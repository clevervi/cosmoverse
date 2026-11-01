"""
Simulation-Based Inference (SBI) / Neural Posterior Estimation (NPE) pipeline (Stage 4 — Phase A).
Facilitates likelihood-free cosmological parameter inference from realistic survey observables
after marginalizing over baryonic (BCEmu) and observational nuisance parameters.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from cosmoverse.inference.flows import ConditionalRealNVP
from cosmoverse.inference.posterior import PosteriorSummary


class SBIEngine:
    """
    Orchestrates the Simulation-Based Inference (SBI) workflow (Phase A):
    1. Sample prior parameters: theta ~ p(theta)
    2. Forward simulations (Emulator + BCEmu + Observational Realism): x = Simulator(theta)
    3. Train neural posterior density estimator: q_phi(theta | x)
    4. Infer posterior p(theta | x_obs) to be validated via SBC and TARP.
    """

    def __init__(
        self,
        simulator_fn: Callable[[np.ndarray], np.ndarray],
        param_bounds: Dict[str, Tuple[float, float]],
        seed: int = 42,
        feature_mode: str = "linear",
        covariance_inflation: float = 1.0,
        density_estimator: str = "regression",
        flow_epochs: int = 800,
        flow_batch_size: int = 64,
        flow_learning_rate: float = 1e-3,
        flow_scale_bound: float = 5.0,
        flow_n_layers: int = 6,
        flow_hidden_dim: int = 0,
    ):
        if feature_mode not in {"linear", "quadratic"}:
            raise ValueError("feature_mode must be 'linear' or 'quadratic'")
        if covariance_inflation <= 0.0:
            raise ValueError("covariance_inflation must be positive")
        if density_estimator not in {"regression", "flow"}:
            raise ValueError("density_estimator must be 'regression' or 'flow'")
        if flow_epochs < 1 or flow_batch_size < 2:
            raise ValueError("invalid flow training hyperparameters")
        if flow_scale_bound <= 0.0:
            raise ValueError("flow_scale_bound must be positive")
        if flow_n_layers < 2:
            raise ValueError("flow_n_layers must be >= 2")
        if flow_hidden_dim < 0:
            raise ValueError("flow_hidden_dim must be non-negative")

        self.simulator_fn = simulator_fn
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.param_dim = len(self.param_names)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.feature_mode = feature_mode
        self.covariance_inflation = float(covariance_inflation)
        self.density_estimator = density_estimator
        self.flow_epochs = int(flow_epochs)
        self.flow_batch_size = int(flow_batch_size)
        self.flow_learning_rate = float(flow_learning_rate)
        self.flow_scale_bound = float(flow_scale_bound)
        self.flow_n_layers = int(flow_n_layers)
        self.flow_hidden_dim = int(flow_hidden_dim)
        self._is_fitted = False

        self.theta_bank: Optional[np.ndarray] = None
        self.x_bank: Optional[np.ndarray] = None
        self.flow: Optional[ConditionalRealNVP] = None

        # Fitted conditional mean & covariance projection from simulation bank
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._theta_mean: Optional[np.ndarray] = None
        self._theta_std: Optional[np.ndarray] = None
        self._reg_weights: Optional[np.ndarray] = None
        self._reg_bias: Optional[np.ndarray] = None
        self._residual_cov_chol: Optional[np.ndarray] = None

    def sample_prior(self, n_simulations: int) -> np.ndarray:
        """Draws parameters uniformly from the prior boundaries."""
        thetas = np.zeros((n_simulations, self.param_dim))
        for j, (_name, (low, high)) in enumerate(self.param_bounds.items()):
            thetas[:, j] = self.rng.uniform(low, high, size=n_simulations)
        return thetas

    def generate_training_bank(
        self,
        n_simulations: int = 50,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs simulator for each drawn parameter vector.
        """
        thetas = self.sample_prior(n_simulations)
        observations = []

        for i in range(n_simulations):
            if verbose and (i % 10 == 0 or i == n_simulations - 1):
                print(f"[SBI Bank] Simulating {i+1}/{n_simulations}...")
            x = np.atleast_1d(np.asarray(self.simulator_fn(thetas[i]), dtype=np.float64))
            observations.append(x)

        self.theta_bank = thetas
        self.x_bank = np.asarray(observations, dtype=np.float64)
        return self.theta_bank, self.x_bank

    def _features(self, x_norm: np.ndarray) -> np.ndarray:
        """Builds the regression feature vector (linear or quadratic)."""
        x_arr = np.asarray(x_norm, dtype=np.float64)
        if self.feature_mode == "linear":
            return x_arr
        return np.hstack([x_arr, x_arr**2])

    def fit_density_estimator(self, n_layers: int = 3, ridge_alpha: float = 1e-2) -> None:
        """
        Initializes and fits the conditional normalizing flow density estimator q_phi(theta | x)
        using the generated simulation bank.
        """
        if self.x_bank is None or self.theta_bank is None:
            raise ValueError("Training bank is empty. Call generate_training_bank() first.")

        # Standardize summary statistics x
        self._x_mean = np.mean(self.x_bank, axis=0)
        self._x_std = np.std(self.x_bank, axis=0) + 1e-8
        x_norm = (self.x_bank - self._x_mean) / self._x_std

        if self.density_estimator == "flow":
            self._theta_mean = np.mean(self.theta_bank, axis=0)
            self._theta_std = np.std(self.theta_bank, axis=0) + 1e-8
            theta_norm = (self.theta_bank - self._theta_mean) / self._theta_std
            self.flow = ConditionalRealNVP(
                param_dim=self.param_dim,
                context_dim=x_norm.shape[1],
                n_layers=self.flow_n_layers,
                seed=self.seed,
                scale_bound=self.flow_scale_bound,
                hidden_dim=self.flow_hidden_dim,
            )
            self.flow.fit(
                theta_norm,
                x_norm,
                n_epochs=self.flow_epochs,
                batch_size=self.flow_batch_size,
                learning_rate=self.flow_learning_rate,
                seed=self.seed,
            )
            self._is_fitted = True
            return

        x_features = self._features(x_norm)
        context_dim = x_features.shape[1]

        self.flow = ConditionalRealNVP(
            param_dim=self.param_dim,
            context_dim=context_dim,
            n_layers=n_layers,
            seed=self.seed,
        )

        # Fit regularized conditional projection E[theta | x] and residual covariance
        n_sim = x_features.shape[0]
        x_aug = np.hstack([x_features, np.ones((n_sim, 1))])
        reg_eye = ridge_alpha * np.eye(context_dim + 1)
        reg_eye[-1, -1] = 0.0  # Do not penalize intercept
        w_full = np.linalg.solve(x_aug.T @ x_aug + reg_eye, x_aug.T @ self.theta_bank)

        self._reg_weights = w_full[:-1, :]
        self._reg_bias = w_full[-1, :]

        residuals = self.theta_bank - (x_aug @ w_full)
        if n_sim > self.param_dim + 2:
            cov = np.cov(residuals, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
        else:
            cov = np.diag([((high - low) * 0.1) ** 2 for (low, high) in self.param_bounds.values()])

        cov_reg = (
            cov * self.covariance_inflation
            + 1e-6 * np.eye(self.param_dim)
        )
        self._residual_cov_chol = np.linalg.cholesky(cov_reg)
        self._is_fitted = True

    def infer(
        self,
        observed_x: np.ndarray,
        n_posterior_samples: int = 2000,
        seed: Optional[int] = None,
    ) -> PosteriorSummary:
        """
        Samples the posterior distribution p(theta | observed_x).
        """
        if not self._is_fitted:
            self.fit_density_estimator()

        obs_arr = np.atleast_1d(np.asarray(observed_x, dtype=np.float64))
        obs_norm = (obs_arr - self._x_mean) / self._x_std

        if self.density_estimator == "flow":
            theta_norm = self.flow.sample(
                obs_norm,
                n_samples=n_posterior_samples,
                seed=seed,
            )
            samples = (
                theta_norm * self._theta_std[np.newaxis, :]
                + self._theta_mean[np.newaxis, :]
            )
            for j, (_name, (low, high)) in enumerate(self.param_bounds.items()):
                samples[:, j] = np.clip(samples[:, j], low, high)
            return PosteriorSummary(samples, param_names=self.param_names)

        obs_features = self._features(obs_norm)

        # Base flow latent samples (near standard normal)
        raw_z = self.flow.sample(
            obs_features,
            n_samples=n_posterior_samples,
            seed=seed if seed is not None else self.seed,
        )
        # Standardize raw flow base draws to zero mean, unit variance per axis
        z_std = (raw_z - np.mean(raw_z, axis=0)) / (np.std(raw_z, axis=0) + 1e-8)

        # Conditional mean prediction + Cholesky-scaled flow fluctuations
        cond_mean = obs_features @ self._reg_weights + self._reg_bias
        samples = cond_mean[np.newaxis, :] + (z_std @ self._residual_cov_chol.T)

        # Clip samples inside physical prior boundaries
        for j, (_name, (low, high)) in enumerate(self.param_bounds.items()):
            samples[:, j] = np.clip(samples[:, j], low, high)

        return PosteriorSummary(samples, param_names=self.param_names)
