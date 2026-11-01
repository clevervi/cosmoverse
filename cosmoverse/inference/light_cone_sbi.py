"""
Level 3 End-to-End Simulation-Based Inference on the Light-Cone Angular
Correlation Function w(theta) (`LightConeSBIPipeline`).

The data vector is the 12-bin angular correlation function measured on the
light cone. The forward model is a fast Limber + BCEmu surrogate:
  w(theta; Omega_m, sigma8, A_AGN) = Hankel[b1^2 S_b(k; A_AGN) P_m(k) / chi^2],
with b1 fixed to the galaxy-bias prior. The posterior is learned by
`SBIEngine` (quadratic conditional projection + conditional RealNVP residuals)
and validated with SBC and TARP before the blocking gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import integrate, special

from cosmoverse.baryons.feedback import BCEmuModel
from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import BaryonParameters, CosmologicalParameters
from cosmoverse.inference.sbi import SBIEngine
from cosmoverse.validation.sbc import SimulationBasedCalibration
from cosmoverse.validation.tarp import TARPCalibration


def compute_light_cone_w_theta(
    omega_m: float,
    sigma8: float,
    b1: float,
    a_agn: float,
    z_eff: float,
    theta_deg: np.ndarray,
    ell: np.ndarray,
) -> np.ndarray:
    """
    Noiseless light-cone angular correlation function from the Limber + BCEmu
    surrogate:
      C_ell = b1^2 S_b(k; A_AGN) P_m(k) / chi^2,
      w(theta) = int ell d ell / (2 pi) C_ell J0(ell theta).
    """
    cosmology = Cosmology(
        CosmologicalParameters(omega_m=float(omega_m), sigma8=float(sigma8))
    )
    chi = max(float(cosmology.comoving_distance(z_eff)), 100.0)
    k = (np.asarray(ell, dtype=np.float64) + 0.5) / chi

    pk_m = np.asarray(
        cosmology.linear_matter_power_spectrum(k, z=z_eff),
        dtype=np.float64,
    ) * (1.0 + 0.45 * ((k / 0.25) ** 0.8))
    suppression = BCEmuModel(
        BaryonParameters(mode="BCEmu3", A_AGN=float(a_agn))
    ).suppression_factor(k, z=z_eff)
    c_ell = (float(b1) ** 2) * pk_m * suppression / (chi**2)

    theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    return np.array(
        [
            integrate.simpson(
                np.asarray(ell) * c_ell * special.j0(np.asarray(ell) * angle)
                / (2.0 * np.pi),
                x=ell,
            )
            for angle in theta_rad
        ]
    )


class LightConeSBIPipeline:
    """
    Fast light-cone forward model + SBI + SBC/TARP calibration for
    (Omega_m, sigma8, A_AGN) with a fixed galaxy bias b1.
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        b1_fixed: float = 1.65,
        z_eff: float = 0.8,
        theta_min_deg: float = 1.0,
        theta_max_deg: float = 20.0,
        n_theta_bins: int = 12,
        noise_fraction: float = 0.05,
        noise_floor: float = 0.005,
        n_training_simulations: int = 200,
        covariance_inflation: float = 1.3,
        seed: int = 42,
    ) -> None:
        self.param_bounds = param_bounds or {
            "omega_m": (0.24, 0.40),
            "sigma8": (0.72, 0.90),
            "A_AGN": (0.5, 2.0),
        }
        if set(self.param_bounds) != {"omega_m", "sigma8", "A_AGN"}:
            raise ValueError(
                "param_bounds must contain exactly omega_m, sigma8 and A_AGN"
            )
        if n_theta_bins < 2:
            raise ValueError("n_theta_bins must be >= 2")
        if theta_min_deg <= 0.0 or theta_max_deg <= theta_min_deg:
            raise ValueError("theta bounds must satisfy 0 < theta_min < theta_max")
        if noise_fraction < 0.0 or noise_floor <= 0.0:
            raise ValueError("noise parameters must be non-negative")
        if n_training_simulations < 10:
            raise ValueError("n_training_simulations must be >= 10")

        self.b1_fixed = float(b1_fixed)
        self.z_eff = float(z_eff)
        self.theta_deg = np.linspace(
            float(theta_min_deg), float(theta_max_deg), int(n_theta_bins)
        )
        self.ell = np.linspace(1.0, 8000.0, 1024)
        self.noise_fraction = float(noise_fraction)
        self.noise_floor = float(noise_floor)
        self.n_training_simulations = int(n_training_simulations)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.sbi = SBIEngine(
            simulator_fn=self.forward_model,
            param_bounds=self.param_bounds,
            seed=seed,
            feature_mode="quadratic",
            covariance_inflation=covariance_inflation,
        )
        self.is_fitted = False

    def model_w_theta(
        self,
        theta: np.ndarray,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        Evaluates the noiseless (or noisy) light-cone w(theta) for
        theta = (Omega_m, sigma8, A_AGN).
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.shape != (3,):
            raise ValueError("theta must have shape (3,) = (omega_m, sigma8, A_AGN)")

        omega_m, sigma8, a_agn = (float(value) for value in theta_arr)
        w_theta = compute_light_cone_w_theta(
            omega_m,
            sigma8,
            self.b1_fixed,
            a_agn,
            self.z_eff,
            self.theta_deg,
            self.ell,
        )
        if add_noise:
            sigma = self.noise_fraction * np.maximum(
                np.abs(w_theta), self.noise_floor
            )
            w_theta = w_theta + self.rng.normal(0.0, sigma)
        return w_theta

    def forward_model(self, theta: np.ndarray) -> np.ndarray:
        """Noisy forward simulator used by SBI training and SBC/TARP."""
        return self.model_w_theta(theta, add_noise=True)

    def fiducial_w_theta(self) -> np.ndarray:
        """Noiseless fiducial w(theta) at (0.315, 0.811, 1.0)."""
        return self.model_w_theta(np.array([0.315, 0.811, 1.0]), add_noise=False)

    def fit(self, n_simulations: int | None = None, verbose: bool = False) -> "LightConeSBIPipeline":
        """Generates the training bank and fits the conditional flow."""
        n_sims = (
            self.n_training_simulations
            if n_simulations is None
            else int(n_simulations)
        )
        self.sbi.generate_training_bank(n_simulations=n_sims, verbose=verbose)
        self.sbi.fit_density_estimator(n_layers=3)
        self.is_fitted = True
        return self

    def infer(
        self,
        w_obs: np.ndarray,
        n_posterior_samples: int = 150,
        seed: int | None = None,
    ):
        """Returns the posterior summary for an observed w(theta) vector."""
        if not self.is_fitted:
            self.fit()
        return self.sbi.infer(
            w_obs,
            n_posterior_samples=n_posterior_samples,
            seed=seed,
        )

    def run_calibration(
        self,
        n_sbc_trials: int = 50,
        n_tarp_trials: int = 50,
        n_posterior_samples: int = 150,
    ) -> dict[str, Any]:
        """Runs SBC and TARP and evaluates the blocking joint gate."""
        if not self.is_fitted:
            self.fit()

        sbc = SimulationBasedCalibration(
            sample_prior_fn=lambda: self.sbi.sample_prior(1)[0],
            simulate_fn=self.forward_model,
            infer_fn=lambda x, n: self.sbi.infer(
                x, n_posterior_samples=n
            ).samples,
            param_names=list(self.param_bounds),
        )
        sbc_metrics = sbc.run_calibration(
            n_trials=int(n_sbc_trials),
            n_posterior_samples=int(n_posterior_samples),
        )

        test_thetas = self.sbi.sample_prior(int(n_tarp_trials))
        test_obs = np.array([self.forward_model(theta) for theta in test_thetas])
        test_posteriors = np.array(
            [
                self.sbi.infer(x, n_posterior_samples=n_posterior_samples).samples
                for x in test_obs
            ]
        )
        tarp = TARPCalibration(param_bounds=self.param_bounds, seed=self.seed)
        tarp_result = tarp.evaluate(test_thetas, test_posteriors)

        passes_sbc = all(
            metric["passes_acceptance_gate"] for metric in sbc_metrics.values()
        )
        return {
            "param_names": list(self.param_bounds),
            "b1_fixed": self.b1_fixed,
            "theta_deg": self.theta_deg,
            "n_training_simulations": self.n_training_simulations,
            "sbc": sbc_metrics,
            "tarp": tarp_result,
            "passes_sbc_gate": bool(passes_sbc),
            "passes_tarp_gate": bool(tarp_result["passes_acceptance_gate"]),
            "passes_joint_sbi_gate": bool(
                passes_sbc and tarp_result["passes_acceptance_gate"]
            ),
        }
