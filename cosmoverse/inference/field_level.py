"""
Differentiable Field-Level Inference Engine (Stage 5 — Phase B / Paper 2 Extension).

Aligns with Section 6.5, Section 9, and Section 12 of docs/MASTER_DOCUMENT.md:
  - Jointly reconstructs the 3D initial density field delta_ini(x) and cosmological
    amplitude scaling parameters using exact analytical gradients of a forward model.
  - Explicitly benchmarks Preconditioned MCLMC against Unpreconditioned MCLMC and
    Metropolis-adjusted HMC (Plan B baseline), reporting:
      1. Fourier cross-correlation r(k) between reconstructed and true initial conditions.
      2. Effective Sample Size per gradient evaluation (ESS / N_grad).
      3. Energy Error Variance Per Dimension (EEVPD) and marginal posterior bias.
"""

from typing import Any

import numpy as np

from cosmoverse.emulators.training import cross_correlation_coefficient
from cosmoverse.inference.mclmc import MicrocanonicalHMC


class FieldLevelInferenceEngine:
    """
    Differentiable field-level Bayesian reconstruction of 3D initial density fields
    delta_ini(x) and growth/clustering amplitude alpha_s8 = sigma_8 / sigma_8,fid.
    """

    def __init__(
        self,
        grid_size: int = 8,
        box_size: float = 100.0,
        noise_sigma: float = 0.25,
        nonlinear_coupling: float = 0.12,
    ):
        self.grid_size = grid_size
        self.box_size = box_size
        self.noise_sigma = noise_sigma
        self.nonlinear_coupling = nonlinear_coupling
        self.n_voxels = grid_size ** 3

        # Build anisotropic prior spectrum scales sqrt(P_prior(x)) to test preconditioning
        rng = np.random.default_rng(101)
        raw_scales = np.exp(rng.uniform(-0.9, 0.9, size=self.n_voxels))
        self.prior_std = np.concatenate([[0.15], raw_scales])  # index 0 = delta_alpha_s8

    def forward_field(self, delta_ini_flat: np.ndarray, delta_alpha: float) -> np.ndarray:
        """
        Differentiable weakly non-linear forward map:
        delta_final(x) = (1 + delta_alpha) * [delta_ini(x) + c_nl * delta_ini(x)^2]
        """
        alpha = 1.0 + delta_alpha
        return alpha * (delta_ini_flat + self.nonlinear_coupling * (delta_ini_flat ** 2))

    def potential_and_grad(
        self, state: np.ndarray, obs_flat: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Computes negative log-posterior U(q) and exact analytical gradient dU/dq
        for state q = [delta_alpha, delta_ini_flat...].
        """
        delta_alpha = float(state[0])
        delta_ini = state[1:]
        alpha = 1.0 + delta_alpha

        pred = self.forward_field(delta_ini, delta_alpha)
        residual = (pred - obs_flat) / self.noise_sigma

        # Prior + Likelihood negative log-density
        prior_term = 0.5 * float(np.sum((state / self.prior_std) ** 2))
        lik_term = 0.5 * float(np.sum(residual ** 2))
        potential = prior_term + lik_term

        # Analytical gradients
        grad_prior = state / (self.prior_std ** 2)
        dpred_dalpha = delta_ini + self.nonlinear_coupling * (delta_ini ** 2)
        dpred_dini = alpha * (1.0 + 2.0 * self.nonlinear_coupling * delta_ini)

        grad_alpha = grad_prior[0] + float(np.sum(residual * dpred_dalpha)) / self.noise_sigma
        grad_ini = grad_prior[1:] + (residual * dpred_dini) / self.noise_sigma

        grad = np.concatenate([[grad_alpha], grad_ini])
        return potential, grad

    @staticmethod
    def _estimate_ess_1d(trace: np.ndarray) -> float:
        """Estimates 1D Effective Sample Size via lag-1 autocorrelation rho_1."""
        n = len(trace)
        if n < 4:
            return float(n)
        centered = trace - np.mean(trace)
        var = float(np.var(centered))
        if var < 1e-14:
            return 1.0
        rho1 = float(np.sum(centered[:-1] * centered[1:]) / ((n - 1) * var))
        rho1_clipped = float(np.clip(rho1, -0.95, 0.98))
        ess = n * (1.0 - rho1_clipped) / (1.0 + rho1_clipped)
        return float(np.clip(ess, 1.0, float(n)))

    def benchmark_mclmc_vs_hmc(
        self,
        n_samples: int = 80,
        n_leapfrog: int = 6,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Executes the Stage 5 benchmark comparing:
          1. Preconditioned MCLMC (whitened latent space via self.prior_std)
          2. Unpreconditioned MCLMC (demonstrating degradation without preconditioning)
          3. Preconditioned HMC (Plan B baseline)
        """
        rng = np.random.default_rng(seed)
        true_delta_alpha = 0.05
        true_ini_flat = rng.normal(0.0, self.prior_std[1:])
        true_obs_flat = self.forward_field(true_ini_flat, true_delta_alpha) + rng.normal(
            0.0, self.noise_sigma, size=self.n_voxels
        )

        def pot_fn(q: np.ndarray) -> float:
            u, _ = self.potential_and_grad(q, true_obs_flat)
            return u

        def grad_fn(q: np.ndarray) -> np.ndarray:
            _, g = self.potential_and_grad(q, true_obs_flat)
            return g

        # Posterior precision preconditioner scale: sigma_post_i = 1 / sqrt(1/sigma_prior^2 + 1/sigma_noise^2)
        post_scale = 1.0 / np.sqrt((1.0 / (self.prior_std ** 2)) + (1.0 / (self.noise_sigma ** 2)))
        post_scale[0] = 0.04

        # Warm start near MAP linear Wiener estimate
        wiener_ini = (true_obs_flat / (self.noise_sigma ** 2)) * (post_scale[1:] ** 2)
        q_init = np.concatenate([[0.04], wiener_ini])

        # 1. Preconditioned MCLMC
        mclmc_pre = MicrocanonicalHMC(
            potential_fn=pot_fn,
            grad_potential_fn=grad_fn,
            step_size=0.020,
            n_leapfrog_steps=n_leapfrog,
            preconditioner_scale=post_scale,
        )
        chain_mclmc_pre = mclmc_pre.sample_mclmc_chain(q_init, n_samples=n_samples, seed=seed)

        # 2. Unpreconditioned MCLMC (same step size, no preconditioner -> degraded EEVPD)
        mclmc_unpre = MicrocanonicalHMC(
            potential_fn=pot_fn,
            grad_potential_fn=grad_fn,
            step_size=0.020,
            n_leapfrog_steps=n_leapfrog,
            preconditioner_scale=None,
        )
        mclmc_unpre.sample_mclmc_chain(
            q_init, n_samples=n_samples, seed=seed + 1
        )

        # 3. Preconditioned HMC (Plan B baseline)
        hmc_pre = MicrocanonicalHMC(
            potential_fn=pot_fn,
            grad_potential_fn=grad_fn,
            step_size=0.025,
            n_leapfrog_steps=n_leapfrog,
            preconditioner_scale=post_scale,
        )
        chain_hmc = hmc_pre.sample_chain(q_init, n_samples=n_samples, seed=seed + 2)

        # Evaluate 3D initial condition reconstruction r(k) for Preconditioned MCLMC
        mean_ini_mclmc = np.mean(chain_mclmc_pre[n_samples // 4 :, 1:], axis=0).reshape(
            (self.grid_size, self.grid_size, self.grid_size)
        )
        true_ini_3d = true_ini_flat.reshape((self.grid_size, self.grid_size, self.grid_size))
        k_bins, r_k = cross_correlation_coefficient(
            mean_ini_mclmc, true_ini_3d, box_size=self.box_size, n_bins=4
        )

        n_grads = float(n_samples * n_leapfrog)
        ess_mclmc = self._estimate_ess_1d(chain_mclmc_pre[:, 0])
        ess_hmc = self._estimate_ess_1d(chain_hmc[:, 0])

        bias_mclmc = float(abs(np.mean(chain_mclmc_pre[n_samples // 4 :, 0]) - true_delta_alpha))
        bias_hmc = float(abs(np.mean(chain_hmc[n_samples // 4 :, 0]) - true_delta_alpha))

        return {
            "dimension": 1 + self.n_voxels,
            "k_bins": k_bins.tolist(),
            "ic_cross_correlation_rk": r_k.tolist(),
            "mean_large_scale_rk": float(np.mean(r_k[:2])),
            "preconditioned_mclmc": {
                "eevpd": mclmc_pre.last_diagnostics["eevpd"],
                "is_energy_stable": mclmc_pre.last_diagnostics["is_energy_stable"],
                "ess_per_grad": ess_mclmc / n_grads,
                "marginal_alpha_bias": bias_mclmc,
            },
            "unpreconditioned_mclmc": {
                "eevpd": mclmc_unpre.last_diagnostics["eevpd"],
                "degradation_ratio_eevpd": (
                    mclmc_unpre.last_diagnostics["eevpd"]
                    / max(mclmc_pre.last_diagnostics["eevpd"], 1e-12)
                ),
            },
            "preconditioned_hmc_plan_b": {
                "acceptance_rate": hmc_pre.last_diagnostics["acceptance_rate"],
                "ess_per_grad": ess_hmc / n_grads,
                "marginal_alpha_bias": bias_hmc,
            },
        }
