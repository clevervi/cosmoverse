"""
Preconditioned Microcanonical Langevin Monte Carlo (MCLMC) and HMC sampling (Stage 5 — Phase B).
Provides high-dimensional gradient-based sampling for field-level inference (e.g. JaxPM + MCLMC).

References:
  - Robnik, J., De Luca, G. B., Silverstein, E., & Seljak, U. (2023) "Microcanonical Hamiltonian Monte Carlo"
  - Simon-Onfroy, A., Lanusse, F., & de Mattia, A. (JCAP, Dec 2025) field-level benchmark on JaxPM.

Methodological Caution (Master Document Section 6.5 & 12):
  MCLMC is a promising candidate for >= 10^6-dimensional field-level inference, NOT a magic bullet.
  It requires explicit latent-space preconditioning (e.g. whitening initial modes by sqrt(P(k)))
  and verification of energy variance and marginal bias against HMC/NUTS baselines (Plan B).
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np


class MicrocanonicalHMC:
    """
    Preconditioned Microcanonical Langevin Monte Carlo (MCLMC) and Metropolis-HMC sampler.
    Supports diagonal/Fourier latent preconditioning q_phys = preconditioner_scale * q_latent.
    """

    def __init__(
        self,
        potential_fn: Callable[[np.ndarray], float],
        grad_potential_fn: Callable[[np.ndarray], np.ndarray],
        step_size: float = 0.05,
        n_leapfrog_steps: int = 10,
        preconditioner_scale: Optional[np.ndarray] = None,
        decoherence_length_L: float = 2.0,
    ):
        self.potential_fn = potential_fn
        self.grad_fn = grad_potential_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog_steps
        self.preconditioner_scale = preconditioner_scale
        self.decoherence_length_L = decoherence_length_L
        self.last_diagnostics: Dict[str, Any] = {}

    def _get_scale(self, dim: int) -> np.ndarray:
        if self.preconditioner_scale is None:
            return np.ones(dim, dtype=np.float64)
        return np.asarray(self.preconditioner_scale, dtype=np.float64)

    @staticmethod
    def _isokinetic_momentum_update(u: np.ndarray, force: np.ndarray, eps_over_d_minus_1: float) -> np.ndarray:
        """
        Exact integrator step for Microcanonical momentum on the unit sphere S^(d-1):
        du/dt = (I - u u^T) * force / (d - 1)
        """
        f_norm = float(np.linalg.norm(force))
        if f_norm < 1e-14:
            return u
        e_vec = force / f_norm
        delta = eps_over_d_minus_1 * f_norm
        u_dot_e = float(np.dot(u, e_vec))
        # Hyperbolic rotation on S^(d-1)
        sinh_d = np.sinh(delta)
        cosh_d = np.cosh(delta)
        num = u + (sinh_d + u_dot_e * (cosh_d - 1.0)) * e_vec
        denom = cosh_d + u_dot_e * sinh_d
        u_new = num / (denom + 1e-15)
        return u_new / (np.linalg.norm(u_new) + 1e-15)

    def sample_mclmc_chain(
        self,
        initial_position: np.ndarray,
        n_samples: int = 500,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Runs Preconditioned Microcanonical Langevin Monte Carlo (MCLMC) without
        Metropolis-Hastings rejections, tracking Energy Error Variance Per Dimension (EEVPD).
        """
        rng = np.random.default_rng(seed)
        q_init = np.asarray(initial_position, dtype=np.float64)
        dim = max(len(q_init), 2)
        scale = self._get_scale(len(q_init))

        # Transform to preconditioned latent space z = q / scale
        z = q_init / scale

        # Unit momentum vector u on S^(d-1)
        u = rng.standard_normal(len(q_init))
        u /= np.linalg.norm(u) + 1e-15

        sqrt_d = np.sqrt(float(dim))
        eps = self.step_size
        nu = np.sqrt((np.exp(2.0 * eps / self.decoherence_length_L) - 1.0) / float(dim))

        samples = np.zeros((n_samples, len(q_init)), dtype=np.float64)
        energy_changes: List[float] = []

        for i in range(n_samples):
            u_prev = self.potential_fn(z * scale)

            for _ in range(self.n_leapfrog):
                # Half drift in latent space
                z = z + 0.5 * eps * sqrt_d * u
                # Preconditioned force in latent space: - dU/dz = - scale * dU/dq
                grad_q = self.grad_fn(z * scale)
                force_z = -scale * grad_q
                u = self._isokinetic_momentum_update(u, force_z, eps * sqrt_d / max(float(dim - 1), 1.0))
                # Second half drift
                z = z + 0.5 * eps * sqrt_d * u

                # Partial momentum refreshment (Langevin noise on unit sphere)
                xi = rng.standard_normal(len(q_init))
                u = u + nu * xi
                u /= np.linalg.norm(u) + 1e-15

            q_curr = z * scale
            u_curr = self.potential_fn(q_curr)
            energy_changes.append((u_curr - u_prev) / float(dim))
            samples[i] = q_curr

        eevpd = float(np.var(energy_changes)) if energy_changes else 0.0
        self.last_diagnostics = {
            "sampler": "Preconditioned-MCLMC",
            "eevpd": eevpd,
            "is_energy_stable": bool(eevpd < 0.1),
            "preconditioned": self.preconditioner_scale is not None,
        }
        return samples

    def sample_chain(
        self,
        initial_position: np.ndarray,
        n_samples: int = 500,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Runs Preconditioned Metropolis-adjusted Hamiltonian Monte Carlo (HMC — Plan B baseline).
        """
        rng = np.random.default_rng(seed)
        q_init = np.asarray(initial_position, dtype=np.float64)
        dim = len(q_init)
        scale = self._get_scale(dim)
        samples = np.zeros((n_samples, dim))

        current_z = q_init / scale
        accepted = 0

        for i in range(n_samples):
            p = rng.standard_normal(dim)
            current_p = np.copy(p)

            current_u = self.potential_fn(current_z * scale)
            current_k = 0.5 * np.sum(current_p ** 2)

            z_prop = np.copy(current_z)
            p_prop = np.copy(current_p)

            # Half step for momentum in preconditioned coordinates
            p_prop -= 0.5 * self.step_size * (scale * self.grad_fn(z_prop * scale))

            for _ in range(self.n_leapfrog):
                z_prop += self.step_size * p_prop
                p_prop -= self.step_size * (scale * self.grad_fn(z_prop * scale))

            p_prop += 0.5 * self.step_size * (scale * self.grad_fn(z_prop * scale))
            p_prop = -p_prop

            prop_u = self.potential_fn(z_prop * scale)
            prop_k = 0.5 * np.sum(p_prop ** 2)

            log_alpha = (current_u + current_k) - (prop_u + prop_k)
            if np.log(rng.uniform(1e-15, 1.0)) < min(0.0, log_alpha):
                current_z = z_prop
                accepted += 1

            samples[i] = current_z * scale

        self.last_diagnostics = {
            "sampler": "Preconditioned-HMC",
            "acceptance_rate": accepted / float(max(n_samples, 1)),
            "preconditioned": self.preconditioner_scale is not None,
        }
        return samples
