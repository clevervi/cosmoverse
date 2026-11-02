"""
CuPy field-level inference benchmark (Stage 5 — Phase B / Paper 2 Extension).

Ports the differentiable field-level negative log-posterior, its exact
analytical gradient and the preconditioned MCLMC chain to CuPy so the same
reconstruction can be benchmarked on a CUDA device against the NumPy engine.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from cosmoverse.acceleration import _import_cupy, cupy_available, device_summary
from cosmoverse.inference.field_level import FieldLevelInferenceEngine
from cosmoverse.inference.mclmc import MicrocanonicalHMC


class GPUFieldLevelEngine:
    """CuPy counterpart of the field-level forward model and its gradient."""

    def __init__(
        self,
        grid_size: int = 64,
        box_size: float = 100.0,
        noise_sigma: float = 0.25,
        nonlinear_coupling: float = 0.12,
        prior_std: np.ndarray | None = None,
    ) -> None:
        if not cupy_available():
            raise RuntimeError("CuPy and a CUDA device are required")
        self.cp = _import_cupy()
        self.grid_size = int(grid_size)
        self.box_size = float(box_size)
        self.noise_sigma = float(noise_sigma)
        self.nonlinear_coupling = float(nonlinear_coupling)
        self.n_voxels = self.grid_size ** 3
        if prior_std is None:
            rng = np.random.default_rng(101)
            raw_scales = np.exp(rng.uniform(-0.9, 0.9, size=self.n_voxels))
            prior_std = np.concatenate([[0.15], raw_scales])
        self.prior_std_np = np.asarray(prior_std, dtype=np.float64)
        if self.prior_std_np.shape != (1 + self.n_voxels,):
            raise ValueError("prior_std has the wrong shape")
        self.prior_std = self.cp.asarray(self.prior_std_np)

    def forward_field_gpu(self, delta_ini_flat: Any, delta_alpha: Any) -> Any:
        alpha = 1.0 + delta_alpha
        return alpha * (
            delta_ini_flat + self.nonlinear_coupling * (delta_ini_flat ** 2)
        )

    def potential_and_grad_gpu(
        self, state: Any, obs_flat: Any
    ) -> tuple[float, Any]:
        cp = self.cp
        state = cp.asarray(state, dtype=cp.float64)
        obs_flat = cp.asarray(obs_flat, dtype=cp.float64)
        delta_alpha = state[0]
        delta_ini = state[1:]
        alpha = 1.0 + delta_alpha

        pred = self.forward_field_gpu(delta_ini, delta_alpha)
        residual = (pred - obs_flat) / self.noise_sigma

        prior_term = 0.5 * cp.sum((state / self.prior_std) ** 2)
        lik_term = 0.5 * cp.sum(residual ** 2)
        potential = prior_term + lik_term

        grad_prior = state / (self.prior_std ** 2)
        dpred_dalpha = delta_ini + self.nonlinear_coupling * (delta_ini ** 2)
        dpred_dini = alpha * (
            1.0 + 2.0 * self.nonlinear_coupling * delta_ini
        )
        grad_alpha = (
            grad_prior[0]
            + cp.sum(residual * dpred_dalpha) / self.noise_sigma
        )
        grad_ini = (
            grad_prior[1:]
            + (residual * dpred_dini) / self.noise_sigma
        )
        grad = cp.concatenate([grad_alpha[None], grad_ini])
        return float(potential), grad

    def _isokinetic_momentum_update_gpu(
        self, u: Any, force: Any, eps_over_d_minus_1: float
    ) -> Any:
        cp = self.cp
        f_norm = float(cp.linalg.norm(force))
        if f_norm < 1e-14:
            return u
        e_vec = force / f_norm
        delta = eps_over_d_minus_1 * f_norm
        u_dot_e = float(cp.dot(u, e_vec))
        sinh_d = math.sinh(delta)
        cosh_d = math.cosh(delta)
        num = u + (sinh_d + u_dot_e * (cosh_d - 1.0)) * e_vec
        denom = cosh_d + u_dot_e * sinh_d
        u_new = num / (denom + 1e-15)
        return u_new / (cp.linalg.norm(u_new) + 1e-15)

    def sample_mclmc_chain_gpu(
        self,
        initial_position: np.ndarray,
        obs_flat: np.ndarray,
        preconditioner_scale: np.ndarray,
        n_samples: int = 12,
        n_leapfrog: int = 4,
        step_size: float = 0.020,
        decoherence_length_L: float = 2.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cp = self.cp
        cp.random.seed(seed)
        q_init = cp.asarray(initial_position, dtype=cp.float64)
        obs = cp.asarray(obs_flat, dtype=cp.float64)
        scale = cp.asarray(preconditioner_scale, dtype=cp.float64)
        dim = max(len(initial_position), 2)

        z = q_init / scale
        u = cp.random.standard_normal(len(initial_position))
        u /= cp.linalg.norm(u) + 1e-15

        sqrt_d = math.sqrt(float(dim))
        eps = step_size
        nu = math.sqrt(
            (math.exp(2.0 * eps / decoherence_length_L) - 1.0) / float(dim)
        )
        samples = cp.zeros((n_samples, len(initial_position)), dtype=cp.float64)
        energy_changes: list[float] = []

        for index in range(n_samples):
            u_prev = self.potential_and_grad_gpu(z * scale, obs)[0]
            for _ in range(n_leapfrog):
                z = z + 0.5 * eps * sqrt_d * u
                _, grad_q = self.potential_and_grad_gpu(z * scale, obs)
                force_z = -scale * grad_q
                u = self._isokinetic_momentum_update_gpu(
                    u,
                    force_z,
                    eps * sqrt_d / max(float(dim - 1), 1.0),
                )
                z = z + 0.5 * eps * sqrt_d * u
                xi = cp.random.standard_normal(len(initial_position))
                u = u + nu * xi
                u /= cp.linalg.norm(u) + 1e-15
            q_curr = z * scale
            u_curr = self.potential_and_grad_gpu(q_curr, obs)[0]
            energy_changes.append((u_curr - u_prev) / float(dim))
            samples[index] = q_curr

        eevpd = float(np.var(energy_changes)) if energy_changes else 0.0
        diagnostics = {
            "sampler": "Preconditioned-MCLMC-GPU",
            "eevpd": eevpd,
            "is_energy_stable": bool(eevpd < 0.1),
            "preconditioned": True,
        }
        return cp.asnumpy(samples), diagnostics


class GPUFieldLevelBenchmark:
    """CPU-vs-GPU field-level posterior and MCLMC benchmark."""

    def __init__(
        self,
        grid_size: int = 64,
        box_size: float = 100.0,
        noise_sigma: float = 0.25,
        nonlinear_coupling: float = 0.12,
        seed: int = 42,
    ) -> None:
        self.grid_size = int(grid_size)
        self.box_size = float(box_size)
        self.noise_sigma = float(noise_sigma)
        self.nonlinear_coupling = float(nonlinear_coupling)
        self.seed = int(seed)

    def _build_problem(self) -> dict[str, Any]:
        cpu_engine = FieldLevelInferenceEngine(
            grid_size=self.grid_size,
            box_size=self.box_size,
            noise_sigma=self.noise_sigma,
            nonlinear_coupling=self.nonlinear_coupling,
        )
        rng = np.random.default_rng(self.seed + 1)
        true_delta_alpha = 0.05
        true_ini = rng.normal(0.0, cpu_engine.prior_std[1:])
        obs = cpu_engine.forward_field(true_ini, true_delta_alpha) + rng.normal(
            0.0, self.noise_sigma, size=cpu_engine.n_voxels
        )
        post_scale = 1.0 / np.sqrt(
            (1.0 / (cpu_engine.prior_std ** 2))
            + (1.0 / (self.noise_sigma ** 2))
        )
        post_scale[0] = 0.04
        wiener_ini = (obs / (self.noise_sigma ** 2)) * (post_scale[1:] ** 2)
        q_init = np.concatenate([[0.04], wiener_ini])
        return {
            "cpu_engine": cpu_engine,
            "obs": obs,
            "q_init": q_init,
            "post_scale": post_scale,
        }

    @staticmethod
    def _time_call(fn: Any, repeats: int) -> float:
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) / float(repeats)

    def run(
        self,
        n_samples: int = 12,
        n_leapfrog: int = 4,
        step_size: float = 0.020,
    ) -> dict[str, Any]:
        if not cupy_available():
            raise RuntimeError("CuPy and a CUDA device are required")
        cp = _import_cupy()
        problem = self._build_problem()
        cpu_engine: FieldLevelInferenceEngine = problem["cpu_engine"]
        obs = problem["obs"]
        q_init = problem["q_init"]
        post_scale = problem["post_scale"]

        gpu_engine = GPUFieldLevelEngine(
            grid_size=self.grid_size,
            box_size=self.box_size,
            noise_sigma=self.noise_sigma,
            nonlinear_coupling=self.nonlinear_coupling,
            prior_std=cpu_engine.prior_std,
        )

        cpu_u, cpu_g = cpu_engine.potential_and_grad(q_init, obs)
        gpu_u, gpu_g = gpu_engine.potential_and_grad_gpu(q_init, obs)
        potential_rel_error = abs(cpu_u - gpu_u) / max(abs(cpu_u), 1e-12)
        grad_max_abs_error = float(
            cp.max(cp.abs(cp.asarray(cpu_g) - gpu_g))
        )

        cpu_pg_seconds = self._time_call(
            lambda: cpu_engine.potential_and_grad(q_init, obs), repeats=3
        )
        for _ in range(2):
            gpu_engine.potential_and_grad_gpu(q_init, obs)
        cp.cuda.Stream.null.synchronize()
        gpu_pg_seconds = self._time_call(
            lambda: gpu_engine.potential_and_grad_gpu(q_init, obs), repeats=5
        )

        def pot_fn(q: np.ndarray) -> float:
            return cpu_engine.potential_and_grad(q, obs)[0]

        def grad_fn(q: np.ndarray) -> np.ndarray:
            return cpu_engine.potential_and_grad(q, obs)[1]

        cpu_sampler = MicrocanonicalHMC(
            potential_fn=pot_fn,
            grad_potential_fn=grad_fn,
            step_size=step_size,
            n_leapfrog_steps=n_leapfrog,
            preconditioner_scale=post_scale,
        )
        start = time.perf_counter()
        cpu_chain = cpu_sampler.sample_mclmc_chain(
            q_init, n_samples=n_samples, seed=self.seed
        )
        cpu_chain_seconds = time.perf_counter() - start

        gpu_engine.sample_mclmc_chain_gpu(
            q_init,
            obs,
            post_scale,
            n_samples=1,
            n_leapfrog=1,
            step_size=step_size,
            seed=self.seed + 99,
        )
        start = time.perf_counter()
        gpu_chain, gpu_diagnostics = gpu_engine.sample_mclmc_chain_gpu(
            q_init,
            obs,
            post_scale,
            n_samples=n_samples,
            n_leapfrog=n_leapfrog,
            step_size=step_size,
            seed=self.seed,
        )
        gpu_chain_seconds = time.perf_counter() - start

        return {
            "grid_size": self.grid_size,
            "dimension": 1 + cpu_engine.n_voxels,
            "n_samples": n_samples,
            "n_leapfrog": n_leapfrog,
            "potential_relative_error": potential_rel_error,
            "grad_max_abs_error": grad_max_abs_error,
            "cpu_potential_grad_seconds": cpu_pg_seconds,
            "gpu_potential_grad_seconds": gpu_pg_seconds,
            "potential_grad_speedup": cpu_pg_seconds / max(gpu_pg_seconds, 1e-12),
            "cpu_chain_seconds": cpu_chain_seconds,
            "gpu_chain_seconds": gpu_chain_seconds,
            "chain_speedup": cpu_chain_seconds / max(gpu_chain_seconds, 1e-12),
            "cpu_diagnostics": cpu_sampler.last_diagnostics,
            "gpu_diagnostics": gpu_diagnostics,
            "cpu_marginal_alpha_mean": float(
                np.mean(cpu_chain[n_samples // 4 :, 0])
            ),
            "gpu_marginal_alpha_mean": float(
                np.mean(gpu_chain[n_samples // 4 :, 0])
            ),
            "gpu_chain_shape": list(gpu_chain.shape),
            "device": device_summary(),
        }
