"""
Particle-Mesh trained matter power spectrum emulator (`PMPowerSpectrumEmulator`).

Trains a quadratic response surface for log P(k; Omega_m, sigma8) on a grid of
real Particle-Mesh simulations and provides a hybrid extrapolation: the emulated
P(k) is used inside the measured k range, while the analytic Eisenstein-Hu shape
with the boundary amplitude correction is used outside it.
"""

from __future__ import annotations

import numpy as np

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import CosmologicalParameters, SimulationParameters
from cosmoverse.simulation.nbody import ParticleMeshSimulator


class PMPowerSpectrumEmulator:
    """
    Quadratic log-P(k) emulator trained on PM simulations.
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        z_eff: float = 0.1,
        box_size: float = 200.0,
        n_mesh: int = 32,
        n_steps: int = 3,
        n_k_bins: int = 8,
        n_grid: int = 5,
        seed: int = 42,
    ) -> None:
        self.param_bounds = param_bounds or {
            "omega_m": (0.24, 0.40),
            "sigma8": (0.72, 0.90),
        }
        if set(self.param_bounds) != {"omega_m", "sigma8"}:
            raise ValueError(
                "param_bounds must contain exactly omega_m and sigma8"
            )
        if n_grid < 3:
            raise ValueError("n_grid must be >= 3 for the quadratic response")
        if n_k_bins < 3:
            raise ValueError("n_k_bins must be >= 3")

        self.z_eff = float(z_eff)
        self.box_size = float(box_size)
        self.n_mesh = int(n_mesh)
        self.n_steps = int(n_steps)
        self.n_k_bins = int(n_k_bins)
        self.n_grid = int(n_grid)
        self.seed = int(seed)

        self.k_grid: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.training_mape_pct: float | None = None
        self.loo_mape_pct: float | None = None
        self.is_trained = False

    @staticmethod
    def _features(omega_m: float, sigma8: float) -> np.ndarray:
        return np.array(
            [
                1.0,
                omega_m,
                sigma8,
                omega_m**2,
                omega_m * sigma8,
                sigma8**2,
            ],
            dtype=np.float64,
        )

    def _design_matrix(self, thetas: np.ndarray) -> np.ndarray:
        return np.array(
            [self._features(float(om), float(s8)) for om, s8 in thetas],
            dtype=np.float64,
        )

    def train(self, verbose: bool = False) -> "PMPowerSpectrumEmulator":
        """Runs the PM grid and fits the quadratic log-P(k) response surface."""
        omega_values = np.linspace(*self.param_bounds["omega_m"], self.n_grid)
        sigma_values = np.linspace(*self.param_bounds["sigma8"], self.n_grid)

        thetas: list[list[float]] = []
        log_pk: list[np.ndarray] = []
        k_grid: np.ndarray | None = None

        for omega_m in omega_values:
            for sigma8 in sigma_values:
                cosmology = Cosmology(
                    CosmologicalParameters(
                        omega_m=float(omega_m), sigma8=float(sigma8)
                    )
                )
                sim_params = SimulationParameters(
                    box_size=self.box_size,
                    n_mesh=self.n_mesh,
                    n_steps=self.n_steps,
                    z_initial=5.0,
                    z_final=self.z_eff,
                    seed=self.seed,
                )
                simulator = ParticleMeshSimulator(
                    cosmology=cosmology, sim_params=sim_params
                )
                simulator.run()
                k, pk = simulator.compute_power_spectrum(n_bins=self.n_k_bins)
                if k_grid is None:
                    k_grid = np.asarray(k, dtype=np.float64)
                    if k_grid.shape != (self.n_k_bins,):
                        raise RuntimeError("Unexpected PM k-grid shape")
                thetas.append([float(omega_m), float(sigma8)])
                log_pk.append(np.log(np.maximum(pk, 1e-12)))
                if verbose:
                    print(
                        f"[PM emulator] Omega_m={omega_m:.3f}, "
                        f"sigma8={sigma8:.3f}"
                    )

        theta_array = np.asarray(thetas, dtype=np.float64)
        log_pk_array = np.asarray(log_pk, dtype=np.float64)
        design = self._design_matrix(theta_array)
        coefficients = np.linalg.lstsq(design, log_pk_array, rcond=None)[0]

        prediction = np.exp(design @ coefficients)
        training_mape = float(
            np.mean(
                np.abs(prediction - np.exp(log_pk_array))
                / np.maximum(np.exp(log_pk_array), 1e-12)
            )
            * 100.0
        )

        loo_errors: list[float] = []
        for index in range(len(theta_array)):
            keep = np.ones(len(theta_array), dtype=bool)
            keep[index] = False
            loo_coefficients = np.linalg.lstsq(
                design[keep], log_pk_array[keep], rcond=None
            )[0]
            loo_prediction = np.exp(design[index] @ loo_coefficients)
            truth = np.exp(log_pk_array[index])
            loo_errors.append(
                float(
                    np.mean(
                        np.abs(loo_prediction - truth) / np.maximum(truth, 1e-12)
                    )
                    * 100.0
                )
            )

        self.k_grid = np.asarray(k_grid, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.training_mape_pct = training_mape
        self.loo_mape_pct = float(np.mean(loo_errors))
        self.is_trained = True
        return self

    def predict_pk(
        self,
        k: np.ndarray,
        omega_m: float,
        sigma8: float,
    ) -> np.ndarray:
        """
        Returns the emulated P(k) in (Mpc/h)^3, using the analytic Eisenstein-Hu
        shape with boundary amplitude corrections outside the PM k range.
        """
        if not self.is_trained or self.k_grid is None or self.coefficients is None:
            raise ValueError("Emulator is not trained; call train() first")

        k_arr = np.asarray(k, dtype=np.float64)
        log_pk_grid = self._features(float(omega_m), float(sigma8)) @ self.coefficients

        cosmology = Cosmology(
            CosmologicalParameters(
                omega_m=float(omega_m), sigma8=float(sigma8)
            )
        )
        analytic = np.asarray(
            cosmology.linear_matter_power_spectrum(k_arr, z=self.z_eff),
            dtype=np.float64,
        )
        analytic_min = float(
            cosmology.linear_matter_power_spectrum(
                self.k_grid[0], z=self.z_eff
            )
        )
        analytic_max = float(
            cosmology.linear_matter_power_spectrum(
                self.k_grid[-1], z=self.z_eff
            )
        )
        ratio_min = float(np.exp(log_pk_grid[0])) / max(analytic_min, 1e-30)
        ratio_max = float(np.exp(log_pk_grid[-1])) / max(analytic_max, 1e-30)

        interpolated = np.exp(
            np.interp(
                np.log(np.maximum(k_arr, 1e-12)),
                np.log(self.k_grid),
                log_pk_grid,
            )
        )
        return np.where(
            k_arr < self.k_grid[0],
            analytic * ratio_min,
            np.where(k_arr <= self.k_grid[-1], interpolated, analytic * ratio_max),
        )

    def evaluate_accuracy(self) -> dict[str, float | bool]:
        """Returns training and leave-one-out MAPE plus the blocking gate."""
        if not self.is_trained:
            raise ValueError("Emulator is not trained; call train() first")
        loo = float(self.loo_mape_pct or 0.0)
        return {
            "training_mape_pct": float(self.training_mape_pct or 0.0),
            "loo_mape_pct": loo,
            "n_training_points": int(self.n_grid**2),
            "passes_pm_emulator_gate": bool(loo < 2.0),
        }
