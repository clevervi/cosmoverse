"""
Level 3 Multi-Probe Simulation-Based Inference: light-cone w(theta) + RSD
multipoles P_0, P_2, P_4 (`MultiProbeSBIPipeline`).

Combining the angular clustering shape with the redshift-space quadrupole and
hexadecapole breaks the sigma8-b1 degeneracy, allowing a four-parameter
inference (Omega_m, sigma8, b1, A_AGN) with the same SBC/TARP blocking gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cosmoverse.baryons.feedback import BCEmuModel
from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import BaryonParameters, CosmologicalParameters
from cosmoverse.inference.light_cone_sbi import compute_light_cone_w_theta
from cosmoverse.inference.sbi import SBIEngine
from cosmoverse.observations.forward_model import RedshiftSpaceDistortionModel
from cosmoverse.validation.sbc import SimulationBasedCalibration
from cosmoverse.validation.tarp import TARPCalibration


def compute_rsd_multipoles(
    omega_m: float,
    sigma8: float,
    b1: float,
    a_agn: float,
    z_eff: float,
    k: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Computes the Kaiser + FoG redshift-space multipoles P_0, P_2, P_4
    for a BCEmu-suppressed matter power spectrum.
    """
    cosmology = Cosmology(
        CosmologicalParameters(omega_m=float(omega_m), sigma8=float(sigma8))
    )
    pk_m = np.asarray(
        cosmology.linear_matter_power_spectrum(k, z=z_eff),
        dtype=np.float64,
    )
    suppression = BCEmuModel(
        BaryonParameters(mode="BCEmu3", A_AGN=float(a_agn))
    ).suppression_factor(k, z=z_eff)
    growth_rate = float(cosmology.growth_rate(z_eff))
    model = RedshiftSpaceDistortionModel(
        b1=float(b1),
        growth_rate_f=growth_rate,
        sigma_fog=3.5,
    )
    return model.compute_multipoles(k, pk_m * suppression)


class MultiProbeSBIPipeline:
    """
    Four-parameter SBI on the joint data vector
    [w(theta) (12 bins), P_0 (8), P_2 (8), P_4 (8)].
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        z_eff: float = 0.8,
        theta_min_deg: float = 1.0,
        theta_max_deg: float = 20.0,
        n_theta_bins: int = 12,
        k_min: float = 0.03,
        k_max: float = 0.5,
        n_k_bins: int = 8,
        noise_fraction: float = 0.05,
        noise_floor: float = 0.005,
        rsd_noise_floor_fraction: float = 0.01,
        n_training_simulations: int = 400,
        covariance_inflation: float = 1.3,
        seed: int = 42,
    ) -> None:
        self.param_bounds = param_bounds or {
            "omega_m": (0.24, 0.40),
            "sigma8": (0.72, 0.90),
            "b1": (1.2, 2.2),
            "A_AGN": (0.5, 2.0),
        }
        if set(self.param_bounds) != {
            "omega_m",
            "sigma8",
            "b1",
            "A_AGN",
        }:
            raise ValueError(
                "param_bounds must contain exactly omega_m, sigma8, b1 and A_AGN"
            )
        if theta_min_deg <= 0.0 or theta_max_deg <= theta_min_deg:
            raise ValueError("theta bounds must satisfy 0 < theta_min < theta_max")
        if n_theta_bins < 2 or n_k_bins < 2:
            raise ValueError("n_theta_bins and n_k_bins must be >= 2")
        if k_min <= 0.0 or k_max <= k_min:
            raise ValueError("k bounds must satisfy 0 < k_min < k_max")
        if noise_fraction < 0.0 or noise_floor <= 0.0:
            raise ValueError("noise parameters must be non-negative")
        if n_training_simulations < 10:
            raise ValueError("n_training_simulations must be >= 10")

        self.z_eff = float(z_eff)
        self.theta_deg = np.linspace(
            float(theta_min_deg), float(theta_max_deg), int(n_theta_bins)
        )
        self.ell = np.linspace(1.0, 8000.0, 1024)
        self.k_rsd = np.linspace(float(k_min), float(k_max), int(n_k_bins))
        self.noise_fraction = float(noise_fraction)
        self.noise_floor = float(noise_floor)
        self.rsd_noise_floor_fraction = float(rsd_noise_floor_fraction)
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

    def model_data_vector(
        self,
        theta: np.ndarray,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        Evaluates the joint noiseless (or noisy) data vector for
        theta = (Omega_m, sigma8, b1, A_AGN).
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.shape != (4,):
            raise ValueError(
                "theta must have shape (4,) = (omega_m, sigma8, b1, A_AGN)"
            )
        omega_m, sigma8, b1, a_agn = (float(value) for value in theta_arr)

        w_theta = compute_light_cone_w_theta(
            omega_m,
            sigma8,
            b1,
            a_agn,
            self.z_eff,
            self.theta_deg,
            self.ell,
        )
        rsd = compute_rsd_multipoles(
            omega_m,
            sigma8,
            b1,
            a_agn,
            self.z_eff,
            self.k_rsd,
        )
        p0 = np.asarray(rsd["P0"], dtype=np.float64)
        p2 = np.asarray(rsd["P2"], dtype=np.float64)
        p4 = np.asarray(rsd["P4"], dtype=np.float64)
        data = np.concatenate([w_theta, p0, p2, p4])

        if add_noise:
            p0_floor = self.rsd_noise_floor_fraction * np.maximum(
                np.abs(p0), 1e-6
            )
            sigma = np.concatenate(
                [
                    self.noise_fraction
                    * np.maximum(np.abs(w_theta), self.noise_floor),
                    self.noise_fraction * np.maximum(np.abs(p0), p0_floor),
                    self.noise_fraction * np.maximum(np.abs(p2), p0_floor),
                    self.noise_fraction * np.maximum(np.abs(p4), p0_floor),
                ]
            )
            data = data + self.rng.normal(0.0, sigma)
        return data

    def forward_model(self, theta: np.ndarray) -> np.ndarray:
        """Noisy multi-probe simulator used by SBI training and SBC/TARP."""
        return self.model_data_vector(theta, add_noise=True)

    def fiducial_data_vector(self) -> np.ndarray:
        """Noiseless fiducial data vector at (0.315, 0.811, 1.65, 1.0)."""
        return self.model_data_vector(
            np.array([0.315, 0.811, 1.65, 1.0]), add_noise=False
        )

    def fit(
        self, n_simulations: int | None = None, verbose: bool = False
    ) -> "MultiProbeSBIPipeline":
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
        data_vector: np.ndarray,
        n_posterior_samples: int = 150,
        seed: int | None = None,
    ):
        """Returns the posterior summary for an observed multi-probe data vector."""
        if not self.is_fitted:
            self.fit()
        return self.sbi.infer(
            data_vector,
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
            "theta_deg": self.theta_deg,
            "k_rsd": self.k_rsd,
            "n_training_simulations": self.n_training_simulations,
            "sbc": sbc_metrics,
            "tarp": tarp_result,
            "passes_sbc_gate": bool(passes_sbc),
            "passes_tarp_gate": bool(tarp_result["passes_acceptance_gate"]),
            "passes_joint_sbi_gate": bool(
                passes_sbc and tarp_result["passes_acceptance_gate"]
            ),
        }
