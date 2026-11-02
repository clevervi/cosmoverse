"""
Target-family domain adaptation for the trained conditional flow
(`DomainAdaptationPipeline`).

The source flow is trained on CAMELS-DMO + BCEmu. A small calibration set from
the target hydrodynamical family is used to fine-tune the flow, and the
before/after transfer metrics are measured on a disjoint target test set.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from cosmoverse.acceleration import device_summary
from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import CosmologicalParameters
from cosmoverse.datasets.camels_spectra import CAMELSHydroSpectraConnector
from cosmoverse.inference.pm_emulator_sbi import PMEmulatorSBIPipeline
from cosmoverse.validation.transfer import (
    CrossSimulationTransferEvaluator,
    DatasetPartitionProtocol,
)


def _target_response_function(
    connector: CAMELSHydroSpectraConnector,
    family: str,
    theta: np.ndarray,
    index: int,
    z_eff: float,
):
    record = {
        "sim_id": f"{family}-LH_{index}",
        "Omega_m": float(theta[0]),
        "sigma_8": float(theta[1]),
        "A_SN1": 1.0,
        "A_AGN1": float(theta[3]),
    }
    spec = connector.load_or_compute_spectrum(
        record, suite_override=family, z=z_eff
    )
    k_spec = np.asarray(spec["k"], dtype=np.float64)
    if "baryonic_response" in spec:
        response = np.asarray(spec["baryonic_response"], dtype=np.float64)
    else:
        cosmology = Cosmology(
            CosmologicalParameters(
                omega_m=float(theta[0]), sigma8=float(theta[1])
            )
        )
        pk_dmo = np.asarray(
            cosmology.linear_matter_power_spectrum(k_spec, z=z_eff),
            dtype=np.float64,
        )
        response = np.asarray(spec["pk_hydro"], dtype=np.float64) / np.maximum(
            pk_dmo, 1e-30
        )

    def response_fn(k: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(k, dtype=np.float64),
            k_spec,
            response,
            left=float(response[0]),
            right=1.0,
        )

    return response_fn, str(spec["source"])


class DomainAdaptationPipeline:
    """Fine-tunes the source flow on a target hydrodynamical family."""

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        source_pipeline: PMEmulatorSBIPipeline | None = None,
        family: str = "CAMELS-SIMBA",
        data_dir: str | None = None,
        n_calibration: int = 200,
        n_test: int = 100,
        n_records: int = 800,
        n_posterior_samples: int = 150,
        fine_tune_epochs: int = 200,
        fine_tune_learning_rate: float = 3e-4,
        seed: int = 42,
        dashboard: Any | None = None,
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
        if family not in {
            "CAMELS-IllustrisTNG",
            "CAMELS-SIMBA",
            "CAMELS-Astrid",
        }:
            raise ValueError("unsupported target family")
        if n_calibration < 20 or n_test < 5:
            raise ValueError("n_calibration >= 20 and n_test >= 5 are required")

        self.family = family
        self.source_pipeline = source_pipeline or PMEmulatorSBIPipeline(
            param_bounds=self.param_bounds,
            n_training_simulations=1600,
            density_estimator="flow",
            flow_n_layers=6,
            flow_epochs=800,
            flow_scale_bound=6.0,
            flow_hidden_dim=16,
            seed=seed,
        )
        self.connector = CAMELSHydroSpectraConnector(data_dir=data_dir)
        self.data_dir = data_dir
        self.n_calibration = int(n_calibration)
        self.n_test = int(n_test)
        self.n_records = int(n_records)
        self.n_posterior_samples = int(n_posterior_samples)
        self.fine_tune_epochs = int(fine_tune_epochs)
        self.fine_tune_learning_rate = float(fine_tune_learning_rate)
        self.seed = int(seed)
        self.dashboard = dashboard

    def _update_dashboard(self, stage: str, payload: dict[str, Any]) -> None:
        if self.dashboard is not None:
            self.dashboard.update(stage, payload)

    def _make_records(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed + 31)
        thetas = [
            np.array(
                [
                    rng.uniform(*self.param_bounds[name])
                    for name in self.param_bounds
                ],
                dtype=np.float64,
            )
            for _ in range(self.n_records)
        ]
        records: list[dict[str, Any]] = []
        for family in ("CAMELS-DMO", self.family):
            for index, theta in enumerate(thetas):
                records.append(
                    {
                        "ic_seed": index,
                        "cosmo_id": f"cosmo-{index}",
                        "family_id": family,
                        "theta": theta,
                    }
                )
        return records

    def run(self) -> dict[str, Any]:
        """Runs source training, target calibration, fine-tuning and evaluation."""
        device = device_summary()
        self._update_dashboard(
            "source_training",
            {
                "progress": 0.05,
                "device": device,
                "note": "Training the source PM + MLP flow on CAMELS-DMO + BCEmu.",
            },
        )

        self.source_pipeline.fit()
        source_calibration = self.source_pipeline.run_calibration(
            n_sbc_trials=100,
            n_tarp_trials=100,
        )
        source_metrics = {
            name: {
                "coverage_68": metric["coverage_68"],
                "coverage_95": metric["coverage_95"],
                "p_value": metric["p_value"],
            }
            for name, metric in source_calibration["sbc"].items()
        }
        self._update_dashboard(
            "source_calibration",
            {
                "progress": 0.25,
                "device": device,
                "coverage": [
                    {
                        "label": "Source",
                        "cov68": source_calibration["tarp"]["coverage_68"],
                        "cov95": source_calibration["tarp"]["coverage_95"],
                    }
                ],
                "metrics": [
                    {
                        "label": "Source SBC/TARP",
                        "median_bias": 0.0,
                        "delta_ice": 0.0,
                        "gate": (
                            "PASS"
                            if source_calibration["passes_joint_sbi_gate"]
                            else "FAIL"
                        ),
                    }
                ],
                "note": f"Source SBC/TARP gate: "
                f"{'PASS' if source_calibration['passes_joint_sbi_gate'] else 'FAIL'}.",
                "source_sbc": source_metrics,
            },
        )

        records = self._make_records()
        source_train, source_test, target_candidates = (
            DatasetPartitionProtocol.partition_by_simulation_family(
                records,
                source_family="CAMELS-DMO",
                target_family=self.family,
                seed=self.seed,
            )
        )
        calibration_records = target_candidates[: self.n_calibration]
        test_records = target_candidates[
            self.n_calibration : self.n_calibration + self.n_test
        ]
        leakage = DatasetPartitionProtocol.verify_zero_leakage(
            source_train, test_records
        )
        calibration_ids = {int(record["ic_seed"]) for record in calibration_records}
        test_ids = {int(record["ic_seed"]) for record in test_records}
        calibration_test_disjoint = calibration_ids.isdisjoint(test_ids)

        source_records = source_test[: self.n_test]
        source_thetas = np.array(
            [record["theta"] for record in source_records], dtype=np.float64
        )
        calibration_thetas = np.array(
            [record["theta"] for record in calibration_records],
            dtype=np.float64,
        )
        test_thetas = np.array(
            [record["theta"] for record in test_records], dtype=np.float64
        )
        test_indices = [int(record["ic_seed"]) for record in test_records]

        source_obs = np.array(
            [
                self.source_pipeline.forward_model(theta)
                for theta in source_thetas
            ]
        )
        calibration_responses = [
            _target_response_function(
                self.connector,
                self.family,
                theta,
                int(record["ic_seed"]),
                self.source_pipeline.z_eff,
            )
            for theta, record in zip(
                calibration_thetas, calibration_records, strict=True
            )
        ]
        calibration_obs = np.array(
            [
                self.source_pipeline.model_data_vector_with_response(
                    theta, response_fn, add_noise=True
                )
                for theta, (response_fn, _) in zip(
                    calibration_thetas, calibration_responses, strict=True
                )
            ]
        )
        test_responses = [
            _target_response_function(
                self.connector,
                self.family,
                theta,
                index,
                self.source_pipeline.z_eff,
            )
            for theta, index in zip(test_thetas, test_indices, strict=True)
        ]
        test_obs = np.array(
            [
                self.source_pipeline.model_data_vector_with_response(
                    theta, response_fn, add_noise=True
                )
                for theta, (response_fn, _) in zip(
                    test_thetas, test_responses, strict=True
                )
            ]
        )
        data_sources = sorted({source for _, source in calibration_responses})

        self._update_dashboard(
            "target_calibration",
            {
                "progress": 0.45,
                "device": device,
                "note": f"Generated {len(calibration_thetas)} {self.family} "
                f"calibration simulations [{', '.join(data_sources)}].",
            },
        )

        original_pipeline = copy.deepcopy(self.source_pipeline)
        evaluator = CrossSimulationTransferEvaluator(
            param_bounds=self.param_bounds,
            source_family_name="CAMELS-DMO+BCEmu",
            target_family_name=self.family,
            seed=self.seed,
        )
        before = evaluator.evaluate_transfer(
            infer_fn=lambda x, n: original_pipeline.infer(
                x, n_posterior_samples=n
            ).samples,
            source_test_thetas=source_thetas,
            source_test_obs=source_obs,
            target_test_thetas=test_thetas,
            target_test_obs=test_obs,
            n_posterior_samples=self.n_posterior_samples,
        )

        fine_tune_result = self.source_pipeline.sbi.fine_tune_flow(
            calibration_thetas,
            calibration_obs,
            n_epochs=self.fine_tune_epochs,
            learning_rate=self.fine_tune_learning_rate,
            seed=self.seed,
        )
        self._update_dashboard(
            "fine_tuning",
            {
                "progress": 0.65,
                "device": device,
                "fine_tune_loss_history": fine_tune_result["loss_history"],
                "note": f"Fine-tuning the flow on {self.family} calibration data.",
            },
        )

        after = evaluator.evaluate_transfer(
            infer_fn=lambda x, n: self.source_pipeline.infer(
                x, n_posterior_samples=n
            ).samples,
            source_test_thetas=source_thetas,
            source_test_obs=source_obs,
            target_test_thetas=test_thetas,
            target_test_obs=test_obs,
            n_posterior_samples=self.n_posterior_samples,
        )

        before_post = before["post_transfer"]
        after_post = after["post_transfer"]
        delta_median = (
            after_post["median_bias_in_sigma"]
            - before_post["median_bias_in_sigma"]
        )
        delta_ice = (
            after["degradation"]["delta_tarp_ice"]
            - before["degradation"]["delta_tarp_ice"]
        )
        improved = bool(
            after_post["median_bias_in_sigma"]
            < before_post["median_bias_in_sigma"]
        )
        passes_adaptation = bool(
            leakage["is_leakage_free"]
            and calibration_test_disjoint
            and source_calibration["passes_joint_sbi_gate"]
            and after_post["median_bias_in_sigma"] < 1.10
            and after_post["coverage_95"] >= 0.70
            and after["degradation"]["delta_tarp_ice"] <= 0.20
            and improved
        )

        posterior = self.source_pipeline.infer(
            test_obs[0], n_posterior_samples=400, seed=self.seed
        ).samples
        self._update_dashboard(
            "done",
            {
                "progress": 1.0,
                "device": device,
                "coverage": [
                    {
                        "label": "Source",
                        "cov68": source_calibration["tarp"]["coverage_68"],
                        "cov95": source_calibration["tarp"]["coverage_95"],
                    },
                    {
                        "label": f"{self.family} before",
                        "cov68": before_post["coverage_68"],
                        "cov95": before_post["coverage_95"],
                    },
                    {
                        "label": f"{self.family} adapted",
                        "cov68": after_post["coverage_68"],
                        "cov95": after_post["coverage_95"],
                    },
                ],
                "metrics": [
                    {
                        "label": "Before adaptation",
                        "median_bias": before_post["median_bias_in_sigma"],
                        "delta_ice": before["degradation"]["delta_tarp_ice"],
                        "gate": (
                            "PASS" if before_post["passes_acceptance_gate"] else "FAIL"
                        ),
                    },
                    {
                        "label": "After adaptation",
                        "median_bias": after_post["median_bias_in_sigma"],
                        "delta_ice": after["degradation"]["delta_tarp_ice"],
                        "gate": "PASS" if passes_adaptation else "FAIL",
                    },
                ],
                "fine_tune_loss_history": fine_tune_result["loss_history"],
                "posterior": {
                    "label": "Omega_m vs sigma8 (adapted)",
                    "truth": [float(test_thetas[0][0]), float(test_thetas[0][1])],
                    "samples": posterior[:, :2].tolist(),
                },
                "note": f"Adaptation gate: "
                f"{'PASS' if passes_adaptation else 'FAIL'}.",
            },
        )

        return {
            "family": self.family,
            "data_source": data_sources,
            "leakage_check": leakage,
            "calibration_test_disjoint": calibration_test_disjoint,
            "source_calibration": source_calibration,
            "before_adaptation": before_post,
            "after_adaptation": after_post,
            "delta_median_bias": float(delta_median),
            "delta_tarp_ice_vs_before": float(delta_ice),
            "fine_tune_loss_history": fine_tune_result["loss_history"],
            "fine_tune_epochs": self.fine_tune_epochs,
            "n_calibration": int(self.n_calibration),
            "n_test": int(self.n_test),
            "improved": improved,
            "passes_adaptation_gate": passes_adaptation,
        }
