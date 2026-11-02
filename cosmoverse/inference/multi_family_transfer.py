"""
Multi-family cross-simulation transfer with the trained flow
(`MultiFamilyTransferPipeline`).

Source: CAMELS-DMO + BCEmu. Targets: CAMELS-IllustrisTNG, CAMELS-SIMBA and
CAMELS-Astrid. By default the calibrated CAMELS response curves are used and
the result is labeled `calibrated_camels_response`. If a `data_dir` with real
`{sim_id}_Pk.txt` tables is provided, the connector loads those tables and the
result is labeled `disk_table`; the tables provide the response shape, not a
full real-CAMELS parameter inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import CosmologicalParameters
from cosmoverse.datasets.camels_spectra import CAMELSHydroSpectraConnector
from cosmoverse.inference.pm_emulator_sbi import PMEmulatorSBIPipeline
from cosmoverse.validation.transfer import (
    CrossSimulationTransferEvaluator,
    DatasetPartitionProtocol,
)


class MultiFamilyTransferPipeline:
    """Runs the trained-flow transfer against TNG, SIMBA and Astrid."""

    FAMILIES = (
        "CAMELS-IllustrisTNG",
        "CAMELS-SIMBA",
        "CAMELS-Astrid",
    )

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        source_pipeline: PMEmulatorSBIPipeline | None = None,
        data_dir: str | None = None,
        n_records: int = 600,
        n_test: int = 100,
        n_posterior_samples: int = 150,
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
        if n_records < 20 or n_test < 5:
            raise ValueError("n_records >= 20 and n_test >= 5 are required")

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
        self.n_records = int(n_records)
        self.n_test = int(n_test)
        self.n_posterior_samples = int(n_posterior_samples)
        self.seed = int(seed)

    def _make_records(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed + 21)
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
        for family in ("CAMELS-DMO",) + self.FAMILIES:
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

    def _response_function(
        self,
        family: str,
        theta: np.ndarray,
        index: int,
    ) -> tuple[Any, str]:
        record = {
            "sim_id": f"{family}-LH_{index}",
            "Omega_m": float(theta[0]),
            "sigma_8": float(theta[1]),
            "A_SN1": 1.0,
            "A_AGN1": float(theta[3]),
        }
        spec = self.connector.load_or_compute_spectrum(
            record,
            suite_override=family,
            z=self.source_pipeline.z_eff,
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
                cosmology.linear_matter_power_spectrum(
                    k_spec, z=self.source_pipeline.z_eff
                ),
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

    def run(self) -> dict[str, Any]:
        """Runs the source calibration and the three target-family transfers."""
        self.source_pipeline.fit()
        source_calibration = self.source_pipeline.run_calibration(
            n_sbc_trials=100,
            n_tarp_trials=100,
        )

        records = self._make_records()
        source_train, source_test, target_test_reference = (
            DatasetPartitionProtocol.partition_by_simulation_family(
                records,
                source_family="CAMELS-DMO",
                target_family="CAMELS-IllustrisTNG",
                seed=self.seed,
            )
        )
        leakage = DatasetPartitionProtocol.verify_zero_leakage(
            source_train, target_test_reference
        )

        source_records = source_test[: self.n_test]
        reference_records = target_test_reference[: self.n_test]
        source_thetas = np.array(
            [record["theta"] for record in source_records], dtype=np.float64
        )
        target_indices = [int(record["ic_seed"]) for record in reference_records]
        target_thetas = np.array(
            [record["theta"] for record in reference_records], dtype=np.float64
        )

        source_obs = np.array(
            [
                self.source_pipeline.forward_model(theta)
                for theta in source_thetas
            ]
        )

        family_results: dict[str, Any] = {}
        all_passed = True
        pre_metrics: dict[str, Any] | None = None

        for family in self.FAMILIES:
            responses = [
                self._response_function(
                    family, theta, index
                )
                for theta, index in zip(target_thetas, target_indices, strict=True)
            ]
            target_obs = np.array(
                [
                    self.source_pipeline.model_data_vector_with_response(
                        theta, response_fn, add_noise=True
                    )
                    for theta, (response_fn, _) in zip(
                        target_thetas, responses, strict=True
                    )
                ]
            )
            data_sources = {source for _, source in responses}

            evaluator = CrossSimulationTransferEvaluator(
                param_bounds=self.param_bounds,
                source_family_name="CAMELS-DMO+BCEmu",
                target_family_name=family,
                seed=self.seed,
            )
            result = evaluator.evaluate_transfer(
                infer_fn=lambda x, n: self.source_pipeline.infer(
                    x, n_posterior_samples=n
                ).samples,
                source_test_thetas=source_thetas,
                source_test_obs=source_obs,
                target_test_thetas=target_thetas,
                target_test_obs=target_obs,
                n_posterior_samples=self.n_posterior_samples,
            )
            if pre_metrics is None:
                pre_metrics = result["pre_transfer"]
            post = result["post_transfer"]
            degradation = result["degradation"]
            family_gate = bool(
                leakage["is_leakage_free"]
                and source_calibration["passes_joint_sbi_gate"]
                and post["median_bias_in_sigma"] < 1.10
                and post["coverage_95"] >= 0.70
                and degradation["delta_tarp_ice"] <= 0.20
            )
            all_passed = all_passed and family_gate
            family_results[family] = {
                "data_source": sorted(data_sources),
                "post_transfer": post,
                "degradation": degradation,
                "passes_family_gate": family_gate,
            }

        methodology_ok = bool(
            leakage["is_leakage_free"]
            and source_calibration["passes_joint_sbi_gate"]
            and len(family_results) == len(self.FAMILIES)
            and all(
                np.isfinite(res["post_transfer"]["median_bias_in_sigma"])
                for res in family_results.values()
            )
        )
        return {
            "families": family_results,
            "source_calibration": source_calibration,
            "leakage_check": leakage,
            "pre_transfer": pre_metrics,
            "n_source_test_used": len(source_thetas),
            "n_target_test_used": len(target_thetas),
            "source_training_simulations": (
                self.source_pipeline.n_training_simulations
            ),
            "data_dir": self.data_dir,
            "passes_degradation_bound_all": bool(all_passed),
            "passes_multi_family_gate": methodology_ok,
        }
