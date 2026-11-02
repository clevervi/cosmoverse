"""
Cross-Simulation Family Transfer with the trained conditional flow
(`FlowTransferPipeline`).

The source family is CAMELS-DMO + BCEmu. The target family applies the
IllustrisTNG-like hydrodynamical response on top of the source P(k). Source and
target test sets are partitioned with the mandatory leakage-free protocol, and
the Result 4 degradation (coverage and bias before/after transfer) is reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cosmoverse.inference.pm_emulator_sbi import PMEmulatorSBIPipeline
from cosmoverse.validation.transfer import (
    CrossSimulationTransferEvaluator,
    DatasetPartitionProtocol,
)


class FlowTransferPipeline:
    """
    Trains the multi-probe flow on the source family and evaluates the
    coverage/bias degradation on a disjoint target family.
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        source_pipeline: Any | None = None,
        mismatch_strength: float = 1.0,
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
        if mismatch_strength < 0.0:
            raise ValueError("mismatch_strength must be non-negative")
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
        if set(self.source_pipeline.param_bounds) != set(self.param_bounds):
            raise ValueError("source_pipeline parameter space does not match")

        self.mismatch_strength = float(mismatch_strength)
        self.n_records = int(n_records)
        self.n_test = int(n_test)
        self.n_posterior_samples = int(n_posterior_samples)
        self.seed = int(seed)

    def _make_records(self) -> list[dict[str, Any]]:
        """Builds paired source/target records with shared IC seeds and cosmologies."""
        rng = np.random.default_rng(self.seed + 11)
        thetas = []
        for _ in range(self.n_records):
            theta = np.array(
                [
                    rng.uniform(*self.param_bounds[name])
                    for name in self.param_bounds
                ],
                dtype=np.float64,
            )
            thetas.append(theta)

        records: list[dict[str, Any]] = []
        for family in ("CAMELS-DMO", "CAMELS-IllustrisTNG"):
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
        """Runs the source training, leakage check, and transfer evaluation."""
        self.source_pipeline.fit()
        source_calibration = self.source_pipeline.run_calibration(
            n_sbc_trials=100,
            n_tarp_trials=100,
        )

        records = self._make_records()
        source_train, source_test, target_test = (
            DatasetPartitionProtocol.partition_by_simulation_family(
                records,
                source_family="CAMELS-DMO",
                target_family="CAMELS-IllustrisTNG",
                seed=self.seed,
            )
        )
        leakage = DatasetPartitionProtocol.verify_zero_leakage(
            source_train, target_test
        )

        source_records = source_test[: self.n_test]
        target_records = target_test[: self.n_test]
        source_thetas = np.array(
            [record["theta"] for record in source_records], dtype=np.float64
        )
        target_thetas = np.array(
            [record["theta"] for record in target_records], dtype=np.float64
        )

        source_obs = np.array(
            [
                self.source_pipeline.forward_model(theta)
                for theta in source_thetas
            ]
        )
        target_obs = np.array(
            [
                self.source_pipeline.model_data_vector(
                    theta,
                    add_noise=True,
                    family="target",
                    mismatch_strength=self.mismatch_strength,
                )
                for theta in target_thetas
            ]
        )

        evaluator = CrossSimulationTransferEvaluator(
            param_bounds=self.param_bounds,
            source_family_name="CAMELS-DMO+BCEmu",
            target_family_name="CAMELS-IllustrisTNG",
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

        post = result["post_transfer"]
        result.update(
            {
                "mismatch_strength": self.mismatch_strength,
                "n_source_train_records": len(source_train),
                "n_source_test_records": len(source_test),
                "n_target_test_records": len(target_test),
                "n_source_test_used": len(source_thetas),
                "n_target_test_used": len(target_thetas),
                "source_training_simulations": (
                    self.source_pipeline.n_training_simulations
                ),
                "source_calibration": source_calibration,
                "leakage_check": leakage,
                "transfer_tolerance": {
                    "median_bias_max": 1.10,
                    "coverage_95_min": 0.70,
                    "delta_tarp_ice_max": 0.20,
                },
                "passes_transfer_gate": bool(
                    leakage["is_leakage_free"]
                    and source_calibration["passes_joint_sbi_gate"]
                    and post["median_bias_in_sigma"] < 1.10
                    and post["coverage_95"] >= 0.70
                    and result["degradation"]["delta_tarp_ice"] <= 0.20
                ),
            }
        )
        return result
