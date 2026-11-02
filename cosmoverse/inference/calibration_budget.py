"""
Calibration-budget curve for target-family domain adaptation
(`CalibrationBudgetAnalyzer`).

The source flow is trained once on CAMELS-DMO + BCEmu. For each target
calibration budget N_cal, a fresh copy of the source flow is fine-tuned on the
first N_cal target simulations and evaluated on a disjoint target test set.
The result is the minimum N_cal that restores the adaptation gate.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from cosmoverse.acceleration import device_summary
from cosmoverse.inference.domain_adaptation import (
    _target_response_function,
)
from cosmoverse.inference.pm_emulator_sbi import PMEmulatorSBIPipeline
from cosmoverse.validation.transfer import (
    CrossSimulationTransferEvaluator,
    DatasetPartitionProtocol,
)


class CalibrationBudgetAnalyzer:
    """Measures coverage, bias and TARP-ICE as a function of N_cal."""

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        source_pipeline: PMEmulatorSBIPipeline | None = None,
        family: str = "CAMELS-SIMBA",
        data_dir: str | None = None,
        budgets: tuple[int, ...] = (25, 50, 100, 200, 400),
        n_test: int = 100,
        n_records: int = 1200,
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
        if not budgets:
            raise ValueError("budgets must not be empty")
        if min(budgets) < 5 or n_test < 5:
            raise ValueError("budgets >= 5 and n_test >= 5 are required")

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
        self.data_dir = data_dir
        self.budgets = tuple(sorted(int(value) for value in budgets))
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
        rng = np.random.default_rng(self.seed + 41)
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
        """Runs the source training and the N_cal sweep."""
        from cosmoverse.datasets.camels_spectra import (
            CAMELSHydroSpectraConnector,
        )

        device = device_summary()
        self._update_dashboard(
            "source_training",
            {
                "progress": 0.05,
                "device": device,
                "note": "Training the source PM + MLP flow for the calibration-budget sweep.",
            },
        )

        self.source_pipeline.fit()
        source_calibration = self.source_pipeline.run_calibration(
            n_sbc_trials=100,
            n_tarp_trials=100,
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
        max_budget = max(self.budgets)
        calibration_pool = target_candidates[:max_budget]
        test_records = target_candidates[
            max_budget : max_budget + self.n_test
        ]
        leakage = DatasetPartitionProtocol.verify_zero_leakage(
            source_train, test_records
        )
        calibration_ids = {
            int(record["ic_seed"]) for record in calibration_pool
        }
        test_ids = {int(record["ic_seed"]) for record in test_records}
        calibration_test_disjoint = calibration_ids.isdisjoint(test_ids)

        source_records = source_test[: self.n_test]
        source_thetas = np.array(
            [record["theta"] for record in source_records], dtype=np.float64
        )
        calibration_thetas = np.array(
            [record["theta"] for record in calibration_pool],
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
        connector = CAMELSHydroSpectraConnector(data_dir=self.data_dir)
        calibration_responses = [
            _target_response_function(
                connector,
                self.family,
                theta,
                int(record["ic_seed"]),
                self.source_pipeline.z_eff,
            )
            for theta, record in zip(
                calibration_thetas, calibration_pool, strict=True
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
                connector,
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
        data_sources = sorted({source for _, source in test_responses})

        original = copy.deepcopy(self.source_pipeline)
        evaluator = CrossSimulationTransferEvaluator(
            param_bounds=self.param_bounds,
            source_family_name="CAMELS-DMO+BCEmu",
            target_family_name=self.family,
            seed=self.seed,
        )
        before = evaluator.evaluate_transfer(
            infer_fn=lambda x, n: original.infer(
                x, n_posterior_samples=n
            ).samples,
            source_test_thetas=source_thetas,
            source_test_obs=source_obs,
            target_test_thetas=test_thetas,
            target_test_obs=test_obs,
            n_posterior_samples=self.n_posterior_samples,
        )
        before_post = before["post_transfer"]

        curve: list[dict[str, Any]] = []
        best_loss: list[float] = []
        for index, budget in enumerate(self.budgets):
            candidate = copy.deepcopy(original)
            fine_tune_result = candidate.sbi.fine_tune_flow(
                calibration_thetas[:budget],
                calibration_obs[:budget],
                n_epochs=self.fine_tune_epochs,
                learning_rate=self.fine_tune_learning_rate,
                seed=self.seed,
            )
            after = evaluator.evaluate_transfer(
                infer_fn=lambda x, n, pipe=candidate: pipe.infer(
                    x, n_posterior_samples=n
                ).samples,
                source_test_thetas=source_thetas,
                source_test_obs=source_obs,
                target_test_thetas=test_thetas,
                target_test_obs=test_obs,
                n_posterior_samples=self.n_posterior_samples,
            )
            post = after["post_transfer"]
            degradation = after["degradation"]
            passes = bool(
                post["median_bias_in_sigma"] < 1.00
                and post["coverage_68"] >= 0.55
                and post["coverage_95"] >= 0.85
                and degradation["delta_tarp_ice"] <= 0.15
                and post["median_bias_in_sigma"]
                < before_post["median_bias_in_sigma"]
            )
            curve.append(
                {
                    "n_cal": int(budget),
                    "coverage_68": float(post["coverage_68"]),
                    "coverage_95": float(post["coverage_95"]),
                    "median_bias_in_sigma": float(
                        post["median_bias_in_sigma"]
                    ),
                    "tarp_ice": float(post["tarp_ice"]),
                    "delta_tarp_ice": float(
                        degradation["delta_tarp_ice"]
                    ),
                    "passes_gate": passes,
                }
            )
            if passes and not best_loss:
                best_loss = list(fine_tune_result["loss_history"])
            self._update_dashboard(
                f"calibration_budget_{budget}",
                {
                    "progress": 0.2 + 0.7 * (index + 1) / len(self.budgets),
                    "device": device,
                    "coverage": [
                        {
                            "label": "Source",
                            "cov68": source_calibration["tarp"][
                                "coverage_68"
                            ],
                            "cov95": source_calibration["tarp"][
                                "coverage_95"
                            ],
                        },
                        {
                            "label": f"N_cal={budget}",
                            "cov68": post["coverage_68"],
                            "cov95": post["coverage_95"],
                        },
                    ],
                    "metrics": [
                        {
                            "label": f"N_cal={budget}",
                            "median_bias": post["median_bias_in_sigma"],
                            "delta_ice": degradation["delta_tarp_ice"],
                            "gate": "PASS" if passes else "FAIL",
                        }
                    ],
                    "budget_curve": curve,
                    "note": f"Calibration budget N_cal={budget}.",
                },
            )

        passing_budgets = [
            entry["n_cal"] for entry in curve if entry["passes_gate"]
        ]
        min_calibration = min(passing_budgets) if passing_budgets else None
        passes_budget_gate = bool(
            leakage["is_leakage_free"]
            and calibration_test_disjoint
            and source_calibration["passes_joint_sbi_gate"]
            and min_calibration is not None
        )

        self._update_dashboard(
            "done",
            {
                "progress": 1.0,
                "device": device,
                "coverage": [
                    {
                        "label": "Before adaptation",
                        "cov68": before_post["coverage_68"],
                        "cov95": before_post["coverage_95"],
                    },
                    {
                        "label": "Best adapted",
                        "cov68": max(entry["coverage_68"] for entry in curve),
                        "cov95": max(entry["coverage_95"] for entry in curve),
                    },
                ],
                "metrics": [
                    {
                        "label": f"N_cal={entry['n_cal']}",
                        "median_bias": entry["median_bias_in_sigma"],
                        "delta_ice": entry["delta_tarp_ice"],
                        "gate": "PASS" if entry["passes_gate"] else "FAIL",
                    }
                    for entry in curve
                ],
                "fine_tune_loss_history": best_loss,
                "budget_curve": curve,
                "note": f"Minimum calibration budget for the gate: {min_calibration}.",
            },
        )

        return {
            "family": self.family,
            "data_source": data_sources,
            "budgets": list(self.budgets),
            "curve": curve,
            "before_adaptation": before_post,
            "min_calibration_for_gate": min_calibration,
            "passes_budget_gate": passes_budget_gate,
            "source_calibration": source_calibration,
            "leakage_check": leakage,
            "calibration_test_disjoint": calibration_test_disjoint,
            "n_test": self.n_test,
            "fine_tune_epochs": self.fine_tune_epochs,
        }
