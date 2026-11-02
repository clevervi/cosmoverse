"""
Data Partitioning Protocol (Section 7) and Cross-Simulation Transfer Evaluator (Resultado 4 — Section 4.4).

Critical Rule (Master Document Section 7):
  DO NOT use naive random 80/20 splits. Simulations sharing initial conditions (ic_seed) or
  nearby cosmologies (cosmo_id) produce deceptively optimistic test metrics due to cosmic
  variance phase leakage rather than true generalization.

Mandatory Protocol:
  1. Partition strictly by initial conditions (ic_seed) AND cosmology (cosmo_id).
  2. Partition by simulation family (family_id, e.g. 'CAMELS-DMO' -> 'CAMELS-IllustrisTNG').
  3. Quantify coverage and bias BEFORE and AFTER cross-simulation family transfer.
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from cosmoverse.validation.tarp import TARPCalibration


class DatasetPartitionProtocol:
    """
    Enforces leakage-free partitioning of cosmological simulation suites by
    initial condition seed (ic_seed), cosmology ID (cosmo_id), and simulation family (family_id).
    """

    @staticmethod
    def partition_by_ic_and_cosmology(
        records: List[Dict[str, Any]],
        train_fraction: float = 0.75,
        split_mode: str = "disjoint_ic_and_cosmo",
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Partitions simulation metadata records into train and test subsets with strictly
        disjoint initial condition seeds (ic_seed) and cosmology IDs (cosmo_id).

        Raises:
            ValueError: if split_mode == 'random_80_20', which violates Section 7 protocol.
        """
        if split_mode.lower() in ("random", "random_80_20"):
            raise ValueError(
                "Naive random 80/20 splits are forbidden by COSMOVERSE Master Document Section 7 "
                "(causes cosmic variance phase leakage across shared ic_seed / cosmo_id)."
            )

        rng = np.random.default_rng(seed)
        unique_cosmos = sorted({str(r["cosmo_id"]) for r in records})
        unique_ics = sorted({int(r["ic_seed"]) for r in records})

        rng.shuffle(unique_cosmos)
        rng.shuffle(unique_ics)

        n_train_cosmo = max(1, int(len(unique_cosmos) * train_fraction))
        n_train_ic = max(1, int(len(unique_ics) * train_fraction))

        train_cosmos = set(unique_cosmos[:n_train_cosmo])
        test_cosmos = set(unique_cosmos[n_train_cosmo:])
        train_ics = set(unique_ics[:n_train_ic])
        test_ics = set(unique_ics[n_train_ic:])

        train_set = [
            r for r in records
            if str(r["cosmo_id"]) in train_cosmos and int(r["ic_seed"]) in train_ics
        ]
        test_set = [
            r for r in records
            if str(r["cosmo_id"]) in test_cosmos and int(r["ic_seed"]) in test_ics
        ]

        return train_set, test_set

    @staticmethod
    def verify_zero_leakage(
        train_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verifies that zero initial condition seeds (ic_seed) and zero cosmology IDs (cosmo_id)
        overlap between training and testing partitions.
        """
        train_ics = {int(r["ic_seed"]) for r in train_records}
        test_ics = {int(r["ic_seed"]) for r in test_records}
        shared_ics = train_ics.intersection(test_ics)

        train_cosmos = {str(r["cosmo_id"]) for r in train_records}
        test_cosmos = {str(r["cosmo_id"]) for r in test_records}
        shared_cosmos = train_cosmos.intersection(test_cosmos)

        is_valid = (len(shared_ics) == 0) and (len(shared_cosmos) == 0)
        return {
            "is_leakage_free": is_valid,
            "shared_ic_seeds": len(shared_ics),
            "shared_cosmo_ids": len(shared_cosmos),
            "n_train": len(train_records),
            "n_test": len(test_records),
        }

    @staticmethod
    def partition_by_simulation_family(
        records: List[Dict[str, Any]],
        source_family: str,
        target_family: str,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits records for Resultado 4 (Cross-Simulation Transfer):
          - source_train: records from source_family with train ic_seeds & cosmo_ids
          - source_test: records from source_family with disjoint test ic_seeds & cosmo_ids
          - target_transfer_test: records from target_family with the disjoint test ic_seeds & cosmo_ids
        """
        source_recs = [r for r in records if str(r.get("family_id")) == source_family]
        target_recs = [r for r in records if str(r.get("family_id")) == target_family]

        source_train, source_test = DatasetPartitionProtocol.partition_by_ic_and_cosmology(
            source_recs, train_fraction=0.70, seed=seed
        )
        train_ics = {int(r["ic_seed"]) for r in source_train}
        train_cosmos = {str(r["cosmo_id"]) for r in source_train}

        target_test = [
            r for r in target_recs
            if int(r["ic_seed"]) not in train_ics and str(r["cosmo_id"]) not in train_cosmos
        ]
        return source_train, source_test, target_test


class CrossSimulationTransferEvaluator:
    """
    Evaluates Resultado 4 (Central Contribution of COSMOVERSE — Master Document Section 4.4):
    Quantifies coverage and bias before and after transferring an inference pipeline
    trained on Simulation Family A (e.g. N-body + BCEmu) to Simulation Family B
    (e.g. Hydrodynamic IllustrisTNG / SIMBA).
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        source_family_name: str = "N-body_DMO+BCEmu",
        target_family_name: str = "Hydro_IllustrisTNG",
        seed: int = 42,
    ):
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.source_family_name = source_family_name
        self.target_family_name = target_family_name
        self.seed = seed
        self.tarp = TARPCalibration(param_bounds=param_bounds, seed=seed)

    def _compute_metrics_on_suite(
        self,
        true_thetas: np.ndarray,
        posterior_draws_all: np.ndarray,
    ) -> Dict[str, Any]:
        """Computes systematic parameter bias (in units of posterior sigma) and TARP coverage."""
        tarp_res = self.tarp.evaluate(true_thetas, posterior_draws_all)
        means = np.mean(posterior_draws_all, axis=1)  # (N_trials, D)
        stds = np.maximum(np.std(posterior_draws_all, axis=1), 1e-3)
        # Systematic bias per parameter: |< (mu_i - theta*_i) / sigma_i >|
        bias_sigma = np.abs(np.mean((means - true_thetas) / stds, axis=0))
        abs_z = np.abs((means - true_thetas) / stds)

        return {
            "mean_bias_in_sigma": float(np.mean(bias_sigma)),
            "median_bias_in_sigma": float(np.median(abs_z)),
            "p95_bias_in_sigma": float(np.percentile(abs_z, 95)),
            "fraction_within_1sigma": float(np.mean(abs_z < 1.0)),
            "per_param_bias_in_sigma": {
                name: float(bias_sigma[j]) for j, name in enumerate(self.param_names)
            },
            "coverage_68": float(tarp_res["coverage_68"]),
            "coverage_95": float(tarp_res["coverage_95"]),
            "tarp_ice": float(tarp_res["integrated_calibration_error"]),
            "passes_acceptance_gate": bool(tarp_res["passes_acceptance_gate"]),
        }

    def evaluate_transfer(
        self,
        infer_fn: Callable[[np.ndarray, int], np.ndarray],
        source_test_thetas: np.ndarray,
        source_test_obs: np.ndarray,
        target_test_thetas: np.ndarray,
        target_test_obs: np.ndarray,
        n_posterior_samples: int = 200,
    ) -> Dict[str, Any]:
        """
        Evaluates posterior calibration BEFORE transfer (on source_family test set)
        and AFTER transfer (on target_family test set), reporting explicit degradation.
        """
        n_src = len(source_test_thetas)
        src_post = np.array([
            infer_fn(source_test_obs[i], n_posterior_samples) for i in range(n_src)
        ])
        pre_metrics = self._compute_metrics_on_suite(source_test_thetas, src_post)

        n_tgt = len(target_test_thetas)
        tgt_post = np.array([
            infer_fn(target_test_obs[i], n_posterior_samples) for i in range(n_tgt)
        ])
        post_metrics = self._compute_metrics_on_suite(target_test_thetas, tgt_post)

        delta_cov_68 = post_metrics["coverage_68"] - pre_metrics["coverage_68"]
        delta_cov_95 = post_metrics["coverage_95"] - pre_metrics["coverage_95"]
        delta_bias_sigma = post_metrics["mean_bias_in_sigma"] - pre_metrics["mean_bias_in_sigma"]

        return {
            "transfer_pair": f"{self.source_family_name} -> {self.target_family_name}",
            "pre_transfer": pre_metrics,
            "post_transfer": post_metrics,
            "degradation": {
                "delta_coverage_68": float(delta_cov_68),
                "delta_coverage_95": float(delta_cov_95),
                "abs_coverage_68_drop": float(abs(post_metrics["coverage_68"] - 0.6827) - abs(pre_metrics["coverage_68"] - 0.6827)),
                "delta_bias_in_sigma": float(delta_bias_sigma),
                "delta_tarp_ice": float(post_metrics["tarp_ice"] - pre_metrics["tarp_ice"]),
            },
            "is_transfer_bounded": bool(post_metrics["mean_bias_in_sigma"] < 1.0),
            "is_transfer_bounded_robust": bool(
                post_metrics["median_bias_in_sigma"] < 1.0
            ),
        }
