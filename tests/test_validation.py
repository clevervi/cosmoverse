"""
Unit tests for validation suites: SBC + fixed-parameter checks, TARP (Resultado 3),
Scale-Dependent Misspecification, Leakage-Free Data Partition Protocol (Section 7),
Cross-Simulation Family Transfer (Resultado 4), and CAMELS Gen-2 benchmarks.
"""

import numpy as np
import pytest

from cosmoverse.validation.benchmarks import (
    camels_gen2_catalog_metadata,
    camels_latin_hypercube_benchmark,
)
from cosmoverse.validation.coverage import EmpiricalCoverage
from cosmoverse.validation.misspecification import ScaleDependentMisspecification
from cosmoverse.validation.sbc import SimulationBasedCalibration
from cosmoverse.validation.tarp import TARPCalibration
from cosmoverse.validation.transfer import (
    CrossSimulationTransferEvaluator,
    DatasetPartitionProtocol,
)


def test_camels_latin_hypercube_and_gen2():
    pts = camels_latin_hypercube_benchmark(n_samples=15, seed=42)
    assert len(pts) == 15
    for p in pts:
        assert 0.1 <= p["omega_m"] <= 0.5
        assert 0.6 <= p["sigma8"] <= 1.0
        assert 0.25 <= p["A_AGN"] <= 4.0

    gen2 = camels_gen2_catalog_metadata(n_records_per_family=10, seed=42)
    assert len(gen2) == 20
    assert gen2[0]["n_total_camels_gen2_params"] == 35
    assert gen2[0]["box_size_mpc_h"] == 50.0


def test_empirical_coverage():
    cov = EmpiricalCoverage(param_names=["p1"])
    true_thetas = np.zeros((20, 1))
    posteriors = [np.random.normal(0, 1, size=(200, 1)) for _ in range(20)]
    results = cov.compute_coverage(true_thetas, posteriors, nominal_levels=[0.6827])
    assert 0.0 <= results["p1"][0.6827] <= 1.0


def test_sbc_and_fixed_parameter_check():
    rng = np.random.default_rng(42)
    sbc = SimulationBasedCalibration(
        sample_prior_fn=lambda: np.array([rng.uniform(0.2, 0.4)]),
        simulate_fn=lambda th: th + rng.normal(0, 0.03, size=1),
        infer_fn=lambda x, n: rng.normal(x, 0.03, size=(n, 1)),
        param_names=["test_param"],
    )
    res = sbc.run_calibration(n_trials=30, n_posterior_samples=100)
    assert "test_param" in res
    assert "p_value" in res["test_param"]
    assert "coverage_68" in res["test_param"]

    fixed_res = sbc.run_fixed_parameter_check(np.array([0.3]), n_repeats=15, n_posterior_samples=100)
    assert "test_param" in fixed_res
    assert fixed_res["test_param"]["mean_bias_in_sigma"] < 1.0


def test_tarp_calibration_result3():
    rng = np.random.default_rng(42)
    bounds = {"omega_m": (0.1, 0.5), "sigma8": (0.6, 1.0)}
    tarp = TARPCalibration(param_bounds=bounds, seed=42)

    n_trials = 40
    true_thetas = np.column_stack([
        rng.uniform(0.15, 0.45, size=n_trials),
        rng.uniform(0.65, 0.95, size=n_trials),
    ])
    obs = true_thetas + rng.normal(0, 0.02, size=(n_trials, 2))
    post = obs[:, np.newaxis, :] + rng.normal(0, 0.02, size=(n_trials, 150, 2))

    res = tarp.evaluate(true_thetas, post)
    assert 0.50 <= res["coverage_68"] <= 0.85
    assert 0.80 <= res["coverage_95"] <= 1.00
    assert res["passes_acceptance_gate"] is True


def test_scale_dependent_misspecification():
    rng = np.random.default_rng(42)
    k = np.array([0.1, 0.2, 0.5, 1.0])
    bank = rng.normal(10.0, 0.2, size=(50, 4))
    detector = ScaleDependentMisspecification(k_grid=k, x_training_bank=bank)

    # In-distribution observation
    in_obs = np.array([10.05, 9.95, 10.1, 9.9])
    res_ok = detector.evaluate_scale_cuts(in_obs, k_cuts=[0.2, 1.0])
    assert res_ok["is_misspecified"] is False

    # Small-scale baryonic misspecification at k >= 0.5 h/Mpc
    bad_obs = np.array([10.0, 10.0, 11.5, 12.5])
    res_bad = detector.evaluate_scale_cuts(bad_obs, k_cuts=[0.2, 1.0])
    assert res_bad["is_misspecified"] is True
    assert res_bad["is_small_scale_driven"] is True


def test_partition_protocol_and_cross_simulation_transfer_result4():
    catalog = camels_gen2_catalog_metadata(n_records_per_family=16, seed=42)

    # Naive random 80/20 split is explicitly forbidden by Section 7
    with pytest.raises(ValueError):
        DatasetPartitionProtocol.partition_by_ic_and_cosmology(
            catalog, split_mode="random_80_20"
        )

    src_train, src_test, tgt_test = DatasetPartitionProtocol.partition_by_simulation_family(
        catalog, source_family="CAMELS-DMO", target_family="CAMELS-IllustrisTNG", seed=42
    )
    leak_src = DatasetPartitionProtocol.verify_zero_leakage(src_train, src_test)
    leak_tgt = DatasetPartitionProtocol.verify_zero_leakage(src_train, tgt_test)
    assert leak_src["is_leakage_free"] is True
    assert leak_tgt["is_leakage_free"] is True

    # Cross-Simulation Transfer Evaluation
    rng = np.random.default_rng(42)
    bounds = {"omega_m": (0.1, 0.5), "sigma8": (0.6, 1.0)}
    evaluator = CrossSimulationTransferEvaluator(param_bounds=bounds, seed=42)

    src_th = np.array([[r["omega_m"], r["sigma8"]] for r in src_test])
    src_obs = src_th + rng.normal(0, 0.02, size=src_th.shape)
    tgt_th = np.array([[r["omega_m"], r["sigma8"]] for r in tgt_test])
    # Target family has a small systematic hydro shift
    tgt_obs = tgt_th + np.array([0.01, -0.015]) + rng.normal(0, 0.02, size=tgt_th.shape)

    res4 = evaluator.evaluate_transfer(
        infer_fn=lambda x, n: x[np.newaxis, :] + rng.normal(0, 0.02, size=(n, 2)),
        source_test_thetas=src_th,
        source_test_obs=src_obs,
        target_test_thetas=tgt_th,
        target_test_obs=tgt_obs,
        n_posterior_samples=120,
    )
    assert "pre_transfer" in res4
    assert "post_transfer" in res4
    assert "degradation" in res4
    assert res4["is_transfer_bounded"] is True
