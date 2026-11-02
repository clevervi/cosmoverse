"""
Regression test for the v3.5 calibration-budget curve (Result 4 extension).

The full SIMBA sweep is expensive (~2-3 min), so this test is marked slow and
reproduced only in the `-Full` local CI gate.
"""

import pytest

from cosmoverse.inference.calibration_budget import CalibrationBudgetAnalyzer


@pytest.mark.slow
def test_simba_calibration_budget_reproduces_minimum_ncal():
    report = CalibrationBudgetAnalyzer(
        family="CAMELS-SIMBA",
        budgets=(25, 50, 100, 200, 400),
        n_test=100,
        fine_tune_epochs=200,
        seed=42,
    ).run()

    assert report["passes_budget_gate"] is True
    assert report["min_calibration_for_gate"] == 50
    assert report["leakage_check"]["is_leakage_free"] is True
    assert report["calibration_test_disjoint"] is True
    assert report["source_calibration"]["passes_joint_sbi_gate"] is True

    curve = report["curve"]
    assert [entry["n_cal"] for entry in curve] == [25, 50, 100, 200, 400]
    assert curve[0]["passes_gate"] is False
    assert curve[0]["coverage_68"] < 0.55
    assert curve[0]["coverage_95"] < 0.85
    assert all(entry["passes_gate"] for entry in curve[1:])
    assert curve[-1]["coverage_95"] >= curve[0]["coverage_95"]
    assert curve[-1]["median_bias_in_sigma"] < curve[0]["median_bias_in_sigma"]
