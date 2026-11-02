"""
Unit tests for target-family domain adaptation with the trained flow and the
live progress dashboard integration.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cosmoverse.inference import SBIEngine
from cosmoverse.inference.domain_adaptation import DomainAdaptationPipeline
from cosmoverse.validation import ProgressDashboard


def test_fine_tune_flow_updates_weights() -> None:
    rng = np.random.default_rng(42)

    def simulator(theta: np.ndarray) -> np.ndarray:
        return theta * 2.0 + rng.normal(0.0, 0.02, size=len(theta))

    bounds = {"omega_m": (0.1, 0.5), "sigma8": (0.6, 1.0)}
    sbi = SBIEngine(
        simulator_fn=simulator,
        param_bounds=bounds,
        seed=42,
        density_estimator="flow",
        flow_epochs=80,
        flow_scale_bound=4.0,
    )
    sbi.generate_training_bank(n_simulations=120)
    sbi.fit_density_estimator(n_layers=4)

    target_theta = sbi.sample_prior(80)
    target_x = np.array([simulator(theta) for theta in target_theta])
    result = sbi.fine_tune_flow(
        target_theta,
        target_x,
        n_epochs=80,
        learning_rate=3e-4,
        seed=42,
    )

    assert result["n_epochs"] == 80
    assert len(result["loss_history"]) == 80
    assert result["loss_history"][-1] < result["loss_history"][0]


@pytest.mark.slow
def test_domain_adaptation_simba_and_dashboard(tmp_path: pathlib.Path) -> None:
    dashboard_path = tmp_path / "domain_adaptation.html"
    dashboard = ProgressDashboard(dashboard_path)
    pipeline = DomainAdaptationPipeline(
        family="CAMELS-SIMBA",
        n_calibration=200,
        n_test=100,
        fine_tune_epochs=200,
        seed=42,
        dashboard=dashboard,
    )
    result = pipeline.run()

    assert result["leakage_check"]["is_leakage_free"] is True
    assert result["calibration_test_disjoint"] is True
    assert result["source_calibration"]["passes_joint_sbi_gate"] is True
    assert result["improved"] is True
    assert result["after_adaptation"]["median_bias_in_sigma"] < result[
        "before_adaptation"
    ]["median_bias_in_sigma"]
    assert result["passes_adaptation_gate"] is True

    assert dashboard_path.exists()
    html = dashboard_path.read_text(encoding="utf-8")
    assert "After adaptation" in html
    assert "adaptation" in html.lower()
