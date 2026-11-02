"""
Unit tests for the PM-trained P(k) emulator and the PM-emulator multi-probe
SBI pipeline with SBC/TARP calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmoverse.emulators import PMPowerSpectrumEmulator
from cosmoverse.inference import PMEmulatorSBIPipeline


def _small_emulator() -> PMPowerSpectrumEmulator:
    return PMPowerSpectrumEmulator(
        n_grid=3,
        n_mesh=16,
        n_steps=2,
        n_k_bins=6,
        seed=42,
    )


def test_pm_pk_emulator_training_and_prediction() -> None:
    emulator = _small_emulator().train()
    accuracy = emulator.evaluate_accuracy()

    assert accuracy["n_training_points"] == 9
    assert accuracy["loo_mape_pct"] < 5.0
    assert accuracy["passes_pm_emulator_gate"] is True

    pk = emulator.predict_pk(np.array([0.05, 0.10, 0.20]), 0.315, 0.811)
    assert pk.shape == (3,)
    assert np.all(np.isfinite(pk))
    assert np.all(pk > 0.0)


def test_pm_emulator_sbi_fit_and_infer() -> None:
    pipeline = PMEmulatorSBIPipeline(
        emulator=_small_emulator(),
        n_training_simulations=80,
        seed=42,
    )
    pipeline.fit()

    posterior = pipeline.infer(
        pipeline.fiducial_data_vector(), n_posterior_samples=100
    )
    assert posterior.samples.shape == (100, 4)

    for name, (low, high) in pipeline.param_bounds.items():
        column = posterior.samples[:, list(pipeline.param_bounds).index(name)]
        assert np.all(column >= low)
        assert np.all(column <= high)

    intervals = posterior.credible_intervals()
    assert set(intervals) == {"omega_m", "sigma8", "b1", "A_AGN"}


@pytest.mark.slow
def test_pm_emulator_flow_calibration_gate() -> None:
    pipeline = PMEmulatorSBIPipeline(
        n_training_simulations=1600,
        density_estimator="flow",
        flow_n_layers=6,
        flow_epochs=800,
        flow_scale_bound=6.0,
        flow_hidden_dim=16,
        seed=42,
    )
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=100,
        n_tarp_trials=100,
        n_posterior_samples=150,
    )

    assert result["passes_sbc_gate"] is True
    assert result["passes_tarp_gate"] is True
    assert result["passes_joint_sbi_gate"] is True


def test_pm_emulator_sbi_calibration_gate() -> None:
    pipeline = PMEmulatorSBIPipeline(n_training_simulations=400, seed=42)
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=50,
        n_tarp_trials=50,
        n_posterior_samples=150,
    )

    assert result["emulator_accuracy"]["passes_pm_emulator_gate"] is True
    assert result["passes_sbc_gate"] is True
    assert result["passes_tarp_gate"] is True
    assert result["passes_joint_sbi_gate"] is True
