"""
Unit tests for end-to-end SBI on the light-cone angular correlation function
w(theta): fast Limber + BCEmu forward model, posterior inference, and
SBC/TARP blocking calibration.
"""

from __future__ import annotations

import numpy as np

from cosmoverse.inference import LightConeSBIPipeline


def test_light_cone_forward_model() -> None:
    pipeline = LightConeSBIPipeline(n_training_simulations=20)

    fiducial = pipeline.model_w_theta(np.array([0.315, 0.811, 1.0]))
    assert fiducial.shape == (12,)
    assert np.all(np.isfinite(fiducial))

    low_agn = pipeline.model_w_theta(np.array([0.315, 0.811, 0.5]))
    high_agn = pipeline.model_w_theta(np.array([0.315, 0.811, 2.0]))
    assert np.any(np.abs(low_agn - high_agn) > 1e-4)

    low_s8 = pipeline.model_w_theta(np.array([0.315, 0.72, 1.0]))
    high_s8 = pipeline.model_w_theta(np.array([0.315, 0.90, 1.0]))
    assert float(high_s8[0]) > float(low_s8[0])


def test_light_cone_sbi_fit_and_infer() -> None:
    pipeline = LightConeSBIPipeline(n_training_simulations=80, seed=42)
    pipeline.fit()

    posterior = pipeline.infer(
        pipeline.fiducial_w_theta(), n_posterior_samples=100
    )
    assert posterior.samples.shape == (100, 3)

    for name, (low, high) in pipeline.param_bounds.items():
        column = posterior.samples[:, list(pipeline.param_bounds).index(name)]
        assert np.all(column >= low)
        assert np.all(column <= high)

    intervals = posterior.credible_intervals()
    assert set(intervals) == {"omega_m", "sigma8", "A_AGN"}


def test_light_cone_sbi_calibration_gate() -> None:
    pipeline = LightConeSBIPipeline(n_training_simulations=200, seed=42)
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=50,
        n_tarp_trials=50,
        n_posterior_samples=150,
    )

    assert result["passes_sbc_gate"] is True
    assert result["passes_tarp_gate"] is True
    assert result["passes_joint_sbi_gate"] is True
    for metric in result["sbc"].values():
        assert metric["passes_acceptance_gate"] is True
