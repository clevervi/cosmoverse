"""
Unit tests for four-parameter multi-probe SBI on the joint light-cone
w(theta) + RSD P_0, P_2, P_4 data vector.
"""

from __future__ import annotations

import numpy as np

from cosmoverse.inference import MultiProbeSBIPipeline


def test_multiprobe_forward_model() -> None:
    pipeline = MultiProbeSBIPipeline(n_training_simulations=20)

    data = pipeline.model_data_vector(np.array([0.315, 0.811, 1.65, 1.0]))
    assert data.shape == (36,)
    assert np.all(np.isfinite(data))

    low_b1 = pipeline.model_data_vector(np.array([0.315, 0.811, 1.2, 1.0]))
    high_b1 = pipeline.model_data_vector(np.array([0.315, 0.811, 2.2, 1.0]))
    assert float(high_b1[12]) > float(low_b1[12])

    low_agn = pipeline.model_data_vector(np.array([0.315, 0.811, 1.65, 0.5]))
    high_agn = pipeline.model_data_vector(np.array([0.315, 0.811, 1.65, 2.0]))
    assert np.any(np.abs(low_agn[:12] - high_agn[:12]) > 1e-4)


def test_multiprobe_sbi_fit_and_infer() -> None:
    pipeline = MultiProbeSBIPipeline(n_training_simulations=100, seed=42)
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


def test_multiprobe_sbi_calibration_gate() -> None:
    pipeline = MultiProbeSBIPipeline(n_training_simulations=400, seed=42)
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
