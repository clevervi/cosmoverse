"""
Unit tests for gradient-based training of the conditional RealNVP flow and
the end-to-end flow-SBI calibration gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmoverse.inference import MultiProbeSBIPipeline
from cosmoverse.inference.flows import ConditionalRealNVP


def test_flow_log_prob_consistency_and_gradients() -> None:
    flow = ConditionalRealNVP(
        param_dim=3,
        context_dim=2,
        n_layers=2,
        seed=42,
        scale_bound=3.0,
    )
    rng = np.random.default_rng(7)
    theta = rng.normal(size=(6, 3))
    context = rng.normal(size=(6, 2))

    z = rng.normal(size=(6, 3))
    current = z
    log_det_forward = np.zeros(6)
    for layer, permutation in zip(
        flow.layers, flow.permutations, strict=True
    ):
        current, log_det = layer.forward(current, context)
        log_det_forward += log_det
        current = current[:, permutation]

    log_base = -0.5 * np.sum(z**2, axis=1) - 0.5 * 3 * np.log(2.0 * np.pi)
    np.testing.assert_allclose(
        flow.log_prob(current, context),
        log_base - log_det_forward,
        rtol=1e-10,
        atol=1e-10,
    )

    def loss() -> float:
        return float(-np.mean(flow.log_prob(theta, context)))

    grads = flow.backward_log_prob(theta, context)
    for layer_index, layer in enumerate(flow.layers):
        for name in ("w_s", "b_s", "w_t", "b_t"):
            parameter = getattr(layer, name)
            index = tuple(0 for _ in parameter.shape)
            original = parameter[index]
            epsilon = 1e-6
            parameter[index] = original + epsilon
            loss_plus = loss()
            parameter[index] = original - epsilon
            loss_minus = loss()
            parameter[index] = original
            finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon)
            np.testing.assert_allclose(
                grads[layer_index][name][index],
                finite_difference,
                rtol=1e-5,
                atol=1e-7,
            )


def test_mlp_coupling_gradients() -> None:
    flow = ConditionalRealNVP(
        param_dim=4,
        context_dim=3,
        n_layers=2,
        seed=42,
        scale_bound=3.0,
        hidden_dim=8,
    )
    rng = np.random.default_rng(11)
    theta = rng.normal(size=(5, 4))
    context = rng.normal(size=(5, 3))

    def loss() -> float:
        return float(-np.mean(flow.log_prob(theta, context)))

    grads = flow.backward_log_prob(theta, context)
    for layer_index, layer in enumerate(flow.layers):
        for name in layer.parameter_names():
            parameter = getattr(layer, name)
            index = tuple(0 for _ in parameter.shape)
            original = parameter[index]
            epsilon = 1e-6
            parameter[index] = original + epsilon
            loss_plus = loss()
            parameter[index] = original - epsilon
            loss_minus = loss()
            parameter[index] = original
            finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon)
            np.testing.assert_allclose(
                grads[layer_index][name][index],
                finite_difference,
                rtol=1e-5,
                atol=1e-7,
            )


@pytest.mark.slow
def test_flow_sbi_calibration_gate() -> None:
    pipeline = MultiProbeSBIPipeline(
        n_training_simulations=400,
        density_estimator="flow",
        seed=42,
    )
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=50,
        n_tarp_trials=50,
        n_posterior_samples=150,
    )

    assert result["passes_sbc_gate"] is True
    assert result["passes_tarp_gate"] is True
    assert result["passes_joint_sbi_gate"] is True
