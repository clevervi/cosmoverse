"""
Unit tests for Stage 5 (Paper 2 Extension) Differentiable Field-Level Inference
and Preconditioned MCLMC vs HMC benchmark (Issue #23).
"""

import numpy as np

from cosmoverse.inference.field_level import FieldLevelInferenceEngine


def test_field_level_analytical_gradient_matches_finite_difference():
    engine = FieldLevelInferenceEngine(grid_size=4, box_size=50.0, noise_sigma=0.3)
    rng = np.random.default_rng(7)
    q = rng.normal(0.0, 0.1, size=1 + engine.n_voxels)
    obs = rng.normal(0.0, 0.2, size=engine.n_voxels)

    _, grad_analytic = engine.potential_and_grad(q, obs)
    eps = 1e-6
    for idx in (0, 1, 5, 10):
        dq = np.zeros_like(q)
        dq[idx] = eps
        u_plus, _ = engine.potential_and_grad(q + dq, obs)
        u_minus, _ = engine.potential_and_grad(q - dq, obs)
        grad_fd = (u_plus - u_minus) / (2.0 * eps)
        assert np.isclose(grad_analytic[idx], grad_fd, rtol=1e-4, atol=1e-5)


def test_field_level_mclmc_vs_hmc_and_preconditioning_advantage():
    engine = FieldLevelInferenceEngine(grid_size=6, box_size=80.0, noise_sigma=0.25)
    res = engine.benchmark_mclmc_vs_hmc(n_samples=50, n_leapfrog=5, seed=42)

    assert res["dimension"] == 1 + 6 ** 3
    # Large-scale initial conditions are strongly correlated with truth
    assert res["mean_large_scale_rk"] > 0.70
    assert res["preconditioned_mclmc"]["is_energy_stable"] is True
    # Preconditioning significantly lowers energy variance per dimension (EEVPD)
    assert (
        res["preconditioned_mclmc"]["eevpd"]
        < res["unpreconditioned_mclmc"]["eevpd"]
    )
    assert res["preconditioned_hmc_plan_b"]["acceptance_rate"] > 0.20
