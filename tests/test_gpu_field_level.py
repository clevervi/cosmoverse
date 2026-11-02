"""CUDA regression tests for the v3.5 CuPy field-level benchmark."""

import numpy as np
import pytest

from cosmoverse.acceleration import cupy_available
from cosmoverse.inference.field_level import FieldLevelInferenceEngine
from cosmoverse.inference.gpu_field_level import (
    GPUFieldLevelBenchmark,
    GPUFieldLevelEngine,
)

pytestmark = pytest.mark.skipif(
    not cupy_available(), reason="CuPy/CUDA device not available"
)


def test_gpu_field_level_potential_and_gradient_match_cpu():
    cpu = FieldLevelInferenceEngine(grid_size=8, box_size=100.0, noise_sigma=0.25)
    gpu = GPUFieldLevelEngine(
        grid_size=8,
        box_size=100.0,
        noise_sigma=0.25,
        prior_std=cpu.prior_std,
    )
    rng = np.random.default_rng(7)
    state = rng.normal(0.0, 0.1, size=1 + cpu.n_voxels)
    obs = rng.normal(0.0, 0.2, size=cpu.n_voxels)

    cpu_potential, cpu_gradient = cpu.potential_and_grad(state, obs)
    gpu_potential, gpu_gradient = gpu.potential_and_grad_gpu(state, obs)

    assert abs(cpu_potential - gpu_potential) < 1e-9 * max(
        1.0, abs(cpu_potential)
    )
    assert np.max(np.abs(cpu_gradient - gpu_gradient.get())) < 1e-10


def test_gpu_field_level_chain_smoke_and_benchmark_report():
    report = GPUFieldLevelBenchmark(grid_size=8, seed=5).run(
        n_samples=4, n_leapfrog=2
    )

    assert report["grid_size"] == 8
    assert report["dimension"] == 1 + 8 ** 3
    assert report["gpu_chain_shape"] == [4, 1 + 8 ** 3]
    assert report["potential_relative_error"] < 1e-8
    assert report["grad_max_abs_error"] < 1e-8
    assert report["gpu_diagnostics"]["eevpd"] >= 0.0
    assert np.isfinite(report["gpu_marginal_alpha_mean"])
    assert report["gpu_chain_seconds"] > 0.0
    assert report["cpu_chain_seconds"] > 0.0
