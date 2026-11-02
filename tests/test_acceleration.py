"""
Unit tests for the device audit and the optional CuPy Particle-Mesh backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmoverse.acceleration import cupy_available, device_summary
from cosmoverse.core.parameters import SimulationParameters
from cosmoverse.simulation.nbody import ParticleMeshSimulator


def test_device_summary() -> None:
    info = device_summary()
    assert "backend" in info
    assert info["backend"] in {"numpy", "cupy"}
    assert "cupy_available" in info
    assert info["gpu_count"] >= 0


def test_gpu_pm_matches_cpu() -> None:
    if not cupy_available():
        pytest.skip("CuPy/CUDA device not available")

    sim_params = SimulationParameters(
        box_size=50.0, n_mesh=16, n_steps=1, seed=42
    )
    cpu_sim = ParticleMeshSimulator(sim_params=sim_params, use_gpu=False)
    cpu_density = cpu_sim.run()

    gpu_sim = ParticleMeshSimulator(sim_params=sim_params, use_gpu=True)
    gpu_density = gpu_sim.run()
    gpu_numpy = np.asarray(gpu_density.get())

    assert gpu_numpy.shape == cpu_density.shape
    assert np.max(np.abs(cpu_density - gpu_numpy)) < 1e-10

    _, cpu_pk = cpu_sim.compute_power_spectrum(n_bins=6)
    _, gpu_pk = gpu_sim.compute_power_spectrum(n_bins=6)
    assert np.allclose(cpu_pk, gpu_pk, rtol=1e-10)
