"""
Unit tests for initial conditions, Cloud-in-Cell deposition, and Particle-Mesh solver.
"""

import numpy as np
import pytest

from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import CosmologicalParameters, SimulationParameters
from cosmoverse.simulation.evolution import cic_deposit
from cosmoverse.simulation.initial_conditions import (
    generate_gaussian_random_field,
    setup_initial_particles,
)
from cosmoverse.simulation.nbody import ParticleMeshSimulator


def test_gaussian_random_field():
    cosmo = Cosmology(CosmologicalParameters.planck2018())
    sim_params = SimulationParameters(box_size=50.0, n_mesh=16, seed=42)
    delta_real, k_mag = generate_gaussian_random_field(cosmo, sim_params)

    assert delta_real.shape == (16, 16, 16)
    # Mean density perturbation should be close to zero
    assert np.mean(delta_real) == pytest.approx(0.0, abs=1e-2)


def test_initial_particles():
    cosmo = Cosmology(CosmologicalParameters.planck2018())
    sim_params = SimulationParameters(box_size=50.0, n_mesh=16, z_initial=50.0, seed=42)
    pos, vel = setup_initial_particles(cosmo, sim_params)

    assert pos.shape == (16**3, 3)
    assert vel.shape == (16**3, 3)
    assert np.all(pos >= 0.0)
    assert np.all(pos <= 50.0)


def test_cic_deposition():
    # 100 particles in a 20 Mpc/h box
    rng = np.random.default_rng(42)
    positions = rng.uniform(0.0, 20.0, size=(1000, 3))
    delta = cic_deposit(positions, box_size=20.0, n_mesh=16)

    assert delta.shape == (16, 16, 16)
    # Conservation of mass: mean(delta) should be approx 0
    assert np.mean(delta) == pytest.approx(0.0, abs=1e-5)


def test_particle_mesh_simulation_run():
    cosmo = Cosmology(CosmologicalParameters.planck2018())
    sim_params = SimulationParameters(box_size=50.0, n_mesh=16, n_steps=3, seed=42)
    sim = ParticleMeshSimulator(cosmology=cosmo, sim_params=sim_params)

    delta_final = sim.run()
    assert delta_final.shape == (16, 16, 16)

    k_bins, pk = sim.compute_power_spectrum(n_bins=6)
    assert len(k_bins) == 6
    assert len(pk) == 6
    assert np.all(pk >= 0.0)
