"""
Unit tests for core cosmology, parameters, and configuration.
"""

import numpy as np
import pytest

from cosmoverse.core.configuration import CosmoverseConfig
from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import (
    CosmologicalParameters,
)


def test_cosmological_parameters_validation():
    params = CosmologicalParameters.planck2018()
    assert params.omega_m == pytest.approx(0.3153, rel=1e-3)
    assert params.h == pytest.approx(0.6736, rel=1e-3)
    assert params.H0 == pytest.approx(67.36, rel=1e-3)
    assert params.omega_cdm == pytest.approx(0.3153 - 0.0493, rel=1e-3)
    assert params.S8 > 0.0

    # Ensure invalid omega_b raises ValueError
    with pytest.raises(ValueError):
        CosmologicalParameters(omega_m=0.1, omega_b=0.2)


def test_prior_sampling():
    samples = CosmologicalParameters.sample_prior(n_samples=10, seed=42)
    assert len(samples) == 10
    for s in samples:
        assert 0.1 <= s.omega_m <= 0.5
        assert 0.6 <= s.sigma8 <= 1.0


def test_cosmology_background():
    cosmo = Cosmology(CosmologicalParameters.planck2018())

    # E(0) must equal 1.0 for flat Lambda-CDM
    assert cosmo.E(0.0) == pytest.approx(1.0, rel=1e-4)
    assert cosmo.E(1.0) > 1.0

    # Distances
    assert cosmo.comoving_distance(0.0) == pytest.approx(0.0)
    chi1 = cosmo.comoving_distance(1.0)
    assert chi1 > 2000.0  # Approx 2300-2400 Mpc/h

    da1 = cosmo.angular_diameter_distance(1.0)
    dl1 = cosmo.luminosity_distance(1.0)
    assert da1 == pytest.approx(chi1 / 2.0)
    assert dl1 == pytest.approx(chi1 * 2.0)


def test_growth_factor():
    cosmo = Cosmology(CosmologicalParameters.planck2018())
    # Normalized so D1(0) = 1.0
    d0 = cosmo.growth_factor(0.0)
    assert d0 == pytest.approx(1.0, rel=1e-3)

    # In matter era D1(z) scales roughly as 1/(1+z)
    d1 = cosmo.growth_factor(1.0)
    assert 0.4 < d1 < 0.7


def test_linear_matter_power_spectrum():
    cosmo = Cosmology(CosmologicalParameters.planck2018())
    k = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0])
    pk = cosmo.linear_matter_power_spectrum(k, z=0.0)

    assert len(pk) == len(k)
    assert np.all(pk > 0.0)
    # P(k) turnover check: P(0.02) should be higher than P(1e-3) and P(10.0)
    pk_turnover = cosmo.linear_matter_power_spectrum(0.02, z=0.0)
    assert pk_turnover > pk[0]
    assert pk_turnover > pk[-1]


def test_configuration_io(tmp_path):
    cfg = CosmoverseConfig(project_name="unit-test-run")
    yaml_file = tmp_path / "config.yaml"
    cfg.to_yaml(yaml_file)
    assert yaml_file.exists()

    loaded = CosmoverseConfig.from_yaml(yaml_file)
    assert loaded.project_name == "unit-test-run"
    assert loaded.cosmology.omega_m == cfg.cosmology.omega_m
