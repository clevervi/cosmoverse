"""
Unit tests for survey masking, galaxy bias sampling, and angular clustering
measured from a particle light-cone catalog.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmoverse.core.parameters import SimulationParameters
from cosmoverse.observations import (
    GalaxyBiasSampler,
    LightConeBuilder,
    LightConeSurveyPipeline,
    SurveyMask,
)
from cosmoverse.simulation.nbody import ParticleMeshSimulator


@pytest.fixture(scope="module")
def light_cone_catalog() -> dict:
    sim_params = SimulationParameters(
        box_size=50.0,
        n_mesh=16,
        n_steps=2,
        z_initial=5.0,
        z_final=0.0,
        seed=42,
    )
    simulator = ParticleMeshSimulator(sim_params=sim_params)
    builder = LightConeBuilder(box_size=50.0, n_mesh=16)
    return builder.build_from_simulation(
        simulator,
        snapshot_redshifts=[0.0, 0.1, 0.2],
        z_max=0.2,
        n_radial_bins=4,
        density_smoothing_cells=3.0,
    )


def test_survey_mask_selection_and_area() -> None:
    catalog = {
        "ra_deg": np.array([0.0, 10.0, 359.0, 10.0]),
        "dec_deg": np.array([80.0, 85.0, 80.0, 60.0]),
        "z_obs": np.array([0.1, 0.1, 0.1, 0.1]),
        "value": np.array([1.0, 2.0, 3.0, 4.0]),
    }
    mask = SurveyMask(
        ra_min=0.0,
        ra_max=20.0,
        dec_min=70.0,
        dec_max=90.0,
        z_min=0.0,
        z_max=0.2,
    )
    selected = mask.apply(catalog)

    assert selected["n_objects"] == 2
    assert np.allclose(selected["value"], [1.0, 2.0])
    assert mask.area_deg2 > 0.0

    with pytest.raises(ValueError):
        SurveyMask(ra_min=20.0, ra_max=10.0)
    with pytest.raises(ValueError):
        SurveyMask(dec_min=90.0, dec_max=70.0)
    with pytest.raises(ValueError):
        SurveyMask(z_min=0.5, z_max=0.2)


def test_galaxy_bias_sampler(light_cone_catalog: dict) -> None:
    matter = SurveyMask().apply(light_cone_catalog)
    sampler = GalaxyBiasSampler(b1=1.5, sampling_rate=0.3, seed=42)
    galaxy = sampler.sample(matter)

    assert 0 < len(galaxy["ra_deg"]) < len(matter["ra_deg"])
    assert len(galaxy["selection_mask"]) == len(matter["ra_deg"])
    assert np.count_nonzero(galaxy["selection_mask"]) == len(galaxy["ra_deg"])
    assert galaxy["selection_probability"].shape == (len(galaxy["ra_deg"]),)
    assert galaxy["b1_input"] == 1.5
    assert np.all(galaxy["selection_probability"] >= 0.0)
    assert np.all(galaxy["selection_probability"] <= 1.0)


def test_light_cone_survey_pipeline(light_cone_catalog: dict) -> None:
    pipeline = LightConeSurveyPipeline(b1=1.5, sampling_rate=0.3, seed=42)
    result = pipeline.evaluate(light_cone_catalog, box_size=50.0)

    assert result["n_matter"] > result["n_galaxies"] > 100
    assert result["b1_effective"] > 1.0
    assert result["pearson_r_gm_mm"] > 0.8
    assert result["passes_light_cone_survey_gate"] is True
    assert result["theta_deg"].shape == result["w_theta_gm"].shape
    assert np.all(np.isfinite(result["w_theta_gm"]))
    assert np.all(np.isfinite(result["w_theta_mm"]))
    assert np.all(np.isfinite(result["w_theta_analytic"]))


def test_no_bias_control_fails_survey_gate(light_cone_catalog: dict) -> None:
    pipeline = LightConeSurveyPipeline(b1=0.0, sampling_rate=0.3, seed=42)
    result = pipeline.evaluate(light_cone_catalog, box_size=50.0)

    assert result["b1_effective"] < 1.0
    assert result["passes_light_cone_survey_gate"] is False
