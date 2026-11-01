"""
Unit tests for Stage 3 Observational Realism: survey specifications, galaxy bias,
Redshift-Space Distortions (RSD multipoles P0, P2, P4), selection functions, and noise.
"""

import numpy as np

from cosmoverse.observations.forward_model import (
    GalaxyBiasModel,
    ObservationalForwardPipeline,
    RedshiftSpaceDistortionModel,
)
from cosmoverse.observations.selection import SelectionFunction
from cosmoverse.observations.survey import SurveySpecification


def test_survey_specs():
    desi = SurveySpecification.desi_lrg()
    assert desi.area_deg2 == 14000.0
    assert 0.0 < desi.f_sky < 1.0

    euclid = SurveySpecification.euclid_wide()
    assert euclid.shape_noise_sigma_e > 0.0

    lsst = SurveySpecification.lsst_rubin()
    assert lsst.area_deg2 == 18000.0


def test_selection_function():
    coords = np.random.uniform(0, 100, size=(500, 3))
    sf = SelectionFunction(completeness_rate=0.90)
    mask = sf.apply_selection(coords, seed=42)
    assert mask.shape == (500,)
    assert 400 < np.sum(mask) < 480


def test_galaxy_bias_and_rsd_multipoles():
    bias = GalaxyBiasModel(b1=1.6, b2=0.2)
    delta_m = np.random.default_rng(42).normal(0, 0.5, size=(8, 8, 8))
    delta_g = bias.apply_to_field(delta_m)
    assert delta_g.shape == (8, 8, 8)
    assert np.isclose(np.mean(delta_g), 0.0, atol=1e-6)

    rsd = RedshiftSpaceDistortionModel(b1=1.6, growth_rate_f=0.55, sigma_fog=3.0)
    k = np.array([0.05, 0.1, 0.3])
    pk_m = np.array([10000.0, 5000.0, 1200.0])
    mp = rsd.compute_multipoles(k, pk_m, shot_noise=100.0)
    assert mp["P0"].shape == (3,)
    assert mp["P2"].shape == (3,)
    assert mp["P4"].shape == (3,)
    # Monopole on large scales is boosted by Kaiser + bias relative to matter P(k)
    assert mp["P0"][0] > pk_m[0]


def test_observational_forward_pipeline_stage3():
    pipeline = ObservationalForwardPipeline()
    k = np.array([0.05, 0.1, 0.2])
    pk = np.array([5000.0, 2500.0, 1000.0])

    obs = pipeline.observe_power_spectrum(k, pk, seed=42)
    assert len(obs["P0_obs"]) == 3
    assert np.all(obs["sigma_P0"] > 0.0)

    field = np.zeros((16, 16, 16))
    obs_field = pipeline.apply_3d_survey_mask_and_noise(field, box_size=100.0, seed=42)
    assert obs_field.shape == (16, 16, 16)
    # Masked edge slice is strictly zero
    assert np.all(obs_field[0, :, :] == 0.0)
