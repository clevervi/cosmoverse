"""
Unit tests for Level 2 relativistic light-cone geometry,
lensing magnification bias, and wide-angle redshift-space mapping.
"""

from __future__ import annotations

import numpy as np

from cosmoverse.observations import (
    LightConeGeometry,
    RelativisticLightConePipeline,
)


def test_light_cone_shell_geometry() -> None:
    geometry = LightConeGeometry()
    shells = geometry.build_shells(0.4, 1.1, n_shells=16)

    assert shells["z_edges"].shape == (17,)
    assert shells["z_centers"].shape == (16,)
    assert np.all(np.diff(shells["z_edges"]) > 0.0)
    assert np.all(np.diff(shells["chi_edges_mpc_h"]) > 0.0)
    assert np.all(shells["volume_element_mpc3_per_sr"] > 0.0)
    assert np.all(shells["shell_volume_mpc3"] > 0.0)
    assert np.all(shells["exact_shell_volume_mpc3"] > 0.0)

    volume_ratio = float(
        np.sum(shells["shell_volume_mpc3"])
        / np.sum(shells["exact_shell_volume_mpc3"])
    )
    assert 0.99 < volume_ratio < 1.01
    assert np.all(np.diff(shells["lookback_time_gyr"]) > 0.0)

    lookback = geometry.lookback_time(np.array([0.0, 1.0, 2.0]))
    assert float(lookback[0]) == 0.0
    assert np.all(np.diff(lookback) > 0.0)
    assert 8.0 < float(lookback[-1]) < 15.0


def test_magnification_bias_spectra() -> None:
    pipeline = RelativisticLightConePipeline(magnification_slope=0.6)
    spectra = pipeline.compute_magnification_bias_spectra()

    assert spectra["ell"].shape == (24,)
    for key in (
        "c_ell_gg_intrinsic",
        "c_ell_gkappa",
        "c_ell_kappa_kappa",
        "c_ell_gg_magnified",
    ):
        values = np.asarray(spectra[key], dtype=np.float64)
        assert values.shape == spectra["ell"].shape
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    assert np.all(
        spectra["c_ell_gg_magnified"] > spectra["c_ell_gg_intrinsic"]
    )

    no_magnification = RelativisticLightConePipeline(
        magnification_slope=0.4
    ).compute_magnification_bias_spectra()
    assert np.allclose(
        no_magnification["c_ell_gg_magnified"],
        no_magnification["c_ell_gg_intrinsic"],
        rtol=1e-12,
        atol=0.0,
    )

    higher_slope = RelativisticLightConePipeline(
        magnification_slope=0.8
    ).compute_magnification_bias_spectra()
    assert float(np.mean(higher_slope["c_ell_gg_magnified"])) > float(
        np.mean(spectra["c_ell_gg_magnified"])
    )


def test_wide_angle_redshift_space_mapping() -> None:
    pipeline = RelativisticLightConePipeline()

    narrow = pipeline.evaluate_wide_angle_mapping_error(opening_angle_deg=1.0)
    wide = pipeline.evaluate_wide_angle_mapping_error(opening_angle_deg=60.0)
    assert narrow["fractional_rms_error"] < 0.05
    assert wide["fractional_rms_error"] > 0.20
    assert wide["fractional_rms_error"] > narrow["fractional_rms_error"]

    positions = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    velocities = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
    s_wide = pipeline.displace_to_light_cone_redshift_space(
        positions, velocities, scale_a=1.0, hubble_z=100.0, h_param=1.0
    )
    assert np.allclose(s_wide, np.array([[11.0, 0.0, 0.0], [0.0, 0.0, 11.0]]))

    s_plane = pipeline.displace_to_plane_parallel_redshift_space(
        positions, velocities, scale_a=1.0, hubble_z=100.0, h_param=1.0
    )
    assert np.isclose(s_plane[0, 0], 10.0)
    assert np.isclose(s_plane[0, 2], 0.0)
    assert np.isclose(s_plane[1, 2], 11.0)


def test_relativistic_light_cone_gate() -> None:
    impact = RelativisticLightConePipeline().evaluate_relativistic_light_cone_impact()

    assert impact["passes_relativistic_light_cone_gate"] is True
    assert float(impact["unmarginalized_s8_bias_sigma"]) > 0.30
    assert float(impact["marginalized_s8_bias_sigma"]) < 0.30
    assert 0.0 < float(impact["mean_magnification_contamination_pct"]) < 20.0
    assert 0.2 < float(impact["wide_angle_fractional_rms_error_60deg"]) < 1.5
