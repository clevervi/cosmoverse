"""
Unit tests for past-light-cone catalog construction from PM snapshots,
periodic tiling, wide-angle RSD, and NPZ export.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cosmoverse.core.parameters import SimulationParameters
from cosmoverse.observations import LightConeBuilder
from cosmoverse.simulation.nbody import ParticleMeshSimulator


def _build_catalog() -> tuple[LightConeBuilder, dict]:
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
    catalog = builder.build_from_simulation(
        simulator,
        snapshot_redshifts=[0.0, 0.1, 0.2],
        z_max=0.2,
        n_radial_bins=4,
    )
    return builder, catalog


def test_light_cone_catalog_from_pm_snapshots() -> None:
    builder, catalog = _build_catalog()
    metrics = builder.evaluate_catalog(catalog)

    assert metrics["n_objects"] > 0
    assert metrics["n_populated_shells"] == metrics["n_shells"] == 4
    assert metrics["passes_light_cone_catalog_gate"] is True
    assert 0.0 < metrics["mean_rsd_displacement_mpc_h"] < 20.0
    assert metrics["redshift_min"] >= 0.0
    assert metrics["redshift_max"] > 0.0

    assert np.all(catalog["ra_deg"] >= -180.0)
    assert np.all(catalog["ra_deg"] <= 180.0)
    assert np.all(catalog["dec_deg"] >= -90.0)
    assert np.all(catalog["dec_deg"] <= 90.0)
    assert np.all(catalog["chi_real_mpc_h"] >= 0.0)
    assert np.all(catalog["chi_redshift_space_mpc_h"] >= 0.0)
    assert np.all(np.isfinite(catalog["z_obs"]))
    assert np.all(np.isfinite(catalog["v_parallel_km_s"]))


def test_light_cone_builder_without_rsd() -> None:
    sim_params = SimulationParameters(
        box_size=50.0,
        n_mesh=16,
        n_steps=2,
        z_initial=5.0,
        z_final=0.0,
        seed=42,
    )
    simulator = ParticleMeshSimulator(sim_params=sim_params)
    snapshots = simulator.run_with_snapshots([0.0, 0.1, 0.2])

    builder = LightConeBuilder(box_size=50.0, n_mesh=16)
    catalog = builder.build_from_snapshots(
        snapshots,
        z_max=0.2,
        n_radial_bins=4,
        apply_wide_angle_rsd=False,
    )

    assert np.allclose(
        catalog["chi_redshift_space_mpc_h"],
        catalog["chi_real_mpc_h"],
        rtol=0.0,
        atol=1e-12,
    )
    assert builder.evaluate_catalog(catalog)["mean_rsd_displacement_mpc_h"] == 0.0


def test_light_cone_catalog_export(tmp_path: pathlib.Path) -> None:
    builder, catalog = _build_catalog()
    manifest = builder.export_catalog(catalog, out_dir=tmp_path / "cone")

    output = pathlib.Path(str(manifest["path"]))
    assert output.exists()
    assert len(str(manifest["sha256"])) == 64
    data = np.load(output)
    assert data["ra_deg"].shape == (manifest["n_objects"],)
    assert data["shell_index"].shape == (manifest["n_objects"],)


def test_light_cone_builder_input_validation() -> None:
    builder = LightConeBuilder(box_size=50.0, n_mesh=16)
    with pytest.raises(ValueError):
        builder.build_from_snapshots({}, z_max=0.2)
    with pytest.raises(ValueError):
        builder.build_from_snapshots({0.0: {}}, z_max=0.0)
    with pytest.raises(ValueError):
        builder.build_from_snapshots({0.0: {}}, z_max=0.2, n_radial_bins=0)
