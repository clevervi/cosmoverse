"""
Unit tests for Level 2 Halo Occupation Distribution (HOD), Secondary Assembly Bias,
Matter-Galaxy Cross-Spectrum P_gm(k), and Stage-IV Synthetic Mock Catalog Exporter.
"""

from __future__ import annotations

import pathlib

import numpy as np

from cosmoverse.observations.hod_and_mocks import (
    HaloOccupationModel,
    HODAssemblyBiasParameters,
    StageIVMockCatalogExporter,
)


def test_hod_occupation_and_cross_spectra() -> None:
    model = HaloOccupationModel(hod_params=HODAssemblyBiasParameters())
    masses = np.logspace(11.0, 15.0, 16)

    n_cen = model.mean_central_occupation(masses, c_rank=0.0)
    n_sat = model.mean_satellite_occupation(masses, c_rank=0.0)
    assert np.all((n_cen >= 0.0) & (n_cen <= 1.0))
    assert float(n_cen[-1]) > 0.99
    assert float(n_sat[-1]) > float(n_sat[0])

    # Higher concentration rank increases central occupation when A_cent > 0
    n_cen_high = model.mean_central_occupation(np.array([1e13]), c_rank=0.5)[0]
    n_cen_low = model.mean_central_occupation(np.array([1e13]), c_rank=-0.5)[0]
    assert n_cen_high > n_cen_low

    k = np.geomspace(0.05, 1.5, 20)
    pk_m = 2500.0 * (k / 0.1) ** (-1.6)
    spectra = model.compute_galaxy_matter_spectra(k, pk_m, z=0.0)
    assert len(spectra["pk_gg"]) == 20
    assert len(spectra["pk_gm"]) == 20
    assert float(spectra["b_eff"]) > 1.0
    assert 0.85 <= float(np.mean(spectra["r_gm"])) <= 1.05

    ab_eval = model.evaluate_assembly_bias_marginalization()
    assert ab_eval["passes_stage4_assembly_bias_gate"] is True
    assert float(ab_eval["marginalized_s8_bias_sigma"]) < 0.30


def test_stage4_mock_catalog_exporter(tmp_path: pathlib.Path) -> None:
    manifest = StageIVMockCatalogExporter.export_mock_catalog(
        out_dir=tmp_path / "mocks", n_galaxies=64, seed=42
    )
    assert manifest["n_galaxies"] == 64
    npz_file = pathlib.Path(str(manifest["npz_file"]))
    csv_file = pathlib.Path(str(manifest["csv_file"]))
    assert npz_file.exists()
    assert csv_file.exists()

    loaded = np.load(npz_file)
    assert loaded["ra_deg"].shape == (64,)
    assert loaded["z_rsd_mpc_h"].shape == (64,)
    assert loaded["gamma1"].shape == (64,)
