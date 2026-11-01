"""
Unit tests for CAMELSHydroSpectraConnector (`IllustrisTNG`, `SIMBA`, `Astrid` power spectra
and multi-suite transfer table).
"""

import pathlib

import numpy as np

from cosmoverse.datasets.camels_gen2 import CAMELSGen2Catalog
from cosmoverse.datasets.camels_spectra import CAMELSHydroSpectraConnector


def test_camels_suite_baryonic_responses_hierarchy():
    k = np.geomspace(1e-4, 5.0, 20)
    r_dmo = CAMELSHydroSpectraConnector.suite_baryonic_response(k, "CAMELS-DMO")
    r_tng = CAMELSHydroSpectraConnector.suite_baryonic_response(k, "CAMELS-IllustrisTNG")
    r_simba = CAMELSHydroSpectraConnector.suite_baryonic_response(k, "CAMELS-SIMBA")
    r_astrid = CAMELSHydroSpectraConnector.suite_baryonic_response(k, "CAMELS-Astrid")

    # Exact mass conservation at large scales (k -> 0)
    assert np.allclose(r_dmo, 1.0)
    assert np.isclose(r_tng[0], 1.0, atol=1e-3)
    assert np.isclose(r_simba[0], 1.0, atol=1e-3)
    assert np.isclose(r_astrid[0], 1.0, atol=1e-3)

    # At intermediate scale k ~ 2.0 h/Mpc, SIMBA has stronger AGN jet suppression than IllustrisTNG
    idx_k2 = int(np.argmin(np.abs(k - 2.0)))
    assert r_simba[idx_k2] < r_tng[idx_k2] < 1.0


def test_camels_disk_table_loading_and_multi_suite_transfer(tmp_path: pathlib.Path):
    catalog = CAMELSGen2Catalog()
    rec = catalog.get_subset("LH")[0]

    # Write a mock CAMELS public Pk table to disk
    k_disk = np.geomspace(0.02, 3.0, 15)
    pk_disk = 1500.0 * (k_disk ** (-1.5))
    table_path = tmp_path / f"{rec.sim_id}_Pk.txt"
    np.savetxt(table_path, np.column_stack([k_disk, pk_disk]))

    conn = CAMELSHydroSpectraConnector(data_dir=tmp_path)
    loaded = conn.load_or_compute_spectrum(rec)
    assert loaded["source"] == "disk_table"
    assert len(loaded["pk_hydro"]) == len(conn.k_grid)

    transfer_rows = conn.evaluate_multi_suite_transfer_table(n_test_sims=15, seed=42)
    assert len(transfer_rows) == 3
    suites = [r["target_suite"] for r in transfer_rows]
    assert suites == ["CAMELS-IllustrisTNG", "CAMELS-SIMBA", "CAMELS-Astrid"]
    for row in transfer_rows:
        assert row["is_leakage_free"] is True
        assert row["marginalized_bias_sigma"] < row["unmarginalized_bias_sigma"]
        assert row["stage4_compatible_after_marg"] is True


def test_paper1_latex_manuscript_and_bib_integrity():
    root = pathlib.Path(__file__).resolve().parent.parent
    main_tex = (root / "paper" / "main.tex").read_text(encoding="utf-8")
    bib_tex = (root / "paper" / "references.bib").read_text(encoding="utf-8")

    assert "\\title{\\textbf{COSMOVERSE I:" in main_tex
    assert "\\input{tables/table_results_summary.tex}" in main_tex
    assert "\\input{tables/table_suite_transfer.tex}" in main_tex
    # Section 8 rule: never cite unverified CosmoRGNet in the manuscript or bibliography
    assert "CosmoRGNet" not in main_tex
    assert "CosmoRGNet" not in bib_tex
    assert "TUNeS2026" in bib_tex
    assert "GiriSchneider2021" in bib_tex
