"""
Unit tests for v1.1.0 features:
  1. Joint Stage-IV 3x2pt + RSD Observational Pipeline (`Joint3x2ptObservationalPipeline`)
  2. Model Checkpoint Persistence & SHA-256 verification (`CheckpointManager`)
  3. CAMELS Multi-Suite Spectrum Cache Fetcher (`CAMELSArchiveFetcher`)
  4. JOSS Manuscript (`paper/paper.md`) citation & frontmatter validation
"""

import pathlib

from cosmoverse.datasets.camels_spectra import CAMELSHydroSpectraConnector
from cosmoverse.datasets.remote import CAMELSArchiveFetcher, CheckpointManager
from cosmoverse.observations.three_x_two_pt import Joint3x2ptObservationalPipeline
from cosmoverse.validation.arxiv_packager import ArxivSubmissionPackager


def test_joint_3x2pt_observational_pipeline_and_degeneracy_breaking():
    pipeline = Joint3x2ptObservationalPipeline()
    vec = pipeline.compute_joint_data_vector(b1=1.45, shear_m=0.01)
    assert len(vec["C_ell_kk"]) == 10
    assert len(vec["C_ell_gk"]) == 10
    assert len(vec["C_ell_gg"]) == 10
    assert len(vec["joint_data_vector"]) == 60

    deg = pipeline.evaluate_probe_degeneracy_breaking()
    assert deg["passes_joint_degeneracy_breaking_gate"] is True
    assert deg["fom_gain_vs_rsd_only"] > 1.8


def test_checkpoint_manager_and_camels_archive_fetcher(tmp_path: pathlib.Path):
    ckpt_dir = tmp_path / "ckpt"
    manifest = CheckpointManager.train_and_save_checkpoint(
        ckpt_dir, grid_size=16, n_sbi_sims=24, seed=42
    )
    assert pathlib.Path(manifest["checkpoint_file"]).exists()
    loaded = CheckpointManager.verify_and_load_checkpoint(ckpt_dir)
    assert loaded["is_verified"] is True
    assert loaded["sha256"] == manifest["sha256"]

    cache_dir = tmp_path / "camels_cache"
    cache_res = CAMELSArchiveFetcher.fetch_and_cache_public_spectra(cache_dir, n_per_suite=2)
    assert cache_res["n_files_cached"] == 6

    # Verify CAMELSHydroSpectraConnector loads directly from the cached disk table
    from cosmoverse.datasets.camels_gen2 import CAMELSGen2Catalog

    first_rec = CAMELSGen2Catalog(family_id="CAMELS-IllustrisTNG", seed=42).get_subset("LH")[0]
    connector = CAMELSHydroSpectraConnector(data_dir=cache_dir)
    spec = connector.load_or_compute_spectrum(first_rec)
    assert spec["source"] == "disk_table"


def test_joss_paper_audit():
    packager = ArxivSubmissionPackager()
    audit = packager.audit_manuscript_integrity()
    assert audit["is_valid"] is True
    assert "joss" in audit["cited_by_paper"]
    assert audit["missing_citations"]["joss"] == []
