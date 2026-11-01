"""
Unit tests for SimulationBudgetAnalyzer (`cosmoverse.validation.budget`) and
ArxivSubmissionPackager (`cosmoverse.validation.arxiv_packager`).
"""

import pathlib
import tarfile

from cosmoverse.validation.arxiv_packager import ArxivSubmissionPackager
from cosmoverse.validation.budget import SimulationBudgetAnalyzer


def test_simulation_budget_scaling_law():
    analyzer = SimulationBudgetAnalyzer(seed=42)
    res = analyzer.evaluate_budget_scaling_curve()
    assert res["min_budget_for_emulator_1pct_mape"] == 8
    assert res["min_budget_for_stage4_calibrated_posteriors"] == 16
    assert all(pt["is_leakage_free"] for pt in res["curve"])
    # Verify monotonic bias improvement with larger simulation budget
    biases = [pt["hybrid_bias_stage4_sigma"] for pt in res["curve"]]
    assert biases == sorted(biases, reverse=True)


def test_arxiv_submission_packager(tmp_path: pathlib.Path):
    packager = ArxivSubmissionPackager()
    audit = packager.audit_manuscript_integrity()
    assert audit["is_valid"] is True
    assert len(audit["leaked_blocked_keys"]) == 0
    assert audit["missing_citations"]["paper1"] == []
    assert audit["missing_citations"]["paper2"] == []

    bundles = packager.build_arxiv_bundles(out_dir=tmp_path / "arxiv")
    p1_tar = pathlib.Path(bundles["paper1_bundle"])
    p2_tar = pathlib.Path(bundles["paper2_bundle"])
    assert p1_tar.exists() and p2_tar.exists()

    with tarfile.open(p1_tar, "r:gz") as tf:
        names = tf.getnames()
        assert "main.tex" in names
        assert "references.bib" in names
        assert "00README.txt" in names
