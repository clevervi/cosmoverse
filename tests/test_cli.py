"""
Unit tests for COSMOVERSE CLI subcommands (info, frontier, camels, field-level, audit-citations).
"""

import json
import pathlib

from cosmoverse import __version__
from cosmoverse.cli import main


def test_cli_version_and_subcommands(tmp_path: pathlib.Path):
    assert __version__ == "1.0.0"
    main(["info"])
    main(["frontier"])
    main(["camels"])
    main(["field-level", "--grid", "4", "--samples", "20"])
    main(["audit-citations"])
    main(["benchmark", "--n", "4"])
    main(["multi-suite"])
    main(["mangrove"])
    main(["budget"])
    main(["dashboard"])
    main(["figures", "--out-dir", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "fig1_result1_emulator.svg").exists()
    assert (tmp_path / "figs" / "fig5_paper2_ic_reconstruction.svg").exists()
    assert (tmp_path / "figs" / "fig6_paper2_mclmc_vs_hmc.svg").exists()
    assert (tmp_path / "figs" / "fig7_simulation_budget_scaling.svg").exists()

    main(["arxiv-bundle", "--out-dir", str(tmp_path / "arxiv")])
    assert (tmp_path / "arxiv" / "cosmoverse_paper1_arxiv.tar.gz").exists()
    assert (tmp_path / "arxiv" / "cosmoverse_paper2_arxiv.tar.gz").exists()

    root = pathlib.Path(__file__).resolve().parents[1]
    assert (root / "docs" / "index.html").exists()
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["version"] == "1.0.0"
    assert zenodo["license"] == "MIT"
