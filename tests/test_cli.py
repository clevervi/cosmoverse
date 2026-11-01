"""
Unit tests for COSMOVERSE CLI subcommands (info, frontier, camels, field-level, audit-citations).
"""

import pathlib

from cosmoverse import __version__
from cosmoverse.cli import main


def test_cli_version_and_subcommands(tmp_path: pathlib.Path):
    assert __version__ == "0.5.0"
    main(["info"])
    main(["frontier"])
    main(["camels"])
    main(["field-level", "--grid", "4", "--samples", "20"])
    main(["audit-citations"])
    main(["benchmark", "--n", "4"])
    main(["figures", "--out-dir", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "fig1_result1_emulator.svg").exists()
