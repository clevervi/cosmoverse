"""
Unit tests for COSMOVERSE CLI subcommands (info, frontier, camels, field-level, audit-citations).
"""

from cosmoverse import __version__
from cosmoverse.cli import main


def test_cli_version_and_subcommands():
    assert __version__ == "0.3.0"
    main(["info"])
    main(["frontier"])
    main(["camels"])
    main(["field-level", "--grid", "4", "--samples", "20"])
    main(["audit-citations"])
    main(["benchmark", "--n", "4"])
