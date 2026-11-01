"""
Unit tests for Paper1FigureGenerator (`cosmoverse.validation.plotting`).
"""

import pathlib

from cosmoverse.validation.plotting import Paper1FigureGenerator


def test_paper1_figure_generator_writes_all_four_svgs(tmp_path: pathlib.Path):
    saved = Paper1FigureGenerator.generate_all_paper1_figures(tmp_path)
    assert len(saved) == 4
    for key, path in saved.items():
        assert path.exists(), f"Missing {key}"
        content = path.read_text(encoding="utf-8")
        assert content.startswith("<svg ")
        assert "</svg>" in content
