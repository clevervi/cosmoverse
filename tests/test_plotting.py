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


def test_figure8_calibration_budget_curve(tmp_path: pathlib.Path):
    curve = [
        {
            "n_cal": 25,
            "coverage_68": 0.49,
            "coverage_95": 0.77,
            "median_bias_in_sigma": 1.00,
            "passes_gate": False,
        },
        {
            "n_cal": 50,
            "coverage_68": 0.58,
            "coverage_95": 0.91,
            "median_bias_in_sigma": 0.72,
            "passes_gate": True,
        },
        {
            "n_cal": 200,
            "coverage_68": 0.67,
            "coverage_95": 0.98,
            "median_bias_in_sigma": 0.68,
            "passes_gate": True,
        },
    ]
    svg = Paper1FigureGenerator.render_figure8_calibration_budget(
        {"curve": curve, "min_calibration_for_gate": 50}
    )
    assert svg.startswith("<svg ")
    assert svg.rstrip().endswith("</svg>")
    assert "N_cal,min = 50" in svg
    assert "Coverage 68%" in svg
    assert "Coverage 95%" in svg
    out = tmp_path / "fig8_calibration_budget.svg"
    out.write_text(svg, encoding="utf-8", newline="\n")
    assert out.exists()
