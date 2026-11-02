"""
Unit tests for ProgressDashboard (`cosmoverse.validation.progress_dashboard`).
"""

import json
import pathlib
import re
import webbrowser

from cosmoverse.validation.progress_dashboard import ProgressDashboard

_PAYLOAD_PATTERN = re.compile(
    r'<script type="application/json" id="payload">(.*?)</script>', re.DOTALL
)


def _state(html_text: str) -> dict:
    match = _PAYLOAD_PATTERN.search(html_text)
    assert match is not None, "embedded payload block is missing"
    return json.loads(match.group(1))


def test_update_writes_standalone_html_with_stage_labels_and_payload(tmp_path: pathlib.Path):
    dashboard = ProgressDashboard(tmp_path / "dist" / "progress.html")

    output = dashboard.update(
        "Stage 4: SBI calibration",
        {
            "progress": 0.42,
            "coverage": [
                {"label": "Source", "cov68": 0.66, "cov95": 0.93},
                {"label": "Target", "cov68": 0.61, "cov95": 0.9},
            ],
            "metrics": [
                {"label": "TNG", "median_bias": 1.39, "delta_ice": 0.225, "gate": "FAIL"}
            ],
        },
    )

    assert output == tmp_path / "dist" / "progress.html"
    assert output.exists()
    text = output.read_text(encoding="utf-8")

    assert text.startswith("<!DOCTYPE html>")
    assert "<html" in text and "</head>" in text and "</body>" in text
    assert text.rstrip().endswith("</html>")
    assert "__PAYLOAD__" not in text
    assert '<meta http-equiv="refresh" content="2">' in text
    assert '<script type="application/json" id="payload">' in text
    assert "<script src=" not in text
    assert "<link" not in text

    for expected in ("Stage 4: SBI calibration", "Source", "Target", "TNG"):
        assert expected in text

    state = _state(text)
    assert state["stage"] == "Stage 4: SBI calibration"
    assert state["payload"]["progress"] == 0.42
    assert state["payload"]["coverage"][1]["label"] == "Target"
    assert state["payload"]["metrics"][0]["gate"] == "FAIL"


def test_second_update_replaces_previous_stage_and_labels(tmp_path: pathlib.Path):
    dashboard = ProgressDashboard(tmp_path / "progress.html")
    dashboard.update(
        "Emulator training",
        {"metrics": [{"label": "OLDGROUP", "median_bias": 0.4, "delta_ice": 0.03, "gate": "PASS"}]},
    )

    output = dashboard.update(
        "Transfer sweep",
        {"metrics": [{"label": "NEWGROUP", "median_bias": 0.9, "delta_ice": 0.12, "gate": "FAIL"}]},
    )
    text = output.read_text(encoding="utf-8")

    assert "Transfer sweep" in text
    assert "NEWGROUP" in text
    assert "Emulator training" not in text
    assert "OLDGROUP" not in text
    assert _state(text)["stage"] == "Transfer sweep"


def test_open_launches_file_uri_without_raising(
    tmp_path: pathlib.Path, monkeypatch
):
    dashboard = ProgressDashboard(tmp_path / "progress.html")
    dashboard.update("Stage 1: partitioning", {"progress": 0.05})

    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)

    dashboard.open()

    assert len(opened) == 1
    assert opened[0].startswith("file://")
    assert opened[0].endswith("progress.html")


def test_payload_with_posterior_and_loss_histories_renders(tmp_path: pathlib.Path):
    dashboard = ProgressDashboard(str(tmp_path / "progress.html"))
    output = dashboard.update(
        "Stage 5: field-level posterior",
        {
            "progress": 0.87,
            "loss_history": [2.8, 2.1, 1.6, 1.3, 1.1],
            "fine_tune_loss_history": [1.2, 1.0, 0.9],
            "posterior": {
                "label": "Omega_m vs sigma8",
                "truth": [0.315, 0.811],
                "samples": [[0.30, 0.80], [0.312, 0.815], [0.32, 0.808]],
            },
            "device": {
                "backend": "cupy",
                "gpu_name": "NVIDIA GeForce RTX 3080 Laptop GPU",
            },
            "note": "Coverage gate pending TARP rerun.",
        },
    )

    text = output.read_text(encoding="utf-8")
    assert 'id="loss-chart"' in text
    assert 'id="posterior-chart"' in text
    assert "NVIDIA GeForce RTX 3080 Laptop GPU" in text

    inner = _state(text)["payload"]
    assert inner["loss_history"] == [2.8, 2.1, 1.6, 1.3, 1.1]
    assert inner["fine_tune_loss_history"] == [1.2, 1.0, 0.9]
    assert inner["posterior"]["truth"] == [0.315, 0.811]
    assert len(inner["posterior"]["samples"]) == 3
    assert inner["device"]["backend"] == "cupy"
    assert inner["note"] == "Coverage gate pending TARP rerun."


def test_non_finite_floats_become_null_in_embedded_json(tmp_path: pathlib.Path):
    dashboard = ProgressDashboard(tmp_path / "progress.html")
    output = dashboard.update(
        "Stage 2: emulator training",
        {"loss_history": [1.0, float("nan"), float("inf")]},
    )

    assert _state(output.read_text(encoding="utf-8"))["payload"]["loss_history"] == [1.0, None, None]
