"""Static checks for the standalone WebGL COSMOVERSE dashboard asset.

The dashboard is a single self-contained HTML file with raw WebGL, so these
tests validate the shipped markup and script markers without needing a browser.
"""

import pathlib
import re

DASHBOARD = pathlib.Path(__file__).resolve().parents[1] / "docs" / "webgl_cosmic_web.html"


def _html() -> str:
    assert DASHBOARD.is_file(), f"missing dashboard asset: {DASHBOARD}"
    return DASHBOARD.read_text(encoding="utf-8")


def _input_tag(html: str, element_id: str) -> str:
    pattern = rf'<input\b[^>]*\bid="{re.escape(element_id)}"[^>]*>'
    match = re.search(pattern, html)
    assert match is not None, f"missing input element #{element_id}"
    return match.group(0)


def test_dashboard_is_a_standalone_html_document():
    assert DASHBOARD.is_file()
    raw = DASHBOARD.read_bytes()
    assert raw[:15] == b"<!DOCTYPE html>"
    assert len(raw) < 64 * 1024

    html = raw.decode("utf-8")
    assert html.rstrip().endswith("</html>")
    assert '<canvas id="cosmic-canvas">' in html


def test_inline_vertex_and_fragment_shader_sources_are_present():
    html = _html()
    assert '<script id="vertex-shader" type="x-shader/x-vertex">' in html
    assert '<script id="fragment-shader" type="x-shader/x-fragment">' in html
    assert "gl_Position" in html
    assert "gl_FragColor" in html
    assert "gl_PointCoord" in html
    assert "gl_PointSize" in html


def test_four_parameter_sliders_and_posterior_toggle_are_present():
    html = _html()
    for element_id in ("omega-m", "sigma8", "b1", "a-agn"):
        tag = _input_tag(html, element_id)
        assert 'type="range"' in tag, f"#{element_id} is not a range input"

    toggle = _input_tag(html, "toggle-posterior")
    assert 'type="checkbox"' in toggle


def test_simba_calibration_summary_is_rendered():
    html = _html()
    assert "SIMBA calibrated-response demo" in html
    for token in ("54%", "57%", "64%", "93%", "2.32", "0.68", "PASS"):
        assert token in html, f"missing calibration token {token}"
    assert "ADAPTATION GATE" in html


def test_no_external_assets_or_network_calls():
    html = _html()
    assert not re.search(r"https?://", html, re.IGNORECASE)
    assert not re.search(r'(?:src|href)\s*=\s*["\']\s*(?:https?:)?//', html, re.IGNORECASE)
    assert "src=\"http" not in html
    assert not re.search(r"<link\b", html, re.IGNORECASE)
    assert not re.search(r"<img\b", html, re.IGNORECASE)
    assert not re.search(r"@import\b", html, re.IGNORECASE)
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html


def test_webgl_render_loop_and_camera_controls_are_wired():
    html = _html()
    assert "requestAnimationFrame" in html
    assert "gl.createProgram" in html
    assert "gl.createShader" in html
    assert "gl.drawArrays" in html
    assert "webglcontextlost" in html
    assert "webglcontextrestored" in html

    assert 'id="reset-view"' in html
    assert "pointerdown" in html
    assert "pointermove" in html
    assert "wheel" in html
