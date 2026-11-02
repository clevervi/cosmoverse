"""Static checks for the immersive 'EL UNIVERSO' WebGL page.

The page is intentionally NOT a dashboard: it must contain no controls,
telemetry panels or external assets, only the full-screen cosmic-web view.
"""

import pathlib
import re

PAGE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "universo.html"


def _html() -> str:
    assert PAGE.is_file(), f"missing universe page: {PAGE}"
    return PAGE.read_text(encoding="utf-8")


def test_universo_is_a_standalone_fullscreen_document():
    assert PAGE.is_file()
    raw = PAGE.read_bytes()
    assert raw[:15] == b"<!DOCTYPE html>"
    assert len(raw) < 96 * 1024
    html = raw.decode("utf-8")
    assert html.rstrip().endswith("</html>")
    assert '<canvas id="universo-canvas"' in html
    assert "EL UNIVERSO" in html
    assert "arrastra" in html


def test_universo_has_layered_webgl_shaders_and_render_loop():
    html = _html()
    assert '<script id="vertex-shader" type="x-shader/x-vertex">' in html
    assert '<script id="fragment-shader" type="x-shader/x-fragment">' in html
    assert '<script id="quad-vertex-shader" type="x-shader/x-vertex">' in html
    assert '<script id="bright-fragment-shader" type="x-shader/x-fragment">' in html
    assert '<script id="blur-fragment-shader" type="x-shader/x-fragment">' in html
    assert '<script id="volume-fragment-shader" type="x-shader/x-fragment">' in html
    assert '<script id="composite-fragment-shader" type="x-shader/x-fragment">' in html
    assert "gl_Position" in html
    assert "gl_PointCoord" in html
    assert "gl_PointSize" in html
    assert "gl.createProgram" in html
    assert "requestAnimationFrame" in html
    assert "gl.drawArrays" in html
    assert "createFramebuffer" in html
    assert "webglcontextlost" in html
    assert 'getContext("webgl2"' in html
    assert "sampler3D" in html


def test_universo_has_layers_bloom_and_cinematic_camera():
    html = _html()
    match = re.search(r"var TARGET_POINTS\s*=\s*(\d+);", html)
    assert match is not None
    assert int(match.group(1)) >= 150000

    assert "uEvolution" in html
    assert "aPositionEarly" in html
    assert "uShowDM" in html
    assert "uShowGas" in html
    assert "uShowGal" in html
    assert "uThreshold" in html
    assert "uBloomStrength" in html
    assert "dblclick" in html
    assert "pointerdown" in html
    assert "pointermove" in html
    assert "wheel" in html
    assert "pinchStartDistance" in html
    assert "prefers-reduced-motion" in html
    assert "visibilitychange" in html
    assert "qualityStage" in html
    assert 'id="pause"' in html
    assert 'aria-label="Pausar movimiento automático"' in html
    assert "togglePause" in html
    assert "ArrowLeft" in html
    assert 'id="tour"' in html
    assert 'id="tour-caption"' in html
    assert "tourStops" in html
    assert "startTour" in html
    assert "uVolume" in html
    assert "volumeTexture" in html
    assert "uCameraPos" in html


def test_universo_documents_the_three_scientific_layers():
    html = _html()
    for label in ("materia oscura", "gas", "galaxias"):
        assert label in html
    assert "SIMULACIÓN ILUSTRATIVA" in html
    assert "isMobile" in html
    assert "maxDpr" in html
    assert "bloomScale" in html


def test_universo_has_no_dashboard_controls_or_telemetry():
    html = _html()
    assert "<input" not in html.lower()
    for forbidden in (
        'type="range"',
        "FPS",
        "SIMBA",
        "coverage",
        "dashboard",
        "omega-m",
        "toggle-posterior",
    ):
        assert forbidden not in html, f"unexpected dashboard token: {forbidden}"


def test_universo_has_no_external_assets_or_network_calls():
    html = _html()
    assert not re.search(r"https?://", html, re.IGNORECASE)
    assert not re.search(r"<link\b", html, re.IGNORECASE)
    assert not re.search(r"<img\b", html, re.IGNORECASE)
    assert not re.search(r"@import\b", html, re.IGNORECASE)
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
