"""Static checks for the immersive 'EL UNIVERSO' WebGL page.

The page is intentionally NOT a dashboard: it must contain no controls,
telemetry panels or external assets, only the full-screen cosmic-web view.
"""

import base64
import pathlib
import re

PAGE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "universo.html"
DENSITY = pathlib.Path(__file__).resolve().parents[1] / "docs" / "universo_camels.js"


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

    assert "uShowDM" in html
    assert "uShowGas" in html
    assert "uShowGal" in html
    assert "COSMOVerseCAMELS" in html
    assert "densityField" in html
    assert "temperatureField" in html
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
    for label in ("gas density", "temperatura", "galaxias"):
        assert label in html
    assert "CAMELS-SIMBA" in html
    assert "EX 96³" in html
    assert "z = 0" in html
    assert "isMobile" in html
    assert "maxDpr" in html
    assert "bloomScale" in html


def test_universo_uses_the_embedded_camels_gas_field():
    assert DENSITY.is_file(), f"missing CAMELS field: {DENSITY}"
    raw = DENSITY.read_text(encoding="utf-8")
    assert raw.startswith("window.COSMOVerseCAMELS")
    size_match = re.search(r"size:\s*(\d+)", raw)
    data_match = re.search(r'data:\s*"([^"]+)"', raw)
    assert size_match is not None
    assert data_match is not None
    assert "SIMBA" in raw
    assert "Mgas" in raw
    size = int(size_match.group(1))
    assert size >= 80
    decoded = base64.b64decode(data_match.group(1))
    assert len(decoded) == 2 * size ** 3
    assert len(raw) < 6 * 1024 * 1024


def test_universo_has_no_dashboard_controls_or_telemetry():
    html = _html()
    assert "<input" not in html.lower()
    for forbidden in (
        'type="range"',
        "FPS",
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
