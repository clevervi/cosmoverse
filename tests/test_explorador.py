"""Static checks for the single-page multi-scale cosmic explorer."""

import base64
import pathlib
import re

PAGE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "explorador.html"
CAMELS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "universo_camels.js"
SDSS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "explorador_sdss.js"


def _html() -> str:
    assert PAGE.is_file(), f"missing explorer page: {PAGE}"
    return PAGE.read_text(encoding="utf-8")


def test_explorer_is_a_single_page_with_six_scales():
    raw = PAGE.read_bytes()
    assert raw[:15] == b"<!DOCTYPE html>"
    assert len(raw) < 200 * 1024
    html = raw.decode("utf-8")
    assert html.rstrip().endswith("</html>")
    assert '<canvas id="space"' in html
    for name in (
        "Tierra",
        "Sistema Solar",
        "Vía Láctea",
        "Grupo Local",
        "Laniakea",
        "Red cósmica",
    ):
        assert name in html
    assert 'id="rail"' in html
    assert 'id="address"' in html


def test_explorer_has_all_scene_builders_and_interactions():
    html = _html()
    for marker in (
        "buildEarth",
        "buildSolar",
        "buildMilkyWay",
        "buildLocalGroup",
        "buildLaniakea",
        "buildCosmic",
        "activateLevel",
        "focusObject",
        "pickObjectAt",
        "requestAnimationFrame",
        "pointerdown",
        "pointermove",
        "wheel",
        "dblclick",
        "pinch",
        "prefers-reduced-motion",
    ):
        assert marker in html
    assert 'getContext("webgl2"' in html
    assert "uniform mat4 uModel" in html


def test_explorer_has_search_info_and_fast_navigation():
    html = _html()
    for marker in (
        'id="search-btn"',
        'id="search-modal"',
        'id="search-input"',
        'id="search-results"',
        'id="info"',
        'id="info-focus"',
        'id="about-modal"',
        "openSearch",
        "renderSearch",
        "selectObject",
        "key >= \"1\" && e.key <= \"6\"",
        "mapoftheuniverse.net",
        "universemap.net",
    ):
        assert marker in html


def test_explorer_uses_camels_cosmic_web_data():
    html = _html()
    assert '<script src="universo_camels.js"></script>' in html
    assert "COSMOVerseCAMELS" in html
    assert "density" in html
    assert "temperatura" in html
    assert CAMELS.is_file()
    raw = CAMELS.read_text(encoding="utf-8")
    match = re.search(r"size:\s*(\d+)", raw)
    data = re.search(r'data:\s*"([^"]+)"', raw)
    assert match is not None
    assert data is not None
    size = int(match.group(1))
    assert len(base64.b64decode(data.group(1))) == 2 * size ** 3


def test_explorer_has_real_sdss_toggle_and_guided_tour():
    html = _html()
    assert '<script src="explorador_sdss.js"></script>' in html
    assert "COSMOVerseSDSS" in html
    assert "buildSDSS" in html
    assert "sdssPicking" in html
    assert 'id="data-btn"' in html
    assert 'id="tour-btn"' in html
    assert 'id="tour-card"' in html
    assert "TOUR_STOPS" in html
    assert "aliases" in html
    assert "Galaxia SDSS" in html
    assert SDSS.is_file()
    raw = SDSS.read_text(encoding="utf-8")
    count = re.search(r"count:\s*(\d+)", raw)
    data = re.search(r'data:\s*"([^"]+)"', raw)
    assert count is not None
    assert data is not None
    assert "SDSS" in raw
    assert len(base64.b64decode(data.group(1))) == int(count.group(1)) * 11
    assert len(raw) < 3 * 1024 * 1024


def test_explorer_has_no_external_assets_or_dashboards():
    html = _html()
    assert not re.search(r'<script[^>]+src="https?://', html, re.IGNORECASE)
    assert not re.search(r'<img\b', html, re.IGNORECASE)
    assert not re.search(r"<link\b", html, re.IGNORECASE)
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert "<input" in html.lower()
    assert 'type="search"' in html
    assert 'type="range"' not in html
    assert "FPS" not in html
