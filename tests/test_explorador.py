"""Static checks for the single-page multi-scale cosmic explorer."""

import base64
import pathlib
import re

PAGE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "explorador.html"
CAMELS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "universo_camels.js"
SDSS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "explorador_sdss.js"
TWOMRS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "explorador_2mrs.js"
CMB = pathlib.Path(__file__).resolve().parents[1] / "docs" / "explorador_cmb.js"
CMB_IMAGE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "explorador_cmb.jpg"


def _html() -> str:
    assert PAGE.is_file(), f"missing explorer page: {PAGE}"
    return PAGE.read_text(encoding="utf-8")


def test_explorer_is_a_single_page_with_seven_scales():
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
        "Universo observable",
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
        "buildObservable",
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
        "key >= \"1\" && e.key <= \"7\"",
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


def test_explorer_has_all_sky_2mrs_and_planck_cmb():
    html = _html()
    assert '<script src="explorador_2mrs.js"></script>' in html
    assert '<script src="explorador_cmb.js"></script>' in html
    assert "COSMOVerse2MRS" in html
    assert "COSMOVerseCMB" in html
    assert "buildObservable" in html
    assert "twomrsPicking" in html
    assert "cmbSphere" in html
    assert "Galaxia 2MRS" in html
    assert "Planck" in html

    assert TWOMRS.is_file()
    raw = TWOMRS.read_text(encoding="utf-8")
    count = re.search(r"count:\s*(\d+)", raw)
    data = re.search(r'data:\s*"([^"]+)"', raw)
    assert count is not None
    assert data is not None
    assert "2MASS Redshift Survey" in raw
    assert "radialNote" in raw
    assert int(count.group(1)) == 43482
    assert len(base64.b64decode(data.group(1))) == int(count.group(1)) * 11
    assert len(raw) < 2 * 1024 * 1024

    assert CMB.is_file()
    cmb_raw = CMB.read_text(encoding="utf-8")
    cmb_data = re.search(r"data:image/jpeg;base64,([^\"]+)", cmb_raw)
    assert cmb_data is not None
    assert len(base64.b64decode(cmb_data.group(1))) > 100 * 1024
    assert CMB_IMAGE.is_file()


def test_explorer_has_free_zoom_transitions_and_black_holes():
    html = _html()
    for marker in (
        'id="fade-canvas"',
        "preserveDrawingBuffer",
        "maybeSwitchLevelByZoom",
        "discoverRandom",
        "info-fidelity",
        "objectTypeLabel",
        "get(\"focus\")",
        "uType > 4.5",
        "Sagitario A*",
        "Cygnus X-1",
        "M87*",
        "TON 618",
        "3C 273",
        "Plutón",
        "Ceres",
        "Eris",
        "Titán",
        "Cinturón de asteroides",
        "Cinturón de Kuiper",
    ):
        assert marker in html
    assert 'id="random-btn"' in html
    assert "dato real" in html
    assert "labelHalf" in html


def test_explorer_has_animated_orbits_and_deep_catalog():
    html = _html()
    for marker in (
        "updateSolarMotion",
        "dynamicPlanets",
        "dynamicMoons",
        "dynamicSaturn",
        "solarTime",
        "Próxima b",
        "TRAPPIST-1e",
        "Púlsar del Cangrejo",
        "1P/Halley",
        "Bennu",
        "Galaxia del Remolino",
        "Centaurus A",
        "Gaia BH1",
        "V404 Cygni",
    ):
        assert marker in html
    assert "orbit: { radius: p.orbit, period: p.period" in html
    assert "focusRadius: .42" in html


def test_explorer_has_local_planet_textures_and_spin():
    html = _html()
    assert '<script src="explorador_planetas.js"></script>' in html
    assert "COSMOVersePlanets" in html
    assert "loadPlanetTextures" in html
    assert "obj.spin" in html
    assert "uSpin" in html
    assert "uNightTexture" in html
    assert "uCloudTexture" in html
    assert "uUseNight" in html
    assert "uUseClouds" in html
    planets_js = pathlib.Path(__file__).resolve().parents[1] / "docs" / "explorador_planetas.js"
    assert planets_js.is_file()
    raw = planets_js.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in raw
    assert "earth:" in raw
    assert "jupiter:" in raw
    assert "earth_clouds:" in raw
    assert "earth_night:" in raw
    assert len(raw) > 200 * 1024
    for name in ("earth", "earth_clouds", "earth_night", "moon", "sun", "saturn"):
        assert (planets_js.parent / "planetas" / f"{name}.jpg").is_file()


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
