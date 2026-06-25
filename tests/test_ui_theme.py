"""
Tests for the "Lunar Graphite" UI theme.

These guard the calm, mission-control redesign of the desktop UI:

- ``THEME`` (Qt widgets) and ``ORBIT_THEME`` (OpenGL orbit preview) expose the
  required token keys,
- the color helper functions behave correctly,
- the global stylesheet builds without error and routes through the palette,
- the old neon-cyan / violet / champagne-gold accents are gone,
- the orbit preview still constructs when OpenGL is unavailable,
- ``lunaris.ui.app`` imports cleanly.

The tests are deliberately lightweight: anything that needs a widget forces the
Qt ``offscreen`` platform so they run headless in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from lunaris.ui.core.ui_commons import (
    LOG_COLORS,
    ORBIT_THEME,
    THEME,
    hex_to_rgba_float,
    rgba_css_to_tuple,
    with_alpha,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "src" / "lunaris" / "ui"


# ---------------------------------------------------------------------------
# 1) Token dictionaries
# ---------------------------------------------------------------------------

# Single compact naming surface (bg_*/fg_*/role). THEME == ColorTokens.as_dict(),
# so these are exactly the token field names; there is no parallel alias surface.
THEME_REQUIRED_KEYS = [
    "bg_space", "bg_shell", "bg_card", "bg_card_alt", "bg_inset", "bg_entry",
    "bg_log", "bg_hover", "bg_overlay",
    "fg_main", "fg_soft", "fg_muted", "fg_disabled", "fg_inverse", "fg_link",
    "accent", "accent_hov", "accent_dim", "accent_deep",
    "secondary", "secondary_hov", "secondary_dim",
    "success", "warning", "error", "critical", "info", "inactive",
    "border", "border_soft", "border_strong", "panel_shadow", "grid_color",
]

ORBIT_THEME_REQUIRED_KEYS = [
    "space_bg",
    "moon_dark", "moon_mid", "moon_light",
    "orbit_line", "orbit_glow", "spacecraft",
    "periapsis", "apoapsis",
    "orbit_plane",
    "axis_x", "axis_y", "axis_z",
]

LOG_COLORS_REQUIRED_KEYS = [
    "error", "warning", "success", "system", "info", "debug", "timestamp", "default",
]


@pytest.mark.parametrize("key", THEME_REQUIRED_KEYS)
def test_theme_has_required_keys(key: str) -> None:
    assert key in THEME, f"THEME is missing required key {key!r}"


@pytest.mark.parametrize("key", ORBIT_THEME_REQUIRED_KEYS)
def test_orbit_theme_has_required_keys(key: str) -> None:
    assert key in ORBIT_THEME, f"ORBIT_THEME is missing required key {key!r}"


@pytest.mark.parametrize("key", LOG_COLORS_REQUIRED_KEYS)
def test_log_colors_has_required_keys(key: str) -> None:
    assert key in LOG_COLORS, f"LOG_COLORS is missing required key {key!r}"


def test_theme_accent_is_lunar_graphite_blue() -> None:
    # The Lunar Graphite primary accent is a restrained orbital blue, not the old
    # ion-cyan. It was refined to the lighter, calmer #6AA9FF (less edge vibration
    # on the graphite canvas, higher contrast for dark text on the accent fill);
    # the teal secondary stays split from the success state color.
    assert THEME["accent"].lower() == "#6aa9ff"
    assert THEME["secondary"].lower() == "#15d6a6"


def test_theme_has_no_legacy_neon_or_gold() -> None:
    """No old neon-cyan / violet / champagne-gold tokens survive in the palette."""
    banned = {
        "#35d0ff", "#7ce7ff",          # ion cyan
        "#8b7cff", "#b0a7ff",          # violet
        "#b9975b", "#c9aa71", "#8b6b3a",  # champagne gold
        "#f4efe6", "#ddd2bd", "#a89d8c",  # cream/tan
    }
    values = {v.lower() for v in THEME.values()}
    leaked = banned & values
    assert not leaked, f"Legacy neon/gold tokens still in THEME: {leaked}"
    # Old cyan/violet rgb() triples must not appear inside rgba(...) tokens either.
    joined = " ".join(THEME.values()).lower()
    assert "53,208,255" not in joined.replace(" ", "")
    assert "139,124,255" not in joined.replace(" ", "")


# ---------------------------------------------------------------------------
# 2) Color helper functions
# ---------------------------------------------------------------------------

def test_hex_to_rgba_float_basic() -> None:
    assert hex_to_rgba_float("#FFFFFF") == (1.0, 1.0, 1.0, 1.0)
    assert hex_to_rgba_float("#000000", 0.5) == (0.0, 0.0, 0.0, 0.5)
    r, g, b, a = hex_to_rgba_float("#6AA9FF")
    assert round(r, 3) == round(106 / 255, 3)
    assert round(g, 3) == round(169 / 255, 3)
    assert b == 1.0 and a == 1.0


def test_hex_to_rgba_float_shorthand_and_clamp() -> None:
    assert hex_to_rgba_float("#fff") == (1.0, 1.0, 1.0, 1.0)
    # Alpha is clamped into [0, 1].
    assert hex_to_rgba_float("#000000", 5.0)[3] == 1.0
    assert hex_to_rgba_float("#000000", -1.0)[3] == 0.0


def test_rgba_css_to_tuple_accepts_hex_and_rgba() -> None:
    assert rgba_css_to_tuple("#FF0000") == (1.0, 0.0, 0.0, 1.0)
    r, g, b, a = rgba_css_to_tuple("rgba(255, 0, 0, 0.5)")
    assert (r, g, b, a) == (1.0, 0.0, 0.0, 0.5)
    # The ORBIT_THEME axis tokens must be parseable for OpenGL.
    for key in ("axis_x", "axis_y", "axis_z", "orbit_plane"):
        parsed = rgba_css_to_tuple(ORBIT_THEME[key])
        assert len(parsed) == 4
        assert all(0.0 <= c <= 1.0 for c in parsed)


def test_with_alpha_emits_css_rgba() -> None:
    assert with_alpha("#6AA9FF", 0.5) == "rgba(106, 169, 255, 0.5)"
    # Accepts an existing rgba() token and rewrites its alpha.
    assert with_alpha("rgba(106,169,255,0.12)", 0.3) == "rgba(106, 169, 255, 0.3)"


@pytest.mark.parametrize("bad", ["", "not-a-color", "#12", "#xyzxyz", "hsl(1,2,3)"])
def test_color_helpers_reject_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        rgba_css_to_tuple(bad)


def test_every_orbit_theme_token_is_parseable() -> None:
    for key, token in ORBIT_THEME.items():
        parsed = rgba_css_to_tuple(token)
        assert len(parsed) == 4, f"ORBIT_THEME[{key!r}] = {token!r} did not parse"


# ---------------------------------------------------------------------------
# 3) Stylesheet generation
# ---------------------------------------------------------------------------

def test_build_app_stylesheet_smoke() -> None:
    from lunaris.ui.theme import build_app_stylesheet

    qss = build_app_stylesheet(THEME, LOG_COLORS)
    assert isinstance(qss, str) and len(qss) > 2000
    # Key object-name selectors must survive the extraction.
    for selector in ("QMainWindow", "QPushButton#primaryBtn", "QListWidget#navDrawer",
                     "QTextEdit#log", "QLabel#statusBadge"):
        assert selector in qss, f"stylesheet missing selector {selector!r}"


def test_build_app_stylesheet_has_no_legacy_neon() -> None:
    from lunaris.ui.theme import build_app_stylesheet

    qss = build_app_stylesheet(THEME, LOG_COLORS).lower().replace(" ", "")
    assert "#35d0ff" not in qss and "#8b7cff" not in qss
    assert "53,208,255" not in qss and "139,124,255" not in qss
    # The new orbital-blue accent should be present (as hex and/or rgb triple).
    assert "#6aa9ff" in qss or "106,169,255" in qss


def test_build_app_stylesheet_reserves_gradients_for_primary_action() -> None:
    """Gradients are reserved for the primary action, not every surface."""
    from lunaris.ui.theme import build_app_stylesheet

    qss = build_app_stylesheet(THEME, LOG_COLORS)
    # A calm theme should use very few gradients in the global sheet.
    assert qss.count("qlineargradient") <= 2


# ---------------------------------------------------------------------------
# 4) Source-level cleanup guarantees
# ---------------------------------------------------------------------------

def test_orbit_page_has_no_champagne_gold_selected_state() -> None:
    src = (UI_DIR / "pages" / "orbit_config_page.py").read_text(encoding="utf-8")
    # The old segmented-control "checked" state used champagne-gold rgba.
    assert "185, 151, 91" not in src
    assert "185,151,91" not in src
    assert "#b9975b" not in src.lower()


def test_app_py_delegates_stylesheet_generation() -> None:
    src = (UI_DIR / "app.py").read_text(encoding="utf-8")
    # app.py should orchestrate, not host the giant QSS block.
    assert "build_app_stylesheet" in src
    assert "GLOBAL FOUNDATION" not in src, "Large inline QSS block still lives in app.py"
    # No raw legacy neon literals left behind in app.py.
    assert "#35D0FF" not in src and "#8B7CFF" not in src
    assert "53,208,255" not in src.replace(" ", "")


def test_no_raw_legacy_accent_in_pages() -> None:
    """No page-local raw cyan/violet rgba accents remain."""
    page_files = list((UI_DIR / "pages").glob("*.py")) + [UI_DIR / "launcher.py"]
    for path in page_files:
        text = path.read_text(encoding="utf-8").replace(" ", "").lower()
        assert "53,208,255" not in text, f"{path.name} still uses raw cyan rgba"
        assert "139,124,255" not in text, f"{path.name} still uses raw violet rgba"
        assert "#35d0ff" not in text, f"{path.name} still uses raw cyan hex"
        assert "#8b7cff" not in text, f"{path.name} still uses raw violet hex"


# ---------------------------------------------------------------------------
# 5) Orbit preview construction (incl. no-OpenGL fallback)
# ---------------------------------------------------------------------------

def _qapp():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_orbit_viz_constructs_with_opengl_unavailable(monkeypatch) -> None:
    app = _qapp()
    from lunaris.ui.pages import orbit_config_page as ocp

    # Pretend OpenGL is unavailable: the widget must still build a fallback card
    # and never raise, so the mission window can always open.
    monkeypatch.setattr(ocp, "HAS_OPENGL", False)
    viz = ocp.OrbitViz3D()
    try:
        assert viz.gl_widget is None
        # These must be safe no-ops without a GL widget.
        viz.set_orbit_params(2000.0, 0.1, 45.0, 0.0, 0.0, 30.0)
        viz.update_orbit()
        viz.reset_view()
        # A designed empty-state card should be present.
        from PySide6 import QtWidgets

        frames = viz.findChildren(QtWidgets.QFrame)
        # The bespoke fallback was migrated to the shared EmptyState primitive.
        assert any(f.objectName() == "emptyState" for f in frames)
        labels = [w.text() for w in viz.findChildren(QtWidgets.QLabel)]
        assert any("unavailable" in t.lower() for t in labels)
    finally:
        viz.deleteLater()
    _ = app


def test_orbit_viz_marker_radius_is_clamped(monkeypatch) -> None:
    app = _qapp()
    from lunaris.ui.pages import orbit_config_page as ocp

    monkeypatch.setattr(ocp, "HAS_OPENGL", False)
    viz = ocp.OrbitViz3D()
    try:
        # Tiny orbit clamps up to the floor; huge orbit clamps to the ceiling.
        viz.set_orbit_params(ocp.R_MOON + 50.0, 0.0, 90.0, 0.0, 0.0, 0.0)
        assert 12.0 <= viz._marker_radius_km() <= 40.0
        viz.set_orbit_params(500000.0, 0.5, 90.0, 0.0, 0.0, 0.0)
        assert viz._marker_radius_km() == 40.0
    finally:
        viz.deleteLater()
    _ = app


def test_orbit_page_metric_strip_has_all_chips() -> None:
    app = _qapp()
    from lunaris.ui.pages.orbit_config_page import OrbitPage

    page = OrbitPage()
    try:
        for attr in ("lbl_period", "lbl_hp", "lbl_ha", "lbl_ecc", "lbl_inc", "lbl_energy"):
            assert hasattr(page, attr), f"OrbitPage missing metric label {attr!r}"
    finally:
        page.deleteLater()
    _ = app


# ---------------------------------------------------------------------------
# 6) Import sanity
# ---------------------------------------------------------------------------

def test_ui_app_imports() -> None:
    import importlib

    mod = importlib.import_module("lunaris.ui.app")
    assert hasattr(mod, "MainWindow")


def test_app_name_renamed_away_from_st_lrps() -> None:
    from lunaris.ui.core.ui_commons import APP_NAME, WINDOW_SETTINGS

    assert APP_NAME == "Lunaris Mission Studio"
    assert APP_NAME != "ST-LRPS Studio"
    # The window title tracks the visible app name.
    assert WINDOW_SETTINGS["title"] == APP_NAME
