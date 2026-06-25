"""Contract tests for the typed desktop design system."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from lunaris.ui.theme.tokens import (
    DESIGN_TOKENS,
    ColorTokens,
    ControlMetrics,
    LayoutTokens,
    RadiusTokens,
    SpacingTokens,
    TypographyTokens,
    VisualizationTokens,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    light, dark = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_typed_token_groups_are_frozen_dataclasses() -> None:
    for token_type in (
        ColorTokens,
        TypographyTokens,
        SpacingTokens,
        RadiusTokens,
        ControlMetrics,
        LayoutTokens,
        VisualizationTokens,
    ):
        assert dataclasses.is_dataclass(token_type)
        assert token_type.__dataclass_params__.frozen is True


def test_core_palette_and_layout_contract() -> None:
    colors = DESIGN_TOKENS.colors
    assert colors.bg_space == "#090C12"
    assert colors.accent == "#3B86FF"
    assert colors.success == "#3DD17E"
    assert DESIGN_TOKENS.layout.nav_width == 216
    assert DESIGN_TOKENS.layout.nav_compact_width < DESIGN_TOKENS.layout.nav_width
    assert DESIGN_TOKENS.layout.console_collapsed_height <= 46
    assert DESIGN_TOKENS.controls.minimum_height >= 34


def test_color_tokens_single_naming_surface_is_complete() -> None:
    colors = DESIGN_TOKENS.colors
    mapping = colors.as_dict()
    # The compact bg_*/fg_*/role names are the single naming surface: every
    # field appears in as_dict() and is non-empty, with no parallel semantic
    # alias surface (surface_*/text_*/divider/as_legacy_dict) to drift apart.
    for name in (
        "bg_space", "bg_shell", "bg_card", "bg_card_alt", "bg_inset",
        "bg_entry", "bg_log", "bg_hover", "bg_overlay",
        "fg_main", "fg_soft", "fg_muted", "fg_disabled", "fg_inverse", "fg_link",
        "accent", "accent_hov", "accent_dim", "accent_deep",
        "secondary", "secondary_hov", "secondary_dim",
        "success", "warning", "error", "critical", "info", "inactive",
        "border", "border_soft", "border_strong", "panel_shadow", "grid_color",
    ):
        assert getattr(colors, name)
        assert mapping[name] == getattr(colors, name)
    assert not hasattr(colors, "surface_app")
    assert not hasattr(colors, "text_primary")
    assert not hasattr(colors, "as_legacy_dict")

    visualization = DESIGN_TOKENS.visualization.as_dict()
    for role in (
        "space_bg",
        "moon_dark",
        "moon_mid",
        "moon_light",
        "orbit_line",
        "spacecraft",
        "terrain",
        "gravity",
        "srp",
        "sun",
        "earth",
        "selected_plot",
        "comparison_plot",
        "grid",
        "axis_text",
    ):
        assert visualization[role]


@pytest.mark.parametrize("surface", ["bg_space", "bg_shell", "bg_card", "bg_entry"])
def test_primary_text_contrast_is_accessible(surface: str) -> None:
    colors = DESIGN_TOKENS.colors
    assert _contrast(colors.fg_main, getattr(colors, surface)) >= 7.0


@pytest.mark.parametrize(
    ("foreground", "background"),
    [
        ("fg_soft", "bg_card"),
        ("fg_muted", "bg_card"),
        ("fg_disabled", "bg_inset"),
        ("fg_inverse", "accent"),
    ],
)
def test_semantic_text_contrast_meets_wcag_aa(foreground: str, background: str) -> None:
    colors = DESIGN_TOKENS.colors
    assert _contrast(getattr(colors, foreground), getattr(colors, background)) >= 4.5


def test_theme_facade_tracks_typed_tokens() -> None:
    pytest.importorskip("PySide6")
    from lunaris.ui.core.ui_commons import THEME

    assert THEME["accent"] == DESIGN_TOKENS.colors.accent
    assert THEME["accent_dim"] == DESIGN_TOKENS.colors.accent_dim
    assert THEME["bg_log"] == DESIGN_TOKENS.colors.bg_log


def test_global_stylesheet_has_shared_component_selectors() -> None:
    pytest.importorskip("PySide6")
    from lunaris.ui.core.ui_commons import LOG_COLORS, THEME
    from lunaris.ui.theme import build_app_stylesheet

    qss = build_app_stylesheet(THEME, LOG_COLORS)
    for selector in (
        "QFrame#pageHeader",
        "QFrame#section",
        "QFrame#inlineNotice",
        "QFrame#toolbar",
        "QLineEdit#compactSearch",
        "QPlainTextEdit#logConsole",
        "QPushButton#navButton",
    ):
        assert selector in qss
    assert "qlineargradient" not in qss


def test_shared_primitives_construct_offscreen() -> None:
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from lunaris.ui.components import (
        CompactSearchField,
        EmptyState,
        FormGrid,
        InlineNotice,
        MetricRow,
        PageShell,
        Section,
        SegmentedControl,
        Toolbar,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    content = Section("Inputs", "Configure the analysis.")
    shell = PageShell("Mission", "Mission setup", content=content)
    widgets = [
        shell,
        FormGrid(),
        MetricRow([("Altitude", "50 km")]),
        InlineNotice("Ready", "success"),
        Toolbar(),
        SegmentedControl(["A", "B"]),
        EmptyState("No results"),
        CompactSearchField(),
    ]
    try:
        assert shell.header.title_label.text() == "Mission"
        assert shell.scroll_area is not None
        assert all(widget.objectName() for widget in widgets)
    finally:
        for widget in widgets:
            widget.deleteLater()
    _ = app


def test_studio_sources_no_longer_contain_legacy_neon_tokens() -> None:
    targets = [
        REPO_ROOT / "src/lunaris/surrogate/st_lrps/ui/dashboard_widgets.py",
        REPO_ROOT / "src/lunaris/surrogate/st_lrps/ui/studio_parts/main_window.py",
        REPO_ROOT / "src/lunaris/surrogate/st_lrps/ui/studio_parts/data_pages.py",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in targets)
    assert "#35d0ff" not in joined
    assert "#8b7cff" not in joined
    assert "{{theme[" not in joined


def test_ui_architecture_documents_exist() -> None:
    for name in ("UI_AUDIT.md", "UI_DESIGN_SYSTEM.md", "UI_PAGE_ARCHITECTURE.md"):
        path = REPO_ROOT / "docs" / name
        assert path.is_file() and path.stat().st_size > 500
