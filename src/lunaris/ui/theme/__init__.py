# -*- coding: utf-8 -*-
"""
Lunar Graphite theme package.

This package keeps the large global QSS generation out of ``lunaris.ui.app`` so
the main window stays an orchestration layer rather than a design-token dump.

Public API
----------
``build_app_stylesheet(theme, log_colors)``
    Return the full application-level Qt Style Sheet string for the given
    ``THEME`` / ``LOG_COLORS`` palettes (see ``lunaris.ui.core.ui_commons``).
"""

from __future__ import annotations

from lunaris.ui_foundation.tokens import (
    DESIGN_TOKENS,
    ColorTokens,
    ControlMetrics,
    DesignTokens,
    LayoutTokens,
    RadiusTokens,
    SpacingTokens,
    TypographyTokens,
    VisualizationTokens,
)


def build_app_stylesheet(theme, log_colors):
    """Import the Qt-facing stylesheet builder only when it is requested."""
    from lunaris.ui_foundation.stylesheet import build_app_stylesheet as _build

    return _build(theme, log_colors)


__all__ = [
    "build_app_stylesheet",
    "DESIGN_TOKENS",
    "ColorTokens",
    "ControlMetrics",
    "DesignTokens",
    "LayoutTokens",
    "RadiusTokens",
    "SpacingTokens",
    "TypographyTokens",
    "VisualizationTokens",
]
