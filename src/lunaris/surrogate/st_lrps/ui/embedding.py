"""Supported Mission-UI entry point for embedding the optional ST-LRPS Studio.

Mission UI code imports this facade instead of reaching into ``studio_parts``.
The facade is intentionally small: Studio internals remain free to move while
the launcher-facing window and theme hook retain one explicit seam.
"""

from __future__ import annotations

from lunaris.surrogate.st_lrps.ui.studio_parts.main_window import MainWindow
from lunaris.surrogate.st_lrps.ui.studio_parts.qt_common import apply_premium_dark_theme

__all__ = ["MainWindow", "apply_premium_dark_theme"]
