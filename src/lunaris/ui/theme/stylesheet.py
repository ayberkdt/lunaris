"""Compatibility export for the binding-neutral stylesheet builder.

Do NOT add QSS here. The single source of truth is
``lunaris.ui_foundation.stylesheet``; this module only re-exports it. See
docs/UI_THEME.md.
"""

from lunaris.ui_foundation.stylesheet import build_app_stylesheet

__all__ = ["build_app_stylesheet"]
