"""Shared application-font loading for both Lunaris desktop apps.

Mission Studio and ST-LRPS Studio must resolve the same typography stack.
The loader registers bundled and platform font files with Qt's font
database (which also makes real glyphs available under the ``offscreen``
platform used for capture verification) and returns the preferred UI font
sized from the typography tokens.

This module imports Qt, so it is intentionally NOT re-exported from
``lunaris.ui_foundation.__init__`` — the token/stylesheet surface stays
importable without a Qt installation.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6 import QtGui

from lunaris.ui_foundation.tokens import DESIGN_TOKENS

# Fallback chain tried against the font database after registration. The
# first entry mirrors the stylesheet's ``family_ui`` token so QSS and the
# QApplication font agree.
PREFERRED_FAMILIES: tuple[str, ...] = (
    DESIGN_TOKENS.typography.family_ui.strip('"'),
    "Inter",
    "Noto Sans",
    "Roboto",
    "DejaVu Sans",
    "Arial",
)

_BASE_POINT_SIZE = int(DESIGN_TOKENS.typography.size_body_pt)

# Windows system font files worth registering explicitly: Qt's offscreen
# platform does not enumerate installed system fonts by itself.
_WINDOWS_FONT_FILES = (
    "segoeui.ttf",
    "segoeuib.ttf",
    "segoeuii.ttf",
    "segoeuil.ttf",
    "consola.ttf",
    "consolab.ttf",
    "consolai.ttf",
    "arial.ttf",
)


def load_app_font(fonts_dir: Path | str | None = None) -> QtGui.QFont:
    """Register available font files and return the preferred app font.

    Parameters
    ----------
    fonts_dir : optional directory of bundled ``.ttf``/``.otf``/``.ttc``
        files to register first (e.g. the mission app's ``assets/fonts``).
        ``None`` skips the bundled step and relies on platform fonts.
    """
    loaded_families: list[str] = []

    if fonts_dir is not None:
        fonts_path = Path(fonts_dir)
        if fonts_path.exists():
            for pattern in ("*.ttf", "*.otf", "*.ttc"):
                for font_file in fonts_path.glob(pattern):
                    font_id = QtGui.QFontDatabase.addApplicationFont(str(font_file))
                    if font_id != -1:
                        loaded_families.extend(
                            QtGui.QFontDatabase.applicationFontFamilies(font_id)
                        )

    if os.name == "nt":
        windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for filename in _WINDOWS_FONT_FILES:
            font_file = windows_fonts / filename
            if font_file.exists():
                font_id = QtGui.QFontDatabase.addApplicationFont(str(font_file))
                if font_id != -1:
                    loaded_families.extend(
                        QtGui.QFontDatabase.applicationFontFamilies(font_id)
                    )

    available = set(QtGui.QFontDatabase.families())

    for family in PREFERRED_FAMILIES:
        if family in available:
            return QtGui.QFont(family, _BASE_POINT_SIZE)

    for family in loaded_families:
        if family in available:
            return QtGui.QFont(family, _BASE_POINT_SIZE)

    return QtGui.QFont(PREFERRED_FAMILIES[0], _BASE_POINT_SIZE)
