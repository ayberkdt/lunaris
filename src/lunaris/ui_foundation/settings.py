"""Persistent Qt settings for both Lunaris desktop apps.

``QSettings(organization, application)`` always opens the platform's native
store (the registry on Windows), and ``QSettings.setPath`` does not redirect
it. When ``LUNARIS_APP_DATA_DIR`` is set (the test suite always sets it, see
``tests/conftest.py``) every store is a file-backed .ini under that folder
instead, so tests, captures and portable installs never read or write the
user's real preferences.

This module imports Qt, so it is intentionally NOT re-exported from
``lunaris.ui_foundation.__init__``.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6 import QtCore

#: Environment override for tests, captures and portable installs (the same
#: variable Mission Studio already honours for its session and layout files).
APP_DATA_ENV = "LUNARIS_APP_DATA_DIR"


def persistent_settings(organization: str, application: str, ini_name: str) -> QtCore.QSettings:
    """The settings store for one application.

    Native store under *organization*/*application* by default, so existing
    user settings stay reachable; ``<LUNARIS_APP_DATA_DIR>/<ini_name>`` when
    the override is set.
    """
    override = os.environ.get(APP_DATA_ENV, "").strip()
    if override:
        folder = Path(override).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        return QtCore.QSettings(str(folder / ini_name), QtCore.QSettings.Format.IniFormat)
    return QtCore.QSettings(organization, application)


__all__ = ["APP_DATA_ENV", "persistent_settings"]
