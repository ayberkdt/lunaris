from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtCore = pytest.importorskip("PySide6.QtCore")
QtGui = pytest.importorskip("PySide6.QtGui")


def wait_until(
    app: QtWidgets.QApplication,
    predicate: Callable[[], bool],
    timeout_s: float = 2.0,
) -> bool:
    """Drive the Qt event loop until *predicate* holds or the timeout passes.

    Shared async-UI test helper (W5.2): animations, single-shot timers, and
    queued signals need the loop pumped before their end state is assertable.
    Returns the final predicate value so callers can ``assert wait_until(...)``.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


@contextlib.contextmanager
def preserved_app_look(app: QtWidgets.QApplication):
    """Restore the shared QApplication's style, font, palette and stylesheet on exit.

    Tests that apply the Studio theme (``apply_premium_dark_theme``) change the
    one process-wide QApplication; left in place, its font leaks into later
    size-hint layout tests and makes them pass or fail by test order. Only what
    changed is restored, because an app-wide restyle is expensive.
    """
    before = (app.styleSheet(), app.style().name(), app.font(), app.palette())
    try:
        yield app
    finally:
        sheet, style_name, font, palette = before
        if app.style().name() != style_name:
            app.setStyle(style_name)
        if app.font() != font:
            app.setFont(font)
        if app.palette() != palette:
            app.setPalette(palette)
        if app.styleSheet() != sheet:
            app.setStyleSheet(sheet)
