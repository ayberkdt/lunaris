from __future__ import annotations

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
