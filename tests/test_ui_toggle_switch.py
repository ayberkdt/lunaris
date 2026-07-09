"""Keyboard-accessibility tests for the custom ToggleSwitch widget (Phase 1)."""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

QtTest = pytest.importorskip('PySide6.QtTest')

import pytest

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from lunaris.ui.core.ui_commons import ToggleSwitch

    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False


def _app():
    return QApplication.instance() or QApplication([])


def test_toggle_switch_is_keyboard_focusable() -> None:
    _app()
    sw = ToggleSwitch()
    # Must be reachable by Tab and able to hold keyboard focus.
    assert sw.focusPolicy() == Qt.StrongFocus
    # Icon-only paint control needs a non-empty accessible name by default.
    assert sw.accessibleName()


def test_toggle_switch_space_and_enter_toggle() -> None:
    _app()
    sw = ToggleSwitch()
    sw.show()
    sw.setFocus()
    assert sw.isChecked() is False

    QTest.keyClick(sw, Qt.Key_Space)
    assert sw.isChecked() is True  # Space toggles (QAbstractButton built-in)

    QTest.keyClick(sw, Qt.Key_Return)
    assert sw.isChecked() is False  # Enter toggles (added in keyPressEvent)

    sw.close()


def test_prefers_reduced_motion_reads_setting() -> None:
    from PySide6.QtCore import QSettings

    from lunaris.ui.core.ui_commons import prefers_reduced_motion

    settings = QSettings("Lunaris", "MissionStudio")
    previous = settings.value("ui/reduce_motion")
    try:
        settings.setValue("ui/reduce_motion", True)
        settings.sync()
        assert prefers_reduced_motion() is True
        settings.setValue("ui/reduce_motion", False)
        settings.sync()
        assert prefers_reduced_motion() is False
    finally:
        if previous is None:
            settings.remove("ui/reduce_motion")
        else:
            settings.setValue("ui/reduce_motion", previous)
        settings.sync()
