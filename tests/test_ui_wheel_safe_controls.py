"""Regression tests for wheel-safe form controls (UI redesign M4).

Scrolling a long configuration form must never silently change a value the
pointer happens to hover. ``NoWheelSpinBox`` / ``NoWheelDoubleSpinBox`` /
``NoWheelComboBox`` therefore ignore wheel events (even when focused) while
keeping keyboard stepping intact.
"""

from __future__ import annotations

import pytest

pytest.importorskip('PySide6.QtWidgets')


import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui.core.ui_commons import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wheel(widget: QtWidgets.QWidget, delta: int = 120) -> QtGui.QWheelEvent:
    pos = QtCore.QPointF(widget.rect().center())
    return QtGui.QWheelEvent(
        pos,
        widget.mapToGlobal(widget.rect().center()),
        QtCore.QPoint(0, delta),
        QtCore.QPoint(0, delta),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )


def test_spinbox_wheel_does_not_change_value() -> None:
    _app()
    sb = NoWheelSpinBox()
    sb.setRange(0, 100)
    sb.setValue(10)
    sb.setFocus()
    QtWidgets.QApplication.sendEvent(sb, _wheel(sb, 120))
    QtWidgets.QApplication.sendEvent(sb, _wheel(sb, -120))
    assert sb.value() == 10


def test_double_spinbox_wheel_does_not_change_value() -> None:
    _app()
    sb = NoWheelDoubleSpinBox()
    sb.setRange(0.0, 1.0)
    sb.setSingleStep(0.1)
    sb.setValue(0.5)
    sb.setFocus()
    QtWidgets.QApplication.sendEvent(sb, _wheel(sb, 120))
    assert sb.value() == pytest.approx(0.5)


def test_combobox_wheel_does_not_change_selection() -> None:
    _app()
    cb = NoWheelComboBox()
    cb.addItems(["a", "b", "c"])
    cb.setCurrentIndex(1)
    cb.setFocus()
    QtWidgets.QApplication.sendEvent(cb, _wheel(cb, 120))
    QtWidgets.QApplication.sendEvent(cb, _wheel(cb, -120))
    assert cb.currentIndex() == 1


def test_wheel_event_is_ignored_so_parent_can_scroll() -> None:
    _app()
    sb = NoWheelSpinBox()
    event = _wheel(sb, 120)
    sb.wheelEvent(event)
    # An ignored event propagates to the parent (the scroll area) instead of
    # being consumed by the control.
    assert not event.isAccepted()


def test_keyboard_stepping_still_works() -> None:
    _app()
    sb = NoWheelSpinBox()
    sb.setRange(0, 100)
    sb.setValue(10)
    sb.stepBy(1)
    assert sb.value() == 11
