"""
Interaction tests for ``NumericDragLineEdit`` (objective 4).

Covers typed commit, invalid temporary input, clamping, drag (including
modifier sensitivity and the click-vs-drag threshold), duplicate-signal
suppression, programmatic synchronization, and cursor reset.
"""

from __future__ import annotations

import pytest

pytest.importorskip('PySide6.QtWidgets')


import os

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

from lunaris.ui.core.ui_commons import NumericDragLineEdit


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _field(**kw) -> NumericDragLineEdit:
    _app()
    kw.setdefault("min_value", 0.0)
    kw.setdefault("max_value", 200.0)
    kw.setdefault("step", 5.0)
    kw.setdefault("decimals", 1)
    return NumericDragLineEdit("100.0", **kw)


def _type(field: NumericDragLineEdit, text: str) -> None:
    """Simulate keyboard editing (does not touch the internal numeric value)."""
    QtWidgets.QLineEdit.setText(field, text)


def test_typed_valid_value_commits_and_emits() -> None:
    f = _field()
    seen = []
    f.value_changed.connect(seen.append)
    _type(f, "55")
    f._commit_text()
    assert f.value() == 55.0
    assert seen == [55.0]


def test_typed_invalid_temporary_input_is_not_committed() -> None:
    f = _field()
    seen = []
    f.value_changed.connect(seen.append)
    for temp in ("", "-", ".", "1e"):
        _type(f, temp)
        f._commit_text()
        assert f.value() == 100.0, temp      # last valid value preserved
        assert f.text() == "100.0"           # display restored
    assert seen == []                        # nothing emitted


def test_clamping_for_typed_and_programmatic_values() -> None:
    f = _field()
    _type(f, "999")
    f._commit_text()
    assert f.value() == 200.0                # clamped to max
    f.set_value(-50)
    assert f.value() == 0.0                  # clamped to min


def test_no_duplicate_signal_when_value_unchanged() -> None:
    f = _field()
    seen = []
    f.value_changed.connect(seen.append)
    f.set_value(120.0)
    f.set_value(120.0)                       # same value
    f.set_value(120.0)
    assert seen == [120.0]


def test_programmatic_setText_keeps_internal_value_in_sync() -> None:
    f = _field()
    seen = []
    f.value_changed.connect(seen.append)
    f.setText("42.5")                        # e.g. session load
    assert f.value() == 42.5                 # not stale
    assert seen == []                        # setText does not emit


def test_set_value_can_suppress_emission() -> None:
    f = _field()
    seen = []
    f.value_changed.connect(seen.append)
    f.set_value(33.0, emit=False)
    assert f.value() == 33.0
    assert seen == []


def _press(field, x):
    pos = QtCore.QPointF(x, 10)
    return QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, pos, field.mapToGlobal(pos.toPoint()),
        QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )


def _move(field, x, modifiers=QtCore.Qt.NoModifier):
    pos = QtCore.QPointF(x, 10)
    return QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove, pos, field.mapToGlobal(pos.toPoint()),
        QtCore.Qt.NoButton, QtCore.Qt.LeftButton, modifiers,
    )


def _release(field, x):
    pos = QtCore.QPointF(x, 10)
    return QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, pos, field.mapToGlobal(pos.toPoint()),
        QtCore.Qt.LeftButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
    )


def test_drag_on_handle_changes_value() -> None:
    f = _field()
    f.resize(120, 38)
    seen = []
    f.value_changed.connect(seen.append)
    handle_x = f.width() - 5            # right-edge drag handle
    f.mousePressEvent(_press(f, handle_x))
    f.mouseMoveEvent(_move(f, handle_x + 10))   # 10 px * step 5.0 = +50
    f.mouseReleaseEvent(_release(f, handle_x))
    assert f.value() > 100.0
    assert seen and seen[-1] == f.value()


def test_modifier_reduces_drag_sensitivity() -> None:
    base = _field()
    base.resize(120, 38)
    hx = base.width() - 5
    base.mousePressEvent(_press(base, hx))
    base.mouseMoveEvent(_move(base, hx + 10))
    base.mouseReleaseEvent(_release(base, hx))
    coarse_delta = base.value() - 100.0

    fine = _field()
    fine.resize(120, 38)
    hx = fine.width() - 5
    fine.mousePressEvent(_press(fine, hx))
    fine.mouseMoveEvent(_move(fine, hx + 10, modifiers=QtCore.Qt.ControlModifier))
    fine.mouseReleaseEvent(_release(fine, hx))
    fine_delta = fine.value() - 100.0

    assert 0 < fine_delta < coarse_delta     # Ctrl = 0.1x


def test_plain_click_on_handle_does_not_change_value_or_emit() -> None:
    f = _field()
    f.resize(120, 38)
    seen = []
    f.value_changed.connect(seen.append)
    hx = f.width() - 5
    f.mousePressEvent(_press(f, hx))
    f.mouseMoveEvent(_move(f, hx + 1))       # below the drag threshold
    f.mouseReleaseEvent(_release(f, hx + 1))
    assert f.value() == 100.0
    assert seen == []


def test_cursor_resets_when_disabled_mid_drag() -> None:
    f = _field()
    f.resize(120, 38)
    hx = f.width() - 5
    f.mousePressEvent(_press(f, hx))
    f.mouseMoveEvent(_move(f, hx + 10))
    assert f._dragging is True
    f.setEnabled(False)                      # e.g. switched to a ghost field
    assert f._dragging is False
    assert f._drag_armed is False
    assert f.cursor().shape() == QtCore.Qt.ArrowCursor
