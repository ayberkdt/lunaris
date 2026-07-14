"""Keyboard and focus behaviour for shared UI primitives (W2).

Covers the deterministic tab-order helper, the SegmentedControl radio-group
arrow navigation, and the ToggleSwitch keyboard contract. Headless via the
offscreen Qt platform.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

pytest.importorskip("PySide6.QtWidgets")
QtTest = pytest.importorskip("PySide6.QtTest")

from lunaris.ui.components.primitives import (  # noqa: E402
    FormGrid,
    SegmentedControl,
    apply_tab_order,
)


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_apply_tab_order_chains_widgets_in_order() -> None:
    _app()
    host = QtWidgets.QWidget()
    a, b, c = (QtWidgets.QLineEdit(host) for _ in range(3))
    try:
        apply_tab_order([a, b, c])
        # nextInFocusChain threads through hidden proxies, so assert the
        # relative order rather than adjacency.
        assert a.nextInFocusChain() is b or _reaches(a, b)
        assert _reaches(b, c)
    finally:
        host.deleteLater()


def _reaches(start: QtWidgets.QWidget, target: QtWidgets.QWidget, limit: int = 20) -> bool:
    node = start
    for _ in range(limit):
        node = node.nextInFocusChain()
        if node is target:
            return True
    return False


def test_apply_tab_order_skips_none_and_duplicates() -> None:
    _app()
    host = QtWidgets.QWidget()
    a, b = QtWidgets.QLineEdit(host), QtWidgets.QLineEdit(host)
    try:
        # Must not raise on None / repeated entries.
        apply_tab_order([a, None, a, b, None])
        assert _reaches(a, b)
    finally:
        host.deleteLater()


def test_formgrid_tracks_fields_and_orders_them() -> None:
    _app()
    grid = FormGrid()
    fields = [QtWidgets.QLineEdit() for _ in range(3)]
    try:
        for i, f in enumerate(fields):
            grid.add_row(f"Field {i}", f)
        assert grid.fields == fields
        # add_row wires the visual sequence automatically; callers can still
        # reapply the public helper after dynamic form changes.
        assert _reaches(fields[0], fields[1])
        assert _reaches(fields[1], fields[2])
        grid.apply_tab_order()  # public reapplication remains safe
    finally:
        grid.deleteLater()


def test_formgrid_mirrors_label_into_accessible_name() -> None:
    _app()
    grid = FormGrid()
    field = QtWidgets.QLineEdit()
    try:
        grid.add_row("Semi-major axis:", field)
        assert field.accessibleName() == "Semi-major axis"
    finally:
        grid.deleteLater()


def test_segmented_control_arrow_keys_move_selection() -> None:
    app = _app()
    seg = SegmentedControl(["Alpha", "Beta", "Gamma"])
    changes: list[int] = []
    seg.current_changed.connect(changes.append)
    try:
        seg.show()
        assert seg.current_index() == 0

        # Right / Down advance and wrap; Left / Up go back and wrap.
        seg.buttons[0].setFocus()
        _press(app, seg.buttons[0], QtCore.Qt.Key_Right)
        assert seg.current_index() == 1
        _press(app, seg.buttons[1], QtCore.Qt.Key_Down)
        assert seg.current_index() == 2
        _press(app, seg.buttons[2], QtCore.Qt.Key_Right)  # wrap to 0
        assert seg.current_index() == 0
        _press(app, seg.buttons[0], QtCore.Qt.Key_Left)  # wrap to 2
        assert seg.current_index() == 2
        _press(app, seg.buttons[2], QtCore.Qt.Key_Up)
        assert seg.current_index() == 1

        assert changes == [1, 2, 0, 2, 1]
    finally:
        seg.deleteLater()


def test_segmented_control_single_button_arrow_is_noop() -> None:
    app = _app()
    seg = SegmentedControl(["Only"])
    try:
        seg.show()
        _press(app, seg.buttons[0], QtCore.Qt.Key_Right)
        assert seg.current_index() == 0
    finally:
        seg.deleteLater()


def test_toggle_switch_is_keyboard_operable() -> None:
    app = _app()
    from lunaris.ui.core.ui_commons import ToggleSwitch

    sw = ToggleSwitch()
    try:
        assert sw.focusPolicy() == QtCore.Qt.StrongFocus
        assert not sw.isChecked()
        sw.show()
        sw.setFocus()
        _press(app, sw, QtCore.Qt.Key_Space)
        assert sw.isChecked()
        _press(app, sw, QtCore.Qt.Key_Return)  # Enter mirrors Space
        assert not sw.isChecked()
    finally:
        sw.deleteLater()


def _press(app, widget, key) -> None:
    QtTest.QTest.keyClick(widget, key)
    app.processEvents()


# ---------------------------------------------------------------------------
# Shortcut inventory (W2.3): ui/core/shortcuts.py is the single source of
# truth consumed by the menu bar and the window-level QShortcut builders.
# ---------------------------------------------------------------------------


def test_shortcut_inventory_ids_and_keys_are_unique() -> None:
    from lunaris.ui.core.shortcuts import SHORTCUTS

    ids = [s.action_id for s in SHORTCUTS]
    assert len(ids) == len(set(ids)), "duplicate action ids in shortcut inventory"

    all_keys = [k for s in SHORTCUTS for k in s.key_sequences]
    assert len(all_keys) == len(set(all_keys)), (
        f"conflicting key bindings in shortcut inventory: {sorted(all_keys)}"
    )


def test_shortcut_inventory_keys_parse_as_key_sequences() -> None:
    _app()
    from lunaris.ui.core.shortcuts import SHORTCUTS

    for s in SHORTCUTS:
        assert s.key_sequences, f"{s.action_id} declares no key sequence"
        assert s.scope in {"menu", "window"}, f"{s.action_id} has unknown scope"
        for key in s.key_sequences:
            seq = QtGui.QKeySequence(key)
            assert not seq.isEmpty(), f"{s.action_id}: {key!r} did not parse"


def test_shortcut_helpers_roundtrip() -> None:
    from lunaris.ui.core import shortcuts

    assert shortcuts.primary_key("analysis.run") == "F5"
    assert shortcuts.keys("view.toggle_log")[0] == shortcuts.primary_key("view.toggle_log")
    assert len(shortcuts.keys("nav.page")) == 9
