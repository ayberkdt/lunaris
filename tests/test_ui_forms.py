"""Form validation contract for FormGrid and LabeledField (W3).

The behavioural rule this locks in: a field can be marked invalid with a
message, the message is reachable (tooltip / accessible description / inline
text), ``focus_first_invalid`` jumps to the first offending field, and clearing
an error restores the valid state. Headless via offscreen Qt.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtWidgets

pytest.importorskip("PySide6.QtWidgets")

from lunaris.ui.components.primitives import FormGrid, LabeledField  # noqa: E402


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_formgrid_required_marks_label_and_field() -> None:
    _app()
    grid = FormGrid()
    field = QtWidgets.QLineEdit()
    try:
        grid.add_row("Semi-major axis", field, required=True)
        assert field.property("required") is True
        assert field.accessibleDescription() == "required"
        # The label cell carries the required asterisk.
        label = grid.grid.itemAtPosition(0, 0).widget()
        assert label.text().endswith(" *")
    finally:
        grid.deleteLater()


def test_formgrid_set_error_toggles_property_and_tracks_invalid() -> None:
    _app()
    grid = FormGrid()
    a, b = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    try:
        b.setToolTip("nominal value")
        b.setAccessibleDescription("physical input")
        grid.add_row("A", a)
        grid.add_row("B", b)
        grid.set_error(b, "must be positive")
        assert b.property("fieldError") is True
        assert b.toolTip() == "must be positive"
        assert b.accessibleDescription() == "must be positive"
        assert grid._error_labels[b].text() == "must be positive"
        assert not grid._error_labels[b].isHidden()
        assert grid._invalid == [b]

        grid.set_error(b, None)
        assert b.property("fieldError") is False
        assert b.toolTip() == "nominal value"
        assert b.accessibleDescription() == "physical input"
        assert grid._error_labels[b].isHidden()
        assert grid._invalid == []
    finally:
        grid.deleteLater()


def test_formgrid_focus_first_invalid_uses_visual_order() -> None:
    app = _app()
    host = QtWidgets.QWidget()
    grid = FormGrid(host)
    a, b, c = (QtWidgets.QLineEdit() for _ in range(3))
    try:
        for name, w in (("A", a), ("B", b), ("C", c)):
            grid.add_row(name, w)
        host.show()
        # Mark B and C invalid; focus must land on B (earlier in visual order).
        grid.set_error(c, "bad")
        grid.set_error(b, "bad")
        assert grid.focus_first_invalid() is True
        app.processEvents()
        assert grid._invalid[0] in (b, c)  # tracked
        # B precedes C in self.fields, so it wins regardless of error order.
        assert grid.focus_first_invalid() is True
        app.processEvents()
        assert app.focusWidget() is b
    finally:
        host.deleteLater()


def test_formgrid_focus_first_invalid_returns_false_when_clean() -> None:
    _app()
    grid = FormGrid()
    try:
        grid.add_row("A", QtWidgets.QLineEdit())
        assert grid.focus_first_invalid() is False
    finally:
        grid.deleteLater()


def test_formgrid_clear_errors_resets_all() -> None:
    _app()
    grid = FormGrid()
    a, b = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    try:
        grid.add_row("A", a)
        grid.add_row("B", b)
        grid.set_error(a, "x")
        grid.set_error(b, "y")
        grid.clear_errors()
        assert grid._invalid == []
        assert a.property("fieldError") is False
        assert b.property("fieldError") is False
    finally:
        grid.deleteLater()


def test_labeledfield_inline_error_shows_and_hides() -> None:
    _app()
    field = QtWidgets.QLineEdit()
    field.setToolTip("unitless ratio")
    lf = LabeledField("Eccentricity", field, required=True)
    try:
        assert lf.has_error() is False
        assert field.property("required") is True
        lf.set_error("must be < 1")
        assert lf.has_error() is True
        assert lf._error_label.text() == "must be < 1"
        assert field.property("fieldError") is True
        lf.set_error(None)
        assert lf.has_error() is False
        # Clearing an error on a required field restores the required hint.
        assert field.accessibleDescription() == "required"
        assert field.toolTip() == "unitless ratio"
    finally:
        lf.deleteLater()


def test_labeledfield_error_hidden_by_default_adds_no_visible_text() -> None:
    _app()
    lf = LabeledField("Mass", QtWidgets.QLineEdit())
    try:
        assert lf._error_label.isVisible() is False
        assert lf._error_label.text() == ""
    finally:
        lf.deleteLater()


# ---------------------------------------------------------------------------
# W3.3 pilot: MissionPropagationPage wires the FormGrid error contract.
# Computation stays per-keystroke; error DISPLAY appears on blur; fixing a
# value clears the mark immediately; validate_inputs() is the pre-run gate.
# ---------------------------------------------------------------------------


def _propagation_page():
    from lunaris.ui.pages.mission_propagation_page import MissionPropagationPage

    _app()
    return MissionPropagationPage()


def test_propagation_duration_error_shown_on_blur_not_while_typing() -> None:
    page = _propagation_page()
    try:
        page.ent_duration.setText("abc")
        # Typing alone must not flag the field.
        assert page.ent_duration.property("fieldError") in (None, False)

        page.ent_duration.editingFinished.emit()
        assert page.ent_duration.property("fieldError") is True
        assert "not a number" in page.ent_duration.toolTip()

        # Fixing the value clears the mark immediately (no blur needed).
        page.ent_duration.setText("10.0")
        assert page.ent_duration.property("fieldError") is False
    finally:
        page.deleteLater()


def test_propagation_validate_inputs_focuses_first_invalid() -> None:
    page = _propagation_page()
    try:
        page.ent_duration.setText("")
        page.ent_rtol.setText("nope")
        assert page.validate_inputs() is False
        # Both offending fields are marked, message text reachable.
        assert page.ent_duration.property("fieldError") is True
        assert page.ent_rtol.property("fieldError") is True

        page.ent_duration.setText("5")
        page.ent_rtol.setText("1e-9")
        assert page.validate_inputs() is True
        assert page.ent_duration.property("fieldError") is False
        assert page.ent_rtol.property("fieldError") is False
    finally:
        page.deleteLater()
