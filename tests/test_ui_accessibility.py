"""Accessibility regression tests (Step B/E closeout).

Locks in the accessible-name coverage on icon-only controls and the primary
mode-selecting combos so future edits cannot silently regress screen-reader
support.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_force_models_icon_buttons_have_accessible_names() -> None:
    _app()
    from lunaris.ui.pages.force_models_page import ForceModelsPage

    page = ForceModelsPage()
    # Icon-only gear buttons must be named for screen readers.
    assert page.btn_gravity_settings.accessibleName()
    assert page.btn_albedo_settings.accessibleName()
    # Every force toggle is icon-only paint → must carry a name.
    for switch in (
        page.sw_gravity, page.sw_sun, page.sw_earth, page.sw_earth_j2,
        page.sw_srp, page.sw_albedo, page.sw_thermal,
        page.sw_tides_k2, page.sw_tides_k3, page.sw_relativity_1pn,
    ):
        assert switch.accessibleName()


def test_batch_page_selectors_have_accessible_names() -> None:
    _app()
    from lunaris.ui.pages.batch_propagation_page import (
        BatchPropagationPage,
        UIBatchPropagationConfig,
    )

    page = BatchPropagationPage(batch_cfg=UIBatchPropagationConfig(use_gpu=False))
    try:
        for combo in (
            page.cb_sampling_method,
            page.cb_batch_gravity_mode,
            page.cb_batch_backend,
            page.cb_format,
        ):
            assert combo.accessibleName()
        assert page.toggle_gpu.accessibleName()
    finally:
        page.shutdown()


# --- Whole-window accessible-identity sweep -------------------------------

_INTERACTIVE = (
    "QAbstractButton",
    "QComboBox",
    "QAbstractSpinBox",
    "QLineEdit",
    "QPlainTextEdit",
    "QTextEdit",
    "QAbstractSlider",
    "QAbstractItemView",
)

_SKIP_OBJECT_NAMES = {"qt_spinbox_lineedit", "qt_toolbar_ext_button"}


def _interactive_types() -> tuple[type, ...]:
    return tuple(getattr(QtWidgets, name) for name in _INTERACTIVE)


def _in_composite_control(w) -> bool:
    """True for internal children of combos/spinboxes/views (Qt names those)."""
    parent = w.parentWidget()
    if isinstance(
        parent,
        (
            QtWidgets.QComboBox,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QAbstractItemView,
            QtWidgets.QCalendarWidget,
        ),
    ):
        return True
    cur = parent
    while cur is not None:
        if isinstance(cur, QtWidgets.QComboBox):
            return True
        if cur.metaObject().className() == "QComboBoxPrivateContainer":
            return True
        cur = cur.parentWidget()
    return False


def _widget_path(w) -> str:
    parts = []
    cur = w
    while cur is not None:
        parts.append(cur.objectName() or cur.metaObject().className())
        cur = cur.parentWidget()
    return "/".join(reversed(parts))


@pytest.mark.slow
def test_main_window_focusable_widgets_have_accessible_identity(tmp_path, monkeypatch) -> None:
    """Every focusable interactive control must expose an accessible identity.

    Identity means, in Qt's own name-resolution order: an explicit
    ``accessibleName``, the control's visible text (buttons), or a buddy
    label. Ghost/derived fields are exempt because they carry
    ``Qt.NoFocus`` and never enter the keyboard path.
    """
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "a11y_runtime"))
    _app()
    from lunaris.ui.app import MainWindow

    window = MainWindow()
    try:
        interactive = _interactive_types()
        buddies = {
            id(lbl.buddy()): lbl
            for lbl in window.findChildren(QtWidgets.QLabel)
            if lbl.buddy() is not None
        }
        missing: list[str] = []
        for w in window.findChildren(QtWidgets.QWidget):
            if not isinstance(w, interactive):
                continue
            if isinstance(w, QtWidgets.QScrollBar):
                continue
            if w.objectName() in _SKIP_OBJECT_NAMES:
                continue
            if w.focusPolicy() == QtCore.Qt.NoFocus:
                continue
            if _in_composite_control(w):
                continue
            if w.accessibleName().strip():
                continue
            if isinstance(w, QtWidgets.QAbstractButton) and w.text().strip():
                continue
            buddy = buddies.get(id(w))
            if buddy is not None and buddy.text().strip():
                continue
            missing.append(f"{type(w).__name__}: {_widget_path(w)}")
        assert not missing, (
            "Focusable widgets without an accessible identity:\n" + "\n".join(missing)
        )
    finally:
        window.close()
