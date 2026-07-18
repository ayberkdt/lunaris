"""Responsive-geometry regressions for the ST-LRPS desktop workspace."""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtWidgets

pytest.importorskip("torch")


def _window(width: int, height: int, *, page_index: int | None = None):
    from lunaris.surrogate.st_lrps.ui.studio_parts.main_window import MainWindow
    from lunaris.surrogate.st_lrps.ui.studio_parts.qt_common import (
        apply_premium_dark_theme,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_premium_dark_theme(app)
    window = MainWindow()
    window.resize(width, height)
    if page_index is not None:
        window._navigate(page_index)
    window.show()
    app.processEvents()
    return app, window


def _assert_button_text_fits(button: QtWidgets.QPushButton) -> None:
    option = QtWidgets.QStyleOptionButton()
    button.initStyleOption(option)
    content = button.style().subElementRect(
        QtWidgets.QStyle.SubElement.SE_PushButtonContents,
        option,
        button,
    )
    required = button.fontMetrics().horizontalAdvance(button.text())
    assert content.width() >= required, (
        f"{button.text()!r} is clipped: content={content.width()}px, "
        f"text={required}px"
    )


def _assert_combo_text_fits(combo: QtWidgets.QComboBox) -> None:
    parent = combo.parentWidget()
    assert parent is not None
    assert parent.contentsRect().contains(combo.geometry()), (
        f"{combo.currentText()!r} extends outside its layout cell: "
        f"combo={combo.geometry().getRect()}, parent={parent.contentsRect().getRect()}"
    )
    option = QtWidgets.QStyleOptionComboBox()
    combo.initStyleOption(option)
    content = combo.style().subControlRect(
        QtWidgets.QStyle.ComplexControl.CC_ComboBox,
        option,
        QtWidgets.QStyle.SubControl.SC_ComboBoxEditField,
        combo,
    )
    required = combo.fontMetrics().horizontalAdvance(combo.currentText())
    assert content.width() >= required, (
        f"{combo.currentText()!r} is clipped: content={content.width()}px, "
        f"text={required}px"
    )


@pytest.mark.parametrize("size", [(1024, 768), (1280, 860)])
def test_training_setup_readiness_and_actions_fit(size) -> None:
    app, window = _window(*size, page_index=1)
    try:
        tab = window._train_tab
        summary = tab._launch_summary
        assert summary.width() >= 240
        if summary.hasHeightForWidth():
            assert summary.height() >= summary.heightForWidth(summary.width())
        _assert_button_text_fits(tab.btn_enqueue_setup)
        _assert_button_text_fits(tab.btn_start_setup)
        _assert_combo_text_fits(tab.workflow_mode)
        if size == (1024, 768):
            assert tab._launch_strip._compact
        context = tab._launch_strip._context
        for child in (tab._launch_strip._workflow, tab._launch_strip._output_mode):
            assert context.contentsRect().contains(child.geometry())
        assert not tab._launch_strip._workflow.geometry().intersects(
            tab._launch_strip._output_mode.geometry()
        )
        plan = tab._launch_plan_values
        assert plan.height() >= plan.sizeHint().height()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("size", [(1024, 768), (1280, 860)])
def test_data_workspace_has_no_form_horizontal_scroll_or_clipped_actions(size) -> None:
    app, window = _window(*size)
    try:
        window._navigate(0)
        app.processEvents()
        page = window._data_page
        cloud = window._cloud_tab
        _assert_button_text_fits(cloud._btn_generate_now)
        for button in cloud.findChildren(QtWidgets.QPushButton):
            if button.text() in {"Show Command", "Copy"} and button.isVisible():
                _assert_button_text_fits(button)
        visible_scrolls = [
            area
            for area in page.findChildren(QtWidgets.QScrollArea)
            if area.isVisible()
        ]
        assert visible_scrolls
        assert all(area.horizontalScrollBar().maximum() == 0 for area in visible_scrolls)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("size", [(1024, 768), (1280, 860)])
def test_training_monitor_reflows_without_clipped_controls(size) -> None:
    app, window = _window(*size, page_index=2)
    try:
        tab = window._train_tab
        splitter = tab._monitor_splitter
        expected = (
            QtCore.Qt.Orientation.Vertical
            if size == (1024, 768)
            else QtCore.Qt.Orientation.Horizontal
        )
        assert splitter.orientation() == expected
        assert splitter.compact is (size == (1024, 768))
        for button in (
            tab.runner.btn_start,
            tab.runner.btn_stop,
            tab.btn_enqueue_monitor,
            tab.btn_clear_log_monitor,
            tab.btn_open_run_monitor,
            tab.btn_preview_cmd_monitor,
            tab.btn_copy_cmd_monitor,
        ):
            _assert_button_text_fits(button)
        assert tab._monitor_scroll.horizontalScrollBar().maximum() == 0
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
