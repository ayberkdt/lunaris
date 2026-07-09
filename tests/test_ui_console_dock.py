"""Smoke tests for the expandable execution console."""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

from lunaris.ui.widgets.log_panel import ExecutionConsoleDock, ExecutionLogPanel


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_historical_console_name_is_compatible() -> None:
    assert ExecutionLogPanel is ExecutionConsoleDock


def test_console_collapses_to_status_strip_and_restores() -> None:
    app = _app()
    panel = ExecutionConsoleDock()
    try:
        panel.set_collapsed(True)
        assert panel.is_collapsed
        assert panel.header.isVisible() or not panel.isVisible()
        assert panel.body.isHidden()
        panel.set_collapsed(False)
        assert not panel.is_collapsed
        assert not panel.body.isHidden()
    finally:
        panel.deleteLater()
    _ = app


def test_console_counts_and_filters_messages() -> None:
    app = _app()
    panel = ExecutionConsoleDock()
    try:
        panel.append("nominal telemetry", severity="info")
        panel.append("temperature warning", severity="warning")
        panel.append("solver failed", severity="error")
        panel._flush()
        assert panel.lbl_warning_count.text() == "W 1"
        assert panel.lbl_error_count.text() == "E 1"
        assert panel.lbl_latest.text() == "solver failed"

        panel.severity_filter.setCurrentText("Errors")
        assert "solver failed" in panel.console.toPlainText()
        assert "temperature warning" not in panel.console.toPlainText()

        panel.severity_filter.setCurrentText("All levels")
        panel.search_field.setText("temperature")
        assert "temperature warning" in panel.console.toPlainText()
        assert "nominal telemetry" not in panel.console.toPlainText()
    finally:
        panel.deleteLater()
    _ = app
