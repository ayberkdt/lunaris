"""Tests for the reusable DataTable primitive (Phase 3 data presentation)."""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

try:
    from PySide6.QtWidgets import QApplication

    from lunaris.ui.components.primitives import DataTable

    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False


def _app():
    return QApplication.instance() or QApplication([])


def test_data_table_unit_headers_and_csv() -> None:
    _app()
    table = DataTable(
        [("Altitude", "km"), ("Max Degree", ""), "Backend"],
        numeric_columns=(0, 1),
    )
    assert table.columnCount() == 3
    assert table._header_text(0) == "Altitude [km]"
    assert table._header_text(1) == "Max Degree"  # empty unit → no brackets
    assert table._header_text(2) == "Backend"

    table.append_row([10.0, 1000, "cpu_sh"])
    table.append_row([50.0, 660, "numba_cuda_sh"])
    assert table.rowCount() == 2

    csv_text = table.to_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0] == "Altitude [km],Max Degree,Backend"
    assert lines[1] == "10.0,1000,cpu_sh"
    assert lines[2] == "50.0,660,numba_cuda_sh"


def test_data_table_copy_selection_is_tsv() -> None:
    app = _app()
    table = DataTable(["A", "B"])
    table.append_row(["x", "y"])
    table.append_row(["p", "q"])
    table.selectAll()
    table.copy_selection()
    text = app.clipboard().text()
    assert text == "x\ty\np\tq"
