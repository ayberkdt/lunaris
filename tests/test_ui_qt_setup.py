"""Both desktop apps set the C number locale and shared GL contexts first.

Regression tests: no locale was set, so on a Turkish desktop a
``QDoubleSpinBox`` showed ``10,000`` for ten and typing ``0.5`` gave 5.0 (``.``
is that locale's group separator) while every line edit, log and config file
uses ``.``. ``AA_ShareOpenGLContexts`` was never set, but pyqtgraph keeps each
compiled shader program id on a module-level object, so a 3D view in a second
unshared context (a floated Mission Monitor dock) draws only its labels.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from lunaris.ui_foundation.qt_setup import prepare_qt_application  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = (
    "src/lunaris/ui/app.py",
    "src/lunaris/ui/launcher.py",
    "src/lunaris/surrogate/st_lrps/ui/studio.py",
)


@pytest.fixture()
def turkish_desktop():
    previous = QtCore.QLocale()
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.Turkish, QtCore.QLocale.Turkey))
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield
    QtCore.QLocale.setDefault(previous)


def test_spin_box_reads_and_writes_dot_decimals_on_a_comma_locale(turkish_desktop) -> None:
    prepare_qt_application()
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(0.0, 100000.0)
    spin.setDecimals(3)
    spin.setValue(10.0)
    assert spin.text() == "10.000"
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "0.5")
    spin.interpretText()
    assert spin.value() == pytest.approx(0.5)
    assert spin.textFromValue(1234.5) == "1234.500"  # no group separator


@pytest.mark.parametrize("relpath", ENTRY_POINTS)
def test_entry_point_prepares_qt_before_creating_the_application(relpath: str) -> None:
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8-sig"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    prepare_lines: list[int] = []
    app_lines: list[int] = []
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "prepare_qt_application":
            prepare_lines.append(node.lineno)
        elif name == "QApplication":
            app_lines.append(node.lineno)
    assert prepare_lines, f"{relpath}: main() never calls prepare_qt_application()"
    assert app_lines, f"{relpath}: main() creates no QApplication (scan is stale)"
    assert min(prepare_lines) < min(app_lines)
