"""Deferred UI callbacks never run on a destroyed widget.

Regression tests: ``QTimer.singleShot(ms, callable)`` has no receiver, so
PySide6 runs it even after the widget is gone. A lambda or local closure fires
after a plain ``deleteLater``. A bound method usually dies with its object, but
not when the shot is scheduled while the widget is being destroyed: the Studio
launch strip schedules ``_refresh_layout`` from its ``event()`` override on the
``LayoutRequest`` its dying children post, and the shot then read a deleted
child (``RuntimeError: Internal C++ object already deleted``). Every single
shot in the UI packages now names its widget as the receiver, so it dies with
the widget.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI_PACKAGES = (
    ROOT / "src" / "lunaris" / "ui",
    ROOT / "src" / "lunaris" / "ui_foundation",
    ROOT / "src" / "lunaris" / "surrogate" / "st_lrps" / "ui",
)


def _single_shot_calls():
    for package in UI_PACKAGES:
        for path in sorted(package.rglob("*.py")):
            if "web" in path.relative_to(package).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "singleShot"
                ):
                    yield path.relative_to(ROOT).as_posix(), node


def test_every_single_shot_names_a_receiver() -> None:
    calls = list(_single_shot_calls())
    assert calls, "the scan found no single shots; the package paths are wrong"
    detached = [f"{path}:{node.lineno}" for path, node in calls if len(node.args) < 3]
    assert detached == [], (
        "QTimer.singleShot without a receiver can run on a destroyed widget; "
        "pass the owning QObject as the second argument: " + ", ".join(detached)
    )


@pytest.fixture()
def qt_errors(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors: list[str] = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(repr(exc[1])))
    monkeypatch.setattr(sys, "unraisablehook", lambda u: errors.append(repr(u.exc_value)))
    return errors


def _drain() -> None:
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    app.processEvents()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(50, loop.quit)
    loop.exec()


def test_receiver_single_shot_dies_with_its_widget(qt_errors) -> None:
    import shiboken6
    from PySide6 import QtCore, QtWidgets

    fired: list[str] = []
    detached = QtWidgets.QWidget()
    bound = QtWidgets.QWidget()
    QtCore.QTimer.singleShot(0, lambda: fired.append(detached.objectName()))
    QtCore.QTimer.singleShot(0, bound, lambda: fired.append(bound.objectName()))
    shiboken6.delete(detached)
    shiboken6.delete(bound)
    # The premise of the fix: a receiver-less lambda still runs (and fails).
    # Depending on the PySide6 build the error reaches sys.excepthook or is
    # re-raised from the event loop; accept either.
    try:
        _drain()
    except RuntimeError as exc:
        qt_errors.append(repr(exc))
    assert any("already deleted" in e for e in qt_errors)
    assert fired == []
    qt_errors.clear()


@pytest.mark.parametrize("how", ["delete", "parent_deleted"])
def test_launch_strip_destroyed_mid_layout_runs_no_callback(qt_errors, how) -> None:
    import shiboken6
    from PySide6 import QtWidgets

    from lunaris.surrogate.st_lrps.ui.studio_parts.training_pages import _ResponsiveLaunchStrip

    host = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(host)
    parts = [QtWidgets.QLabel("workflow"), QtWidgets.QLabel("mode"), QtWidgets.QLabel("badge"),
             QtWidgets.QPushButton("Queue"), QtWidgets.QPushButton("Start"), QtWidgets.QLabel("note")]
    strip = _ResponsiveLaunchStrip(*parts)
    layout.addWidget(strip)
    host.show()
    # One pass only: the strip's own deferred refresh is still pending when it
    # is destroyed, which is the window the receiver-less shot fired in.
    QtWidgets.QApplication.instance().processEvents()
    if how == "delete":
        shiboken6.delete(strip)
    else:
        host.deleteLater()
    _drain()
    assert qt_errors == []
    if how == "delete":
        host.deleteLater()
        _drain()
