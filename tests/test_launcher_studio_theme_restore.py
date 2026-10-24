"""Launcher: closing the ST-LRPS Studio gives the application its look back.

Regression test: the launcher hosts both workspaces in one process, and
opening the Studio ran ``apply_premium_dark_theme(app)``, which sets the
application-wide style (Fusion), font, palette and stylesheet. Nothing undid
it, so the launcher and a Mission Studio opened afterwards inherited the
Studio's application stylesheet (parentless dialogs, menus, anything the
window stylesheet does not cover), its style and its font.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtGui, QtWidgets  # noqa: E402


def _look(app: QtWidgets.QApplication) -> tuple:
    return (app.styleSheet(), app.style().name(), app.font().toString(),
            app.palette().color(QtGui.QPalette.ColorRole.Base).name())


@pytest.fixture()
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sheet, style = application.styleSheet(), application.style().name()
    font, palette = QtGui.QFont(application.font()), QtGui.QPalette(application.palette())
    yield application
    application.setStyleSheet(sheet)
    application.setStyle(style)
    application.setFont(font)
    application.setPalette(palette)


def test_closing_the_studio_restores_the_application_look(app, monkeypatch) -> None:
    from lunaris.surrogate.st_lrps.ui import embedding
    from lunaris.ui.launcher import LauncherWindow

    # A stand-in window: the test is about the theme hand-off, not the Studio.
    monkeypatch.setattr(embedding, "MainWindow", QtWidgets.QWidget)
    app.setStyleSheet("")
    before = _look(app)

    win = SimpleNamespace(show=lambda: None, raise_=lambda: None, activateWindow=lambda: None)
    for name in ("_build_studio_window", "_on_workspace_closed", "_restore_app_look"):
        if hasattr(LauncherWindow, name):
            setattr(win, name, getattr(LauncherWindow, name).__get__(win))

    studio = win._build_studio_window()
    assert _look(app) != before, "the Studio theme was not applied (test premise)"
    studio.deleteLater()
    win._on_workspace_closed()

    assert _look(app) == before


def test_launchers_share_one_application_wheel_guard(app, monkeypatch) -> None:
    """Every app-level filter runs in Python on every event of every widget;
    one per launcher piled up in a long-lived application."""
    from lunaris.surrogate.st_lrps.ui import embedding
    from lunaris.ui.launcher import LauncherWindow
    from lunaris.ui_foundation.event_filters import NoWheelOnSpinFilter

    monkeypatch.setattr(embedding, "MainWindow", QtWidgets.QWidget)
    # The guard is the subject, not the theme: an application-wide style change
    # re-polishes every widget the suite has left alive, which is slow.
    monkeypatch.setattr(embedding, "apply_premium_dark_theme", lambda _app: None)
    guards = []
    for _ in range(3):  # three launchers in one application
        win = SimpleNamespace(show=lambda: None, raise_=lambda: None, activateWindow=lambda: None)
        win._build_studio_window = LauncherWindow._build_studio_window.__get__(win)
        win._build_studio_window().deleteLater()
        guards.append(win._studio_wheel_guard)

    assert len(app.findChildren(NoWheelOnSpinFilter)) == 1
    assert guards[0] is guards[1] is guards[2]
