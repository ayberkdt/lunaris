"""Mission Monitor: restoring a saved dashboard never doubles its panels.

Regression test: ``clear_docks`` removed the old docks and scheduled them with
``deleteLater``, but until the event loop ran they were still children of the
dock host under the same object names. ``restoreState`` matches docks by
object name, found the old dock and put it back beside its replacement, so
every relaunch showed each panel twice.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtWidgets

from lunaris.ui.monitor.persistence import layout_from_payload, layout_to_payload
from lunaris.ui.monitor.workspace import MonitorController, MonitorPage


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def page():
    app = _app()
    p = MonitorPage(MonitorController())
    yield p
    p.deleteLater()
    app.processEvents()


def _shown_docks(workspace) -> list[str]:
    """Object names of every dock the host would display (not explicitly hidden)."""
    host = workspace._host
    return sorted(
        d.objectName()
        for d in host.findChildren(QtWidgets.QDockWidget)
        if not d.isHidden()
    )


def test_restore_layout_shows_each_panel_once(page) -> None:
    app = _app()
    workspace = page.workspace
    expected = sorted(f"monitor_dock_{wid}" for wid in workspace._docks)
    assert _shown_docks(workspace) == expected

    layout = layout_from_payload(layout_to_payload(page.capture_layout()))
    page.restore_layout(layout)  # what startup does, before any event processing
    assert _shown_docks(page.workspace) == expected

    app.processEvents()
    assert _shown_docks(page.workspace) == expected


def test_restore_tab_layout_twice_in_place_shows_each_panel_once(page) -> None:
    workspace = page.workspace
    tab = workspace.capture_tab_layout("Dashboard")
    expected = sorted(f"monitor_dock_{wid}" for wid in tab.widget_ids)
    workspace.restore_tab_layout(tab)
    workspace.restore_tab_layout(tab)
    assert _shown_docks(workspace) == expected
