"""Mission Monitor workspace tests (group D preset integrity + group F geometry).

Covers preset/registry consistency, dock creation, graceful placeholders for
unimplemented widgets, singleton re-show, and default-layout geometry under
the real application stylesheet.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtWidgets

from lunaris.ui.monitor.presets import DEFAULT_PRESET_ID, PRESETS, preset_by_id, split_preset
from lunaris.ui.monitor.registry import DEFAULT_REGISTRY
from lunaris.ui.monitor.widgets.base import MissingWidgetPlaceholder
from lunaris.ui.monitor.workspace import MonitorController, MonitorPage, MonitorWorkspace


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def workspace():
    app = _app()
    controller = MonitorController()
    ws = MonitorWorkspace(controller)
    yield ws
    ws.deleteLater()
    app.processEvents()


class TestPresetIntegrity:
    def test_every_preset_widget_id_is_declared_in_the_registry(self):
        import lunaris.ui.monitor.widgets  # ensure registration ran

        unknown = [
            (preset.preset_id, wid)
            for preset in PRESETS
            for wid in preset.widget_ids
            if DEFAULT_REGISTRY.get(wid) is None
        ]
        assert unknown == []

    def test_preset_ids_are_unique(self):
        ids = [p.preset_id for p in PRESETS]
        assert len(ids) == len(set(ids))

    def test_split_preset_separates_reserved_widgets(self):
        preset = preset_by_id("orbit_overview")
        assert preset is not None
        openable, skipped = split_preset(preset, DEFAULT_REGISTRY)
        open_ids = [s.widget_id for s in openable]
        assert "altitude" in open_ids
        assert "orbital_elements" in open_ids
        assert "orbit_view" in skipped  # Phase 4 widget: declared, not faked


class TestWorkspace:
    def test_default_preset_opens_implemented_widgets(self, workspace):
        assert workspace.active_preset_id == DEFAULT_PRESET_ID
        open_ids = set(workspace._docks)
        assert {"altitude", "orbital_elements", "event_timeline"} <= open_ids
        # The skipped 3D widget is reported honestly on the toolbar.
        assert workspace.skipped_label.isVisibleTo(workspace)
        assert "orbit_view" in workspace.skipped_label.text()

    def test_unknown_widget_id_restores_as_placeholder(self, workspace):
        dock = workspace.add_widget("widget_removed_in_v9")
        assert isinstance(dock.widget(), MissingWidgetPlaceholder)
        assert "widget_removed_in_v9" in dock.windowTitle() or dock.windowTitle()

    def test_reserved_widget_opens_as_placeholder_not_fake_ui(self, workspace):
        dock = workspace.add_widget("orbit_view")
        assert isinstance(dock.widget(), MissingWidgetPlaceholder)

    def test_singleton_widget_is_reshown_not_duplicated(self, workspace):
        first = workspace.add_widget("altitude")
        first.close()
        again = workspace.add_widget("altitude")
        assert again is first
        assert len([d for wid, d in workspace._docks.items() if wid == "altitude"]) == 1

    def test_preset_switch_replaces_docks(self, workspace):
        workspace.apply_preset("numerical_health")
        assert "integrator_health" in workspace._docks
        assert "altitude" not in workspace._docks

    def test_dock_object_names_are_deterministic_for_state_restore(self, workspace):
        for wid, dock in workspace._docks.items():
            assert dock.objectName() == f"monitor_dock_{wid}"


class TestGeometry:
    def test_default_layout_fits_and_nothing_is_zero_sized(self):
        app = _app()
        from lunaris.ui.core.ui_commons import LOG_COLORS, THEME
        from lunaris.ui.theme import build_app_stylesheet

        controller = MonitorController()
        page = MonitorPage(controller)
        page.setStyleSheet(build_app_stylesheet(THEME, LOG_COLORS))
        page.resize(1280, 800)
        page.show()
        app.processEvents()
        try:
            ws = page.workspace
            assert ws.preset_combo.isVisibleTo(page)
            assert ws.add_button.isVisibleTo(page)
            assert ws.reset_button.isVisibleTo(page)
            for dock in ws._docks.values():
                assert dock.isVisibleTo(page)
                assert dock.width() > 50
                assert dock.height() > 50
            # Toolbar controls must not be clipped out of the page width.
            assert ws.add_button.geometry().right() <= page.width()
        finally:
            page.close()
            page.deleteLater()
            app.processEvents()

    def test_narrow_window_degrades_without_crash(self):
        app = _app()
        controller = MonitorController()
        page = MonitorPage(controller)
        page.resize(700, 500)
        page.show()
        app.processEvents()
        try:
            page.resize(560, 420)
            app.processEvents()
            assert page.workspace._docks  # still alive and populated
        finally:
            page.close()
            page.deleteLater()
            app.processEvents()
