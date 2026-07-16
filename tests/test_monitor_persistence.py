"""Layout persistence tests (Mission Monitor phase 5).

Covers the versioned lunaris_monitor_layout_v1 schema (roundtrip, foreign
schema rejection), corrupt-file quarantine (the app must still open), the
multi-tab capture/restore path including dock geometry and unknown-widget
placeholders, and the end-to-end MainWindow save-on-close / restore-on-start
cycle.
"""

from __future__ import annotations

import json

import pytest
from tests.ui_qt_helpers import QtWidgets

from lunaris.ui.monitor.persistence import (
    MONITOR_LAYOUT_SCHEMA_VERSION,
    LayoutError,
    MonitorLayout,
    TabLayout,
    layout_from_payload,
    layout_to_payload,
    load_layout,
    load_layout_or_quarantine,
    save_layout,
)
from lunaris.ui.monitor.widgets.base import MissingWidgetPlaceholder
from lunaris.ui.monitor.workspace import MonitorController, MonitorPage


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def make_layout(**overrides) -> MonitorLayout:
    base = dict(
        tabs=(
            TabLayout("Ops", "orbit_overview", ("altitude", "event_timeline")),
            TabLayout("Numerics", "numerical_health", ("integrator_health",)),
        ),
        active_tab=1,
        last_replay_path="C:/runs/run_x/telemetry.ndjson",
    )
    base.update(overrides)
    return MonitorLayout(**base)


class TestSchema:
    def test_payload_round_trip(self):
        layout = make_layout()
        restored = layout_from_payload(layout_to_payload(layout))
        assert restored == layout

    def test_foreign_schema_is_rejected(self):
        payload = layout_to_payload(make_layout())
        payload["schema_version"] = "lunaris_monitor_layout_v99"
        with pytest.raises(LayoutError, match="v99"):
            layout_from_payload(payload)

    def test_layout_requires_at_least_one_tab(self):
        with pytest.raises(LayoutError):
            MonitorLayout(tabs=())

    def test_out_of_range_active_tab_falls_back_to_zero(self):
        payload = layout_to_payload(make_layout())
        payload["active_tab"] = 99
        assert layout_from_payload(payload).active_tab == 0

    def test_invalid_dock_blob_decodes_to_empty(self):
        tab = TabLayout("T", "orbit_overview", (), dock_state_b64="&&&not-base64")
        assert tab.dock_state_bytes() == b""


class TestFiles:
    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "monitor_layout.json"
        layout = make_layout()
        save_layout(path, layout)
        assert load_layout(path) == layout
        assert not path.with_suffix(".json.tmp").exists()  # atomic write cleaned up

    def test_missing_file_is_silent_first_run(self, tmp_path):
        assert load_layout_or_quarantine(tmp_path / "nope.json") is None

    def test_corrupt_file_is_quarantined_not_fatal(self, tmp_path):
        path = tmp_path / "monitor_layout.json"
        path.write_text("{not json at all", encoding="utf-8")
        warnings: list[str] = []
        result = load_layout_or_quarantine(path, log_warning=warnings.append)
        assert result is None
        assert not path.exists()
        assert path.with_suffix(".json.bak").exists()  # user data preserved
        assert warnings and "backed up" in warnings[0]

    def test_foreign_schema_file_is_quarantined(self, tmp_path):
        path = tmp_path / "monitor_layout.json"
        payload = layout_to_payload(make_layout())
        payload["schema_version"] = "v_future"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_layout_or_quarantine(path) is None
        assert path.with_suffix(".json.bak").exists()


@pytest.fixture()
def page():
    app = _app()
    controller = MonitorController()
    p = MonitorPage(controller)
    yield p
    p.deleteLater()
    app.processEvents()


class TestMonitorPageTabs:
    def test_starts_with_one_tab_that_cannot_be_closed_away(self, page):
        assert page.tabs.count() == 1
        page._on_tab_close_requested(0)
        assert page.tabs.count() == 1  # the last dashboard survives

    def test_add_and_close_tabs(self, page):
        page.add_dashboard_tab("Second")
        assert page.tabs.count() == 2
        assert page.tabs.currentIndex() == 1
        page._on_tab_close_requested(1)
        assert page.tabs.count() == 1

    def test_capture_restore_round_trip(self, page):
        _app()
        page.add_dashboard_tab("Numerics")
        page.workspace.apply_preset("numerical_health")
        page.tabs.setCurrentIndex(0)
        page.controller.last_replay_path = "X:/some/telemetry.ndjson"

        layout = page.capture_layout()
        assert layout.schema_version == MONITOR_LAYOUT_SCHEMA_VERSION
        assert len(layout.tabs) == 2
        assert layout.tabs[1].preset_id == "numerical_health"
        assert layout.tabs[0].dock_state_b64  # dock geometry captured

        # Serialize through JSON (the real on-disk trip) and rebuild.
        rebuilt = layout_from_payload(layout_to_payload(layout))
        page.restore_layout(rebuilt)
        assert page.tabs.count() == 2
        assert page.tabs.tabText(1) == "Numerics"
        assert page.tabs.currentIndex() == 0
        second = page.tabs.widget(1)
        assert set(second._docks) == set(layout.tabs[1].widget_ids)
        assert page.controller.last_replay_path == "X:/some/telemetry.ndjson"

    def test_restore_with_unknown_widget_id_yields_placeholder(self, page):
        layout = MonitorLayout(tabs=(
            TabLayout("Old", "orbit_overview", ("altitude", "widget_gone_in_v9")),
        ))
        page.restore_layout(layout)
        dock = page.workspace._docks["widget_gone_in_v9"]
        assert isinstance(dock.widget(), MissingWidgetPlaceholder)

    def test_restore_with_unknown_preset_falls_back_to_default(self, page):
        layout = MonitorLayout(tabs=(
            TabLayout("Odd", "preset_that_never_existed", ("altitude",)),
        ))
        page.restore_layout(layout)
        assert page.workspace.active_preset_id == "orbit_overview"


class TestMainWindowIntegration:
    def test_layout_survives_a_close_and_reopen_cycle(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
        app = _app()
        from lunaris.ui.app import MainWindow

        w1 = MainWindow()
        try:
            w1.page_monitor.add_dashboard_tab("Custom")
            w1.page_monitor.workspace.apply_preset("numerical_health")
            w1._save_monitor_layout()
        finally:
            w1.close()
            w1.deleteLater()
            app.processEvents()

        assert (tmp_path / "appdata" / "monitor_layout.json").is_file()

        w2 = MainWindow()
        try:
            assert w2.page_monitor.tabs.count() == 2
            assert w2.page_monitor.tabs.tabText(1) == "Custom"
            second = w2.page_monitor.tabs.widget(1)
            assert second.active_preset_id == "numerical_health"
        finally:
            w2.close()
            w2.deleteLater()
            app.processEvents()

    def test_corrupt_layout_file_does_not_block_startup(self, tmp_path, monkeypatch):
        appdata = tmp_path / "appdata"
        appdata.mkdir(parents=True)
        (appdata / "monitor_layout.json").write_text("garbage{{{", encoding="utf-8")
        monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(appdata))
        app = _app()
        from lunaris.ui.app import MainWindow

        w = MainWindow()
        try:
            # Default dashboard opened; broken file preserved as .bak.
            assert w.page_monitor.tabs.count() == 1
            assert (appdata / "monitor_layout.json.bak").is_file()
        finally:
            w.close()
            w.deleteLater()
            app.processEvents()
