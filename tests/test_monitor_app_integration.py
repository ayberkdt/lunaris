"""MainWindow ↔ Mission Monitor integration (stdout fan-out, group B/H edges).

Uses the same real-MainWindow fixture pattern as test_ui_stdout_pipeline: the
new [TELEMETRY]/[TELEMETRY_META] routing must feed the monitor store *and*
keep the legacy surfaces (progress bar, telemetry plot, console behavior)
working unchanged.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtWidgets

from lunaris.common.telemetry_contract import (
    TelemetryProvenance,
    TelemetrySample,
    encode_meta_line,
    encode_sample_line,
)


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def win(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
    app = _app()
    from lunaris.ui.app import MainWindow

    w = MainWindow()
    yield w, app
    w.close()
    w.deleteLater()
    app.processEvents()


def _log_count(w) -> int:
    return len(w.log_panel._pending) + len(w.log_panel._entries)


def _sample_line(seq: int, t: float, alt_m: float = 50_000.0) -> str:
    return encode_sample_line(TelemetrySample(
        run_id="run_it", sequence_id=seq, simulation_time_s=t,
        altitude_m=alt_m, radius_m=1_737_400.0 + alt_m, speed_m_s=1650.0,
        orbital_elements={"ecc": 0.01, "sma_m": 1.8e6},
    ))


def test_v1_sample_feeds_monitor_and_legacy_progress(win) -> None:
    w, _ = win
    w.sim_state.total_duration = 200.0
    before = _log_count(w)

    w._consume_stdout_line(_sample_line(0, 100.0))

    # Not echoed to the console (same rule as legacy telemetry).
    assert _log_count(w) == before
    # Legacy surfaces still update from the same line.
    assert w._last_telem_t_s == 100.0
    assert w.progress_bar.value() == 500
    # And the monitor store received the typed sample.
    store = w.monitor_controller.store
    assert store.counters.accepted == 1
    assert store.has_channel("altitude_m")
    assert store.has_channel("elements.ecc")


def test_meta_line_sets_provenance_and_is_acknowledged(win) -> None:
    w, _ = win
    before = _log_count(w)
    w._consume_stdout_line(encode_meta_line(TelemetryProvenance(
        run_id="run_it", integrator="DOP853", gravity_backend="classic_sh",
    )))
    assert _log_count(w) == before + 1  # one friendly system line, no raw JSON
    prov = w.monitor_controller.store.provenance
    assert prov is not None and prov.integrator == "DOP853"


def test_malformed_v1_line_fails_closed_to_console(win) -> None:
    w, _ = win
    before = _log_count(w)
    w._consume_stdout_line("[TELEMETRY] {broken json")
    assert _log_count(w) == before + 1  # surfaced as a warning line
    assert w.monitor_controller.store.counters.accepted == 0


def test_unsupported_schema_raises_monitor_banner(win) -> None:
    w, app = win
    w._consume_stdout_line(
        '[TELEMETRY] {"schema_version": "lunaris_telemetry_v99", '
        '"run_id": "x", "sequence_id": 0, "simulation_time_s": 1.0}'
    )
    app.processEvents()
    assert w.page_monitor.problem_banner.isVisibleTo(w.page_monitor)
    assert "v99" in w.page_monitor.problem_banner.text()


def test_legacy_bare_json_also_lands_in_the_monitor_store(win) -> None:
    w, _ = win
    w._consume_stdout_line('{"t_s": 30.0, "alt_km": 49.0, "v_km_s": 1.6, "ecc": 0.02}')
    store = w.monitor_controller.store
    assert store.counters.accepted == 1
    t, v = store.snapshot("altitude_m")
    assert v[0] == pytest.approx(49_000.0)


def test_diag_payload_reaches_monitor_and_results_page(win) -> None:
    w, _ = win
    w._consume_stdout_line('[DIAG] {"wall_time_s": 5.0, "nfev": 100.0, "impacted": false}')
    assert w.monitor_controller.store.run_diagnostics is not None
    assert w.monitor_controller.store.run_diagnostics["nfev"] == 100.0


def test_run_lifecycle_hooks_reset_and_finish_the_store(win) -> None:
    w, _ = win
    ctrl = w.monitor_controller
    ctrl.begin_live_run(expected_duration_s=100.0)
    w._consume_stdout_line(_sample_line(0, 10.0))
    assert ctrl.store.mode == "live"
    ctrl.finish_live_run(exit_code=0)
    assert ctrl.store.outcome is not None
    assert ctrl.store.outcome.success is True
    # A fresh run starts a clean store (bounded, no leakage across runs).
    ctrl.begin_live_run()
    assert ctrl.store.counters.accepted == 0
    assert ctrl.store.outcome is None


def test_terrain_sample_drives_the_collision_watchdog_fields(win) -> None:
    w, _ = win
    line = encode_sample_line(TelemetrySample(
        run_id="run_it", sequence_id=0, simulation_time_s=5.0,
        altitude_m=1200.0, radius_m=1_738_600.0,
        surface_radius_m=1_738_000.0, terrain_clearance_m=600.0,
    ))
    w._consume_stdout_line(line)
    # The legacy dict derived for _check_collision carries the surface-relative
    # altitude: alt_km - clearance_km = surface_alt_km (0.6 km here).
    store = w.monitor_controller.store
    assert store.has_channel("terrain_clearance_m")
    assert w._last_telem_t_s == 5.0
