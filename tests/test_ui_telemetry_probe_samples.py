"""Mission Studio legacy telemetry surfaces and ``rhs_probe`` samples.

Regression tests for ``MainWindow._apply_telemetry_sample``:

* an ``rhs_probe`` sample (a solver evaluation that may belong to a rejected
  adaptive step) reached the impact watchdog, so one probe below the surface
  stopped a run whose accepted trajectory never went there;
* the output states the engine emits from t0 after the integration were drawn
  after the live probe trace (the orbit twice, with a line back to t0) and
  drove the progress bar from ~100 % back to 0.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from lunaris.common.telemetry_contract import TelemetrySample  # noqa: E402
from lunaris.ui.app import MainWindow  # noqa: E402


class _Plot:
    def __init__(self) -> None:
        self.points: list[float] = []
        self.clears = 0

    def add_datapoint(self, legacy: dict) -> None:
        self.points.append(legacy["t_s"])

    def clear_all(self) -> None:
        self.points.clear()
        self.clears += 1


def _window() -> SimpleNamespace:
    win = SimpleNamespace(
        page_telemetry=SimpleNamespace(telemetry_multiplot=_Plot()),
        collision_checks=[],
        progress=[],
        _telemetry_holds_probes=False,
        _telemetry_saw_probes=False,
    )
    win._check_collision = lambda legacy: win.collision_checks.append(legacy)
    win._update_run_progress = lambda t_s: win.progress.append(t_s)
    return win


def _sample(t_s: float, kind: str, alt_m: float = 100e3) -> TelemetrySample:
    return TelemetrySample(
        run_id="r", sequence_id=int(t_s), simulation_time_s=t_s, sample_kind=kind, altitude_m=alt_m
    )


def test_probe_below_the_surface_does_not_reach_the_impact_watchdog() -> None:
    win = _window()
    MainWindow._apply_telemetry_sample(win, _sample(10.0, "rhs_probe", alt_m=-500.0))
    assert win.collision_checks == []


def test_output_state_still_reaches_the_impact_watchdog() -> None:
    win = _window()
    MainWindow._apply_telemetry_sample(win, _sample(10.0, "output_state", alt_m=-500.0))
    assert len(win.collision_checks) == 1
    assert win.collision_checks[0]["alt_km"] == pytest.approx(-0.5)


def test_output_states_replace_the_probe_trace_and_keep_progress_monotonic() -> None:
    win = _window()
    for t in (100.0, 200.0, 300.0):
        MainWindow._apply_telemetry_sample(win, _sample(t, "rhs_probe"))
    for t in (0.0, 150.0, 300.0):
        MainWindow._apply_telemetry_sample(win, _sample(t, "output_state"))
    plot = win.page_telemetry.telemetry_multiplot
    assert plot.clears == 1
    assert plot.points == [0.0, 150.0, 300.0]
    assert win.progress == [100.0, 200.0, 300.0]


def test_impact_watchdog_is_rearmed_for_the_next_run() -> None:
    """Regression: ``_collision_triggered`` latched for the window's lifetime."""
    import inspect

    stops: list[int] = []
    win = SimpleNamespace(
        _collision_triggered=False,
        _try_float=MainWindow._try_float,
        _stop_process=lambda: stops.append(1),
        _log_message=lambda *a, **k: None,
    )
    MainWindow._check_collision(win, {"alt_km": -1.0})
    assert len(stops) == 1
    MainWindow._check_collision(win, {"alt_km": -1.0})  # same run: latched
    assert len(stops) == 1
    MainWindow._reset_run_watch_state(win)  # what a new run does
    MainWindow._check_collision(win, {"alt_km": -1.0})
    assert len(stops) == 2
    # The run launcher performs that reset.
    assert "self._reset_run_watch_state()" in inspect.getsource(MainWindow._run_process)


def test_run_without_probes_draws_and_reports_output_states() -> None:
    win = _window()
    for t in (0.0, 60.0):
        MainWindow._apply_telemetry_sample(win, _sample(t, "output_state"))
    assert win.page_telemetry.telemetry_multiplot.clears == 0
    assert win.progress == [0.0, 60.0]
