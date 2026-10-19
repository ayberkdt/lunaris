"""Event and start epochs count from ``start_date`` (solver time 0), not from the first sample.

``TimeConfig.start_date`` is solver time 0 and ``PropagationResult.t`` holds
absolute solver times; the analysis added ``t - t[0]`` to ``start_date`` for
event epochs and reported ``start_date`` as the start epoch, so a run with
``t0_s > 0`` was early by ``t0_s``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from tests.test_orbit_analysis_contract import MU, RADIUS_M, _circular_result, _config

from lunaris.analysis.orbit_analysis import build_orbit_analysis

_META = {"mu_m3s2": MU, "body_radius_m": RADIUS_M}


def _start_metric(analysis):
    return analysis.metric_map["mission.start_epoch"]


def test_event_epochs_count_from_start_date_not_from_the_first_sample() -> None:
    """start_date is solver time 0; a run with t0_s = 600 s starts ten minutes later."""
    result = _circular_result()
    result.t = result.t + 600.0
    analysis = build_orbit_analysis(result, config=_config(), meta=_META, run_id="t0")
    start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    assert _start_metric(analysis).value == "2026-01-01T00:10:00.000Z"
    assert _start_metric(analysis).time_system == "UTC"
    assert analysis.events
    for event in analysis.events:
        expected = start + timedelta(seconds=event.simulation_time_s)
        assert event.epoch_utc == expected.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def test_a_run_from_solver_time_zero_keeps_the_configured_start_string() -> None:
    analysis = build_orbit_analysis(_circular_result(), config=_config(), meta=_META, run_id="t0")
    assert _start_metric(analysis).value == "2026-01-01T00:00:00Z"


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ("2026-01-01 03:00:00+03:00", "2026-01-01T00:10:00.000Z"),
        ("2026-01-01 00:00", "2026-01-01T00:10:00.000Z"),
    ],
)
def test_iso_start_dates_the_run_accepts_give_utc_epochs(start: str, expected: str) -> None:
    """The repository's civil-time parser reads offsets and short forms the CLI accepts."""
    cfg = _config()
    cfg.time.start_date = start
    result = _circular_result()
    result.t = result.t + 600.0
    analysis = build_orbit_analysis(result, config=cfg, meta=_META, run_id="iso")
    assert _start_metric(analysis).value == expected
    assert all(event.epoch_utc is not None for event in analysis.events)


def test_a_tdb_start_keeps_its_label_and_never_reports_a_utc_event_epoch() -> None:
    cfg = _config()
    cfg.time.start_date = "2026-01-01T00:01:09.184 TDB"
    result = _circular_result()
    result.t = result.t + 600.0
    analysis = build_orbit_analysis(result, config=cfg, meta=_META, run_id="tdb")
    metric = _start_metric(analysis)
    assert metric.time_system == "TDB"
    assert metric.value == "2026-01-01T00:11:09.184 TDB"
    # No leap-second table here: a UTC epoch would be a TDB value mislabelled.
    assert analysis.events
    assert all(event.epoch_utc is None for event in analysis.events)


def test_an_unparseable_start_with_t0_is_unavailable_not_early() -> None:
    cfg = _config()
    cfg.time.start_date = "2026 JAN 01 00:00:00"
    result = _circular_result()
    result.t = result.t + 600.0
    analysis = build_orbit_analysis(result, config=cfg, meta=_META, run_id="spice")
    metric = _start_metric(analysis)
    assert metric.value is None
    assert metric.status == "unavailable"
    assert "could not be parsed" in (metric.availability_reason or "")
