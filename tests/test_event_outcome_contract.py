from __future__ import annotations

import numpy as np
import pytest

from lunaris.core.propagation.events import EventOutcome, event_outcome_from_solver_events


def _event(*, role: str | None = None, terminal: bool = True):
    def ev(_t, _y):
        return 1.0

    ev.terminal = terminal  # type: ignore[attr-defined]
    ev.direction = 0.0  # type: ignore[attr-defined]
    if role is not None:
        ev._event_role = role  # type: ignore[attr-defined]
    return ev


def test_event_outcome_normalizes_reason_to_stopped_early() -> None:
    outcome = EventOutcome(
        impacted=False,
        t_impact_s=None,
        y_impact=None,
        stopped_early=False,
        stop_reason="event",
        t_stop_s=12.5,
    )

    assert outcome.stopped_early is True
    assert outcome.stop_reason == "event"


def test_solver_event_outcome_preserves_impact_event() -> None:
    y_imp = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    outcome = event_outcome_from_solver_events(
        events=[_event(role="impact", terminal=True)],
        t_events=[np.asarray([4.0], dtype=np.float64)],
        y_events=[y_imp.reshape(1, -1)],
        stopped_early=True,
        stop_reason=None,
        stop_file=None,
        stop_event_in_scipy=False,
    )

    assert outcome.impacted is True
    assert outcome.stopped_early is True
    assert outcome.stop_reason == "impact"
    assert outcome.t_impact_s == pytest.approx(4.0)
    assert outcome.t_stop_s == pytest.approx(4.0)
    np.testing.assert_allclose(outcome.y_impact, y_imp)


def test_solver_event_outcome_preserves_generic_terminal_event() -> None:
    outcome = event_outcome_from_solver_events(
        events=[_event(terminal=True)],
        t_events=[np.asarray([7.0], dtype=np.float64)],
        y_events=[np.zeros((1, 6), dtype=np.float64)],
        stopped_early=True,
        stop_reason=None,
        stop_file=None,
        stop_event_in_scipy=False,
    )

    assert outcome.impacted is False
    assert outcome.stopped_early is True
    assert outcome.stop_reason == "event"
    assert outcome.t_stop_s == pytest.approx(7.0)


def test_solver_event_outcome_does_not_treat_nonterminal_event_as_stop_reason() -> None:
    outcome = event_outcome_from_solver_events(
        events=[_event(role="peri", terminal=False)],
        t_events=[np.asarray([3.0], dtype=np.float64)],
        y_events=[np.zeros((1, 6), dtype=np.float64)],
        stopped_early=False,
        stop_reason=None,
        stop_file=None,
        stop_event_in_scipy=False,
    )

    assert outcome.stopped_early is False
    assert outcome.stop_reason is None
    assert outcome.t_stop_s is None


def test_solver_event_outcome_preserves_stop_file_event() -> None:
    outcome = event_outcome_from_solver_events(
        events=[_event(role="stop", terminal=True)],
        t_events=[np.asarray([9.0], dtype=np.float64)],
        y_events=[np.zeros((1, 6), dtype=np.float64)],
        stopped_early=True,
        stop_reason=None,
        stop_file="STOP",
        stop_event_in_scipy=True,
    )

    assert outcome.stopped_early is True
    assert outcome.stop_reason == "stop file"
    assert outcome.t_stop_s == pytest.approx(9.0)
