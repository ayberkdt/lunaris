"""Surface/eclipse event start guards and the max-eclipse-duration event.

Regression tests:

* Terminator, node, target-flyover, occultation, stability and solar-eclipse
  events returned a constant for absolute ``t < 1 s``. A run starting on the
  other side of the surface reported a fake crossing at t = 1 s (a run
  starting in shadow got a false eclipse entry), a real crossing in the first
  second was lost, and a run whose t0 is an epoch never got the window. The
  window is now start-relative, returns NaN (never a crossing) and defaults
  to 0.
* ``make_max_eclipse_duration_event`` returned the same stale value at both
  ends of SciPy's root bracket, so plain ``solve_ivp`` raised
  "f(a) and f(b) must have different signs".
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lunaris.core.events import (
    make_max_eclipse_duration_event,
    make_node_crossing_event,
    make_occultation_event,
    make_solar_eclipse_event,
    make_stability_violation_event,
)


def _drift(_t: float, y: np.ndarray) -> np.ndarray:
    return np.concatenate([y[3:], np.zeros(3)])


def test_node_event_moving_away_from_the_plane_reports_no_crossing() -> None:
    ev = make_node_crossing_event(which="both")
    y0 = np.array([1.0e6, 0.0, -1000.0, 0.0, 100.0, -10.0])
    sol = solve_ivp(_drift, (0.0, 100.0), y0, events=[ev], max_step=0.25)
    assert sol.t_events[0].size == 0


def test_node_crossing_in_the_first_second_is_found() -> None:
    ev = make_node_crossing_event(which="asc")
    y0 = np.array([1.0e6, 0.0, -1.0, 0.0, 100.0, 2.0])  # z = 0 at t = 0.5 s
    sol = solve_ivp(_drift, (0.0, 10.0), y0, events=[ev], max_step=0.1, rtol=1e-10, atol=1e-12)
    assert sol.t_events[0].size == 1
    assert sol.t_events[0][0] == pytest.approx(0.5, abs=1e-8)


def test_solar_eclipse_event_starting_in_shadow_reports_no_entry() -> None:
    sun = np.array([1.5e11, 0.0, 0.0])
    ev = make_solar_eclipse_event(
        get_sun_vec_m=lambda t: sun, R_moon_m=1.7374e6, include_earth=False
    )
    # Behind the Moon, on the anti-Sun side, moving slowly: shadow throughout.
    y0 = np.array([-2.0e6, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert ev(0.0, y0) < 0.0
    sol = solve_ivp(_drift, (0.0, 30.0), y0, events=[ev], max_step=0.5)
    assert sol.t_events[0].size == 0


def test_guard_window_is_start_relative_and_nan() -> None:
    ev = make_occultation_event(
        body="earth", get_body_vec_m=lambda t: np.array([3.84e8, 0.0, 0.0]), R_ref_m=1.7374e6,
        t_guard_s=5.0,
    )
    y = np.array([2.0e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    t0 = 7.5e8  # an epoch-like start: the old absolute window never applied
    assert math.isnan(ev(t0, y))
    assert math.isnan(ev(t0 + 4.0, y))
    assert math.isfinite(ev(t0 + 5.0, y))


def test_stability_event_default_has_no_window() -> None:
    ev = make_stability_violation_event(mu=4.9028e12, R_ref_m=1.7374e6, e_max=0.01)
    r = 1.8374e6
    v = math.sqrt(4.9028e12 / r) * 1.1  # e ~ 0.21 > e_max at the start
    assert ev(0.0, np.array([r, 0.0, 0.0, 0.0, v, 0.0])) > 0.0


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_guard_window_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="t_guard_s"):
        make_node_crossing_event(t_guard_s=bad)


def test_max_eclipse_duration_event_localises_entry_plus_limit() -> None:
    first_eclipsed: list[float] = []

    def is_eclipsed(t: float, _y: np.ndarray) -> bool:
        if t >= 100.0:
            if not first_eclipsed:
                first_eclipsed.append(float(t))
            return True
        return False

    ev = make_max_eclipse_duration_event(is_eclipsed=is_eclipsed, max_duration_s=50.0)
    sol = solve_ivp(lambda t, y: [1.0], (0.0, 400.0), [0.0], events=[ev], max_step=40.0)
    assert sol.status == 1  # terminal event
    (t_hit,) = sol.t_events[0]
    assert t_hit == pytest.approx(first_eclipsed[0] + 50.0, abs=1e-6)
    assert 150.0 <= t_hit <= 150.0 + 40.0


def test_max_eclipse_duration_resets_when_eclipse_ends() -> None:
    ev = make_max_eclipse_duration_event(
        is_eclipsed=lambda t, _y: 10.0 <= t < 30.0, max_duration_s=50.0
    )
    y = np.zeros(1)
    assert ev(0.0, y) == 50.0
    assert ev(10.0, y) == 50.0
    assert ev(20.0, y) == pytest.approx(40.0)
    assert ev(15.0, y) == pytest.approx(45.0)  # probe: read-only, linear in t
    assert ev(30.0, y) == 50.0  # left the shadow
