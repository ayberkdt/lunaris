"""SciPy event root-finding on a circular orbit.

``r . v`` is at rounding level on a circular orbit, so the solver state and the
step interpolant can give it opposite signs at the same step end; brentq then
raised "f(a) and f(b) must have different signs" inside solve_ivp for RK45,
Radau and LSODA. The step-consistent event wrapper must prevent that without
changing any run where the inconsistency does not occur.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import lunaris.core.propagation.scipy_runner as scipy_runner
from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import PropagatorConfig, TimeConfig
from lunaris.core.propagation.propagator import propagate

MU = float(MU_MOON)
R = float(R_MOON)


class _PointMass:
    grav = None
    ephem = None
    flags = SimpleNamespace()

    def build_rhs(self):
        def rhs(_t, y):
            r = y[:3]
            rn = float(np.linalg.norm(r))
            dy = np.empty_like(y)
            dy[:3] = y[3:6]
            dy[3:6] = -MU * r / rn**3
            return dy

        return rhs


def _state(speed_factor: float) -> np.ndarray:
    r0 = R + 100e3
    v = math.sqrt(MU / r0) * speed_factor
    return np.array([r0, 0.0, 0.0, 0.0, 0.6 * v, 0.8 * v])


def _run(method: str, y0: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return propagate(
            _PointMass(),
            y0,
            PropagatorConfig(method=method, verbose=False, compute_2body_baseline=False),
            time_cfg=TimeConfig(duration_s=86400.0, output_dt_s=600.0),
        )


@pytest.mark.parametrize("method", ["RK45", "RADAU", "LSODA"])
def test_circular_orbit_events_do_not_crash_root_finding(method: str) -> None:
    res = _run(method, _state(1.0))
    assert np.all(np.isfinite(res.y))


def test_unwrapped_events_reproduce_the_brentq_failure(monkeypatch) -> None:
    monkeypatch.setattr(scipy_runner, "_step_consistent_events", lambda events: list(events))
    with pytest.raises(ValueError, match="different signs"):
        _run("RK45", _state(1.0))


def test_wrapper_is_bit_identical_when_signs_are_consistent(monkeypatch) -> None:
    wrapped = _run("RK45", _state(1.05))
    monkeypatch.setattr(scipy_runner, "_step_consistent_events", lambda events: list(events))
    plain = _run("RK45", _state(1.05))
    assert np.array_equal(wrapped.t, plain.t)
    assert np.array_equal(wrapped.y, plain.y)
    assert len(wrapped.t_events) == len(plain.t_events)
    for a, b in zip(wrapped.t_events, plain.t_events, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_a_root_exactly_on_a_step_end_is_reported_once() -> None:
    # SciPy reads g = 0 at a step end as a crossing at the end of that step
    # and again at the start of the next. A time limit that is a whole number
    # of steps after a step end puts its root there.
    from scipy.integrate import solve_ivp

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -y[0]])

    kwargs = dict(method="DOP853", rtol=1e-9, atol=1e-12, max_step=60.0)
    probe = solve_ivp(rhs, (0.0, 600.0), [1.0, 0.0], dense_output=True, **kwargs)
    root = float(probe.sol.ts[len(probe.sol.ts) // 2])  # a step end of the same solve

    def event(t: float, y: np.ndarray) -> float:
        return root - t

    event.terminal = False  # type: ignore[attr-defined]
    event.direction = -1.0  # type: ignore[attr-defined]
    plain = solve_ivp(rhs, (0.0, 600.0), [1.0, 0.0], events=[event], **kwargs)
    assert plain.t_events[0].size == 2  # SciPy alone: the same root twice
    wrapped = solve_ivp(rhs, (0.0, 600.0), [1.0, 0.0], events=scipy_runner._step_consistent_events([event]), **kwargs)
    assert wrapped.t_events[0].size == 1
    assert wrapped.t_events[0][0] == pytest.approx(root, abs=1e-9)  # abs: brentq's xtol on a 60 s step


def test_event_wrapper_reads_an_exact_zero_step_end_as_just_past_the_root() -> None:
    def event(_t: float, y: np.ndarray) -> float:
        return float(y[0])

    event.terminal = False  # type: ignore[attr-defined]
    (wrapped,) = scipy_runner._step_consistent_events([event])
    assert wrapped(0.0, np.array([1.0])) == 1.0
    # Away from the previous step end's +1, by the smallest normal float.
    assert wrapped(1.0, np.array([0.0])) == -float(np.finfo(np.float64).tiny)
    # The root search re-evaluates the step end on the interpolant: the side
    # chosen above wins over a rounding-level fresh value of the other sign.
    assert wrapped(1.0, np.array([2e-10])) == -float(np.finfo(np.float64).tiny)
    # Interior times are never substituted.
    assert wrapped(0.5, np.array([0.25])) == 0.25
