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
