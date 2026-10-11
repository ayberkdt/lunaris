"""A longitude-crossing event fires once per revolution, not also at the antipode.

``wrap_pi(lon - lon0)`` jumps from +pi to -pi at ``lon0 + 180 deg``; with the
default ``direction=0`` SciPy (and the fixed-step driver) read that jump as a
root, so every revolution reported the meridian and its antipode. A start-time
``+1`` guard also faked a crossing at ~1 s when ``lon < lon0`` at the start.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.core.events import make_longitude_crossing_event

MU = float(MU_MOON)
R0 = float(R_MOON) + 100e3
V0 = math.sqrt(MU / R0)
PERIOD = 2.0 * math.pi * R0 / V0


def _rhs(_t, y):
    dy = np.empty_like(y)
    dy[:3] = y[3:6]
    dy[3:6] = -MU * y[:3] / float(np.linalg.norm(y[:3])) ** 3
    return dy


def _identity(_t, r):
    return np.asarray(r, dtype=np.float64)


def _state(lon_deg: float) -> np.ndarray:
    lam = math.radians(lon_deg)
    return np.array([R0 * math.cos(lam), R0 * math.sin(lam), 0.0,
                     -V0 * math.sin(lam), V0 * math.cos(lam), 0.0])


def _crossings(lon_start_deg: float, direction: float, **kw) -> np.ndarray:
    ev = make_longitude_crossing_event(lon0_deg=0.0, r_i_to_bf=_identity, direction=direction, **kw)
    sol = solve_ivp(_rhs, (0.0, 2.5 * PERIOD), _state(lon_start_deg), method="DOP853",
                    rtol=1e-12, atol=1e-6, max_step=60.0, events=[ev])
    return np.asarray(sol.t_events[0])


@pytest.mark.parametrize("direction", [0.0, 1.0, -1.0])
def test_meridian_crossed_once_per_revolution(direction):
    # Prograde equatorial circular orbit from lon = 10 deg: the inertial
    # longitude reaches 360 and 720 deg in 2.5 periods (identity body frame).
    t = _crossings(10.0, direction)
    expected = np.array([350.0, 710.0]) / 360.0 * PERIOD if direction >= 0.0 else np.array([])
    np.testing.assert_allclose(t, expected, rtol=0.0, atol=1e-3)


def test_start_west_of_meridian_logs_no_crossing_at_one_second():
    # lon = -10 deg at the start: the meridian is 10 deg ahead, not at t ~ 1 s.
    t = _crossings(-10.0, 0.0)
    expected = np.array([10.0, 370.0, 730.0]) / 360.0 * PERIOD
    expected = expected[expected <= 2.5 * PERIOD]
    np.testing.assert_allclose(t, expected, rtol=0.0, atol=1e-3)
