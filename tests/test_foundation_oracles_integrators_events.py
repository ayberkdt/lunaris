"""Foundation oracles: integrator order on a nonlinear arc and apsis events.

An independent two-body oracle (its own Kepler solver; no lunaris maths) is the
truth for everything here.

All orbits: lunar point mass, inertial axes, SI units, solver time in seconds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.core.propagation.integrators.fixed_step import _accel_stepper
from lunaris.core.propagation.integrators.rk import _rk4_step_full
from lunaris.core.propagation.integrators.symplectic import _composed_step, _composition_weights

MU = float(MU_MOON)
R = float(R_MOON)


# -----------------------------------------------------------------------------
# Independent two-body oracle
# -----------------------------------------------------------------------------

def _coe_state(a: float, e: float, inc: float, raan: float, argp: float, mean_anom: float) -> np.ndarray:
    """Inertial (r, v) from classical elements and mean anomaly (own Newton solver)."""
    ecc_anom = mean_anom if e < 0.8 else math.pi
    for _ in range(100):
        d_e = (ecc_anom - e * math.sin(ecc_anom) - mean_anom) / (1.0 - e * math.cos(ecc_anom))
        ecc_anom -= d_e
        if abs(d_e) < 1e-15:
            break
    c_e, s_e = math.cos(ecc_anom), math.sin(ecc_anom)
    b = a * math.sqrt(1.0 - e * e)
    n = math.sqrt(MU / a ** 3)
    xp, yp = a * (c_e - e), b * s_e
    vxp = -a * n * s_e / (1.0 - e * c_e)
    vyp = b * n * c_e / (1.0 - e * c_e)
    co, so, ci, si = math.cos(raan), math.sin(raan), math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    p_hat = np.array([co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si])
    q_hat = np.array([-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si])
    return np.concatenate([xp * p_hat + yp * q_hat, vxp * p_hat + vyp * q_hat])


class _Orbit:
    def __init__(self, a: float, e: float, inc: float, raan: float, argp: float, m0: float) -> None:
        self.a, self.e, self.inc, self.raan, self.argp, self.m0 = a, e, inc, raan, argp, m0
        self.n = math.sqrt(MU / a ** 3)
        self.period = 2.0 * math.pi / self.n

    def state(self, t: float) -> np.ndarray:
        return _coe_state(self.a, self.e, self.inc, self.raan, self.argp, self.m0 + self.n * t)

    def apsis_times(self, t0: float, t1: float, which: str) -> np.ndarray:
        """Epochs in [t0, t1] where the mean anomaly is 0 (peri) or pi (apo)."""
        target = 0.0 if which == "peri" else math.pi
        k0 = math.floor((self.m0 + self.n * t0 - target) / (2.0 * math.pi)) - 1
        out = []
        for k in range(k0, k0 + 10000):
            t = (target + 2.0 * math.pi * k - self.m0) / self.n
            if t > t1:
                break
            if t >= t0:
                out.append(t)
        return np.asarray(out)


# -----------------------------------------------------------------------------
# 1. Fixed-step integrator order on a nonlinear Kepler arc
# -----------------------------------------------------------------------------

_ARC_ORBIT = _Orbit(6.0e6, 0.5, 0.4, 0.3, 0.2, -0.6)
_ARC_DURATION = 0.45 * _ARC_ORBIT.period
_ARC_Y0 = _ARC_ORBIT.state(0.0)
_ARC_YT = _ARC_ORBIT.state(_ARC_DURATION)
# Position scale ~1e7 m times float64 eps ~1e-16 gives a ~1e-9 m error floor; every
# error in the order fits below must exceed it by _FLOOR_MARGIN.
_ERR_FLOOR_M = 1e-9
_FLOOR_MARGIN = 1e2


def _arc_accel(_t: float, y: np.ndarray) -> np.ndarray:
    r = y[:3]
    return -MU * r / float(np.linalg.norm(r)) ** 3


def _arc_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    out = np.empty(6)
    out[:3] = y[3:6]
    out[3:] = _arc_accel(_t, y)
    return out


def _arc_error(step, n: int) -> float:
    """Final position error [m] vs the oracle after ``n`` equal steps over the arc."""
    h = _ARC_DURATION / n
    y = _ARC_Y0.copy()
    t = 0.0
    for _ in range(n):
        y = step(t, y, h)
        t += h
    return float(np.linalg.norm(y[:3] - _ARC_YT[:3]))


def _observed_order(step, ns: list[int]) -> float:
    """Slope of log(error) vs log(1/n) across the whole sweep (end points)."""
    errs = [_arc_error(step, n) for n in ns]
    assert min(errs) > _FLOOR_MARGIN * _ERR_FLOOR_M, f"errors {errs} too close to the float64 floor"
    return math.log(errs[0] / errs[-1]) / math.log(ns[-1] / ns[0])


def _library_step(method: str):
    if method == "RK4":
        return lambda t, y, h: _rk4_step_full(_arc_rhs, t, y, h)
    stepper = _accel_stepper(method)
    return lambda t, y, h: stepper(_arc_accel, t, y, h)


def _outer_nested_weights(levels: int) -> tuple[float, ...]:
    """Yoshida (1990) triple jump with the NEW coefficients as the outer ones.

    Psi_{2k+2}(h) = Psi_{2k}(x1 h) Psi_{2k}(x0 h) Psi_{2k}(x1 h), so each level's
    coefficient multiplies the whole lower-order composition. Written out here
    independently of the library so the control below does not share its code.
    """
    weights = [1.0]
    for k in range(1, levels + 1):
        root = 2.0 ** (1.0 / (2 * k + 1))
        x1 = 1.0 / (2.0 - root)
        x0 = -root / (2.0 - root)
        weights = [x1 * w for w in weights] + [x0 * w for w in weights] + [x1 * w for w in weights]
    return tuple(weights)


# Step-count sweeps keep every error >= ~1e-6 m (>> float64 floor), so the fit
# sees the asymptotic truncation regime rather than round-off.
_SWEEPS = {
    "VV": [400, 800, 1600, 3200],
    "PEFRL": [100, 200, 400, 800],
    "Y4": [100, 200, 400, 800],
    "RKN4": [100, 200, 400, 800],
    "RK4": [100, 200, 400, 800],
    "Y6": [240, 480, 960, 1920],
    "Y8": [60, 120, 240, 480],
}


@pytest.mark.parametrize("method,expected,tol", [
    # (canonical method, nominal order, tol); tol is relative to the nominal order and
    # covers the sweep's residual higher-order contamination.
    ("VV", 2.0, 0.3),
    ("PEFRL", 4.0, 0.4),
    ("Y4", 4.0, 0.4),
    ("RKN4", 4.0, 0.4),
    ("RK4", 4.0, 0.4),
])
def test_kepler_arc_order_of_fourth_and_second_order_integrators(method, expected, tol):
    order = _observed_order(_library_step(method), _SWEEPS[method])
    assert abs(order - expected) < tol, f"{method} observed order {order:.3f}"


@pytest.mark.parametrize("method,expected,tol", [
    # The inward-nested weights (each Verlet sub-step split by the newest level) measured
    # 4.67 (Y6) / 4.81 (Y8) on these sweeps: order 6/8 only on a linear problem.
    ("Y6", 6.0, 0.4),
    ("Y8", 8.0, 0.5),
])
def test_kepler_arc_order_of_high_order_yoshida(method, expected, tol):
    order = _observed_order(_library_step(method), _SWEEPS[method])
    assert abs(order - expected) < tol, f"{method} observed order {order:.3f}"


@pytest.mark.parametrize("levels,method,expected,tol", [(2, "Y6", 6.0, 0.4), (3, "Y8", 8.0, 0.5)])
def test_outer_nested_weights_reach_nominal_order_control(levels, method, expected, tol):
    """Control: the independently written outer nesting is order 6 / 8 and equals the library's."""
    nested = _outer_nested_weights(levels)
    assert np.allclose(nested, _composition_weights(levels), rtol=0, atol=1e-14)  # atol: float64 round-off on O(1) weights
    assert sum(nested) == pytest.approx(1.0, abs=1e-13)  # abs: float64 round-off in a 3**levels-term sum
    step = lambda t, y, h: _composed_step(_arc_accel, t, y, h, nested)  # noqa: E731
    order = _observed_order(step, _SWEEPS[method])
    assert abs(order - expected) < tol, f"outer-nested {method} observed order {order:.3f}"
