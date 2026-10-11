"""Periapsis/apoapsis events against analytic Kepler apsis times.

Regression for the absolute ``t < 1 s`` guard in ``_make_rdot_event``: it
returned +1 for the first second of absolute time, so a run starting with
``r . v < 0`` logged an apoapsis at about 1 s and a real apsis inside the first
second was hidden; for ``t0 > 1 s`` it did nothing. The guard is now relative
to the run start, and a start that sits on an apsis is not itself an event.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.events import make_aposelene_event, make_periselene_event
from lunaris.core.propagation.propagator import propagate

MU = float(MU_MOON)
RP = float(R_MOON) + 300e3
RA = float(R_MOON) + 3000e3
A = 0.5 * (RP + RA)
E_ECC = (RA - RP) / (RA + RP)
PERIOD = 2.0 * math.pi * math.sqrt(A**3 / MU)
N_MEAN = 2.0 * math.pi / PERIOD


class _PointMass:
    grav = None
    ephem = None

    def build_rhs(self):
        def rhs(_t, y):
            y = np.asarray(y, dtype=np.float64)
            dy = np.empty_like(y)
            dy[:3] = y[3:6]
            dy[3:6] = -MU * y[:3] / float(np.linalg.norm(y[:3])) ** 3
            return dy

        return rhs


def _state_at_mean_anomaly(m: float) -> np.ndarray:
    """Inertial state of the test orbit at mean anomaly ``m`` [rad] (periapsis on +x)."""
    big_e = m
    for _ in range(50):
        big_e -= (big_e - E_ECC * math.sin(big_e) - m) / (1.0 - E_ECC * math.cos(big_e))
    nu = 2.0 * math.atan2(math.sqrt(1.0 + E_ECC) * math.sin(0.5 * big_e),
                          math.sqrt(1.0 - E_ECC) * math.cos(0.5 * big_e))
    p = A * (1.0 - E_ECC**2)
    r = p / (1.0 + E_ECC * math.cos(nu))
    h = math.sqrt(MU * p)
    vr = MU / h * E_ECC * math.sin(nu)
    vt = h / r
    c, s = math.cos(nu), math.sin(nu)
    return np.array([r * c, r * s, 0.0, vr * c - vt * s, vr * s + vt * c, 0.0])


def _cfg(method: str) -> PropagatorConfig:
    return PropagatorConfig(
        method=method,
        rtol=1e-12,
        atol=1e-9,
        verbose=False,
        compute_2body_baseline=False,
        use_nyquist_max_step=False,
        user_max_step_s=10.0,
        events=EventConfig(detect_impact=False, enable_peri_apo_events=True),
    )


def _apsis_times(m0: float, t0: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic periapsis and apoapsis times in (t0, t0 + duration] for a start at mean anomaly m0."""
    peri, apo = [], []
    for k in range(-1, 5):
        tp = t0 + (2.0 * math.pi * k - m0) / N_MEAN
        ta = t0 + (2.0 * math.pi * k + math.pi - m0) / N_MEAN
        if t0 < tp <= t0 + duration:
            peri.append(tp)
        if t0 < ta <= t0 + duration:
            apo.append(ta)
    return np.array(peri), np.array(apo)


def _run(method: str, m0: float, t0: float, duration: float):
    y0 = _state_at_mean_anomaly(m0)
    tc = TimeConfig(t0_s=t0, duration_s=duration, output_dt_s=duration / 50.0, samples_per_period=50)
    res = propagate(_PointMass(), y0, _cfg(method), time_cfg=tc)
    return np.asarray(res.t_events[0]), np.asarray(res.t_events[1])


# Event-time tolerance [s]: DOP853 at rtol 1e-12 locates the root to well below
# 1 ms; RK4 with a 10 s step and bisection refinement to ~1e-3 s.
_TOL = {"DOP853": 1e-3, "RK4": 5e-2}


@pytest.mark.parametrize("method", ["DOP853", "RK4"])
def test_inbound_start_logs_no_fake_apoapsis(method):
    # Start inbound (r.v < 0), half-way from apoapsis to periapsis; no apsis in 600 s.
    t_peri, t_apo = _run(method, 1.5 * math.pi, 0.0, 600.0)
    assert t_peri.size == 0
    assert t_apo.size == 0


@pytest.mark.parametrize("method", ["DOP853", "RK4"])
@pytest.mark.parametrize("t0", [0.0, 5.0e4])
def test_apsis_inside_first_second_is_reported(method, t0):
    # Periapsis 0.5 s after the start.
    m0 = -0.5 * N_MEAN
    t_peri, t_apo = _run(method, m0, t0, 600.0)
    np.testing.assert_allclose(t_peri, [t0 + 0.5], rtol=0.0, atol=_TOL[method])
    assert t_apo.size == 0


@pytest.mark.parametrize("method", ["DOP853", "RK4"])
@pytest.mark.parametrize("m0", [0.0, math.pi])
def test_start_on_apsis_is_not_an_event_and_later_apses_match_kepler(method, m0):
    duration = 1.6 * PERIOD
    t_peri, t_apo = _run(method, m0, 0.0, duration)
    exp_peri, exp_apo = _apsis_times(m0, 0.0, duration)
    np.testing.assert_allclose(t_peri, exp_peri, rtol=0.0, atol=_TOL[method])
    np.testing.assert_allclose(t_apo, exp_apo, rtol=0.0, atol=_TOL[method])


def test_guard_is_relative_to_the_first_evaluation():
    inbound = _state_at_mean_anomaly(1.5 * math.pi)
    peri = make_periselene_event(t_guard_s=5.0)
    assert math.isnan(peri(1000.0, inbound))  # start of the run
    assert math.isnan(peri(1004.0, inbound))  # inside the window
    assert peri(1006.0, inbound) < 0.0  # after it, the true r.v sign
    # Default: no window at all.
    apo = make_aposelene_event()
    assert apo(1000.0, inbound) < 0.0
