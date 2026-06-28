# tests/test_integrators.py
"""
Integrator verification for ``lunaris.core.propagation.propagator``.

Covers the in-house fixed-step family added alongside the SciPy adaptive
methods:

  * Symplectic: velocity-Verlet (VV), PEFRL, and the Yoshida triple-jump
    compositions Y4 / Y6 / Y8.
  * Runge-Kutta-Nystrom: RKN4.
  * Classical explicit Runge-Kutta: RK4 (full-state).

Truth is an *exact* two-body Kepler propagation (solve Kepler's equation, then
rotate the perifocal state with the library ``coe_to_rv``), so global-error
convergence orders can be measured directly rather than against another
numerical integrator. We also check the symplectic methods' hallmark bounded
energy drift, per-method circular-orbit consistency, and the state-dimension
contracts (RK4 accepts augmented states; symplectic/RKN reject them).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.math_utils import coe_to_rv, rv_to_coe_select
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.propagation import propagator as P
from lunaris.core.propagation.propagator import propagate

MU = float(MU_MOON)
R = float(R_MOON)


# -----------------------------------------------------------------------------
# Minimal exact point-mass dynamics (matches the propagator fallback constants)
# -----------------------------------------------------------------------------

class FakePointMassDynamics:
    grav = None
    ephem = None

    def build_rhs(self):
        def rhs(t, y):
            y = np.asarray(y, dtype=np.float64)
            r = y[:3]
            v = y[3:6]
            rn = float(np.linalg.norm(r))
            a = -MU * r / (rn ** 3)
            dy = np.empty_like(y)
            dy[:3] = v
            dy[3:6] = a
            if y.size > 6:
                dy[6:] = 0.0
            return dy

        return rhs


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _state_from_coe(alt_peri_m: float, e: float) -> tuple[np.ndarray, float]:
    """Build an initial state at periapsis for a given altitude and eccentricity."""
    rp = R + alt_peri_m
    a = rp / (1.0 - e)
    r, v = coe_to_rv(a, e, math.radians(35.0), math.radians(50.0), math.radians(40.0), 0.0, MU)
    T = 2.0 * math.pi * math.sqrt(a ** 3 / MU)
    return np.concatenate([r, v]), T


def _kepler_propagate(y0: np.ndarray, dt: float) -> np.ndarray:
    """Exact two-body propagation of (r, v) by dt seconds."""
    r0 = y0[:3]
    v0 = y0[3:6]
    a, e, inc, raan, argp, nu0 = rv_to_coe_select(r0, v0, MU, mode="coe6")

    n = math.sqrt(MU / a ** 3)
    # true -> eccentric -> mean anomaly
    E0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu0 / 2.0),
                          math.sqrt(1.0 + e) * math.cos(nu0 / 2.0))
    M0 = E0 - e * math.sin(E0)
    M = M0 + n * dt

    # Newton solve of Kepler's equation
    E = M if e < 0.8 else math.pi
    for _ in range(200):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < 1e-15:
            break

    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                          math.sqrt(1.0 - e) * math.cos(E / 2.0))
    r, v = coe_to_rv(a, e, inc, raan, argp, nu, MU)
    return np.concatenate([r, v])


def _cfg(method: str, h: float) -> PropagatorConfig:
    return PropagatorConfig(
        method=method,
        verbose=False,
        compute_2body_baseline=False,
        use_nyquist_max_step=False,
        user_max_step_s=h,
        events=EventConfig(detect_impact=False, impact_alt_km=0.0, enable_peri_apo_events=False),
    )


def _run_final_state(method: str, y0: np.ndarray, duration_s: float, n_steps: int) -> np.ndarray:
    """Integrate ``duration_s`` with exactly ``n_steps`` fixed sub-steps; return final state."""
    h = duration_s / n_steps
    # output_dt_s == duration -> a 2-point grid, so the whole span is one segment
    # subdivided into n_steps sub-steps of size h.
    tc = TimeConfig(duration_s=duration_s, output_dt_s=duration_s, samples_per_period=2)
    res = propagate(FakePointMassDynamics(), y0, _cfg(method, h), time_cfg=tc)
    return np.asarray(res.y[-1, :6], dtype=np.float64)


def _global_error(method: str, y0: np.ndarray, duration_s: float, n_steps: int) -> float:
    y_num = _run_final_state(method, y0, duration_s, n_steps)
    y_true = _kepler_propagate(y0, duration_s)
    # Position error, normalized by orbit scale.
    return float(np.linalg.norm(y_num[:3] - y_true[:3]))


def _measured_order(method: str, y0: np.ndarray, duration_s: float, n_coarse: int) -> float:
    e1 = _global_error(method, y0, duration_s, n_coarse)
    e2 = _global_error(method, y0, duration_s, 2 * n_coarse)
    assert e1 > 0.0 and e2 > 0.0
    return math.log2(e1 / e2)


# -----------------------------------------------------------------------------
# Raw-stepper order harness (1-D harmonic oscillator x'' = -x, exact truth)
# -----------------------------------------------------------------------------
# This isolates each stepper's algebraic order from the propagator machinery and
# from any analytic-Kepler reference floor: the oscillator solution is known in
# closed form to machine precision, so high orders (incl. Y8) measure cleanly.

def _sho_accel(t, y6):
    return np.array([-y6[0], 0.0, 0.0], dtype=np.float64)


def _sho_rhs(t, y):
    return np.array([y[3], 0.0, 0.0, -y[0], 0.0, 0.0], dtype=np.float64)


def _sho_error(method: str, n: int) -> float:
    """Global error after one period of x''=-x from x0=1, v0=0 (exact: x=1, v=0)."""
    T = 2.0 * math.pi
    h = T / n
    y = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    t = 0.0
    m = P._norm_method(method)
    rhs_steppers = {"RK4": P._rk4_step_full, "RK8": P._rk8_step_full}
    for _ in range(n):
        if m in P._RHS_METHODS:
            y = rhs_steppers[P._RHS_METHODS[m]](_sho_rhs, t, y, h)
        else:
            canonical = P._ACCEL_METHODS[m]
            y = P._accel_stepper(canonical)(_sho_accel, t, y, h)
        t += h
    return abs(y[0] - 1.0) + abs(y[3] - 0.0)


def _sho_order(method: str, n_coarse: int) -> float:
    e1 = _sho_error(method, n_coarse)
    e2 = _sho_error(method, 2 * n_coarse)
    assert e1 > 0.0 and e2 > 0.0
    return math.log2(e1 / e2)


# (method, coarse step count, expected order, tolerance)
_RAW_ORDER_CASES = [
    ("VV", 200, 2.0, 0.3),
    ("YOSHIDA4", 50, 4.0, 0.4),
    ("PEFRL", 50, 4.0, 0.4),
    ("RKN4", 50, 4.0, 0.4),
    ("RK4", 50, 4.0, 0.4),
    ("YOSHIDA6", 30, 6.0, 0.5),
    ("YOSHIDA8", 10, 8.0, 0.6),
    ("RK8", 6, 8.0, 0.6),
]


@pytest.mark.parametrize("method,n_coarse,expected,tol", _RAW_ORDER_CASES)
def test_raw_integrator_order(method, n_coarse, expected, tol):
    order = _sho_order(method, n_coarse)
    assert abs(order - expected) < tol, f"{method}: measured order {order:.2f}, expected ~{expected}"


# -----------------------------------------------------------------------------
# Composition framework
# -----------------------------------------------------------------------------

def test_composition_weights_sum_to_one_and_count():
    assert sum(P._Y4_WEIGHTS) == pytest.approx(1.0, abs=1e-13)
    assert sum(P._Y6_WEIGHTS) == pytest.approx(1.0, abs=1e-13)
    assert sum(P._Y8_WEIGHTS) == pytest.approx(1.0, abs=1e-13)
    # Triple-jump: 3^levels sub-steps (orders 4/6/8 -> 1/2/3 levels).
    assert len(P._Y4_WEIGHTS) == 3
    assert len(P._Y6_WEIGHTS) == 9
    assert len(P._Y8_WEIGHTS) == 27


def test_y4_weights_match_classic_yoshida_coefficients():
    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
    assert P._Y4_WEIGHTS == pytest.approx((w1, w0, w1), rel=0, abs=1e-15)


# -----------------------------------------------------------------------------
# Convergence order (global error vs exact two-body)
# -----------------------------------------------------------------------------

# (method, coarse step count, expected order, tolerance)
_ORDER_CASES = [
    ("VV", 60, 2.0, 0.4),
    ("YOSHIDA4", 24, 4.0, 0.5),
    ("PEFRL", 24, 4.0, 0.5),
    ("RKN4", 24, 4.0, 0.5),
    ("RK4", 24, 4.0, 0.5),
    ("YOSHIDA6", 12, 6.0, 0.7),
]


@pytest.mark.parametrize("method,n_coarse,expected,tol", _ORDER_CASES)
def test_convergence_order(method, n_coarse, expected, tol):
    # Moderately eccentric orbit exercises the higher-order error terms.
    y0, T = _state_from_coe(alt_peri_m=500e3, e=0.3)
    duration = T / 6.0  # a short arc keeps the error in the asymptotic regime
    order = _measured_order(method, y0, duration, n_coarse)
    assert abs(order - expected) < tol, f"{method}: measured order {order:.2f}, expected ~{expected}"


# -----------------------------------------------------------------------------
# Symplectic energy behaviour
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("method,max_drift", [
    ("VV", 5e-6),
    ("PEFRL", 1e-8),
    ("YOSHIDA4", 1e-8),
    ("YOSHIDA6", 1e-10),
])
def test_symplectic_energy_drift_is_bounded(method, max_drift):
    # Circular orbit over many revolutions: symplectic methods keep the specific
    # energy bounded (no secular drift), unlike a generic RK method.
    r0 = R + 300e3
    v_circ = math.sqrt(MU / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
    T = 2.0 * math.pi * math.sqrt(r0 ** 3 / MU)

    n_rev = 20
    duration = n_rev * T
    h = T / 200.0
    tc = TimeConfig(duration_s=duration, output_dt_s=T / 20.0, samples_per_period=2,
                    max_points_cap=1_000_000)
    res = propagate(FakePointMassDynamics(), y0, _cfg(method, h), time_cfg=tc)

    r = res.y[:, :3]
    v = res.y[:, 3:6]
    rn = np.linalg.norm(r, axis=1)
    eps = 0.5 * np.sum(v ** 2, axis=1) - MU / rn
    rel_drift = abs((eps.max() - eps.min()) / eps.mean())
    assert rel_drift < max_drift, f"{method}: energy drift {rel_drift:.2e} exceeds {max_drift:.0e}"


# -----------------------------------------------------------------------------
# Per-method consistency: a circular orbit stays circular
# -----------------------------------------------------------------------------

# Radial-drift tolerance scales with the method's order (VV is only 2nd order, so
# its O(h^2) radial oscillation at h=T/400 is the largest).
_CIRCULAR_DRIFT_TOL = {
    "VV": 2e-4,
    "PEFRL": 1e-5,
    "YOSHIDA4": 1e-5,
    "YOSHIDA6": 1e-6,
    "YOSHIDA8": 1e-6,
    "RKN4": 1e-5,
    "RK4": 1e-5,
    "RK8": 1e-6,
}


@pytest.mark.parametrize("method", list(_CIRCULAR_DRIFT_TOL))
def test_circular_orbit_radius_is_preserved(method):
    r0 = R + 250e3
    v_circ = math.sqrt(MU / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
    T = 2.0 * math.pi * math.sqrt(r0 ** 3 / MU)

    h = T / 400.0
    tc = TimeConfig(duration_s=T, output_dt_s=T / 200.0, samples_per_period=2)
    res = propagate(FakePointMassDynamics(), y0, _cfg(method, h), time_cfg=tc)

    radii = np.linalg.norm(res.y[:, :3], axis=1)
    rel_drift = (radii.max() - radii.min()) / r0
    assert rel_drift < _CIRCULAR_DRIFT_TOL[method], f"{method}: radial drift {rel_drift:.2e}"
    assert not res.impacted


# -----------------------------------------------------------------------------
# State-dimension contracts
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["RK4", "RK8"])
def test_rk_methods_support_augmented_state(method):
    # 7-element state [x,y,z,vx,vy,vz,mass]; mass is inert in the fake RHS.
    r0 = R + 200e3
    v_circ = math.sqrt(MU / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0, 500.0])
    T = 2.0 * math.pi * math.sqrt(r0 ** 3 / MU)
    tc = TimeConfig(duration_s=T / 4.0, output_dt_s=T / 40.0, samples_per_period=2)
    res = propagate(FakePointMassDynamics(), y0, _cfg(method, T / 200.0), time_cfg=tc)
    assert res.y.shape[1] == 7
    # Mass column is carried through unchanged.
    assert np.allclose(res.y[:, 6], 500.0, atol=1e-9)


@pytest.mark.parametrize("method", ["VV", "PEFRL", "YOSHIDA4", "YOSHIDA6", "YOSHIDA8", "RKN4"])
def test_acceleration_methods_reject_augmented_state(method):
    r0 = R + 200e3
    v_circ = math.sqrt(MU / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0, 500.0])
    T = 2.0 * math.pi * math.sqrt(r0 ** 3 / MU)
    tc = TimeConfig(duration_s=T / 4.0, output_dt_s=T / 40.0, samples_per_period=2)
    with pytest.raises(ValueError):
        propagate(FakePointMassDynamics(), y0, _cfg(method, T / 200.0), time_cfg=tc)


# -----------------------------------------------------------------------------
# SciPy adaptive methods must all run end-to-end through propagate()
# -----------------------------------------------------------------------------
# Regression for the RADAU casing bug: solve_ivp wants the exact name "Radau",
# but the propagator normalizes method tokens to upper case. Every adaptive
# method the UI exposes must integrate without raising.

@pytest.mark.parametrize("method", ["DOP853", "RK45", "RK23", "RADAU", "BDF", "LSODA"])
def test_scipy_adaptive_methods_run(method):
    y0, T = _state_from_coe(alt_peri_m=400e3, e=0.1)
    tc = TimeConfig(duration_s=T / 4.0, output_dt_s=T / 40.0, samples_per_period=2)
    cfg = PropagatorConfig(
        method=method, rtol=1e-8, atol=1e-10, verbose=False,
        compute_2body_baseline=False, use_nyquist_max_step=False,
        events=EventConfig(detect_impact=False, impact_alt_km=0.0, enable_peri_apo_events=False),
    )
    res = propagate(FakePointMassDynamics(), y0, cfg, time_cfg=tc)
    assert np.all(np.isfinite(res.y))
    # Compare the final state to the exact two-body solution (loose, just sanity).
    y_true = _kepler_propagate(y0, T / 4.0)
    assert np.linalg.norm(res.y[-1, :3] - y_true[:3]) < 1e4


@pytest.mark.parametrize("method", ["VV", "PEFRL", "YOSHIDA4", "YOSHIDA6", "YOSHIDA8", "RKN4", "RK4", "RK8"])
def test_fixed_step_impact_event_is_detected(method):
    # Exercises the fixed-step event-refinement path (the `step` closure) with a
    # geometric terminal impact event for every in-house method.
    rp = R + 50e3
    v0 = math.sqrt(MU / rp) * 0.3  # sub-circular -> descends and impacts
    y0 = np.array([rp, 0.0, 0.0, 0.0, v0, 0.0])
    tc = TimeConfig(duration_s=3600.0, output_dt_s=30.0, samples_per_period=2)
    cfg = PropagatorConfig(
        method=method, verbose=False, compute_2body_baseline=False,
        use_nyquist_max_step=False, user_max_step_s=10.0,
        events=EventConfig(detect_impact=True, impact_alt_km=0.0, enable_peri_apo_events=True),
    )
    res = propagate(FakePointMassDynamics(), y0, cfg, time_cfg=tc)
    assert res.impacted and res.stop_reason == "impact"
    alt_km = (float(np.linalg.norm(res.y_impact[:3])) - R) / 1000.0
    assert abs(alt_km) < 1.0  # impact recorded at ~surface


if __name__ == "__main__":
    import sys
    print("Run with: python -m pytest -q tests/test_integrators.py")
    sys.exit(0)
