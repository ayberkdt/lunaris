"""Integration-level invariants for the assembled RHS.

Beyond per-kernel correctness, the equations of motion must behave as a physical
system: a closed circular orbit must conserve energy, a fixed-order integrator on
that RHS must show its theoretical convergence order when the step is refined, and
the engine's acceleration breakdown must agree with the RHS it reports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine

_SC = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
_R0 = 1.9e6  # circular-orbit radius [m]


def _point_mass_rhs():
    eng = DynamicsEngine(
        sc_props=_SC,
        flags=PerturbationFlags(
            enable_sh=False, enable_3rd_body_sun=False, enable_3rd_body_earth=False,
            enable_srp=False, enable_relativity_1pn=False, enable_earth_j2=False,
        ),
        gravity_model=None,
        ephem_manager=None,
        allow_identity_rotation=True,
    )
    rhs = eng.build_rhs(force_rebuild=True)
    rhs(0.0, np.zeros(6))  # warm up JIT
    return eng, rhs


def _circular_state() -> np.ndarray:
    v = math.sqrt(MU_MOON / _R0)
    return np.array([_R0, 0.0, 0.0, 0.0, v, 0.0], dtype=np.float64)


def _rk4(rhs, y0: np.ndarray, t0: float, tf: float, dt: float) -> np.ndarray:
    y = y0.astype(np.float64).copy()
    n = int(round((tf - t0) / dt))
    t = t0
    for _ in range(n):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
    return y


def _specific_energy(y: np.ndarray) -> float:
    r = np.linalg.norm(y[:3])
    v2 = float(y[3:6] @ y[3:6])
    return 0.5 * v2 - MU_MOON / r


def test_acceleration_breakdown_matches_rhs_for_point_mass() -> None:
    eng, rhs = _point_mass_rhs()
    y = _circular_state()
    comp = eng.get_acceleration_breakdown(0.0, y)
    rhs_accel = np.linalg.norm(rhs(0.0, y)[3:6])
    # Only one force is active, so its component norm is the total RHS accel.
    assert "Gravity (PM)" in comp
    assert comp["Gravity (PM)"] == pytest.approx(rhs_accel, rel=1e-12)


def test_energy_conserved_on_circular_orbit() -> None:
    _, rhs = _point_mass_rhs()
    y0 = _circular_state()
    period = 2.0 * math.pi * math.sqrt(_R0**3 / MU_MOON)
    yf = _rk4(rhs, y0, 0.0, period, dt=period / 4000.0)

    e0 = _specific_energy(y0)
    ef = _specific_energy(yf)
    assert abs(ef - e0) <= 1e-9 * abs(e0)
    # Closed orbit returns near its start after one period.
    assert np.linalg.norm(yf[:3] - y0[:3]) <= 1e-3 * _R0


def test_rk4_shows_fourth_order_convergence() -> None:
    """Halving dt must cut the global error by ~2^4 = 16 against the analytic
    circular-orbit solution (a clean check that the RHS feeds a 4th-order method
    a smooth, correct vector field)."""
    _, rhs = _point_mass_rhs()
    y0 = _circular_state()
    w = math.sqrt(MU_MOON / _R0**3)
    tf = 600.0
    r_exact = _R0 * np.array([math.cos(w * tf), math.sin(w * tf), 0.0])

    def err(dt: float) -> float:
        yf = _rk4(rhs, y0, 0.0, tf, dt)
        return float(np.linalg.norm(yf[:3] - r_exact))

    e_coarse = err(4.0)
    e_fine = err(2.0)
    ratio = e_coarse / e_fine
    # Theoretical 16x; allow a generous band for finite-step/roundoff effects.
    assert 12.0 <= ratio <= 20.0


def test_state_derivative_couples_position_to_velocity() -> None:
    """dy/dt[:3] must be exactly the velocity sub-vector (kinematic identity)."""
    _, rhs = _point_mass_rhs()
    y = _circular_state()
    dy = rhs(0.0, y)
    np.testing.assert_allclose(dy[:3], y[3:6], rtol=0.0, atol=0.0)
