"""Analytic, model-independent tests for the ST-LRPS orbit-drift harness."""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.surrogate.st_lrps.evaluation.orbit_drift import (
    circular_orbit_state,
    energy_drift,
    orbit_drift,
    propagate_orbit,
)

MU = 4.9048695e12  # Moon GM [m^3/s^2]
RADIUS = 1.938e6  # ~200 km altitude [m]


def _kepler(mu: float):
    def accel(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        return -mu * r / rn ** 3
    return accel


def _kepler_potential(mu: float):
    def potential(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        return float(-mu / np.linalg.norm(r))
    return potential


def _period(mu: float, radius: float) -> float:
    return 2.0 * np.pi * np.sqrt(radius ** 3 / mu)


def test_propagate_circular_orbit_closes_and_conserves_energy():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    period = _period(MU, RADIUS)
    n = 2000
    orbit = propagate_orbit(_kepler(MU), r0, v0, dt_s=period / n, n_steps=n)

    # After one period the orbit should return close to its start.
    closure = np.linalg.norm(orbit["positions"][-1] - r0)
    assert closure / RADIUS < 1e-4

    # And RK4 should conserve mechanical energy to truncation floor.
    e = energy_drift(_kepler(MU), _kepler_potential(MU), r0, v0, dt_s=period / n, n_steps=n)
    assert e["relative_energy_drift_max_abs"] < 1e-6


def test_orbit_drift_identical_models_is_zero():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    period = _period(MU, RADIUS)
    out = orbit_drift(_kepler(MU), _kepler(MU), r0, v0, dt_s=period / 500, n_steps=500)
    assert out["position_drift_summary_m"]["max"] == 0.0
    assert out["velocity_drift_summary_m_s"]["max"] == 0.0
    assert out["position_drift_rel_max"] == 0.0


def test_orbit_drift_grows_for_mismatched_models():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    period = _period(MU, RADIUS)
    n = 1500  # 3 orbits
    out = orbit_drift(
        _kepler(MU), _kepler(MU * 1.001), r0, v0, dt_s=3 * period / n, n_steps=n,
    )
    drift = out["position_drift_m"]
    assert drift[0] == 0.0
    assert out["position_drift_summary_m"]["final"] > 0.0
    # Divergence accumulates: the last quarter drifts more than the first quarter.
    assert drift[-1] > drift[len(drift) // 4]


def test_energy_drift_detects_nonconservative_field():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    period = _period(MU, RADIUS)
    n = 4000  # 5 orbits
    dt = 5 * period / n

    pure = energy_drift(_kepler(MU), _kepler_potential(MU), r0, v0, dt_s=dt, n_steps=n)

    # Add a position-only rotational (non-conservative) perturbation a = eps*(-y, x, 0).
    eps = 1.0e-9

    def nonconservative(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        base = -MU * r / rn ** 3
        rot = np.zeros_like(r)
        rot[:, 0] = -eps * r[:, 1]
        rot[:, 1] = eps * r[:, 0]
        return base + rot

    drifted = energy_drift(nonconservative, _kepler_potential(MU), r0, v0, dt_s=dt, n_steps=n)

    # The conservative field stays at the integrator floor; the rotational field
    # pumps energy secularly and is unmistakably larger.
    assert pure["relative_energy_drift_max_abs"] < 1e-6
    assert drifted["relative_energy_drift_max_abs"] > 1e-4
    assert drifted["relative_energy_drift_max_abs"] > 100.0 * pure["relative_energy_drift_max_abs"]


def test_propagate_orbit_rejects_bad_input():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    with pytest.raises(ValueError, match="dt_s must be positive"):
        propagate_orbit(_kepler(MU), r0, v0, dt_s=0.0, n_steps=10)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        propagate_orbit(_kepler(MU), r0, v0, dt_s=1.0, n_steps=0)
    with pytest.raises(ValueError, match="3 elements"):
        propagate_orbit(_kepler(MU), [1.0, 2.0], v0, dt_s=1.0, n_steps=10)


def test_circular_orbit_state_validation():
    with pytest.raises(ValueError, match="must be positive"):
        circular_orbit_state(-1.0, RADIUS)
    r0, v0 = circular_orbit_state(MU, RADIUS)
    # Circular speed: |v| = sqrt(mu / r), perpendicular to r.
    np.testing.assert_allclose(np.linalg.norm(v0), np.sqrt(MU / RADIUS), rtol=1e-12)
    assert abs(float(np.dot(r0, v0))) < 1e-6
