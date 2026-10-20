"""Analytic, model-independent tests for the ST-LRPS orbit-drift harness."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

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


class _TotalOnlyRuntime:
    """Runtime whose total differs from any point-mass-plus-residual reconstruction."""

    def domain_status(self, r):
        return {"recommended_fallback": False, "reason": "ok"}

    def predict_total_accel_fixed(self, r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        return np.full_like(r, 7.0e-3)

    def predict_residual_accel_fixed(self, r):  # pragma: no cover - must not be used
        raise AssertionError("orbit_drift must take the runtime's total, not rebuild it")


def test_domain_tracking_total_accel_returns_the_runtimes_total_unchanged():
    """Ported from Sidera 4a8784c1: no point mass is added on top of the runtime total.

    The old wrapper returned ``-mu r/r^3 + residual`` for every artifact, counting
    the monopole twice for a full-field artifact and dropping degrees
    2..base_degree for an SH-baseline residual artifact.
    """
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _DomainTrackingTotalAccel

    accel = _DomainTrackingTotalAccel(_TotalOnlyRuntime(), strict_domain=False)
    out = accel(np.array([[RADIUS, 0.0, 0.0], [0.0, RADIUS, 0.0]]))
    np.testing.assert_array_equal(out, np.full((2, 3), 7.0e-3))


def test_orbit_drift_runtime_uses_the_artifacts_own_base(tmp_path):
    """Real artifacts: a full-field model gets no extra monopole; an SH baseline is refused."""
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _runtime_total_accel
    from st_lrps_contract_test_utils import make_contract_run

    r = np.array([[RADIUS, 0.0, 0.0]])  # Moon-fixed, m

    # Residual over an SH degree-20 baseline: the runtime cannot supply the
    # baseline, so propagation is refused instead of silently using a point mass.
    sh_run = make_contract_run(tmp_path / "sh", degree_min=20, degree_max=60)
    sh_accel, _, _ = _runtime_total_accel(str(sh_run["run_dir"]), "cpu")
    with pytest.raises(ValueError, match="base_accel_fn"):
        sh_accel(r)

    # Full field with no baseline: the zero-weight network's output is the total.
    # (The fixture builder only writes residual datasets, so the loaded runtime's
    # target contract is switched to the full-field semantics it would carry.)
    import dataclasses

    full_accel, mu, _ = _runtime_total_accel(str(sh_run["run_dir"]), "cpu")
    full_accel.runtime.target_contract = dataclasses.replace(
        full_accel.runtime.target_contract, target_mode="full", baseline_kind="none"
    )
    a_net = np.asarray(
        full_accel.runtime.predict_residual_accel_fixed(r), dtype=np.float64
    ).reshape(-1, 3)
    out = full_accel(r)
    np.testing.assert_allclose(out, a_net, rtol=0, atol=1e-12)
    # The old wrapper added mu/r^2 ~ 1.3 m/s^2 here; the total must not carry it.
    assert np.max(np.abs(out)) < 1e-6 * mu / RADIUS ** 2

    # Point-mass base: exactly one monopole, -mu r / r^3.
    full_accel.runtime.target_contract = dataclasses.replace(
        full_accel.runtime.target_contract, baseline_kind="point_mass"
    )
    np.testing.assert_allclose(
        full_accel(r), a_net - full_accel.runtime.mu_si * r / RADIUS ** 3, rtol=1e-12, atol=0
    )
