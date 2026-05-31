# tests/test_propagator_physics.py
# -*- coding: utf-8 -*-
"""
Physics & orchestration tests for ``lunaris.core.propagator``.

These tests use a *minimal fake* point-mass dynamics object rather than the full
Numba ``DynamicsEngine`` (whose RHS is already covered in ``test_dynamics.py``).
The fake exposes exactly the surface the propagator relies on
(``build_rhs()``, ``grav``, ``ephem``), which keeps these tests fast, fully
deterministic, and focused on the propagator's own contracts: two-body
invariants, terminal impact detection, peri/apo event bookkeeping, the fixed-step
6-D guard, time-grid helpers, and stop-file / checkpoint behaviour.

The fake's gravitational parameter matches the propagator's fallback constants
(``grav is None`` -> ``MU_MOON`` / ``R_MOON``) so the analytic expectations and the
propagator's internal reference values agree.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.propagator import _clamp_output_dt, make_time_grid, propagate

MU = float(MU_MOON)
R = float(R_MOON)


class FakePointMassDynamics:
    """Minimal stand-in for DynamicsEngine: pure-Python point-mass RHS.

    ``grav = None`` makes the propagator fall back to (R_MOON, MU_MOON) for its
    reference radius / GM, which is exactly the field this RHS integrates.
    """

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


def _circular_state(alt_m: float):
    r0 = R + alt_m
    v_circ = math.sqrt(MU / r0)
    return np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0]), r0, v_circ


def _period(a: float) -> float:
    return 2.0 * math.pi * math.sqrt(a ** 3 / MU)


def _cfg(**kw) -> PropagatorConfig:
    base = dict(
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        verbose=False,
        compute_2body_baseline=False,
        use_nyquist_max_step=False,
        events=EventConfig(detect_impact=True, impact_alt_km=0.0, enable_peri_apo_events=True),
    )
    base.update(kw)
    return PropagatorConfig(**base)


# =============================================================================
# Two-body invariants
# =============================================================================

def test_circular_orbit_stays_circular_over_one_period():
    y0, r0, _ = _circular_state(100e3)
    T = _period(r0)
    tc = TimeConfig(duration_s=T, output_dt_s=T / 200.0, samples_per_period=200)
    res = propagate(FakePointMassDynamics(), y0, _cfg(), time_cfg=tc)

    radii = np.linalg.norm(res.y[:, :3], axis=1)
    # A circular orbit must keep |r| constant to a few parts in 1e-9 (DOP853, tight tol).
    rel_drift = (radii.max() - radii.min()) / r0
    assert rel_drift < 1e-6
    assert not res.impacted


def test_specific_energy_is_conserved():
    y0, r0, _ = _circular_state(250e3)
    T = _period(r0)
    tc = TimeConfig(duration_s=T, output_dt_s=T / 200.0, samples_per_period=200)
    res = propagate(FakePointMassDynamics(), y0, _cfg(), time_cfg=tc)

    r = res.y[:, :3]
    v = res.y[:, 3:6]
    rn = np.linalg.norm(r, axis=1)
    eps = 0.5 * np.sum(v ** 2, axis=1) - MU / rn
    rel_drift = abs((eps.max() - eps.min()) / eps.mean())
    assert rel_drift < 1e-8


def test_angular_momentum_norm_is_conserved():
    y0, r0, _ = _circular_state(180e3)
    T = _period(r0)
    tc = TimeConfig(duration_s=T, output_dt_s=T / 200.0, samples_per_period=200)
    res = propagate(FakePointMassDynamics(), y0, _cfg(), time_cfg=tc)

    h = np.cross(res.y[:, :3], res.y[:, 3:6])
    hn = np.linalg.norm(h, axis=1)
    rel_drift = (hn.max() - hn.min()) / hn.mean()
    assert rel_drift < 1e-8


# =============================================================================
# Events
# =============================================================================

def test_impact_event_triggers_on_descending_trajectory():
    # Start at apoapsis above the surface with periapsis below it -> guaranteed impact.
    ra = R + 200e3
    rp = R - 50e3
    a = 0.5 * (ra + rp)
    v_apo = math.sqrt(MU * (2.0 / ra - 1.0 / a))
    y0 = np.array([ra, 0.0, 0.0, 0.0, v_apo, 0.0])

    T = _period(a)
    tc = TimeConfig(duration_s=T, output_dt_s=T / 400.0, samples_per_period=400)
    res = propagate(FakePointMassDynamics(), y0, _cfg(), time_cfg=tc)

    assert res.impacted is True
    assert res.stop_reason == "impact"
    assert res.t_impact_s is not None and 0.0 < res.t_impact_s < T
    # The reported impact state sits at (approximately) the impact altitude (0 km).
    impact_alt_km = (np.linalg.norm(res.y_impact[:3]) - R) / 1000.0
    assert impact_alt_km == pytest.approx(0.0, abs=1.0)


def test_peri_apo_events_are_collected_without_crashing():
    # Eccentric orbit fully above the surface -> no impact, but peri/apo crossings.
    ra = R + 3000e3
    rp = R + 300e3
    a = 0.5 * (ra + rp)
    v_apo = math.sqrt(MU * (2.0 / ra - 1.0 / a))
    y0 = np.array([ra, 0.0, 0.0, 0.0, v_apo, 0.0])

    T = _period(a)
    tc = TimeConfig(duration_s=1.5 * T, output_dt_s=T / 300.0, samples_per_period=300)
    res = propagate(FakePointMassDynamics(), y0, _cfg(), time_cfg=tc)

    assert not res.impacted
    # t_events mirrors SciPy: [impact, peri, apo]; each entry is an array.
    assert isinstance(res.t_events, list) and len(res.t_events) >= 3
    for arr in res.t_events:
        assert isinstance(np.asarray(arr), np.ndarray)
    # Over 1.5 periods at least one periapsis or apoapsis crossing should be found.
    total_peri_apo = sum(int(np.asarray(res.t_events[i]).size) for i in (1, 2))
    assert total_peri_apo >= 1


# =============================================================================
# Solver-selection / input-validation contracts
# =============================================================================

def test_fixed_step_symplectic_rejects_augmented_state():
    y0_aug = np.array([R + 100e3, 0.0, 0.0, 0.0, math.sqrt(MU / (R + 100e3)), 0.0, 1000.0])
    tc = TimeConfig(duration_s=100.0, output_dt_s=10.0, samples_per_period=10)
    with pytest.raises(ValueError, match="6D|6-D|six"):
        propagate(FakePointMassDynamics(), y0_aug, _cfg(method="YOSHIDA4"), time_cfg=tc)


def test_propagate_requires_time_cfg():
    y0, _, _ = _circular_state(100e3)
    with pytest.raises(ValueError, match="time_cfg is required"):
        propagate(FakePointMassDynamics(), y0, _cfg(), time_cfg=None)


def test_unknown_method_falls_back_to_dop853_and_runs():
    y0, r0, _ = _circular_state(120e3)
    T = _period(r0)
    tc = TimeConfig(duration_s=T / 4.0, output_dt_s=T / 200.0, samples_per_period=200)
    # An unrecognised non-symplectic method must degrade gracefully, not crash.
    res = propagate(FakePointMassDynamics(), y0, _cfg(method="NOT_A_METHOD"), time_cfg=tc)
    assert res.t.size >= 2
    assert np.all(np.isfinite(res.y))


# =============================================================================
# Time-grid helpers
# =============================================================================

def test_make_time_grid_regular_and_degenerate():
    grid = make_time_grid(0.0, 10.0, 2.0)
    np.testing.assert_allclose(grid, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert grid[-1] == 10.0
    # Degenerate inputs collapse to the 2-point span rather than raising.
    np.testing.assert_allclose(make_time_grid(0.0, 10.0, 0.0), [0.0, 10.0])
    np.testing.assert_allclose(make_time_grid(10.0, 0.0, 1.0), [10.0, 0.0])


def test_clamp_output_dt_rejects_nonpositive_and_enforces_cap():
    with pytest.raises(ValueError, match="positive"):
        _clamp_output_dt(0.0, 100.0, -1.0, cap=100, verbose=False)
    # Requesting more points than the cap allows -> dt is increased.
    dt = _clamp_output_dt(0.0, 100.0, 1.0, cap=10, verbose=False)
    n_points = int(math.ceil(100.0 / dt)) + 1
    assert dt > 1.0
    assert n_points <= 10


# =============================================================================
# Stop-file & checkpoint behaviour (temp dirs)
# =============================================================================

def test_stop_file_halts_fixed_step_integration(tmp_path):
    stop_file = tmp_path / "STOP"
    stop_file.write_text("stop", encoding="utf-8")  # exists from the start

    y0, r0, _ = _circular_state(100e3)
    T = _period(r0)
    tc = TimeConfig(duration_s=T, output_dt_s=T / 100.0, samples_per_period=100)
    cfg = _cfg(method="VV", stop_file=str(stop_file), stop_event_in_scipy=False)

    res = propagate(FakePointMassDynamics(), y0, cfg, time_cfg=tc)
    assert res.stopped_early is True
    assert res.stop_reason == "stop file"


def test_checkpoint_npz_is_written(tmp_path):
    ckpt = tmp_path / "ckpt.npz"
    y0, r0, _ = _circular_state(150e3)
    T = _period(r0)
    tc = TimeConfig(duration_s=T / 4.0, output_dt_s=T / 100.0, samples_per_period=100)
    cfg = _cfg(checkpoint_path=str(ckpt))

    res = propagate(FakePointMassDynamics(), y0, cfg, time_cfg=tc)
    assert ckpt.exists()
    with np.load(ckpt) as data:
        assert "t" in data and "y_row" in data
        assert data["t"].shape[0] == res.t.shape[0]
