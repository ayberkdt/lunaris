"""Yoshida stage overhang vs. the ephemeris table edges.

Yoshida compositions of velocity-Verlet evaluate the force at the partial sums
of their weights, which leave ``[t, t+h]`` by 0.351 h (Y4), 0.649 h (Y6) and
0.916 h (Y8). The first step of every run therefore samples the ephemeris before
``t0 = 0`` (the table start), where the Hermite position and SLERP attitude
samplers held the first row. Measured on the real DE/PCK tables (100 km, 60 deg
orbit, output_dt 60 s, same epoch and state, table origin moved 600 s earlier as
the reference): 0.49 m (Y4) and 1.15 m (Y8) final-position error after one day,
versus 1.1e-7 m for RK4, whose stages stay inside the step.

The samplers now continue the edge segment for one table interval, and the
coverage check accounts for the overhang instead of failing every t0 = 0 run.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.core.propagation.integrators.fixed_step import fixed_step_stage_overhang
from lunaris.core.propagation.integrators.symplectic import (
    _pefrl_step,
    _vv_step,
    _y4_step,
    _y6_step,
    _y8_step,
    stage_time_overhang,
)
from lunaris.core.propagation.propagator import _require_ephemeris_coverage
from lunaris.physics.ephemeris import interp_quat_safe, interp_vec3_hermite

_STEPPERS = {"VV": _vv_step, "PEFRL": _pefrl_step, "Y4": _y4_step, "Y6": _y6_step, "Y8": _y8_step}


@pytest.mark.parametrize("method", sorted(_STEPPERS))
def test_overhang_matches_instrumented_stage_times(method: str) -> None:
    times: list[float] = []

    def accel(t: float, y6: np.ndarray) -> np.ndarray:
        times.append(float(t))
        return -y6[:3]

    _STEPPERS[method](accel, 0.0, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 1.0)
    measured = max(0.0, -min(times), max(times) - 1.0)
    assert stage_time_overhang(method) == pytest.approx(measured, abs=1e-12)


def test_overhang_values_and_aliases() -> None:
    assert fixed_step_stage_overhang("YOSHIDA4") == pytest.approx(0.3512071919596576)
    assert fixed_step_stage_overhang("y8") == pytest.approx(0.9155, abs=1e-4)
    for m in ("RK4", "RK8", "VV", "PEFRL", "RKN4", "DOP853"):
        assert fixed_step_stage_overhang(m) == 0.0


# ---------------------------------------------------------------- samplers

_DT = 60.0
_W = 2.0 * math.pi / 3600.0  # 1-hour circular motion / rotation


_WV = 2.0 * math.pi / (27.32 * 86400.0)  # Earth about the Moon (sidereal month)


def _circle(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = 3.844e8
    p = np.stack([r * np.cos(_WV * t), r * np.sin(_WV * t), 0.0 * t], axis=-1)
    v = np.stack([-r * _WV * np.sin(_WV * t), r * _WV * np.cos(_WV * t), 0.0 * t], axis=-1)
    return p, v


def _quat_z(t: float) -> np.ndarray:
    return np.array([math.cos(0.5 * _W * t), 0.0, 0.0, math.sin(0.5 * _W * t)])


def test_hermite_continues_edge_segment_one_interval() -> None:
    t_tab = np.arange(11) * _DT
    p, v = _circle(t_tab)
    hi = t_tab[-1]
    for t in (-0.351 * _DT, -0.916 * _DT, hi + 0.351 * _DT, hi + 0.99 * _DT):
        got = np.array(interp_vec3_hermite(t, _DT, p, v, True))
        want = _circle(np.array([t]))[0][0]
        # Clamping is off by |v| * |t - edge| (~1 km/s * 21-59 s = 21-59 km); the
        # cubic continuation is within its O((w dt)^4 r) remainder, ~1e-7 m here.
        assert np.linalg.norm(got - want) < 1.0e-3
        edge = p[0] if t < 0 else p[-1]
        assert np.linalg.norm(edge - want) > 1.0e4
        # Default (GPU-parity contract) still holds the end row.
        assert np.array_equal(np.array(interp_vec3_hermite(t, _DT, p, v)), edge)
    # Beyond one interval: clamped to the end row as before.
    assert np.array_equal(np.array(interp_vec3_hermite(-2.0 * _DT, _DT, p, v, True)), p[0])
    assert np.array_equal(np.array(interp_vec3_hermite(hi + 2.0 * _DT, _DT, p, v, True)), p[-1])


def test_hermite_inside_table_is_unchanged() -> None:
    t_tab = np.arange(11) * _DT
    p, v = _circle(t_tab)
    for t in (0.0, 17.3, 299.99, t_tab[-1]):
        u = t / _DT
        i = min(int(u), 9)
        f = u - i
        h00, h10 = 2 * f**3 - 3 * f**2 + 1, f**3 - 2 * f**2 + f
        h01, h11 = -2 * f**3 + 3 * f**2, f**3 - f**2
        ref = h00 * p[i] + h10 * _DT * v[i] + h01 * p[i + 1] + h11 * _DT * v[i + 1]
        np.testing.assert_allclose(interp_vec3_hermite(t, _DT, p, v, True), ref, rtol=0, atol=1e-6)
        assert interp_vec3_hermite(t, _DT, p, v, True) == interp_vec3_hermite(t, _DT, p, v)


def test_attitude_continues_edge_arc_one_interval() -> None:
    q_tab = np.array([_quat_z(t) for t in np.arange(11) * _DT])
    hi = 10 * _DT
    for t in (-0.351 * _DT, -0.9 * _DT, hi + 0.5 * _DT):
        got = np.array(interp_quat_safe(t, _DT, q_tab, True))
        assert np.allclose(got, _quat_z(t), atol=1e-12)
        edge = q_tab[0] if t < 0 else q_tab[-1]
        assert np.allclose(interp_quat_safe(t, _DT, q_tab), edge, atol=0)
    # Inside: SLERP unchanged; beyond one interval: clamped.
    assert np.allclose(interp_quat_safe(90.0, _DT, q_tab, True), _quat_z(90.0), atol=1e-12)
    assert np.allclose(interp_quat_safe(-3.0 * _DT, _DT, q_tab, True), q_tab[0], atol=0)


def test_rhs_sampler_extrapolates_at_the_yoshida_first_stage() -> None:
    from lunaris.physics.ephemeris import get_ephem_state_with_velocity

    t_tab = np.arange(11) * _DT
    p, v = _circle(t_tab)
    q_tab = np.array([_quat_z(t) for t in t_tab])
    t = -0.351 * _DT
    out = get_ephem_state_with_velocity(t, _DT, p, p, v, v, q_tab, True)
    assert np.linalg.norm(np.array(out[0:3]) - _circle(np.array([t]))[0][0]) < 1.0e-3
    assert np.allclose(np.array(out[6:10]), _quat_z(t), atol=1e-12)


# ---------------------------------------------------------------- coverage


def _dyn(n_rows: int) -> SimpleNamespace:
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n_rows, 1))
    tabs = {"dt_s": _DT, "r_sun_tab_m": np.ones((n_rows, 3)), "r_earth_tab_m": np.ones((n_rows, 3)),
            "q_i2f_tab": q}
    return SimpleNamespace(ephem=SimpleNamespace(get_data_provider=lambda: tabs))


def test_coverage_accepts_t0_zero_overhang_within_one_interval() -> None:
    dyn = _dyn(101)  # table [0, 6000] s
    _require_ephemeris_coverage(dyn, 0.0, 6000.0, stage_overhang_s=0.916 * _DT)


def test_coverage_rejects_overhang_beyond_extrapolation_band() -> None:
    dyn = _dyn(101)
    with pytest.raises(ValueError, match="extrapolation band"):
        _require_ephemeris_coverage(dyn, 0.0, 6000.0, stage_overhang_s=0.916 * 120.0)
    with pytest.raises(ValueError, match="outside the"):
        _require_ephemeris_coverage(dyn, -1.0, 6000.0)
