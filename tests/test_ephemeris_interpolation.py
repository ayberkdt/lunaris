"""Ephemeris interpolation correctness and CPU/GPU scheme parity.

Covers the runtime samplers that feed the integrator with Sun/Earth positions and
the Moon-fixed attitude quaternion:

* ``interp_vec3_safe``  - constant / linear / Catmull-Rom by table size,
* ``interp_quat_safe``  - SLERP with unit-norm and shortest-path guarantees,
* ``EphemerisManager``  - inertial<->fixed frame round-trip.

It also pins the GPU device kernel's interpolation *formula*. The CUDA
``_interp3_cuda`` (batch_propagator) was aligned to use the same constant/linear/
Catmull-Rom scheme as the CPU path; this module reproduces that exact device
logic as a numba.njit mirror and asserts it matches ``interp_vec3_safe`` to
machine precision. Identical f64 arithmetic on both backends means agreement here
implies agreement on a real GPU (end-to-end CPU/GPU parity is exercised
separately by tests/test_real_asset_cpu_gpu_validation.py).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numba import njit

from lunaris.common.math_utils import quat_conj, quat_rotate_vec
from lunaris.physics.ephemeris import (
    EphemerisManager,
    EphemerisTables,
    get_ephem_state,
    interp_quat_safe,
    interp_vec3_safe,
)


# -----------------------------------------------------------------------------
# GPU device-kernel mirror (must stay character-identical to _interp3_cuda)
# -----------------------------------------------------------------------------
@njit(cache=True)
def _interp3_cuda_mirror(t, dt_s, tab, n_tab, result):
    if n_tab <= 1 or dt_s <= 0.0:
        result[0] = tab[0, 0]; result[1] = tab[0, 1]; result[2] = tab[0, 2]
        return
    tmax = dt_s * (n_tab - 1)
    if t <= 0.0:
        result[0] = tab[0, 0]; result[1] = tab[0, 1]; result[2] = tab[0, 2]
        return
    if t >= tmax:
        j = n_tab - 1
        result[0] = tab[j, 0]; result[1] = tab[j, 1]; result[2] = tab[j, 2]
        return
    u = t / dt_s
    i = int(u)
    if i < 0:
        i = 0
    elif i > n_tab - 2:
        i = n_tab - 2
    f = u - float(i)
    if f < 0.0:
        f = 0.0
    elif f > 1.0:
        f = 1.0
    if n_tab < 4:
        result[0] = tab[i, 0] * (1.0 - f) + tab[i + 1, 0] * f
        result[1] = tab[i, 1] * (1.0 - f) + tab[i + 1, 1] * f
        result[2] = tab[i, 2] * (1.0 - f) + tab[i + 1, 2] * f
        return
    i0 = i - 1 if i > 0 else 0
    i3 = i + 2 if i < n_tab - 2 else n_tab - 1
    f2 = f * f
    f3 = f2 * f
    w0 = -f + 2.0 * f2 - f3
    w1 = 2.0 - 5.0 * f2 + 3.0 * f3
    w2 = f + 4.0 * f2 - 3.0 * f3
    w3 = -f2 + f3
    result[0] = 0.5 * (tab[i0, 0] * w0 + tab[i, 0] * w1 + tab[i + 1, 0] * w2 + tab[i3, 0] * w3)
    result[1] = 0.5 * (tab[i0, 1] * w0 + tab[i, 1] * w1 + tab[i + 1, 1] * w2 + tab[i3, 1] * w3)
    result[2] = 0.5 * (tab[i0, 2] * w0 + tab[i, 2] * w1 + tab[i + 1, 2] * w2 + tab[i3, 2] * w3)


def _smooth_vec_table(n: int, dt: float) -> np.ndarray:
    t = np.arange(n) * dt
    return np.ascontiguousarray(
        np.stack(
            [3.84e8 * np.cos(2e-6 * t), 3.84e8 * np.sin(2e-6 * t), 1.0e7 * np.sin(1e-6 * t)],
            axis=1,
        )
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 64])
def test_gpu_vec3_kernel_matches_cpu_sampler(n: int) -> None:
    """The GPU device-kernel scheme reproduces the CPU sampler to machine
    precision across table sizes and the whole time span (incl. clamped edges)."""
    dt = 60.0
    tab = _smooth_vec_table(n, dt)
    scale = float(np.max(np.abs(tab)))
    times = np.concatenate([[-5.0, 0.0], np.linspace(0.0, dt * (n - 1), 53), [dt * (n - 1) + 5.0]])
    out = np.zeros(3)
    worst = 0.0
    for t in times:
        ref = np.array(interp_vec3_safe(float(t), dt, tab))
        _interp3_cuda_mirror(float(t), dt, tab, n, out)
        worst = max(worst, float(np.max(np.abs(out - ref))))
    # ~1 ULP at position scale; the previous linear-only kernel diverged ~1e-9 rel.
    assert worst <= 1e-12 * scale


def test_vec3_is_exact_at_table_nodes() -> None:
    dt = 60.0
    tab = _smooth_vec_table(10, dt)
    for k in range(10):
        v = np.array(interp_vec3_safe(k * dt, dt, tab))
        np.testing.assert_allclose(v, tab[k], rtol=0.0, atol=1e-6)


def test_linear_branch_is_exact_on_linear_data() -> None:
    """Small tables (2<=n<4) use linear interpolation, exact on affine data
    everywhere in the span."""
    dt = 30.0
    n = 3
    t = np.arange(n) * dt
    slope = np.array([12.0, -3.0, 7.5])
    base = np.array([1.0e6, -2.0e6, 5.0e5])
    tab = np.ascontiguousarray(base[None, :] + np.outer(t, slope))
    for tq in np.linspace(0.0, dt * (n - 1), 40):
        v = np.array(interp_vec3_safe(float(tq), dt, tab))
        np.testing.assert_allclose(v, base + slope * tq, rtol=1e-12, atol=1e-6)


def test_catmull_rom_is_exact_on_linear_data_in_interior() -> None:
    """Catmull-Rom (n>=4) reproduces affine data exactly in interior segments;
    boundary segments are not linear-exact because the endpoint control point is
    clamped (a documented, acceptable edge property for smooth ephemerides)."""
    dt = 30.0
    n = 12
    t = np.arange(n) * dt
    slope = np.array([12.0, -3.0, 7.5])
    base = np.array([1.0e6, -2.0e6, 5.0e5])
    tab = np.ascontiguousarray(base[None, :] + np.outer(t, slope))
    # Sample strictly inside [dt, dt*(n-2)] (skip first and last segments).
    for tq in np.linspace(dt, dt * (n - 2), 40):
        v = np.array(interp_vec3_safe(float(tq), dt, tab))
        np.testing.assert_allclose(v, base + slope * tq, rtol=1e-10, atol=1e-4)


def test_vec3_constant_row_degeneracy() -> None:
    """A single constant row (third-body disabled) returns that row for any t."""
    tab = np.array([[1.0, -2.0, 3.0]], dtype=np.float64)
    for t in (0.0, 123.0, -50.0):
        assert tuple(interp_vec3_safe(t, 60.0, tab)) == (1.0, -2.0, 3.0)


def test_vec3_clamps_outside_span() -> None:
    dt = 60.0
    tab = _smooth_vec_table(6, dt)
    np.testing.assert_allclose(interp_vec3_safe(-100.0, dt, tab), tab[0], atol=1e-6)
    np.testing.assert_allclose(interp_vec3_safe(1e9, dt, tab), tab[-1], atol=1e-6)


# -----------------------------------------------------------------------------
# Quaternion interpolation
# -----------------------------------------------------------------------------
def _axis_angle_quat(axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.array([math.cos(ang / 2), *(math.sin(ang / 2) * axis)], dtype=np.float64)


def _quat_table(n: int, dt: float) -> np.ndarray:
    axis = np.array([0.2, -0.5, 0.84])
    return np.ascontiguousarray(
        np.stack([_axis_angle_quat(axis, 1e-4 * (k * dt)) for k in range(n)])
    )


def test_quat_interpolation_stays_unit_norm() -> None:
    dt = 60.0
    tab = _quat_table(20, dt)
    for tq in np.linspace(0.0, dt * 19, 60):
        q = np.array(interp_quat_safe(float(tq), dt, tab))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_quat_is_exact_at_nodes() -> None:
    dt = 60.0
    tab = _quat_table(15, dt)
    for k in range(15):
        q = np.array(interp_quat_safe(k * dt, dt, tab))
        # q and -q are the same rotation; compare up to sign.
        assert min(np.linalg.norm(q - tab[k]), np.linalg.norm(q + tab[k])) < 1e-12


def test_quat_slerp_takes_shortest_path_for_antipodal_endpoints() -> None:
    """SLERP must flip to the nearer hemisphere; the midpoint of q and -q' must
    stay a unit quaternion close to q (not collapse through zero)."""
    dt = 1.0
    q = _axis_angle_quat(np.array([0.0, 0.0, 1.0]), 0.2)
    tab = np.ascontiguousarray(np.stack([q, -q]))  # same rotation, opposite sign
    mid = np.array(interp_quat_safe(0.5, dt, tab))
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-12
    assert min(np.linalg.norm(mid - q), np.linalg.norm(mid + q)) < 1e-9


def test_quat_constant_row_degeneracy() -> None:
    q = _axis_angle_quat(np.array([1.0, 1.0, 0.0]), 0.5)
    tab = np.ascontiguousarray(q[None, :])
    out = np.array(interp_quat_safe(999.0, 60.0, tab))
    assert min(np.linalg.norm(out - q), np.linalg.norm(out + q)) < 1e-12


def test_get_ephem_state_matches_component_samplers() -> None:
    dt = 60.0
    n = 16
    sun = _smooth_vec_table(n, dt)
    earth = _smooth_vec_table(n, dt) * 0.001
    q = _quat_table(n, dt)
    for tq in (0.0, 137.0, dt * (n - 1) * 0.5, dt * (n - 1)):
        sx, sy, sz, ex, ey, ez, qw, qx, qy, qz = get_ephem_state(
            float(tq), dt, sun, np.ascontiguousarray(earth), q
        )
        np.testing.assert_allclose((sx, sy, sz), interp_vec3_safe(float(tq), dt, sun), atol=1e-6)
        np.testing.assert_allclose((ex, ey, ez), interp_vec3_safe(float(tq), dt, np.ascontiguousarray(earth)), atol=1e-9)
        np.testing.assert_allclose((qw, qx, qy, qz), interp_quat_safe(float(tq), dt, q), atol=1e-12)


# -----------------------------------------------------------------------------
# Frame round-trip through the manager
# -----------------------------------------------------------------------------
def _build_manager(n: int = 12, dt: float = 60.0) -> EphemerisManager:
    t_tab = np.arange(n, dtype=np.float64) * dt
    tables = EphemerisTables(
        dt_s=dt,
        t_tab_s=t_tab,
        et0=0.0,
        q_i2f_tab=_quat_table(n, dt),
        r_earth_tab_m=_smooth_vec_table(n, dt) * 0.001,
        r_sun_tab_m=_smooth_vec_table(n, dt),
        mu_earth_m3s2=3.986004418e14,
        mu_sun_m3s2=1.32712440018e20,
    )
    return EphemerisManager(tables)


def test_inertial_to_fixed_round_trip_is_identity() -> None:
    mgr = _build_manager()
    rng = np.random.default_rng(7)
    for tq in (0.0, 95.0, 360.0, 659.0):
        for _ in range(5):
            v = rng.normal(scale=1e6, size=3)
            v_fixed = mgr.transform_inertial_to_fixed(float(tq), v)
            v_back = mgr.transform_fixed_to_inertial(float(tq), v_fixed)
            np.testing.assert_allclose(v_back, v, rtol=1e-12, atol=1e-6)


def test_manager_rotation_matches_raw_quaternion() -> None:
    """transform_inertial_to_fixed must apply exactly the interpolated q_i2f."""
    mgr = _build_manager()
    tq = 211.0
    v = np.array([1.2e6, -3.4e5, 7.8e5])
    q = mgr.get_inertial_to_fixed_rotation(tq)
    expected = np.array(quat_rotate_vec(q[0], q[1], q[2], q[3], v[0], v[1], v[2]))
    np.testing.assert_allclose(mgr.transform_inertial_to_fixed(tq, v), expected, rtol=1e-12, atol=1e-9)
    # And the inverse uses the conjugate, not the same quaternion.
    cw, cx, cy, cz = quat_conj(q[0], q[1], q[2], q[3])
    expected_inv = np.array(quat_rotate_vec(cw, cx, cy, cz, v[0], v[1], v[2]))
    np.testing.assert_allclose(mgr.transform_fixed_to_inertial(tq, v), expected_inv, rtol=1e-12, atol=1e-9)
