# tests/test_math_utils.py
# -*- coding: utf-8 -*-
"""
Unit tests for common.math_utils
================================

Goal: high-signal, fast, and deterministic tests for the project's low-level math
primitives (scalar utilities, quaternions, interpolation, orbital elements, and
grid samplers).

Run:
    pytest -q
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_allclose

# -----------------------------------------------------------------------------
# Import helper
# -----------------------------------------------------------------------------
# These tests are intended to be runnable from the repo root without installing
# the package. We add the repo root to sys.path as a fallback.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from common import math_utils  # project layout: <root>/common/math_utils.py
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import 'common.math_utils'. "
        "Run pytest from the repository root (the folder that contains 'common/')."
    ) from e


# -----------------------------------------------------------------------------
# Constants (common.constants)
# -----------------------------------------------------------------------------
# math_utils intentionally does NOT re-export numeric EPS values; they live in
# common.constants to keep a single source of truth.

try:
    from common import constants as C
except Exception:  # pragma: no cover
    C = None


def test_constants_eps_values_exist_and_match_expected():
    if C is None:
        pytest.skip("common.constants not importable in this environment")

    # Existence
    for name in ("EPS_1E12", "EPS_1E15", "EPS_1E18"):
        assert hasattr(C, name), f"Missing constant: {name}"

    # Values (kept explicit so refactors don't silently change semantics)
    assert_allclose(C.EPS_1E12, 1e-12, rtol=0.0, atol=0.0)
    assert_allclose(C.EPS_1E15, 1e-15, rtol=0.0, atol=0.0)
    assert_allclose(C.EPS_1E18, 1e-18, rtol=0.0, atol=0.0)


# -----------------------------------------------------------------------------
# Small helpers for robust comparisons
# -----------------------------------------------------------------------------
_TWOPI = 2.0 * math.pi


def _angdiff(a: float, b: float) -> float:
    """Smallest signed angular difference a-b in [-pi, pi]."""
    return (a - b + math.pi) % _TWOPI - math.pi


def assert_angle_close(a: float, b: float, *, atol: float = 1e-10) -> None:
    assert abs(_angdiff(a, b)) <= atol


def axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return scalar-first quaternion [w,x,y,z] for rotation about axis."""
    axis = np.asarray(axis, dtype=np.float64).ravel()
    if axis.size != 3:
        raise ValueError("axis must be 3D")
    n = np.linalg.norm(axis)
    if n == 0:
        raise ValueError("axis must be nonzero")
    a = axis / n
    s = math.sin(angle_rad / 2.0)
    return np.array([math.cos(angle_rad / 2.0), a[0] * s, a[1] * s, a[2] * s], dtype=np.float64)


def coe_to_rv(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    nu: float,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classical elements -> inertial Cartesian (r,v). Elliptic (e<1) assumed."""
    p = a * (1.0 - e * e)
    r_pf = p / (1.0 + e * math.cos(nu)) * np.array([math.cos(nu), math.sin(nu), 0.0], dtype=np.float64)
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0], dtype=np.float64)

    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)

    # Perifocal -> inertial: R3(raan) * R1(inc) * R3(argp)
    R = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ],
        dtype=np.float64,
    )

    r = R @ r_pf
    v = R @ v_pf
    return r, v


# =============================================================================
# 1) Scalar & geometry utilities
# =============================================================================


@pytest.mark.parametrize("x,y,z", [(3.0, 4.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 2.0, -2.0)])
def test_norm3_matches_numpy(x, y, z):
    got = math_utils.norm3(x, y, z)
    expected = float(np.linalg.norm([x, y, z]))
    assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "val, lo, hi, expected",
    [
        (0.5, 0.0, 1.0, 0.5),     # inside
        (-5.0, 0.0, 1.0, 0.0),    # underflow
        (5.0, 0.0, 1.0, 1.0),     # overflow
        (10.0, 10.0, 10.0, 10.0), # degenerate interval
    ],
)
def test_clamp(val, lo, hi, expected):
    assert math_utils.clamp(val, lo, hi) == expected



@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, 0.0),
        (_TWOPI, 0.0),
        (-_TWOPI, 0.0),
        (-0.1, _TWOPI - 0.1),
        (10.0, 10.0 % _TWOPI),
    ],
)
def test_wrap_angle_2pi(angle, expected):
    out = math_utils.wrap_angle_2pi(angle)
    assert 0.0 <= out < _TWOPI
    assert_allclose(out, expected, atol=0.0, rtol=0.0)


def test_wrap_lon_deg_degenerate_interval():
    # If span is degenerate, function returns input unchanged (per implementation).
    assert_allclose(math_utils.wrap_lon_deg(123.0, west=10.0, east=10.0), 123.0)


@pytest.mark.parametrize(
    "lon, west, east, expected",
    [
        (370.0, 0.0, 360.0, 10.0),
        (-10.0, 0.0, 360.0, 350.0),
        (-10.0, -180.0, 180.0, -10.0),
        (190.0, -180.0, 180.0, -170.0),
    ],
)
def test_wrap_lon_deg(lon, west, east, expected):
    assert_allclose(math_utils.wrap_lon_deg(lon, west=west, east=east), expected)


def test_latlon_from_xyz_m_basic_and_origin():
    R = 1737.4e3

    # X-axis
    lat, lon, r = math_utils.latlon_from_xyz_m(R, 0.0, 0.0)
    assert_allclose([lat, lon, r], [0.0, 0.0, R], atol=1e-9)

    # Z-axis
    lat, lon, r = math_utils.latlon_from_xyz_m(0.0, 0.0, R)
    assert_allclose([lat, lon, r], [90.0, 0.0, R], atol=1e-9)

    # Origin protection
    lat, lon, r = math_utils.latlon_from_xyz_m(0.0, 0.0, 0.0)
    assert_allclose([lat, lon, r], [0.0, 0.0, 0.0], atol=0.0, rtol=0.0)


def test_latlon_from_xyz_m_lon_wrapping_toggle():
    R = 1737.4e3
    # Point on -Y axis => lon = -90 deg (or 270 deg if wrapped)
    lat, lon_wrapped, _ = math_utils.latlon_from_xyz_m(0.0, -R, 0.0, lon_0_360=True)
    lat, lon_signed, _ = math_utils.latlon_from_xyz_m(0.0, -R, 0.0, lon_0_360=False)
    assert_allclose(lat, 0.0, atol=1e-12)
    assert_allclose(lon_wrapped, 270.0, atol=1e-12)
    assert_allclose(lon_signed, -90.0, atol=1e-12)


# =============================================================================
# 2) Quaternions
# =============================================================================



def test_quat_conj_basic():
    q = axis_angle_quat(np.array([0.0, 1.0, 0.0]), 0.7)
    qc = math_utils.quat_conj(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    # Conjugate should flip the vector part (scalar-first convention).
    assert_allclose(qc, [q[0], -q[1], -q[2], -q[3]], atol=0.0, rtol=0.0)




def test_quat_rotate_vec_matches_numpy_wrapper():
    angle = math.pi / 2.0
    q = axis_angle_quat(np.array([0.0, 0.0, 1.0]), angle)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    out_np = math_utils.quat_rotate_np(q, v)
    x, y, z = math_utils.quat_rotate_vec(float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                                         float(v[0]), float(v[1]), float(v[2]))
    out_kernel = np.array([x, y, z], dtype=np.float64)
    assert_allclose(out_kernel, out_np, atol=1e-14)
    assert_allclose(out_np, [0.0, 1.0, 0.0], atol=1e-14)


@pytest.mark.parametrize(
    "bad_q, bad_v",
    [
        (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0])),
    ],
)
def test_quat_rotate_np_input_validation(bad_q, bad_v):
    with pytest.raises(ValueError):
        math_utils.quat_rotate_np(bad_q, bad_v)


def test_quat_slerp_endpoints_and_shortest_path():
    qA = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    angle = math.pi / 2.0
    qB = axis_angle_quat(np.array([0.0, 0.0, 1.0]), angle)

    assert_allclose(math_utils.quat_slerp_np(qA, qB, 0.0), qA, atol=0.0, rtol=0.0)
    assert_allclose(math_utils.quat_slerp_np(qA, qB, 1.0), qB, atol=0.0, rtol=0.0)

    # Shortest path: slerp between q and -q should return q (because -q represents same rotation).
    qC = -qB
    mid = math_utils.quat_slerp_np(qB, qC, 0.5)
    assert_allclose(mid, qB, atol=1e-14)
    assert_allclose(np.linalg.norm(mid), 1.0, atol=1e-14)


def test_interp_quat_slerp_table_behavior():
    # Two quaternions: identity -> 90deg about Z. dt=1s
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = axis_angle_quat(np.array([0.0, 0.0, 1.0]), math.pi / 2.0)
    q_tab = np.vstack([q0, q1])

    # t<=0 clamps to first
    out0 = math_utils.interp_quat_slerp(-1.0, 1.0, q_tab)
    assert_allclose(out0, tuple(q0), atol=1e-14)

    # t>=tmax clamps to last
    out1 = math_utils.interp_quat_slerp(2.0, 1.0, q_tab)
    assert_allclose(out1, tuple(q1), atol=1e-14)

    # midpoint => 45deg about Z
    mid = math_utils.interp_quat_slerp(0.5, 1.0, q_tab)
    expected = axis_angle_quat(np.array([0.0, 0.0, 1.0]), math.pi / 4.0)
    assert_allclose(mid, tuple(expected), atol=1e-14)

    # dt<=0 => safe fallback to first row
    out_dt0 = math_utils.interp_quat_slerp(0.5, 0.0, q_tab)
    assert_allclose(out_dt0, tuple(q0), atol=1e-14)

    # empty table => identity
    out_empty = math_utils.interp_quat_slerp(0.1, 1.0, np.zeros((0, 4)))
    assert_allclose(out_empty, (1.0, 0.0, 0.0, 0.0), atol=0.0, rtol=0.0)


# =============================================================================
# 3) Interpolation (vectors)
# =============================================================================

def test_interp_vec3_catmull_degenerate_tables_and_clamps():
    dt = 1.0

    # Empty
    out = math_utils.interp_vec3_catmull(0.2, dt, np.zeros((0, 3)))
    assert_allclose(out, (0.0, 0.0, 0.0), atol=0.0, rtol=0.0)

    # Single row
    tab = np.array([[3.0, -1.0, 2.0]])
    out = math_utils.interp_vec3_catmull(100.0, dt, tab)
    assert_allclose(out, (3.0, -1.0, 2.0), atol=0.0, rtol=0.0)

    # Clamp outside range
    tab = np.array([[0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0]])
    out_lo = math_utils.interp_vec3_catmull(-5.0, dt, tab)
    out_hi = math_utils.interp_vec3_catmull(99.0, dt, tab)
    assert_allclose(out_lo, (0.0, 0.0, 0.0), atol=0.0, rtol=0.0)
    assert_allclose(out_hi, (30.0, 0.0, 0.0), atol=0.0, rtol=0.0)

    # dt <= 0 => first row
    out_dt0 = math_utils.interp_vec3_catmull(0.5, 0.0, tab)
    assert_allclose(out_dt0, (0.0, 0.0, 0.0), atol=0.0, rtol=0.0)


def test_interp_vec3_catmull_linear_midpoint_property():
    dt = 1.0
    tab = np.array([[0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0]])
    out = math_utils.interp_vec3_catmull(1.5, dt, tab)
    assert_allclose(out, (15.0, 0.0, 0.0), atol=1e-12)


# =============================================================================
# 4) Step size helper (Nyquist)
# =============================================================================

@pytest.mark.parametrize("degree", [0, 1])
def test_nyquist_degree_below_2_returns_inf(degree):
    out = math_utils.nyquist_max_step_s(1737.4e3, 4.9e12, degree, 50.0)
    assert math.isinf(out)


def test_nyquist_input_validation():
    with pytest.raises(ValueError):
        math_utils.nyquist_max_step_s(-1.0, 4.9e12, 10, 50.0)
    with pytest.raises(ValueError):
        math_utils.nyquist_max_step_s(1737.4e3, -1.0, 10, 50.0)
    with pytest.raises(ValueError):
        math_utils.nyquist_max_step_s(1737.4e3, 4.9e12, 10, 50.0, safety_div=0.0)
    with pytest.raises(ValueError):
        math_utils.nyquist_max_step_s(1737.4e3, 4.9e12, 10, 50.0, v_margin=0.0)


def test_nyquist_monotonicity_and_altitude_clamp():
    R = 1737.4e3
    mu = 4.9048695e12

    # Negative altitude is clamped to at least +1 km in implementation, so these match:
    dt_neg = math_utils.nyquist_max_step_s(R, mu, 100, -1000.0)
    dt_zero = math_utils.nyquist_max_step_s(R, mu, 100, 0.0)
    assert_allclose(dt_neg, dt_zero, rtol=0.0, atol=0.0)

    dt_100 = math_utils.nyquist_max_step_s(R, mu, 100, 50.0)
    dt_200 = math_utils.nyquist_max_step_s(R, mu, 200, 50.0)
    assert dt_200 < dt_100  # higher degree => smaller wavelengths => smaller dt
    assert_allclose(dt_100 / dt_200, 2.0, rtol=0.05)  # ~inverse proportional to degree


# =============================================================================
# 5) Orbital mechanics (RV -> COE)
# =============================================================================

@pytest.fixture
def moon_data():
    return {"mu": 4.9048695e12, "R": 1737.4e3}


def test_rv_to_coe_input_validation(moon_data):
    mu = moon_data["mu"]
    with pytest.raises(ValueError):
        math_utils.rv_to_coe_select([1.0, 2.0], [0.0, 0.0, 0.0], mu)  # bad r size
    with pytest.raises(ValueError):
        math_utils.rv_to_coe_select([1.0, 0.0, 0.0], [0.0, 1.0], mu)  # bad v size
    with pytest.raises(ValueError):
        math_utils.rv_to_coe_select([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], -1.0)  # bad mu
    with pytest.raises(ValueError):
        math_utils.rv_to_coe_select([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], mu, mode="bad")  # type: ignore[arg-type]


def test_rv_to_coe_circular_equatorial(moon_data):
    mu = moon_data["mu"]
    r_mag = moon_data["R"] + 100e3
    v_circ = math.sqrt(mu / r_mag)

    r = np.array([r_mag, 0.0, 0.0])
    v = np.array([0.0, v_circ, 0.0])

    a, e, inc, raan, argp, nu, eps, rnorm, vnorm, hnorm = math_utils.rv_to_coe_select(r, v, mu, mode="coe10")

    assert_allclose([rnorm, vnorm], [r_mag, v_circ], rtol=1e-12)
    assert_allclose(a, r_mag, rtol=1e-6)
    assert_allclose(e, 0.0, atol=1e-8)
    assert_allclose(inc, 0.0, atol=1e-10)
    assert_allclose(raan, 0.0, atol=0.0, rtol=0.0)
    assert_allclose(argp, 0.0, atol=0.0, rtol=0.0)
    assert_angle_close(nu, 0.0, atol=1e-10)

    # Invariants
    assert_allclose(eps, -mu / (2.0 * a), rtol=1e-12)
    assert_allclose(hnorm, r_mag * v_circ, rtol=1e-12)


def test_rv_to_coe_circular_inclined_node(moon_data):
    mu = moon_data["mu"]
    r_mag = moon_data["R"] + 80e3
    v_mag = math.sqrt(mu / r_mag)

    inc = math.radians(35.0)

    # r at ascending node (x-axis), velocity rotated by inc about x-axis
    r = np.array([r_mag, 0.0, 0.0])
    v = np.array([0.0, v_mag * math.cos(inc), v_mag * math.sin(inc)])

    a, e, inc_out, raan, argp, nu = math_utils.rv_to_coe_select(r, v, mu, mode="coe6")

    assert_allclose(a, r_mag, rtol=1e-6)
    assert_allclose(e, 0.0, atol=1e-8)
    assert_allclose(inc_out, inc, atol=1e-10)
    # By construction: ascending node along +x => RAAN=0 and argument-of-latitude u=0
    assert_angle_close(raan, 0.0, atol=1e-10)
    assert_angle_close(argp, 0.0, atol=1e-10)
    assert_angle_close(nu, 0.0, atol=1e-10)


def test_rv_to_coe_equatorial_eccentric_periapsis(moon_data):
    mu = moon_data["mu"]
    a = moon_data["R"] + 300e3
    e = 0.2
    rp = a * (1.0 - e)
    vp = math.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))

    r = np.array([rp, 0.0, 0.0])
    v = np.array([0.0, vp, 0.0])

    a_out, e_out, inc, raan, argp, nu = math_utils.rv_to_coe_select(r, v, mu, mode="coe6")
    assert_allclose(a_out, a, rtol=1e-8)
    assert_allclose(e_out, e, rtol=1e-10)
    assert_allclose(inc, 0.0, atol=1e-12)
    assert_angle_close(raan, 0.0, atol=1e-12)
    assert_angle_close(argp, 0.0, atol=1e-10)
    assert_angle_close(nu, 0.0, atol=1e-10)


def test_rv_to_coe_general_case_roundtrip(moon_data):
    mu = moon_data["mu"]
    a = moon_data["R"] + 400e3
    e = 0.15
    inc = math.radians(40.0)
    raan = math.radians(30.0)
    argp = math.radians(60.0)
    nu = math.radians(10.0)

    r, v = coe_to_rv(a, e, inc, raan, argp, nu, mu)
    a2, e2, inc2, raan2, argp2, nu2 = math_utils.rv_to_coe_select(r, v, mu, mode="coe6")

    assert_allclose(a2, a, rtol=1e-9)
    assert_allclose(e2, e, rtol=1e-10)
    assert_allclose(inc2, inc, atol=1e-10)
    assert_angle_close(raan2, raan, atol=1e-10)
    assert_angle_close(argp2, argp, atol=1e-10)
    assert_angle_close(nu2, nu, atol=1e-10)


def test_batch_y_to_elements_shapes_and_consistency(moon_data):
    mu = moon_data["mu"]
    N = 8
    y = np.zeros((6, N), dtype=np.float64)
    r_mag = moon_data["R"] + 100e3
    v_mag = math.sqrt(mu / r_mag)
    y[0, :] = r_mag
    y[4, :] = v_mag

    a_arr, e_arr, inc_arr, argp_arr, eps_arr = math_utils.batch_y_to_elements(y, mu, mode="kepler5")
    assert a_arr.shape == (N,)
    assert e_arr.shape == (N,)
    assert inc_arr.shape == (N,)
    assert argp_arr.shape == (N,)
    assert eps_arr.shape == (N,)

    # Identical inputs => identical outputs
    assert_allclose(a_arr, a_arr[0], rtol=0.0, atol=0.0)
    assert_allclose(e_arr, 0.0, atol=1e-8)
    assert_allclose(inc_arr, 0.0, atol=1e-10)

    # Validation: wrong shape
    with pytest.raises(ValueError):
        math_utils.batch_y_to_elements(np.zeros((5, N)), mu)  # type: ignore[arg-type]


# =============================================================================
# 6) Grid samplers
# =============================================================================

@pytest.mark.parametrize("as_flat", [False, True])
def test_sample_2d_nearest_boundaries_and_wrapping(as_flat):
    data2d = np.arange(9, dtype=np.float64).reshape(3, 3)
    data = data2d.ravel() if as_flat else data2d

    # Clamp top/bottom rows
    assert math_utils.sample_2d_nearest(data, -1.0, 1.0, 3, 3) == 1.0
    assert math_utils.sample_2d_nearest(data, 99.0, 1.0, 3, 3) == 7.0

    # Wrap columns
    assert math_utils.sample_2d_nearest(data, 1.0, 3.0, 3, 3) == 3.0  # col wraps to 0

    # Avoid tie-breaking assumptions: use values slightly off .5
    assert math_utils.sample_2d_nearest(data, 0.49, 0.49, 3, 3) == 0.0
    assert math_utils.sample_2d_nearest(data, 0.51, 0.51, 3, 3) == 4.0

@pytest.mark.parametrize("as_flat", [False, True])
def test_sample_2d_bilinear_center_and_seam(as_flat):
    # value = 10*row + col
    grid = np.array([[0.0, 1.0, 2.0, 3.0],
                     [10.0, 11.0, 12.0, 13.0],
                     [20.0, 21.0, 22.0, 23.0]], dtype=np.float64)
    data = grid.ravel() if as_flat else grid

    # Center between (0,0),(0,1),(1,0),(1,1) at row=0.5, col=0.5 => 5.5
    out = math_utils.sample_2d_bilinear(data, 0.5, 0.5, 3, 4)
    assert_allclose(out, 5.5, atol=1e-12)

    # Longitude seam: col_f=-0.2 => floor=-1, dc=0.8 => between col=3 and col=0
    out = math_utils.sample_2d_bilinear(data, 0.0, -0.2, 3, 4)
    # row clamped to 0; interpolate between 3 and 0 with dc=0.8 => 3*(0.2)+0*(0.8)=0.6
    assert_allclose(out, 0.6, atol=1e-12)


def test_sample_grid_bilinear_mapping_and_lon_wrap():
    # 3x4 grid: value = 10*row + col
    grid = np.array([[0.0, 1.0, 2.0, 3.0],
                     [10.0, 11.0, 12.0, 13.0],
                     [20.0, 21.0, 22.0, 23.0]], dtype=np.float64)

    nlines, nsamples = grid.shape
    res = 10.0
    lat0 = 20.0
    lon0 = 0.0

    # (lat,lon)=(15,5) -> (row_f,col_f)=(0.5,0.5) => 5.5
    out = math_utils.sample_grid_bilinear(15.0, 5.0, grid, nlines, nsamples, res, lon0, lat0)
    assert_allclose(out, 5.5, atol=1e-12)

    # lon wrap: lon=-5 => 355 => col_f=35.5 => seam between col=3 and col=0 with dc=0.5
    out = math_utils.sample_grid_bilinear(20.0, -5.0, grid, nlines, nsamples, res, lon0, lat0)
    assert_allclose(out, 1.5, atol=1e-12)


@pytest.mark.parametrize("as_flat", [False, True])
def test_scaled_samplers_basic_missing_and_tie_rule(as_flat):
    # DN grid, with a missing DN
    dn = np.array([[0.0, 1.0],
                   [2.0, 3.0]], dtype=np.float64)
    data = dn.ravel() if as_flat else dn

    scale, offset = 2.0, 10.0
    missing = 2.0

    # Nearest (half-up): (0.5,0.5) => (1,1) => DN=3 => 3*2+10 = 16
    out = math_utils.sample_2d_scaled_nearest(data, 0.5, 0.5, 2, 2, scale, offset, missing)
    assert_allclose(out, 16.0, atol=0.0, rtol=0.0)

    # Missing DN -> NaN
    out_missing = math_utils.sample_2d_scaled_nearest(data, 1.0, 0.0, 2, 2, scale, offset, missing)
    assert math.isnan(out_missing)

    # Scaled bilinear: if ANY neighbor missing => fallback to nearest
    # At center, neighbors include DN=2 (missing) => fallback to nearest => DN=3 => 16
    out_bilin = math_utils.sample_2d_scaled_bilinear(data, 0.5, 0.5, 2, 2, scale, offset, missing)
    assert_allclose(out_bilin, 16.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "fn_name, args",
    [
        ("sample_2d_nearest", (np.zeros((2, 2)), 0.0, 0.0, 0, 2)),
        ("sample_2d_bilinear", (np.zeros((2, 2)), 0.0, 0.0, 2, 0)),
        ("sample_grid_bilinear", (0.0, 0.0, np.zeros((2, 2)), 2, 2, 0.0, 0.0, 0.0)),
        ("sample_2d_scaled_nearest", (np.zeros((2, 2)), 0.0, 0.0, 0, 2, 1.0, 0.0, -9999.0)),
    ],
)
def test_sampler_invalid_dims_raise(fn_name, args):
    fn = getattr(math_utils, fn_name)
    with pytest.raises(ValueError):
        fn(*args)


@pytest.mark.parametrize(
    "fn_name, good_args, bad_data",
    [
        ("sample_2d_nearest", (0.0, 0.0, 2, 2), np.zeros((3, 3))),
        ("sample_2d_bilinear", (0.0, 0.0, 2, 2), np.zeros((3, 3))),
        ("sample_2d_scaled_nearest", (0.0, 0.0, 2, 2, 1.0, 0.0, -9999.0), np.zeros((3, 3))),
        ("sample_2d_scaled_bilinear", (0.0, 0.0, 2, 2, 1.0, 0.0, -9999.0), np.zeros((3, 3))),
    ],
)
def test_sampler_shape_mismatch_raises(fn_name, good_args, bad_data):
    fn = getattr(math_utils, fn_name)
    with pytest.raises(ValueError):
        fn(bad_data, *good_args)



if __name__ == "__main__":
    import sys

    print("This is a pytest test module. Run it with:")
    print("python -m pytest -vv -rA --durations=10 tests/test_math_utils.py")
    sys.exit(0)