# tests/test_ephemeris.py
# -*- coding: utf-8 -*-
"""
Pytest port of the "Premium self-test" that used to live under:

    if __name__ == "__main__":

in the ephemeris module.

Notes
-----
- No SPICE required (tables are synthetic).
- Focuses on interpolation, clamp behavior, degenerate (N=1) tables,
  quaternion sign-flip continuity, "out" buffer semantics, and high-level vs
  Numba-kernel consistency.

Run:
    pytest -q
or:
    python -m pytest -q
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import sys
from pathlib import Path


# -----------------------------------------------------------------------------
# Import helper
# -----------------------------------------------------------------------------
# These tests are intended to be runnable from the repo root without installing
# the package. We add the repo root to sys.path as a fallback.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from models.ephemeris import (
    EphemerisTables,
    EphemerisManager,
    get_ephem_state,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not (n > 0.0):
        raise AssertionError("Quaternion norm is zero.")
    return q / n


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion [w,x,y,z] to rotation matrix."""
    w, x, y, z = map(float, q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not (n > 0.0):
        raise AssertionError("Quaternion norm is zero for quat_to_R.")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _assert_same_rotation(q1: np.ndarray, q2: np.ndarray, *, atol: float = 1e-10) -> None:
    """q and -q represent same rotation; compare via rotation matrices."""
    R1 = _quat_to_R(q1)
    R2 = _quat_to_R(q2)
    if not np.allclose(R1, R2, atol=atol):
        raise AssertionError(f"Rotation mismatch.\nR1=\n{R1}\nR2=\n{R2}\nR1-R2=\n{R1 - R2}")


def _make_tables(
    dt_s: float,
    t_tab_s: np.ndarray,
    r_sun_tab_m: np.ndarray,
    r_earth_tab_m: np.ndarray,
    q_i2f_tab: np.ndarray,
) -> EphemerisTables:
    return EphemerisTables(
        dt_s=float(dt_s),
        t_tab_s=np.asarray(t_tab_s, dtype=np.float64),
        et0=0.0,
        q_i2f_tab=np.asarray(q_i2f_tab, dtype=np.float64),
        r_earth_tab_m=np.asarray(r_earth_tab_m, dtype=np.float64),
        r_sun_tab_m=np.asarray(r_sun_tab_m, dtype=np.float64),
        mu_earth_m3s2=3.986004418e14,
        mu_sun_m3s2=1.32712440018e20,
        inertial_frame="MOCK_INERTIAL",
        fixed_frame="MOCK_FIXED",
        observer="MOCK_OBSERVER",
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_ephemeris_case_a_linear_interp_clamp_and_out_buffer() -> None:
    """Case A: Basic linear interpolation + clamp tests (N=2) + out-buffer semantics."""
    dt_s = 10.0
    t_tab_s = np.array([0.0, 10.0], dtype=np.float64)

    # Sun: [0,0,0] -> [10,0,0] over 10 s => at t=5 => [5,0,0]
    r_sun_tab_m = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)

    # Earth: [0,100,0] -> [0,200,0] => at t=5 => [0,150,0]
    r_earth_tab_m = np.array([[0.0, 100.0, 0.0], [0.0, 200.0, 0.0]], dtype=np.float64)

    # Quaternion: identity constant
    q_i2f_tab = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    tables = _make_tables(dt_s, t_tab_s, r_sun_tab_m, r_earth_tab_m, q_i2f_tab)
    mgr = EphemerisManager(tables)

    # In-range interpolation
    t_query = 5.0
    np.testing.assert_allclose(mgr.get_sun_position(t_query), np.array([5.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(mgr.get_earth_position(t_query), np.array([0.0, 150.0, 0.0]), atol=1e-12)

    # Boundary points
    np.testing.assert_allclose(mgr.get_sun_position(0.0), np.array([0.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(mgr.get_sun_position(10.0), np.array([10.0, 0.0, 0.0]), atol=1e-12)

    # Clamp behavior
    np.testing.assert_allclose(mgr.get_sun_position(-5.0), np.array([0.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(mgr.get_sun_position(999.0), np.array([10.0, 0.0, 0.0]), atol=1e-12)

    # Out-buffer behavior
    out3 = np.empty(3, dtype=np.float64)
    ret3 = mgr.get_sun_position(t_query, out=out3)
    assert ret3 is out3
    np.testing.assert_allclose(out3, np.array([5.0, 0.0, 0.0]), atol=1e-12)

    out4 = np.empty(4, dtype=np.float64)
    ret4 = mgr.get_inertial_to_fixed_rotation(t_query, out=out4)
    assert ret4 is out4
    np.testing.assert_allclose(out4, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-12)


@pytest.mark.parametrize("tq", [-123.0, 0.0, 1.0, 999.0])
def test_ephemeris_case_b_degenerate_n1_tables(tq: float) -> None:
    """Case B: N=1 degenerate tables (third-body disabled scenario)."""
    dt_s = 10.0
    t_tab_s_1 = np.array([0.0], dtype=np.float64)

    r_sun_1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    r_earth_1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    # Non-identity quat, constant (~45deg about +Y)
    q_const = _norm_quat(np.array([0.9238795325, 0.0, 0.3826834324, 0.0], dtype=np.float64))
    q_1 = np.array([q_const], dtype=np.float64)

    tables1 = _make_tables(dt_s, t_tab_s_1, r_sun_1, r_earth_1, q_1)
    mgr1 = EphemerisManager(tables1)

    np.testing.assert_allclose(mgr1.get_sun_position(tq), r_sun_1[0], atol=1e-12)
    np.testing.assert_allclose(mgr1.get_earth_position(tq), r_earth_1[0], atol=1e-12)

    # Quaternion should represent the same rotation (even if implementation returns q or -q)
    qg = mgr1.get_inertial_to_fixed_rotation(tq)
    _assert_same_rotation(qg, q_const, atol=1e-10)


def test_ephemeris_case_c_quaternion_small_angle_and_sign_flip_continuity() -> None:
    """
    Case C: Quaternion small-angle stability + sign flip continuity.

    The table has a deliberate sign flip between samples: q0 -> -q1.
    SLERP continuity should prevent a "long way" interpolation jump.
    """
    dt_s = 10.0
    t_tab_s = np.array([0.0, 10.0], dtype=np.float64)

    angle = 1e-6  # rad
    half = 0.5 * angle
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1p = _norm_quat(np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64))  # small rot about Z
    q1m = -q1p  # same rotation, flipped sign

    q_flip = np.vstack([q0, q1m]).astype(np.float64)

    r_sun_c = np.vstack([np.zeros(3), np.zeros(3)]).astype(np.float64)
    r_earth_c = np.vstack([np.zeros(3), np.zeros(3)]).astype(np.float64)

    tablesC = _make_tables(dt_s, t_tab_s, r_sun_c, r_earth_c, q_flip)
    mgrC = EphemerisManager(tablesC)

    q_mid = mgrC.get_inertial_to_fixed_rotation(5.0)

    # Expected halfway rotation (~angle/2 about Z => half-angle is angle/4 in quaternion sin/cos)
    q_half = _norm_quat(np.array([math.cos(0.25 * angle), 0.0, 0.0, math.sin(0.25 * angle)], dtype=np.float64))
    _assert_same_rotation(q_mid, q_half, atol=1e-7)  # looser tol: depends on SLERP implementation details


@pytest.mark.slow
def test_ephemeris_case_d_high_level_vs_kernel_random_regression() -> None:
    """Case D: High-level vs Numba kernel consistency (random regression)."""
    rng = np.random.default_rng(12345)

    N = 50
    dtD = 2.0
    t_tab_D = (np.arange(N, dtype=np.float64) * dtD)

    r_sun_D = rng.normal(size=(N, 3)).astype(np.float64) * 1e7
    r_earth_D = rng.normal(size=(N, 3)).astype(np.float64) * 1e7

    qD = rng.normal(size=(N, 4)).astype(np.float64)
    qD /= np.linalg.norm(qD, axis=1, keepdims=True)

    # Enforce continuity (q and -q same; keep dot positive)
    for i in range(1, N):
        if float(np.dot(qD[i - 1], qD[i])) < 0.0:
            qD[i] *= -1.0

    tablesD = _make_tables(dtD, t_tab_D, r_sun_D, r_earth_D, qD)
    mgrD = EphemerisManager(tablesD)

    for _ in range(200):
        tq = float(rng.uniform(-10.0, t_tab_D[-1] + 10.0))

        sun_h = mgrD.get_sun_position(tq)
        earth_h = mgrD.get_earth_position(tq)
        quat_h = mgrD.get_inertial_to_fixed_rotation(tq)

        # Kernel returns floats (sx,sy,sz, ex,ey,ez, qw,qx,qy,qz)
        sx, sy, sz, ex, ey, ez, qw, qx, qy, qz = get_ephem_state(
            float(tq),
            float(dtD),
            r_sun_D,
            r_earth_D,
            qD,
        )
        sun_k = np.array([sx, sy, sz], dtype=np.float64)
        earth_k = np.array([ex, ey, ez], dtype=np.float64)
        quat_k = np.array([qw, qx, qy, qz], dtype=np.float64)

        np.testing.assert_allclose(sun_h, sun_k, atol=1e-10)
        np.testing.assert_allclose(earth_h, earth_k, atol=1e-10)
        _assert_same_rotation(quat_h, quat_k, atol=1e-9)


if __name__ == "__main__":
    import sys

    print("This is a pytest test module. Run it with:")
    print("python -m pytest -vv -rA --durations=10 tests/test_ephemeris.py")
    sys.exit(0)

