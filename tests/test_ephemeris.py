# tests/test_ephemeris.py
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

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Import helper
# -----------------------------------------------------------------------------
# These tests are intended to be runnable from the repo root without installing
# the package. We add the repo root to sys.path as a fallback.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from lunaris.physics.ephemeris import (
    EphemerisManager,
    EphemerisTables,
    _build_time_grid,
    _capture_kernel_provenance,
    _classify_kernel,
    _try_fill_spkezr_state_tables_si,
    get_ephem_state,
    get_ephem_state_with_velocity,
    interp_vec3_derivative_safe,
    load_ephemeris_tables_npz,
    save_ephemeris_tables_npz,
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
    sun = np.asarray(r_sun_tab_m, dtype=np.float64)
    earth = np.asarray(r_earth_tab_m, dtype=np.float64)
    edge_order = 2 if sun.shape[0] >= 3 else 1
    sun_v = (
        np.gradient(sun, float(dt_s), axis=0, edge_order=edge_order)
        if sun.shape[0] > 1
        else np.zeros_like(sun)
    )
    earth_v = (
        np.gradient(earth, float(dt_s), axis=0, edge_order=edge_order)
        if earth.shape[0] > 1
        else np.zeros_like(earth)
    )
    return EphemerisTables(
        dt_s=float(dt_s),
        t_tab_s=np.asarray(t_tab_s, dtype=np.float64),
        et0=0.0,
        q_i2f_tab=np.asarray(q_i2f_tab, dtype=np.float64),
        r_earth_tab_m=earth,
        r_sun_tab_m=sun,
        v_earth_tab_m_s=earth_v,
        v_sun_tab_m_s=sun_v,
        mu_earth_m3s2=3.986004418e14,
        mu_sun_m3s2=1.32712440018e20,
        inertial_frame="MOCK_INERTIAL",
        fixed_frame="MOCK_FIXED",
        observer="MOCK_OBSERVER",
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_build_time_grid_covers_noninteger_duration() -> None:
    """Table grids remain uniform but extend far enough to cover the run span."""
    np.testing.assert_allclose(_build_time_grid(100.0, 60.0), np.array([0.0, 60.0, 120.0]))
    np.testing.assert_allclose(_build_time_grid(120.0, 60.0), np.array([0.0, 60.0, 120.0]))


def test_vectorized_spice_probe_reports_the_fallback_cause() -> None:
    def failing_spkezr(*_args, **_kwargs):
        raise RuntimeError("vectorized ET arrays unsupported")

    ok, reason = _try_fill_spkezr_state_tables_si(
        np.empty((2, 3), dtype=np.float64),
        np.empty((2, 3), dtype=np.float64),
        spkezr=failing_spkezr,
        target="EARTH",
        et_tab=np.asarray([0.0, 60.0]),
        frame="J2000",
        observer="MOON",
    )

    assert not ok
    assert reason is not None
    assert "RuntimeError" in reason
    assert "vectorized ET arrays unsupported" in reason


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


def test_ephemeris_vec3_derivative_safe_linear_and_degenerate() -> None:
    dt_s = 10.0
    tab = np.array(
        [
            [0.0, 2.0, -4.0],
            [10.0, 12.0, 6.0],
            [20.0, 22.0, 16.0],
        ],
        dtype=np.float64,
    )

    got = np.array(interp_vec3_derivative_safe(10.0, dt_s, tab), dtype=np.float64)
    np.testing.assert_allclose(got, np.array([1.0, 1.0, 1.0]), rtol=0.0, atol=0.0)

    one_row = np.array([[5.0, 6.0, 7.0]], dtype=np.float64)
    got_zero = np.array(interp_vec3_derivative_safe(0.0, dt_s, one_row), dtype=np.float64)
    np.testing.assert_allclose(got_zero, np.zeros(3), rtol=0.0, atol=0.0)
    one_velocity = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
    np.testing.assert_allclose(
        interp_vec3_derivative_safe(0.0, dt_s, one_row, one_velocity),
        one_velocity[0],
    )


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
        sx, sy, sz, ex, ey, ez, qw, qx, qy, qz = get_ephem_state_with_velocity(
            float(tq),
            float(dtD),
            r_sun_D,
            r_earth_D,
            tablesD.v_sun_tab_m_s,
            tablesD.v_earth_tab_m_s,
            qD,
            True,
        )
        sun_k = np.array([sx, sy, sz], dtype=np.float64)
        earth_k = np.array([ex, ey, ez], dtype=np.float64)
        quat_k = np.array([qw, qx, qy, qz], dtype=np.float64)

        np.testing.assert_allclose(sun_h, sun_k, atol=1e-10)
        np.testing.assert_allclose(earth_h, earth_k, atol=1e-10)
        _assert_same_rotation(quat_h, quat_k, atol=1e-9)


# -----------------------------------------------------------------------------
# Strict contract validation (EphemerisTables.__post_init__)
# -----------------------------------------------------------------------------
def _valid_table_kwargs(n: int = 4, dt: float = 60.0) -> dict:
    q = np.zeros((n, 4), dtype=np.float64)
    q[:, 0] = 1.0
    return {
        "dt_s": dt,
        "t_tab_s": np.arange(n, dtype=np.float64) * dt,
        "et0": 0.0,
        "q_i2f_tab": q,
        "r_earth_tab_m": np.ones((n, 3), dtype=np.float64),
        "r_sun_tab_m": np.ones((n, 3), dtype=np.float64),
        "v_earth_tab_m_s": np.zeros((n, 3), dtype=np.float64),
        "v_sun_tab_m_s": np.zeros((n, 3), dtype=np.float64),
        "mu_earth_m3s2": 3.986004418e14,
        "mu_sun_m3s2": 1.32712440018e20,
    }


def test_tables_valid_construction_passes():
    EphemerisTables(**_valid_table_kwargs())


def test_schema_v2_npz_round_trip_and_legacy_fail_closed(tmp_path):
    tables = EphemerisTables(**_valid_table_kwargs())
    path = save_ephemeris_tables_npz(tmp_path / "ephem", tables)
    loaded = load_ephemeris_tables_npz(path)
    assert loaded.schema_version == 2
    assert loaded.interpolation_kind == "cubic_hermite_position_velocity"
    np.testing.assert_array_equal(loaded.r_sun_tab_m, tables.r_sun_tab_m)
    np.testing.assert_array_equal(loaded.v_sun_tab_m_s, tables.v_sun_tab_m_s)

    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(
        legacy,
        schema_version=np.asarray(1),
        dt_s=np.asarray(60.0),
        t_tab_s=np.arange(2) * 60.0,
    )
    with pytest.raises(ValueError, match="not schema v2"):
        load_ephemeris_tables_npz(legacy)


def test_tracked_ephemeris_evidence_has_no_workstation_absolute_paths():
    evidence_path = (
        _REPO_ROOT
        / "validation"
        / "ephemeris"
        / "interpolation_validation_2026_07_18.json"
    )
    text = evidence_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert re.search(r"(?i)[a-z]:[\\/]", text) is None
    assert not str(payload["configuration"]["kernel_dir_hint"]).startswith(("/", "\\"))
    for record in payload["ephemeris_contract"]["kernel_provenance"]:
        assert "path" not in record
        assert not str(record["path_hint"]).startswith(("/", "\\"))


# -----------------------------------------------------------------------------
# SPICE kernel provenance (P1a): kernels in the manifest chain
# -----------------------------------------------------------------------------
def test_classify_kernel_by_extension():
    assert _classify_kernel("naif0012.tls") == "LSK"
    assert _classify_kernel("de440.bsp") == "SPK"
    assert _classify_kernel("moon_pa_de440_200625.bpc") == "PCK"
    assert _classify_kernel("moon_080317.tf") == "FK"
    assert _classify_kernel("de440.bsp.txt") == "SPK"  # detached-label style
    assert _classify_kernel("mystery.xyz") == "UNKNOWN"


def test_capture_kernel_provenance_hashes_files(tmp_path):
    k1 = tmp_path / "naif0012.tls"
    k1.write_text("LEAPSECONDS", encoding="utf-8")
    missing = tmp_path / "de440.bsp"  # never created

    prov = _capture_kernel_provenance([str(k1), str(missing)])
    assert prov[0]["name"] == "naif0012.tls"
    assert prov[0]["kind"] == "LSK"
    assert prov[0]["sha256"] and len(prov[0]["sha256"]) == 64
    # Missing file: hash is None, capture still succeeds (never aborts a build).
    assert prov[1]["kind"] == "SPK"
    assert prov[1]["sha256"] is None


def test_manager_kernel_provenance_reports_window_and_kernels():
    kwargs = _valid_table_kwargs(n=5, dt=60.0)
    kwargs["et0"] = 1000.0
    kwargs["kernel_provenance"] = (
        {"name": "naif0012.tls", "path": "/k/naif0012.tls", "kind": "LSK", "sha256": "ab"},
    )
    kwargs["time_scale_note"] = "UTC->ET via LSK"
    mgr = EphemerisManager.from_tables(EphemerisTables(**kwargs))

    prov = mgr.kernel_provenance()
    assert prov["kernels"][0]["kind"] == "LSK"
    assert prov["time_scale_note"] == "UTC->ET via LSK"
    assert prov["et_start"] == 1000.0
    # et_end = et0 + last table time (4 * 60 s).
    assert prov["et_end"] == 1000.0 + 240.0


def test_tables_reject_empty_time_grid():
    kwargs = _valid_table_kwargs()
    kwargs["t_tab_s"] = np.zeros((0,), dtype=np.float64)
    kwargs["q_i2f_tab"] = np.zeros((0, 4), dtype=np.float64)
    kwargs["r_earth_tab_m"] = np.zeros((1, 3), dtype=np.float64)
    kwargs["r_sun_tab_m"] = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="N >= 1"):
        EphemerisTables(**kwargs)


@pytest.mark.parametrize("bad_dt", [float("nan"), float("inf"), 0.0, -1.0])
def test_tables_reject_bad_dt(bad_dt):
    kwargs = _valid_table_kwargs()
    kwargs["dt_s"] = bad_dt
    with pytest.raises(ValueError, match="dt_s"):
        EphemerisTables(**kwargs)


def test_tables_reject_non_finite_time_grid():
    kwargs = _valid_table_kwargs()
    kwargs["t_tab_s"] = np.array([0.0, 60.0, np.nan, 180.0])
    with pytest.raises(ValueError, match="finite"):
        EphemerisTables(**kwargs)


def test_tables_reject_time_grid_not_starting_at_zero():
    kwargs = _valid_table_kwargs()
    kwargs["t_tab_s"] = kwargs["t_tab_s"] + 60.0
    with pytest.raises(ValueError, match="start at 0.0"):
        EphemerisTables(**kwargs)


def test_tables_reject_non_uniform_time_grid():
    kwargs = _valid_table_kwargs()
    t = kwargs["t_tab_s"].copy()
    t[2] += 1.0  # break uniform dt_s spacing
    kwargs["t_tab_s"] = t
    with pytest.raises(ValueError, match="uniformly spaced"):
        EphemerisTables(**kwargs)


def test_tables_reject_zero_quaternion_rows():
    kwargs = _valid_table_kwargs()
    q = kwargs["q_i2f_tab"].copy()
    q[1] = 0.0
    kwargs["q_i2f_tab"] = q
    with pytest.raises(ValueError, match="nonzero quaternions"):
        EphemerisTables(**kwargs)


def test_tables_reject_non_finite_quaternions():
    kwargs = _valid_table_kwargs()
    q = kwargs["q_i2f_tab"].copy()
    q[1, 2] = np.inf
    kwargs["q_i2f_tab"] = q
    with pytest.raises(ValueError, match="finite"):
        EphemerisTables(**kwargs)


def test_tables_reject_non_unit_quaternions():
    kwargs = _valid_table_kwargs()
    q = kwargs["q_i2f_tab"].copy()
    q[1] *= 1.01
    kwargs["q_i2f_tab"] = q
    with pytest.raises(ValueError, match="unit quaternions"):
        EphemerisTables(**kwargs)


@pytest.mark.parametrize("field", ["r_earth_tab_m", "r_sun_tab_m"])
def test_tables_reject_non_finite_position_tables(field):
    kwargs = _valid_table_kwargs()
    arr = kwargs[field].copy()
    arr[0, 1] = np.nan
    kwargs[field] = arr
    with pytest.raises(ValueError, match="finite"):
        EphemerisTables(**kwargs)


@pytest.mark.parametrize("field", ["mu_earth_m3s2", "mu_sun_m3s2"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_tables_reject_bad_gm(field, bad):
    kwargs = _valid_table_kwargs()
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        EphemerisTables(**kwargs)


if __name__ == "__main__":
    import sys

    print("This is a pytest test module. Run it with:")
    print("python -m pytest -vv -rA --durations=10 tests/test_ephemeris.py")
    sys.exit(0)
