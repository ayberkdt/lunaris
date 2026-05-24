"""
Pytest-based regression tests for the spherical harmonics gravity kernels.

Goals
-----
- Deterministic (fixed RNG seed)
- Robust across CPU/Numba variations (conservative tolerances)
- Fast enough for CI while still exercising key code paths

Run:
    pytest -q

Optional:
    # Force a specific module path:
    export LUNAR_SH_MODULE=models.spherical_harmonics
    pytest -q
"""

from __future__ import annotations

from pathlib import Path
import importlib
import importlib.util
import os
import sys
import math
from typing import Tuple

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Import helper
# -----------------------------------------------------------------------------
def _repo_root_from_this_file() -> "Path":
    """
    Resolve repository root from this test file location:
        repo_root/tests/test_spherical_harmonics.py  -> repo_root
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "tests":
            return p.parent
    return here.parent


def _ensure_repo_on_syspath() -> None:
    """
    Ensure the repo root is importable.
    Also add repo_root/LUNAR_SIMULATION if it exists (common in this project layout).
    """
    root = _repo_root_from_this_file()

    candidates = [root]
    lunar_dir = root / "LUNAR_SIMULATION"
    if lunar_dir.exists():
        candidates.append(lunar_dir)

    for c in candidates:
        s = str(c)
        if s not in sys.path:
            sys.path.insert(0, s)


def _import_sh_module():
    """
    Import spherical harmonics module robustly.

    Priority:
    1) If env var LUNAR_SH_MODULE is set, import that exact module path.
       Example:
           export LUNAR_SH_MODULE=models.spherical_harmonics
           pytest -q
    2) Otherwise, try common module paths.
    3) Otherwise, try importing by file path from likely locations inside the repo.
    """
    _ensure_repo_on_syspath()

    env_name = os.environ.get("LUNAR_SH_MODULE", "").strip()
    if env_name:
        return importlib.import_module(env_name)

    candidates = (
        "models.spherical_harmonics",
        "models.spherical_harmonics_v2",
        "core.spherical_harmonics",
        "spherical_harmonics",
        "spherical_harmonics_v2",
        "spherical_harmonics_fixed",
    )
    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            last_err = e

    # Path-based fallback (no package install needed)
    root = _repo_root_from_this_file()
    path_candidates = [
        root / "models" / "spherical_harmonics.py",
        root / "models" / "spherical_harmonics_v2.py",
        root / "models" / "spherical_harmonics_fixed.py",
        root / "spherical_harmonics.py",
        root / "spherical_harmonics_v2.py",
        root / "spherical_harmonics_fixed.py",
        (root / "LUNAR_SIMULATION" / "models" / "spherical_harmonics.py"),
        (root / "LUNAR_SIMULATION" / "models" / "spherical_harmonics_v2.py"),
        (root / "LUNAR_SIMULATION" / "models" / "spherical_harmonics_fixed.py"),
    ]
    for p in path_candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("spherical_harmonics_under_test", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                # dataclasses + string annotations need module in sys.modules before exec_module
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                return mod

    raise ImportError(
        "Could not import spherical harmonics module. Tried module names: "
        + ", ".join(candidates)
        + " and file paths: "
        + ", ".join(str(p) for p in path_candidates)
    ) from last_err


# -----------------------------------------------------------------------------
# Small math helpers
# -----------------------------------------------------------------------------
def _norm3(ax: float, ay: float, az: float) -> float:
    return math.sqrt(ax * ax + ay * ay + az * az)


def _is_finite3(ax: float, ay: float, az: float) -> bool:
    return math.isfinite(ax) and math.isfinite(ay) and math.isfinite(az)


def _rel_err(a: float, b: float) -> float:
    denom = max(1e-30, abs(b))
    return abs(a - b) / denom


def _rel_err_vec(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    na = _norm3(*a)
    nb = _norm3(*b)
    denom = max(1e-30, nb)
    return abs(na - nb) / denom


# -----------------------------------------------------------------------------
# Compatibility helpers (old/new API)
# -----------------------------------------------------------------------------
def _get_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"None of these attributes exist on {type(obj).__name__}: {names}")


def _point_mass(sh, x: float, y: float, z: float, mu: float):
    """
    Support both names:
      - compute_point_mass_acceleration(x,y,z,mu)
      - GravityModel (degree 0) fallback
    """
    if hasattr(sh, "compute_point_mass_acceleration"):
        return sh.compute_point_mass_acceleration(x, y, z, mu)
    if hasattr(sh, "_compute_point_mass_acceleration"):
        return sh.compute_point_mass_acceleration(x, y, z, mu)
    pytest.skip("Module under test has no point-mass acceleration function.")


def _model_fields(model):
    """
    Return a dict of the model fields used by kernels, supporting old/new naming.
    """
    return {
        "r_ref": _get_attr(model, "r_ref", "R_ref_m"),
        "mu": _get_attr(model, "mu", "GM_m3s2"),
        "C": _get_attr(model, "c_coeffs", "Cnm"),
        "S": _get_attr(model, "s_coeffs", "Snm"),
        "diag": _get_attr(model, "diag_coeffs", "diag"),
        "subdiag": _get_attr(model, "subdiag_coeffs", "subdiag"),
        "A": _get_attr(model, "a_coeffs", "A"),
        "B": _get_attr(model, "b_coeffs", "B"),
        "scale_m": _get_attr(model, "scale_m_table", "scale_m"),
        "ws": _get_attr(model, "workspace", "ws"),
    }


def _call_fixed_numba(sh, model, x: float, y: float, z: float, deg: int):
    """
    Deterministic serial kernel (Numba wrapper).
    """
    if not hasattr(sh, "sh_accel_fixed_numba"):
        pytest.skip("Module does not expose sh_accel_fixed_numba.")
    f = _model_fields(model)
    ws = f["ws"]
    return sh.sh_accel_fixed_numba(
        float(x), float(y), float(z), int(deg),
        float(f["r_ref"]), float(f["mu"]),
        f["C"], f["S"],
        f["diag"], f["subdiag"],
        f["A"], f["B"], f["scale_m"],
        ws.P, ws.dP, ws.cos_m, ws.sin_m,
    )


def _call_fixed_parallel(sh, model, x: float, y: float, z: float, deg: int):
    """
    Explicit parallel path via the Python dispatch wrapper, if available.
    """
    if not hasattr(sh, "sh_accel_fixed"):
        pytest.skip("Module does not expose sh_accel_fixed (python dispatch).")

    f = _model_fields(model)
    ws = f["ws"]

    try:
        return sh.sh_accel_fixed(
            float(x), float(y), float(z), int(deg),
            float(f["r_ref"]), float(f["mu"]),
            f["C"], f["S"],
            f["diag"], f["subdiag"],
            f["A"], f["B"], f["scale_m"],
            ws.P, ws.dP, ws.cos_m, ws.sin_m,
            use_parallel=True,
            parallel_threshold=-1,  # always force parallel when available
        )
    except TypeError:
        pytest.skip("sh_accel_fixed signature does not support explicit parallel dispatch.")
    except AttributeError:
        pytest.skip("Parallel dispatch not available in this build/module.")


def _make_model(sh, deg: int, constants, *, c20: float = 0.0, extra_terms=()):
    """
    Build an in-memory GravityModel with optional small coefficients.

    extra_terms: iterable of (is_c, n, m, value)
      - is_c True -> C[n,m] = value
      - is_c False -> S[n,m] = value
    """
    if not hasattr(sh, "GravityModel"):
        pytest.skip("GravityModel not available in module under test")

    R_ref, GM = constants
    C = np.zeros((deg + 1, deg + 1), dtype=np.float64)
    S = np.zeros_like(C)

    if c20 != 0.0:
        C[2, 0] = float(c20)

    for is_c, n, m, v in extra_terms:
        n = int(n)
        m = int(m)
        if is_c:
            C[n, m] = float(v)
        else:
            S[n, m] = float(v)

    # Keep positional args to support both old/new signatures
    return sh.GravityModel.from_arrays(int(deg), float(R_ref), float(GM), C, S)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sh():
    return _import_sh_module()


@pytest.fixture(scope="session")
def constants():
    # Mock Moon-like numbers (consistent across tests)
    R_ref = 1_737_400.0       # [m]
    GM = 4.904_869_5e12       # [m^3/s^2]
    return R_ref, GM


@pytest.fixture()
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session", autouse=True)
def _warmup_numba(sh, constants):
    """
    Trigger a tiny JIT warmup once per test session so the first real test
    doesn't pay compilation cost (and to fail early if Numba isn't working).
    """
    R_ref, _GM = constants
    deg = 6
    model = _make_model(sh, deg, constants, c20=0.0)

    # call once
    _ = _call_fixed_numba(sh, model, R_ref + 100e3, 0.0, 0.0, deg)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_zero_coeffs_matches_point_mass(sh, constants, rng):
    """
    With all coefficients set to zero, SH acceleration should reduce to a point-mass field.
    """
    R_ref, GM = constants
    deg = 10
    model = _make_model(sh, deg, constants, c20=0.0)

    tol_rel_strict = 1e-12
    n_pts = 20
    max_rel = 0.0

    for _ in range(n_pts):
        alt = float(rng.uniform(50e3, 2_000e3))
        r = R_ref + alt

        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        x, y, z = float(r * v[0]), float(r * v[1]), float(r * v[2])

        ax_ref, ay_ref, az_ref = _point_mass(sh, x, y, z, GM)
        ax_sh, ay_sh, az_sh = _call_fixed_numba(sh, model, x, y, z, deg)

        assert _is_finite3(ax_sh, ay_sh, az_sh)

        max_rel = max(
            max_rel,
            _rel_err(ax_sh, ax_ref),
            _rel_err(ay_sh, ay_ref),
            _rel_err(az_sh, az_ref),
        )

    assert max_rel < tol_rel_strict, f"max componentwise rel err too large: {max_rel:.3e}"


def test_harmonic_perturbation_detected(sh, constants):
    """
    A non-zero C20-like coefficient should perturb the point-mass acceleration measurably.
    """
    R_ref, GM = constants
    deg = 10
    model_j2 = _make_model(sh, deg, constants, c20=-2.0e-4)

    x, y, z = R_ref + 500e3, 200e3, -100e3  # avoid symmetry points

    a_ref = _point_mass(sh, x, y, z, GM)
    a_j2 = _call_fixed_numba(sh, model_j2, x, y, z, deg)

    diff = (a_j2[0] - a_ref[0], a_j2[1] - a_ref[1], a_j2[2] - a_ref[2])
    a0 = _norm3(*a_ref)
    d0 = _norm3(*diff)

    assert (d0 / max(1e-30, a0)) > 1e-12, "Perturbation too small / not detected"


def test_serial_parallel_consistency_high_degree(sh, constants, rng):
    """
    For a higher degree model, the deterministic serial kernel and the explicit parallel
    kernel should match closely (within loose tolerance due to parallel reductions / fastmath).
    """
    R_ref, _GM = constants
    deg_hi = 120
    tol_rel_loose = 1e-10

    extra = (
        (True, 2, 0, -2.0e-4),   # C20
        (True, 3, 1, 1.0e-6),    # C31
        (False, 4, 2, -2.0e-6),  # S42
    )
    model = _make_model(sh, deg_hi, constants, c20=0.0, extra_terms=extra)

    alt = 300e3
    r = R_ref + alt
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    x, y, z = float(r * v[0]), float(r * v[1]), float(r * v[2])

    a_s = _call_fixed_numba(sh, model, x, y, z, deg_hi)
    a_p = _call_fixed_parallel(sh, model, x, y, z, deg_hi)

    assert _is_finite3(*a_s) and _is_finite3(*a_p)

    e = _rel_err_vec(a_p, a_s)
    assert e < tol_rel_loose, f"serial/parallel mismatch: {e:.3e}"


def test_pole_robustness(sh, constants):
    """
    Near the poles, lambda is ill-defined (rxy ~ 0). The kernel must stay finite.
    """
    R_ref, _GM = constants
    deg = 10
    model_j2 = _make_model(sh, deg, constants, c20=-2.0e-4)

    x, y, z = 1e-6, -1e-6, R_ref + 200e3
    a = _call_fixed_numba(sh, model_j2, x, y, z, deg)

    assert _is_finite3(*a), "Non-finite acceleration near pole"


def test_adaptive_blend_boundaries_and_continuity(sh, constants):
    """
    Adaptive blending should:
    (A) Match the far/near fixed-degree accelerations exactly at alt_far/alt_near
    (B) Be continuous at "degree bracket" switches.
    """
    if not hasattr(sh, "sh_accel_adaptive_blend_numba"):
        pytest.skip("Module does not expose sh_accel_adaptive_blend_numba.")

    R_ref, GM = constants

    alt_far = 1000e3
    alt_near = 100e3
    deg_far = 2
    deg_near = 10
    step = 2

    model = _make_model(sh, deg_near, constants, c20=-2.0e-4)
    f = _model_fields(model)
    ws = f["ws"]

    v = np.array([1.0, 0.2, -0.1], dtype=np.float64)
    v /= np.linalg.norm(v)

    def accel_adapt_at_alt(alt_m: float):
        rr = R_ref + float(alt_m)
        x = float(rr * v[0])
        y = float(rr * v[1])
        z = float(rr * v[2])

        return sh.sh_accel_adaptive_blend_numba(
            x, y, z,
            int(deg_far), int(deg_near),
            float(alt_far), float(alt_near),
            int(step),
            float(f["r_ref"]), float(f["mu"]),
            f["C"], f["S"],
            f["diag"], f["subdiag"],
            f["A"], f["B"], f["scale_m"],
            ws.P, ws.dP, ws.cos_m, ws.sin_m,
        )

    def accel_fixed_deg(alt_m: float, deg_use: int):
        rr = R_ref + float(alt_m)
        x = float(rr * v[0])
        y = float(rr * v[1])
        z = float(rr * v[2])
        return _call_fixed_numba(sh, model, x, y, z, int(deg_use))

    def bracket_degrees_at_alt(alt_m: float) -> Tuple[int, int]:
        denom = (alt_far - alt_near)
        t = (alt_far - alt_m) / denom
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        s = t * t * (3.0 - 2.0 * t)
        deg_des = deg_far + s * (deg_near - deg_far)

        steps_from_base = int(math.floor((deg_des - deg_far) / step))
        deg_lo = deg_far + steps_from_base * step
        deg_hi = deg_lo + step

        if deg_lo > deg_near:
            deg_lo = deg_near
        if deg_hi > deg_near:
            deg_hi = deg_near
        if deg_hi < deg_lo:
            deg_hi = deg_lo
        return int(deg_lo), int(deg_hi)

    # (A) Boundary matching
    a_b_far = accel_adapt_at_alt(alt_far)
    a_d_far = accel_fixed_deg(alt_far, deg_far)
    assert _rel_err_vec(a_b_far, a_d_far) < 1e-12

    a_b_near = accel_adapt_at_alt(alt_near)
    a_d_near = accel_fixed_deg(alt_near, deg_near)
    assert _rel_err_vec(a_b_near, a_d_near) < 1e-12

    # (B) Continuity at bracket switches
    alts = np.linspace(alt_near, alt_far, 2001)
    prev_pair = bracket_degrees_at_alt(float(alts[0]))
    switch_alts = []

    for a in alts[1:]:
        pair = bracket_degrees_at_alt(float(a))
        if pair != prev_pair:
            switch_alts.append(float(a))
            prev_pair = pair

    eps = 1e-3  # 1 mm altitude perturbation
    max_jump = 0.0

    for a0 in switch_alts:
        a_minus = max(alt_near, a0 - eps)
        a_plus = min(alt_far, a0 + eps)

        am = accel_adapt_at_alt(a_minus)
        ap = accel_adapt_at_alt(a_plus)

        assert _is_finite3(*am) and _is_finite3(*ap)

        jump = _norm3(ap[0] - am[0], ap[1] - am[1], ap[2] - am[2])
        max_jump = max(max_jump, jump)

    assert max_jump < 1e-8, f"Discontinuity detected (max |Δa|={max_jump:.3e})"


if __name__ == "__main__":
    print("This is a pytest test module. Run it with:")
    print("  python -m pytest -vv -rA --durations=10 tests/test_spherical_harmonics.py")
    raise SystemExit(0)
