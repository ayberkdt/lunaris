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
    export LUNAR_SH_MODULE=lunaris.physics.spherical_harmonics
    pytest -q
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from lunaris.validation.gravity_reference.independent_field_oracle import (
    acceleration as independent_oracle_acceleration,
)


# -----------------------------------------------------------------------------
# Import helper
# -----------------------------------------------------------------------------
def _repo_root_from_this_file() -> Path:
    """
    Resolve repository root from this test file location:
        repo_root/tests/test_spherical_harmonics.py  -> repo_root
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "tests":
            return p.parent
    return here.parent


def _import_sh_module():
    """
    Import the canonical spherical harmonics module.

    If the env var ``LUNAR_SH_MODULE`` is set, import that exact module path
    instead (useful for benchmarking an alternative implementation).
    """
    env_name = os.environ.get("LUNAR_SH_MODULE", "").strip()
    if env_name:
        return importlib.import_module(env_name)

    try:
        return importlib.import_module("lunaris.physics.spherical_harmonics")
    except Exception as exc:
        # Path-based fallback for a non-installed source checkout.
        path = _repo_root_from_this_file() / "src" / "lunaris" / "physics" / "spherical_harmonics.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("spherical_harmonics_under_test", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                # dataclasses + string annotations need module in sys.modules before exec_module
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                return mod
        raise ImportError("Could not import lunaris.physics.spherical_harmonics.") from exc


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


def _rel_err_vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
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


def test_internal_sh_contract_docstrings_match_phase_and_pole_guards() -> None:
    """Prevent reviewer-found comment drift in the SH kernel contracts."""
    sh = _import_sh_module()

    normalize = getattr(sh._apply_legendre_normalization, "py_func", sh._apply_legendre_normalization)
    normalize_doc = normalize.__doc__ or ""
    assert "No Condon-Shortley phase is applied" in normalize_doc
    assert "and Condon" not in normalize_doc

    pole_guard = getattr(sh._compute_pole_safe_inv_rho_sq, "py_func", sh._compute_pole_safe_inv_rho_sq)
    pole_doc = pole_guard.__doc__ or ""
    assert "absolute ``EPS_1E24`` rho^2 floor" in pole_doc
    assert "no radial relative softening is applied" in pole_doc


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


def _position_from_lat_lon_alt(constants, lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    R_ref, _GM = constants
    r = float(R_ref) + float(alt_km) * 1000.0
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    return np.array(
        [
            r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat),
        ],
        dtype=np.float64,
    )


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


def test_cpu_sh_matches_independent_finite_difference_oracle_on_tesseral_field(sh, constants):
    """Pin the CPU SH acceleration against an independent U -> grad(U) oracle.

    This is deliberately not a zonal-only field: odd-order tesseral/sectoral
    terms exercise the no-Condon-Shortley phase convention that a C20/J2 smoke
    test cannot see.
    """
    R_ref, GM = constants
    degree = 8
    C = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    S = np.zeros_like(C)
    terms = (
        ("C", 2, 0, -9.0e-5),
        ("C", 2, 2, 1.3e-5),
        ("S", 2, 2, -1.1e-5),
        ("C", 3, 1, 2.2e-6),
        ("S", 3, 1, -1.7e-6),
        ("C", 4, 3, -8.0e-7),
        ("S", 5, 5, 5.0e-7),
        ("C", 7, 3, -2.0e-7),
        ("S", 8, 1, 1.5e-7),
    )
    for kind, n, m, value in terms:
        (C if kind == "C" else S)[n, m] = float(value)
    assert np.any(np.abs(C[3:, 1::2]) > 0.0) or np.any(np.abs(S[3:, 1::2]) > 0.0)

    model = sh.GravityModel.from_arrays(degree, R_ref, GM, C, S)
    sample_points = (
        (0.0, 0.0, 50.0),
        (0.0, 45.0, 100.0),
        (37.0, 120.0, 500.0),
        (-60.0, 200.0, 2000.0),
        (82.0, 15.0, 100.0),
    )

    rows = []
    for lat_deg, lon_deg, alt_km in sample_points:
        pos = _position_from_lat_lon_alt(constants, lat_deg, lon_deg, alt_km)
        analytic = np.asarray(model.accel_fixed(pos, degree=degree), dtype=np.float64)
        numerical = independent_oracle_acceleration(
            pos,
            mu_m3_s2=GM,
            reference_radius_m=R_ref,
            c_coeffs=C,
            s_coeffs=S,
            degree=degree,
            rel_step=3.0e-4,
        )
        rel_err = float(np.linalg.norm(analytic - numerical) / max(np.linalg.norm(analytic), 1e-30))
        abs_err = float(np.max(np.abs(analytic - numerical)))
        rows.append((lat_deg, lon_deg, alt_km, rel_err, abs_err))

    max_rel = max(row[3] for row in rows)
    assert max_rel < 1.0e-9, f"CPU SH vs independent oracle mismatch: {rows!r}"


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

    def bracket_degrees_at_alt(alt_m: float) -> tuple[int, int]:
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


def _py_func(f):
    return f.py_func if hasattr(f, "py_func") else f


def test_legendre_derivative_matches_finite_difference(sh):
    """
    Directly test the internal Legendre derivative table `dP` against a finite-difference derivative of `P`.

    This covers all valid (n, m) orders, not only m=0; odd-order tesseral terms
    depend on these derivatives and are exactly where phase/sign regressions hide.
    """
    max_degree = 40
    delta = 1e-7

    diag, subdiag, A, B, scale_m = sh.build_legendre_coeffs(max_degree)

    P0 = np.zeros((max_degree + 1, max_degree + 1))
    dP0 = np.zeros_like(P0)

    Pm = np.zeros_like(P0)
    dPm = np.zeros_like(P0)

    Pp = np.zeros_like(P0)
    dPp = np.zeros_like(P0)

    compute_stable_m = _py_func(sh._compute_stable_m_limit)
    compute_legendre = _py_func(sh._compute_legendre_polynomials_inplace)

    test_phis = [-0.9, -0.3, 0.3, 0.7]

    for phi in test_phis:
        for arr in (P0, dP0, Pm, dPm, Pp, dPp):
            arr.fill(0.0)

        # P, dP @ phi
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        stable_m0 = compute_stable_m(cos_phi, max_degree)
        compute_legendre(
            sin_phi, cos_phi, max_degree, max_degree, stable_m0,
            diag, subdiag, A, B, scale_m, P0, dP0
        )

        # P @ phi-delta
        sin_m = math.sin(phi - delta)
        cos_m = math.cos(phi - delta)
        stable_m_m = compute_stable_m(cos_m, max_degree)
        compute_legendre(
            sin_m, cos_m, max_degree, max_degree, stable_m_m,
            diag, subdiag, A, B, scale_m, Pm, dPm
        )

        # P @ phi+delta
        sin_p = math.sin(phi + delta)
        cos_p = math.cos(phi + delta)
        stable_m_p = compute_stable_m(cos_p, max_degree)
        compute_legendre(
            sin_p, cos_p, max_degree, max_degree, stable_m_p,
            diag, subdiag, A, B, scale_m, Pp, dPp
        )

        # Finite difference
        dP_fd = (Pp - Pm) / (2.0 * delta)

        valid_triangle = np.fromfunction(lambda n, m: m <= n, P0.shape, dtype=int)
        mask = valid_triangle & (np.abs(dP_fd) > 1e-9)

        if np.any(mask):
            np.testing.assert_allclose(
                dP0[mask],
                dP_fd[mask],
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"Legendre derivative mismatch at phi={phi}"
            )


# -----------------------------------------------------------------------------
# GravityModel high-level API: properties, factories, and the object-oriented
# acceleration methods (the lower-level kernels are tested above; these wrappers
# were previously uncovered).
# -----------------------------------------------------------------------------


def test_gravity_model_properties_expose_underlying_arrays(sh, constants):
    R_ref, GM = constants
    model = _make_model(sh, 3, constants, c20=-9.0e-5)
    assert model.degree_max == 3
    assert model.R_ref_m == R_ref
    assert model.GM_m3s2 == GM
    # Property aliases must return the backing kernel arrays unchanged.
    assert model.Cnm is model.c_coeffs
    assert model.Snm is model.s_coeffs
    assert model.diag is model.diag_coeffs
    assert model.subdiag is model.subdiag_coeffs
    assert model.A is model.a_coeffs
    assert model.B is model.b_coeffs
    assert model.scale_m is model.scale_m_table


def test_gravity_model_from_arrays_rejects_nonfinite_params(sh, constants):
    R_ref, GM = constants
    C = np.zeros((3, 3))
    S = np.zeros_like(C)
    with pytest.raises(ValueError):
        sh.GravityModel.from_arrays(2, math.inf, GM, C, S)
    with pytest.raises(ValueError):
        sh.GravityModel.from_arrays(2, R_ref, math.nan, C, S)


@pytest.mark.parametrize("bad_scalar", [0.0, -1.0])
def test_gravity_model_from_arrays_rejects_nonpositive_physical_scalars(
    sh, constants, bad_scalar
):
    R_ref, GM = constants
    C = np.zeros((3, 3))
    S = np.zeros_like(C)
    with pytest.raises(ValueError, match="finite and > 0"):
        sh.GravityModel.from_arrays(2, bad_scalar, GM, C, S)
    with pytest.raises(ValueError, match="finite and > 0"):
        sh.GravityModel.from_arrays(2, R_ref, bad_scalar, C, S)


@pytest.mark.parametrize("coefficient_name", ["C", "S"])
def test_gravity_model_from_arrays_rejects_nonfinite_coefficients(
    sh, constants, coefficient_name
):
    R_ref, GM = constants
    C = np.zeros((3, 3))
    S = np.zeros_like(C)
    target = C if coefficient_name == "C" else S
    target[2, 1] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        sh.GravityModel.from_arrays(2, R_ref, GM, C, S)


def test_gravity_model_accel_fixed_matches_point_mass_for_zero_coeffs(sh, constants):
    R_ref, GM = constants
    model = _make_model(sh, 2, constants)  # all coeffs zero -> point mass
    r = np.array([R_ref + 200_000.0, 0.0, 0.0])
    a = model.accel_fixed(r)
    assert a.shape == (3,)
    expected_mag = GM / (r[0] ** 2)
    assert np.linalg.norm(a) == pytest.approx(expected_mag, rel=1e-6)
    # acceleration points inward (toward the body)
    assert a[0] < 0.0
    # degree=0 explicitly and a user-supplied workspace exercise both branches
    ws = model.make_workspace()
    a0 = model.accel_fixed(r, degree=0, workspace=ws)
    assert np.linalg.norm(a0) == pytest.approx(expected_mag, rel=1e-6)


def test_compute_point_mass_acceleration_origin_guard_and_inverse_square(sh, constants):
    _, GM = constants
    # At/near the origin the monopole returns zero (singularity guard).
    assert sh.compute_point_mass_acceleration(0.0, 0.0, 0.0, GM) == (0.0, 0.0, 0.0)
    # Off-origin: magnitude = GM/r^2, pointing inward.
    r = 2.0e6
    ax, ay, az = sh.compute_point_mass_acceleration(r, 0.0, 0.0, GM)
    assert (ax**2 + ay**2 + az**2) ** 0.5 == pytest.approx(GM / r**2, rel=1e-12)
    assert ax < 0.0


def test_slice_gravity_model_validation(sh):
    good = np.zeros((4, 4))
    with pytest.raises(ValueError):
        sh.slice_gravity_model(good, good, -1)              # negative degree
    with pytest.raises(ValueError):
        sh.slice_gravity_model(np.zeros(4), good, 2)        # not 2D
    with pytest.raises(ValueError):
        sh.slice_gravity_model(good, np.zeros((4, 3)), 2)   # shape mismatch
    with pytest.raises(ValueError):
        sh.slice_gravity_model(good, good, 10)              # not enough rows
    # Rectangular input is zero-padded in the order dimension to a square block.
    C = np.ones((3, 2))
    cS, sS = sh.slice_gravity_model(C, C, 2)
    assert cS.shape == (3, 3) and sS.shape == (3, 3)
    assert cS[0, 2] == 0.0  # padded column


def test_gravity_model_accel_adaptive_runs_and_is_finite(sh, constants):
    R_ref, GM = constants
    model = _make_model(sh, 4, constants, c20=-9.0e-5)
    r = np.array([0.0, R_ref + 150_000.0, 0.0])
    a = model.accel_adaptive(
        r, degree_far=2, degree_near=4, alt_far=300_000.0, alt_near=50_000.0, degree_step=2
    )
    assert a.shape == (3,)
    assert np.all(np.isfinite(a))


if __name__ == "__main__":
    print("This is a pytest test module. Run it with:")
    print("  python -m pytest -vv -rA --durations=10 tests/test_spherical_harmonics.py")
    raise SystemExit(0)
