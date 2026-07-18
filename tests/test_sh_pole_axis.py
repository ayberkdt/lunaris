"""Polar-axis limit of the spherical-harmonic acceleration kernels.

On the rotation axis (x = y = 0) the longitude is undefined and the transverse
acceleration is the removable-singularity limit of the m=1 sector:

    a_x = sum_n (mu/r^2) (R/r)^n * sqrt(n(n+1)(2n+1)/2) * sigma_n * C_n1
    a_y = ... same with S_n1,   sigma_n = 1 (north) / (-1)^(n+1) (south)

Before the fix both kernels returned a wrong (and mutually inconsistent) limit
there: the numba kernel dropped all m>=1 terms (zero transverse), the torch
evaluator kept a lambda=0 half-term. These tests pin the analytic limit with an
independent implementation, verify it against the off-axis approach limit, and
lock numba/torch backend parity on the axis.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.physics.spherical_harmonics import GravityModel

R_REF = 1.738e6
MU = 4.9028001e12
DEGREE = 4


def _synthetic_model() -> GravityModel:
    """Small fully-normalized model exercising m=0, m=1 and m>=2 sectors."""
    n = DEGREE + 1
    c = np.zeros((n, n))
    s = np.zeros((n, n))
    c[2, 0] = -9.0e-5   # J2-like zonal: axial only, no transverse at the pole
    c[2, 1] = 3.0e-5
    s[3, 1] = -2.0e-5
    c[4, 1] = 1.0e-5
    s[4, 1] = 4.0e-6
    c[2, 2] = 2.4e-5    # sectoral terms: must not contribute on the axis
    s[3, 3] = -1.1e-5
    c[4, 3] = 7.0e-6
    return GravityModel.from_arrays(
        degree_max=DEGREE, r_ref=R_REF, mu=MU, c_coeffs_full=c, s_coeffs_full=s
    )


def _axis_transverse_expected(model: GravityModel, r: float, *, north: bool) -> tuple[float, float]:
    """Independent implementation of the closed-form m=1 axis limit."""
    ax = ay = 0.0
    for n in range(1, model.max_degree + 1):
        lam = math.sqrt(0.5 * n * (n + 1.0) * (2.0 * n + 1.0))
        sigma = 1.0 if north or (n % 2 == 1) else -1.0
        k = (model.mu / r**2) * (model.r_ref / r) ** n * lam * sigma
        ax += k * float(model.Cnm[n, 1])
        ay += k * float(model.Snm[n, 1])
    return ax, ay


@pytest.fixture(scope="module")
def model() -> GravityModel:
    return _synthetic_model()


@pytest.mark.parametrize("north", [True, False])
def test_axis_matches_closed_form(model, north):
    r = R_REF + 100e3
    z = r if north else -r
    a = model.accel_fixed(np.array([0.0, 0.0, z]))
    ax_exp, ay_exp = _axis_transverse_expected(model, r, north=north)

    scale = math.hypot(ax_exp, ay_exp)
    assert scale > 1e-7  # the synthetic model must actually exercise the limit
    assert a[0] == pytest.approx(ax_exp, rel=1e-13, abs=1e-20)
    assert a[1] == pytest.approx(ay_exp, rel=1e-13, abs=1e-20)
    # Axial component is the m=0 sector; monopole must dominate and be finite.
    assert math.isfinite(a[2]) and abs(a[2]) > 0.9 * MU / r**2


@pytest.mark.parametrize("lam_lon", [0.0, 1.1, 2.7])
def test_axis_value_is_approach_limit(model, lam_lon):
    """The perturbation acceleration approaching the pole converges to the axis
    value, independently of the longitude of approach (removable singularity).

    The comparison is on the perturbation vector a - a_pointmass: the full
    Cartesian vector picks up the O(theta) geometric tilt of the (dominant)
    monopole when approaching along a meridian, which would mask the m=1 limit
    being tested.
    """
    r = R_REF + 100e3

    def perturbation(pos: np.ndarray) -> np.ndarray:
        rr = float(np.linalg.norm(pos))
        return model.accel_fixed(pos) + MU / rr**3 * pos

    p_axis = perturbation(np.array([0.0, 0.0, r]))

    errs = []
    for theta in (1e-4, 1e-5, 1e-6):
        pos = np.array(
            [
                r * math.sin(theta) * math.cos(lam_lon),
                r * math.sin(theta) * math.sin(lam_lon),
                r * math.cos(theta),
            ]
        )
        errs.append(float(np.linalg.norm(perturbation(pos) - p_axis)))

    scale = math.hypot(p_axis[0], p_axis[1])
    # Errors shrink linearly with colatitude; the tightest approach agrees to
    # ~1e-5 relative (residual is the O(theta) tilt of the J2 axial term).
    assert errs[2] < 0.15 * errs[1] < 0.03 * errs[0]
    assert errs[2] < 1e-5 * scale


def test_continuity_across_pole_safe_cutoff(model):
    """The analytic branch (rho below the pole-safe cutoff) must join the
    regular evaluation path smoothly. The residual difference is the known
    EPS_1E24 softening of the longitudinal projection (~1e-8 relative at
    rho = 1e-8 m), not a physical discontinuity."""
    r = R_REF + 100e3
    a_axis = model.accel_fixed(np.array([0.0, 0.0, r]))
    rho = 1e-8  # metres: regular path
    a_near = model.accel_fixed(np.array([rho, 0.0, math.sqrt(r * r - rho * rho)]))
    np.testing.assert_allclose(a_near, a_axis, rtol=1e-7)


def test_south_pole_parity_structure(model):
    """sigma_n = (-1)^(n+1): odd-degree m=1 terms keep sign at the south pole,
    even-degree terms flip. Check with single-coefficient models."""
    r = R_REF + 100e3
    for n, attr, expect_flip in ((3, "s", False), (2, "c", True), (4, "c", True)):
        c = np.zeros((DEGREE + 1, DEGREE + 1))
        s = np.zeros((DEGREE + 1, DEGREE + 1))
        (c if attr == "c" else s)[n, 1] = 1.0e-5
        single = GravityModel.from_arrays(
            degree_max=DEGREE, r_ref=R_REF, mu=MU, c_coeffs_full=c, s_coeffs_full=s
        )
        comp = 0 if attr == "c" else 1
        a_n = single.accel_fixed(np.array([0.0, 0.0, r]))[comp]
        a_s = single.accel_fixed(np.array([0.0, 0.0, -r]))[comp]
        assert abs(a_n) > 0.0
        if expect_flip:
            assert a_s == pytest.approx(-a_n, rel=1e-13)
        else:
            assert a_s == pytest.approx(a_n, rel=1e-13)


def test_torch_matches_numba_on_axis_and_off_axis(model):
    torch = pytest.importorskip("torch")
    from lunaris.physics.torch_spherical_harmonics import TorchSHGravityEvaluator

    evaluator = TorchSHGravityEvaluator(
        model, degree=DEGREE, device="cpu", dtype=torch.float64
    )
    r = R_REF + 100e3
    points = np.array(
        [
            [0.0, 0.0, r],                       # north axis
            [0.0, 0.0, -r],                      # south axis
            [r * 0.6, r * 0.3, r * 0.74],        # generic off-axis control
        ]
    )
    a_torch = evaluator.acceleration(torch.tensor(points, dtype=torch.float64)).numpy()
    a_numba = np.array([model.accel_fixed(p) for p in points])
    np.testing.assert_allclose(a_torch, a_numba, rtol=1e-12, atol=1e-16)
