"""SH potential path (``potential_accel_fixed``) against the propagation kernel near the axis.

Oracle: ``GravityModel.accel_fixed`` (the propagation kernel), which takes dP/dphi from the
same-degree order recurrence and has no division by cos(phi). The potential path is the
ST-LRPS dataset generator, so a near-axis error there is a label error.

History: the potential kernel used to take dP/dphi from
(-n sin(phi) P + (n+m) k P_prev) / cos(phi), which cancels catastrophically near the polar
axis (measured on a degree-30 field: 6.7e-5 m/s^2, 11 % of the transverse field, at
rho = 2e-8 m; 1.3e-6 m/s^2 at 1e-6 m; 1.3e-9 m/s^2 at 1 mm; growing like 1/rho). The axis
limit was used only for cos(lat) <= 1e-14. It now uses the propagation kernel's recurrence.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.physics.spherical_harmonics import GravityModel

MU = 4.9028e12  # lunar GM [m^3/s^2]; any positive value works, the checks are linear in it
R_REF = 1738.0e3  # reference radius [m]
_Z_AXIS = 1.9e6  # height on the axis [m], 1.09 reference radii (Moon-fixed frame)


def _random_field(degree: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Fully normalised coefficients with a Kaula-like 1e-4 / n^2 amplitude and C00 = 1."""
    rng = np.random.default_rng(seed)
    c = np.zeros((degree + 1, degree + 1))
    s = np.zeros((degree + 1, degree + 1))
    c[0, 0] = 1.0
    for n in range(1, degree + 1):
        for m in range(n + 1):
            c[n, m] = rng.normal() * 1e-4 / n**2
            if m > 0:
                s[n, m] = rng.normal() * 1e-4 / n**2
    return c, s


def _potential_vs_kernel(degree: int, rho: float) -> float:
    """Relative difference |a_potential - a_kernel| / |a_kernel| at distance ``rho`` from the axis."""
    c, s = _random_field(degree)
    model = GravityModel.from_arrays(degree, R_REF, MU, c, s)
    point = np.array([0.6 * rho, 0.8 * rho, _Z_AXIS])
    reference = model.accel_fixed(point)
    _, via_potential = model.potential_accel_fixed(point)
    return float(np.linalg.norm(via_potential - reference) / np.linalg.norm(reference))


@pytest.mark.parametrize("degree", [12, 200])
@pytest.mark.parametrize("rho", [1e-12, 1e-9, 2e-8, 1e-7, 1e-6, 1e-3, 1.0, 1e3], ids=lambda v: f"rho={v:g}")
def test_potential_path_matches_the_propagation_kernel_near_the_axis(degree: int, rho: float) -> None:
    # 1e-12 relative to |a_kernel| [m/s^2 / m/s^2]: the two paths agree to ~1e-15 at every rho
    # (both are float64 sums of the same terms), so 1e-12 leaves three decades of margin yet
    # sits seven decades below the old defect (~1e-5 relative at 2e-8 m).
    assert _potential_vs_kernel(degree, rho) <= 1e-12


@pytest.mark.parametrize("degree_min", [0, 1, 3])
@pytest.mark.parametrize("rho", [1e-9, 2e-8, 1e-7, 1e-6, 1e-3, 1.0, 1e3], ids=lambda v: f"rho={v:g}")
def test_potential_path_degree_window_matches_the_kernel_near_the_axis(degree_min: int, rho: float) -> None:
    """The residual band (degree_min, N] equals kernel(N) - kernel(degree_min)."""
    degree = 12
    c, s = _random_field(degree)
    model = GravityModel.from_arrays(degree, R_REF, MU, c, s)
    point = np.array([0.6 * rho, 0.8 * rho, _Z_AXIS])
    window = model.accel_fixed(point) - model.accel_fixed(point, degree=degree_min)
    _, via_potential = model.potential_accel_fixed(point, degree_min=degree_min)
    # The window is a difference of two ~1.3 m/s^2 sums (monopole), so the reference carries
    # ~2e-16 m/s^2 of rounding, ~1e-12 of the window itself; 1e-10 of |window| leaves a factor
    # 100 and sits below the old defect from rho = 1 m upward.
    assert np.linalg.norm(via_potential - window) <= 1e-10 * np.linalg.norm(window)


def test_potential_path_acceleration_is_the_gradient_of_its_potential() -> None:
    """a = +grad(V) by a 4th-order central difference of V, including 1 km from the axis.

    The potential sum does not read the latitude derivative, so V is an oracle for the
    acceleration that shares only the ALF values with it.
    """
    degree = 20
    c, s = _random_field(degree)
    model = GravityModel.from_arrays(degree, R_REF, MU, c, s)
    rng = np.random.default_rng(3)
    points = [rng.normal(size=3) for _ in range(8)]
    points = [p / np.linalg.norm(p) * 1.9e6 for p in points]
    points += [np.array([600.0, 800.0, _Z_AXIS]), np.array([0.0, 1.0e3, -_Z_AXIS])]
    h = 5.0  # step [m], body-fixed Cartesian
    for p in points:
        _, a = model.potential_accel_fixed(p, degree_min=1)
        fd = np.zeros(3)
        for axis in range(3):
            e = np.zeros(3)
            e[axis] = h
            v = [model.potential_accel_fixed(p + k * e, degree_min=1)[0] for k in (-2, -1, 1, 2)]
            fd[axis] = (v[0] - 8.0 * v[1] + 8.0 * v[2] - v[3]) / (12.0 * h)
        # 1e-7 relative to |a| [m/s^2] (degree 2..20 residual band): the difference quotient
        # loses ~1e-16 * |V| / (h |a|) ~ 1e-10 to cancellation (measured 2.7e-10 at degree 30);
        # the old near-axis defect was ~1e-1 of the transverse field at 2e-8 m.
        assert np.linalg.norm(a - fd) <= 1e-7 * np.linalg.norm(a)
