"""Model-free frame-handling invariants for the SH gravity engine.

These need no external data or kernels: they assert mathematical identities the
body-fixed evaluation must satisfy, catching position->frame transposition or
sign defects that a single-point value check can miss. They are the
kernel-independent core of "does the engine handle the body frame correctly",
complementary to the (kernel-gated) rotating-frame trajectory reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.physics.spherical_harmonics import GravityModel

MU = float(MU_MOON)
R = float(R_MOON)


def _zonal_model(degree: int = 6) -> GravityModel:
    """A purely zonal (m=0) field is axisymmetric about the z-axis."""
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    c[2, 0] = -9.08e-5   # J2
    c[3, 0] = 1.0e-5     # J3
    c[4, 0] = -8.0e-6    # J4
    return GravityModel.from_arrays(degree, R, MU, c, s)


def _rot_z(alpha: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])


def _points() -> list[np.ndarray]:
    def unit(v):
        a = np.asarray(v, float)
        return a / np.linalg.norm(a)
    return [
        unit((1.0, 0.0, 0.2)) * (R + 200e3),
        unit((0.6, 0.4, 0.7)) * (R + 800e3),
        unit((0.2, -0.9, 0.3)) * (R + 100e3),
    ]


@pytest.mark.parametrize("alpha_deg", [17.0, 90.0, 213.0, -47.0])
def test_zonal_field_is_z_rotation_equivariant(alpha_deg: float) -> None:
    """For an axisymmetric field, a(R_z r) == R_z a(r) (identity, not a fit).

    A longitude/x-y transposition or a sign error in the body-fixed evaluation
    breaks this even though the field has no tesseral terms.
    """
    model = _zonal_model()
    Rz = _rot_z(math.radians(alpha_deg))
    for p in _points():
        a_p = model.accel_fixed(p)
        a_rot = model.accel_fixed(Rz @ p)
        expected = Rz @ a_p
        err = np.linalg.norm(a_rot - expected)
        scale = np.linalg.norm(a_p)
        assert err / scale < 1e-10, f"equivariance broken: rel {err / scale:.2e}"


def test_zonal_field_has_no_longitudinal_acceleration() -> None:
    """Axisymmetry => the east-west (longitudinal) acceleration component is zero."""
    model = _zonal_model()
    for p in _points():
        a = model.accel_fixed(p)
        lam = math.atan2(p[1], p[0])
        e_lon = np.array([-math.sin(lam), math.cos(lam), 0.0])  # east unit vector
        a_lon = abs(float(np.dot(a, e_lon)))
        assert a_lon / np.linalg.norm(a) < 1e-10, f"spurious longitudinal accel {a_lon:.2e}"


def test_point_mass_is_purely_radial_and_inward() -> None:
    """Monopole sanity: a = -mu/r^2 * r_hat (inward, no transverse component)."""
    model = GravityModel.from_arrays(0, R, MU, np.array([[1.0]]), np.array([[0.0]]))
    for p in _points():
        a = model.accel_fixed(p)
        r = float(np.linalg.norm(p))
        expected = -MU / r**2 * (p / r)
        assert np.linalg.norm(a - expected) / np.linalg.norm(expected) < 1e-12
