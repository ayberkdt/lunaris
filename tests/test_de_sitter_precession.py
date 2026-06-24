"""Physical validation of the de Sitter (geodetic) precession term.

The kernel-vs-its-own-formula test in ``test_relativity_effects.py`` proves
internal consistency. This module instead pins the *physics*:

1. **Magnitude** - for the real Moon/Sun geometry the geodetic rate must be the
   canonical ~19.2 mas/yr (orbit-height independent), which validates the
   ``(3/2) mu/(c^2 R^3) (R x V)`` factor.
2. **Direction** - integrating a lunar orbit with the de Sitter term in isolation,
   the orbit plane must precess **prograde** (about +Omega), i.e. ``a = +2 Omega x v``.
   A retrograde result (``-2 Omega x v``) fails this test.

The direction check is memory-independent: it does not rely on any quoted form of
the IERS equation, only on the established fact that geodetic precession is
prograde.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.constants import C_LIGHT, MU_MOON, MU_SUN
from lunaris.physics.relativity_effects import C_SQ, _de_sitter_components

_RAD_TO_MAS = 180.0 / np.pi * 3600.0 * 1000.0
_SEC_PER_YEAR = 365.25 * 86400.0


def _omega_geo(body: np.ndarray, body_v: np.ndarray, mu: float) -> np.ndarray:
    """Gyroscope geodetic precession vector from the formula the kernel uses."""
    r_bm = -body          # external body -> Moon
    v_bm = -body_v
    return 1.5 * mu * np.cross(r_bm, v_bm) / (C_SQ * np.linalg.norm(r_bm) ** 3)


def test_de_sitter_rate_matches_canonical_19_mas_per_year() -> None:
    # Sun at 1 AU on +x; Moon's heliocentric speed ~29.78 km/s (Earth-Moon mean).
    body = np.array([1.495978707e11, 0.0, 0.0])   # Moon -> Sun
    body_v = np.array([0.0, -29_780.0, 0.0])      # d/dt(Moon -> Sun)
    mu = float(MU_SUN)

    omega = _omega_geo(body, body_v, mu)
    rate_mas_per_yr = np.linalg.norm(omega) * _RAD_TO_MAS * _SEC_PER_YEAR
    # Canonical de Sitter precession of the Earth-Moon system: ~19.2 mas/yr.
    assert 18.5 < rate_mas_per_yr < 20.0

    # Tie the rate to the kernel: for v perpendicular to Omega, |a| = 2|Omega||v|.
    v = np.array([0.0, 1600.0, 0.0])  # in xy-plane; Omega is along z here
    assert abs(float(omega[0])) < 1e-30 and abs(float(omega[1])) < 1e-30  # Omega || z
    a = np.array(_de_sitter_components(
        float(v[0]), float(v[1]), float(v[2]),
        float(body[0]), float(body[1]), float(body[2]),
        float(body_v[0]), float(body_v[1]), float(body_v[2]),
        mu,
    ))
    assert np.linalg.norm(a) == pytest.approx(2.0 * np.linalg.norm(omega) * np.linalg.norm(v), rel=1e-9)


def test_de_sitter_precesses_the_orbit_prograde() -> None:
    """Integrate a lunar orbit with ONLY point-mass gravity + the de Sitter term
    (Omega amplified for measurability) and confirm the orbit normal precesses in
    the +Omega sense at rate |Omega|."""
    # Perturber geometry giving Omega along +x (amplified via a synthetic mu_body).
    R, V, mu_body = 1.0e8, 3.0e4, 1.2e22
    body = np.array([0.0, -R, 0.0])
    body_v = np.array([0.0, 0.0, -V])
    omega = _omega_geo(body, body_v, mu_body)
    omega_mag = float(np.linalg.norm(omega))
    assert omega[0] > 0.0 and abs(omega[1]) < 1e-30 and abs(omega[2]) < 1e-30  # +x

    # Circular lunar orbit in the xy-plane (L = +z).
    a0 = 2.0e6
    vc = np.sqrt(MU_MOON / a0)
    y = np.array([a0, 0.0, 0.0, 0.0, vc, 0.0])

    def accel(state: np.ndarray) -> np.ndarray:
        r = state[:3]
        v = state[3:]
        ag = -MU_MOON * r / np.linalg.norm(r) ** 3
        dsx, dsy, dsz = _de_sitter_components(
            v[0], v[1], v[2],
            body[0], body[1], body[2],
            body_v[0], body_v[1], body_v[2],
            mu_body,
        )
        return np.concatenate([v, ag + np.array([dsx, dsy, dsz])])

    dt = 4.0
    period = 2.0 * np.pi * a0 / vc
    n = int(12 * period / dt)
    L0 = np.cross(y[:3], y[3:])
    for _ in range(n):
        k1 = accel(y)
        k2 = accel(y + 0.5 * dt * k1)
        k3 = accel(y + 0.5 * dt * k2)
        k4 = accel(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    Lf = np.cross(y[:3], y[3:])

    total_t = n * dt
    dL = (Lf - L0) / total_t
    # Omega_prec x L0 = dL; with L0 ~ +z and Omega ~ +x, dL ~ -omega_p |L| y.
    omega_p = -dL[1] / np.linalg.norm(L0)
    # Prograde and correct magnitude (the retrograde sign bug gives -omega_mag).
    assert omega_p > 0.0
    assert omega_p == pytest.approx(omega_mag, rel=0.05)
