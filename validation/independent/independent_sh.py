"""Independent spherical-harmonic gravity reference (scipy + numerical gradient).

The Lunaris production evaluator
(:mod:`lunaris.physics.spherical_harmonics`) computes the *analytic*
acceleration with a hand-written, Kahan-compensated, pole-safe Numba recurrence
for the fully-normalized associated Legendre functions (ALFs). To validate it
*independently* this module:

1. evaluates the geopotential ``U`` using a different ALF source — a
   self-contained classical associated-Legendre recurrence (NOT the optimized
   fully-normalized Numba recurrence under test) — and an explicitly written
   4π-normalization factor, and
2. obtains the acceleration as the **numerical gradient** of ``U``
   (a high-order central-difference stencil), never the analytic recurrence.

Because the only thing shared with the code under test is the *mathematical
convention* (4π-normalized real harmonics, body-fixed frame, ``a = +∇U``), an
implementation bug in the Lunaris recurrence, its derivative terms, or its
summation order shows up as a cross-check failure here.

Conventions (verified against closed-form anchors in the tests)
---------------------------------------------------------------
* Frame: body-fixed Cartesian ``(x, y, z)``; output acceleration in the same
  frame. Upstream rotations are out of scope (identical to the Lunaris module).
* Latitude is geocentric: ``sin φ = z / r``. Longitude ``λ = atan2(y, x)``.
* ALFs are fully 4π-normalized including the Condon–Shortley phase, matching the
  Lunaris ``scale_m`` table. The recurrence here bakes the ``(-1)^m`` phase into
  the sectoral term (``P_1^1 < 0``), so no extra phase factor is applied — see
  :func:`normalized_alf_column`. This is pinned by the cross-check tests against
  the analytic Lunaris accelerations and the J2/(2,2) closed forms.
* Potential:
  ``U = (μ/r) [ 1 + Σ_{n≥2} (R/r)^n Σ_{m} P̄_{nm}(sinφ)(C̄_{nm} cos mλ + S̄_{nm} sin mλ) ]``
  and gravitational acceleration ``a = +∇U`` (point-mass limit
  ``U = μ/r → a = -μ r/r³``). The leading ``1`` is the structural monopole; the
  perturbation sum starts at degree 2, mirroring the Lunaris kernel (which adds
  ``-μ/r²`` unconditionally and ignores the stored degree-0/degree-1
  coefficients). See :func:`geopotential`.

Units are SI throughout (metres, m³/s², m/s²).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Any

import numpy as np

__all__ = [
    "normalized_alf_column",
    "geopotential",
    "acceleration",
    "GravityCrossCheckResult",
    "crosscheck_gravity_model",
]

def _normalization_factor(n: int, m: int) -> float:
    """4π (geodesy) full-normalization factor for degree ``n`` order ``m``.

    ``N_nm = sqrt((2 - δ_{m0}) (2n+1) (n-m)! / (n+m)!)`` computed via log-gamma
    so it stays finite at high degree.
    """
    delta = 1.0 if m == 0 else 0.0
    log_fac = lgamma(n - m + 1) - lgamma(n + m + 1)
    return float(np.sqrt((2.0 - delta) * (2.0 * n + 1.0) * np.exp(log_fac)))


def _associated_legendre_all(n_max: int, t: float) -> np.ndarray:
    """Un-normalized associated Legendre functions ``P_n^m(t)`` as ``P[n, m]``.

    A self-contained classical recurrence (deliberately independent of both the
    Lunaris fully-normalized recurrence and any ``scipy.special`` helper, which
    keeps this reference version-proof). The Condon–Shortley ``(-1)^m`` phase is
    folded into the sectoral term, matching the Lunaris convention. Valid for the
    moderate degrees used in cross-checks (``|t| <= 1``).
    """
    p = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    p[0, 0] = 1.0
    somx2 = float(np.sqrt(max(0.0, 1.0 - t * t)))  # sin(colatitude)
    # Sectoral terms P_m^m = (-1)^m (2m-1)!! (1 - t^2)^(m/2).
    for m in range(1, n_max + 1):
        p[m, m] = -(2 * m - 1) * somx2 * p[m - 1, m - 1]
    # First off-diagonal P_{m+1}^m = t (2m+1) P_m^m.
    for m in range(n_max):
        p[m + 1, m] = t * (2 * m + 1) * p[m, m]
    # Vertical recurrence P_n^m = [(2n-1) t P_{n-1}^m - (n+m-1) P_{n-2}^m] / (n-m).
    for m in range(n_max + 1):
        for n in range(m + 2, n_max + 1):
            p[n, m] = ((2 * n - 1) * t * p[n - 1, m] - (n + m - 1) * p[n - 2, m]) / (n - m)
    return p


def normalized_alf_column(n_max: int, sin_phi: float) -> np.ndarray:
    """Return the 4π-normalized ALF matrix ``P̄[n, m]`` for ``n,m = 0..n_max``.

    Built from the self-contained :func:`_associated_legendre_all` recurrence
    (an independent ALF source, version-proof — no ``scipy.special`` dependency),
    then scaled by :func:`_normalization_factor`. The recurrence already carries
    the Condon–Shortley ``(-1)^m`` phase that matches the Lunaris convention, so
    NO extra phase is applied here. Entries with ``m > n`` are zero.
    """
    n_max = int(n_max)
    if n_max < 0:
        raise ValueError(f"n_max must be >= 0, got {n_max}")
    p_nm = _associated_legendre_all(n_max, float(sin_phi))
    p_bar = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    for n in range(n_max + 1):
        for m in range(n + 1):
            p_bar[n, m] = _normalization_factor(n, m) * p_nm[n, m]
    return p_bar


def geopotential(
    position: np.ndarray,
    *,
    mu: float,
    r_ref: float,
    c_coeffs: np.ndarray,
    s_coeffs: np.ndarray,
    degree: int | None = None,
) -> float:
    """Independent body-fixed geopotential ``U`` at a Cartesian position.

    ``c_coeffs``/``s_coeffs`` are ``(N+1, N+1)`` fully-normalized coefficient
    matrices in the same convention the Lunaris :class:`GravityModel` stores.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = float(np.sqrt(x * x + y * y + z * z))
    if r <= 0.0:
        raise ValueError("position must be non-zero")

    c = np.asarray(c_coeffs, dtype=np.float64)
    s = np.asarray(s_coeffs, dtype=np.float64)
    n_avail = min(c.shape[0], s.shape[0]) - 1
    n_max = n_avail if degree is None else min(int(degree), n_avail)

    sin_phi = z / r
    lam = float(np.arctan2(y, x))
    p_bar = normalized_alf_column(n_max, sin_phi)

    cos_m = np.cos(np.arange(n_max + 1) * lam)
    sin_m = np.sin(np.arange(n_max + 1) * lam)
    ratio = r_ref / r

    # The central monopole (μ/r) is structural and ALWAYS present, exactly like
    # the Lunaris kernel (``dv_dr = -μ/r²`` regardless of the stored C₀₀). Real
    # gravity files commonly store C₀₀ = 0, expecting the implementation to
    # supply the monopole. The Lunaris perturbation sum starts at degree 2
    # (degree-0 and degree-1 coefficients are ignored — degree 1 vanishes in a
    # centre-of-mass frame), so this reference mirrors that to compare like with
    # like.
    total = 1.0
    for n in range(2, n_max + 1):
        rn = ratio ** n
        inner = 0.0
        for m in range(n + 1):
            inner += p_bar[n, m] * (c[n, m] * cos_m[m] + s[n, m] * sin_m[m])
        total += rn * inner
    return float(mu / r * total)


# 5-point central-difference stencil coefficients for a first derivative,
# 4th-order accurate: f'(x) ≈ (f(x-2h) - 8 f(x-h) + 8 f(x+h) - f(x+2h)) / (12 h).
_STENCIL_OFFSETS = (-2.0, -1.0, 1.0, 2.0)
_STENCIL_WEIGHTS = (1.0, -8.0, 8.0, -1.0)


def acceleration(
    position: np.ndarray,
    *,
    mu: float,
    r_ref: float,
    c_coeffs: np.ndarray,
    s_coeffs: np.ndarray,
    degree: int | None = None,
    rel_step: float = 1e-4,
) -> np.ndarray:
    """Independent acceleration ``a = +∇U`` via a 4th-order numerical gradient.

    The step is scaled to the position magnitude (``h = rel_step * r``) so the
    stencil stays well-conditioned across altitudes. This deliberately avoids the
    analytic derivative recurrence used by the code under test.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    r = float(np.linalg.norm(pos))
    h = float(rel_step) * r
    if h <= 0.0:
        raise ValueError("position magnitude and rel_step must be positive")

    grad = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        deriv = 0.0
        for offset, weight in zip(_STENCIL_OFFSETS, _STENCIL_WEIGHTS, strict=True):
            probe = pos.copy()
            probe[axis] += offset * h
            deriv += weight * geopotential(
                probe, mu=mu, r_ref=r_ref, c_coeffs=c_coeffs, s_coeffs=s_coeffs, degree=degree
            )
        grad[axis] = deriv / (12.0 * h)
    return grad


@dataclass(frozen=True)
class GravityCrossCheckResult:
    """Per-run summary of the independent-vs-Lunaris acceleration comparison."""

    n_points: int
    degree: int
    max_abs_error: float        # m/s^2
    max_rel_error: float        # dimensionless
    median_rel_error: float
    rel_errors: np.ndarray
    abs_errors: np.ndarray

    def passes(self, *, rel_tol: float) -> bool:
        return bool(self.max_rel_error <= rel_tol)


def crosscheck_gravity_model(
    model: Any,
    points: np.ndarray,
    *,
    degree: int | None = None,
    rel_step: float = 1e-4,
) -> GravityCrossCheckResult:
    """Compare the independent reference against a Lunaris ``GravityModel``.

    ``model`` is duck-typed: it must expose ``mu``/``r_ref``/``c_coeffs``/
    ``s_coeffs`` and ``accel_fixed(position, degree)`` (the Lunaris
    :class:`lunaris.physics.spherical_harmonics.GravityModel` API). It is the
    object *under test*; the reference values come only from this module.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    eval_degree = int(model.max_degree if degree is None else degree)

    abs_errors = np.zeros(len(pts), dtype=np.float64)
    rel_errors = np.zeros(len(pts), dtype=np.float64)
    for i, pos in enumerate(pts):
        ref = acceleration(
            pos,
            mu=float(model.mu),
            r_ref=float(model.r_ref),
            c_coeffs=np.asarray(model.c_coeffs, dtype=np.float64),
            s_coeffs=np.asarray(model.s_coeffs, dtype=np.float64),
            degree=eval_degree,
            rel_step=rel_step,
        )
        got = np.asarray(model.accel_fixed(pos, degree=eval_degree), dtype=np.float64)
        err = float(np.linalg.norm(got - ref))
        denom = float(np.linalg.norm(ref))
        abs_errors[i] = err
        rel_errors[i] = err / denom if denom > 0.0 else err

    return GravityCrossCheckResult(
        n_points=len(pts),
        degree=eval_degree,
        max_abs_error=float(abs_errors.max(initial=0.0)),
        max_rel_error=float(rel_errors.max(initial=0.0)),
        median_rel_error=float(np.median(rel_errors)) if len(rel_errors) else 0.0,
        rel_errors=rel_errors,
        abs_errors=abs_errors,
    )
