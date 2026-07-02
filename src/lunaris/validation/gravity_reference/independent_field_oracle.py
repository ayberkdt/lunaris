"""Independent direct-formula spherical-harmonic field oracle.

This module deliberately does not call the production
``lunaris.physics.spherical_harmonics`` recurrence. It evaluates the potential
with a classical associated-Legendre recurrence and obtains acceleration from a
fourth-order finite-difference gradient of that potential.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def normalization_factor(n: int, m: int) -> float:
    """4-pi/geodesy full-normalization factor."""
    delta = 1.0 if m == 0 else 0.0
    log_fac = math.lgamma(n - m + 1) - math.lgamma(n + m + 1)
    return float(math.sqrt((2.0 - delta) * (2.0 * n + 1.0) * math.exp(log_fac)))


def _associated_legendre_all(n_max: int, sin_phi: float) -> np.ndarray:
    p = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    p[0, 0] = 1.0
    somx2 = math.sqrt(max(0.0, 1.0 - sin_phi * sin_phi))
    for m in range(1, n_max + 1):
        p[m, m] = (2 * m - 1) * somx2 * p[m - 1, m - 1]
    for m in range(n_max):
        p[m + 1, m] = sin_phi * (2 * m + 1) * p[m, m]
    for m in range(n_max + 1):
        for n in range(m + 2, n_max + 1):
            p[n, m] = (
                (2 * n - 1) * sin_phi * p[n - 1, m] - (n + m - 1) * p[n - 2, m]
            ) / (n - m)
    return p


def normalized_alf(n_max: int, sin_phi: float) -> np.ndarray:
    """Return fully normalized associated Legendre values."""
    if n_max < 0:
        raise ValueError("n_max must be non-negative.")
    raw = _associated_legendre_all(n_max, float(sin_phi))
    out = np.zeros_like(raw)
    for n in range(n_max + 1):
        for m in range(n + 1):
            out[n, m] = normalization_factor(n, m) * raw[n, m]
    return out


def geopotential(
    position_m: np.ndarray,
    *,
    mu_m3_s2: float,
    reference_radius_m: float,
    c_coeffs: np.ndarray,
    s_coeffs: np.ndarray,
    degree: int | None = None,
) -> float:
    """Evaluate the positive geodesy potential ``U`` in SI units."""
    pos = np.asarray(position_m, dtype=np.float64).reshape(3)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0:
        raise ValueError("position must be non-zero.")
    c = np.asarray(c_coeffs, dtype=np.float64)
    s = np.asarray(s_coeffs, dtype=np.float64)
    n_avail = min(c.shape[0], s.shape[0]) - 1
    n_max = n_avail if degree is None else min(int(degree), n_avail)
    sin_phi = z / r
    lam = math.atan2(y, x)
    p_bar = normalized_alf(n_max, sin_phi)
    cos_m = np.cos(np.arange(n_max + 1, dtype=np.float64) * lam)
    sin_m = np.sin(np.arange(n_max + 1, dtype=np.float64) * lam)
    ratio = float(reference_radius_m) / r

    # Structural monopole plus the perturbation sum from degree 1: the Lunaris
    # kernels apply stored degree-1 coefficients when non-zero (they are zero in
    # centre-of-mass-frame products), so the oracle must include them too.
    total = 1.0
    for n in range(1, n_max + 1):
        inner = 0.0
        for m in range(n + 1):
            inner += p_bar[n, m] * (c[n, m] * cos_m[m] + s[n, m] * sin_m[m])
        total += ratio**n * inner
    return float(float(mu_m3_s2) / r * total)


_STENCIL_OFFSETS = (-2.0, -1.0, 1.0, 2.0)
_STENCIL_WEIGHTS = (1.0, -8.0, 8.0, -1.0)


def acceleration(
    position_m: np.ndarray,
    *,
    mu_m3_s2: float,
    reference_radius_m: float,
    c_coeffs: np.ndarray,
    s_coeffs: np.ndarray,
    degree: int | None = None,
    rel_step: float = 1e-4,
) -> np.ndarray:
    """Evaluate ``a = +grad(U)`` via an independent fourth-order stencil."""
    pos = np.asarray(position_m, dtype=np.float64).reshape(3)
    radius = float(np.linalg.norm(pos))
    h = float(rel_step) * radius
    if h <= 0.0:
        raise ValueError("position magnitude and rel_step must be positive.")
    grad = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        deriv = 0.0
        for offset, weight in zip(_STENCIL_OFFSETS, _STENCIL_WEIGHTS, strict=True):
            probe = pos.copy()
            probe[axis] += offset * h
            deriv += weight * geopotential(
                probe,
                mu_m3_s2=mu_m3_s2,
                reference_radius_m=reference_radius_m,
                c_coeffs=c_coeffs,
                s_coeffs=s_coeffs,
                degree=degree,
            )
        grad[axis] = deriv / (12.0 * h)
    return grad


@dataclass(frozen=True, slots=True)
class OracleEvaluation:
    potential_m2_s2: float
    acceleration_m_s2: np.ndarray


def evaluate_point(
    position_m: np.ndarray,
    *,
    mu_m3_s2: float,
    reference_radius_m: float,
    c_coeffs: np.ndarray,
    s_coeffs: np.ndarray,
    degree: int | None = None,
    rel_step: float = 1e-4,
    precision_digits: int = 16,
) -> OracleEvaluation:
    """Evaluate potential and acceleration for one point.

    ``precision_digits`` is recorded by manifests. The built-in backend is
    float64; installing an arbitrary-precision backend is intentionally left as a
    regeneration workflow rather than a runtime dependency.
    """
    if int(precision_digits) > 17:
        raise RuntimeError(
            "This environment has no bundled arbitrary-precision SH backend. "
            "Regenerate high-precision references in an external pinned workflow."
        )
    u = geopotential(
        position_m,
        mu_m3_s2=mu_m3_s2,
        reference_radius_m=reference_radius_m,
        c_coeffs=c_coeffs,
        s_coeffs=s_coeffs,
        degree=degree,
    )
    a = acceleration(
        position_m,
        mu_m3_s2=mu_m3_s2,
        reference_radius_m=reference_radius_m,
        c_coeffs=c_coeffs,
        s_coeffs=s_coeffs,
        degree=degree,
        rel_step=rel_step,
    )
    return OracleEvaluation(potential_m2_s2=u, acceleration_m_s2=a)


def coefficients_from_json(payload: dict[str, Any]) -> tuple[int, float, float, np.ndarray, np.ndarray]:
    """Build coefficient arrays from a small validation JSON fixture."""
    if payload.get("schema_version") != "lunaris_synthetic_sh_coefficients_v1":
        raise ValueError("Unsupported coefficient fixture schema_version.")
    degree = int(payload["degree"])
    r_ref = float(payload["reference_radius_m"])
    mu = float(payload["mu_m3_s2"])
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    for row in payload.get("coefficients", []):
        n = int(row["n"])
        m = int(row["m"])
        if n < 0 or m < 0 or m > n or n > degree:
            raise ValueError(f"Invalid coefficient index n={n}, m={m}.")
        c[n, m] = float(row.get("c", 0.0))
        s[n, m] = float(row.get("s", 0.0))
    return degree, r_ref, mu, c, s

