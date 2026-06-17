"""Spherical-harmonic normalization helpers used by validation tests."""

from __future__ import annotations

import math

SUPPORTED_NORMALIZATIONS = {
    "fully_normalized_4pi",
    "fully_normalized",
    "unnormalized",
}


def require_supported_normalization(value: str) -> str:
    norm = str(value).strip().lower()
    if norm not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(f"Unsupported normalization: {value!r}")
    return norm


def full_normalization_factor(n: int, m: int) -> float:
    """Return the 4-pi/geodesy full-normalization factor."""
    n_i = int(n)
    m_i = int(m)
    if n_i < 0 or m_i < 0 or m_i > n_i:
        raise ValueError(f"Invalid spherical-harmonic index n={n}, m={m}")
    delta = 1.0 if m_i == 0 else 0.0
    log_fac = math.lgamma(n_i - m_i + 1) - math.lgamma(n_i + m_i + 1)
    return math.sqrt((2.0 - delta) * (2.0 * n_i + 1.0) * math.exp(log_fac))


def unnormalized_to_fully_normalized(value: float, n: int, m: int) -> float:
    """Convert an unnormalized coefficient to the fully normalized convention."""
    return float(value) / full_normalization_factor(n, m)


def fully_normalized_to_unnormalized(value: float, n: int, m: int) -> float:
    """Convert a fully normalized coefficient to the unnormalized convention."""
    return float(value) * full_normalization_factor(n, m)

