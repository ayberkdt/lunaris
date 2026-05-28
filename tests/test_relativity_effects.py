# tests/test_relativity_effects.py
"""
Pytest suite for the 1PN Schwarzschild relativistic acceleration correction.

This file is migrated from the module-level self-test so it can run in CI.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import the module under test.
# We add repo root to sys.path to support both "package" and "flat" layouts.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Try a couple of likely import paths (adjust if your project layout differs).
try:
    from lunar_simulation.models.relativity_effects import (  # type: ignore
        calc_schwarzschild_accel,
        MU_MOON,
        C_SQ,
        EPS_1E12,
    )
except Exception:  # pragma: no cover
    try:
        from lunaris.physics.relativity_effects import (  # type: ignore
            calc_schwarzschild_accel,
            MU_MOON,
            C_SQ,
            EPS_1E12,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import relativity_effects. "
            "Update the import path in tests/test_relativity_effects.py to match your repo layout."
        ) from e


# ---------------------------------------------------------------------------
# Pure-Python reference implementation (no numba), same formula as kernel
# ---------------------------------------------------------------------------

def _schwarzschild_components_ref(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    mu: float,
) -> tuple[float, float, float]:
    """
    Pure-Python reference (no numba), same formula as the kernel.
    Used only for test comparisons.
    """
    r2 = rx * rx + ry * ry + rz * rz
    if r2 <= EPS_1E12:
        return 0.0, 0.0, 0.0

    r = math.sqrt(r2)
    v2 = vx * vx + vy * vy + vz * vz
    rv = rx * vx + ry * vy + rz * vz

    term_common = mu / (C_SQ * r2 * r)
    alpha = (4.0 * mu / r) - v2
    beta = 4.0 * rv

    ax = term_common * (alpha * rx + beta * vx)
    ay = term_common * (alpha * ry + beta * vy)
    az = term_common * (alpha * rz + beta * vz)
    return ax, ay, az


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _warmup_numba() -> None:
    """
    Trigger JIT compilation once per test session so timing/noise isn't repeated.
    """
    r_warm = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_warm = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    _ = calc_schwarzschild_accel(r_warm, v_warm, float(MU_MOON))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calc_returns_float64_vec3() -> None:
    r = np.array([2.0e6, -1.0e6, 3.0e6], dtype=np.float64)
    v = np.array([1200.0, -800.0, 50.0], dtype=np.float64)

    a = calc_schwarzschild_accel(r, v, float(MU_MOON))

    assert isinstance(a, np.ndarray)
    assert a.shape == (3,)
    assert a.dtype == np.float64


def test_singularity_protection_near_zero_r() -> None:
    r0 = np.zeros(3, dtype=np.float64)
    v0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    a0 = calc_schwarzschild_accel(r0, v0, float(MU_MOON))

    assert np.allclose(a0, 0.0), "Expected zero accel for near-zero position vector"


def test_v_zero_parallel_and_magnitude() -> None:
    # v = 0 case: accel must be parallel to r_vec and match closed-form magnitude
    r1 = np.array([1.2e6, -2.3e6, 0.7e6], dtype=np.float64)
    v1 = np.zeros(3, dtype=np.float64)

    a1 = calc_schwarzschild_accel(r1, v1, float(MU_MOON))

    # Parallel check: r x a ≈ 0
    cross = np.cross(r1, a1)
    # Scaled tolerance: avoid false fails when magnitudes are tiny
    assert np.linalg.norm(cross) <= 1e-12 * (np.linalg.norm(r1) * np.linalg.norm(a1) + 1.0), (
        "For v=0, accel should be parallel to r_vec"
    )

    # Magnitude check:
    # |a| = 4*mu^2 / (c^2 * r^3)  since a_vec = 4*mu^2/(c^2*r^4) * r_vec
    rmag = float(np.linalg.norm(r1))
    expected_mag = 4.0 * float(MU_MOON) ** 2 / (C_SQ * (rmag ** 3))
    got_mag = float(np.linalg.norm(a1))
    assert abs(got_mag - expected_mag) <= 1e-12 * max(expected_mag, 1.0), (
        f"v=0 magnitude mismatch: got {got_mag}, expected {expected_mag}"
    )


def test_reference_agreement() -> None:
    # Deterministic vectors (no randomness) to keep CI stable
    r2 = np.array([2.1e6, 1.7e6, -0.9e6], dtype=np.float64)
    v2 = np.array([-500.0, 1250.0, 200.0], dtype=np.float64)
    mu = float(MU_MOON)

    a_numba = calc_schwarzschild_accel(r2, v2, mu)

    axr, ayr, azr = _schwarzschild_components_ref(
        float(r2[0]), float(r2[1]), float(r2[2]),
        float(v2[0]), float(v2[1]), float(v2[2]),
        mu,
    )
    a_ref = np.array([axr, ayr, azr], dtype=np.float64)

    np.testing.assert_allclose(a_numba, a_ref, rtol=1e-13, atol=0.0)



if __name__ == "__main__":
    import sys

    print("This is a pytest test module. Run it with:")
    print("  python -m pytest -vv -rA --durations=10 tests/test_relativity_effects.py")
    sys.exit(0)


 