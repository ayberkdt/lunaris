# tests/test_solar_effects.py
"""
Pytest suite for Solar Radiation Pressure (SRP) + eclipse geometry.

This file is migrated from the module-level "SMOKE TEST" in models/solar_effects.py
so it can run in CI and be executed deterministically.

Run (repo root):
  python -m pytest -vv -rA --durations=10 tests/test_solar_effects.py
"""

from __future__ import annotations

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
    from lunar_simulation.models.solar_effects import (  # type: ignore
        compute_srp_accel,
        moon_shadow_factor_conical,
        SRPConfig,
    )
    from lunar_simulation.common.constants import AU, R_MOON_MEAN, R_EARTH_MEAN  # type: ignore
    from lunar_simulation.common.type_defs import SpacecraftProps  # type: ignore
except Exception:  # pragma: no cover
    try:
        from models.solar_effects import (  # type: ignore
            compute_srp_accel,
            moon_shadow_factor_conical,
            SRPConfig,
        )
        from common.constants import AU, R_MOON_MEAN, R_EARTH_MEAN  # type: ignore
        from common.type_defs import SpacecraftProps  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import solar_effects. Update the import path in "
            "tests/test_solar_effects.py to match your repo layout."
        ) from e


# ---------------------------------------------------------------------------
# Shared test inputs (deterministic)
# ---------------------------------------------------------------------------

AU_TEST = float(AU)
R_MOON_TEST = float(R_MOON_MEAN)
R_EARTH_TEST = float(R_EARTH_MEAN)

# Common "solar radiation pressure at 1 AU" constant used in your smoke test
P0_TEST = 4.56e-6  # [N/m^2]

# Spacecraft (cannonball model)
# 1000 kg, 10 m^2, Cr=1.8
TEST_PROPS = SpacecraftProps(
    mass_kg=1000.0,
    area_m2=10.0,
    cr=1.8,
    cd=2.2,  # required by dataclass but unused by SRP
)

# Moon-centered Sun vector (simple synthetic geometry)
R_SUN_VEC = np.array([AU_TEST, 0.0, 0.0], dtype=np.float64)


def _expected_srp_mag(r_sc_moon: np.ndarray) -> float:
    """
    Expected SRP magnitude for full sunlight (no eclipse), based on the same scaling
    used by the kernel: (AU / d)^2, where d = ||r_sc - r_sun||.
    """
    d = float(np.linalg.norm(r_sc_moon - R_SUN_VEC))
    if d <= 0.0:
        return 0.0
    scaling = (AU_TEST / d) ** 2
    return (P0_TEST * float(TEST_PROPS.cr) * float(TEST_PROPS.area_m2) / float(TEST_PROPS.mass_kg)) * scaling


def _cos_alignment(acc: np.ndarray, d_vec: np.ndarray) -> float:
    """
    Cosine of angle between acceleration and d_vec. Both must be non-zero.
    """
    a_norm = float(np.linalg.norm(acc))
    d_norm = float(np.linalg.norm(d_vec))
    if a_norm <= 0.0 or d_norm <= 0.0:
        return 0.0
    return float(np.dot(acc / a_norm, d_vec / d_norm))


@pytest.fixture(scope="session", autouse=True)
def _warmup_numba() -> None:
    """
    Trigger Numba compilation once per session (if njit is active).
    """
    cfg = SRPConfig(
        P0=P0_TEST,
        AU_m=AU_TEST,
        enable_moon_eclipse=False,
        enable_earth_eclipse=False,
        shadow_model="conical",
        R_moon_m=R_MOON_TEST,
        R_earth_m=R_EARTH_TEST,
    )
    r_sc = np.array([R_MOON_TEST + 1.0e6, 0.0, 0.0], dtype=np.float64)
    r_earth = np.zeros(3, dtype=np.float64)
    _ = compute_srp_accel(r_sc, R_SUN_VEC, r_earth, TEST_PROPS, cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_sunlight_day_side_magnitude_and_direction() -> None:
    """
    Day-side point: should be fully illuminated (moon eclipse enabled but not active),
    magnitude should match closed-form, direction should align with d_vec = r_sc - r_sun.
    """
    cfg = SRPConfig(
        P0=P0_TEST,
        AU_m=AU_TEST,
        enable_moon_eclipse=True,
        enable_earth_eclipse=False,
        shadow_model="conical",
        R_moon_m=R_MOON_TEST,
        R_earth_m=R_EARTH_TEST,
    )

    r_day = np.array([R_MOON_TEST + 1000e3, 0.0, 0.0], dtype=np.float64)
    r_earth = np.zeros(3, dtype=np.float64)  # unused when earth eclipse disabled

    acc = compute_srp_accel(r_day, R_SUN_VEC, r_earth, TEST_PROPS, cfg)

    expected_mag = _expected_srp_mag(r_day)
    got_mag = float(np.linalg.norm(acc))

    # Use relative tolerance (more robust than absolute 1e-12)
    np.testing.assert_allclose(got_mag, expected_mag, rtol=1e-12, atol=0.0)

    d_vec = r_day - R_SUN_VEC
    cosang = _cos_alignment(acc, d_vec)
    assert cosang >= 1.0 - 1e-12, f"Direction mismatch: cos={cosang:.16f}"


def test_full_shadow_umbra_returns_zero() -> None:
    """
    Night-side point directly behind the Moon: should be in umbra -> zero accel.
    """
    cfg = SRPConfig(
        P0=P0_TEST,
        AU_m=AU_TEST,
        enable_moon_eclipse=True,
        enable_earth_eclipse=False,
        shadow_model="conical",
        R_moon_m=R_MOON_TEST,
        R_earth_m=R_EARTH_TEST,
    )

    r_night = np.array([-R_MOON_TEST - 1000e3, 0.0, 0.0], dtype=np.float64)
    r_earth = np.zeros(3, dtype=np.float64)

    acc = compute_srp_accel(r_night, R_SUN_VEC, r_earth, TEST_PROPS, cfg)
    assert float(np.linalg.norm(acc)) < 1e-15


def test_penumbra_shadow_factor_between_0_and_1() -> None:
    """
    Construct a deterministic point in the penumbra band (0 < nu < 1).

    We choose an x behind the Moon and pick rho midway between the local umbra
    and penumbra radii (derived from the same conical geometry relations).
    """
    # Conical geometry (mirrors the kernel formulas)
    # Sun distance from Moon:
    dist_sun = AU_TEST

    denom_u = (6.9634e8) - R_MOON_TEST  # approx R_SUN_MEAN - R_occult, keep stable even if constants differ
    denom_p = (6.9634e8) + R_MOON_TEST  # approx R_SUN_MEAN + R_occult

    # If your project uses a different R_SUN_MEAN constant, this still tends to be close enough
    # to land inside the penumbra for the chosen x (band widths are O(km)).
    Lu = (R_MOON_TEST * dist_sun) / denom_u
    Lp = (R_MOON_TEST * dist_sun) / denom_p

    x = 1.0e6  # 1000 km behind the Moon (x < Lu, so umbra radius > 0)
    r_u = R_MOON_TEST * (1.0 - x / Lu)
    r_p = R_MOON_TEST * (1.0 + x / Lp)

    rho = 0.5 * (r_u + r_p)  # mid-penumbra
    r_sc = np.array([-x, rho, 0.0], dtype=np.float64)

    nu = float(moon_shadow_factor_conical(
        float(r_sc[0]), float(r_sc[1]), float(r_sc[2]),
        float(R_SUN_VEC[0]), float(R_SUN_VEC[1]), float(R_SUN_VEC[2]),
        float(R_MOON_TEST),
    ))

    assert 0.0 < nu < 1.0, f"Expected penumbra (0<nu<1), got nu={nu:.6f}"


def test_earth_eclipse_enabled_can_zero_out_accel_for_behind_earth_geometry() -> None:
    """
    Deterministic Earth eclipse: place spacecraft directly behind Earth (Earth-centered rho=0),
    and enable only Earth eclipse. Should yield ~0 acceleration.
    """
    cfg_earth = SRPConfig(
        P0=P0_TEST,
        AU_m=AU_TEST,
        enable_moon_eclipse=False,
        enable_earth_eclipse=True,
        shadow_model="conical",
        R_moon_m=R_MOON_TEST,
        R_earth_m=R_EARTH_TEST,
    )

    # Moon-centered Earth position (rough)
    r_earth_moon = np.array([3.84e8, 0.0, 0.0], dtype=np.float64)

    # Spacecraft behind Earth, along -x relative to Earth (umbra line)
    r_sc = r_earth_moon + np.array([-R_EARTH_TEST - 1000e3, 0.0, 0.0], dtype=np.float64)

    acc = compute_srp_accel(r_sc, R_SUN_VEC, r_earth_moon, TEST_PROPS, cfg_earth)
    assert float(np.linalg.norm(acc)) < 1e-15


if __name__ == "__main__":
    import sys

    print("This is a pytest test module. Run it with:")
    print("python -m pytest -vv -rA --durations=10 tests/test_solar_effects.py")
    sys.exit(0)
