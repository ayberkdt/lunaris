"""test_third_body_effects_v2.py

Pytest smoke/regression tests for `lunaris.physics.third_body_effects`.

These tests are intentionally lightweight:
- They validate sign/direction and a 1D closed-form case for differential 3rd-body gravity.
- They sanity-check solid-tide and Earth-J2 differential contributions.
- They validate the `ThirdBodyModel` facade matches a manual composition of the same kernels.

Run from project root:
    pytest -q
"""

from __future__ import annotations

import importlib
import math
from typing import Tuple, Any

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Import helpers
# -----------------------------------------------------------------------------
def _import_third_body_module():
    """Import `lunaris.physics.third_body_effects` with a small fallback list."""
    last_err: Exception | None = None
    for modname in ("lunaris.physics.third_body_effects", "models.third_body_effects", "third_body_effects"):
        try:
            return importlib.import_module(modname)
        except Exception as e:  # pragma: no cover
            last_err = e
    assert last_err is not None
    raise ImportError("Could not import third_body_effects module") from last_err


# -----------------------------------------------------------------------------
# Small math helpers
# -----------------------------------------------------------------------------
def _norm3(v: np.ndarray) -> float:
    return float(math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])))


def _rel_err(a: float, b: float, floor: float = 1e-30) -> float:
    denom = max(float(floor), abs(float(b)))
    return abs(float(a) - float(b)) / denom


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tbe():
    return _import_third_body_module()


@pytest.fixture(scope="session")
def constants() -> Tuple[float, float, float, float]:
    """Best-effort import of common constants with safe numeric fallbacks."""
    # Defaults (SI)
    MU_EARTH = 3.986004418e14           # [m^3/s^2]
    MU_SUN = 1.32712440018e20           # [m^3/s^2]
    R_MOON = 1_737_400.0                # [m] mean radius
    R_EARTH_EQ = 6_378_137.0            # [m] WGS-84 equatorial radius

    try:  # Prefer project SSOT if available
        from lunaris.common.constants import MU_EARTH as _MU_EARTH  # type: ignore
        from lunaris.common.constants import MU_SUN as _MU_SUN      # type: ignore
        # Moon radius name varies a bit across codebases
        try:
            from lunaris.common.constants import R_MOON_MEAN as _R_MOON  # type: ignore
        except Exception:  # pragma: no cover
            from lunaris.common.constants import R_MOON_MEAN_M as _R_MOON  # type: ignore
        try:
            from lunaris.common.constants import R_EARTH_EQUATORIAL as _R_EARTH_EQ  # type: ignore
        except Exception:  # pragma: no cover
            try:
                from lunaris.common.constants import R_EARTH_EQ as _R_EARTH_EQ  # type: ignore
            except Exception:  # pragma: no cover
                from lunaris.common.constants import R_EARTH_MEAN as _R_EARTH_EQ  # type: ignore

        MU_EARTH = float(_MU_EARTH)
        MU_SUN = float(_MU_SUN)
        R_MOON = float(_R_MOON)
        R_EARTH_EQ = float(_R_EARTH_EQ)

    except Exception:
        # Fall back to the local numeric defaults above.
        pass

    return MU_EARTH, MU_SUN, R_MOON, R_EARTH_EQ


@pytest.fixture(scope="session", autouse=True)
def _warmup_numba(tbe, constants):
    """Trigger a tiny JIT warmup once per session to fail early if Numba is broken."""
    MU_EARTH, _, R_MOON, R_EARTH_EQ = constants

    # Simple collinear geometry for warmup
    r_sc = np.array((100e3, 0.0, 0.0), dtype=np.float64)
    r_earth = np.array((384_400e3, 0.0, 0.0), dtype=np.float64)

    # Core kernels should exist and be callable
    _ = tbe.accel_third_body_numba(
        float(r_sc[0]), float(r_sc[1]), float(r_sc[2]),
        float(r_earth[0]), float(r_earth[1]), float(r_earth[2]),
        float(MU_EARTH),
    )

    # Solid tide kernel is intentionally not in __all__, but should exist if implemented
    if hasattr(tbe, "accel_solid_tide"):
        _ = tbe.accel_solid_tide(
            float(r_sc[0]), float(r_sc[1]), float(r_sc[2]),
            float(r_earth[0]), float(r_earth[1]), float(r_earth[2]),
            float(MU_EARTH), float(R_MOON),
            0.024, 0.0,
        )

    # Earth-J2 differential kernel
    if hasattr(tbe, "accel_j2_oblate_diff_numba"):
        _ = tbe.accel_j2_oblate_diff_numba(
            float(r_sc[0]), float(r_sc[1]), float(r_sc[2]),
            float(r_earth[0]), float(r_earth[1]), float(r_earth[2]),
            float(MU_EARTH),
            float(R_EARTH_EQ),
            1.082_626_68e-3,
            0.0, 0.0, 1.0,
        )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_third_body_1d_analytic_and_scaling(tbe, constants):
    MU_EARTH, _, _, _ = constants

    # Collinear geometry: Earth on +X axis, spacecraft between Moon and Earth.
    r_sc = np.array((100e3, 0.0, 0.0), dtype=np.float64)
    r_sc_far = np.array((300e3, 0.0, 0.0), dtype=np.float64)  # closer to Earth
    r_earth = np.array((384_400e3, 0.0, 0.0), dtype=np.float64)

    def ax_1d_ref(mu: float, b: float, r: float) -> float:
        # Differential 3B accel on +X for 0<r<b:
        #   ax = mu * ( 1/(b-r)^2 - 1/b^2 )
        d = b - r
        if d <= 0.0 or b <= 0.0:
            return 0.0
        return mu * (1.0 / (d * d) - 1.0 / (b * b))

    acc_3b = tbe.calc_3rd_body_accel(r_sc, r_earth, MU_EARTH)
    acc_3b_far = tbe.calc_3rd_body_accel(r_sc_far, r_earth, MU_EARTH)

    ax_ref = ax_1d_ref(float(MU_EARTH), float(r_earth[0]), float(r_sc[0]))
    ax_ref_far = ax_1d_ref(float(MU_EARTH), float(r_earth[0]), float(r_sc_far[0]))

    # Direction check: +X expected, y/z ~ 0 for this setup.
    assert acc_3b[0] > 0.0
    assert abs(float(acc_3b[1])) < 1e-12
    assert abs(float(acc_3b[2])) < 1e-12

    # Analytic check (tight but not fragile; fastmath can change last bits)
    assert _rel_err(float(acc_3b[0]), ax_ref) < 1e-10
    assert _rel_err(float(acc_3b_far[0]), ax_ref_far) < 1e-10

    # Scaling: closer to Earth -> larger differential acceleration magnitude.
    assert _norm3(acc_3b_far) > _norm3(acc_3b)


def test_solid_tide_nonzero_and_small(tbe, constants):
    MU_EARTH, _, R_MOON, _ = constants
    if not hasattr(tbe, "accel_solid_tide"):
        pytest.skip("accel_solid_tide not implemented in third_body_effects")

    r_sc = np.array((100e3, 0.0, 0.0), dtype=np.float64)
    r_earth = np.array((384_400e3, 0.0, 0.0), dtype=np.float64)

    acc_3b = tbe.calc_3rd_body_accel(r_sc, r_earth, MU_EARTH)

    tx, ty, tz = tbe.accel_solid_tide(
        float(r_sc[0]), float(r_sc[1]), float(r_sc[2]),
        float(r_earth[0]), float(r_earth[1]), float(r_earth[2]),
        float(MU_EARTH), float(R_MOON),
        0.024, 0.0,
    )
    acc_tide = np.array((tx, ty, tz), dtype=np.float64)

    assert _norm3(acc_tide) > 0.0
    # Heuristic: for this geometry, the tide term should be smaller than the 3B differential term.
    assert _norm3(acc_tide) < _norm3(acc_3b)


def test_earth_j2_differential_explicit_and_params(tbe, constants):
    MU_EARTH, _, _, R_EARTH_EQ = constants

    r_sc = np.array((100e3, 0.0, 0.0), dtype=np.float64)
    r_earth = np.array((384_400e3, 0.0, 0.0), dtype=np.float64)

    k_hat = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    J2_EARTH = 1.082_626_68e-3

    acc_j2 = tbe.calc_j2_oblate_diff_accel(
        r_sc, r_earth,
        mu_body=float(MU_EARTH),
        r_ref=float(R_EARTH_EQ),
        j2=float(J2_EARTH),
        k_hat=k_hat,
    )

    assert _norm3(acc_j2) > 0.0

    # Heuristic: J2 correction is usually smaller than the point-mass differential term here.
    acc_3b = tbe.calc_3rd_body_accel(r_sc, r_earth, MU_EARTH)
    assert _norm3(acc_j2) < _norm3(acc_3b)

    # Also verify the params=EarthJ2Params path (if available) matches explicit inputs.
    if hasattr(tbe, "EarthJ2Params"):
        params = tbe.EarthJ2Params(j2_coeff=float(J2_EARTH), r_eq_m=float(R_EARTH_EQ), spin_axis_i=(0.0, 0.0, 1.0))
        acc_j2_p = tbe.calc_j2_oblate_diff_accel(
            r_sc, r_earth,
            mu_body=float(MU_EARTH),
            params=params,
        )
        assert np.allclose(acc_j2_p, acc_j2, rtol=1e-12, atol=1e-15)


def test_manager_matches_manual_composition(tbe, constants):
    MU_EARTH, _, R_MOON, _ = constants

    if not hasattr(tbe, "ThirdBodyModel") or not hasattr(tbe, "LoveParams") or not hasattr(tbe, "accel_solid_tide"):
        pytest.skip("ThirdBodyModel/LoveParams/accel_solid_tide not available")

    r_sc = np.array((100e3, 0.0, 0.0), dtype=np.float64)
    r_earth = np.array((384_400e3, 0.0, 0.0), dtype=np.float64)

    k2_test = 0.024
    k3_test = 0.0

    model = tbe.ThirdBodyModel(
        love=tbe.LoveParams(k2=k2_test, k3=k3_test, apply_earth_tide=True, apply_sun_tide=False),
        R_ref=float(R_MOON),
    )

    a_model = model.compute_earth(r_sc, r_earth)

    a_3b = tbe.calc_3rd_body_accel(r_sc, r_earth, MU_EARTH)
    tx, ty, tz = tbe.accel_solid_tide(
        float(r_sc[0]), float(r_sc[1]), float(r_sc[2]),
        float(r_earth[0]), float(r_earth[1]), float(r_earth[2]),
        float(MU_EARTH), float(R_MOON),
        float(k2_test), float(k3_test),
    )
    a_manual = a_3b + np.array((tx, ty, tz), dtype=np.float64)

    # Allow tiny last-bit differences (fastmath / evaluation order)
    assert np.allclose(a_model, a_manual, rtol=1e-7, atol=1e-11)
