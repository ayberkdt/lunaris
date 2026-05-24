# tests/test_dynamics.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Imports (skip cleanly if the package layout isn't available in this context)
# -----------------------------------------------------------------------------
try:
    from core.dynamics import DynamicsEngine, extract_ephem_tables_strict
except Exception as e:  # pragma: no cover
    pytest.skip(f"core.dynamics not importable: {e}", allow_module_level=True)

try:
    from common.type_defs import SpacecraftProps, PerturbationFlags
except Exception as e:  # pragma: no cover
    pytest.skip(f"common.type_defs not importable: {e}", allow_module_level=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm3(x: float, y: float, z: float) -> float:
    return float(math.sqrt(x * x + y * y + z * z))


def _build_default_state(*, r_km: float = 1837.4, v_ms: float = 1600.0) -> np.ndarray:
    """Simple planar state: r=[r,0,0], v=[0,v,0] in SI units."""
    r_m = float(r_km) * 1000.0
    y = np.zeros(6, dtype=np.float64)
    y[0] = r_m
    y[4] = float(v_ms)
    return y


@pytest.fixture(scope="module")
def engine_point_mass() -> Tuple[DynamicsEngine, callable]:
    """
    Build a minimal engine configuration with point-mass gravity only.
    Keeps compilation cost to a minimum by reusing the same RHS for the module.
    """
    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)

    # Everything disabled => ephemeris/gravity model not required.
    flags = PerturbationFlags(
        enable_sh=False,
        enable_3rd_body_sun=False,
        enable_3rd_body_earth=False,
        enable_srp=False,
        enable_albedo=False,
        enable_relativity_1pn=False,
        enable_earth_j2=False,
    )

    eng = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        gravity_model=None,
        ephem_manager=None,
        surface_provider=None,
        earth_j2=None,
        allow_identity_rotation=True,  # OK since SH/albedo are disabled.
    )

    rhs = eng.build_rhs(force_rebuild=True)

    # Warm-up (Numba compile) once at module scope.
    y0 = _build_default_state()
    _ = rhs(0.0, y0)

    return eng, rhs


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_rhs_shape_and_inward_acceleration(engine_point_mass: Tuple[DynamicsEngine, callable]) -> None:
    eng, rhs = engine_point_mass

    t0 = 0.0
    y0 = _build_default_state()

    dy0 = rhs(t0, y0)
    assert dy0.shape == y0.shape

    ax, ay, az = float(dy0[3]), float(dy0[4]), float(dy0[5])
    a_norm = _norm3(ax, ay, az)
    assert math.isfinite(a_norm) and a_norm > 0.0

    r = y0[0:3]
    r_norm = float(np.linalg.norm(r))
    assert r_norm > 0.0

    # For point-mass gravity only, acceleration should point roughly inward: a · r < 0
    dot_ar = float(ax * r[0] + ay * r[1] + az * r[2])
    assert dot_ar < 0.0


def test_one_step_consistency_smoke(engine_point_mass: Tuple[DynamicsEngine, callable]) -> None:
    _, rhs = engine_point_mass

    t0 = 0.0
    dt = 1.0
    y0 = _build_default_state()

    dy0 = rhs(t0, y0)
    y1 = y0 + dt * dy0
    dy1 = rhs(t0 + dt, y1)

    assert dy1.shape == y0.shape
    assert np.all(np.isfinite(dy1))

    # Algebraic identity check (should be ~0, only roundoff left)
    dv_res = float(np.linalg.norm((y1[3:6] - y0[3:6]) - dt * dy0[3:6]))
    assert dv_res < 1e-10


def test_extract_ephem_tables_accepts_constant_vector_rows_with_full_quaternion_timeline() -> None:
    class _MockEphem:
        def get_data_provider(self):
            return {
                "dt_s": 60.0,
                "r_sun_tab_m": np.zeros((1, 3), dtype=np.float64),
                "r_earth_tab_m": np.zeros((1, 3), dtype=np.float64),
                "q_i2f_tab": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (8, 1)),
            }

    dt_s, sun_tab, earth_tab, q_tab = extract_ephem_tables_strict(_MockEphem())

    assert dt_s == 60.0
    assert sun_tab.shape == (1, 3)
    assert earth_tab.shape == (1, 3)
    assert q_tab.shape == (8, 4)


def test_acceleration_breakdown_smoke(engine_point_mass: Tuple[DynamicsEngine, callable]) -> None:
    eng, _ = engine_point_mass

    if not hasattr(eng, "get_acceleration_breakdown"):
        pytest.skip("DynamicsEngine.get_acceleration_breakdown not available in this build.")

    y0 = _build_default_state()
    comp = eng.get_acceleration_breakdown(0.0, y0)

    assert isinstance(comp, dict)
    assert len(comp) >= 1

    # Expect at least a gravity term in minimal config.
    has_gravity = any("gravity" in k.lower() for k in comp.keys())
    assert has_gravity

    # All norms should be finite and non-negative.
    for k, v in comp.items():
        assert math.isfinite(float(v))
        assert float(v) >= 0.0


def test_surrogate_gravity_provider_can_drive_python_rhs() -> None:
    class _StubSurrogateGravity:
        model_kind = "st_lrps"
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12

        def surrogate_forward(self, *args, **kwargs):
            return np.array([-1.25, 0.0, 0.0], dtype=np.float64)

        def acceleration_fixed(self, _r_fixed):
            return np.array([-1.25, 0.0, 0.0], dtype=np.float64)

    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
    flags = PerturbationFlags(enable_sh=True)
    eng = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        gravity_model=_StubSurrogateGravity(),
        ephem_manager=None,
        surface_provider=None,
        earth_j2=None,
        allow_identity_rotation=True,
    )

    rhs = eng.build_rhs(force_rebuild=True)
    y0 = _build_default_state()
    dy0 = rhs(0.0, y0)

    assert dy0.shape == y0.shape
    assert float(dy0[3]) == pytest.approx(-1.25)
    assert float(dy0[4]) == pytest.approx(0.0)
    assert float(dy0[5]) == pytest.approx(0.0)

    comp = eng.get_acceleration_breakdown(0.0, y0)
    assert "Gravity (ST-LRPS)" in comp


@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="Set RUN_SLOW=1 to run slow integration test.")
def test_solve_ivp_mini_run(engine_point_mass: Tuple[DynamicsEngine, callable]) -> None:
    scipy = pytest.importorskip("scipy")
    from scipy.integrate import solve_ivp  # type: ignore

    _, rhs = engine_point_mass
    t0 = 0.0
    tf = 10.0
    y0 = _build_default_state()

    sol = solve_ivp(rhs, (t0, tf), y0, rtol=1e-9, atol=1e-12, max_step=1.0)

    assert sol.status in (0, 1)  # 0: success, 1: terminated (shouldn't happen here)
    assert sol.t.size >= 2

    y_end = sol.y[:, -1]
    assert y_end.shape[0] == 6
    assert np.all(np.isfinite(y_end))
