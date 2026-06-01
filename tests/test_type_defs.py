# tests/test_type_defs.py
# -*- coding: utf-8 -*-
"""
Unit tests for common.type_defs
==============================

These tests are the pytest version of the legacy in-module self-test, aligned to
the *current* common/type_defs.py API and validation rules.

Run:
    pytest -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

# -----------------------------------------------------------------------------
# Import helper (run from repo root without installing)
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from lunaris.common.type_defs import (
        SpacecraftProps,
        AdaptiveDegreeConfig,
        GravityConfig,
        PerturbationFlags,
        SolidTideConfig,
        TimeConfig,
        InitialState,
        EventConfig,
        PropagatorConfig,
        SimulationHistory,
        PropagationResult,
    )
    from lunaris.common.constants import DAY_S, R_MOON_MEAN
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import 'common.type_defs'. "
        "Run pytest from the repository root (the folder that contains 'common/')."
    ) from e


# =============================================================================
# 1) SpacecraftProps
# =============================================================================

def test_spacecraft_props_ballistic_coefficient():
    sp = SpacecraftProps(mass_kg=100.0, area_m2=2.0, cd=2.0, cr=1.8)
    assert_allclose(sp.ballistic_coefficient, 25.0, atol=0.0, rtol=0.0)

    sp_inf = SpacecraftProps(mass_kg=100.0, area_m2=0.0, cd=2.0, cr=1.8)
    assert math.isinf(sp_inf.ballistic_coefficient)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mass_kg=0.0, area_m2=1.0, cd=2.0, cr=1.0),   # mass_kg must be >0
        dict(mass_kg=1.0, area_m2=-1.0, cd=2.0, cr=1.0),  # area_m2 must be >=0
        dict(mass_kg=1.0, area_m2=1.0, cd=-0.1, cr=1.0),  # cd must be >=0
        dict(mass_kg=1.0, area_m2=1.0, cd=2.0, cr=-0.1),  # cr must be >=0
    ],
)
def test_spacecraft_props_validation(kwargs):
    with pytest.raises(ValueError):
        SpacecraftProps(**kwargs)


# =============================================================================
# 2) AdaptiveDegreeConfig
# =============================================================================

def test_adaptive_degree_config_defaults_construct():
    _ = AdaptiveDegreeConfig()  # should not raise


def test_adaptive_degree_config_rejects_bad_params():
    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(quantization_step=0)
    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(min_degree=-1)
    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(power=0.0)


def test_adaptive_degree_config_table_validation():
    _ = AdaptiveDegreeConfig(altitude_table=((100.0, 60), (200.0, 40)))  # ok

    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(altitude_table=())  # empty is forbidden; use None

    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(altitude_table=((-1.0, 50), (100.0, 40)))  # negative altitude

    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(altitude_table=((100.0, -1), (200.0, 40)))  # negative degree

    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(altitude_table=((200.0, 40), (100.0, 60)))  # not strictly increasing

    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(altitude_table=((100.0, 60, 0), (200.0, 40)))  # wrong row shape


# =============================================================================
# 3) GravityConfig
# =============================================================================

def test_gravity_config_accepts_valid():
    _ = GravityConfig(file_path="gravity.gfc", degree=None)
    _ = GravityConfig(file_path="gravity.gfc", degree=120)
    _ = GravityConfig(
        file_path="gravity.gfc",
        degree=120,
        backend="st_lrps",
        st_lrps_model_dir="st_lrps/runs/example_run",
    )


@pytest.mark.parametrize("file_path", ["", "   "])
def test_gravity_config_rejects_empty_path(file_path):
    with pytest.raises(ValueError):
        GravityConfig(file_path=file_path, degree=100)


def test_gravity_config_rejects_negative_degree():
    with pytest.raises(ValueError):
        GravityConfig(file_path="gravity.gfc", degree=-1)


def test_gravity_config_rejects_surrogate_backend_without_run_dir():
    with pytest.raises(ValueError):
        GravityConfig(file_path="gravity.gfc", backend="st_lrps", st_lrps_model_dir="")


def test_gravity_config_rejects_adaptive_min_degree_exceeding_degree():
    with pytest.raises(ValueError):
        GravityConfig(
            file_path="gravity.gfc",
            degree=20,
            adaptive=AdaptiveDegreeConfig(min_degree=30),
        )


# =============================================================================
# 4) PerturbationFlags
# =============================================================================

def test_perturbation_flags_tides_guards_and_properties():
    with pytest.raises(ValueError):
        PerturbationFlags(enable_tides_k3=True, enable_tides_k2=False)

    pf = PerturbationFlags(enable_tides_k3=True, enable_tides_k2=True)
    assert pf.enable_tides_k2 is True and pf.enable_tides_k3 is True
    assert pf.tides_degree == 3
    assert pf.tides_kind == "k3"

    pf2 = PerturbationFlags(enable_tides_k2=True)
    assert pf2.tides_degree == 2
    assert pf2.tides_kind == "k2"

    pf0 = PerturbationFlags()
    assert pf0.tides_degree == 0
    assert pf0.tides_kind == "none"


def test_solid_tide_config_validation_and_normalization():
    cfg = SolidTideConfig(tide_bodies=("Earth", "sun", "earth"), k2=0.0, k3=0.01)
    assert cfg.tide_bodies == ("earth", "sun")
    assert cfg.k2 == 0.0
    assert cfg.k3 == 0.01

    with pytest.raises(ValueError):
        SolidTideConfig(tide_bodies=("mars",))
    with pytest.raises(ValueError):
        SolidTideConfig(tide_bodies=())
    with pytest.raises(ValueError):
        SolidTideConfig(k2=-1.0)
    with pytest.raises(ValueError):
        SolidTideConfig(r_ref_m=0.0)


def test_perturbation_flags_enable_third_body_and_surface_forces():
    pf_tb = PerturbationFlags(enable_3rd_body_sun=True)
    assert pf_tb.enable_third_body is True

    pf_tb2 = PerturbationFlags(enable_3rd_body_earth=True)
    assert pf_tb2.enable_third_body is True

    pf_tb3 = PerturbationFlags(enable_earth_j2=True)
    assert pf_tb3.enable_third_body is True

    pf_sf = PerturbationFlags(enable_albedo=True)
    assert pf_sf.enable_surface_forces is True

    pf_sf2 = PerturbationFlags(enable_thermal=True)
    assert pf_sf2.enable_surface_forces is True


# =============================================================================
# 5) TimeConfig
# =============================================================================

def test_time_config_duration_days_and_validation():
    tc = TimeConfig(duration_s=2.0 * DAY_S, output_dt_s=60.0, samples_per_period=120, max_points_cap=200_000)
    assert_allclose(tc.duration_days, 2.0, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError):
        TimeConfig(duration_s=0.0)
    with pytest.raises(ValueError):
        TimeConfig(output_dt_s=0.0)
    with pytest.raises(ValueError):
        TimeConfig(samples_per_period=1)
    with pytest.raises(ValueError):
        TimeConfig(max_points_cap=0)


def test_time_config_enforces_output_grid_cap_when_output_dt_is_set():
    # duration/output_dt = 100001 points -> exceeds 100000 cap
    with pytest.raises(ValueError):
        TimeConfig(duration_s=100000.0, output_dt_s=1.0, max_points_cap=100_000)


# =============================================================================
# 6) InitialState
# =============================================================================

def test_initial_state_array_helpers():
    st = InitialState(x=1.0, y=2.0, z=3.0, vx=4.0, vy=5.0, vz=6.0)
    a = st.to_array()
    r = st.r_vec()
    v = st.v_vec()

    assert a.shape == (6,)
    assert a.dtype == np.float64
    assert r.shape == (3,)
    assert v.shape == (3,)
    assert np.allclose(a, np.array([1, 2, 3, 4, 5, 6], dtype=np.float64))


# =============================================================================
# 7) EventConfig
# =============================================================================

def test_event_config_validation():
    _ = EventConfig()
    with pytest.raises(ValueError):
        EventConfig(impact_alt_km=-0.1)

    with pytest.raises(ValueError):
        EventConfig(detect_impact=False, impact_alt_km=1.0)


# =============================================================================
# 8) PropagatorConfig
# =============================================================================

def test_propagator_config_defaults_and_validation():
    pc = PropagatorConfig()
    assert hasattr(pc, "rtol") and hasattr(pc, "atol")
    assert pc.rtol > 0.0 and pc.atol > 0.0
    assert isinstance(pc.events, EventConfig)

    pc2 = PropagatorConfig(rtol=1e-9, atol=1e-11)
    assert pc2.rtol == 1e-9
    assert pc2.atol == 1e-11

    with pytest.raises(ValueError):
        PropagatorConfig(rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError):
        PropagatorConfig(rtol=1e-12, atol=0.0)

    with pytest.raises(ValueError):
        PropagatorConfig(heartbeat_hours=0.0)

    with pytest.raises(ValueError):
        PropagatorConfig(user_max_step_s=0.0)

    with pytest.raises(ValueError):
        PropagatorConfig(nyquist_safety_div=0.0)

    with pytest.raises(ValueError):
        PropagatorConfig(max_internal_steps=10)

    with pytest.raises(ValueError):
        PropagatorConfig(chunk_s=0.0)

    with pytest.raises(ValueError):
        PropagatorConfig(compute_2body_baseline=True, baseline_rtol=0.0)

    with pytest.raises(ValueError):
        PropagatorConfig(hybrid_switch_alt_m=-1.0)


# =============================================================================
# 9) SimulationHistory
# =============================================================================

def test_simulation_history_dtype_and_shape_validation():
    N = 5
    t_days = np.arange(N, dtype=np.float32)
    pos_km = np.zeros((N, 3), dtype=np.float32)
    vel_km_s = np.zeros((N, 3), dtype=np.float32)
    alt_km = np.zeros(N, dtype=np.float32)

    hist = SimulationHistory(t_days=t_days, pos_km=pos_km, vel_km_s=vel_km_s, alt_km=alt_km)
    assert hist.t_days.dtype == np.float64
    assert hist.pos_km.dtype == np.float64
    assert hist.pos_km.shape == (N, 3)

    with pytest.raises(ValueError):
        SimulationHistory(t_days=t_days, pos_km=np.zeros((N, 2)), vel_km_s=vel_km_s, alt_km=alt_km)

    with pytest.raises(ValueError):
        SimulationHistory(t_days=t_days, pos_km=pos_km, vel_km_s=np.zeros((N + 1, 3)), alt_km=alt_km)


# =============================================================================
# 10) PropagationResult + to_history
# =============================================================================

def test_propagation_result_shapes_and_to_history_contract():
    t = np.array([0.0, DAY_S], dtype=np.float32)

    y = np.array(
        [
            [R_MOON_MEAN + 1000.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
            [R_MOON_MEAN + 1000.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    res = PropagationResult(t=t, y=y)
    assert res.t.dtype == np.float64
    assert res.y.dtype == np.float64
    assert res.t.ndim == 1
    assert res.y.ndim == 2
    assert res.y_col.shape == (6, 2)
    assert np.allclose(res.y_col, res.y.T)

    with pytest.raises(ValueError):
        PropagationResult(t=np.zeros((2, 1)), y=y)  # t not 1D

    with pytest.raises(ValueError):
        PropagationResult(t=t, y=np.zeros(6))  # y not 2D

    with pytest.raises(ValueError):
        PropagationResult(t=np.zeros(3), y=y)  # length mismatch

    with pytest.raises(ValueError):
        PropagationResult(t=t, y=np.zeros((2, 5)))  # <6 states

    h = res.to_history(r_ref_m=R_MOON_MEAN)
    assert_allclose(float(h.t_days[1]), 1.0, atol=0.0, rtol=0.0)
    assert_allclose(float(h.alt_km[0]), 1.0, atol=1e-9, rtol=0.0)
    assert_allclose(float(h.vel_km_s[0, 0]), 1.0, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    import sys

    print("This is a pytest test module. Run it with:")
    print("python -m pytest -vv -rA --durations=10 tests/test_type_defs.py")
    sys.exit(0)
