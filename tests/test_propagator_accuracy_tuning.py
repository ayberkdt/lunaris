"""Accuracy-tuning surface of the propagator:

1. Per-component (vector) ``atol`` resolution for solve_ivp.
2. Energy / angular-momentum drift diagnostics (on the run and the 2-body baseline).
3. SH truncation-degree adequacy warning for the orbit periapsis altitude.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import (
    EventConfig,
    PerturbationFlags,
    PropagatorConfig,
    SpacecraftProps,
    TimeConfig,
)
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.propagation import propagator as prop_mod
from lunaris.core.propagation.propagator import _osculating_periapsis_alt_km, _resolve_atol

# ---------------------------------------------------------------------------
# 1) Vector atol
# ---------------------------------------------------------------------------

def test_resolve_atol_scalar_by_default():
    cfg = PropagatorConfig(atol=1e-12)
    out = _resolve_atol(cfg, 6)
    assert isinstance(out, float)
    assert out == 1e-12


def test_resolve_atol_builds_position_velocity_vector():
    cfg = PropagatorConfig(atol=1e-12, atol_pos=1e-6, atol_vel=1e-9)
    out = _resolve_atol(cfg, 6)
    assert isinstance(out, np.ndarray)
    assert out.shape == (6,)
    np.testing.assert_array_equal(out[0:3], 1e-6)
    np.testing.assert_array_equal(out[3:6], 1e-9)


def test_resolve_atol_augmented_state_keeps_scalar_for_extra_components():
    cfg = PropagatorConfig(atol=1e-12, atol_pos=1e-6, atol_vel=1e-9)
    out = _resolve_atol(cfg, 9)  # 6D state + 3 augmented (e.g. STM/surrogate) channels
    assert out.shape == (9,)
    np.testing.assert_array_equal(out[0:3], 1e-6)
    np.testing.assert_array_equal(out[3:6], 1e-9)
    np.testing.assert_array_equal(out[6:9], 1e-12)  # extra components keep scalar atol


def test_resolve_atol_only_one_of_pos_vel():
    cfg = PropagatorConfig(atol=1e-12, atol_pos=1e-6)  # vel left as scalar
    out = _resolve_atol(cfg, 6)
    np.testing.assert_array_equal(out[0:3], 1e-6)
    np.testing.assert_array_equal(out[3:6], 1e-12)


def test_propagator_config_rejects_nonpositive_component_atol():
    with pytest.raises(ValueError):
        PropagatorConfig(atol_pos=0.0)
    with pytest.raises(ValueError):
        PropagatorConfig(atol_vel=-1e-9)


# ---------------------------------------------------------------------------
# 2) Osculating periapsis for Nyquist cap and degree diagnostics
# ---------------------------------------------------------------------------

def _apoapsis_state_for_altitudes(peri_alt_km: float, apo_alt_km: float) -> np.ndarray:
    rp = float(R_MOON) + float(peri_alt_km) * 1000.0
    ra = float(R_MOON) + float(apo_alt_km) * 1000.0
    sma = 0.5 * (rp + ra)
    v_apo = float(np.sqrt(float(MU_MOON) * (2.0 / ra - 1.0 / sma)))
    return np.asarray([ra, 0.0, 0.0, 0.0, v_apo, 0.0], dtype=np.float64)


def test_osculating_periapsis_altitude_from_apoapsis_state():
    y0 = _apoapsis_state_for_altitudes(500.0, 2000.0)

    alt_km = _osculating_periapsis_alt_km(y0, float(MU_MOON), float(R_MOON))

    assert alt_km == pytest.approx(500.0, rel=0.0, abs=1e-9)


def test_nyquist_step_cap_uses_osculating_periapsis(monkeypatch):
    calls: list[float] = []

    def _fake_nyquist_max_step_s(**kwargs):
        calls.append(float(kwargs["r_min_alt_km"]))
        return 10_000.0

    monkeypatch.setattr(prop_mod, "nyquist_max_step_s", _fake_nyquist_max_step_s)
    dyn = DynamicsEngine(
        sc_props=SpacecraftProps(),
        flags=PerturbationFlags(enable_sh=False),
        allow_identity_rotation=True,
    )
    y0 = _apoapsis_state_for_altitudes(500.0, 2000.0)
    cfg = PropagatorConfig(
        verbose=False,
        compute_2body_baseline=False,
        events=EventConfig(detect_impact=False, enable_peri_apo_events=False),
    )

    res = prop_mod.propagate(dyn, y0, cfg, time_cfg=TimeConfig(duration_s=1.0, output_dt_s=1.0))

    assert calls == pytest.approx([500.0], rel=0.0, abs=1e-9)
    assert res.diagnostics["nyquist_r_min_alt_km"] == pytest.approx(500.0, rel=0.0, abs=1e-9)
