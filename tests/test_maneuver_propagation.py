from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON, STANDARD_GRAVITY
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.propagation import ImpulsiveManeuver, ManeuverPlan, propagate
from lunaris.core.propagation.plans import apply_impulsive_maneuver, ric_to_inertial_dv


class _PointMassDynamics:
    grav = None
    ephem = None
    flags = None

    def build_rhs(self):
        def rhs(_t, y):
            state = np.asarray(y, dtype=np.float64)
            r = state[:3]
            rn = float(np.linalg.norm(r))
            out = np.empty_like(state)
            out[:3] = state[3:6]
            out[3:6] = -MU_MOON * r / rn**3
            if state.size == 7:
                out[6] = 0.0
            return out

        return rhs


def _cfg(**changes) -> PropagatorConfig:
    values = {
        "method": "DOP853",
        "rtol": 1.0e-12,
        "atol": 1.0e-13,
        "verbose": False,
        "use_nyquist_max_step": False,
        "events": EventConfig(detect_impact=False, enable_peri_apo_events=False),
    }
    values.update(changes)
    return PropagatorConfig(**values)


def test_impulse_mass_and_ric_frame_contracts() -> None:
    state = np.array([2.0e6, 0.0, 0.0, 0.0, 1.5e3, 0.0, 500.0])
    maneuver = ImpulsiveManeuver(10.0, (2.0, 3.0, 4.0), frame="ric", isp_s=320.0)
    dv = ric_to_inertial_dv(state, maneuver.dv_mps)
    np.testing.assert_allclose(dv, [2.0, 3.0, 4.0], rtol=0.0, atol=1.0e-15)

    post, record = apply_impulsive_maneuver(state, maneuver)
    expected_mass = 500.0 * math.exp(-math.sqrt(29.0) / (STANDARD_GRAVITY * 320.0))
    np.testing.assert_allclose(post[3:6], [2.0, 1503.0, 4.0], rtol=0.0, atol=1.0e-15)
    assert post[6] == pytest.approx(expected_mass, rel=1.0e-15)
    assert record["output_state_semantics"] == "post_burn_single_timestamp"


def test_ric_rejects_degenerate_orbit_geometry() -> None:
    state = np.array([2.0e6, 0.0, 0.0, 100.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="RIC frame is undefined"):
        ric_to_inertial_dv(state, (1.0, 0.0, 0.0))


def test_segmented_hohmann_transfer_reaches_target_circular_orbit() -> None:
    r1 = float(R_MOON + 100_000.0)
    r2 = float(R_MOON + 300_000.0)
    transfer_a = 0.5 * (r1 + r2)
    v1 = math.sqrt(MU_MOON / r1)
    v2 = math.sqrt(MU_MOON / r2)
    vt1 = math.sqrt(MU_MOON * (2.0 / r1 - 1.0 / transfer_a))
    vt2 = math.sqrt(MU_MOON * (2.0 / r2 - 1.0 / transfer_a))
    transfer_time = math.pi * math.sqrt(transfer_a**3 / MU_MOON)

    y0 = np.array([r1, 0.0, 0.0, 0.0, v1, 0.0])
    plan = ManeuverPlan(
        (
            ImpulsiveManeuver(0.0, (0.0, vt1 - v1, 0.0), frame="ric"),
            ImpulsiveManeuver(transfer_time, (0.0, v2 - vt2, 0.0), frame="ric"),
        )
    )
    result = propagate(
        _PointMassDynamics(),
        y0,
        _cfg(),
        time_cfg=TimeConfig(duration_s=transfer_time, output_dt_s=transfer_time / 80.0),
        maneuver_plan=plan,
    )

    assert np.all(np.diff(result.t) > 0.0)
    assert np.count_nonzero(np.isclose(result.t, 0.0)) == 1
    assert np.count_nonzero(np.isclose(result.t, transfer_time)) == 1
    assert len(result.diagnostics["maneuvers_applied"]) == 2
    final = result.y[-1]
    radius = float(np.linalg.norm(final[:3]))
    speed = float(np.linalg.norm(final[3:6]))
    energy = 0.5 * speed**2 - MU_MOON / radius
    sma = -MU_MOON / (2.0 * energy)
    assert abs(sma - r2) / r2 < 2.0e-9
    assert abs(speed - v2) / v2 < 2.0e-9
    assert result.diagnostics["maneuver_output_contract"] == "single_timestamp_post_burn_v1"


def test_maneuver_checkpoint_and_strict_symplectic_combinations_fail_closed(tmp_path) -> None:
    y0 = np.array([R_MOON + 100_000.0, 0.0, 0.0, 0.0, 1600.0, 0.0])
    time_cfg = TimeConfig(duration_s=20.0, output_dt_s=5.0)
    plan = ManeuverPlan((ImpulsiveManeuver(10.0, (0.0, 1.0, 0.0)),))
    with pytest.raises(ValueError, match="checkpoint schema"):
        propagate(
            _PointMassDynamics(),
            y0,
            _cfg(checkpoint_path=str(tmp_path / "state.npz")),
            time_cfg=time_cfg,
            maneuver_plan=plan,
        )
    with pytest.raises(ValueError, match="strict_symplectic"):
        propagate(
            _PointMassDynamics(),
            y0,
            _cfg(method="VV", strict_symplectic=True),
            time_cfg=time_cfg,
            maneuver_plan=plan,
        )


def test_terminal_event_at_burn_time_takes_precedence() -> None:
    y0 = np.array([R_MOON + 100_000.0, 0.0, 0.0, 0.0, 1600.0, 0.0])

    def stop_at_burn(t, _y):
        return float(t) - 10.0

    stop_at_burn.terminal = True
    stop_at_burn.direction = 1.0
    result = propagate(
        _PointMassDynamics(),
        y0,
        _cfg(),
        time_cfg=TimeConfig(duration_s=20.0, output_dt_s=2.0),
        extra_events=[stop_at_burn],
        maneuver_plan=ManeuverPlan((ImpulsiveManeuver(10.0, (0.0, 5.0, 0.0)),)),
    )
    assert result.stopped_early
    assert result.diagnostics["maneuvers_applied"] == []
    assert result.diagnostics["maneuver_event_precedence"] == "terminal_event_before_burn"
