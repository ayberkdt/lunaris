from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.propagation.plans import (
    resolve_integration_plan,
    resolve_step_size_policy,
    resolve_time_grid_plan,
)


class _PointMassDynamics:
    grav = None


def _state(alt_km: float = 100.0) -> np.ndarray:
    r0 = float(R_MOON) + float(alt_km) * 1000.0
    v0 = float(np.sqrt(float(MU_MOON) / r0))
    return np.asarray([r0, 0.0, 0.0, 0.0, v0, 0.0], dtype=np.float64)


def _cfg(**kwargs) -> PropagatorConfig:
    base = dict(
        verbose=False,
        use_nyquist_max_step=True,
        compute_2body_baseline=False,
        events=EventConfig(detect_impact=False, enable_peri_apo_events=False),
    )
    base.update(kwargs)
    return PropagatorConfig(**base)


def test_time_grid_plan_uses_exact_final_epoch_and_uniform_realized_spacing() -> None:
    plan = resolve_time_grid_plan(
        dynamics=_PointMassDynamics(),
        y0=_state(),
        cfg=_cfg(),
        time_cfg=TimeConfig(duration_s=1000.0, output_dt_s=600.0, t0_s=50.0),
        verbose=False,
    )

    np.testing.assert_allclose(plan.t_eval, [50.0, 550.0, 1050.0])
    assert plan.tf == pytest.approx(1050.0)
    assert plan.realized_output_dt_s == pytest.approx(500.0)
    assert np.all(np.diff(plan.t_eval) > 0.0)


def test_time_grid_plan_enforces_max_point_cap() -> None:
    time_cfg = SimpleNamespace(duration_s=100.0, output_dt_s=1.0, t0_s=25.0, max_points_cap=10)

    plan = resolve_time_grid_plan(
        dynamics=_PointMassDynamics(),
        y0=_state(),
        cfg=_cfg(),
        time_cfg=time_cfg,
        verbose=False,
    )

    assert plan.t_eval.size <= 10
    assert plan.t_eval[0] == pytest.approx(25.0)
    assert plan.t_eval[-1] == pytest.approx(125.0)
    assert np.all(np.diff(plan.t_eval) > 0.0)


def test_step_size_policy_uses_nyquist_without_user_cap() -> None:
    calls: list[float] = []

    def fake_nyquist(**kwargs) -> float:
        calls.append(float(kwargs["r_min_alt_km"]))
        return 42.0

    plan = resolve_step_size_policy(
        cfg=_cfg(),
        y0=_state(500.0),
        R_ref_m=float(R_MOON),
        mu_m3s2=float(MU_MOON),
        sh_degree=100,
        output_dt_s=300.0,
        topo_present=False,
        nyquist_func=fake_nyquist,
    )

    assert calls == pytest.approx([500.0], rel=0.0, abs=1e-9)
    assert plan.actual_max_step_s == pytest.approx(42.0)
    assert plan.limiting_reason == "nyquist"
    assert plan.sh_degree == 100


def test_step_size_policy_user_cap_smaller_than_nyquist_wins() -> None:
    plan = resolve_step_size_policy(
        cfg=_cfg(user_max_step_s=12.0),
        y0=_state(),
        R_ref_m=float(R_MOON),
        mu_m3s2=float(MU_MOON),
        sh_degree=10,
        output_dt_s=300.0,
        topo_present=False,
        nyquist_func=lambda **_kwargs: 50.0,
    )

    assert plan.actual_max_step_s == pytest.approx(12.0)
    assert plan.limiting_reason == "user"


def test_step_size_policy_user_cap_larger_than_nyquist_is_limited() -> None:
    plan = resolve_step_size_policy(
        cfg=_cfg(user_max_step_s=120.0),
        y0=_state(),
        R_ref_m=float(R_MOON),
        mu_m3s2=float(MU_MOON),
        sh_degree=10,
        output_dt_s=300.0,
        topo_present=False,
        nyquist_func=lambda **_kwargs: 50.0,
    )

    assert plan.actual_max_step_s == pytest.approx(50.0)
    assert plan.limiting_reason == "nyquist"


def test_step_size_policy_invalid_nyquist_falls_back_to_output_dt() -> None:
    plan = resolve_step_size_policy(
        cfg=_cfg(),
        y0=_state(),
        R_ref_m=float(R_MOON),
        mu_m3s2=float(MU_MOON),
        sh_degree=10,
        output_dt_s=300.0,
        topo_present=False,
        nyquist_func=lambda **_kwargs: float("nan"),
    )

    assert plan.actual_max_step_s == pytest.approx(300.0)
    assert plan.nyquist_max_step_s == pytest.approx(300.0)
    assert plan.limiting_reason == "output_dt_fallback"


def test_integration_plan_distinguishes_fixed_step_and_chunked_scipy() -> None:
    fixed = resolve_integration_plan(_cfg(method="VV"), duration_s=100.0)
    assert fixed.backend == "fixed_step"
    assert fixed.method == "VV"
    assert fixed.chunk_s is None

    chunked = resolve_integration_plan(_cfg(method="DOP853", chunk_s=25.0), duration_s=100.0)
    assert chunked.backend == "scipy"
    assert chunked.method == "DOP853"
    assert chunked.chunk_s == pytest.approx(25.0)

    too_large = resolve_integration_plan(_cfg(method="DOP853", chunk_s=1000.0), duration_s=100.0)
    assert too_large.chunk_s is None
