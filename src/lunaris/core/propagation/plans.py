"""Planning helpers for single-run propagation orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from lunaris.common.type_defs import PropagatorConfig, TimeConfig
from lunaris.core.propagation.events import _get_detect_impact, _get_impact_alt_km
from lunaris.core.propagation.integrators.fixed_step import _is_fixed_step_method
from lunaris.core.propagation.integrators.scipy import _resolve_scipy_method
from lunaris.core.propagation.time_grid import (
    _clamp_output_dt,
    _get_ref_radius_and_mu,
    make_time_grid,
)


@dataclass(frozen=True, slots=True)
class TimeGridPlan:
    """Resolved output-grid contract for a single propagation run."""

    t0: float
    tf: float
    duration_s: float
    requested_output_dt_s: float | None
    realized_output_dt_s: float
    t_eval: np.ndarray
    max_points_cap: int


@dataclass(frozen=True, slots=True)
class StepSizePlan:
    """Resolved internal max-step policy."""

    user_max_step_s: float | None
    nyquist_max_step_s: float
    actual_max_step_s: float
    limiting_reason: str
    sh_degree: int
    periapsis_alt_km: float | None
    nyquist_r_min_alt_km: float | None


@dataclass(frozen=True, slots=True)
class IntegrationPlan:
    """Resolved integration backend and normalized method token."""

    method: str
    backend: Literal["scipy", "fixed_step"]
    chunk_s: float | None


def _osculating_periapsis_alt_km(y: Any, mu_m3s2: float, R_ref_m: float) -> float | None:
    """Return osculating conic periapsis altitude from a Cartesian state."""
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        return None

    mu = float(mu_m3s2)
    R_ref = float(R_ref_m)
    if (not np.isfinite(mu)) or mu <= 0.0 or (not np.isfinite(R_ref)) or R_ref <= 0.0:
        return None

    r = arr[0:3]
    v = arr[3:6]
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        return None

    rn = float(np.linalg.norm(r))
    h_vec = np.cross(r, v)
    h2 = float(np.dot(h_vec, h_vec))
    if rn <= 0.0 or h2 <= 0.0:
        return None

    ecc_vec = np.cross(v, h_vec) / mu - r / rn
    ecc = float(np.linalg.norm(ecc_vec))
    if (not np.isfinite(ecc)) or ecc < 0.0:
        return None

    rp = (h2 / mu) / (1.0 + ecc)
    if (not np.isfinite(rp)) or rp <= 0.0:
        return None
    return float((rp - R_ref) / 1000.0)


def resolve_time_grid_plan(
    *,
    dynamics: Any,
    y0: np.ndarray,
    cfg: PropagatorConfig,
    time_cfg: TimeConfig | Any | None,
    verbose: bool,
) -> TimeGridPlan:
    """Resolve the output time grid from ``TimeConfig`` and cap policy."""
    if time_cfg is None:
        raise ValueError("time_cfg is required (STRICT). Provide TimeConfig(duration_s=..., output_dt_s=...).")

    if getattr(time_cfg, "duration_s", None) is None:
        raise ValueError("time_cfg.duration_s is required and must be finite/positive.")

    dur_s = float(time_cfg.duration_s)
    if dur_s <= 0.0 or (not np.isfinite(dur_s)):
        raise ValueError("Duration must be positive and finite.")

    t0 = float(getattr(time_cfg, "t0_s", 0.0) or 0.0)
    if not np.isfinite(t0):
        raise ValueError("time_cfg.t0_s must be finite.")
    tf = t0 + dur_s

    dt_out_raw = getattr(time_cfg, "output_dt_s", None)
    if dt_out_raw is None:
        _, mu = _get_ref_radius_and_mu(dynamics)
        mu = float(mu)

        r0 = float(np.linalg.norm(y0[:3]))
        v0 = float(np.linalg.norm(y0[3:6]))
        if not (math.isfinite(r0) and math.isfinite(v0) and r0 > 0.0 and mu > 0.0):
            raise ValueError("Cannot derive output_dt_s: invalid initial state or mu.")

        denom = (2.0 / r0) - (v0 * v0 / mu)
        if denom <= 0.0 or (not math.isfinite(denom)):
            raise ValueError(
                "time_cfg.output_dt_s is None, but the orbit appears unbound/degenerate. "
                "Set output_dt_s explicitly."
            )

        a = 1.0 / denom
        period_s = 2.0 * math.pi * math.sqrt((a * a * a) / mu)
        samples_per_period = int(getattr(time_cfg, "samples_per_period", 360) or 360)
        samples_per_period = max(1, samples_per_period)
        dt_out_user = float(period_s) / float(samples_per_period)
    else:
        dt_out_user = float(dt_out_raw)

    if dt_out_user <= 0.0 or (not np.isfinite(dt_out_user)):
        raise ValueError("time_cfg.output_dt_s must be positive and finite.")

    max_points_cap = int(getattr(time_cfg, "max_points_cap", getattr(cfg, "max_points_cap", 200_000)))
    dt_out = _clamp_output_dt(t0, tf, float(dt_out_user), max_points_cap, verbose)
    t_eval = make_time_grid(t0, tf, dt_out)
    diffs = np.diff(t_eval)
    realized_dt = float(diffs[0]) if diffs.size else float(dur_s)

    return TimeGridPlan(
        t0=float(t0),
        tf=float(tf),
        duration_s=float(dur_s),
        requested_output_dt_s=float(dt_out_user),
        realized_output_dt_s=realized_dt,
        t_eval=np.ascontiguousarray(t_eval, dtype=np.float64),
        max_points_cap=int(max_points_cap),
    )


def resolve_step_size_policy(
    *,
    cfg: PropagatorConfig,
    y0: np.ndarray,
    R_ref_m: float,
    mu_m3s2: float,
    sh_degree: int,
    output_dt_s: float,
    topo_present: bool,
    nyquist_func: Callable[..., float],
) -> StepSizePlan:
    """Resolve Nyquist/user max-step policy into an explicit plan."""
    degree = int(max(1, sh_degree))
    output_dt = float(output_dt_s)
    nyq_max: float | None = None
    nyq_r_min_alt_km: float | None = None
    peri_alt_km: float | None = None
    limiting_base = "output_dt_fallback"

    if bool(getattr(cfg, "use_nyquist_max_step", False)):
        try:
            guard_alt_km = float(_get_impact_alt_km(cfg) if _get_detect_impact(cfg) else 0.0)
            if topo_present:
                guard_alt_km = 0.0
            peri_alt_km = _osculating_periapsis_alt_km(y0, float(mu_m3s2), float(R_ref_m))
            nyq_r_min_alt_km = (
                max(float(guard_alt_km), float(peri_alt_km))
                if peri_alt_km is not None
                else float(guard_alt_km)
            )
            nyq_max = float(nyquist_func(
                R_ref_m=float(R_ref_m),
                mu_m3s2=float(mu_m3s2),
                degree=degree,
                r_min_alt_km=float(nyq_r_min_alt_km),
                safety_div=float(getattr(cfg, "nyquist_safety_div", 8.0)),
                v_margin=float(getattr(cfg, "nyquist_v_margin", 1.2)),
            ))
            limiting_base = "nyquist"
        except Exception:
            nyq_max = None
            limiting_base = "output_dt_fallback"

    if nyq_max is None or (not np.isfinite(nyq_max)) or nyq_max <= 0.0:
        nyq_max = output_dt
        limiting_base = "output_dt_fallback"

    user_raw = getattr(cfg, "user_max_step_s", None)
    if user_raw is None:
        user_max_step_s = None
        actual = float(nyq_max)
        limiting_reason = limiting_base
    else:
        try:
            user_max_step_s = float(user_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"user_max_step_s must be positive and finite, got {user_raw!r}"
            ) from exc
        if not np.isfinite(user_max_step_s) or user_max_step_s <= 0.0:
            raise ValueError(
                f"user_max_step_s must be positive and finite, got {user_max_step_s!r}"
            )
        if user_max_step_s <= float(nyq_max):
            actual = float(user_max_step_s)
            limiting_reason = "user"
        else:
            actual = float(nyq_max)
            limiting_reason = limiting_base

    return StepSizePlan(
        user_max_step_s=user_max_step_s,
        nyquist_max_step_s=float(nyq_max),
        actual_max_step_s=float(actual),
        limiting_reason=str(limiting_reason),
        sh_degree=degree,
        periapsis_alt_km=peri_alt_km,
        nyquist_r_min_alt_km=nyq_r_min_alt_km,
    )


def resolve_integration_plan(cfg: PropagatorConfig, *, duration_s: float) -> IntegrationPlan:
    """Resolve fixed-step vs SciPy dispatch and normalize chunk duration."""
    raw_method = str(getattr(cfg, "method", "DOP853"))
    if _is_fixed_step_method(raw_method):
        return IntegrationPlan(method=raw_method, backend="fixed_step", chunk_s=None)

    chunk_s = getattr(cfg, "chunk_s", None)
    if chunk_s is not None:
        chunk_s = float(chunk_s)
        if chunk_s <= 0.0 or chunk_s >= float(duration_s):
            chunk_s = None

    return IntegrationPlan(
        method=_resolve_scipy_method(raw_method),
        backend="scipy",
        chunk_s=(float(chunk_s) if chunk_s is not None else None),
    )


__all__ = [
    "IntegrationPlan",
    "StepSizePlan",
    "TimeGridPlan",
    "_osculating_periapsis_alt_km",
    "resolve_integration_plan",
    "resolve_step_size_policy",
    "resolve_time_grid_plan",
]
