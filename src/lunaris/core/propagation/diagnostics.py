"""Diagnostics assembly for high-level propagation runs."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from lunaris.common.contracts.diagnostics import PROPAGATION_DIAGNOSTICS_SCHEMA_VERSION
from lunaris.common.math_utils import recommended_sh_degree, specific_energy_drift_stats
from lunaris.common.type_defs import PropagationResult, PropagatorConfig
from lunaris.core.propagation.integrators.fixed_step import _is_symplectic_method
from lunaris.core.propagation.plans import (
    IntegrationPlan,
    StepSizePlan,
    TimeGridPlan,
    _osculating_periapsis_alt_km,
)


def _uses_surrogate_python_autograd(dynamics: Any, rhs_path: str) -> bool:
    """True when a single-trajectory run is using the interpreted ST-LRPS path."""

    if rhs_path == "surrogate_python_autograd":
        return True
    grav = getattr(dynamics, "grav", None)
    return bool(
        getattr(grav, "model_kind", None) == "st_lrps"
        and "surrogate" in str(rhs_path)
    )


def build_propagation_diagnostics(
    *,
    dynamics: Any,
    cfg: PropagatorConfig,
    result: PropagationResult,
    time_plan: TimeGridPlan,
    step_plan: StepSizePlan,
    integration_plan: IntegrationPlan,
    degree: int,
    output_dt_s: float,
    max_step_s: float,
    wall_time_s: float,
    rhs_path: str,
    symplectic_violations: Sequence[str],
    y0: np.ndarray,
    R_ref_m: float,
    mu_m3s2: float,
    verbose: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Build the versioned diagnostics payload for a completed propagation."""

    nfev = (
        float(getattr(result.ode, "nfev", np.nan))
        if result.ode is not None
        else np.nan
    )
    diagnostics: dict[str, Any] = {
        "diagnostics_schema_version": PROPAGATION_DIAGNOSTICS_SCHEMA_VERSION,
        "wall_time_s": float(wall_time_s),
        "output_dt_s": float(output_dt_s),
        "requested_output_dt_s": time_plan.requested_output_dt_s,
        "realized_output_dt_s": time_plan.realized_output_dt_s,
        "output_grid_max_points_cap": float(time_plan.max_points_cap),
        "max_step_s": float(max_step_s),
        "requested_user_max_step_s": step_plan.user_max_step_s,
        "nyquist_max_step_s": float(step_plan.nyquist_max_step_s),
        "actual_max_step_s": float(step_plan.actual_max_step_s),
        "max_step_limiting_reason": step_plan.limiting_reason,
        "nyquist_r_min_alt_km": step_plan.nyquist_r_min_alt_km,
        "sh_degree_for_step_policy": float(step_plan.sh_degree),
        "periapsis_alt_km_for_step_policy": step_plan.periapsis_alt_km,
        "degree": float(degree),
        "n_points": float(result.t.size),
        "nfev": float(nfev) if np.isfinite(nfev) else float("nan"),
        "integration_backend": integration_plan.backend,
        "integrator": integration_plan.method,
        "integration_chunk_s": integration_plan.chunk_s,
        "rhs_path": rhs_path,
        "method_symplectic": float(
            1.0 if _is_symplectic_method(getattr(cfg, "method", "DOP853")) else 0.0
        ),
        "symplectic_violation": float(1.0 if symplectic_violations else 0.0),
    }
    if _uses_surrogate_python_autograd(dynamics, rhs_path):
        diagnostics["single_run_stlrps_cpu_warning"] = True
        diagnostics["benchmark_comparable_to_numba_sh"] = False
    if symplectic_violations:
        diagnostics["symplectic_violation_forces"] = list(symplectic_violations)

    # Energy / angular-momentum drift over the trajectory. This is a combined
    # physical+numerical drift on the full run (a bounded oscillation is physical;
    # a monotone secular trend signals numerical error).
    try:
        for key, value in specific_energy_drift_stats(
            result.t,
            result.y,
            float(mu_m3s2),
        ).items():
            diagnostics[key] = float(value)
    except Exception:
        # R29b-justified: optional diagnostics enrichment; the trajectory itself
        # is already computed and returned unchanged.
        pass

    # SH truncation-degree adequacy for the orbit's periapsis altitude. Below the
    # recommended degree, low-degree truncation (not the integrator) is the
    # dominant position-error term, so we surface it rather than let it pass.
    try:
        y0_arr = np.asarray(y0, dtype=np.float64)
        if int(degree) >= 2 and y0_arr.size >= 6 and float(mu_m3s2) > 0.0:
            alt_peri_km = _osculating_periapsis_alt_km(
                y0_arr,
                float(mu_m3s2),
                float(R_ref_m),
            )
            if alt_peri_km is not None:
                rec_deg = recommended_sh_degree(alt_peri_km, float(R_ref_m))
                diagnostics["periapsis_alt_km"] = float(alt_peri_km)
                diagnostics["recommended_degree"] = float(rec_deg)
                if rec_deg > int(degree):
                    deg_msg = (
                        f"SH truncation degree={int(degree)} may be too low for periapsis "
                        f"altitude {alt_peri_km:.1f} km: upward continuation suggests degree "
                        f">= {rec_deg} to retain gravity signal above the 1e-3 floor. "
                        "Low-degree truncation, not the integrator, is then the dominant "
                        "position-error term."
                    )
                    warnings.warn(deg_msg, RuntimeWarning, stacklevel=2)
                    if verbose:
                        logger.info("[GRAV] %s", deg_msg)
    except Exception:
        # R29b-justified: advisory degree-adequacy diagnostics only; failure
        # here never alters the propagated trajectory.
        pass

    return diagnostics


__all__ = ["build_propagation_diagnostics"]
