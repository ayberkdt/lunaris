"""Time-grid, method-token, and gravity metadata helpers."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.integrator_methods import normalize_integrator_method
from lunaris.common.time_grid_contract import build_output_time_grid
from lunaris.core.dynamics import DynamicsEngine

logger = logging.getLogger(__name__)


def _norm_method(method: Any) -> str:
    """Normalize integrator method names to a stable canonical form."""
    return normalize_integrator_method(method)

def make_time_grid(t0: float, tf: float, dt: float) -> np.ndarray:
    t0 = float(t0)
    tf = float(tf)
    dt = float(dt)
    if (not np.isfinite(t0)) or (not np.isfinite(tf)) or (tf <= t0) or (dt <= 0.0) or (not np.isfinite(dt)):
        return np.array([t0, tf], dtype=np.float64)

    t_rel, _n_snaps, _snap = build_output_time_grid(tf - t0, dt)
    return np.ascontiguousarray(t0 + t_rel, dtype=np.float64)

def _clamp_output_dt(t0: float, tf: float, dt_out: float, cap: int, verbose: bool) -> float:
    dt = float(dt_out)
    if dt <= 0.0 or (not np.isfinite(dt)):
        raise ValueError("output_dt_s must be positive and finite.")
    if tf <= t0:
        return dt

    n = int(math.ceil((tf - t0) / dt)) + 1
    if n > int(cap):
        dt = (tf - t0) / max(2, int(cap) - 1)
        if verbose:
            logger.info("[OUT] max_points_cap exceeded -> increasing output_dt_s to %g s", dt)
    return dt

def get_ref_radius_and_mu(dynamics: DynamicsEngine) -> tuple[float, float]:
    """
    STRICT gravity SSOT:
      - grav.R_ref_m
      - grav.GM_m3s2
    Falls back to constants only if dynamics.grav is None (point-mass baseline).
    """
    grav = getattr(dynamics, "grav", None)
    if grav is None:
        return float(R_MOON), float(MU_MOON)

    missing = [name for name in ("R_ref_m", "GM_m3s2") if not hasattr(grav, name)]
    if missing:
        raise AttributeError(
            "Gravity model attached to DynamicsEngine is missing required strict attributes: "
            + ", ".join(missing)
            + ". Expected SSOT fields: R_ref_m, GM_m3s2."
        )

    return float(grav.R_ref_m), float(grav.GM_m3s2)

def get_sh_degree(dynamics: DynamicsEngine) -> int:
    """
    STRICT gravity SSOT:
      - grav.degree_max
    Returns 1 only if no gravity model is attached.
    """
    grav = getattr(dynamics, "grav", None)
    if grav is None:
        return 1

    if not hasattr(grav, "degree_max"):
        raise AttributeError("Gravity model missing required strict attribute: degree_max.")

    d = int(grav.degree_max)
    return max(1, d)


__all__ = ["_norm_method", "make_time_grid", "_clamp_output_dt", "get_ref_radius_and_mu", "get_sh_degree"]
