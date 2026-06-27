"""Compatibility shim for the historical import path.

The canonical implementation lives in ``lunaris.core.propagation``. The original
``lunaris.core.propagator`` module exposed a single flat namespace, so this shim
reconstructs that surface.
"""

from __future__ import annotations

from lunaris.core.propagation.propagator import (
    propagate,
    PropagationResult,
)
from lunaris.core.propagation.events import (
    build_events,
    _find_event_index,
    _build_r_i_to_bf_from_rot_table,
)
from lunaris.core.propagation.time_grid import (
    make_time_grid,
    _norm_method,
    _get_ref_radius_and_mu,
    _get_sh_degree,
    _clamp_output_dt,
)
from lunaris.core.propagation.telemetry import _make_telem_dict
from lunaris.core.propagation.integrators.fixed_step import (
    _ACCEL_METHODS,
    _RHS_METHODS,
    _accel_stepper,
    _is_fixed_step_method,
    _is_symplectic_method,
)
from lunaris.core.propagation.integrators.rk import (
    _rk4_step_full,
    _rk8_step_full,
)
from lunaris.core.propagation.integrators.symplectic import (
    _Y4_WEIGHTS,
    _Y6_WEIGHTS,
    _Y8_WEIGHTS,
    _composition_weights,
)
from scipy.integrate import solve_ivp

__all__ = [
    "propagate",
    "PropagationResult",
    "build_events",
    "make_time_grid",
    "_get_ref_radius_and_mu",
    "_get_sh_degree",
    "_find_event_index",
    "_build_r_i_to_bf_from_rot_table",
    "_ACCEL_METHODS",
    "_RHS_METHODS",
    "_Y4_WEIGHTS",
    "_Y6_WEIGHTS",
    "_Y8_WEIGHTS",
    "_accel_stepper",
    "_composition_weights",
    "_clamp_output_dt",
    "_is_fixed_step_method",
    "_is_symplectic_method",
    "_make_telem_dict",
    "_norm_method",
    "_rk4_step_full",
    "_rk8_step_full",
    "solve_ivp",
]
