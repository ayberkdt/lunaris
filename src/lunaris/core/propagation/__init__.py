"""Public facade for propagation orchestration and planning contracts."""

from __future__ import annotations

from lunaris.common.type_defs import PropagationResult

from .events import EventOutcome, EventSpec, build_events, event_outcome_from_solver_events
from .plans import (
    ImpulsiveManeuver,
    IntegrationPlan,
    ManeuverPlan,
    StepSizePlan,
    TimeGridPlan,
    resolve_integration_plan,
    resolve_step_size_policy,
    resolve_time_grid_plan,
)
from .propagator import propagate
from .time_grid import make_time_grid

__all__ = [
    "propagate",
    "PropagationResult",
    "EventOutcome",
    "EventSpec",
    "TimeGridPlan",
    "StepSizePlan",
    "IntegrationPlan",
    "ImpulsiveManeuver",
    "ManeuverPlan",
    "build_events",
    "make_time_grid",
    "resolve_time_grid_plan",
    "resolve_step_size_policy",
    "resolve_integration_plan",
    "event_outcome_from_solver_events",
]
