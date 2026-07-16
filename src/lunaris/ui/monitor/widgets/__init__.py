"""Mission Monitor widgets: import-time registration into the default registry.

Importing this package registers every implemented widget spec plus the
reserved (declared, not yet implemented) specs, so the workspace, presets and
layout restore all see one consistent catalogue.
"""

from __future__ import annotations

from lunaris.ui.monitor.registry import DEFAULT_REGISTRY, MonitorWidgetSpec
from lunaris.ui.monitor.widgets.altitude import ALTITUDE_SPEC
from lunaris.ui.monitor.widgets.backend_provenance import BACKEND_PROVENANCE_SPEC
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame
from lunaris.ui.monitor.widgets.event_timeline import EVENT_TIMELINE_SPEC
from lunaris.ui.monitor.widgets.integrator_health import INTEGRATOR_HEALTH_SPEC
from lunaris.ui.monitor.widgets.orbit_view import ORBIT_VIEW_SPEC
from lunaris.ui.monitor.widgets.orbital_elements import ORBITAL_ELEMENTS_SPEC
from lunaris.ui.monitor.widgets.state_vector import STATE_VECTOR_SPEC

#: Widgets that are part of the roadmap (presets may reference them) but have
#: no implementation yet. They restore as honest placeholders — never as
#: decorative stand-ins with fake content.
RESERVED_SPECS: tuple[MonitorWidgetSpec, ...] = (
    MonitorWidgetSpec(
        widget_id="invariant_monitor",
        title="Invariant Monitor",
        category="Numerics",
        description="Energy / angular-momentum drift diagnostics (Phase 4+).",
        required_channels=("state_inertial",),
        factory=None,
    ),
    MonitorWidgetSpec(
        widget_id="force_budget",
        title="Force Contribution",
        category="Forces",
        description="Per-force acceleration budget (opt-in producer channel; Phase 6).",
        required_channels=("force_components",),
        factory=None,
    ),
    MonitorWidgetSpec(
        widget_id="batch_progress",
        title="Batch Progress",
        category="Batch",
        description="Ensemble completion / impact counters from [BATCH_PROGRESS] (Phase 6).",
        required_channels=("batch_progress",),
        factory=None,
    ),
    MonitorWidgetSpec(
        widget_id="st_lrps_domain",
        title="ST-LRPS Domain Status",
        category="ST-LRPS",
        description="Surrogate domain / OOD indicators (Phase 6).",
        required_channels=("provenance",),
        factory=None,
    ),
)


def register_all_specs() -> None:
    """Idempotently register implemented + reserved specs."""
    for spec in (
        ORBIT_VIEW_SPEC,
        ALTITUDE_SPEC,
        ORBITAL_ELEMENTS_SPEC,
        STATE_VECTOR_SPEC,
        INTEGRATOR_HEALTH_SPEC,
        EVENT_TIMELINE_SPEC,
        BACKEND_PROVENANCE_SPEC,
        *RESERVED_SPECS,
    ):
        if DEFAULT_REGISTRY.get(spec.widget_id) is None:
            DEFAULT_REGISTRY.register(spec)


register_all_specs()

__all__ = [
    "ALTITUDE_SPEC",
    "BACKEND_PROVENANCE_SPEC",
    "EVENT_TIMELINE_SPEC",
    "INTEGRATOR_HEALTH_SPEC",
    "ORBITAL_ELEMENTS_SPEC",
    "ORBIT_VIEW_SPEC",
    "RESERVED_SPECS",
    "STATE_VECTOR_SPEC",
    "MonitorWidgetFrame",
    "register_all_specs",
]
