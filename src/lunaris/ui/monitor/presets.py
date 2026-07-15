"""Workspace presets for the Mission Monitor (Qt-free definitions).

A preset is a named, ordered list of widget ids. Presets may reference widgets
that are declared but not yet implemented (reserved specs): the workspace
opens the implemented ones and reports the skipped ones honestly — it never
fills the gap with decorative placeholders pretending to be data.
"""

from __future__ import annotations

from dataclasses import dataclass

from lunaris.ui.monitor.registry import MonitorWidgetRegistry, MonitorWidgetSpec


@dataclass(frozen=True, slots=True)
class MonitorPreset:
    preset_id: str
    title: str
    description: str
    widget_ids: tuple[str, ...]


PRESETS: tuple[MonitorPreset, ...] = (
    MonitorPreset(
        "orbit_overview",
        "Orbit Overview",
        "Trajectory-first view: orbit, altitude, elements and events.",
        ("orbit_view", "altitude", "orbital_elements", "event_timeline"),
    ),
    MonitorPreset(
        "numerical_health",
        "Numerical Health",
        "Integrator behaviour, state inspection and events.",
        ("integrator_health", "state_vector", "event_timeline", "invariant_monitor"),
    ),
    MonitorPreset(
        "force_model",
        "Force Model Monitor",
        "Force budget with altitude context and provenance.",
        ("force_budget", "altitude", "backend_provenance", "event_timeline"),
    ),
    MonitorPreset(
        "batch_ensemble",
        "Batch / Ensemble",
        "Ensemble progress and provenance (batch channels land in a later phase).",
        ("batch_progress", "backend_provenance", "event_timeline"),
    ),
    MonitorPreset(
        "st_lrps",
        "ST-LRPS",
        "Surrogate-backend observability (domain status lands in a later phase).",
        ("backend_provenance", "st_lrps_domain", "integrator_health", "event_timeline"),
    ),
)

DEFAULT_PRESET_ID = "orbit_overview"


def preset_by_id(preset_id: str) -> MonitorPreset | None:
    for preset in PRESETS:
        if preset.preset_id == preset_id:
            return preset
    return None


def split_preset(
    preset: MonitorPreset,
    registry: MonitorWidgetRegistry,
) -> tuple[tuple[MonitorWidgetSpec, ...], tuple[str, ...]]:
    """(openable specs, skipped widget ids) for a preset against a registry.

    Skipped ids are either unknown (removed widget) or reserved (declared but
    not implemented in this build); the workspace reports them to the user.
    """
    openable: list[MonitorWidgetSpec] = []
    skipped: list[str] = []
    for widget_id in preset.widget_ids:
        spec = registry.get(widget_id)
        if spec is not None and spec.implemented:
            openable.append(spec)
        else:
            skipped.append(widget_id)
    return tuple(openable), tuple(skipped)


__all__ = ["DEFAULT_PRESET_ID", "PRESETS", "MonitorPreset", "preset_by_id", "split_preset"]
