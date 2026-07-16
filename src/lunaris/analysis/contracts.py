"""Typed contracts for post-propagation mission analysis.

The reporting layer must never be the owner of a scientific calculation.  The
objects in this module are the boundary between full-resolution analysis and
all presentation/export surfaces (PDF, Markdown, JSON, CSV, and the desktop
UI).  Source arrays remain SI; display-unit conversion belongs to reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ANALYSIS_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 1

_METRIC_STATUSES = frozenset({"ok", "warning", "critical", "unavailable"})
_METRIC_KINDS = frozenset(
    {
        "measured",
        "derived",
        "invariant",
        "diagnostic",
        "configuration",
    }
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One scalar/string metric with explicit scientific interpretation."""

    metric_id: str
    label: str
    value: float | int | str | bool | None
    unit: str | None
    status: str
    source: str
    kind: str = "derived"
    frame: str | None = None
    time_system: str | None = None
    interpretation: str | None = None
    availability_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.metric_id.strip():
            raise ValueError("metric_id cannot be empty")
        if self.status not in _METRIC_STATUSES:
            raise ValueError(f"unsupported metric status: {self.status!r}")
        if self.kind not in _METRIC_KINDS:
            raise ValueError(f"unsupported metric kind: {self.kind!r}")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(f"metric {self.metric_id!r} cannot contain NaN or Inf")
        if self.value is None and self.status != "unavailable":
            raise ValueError(
                f"metric {self.metric_id!r} has no value and must be unavailable"
            )
        if self.value is None and not self.availability_reason:
            raise ValueError(
                f"metric {self.metric_id!r} requires an availability_reason"
            )
        if isinstance(self.value, int | float) and self.unit is None:
            raise ValueError(f"numeric metric {self.metric_id!r} requires a unit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "label": self.label,
            "value": _json_value(self.value),
            "unit": self.unit,
            "status": self.status,
            "source": self.source,
            "kind": self.kind,
            "frame": self.frame,
            "time_system": self.time_system,
            "interpretation": self.interpretation,
            "availability_reason": self.availability_reason,
        }


@dataclass(frozen=True, slots=True)
class AnalysisEvent:
    """A chronological event tied to a state in the integration frame."""

    event_id: str
    event_type: str
    simulation_time_s: float
    epoch_utc: str | None
    state_m_mps: tuple[float, ...] | None
    altitude_m: float | None
    frame: str
    source: str
    severity: str = "normal"
    note: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.simulation_time_s)):
            raise ValueError("event simulation_time_s must be finite")
        if self.altitude_m is not None and not math.isfinite(float(self.altitude_m)):
            raise ValueError("event altitude_m must be finite when available")
        if self.state_m_mps is not None:
            if len(self.state_m_mps) < 6 or not all(
                math.isfinite(float(value)) for value in self.state_m_mps
            ):
                raise ValueError("event state must contain at least six finite values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "simulation_time_s": self.simulation_time_s,
            "epoch_utc": self.epoch_utc,
            "state_m_mps": _json_value(self.state_m_mps),
            "altitude_m": self.altitude_m,
            "frame": self.frame,
            "source": self.source,
            "severity": self.severity,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class ForceContribution:
    """Full-run magnitude statistics for one engine-reported force component."""

    force_id: str
    label: str
    active: bool
    available: bool
    minimum_m_s2: float | None
    median_m_s2: float | None
    p95_m_s2: float | None
    maximum_m_s2: float | None
    sample_count: int
    source: str
    included_in_noncentral_ranking: bool = True
    availability_reason: str | None = None
    interpretation: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.minimum_m_s2,
            self.median_m_s2,
            self.p95_m_s2,
            self.maximum_m_s2,
        )
        if any(value is not None and not math.isfinite(float(value)) for value in values):
            raise ValueError(f"force contribution {self.force_id!r} contains NaN or Inf")
        if self.available and any(value is None for value in values):
            raise ValueError(
                f"available force contribution {self.force_id!r} needs all statistics"
            )
        if not self.available and not self.availability_reason:
            raise ValueError(
                f"unavailable force contribution {self.force_id!r} needs a reason"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_id": self.force_id,
            "label": self.label,
            "active": self.active,
            "available": self.available,
            "minimum_m_s2": self.minimum_m_s2,
            "median_m_s2": self.median_m_s2,
            "p95_m_s2": self.p95_m_s2,
            "maximum_m_s2": self.maximum_m_s2,
            "sample_count": self.sample_count,
            "source": self.source,
            "included_in_noncentral_ranking": self.included_in_noncentral_ranking,
            "availability_reason": self.availability_reason,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True, slots=True)
class OrbitSeries:
    """Full-resolution derived series in SI units."""

    t_s: np.ndarray
    state_m_mps: np.ndarray
    semi_major_axis_m: np.ndarray
    eccentricity: np.ndarray
    inclination_rad: np.ndarray
    raan_rad: np.ndarray
    argument_of_periapsis_rad: np.ndarray
    true_anomaly_rad: np.ndarray
    altitude_m: np.ndarray
    radius_m: np.ndarray
    speed_m_s: np.ndarray
    specific_energy_j_kg: np.ndarray
    angular_momentum_m2_s: np.ndarray
    eccentricity_vector_norm: np.ndarray
    eclipse_mask: np.ndarray | None = None
    latitude_rad: np.ndarray | None = None
    longitude_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.t_s, dtype=np.float64)
        state = np.asarray(self.state_m_mps, dtype=np.float64)
        if t.ndim != 1 or state.ndim != 2 or state.shape[0] != t.size or state.shape[1] < 6:
            raise ValueError("OrbitSeries requires t[N] and row-major state[N,>=6]")
        for name in (
            "semi_major_axis_m",
            "eccentricity",
            "inclination_rad",
            "raan_rad",
            "argument_of_periapsis_rad",
            "true_anomaly_rad",
            "altitude_m",
            "radius_m",
            "speed_m_s",
            "specific_energy_j_kg",
            "angular_momentum_m2_s",
            "eccentricity_vector_norm",
        ):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1 or arr.size != t.size:
                raise ValueError(f"OrbitSeries.{name} must be a length-N vector")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(state[:, :6])):
            raise ValueError("OrbitSeries source time/state must be finite")


@dataclass(frozen=True, slots=True)
class OrbitAnalysisResult:
    """Canonical analysis result consumed by every report/export surface."""

    run_id: str
    preset: str
    generated_at_utc: str
    frame: str
    time_system: str
    metrics: tuple[MetricValue, ...]
    events: tuple[AnalysisEvent, ...]
    series: OrbitSeries
    force_contributions: tuple[ForceContribution, ...] = ()
    force_time_s: np.ndarray | None = None
    force_magnitudes_m_s2: dict[str, np.ndarray] = field(default_factory=dict)
    force_vectors_m_s2: dict[str, np.ndarray] = field(default_factory=dict)
    force_ric_m_s2: dict[str, np.ndarray] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    diagnostics_snapshot: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: int = ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.preset not in {"quick", "standard", "paper"}:
            raise ValueError(f"unsupported report preset: {self.preset!r}")
        ids = [metric.metric_id for metric in self.metrics]
        if len(ids) != len(set(ids)):
            raise ValueError("metric_id values must be unique")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique")
        if list(self.events) != sorted(
            self.events,
            key=lambda event: (event.simulation_time_s, event.event_id),
        ):
            raise ValueError("events must be sorted chronologically")
        if self.force_time_s is not None:
            ft = np.asarray(self.force_time_s, dtype=np.float64)
            if ft.ndim != 1 or not np.all(np.isfinite(ft)):
                raise ValueError("force_time_s must be a finite 1D array")
            for name, values in self.force_magnitudes_m_s2.items():
                arr = np.asarray(values, dtype=np.float64)
                if arr.ndim != 1 or arr.size != ft.size:
                    raise ValueError(f"force series {name!r} does not align with force_time_s")
                if np.any(np.isinf(arr)):
                    raise ValueError(f"force series {name!r} contains infinity")
            for collection_name, collection in (
                ("inertial force vector", self.force_vectors_m_s2),
                ("RIC force vector", self.force_ric_m_s2),
            ):
                for name, values in collection.items():
                    arr = np.asarray(values, dtype=np.float64)
                    if arr.ndim != 2 or arr.shape != (ft.size, 3):
                        raise ValueError(
                            f"{collection_name} series {name!r} must have shape (N, 3)"
                        )
                    if np.any(np.isinf(arr)):
                        raise ValueError(f"{collection_name} series {name!r} contains infinity")

    @property
    def metric_map(self) -> dict[str, MetricValue]:
        return {metric.metric_id: metric for metric in self.metrics}

    def metrics_payload(self) -> dict[str, Any]:
        return {
            "analysis_schema_version": self.schema_version,
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "preset": self.preset,
            "generated_at_utc": self.generated_at_utc,
            "frame": self.frame,
            "time_system": self.time_system,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "force_contributions": [item.to_dict() for item in self.force_contributions],
            "warnings": list(self.warnings),
        }


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "AnalysisEvent",
    "ForceContribution",
    "MetricValue",
    "OrbitAnalysisResult",
    "OrbitSeries",
]
