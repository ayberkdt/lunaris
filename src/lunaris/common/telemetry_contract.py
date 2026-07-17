"""Typed, versioned telemetry contract shared by the propagation engine and the UI.

The Mission Monitor consumes structured telemetry lines emitted by the
propagation subprocess on ``stdout``:

* ``[TELEMETRY_META] {json}`` — once per run, carries :class:`TelemetryProvenance`
  (backends, integrator, gravity model, hashes) so provenance never has to be
  repeated on every sample.
* ``[TELEMETRY] {json}`` — one :class:`TelemetrySample` per emitted sample
  (schema ``lunaris_telemetry_v1``). ``sample_kind`` declares what each sample
  scientifically is: an accepted integrator state, an output-grid state, or a
  transient ``rhs_probe`` solver observation.

This module is the single source of truth for that wire format: the producer
(:mod:`lunaris.core.propagation`) encodes through it and the consumer
(:mod:`lunaris.ui.monitor`) decodes through it. It lives in ``lunaris.common``
so both sides share one type without adding an import edge between ``core`` and
``ui`` (``common`` imports neither).

Design rules
------------
* stdlib-only (no numpy) — ``common`` stays dependency-light.
* Frozen dataclasses; constructors are strict (invalid values raise), while the
  decode helpers sanitize untrusted payloads *before* construction so a
  malformed line can never produce a half-valid sample.
* Finite-value policy: encoding drops non-finite floats (JSON is emitted with
  ``allow_nan=False``); decoding maps non-finite floats to "channel missing"
  (``None`` / dropped mapping entry) instead of fake zeros.
* Unknown ``schema_version`` fails closed via
  :class:`UnsupportedTelemetrySchemaError` — the UI surfaces a warning instead
  of guessing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

TELEMETRY_SCHEMA_VERSION = "lunaris_telemetry_v1"
TELEMETRY_SAMPLE_PREFIX = "[TELEMETRY]"
TELEMETRY_META_PREFIX = "[TELEMETRY_META]"
#: File name of the optional per-run telemetry artifact (newline-delimited
#: JSON: one meta line followed by the persisted sample lines). The artifact
#: is a *subset* of the stdout stream: transient ``rhs_probe`` samples are
#: emitted on stdout only and never written here (see
#: :meth:`lunaris.core.propagation.telemetry_emitter` / the replay loader).
TELEMETRY_ARTIFACT_NAME = "telemetry.ndjson"

#: Diagnostic value types allowed on the wire.
DiagnosticValue = float | int | str | bool

#: Scientific meaning of a structured sample.  ``legacy_unknown`` is decoder-
#: only: current producers must always declare one of the first three values,
#: while old v1/bare-JSON records are kept readable without pretending that an
#: historical RHS-cadence sample was an accepted trajectory state.
SampleKind = Literal["accepted_state", "output_state", "rhs_probe", "legacy_unknown"]
SCIENTIFIC_SAMPLE_KINDS: frozenset[SampleKind] = frozenset(
    {"accepted_state", "output_state"}
)
_SAMPLE_KINDS: frozenset[str] = frozenset(
    {"accepted_state", "output_state", "rhs_probe", "legacy_unknown"}
)

#: Time keys accepted from legacy (pre-v1) bare-JSON telemetry lines, in
#: priority order. Mirrors the historical desktop-UI parser.
LEGACY_TIME_KEYS = ("t_s", "time_s", "t_sec", "t")


class TelemetryDecodeError(ValueError):
    """A telemetry payload could not be decoded into the typed contract."""


class UnsupportedTelemetrySchemaError(TelemetryDecodeError):
    """The payload declares a schema version this build does not understand."""

    def __init__(self, found_version: object) -> None:
        self.found_version = str(found_version)
        super().__init__(
            f"Unsupported telemetry schema version {self.found_version!r} "
            f"(supported: {TELEMETRY_SCHEMA_VERSION!r})."
        )


# =============================================================================
# Typed payloads
# =============================================================================

@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """One discrete mission event (periapsis pass, impact, fallback, stop...)."""

    event_type: str
    simulation_time_s: float
    message: str = ""
    severity: str = "info"  # "info" | "warning" | "critical"

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("TelemetryEvent.event_type must be non-empty")
        if not math.isfinite(self.simulation_time_s):
            raise ValueError(
                f"TelemetryEvent.simulation_time_s must be finite, got {self.simulation_time_s!r}"
            )
        if self.severity not in ("info", "warning", "critical"):
            raise ValueError(f"TelemetryEvent.severity invalid: {self.severity!r}")


@dataclass(frozen=True, slots=True)
class TelemetryProvenance:
    """Run-level provenance, emitted once per run on the meta line.

    Every field except ``run_id`` is optional: a channel that the active
    backend cannot provide is *absent*, never faked. The UI renders absent
    fields as "Unavailable".
    """

    run_id: str
    schema_version: str = TELEMETRY_SCHEMA_VERSION
    requested_backend: str | None = None
    effective_backend: str | None = None
    device: str | None = None
    integrator: str | None = None
    gravity_backend: str | None = None
    gravity_model: str | None = None
    sh_degree: int | None = None
    adaptive_degree: bool | None = None
    st_lrps_artifact: str | None = None
    fallback_reason: str | None = None
    config_sha256: str | None = None
    git_commit: str | None = None
    frame_inertial: str | None = None
    time_system: str = "relative_s"
    reference_radius_m: float | None = None
    mu_m3s2: float | None = None
    telemetry_cadence_s: float | None = None
    replay_policy: str | None = None
    data_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("TelemetryProvenance.run_id must be non-empty")
        object.__setattr__(self, "data_hashes", dict(self.data_hashes))


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One structured telemetry sample.

    ``sample_kind`` declares what the sample scientifically is (accepted
    integrator state, output-grid state, or transient RHS probe) and has no
    default: every producer must state the kind explicitly, so a new producer
    cannot silently promote solver probes to trajectory science. Only the
    decoder maps a *missing* kind (pre-``sample_kind`` artifacts) to
    ``legacy_unknown``.

    All physical quantities use SI (m, m/s, s, rad). Unit conversion is a
    presentation-layer concern. Optional channels are ``None`` (scalars),
    absent mapping keys (``orbital_elements`` / ``diagnostics``) or empty
    tuples (``events``) — never zero-filled placeholders.
    """

    run_id: str
    sequence_id: int
    simulation_time_s: float
    sample_kind: SampleKind
    schema_version: str = TELEMETRY_SCHEMA_VERSION
    wall_time_s: float | None = None
    epoch_et_s: float | None = None
    time_system: str = "relative_s"
    frame_inertial: str | None = None
    frame_fixed: str | None = None
    state_inertial: tuple[float, float, float, float, float, float] | None = None
    state_fixed: tuple[float, float, float, float, float, float] | None = None
    radius_m: float | None = None
    altitude_m: float | None = None
    speed_m_s: float | None = None
    surface_radius_m: float | None = None
    terrain_clearance_m: float | None = None
    orbital_elements: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, DiagnosticValue] = field(default_factory=dict)
    events: tuple[TelemetryEvent, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("TelemetrySample.run_id must be non-empty")
        if self.sequence_id < 0:
            raise ValueError(f"TelemetrySample.sequence_id must be >= 0, got {self.sequence_id!r}")
        if not math.isfinite(self.simulation_time_s):
            raise ValueError(
                f"TelemetrySample.simulation_time_s must be finite, got {self.simulation_time_s!r}"
            )
        if self.sample_kind not in _SAMPLE_KINDS:
            raise ValueError(f"TelemetrySample.sample_kind invalid: {self.sample_kind!r}")
        for name in ("state_inertial", "state_fixed"):
            state = getattr(self, name)
            if state is None:
                continue
            state_t = tuple(float(x) for x in state)
            if len(state_t) != 6 or not all(math.isfinite(x) for x in state_t):
                raise ValueError(f"TelemetrySample.{name} must be 6 finite floats, got {state!r}")
            object.__setattr__(self, name, state_t)
        for name in ("wall_time_s", "epoch_et_s", "radius_m", "altitude_m", "speed_m_s",
                     "surface_radius_m", "terrain_clearance_m"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"TelemetrySample.{name} must be finite or None, got {value!r}")
        elements = dict(self.orbital_elements)
        for key, value in elements.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"TelemetrySample.orbital_elements[{key!r}] must be finite, got {value!r}"
                )
        object.__setattr__(self, "orbital_elements", elements)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "events", tuple(self.events))


# =============================================================================
# Encoding (producer side)
# =============================================================================

def _put_finite(payload: dict[str, Any], key: str, value: float | None) -> None:
    if value is not None and math.isfinite(value):
        payload[key] = float(value)


def sample_to_payload(sample: TelemetrySample) -> dict[str, Any]:
    """Flatten a sample into a JSON-safe dict, dropping absent/non-finite channels."""
    payload: dict[str, Any] = {
        "schema_version": sample.schema_version,
        "run_id": sample.run_id,
        "sequence_id": int(sample.sequence_id),
        "simulation_time_s": float(sample.simulation_time_s),
        "sample_kind": sample.sample_kind,
        "time_system": sample.time_system,
    }
    _put_finite(payload, "wall_time_s", sample.wall_time_s)
    _put_finite(payload, "epoch_et_s", sample.epoch_et_s)
    _put_finite(payload, "radius_m", sample.radius_m)
    _put_finite(payload, "altitude_m", sample.altitude_m)
    _put_finite(payload, "speed_m_s", sample.speed_m_s)
    _put_finite(payload, "surface_radius_m", sample.surface_radius_m)
    _put_finite(payload, "terrain_clearance_m", sample.terrain_clearance_m)
    if sample.frame_inertial:
        payload["frame_inertial"] = sample.frame_inertial
    if sample.frame_fixed:
        payload["frame_fixed"] = sample.frame_fixed
    if sample.state_inertial is not None:
        payload["state_inertial"] = [float(x) for x in sample.state_inertial]
    if sample.state_fixed is not None:
        payload["state_fixed"] = [float(x) for x in sample.state_fixed]
    elements = {k: float(v) for k, v in sample.orbital_elements.items() if math.isfinite(v)}
    if elements:
        payload["orbital_elements"] = elements
    diagnostics: dict[str, DiagnosticValue] = {}
    for key, value in sample.diagnostics.items():
        if isinstance(value, bool | int | str) or (
            isinstance(value, float) and math.isfinite(value)
        ):
            diagnostics[key] = value
    if diagnostics:
        payload["diagnostics"] = diagnostics
    if sample.events:
        payload["events"] = [
            {
                "event_type": e.event_type,
                "simulation_time_s": float(e.simulation_time_s),
                "message": e.message,
                "severity": e.severity,
            }
            for e in sample.events
        ]
    return payload


def encode_sample_line(sample: TelemetrySample) -> str:
    """Encode one sample as a complete ``[TELEMETRY] {json}`` stdout line."""
    body = json.dumps(
        sample_to_payload(sample), separators=(",", ":"), sort_keys=True, allow_nan=False
    )
    return f"{TELEMETRY_SAMPLE_PREFIX} {body}"


def provenance_to_payload(provenance: TelemetryProvenance) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": provenance.schema_version,
        "run_id": provenance.run_id,
        "time_system": provenance.time_system,
    }
    for key in (
        "requested_backend", "effective_backend", "device", "integrator",
        "gravity_backend", "gravity_model", "st_lrps_artifact", "fallback_reason",
        "config_sha256", "git_commit", "frame_inertial", "replay_policy",
    ):
        value = getattr(provenance, key)
        if value is not None:
            payload[key] = str(value)
    if provenance.sh_degree is not None:
        payload["sh_degree"] = int(provenance.sh_degree)
    if provenance.adaptive_degree is not None:
        payload["adaptive_degree"] = bool(provenance.adaptive_degree)
    _put_finite(payload, "reference_radius_m", provenance.reference_radius_m)
    _put_finite(payload, "mu_m3s2", provenance.mu_m3s2)
    _put_finite(payload, "telemetry_cadence_s", provenance.telemetry_cadence_s)
    if provenance.data_hashes:
        payload["data_hashes"] = {str(k): str(v) for k, v in provenance.data_hashes.items()}
    return payload


def encode_meta_line(provenance: TelemetryProvenance) -> str:
    """Encode run provenance as a complete ``[TELEMETRY_META] {json}`` line."""
    body = json.dumps(
        provenance_to_payload(provenance), separators=(",", ":"), sort_keys=True, allow_nan=False
    )
    return f"{TELEMETRY_META_PREFIX} {body}"


# =============================================================================
# Decoding (consumer side) — sanitize untrusted payloads before construction
# =============================================================================

def _require_version(payload: Mapping[str, Any]) -> str:
    version = payload.get("schema_version")
    if version != TELEMETRY_SCHEMA_VERSION:
        raise UnsupportedTelemetrySchemaError(version)
    return str(version)


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def _state6_or_none(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, list | tuple) or len(value) != 6:
        return None
    out: list[float] = []
    for item in value:
        item_f = _finite_or_none(item)
        if item_f is None:
            return None
        out.append(item_f)
    return tuple(out)


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _decode_events(value: object) -> tuple[TelemetryEvent, ...]:
    if not isinstance(value, list):
        return ()
    events: list[TelemetryEvent] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        event_type = _str_or_none(item.get("event_type"))
        t_s = _finite_or_none(item.get("simulation_time_s"))
        if event_type is None or t_s is None:
            continue
        severity = item.get("severity")
        if severity not in ("info", "warning", "critical"):
            severity = "info"
        message = item.get("message")
        events.append(
            TelemetryEvent(
                event_type=event_type,
                simulation_time_s=t_s,
                message=message if isinstance(message, str) else "",
                severity=severity,
            )
        )
    return tuple(events)


def sample_from_payload(payload: Mapping[str, Any]) -> TelemetrySample:
    """Build a sample from an untrusted decoded-JSON payload.

    Raises :class:`UnsupportedTelemetrySchemaError` for a foreign schema and
    :class:`TelemetryDecodeError` when required fields are unusable. Optional
    channels that fail sanitization are dropped, never zero-filled.
    """
    version = _require_version(payload)
    run_id = _str_or_none(payload.get("run_id"))
    if run_id is None:
        raise TelemetryDecodeError("telemetry sample is missing run_id")
    sequence_raw = payload.get("sequence_id")
    if isinstance(sequence_raw, bool) or not isinstance(sequence_raw, int) or sequence_raw < 0:
        raise TelemetryDecodeError(f"telemetry sample has invalid sequence_id: {sequence_raw!r}")
    t_s = _finite_or_none(payload.get("simulation_time_s"))
    if t_s is None:
        raise TelemetryDecodeError("telemetry sample is missing a finite simulation_time_s")
    sample_kind_raw = payload.get("sample_kind")
    if sample_kind_raw is None:
        # Historical v1 artifacts were emitted from an RHS cadence gate.  They
        # remain decodable, but their samples are deliberately not promoted to
        # accepted/output trajectory science.
        sample_kind: SampleKind = "legacy_unknown"
    elif isinstance(sample_kind_raw, str) and sample_kind_raw in _SAMPLE_KINDS:
        sample_kind = cast(SampleKind, sample_kind_raw)
    else:
        raise TelemetryDecodeError(
            f"telemetry sample has invalid sample_kind: {sample_kind_raw!r}"
        )

    elements_raw = payload.get("orbital_elements")
    elements: dict[str, float] = {}
    if isinstance(elements_raw, dict):
        for key, value in elements_raw.items():
            value_f = _finite_or_none(value)
            if isinstance(key, str) and value_f is not None:
                elements[key] = value_f

    diagnostics_raw = payload.get("diagnostics")
    diagnostics: dict[str, DiagnosticValue] = {}
    if isinstance(diagnostics_raw, dict):
        for key, value in diagnostics_raw.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, bool | int | str):
                diagnostics[key] = value
            else:
                value_f = _finite_or_none(value)
                if value_f is not None:
                    diagnostics[key] = value_f

    time_system = _str_or_none(payload.get("time_system")) or "relative_s"
    state_inertial = _state6_or_none(payload.get("state_inertial"))
    state_fixed = _state6_or_none(payload.get("state_fixed"))
    return TelemetrySample(
        run_id=run_id,
        sequence_id=sequence_raw,
        simulation_time_s=t_s,
        sample_kind=sample_kind,
        schema_version=version,
        wall_time_s=_finite_or_none(payload.get("wall_time_s")),
        epoch_et_s=_finite_or_none(payload.get("epoch_et_s")),
        time_system=time_system,
        frame_inertial=_str_or_none(payload.get("frame_inertial")),
        frame_fixed=_str_or_none(payload.get("frame_fixed")),
        state_inertial=state_inertial,  # type: ignore[arg-type]
        state_fixed=state_fixed,  # type: ignore[arg-type]
        radius_m=_finite_or_none(payload.get("radius_m")),
        altitude_m=_finite_or_none(payload.get("altitude_m")),
        speed_m_s=_finite_or_none(payload.get("speed_m_s")),
        surface_radius_m=_finite_or_none(payload.get("surface_radius_m")),
        terrain_clearance_m=_finite_or_none(payload.get("terrain_clearance_m")),
        orbital_elements=elements,
        diagnostics=diagnostics,
        events=_decode_events(payload.get("events")),
    )


def provenance_from_payload(payload: Mapping[str, Any]) -> TelemetryProvenance:
    """Build run provenance from an untrusted decoded-JSON meta payload."""
    version = _require_version(payload)
    run_id = _str_or_none(payload.get("run_id"))
    if run_id is None:
        raise TelemetryDecodeError("telemetry meta is missing run_id")
    sh_degree_raw = payload.get("sh_degree")
    sh_degree = (
        sh_degree_raw
        if isinstance(sh_degree_raw, int) and not isinstance(sh_degree_raw, bool)
        else None
    )
    adaptive_raw = payload.get("adaptive_degree")
    adaptive_degree = adaptive_raw if isinstance(adaptive_raw, bool) else None
    hashes_raw = payload.get("data_hashes")
    data_hashes: dict[str, str] = {}
    if isinstance(hashes_raw, dict):
        data_hashes = {
            str(k): str(v) for k, v in hashes_raw.items() if isinstance(v, str | int | float)
        }
    return TelemetryProvenance(
        run_id=run_id,
        schema_version=version,
        requested_backend=_str_or_none(payload.get("requested_backend")),
        effective_backend=_str_or_none(payload.get("effective_backend")),
        device=_str_or_none(payload.get("device")),
        integrator=_str_or_none(payload.get("integrator")),
        gravity_backend=_str_or_none(payload.get("gravity_backend")),
        gravity_model=_str_or_none(payload.get("gravity_model")),
        sh_degree=sh_degree,
        adaptive_degree=adaptive_degree,
        st_lrps_artifact=_str_or_none(payload.get("st_lrps_artifact")),
        fallback_reason=_str_or_none(payload.get("fallback_reason")),
        config_sha256=_str_or_none(payload.get("config_sha256")),
        git_commit=_str_or_none(payload.get("git_commit")),
        frame_inertial=_str_or_none(payload.get("frame_inertial")),
        time_system=_str_or_none(payload.get("time_system")) or "relative_s",
        reference_radius_m=_finite_or_none(payload.get("reference_radius_m")),
        mu_m3s2=_finite_or_none(payload.get("mu_m3s2")),
        telemetry_cadence_s=_finite_or_none(payload.get("telemetry_cadence_s")),
        replay_policy=_str_or_none(payload.get("replay_policy")),
        data_hashes=data_hashes,
    )


def _strip_prefix(line: str, prefix: str) -> str:
    stripped = line.strip()
    if stripped.startswith(prefix):
        stripped = stripped[len(prefix):].strip()
    return stripped


def decode_sample_line(line: str) -> TelemetrySample:
    """Decode a (possibly prefixed) sample line; raises TelemetryDecodeError."""
    body = _strip_prefix(line, TELEMETRY_SAMPLE_PREFIX)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TelemetryDecodeError(f"telemetry sample line is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TelemetryDecodeError("telemetry sample payload must be a JSON object")
    return sample_from_payload(payload)


def decode_meta_line(line: str) -> TelemetryProvenance:
    """Decode a (possibly prefixed) meta line; raises TelemetryDecodeError."""
    body = _strip_prefix(line, TELEMETRY_META_PREFIX)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TelemetryDecodeError(f"telemetry meta line is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TelemetryDecodeError("telemetry meta payload must be a JSON object")
    return provenance_from_payload(payload)


# =============================================================================
# Legacy adapter (pre-v1 bare-JSON lines: t_s / alt_km / v_km_s / ecc / ...)
# =============================================================================

def _legacy_time_s(payload: Mapping[str, Any]) -> float | None:
    for key in LEGACY_TIME_KEYS:
        if key not in payload:
            continue
        t_s = _finite_or_none(payload[key])
        if t_s is None:
            return None
        if key == "t":
            unit = str(payload.get("t_unit", "")).strip().lower()
            if unit.startswith("h"):
                t_s *= 3600.0
            elif unit.startswith("d"):
                t_s *= 86400.0
        return t_s
    return None


def _legacy_km_to_m(payload: Mapping[str, Any], key: str) -> float | None:
    value = _finite_or_none(payload.get(key))
    return None if value is None else value * 1000.0


def sample_from_legacy_dict(
    payload: Mapping[str, Any],
    *,
    run_id: str = "legacy",
    sequence_id: int,
) -> TelemetrySample | None:
    """Map a legacy bare-JSON telemetry dict onto the v1 contract.

    Legacy lines carry no schema/run/sequence information, so the caller
    supplies a synthetic ``run_id`` and a monotonically increasing
    ``sequence_id``. Returns ``None`` when the dict has no usable time field
    (the line was probably not telemetry at all).
    """
    t_s = _legacy_time_s(payload)
    if t_s is None:
        return None
    elements: dict[str, float] = {}
    ecc = _finite_or_none(payload.get("ecc"))
    if ecc is not None:
        elements["ecc"] = ecc
    speed_km_s = _finite_or_none(payload.get("v_km_s"))
    return TelemetrySample(
        run_id=run_id,
        sequence_id=sequence_id,
        simulation_time_s=t_s,
        sample_kind="legacy_unknown",
        altitude_m=_legacy_km_to_m(payload, "alt_km"),
        speed_m_s=None if speed_km_s is None else speed_km_s * 1000.0,
        surface_radius_m=_legacy_km_to_m(payload, "surface_r_km"),
        terrain_clearance_m=_legacy_km_to_m(payload, "terrain_clearance_km"),
        orbital_elements=elements,
    )


__all__ = [
    "LEGACY_TIME_KEYS",
    "TELEMETRY_ARTIFACT_NAME",
    "TELEMETRY_META_PREFIX",
    "TELEMETRY_SAMPLE_PREFIX",
    "TELEMETRY_SCHEMA_VERSION",
    "DiagnosticValue",
    "SCIENTIFIC_SAMPLE_KINDS",
    "SampleKind",
    "TelemetryDecodeError",
    "TelemetryEvent",
    "TelemetryProvenance",
    "TelemetrySample",
    "UnsupportedTelemetrySchemaError",
    "decode_meta_line",
    "decode_sample_line",
    "encode_meta_line",
    "encode_sample_line",
    "provenance_from_payload",
    "provenance_to_payload",
    "sample_from_legacy_dict",
    "sample_from_payload",
    "sample_to_payload",
]
