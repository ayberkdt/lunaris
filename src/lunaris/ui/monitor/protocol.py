"""Stdout-line classification for the Mission Monitor (Qt-free).

The desktop UI already assembles complete stdout lines per stream via
:class:`lunaris.ui.core.log_stream.LineAssembler` (stdout and stderr never
share an assembler, so fragments cannot interleave). This module takes those
*complete* lines and decides whether each one is monitor telemetry:

* ``[TELEMETRY_META] {json}`` → :class:`MetaMessage`
* ``[TELEMETRY] {json}``      → :class:`SampleMessage`
* legacy bare-JSON telemetry (``{"t_s": ...}``) → :class:`SampleMessage`
  through the v1 legacy adapter
* a *prefixed* line that fails to decode → :class:`ProtocolProblem`
  (malformed or unsupported schema — fail-closed, surfaced to the user)
* anything else → ``None`` (an ordinary log line; the caller keeps routing it
  to the Execution Console exactly as before)

Malformed **legacy-looking** lines return ``None`` rather than a problem:
arbitrary JSON in log output must not raise telemetry alarms. Only lines that
claim to be telemetry (via prefix) are held to the contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from lunaris.common.telemetry_contract import (
    LEGACY_TIME_KEYS,
    TELEMETRY_META_PREFIX,
    TELEMETRY_SAMPLE_PREFIX,
    TelemetryDecodeError,
    TelemetryProvenance,
    TelemetrySample,
    UnsupportedTelemetrySchemaError,
    decode_meta_line,
    decode_sample_line,
    sample_from_legacy_dict,
)


@dataclass(frozen=True, slots=True)
class SampleMessage:
    sample: TelemetrySample
    legacy: bool = False


@dataclass(frozen=True, slots=True)
class MetaMessage:
    provenance: TelemetryProvenance


@dataclass(frozen=True, slots=True)
class ProtocolProblem:
    kind: Literal["malformed", "unsupported_schema"]
    detail: str
    line: str


TelemetryMessage = SampleMessage | MetaMessage | ProtocolProblem

#: Quick membership probes reused from the legacy desktop parser: a bare JSON
#: object is only *considered* telemetry when it names a known time key.
_LEGACY_KEY_TOKENS = tuple(f'"{key}"' for key in LEGACY_TIME_KEYS) + tuple(
    f"'{key}'" for key in LEGACY_TIME_KEYS
)


@dataclass
class ProtocolCounters:
    samples: int = 0
    metas: int = 0
    legacy_samples: int = 0
    malformed: int = 0
    unsupported_schema: int = 0


class TelemetryLineClassifier:
    """Stateful classifier for one telemetry stream.

    State is limited to what legacy lines lack: the run id and a synthetic,
    monotonically increasing sequence id. ``begin_run`` must be called when a
    new process starts so legacy samples land in the right store run.
    """

    def __init__(self, *, legacy_run_id: str = "legacy") -> None:
        self.counters = ProtocolCounters()
        self._legacy_run_id = legacy_run_id
        self._legacy_seq = 0

    def begin_run(self, legacy_run_id: str) -> None:
        self._legacy_run_id = legacy_run_id
        self._legacy_seq = 0

    def classify(self, line: str) -> TelemetryMessage | None:
        stripped = line.strip()
        if not stripped:
            return None
        if stripped.startswith(TELEMETRY_META_PREFIX):
            return self._decode_meta(stripped)
        if stripped.startswith(TELEMETRY_SAMPLE_PREFIX):
            return self._decode_sample(stripped)
        if stripped.startswith("{") and any(token in stripped for token in _LEGACY_KEY_TOKENS):
            return self._decode_legacy(stripped)
        return None

    # ------------------------------------------------------------- internals
    def _decode_meta(self, line: str) -> TelemetryMessage:
        try:
            provenance = decode_meta_line(line)
        except UnsupportedTelemetrySchemaError as exc:
            self.counters.unsupported_schema += 1
            return ProtocolProblem("unsupported_schema", str(exc), line)
        except TelemetryDecodeError as exc:
            self.counters.malformed += 1
            return ProtocolProblem("malformed", str(exc), line)
        self.counters.metas += 1
        return MetaMessage(provenance)

    def _decode_sample(self, line: str) -> TelemetryMessage:
        try:
            sample = decode_sample_line(line)
        except UnsupportedTelemetrySchemaError as exc:
            self.counters.unsupported_schema += 1
            return ProtocolProblem("unsupported_schema", str(exc), line)
        except TelemetryDecodeError as exc:
            self.counters.malformed += 1
            return ProtocolProblem("malformed", str(exc), line)
        self.counters.samples += 1
        return SampleMessage(sample)

    def _decode_legacy(self, line: str) -> TelemetryMessage | None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        sample = sample_from_legacy_dict(
            payload, run_id=self._legacy_run_id, sequence_id=self._legacy_seq
        )
        if sample is None:
            return None
        self._legacy_seq += 1
        self.counters.samples += 1
        self.counters.legacy_samples += 1
        return SampleMessage(sample, legacy=True)


__all__ = [
    "MetaMessage",
    "ProtocolCounters",
    "ProtocolProblem",
    "SampleMessage",
    "TelemetryLineClassifier",
    "TelemetryMessage",
]
