"""Bounded, sequence-aware telemetry store feeding the Mission Monitor widgets.

One store instance holds the telemetry of exactly one run (live or replay).
Design contract:

* **Bounded** — every channel lives in a pre-sized ring buffer; a multi-day
  run can never grow UI memory past ``capacity`` samples. Full-resolution
  science stays in the run artifacts, never here.
* **Qt-free, single-writer** — the store has no locks because all writes come
  from the Qt main thread (QProcess ``readyRead`` handlers and queued replay
  batches are both delivered there). Snapshots return copies, so widgets can
  never observe a half-written ring.
* **Sequence-honest** — duplicates, out-of-order and gap counts are tracked
  from ``sequence_id`` and surfaced instead of silently papered over.
* **Availability-honest** — a channel that never received a value yields an
  empty snapshot; missing values inside a live channel are NaN internally and
  are filtered out of snapshots. Nothing is zero-filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from lunaris.common.telemetry_contract import (
    TelemetryEvent,
    TelemetryProvenance,
    TelemetrySample,
)
from lunaris.ui.monitor.channels import ELEMENT_CHANNEL_PREFIX, SCALAR_SAMPLE_FIELDS
from lunaris.ui.monitor.downsample import decimate_indices, envelope_downsample

StoreMode = Literal["idle", "live", "replay"]
AppendStatus = Literal[
    "appended", "rhs_probe", "uncertain_sample", "duplicate", "out_of_order", "foreign_run"
]

DEFAULT_CAPACITY = 50_000
DEFAULT_MAX_EVENTS = 10_000


@dataclass
class StoreCounters:
    """Bookkeeping the UI can surface (sequence health, drops)."""

    accepted: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    #: Number of samples known to be missing (from sequence_id gaps).
    gap_samples: int = 0
    #: Samples rejected because they belong to a different run_id.
    foreign_run: int = 0
    events_dropped: int = 0
    rhs_probes: int = 0
    uncertain_samples: int = 0


@dataclass(frozen=True)
class RunOutcome:
    """How the observed run ended (exit code / stop reason, if known)."""

    reason: str = ""
    exit_code: int | None = None
    success: bool | None = None


@dataclass
class _ChannelBuffer:
    values: np.ndarray
    #: True once at least one finite value was written (channel "exists").
    seen: bool = False


def _new_ring(capacity: int) -> np.ndarray:
    return np.full(capacity, np.nan, dtype=np.float64)


@dataclass
class TelemetryStore:
    capacity: int = DEFAULT_CAPACITY
    max_events: int = DEFAULT_MAX_EVENTS

    mode: StoreMode = "idle"
    run_id: str | None = None
    provenance: TelemetryProvenance | None = None
    counters: StoreCounters = field(default_factory=StoreCounters)
    latest_sample: TelemetrySample | None = None
    #: Most recent transient solver observation; never copied into trajectory buffers.
    latest_probe: TelemetrySample | None = None
    #: Latest per-sample diagnostics, merged key-wise as samples arrive.
    latest_diagnostics: dict[str, Any] = field(default_factory=dict)
    #: End-of-run engine diagnostics ([DIAG] payload), set once when available.
    run_diagnostics: dict[str, Any] | None = None
    outcome: RunOutcome | None = None
    #: Optional hint (seconds) of the configured run duration, for progress/ETA.
    expected_duration_s: float | None = None

    def __post_init__(self) -> None:
        if self.capacity < 2:
            raise ValueError(f"capacity must be >= 2, got {self.capacity!r}")
        self._t = _new_ring(self.capacity)
        self._wall = _new_ring(self.capacity)
        self._scalars: dict[str, _ChannelBuffer] = {}
        self._states: dict[str, np.ndarray] = {}
        self._stored = 0
        self._last_seq: int | None = None
        self._last_probe_seq: int | None = None
        self._events: list[TelemetryEvent] = []
        self._event_keys: set[tuple[str, float, str]] = set()

    # ------------------------------------------------------------------ runs
    def begin_run(
        self,
        run_id: str,
        *,
        mode: StoreMode = "live",
        expected_duration_s: float | None = None,
    ) -> None:
        """Reset all buffers and start collecting a new run."""
        self.__post_init__()
        self.mode = mode
        self.run_id = run_id
        self.provenance = None
        self.counters = StoreCounters()
        self.latest_sample = None
        self.latest_probe = None
        self.latest_diagnostics = {}
        self.run_diagnostics = None
        self.outcome = None
        self.expected_duration_s = expected_duration_s

    def reset(self) -> None:
        """Drop everything and return to idle."""
        self.begin_run("", mode="idle")
        self.run_id = None

    def set_provenance(self, provenance: TelemetryProvenance) -> None:
        self.provenance = provenance
        # Meta arriving before any sample fixes the authoritative run id.
        if self._stored == 0:
            self.run_id = provenance.run_id

    def set_run_diagnostics(self, payload: dict[str, Any]) -> None:
        self.run_diagnostics = dict(payload)

    def finish_run(self, outcome: RunOutcome | None = None) -> None:
        self.outcome = outcome or RunOutcome()

    def enter_replay(self) -> None:
        """Switch a finished live session (or a fresh store) to replay mode."""
        self.mode = "replay"

    # --------------------------------------------------------------- appends
    def append(self, sample: TelemetrySample) -> AppendStatus:
        if self.run_id in (None, ""):
            self.run_id = sample.run_id
            if self.mode == "idle":
                self.mode = "live"
        if sample.run_id != self.run_id:
            if self._stored == 0 and self.provenance is None:
                self.run_id = sample.run_id
            else:
                self.counters.foreign_run += 1
                return "foreign_run"

        if sample.sample_kind == "rhs_probe":
            self.latest_probe = sample
            self._last_probe_seq = sample.sequence_id
            self.counters.rhs_probes += 1
            for key, value in sample.diagnostics.items():
                self.latest_diagnostics[key] = value
            for event in sample.events:
                self.add_event(event)
            return "rhs_probe"
        if sample.sample_kind == "legacy_unknown":
            # Old v1 samples may be rejected RK stages/RHS probes.  Preserve
            # observability of their existence without feeding trajectory,
            # cursor, state-vector, altitude, or element channels.
            self.counters.uncertain_samples += 1
            return "uncertain_sample"

        seq = sample.sequence_id
        if self._last_seq is not None:
            if seq == self._last_seq:
                self.counters.duplicates += 1
                return "duplicate"
            if seq < self._last_seq:
                self.counters.out_of_order += 1
                return "out_of_order"
            if seq > self._last_seq + 1:
                self.counters.gap_samples += seq - self._last_seq - 1
        self._last_seq = seq

        pos = self._stored % self.capacity
        self._t[pos] = sample.simulation_time_s
        self._wall[pos] = np.nan if sample.wall_time_s is None else sample.wall_time_s
        # A ring slot being overwritten must not leak the evicted sample's
        # values into channels this sample does not carry.
        for buffer in self._scalars.values():
            buffer.values[pos] = np.nan
        for state in self._states.values():
            state[pos, :] = np.nan

        for channel_id, attr in SCALAR_SAMPLE_FIELDS.items():
            value = getattr(sample, attr)
            if value is not None:
                self._write_scalar(channel_id, pos, float(value))
        for key, value in sample.orbital_elements.items():
            self._write_scalar(f"{ELEMENT_CHANNEL_PREFIX}{key}", pos, float(value))
        if sample.state_inertial is not None:
            self._write_state("state_inertial", pos, sample.state_inertial)
        if sample.state_fixed is not None:
            self._write_state("state_fixed", pos, sample.state_fixed)

        for key, value in sample.diagnostics.items():
            self.latest_diagnostics[key] = value
        for event in sample.events:
            self.add_event(event)

        self.latest_sample = sample
        self._stored += 1
        self.counters.accepted += 1
        return "appended"

    def extend(self, samples: list[TelemetrySample] | tuple[TelemetrySample, ...]) -> int:
        """Append many samples; returns how many were accepted."""
        accepted = 0
        for sample in samples:
            if self.append(sample) == "appended":
                accepted += 1
        return accepted

    def add_event(self, event: TelemetryEvent) -> bool:
        """Record a discrete event with duplicate suppression. True if kept."""
        key = (event.event_type, round(event.simulation_time_s, 6), event.message)
        if key in self._event_keys:
            return False
        if len(self._events) >= self.max_events:
            self.counters.events_dropped += 1
            return False
        self._event_keys.add(key)
        self._events.append(event)
        return True

    def _write_scalar(self, channel_id: str, pos: int, value: float) -> None:
        buffer = self._scalars.get(channel_id)
        if buffer is None:
            buffer = _ChannelBuffer(values=_new_ring(self.capacity))
            self._scalars[channel_id] = buffer
        buffer.values[pos] = value
        if np.isfinite(value):
            buffer.seen = True

    def _write_state(self, channel_id: str, pos: int, state: tuple[float, ...]) -> None:
        table = self._states.get(channel_id)
        if table is None:
            table = np.full((self.capacity, 6), np.nan, dtype=np.float64)
            self._states[channel_id] = table
        table[pos, :] = state

    # -------------------------------------------------------------- queries
    @property
    def n_samples(self) -> int:
        return min(self._stored, self.capacity)

    @property
    def total_appended(self) -> int:
        return self.counters.accepted

    def _ordered_indices(self) -> np.ndarray:
        if self._stored <= self.capacity:
            return np.arange(self._stored, dtype=np.int64)
        head = self._stored % self.capacity
        return np.concatenate(
            (np.arange(head, self.capacity, dtype=np.int64),
             np.arange(head, dtype=np.int64))
        )

    def times(self) -> np.ndarray:
        """Simulation times of the retained samples, oldest to newest (copy)."""
        return self._t[self._ordered_indices()].copy()

    def has_channel(self, channel_id: str) -> bool:
        if channel_id in ("events",):
            return bool(self._events)
        if channel_id in ("diagnostics",):
            return bool(self.latest_diagnostics) or bool(self.run_diagnostics)
        if channel_id in ("provenance",):
            return self.provenance is not None
        if channel_id in self._states:
            return True
        buffer = self._scalars.get(channel_id)
        return buffer is not None and buffer.seen

    def available_channels(self) -> tuple[str, ...]:
        channels = [cid for cid, buf in self._scalars.items() if buf.seen]
        channels.extend(self._states.keys())
        if self._events:
            channels.append("events")
        if self.latest_diagnostics or self.run_diagnostics:
            channels.append("diagnostics")
        if self.provenance is not None:
            channels.append("provenance")
        return tuple(sorted(channels))

    def snapshot(
        self,
        channel_id: str,
        *,
        max_points: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(t, value) copies for one scalar channel, NaN-filtered and ordered.

        An unknown or never-seen channel yields two empty arrays — the widget
        renders its "channel unavailable" state, never fake zeros.
        """
        buffer = self._scalars.get(channel_id)
        if buffer is None or self._stored == 0:
            return np.empty(0), np.empty(0)
        order = self._ordered_indices()
        t = self._t[order]
        v = buffer.values[order]
        mask = np.isfinite(t) & np.isfinite(v)
        t, v = t[mask], v[mask]
        if max_points is not None and t.shape[0] > max_points:
            return envelope_downsample(t, v, max_points)
        return t.copy(), v.copy()

    def snapshot_state(
        self,
        channel_id: str = "state_inertial",
        *,
        max_points: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(t, states[n,6]) copies for a state channel, row-complete rows only."""
        table = self._states.get(channel_id)
        if table is None or self._stored == 0:
            return np.empty(0), np.empty((0, 6))
        order = self._ordered_indices()
        t = self._t[order]
        y = table[order, :]
        mask = np.isfinite(t) & np.all(np.isfinite(y), axis=1)
        t, y = t[mask], y[mask]
        if max_points is not None and t.shape[0] > max_points:
            idx = decimate_indices(int(t.shape[0]), max_points)
            return t[idx].copy(), y[idx].copy()
        return t.copy(), y.copy()

    def value_at_or_before(self, channel_id: str, t_s: float) -> float | None:
        """Latest finite value of a channel at or before ``t_s`` (replay cursor)."""
        t, v = self.snapshot(channel_id)
        if t.shape[0] == 0:
            return None
        idx = int(np.searchsorted(t, t_s, side="right")) - 1
        if idx < 0:
            return None
        return float(v[idx])

    def state_at_or_before(
        self, t_s: float, channel_id: str = "state_inertial"
    ) -> tuple[float, np.ndarray] | None:
        """(t, state6) of the latest complete state at or before ``t_s``."""
        t, y = self.snapshot_state(channel_id)
        if t.shape[0] == 0:
            return None
        idx = int(np.searchsorted(t, t_s, side="right")) - 1
        if idx < 0:
            return None
        return float(t[idx]), y[idx].copy()

    def events(self) -> tuple[TelemetryEvent, ...]:
        """All recorded events ordered by simulation time (stable)."""
        return tuple(sorted(self._events, key=lambda e: e.simulation_time_s))

    def time_bounds(self) -> tuple[float, float] | None:
        """(t_min, t_max) over retained samples, or None when empty."""
        if self._stored == 0:
            return None
        order = self._ordered_indices()
        t = self._t[order]
        t = t[np.isfinite(t)]
        if t.shape[0] == 0:
            return None
        return float(t[0]), float(t[-1])


__all__ = [
    "DEFAULT_CAPACITY",
    "DEFAULT_MAX_EVENTS",
    "AppendStatus",
    "RunOutcome",
    "StoreCounters",
    "StoreMode",
    "TelemetryStore",
]
