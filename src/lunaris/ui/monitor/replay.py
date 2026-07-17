"""Replay: ndjson artifact loading (off the UI thread) and the timeline clock.

The replay artifact (``telemetry.ndjson`` in a run directory) is the disk
mirror of the stdout protocol: one ``[TELEMETRY_META]`` line followed by
``[TELEMETRY]`` sample lines. :class:`ReplayLoader` parses it in a worker
QThread and hands immutable payloads to the controller through queued
signals — the UI thread never blocks on file IO, and widgets never see the
loader.

:class:`TimelineController` is the single source of truth for the replay
cursor: play/pause, speed, stepping and event jumps all funnel through it into
``MonitorController.jump_to_time``, so every widget renders the same instant.
"""

from __future__ import annotations

from PySide6 import QtCore

from lunaris.common.telemetry_contract import (
    SCIENTIFIC_SAMPLE_KINDS,
    TELEMETRY_META_PREFIX,
    TELEMETRY_SAMPLE_PREFIX,
    TelemetryDecodeError,
    TelemetrySample,
    UnsupportedTelemetrySchemaError,
    decode_meta_line,
    decode_sample_line,
)

#: Samples delivered to the store per queued signal.
_BATCH_SIZE = 2000
#: Absolute cap on replay samples held in UI memory. Files beyond this are
#: truncated to the *first* HARD_CAP samples and the user is told explicitly.
REPLAY_SAMPLE_HARD_CAP = 200_000


class ReplayLoader(QtCore.QThread):
    """Streams one telemetry.ndjson artifact into sample batches."""

    #: Number of sample lines found (pre-scan), before parsing starts.
    count_ready = QtCore.Signal(int)
    meta_ready = QtCore.Signal(object)    # TelemetryProvenance
    batch_ready = QtCore.Signal(object)   # list[TelemetrySample]
    finished_ok = QtCore.Signal(int)      # delivered sample count
    failed = QtCore.Signal(str)           # human-readable, fail-closed reason
    warning = QtCore.Signal(str)          # readable artifact with reduced trust/content

    def __init__(self, path: str, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._path = str(path)

    def run(self) -> None:  # noqa: N802 (QThread API)
        try:
            self._run_inner()
        except OSError as exc:
            self.failed.emit(f"Run artifact could not be read: {exc}")
        except UnsupportedTelemetrySchemaError as exc:
            # Fail-closed: an artifact from a newer/foreign schema is not
            # partially guessed at.
            self.failed.emit(str(exc))

    def _run_inner(self) -> None:
        # Pass 1: count sample lines so the store can be sized before data
        # arrives (bounded memory is decided up front, not discovered late).
        sample_lines = 0
        probe_lines = 0
        uncertain_lines = 0
        with open(self._path, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith(TELEMETRY_SAMPLE_PREFIX):
                    continue
                try:
                    sample = decode_sample_line(stripped)
                except UnsupportedTelemetrySchemaError:
                    raise
                except TelemetryDecodeError:
                    continue
                if sample.sample_kind in SCIENTIFIC_SAMPLE_KINDS:
                    sample_lines += 1
                elif sample.sample_kind == "rhs_probe":
                    probe_lines += 1
                else:
                    uncertain_lines += 1
        if self.isInterruptionRequested():
            return
        self.count_ready.emit(min(sample_lines, REPLAY_SAMPLE_HARD_CAP))

        delivered = 0
        malformed = 0
        batch: list[TelemetrySample] = []
        with open(self._path, encoding="utf-8") as handle:
            for line in handle:
                if self.isInterruptionRequested():
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(TELEMETRY_META_PREFIX):
                    # Schema errors propagate (fail-closed); other decode
                    # problems only cost the provenance panel.
                    try:
                        self.meta_ready.emit(decode_meta_line(stripped))
                    except UnsupportedTelemetrySchemaError:
                        raise
                    except TelemetryDecodeError:
                        malformed += 1
                    continue
                if not stripped.startswith(TELEMETRY_SAMPLE_PREFIX):
                    continue
                try:
                    sample = decode_sample_line(stripped)
                except UnsupportedTelemetrySchemaError:
                    raise
                except TelemetryDecodeError:
                    malformed += 1
                    continue
                if sample.sample_kind not in SCIENTIFIC_SAMPLE_KINDS:
                    continue
                batch.append(sample)
                delivered += 1
                if len(batch) >= _BATCH_SIZE:
                    self.batch_ready.emit(batch)
                    batch = []
                if delivered >= REPLAY_SAMPLE_HARD_CAP:
                    break
        if batch:
            self.batch_ready.emit(batch)

        if probe_lines:
            self.warning.emit(
                f"Excluded {probe_lines} RHS probe sample(s) from scientific replay."
            )
        if uncertain_lines:
            self.warning.emit(
                f"Excluded {uncertain_lines} legacy sample(s): accepted-state semantics "
                "cannot be established for this artifact."
            )

        if delivered == 0:
            if uncertain_lines:
                self.failed.emit(
                    "This legacy artifact is readable, but its samples have uncertain RHS/accepted-state "
                    "semantics and cannot be shown as a scientific trajectory."
                )
                return
            if probe_lines:
                self.failed.emit("The artifact contains RHS probes but no accepted/output trajectory states.")
                return
            if malformed:
                self.failed.emit(
                    "The artifact contains telemetry lines but none could be decoded."
                )
            else:
                self.failed.emit("The artifact contains no telemetry samples.")
            return
        self.finished_ok.emit(delivered)


class TimelineController(QtCore.QObject):
    """Playback clock + shared cursor for replay mode.

    Speed semantics: **1× replays the full run in about one minute of wall
    time** (independent of mission duration), and the other speeds scale that
    rate. This is deterministic and stated in the UI tooltip — "1×" does not
    pretend to be real time for multi-day missions.
    """

    SPEEDS: tuple[float, ...] = (0.25, 1.0, 5.0, 20.0)
    NOMINAL_REPLAY_WALL_S = 60.0
    _TICK_MS = 50

    cursor_changed = QtCore.Signal(float)
    playing_changed = QtCore.Signal(bool)

    def __init__(self, monitor, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._monitor = monitor
        self._speed = 1.0
        self._playing = False
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self._TICK_MS)
        self._timer.timeout.connect(self._on_tick)
        self._elapsed = QtCore.QElapsedTimer()

    # ------------------------------------------------------------- properties
    @property
    def speed(self) -> float:
        return self._speed

    @property
    def playing(self) -> bool:
        return self._playing

    def bounds(self) -> tuple[float, float] | None:
        return self._monitor.store.time_bounds()

    def cursor(self) -> float | None:
        """Current cursor; None (follow-latest) resolves to the run end."""
        if self._monitor.cursor_time_s is not None:
            return float(self._monitor.cursor_time_s)
        b = self.bounds()
        return b[1] if b is not None else None

    # ---------------------------------------------------------------- control
    def set_speed(self, speed: float) -> None:
        if speed > 0.0:
            self._speed = float(speed)

    def play(self) -> None:
        if self._monitor.store.mode != "replay" or self.bounds() is None:
            return
        b = self.bounds()
        current = self.cursor()
        # Playing from the end restarts from the beginning.
        if b is not None and current is not None and current >= b[1]:
            self.set_cursor(b[0])
        self._playing = True
        self._elapsed.restart()
        self._timer.start()
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self.playing_changed.emit(False)

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def set_cursor(self, t_s: float) -> None:
        b = self.bounds()
        if b is None:
            return
        clamped = min(max(float(t_s), b[0]), b[1])
        self._monitor.jump_to_time(clamped)
        self.cursor_changed.emit(clamped)

    def step(self, direction: int) -> None:
        """Move to the adjacent retained sample (±1)."""
        import numpy as np

        times = self._monitor.store.times()
        if times.shape[0] == 0:
            return
        current = self.cursor()
        if current is None:
            current = float(times[-1])
        idx = int(np.searchsorted(times, current, side="right")) - 1
        idx = max(0, min(times.shape[0] - 1, idx + int(direction)))
        self.set_cursor(float(times[idx]))

    def jump_start(self) -> None:
        b = self.bounds()
        if b is not None:
            self.set_cursor(b[0])

    def jump_end(self) -> None:
        b = self.bounds()
        if b is not None:
            self.set_cursor(b[1])

    def jump_to_event(self, simulation_time_s: float) -> None:
        self.set_cursor(float(simulation_time_s))

    # ------------------------------------------------------------- tick clock
    def rate_sim_per_wall(self) -> float:
        b = self.bounds()
        if b is None or b[1] <= b[0]:
            return self._speed
        return (b[1] - b[0]) / self.NOMINAL_REPLAY_WALL_S * self._speed

    def advance(self, wall_dt_s: float) -> None:
        """Advance the cursor by a wall-clock delta (also used by tests)."""
        b = self.bounds()
        if b is None:
            return
        current = self.cursor()
        if current is None:
            current = b[0]
        target = current + self.rate_sim_per_wall() * float(wall_dt_s)
        if target >= b[1]:
            self.set_cursor(b[1])
            self.pause()
        else:
            self.set_cursor(target)

    def _on_tick(self) -> None:
        if not self._playing:
            return
        dt = self._elapsed.restart() / 1000.0
        self.advance(dt)


__all__ = ["REPLAY_SAMPLE_HARD_CAP", "ReplayLoader", "TimelineController"]
