"""Structured (lunaris_telemetry_v1) telemetry emission for propagation runs.

Live RHS cadence samples are explicit ``rhs_probe`` observations and are never
mirrored into the scientific replay sink.  After integration, solver-returned
output states are emitted as ``output_state`` samples and form the replay
trajectory.

Emission is strictly an observation layer: it is best-effort (any failure is
swallowed so telemetry can never break a propagation), it never mutates the
state, and all serialization work happens only at telemetry cadence — never
per RHS evaluation.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lunaris.common.math_utils import rv_to_coe_select
from lunaris.common.telemetry_contract import (
    SampleKind,
    TelemetrySample,
    encode_sample_line,
)

#: Below this eccentricity the argument of periapsis is undefined; the channel
#: is omitted (the UI shows "undefined (circular)"), never a fake zero.
_CIRCULAR_ECC_EPS = 1e-11
#: Within this distance (rad) of i=0 or i=pi the RAAN is undefined; omitted.
_EQUATORIAL_INC_EPS = 1e-9

#: Frame label for the mission propagator's integration frame.
INERTIAL_FRAME_LABEL = "moon_centered_inertial"

_MAX_FAILURE_MESSAGE = 240
_SINK_FAILURE_LIMIT = 5


def _bounded_failure(exc: Exception) -> tuple[str, str]:
    """Return bounded diagnostics without leaking local absolute paths."""
    message = " ".join(str(exc).split())
    message = re.sub(r"(?:[A-Za-z]:\\|/)[^\s]+", "<path>", message)
    return type(exc).__name__, message[:_MAX_FAILURE_MESSAGE]


@dataclass(slots=True)
class TelemetryDiagnostics:
    sample_build_failures: int = 0
    serialization_failures: int = 0
    writer_failures: int = 0
    sink_write_failures: int = 0
    terrain_enrichment_failures: int = 0
    first_failure_type: str | None = None
    first_failure_message: str | None = None
    sink_disabled: bool = False

    def record(self, counter: str, exc: Exception) -> None:
        setattr(self, counter, int(getattr(self, counter)) + 1)
        if self.first_failure_type is None:
            self.first_failure_type, self.first_failure_message = _bounded_failure(exc)

    def as_dict(self) -> dict[str, int | str | bool]:
        payload: dict[str, int | str | bool] = {
            "sample_build_failures": self.sample_build_failures,
            "serialization_failures": self.serialization_failures,
            "writer_failures": self.writer_failures,
            "sink_write_failures": self.sink_write_failures,
            "terrain_enrichment_failures": self.terrain_enrichment_failures,
            "sink_disabled": self.sink_disabled,
        }
        payload["dropped_samples"] = (
            self.sample_build_failures + self.serialization_failures + self.writer_failures
        )
        if self.first_failure_type is not None:
            payload["first_failure_type"] = self.first_failure_type
        if self.first_failure_message is not None:
            payload["first_failure_message"] = self.first_failure_message
        return payload


def generate_run_id(prefix: str = "run") -> str:
    """Collision-safe, human-sortable run identifier."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class TelemetryEmitter:
    """Builds and writes v1 telemetry samples for one propagation run."""

    def __init__(
        self,
        *,
        run_id: str | None,
        t0_s: float,
        reference_radius_m: float,
        mu_m3s2: float,
        frame_inertial: str = INERTIAL_FRAME_LABEL,
        r_i_to_bf: Callable[[float, np.ndarray], np.ndarray] | None = None,
        surface_radius_m: Callable[[float, float], float] | None = None,
        sink_path: str | None = None,
        writer: Callable[[str], None] | None = None,
    ) -> None:
        self.run_id = run_id or generate_run_id()
        self._t0_s = float(t0_s)
        self._r_ref_m = float(reference_radius_m)
        self._mu = float(mu_m3s2)
        self._frame_inertial = frame_inertial
        self._r_i_to_bf = r_i_to_bf
        self._surface_radius_m = surface_radius_m
        self._sink_path = str(sink_path) if sink_path else None
        self._sink_failures = 0
        self._writer = writer if writer is not None else self._print_line
        self._trajectory_sequence = 0
        self._probe_sequence = 0
        self._wall0 = time.perf_counter()
        self.diagnostics = TelemetryDiagnostics()

    @staticmethod
    def _print_line(line: str) -> None:
        print(line, flush=True)

    def write_raw_line(self, line: str) -> None:
        """Mirror an already-encoded protocol line (e.g. the meta line) to the sink."""
        self._sink_write(line)

    def emit(
        self,
        t_frame_s: float,
        y: np.ndarray,
        *,
        sample_kind: SampleKind = "output_state",
        persist: bool | None = None,
    ) -> bool:
        """Best-effort emission; return whether the sample was built/encoded.

        ``rhs_probe`` defaults to stdout-only.  Scientific samples default to
        stdout plus the replay sink.  Writer and sink failures are independent:
        a closed stdout pipe cannot destroy an otherwise writable replay file.
        """
        if persist is None:
            persist = sample_kind != "rhs_probe"
        try:
            sample = self._build_sample(
                float(t_frame_s), np.asarray(y, dtype=np.float64), sample_kind=sample_kind
            )
            if sample is None:
                self.diagnostics.record(
                    "sample_build_failures", ValueError("state unavailable for telemetry")
                )
                return False
        except Exception as exc:
            self.diagnostics.record("sample_build_failures", exc)
            return False
        try:
            line = encode_sample_line(sample)
        except (TypeError, ValueError, OverflowError) as exc:
            self.diagnostics.record("serialization_failures", exc)
            return False
        try:
            self._writer(line)
        except Exception as exc:
            # ``writer`` is an injected callback (stdout by default), so its
            # implementation can raise any ordinary Exception. This is one of
            # the few intentionally broad boundaries: telemetry must never
            # terminate an otherwise valid propagation.
            self.diagnostics.record("writer_failures", exc)
        if persist:
            self._sink_write(line)
        return True

    def emit_rhs_probe(self, t_frame_s: float, y: np.ndarray) -> bool:
        """Emit a transient solver probe; never write it to ``telemetry.ndjson``."""
        return self.emit(t_frame_s, y, sample_kind="rhs_probe", persist=False)

    def emit_trajectory(self, times_s: np.ndarray, states: np.ndarray) -> int:
        """Emit solver-returned trajectory rows as one contiguous replay stream."""
        times = np.asarray(times_s, dtype=np.float64).reshape(-1)
        table = np.asarray(states, dtype=np.float64)
        if table.ndim != 2 or table.shape[0] != times.size:
            self.diagnostics.record(
                "sample_build_failures", ValueError("trajectory time/state shape mismatch")
            )
            return 0
        emitted = 0
        last_t: float | None = None
        for t_frame_s, state in zip(times, table, strict=True):
            t_value = float(t_frame_s)
            if last_t is not None and t_value <= last_t:
                self.diagnostics.record(
                    "sample_build_failures",
                    ValueError("trajectory times are not strictly increasing"),
                )
                continue
            if self.emit(t_value, state, sample_kind="output_state", persist=True):
                emitted += 1
                last_t = t_value
        return emitted

    # ------------------------------------------------------------- internals
    def _build_sample(
        self, t_frame_s: float, y: np.ndarray, *, sample_kind: SampleKind
    ) -> TelemetrySample | None:
        if y.size < 6:
            return None
        r = y[0:3]
        v = y[3:6]
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
            return None

        elements: dict[str, float] = {}
        radius_m: float | None = None
        speed_m_s: float | None = None
        if np.isfinite(self._mu) and self._mu > 0.0:
            a, ecc, inc, raan, argp, nu, _eps, rnorm, vnorm, _h = rv_to_coe_select(
                r, v, self._mu, mode="coe10"
            )
            radius_m = float(rnorm)
            speed_m_s = float(vnorm)
            if np.isfinite(a):
                elements["sma_m"] = float(a)
            if np.isfinite(ecc):
                elements["ecc"] = float(ecc)
            if np.isfinite(inc):
                elements["inc_rad"] = float(inc)
                # RAAN is undefined for (near-)equatorial orbits: omit instead
                # of publishing the kernel's 0.0 placeholder.
                near_equatorial = (
                    inc < _EQUATORIAL_INC_EPS or inc > (np.pi - _EQUATORIAL_INC_EPS)
                )
                if np.isfinite(raan) and not near_equatorial:
                    elements["raan_rad"] = float(raan)
            # Arg. of periapsis is undefined for (near-)circular orbits.
            if np.isfinite(argp) and np.isfinite(ecc) and ecc >= _CIRCULAR_ECC_EPS:
                elements["argp_rad"] = float(argp)
            if np.isfinite(nu):
                elements["nu_rad"] = float(nu)
        else:
            rnorm_f = float(np.linalg.norm(r))
            if np.isfinite(rnorm_f) and rnorm_f > 0.0:
                radius_m = rnorm_f
            vnorm_f = float(np.linalg.norm(v))
            if np.isfinite(vnorm_f):
                speed_m_s = vnorm_f
        if radius_m is None or radius_m <= 0.0:
            return None

        altitude_m: float | None = None
        if np.isfinite(self._r_ref_m) and self._r_ref_m > 0.0:
            altitude_m = radius_m - self._r_ref_m

        surface_radius_m, terrain_clearance_m = self._terrain_channels(t_frame_s, r, radius_m)

        sequence_id = (
            self._probe_sequence if sample_kind == "rhs_probe" else self._trajectory_sequence
        )
        sample = TelemetrySample(
            run_id=self.run_id,
            sequence_id=sequence_id,
            simulation_time_s=t_frame_s - self._t0_s,
            sample_kind=sample_kind,
            wall_time_s=time.perf_counter() - self._wall0,
            frame_inertial=self._frame_inertial,
            state_inertial=tuple(float(x) for x in y[0:6]),  # type: ignore[arg-type]
            radius_m=radius_m,
            altitude_m=altitude_m,
            speed_m_s=speed_m_s,
            surface_radius_m=surface_radius_m,
            terrain_clearance_m=terrain_clearance_m,
            orbital_elements=elements,
        )
        if sample_kind == "rhs_probe":
            self._probe_sequence += 1
        else:
            self._trajectory_sequence += 1
        return sample

    def _terrain_channels(
        self, t_frame_s: float, r: np.ndarray, radius_m: float
    ) -> tuple[float | None, float | None]:
        """Topography-sampled local surface radius / clearance (optional)."""
        if self._r_i_to_bf is None or self._surface_radius_m is None:
            return None, None
        try:
            from lunaris.core.propagation.telemetry import _latlon_from_r_bf

            r_bf = np.asarray(self._r_i_to_bf(t_frame_s, r), dtype=np.float64).reshape(3)
            lat_rad, lon_rad = _latlon_from_r_bf(r_bf)
            terrain_r_m = float(self._surface_radius_m(lat_rad, lon_rad))
            if not np.isfinite(terrain_r_m) or terrain_r_m <= 0.0:
                return None, None
            return terrain_r_m, radius_m - terrain_r_m
        except Exception as exc:
            # Optional terrain enrichment stays best-effort (mirrors the
            # legacy _make_telem_dict behavior). These are injected frame and
            # surface callbacks, so protect the observation boundary broadly
            # while recording the failure instead of hiding it.
            self.diagnostics.record("terrain_enrichment_failures", exc)
            return None, None

    def _sink_write(self, line: str) -> None:
        if self._sink_path is None:
            return
        try:
            # Open-per-write keeps the artifact crash-safe at telemetry
            # cadence (seconds); after repeated failures the sink is disabled
            # so a broken disk cannot slow the run down.
            with open(self._sink_path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            self._sink_failures += 1
            self.diagnostics.record("sink_write_failures", exc)
            if self._sink_failures >= _SINK_FAILURE_LIMIT:
                self._sink_path = None
                self.diagnostics.sink_disabled = True


def build_emitter_from_config(
    cfg: Any,
    *,
    t0_s: float,
    reference_radius_m: float,
    mu_m3s2: float,
    r_i_to_bf: Callable[[float, np.ndarray], np.ndarray] | None = None,
    surface_radius_m: Callable[[float, float], float] | None = None,
) -> TelemetryEmitter:
    """Construct an emitter from PropagatorConfig-style attributes."""
    run_id = str(getattr(cfg, "telemetry_run_id", "") or "") or None
    sink_path = str(getattr(cfg, "telemetry_sink_path", "") or "") or None
    return TelemetryEmitter(
        run_id=run_id,
        t0_s=t0_s,
        reference_radius_m=reference_radius_m,
        mu_m3s2=mu_m3s2,
        r_i_to_bf=r_i_to_bf,
        surface_radius_m=surface_radius_m,
        sink_path=sink_path,
    )


__all__ = [
    "INERTIAL_FRAME_LABEL",
    "TelemetryDiagnostics",
    "TelemetryEmitter",
    "build_emitter_from_config",
    "generate_run_id",
]
