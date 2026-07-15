"""Structured (lunaris_telemetry_v1) telemetry emission for propagation runs.

The propagator wraps its RHS with a cadence gate (one float comparison per
evaluation); when the gate opens it hands the raw state to this emitter, which
builds a typed :class:`~lunaris.common.telemetry_contract.TelemetrySample`,
prints it as a ``[TELEMETRY] {json}`` stdout line, and optionally mirrors the
line into an ndjson sink file (the replay artifact).

Emission is strictly an observation layer: it is best-effort (any failure is
swallowed so telemetry can never break a propagation), it never mutates the
state, and all serialization work happens only at telemetry cadence — never
per RHS evaluation.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from lunaris.common.math_utils import rv_to_coe_select
from lunaris.common.telemetry_contract import TelemetrySample, encode_sample_line

#: Below this eccentricity the argument of periapsis is undefined; the channel
#: is omitted (the UI shows "undefined (circular)"), never a fake zero.
_CIRCULAR_ECC_EPS = 1e-11
#: Within this distance (rad) of i=0 or i=pi the RAAN is undefined; omitted.
_EQUATORIAL_INC_EPS = 1e-9

#: Frame label for the mission propagator's integration frame.
INERTIAL_FRAME_LABEL = "moon_centered_inertial"


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
        self._sequence = 0
        self._wall0 = time.perf_counter()

    @staticmethod
    def _print_line(line: str) -> None:
        print(line, flush=True)

    def write_raw_line(self, line: str) -> None:
        """Mirror an already-encoded protocol line (e.g. the meta line) to the sink."""
        self._sink_write(line)

    def emit(self, t_frame_s: float, y: np.ndarray) -> None:
        """Best-effort: build and write one sample; failures never propagate."""
        try:
            sample = self._build_sample(float(t_frame_s), np.asarray(y, dtype=np.float64))
            if sample is None:
                return
            line = encode_sample_line(sample)
        except Exception:
            # Telemetry must stay an observation layer: a bad state or an
            # encoding surprise skips one sample instead of killing the run.
            return
        try:
            self._writer(line)
        except Exception:
            return
        self._sink_write(line)

    # ------------------------------------------------------------- internals
    def _build_sample(self, t_frame_s: float, y: np.ndarray) -> TelemetrySample | None:
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

        sample = TelemetrySample(
            run_id=self.run_id,
            sequence_id=self._sequence,
            simulation_time_s=t_frame_s - self._t0_s,
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
        self._sequence += 1
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
        except Exception:
            # Optional terrain enrichment stays best-effort (mirrors the
            # legacy _make_telem_dict behavior).
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
        except OSError:
            self._sink_failures += 1
            if self._sink_failures >= 5:
                self._sink_path = None


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
    "TelemetryEmitter",
    "build_emitter_from_config",
    "generate_run_id",
]
