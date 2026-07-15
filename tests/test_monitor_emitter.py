"""Producer-side telemetry emitter tests (v1 [TELEMETRY] line generation).

Verifies the emitter builds contract-valid samples from raw states, applies
the singularity-omission policy (no fake zeros for undefined angles), never
raises into the propagation hot path, and mirrors lines into the ndjson sink.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.telemetry_contract import (
    TELEMETRY_SAMPLE_PREFIX,
    decode_sample_line,
)
from lunaris.core.propagation.telemetry_emitter import TelemetryEmitter, generate_run_id

MU_MOON = 4.9028e12
R_MOON = 1_737_400.0


def make_emitter(**overrides) -> tuple[TelemetryEmitter, list[str]]:
    lines: list[str] = []
    kwargs = dict(
        run_id="run_e",
        t0_s=0.0,
        reference_radius_m=R_MOON,
        mu_m3s2=MU_MOON,
        writer=lines.append,
    )
    kwargs.update(overrides)
    return TelemetryEmitter(**kwargs), lines


def circular_state(alt_m: float = 100_000.0) -> np.ndarray:
    r = R_MOON + alt_m
    v = math.sqrt(MU_MOON / r)
    # Inclined circular orbit (i = 45 deg) so RAAN stays defined.
    return np.array([r, 0.0, 0.0, 0.0, v / math.sqrt(2.0), v / math.sqrt(2.0)])


class TestEmission:
    def test_emitted_line_is_contract_valid_and_complete(self):
        emitter, lines = make_emitter()
        emitter.emit(60.0, circular_state())
        assert len(lines) == 1
        assert lines[0].startswith(TELEMETRY_SAMPLE_PREFIX + " ")
        sample = decode_sample_line(lines[0])
        assert sample.run_id == "run_e"
        assert sample.sequence_id == 0
        assert sample.simulation_time_s == pytest.approx(60.0)
        assert sample.radius_m == pytest.approx(R_MOON + 100_000.0)
        assert sample.altitude_m == pytest.approx(100_000.0)
        assert sample.state_inertial is not None
        assert sample.frame_inertial == "moon_centered_inertial"
        assert sample.orbital_elements["sma_m"] == pytest.approx(R_MOON + 100_000.0, rel=1e-6)
        assert sample.orbital_elements["inc_rad"] == pytest.approx(math.pi / 4.0, rel=1e-6)

    def test_sequence_ids_increase_monotonically(self):
        emitter, lines = make_emitter()
        emitter.emit(0.0, circular_state())
        emitter.emit(60.0, circular_state())
        seqs = [decode_sample_line(line).sequence_id for line in lines]
        assert seqs == [0, 1]

    def test_circular_orbit_omits_argp_instead_of_zero(self):
        emitter, lines = make_emitter()
        emitter.emit(0.0, circular_state())
        sample = decode_sample_line(lines[0])
        assert "argp_rad" not in sample.orbital_elements  # undefined, not 0.0
        assert "raan_rad" in sample.orbital_elements       # inclined -> defined
        assert sample.orbital_elements["ecc"] == pytest.approx(0.0, abs=1e-9)

    def test_equatorial_orbit_omits_raan(self):
        r = R_MOON + 100_000.0
        v = math.sqrt(MU_MOON / r)
        state = np.array([r, 0.0, 0.0, 0.0, v * 1.1, 0.0])  # equatorial, eccentric
        emitter, lines = make_emitter()
        emitter.emit(0.0, state)
        sample = decode_sample_line(lines[0])
        assert "raan_rad" not in sample.orbital_elements
        assert "argp_rad" in sample.orbital_elements  # eccentric -> defined

    def test_time_is_relative_to_t0(self):
        emitter, lines = make_emitter(t0_s=1000.0)
        emitter.emit(1060.0, circular_state())
        assert decode_sample_line(lines[0]).simulation_time_s == pytest.approx(60.0)


class TestHotPathSafety:
    def test_bad_state_is_skipped_silently(self):
        emitter, lines = make_emitter()
        emitter.emit(0.0, np.array([np.nan] * 6))
        emitter.emit(0.0, np.array([1.0, 2.0]))  # too short
        emitter.emit(0.0, np.zeros(6))           # degenerate radius
        assert lines == []

    def test_writer_failure_never_propagates(self):
        def broken_writer(_line: str) -> None:
            raise OSError("stdout closed")

        emitter, _ = make_emitter(writer=broken_writer)
        emitter.emit(0.0, circular_state())  # must not raise

    def test_zero_mu_still_emits_radius_and_speed(self):
        emitter, lines = make_emitter(mu_m3s2=0.0)
        emitter.emit(0.0, circular_state())
        sample = decode_sample_line(lines[0])
        assert sample.radius_m is not None
        assert sample.speed_m_s is not None
        assert sample.orbital_elements == {}  # honestly absent, not zeros


class TestSink:
    def test_lines_are_mirrored_to_the_ndjson_sink(self, tmp_path):
        sink = tmp_path / "telemetry.ndjson"
        emitter, lines = make_emitter(sink_path=str(sink))
        emitter.write_raw_line('[TELEMETRY_META] {"schema_version":"lunaris_telemetry_v1","run_id":"run_e"}')
        emitter.emit(0.0, circular_state())
        emitter.emit(60.0, circular_state())
        content = sink.read_text(encoding="utf-8").splitlines()
        assert len(content) == 3
        assert content[0].startswith("[TELEMETRY_META]")
        assert content[1] == lines[0]

    def test_broken_sink_disables_itself_but_keeps_stdout(self, tmp_path):
        emitter, lines = make_emitter(sink_path=str(tmp_path))  # a directory: open() fails
        for i in range(7):
            emitter.emit(float(i), circular_state())
        assert len(lines) == 7  # stdout emission unaffected
        assert emitter._sink_path is None  # sink disabled after repeated failures


def test_generate_run_id_is_unique_and_sortable():
    a, b = generate_run_id(), generate_run_id()
    assert a != b
    assert a.startswith("run-")
