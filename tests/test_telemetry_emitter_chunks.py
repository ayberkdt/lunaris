"""emit_trajectory writes in chunks without changing the telemetry stream.

The trajectory rows arrive after the solver has finished; they are written with
one stdout write and one ``telemetry.ndjson`` append per chunk instead of one
print and one file open per row. Every line must still reach stdout and the
sink, in order, and the desktop UI's line pipeline must still decode each one.
"""

from __future__ import annotations

import numpy as np

import lunaris.core.propagation.telemetry_emitter as telemetry
from lunaris.ui.core.log_stream import LineAssembler
from lunaris.ui.monitor.protocol import SampleMessage, TelemetryLineClassifier

MU_MOON = 4.9028e12
R_MOON = 1_737_400.0


def _circular_trajectory(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) * 60.0
    radius = R_MOON + 100_000.0
    rate = np.sqrt(MU_MOON / radius**3)
    states = np.stack(
        [
            radius * np.cos(rate * t),
            radius * np.sin(rate * t),
            0 * t,
            -radius * rate * np.sin(rate * t),
            radius * rate * np.cos(rate * t),
            0 * t,
        ],
        1,
    )
    return t, states


def _sequence_ids(lines: list[str]) -> list[int]:
    return [int(line.split('"sequence_id":')[1].split(",")[0]) for line in lines]


def test_a_trajectory_is_written_in_chunks_with_every_line_in_order(tmp_path, monkeypatch):
    """One sink append per chunk (not per row); the stream itself is unchanged."""
    n = 2 * telemetry._TRAJECTORY_CHUNK_LINES + 7
    t, states = _circular_trajectory(n)
    sink = tmp_path / "telemetry.ndjson"
    lines: list[str] = []
    emitter = telemetry.TelemetryEmitter(
        run_id="chunks",
        t0_s=0.0,
        reference_radius_m=R_MOON,
        mu_m3s2=MU_MOON,
        sink_path=str(sink),
        writer=lines.append,
    )
    opened: list[str] = []
    real_open = open

    def counting_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    # _sink_write_many looks `open` up as a module global first.
    monkeypatch.setattr(telemetry, "open", counting_open, raising=False)
    assert emitter.emit_trajectory(t, states) == n
    assert len(lines) == n  # an injected writer still receives one protocol line per call
    written = sink.read_text(encoding="utf-8").splitlines()
    assert written == lines
    assert _sequence_ids(written) == list(range(n))
    assert len(opened) == 3  # ceil(n / chunk) appends, not n
    # Outside emit_trajectory a single sample is still written immediately.
    assert emitter.emit(float(t[-1] + 60.0), states[0])
    assert len(lines) == n + 1
    assert len(sink.read_text(encoding="utf-8").splitlines()) == n + 1


def test_chunked_stdout_reaches_the_ui_line_pipeline_line_by_line(tmp_path, capsys):
    """The default print writer emits whole chunks; the UI still decodes every sample."""
    n = telemetry._TRAJECTORY_CHUNK_LINES + 5
    t, states = _circular_trajectory(n)
    sink = tmp_path / "telemetry.ndjson"
    emitter = telemetry.TelemetryEmitter(
        run_id="stdout_chunks",
        t0_s=0.0,
        reference_radius_m=R_MOON,
        mu_m3s2=MU_MOON,
        sink_path=str(sink),
    )
    assert emitter.emit_trajectory(t, states) == n
    out = capsys.readouterr().out
    stdout_lines = out.splitlines()
    assert stdout_lines == sink.read_text(encoding="utf-8").splitlines()
    assert out.endswith("\n")

    assembler = LineAssembler()
    classifier = TelemetryLineClassifier()
    # Feed in arbitrary read sizes, as QProcess delivers stdout.
    decoded = []
    for start in range(0, len(out), 4093):
        for line in assembler.push(out[start : start + 4093]):
            message = classifier.classify(line)
            assert isinstance(message, SampleMessage)
            decoded.append(message.sample)
    assert assembler.pending == ""
    assert [sample.sequence_id for sample in decoded] == list(range(n))
    assert all(sample.run_id == "stdout_chunks" for sample in decoded)


def test_a_failing_writer_counts_every_dropped_line_and_keeps_the_sink(tmp_path, monkeypatch):
    n = telemetry._TRAJECTORY_CHUNK_LINES + 3
    t, states = _circular_trajectory(n)
    sink = tmp_path / "telemetry.ndjson"
    emitter = telemetry.TelemetryEmitter(
        run_id="broken_pipe",
        t0_s=0.0,
        reference_radius_m=R_MOON,
        mu_m3s2=MU_MOON,
        sink_path=str(sink),
    )

    def broken_print(*_args, **_kwargs):
        raise BrokenPipeError("stdout closed")

    monkeypatch.setattr("builtins.print", broken_print)
    assert emitter.emit_trajectory(t, states) == n
    assert emitter.diagnostics.writer_failures == n
    assert emitter.diagnostics.as_dict()["dropped_samples"] == n
    assert _sequence_ids(sink.read_text(encoding="utf-8").splitlines()) == list(range(n))
