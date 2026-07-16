"""Telemetry protocol parser tests (Mission Monitor test group B).

Covers complete/partial lines through the existing LineAssembler, multiple
messages per chunk, malformed JSON, unknown schema (fail-closed), the legacy
bare-JSON adapter, and stdout/stderr stream separation.
"""

from __future__ import annotations

import json

from lunaris.common.telemetry_contract import (
    TelemetryProvenance,
    TelemetrySample,
    encode_meta_line,
    encode_sample_line,
)
from lunaris.ui.core.log_stream import LineAssembler
from lunaris.ui.monitor.protocol import (
    MetaMessage,
    ProtocolProblem,
    SampleMessage,
    TelemetryLineClassifier,
)


def sample_line(seq: int = 0, t: float = 10.0) -> str:
    return encode_sample_line(
        TelemetrySample(run_id="run_a", sequence_id=seq, simulation_time_s=t, altitude_m=1000.0)
    )


def meta_line() -> str:
    return encode_meta_line(TelemetryProvenance(run_id="run_a", integrator="DOP853"))


class TestClassification:
    def test_sample_line_classifies_as_sample(self):
        msg = TelemetryLineClassifier().classify(sample_line())
        assert isinstance(msg, SampleMessage)
        assert msg.sample.run_id == "run_a"
        assert not msg.legacy

    def test_meta_line_classifies_as_meta(self):
        msg = TelemetryLineClassifier().classify(meta_line())
        assert isinstance(msg, MetaMessage)
        assert msg.provenance.integrator == "DOP853"

    def test_ordinary_log_lines_pass_through_as_none(self):
        classifier = TelemetryLineClassifier()
        for line in (
            "[System] Launching mission analysis...",
            "[STEP] max_step_s=12.5 (reason=nyquist, deg=120)",
            "",
            '{"config": {"a": 1}}',  # JSON but no telemetry time key
        ):
            assert classifier.classify(line) is None
        assert classifier.counters.malformed == 0

    def test_diag_lines_are_not_claimed_by_the_monitor(self):
        # [DIAG] routing must stay owned by the existing Results-page path.
        line = '[DIAG] {"wall_time_s": 5.0, "nfev": 100}'
        assert TelemetryLineClassifier().classify(line) is None


class TestFailClosed:
    def test_malformed_prefixed_json_is_reported_not_crashed(self):
        classifier = TelemetryLineClassifier()
        msg = classifier.classify("[TELEMETRY] {not json at all")
        assert isinstance(msg, ProtocolProblem)
        assert msg.kind == "malformed"
        assert classifier.counters.malformed == 1

    def test_unknown_schema_version_is_fail_closed(self):
        classifier = TelemetryLineClassifier()
        body = json.dumps({
            "schema_version": "lunaris_telemetry_v2",
            "run_id": "r", "sequence_id": 0, "simulation_time_s": 1.0,
        })
        msg = classifier.classify(f"[TELEMETRY] {body}")
        assert isinstance(msg, ProtocolProblem)
        assert msg.kind == "unsupported_schema"
        assert classifier.counters.unsupported_schema == 1

    def test_prefixed_payload_missing_required_fields_is_malformed(self):
        msg = TelemetryLineClassifier().classify(
            '[TELEMETRY] {"schema_version": "lunaris_telemetry_v1"}'
        )
        assert isinstance(msg, ProtocolProblem)
        assert msg.kind == "malformed"

    def test_malformed_legacy_lookalike_falls_back_to_log(self):
        # Non-contract JSON in logs must not raise telemetry alarms.
        classifier = TelemetryLineClassifier()
        assert classifier.classify('{"t_s": broken json') is None
        assert classifier.counters.malformed == 0


class TestLegacyAdapter:
    def test_legacy_bare_json_line_becomes_v1_sample(self):
        classifier = TelemetryLineClassifier()
        classifier.begin_run("legacy_run_7")
        msg = classifier.classify('{"t_s": 60.0, "alt_km": 49.5, "v_km_s": 1.62, "ecc": 0.01}')
        assert isinstance(msg, SampleMessage)
        assert msg.legacy
        assert msg.sample.run_id == "legacy_run_7"
        assert msg.sample.altitude_m is not None

    def test_legacy_sequence_ids_are_monotonic_and_reset_per_run(self):
        classifier = TelemetryLineClassifier()
        classifier.begin_run("r1")
        first = classifier.classify('{"t_s": 1.0}')
        second = classifier.classify('{"t_s": 2.0}')
        assert isinstance(first, SampleMessage) and isinstance(second, SampleMessage)
        assert (first.sample.sequence_id, second.sample.sequence_id) == (0, 1)
        classifier.begin_run("r2")
        third = classifier.classify('{"t_s": 3.0}')
        assert isinstance(third, SampleMessage)
        assert third.sample.sequence_id == 0
        assert third.sample.run_id == "r2"


class TestStreamAssembly:
    def test_partial_lines_assemble_before_classification(self):
        assembler = LineAssembler()
        classifier = TelemetryLineClassifier()
        line = sample_line(seq=1, t=42.0)
        head, tail = line[:17], line[17:]

        assert assembler.push(head) == []
        completed = assembler.push(tail + "\n")
        assert completed == [line]
        msg = classifier.classify(completed[0])
        assert isinstance(msg, SampleMessage)
        assert msg.sample.simulation_time_s == 42.0

    def test_multiple_messages_in_one_chunk(self):
        assembler = LineAssembler()
        classifier = TelemetryLineClassifier()
        chunk = meta_line() + "\n" + sample_line(0) + "\n[OK] Finished.\n"
        messages = [classifier.classify(line) for line in assembler.push(chunk)]
        assert isinstance(messages[0], MetaMessage)
        assert isinstance(messages[1], SampleMessage)
        assert messages[2] is None  # ordinary log line

    def test_stdout_and_stderr_use_independent_assemblers(self):
        # One assembler per stream is the existing console contract; telemetry
        # inherits it: interleaved chunks on separate assemblers never merge.
        stdout, stderr = LineAssembler(), LineAssembler()
        line = sample_line(2)
        stdout.push(line[:10])
        assert stderr.push("Traceback (most recent call last):\n") == [
            "Traceback (most recent call last):"
        ]
        completed = stdout.push(line[10:] + "\n")
        assert completed == [line]
        msg = TelemetryLineClassifier().classify(completed[0])
        assert isinstance(msg, SampleMessage)
