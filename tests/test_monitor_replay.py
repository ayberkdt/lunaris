"""Replay tests (Mission Monitor test group E).

Covers ndjson artifact loading (via the real emitter sink, end-to-end),
fail-closed unsupported-schema handling, timeline synchronization through the
single shared cursor, play/pause/speed semantics, event jumps, stepping,
final-sample parity, and the live→replay transition.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.ui_qt_helpers import QtWidgets

from lunaris.common.telemetry_contract import (
    TelemetryProvenance,
    TelemetrySample,
    encode_meta_line,
    encode_sample_line,
)
from lunaris.core.propagation.telemetry_emitter import TelemetryEmitter
from lunaris.ui.monitor.replay import ReplayLoader, TimelineController
from lunaris.ui.monitor.workspace import MonitorController

MU_MOON = 4.9028e12
R_MOON = 1_737_400.0


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def controller():
    app = _app()
    ctrl = MonitorController()
    yield ctrl
    ctrl.stop_replay_loader()
    app.processEvents()


def write_artifact(path, n_samples: int = 25, cadence_s: float = 60.0) -> None:
    """Produce a real artifact through the producer-side emitter sink."""
    meta = encode_meta_line(TelemetryProvenance(
        run_id="run_replay", integrator="DOP853", gravity_backend="classic_sh",
        telemetry_cadence_s=cadence_s,
    ))
    emitter = TelemetryEmitter(
        run_id="run_replay", t0_s=0.0, reference_radius_m=R_MOON, mu_m3s2=MU_MOON,
        sink_path=str(path), writer=lambda _line: None,  # sink only
    )
    emitter.write_raw_line(meta)
    r = R_MOON + 100_000.0
    v = math.sqrt(MU_MOON / r)
    for i in range(n_samples):
        emitter.emit(i * cadence_s, np.array([r, 0.0, 0.0, 0.0, v / 2**0.5, v / 2**0.5]))


def run_loader_sync(path) -> dict:
    """Execute the loader body synchronously and collect its signals."""
    loader = ReplayLoader(str(path))
    out: dict = {"count": None, "meta": None, "batches": [], "ok": None, "failed": None}
    loader.count_ready.connect(lambda n: out.__setitem__("count", n))
    loader.meta_ready.connect(lambda p: out.__setitem__("meta", p))
    loader.batch_ready.connect(lambda b: out["batches"].append(list(b)))
    loader.finished_ok.connect(lambda n: out.__setitem__("ok", n))
    loader.failed.connect(lambda d: out.__setitem__("failed", d))
    loader.run()  # same-thread execution -> direct signal delivery
    return out


class TestReplayLoader:
    def test_artifact_round_trip_through_the_real_emitter(self, tmp_path):
        _app()
        artifact = tmp_path / "telemetry.ndjson"
        write_artifact(artifact, n_samples=25)
        out = run_loader_sync(artifact)
        assert out["failed"] is None
        assert out["count"] == 25
        assert out["ok"] == 25
        assert out["meta"] is not None and out["meta"].run_id == "run_replay"
        samples = [s for batch in out["batches"] for s in batch]
        assert len(samples) == 25
        assert isinstance(samples[0], TelemetrySample)
        assert samples[-1].simulation_time_s == pytest.approx(24 * 60.0)

    def test_unsupported_schema_fails_closed(self, tmp_path):
        _app()
        artifact = tmp_path / "telemetry.ndjson"
        artifact.write_text(
            '[TELEMETRY] {"schema_version": "lunaris_telemetry_v99", '
            '"run_id": "x", "sequence_id": 0, "simulation_time_s": 1.0}\n',
            encoding="utf-8",
        )
        out = run_loader_sync(artifact)
        assert out["failed"] is not None
        assert "v99" in out["failed"]
        assert out["ok"] is None

    def test_missing_file_fails_gracefully(self, tmp_path):
        _app()
        out = run_loader_sync(tmp_path / "deleted_mid_session.ndjson")
        assert out["failed"] is not None
        assert out["ok"] is None

    def test_malformed_lines_are_skipped_but_all_malformed_fails(self, tmp_path):
        _app()
        artifact = tmp_path / "telemetry.ndjson"
        good = encode_sample_line(TelemetrySample(
            run_id="r", sequence_id=0, simulation_time_s=1.0, altitude_m=10.0,
        ))
        artifact.write_text(f"[TELEMETRY] {{broken\n{good}\n", encoding="utf-8")
        out = run_loader_sync(artifact)
        assert out["ok"] == 1  # good sample survives its malformed sibling

        artifact.write_text("[TELEMETRY] {broken\n", encoding="utf-8")
        out = run_loader_sync(artifact)
        assert out["failed"] is not None

    def test_controller_end_to_end_async_load(self, tmp_path, controller):
        app = _app()
        artifact = tmp_path / "telemetry.ndjson"
        write_artifact(artifact, n_samples=10)
        done: list = []
        controller.replay_loaded.connect(lambda: done.append(True))
        controller.replay_failed.connect(lambda d: done.append(d))
        controller.open_replay_file(str(artifact))
        import time as _time

        t0 = _time.monotonic()
        while not done and _time.monotonic() - t0 < 10.0:
            app.processEvents()
            _time.sleep(0.01)
        assert done == [True]
        controller.flush_now()
        assert controller.store.mode == "replay"
        assert controller.store.n_samples == 10
        assert controller.store.provenance is not None
        # The replay store is sized to the artifact, not the live default.
        assert controller.store.capacity == 16  # max(10, 16)


def load_replay(controller: MonitorController, tmp_path, n: int = 25) -> None:
    """Synchronous load through the controller's own signal handlers."""
    artifact = tmp_path / "telemetry.ndjson"
    write_artifact(artifact, n_samples=n)
    out = run_loader_sync(artifact)
    controller._on_replay_count(out["count"])
    controller._on_replay_meta(out["meta"])
    for batch in out["batches"]:
        controller._on_replay_batch(batch)
    controller._on_replay_finished(out["ok"])
    controller.flush_now()


class TestTimeline:
    def test_all_widget_reads_share_one_cursor(self, controller, tmp_path):
        load_replay(controller, tmp_path)
        timeline = TimelineController(controller)
        timeline.set_cursor(600.0)
        assert controller.cursor_time_s == pytest.approx(600.0)
        # Widgets resolve values through value_at_or_before on the same cursor.
        alt = controller.store.value_at_or_before("altitude_m", controller.cursor_time_s)
        assert alt == pytest.approx(100_000.0, rel=1e-3)

    def test_cursor_is_clamped_to_run_bounds(self, controller, tmp_path):
        load_replay(controller, tmp_path)
        timeline = TimelineController(controller)
        timeline.set_cursor(-100.0)
        assert controller.cursor_time_s == pytest.approx(0.0)
        timeline.set_cursor(1e9)
        assert controller.cursor_time_s == pytest.approx(24 * 60.0)

    def test_play_advances_and_pause_stops(self, controller, tmp_path):
        load_replay(controller, tmp_path)
        timeline = TimelineController(controller)
        timeline.jump_start()
        timeline.play()
        assert timeline.playing
        timeline.advance(1.0)  # deterministic tick
        first = controller.cursor_time_s
        assert first > 0.0
        timeline.pause()
        assert not timeline.playing
        timeline.advance(0.0)
        assert controller.cursor_time_s == pytest.approx(first)

    def test_speed_scales_the_advance_rate(self, controller, tmp_path):
        load_replay(controller, tmp_path)
        timeline = TimelineController(controller)
        span = 24 * 60.0
        timeline.jump_start()
        timeline.set_speed(1.0)
        timeline.advance(1.0)
        at_1x = controller.cursor_time_s
        assert at_1x == pytest.approx(span / timeline.NOMINAL_REPLAY_WALL_S)
        timeline.jump_start()
        timeline.set_speed(20.0)
        timeline.advance(1.0)
        assert controller.cursor_time_s == pytest.approx(20.0 * at_1x)

    def test_playback_pauses_at_the_end_with_final_sample_parity(
        self, controller, tmp_path
    ):
        load_replay(controller, tmp_path)
        timeline = TimelineController(controller)
        timeline.jump_start()
        timeline.play()
        timeline.advance(1e6)  # far past the end
        assert not timeline.playing
        # Final parity: the cursor lands exactly on the artifact's last sample.
        t, _v = controller.store.snapshot("altitude_m")
        assert controller.cursor_time_s == pytest.approx(float(t[-1]))

    def test_step_moves_between_adjacent_samples(self, controller, tmp_path):
        load_replay(controller, tmp_path)
        timeline = TimelineController(controller)
        timeline.jump_start()
        timeline.step(+1)
        assert controller.cursor_time_s == pytest.approx(60.0)
        timeline.step(+1)
        assert controller.cursor_time_s == pytest.approx(120.0)
        timeline.step(-1)
        assert controller.cursor_time_s == pytest.approx(60.0)
        timeline.step(-1)
        timeline.step(-1)  # clamped at the first sample
        assert controller.cursor_time_s == pytest.approx(0.0)

    def test_jump_to_event(self, controller, tmp_path):
        from lunaris.common.telemetry_contract import TelemetryEvent

        load_replay(controller, tmp_path)
        controller.store.add_event(TelemetryEvent("periselene", 300.0, "pass"))
        timeline = TimelineController(controller)
        timeline.jump_to_event(300.0)
        assert controller.cursor_time_s == pytest.approx(300.0)

    def test_replaying_from_the_end_restarts(self, controller, tmp_path):
        load_replay(controller, tmp_path)
        timeline = TimelineController(controller)
        timeline.jump_end()
        timeline.play()
        assert controller.cursor_time_s == pytest.approx(0.0)
        timeline.pause()

    def test_live_mode_ignores_cursor_and_play(self, controller):
        from lunaris.common.telemetry_contract import encode_sample_line

        controller.begin_live_run()
        controller.feed_line(encode_sample_line(TelemetrySample(
            run_id="live", sequence_id=0, simulation_time_s=10.0, altitude_m=1.0,
        )))
        timeline = TimelineController(controller)
        timeline.play()
        assert not timeline.playing  # playback is a replay-only affordance
        controller.jump_to_time(5.0)
        assert controller.cursor_time_s is None  # live mode follows latest


class TestLiveToReplayTransition:
    def test_finished_live_run_reviews_in_place(self, controller):
        controller.begin_live_run(expected_duration_s=100.0)
        for i in range(5):
            controller.feed_line(encode_sample_line(TelemetrySample(
                run_id="live", sequence_id=i, simulation_time_s=float(i * 10),
                altitude_m=1000.0 + i,
            )))
        controller.finish_live_run(exit_code=0)
        controller.enter_replay_of_live()
        controller.flush_now()
        assert controller.store.mode == "replay"
        assert controller.store.n_samples == 5  # same data, no reload
        timeline = TimelineController(controller)
        timeline.jump_start()
        assert controller.cursor_time_s == pytest.approx(0.0)
        timeline.jump_end()
        assert controller.cursor_time_s == pytest.approx(40.0)

    def test_empty_live_session_cannot_enter_replay(self, controller):
        controller.begin_live_run()
        controller.enter_replay_of_live()
        assert controller.store.mode == "live"
