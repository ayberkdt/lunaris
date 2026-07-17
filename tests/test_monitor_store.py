"""Bounded telemetry store tests (Mission Monitor test group C + performance I).

Covers bounded capacity, sequence ordering/duplicates/gaps, reset, the
live-to-replay transition, snapshot isolation, and the algorithmic memory
contract (append count never grows the buffers).
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.telemetry_contract import TelemetryEvent, TelemetryProvenance, TelemetrySample
from lunaris.ui.monitor.downsample import decimate_indices, envelope_downsample
from lunaris.ui.monitor.store import RunOutcome, TelemetryStore


def sample(seq: int, t: float | None = None, run_id: str = "run_a", **extra) -> TelemetrySample:
    extra.setdefault("sample_kind", "output_state")
    return TelemetrySample(
        run_id=run_id,
        sequence_id=seq,
        simulation_time_s=float(seq) if t is None else t,
        **extra,
    )


class TestBoundedCapacity:
    def test_capacity_evicts_oldest_and_never_grows(self):
        store = TelemetryStore(capacity=100)
        store.begin_run("run_a")
        for i in range(250):
            store.append(sample(i, altitude_m=float(i)))
        assert store.n_samples == 100
        t, v = store.snapshot("altitude_m")
        assert t.shape == (100,)
        assert t[0] == pytest.approx(150.0)  # oldest retained
        assert t[-1] == pytest.approx(249.0)
        assert v[0] == pytest.approx(150.0)

    def test_buffer_allocation_is_capacity_bound_not_append_bound(self):
        store = TelemetryStore(capacity=64)
        store.begin_run("run_a")
        for i in range(100_000):
            store.append(sample(i, altitude_m=1.0))
        # The memory contract: internal arrays stay at capacity regardless of
        # how many samples were appended.
        assert store._t.shape == (64,)
        assert all(buf.values.shape == (64,) for buf in store._scalars.values())
        assert store.counters.accepted == 100_000

    def test_event_list_is_bounded(self):
        store = TelemetryStore(capacity=16, max_events=5)
        store.begin_run("run_a")
        for i in range(10):
            store.add_event(TelemetryEvent("periselene", float(i)))
        assert len(store.events()) == 5
        assert store.counters.events_dropped == 5


class TestSequenceHandling:
    def test_duplicates_are_dropped_and_counted(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        assert store.append(sample(0)) == "appended"
        assert store.append(sample(0)) == "duplicate"
        assert store.counters.duplicates == 1
        assert store.n_samples == 1

    def test_out_of_order_is_dropped_and_counted(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.append(sample(5))
        assert store.append(sample(3)) == "out_of_order"
        assert store.counters.out_of_order == 1

    def test_gaps_are_measured_in_missing_samples(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.append(sample(0))
        store.append(sample(4))  # 1,2,3 missing
        assert store.counters.gap_samples == 3

    def test_foreign_run_samples_are_rejected(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.set_provenance(TelemetryProvenance(run_id="run_a"))
        assert store.append(sample(0, run_id="run_b")) == "foreign_run"
        assert store.counters.foreign_run == 1


class TestChannels:
    def test_rhs_probe_and_uncertain_legacy_never_enter_trajectory_channels(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        state = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        probe = sample(
            0, sample_kind="rhs_probe", altitude_m=999.0, state_inertial=state
        )
        uncertain = sample(
            0, sample_kind="legacy_unknown", altitude_m=888.0, state_inertial=state
        )
        assert store.append(probe) == "rhs_probe"
        assert store.append(uncertain) == "uncertain_sample"
        assert store.n_samples == 0
        assert store.snapshot("altitude_m")[0].size == 0
        assert store.snapshot_state()[0].size == 0
        assert store.counters.rhs_probes == 1
        assert store.counters.uncertain_samples == 1

    def test_never_seen_channel_yields_empty_not_zeros(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.append(sample(0, altitude_m=100.0))
        t, v = store.snapshot("terrain_clearance_m")
        assert t.shape == (0,)
        assert v.shape == (0,)
        assert not store.has_channel("terrain_clearance_m")
        assert store.has_channel("altitude_m")

    def test_intermittent_channel_values_are_nan_filtered(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.append(sample(0, altitude_m=10.0))
        store.append(sample(1))  # no altitude on this tick
        store.append(sample(2, altitude_m=30.0))
        t, v = store.snapshot("altitude_m")
        assert list(t) == [0.0, 2.0]
        assert list(v) == [10.0, 30.0]

    def test_orbital_elements_become_namespaced_channels(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.append(sample(0, orbital_elements={"ecc": 0.02, "sma_m": 1.8e6}))
        t, v = store.snapshot("elements.ecc")
        assert v[0] == pytest.approx(0.02)
        assert "elements.sma_m" in store.available_channels()

    def test_state_snapshot_and_cursor_lookup(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        state = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        store.append(sample(0, t=10.0, state_inertial=state, altitude_m=100.0))
        store.append(sample(1, t=20.0, state_inertial=tuple(x * 2 for x in state)))
        t, y = store.snapshot_state("state_inertial")
        assert y.shape == (2, 6)
        hit = store.state_at_or_before(15.0)
        assert hit is not None
        assert hit[0] == pytest.approx(10.0)
        assert store.value_at_or_before("altitude_m", 15.0) == pytest.approx(100.0)
        assert store.value_at_or_before("altitude_m", 5.0) is None

    def test_ring_reuse_does_not_leak_stale_values_into_new_rows(self):
        store = TelemetryStore(capacity=2)
        store.begin_run("run_a")
        store.append(sample(0, altitude_m=10.0))
        store.append(sample(1, altitude_m=20.0))
        store.append(sample(2))  # overwrites slot 0 and carries no altitude
        t, v = store.snapshot("altitude_m")
        assert list(t) == [1.0]
        assert list(v) == [20.0]


class TestSnapshotIsolation:
    def test_snapshot_mutation_does_not_affect_store(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.append(sample(0, altitude_m=10.0))
        t, v = store.snapshot("altitude_m")
        v[0] = 999.0
        _, v2 = store.snapshot("altitude_m")
        assert v2[0] == pytest.approx(10.0)


class TestRunLifecycle:
    def test_reset_clears_everything(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.append(sample(0, altitude_m=1.0))
        store.set_run_diagnostics({"wall_time_s": 5.0})
        store.reset()
        assert store.mode == "idle"
        assert store.run_id is None
        assert store.n_samples == 0
        assert store.run_diagnostics is None
        assert store.available_channels() == ()

    def test_begin_run_starts_a_fresh_sequence_space(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a")
        store.append(sample(10))
        store.begin_run("run_b")
        assert store.append(sample(0, run_id="run_b")) == "appended"
        assert store.counters.out_of_order == 0

    def test_live_to_replay_transition_retains_data(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("run_a", expected_duration_s=100.0)
        for i in range(5):
            store.append(sample(i, altitude_m=float(i)))
        store.finish_run(RunOutcome(reason="completed", exit_code=0, success=True))
        store.enter_replay()
        assert store.mode == "replay"
        assert store.n_samples == 5
        assert store.outcome is not None and store.outcome.success is True
        assert store.time_bounds() == (0.0, 4.0)

    def test_meta_before_samples_pins_the_run_id(self):
        store = TelemetryStore(capacity=16)
        store.begin_run("placeholder")
        store.set_provenance(TelemetryProvenance(run_id="run_real"))
        assert store.run_id == "run_real"
        assert store.append(sample(0, run_id="run_real")) == "appended"


class TestDownsampling:
    def test_envelope_output_is_bounded(self):
        t = np.linspace(0.0, 1.0, 10_000)
        v = np.sin(t * 40.0)
        t_ds, v_ds = envelope_downsample(t, v, 500)
        assert t_ds.shape[0] <= 500
        assert t_ds[0] == t[0] and t_ds[-1] == t[-1]

    def test_envelope_preserves_isolated_spike(self):
        t = np.arange(10_000, dtype=float)
        v = np.zeros(10_000)
        v[7_321] = -55.0  # single-sample dip a stride would likely skip
        _, v_ds = envelope_downsample(t, v, 200)
        assert v_ds.min() == pytest.approx(-55.0)

    def test_small_series_pass_through_unchanged(self):
        t = np.arange(10, dtype=float)
        v = t * 2
        t_ds, v_ds = envelope_downsample(t, v, 100)
        assert np.array_equal(t_ds, t)
        assert np.array_equal(v_ds, v)

    def test_snapshot_applies_display_budget(self):
        store = TelemetryStore(capacity=10_000)
        store.begin_run("run_a")
        for i in range(8_000):
            store.append(sample(i, altitude_m=float(i % 97)))
        t, v = store.snapshot("altitude_m", max_points=512)
        assert t.shape[0] <= 512
        assert v.max() == pytest.approx(96.0)

    def test_decimate_indices_keeps_endpoints(self):
        idx = decimate_indices(10_000, 100)
        assert idx[0] == 0 and idx[-1] == 9_999
        assert idx.shape[0] <= 100
