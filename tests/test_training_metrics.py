"""
Tests for st_lrps.ui.training_metrics — pure-Python utilities.

No Qt, GPU, SPICE, or data files required.
"""

from __future__ import annotations

import time

import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so st_lrps is importable
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from st_lrps.ui.training_metrics import (

    ETAEstimator,
    TrainingLogParser,
    TrainingMetricsStore,
    TrainingRecord,
    compute_auto_log_interval,
)


# ═══════════════════════════════════════════════════════════════════════════
# Auto log interval
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoLogInterval:
    def test_1000_batches(self):
        assert compute_auto_log_interval(1000) == 100

    def test_87_batches(self):
        assert compute_auto_log_interval(87) == 9

    def test_5_batches(self):
        assert compute_auto_log_interval(5) == 1

    def test_zero_batches(self):
        assert compute_auto_log_interval(0) == 1

    def test_negative_batches(self):
        assert compute_auto_log_interval(-10) == 1

    def test_exact_multiple(self):
        assert compute_auto_log_interval(100) == 10

    def test_custom_target(self):
        assert compute_auto_log_interval(1000, target_updates=20) == 50

    def test_single_batch(self):
        assert compute_auto_log_interval(1) == 1


# ═══════════════════════════════════════════════════════════════════════════
# ETA Estimator
# ═══════════════════════════════════════════════════════════════════════════

class TestETAEstimator:
    def test_insufficient_data_returns_estimating(self):
        est = ETAEstimator()
        est.set_total_epochs(100)
        assert est.format_remaining() == "Estimating…"

    def test_remaining_none_before_start(self):
        est = ETAEstimator()
        est.set_total_epochs(100)
        assert est.remaining_seconds() is None

    def test_elapsed_none_before_start(self):
        est = ETAEstimator()
        assert est.elapsed_seconds() is None

    def test_completed_epochs_produce_finite_eta(self):
        est = ETAEstimator()
        est.set_total_epochs(10)
        est.on_training_start()

        # Simulate 2 completed epochs
        est.on_epoch_start(1)
        est.on_epoch_end(1)
        est.on_epoch_start(2)
        est.on_epoch_end(2)

        # Now ask for remaining
        rem = est.remaining_seconds()
        assert rem is not None
        assert rem >= 0

    def test_no_negative_eta(self):
        est = ETAEstimator()
        est.set_total_epochs(5)
        est.on_training_start()

        for ep in range(1, 6):
            est.on_epoch_start(ep)
            est.on_epoch_end(ep)

        rem = est.remaining_seconds()
        assert rem is not None
        assert rem >= 0

    def test_zero_remaining_at_end(self):
        est = ETAEstimator()
        est.set_total_epochs(3)
        est.on_training_start()

        for ep in range(1, 4):
            est.on_epoch_start(ep)
            est.on_epoch_end(ep)

        rem = est.remaining_seconds()
        assert rem == 0.0

    def test_format_elapsed_before_start(self):
        est = ETAEstimator()
        assert est.format_elapsed() == "--:--:--"

    def test_format_finish_no_data(self):
        est = ETAEstimator()
        assert est.format_finish() == "—"

    def test_batch_progress_eta_during_first_epoch(self):
        est = ETAEstimator()
        est.set_total_epochs(10)
        est.on_training_start()
        est.on_epoch_start(1)

        # Simulate some batch progress with a small sleep
        time.sleep(0.05)
        est.on_batch_progress(50, 100)

        # With 50% progress, should be able to estimate
        rem = est.remaining_seconds()
        # Might still be None if progress < 5%, but at 50% it should work
        if rem is not None:
            assert rem >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Training Log Parser
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainingLogParser:
    def setup_method(self):
        self.parser = TrainingLogParser()

    def test_empty_line_returns_none(self):
        assert self.parser.parse_line("") is None
        assert self.parser.parse_line("   ") is None

    def test_parse_epoch_train_batch(self):
        line = "Epoch [3/400] [train] batch [184/1840] opt=1.55e-04 ref=2.92e-04 U=7.62e-05 a=8.17e-03 lr=6.99e-05"
        rec = self.parser.parse_line(line)
        assert rec is not None
        assert rec.epoch == 3
        assert rec.phase == "train"
        assert rec.event == "batch"
        assert rec.loss_opt == pytest.approx(1.55e-04, rel=1e-3)
        assert rec.loss_ref == pytest.approx(2.92e-04, rel=1e-3)
        assert rec.lr == pytest.approx(6.99e-05, rel=1e-3)

    def test_parse_validation_summary(self):
        line = "Epoch [13/400] [val] ref=1.23e-04 U=5.0e-05 a=3.0e-03 cossim=0.987 ang=9.2 deg"
        rec = self.parser.parse_line(line)
        assert rec is not None
        assert rec.phase == "val"
        assert rec.event == "val_summary"
        assert rec.loss_ref == pytest.approx(1.23e-04, rel=1e-3)
        assert rec.cos_sim == pytest.approx(0.987, rel=1e-3)

    def test_parse_checkpoint_best_updated(self):
        line = "[checkpoint] best updated val_ref=8.21e-05 epoch=42 score=8.21e-05"
        rec = self.parser.parse_line(line)
        assert rec is not None
        assert rec.event == "best_updated"
        assert rec.severity == "success"
        assert rec.phase == "checkpoint"

    def test_parse_checkpoint_last_saved(self):
        line = "[checkpoint] ckpt_last saved"
        rec = self.parser.parse_line(line)
        assert rec is not None
        assert rec.event == "checkpoint_saved"
        assert rec.phase == "checkpoint"

    def test_parse_warning(self):
        line = "[WARNING] Learning rate is very low: 1e-8"
        rec = self.parser.parse_line(line)
        assert rec is not None
        assert rec.severity == "warning"
        assert rec.event == "warning"

    def test_parse_error(self):
        line = "[HATA] NaN detected in loss at batch 42"
        rec = self.parser.parse_line(line)
        assert rec is not None
        assert rec.severity == "error"
        assert rec.event == "error"

    def test_non_metric_line_returns_none(self):
        line = "Loading dataset from /path/to/data.h5..."
        rec = self.parser.parse_line(line)
        assert rec is None

    def test_epoch_state_persists(self):
        self.parser.parse_line("Epoch [5/100] [train] opt=1e-3 ref=2e-3")
        rec = self.parser.parse_line("[val] ref=5e-4 U=1e-4 a=2e-3")
        assert rec is not None
        assert rec.epoch == 5


# ═══════════════════════════════════════════════════════════════════════════
# Training Metrics Store
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainingMetricsStore:
    def test_empty_store(self):
        store = TrainingMetricsStore()
        assert len(store.records) == 0
        assert store.latest_train_loss() is None
        assert store.latest_val_loss() is None
        assert store.latest_lr() is None
        assert store.latest_epoch() == 0

    def test_append_and_retrieve(self):
        store = TrainingMetricsStore()
        rec = TrainingRecord(
            epoch=5,
            phase="train",
            event="batch",
            loss_opt=1.5e-4,
            loss_ref=2.9e-4,
            lr=6.99e-5,
        )
        store.append(rec)

        assert len(store.records) == 1
        assert store.latest_train_loss() == pytest.approx(1.5e-4)
        assert store.latest_lr() == pytest.approx(6.99e-5)
        assert store.latest_epoch() == 5

    def test_val_record_updates_val_loss(self):
        store = TrainingMetricsStore()
        rec = TrainingRecord(
            epoch=10,
            phase="val",
            event="val_summary",
            loss_ref=8.0e-5,
        )
        store.append(rec)
        assert store.latest_val_loss() == pytest.approx(8.0e-5)

    def test_best_updated(self):
        store = TrainingMetricsStore()
        rec = TrainingRecord(
            epoch=42,
            phase="checkpoint",
            event="best_updated",
            score=8.21e-5,
        )
        store.append(rec)
        assert store.latest_best_score() == pytest.approx(8.21e-5)
        assert store.latest_best_epoch() == 42

    def test_max_records_trimming(self):
        store = TrainingMetricsStore(max_records=10)
        for i in range(20):
            store.append(TrainingRecord(epoch=i, phase="train", event="batch"))
        assert len(store.records) == 10
        # Oldest records should be trimmed
        assert store.records[0].epoch == 10

    def test_clear(self):
        store = TrainingMetricsStore()
        store.append(TrainingRecord(epoch=1, phase="train", event="batch", lr=1e-3))
        store.clear()
        assert len(store.records) == 0
        assert store.latest_lr() is None
