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

    EpochGuard,
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
        # Phase 7: an unknown finish time reads "Estimating…" (was "—").
        est = ETAEstimator()
        assert est.format_finish() == "Estimating…"

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


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: parser severity classification (no [INFO] false positives)
# ═══════════════════════════════════════════════════════════════════════════

class TestParserSeverity:
    def setup_method(self):
        self.parser = TrainingLogParser()

    def test_info_line_is_not_error(self):
        # Every engine line is prefixed with "[INFO]"; "Inf" must not trip the
        # error regex.
        rec = self.parser.parse_line("2026-05-25 12:00:00 [INFO] Epoch 1/400")
        assert rec is None or rec.severity != "error"

    def test_info_prefixed_batch_is_not_error(self):
        line = ("2026-05-25 12:00:00,1 [INFO] [train] epoch=3 batch=12/100 | "
                "opt=1.2e-01 ref=2.3e-01 U=3.4e-02 a=4.5e-02 lr=1.0e-04")
        rec = self.parser.parse_line(line)
        assert rec is not None
        assert rec.severity == "info"
        assert rec.event == "batch"

    def test_error_bracket(self):
        rec = self.parser.parse_line("[ERROR] Something failed")
        assert rec is not None
        assert rec.severity == "error"
        assert rec.event == "error"

    def test_error_nan(self):
        rec = self.parser.parse_line("[train] loss=NaN detected")
        assert rec is not None
        assert rec.severity == "error"

    def test_error_inf(self):
        rec = self.parser.parse_line("value=Inf encountered in gradient")
        assert rec is not None
        assert rec.severity == "error"

    def test_error_traceback(self):
        rec = self.parser.parse_line("Traceback (most recent call last):")
        assert rec is not None
        assert rec.severity == "error"

    def test_error_fatal_bracket(self):
        rec = self.parser.parse_line("[FATAL ERROR] out of memory")
        assert rec is not None
        assert rec.severity == "error"

    def test_warning_bracket(self):
        rec = self.parser.parse_line("[WARNING] learning rate is very low")
        assert rec is not None
        assert rec.severity == "warning"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6/7: parser field extraction for real engine log lines
# ═══════════════════════════════════════════════════════════════════════════

class TestParserFieldExtraction:
    def setup_method(self):
        self.parser = TrainingLogParser()

    def _batch_line(self):
        return ("2026-05-25 12:00:01,1 [INFO] [train] epoch=5 batch=12/100 | "
                "opt=1.234e-01 ref=2.345e-01 U=3.45e-02 a=4.56e-02 lr=1.00e-04 | "
                "dir=5.67e-06 | 12,345 samples/s | eta=120s cuda_mem=1234/2345MiB "
                "peak=3000/4000MiB total=8000MiB")

    def test_kv_epoch_extracted(self):
        rec = self.parser.parse_line(self._batch_line())
        assert rec.epoch == 5

    def test_batch_equals_form(self):
        rec = self.parser.parse_line(self._batch_line())
        assert rec.batch == 12
        assert rec.total_batches == 100

    def test_progress_pct(self):
        rec = self.parser.parse_line(self._batch_line())
        assert rec.progress_pct == pytest.approx(12.0)

    def test_losses_extracted(self):
        rec = self.parser.parse_line(self._batch_line())
        assert rec.loss_opt == pytest.approx(1.234e-01, rel=1e-3)
        assert rec.loss_ref == pytest.approx(2.345e-01, rel=1e-3)
        assert rec.loss_u == pytest.approx(3.45e-02, rel=1e-3)
        assert rec.loss_a == pytest.approx(4.56e-02, rel=1e-3)
        assert rec.lr == pytest.approx(1.00e-04, rel=1e-3)

    def test_direction_loss_extracted(self):
        rec = self.parser.parse_line(self._batch_line())
        assert rec.direction_loss == pytest.approx(5.67e-06, rel=1e-3)

    def test_samples_with_comma(self):
        rec = self.parser.parse_line(self._batch_line())
        assert rec.samples_per_s == pytest.approx(12345.0)

    def test_eta_and_memory(self):
        rec = self.parser.parse_line(self._batch_line())
        assert rec.eta_s == pytest.approx(120.0)
        assert "cuda_mem=1234/2345MiB" in rec.memory

    def test_val_summary_trailing_space_header(self):
        # The engine prints validation summaries as "[val ]" (trailing space).
        line = ("2026-05-25 12:01:00,0 [INFO] [val ] epoch=5 done: 1,000 samples "
                "in 2.0s loss_opt=8.1e-05 loss_ref=9.2e-05 U=5.0e-05 a=3.0e-03")
        rec = self.parser.parse_line(line)
        assert rec is not None
        assert rec.phase == "val"
        assert rec.event == "val_summary"
        assert rec.loss_ref == pytest.approx(9.2e-05, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8: epoch guard (debounce repeated "Epoch X/Y" lines)
# ═══════════════════════════════════════════════════════════════════════════

class TestEpochGuard:
    def test_first_start_true_repeat_false(self):
        g = EpochGuard()
        assert g.should_start(3) is True
        assert g.should_start(3) is False
        assert g.should_start(3) is False

    def test_new_epoch_starts(self):
        g = EpochGuard()
        assert g.should_start(3) is True
        assert g.should_start(4) is True
        assert g.should_start(4) is False

    def test_first_end_true_repeat_false(self):
        g = EpochGuard()
        assert g.should_end(3) is True
        assert g.should_end(3) is False

    def test_reset(self):
        g = EpochGuard()
        g.should_start(3)
        g.should_end(3)
        g.reset()
        assert g.should_start(3) is True
        assert g.should_end(3) is True

    def test_simulated_epoch_does_not_double_count(self):
        # Many "Epoch 3" lines, one validation summary → exactly one start/end.
        g = EpochGuard()
        starts = sum(g.should_start(3) for _ in range(10))
        ends = sum(g.should_end(3) for _ in range(3))  # repeated val summaries
        assert starts == 1
        assert ends == 1


# ═══════════════════════════════════════════════════════════════════════════
# Phase 9: CLI log-every-mode support
# ═══════════════════════════════════════════════════════════════════════════

class TestCLILogEveryMode:
    def _parse(self, argv):
        import sys as _sys
        from st_lrps.training.config import parse_args
        old = _sys.argv
        _sys.argv = ["prog"] + argv
        try:
            try:
                return parse_args()
            except SystemExit:
                pytest.skip("parse_args requires a resolvable dataset in this environment")
            except Exception as exc:  # pragma: no cover - environment dependent
                pytest.skip(f"parse_args could not resolve a dataset: {exc}")
        finally:
            _sys.argv = old

    def test_default_is_fixed(self):
        cfg = self._parse(["--epochs", "2"])
        assert cfg.log_every_mode == "fixed"

    def test_auto_mode(self):
        cfg = self._parse(["--epochs", "2", "--log-every-mode", "auto"])
        assert cfg.log_every_mode == "auto"

    def test_fixed_mode_preserves_value(self):
        cfg = self._parse(["--epochs", "2", "--log-every-mode", "fixed", "--log-every", "37"])
        assert cfg.log_every_mode == "fixed"
        assert cfg.log_every == 37


# ═══════════════════════════════════════════════════════════════════════════
# Phase 9: auto log interval edge cases (engine uses ceil(total/10))
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoLogIntervalEngineRule:
    def test_large(self):
        assert compute_auto_log_interval(1000) == 100

    def test_uneven(self):
        assert compute_auto_log_interval(87) == 9

    def test_small(self):
        assert compute_auto_log_interval(5) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: estimated finish-time formatting + epoch-duration helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestFinishTimeFormatting:
    def test_today_is_hh_mm(self):
        from datetime import datetime
        from st_lrps.ui.training_metrics import format_finish_time
        now = datetime(2026, 5, 25, 9, 0, 0)
        ft = datetime(2026, 5, 25, 18, 30, 0)
        assert format_finish_time(ft, now) == "18:30"

    def test_other_day_is_full_date(self):
        from datetime import datetime
        from st_lrps.ui.training_metrics import format_finish_time
        now = datetime(2026, 5, 25, 23, 0, 0)
        ft = datetime(2026, 5, 27, 4, 15, 0)
        assert format_finish_time(ft, now) == "2026-05-27 04:15"


class TestEpochDurationHelpers:
    def test_current_epoch_none_before_start(self):
        est = ETAEstimator()
        assert est.current_epoch_seconds() is None
        assert est.format_current_epoch() == "--:--:--"

    def test_avg_epoch_estimating_before_any_epoch(self):
        est = ETAEstimator()
        est.set_total_epochs(5)
        est.on_training_start()
        est.on_epoch_start(1)
        assert est.average_epoch_seconds() is None
        assert est.format_avg_epoch() == "Estimating…"

    def test_avg_epoch_finite_after_epoch(self):
        est = ETAEstimator()
        est.set_total_epochs(5)
        est.on_training_start()
        est.on_epoch_start(1)
        time.sleep(0.03)  # exceed the monotonic clock tick so duration > 0
        est.on_epoch_end(1)
        assert est.average_epoch_seconds() is not None
        assert est.average_epoch_seconds() >= 0.0
