# tests/test_st_lrps_checkpoint_manager.py
"""
Unit tests for ``BestCheckpointTracker`` (the CheckpointManager policy extracted
from the ST-LRPS training loop).

These pin the best-selection / burn-in / patience / early-stop policy in
isolation — no torch, no run directory — so the behaviour the engine relies on is
verified independently of an end-to-end training run.
"""

from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import math
from dataclasses import dataclass

from lunaris.surrogate.st_lrps.training.checkpoint_manager import (
    BestCheckpointTracker,
    BestUpdate,
    resolve_best_ckpt_start_epoch,
)


@dataclass
class _StartCfg:
    """Minimal duck-typed cfg for resolve_best_ckpt_start_epoch."""

    best_ckpt_start_epoch: int = -1
    direction_loss_start_epoch: int = 10
    direction_loss_ramp_epochs: int = 40
    checkpoint_settle_epochs: int = 5
    direction_loss_weight: float = 0.20


def test_initial_state_has_no_best():
    t = BestCheckpointTracker(start_epoch=0, patience=5)
    assert not t.has_best
    assert t.best_epoch == -1
    assert t.best_epoch_display is None
    assert math.isinf(t.best_val)
    assert t.best_val_or_none is None
    assert t.epochs_without_improve == 0


def test_first_eligible_epoch_is_always_best():
    t = BestCheckpointTracker(start_epoch=0, patience=5)
    upd = t.update(epoch=0, score=10.0)
    assert isinstance(upd, BestUpdate)
    assert upd.eligible and upd.is_best
    assert t.best_val == 10.0
    assert t.best_epoch == 0
    assert t.best_epoch_display == 1
    assert t.epochs_without_improve == 0


def test_lower_score_updates_best_and_resets_patience():
    t = BestCheckpointTracker(start_epoch=0, patience=5)
    t.update(epoch=0, score=10.0)
    t.update(epoch=1, score=11.0)        # worse -> patience 1
    assert t.epochs_without_improve == 1
    upd = t.update(epoch=2, score=9.0)   # better -> new best, patience reset
    assert upd.is_best and t.best_epoch == 2 and t.best_val == 9.0
    assert t.epochs_without_improve == 0


def test_non_improving_increments_patience_only():
    t = BestCheckpointTracker(start_epoch=0, patience=5)
    t.update(epoch=0, score=5.0)
    for i in range(1, 4):
        upd = t.update(epoch=i, score=5.0 + i)  # strictly worse each time
        assert not upd.is_best
    assert t.epochs_without_improve == 3
    assert t.best_epoch == 0 and t.best_val == 5.0


def test_equal_score_is_not_a_new_best():
    # Selection is strict (<): an equal score must not win, mirroring the engine.
    t = BestCheckpointTracker(start_epoch=0, patience=5)
    t.update(epoch=0, score=4.0)
    upd = t.update(epoch=1, score=4.0)
    assert not upd.is_best
    assert t.best_epoch == 0
    assert t.epochs_without_improve == 1


def test_burn_in_epochs_are_not_eligible():
    t = BestCheckpointTracker(start_epoch=3, patience=5)
    for e in range(3):
        upd = t.update(epoch=e, score=1.0)   # would be great scores, but burn-in
        assert not upd.eligible and not upd.is_best
        assert t.is_burn_in(e)
    assert not t.has_best
    assert t.epochs_without_improve == 0
    # First eligible epoch wins even with a worse score than the burn-in epochs.
    upd = t.update(epoch=3, score=100.0)
    assert upd.eligible and upd.is_best
    assert t.best_epoch == 3 and t.best_val == 100.0


def test_early_stop_triggers_at_patience():
    t = BestCheckpointTracker(start_epoch=0, patience=3)
    t.update(epoch=0, score=1.0)
    assert not t.should_early_stop()
    for i in range(1, 4):
        t.update(epoch=i, score=2.0 + i)     # 3 non-improving epochs
    assert t.epochs_without_improve == 3
    assert t.should_early_stop()


def test_restore_resumes_counters():
    t = BestCheckpointTracker(start_epoch=0, patience=5)
    t.restore(best_val=2.5, best_epoch=7, epochs_without_improve=2)
    assert t.best_val == 2.5
    assert t.best_epoch == 7
    assert t.best_epoch_display == 8
    assert t.epochs_without_improve == 2
    # A better score after resume still wins and resets patience.
    upd = t.update(epoch=8, score=2.0)
    assert upd.is_best and t.best_epoch == 8 and t.epochs_without_improve == 0


# =============================================================================
# resolve_best_ckpt_start_epoch
# =============================================================================

def test_auto_start_waits_for_direction_ready_when_direction_active():
    res = resolve_best_ckpt_start_epoch(_StartCfg(best_ckpt_start_epoch=-1))
    # 10 + 40 + 5 = 55
    assert res.start_epoch == 55
    assert "settled" in res.reason
    assert res.warning is None


def test_auto_start_is_zero_when_direction_disabled():
    res = resolve_best_ckpt_start_epoch(
        _StartCfg(best_ckpt_start_epoch=-1, direction_loss_weight=0.0)
    )
    assert res.start_epoch == 0
    assert "disabled" in res.reason
    assert res.warning is None


def test_manual_start_is_honored_without_warning_when_late_enough():
    res = resolve_best_ckpt_start_epoch(_StartCfg(best_ckpt_start_epoch=60))
    assert res.start_epoch == 60
    assert res.reason == "manual best_ckpt_start_epoch"
    assert res.warning is None


def test_manual_start_earlier_than_direction_ready_warns():
    res = resolve_best_ckpt_start_epoch(_StartCfg(best_ckpt_start_epoch=5))
    assert res.start_epoch == 5
    assert res.warning is not None and "5 < 55" in res.warning


def test_manual_early_start_does_not_warn_when_direction_disabled():
    res = resolve_best_ckpt_start_epoch(
        _StartCfg(best_ckpt_start_epoch=5, direction_loss_weight=0.0)
    )
    assert res.start_epoch == 5
    assert res.warning is None
