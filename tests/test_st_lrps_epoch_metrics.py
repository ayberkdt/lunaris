"""Epoch-summary metrics are sample-weighted means, not means of batch means.

``STLRPSTrainer._update_metrics`` / ``_write_history_summary`` reduce the
per-batch statistics into one value per epoch. The validation loader keeps its
last, smaller batch (``drop_last=False``), so a plain mean of per-batch means
over-weights that remainder batch; ``val_total_loss`` drives best-checkpoint
selection. The direction-fix statistics (``loss_rel``, ``rel_a_mean``,
``wtz_frac``) follow the same rule.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.training.engine import STLRPSTrainer, _EpochState  # noqa: E402


def _bare_trainer() -> STLRPSTrainer:
    """A trainer with only the attributes the two methods under test read."""
    inst = STLRPSTrainer.__new__(STLRPSTrainer)
    inst.cfg = SimpleNamespace(
        use_radial_cross_loss=True,
        use_laplacian_regularization=False,
        use_altitude_balanced_loss=False,
        collocation_laplacian_weight=0.0,
        w_u=1.0,
        w_a=1.0,
    )
    inst.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}])
    return inst


def _stats(v: float) -> dict[str, float]:
    return {
        "loss_ref": v, "loss_opt": v, "mse_u": v, "mse_a": v,
        "loss_rel": v, "rel_a_mean": v, "wtz_frac": v,
        "loss_radial": v, "loss_cross": v, "cossim_mean": v, "mask_frac": v,
    }


def test_epoch_summary_is_sample_weighted():
    trainer = _bare_trainer()
    state = _EpochState()
    # 1024-row batch with value 1, then a 1-row remainder batch with value 2.
    trainer._update_metrics(state, _stats(1.0), torch.zeros(1024, 3), torch.ones(1024, 3))
    trainer._update_metrics(state, _stats(2.0), torch.zeros(1, 3), 2.0 * torch.ones(1, 3))
    assert state.samples_done == 1025

    summary = trainer._write_history_summary(
        state, phase="val ", epoch=0, accel_factor=1.0, lambda_dir_eff=0.0, phase_time=1.0,
    )
    expected = (1024 * 1.0 + 1 * 2.0) / 1025   # 1.000976...; the batch mean is 1.5
    for key in (
        "loss", "val_total_loss", "objective_loss", "mse_u", "mse_a",
        "loss_rel", "rel_a_mean", "wtz_frac", "loss_radial", "loss_cross",
        "cossim_mean", "mask_frac",
    ):
        assert summary[key] == pytest.approx(expected, rel=1e-12), key
    assert summary["val_base_loss"] == pytest.approx(2.0 * expected, rel=1e-12)
    # |a| per batch: sqrt(3) and 2*sqrt(3), weighted the same way.
    assert summary["a_norm_mean"] == pytest.approx(3.0 ** 0.5 * expected, rel=1e-6)
    # physics loss = (dir + radial + cross + lap + rel), each sample-weighted.
    assert summary["val_physics_loss"] == pytest.approx(3.0 * expected, rel=1e-12)


def test_equal_batches_reduce_to_the_batch_mean():
    trainer = _bare_trainer()
    state = _EpochState()
    trainer._update_metrics(state, _stats(1.0), torch.zeros(8, 3), torch.ones(8, 3))
    trainer._update_metrics(state, _stats(3.0), torch.zeros(8, 3), torch.ones(8, 3))
    summary = trainer._write_history_summary(
        state, phase="train", epoch=0, accel_factor=1.0, lambda_dir_eff=0.0, phase_time=1.0,
    )
    assert summary["loss"] == pytest.approx(2.0, rel=1e-12)
    assert summary["wtz_frac"] == pytest.approx(2.0, rel=1e-12)
