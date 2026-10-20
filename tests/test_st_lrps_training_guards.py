"""Training-engine guards: invalid programmatic settings fail instead of being replaced."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
from lunaris.surrogate.st_lrps.training.config import TrainConfig
from lunaris.surrogate.st_lrps.training.engine import STLRPSTrainer
from lunaris.surrogate.st_lrps.training.losses import GradNormWeights, SobolevLoss


def _trainer(cfg: TrainConfig) -> STLRPSTrainer:
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    return STLRPSTrainer(
        model, SobolevLoss(sp, a_sign=1.0), opt, GradNormWeights(mode="fixed"),
        torch.device("cpu"), cfg,
    )


def test_trainer_rejects_an_invalid_laplacian_mode():
    # "trian" used to become "diagnostic" silently: the run asked for a trained
    # Laplacian penalty and got diagnostics only.
    cfg = TrainConfig(data="x.h5", out="o", epochs=1, batch_size=8, laplacian_mode="trian", amp=False)
    with pytest.raises(ValueError, match="laplacian_mode='trian' is not valid"):
        _trainer(cfg)


def _tiny_cfg(data_path, out_dir, **overrides) -> TrainConfig:
    kwargs = dict(
        data=str(data_path), out=str(out_dir), epochs=1, batch_size=8,
        num_workers=0, pin_memory=False, quick_check=True, max_train_batches=1,
        max_val_batches=1, activation="silu", hidden=8, depth=2, model_preset="custom",
        use_residual_blocks=False, n_bands=1, val_ratio=0.25, split_seed=11,
        a_sign=1.0, gradnorm_mode="fixed", direction_loss_weight=0.0,
        use_altitude_balanced_loss=False, use_radial_cross_loss=False,
        preload_policy="never", auto_preload_mb=0.0, fit_rows=32, fit_chunk_rows=32,
        cache_rows=16, sampler_block_size=16, warmup_epochs=0, best_ckpt_start_epoch=0,
        log_every=0,
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_stop_signal_handlers_are_restored_when_the_training_loop_raises(tmp_path, monkeypatch):
    import signal

    from dataset_pipeline_test_utils import write_toy_contract_h5
    from lunaris.surrogate.st_lrps.training import engine

    data_path = write_toy_contract_h5(tmp_path / "toy_train.h5", n=48)
    cfg = _tiny_cfg(data_path, tmp_path / "run")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected failure inside the epoch loop")

    # The first statement of every epoch; raising here fails the loop after the
    # stop-signal handlers were installed.
    monkeypatch.setattr(engine, "_lr_multiplier_for_epoch", _boom)
    before = {sig: signal.getsignal(sig) for sig in engine._stop_signal_list()}
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            engine.train(cfg)
        after = {sig: signal.getsignal(sig) for sig in engine._stop_signal_list()}
    finally:
        engine._restore_stop_signal_handlers(before)
    assert after == before


def test_resume_counts_malformed_history_lines(tmp_path, caplog):
    import logging

    from dataset_pipeline_test_utils import write_toy_contract_h5
    from lunaris.surrogate.st_lrps.training import engine

    data_path = write_toy_contract_h5(tmp_path / "toy_train.h5", n=48)
    run_dir = tmp_path / "run"
    engine.train(_tiny_cfg(data_path, run_dir))
    history = run_dir / "history.jsonl"
    assert history.exists()
    with history.open("a", encoding="utf-8") as handle:
        handle.write('{"epoch": 7, "truncated\n')  # a half-written row from a killed run

    caplog.set_level(logging.WARNING, logger=engine.logger.name)
    engine.train(_tiny_cfg(data_path, run_dir, epochs=2, resume_from=str(run_dir)))
    assert "skipped 1 malformed history line(s)" in caplog.text


@pytest.mark.parametrize("mode", ["off", "diagnostic", "train", " Train "])
def test_trainer_accepts_the_valid_laplacian_modes(mode):
    cfg = TrainConfig(data="x.h5", out="o", epochs=1, batch_size=8, laplacian_mode=mode, amp=False)
    assert _trainer(cfg).laplacian_mode == mode.strip().lower()
