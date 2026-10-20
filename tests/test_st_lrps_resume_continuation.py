"""A stopped run must continue with the settings it was started with.

Resume locks only the architecture/scaler fields; everything else comes from the
resuming command line. These tests pin the three pieces that keep a walltime
stop + resume faithful: the printed resume command repeats the original flags,
a bare ``--resume-from`` restores independent train/val files (it used to
re-split the train file and fail the scaler provenance check), and settings a
resume would silently change are listed.
"""

from __future__ import annotations

import json
import shlex
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch.nn")

from dataset_pipeline_test_utils import write_toy_contract_h5
from lunaris.surrogate.st_lrps.training.config import parse_args
from lunaris.surrogate.st_lrps.training.engine import (
    resume_command_hint,
    resume_setting_drift,
)


def test_resume_hint_repeats_the_original_flags_and_replaces_output_and_epochs():
    argv = [
        "--train-data", "/d/train.h5", "--val-data=/d/val.h5", "--batch-size", "256",
        "--lr", "3e-4", "--out", "/runs/a", "--epochs", "40", "--overwrite",
        "--resume-from", "/runs/old",
    ]
    hint = resume_command_hint(argv, "/runs/a", 80)
    tokens = shlex.split(hint)
    assert tokens[:3] == ["python", "-m", "lunaris.surrogate.st_lrps.training.cli"]
    rest = tokens[3:]
    assert rest[:6] == ["--train-data", "/d/train.h5", "--val-data=/d/val.h5", "--batch-size", "256", "--lr"]
    assert rest[-4:] == ["--resume-from", "/runs/a", "--epochs", "80"]
    for dropped in ("--out", "--overwrite", "/runs/old", "40"):
        assert dropped not in rest
    assert rest.count("--epochs") == 1 and rest.count("--resume-from") == 1


def test_resume_hint_drops_equals_forms():
    hint = resume_command_hint(["--out-dir=/x", "--epochs=5", "--hidden", "64"], "/x", 9)
    assert shlex.split(hint)[3:] == ["--hidden", "64", "--resume-from", "/x", "--epochs", "9"]


def test_setting_drift_lists_changed_optimisation_settings_only():
    prev = {
        "batch_size": 256, "lr": 3e-4, "hidden": 64, "epochs": 40, "out": "/a",
        "train_data": "/t.h5", "w0_bands": [10.0, 16.4], "altitude_max_km": 300.0,
    }
    cur = SimpleNamespace(
        batch_size=8192, lr=3e-4, hidden=128, epochs=80, out="/b",
        train_data="/u.h5", w0_bands=None, altitude_max_km=1000.0,
    )
    drift = resume_setting_drift(prev, cur, locked=("hidden",))
    assert drift == [("batch_size", 256, 8192)]


def test_bare_resume_restores_independent_train_and_val_files(tmp_path, monkeypatch):
    train = write_toy_contract_h5(tmp_path / "train.h5", n=48)
    val = write_toy_contract_h5(tmp_path / "val.h5", n=16)
    test = write_toy_contract_h5(tmp_path / "test.h5", n=16)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    # What a run on independent files records: ``data`` mirrors the train file.
    (run_dir / "config.json").write_text(
        json.dumps({
            "data": str(train), "train_data": str(train), "val_data": str(val),
            "test_data": str(test),
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["lunaris-train", "--resume-from", str(run_dir), "--epochs", "5"])
    cfg = parse_args()
    assert cfg.train_data == str(train)
    assert cfg.val_data == str(val)
    assert cfg.test_data == str(test)
    assert cfg.data in (None, str(train))


def test_drift_does_not_hide_schedule_metric_or_runtime_settings():
    prev = {"warmup_epochs": 5, "best_metric": "val_total_loss", "runtime_model_kind": "potential_autograd"}
    cur = SimpleNamespace(warmup_epochs=0, best_metric="hybrid", runtime_model_kind="force_direct")
    keys = [k for k, _, _ in resume_setting_drift(prev, cur)]
    assert keys == ["best_metric", "runtime_model_kind", "warmup_epochs"]


def test_a_repeated_stop_signal_within_the_grace_window_is_ignored():
    import signal

    from lunaris.surrogate.st_lrps.training.engine import build_stop_signal_handler

    now = {"t": 100.0}
    restored = []
    interrupt: dict = {"flag": False}
    handler = build_stop_signal_handler(
        interrupt, lambda: restored.append(True), grace_s=5.0, clock=lambda: now["t"]
    )
    handler(int(signal.SIGTERM), None)  # Slurm's TERM: graceful stop
    assert interrupt["flag"] is True
    now["t"] = 101.0
    handler(int(signal.SIGTERM), None)  # the relayed copy of the same stop: ignored
    assert restored == []
    now["t"] = 106.5
    with pytest.raises(KeyboardInterrupt):  # a deliberate second stop: force-quit
        handler(int(signal.SIGINT), None)
    assert restored == [True]
