"""Tests for the optional Periodic Evaluation During Training feature.

Covers the pure schedule helper, the command builder, plan resolution, resume
skip behavior, and the subprocess runner (with a mocked subprocess so no real
evaluation runs). A lightweight UI command-builder check is included and skipped
when PyQt is unavailable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from st_lrps.training.periodic_eval import (
    PeriodicEvalPlan,
    build_periodic_eval_command,
    completed_periodic_eval_epochs,
    compute_periodic_eval_epochs,
    epoch_output_dir,
    history_path,
    load_periodic_eval_history,
    resolve_eval_dataset_path,
    resolve_periodic_eval_plan,
    run_periodic_eval,
)


# ---------------------------------------------------------------------------
# 1. Schedule helper
# ---------------------------------------------------------------------------
def test_schedule_count_spreads_across_horizon():
    assert compute_periodic_eval_epochs(400, 10, None) == [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]


def test_schedule_count_includes_final_epoch():
    sched = compute_periodic_eval_epochs(400, 10, None)
    assert sched[-1] == 400


def test_schedule_every_k_epochs():
    assert compute_periodic_eval_epochs(100, None, 25) == [25, 50, 75, 100]


def test_schedule_count_disabled_is_empty():
    assert compute_periodic_eval_epochs(400, None, None) == []


def test_schedule_zero_or_negative_count_is_empty():
    assert compute_periodic_eval_epochs(400, 0, None) == []
    assert compute_periodic_eval_epochs(400, -3, None) == []


def test_schedule_count_and_every_mutually_exclusive():
    with pytest.raises(ValueError):
        compute_periodic_eval_epochs(400, 10, 25)


def test_schedule_no_duplicates_and_sorted():
    # count > total collapses to one-per-epoch without duplicates.
    sched = compute_periodic_eval_epochs(5, 100, None)
    assert sched == sorted(set(sched))
    assert sched == [1, 2, 3, 4, 5]


def test_schedule_start_epoch_filters_past_epochs():
    # Resuming from epoch 287: only future scheduled epochs remain.
    assert compute_periodic_eval_epochs(400, 10, None, start_epoch=288) == [320, 360, 400]


# ---------------------------------------------------------------------------
# 2. Plan resolution + command builder
# ---------------------------------------------------------------------------
def _cfg(**overrides):
    base = dict(
        epochs=400,
        batch_size=8192,
        data=None,
        val_data=None,
        test_data=None,
        ood_data=None,
        periodic_eval_count=None,
        periodic_eval_every_epochs=None,
        periodic_eval_dataset="val",
        periodic_eval_max_samples=200_000,
        periodic_eval_batch_size=None,
        periodic_eval_device="auto",
        periodic_eval_prefer_checkpoint="last",
        periodic_eval_timeout_sec=None,
        periodic_eval_continue_on_fail=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_plan_disabled_by_default():
    plan = resolve_periodic_eval_plan(_cfg())
    assert plan.enabled is False
    assert plan.epochs == []


def test_plan_count_enabled():
    plan = resolve_periodic_eval_plan(_cfg(periodic_eval_count=10))
    assert plan.enabled is True
    assert plan.epochs == [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
    assert plan.prefer_checkpoint == "last"


def test_plan_every_enabled():
    plan = resolve_periodic_eval_plan(_cfg(epochs=100, periodic_eval_every_epochs=25))
    assert plan.enabled is True
    assert plan.epochs == [25, 50, 75, 100]


def test_plan_batch_size_falls_back_to_training_batch():
    plan = resolve_periodic_eval_plan(_cfg(periodic_eval_count=2, batch_size=4096))
    assert plan.batch_size == 4096
    plan2 = resolve_periodic_eval_plan(_cfg(periodic_eval_count=2, periodic_eval_batch_size=1024))
    assert plan2.batch_size == 1024


def test_plan_resume_start_epoch_drops_past():
    plan = resolve_periodic_eval_plan(_cfg(periodic_eval_count=10), start_epoch=288)
    assert plan.epochs == [320, 360, 400]


def test_command_builder_emits_supported_flags():
    cmd = build_periodic_eval_command(
        run_dir="runs/r",
        data_path="v.h5",
        out_dir="runs/r/periodic_evals/epoch_0040",
        prefer_checkpoint="last",
        max_samples=10_000,
        batch_size=4096,
        device="cuda",
        python_exe="python",
    )
    assert cmd[:4] == ["python", "-u", "-m", "st_lrps.evaluation.cli"]
    for flag, value in (
        ("--model-dir", "runs/r"),
        ("--data", "v.h5"),
        ("--out", "runs/r/periodic_evals/epoch_0040"),
        ("--checkpoint-prefer", "last"),
        ("--max-samples", "10000"),
        ("--batch-size", "4096"),
        ("--device", "cuda"),
    ):
        assert flag in cmd
        assert cmd[cmd.index(flag) + 1] == value


def test_dataset_resolution():
    cfg = _cfg(val_data="V.h5", test_data="T.h5", ood_data="O.h5")
    assert resolve_eval_dataset_path(cfg, "val") == "V.h5"
    assert resolve_eval_dataset_path(cfg, "test") == "T.h5"
    assert resolve_eval_dataset_path(cfg, "ood") == "O.h5"
    # val falls back to --data for single-dataset runs.
    assert resolve_eval_dataset_path(_cfg(data="D.h5"), "val") == "D.h5"
    # missing test/ood -> None (caller skips).
    assert resolve_eval_dataset_path(_cfg(), "test") is None


# ---------------------------------------------------------------------------
# 3. Resume history
# ---------------------------------------------------------------------------
def test_completed_epochs_from_history(tmp_path):
    run_dir = tmp_path / "run"
    hpath = history_path(run_dir)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    with hpath.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"epoch": 40, "status": "success"}) + "\n")
        fh.write(json.dumps({"epoch": 80, "status": "skipped"}) + "\n")
        fh.write(json.dumps({"epoch": 120, "status": "failure"}) + "\n")
    completed = completed_periodic_eval_epochs(run_dir)
    # success + skipped are not re-run; a past failure does not block training.
    assert completed == {40, 80}
    assert load_periodic_eval_history(run_dir)[120] == "failure"


def test_completed_epochs_empty_when_no_history(tmp_path):
    assert completed_periodic_eval_epochs(tmp_path / "no_run") == set()


# ---------------------------------------------------------------------------
# 4. Subprocess runner (mocked) — training-loop safety
# ---------------------------------------------------------------------------
def _make_plan(**overrides):
    base = dict(
        enabled=True,
        epochs=[40],
        dataset="val",
        prefer_checkpoint="last",
        max_samples=10_000,
        batch_size=4096,
        device="cpu",
        timeout_sec=None,
        continue_on_fail=True,
    )
    base.update(overrides)
    return PeriodicEvalPlan(**base)


def test_runner_skips_missing_dataset(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _cfg(periodic_eval_count=10, val_data=str(tmp_path / "does_not_exist.h5"))
    plan = _make_plan()

    def _fail_runner(*a, **k):  # must NOT be called
        raise AssertionError("subprocess should not run for a missing dataset")

    ok = run_periodic_eval(cfg, run_dir, 40, plan, _runner=_fail_runner)
    assert ok is True  # a skip is not a failure
    hist = load_periodic_eval_history(run_dir)
    assert hist[40] == "skipped"


def test_runner_failure_does_not_abort_and_records(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    val = tmp_path / "val.h5"
    val.write_bytes(b"\x00")
    cfg = _cfg(periodic_eval_count=10, val_data=str(val))
    plan = _make_plan(continue_on_fail=True)

    def _bad_runner(cmd, **k):
        return SimpleNamespace(returncode=1, stdout="boom-out", stderr="boom-err")

    ok = run_periodic_eval(cfg, run_dir, 40, plan, _runner=_bad_runner)
    assert ok is False  # failure reported; engine continues because continue_on_fail=True
    hist = load_periodic_eval_history(run_dir)
    assert hist[40] == "failure"
    # Captured stderr is persisted for debugging.
    assert (epoch_output_dir(run_dir, 40) / "eval_stderr.log").read_text(encoding="utf-8") == "boom-err"


def test_runner_success_records_and_extracts_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    val = tmp_path / "val.h5"
    val.write_bytes(b"\x00")
    cfg = _cfg(periodic_eval_count=10, val_data=str(val))
    plan = _make_plan()

    def _good_runner(cmd, **k):
        # The real eval CLI writes summary_metrics.json into --out; emulate it.
        out_dir = epoch_output_dir(run_dir, 40)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary_metrics.json").write_text(
            json.dumps([{"rmse_u": 1.5e-3, "rmse_a_vec": 2.0e-6, "angular_mean_deg": 3.1, "n_samples": 10000}]),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    ok = run_periodic_eval(cfg, run_dir, 40, plan, _runner=_good_runner)
    assert ok is True
    hist = load_periodic_eval_history(run_dir)
    assert hist[40] == "success"
    # Metrics summary was parsed into the record.
    records = [
        json.loads(line)
        for line in history_path(run_dir).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rec = records[-1]
    assert rec["metrics"]["rmse_u"] == 1.5e-3
    assert rec["metrics"]["rmse_a"] == 2.0e-6


# ---------------------------------------------------------------------------
# 5. UI command builder (skipped without PyQt)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_ui_disabled_emits_no_periodic_flags(qapp):
    from st_lrps.ui.studio import STLRPSTrainTab

    tab = STLRPSTrainTab()
    args = tab._build_args(show_errors=False)
    assert args is not None
    assert not any(a.startswith("--periodic-eval") for a in args)
    tab.deleteLater()


def test_ui_count_mode_emits_count_flag(qapp):
    from st_lrps.ui.studio import STLRPSTrainTab

    tab = STLRPSTrainTab()
    tab.periodic_eval_enabled.setChecked(True)
    tab.periodic_eval_mode.setCurrentIndex(tab.periodic_eval_mode.findData("count"))
    tab.periodic_eval_count.setValue(10)
    args = tab._build_args(show_errors=False)
    assert "--periodic-eval-count" in args
    assert args[args.index("--periodic-eval-count") + 1] == "10"
    assert "--periodic-eval-every-epochs" not in args
    assert "--periodic-eval-dataset" in args
    tab.deleteLater()


def test_ui_every_mode_emits_every_flag(qapp):
    from st_lrps.ui.studio import STLRPSTrainTab

    tab = STLRPSTrainTab()
    tab.periodic_eval_enabled.setChecked(True)
    tab.periodic_eval_mode.setCurrentIndex(tab.periodic_eval_mode.findData("every"))
    tab.periodic_eval_every.setValue(25)
    args = tab._build_args(show_errors=False)
    assert "--periodic-eval-every-epochs" in args
    assert args[args.index("--periodic-eval-every-epochs") + 1] == "25"
    assert "--periodic-eval-count" not in args
    tab.deleteLater()
