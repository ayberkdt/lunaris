"""Lightweight ST-LRPS Studio command-builder tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from lunaris.surrogate.st_lrps.ui.studio import (  # noqa: E402
    PROFILE_CLI_MODULE,
    STLRPSProfilingTab,
    STLRPSTrainTab,
    TRAIN_CLI_MODULE,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def _set_combo_data(combo, value: str) -> None:
    idx = combo.findData(value)
    assert idx >= 0
    combo.setCurrentIndex(idx)


def test_train_tab_builds_resume_training_args(qapp, tmp_path):
    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()

    tab = STLRPSTrainTab()
    _set_combo_data(tab.workflow_mode, "train_only")
    _set_combo_data(tab.dataset_mode, "single")
    tab.data.setText("")
    tab.out_dir.setText("")
    tab.resume_enabled.setChecked(True)
    tab.resume_from.setText(str(run_dir))
    _set_combo_data(tab.resume_checkpoint, "last")
    _set_combo_data(tab.resume_history_mode, "append")

    args = tab._build_args(show_errors=False)

    assert args is not None
    assert "-m" in args
    assert TRAIN_CLI_MODULE in args
    assert "--resume-from" in args
    assert args[args.index("--resume-from") + 1] == str(run_dir)
    assert "--resume-checkpoint" in args
    assert args[args.index("--resume-checkpoint") + 1] == "last"
    assert "--resume-append-history" in args
    assert "--train-data" not in args
    assert "--val-data" not in args
    tab.deleteLater()


def test_train_tab_omits_debug_flags(qapp):
    """Normal Studio command generation must not emit dev/debug flags."""
    tab = STLRPSTrainTab()
    _set_combo_data(tab.workflow_mode, "train_then_eval")
    _set_combo_data(tab.dataset_mode, "single")
    tab.data.setText("")
    tab.out_dir.setText("")

    args = tab._build_args(show_errors=False)
    assert args is not None
    for forbidden in ("--quick-check", "--max-train-batches", "--max-val-batches"):
        assert forbidden not in args, f"{forbidden} should not be emitted"
    # The normal workflow still carries the supported model/logging flags.
    assert "--model-preset" in args
    assert "--log-every-mode" in args
    tab.deleteLater()


def test_workflow_selector_has_no_quick_check(qapp):
    tab = STLRPSTrainTab()
    values = {tab.workflow_mode.itemData(i) for i in range(tab.workflow_mode.count())}
    assert "quick_check" not in values
    assert values == {"train_only", "eval_only", "train_then_eval", "queue"}
    tab.deleteLater()


def test_old_profile_with_debug_fields_loads(qapp):
    """Backward compatibility: profiles carrying removed debug fields still load."""
    tab = STLRPSTrainTab()
    legacy = {
        "hidden": 256, "depth": 4, "epochs": 12,
        "quick_check": True,          # removed-from-UI debug field
        "max_train_batches": 5,       # removed-from-UI debug field
        "max_val_batches": 2,         # removed-from-UI debug field
        "obsolete_field_xyz": 123,    # unknown field must be ignored
    }
    tab._apply_config(legacy)  # must not raise
    assert tab.hidden.value() == 256
    assert tab.depth.value() == 4
    # Debug controls remain hidden regardless of the loaded value.
    assert not tab.quick_check.isVisible()
    tab.deleteLater()


def test_profiling_tab_builds_runtime_profile_args(qapp, tmp_path):
    model_dir = tmp_path / "model_run"
    model_dir.mkdir()
    out_dir = tmp_path / "profile_out"

    tab = STLRPSProfilingTab()
    tab.profile_model_dir.setText(str(model_dir))
    tab.profile_out_dir.setText(str(out_dir))
    _set_combo_data(tab.profile_input_source, "synthetic")

    args = tab._build_profile_args(show_errors=False)

    assert args is not None
    assert "-m" in args
    assert PROFILE_CLI_MODULE in args
    assert "--model-dir" in args
    assert args[args.index("--model-dir") + 1] == str(model_dir)
    assert "--batch-sizes" in args
    assert "--chunk-sizes" in args
    assert "--out-dir" in args
    assert args[args.index("--out-dir") + 1] == str(out_dir)
    assert "--data" not in args
    tab.deleteLater()
