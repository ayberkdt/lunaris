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

from st_lrps.ui.studio import (  # noqa: E402
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
