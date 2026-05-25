"""Offscreen smoke tests for the ST-LRPS Studio live loss plot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from st_lrps.ui.studio import LiveLossPlot  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_live_loss_plot_handles_history_gaps_and_duplicate_epochs(qapp, tmp_path):
    history_path = tmp_path / "history.jsonl"
    rows = [
        {
            "epoch": 0,
            "train_loss_total": 10.0,
            "train_loss_objective": 9.0,
            "train_loss_u": 4.0,
            "train_loss_a": 6.0,
            "val_loss_total": 8.0,
            "val_loss_base": 7.5,
            "val_loss_a": 3.0,
            "val_cos_sim": 0.8,
            "checkpoint_score": 8.0,
            "best_score": 8.0,
            "lr": 1e-4,
        },
        {
            "epoch": 0,
            "train_loss_total": 5.0,
            "val_loss_total": 4.0,
            "val_loss_a": "inf",
            "checkpoint_score": 4.0,
            "best_score": 4.0,
            "lr": 5e-5,
        },
        {
            "epoch": 1,
            "train_loss_total": 0.0,
            "train_loss_a": 2.0,
            "train_loss_dir": 0.05,
            "train_cos_sim": 0.9,
            "val_loss_total": 2.0,
            "val_loss_a": 1.0,
            "val_loss_dir": 0.04,
            "val_cos_sim": 0.92,
            "val_angular_mean_deg": 12.0,
            "checkpoint_score": 2.0,
            "best_score": 2.0,
            "lr": 2e-5,
        },
    ]
    history_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    widget = LiveLossPlot()
    widget.load_history_file(str(history_path))
    widget._chk_smooth.setChecked(True)
    widget._smooth_window.setValue(3)
    widget._chk_log_y.setChecked(True)
    widget._auto_range()
    widget.parse_line("[train] epoch=2 batch=1/1 loss_opt=1.0e+00 loss_ref=1.0e+00 U=5.0e-01 a=5.0e-01 dir=0.0e+00 cossim=0.95 lr=1e-5")

    assert widget._epochs[:2] == [1, 2]
    assert len(widget._epochs) >= 2
    assert len(widget._train_loss) == len(widget._epochs)
    assert len(widget._val_loss_a) == len(widget._epochs)
    widget.deleteLater()
