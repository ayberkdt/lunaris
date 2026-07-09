"""
UI regression tests for the ensemble analysis workspace and log-panel collapse.

These tests deliberately stay lightweight while still touching the concrete Qt
widgets that the desktop user interacts with.  They provide coverage for two
user-visible behaviors that recently regressed:

1. the lower terminal/log panel should collapse to a compact dock rail instead
   of leaving a large dead area behind
2. the ensemble analysis workspace should be able to load a saved archive
   and populate its summary metrics without requiring a full app launch
"""

from __future__ import annotations

import pytest

pytest.importorskip('PySide6.QtWidgets')


import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from lunaris.analysis.ensemble.plotting import plot_ensemble_report
from lunaris.analysis.ensemble.statistics import compute_ensemble_report
from lunaris.common.batch_defs import BatchPropagationResult
from lunaris.ui.app import MainWindow
from lunaris.ui.components.ensemble_analysis_panel import EnsembleAnalysisPanel


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_log_panel_collapse_reduces_splitter_footprint() -> None:
    app = _app()
    win = MainWindow()
    try:
        win.show()
        app.processEvents()

        if not win.log_panel.is_collapsed:
            win._toggle_log_collapsed()
            app.processEvents()

        collapsed_sizes = win.main_splitter.sizes()
        assert collapsed_sizes[1] <= 60
        assert win.log_panel.minimumHeight() >= 34
        assert win.log_panel.is_collapsed is True

        win._toggle_log_collapsed()
        app.processEvents()

        restored_sizes = win.main_splitter.sizes()
        assert restored_sizes[1] >= 120
        assert win.log_panel.minimumHeight() >= 150
        assert win.log_panel.is_collapsed is False
    finally:
        win.tick_timer.stop()
        win.hide()
        win.deleteLater()
        app.processEvents()


def test_batch_analysis_panel_loads_npz_archive_and_populates_summary(tmp_path: Path) -> None:
    app = _app()

    t = np.asarray([0.0, 60.0, 120.0], dtype=np.float64)
    Y = np.asarray(
        [
            [
                [1_837_500.0, 0.0, 0.0, 0.0, 0.0, 1_630.0],
                [1_837_650.0, 0.0, 0.0, 0.0, 0.0, 1_631.0],
                [1_837_800.0, 0.0, 0.0, 0.0, 0.0, 1_632.0],
            ],
            [
                [1_837_400.0, 1_500.0, 0.0, 0.0, 0.0, 1_629.5],
                [1_837_520.0, 1_550.0, 0.0, 0.0, 0.0, 1_630.5],
                [1_837_700.0, 1_600.0, 0.0, 0.0, 0.0, 1_631.5],
            ],
            [
                [1_837_250.0, 3_000.0, 0.0, 0.0, 0.0, 1_628.5],
                [1_837_430.0, 3_050.0, 0.0, 0.0, 0.0, 1_629.5],
                [1_737_200.0, 3_100.0, 0.0, 0.0, 0.0, 10.0],
            ],
        ],
        dtype=np.float64,
    )
    sc_samples = np.asarray(
        [
            [1000.0, 5.0, 2.2, 1.5],
            [1000.0, 5.0, 2.2, 1.5],
            [1000.0, 5.0, 2.2, 1.5],
        ],
        dtype=np.float64,
    )
    impact_mask = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    t_impact = np.asarray([np.nan, np.nan, 120.0], dtype=np.float64)

    result = BatchPropagationResult(
        t=t,
        Y=Y,
        sc_samples=sc_samples,
        impact_mask=impact_mask,
        t_impact=t_impact,
    )

    output_path = tmp_path / "ensemble_analysis_test.npz"
    np.savez_compressed(
        str(output_path),
        t=result.t,
        Y=result.Y,
        sc_samples=result.sc_samples,
        impact_flags=result.impact_mask,
        t_impact=result.t_impact,
        valid_mask=np.ones(3, dtype=np.float64),
        impact_position_inertial_m=np.full((3, 3), np.nan, dtype=np.float64),
        impact_position_fixed_m=np.full((3, 3), np.nan, dtype=np.float64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "archive_schema_version": 2,
                    "n_samples": 3,
                    "seed": 42,
                    "duration_s": 120.0,
                    "output_dt_s": 60.0,
                    "backend": "CPU",
                    "requested_batch_backend": "cpu_sh",
                    "actual_batch_backend": "cpu_sh",
                    "batch_backend": "cpu_sh",
                    "detect_impact": True,
                    "compute_impact_statistics": True,
                }
            ),
            dtype=np.str_,
        ),
    )

    panel = EnsembleAnalysisPanel()
    panel.ent_result_path.setText(str(output_path))
    panel._start_analysis()

    deadline = time.time() + 10.0
    while panel._worker is not None and panel._worker.isRunning() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()

    assert panel._stats is not None
    assert panel._summary_labels["archive"].text() == output_path.name
    assert panel._summary_labels["n_samples"].text() == "3 scenarios"
    assert panel._summary_labels["n_impacts"].text() == "1 (33.33%)"
    assert panel.btn_export_report.isEnabled() is True
    assert panel.btn_refresh_plot.isEnabled() is True


def test_plot_ensemble_report_writes_pdf_with_summary_page(tmp_path: Path) -> None:
    t = np.asarray([0.0, 60.0, 120.0], dtype=np.float64)
    Y = np.asarray(
        [
            [
                [1_837_500.0, 0.0, 0.0, 0.0, 0.0, 1_630.0],
                [1_837_650.0, 0.0, 0.0, 0.0, 0.0, 1_631.0],
                [1_837_800.0, 0.0, 0.0, 0.0, 0.0, 1_632.0],
            ],
            [
                [1_837_400.0, 1_500.0, 0.0, 0.0, 0.0, 1_629.5],
                [1_837_520.0, 1_550.0, 0.0, 0.0, 0.0, 1_630.5],
                [1_837_700.0, 1_600.0, 0.0, 0.0, 0.0, 1_631.5],
            ],
            [
                [1_837_250.0, 3_000.0, 0.0, 0.0, 0.0, 1_628.5],
                [1_837_430.0, 3_050.0, 0.0, 0.0, 0.0, 1_629.5],
                [1_737_200.0, 3_100.0, 0.0, 0.0, 0.0, 10.0],
            ],
        ],
        dtype=np.float64,
    )
    sc_samples = np.asarray(
        [
            [1000.0, 5.0, 2.2, 1.5],
            [1000.0, 5.0, 2.2, 1.5],
            [1000.0, 5.0, 2.2, 1.5],
        ],
        dtype=np.float64,
    )
    impact_mask = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    t_impact = np.asarray([np.nan, np.nan, 120.0], dtype=np.float64)
    result = BatchPropagationResult(
        t=t,
        Y=Y,
        sc_samples=sc_samples,
        impact_mask=impact_mask,
        t_impact=t_impact,
    )

    stats = compute_ensemble_report(result, compute_oe=False)
    out_pdf = tmp_path / "ensemble_report.pdf"
    figures = plot_ensemble_report(result, stats, output_path=str(out_pdf), show=False)

    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
    assert len(figures) >= 6
