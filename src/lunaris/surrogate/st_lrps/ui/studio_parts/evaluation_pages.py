# -*- coding: utf-8 -*-
"""
ST-LRPS Studio.

PyQt6 dashboard for the lunar scalar potential surrogate codebase.

The model predicts residual potential dU(x); residual acceleration da is
computed from the gradient of that scalar field. ST-LRPS is a Sobolev-trained
lunar residual potential surrogate, not a classical q,p state-space model.

What you can do from the UI
---------------------------
- Train a potential surrogate   (runs `python -m lunaris.surrogate.st_lrps.training.cli` as a subprocess)
- Resume an interrupted training run
- Evaluate a surrogate run      (runs `python -m lunaris.surrogate.st_lrps.evaluation.cli` as a subprocess)
- Profile ST-LRPS runtime inference (runs `python -m lunaris.surrogate.st_lrps.runtime.profiling`)
- Browse evaluation plots inline (post-processing dashboard)
- Inspect runtime profiling summaries and plots
- Watch live loss curves during training (pyqtgraph)
- Queue multiple training runs for overnight execution

UX Architecture — Core features
---------------------------------
1–6.  Groups, Grid, Tooltips, Collapsible, QSettings, Path validation.

UX Architecture — Productivity features
---------------------------------
7–12. Image gallery, Log highlight, Auto-scroll, Presets, Post-run, Dependent params.

UX Architecture — Live & introspection features
-----------------------------
13. Live Loss Plotting: real-time pyqtgraph chart of train/val loss parsed from logs.
14. Dataset Introspection: auto-read HDF5 metadata (row count, attrs) on path selection.
15. Training Queue: enqueue multiple configs and run them sequentially overnight.

Run
---
  python -m lunaris.surrogate.st_lrps.ui.studio
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

from __future__ import annotations

import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from lunaris.common.paths import project_root_from_file
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys

from .qt_common import *
from .qt_common import _USE_PYSIDE


# pyqtgraph — optional, graceful fallback
try:
    import pyqtgraph as pg

    _HAS_PYQTGRAPH = pyqtgraph_matches_qt(pg)
except ImportError:
    _HAS_PYQTGRAPH = False

# h5py — optional, for dataset introspection
try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        CHECKPOINT_SCHEMA_VERSION,
        CRITICAL_CONFIG_FIELDS,
        compute_payload_sha256,
        load_checkpoint as load_artifact_checkpoint,
        make_run_layout,
        read_run_manifest,
        resolve_run_dir as resolve_artifact_run_dir,
    )
except Exception:  # pragma: no cover - UI remains usable without artifact deps
    CHECKPOINT_SCHEMA_VERSION = "st_lrps_checkpoint_v2"  # type: ignore[assignment]
    CRITICAL_CONFIG_FIELDS = tuple()  # type: ignore[assignment]
    compute_payload_sha256 = None  # type: ignore[assignment]
    load_artifact_checkpoint = None  # type: ignore[assignment]
    make_run_layout = None  # type: ignore[assignment]
    read_run_manifest = None  # type: ignore[assignment]
    resolve_artifact_run_dir = None  # type: ignore[assignment]

# Dashboard widgets and training metrics (Phase 1-8 redesign)
try:
    from lunaris.surrogate.st_lrps.ui.dashboard_widgets import (
        ExperimentHeader,
        KPIStrip,
        MetricCard,
        StructuredLogView,
        TimeMetricsStrip,
    )
    from lunaris.surrogate.st_lrps.ui.training_metrics import (
        EpochGuard,
        ETAEstimator,
        TrainingLogParser,
        TrainingMetricsStore,
        compute_auto_log_interval,
    )
    _HAS_DASHBOARD_V2 = True
except Exception:  # pragma: no cover
    _HAS_DASHBOARD_V2 = False



SCRIPT_DIR = Path(__file__).resolve().parent
_PRESETS_DIR = SCRIPT_DIR / "presets"

# The training/evaluation entry points are now subpackage modules launched via
# ``python -m``. Module execution requires the repo root (which contains the
# importable ``st_lrps`` package) as the subprocess working directory.
_REPO_ROOT = project_root_from_file(__file__)
_STLRPS_ROOT = SCRIPT_DIR.parents[1]
TRAIN_CLI_MODULE = "lunaris.surrogate.st_lrps.training.cli"
EVAL_CLI_MODULE = "lunaris.surrogate.st_lrps.evaluation.cli"

# Short, header-friendly labels for the model-representation presets.
_PRESET_SHORT = {
    "baseline_raw": "baseline",
    "recommended_physical_radial_decay": "phys-radial",
    "ablation_radial_separation": "abl:radial-sep",
    "ablation_radial_decay_scaled": "abl:radial-decay",
    "ablation_real_sh_low_degree": "abl:real-sh",
    "custom": "custom",
}
PROFILE_CLI_MODULE = "lunaris.surrogate.st_lrps.runtime.profiling"
# Filesystem locations are used only for preflight existence checks; launching
# always goes through ``-m`` so package-relative imports resolve correctly.
TRAIN_CLI_PATH = _STLRPS_ROOT / "training" / "cli.py"
EVAL_CLI_PATH = _STLRPS_ROOT / "evaluation" / "cli.py"
PROFILE_CLI_PATH = _STLRPS_ROOT / "runtime" / "profiling.py"

OUTPUT_ROOT = _REPO_ROOT / "outputs"
TRAINING_OUTPUT_ROOT = OUTPUT_ROOT / "training"
DATASET_REPORTS_OUTPUT_ROOT = OUTPUT_ROOT / "dataset_reports"
RUNTIME_PERFORMANCE_OUTPUT_ROOT = OUTPUT_ROOT / "runtime"
EVALUATION_OUTPUT_ROOT = OUTPUT_ROOT / "evaluations"
DATASET_SUITE_OUTPUT_ROOT = OUTPUT_ROOT / "datasets" / "cloud_suites"

# UI defaults are intentionally read from the generator configuration module.
# This keeps the dashboard from drifting away from the command-line SSOT when
# dataset-suite sizes, altitude ranges, seeds, or sampling knobs are tuned.
try:
    from lunaris.surrogate.st_lrps.data.spatial_cloud_parameters import (
        DEFAULT_CLOUD_SUITE_CONFIG,
        DEFAULT_SPATIAL_CLOUD_CONFIG,
        SUITE_PRESETS,
    )
except Exception:  # pragma: no cover - UI remains usable without generator deps
    DEFAULT_CLOUD_SUITE_CONFIG = None  # type: ignore[assignment]
    DEFAULT_SPATIAL_CLOUD_CONFIG = None  # type: ignore[assignment]
    SUITE_PRESETS = {}  # type: ignore[assignment]


from .common_widgets import *
from .runtime_pages import ModelReportPanel

from .common_widgets import _tune_form, _tune_inputs, _row_lineedit_with_button, _scroll_wrap, _settings, _read_json_if_exists, _split_cli_args, _format_command, _send_os_notification, _apply_status_tips, _cfg_value, _norm_path, _timestamp_slug, _safe_slug, _default_training_output_dir, _default_runtime_output_dir, _default_dataset_report_dir, _output_standard_text, _mono_font, _inspect_run_artifacts, _NoWheelOnSpinFilter


from .data_pages import *
from .data_pages import _introspect_h5


class STLRPSEvalTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        grp_input = QGroupBox("Input Files")
        form_input = QFormLayout()
        _tune_form(form_input)

        self.model_dir = ValidatedPathEdit(
            placeholder="Empty → latest run folder", check_file=False
        )
        btn_model = QPushButton("Select…")
        btn_model.clicked.connect(self._pick_model_dir)
        model_row = _row_lineedit_with_button(self.model_dir, btn_model)

        self.data = ValidatedPathEdit(
            placeholder="Empty → auto-detected", check_file=True
        )
        btn_data = QPushButton("Select…")
        btn_data.clicked.connect(self._pick_data)
        data_row = _row_lineedit_with_button(self.data, btn_data)

        self.test_data = ValidatedPathEdit(
            placeholder="Optional independent in-band test dataset", check_file=True
        )
        btn_test_data = QPushButton("Select...")
        btn_test_data.clicked.connect(lambda: self._pick_eval_dataset_path(self.test_data, "Select test dataset"))
        test_data_row = _row_lineedit_with_button(self.test_data, btn_test_data)

        self.ood_data = ValidatedPathEdit(
            placeholder="Optional OOD/extrapolation dataset", check_file=True
        )
        btn_ood_data = QPushButton("Select...")
        btn_ood_data.clicked.connect(lambda: self._pick_eval_dataset_path(self.ood_data, "Select OOD dataset"))
        ood_data_row = _row_lineedit_with_button(self.ood_data, btn_ood_data)

        self.use_config_datasets = QCheckBox("Use test/OOD dataset paths from training config if available")
        self.use_config_datasets.setChecked(False)

        self.export_hard_samples = QCheckBox("Export hard samples")
        self.export_hard_samples.setChecked(False)
        self.export_hard_samples.setEnabled(False)
        self.export_hard_samples.setToolTip(
            "Reserved for the active-learning exporter. Use Extra CLI arguments if your evaluator build exposes it."
        )
        self.hard_sample_count = QSpinBox()
        self.hard_sample_count.setRange(1, 10_000_000)
        self.hard_sample_count.setValue(10000)
        self.hard_sample_count.setEnabled(False)
        self.hard_sample_metric = QComboBox()
        self.hard_sample_metric.addItems(["accel", "angular", "cross_radial"])
        self.hard_sample_metric.setEnabled(False)

        self.dataset_name = QLineEdit("data")
        self.out_dir = ValidatedPathEdit(
            placeholder="Empty -> selected training run/evals/eval_<dataset>_<timestamp>", check_file=False
        )
        self.out_dir.setToolTip(
            f"Leave empty for run-local evaluation output. Standalone evaluation folders should use {EVALUATION_OUTPUT_ROOT}."
        )
        btn_out = QPushButton("Select…")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)
        self.run_artifact_badge = QLabel("No run selected")
        self.run_artifact_badge.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.run_artifact_summary = QPlainTextEdit()
        self.run_artifact_summary.setReadOnly(True)
        self.run_artifact_summary.setFont(_mono_font())
        self.run_artifact_summary.setMaximumHeight(150)
        self.run_artifact_summary.setPlaceholderText(
            "run_manifest.json-aware artifact summary will appear here."
        )

        form_input.addRow("Model Folder", model_row)
        form_input.addRow("Test Dataset", data_row)
        form_input.addRow("Independent Test Dataset", test_data_row)
        form_input.addRow("OOD Dataset", ood_data_row)
        form_input.addRow(self.use_config_datasets)
        form_input.addRow(self.export_hard_samples)
        form_input.addRow("Hard sample count", self.hard_sample_count)
        form_input.addRow("Hard sample metric", self.hard_sample_metric)
        form_input.addRow("HDF5 Dataset Name", self.dataset_name)
        form_input.addRow("Output Folder", out_row)
        form_input.addRow("Artifact Status", self.run_artifact_badge)
        form_input.addRow("Artifact Summary", self.run_artifact_summary)
        grp_input.setLayout(form_input)

        grp_hw = QGroupBox("Hardware and Processing")
        form_hw = QFormLayout()
        _tune_form(form_hw)
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda", "mps"])
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 10_000_000)
        self.batch_size.setValue(8192)
        self.a_sign = QDoubleSpinBox()
        self.a_sign.setDecimals(1)
        self.a_sign.setRange(-10.0, 10.0)
        self.a_sign.setValue(1.0)
        form_hw.addRow("Device", self.device)
        form_hw.addRow("Batch Size", self.batch_size)
        form_hw.addRow("Acceleration Sign", self.a_sign)
        grp_hw.setLayout(form_hw)

        grp_spatial = QGroupBox("Spatial Analysis")
        form_spatial = QFormLayout()
        _tune_form(form_spatial)
        self.r_ref_m = QLineEdit("")
        self.r_ref_m.setPlaceholderText("Empty → lunar radius")
        self.alt_bin_km = QDoubleSpinBox()
        self.alt_bin_km.setDecimals(2)
        self.alt_bin_km.setRange(1.0, 10_000.0)
        self.alt_bin_km.setValue(50.0)
        self.start = QSpinBox()
        self.start.setRange(0, 2_147_483_647)
        self.start.setValue(0)
        self.end = QLineEdit("")
        self.end.setPlaceholderText("Empty → EOF")
        self.max_points = QSpinBox()
        self.max_points.setRange(10_000, 50_000_000)
        self.max_points.setValue(500_000)
        form_spatial.addRow("Reference Radius (m)", self.r_ref_m)
        form_spatial.addRow("Altitude Bin (km)", self.alt_bin_km)
        form_spatial.addRow("Start", self.start)
        form_spatial.addRow("End", self.end)
        form_spatial.addRow("Point Limit", self.max_points)
        grp_spatial.setLayout(form_spatial)

        self.extra_args = QLineEdit("")
        self.extra_args.setPlaceholderText("Extra CLI arguments")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(grp_input, 0, 0, 1, 2)
        grid.addWidget(grp_hw, 1, 0)
        grid.addWidget(grp_spatial, 1, 1)
        extra_f = QFormLayout()
        _tune_form(extra_f)
        extra_f.addRow("Extra CLI", self.extra_args)
        extra_w = QWidget()
        extra_w.setLayout(extra_f)
        grid.addWidget(extra_w, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for g in (grp_input, grp_hw, grp_spatial):
            _tune_inputs(g)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Start Evaluation")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_progress_parser(self._parse_progress)
        self.runner.set_finished_hook(self._on_eval_finished)
        self._gallery = ImageGallery()

        top = QWidget()
        top_l = QVBoxLayout()
        top_l.setContentsMargins(8, 8, 8, 8)
        top_l.addLayout(grid)
        top.setLayout(top_l)

        bottom = QWidget()
        bl = QVBoxLayout()
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        bl.addWidget(self.runner, 1)
        bl.addWidget(self._gallery, 1)
        bottom.setLayout(bl)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(_scroll_wrap(top))
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 560])

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)
        self._effective_out_dir = ""
        self.model_dir.textChanged.connect(self._refresh_run_artifact_summary)
        self._restore_settings()
        self._refresh_run_artifact_summary()

    def _pick_model_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Model Folder", self.model_dir.text() or str(SCRIPT_DIR)
        )
        if d:
            self.model_dir.setText(_norm_path(d))

    def _refresh_run_artifact_summary(self) -> None:
        run_dir = self.model_dir.text().strip()
        if not run_dir:
            self.run_artifact_badge.setText("No run selected")
            self.run_artifact_badge.setStyleSheet("color: #94a3b8; font-size: 10px;")
            self.run_artifact_summary.setPlainText("")
            return
        status = _inspect_run_artifacts(run_dir)
        warnings = list(status.get("warnings") or [])
        badge_text = "Ready" if not warnings else f"Warnings: {len(warnings)}"
        badge_color = "#6ee7b7" if not warnings else "#f59e0b"
        if any(
            str(item).startswith(("missing_", "checkpoint_load_failed", "config_checkpoint_mismatch"))
            for item in warnings
        ):
            badge_color = "#f87171"
        self.run_artifact_badge.setText(badge_text)
        self.run_artifact_badge.setStyleSheet(f"color: {badge_color}; font-size: 10px;")

        summary_lines = [
            f"source: {status.get('source', 'fallback')}",
            f"run_dir: {status.get('run_dir') or run_dir}",
            f"best_epoch: {status.get('best_epoch')}",
            f"best_score: {status.get('best_score')}",
            f"architecture_signature: {status.get('architecture_signature')}",
            f"w0_bands: {status.get('w0_bands')}",
            f"checkpoint_schema_version: {status.get('checkpoint_schema_version')}",
            f"checkpoint_path: {status.get('checkpoint_path')}",
            f"scaler_hash: {status.get('scaler_hash')}",
            f"scaler_status: {status.get('scaler_status')}",
        ]
        if warnings:
            summary_lines.append("warnings:")
            summary_lines.extend(f"  - {warning}" for warning in warnings)
        self.run_artifact_summary.setPlainText("\n".join(summary_lines))

    def _pick_data(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Dataset",
            self.data.text() or str(SCRIPT_DIR),
            "HDF5 (*.h5 *.hdf5);;PT (*.pt);;All (*.*)",
        )
        if fn:
            self.data.setText(_norm_path(fn))

    def _pick_eval_dataset_path(self, target: ValidatedPathEdit, title: str):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            title,
            target.text() or str(SCRIPT_DIR),
            "HDF5 (*.h5 *.hdf5);;PT (*.pt);;All (*.*)",
        )
        if fn:
            target.setText(_norm_path(fn))

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Output", self.out_dir.text() or str(EVALUATION_OUTPUT_ROOT)
        )
        if d:
            self.out_dir.setText(_norm_path(d))

    def _save_settings(self):
        s = _settings()
        s.beginGroup("eval")
        s.setValue("model_dir", self.model_dir.text())
        s.setValue("data_path", self.data.text())
        s.setValue("test_data", self.test_data.text())
        s.setValue("ood_data", self.ood_data.text())
        s.setValue("use_config_datasets", self.use_config_datasets.isChecked())
        s.setValue("dataset_name", self.dataset_name.text())
        s.setValue("out_dir", self.out_dir.text())
        s.setValue("device", self.device.currentText())
        s.setValue("batch_size", self.batch_size.value())
        s.setValue("a_sign", self.a_sign.value())
        s.setValue("r_ref_m", self.r_ref_m.text())
        s.setValue("alt_bin_km", self.alt_bin_km.value())
        s.setValue("start", self.start.value())
        s.setValue("end", self.end.text())
        s.setValue("max_points", self.max_points.value())
        s.endGroup()
        s.sync()

    def _restore_settings(self):
        s = _settings()
        s.beginGroup("eval")
        if s.contains("model_dir"):
            self.model_dir.setText(str(s.value("model_dir", "")))
        if s.contains("data_path"):
            self.data.setText(str(s.value("data_path", "")))
        if s.contains("test_data"):
            self.test_data.setText(str(s.value("test_data", "")))
        if s.contains("ood_data"):
            self.ood_data.setText(str(s.value("ood_data", "")))
        if s.contains("use_config_datasets"):
            self.use_config_datasets.setChecked(str(s.value("use_config_datasets", "false")).lower() == "true")
        if s.contains("dataset_name"):
            self.dataset_name.setText(str(s.value("dataset_name", "data")))
        if s.contains("out_dir"):
            self.out_dir.setText(str(s.value("out_dir", "")))
        if s.contains("device"):
            self.device.setCurrentText(str(s.value("device", "auto")))
        if s.contains("batch_size"):
            self.batch_size.setValue(int(s.value("batch_size", 8192)))
        if s.contains("a_sign"):
            self.a_sign.setValue(float(s.value("a_sign", 1.0)))
        if s.contains("r_ref_m"):
            self.r_ref_m.setText(str(s.value("r_ref_m", "")))
        if s.contains("alt_bin_km"):
            self.alt_bin_km.setValue(float(s.value("alt_bin_km", 50.0)))
        if s.contains("start"):
            self.start.setValue(int(s.value("start", 0)))
        if s.contains("end"):
            self.end.setText(str(s.value("end", "")))
        if s.contains("max_points"):
            self.max_points.setValue(int(s.value("max_points", 500_000)))
        s.endGroup()

    def _start(self):
        if not EVAL_CLI_PATH.exists():
            QMessageBox.critical(self, "Not Found", "st_lrps/evaluation/cli.py is required.")
            return
        args = ["-u", "-m", EVAL_CLI_MODULE]
        md = self.model_dir.text().strip()
        if md:
            if not Path(md).exists():
                QMessageBox.critical(self, "Not Found", f"Model:\n{md}")
                return
            args += ["--model-dir", md]
        dp = self.data.text().strip()
        if dp:
            if not Path(dp).exists():
                QMessageBox.critical(self, "Not Found", f"Dataset:\n{dp}")
                return
            args += ["--data", dp]
        for flag, path in (
            ("--test-data", self.test_data.text().strip()),
            ("--ood-data", self.ood_data.text().strip()),
        ):
            if path:
                if not Path(path).exists():
                    QMessageBox.critical(self, "Not Found", f"{flag}:\n{path}")
                    return
                args += [flag, path]
        if self.use_config_datasets.isChecked():
            args += ["--use-config-datasets"]
        args += ["--dataset-name", self.dataset_name.text().strip() or "data"]
        od = self.out_dir.text().strip()
        if od:
            args += ["--out", od]
        args += [
            "--device",
            self.device.currentText(),
            "--batch-size",
            str(self.batch_size.value()),
            "--a-sign",
            str(self.a_sign.value()),
            "--alt-bin-km",
            str(self.alt_bin_km.value()),
            "--start",
            str(self.start.value()),
        ]
        if self.end.text().strip():
            args += ["--end", self.end.text().strip()]
        args += ["--max-points-for-plots", str(self.max_points.value())]
        if self.r_ref_m.text().strip():
            args += ["--r-ref-m", self.r_ref_m.text().strip()]
        extra = self.extra_args.text().strip()
        if extra:
            extra_args, err = _split_cli_args(extra)
            if err:
                QMessageBox.critical(self, "Invalid extra CLI arguments", err)
                return
            args += extra_args or []
        self.runner.progress.setRange(0, 0)
        self._effective_out_dir = od
        self.runner.set_output_dir(od)
        self._gallery.clear_gallery()
        self._save_settings()
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _on_eval_finished(self, exit_code, exit_status):
        if exit_status != QProcess.ExitStatus.NormalExit:
            return
        out_dir = self._effective_out_dir
        if not out_dir or not Path(out_dir).is_dir():
            text = self.runner.log.toPlainText()
            for pat in [
                r"(?:out_dir|Output dir|Saving to|Results saved to)\s*[:=]\s*(.+)",
                r"Plots saved to\s*[:=]?\s*(.+)",
            ]:
                m = re.search(pat, text)
                if m:
                    c = m.group(1).strip().strip("'\"")
                    if Path(c).is_dir():
                        out_dir = c
                        break
        if out_dir and Path(out_dir).is_dir():
            # Load plots in priority order
            _PRIORITY_PLOTS = [
                "parity_U",
                "hist_rel_err_accel_pct",
                "hist_angular_err_deg",
                "binned_mape_accel_vs_alt",
                "binned_mape_U_vs_alt",
                "scatter_relerr_accel_vs_alt",
                "ood_bar_accel_rmse",
            ]
            all_imgs: List[Path] = []
            for sub in ("", "test", "ood"):
                sd = Path(out_dir) / sub if sub else Path(out_dir)
                if sd.is_dir():
                    all_imgs += list(sd.glob("*.png")) + list(sd.glob("*.jpg"))

            # Sort: priority plots first, then rest alphabetically
            def _sort_key(p: Path) -> tuple:
                stem = p.stem.lower()
                for i, pn in enumerate(_PRIORITY_PLOTS):
                    if pn.lower() in stem:
                        return (0, i, stem)
                return (1, 0, stem)

            all_imgs.sort(key=_sort_key)
            cnt = self._gallery.load_images(all_imgs)
            if cnt:
                self.runner.append(f"\n[UI] {cnt} plots loaded: {out_dir}")
            self.runner.set_output_dir(out_dir)
            self.runner.btn_open_folder.setVisible(True)

            # Parse eval_report.json for metric summary card
            self._show_eval_metrics(out_dir)

    def _show_eval_metrics(self, out_dir: str) -> None:
        """Parse eval_report.json and append a metric summary to the log."""
        report_path = Path(out_dir) / "eval_report.json"
        if not report_path.exists():
            return
        try:
            with open(report_path, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
        except Exception as exc:
            self.runner.append(f"[UI] eval_report.json parse error: {exc}")
            return

        lines = ["\n[UI] ─── Evaluation Metrics Summary ───"]

        def _fmt(section: str, label: str) -> None:
            s = rep.get(section)
            if not isinstance(s, dict):
                return
            parts = []
            for key in ("rmse", "rel_mean", "rel_p50", "rel_p90", "mean", "p50", "p90"):
                if key in s:
                    parts.append(f"{key}={s[key]:.4g}")
            if parts:
                lines.append(f"  {label}: " + "  ".join(parts))

        _fmt("U",    "Potential U")
        _fmt("accel","Acceleration |a|")
        _fmt("angle","Angular error")

        # OOD check
        ood_warn = []
        for band in ("lower_ood", "upper_ood"):
            sec = rep.get(band)
            if isinstance(sec, dict):
                n = sec.get("N", sec.get("n", -1))
                if isinstance(n, (int, float)) and int(n) == 0:
                    ood_warn.append(band)
        if ood_warn:
            lines.append(
                f"  [WARNING] OOD bands {ood_warn} have N=0 — "
                "OOD dataset was not evaluated or contains no points in the OOD band."
            )

        self.runner.append("\n".join(lines))

    def _parse_progress(self, line):
        if line.strip() and self.runner.progress.maximum() != 0:
            self.runner.progress.setValue(
                min(self.runner.progress.value() + 1, self.runner.progress.maximum())
            )
        if "EVAL SUMMARY" in line or "Evaluation completed" in line or "Done" in line:
            self.runner.progress.setRange(0, 100)
            self.runner.progress.setValue(100)
        if not self._effective_out_dir:
            m = re.search(r"(?:out_dir|Output dir|Saving to)\s*[:=]\s*(.+)", line)
            if m:
                c = m.group(1).strip().strip("'\"")
                if Path(c).is_dir():
                    self._effective_out_dir = c
                    self.runner.set_output_dir(c)


class EvaluationPage(QWidget):
    """Evaluation workspace: model quality and reporting."""

    def __init__(self, eval_tab: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.report_panel = ModelReportPanel()
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(_scroll_wrap(self.report_panel), "Model Report")
        tabs.addTab(eval_tab, "Accuracy Evaluation")
        self._tabs = tabs
        lo = QVBoxLayout()
        lo.setContentsMargins(22, 20, 22, 20)
        lo.setSpacing(14)
        title = QLabel("Evaluation")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #e8ecf8;")
        subtitle = QLabel("Inspect artifacts and evaluate pointwise surrogate accuracy.")
        subtitle.setStyleSheet("color: #94a3b8; font-size: 12px;")
        lo.addWidget(title)
        lo.addWidget(subtitle)
        lo.addWidget(tabs, 1)
        self.setLayout(lo)

