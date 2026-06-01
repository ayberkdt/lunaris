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
from .common_widgets import _tune_form, _tune_inputs, _row_lineedit_with_button, _scroll_wrap, _settings, _read_json_if_exists, _split_cli_args, _format_command, _send_os_notification, _apply_status_tips, _cfg_value, _norm_path, _timestamp_slug, _safe_slug, _default_training_output_dir, _default_runtime_output_dir, _default_dataset_report_dir, _output_standard_text, _mono_font, _make_page_header, _style_command_preview, _inspect_run_artifacts, _NoWheelOnSpinFilter


from .data_pages import *
from .data_pages import _introspect_h5


class STLRPSProfilingTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(520)

        grp_model = QGroupBox("Model / Run")
        form_model = QFormLayout()
        _tune_form(form_model)

        self.profile_model_dir = ValidatedPathEdit(
            placeholder="Trained run directory or checkpoint", check_file=False
        )
        btn_model_dir = QPushButton("Browse Run...")
        btn_model_dir.clicked.connect(self._pick_profile_model_dir)
        btn_model_ckpt = QPushButton("Browse Checkpoint...")
        btn_model_ckpt.clicked.connect(self._pick_profile_checkpoint)
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(6)
        model_row.addWidget(self.profile_model_dir, 1)
        model_row.addWidget(btn_model_dir)
        model_row.addWidget(btn_model_ckpt)
        model_widget = QWidget()
        model_widget.setLayout(model_row)
        form_model.addRow("Model/run directory", model_widget)
        grp_model.setLayout(form_model)

        grp_runtime = QGroupBox("Runtime Sweep")
        form_runtime = QFormLayout()
        _tune_form(form_runtime)
        self.profile_device = QComboBox()
        self.profile_device.addItems(["auto", "cpu", "cuda"])
        self.profile_batch_sizes = QLineEdit("1,16,128,1024,8192")
        self.profile_chunk_sizes = QLineEdit("none,512,1024,4096,8192")
        self.profile_n_warmup = QSpinBox()
        self.profile_n_warmup.setRange(0, 100000)
        self.profile_n_warmup.setValue(10)
        self.profile_n_repeat = QSpinBox()
        self.profile_n_repeat.setRange(1, 100000)
        self.profile_n_repeat.setValue(50)
        self.profile_seed = QSpinBox()
        self.profile_seed.setRange(0, 2_147_483_647)
        self.profile_seed.setValue(42)
        form_runtime.addRow("Device", self.profile_device)
        form_runtime.addRow("Batch sizes", self.profile_batch_sizes)
        form_runtime.addRow("Chunk sizes", self.profile_chunk_sizes)
        form_runtime.addRow("Warmup calls", self.profile_n_warmup)
        form_runtime.addRow("Repeat calls", self.profile_n_repeat)
        form_runtime.addRow("Seed", self.profile_seed)
        grp_runtime.setLayout(form_runtime)

        grp_input = QGroupBox("Input Queries")
        form_input = QFormLayout()
        _tune_form(form_input)
        self.profile_input_source = QComboBox()
        self.profile_input_source.addItem("synthetic", "synthetic")
        self.profile_input_source.addItem("dataset", "dataset")
        self.profile_data = ValidatedPathEdit(placeholder="HDF5 dataset path", check_file=True)
        btn_data = QPushButton("Browse...")
        btn_data.clicked.connect(self._pick_profile_data)
        self.profile_dataset_name = QLineEdit("data")
        self.profile_alt_min_km = QDoubleSpinBox()
        self.profile_alt_min_km.setDecimals(2)
        self.profile_alt_min_km.setRange(-10000.0, 1_000_000.0)
        self.profile_alt_min_km.setValue(100.0)
        self.profile_alt_max_km = QDoubleSpinBox()
        self.profile_alt_max_km.setDecimals(2)
        self.profile_alt_max_km.setRange(-10000.0, 1_000_000.0)
        self.profile_alt_max_km.setValue(2000.0)
        form_input.addRow("Input source", self.profile_input_source)
        form_input.addRow("Dataset", _row_lineedit_with_button(self.profile_data, btn_data))
        form_input.addRow("Dataset name", self.profile_dataset_name)
        form_input.addRow("Altitude min (km)", self.profile_alt_min_km)
        form_input.addRow("Altitude max (km)", self.profile_alt_max_km)
        grp_input.setLayout(form_input)

        grp_output = QGroupBox("Output / Options")
        form_output = QFormLayout()
        _tune_form(form_output)
        self.profile_out_dir = ValidatedPathEdit(
            placeholder=f"{RUNTIME_PERFORMANCE_OUTPUT_ROOT}/<run>_<timestamp>", check_file=False
        )
        self.profile_out_dir.setText(str(_default_runtime_output_dir()))
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._pick_profile_out_dir)
        self.profile_compare_classic_sh = QCheckBox("Compare classic SH")
        self.profile_classic_sh_degree = QSpinBox()
        self.profile_classic_sh_degree.setRange(1, 10000)
        self.profile_classic_sh_degree.setValue(60)
        self.profile_json_only = QCheckBox("JSON only")
        self.profile_verbose = QCheckBox("Verbose output")
        self.profile_extra_args = QLineEdit("")
        self.profile_extra_args.setPlaceholderText("Extra profiling CLI arguments")
        form_output.addRow("Output directory", _row_lineedit_with_button(self.profile_out_dir, btn_out))
        form_output.addRow(self.profile_compare_classic_sh)
        form_output.addRow("Classic SH degree", self.profile_classic_sh_degree)
        form_output.addRow(self.profile_json_only)
        form_output.addRow(self.profile_verbose)
        form_output.addRow("Extra profiling args", self.profile_extra_args)
        grp_output.setLayout(form_output)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(grp_runtime, 0, 0)
        grid.addWidget(grp_input, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for group in (grp_model, grp_runtime, grp_input, grp_output):
            _tune_inputs(group)

        self.command_preview = QPlainTextEdit()
        _style_command_preview(self.command_preview, min_h=82, max_h=120)
        self.command_preview.setPlaceholderText(
            f"Click Preview Command to see the exact python -m {PROFILE_CLI_MODULE} command."
        )
        self.command_warning = QLabel("")
        self.command_warning.setWordWrap(True)
        self.command_warning.setStyleSheet("color: #fbbf24; font-size: 11px;")
        btn_preview = QPushButton("Preview Command")
        btn_preview.clicked.connect(self._refresh_profile_preview)
        btn_copy = QPushButton("Copy Command")
        btn_copy.clicked.connect(self._copy_profile_command)
        preview_buttons = QHBoxLayout()
        preview_buttons.setContentsMargins(0, 0, 0, 0)
        preview_buttons.addWidget(btn_preview)
        preview_buttons.addWidget(btn_copy)
        preview_buttons.addStretch(1)

        preview_form = QFormLayout()
        _tune_form(preview_form)
        preview_buttons_widget = QWidget()
        preview_buttons_widget.setLayout(preview_buttons)
        preview_form.addRow("", preview_buttons_widget)

        # Rebuild Runtime Performance page layout (Phase 2):
        # We refactor the profiling dashboard to place Configuration in a scrollable panel,
        # Action controls in a dedicated row, and output logs + markdown preview + plots
        # in a spacious splitter workspace below.
        
        # Top Config Container (Model and Sweep parameters stacked vertically or in two-column cards)
        config_card = QFrame()
        config_card.setObjectName("profileConfigCard")
        config_card.setStyleSheet(
            "QFrame#profileConfigCard {"
            "  background: rgba(11, 16, 32, 0.72);"
            "  border: 1px solid rgba(185, 194, 221, 0.12);"
            "  border-radius: 12px;"
            "}"
        )
        config_l = QVBoxLayout()
        config_l.setContentsMargins(14, 14, 14, 14)
        config_l.setSpacing(10)
        config_heading = QLabel("Sweep Configuration")
        config_heading.setStyleSheet("font-size: 14px; font-weight: 700; color: #e8ecf8;")
        config_l.addWidget(config_heading)
        config_l.addLayout(grid)
        config_card.setLayout(config_l)

        launch_card = QFrame()
        launch_card.setObjectName("profileLaunchCard")
        launch_card.setStyleSheet(
            "QFrame#profileLaunchCard {"
            "  background: rgba(11, 16, 32, 0.82);"
            "  border: 1px solid rgba(185, 194, 221, 0.13);"
            "  border-radius: 12px;"
            "}"
        )
        launch_l = QVBoxLayout()
        launch_l.setContentsMargins(16, 16, 16, 16)
        launch_l.setSpacing(12)
        launch_heading = QLabel("Model / Options")
        launch_heading.setStyleSheet("font-size: 14px; font-weight: 700; color: #e8ecf8;")
        launch_l.addWidget(launch_heading)
        launch_l.addWidget(grp_model)
        launch_l.addWidget(grp_output)
        launch_card.setLayout(launch_l)

        # Upper setup: side-by-side configurations
        upper_setup = QWidget()
        upper_setup_lo = QHBoxLayout()
        upper_setup_lo.setContentsMargins(0, 0, 0, 0)
        upper_setup_lo.setSpacing(14)
        upper_setup_lo.addWidget(launch_card, 1)
        upper_setup_lo.addWidget(config_card, 1)
        upper_setup.setLayout(upper_setup_lo)
        
        # Wrap upper config in a roomy scroll area
        scroll_config = _scroll_wrap(upper_setup)
        scroll_config.setMinimumHeight(300)
        scroll_config.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Actions & Command preview panel
        actions_card = QFrame()
        actions_card.setObjectName("profileActionsCard")
        actions_card.setStyleSheet(
            "QFrame#profileActionsCard {"
            "  background: rgba(8, 13, 26, 0.82);"
            "  border: 1px solid rgba(53, 208, 255, 0.18);"
            "  border-radius: 12px;"
            "}"
        )
        actions_l = QVBoxLayout()
        actions_l.setContentsMargins(12, 10, 12, 10)
        actions_l.setSpacing(8)
        
        # Command preview directly inside actions
        cmd_head = QLabel("Generated CLI Command")
        cmd_head.setStyleSheet("font-size: 11px; font-weight: 800; color: #35d0ff;")
        
        actions_l.addWidget(cmd_head)
        actions_l.addWidget(self.command_preview)
        actions_l.addWidget(self.command_warning)
        actions_l.addLayout(preview_form)
        actions_card.setLayout(actions_l)

        # Instantiate separate controls for Profiling (Phase 5 parenting)
        self.runner = ProcessPane()
        self.runner.btn_start.setText("Run Profiling")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_finished_hook(self._on_profile_finished)
        self.runner.set_stop_hint("")
        
        self.btn_run_profiling = QPushButton("Run Profiling")
        self.btn_run_profiling.setProperty("kind", "primary")
        self.btn_run_profiling.clicked.connect(self._start)
        
        self.btn_stop_profiling = QPushButton("Stop")
        self.btn_stop_profiling.setProperty("kind", "danger")
        self.btn_stop_profiling.setEnabled(False)
        self.btn_stop_profiling.clicked.connect(lambda: (setattr(self, "_user_stopped_profile", True), self.runner.stop()))
        
        self.btn_clear_profiling = QPushButton("Clear Log")
        self.btn_clear_profiling.setProperty("kind", "ghost")
        self.btn_clear_profiling.clicked.connect(self.runner.log.clear)

        # Profiling actions row
        profiling_bar = QHBoxLayout()
        profiling_bar.setContentsMargins(4, 4, 4, 4)
        profiling_bar.setSpacing(10)
        profiling_bar.addWidget(self.btn_run_profiling)
        profiling_bar.addWidget(self.btn_stop_profiling)
        profiling_bar.addWidget(self.runner.progress, 1)  # Direct progress synchronization
        profiling_bar.addWidget(self.btn_clear_profiling)

        self.profile_summary = QPlainTextEdit()
        _style_command_preview(self.profile_summary, min_h=180)
        self.profile_summary.setPlaceholderText("runtime_profile_summary.md will appear here after profiling.")
        self._gallery = ImageGallery()
        self._gallery._placeholder.setText("Runtime profile plots will appear here.")

        # Bottom Pane: tabbed log and result view to prevent squishing
        bottom_tabs = QTabWidget()
        bottom_tabs.setDocumentMode(True)
        bottom_tabs.addTab(self.runner.raw_log_widget(), "Process Output Log")
        bottom_tabs.addTab(self.profile_summary, "Markdown Profile Summary")
        bottom_tabs.addTab(self._gallery, "Visual Latency & Throughput Plots")

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(scroll_config)
        main_splitter.addWidget(actions_card)
        
        lower_workspace = QWidget()
        lower_workspace_lo = QVBoxLayout()
        lower_workspace_lo.setContentsMargins(0, 0, 0, 0)
        lower_workspace_lo.setSpacing(6)
        lower_workspace_lo.addLayout(profiling_bar)
        lower_workspace_lo.addWidget(bottom_tabs, 1)
        lower_workspace.setLayout(lower_workspace_lo)
        
        main_splitter.addWidget(lower_workspace)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setStretchFactor(2, 4)
        main_splitter.setSizes([320, 110, 420])

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(main_splitter, 1)
        self.setLayout(layout)

        self._effective_out_dir = ""
        self.profile_input_source.currentIndexChanged.connect(self._on_input_source_changed)
        self._restore_settings()
        self._on_input_source_changed()
        self._refresh_profile_preview()

    def _pick_profile_model_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Model/run directory", self.profile_model_dir.text() or str(_REPO_ROOT)
        )
        if d:
            self.profile_model_dir.setText(_norm_path(d))

    def _pick_profile_checkpoint(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Model checkpoint",
            self.profile_model_dir.text() or str(_REPO_ROOT),
            "PyTorch checkpoints (*.pt);;All (*.*)",
        )
        if fn:
            self.profile_model_dir.setText(_norm_path(fn))

    def _pick_profile_data(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Profiling dataset",
            self.profile_data.text() or str(_REPO_ROOT),
            "HDF5 (*.h5 *.hdf5);;All (*.*)",
        )
        if fn:
            self.profile_data.setText(_norm_path(fn))

    def _pick_profile_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self,
            "Profiling output directory",
            self.profile_out_dir.text() or str(RUNTIME_PERFORMANCE_OUTPUT_ROOT),
        )
        if d:
            self.profile_out_dir.setText(_norm_path(d))

    def _on_input_source_changed(self, *_args) -> None:
        is_dataset = self.profile_input_source.currentData() == "dataset"
        self.profile_data.setEnabled(is_dataset)
        self._refresh_profile_preview()

    def _save_settings(self) -> None:
        s = _settings()
        s.beginGroup("profiling")
        s.setValue("profile_model_dir", self.profile_model_dir.text())
        s.setValue("profile_device", self.profile_device.currentText())
        s.setValue("profile_batch_sizes", self.profile_batch_sizes.text())
        s.setValue("profile_chunk_sizes", self.profile_chunk_sizes.text())
        s.setValue("profile_n_warmup", self.profile_n_warmup.value())
        s.setValue("profile_n_repeat", self.profile_n_repeat.value())
        s.setValue("profile_seed", self.profile_seed.value())
        s.setValue("profile_input_source", self.profile_input_source.currentData() or "synthetic")
        s.setValue("profile_data", self.profile_data.text())
        s.setValue("profile_dataset_name", self.profile_dataset_name.text())
        s.setValue("profile_alt_min_km", self.profile_alt_min_km.value())
        s.setValue("profile_alt_max_km", self.profile_alt_max_km.value())
        s.setValue("profile_out_dir", self.profile_out_dir.text())
        s.setValue("profile_compare_classic_sh", self.profile_compare_classic_sh.isChecked())
        s.setValue("profile_classic_sh_degree", self.profile_classic_sh_degree.value())
        s.setValue("profile_json_only", self.profile_json_only.isChecked())
        s.setValue("profile_verbose", self.profile_verbose.isChecked())
        s.setValue("profile_extra_args", self.profile_extra_args.text())
        s.endGroup()
        s.sync()

    def _restore_settings(self) -> None:
        s = _settings()
        s.beginGroup("profiling")

        def _st(key: str, default: str = "") -> str:
            return str(s.value(key, default)) if s.contains(key) else default

        def _i(key: str, default: int) -> int:
            try:
                return int(s.value(key, default))
            except Exception:
                return default

        def _f(key: str, default: float) -> float:
            try:
                return float(s.value(key, default))
            except Exception:
                return default

        def _b(key: str, default: bool = False) -> bool:
            return str(s.value(key, str(default).lower())).lower() == "true"

        self.profile_model_dir.setText(_st("profile_model_dir", ""))
        self.profile_device.setCurrentText(_st("profile_device", "auto"))
        self.profile_batch_sizes.setText(_st("profile_batch_sizes", "1,16,128,1024,8192"))
        self.profile_chunk_sizes.setText(_st("profile_chunk_sizes", "none,512,1024,4096,8192"))
        self.profile_n_warmup.setValue(_i("profile_n_warmup", 10))
        self.profile_n_repeat.setValue(_i("profile_n_repeat", 50))
        self.profile_seed.setValue(_i("profile_seed", 42))
        source = _st("profile_input_source", "synthetic")
        idx = self.profile_input_source.findData(source)
        if idx >= 0:
            self.profile_input_source.setCurrentIndex(idx)
        self.profile_data.setText(_st("profile_data", ""))
        self.profile_dataset_name.setText(_st("profile_dataset_name", "data"))
        self.profile_alt_min_km.setValue(_f("profile_alt_min_km", 100.0))
        self.profile_alt_max_km.setValue(_f("profile_alt_max_km", 2000.0))
        saved_profile_out = _st("profile_out_dir", str(_default_runtime_output_dir()))
        if saved_profile_out.replace("\\", "/") == "results/profiling/st_lrps_runtime":
            saved_profile_out = str(_default_runtime_output_dir())
        self.profile_out_dir.setText(saved_profile_out)
        self.profile_compare_classic_sh.setChecked(_b("profile_compare_classic_sh", False))
        self.profile_classic_sh_degree.setValue(_i("profile_classic_sh_degree", 60))
        self.profile_json_only.setChecked(_b("profile_json_only", False))
        self.profile_verbose.setChecked(_b("profile_verbose", False))
        self.profile_extra_args.setText(_st("profile_extra_args", ""))
        s.endGroup()

    def _build_profile_args(self, show_errors: bool = True) -> Optional[List[str]]:
        def fail(title: str, message: str) -> Optional[List[str]]:
            if show_errors:
                QMessageBox.critical(self, title, message)
            else:
                self.command_warning.setText(message)
            return None

        if not show_errors:
            self.command_warning.setText("")

        if not PROFILE_CLI_PATH.exists():
            return fail("Missing script", "st_lrps/runtime/profiling.py not found in the repository.")

        model_dir = self.profile_model_dir.text().strip()
        if not model_dir:
            return fail("Missing model", "Runtime profiling requires --model-dir.")
        if not Path(model_dir).exists():
            return fail("Missing model", f"Model/run path not found:\n{model_dir}")

        batch_sizes = self.profile_batch_sizes.text().strip()
        chunk_sizes = self.profile_chunk_sizes.text().strip()
        if not batch_sizes:
            return fail("Missing batch sizes", "Batch sizes must be a comma-separated list.")
        if not chunk_sizes:
            return fail("Missing chunk sizes", "Chunk sizes must be a comma-separated list, e.g. none,512,1024.")

        input_source = self.profile_input_source.currentData() or "synthetic"
        out_dir = self.profile_out_dir.text().strip()
        if not out_dir:
            out_dir = str(_default_runtime_output_dir(model_dir))
            self.profile_out_dir.setText(out_dir)

        args = [
            "-u",
            "-m",
            PROFILE_CLI_MODULE,
            "--model-dir",
            model_dir,
            "--device",
            self.profile_device.currentText(),
            "--batch-sizes",
            batch_sizes,
            "--chunk-sizes",
            chunk_sizes,
            "--n-warmup",
            str(self.profile_n_warmup.value()),
            "--n-repeat",
            str(self.profile_n_repeat.value()),
            "--input-source",
            input_source,
            "--dataset-name",
            self.profile_dataset_name.text().strip() or "data",
            "--alt-min-km",
            str(self.profile_alt_min_km.value()),
            "--alt-max-km",
            str(self.profile_alt_max_km.value()),
            "--seed",
            str(self.profile_seed.value()),
            "--out-dir",
            out_dir,
            "--classic-sh-degree",
            str(self.profile_classic_sh_degree.value()),
        ]

        if input_source == "dataset":
            data_path = self.profile_data.text().strip()
            if not data_path:
                return fail("Missing dataset", "Dataset input mode requires --data.")
            if not Path(data_path).is_file():
                return fail("Missing dataset", f"Dataset not found:\n{data_path}")
            args += ["--data", data_path]

        if self.profile_compare_classic_sh.isChecked():
            args += ["--compare-classic-sh"]
        if self.profile_json_only.isChecked():
            args += ["--json-only"]
        if self.profile_verbose.isChecked():
            args += ["--verbose"]

        extra = self.profile_extra_args.text().strip()
        if extra:
            extra_args, err = _split_cli_args(extra)
            if err:
                return fail("Invalid extra CLI arguments", err)
            args += extra_args or []
        return args

    def _refresh_profile_preview(self) -> None:
        args = self._build_profile_args(show_errors=False)
        if args is None:
            self.command_preview.clear()
            return
        self.command_preview.setPlainText(_format_command(sys.executable, args))
        if not self.command_warning.text():
            self.command_warning.setText("Command is valid for the current UI fields.")

    def _copy_profile_command(self) -> None:
        if not self.command_preview.toPlainText().strip():
            self._refresh_profile_preview()
        QGuiApplication.clipboard().setText(self.command_preview.toPlainText())

    def _resolved_out_dir(self) -> Path:
        out_text = self._effective_out_dir or self.profile_out_dir.text().strip() or str(_default_runtime_output_dir())
        p = Path(out_text)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _start(self) -> None:
        args = self._build_profile_args()
        if args is None:
            return
        self._save_settings()
        self._effective_out_dir = self.profile_out_dir.text().strip() or str(_default_runtime_output_dir(self.profile_model_dir.text().strip()))
        out_path = self._resolved_out_dir()
        self.profile_summary.clear()
        self._gallery.clear_gallery()
        self._gallery._placeholder.setText("Runtime profile plots will appear here.")
        self.btn_run_profiling.setEnabled(False)
        self.btn_stop_profiling.setEnabled(True)
        self.runner.progress.setRange(0, 0)
        self.runner.set_output_dir(str(out_path))
        self.runner.set_stop_hint("")
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _on_profile_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.btn_run_profiling.setEnabled(True)
        self.btn_stop_profiling.setEnabled(False)
        out_path = self._resolved_out_dir()
        if exit_status != QProcess.ExitStatus.NormalExit:
            return
        summary_path = out_path / "runtime_profile_summary.md"
        json_path = out_path / "runtime_profile.json"
        csv_path = out_path / "runtime_profile.csv"
        if summary_path.is_file():
            try:
                self.profile_summary.setPlainText(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.profile_summary.setPlainText(f"Could not read {summary_path}: {exc}")
        else:
            present = [p.name for p in (json_path, csv_path) if p.is_file()]
            if present:
                self.profile_summary.setPlainText(
                    "Markdown summary was not generated.\nFound: " + ", ".join(present)
                )
            else:
                self.profile_summary.setPlainText(
                    "No profiling output files found yet. Check the process log for CLI errors."
                )
                self.runner.append(f"[UI] Profiling outputs not found in: {out_path}")

        images = [
            out_path / "runtime_profile_latency.png",
            out_path / "runtime_profile_throughput.png",
        ]
        loaded = self._gallery.load_images([p for p in images if p.is_file()])
        if loaded:
            self.runner.append(f"[UI] Loaded {loaded} profiling plot(s): {out_path}")
        else:
            self._gallery._placeholder.setText("No runtime profile plots found.")
            self.runner.append("[UI] No profiling PNG plots found. This is OK when matplotlib is unavailable or --json-only is set.")
        if out_path.is_dir():
            self.runner.set_output_dir(str(out_path))
            self.runner.btn_open_folder.setVisible(True)


class RuntimePerformancePage(QWidget):
    """Runtime inference performance workspace."""

    def __init__(self, profile_tab: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lo = QVBoxLayout()
        lo.setContentsMargins(22, 20, 22, 20)
        lo.setSpacing(14)
        lo.addWidget(_make_page_header(
            "Runtime Performance",
            "Profile loading latency, throughput, batching behavior, chunk effects, and hardware acceleration.",
            "Inference Workbench",
        ))
        lo.addWidget(profile_tab, 1)
        self.setLayout(lo)


class ModelReportPanel(QWidget):
    """Read-only artifact report for a trained ST-LRPS run directory."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        title = QLabel("Model Report")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #e6edf7;")

        self.run_edit = ValidatedPathEdit(
            placeholder="Select a trained run directory", check_file=False
        )
        btn_browse = QPushButton("Select...")
        btn_browse.clicked.connect(self._pick)
        btn_refresh = QPushButton("Refresh Report")
        btn_refresh.setProperty("kind", "primary")
        btn_refresh.clicked.connect(self._refresh)
        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(8)
        path_row.addWidget(self.run_edit, 1)
        path_row.addWidget(btn_browse)
        path_row.addWidget(btn_refresh)

        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setFont(_mono_font())
        self._report.setPlaceholderText("Select a run directory to inspect model artifacts.")
        self._report.setMinimumHeight(320)

        # Open-file buttons
        open_row = QHBoxLayout()
        open_row.setContentsMargins(0, 0, 0, 0)
        open_row.setSpacing(8)
        self._open_buttons = {}
        for label, fname in (
            ("Open run folder", ""),
            ("config.json", "config.json"),
            ("history.csv", "history.csv"),
            ("history.jsonl", "history.jsonl"),
            ("train.log", "train.log"),
        ):
            b = QPushButton(label)
            b.setProperty("kind", "ghost")
            b.clicked.connect(lambda _c=False, f=fname: self._open(f))
            self._open_buttons[label] = b
            open_row.addWidget(b)
        open_row.addStretch(1)

        lo = QVBoxLayout()
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setSpacing(10)
        lo.addWidget(title)
        lo.addLayout(path_row)
        lo.addLayout(open_row)
        lo.addWidget(self._report, 1)
        self.setLayout(lo)

    def _pick(self) -> None:
        start = self.run_edit.text().strip() or str(_REPO_ROOT)
        path = QFileDialog.getExistingDirectory(self, "Select run directory", start)
        if path:
            self.run_edit.setText(path)
            self._refresh()

    def _open(self, fname: str) -> None:
        run_dir = self.run_edit.text().strip()
        if not run_dir or not Path(run_dir).is_dir():
            QMessageBox.information(self, "No run", "Select a run directory first.")
            return
        target = Path(run_dir) if not fname else (Path(run_dir) / fname)
        if not target.exists():
            QMessageBox.information(self, "Not available", f"{target.name} not found in this run.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _refresh(self) -> None:
        run_dir = self.run_edit.text().strip()
        if not run_dir or not Path(run_dir).is_dir():
            self._report.setPlainText("Select a run directory to inspect model artifacts.")
            return
        root = Path(run_dir)
        cfg = _read_json_if_exists(root / "config.json")
        manifest = _read_json_if_exists(root / "run_manifest.json")
        scaler = _read_json_if_exists(root / "scaler.json")
        feat = _read_json_if_exists(root / "provenance" / "feature_summary.json")
        dsm = _read_json_if_exists(root / "provenance" / "dataset_meta.json")
        status = _inspect_run_artifacts(run_dir)

        def g(*keys, src=cfg, default="not available"):
            for k in keys:
                if isinstance(src, dict) and k in src and src[k] is not None:
                    return src[k]
            return default

        lines: List[str] = []
        lines.append(f"Run directory : {root}")
        lines.append(f"Run status    : {manifest.get('status', 'not available') if manifest else 'not available'}")
        lines.append("")
        lines.append("── Architecture ──")
        lines.append(f"model_preset     : {g('model_preset')}")
        lines.append(f"hidden / depth   : {g('hidden')} / {g('depth')}")
        lines.append(f"n_bands          : {g('n_bands')}")
        lines.append(f"activation       : {g('activation')}")
        lines.append(f"degree_min/max   : {g('degree_min')} / {g('degree_max', 'requested_degree')}")
        lines.append(f"embedding_type   : {g('embedding_type', src=feat) if feat else g('embedding_type')}")
        lines.append(f"input_feature_dim: {g('input_feature_dim', src=feat) if feat else g('input_feature_dim')}")
        lines.append(f"arch signature   : {status.get('architecture_signature') or 'not available'}")
        lines.append("")
        lines.append("── Checkpoint ──")
        lines.append(f"checkpoint     : {status.get('checkpoint_path') or 'not available'}")
        lines.append(f"schema version : {status.get('checkpoint_schema_version') or 'not available'}")
        lines.append(f"best epoch     : {status.get('best_epoch') if status.get('best_epoch') is not None else 'not available'}")
        lines.append(f"best score     : {status.get('best_score') if status.get('best_score') is not None else 'not available'}")
        lines.append(f"scaler status  : {status.get('scaler_status')}")
        if scaler:
            lines.append(f"scaler keys    : {', '.join(list(scaler.keys())[:8])}")
        lines.append("")
        lines.append("── Target contract ──")
        tc = cfg.get("target_contract") if isinstance(cfg, dict) else None
        if isinstance(tc, dict):
            for k, v in tc.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append("  not available")
        if dsm:
            lines.append("")
            lines.append("── Dataset meta (provenance) ──")
            for k in ("unit_system", "central_body", "target_mode", "degree_min", "degree_max"):
                if k in dsm:
                    lines.append(f"  {k}: {dsm[k]}")
        if status.get("warnings"):
            lines.append("")
            lines.append("── Warnings ──")
            for w in status["warnings"]:
                lines.append(f"  ! {w}")

        # Disable open buttons for missing files
        self._open_buttons["config.json"].setEnabled((root / "config.json").exists())
        self._open_buttons["history.csv"].setEnabled((root / "history.csv").exists())
        self._open_buttons["history.jsonl"].setEnabled((root / "history.jsonl").exists())
        self._open_buttons["train.log"].setEnabled((root / "train.log").exists())

        self._report.setPlainText("\n".join(lines))

