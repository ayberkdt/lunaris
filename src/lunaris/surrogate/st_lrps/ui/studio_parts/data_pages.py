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
_STLRPS_DATA_MODULE_DIR = _STLRPS_ROOT / "data"
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
from .common_widgets import _tune_form, _tune_inputs, _row_lineedit_with_button, _scroll_wrap, _settings, _read_json_if_exists, _split_cli_args, _format_command, _send_os_notification, _apply_status_tips, _cfg_value, _norm_path, _timestamp_slug, _safe_slug, _default_training_output_dir, _default_runtime_output_dir, _default_dataset_report_dir, _output_standard_text, _mono_font, _make_page_header, _style_command_preview, _style_surface, _inspect_run_artifacts, _NoWheelOnSpinFilter


def _introspect_h5(path: str) -> Optional[Dict[str, Any]]:
    """
    Read metadata from an HDF5 file without loading the full dataset.
    Returns a dict with: rows, cols, col_names (if stored), attrs, is_si.
    Returns None if h5py is unavailable or the file can't be read.
    """
    if not _HAS_H5PY:
        return None
    try:
        with h5py.File(path, "r") as f:
            info: Dict[str, Any] = {"attrs": {}}

            # Gather file-level attributes
            for key in f.attrs:
                val = f.attrs[key]
                if hasattr(val, "item"):
                    val = val.item()
                elif isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                info["attrs"][key] = val

            # Find the primary dataset (try 'data', else first 2D dataset)
            ds = None
            ds_name = ""
            for name in ("data", "dataset", "train"):
                if name in f:
                    ds = f[name]
                    ds_name = name
                    break
            if ds is None:
                for name in f:
                    if isinstance(f[name], h5py.Dataset) and len(f[name].shape) >= 2:
                        ds = f[name]
                        ds_name = name
                        break

            if ds is not None:
                info["dataset_name"] = ds_name
                info["rows"] = ds.shape[0]
                info["cols"] = ds.shape[1] if len(ds.shape) > 1 else 1
                info["dtype"] = str(ds.dtype)
                info["shape"] = list(ds.shape)

                # Dataset-level attributes
                for key in ds.attrs:
                    val = ds.attrs[key]
                    if hasattr(val, "item"):
                        val = val.item()
                    elif isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    info["attrs"][key] = val

            # Heuristic: detect SI vs canonical units
            all_attrs_str = json.dumps(info["attrs"]).lower()
            if (
                "si" in all_attrs_str
                or "meter" in all_attrs_str
                or "m/s" in all_attrs_str
            ):
                info["is_si"] = True
            elif "canonical" in all_attrs_str or "dimensionless" in all_attrs_str:
                info["is_si"] = False
            else:
                info["is_si"] = None  # Unknown

            # Check for column names
            if "columns" in info["attrs"]:
                info["col_names"] = info["attrs"]["columns"]
            elif "column_names" in info["attrs"]:
                info["col_names"] = info["attrs"]["column_names"]

            return info
    except Exception:
        return None


def _attr_lookup(attrs: Dict[str, Any], *keys: str) -> Any:
    """Return the first present metadata value across common naming variants."""
    for key in keys:
        if key in attrs:
            return attrs[key]
        lower = key.lower()
        for candidate, value in attrs.items():
            if str(candidate).lower() == lower:
                return value
    return None


def _data_action_card(
    title: str,
    subtitle: str,
    primary_button: QPushButton,
    *,
    secondary_buttons: Optional[List[QPushButton]] = None,
    detail: Optional[QWidget] = None,
    object_name: str = "dataActionCard",
) -> QFrame:
    """Create a compact, action-first card for the Data workspace."""
    card = QFrame()
    card.setObjectName(object_name)
    card.setStyleSheet(
        f"QFrame#{object_name} {{"
        "  background: rgba(8, 13, 26, 0.84);"
        "  border: 1px solid rgba(53, 208, 255, 0.18);"
        "  border-radius: 14px;"
        "}"
    )
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(12)

    top = QHBoxLayout()
    top.setContentsMargins(0, 0, 0, 0)
    top.setSpacing(14)

    text_col = QVBoxLayout()
    text_col.setContentsMargins(0, 0, 0, 0)
    text_col.setSpacing(3)
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(
        "color: #f3f7ff; font-size: 16px; font-weight: 800; "
        "background: transparent; border: none;"
    )
    subtitle_lbl = QLabel(subtitle)
    subtitle_lbl.setWordWrap(True)
    subtitle_lbl.setStyleSheet(
        "color: #8fa0bf; font-size: 12px; background: transparent; border: none;"
    )
    text_col.addWidget(title_lbl)
    text_col.addWidget(subtitle_lbl)

    primary_button.setProperty("kind", "primary")
    primary_button.setMinimumHeight(42)
    primary_button.setMinimumWidth(150)

    top.addLayout(text_col, 1)
    top.addWidget(primary_button, 0, Qt.AlignmentFlag.AlignTop)
    layout.addLayout(top)

    if secondary_buttons:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        for button in secondary_buttons:
            button.setProperty("kind", button.property("kind") or "ghost")
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

    if detail is not None:
        layout.addWidget(detail)
    return card


def _compact_path_label(empty_text: str) -> QLabel:
    label = QLabel(empty_text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setStyleSheet(
        "QLabel { color: #9fb0cc; font-size: 11px; padding: 8px 10px;"
        " background: rgba(4, 8, 16, 0.55);"
        " border: 1px solid rgba(185, 194, 221, 0.12);"
        " border-radius: 9px; }"
    )
    return label


def _set_path_label(label: QLabel, path: str, *, empty_text: str) -> None:
    path = path.strip()
    if not path:
        label.setText(empty_text)
        return
    p = Path(path)
    shown = p.name if p.name else path
    label.setText(f"{shown}\n{path}")


class CloudGenTab(QWidget):
    """Tab for generating spatial point-cloud datasets via spatial_cloud_generator.py.

    Supports two modes:
    - Single Cloud: single HDF5/PT file output (existing behaviour)
    - Dataset Suite: full train/val/test/ood suite with manifest.json

    Emits ``cloud_params_changed(alt_min_km, alt_max_km, deg_min, deg_max)``
    whenever the altitude or degree range widgets change, so the Train tab can
    stay in sync without manual copy-paste.
    """

    cloud_params_changed = pyqtSignal(float, float, int, int)

    # Mode indices for the stacked widget
    _MODE_SINGLE = 0
    _MODE_SUITE  = 1

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._train_tab_ref: Optional[Any] = None  # set by MainWindow
        self._last_suite_dir: Optional[str] = None  # set after suite completes
        self._analysis_queue: deque[Tuple[str, str, str]] = deque()
        self._active_analysis: Optional[Tuple[str, str, str]] = None

        import os as _os_cloudgen
        _cpu_count = max(1, _os_cloudgen.cpu_count() or 1)

        # ── Mode selector ────────────────────────────────────────────────────
        mode_bar = QHBoxLayout()
        mode_bar.setContentsMargins(0, 0, 0, 4)
        mode_bar.setSpacing(8)
        mode_lbl = QLabel("Workflow")
        mode_lbl.setStyleSheet("font-weight: 700; color: #c4ccff;")
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Single Cloud", self._MODE_SINGLE)
        self._mode_combo.addItem("Dataset Suite", self._MODE_SUITE)
        self._mode_combo.setMinimumWidth(220)
        self._mode_combo.setToolTip(
            "Single Cloud: generates a single .h5/.pt file.\n"
            "Dataset Suite: generates a train/val/test/ood + manifest.json set."
        )
        mode_bar.addWidget(mode_lbl)
        mode_bar.addWidget(self._mode_combo)
        mode_bar.addStretch(1)

        # ── Sync banner (shared) ─────────────────────────────────────────────
        self._sync_banner = QLabel("")
        self._sync_banner.setWordWrap(True)
        self._sync_banner.setStyleSheet(
            "QLabel { color: #34d399; background: rgba(52,211,153,0.08); "
            "border: 1px solid rgba(52,211,153,0.3); border-radius: 8px; "
            "padding: 6px 12px; font-size: 11px; }"
        )
        self._sync_banner.setVisible(False)

        # ── Single cloud page ────────────────────────────────────────────────
        single_page = self._build_single_cloud_page(_cpu_count)

        # ── Suite page ────────────────────────────────────────────────────────
        suite_page = self._build_suite_page(_cpu_count)

        # ── Stacked widget ───────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.addWidget(single_page)  # page 0
        self._stack.addWidget(suite_page)   # page 1
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        # ── Command preview (shared) ─────────────────────────────────────────
        self.command_preview = QPlainTextEdit()
        _style_command_preview(self.command_preview, min_h=76, max_h=96)
        self.command_preview.setPlaceholderText(
            "Generated CLI command"
        )
        self.command_preview.setVisible(False)
        btn_preview = QPushButton("Show Command")
        btn_preview.clicked.connect(self._show_command_preview)
        btn_copy_cmd = QPushButton("Copy")
        btn_copy_cmd.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.command_preview.toPlainText())
        )
        self._btn_generate_now = QPushButton("Start Cloud Generation")
        self._btn_generate_now.clicked.connect(self._start)
        preview_btns = QHBoxLayout()
        preview_btns.setContentsMargins(0, 0, 0, 0)
        preview_btns.setSpacing(8)
        preview_btns.addLayout(mode_bar)
        preview_btns.addStretch(1)
        preview_btns.addWidget(btn_preview)
        preview_btns.addWidget(btn_copy_cmd)

        preview_w = QWidget()
        preview_vbox = QVBoxLayout()
        preview_vbox.setContentsMargins(0, 0, 0, 0)
        preview_vbox.setSpacing(6)
        preview_vbox.addLayout(preview_btns)
        preview_vbox.addWidget(self.command_preview)
        preview_w.setLayout(preview_vbox)
        generator_card = _data_action_card(
            "Generate Dataset",
            "Pick a workflow, tune the cards below, then launch the dataset job.",
            self._btn_generate_now,
            detail=preview_w,
        )
        analysis_panel = self._build_analysis_panel()

        # ── ProcessPane ──────────────────────────────────────────────────────
        self.runner = ProcessPane()
        self.runner.btn_start.setText("Start Cloud Generation")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_finished_hook(self._on_finished)
        self.runner.set_progress_parser(self._parse_progress)

        top = QWidget()
        top_l = QVBoxLayout()
        top_l.setContentsMargins(8, 8, 8, 4)
        top_l.setSpacing(10)
        top_l.addWidget(generator_card)
        top_l.addWidget(self._sync_banner)
        top_l.addWidget(self._stack, 1)
        top_l.addWidget(analysis_panel)
        top.setLayout(top_l)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(_scroll_wrap(top))
        splitter.addWidget(self.runner)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 300])

        main_lo = QVBoxLayout()
        main_lo.setContentsMargins(10, 10, 10, 10)
        main_lo.addWidget(splitter, 1)
        self.setLayout(main_lo)

        # Wire param-change signals
        self.alt_min_km.valueChanged.connect(self._emit_params_changed)
        self.alt_max_km.valueChanged.connect(self._emit_params_changed)
        self.degree_min.valueChanged.connect(self._emit_params_changed)
        self.degree_max.valueChanged.connect(self._emit_params_changed)

        self._restore_settings()

    # ------------------------------------------------------------------
    # Single cloud page builder
    # ------------------------------------------------------------------
    def _build_single_cloud_page(self, cpu_count: int) -> QWidget:
        cloud_cfg = DEFAULT_SPATIAL_CLOUD_CONFIG

        # ── Group 1: Gravity Field ─────────────────────────────────────────
        grp_grav = QGroupBox("Gravity Field")
        form_grav = QFormLayout()
        _tune_form(form_grav)

        self.degree_max = QSpinBox()
        self.degree_max.setRange(1, 1800)
        self.degree_max.setValue(int(_cfg_value(cloud_cfg, "degree_max", 100)))
        self.degree_max.setToolTip("Hedef SH derecesi (ust sinir).")

        self.degree_min = QSpinBox()
        self.degree_min.setSpecialValueText("Tam alan (-1)")
        self.degree_min.setRange(-1, 1800)
        self.degree_min.setValue(int(_cfg_value(cloud_cfg, "degree_min", 20)))
        self.degree_min.setToolTip(
            "Taban model derecesi. -1 = nokta kutlesi dahil tam alan."
        )

        self.gfc_path = ValidatedPathEdit(
            placeholder="Empty -> default lunar gravity model", check_file=True
        )
        btn_gfc = QPushButton("Choose...")
        btn_gfc.clicked.connect(self._pick_gfc_path)
        gfc_row = _row_lineedit_with_button(self.gfc_path, btn_gfc)

        form_grav.addRow("Max SH Degree", self.degree_max)
        form_grav.addRow("Min SH Degree (base)", self.degree_min)
        form_grav.addRow("Gravity Model", gfc_row)
        grp_grav.setLayout(form_grav)

        # ── Group 2: Spatial Sampling ──────────────────────────────────────
        grp_spatial = QGroupBox("Spatial Sampling")
        form_spatial = QFormLayout()
        _tune_form(form_spatial)

        self.n_samples = QSpinBox()
        self.n_samples.setRange(1_000, 100_000_000)
        self.n_samples.setValue(int(_cfg_value(cloud_cfg, "n_samples", 2_000_000)))
        self.n_samples.setSingleStep(100_000)
        self.n_samples.setToolTip("Uretilecek toplam nokta sayisi.")

        self.alt_min_km = QDoubleSpinBox()
        self.alt_min_km.setDecimals(1)
        self.alt_min_km.setRange(0.0, 100_000.0)
        self.alt_min_km.setValue(float(_cfg_value(cloud_cfg, "alt_min_km", 200.0)))
        self.alt_min_km.setSingleStep(10.0)
        self.alt_min_km.setSuffix(" km")

        self.alt_max_km = QDoubleSpinBox()
        self.alt_max_km.setDecimals(1)
        self.alt_max_km.setRange(0.1, 100_000.0)
        self.alt_max_km.setValue(float(_cfg_value(cloud_cfg, "alt_max_km", 600.0)))
        self.alt_max_km.setSingleStep(10.0)
        self.alt_max_km.setSuffix(" km")

        self.sampling_strategy = QComboBox()
        self.sampling_strategy.addItem("Mixed - recommended", "mixed")
        self.sampling_strategy.addItem("Uniform volume", "uniform")
        self.sampling_strategy.addItem("Inverse-r2 surface focus", "inverse_r2")
        _strategy_idx = self.sampling_strategy.findData(str(_cfg_value(cloud_cfg, "sampling_strategy", "mixed")))
        if _strategy_idx >= 0:
            self.sampling_strategy.setCurrentIndex(_strategy_idx)

        self.surface_bias_ratio = QDoubleSpinBox()
        self.surface_bias_ratio.setDecimals(2)
        self.surface_bias_ratio.setRange(0.0, 1.0)
        self.surface_bias_ratio.setValue(float(_cfg_value(cloud_cfg, "surface_bias_ratio", 0.70)))
        self.surface_bias_ratio.setSingleStep(0.05)

        form_spatial.addRow("Sample Count", self.n_samples)
        form_spatial.addRow("Min Altitude", self.alt_min_km)
        form_spatial.addRow("Max Altitude", self.alt_max_km)
        form_spatial.addRow("Sampling Strategy", self.sampling_strategy)
        form_spatial.addRow("Surface Bias", self.surface_bias_ratio)
        grp_spatial.setLayout(form_spatial)

        # ── Group 3: Output ────────────────────────────────────────────────
        grp_out = QGroupBox("Output Settings")
        form_out = QFormLayout()
        _tune_form(form_out)

        self.out_format = QComboBox()
        self.out_format.addItem("HDF5 (.h5) - recommended", "h5")
        self.out_format.addItem("PyTorch (.pt)", "pt")

        self.out_path = QLineEdit("")
        self.out_path.setPlaceholderText("Auto when empty")
        btn_out_save = QPushButton("Choose...")
        btn_out_save.clicked.connect(self._pick_out_path)
        out_row = _row_lineedit_with_button(self.out_path, btn_out_save)

        self.dtype = QComboBox()
        self.dtype.addItem("float32 - recommended", "float32")
        self.dtype.addItem("float64 - high precision", "float64")

        self.canonical = QCheckBox("Canonical units")
        self.canonical.setChecked(bool(_cfg_value(cloud_cfg, "canonical", False)))

        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setValue(int(_cfg_value(cloud_cfg, "seed", 12345)))

        form_out.addRow("Output Format", self.out_format)
        form_out.addRow("Output File", out_row)
        form_out.addRow("Dtype", self.dtype)
        form_out.addRow(self.canonical)
        form_out.addRow("Seed", self.seed)
        grp_out.setLayout(form_out)

        # ── Group 4: Performance ───────────────────────────────────────────
        grp_perf = QGroupBox("Performance")
        form_perf = QFormLayout()
        _tune_form(form_perf)

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1_000, 10_000_000)
        self.chunk_size.setValue(int(_cfg_value(cloud_cfg, "chunk_size", 50_000)))
        self.chunk_size.setSingleStep(10_000)

        self.workers = QSpinBox()
        self.workers.setRange(1, 256)
        self.workers.setValue(min(cpu_count, int(_cfg_value(cloud_cfg, "workers", 8))))

        self.no_multiprocessing = QCheckBox("Single-process mode")
        self.no_multiprocessing.setChecked(bool(_cfg_value(cloud_cfg, "no_multiprocessing", False)))

        form_perf.addRow("Chunk Size", self.chunk_size)
        form_perf.addRow(f"Worker Count (system: {cpu_count})", self.workers)
        form_perf.addRow(self.no_multiprocessing)
        grp_perf.setLayout(form_perf)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(grp_grav,    0, 0)
        grid.addWidget(grp_spatial, 0, 1)
        grid.addWidget(grp_out,     1, 0)
        grid.addWidget(grp_perf,    1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for g in (grp_grav, grp_spatial, grp_out, grp_perf):
            _tune_inputs(g)

        w = QWidget()
        lo = QVBoxLayout()
        lo.setContentsMargins(4, 4, 4, 4)
        lo.setSpacing(8)
        lo.addLayout(grid)
        w.setLayout(lo)
        return w

    # ------------------------------------------------------------------
    # Suite page builder
    # ------------------------------------------------------------------
    def _build_suite_page(self, cpu_count: int) -> QWidget:
        suite_cfg = DEFAULT_CLOUD_SUITE_CONFIG

        # A) Physics group
        grp_phys = QGroupBox("Physics / Degree / Altitude")
        form_phys = QFormLayout()
        _tune_form(form_phys)

        self.s_degree_min = QSpinBox()
        self.s_degree_min.setRange(0, 1800)
        self.s_degree_min.setValue(int(_cfg_value(suite_cfg, "degree_min", 20)))
        self.s_degree_min.setToolTip("Taban model derecesi (bas derece).")

        self.s_degree_max = QSpinBox()
        self.s_degree_max.setRange(1, 1800)
        self.s_degree_max.setValue(int(_cfg_value(suite_cfg, "degree_max", 100)))
        self.s_degree_max.setToolTip("Hedef SH derecesi (ust sinir).")

        self.s_train_alt_min_km = QDoubleSpinBox()
        self.s_train_alt_min_km.setDecimals(1)
        self.s_train_alt_min_km.setRange(0.0, 50_000.0)
        self.s_train_alt_min_km.setValue(float(_cfg_value(suite_cfg, "train_alt_min_km", 200.0)))
        self.s_train_alt_min_km.setSuffix(" km")
        self.s_train_alt_min_km.setToolTip("Lower bound of the training altitude range.")

        self.s_train_alt_max_km = QDoubleSpinBox()
        self.s_train_alt_max_km.setDecimals(1)
        self.s_train_alt_max_km.setRange(1.0, 50_000.0)
        self.s_train_alt_max_km.setValue(float(_cfg_value(suite_cfg, "train_alt_max_km", 600.0)))
        self.s_train_alt_max_km.setSuffix(" km")
        self.s_train_alt_max_km.setToolTip("Upper bound of the training altitude range.")

        self.s_ood_margin_km = QDoubleSpinBox()
        self.s_ood_margin_km.setDecimals(1)
        self.s_ood_margin_km.setRange(1.0, 5_000.0)
        self.s_ood_margin_km.setValue(float(_cfg_value(suite_cfg, "ood_margin_km", 40.0)))
        self.s_ood_margin_km.setSuffix(" km")
        self.s_ood_margin_km.setToolTip(
            "OOD bolge genisligi. OOD low = [alt_min - margin, alt_min]; "
            "OOD high = [alt_max, alt_max + margin]."
        )

        self.s_gfc_path = ValidatedPathEdit(
            placeholder="Empty -> default lunar gravity model", check_file=True
        )
        btn_s_gfc = QPushButton("Choose...")
        btn_s_gfc.clicked.connect(self._pick_suite_gfc_path)
        s_gfc_row = _row_lineedit_with_button(self.s_gfc_path, btn_s_gfc)

        form_phys.addRow("Degree Min (base)", self.s_degree_min)
        form_phys.addRow("Degree Max (target)", self.s_degree_max)
        form_phys.addRow("Training Alt Min", self.s_train_alt_min_km)
        form_phys.addRow("Training Alt Max", self.s_train_alt_max_km)
        form_phys.addRow("OOD Margin", self.s_ood_margin_km)
        form_phys.addRow("Gravity Model", s_gfc_row)
        grp_phys.setLayout(form_phys)

        # B) Train hybrid allocation
        grp_train = QGroupBox("Training Hybrid Distribution")
        form_train = QFormLayout()
        _tune_form(form_train)

        self.s_train_su_n = QSpinBox()
        self.s_train_su_n.setRange(0, 100_000_000)
        self.s_train_su_n.setValue(int(_cfg_value(suite_cfg, "train_stratified_uniform_n", 2_000_000)))
        self.s_train_su_n.setSingleStep(100_000)
        self.s_train_su_n.setToolTip("Number of stratified-uniform points.")

        self.s_train_ir2_n = QSpinBox()
        self.s_train_ir2_n.setRange(0, 100_000_000)
        self.s_train_ir2_n.setValue(int(_cfg_value(suite_cfg, "train_inverse_r2_n", 1_000_000)))
        self.s_train_ir2_n.setSingleStep(100_000)
        self.s_train_ir2_n.setToolTip("Number of inverse-r² (surface-focused) points.")

        self.s_train_rm_n = QSpinBox()
        self.s_train_rm_n.setRange(0, 100_000_000)
        self.s_train_rm_n.setValue(int(_cfg_value(suite_cfg, "train_residual_mag_n", 1_000_000)))
        self.s_train_rm_n.setSingleStep(100_000)
        self.s_train_rm_n.setToolTip("Number of points sampled with weighting by residual-acceleration magnitude.")

        self.s_train_bb_n = QSpinBox()
        self.s_train_bb_n.setRange(0, 100_000_000)
        self.s_train_bb_n.setValue(int(_cfg_value(suite_cfg, "train_boundary_n", 1_000_000)))
        self.s_train_bb_n.setSingleStep(100_000)
        self.s_train_bb_n.setToolTip("Number of boundary-buffer points (lower/upper altitude edges).")

        self._suite_total_lbl = QLabel("")
        self._suite_total_lbl.setStyleSheet("color: #7c8dc7; font-weight: bold;")

        for sb in (self.s_train_su_n, self.s_train_ir2_n, self.s_train_rm_n, self.s_train_bb_n):
            sb.valueChanged.connect(self._update_suite_total_label)

        self.s_residual_mag_candidate_multiplier = QSpinBox()
        self.s_residual_mag_candidate_multiplier.setRange(1, 100)
        self.s_residual_mag_candidate_multiplier.setValue(int(_cfg_value(suite_cfg, "residual_mag_candidate_multiplier", 5)))
        self.s_residual_mag_candidate_multiplier.setToolTip(
            "Candidate multiplier for residual-mag weighted sampling. "
            "N_candidates = n_samples * multiplier. Default: 5."
        )

        self.s_residual_mag_weight_power = QDoubleSpinBox()
        self.s_residual_mag_weight_power.setDecimals(2)
        self.s_residual_mag_weight_power.setRange(0.01, 4.0)
        self.s_residual_mag_weight_power.setValue(float(_cfg_value(suite_cfg, "residual_mag_weight_power", 0.5)))
        self.s_residual_mag_weight_power.setSingleStep(0.1)
        self.s_residual_mag_weight_power.setToolTip(
            "Weighting exponent for residual-mag sampling. "
            "p proportional to floor + (score/median)^power. Default: 0.5."
        )

        self.s_boundary_mode = QComboBox()
        self.s_boundary_mode.addItem("Strict in train range", "strict")
        self.s_boundary_mode.addItem("Soft around edge", "soft")
        _boundary_idx = self.s_boundary_mode.findData(str(_cfg_value(suite_cfg, "boundary_mode", "strict")))
        if _boundary_idx >= 0:
            self.s_boundary_mode.setCurrentIndex(_boundary_idx)
        self.s_boundary_mode.setToolTip(
            "strict: boundary points inside [alt_min, alt_min+bw] and [alt_max-bw, alt_max]. "
            "soft: boundary band straddles the edge."
        )

        self.s_boundary_width_km = QDoubleSpinBox()
        self.s_boundary_width_km.setDecimals(1)
        self.s_boundary_width_km.setRange(1.0, 500.0)
        self.s_boundary_width_km.setValue(float(_cfg_value(suite_cfg, "boundary_width_km", 20.0)))
        self.s_boundary_width_km.setSuffix(" km")
        self.s_boundary_width_km.setToolTip("Width of the boundary buffer band at each edge. Default: 20 km.")

        form_train.addRow("Stratified Uniform", self.s_train_su_n)
        form_train.addRow("Inverse-r2", self.s_train_ir2_n)
        form_train.addRow("Residual Mag Weighted", self.s_train_rm_n)
        form_train.addRow("  ResidMag Candidate Mult.", self.s_residual_mag_candidate_multiplier)
        form_train.addRow("  ResidMag Weight Power", self.s_residual_mag_weight_power)
        form_train.addRow("Boundary Buffer", self.s_train_bb_n)
        form_train.addRow("  Boundary Mode", self.s_boundary_mode)
        form_train.addRow("  Boundary Width", self.s_boundary_width_km)
        form_train.addRow("", self._suite_total_lbl)
        grp_train.setLayout(form_train)

        # C) Val/Test/OOD sizes
        grp_vto = QGroupBox("Validation / Test / OOD")
        form_vto = QFormLayout()
        _tune_form(form_vto)

        self.s_val_n = QSpinBox()
        self.s_val_n.setRange(0, 100_000_000)
        self.s_val_n.setValue(int(_cfg_value(suite_cfg, "val_n", 1_000_000)))
        self.s_val_n.setSingleStep(100_000)

        self.s_test_n = QSpinBox()
        self.s_test_n.setRange(0, 100_000_000)
        self.s_test_n.setValue(int(_cfg_value(suite_cfg, "test_n", 1_000_000)))
        self.s_test_n.setSingleStep(100_000)

        self.s_ood_low_n = QSpinBox()
        self.s_ood_low_n.setRange(0, 100_000_000)
        self.s_ood_low_n.setValue(int(_cfg_value(suite_cfg, "ood_low_n", 250_000)))
        self.s_ood_low_n.setSingleStep(50_000)

        self.s_ood_high_n = QSpinBox()
        self.s_ood_high_n.setRange(0, 100_000_000)
        self.s_ood_high_n.setValue(int(_cfg_value(suite_cfg, "ood_high_n", 250_000)))
        self.s_ood_high_n.setSingleStep(50_000)

        self.s_combine_ood = QCheckBox("Combine OOD low + high")
        self.s_combine_ood.setChecked(bool(_cfg_value(suite_cfg, "combine_ood", True)))

        form_vto.addRow("Validation", self.s_val_n)
        form_vto.addRow("Test", self.s_test_n)
        form_vto.addRow("OOD Low", self.s_ood_low_n)
        form_vto.addRow("OOD High", self.s_ood_high_n)
        form_vto.addRow(self.s_combine_ood)
        grp_vto.setLayout(form_vto)

        # D) Seeds
        grp_seeds = QGroupBox("Seeds")
        form_seeds = QFormLayout()
        _tune_form(form_seeds)

        def _make_seed_spin(default: int) -> QSpinBox:
            sb = QSpinBox()
            sb.setRange(0, 99_999_999)
            sb.setValue(default)
            return sb

        self.s_seed_base          = _make_seed_spin(int(_cfg_value(suite_cfg, "base_seed", 42)))
        self.s_seed_train_uniform = _make_seed_spin(int(_cfg_value(suite_cfg, "train_uniform_seed", 42)))
        self.s_seed_train_ir2     = _make_seed_spin(int(_cfg_value(suite_cfg, "train_inverse_r2_seed", 142)))
        self.s_seed_train_rm      = _make_seed_spin(int(_cfg_value(suite_cfg, "train_residual_mag_seed", 242)))
        self.s_seed_train_bb      = _make_seed_spin(int(_cfg_value(suite_cfg, "train_boundary_seed", 342)))
        self.s_seed_val           = _make_seed_spin(int(_cfg_value(suite_cfg, "val_seed", 1042)))
        self.s_seed_test          = _make_seed_spin(int(_cfg_value(suite_cfg, "test_seed", 2042)))
        self.s_seed_ood_low       = _make_seed_spin(int(_cfg_value(suite_cfg, "ood_low_seed", 3042)))
        self.s_seed_ood_high      = _make_seed_spin(int(_cfg_value(suite_cfg, "ood_high_seed", 4042)))

        btn_auto_seeds = QPushButton("Assign Independent Seeds")
        btn_auto_seeds.setToolTip("Tum tohumlara base_seed bazli bagimsiz degerler atar.")
        btn_auto_seeds.clicked.connect(self._auto_assign_seeds)

        form_seeds.addRow("Base Seed", self.s_seed_base)
        form_seeds.addRow("Train Uniform Seed", self.s_seed_train_uniform)
        form_seeds.addRow("Train Inv-r2 Seed", self.s_seed_train_ir2)
        form_seeds.addRow("Train ResidMag Seed", self.s_seed_train_rm)
        form_seeds.addRow("Train Boundary Seed", self.s_seed_train_bb)
        form_seeds.addRow("Val Seed", self.s_seed_val)
        form_seeds.addRow("Test Seed", self.s_seed_test)
        form_seeds.addRow("OOD Low Seed", self.s_seed_ood_low)
        form_seeds.addRow("OOD High Seed", self.s_seed_ood_high)
        form_seeds.addRow("", btn_auto_seeds)
        grp_seeds.setLayout(form_seeds)

        # E) Presets
        grp_presets = QGroupBox("Suite Presets")
        form_presets = QFormLayout()
        _tune_form(form_presets)

        self.s_preset_combo = QComboBox()
        self.s_preset_combo.addItem("Choose preset...", "")
        self.s_preset_combo.addItem("Debug Suite (100k)", "debug_suite")
        self.s_preset_combo.addItem("Baseline Uniform (2M train)", "baseline_uniform_suite")
        self.s_preset_combo.addItem("Recommended Hybrid 5M", "recommended_hybrid_5M")
        self.s_preset_combo.addItem("High Accuracy 10M", "high_accuracy_10M")

        btn_apply_preset = QPushButton("Apply Preset")
        btn_apply_preset.clicked.connect(self._apply_suite_preset)

        form_presets.addRow("Preset", self.s_preset_combo)
        form_presets.addRow("", btn_apply_preset)
        grp_presets.setLayout(form_presets)

        # F) Suite output
        grp_suite_out = QGroupBox("Suite Output")
        form_suite_out = QFormLayout()
        _tune_form(form_suite_out)

        self.s_suite_name = QLineEdit("")
        self.s_suite_name.setPlaceholderText("Auto-named when empty")

        self.s_suite_out_dir = QLineEdit("")
        self.s_suite_out_dir.setPlaceholderText(
            f"Empty -> {DATASET_SUITE_OUTPUT_ROOT}"
        )
        btn_suite_out = QPushButton("Choose...")
        btn_suite_out.clicked.connect(self._pick_suite_out_dir)
        suite_out_row = _row_lineedit_with_button(self.s_suite_out_dir, btn_suite_out)

        self.s_auto_apply = QCheckBox("Auto-fill Training when done")
        self.s_auto_apply.setChecked(True)
        self.s_auto_apply.setToolTip(
            "When enabled, manifest train/val/test/ood paths are written\n"
            "automatically into the Training tab."
        )

        self.s_chunk_size = QSpinBox()
        self.s_chunk_size.setRange(1_000, 10_000_000)
        self.s_chunk_size.setValue(int(_cfg_value(suite_cfg, "chunk_size", 50_000)))
        self.s_chunk_size.setSingleStep(10_000)

        self.s_dtype = QComboBox()
        self.s_dtype.addItem("float32 - recommended", "float32")
        self.s_dtype.addItem("float64 - high precision", "float64")

        form_suite_out.addRow("Suite Name", self.s_suite_name)
        form_suite_out.addRow("Output Folder", suite_out_row)
        form_suite_out.addRow(self.s_auto_apply)
        form_suite_out.addRow("Chunk Size", self.s_chunk_size)
        form_suite_out.addRow("Dtype", self.s_dtype)
        grp_suite_out.setLayout(form_suite_out)

        # G) Suite actions
        btn_open_suite_folder = QPushButton("Open Suite Folder")
        btn_open_suite_folder.clicked.connect(self._open_suite_folder)
        btn_apply_to_train = QPushButton("Apply to Training Tab")
        btn_apply_to_train.clicked.connect(self._apply_suite_to_train)
        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)
        actions_row.addWidget(btn_open_suite_folder)
        actions_row.addWidget(btn_apply_to_train)
        actions_row.addStretch(1)

        # Layout
        left = QVBoxLayout()
        left.setSpacing(8)
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(grp_phys)
        left.addWidget(grp_presets)
        left.addWidget(grp_suite_out)
        left.addLayout(actions_row)
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(8)
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(grp_train)
        right.addWidget(grp_vto)
        right.addWidget(grp_seeds)
        right.addStretch(1)

        cols = QHBoxLayout()
        cols.setSpacing(12)
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)

        for grp in (grp_phys, grp_train, grp_vto, grp_seeds, grp_presets, grp_suite_out):
            _tune_inputs(grp)

        self._update_suite_total_label()

        w = QWidget()
        lo = QVBoxLayout()
        lo.setContentsMargins(4, 4, 4, 4)
        lo.setSpacing(6)
        lo.addLayout(cols)
        w.setLayout(lo)
        return w

    # ------------------------------------------------------------------
    # Cloud analysis panel
    # ------------------------------------------------------------------
    def _build_analysis_panel(self) -> QWidget:
        self.analysis_input = ValidatedPathEdit(
            placeholder="Analyze a generated .h5/.hdf5/.pt dataset", check_file=True
        )
        self.analysis_input.setVisible(False)
        self._quick_analysis_file_label = _compact_path_label("No analysis dataset selected")
        self.analysis_input.textChanged.connect(
            lambda text: _set_path_label(
                self._quick_analysis_file_label,
                text,
                empty_text="No analysis dataset selected",
            )
        )
        btn_pick = QPushButton("Choose...")
        btn_pick.clicked.connect(self._pick_analysis_input)
        btn_path = QPushButton("Path")
        btn_path.setCheckable(True)
        btn_path.toggled.connect(self.analysis_input.setVisible)

        self.analysis_outdir = QLineEdit("")
        self.analysis_outdir.setPlaceholderText(
            f"Empty -> {DATASET_REPORTS_OUTPUT_ROOT}/<dataset>_<timestamp>"
        )
        btn_out = QPushButton("Output...")
        btn_out.clicked.connect(self._pick_analysis_outdir)
        out_row = _row_lineedit_with_button(self.analysis_outdir, btn_out)

        self.analysis_sample = QSpinBox()
        self.analysis_sample.setRange(100, 20_000_000)
        self.analysis_sample.setValue(200_000)
        self.analysis_sample.setSingleStep(10_000)

        self.analysis_scatter_n = QSpinBox()
        self.analysis_scatter_n.setRange(0, 2_000_000)
        self.analysis_scatter_n.setValue(50_000)
        self.analysis_scatter_n.setSingleStep(10_000)

        self.analysis_make_plots = QCheckBox("Create plots")
        self.analysis_make_plots.setChecked(True)

        self.analysis_auto_after_suite = QCheckBox("Analyze suite after generation")
        self.analysis_auto_after_suite.setChecked(False)
        self.analysis_auto_after_suite.setToolTip(
            "Runs analysis for train_hybrid, val_uniform, test_uniform, and ood_combined "
            "after suite generation. Disabled by default for very large suites."
        )

        btn_use_latest = QPushButton("Use Latest Output")
        btn_use_latest.clicked.connect(self._use_latest_analysis_candidate)
        btn_run = QPushButton("Analyze Dataset")
        btn_run.clicked.connect(lambda: self._run_cloud_analysis())
        btn_suite = QPushButton("Analyze Suite")
        btn_suite.clicked.connect(self._run_suite_analysis_from_last_dir)

        self.analysis_summary = QPlainTextEdit()
        _style_command_preview(self.analysis_summary, min_h=92, max_h=150)
        self.analysis_summary.setMaximumHeight(150)
        self.analysis_summary.setPlaceholderText("Analysis summary")

        form = QFormLayout()
        _tune_form(form)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Analysis Output", out_row)
        form.addRow("Sample Rows", self.analysis_sample)
        form.addRow("Scatter Points", self.analysis_scatter_n)

        toggles = QHBoxLayout()
        toggles.setContentsMargins(0, 0, 0, 0)
        toggles.setSpacing(10)
        toggles.addWidget(self.analysis_make_plots)
        toggles.addWidget(self.analysis_auto_after_suite)
        toggles.addStretch(1)

        detail = QWidget()
        detail_l = QVBoxLayout(detail)
        detail_l.setContentsMargins(0, 0, 0, 0)
        detail_l.setSpacing(8)
        detail_l.addWidget(self._quick_analysis_file_label)
        detail_l.addWidget(self.analysis_input)
        detail_l.addLayout(form)
        detail_l.addLayout(toggles)
        detail_l.addWidget(self.analysis_summary)

        panel = _data_action_card(
            "Quick Analysis",
            "Run a compact report on the latest generated dataset.",
            btn_run,
            secondary_buttons=[btn_use_latest, btn_pick, btn_path, btn_suite],
            detail=detail,
            object_name="quickAnalysisCard",
        )
        _tune_inputs(panel)

        self._analysis_proc = QProcess(self)
        self._analysis_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._analysis_proc.readyReadStandardOutput.connect(self._on_analysis_output)
        self._analysis_proc.finished.connect(self._on_analysis_finished)
        return panel

    def _pick_analysis_input(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Cloud Dataset",
            self.analysis_input.text() or str(SCRIPT_DIR),
            "Cloud datasets (*.h5 *.hdf5 *.pt);;All (*.*)",
        )
        if fn:
            self.analysis_input.setText(_norm_path(fn))

    def _pick_analysis_outdir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self,
            "Analysis Output Folder",
            self.analysis_outdir.text() or str(DATASET_REPORTS_OUTPUT_ROOT),
        )
        if d:
            self.analysis_outdir.setText(_norm_path(d))

    def _analysis_default_outdir(self, dataset_path: Path, label: str = "") -> Path:
        base = self.analysis_outdir.text().strip()
        if base:
            out = Path(base)
            if not out.is_absolute():
                out = (SCRIPT_DIR / out).resolve()
        else:
            out = _default_dataset_report_dir(str(dataset_path))
        return out / label if label else out

    def _suite_manifest_files(self, suite_dir: Path) -> Dict[str, str]:
        manifest_path = suite_dir / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        files = manifest.get("output_files", {})
        if not isinstance(files, dict):
            return {}

        def _resolve(value: object) -> str:
            if not value:
                return ""
            p = Path(str(value))
            if p.is_absolute():
                return str(p)
            for candidate in ((SCRIPT_DIR / p).resolve(), (suite_dir / p).resolve(), (suite_dir / p.name).resolve()):
                if candidate.exists():
                    return str(candidate)
            return str((suite_dir / p.name).resolve())

        return {str(k): _resolve(v) for k, v in files.items()}

    def _analysis_candidates(self) -> List[str]:
        candidates: List[str] = []
        if self._mode_combo.currentData() == self._MODE_SINGLE:
            out = self.out_path.text().strip()
            if out:
                candidates.append(out)
        if self._last_suite_dir:
            files = self._suite_manifest_files(Path(self._last_suite_dir))
            for key in ("train", "val", "test", "ood_combined", "ood_high", "ood_low"):
                if files.get(key):
                    candidates.append(files[key])
        candidates.extend([
            self.analysis_input.text().strip(),
            self.out_path.text().strip() if hasattr(self, "out_path") else "",
        ])
        seen: set[str] = set()
        resolved: List[str] = []
        for item in candidates:
            if not item:
                continue
            p = Path(item)
            if not p.is_absolute():
                p = (SCRIPT_DIR / p).resolve()
            s = str(p)
            if s not in seen and p.exists():
                seen.add(s)
                resolved.append(s)
        return resolved

    def _use_latest_analysis_candidate(self) -> None:
        candidates = self._analysis_candidates()
        if not candidates:
            QMessageBox.information(self, "Cloud Analysis", "No generated dataset was found.")
            return
        self.analysis_input.setText(candidates[0])

    def _build_analysis_args(self, dataset_path: Path, outdir: Path) -> List[str]:
        script = _STLRPS_DATA_MODULE_DIR / "spatial_cloud_analysis.py"
        args = [
            "-u", str(script), str(dataset_path),
            "--sample", str(self.analysis_sample.value()),
            "--seed", "123",
            "--scatter-n", str(self.analysis_scatter_n.value()),
            "--outdir", str(outdir),
            "--dump-json",
        ]
        if not self.analysis_make_plots.isChecked():
            args.append("--no-plots")
        return args

    def _run_cloud_analysis(self, dataset_path: Optional[str] = None, label: str = "dataset") -> None:
        if self._analysis_proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Cloud Analysis", "Analysis is already running.")
            return
        path_text = dataset_path or self.analysis_input.text().strip()
        if not path_text:
            self._use_latest_analysis_candidate()
            path_text = self.analysis_input.text().strip()
        if not path_text:
            return
        p = Path(path_text)
        if not p.is_absolute():
            p = (SCRIPT_DIR / p).resolve()
        if not p.exists():
            QMessageBox.warning(self, "Cloud Analysis", f"Dataset not found:\n{p}")
            return
        outdir = self._analysis_default_outdir(p, label if label != "dataset" else "")
        self._active_analysis = (str(p), str(outdir), label)
        self.analysis_summary.appendPlainText(f"\n[analysis] starting {label}: {p.name}")
        self._analysis_proc.setWorkingDirectory(str(_REPO_ROOT))
        self._analysis_proc.start(sys.executable, self._build_analysis_args(p, outdir))

    def _run_suite_analysis_from_last_dir(self) -> None:
        if not self._last_suite_dir:
            QMessageBox.information(self, "Cloud Analysis", "Generate or choose a dataset suite first.")
            return
        self._run_suite_analysis(Path(self._last_suite_dir))

    def _run_suite_analysis(self, suite_dir: Path) -> None:
        if self._analysis_proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Cloud Analysis", "Analysis is already running.")
            return
        suite_dir = Path(suite_dir)
        if not suite_dir.is_absolute():
            suite_dir = (SCRIPT_DIR / suite_dir).resolve()
        files = self._suite_manifest_files(suite_dir)
        order = [
            ("train", files.get("train", "")),
            ("val", files.get("val", "")),
            ("test", files.get("test", "")),
            ("ood", files.get("ood_combined", "") or files.get("ood_high", "") or files.get("ood_low", "")),
        ]
        self._analysis_queue.clear()
        for label, value in order:
            if value and Path(value).exists():
                outdir = self._analysis_default_outdir(Path(value), label)
                self._analysis_queue.append((value, str(outdir), label))
        if not self._analysis_queue:
            QMessageBox.warning(self, "Cloud Analysis", "No analyzable HDF5 file was found in the suite.")
            return
        self.analysis_summary.setPlainText("[analysis] suite analysis queued...")
        self._start_next_analysis_job()

    def _start_next_analysis_job(self) -> None:
        if not self._analysis_queue:
            return
        dataset_path, outdir, label = self._analysis_queue.popleft()
        self._active_analysis = (dataset_path, outdir, label)
        self.analysis_summary.appendPlainText(f"\n[analysis] starting {label}: {Path(dataset_path).name}")
        self._analysis_proc.setWorkingDirectory(str(_REPO_ROOT))
        self._analysis_proc.start(
            sys.executable,
            self._build_analysis_args(Path(dataset_path), Path(outdir)),
        )

    def _on_analysis_output(self) -> None:
        data = bytes(self._analysis_proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data.strip():
            self.analysis_summary.appendPlainText(data.rstrip())

    def _format_analysis_summary(self, summary_path: Path, label: str) -> str:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return f"[analysis] {label}: summary.json could not be read ({exc})"
        meta = data.get("meta", {})
        analyzed = data.get("analyzed", {})
        stats = data.get("stats", {})
        quality = data.get("quality", {})
        finite = quality.get("finite", {})
        direction = quality.get("spatial_direction_balance", {})
        alt_balance = quality.get("altitude_balance", {})
        geom = quality.get("acceleration_geometry", {})
        field = quality.get("field_dynamic_range", {})
        warnings = quality.get("warnings", [])

        def _fmt(value: object, digits: int = 3) -> str:
            try:
                return f"{float(value):.{digits}g}"
            except Exception:
                return "-"

        lines = [
            f"[summary:{label}] rows={meta.get('n_total', '-')} analyzed={analyzed.get('rows_after_filter', '-')}",
            f"  role={meta.get('dataset_role') or 'unknown'} target={meta.get('target_mode') or 'unknown'} degree={meta.get('degree_min')}-{meta.get('degree_max')}",
            f"  finite={_fmt(finite.get('finite_row_fraction'))} nonfinite_rows={finite.get('nonfinite_rows', 0)}",
            f"  altitude_balance: empty_bins={alt_balance.get('empty_bins', '-')} cv={_fmt(alt_balance.get('coefficient_of_variation'))} entropy={_fmt(alt_balance.get('entropy_score'))}",
            f"  direction_balance: octant_entropy={_fmt(direction.get('octant_entropy_score'))} max_octant={_fmt(direction.get('octant_max_fraction'))} mean_dir_norm={_fmt(direction.get('mean_unit_vector_norm'))}",
            f"  field_range: |U|p99/p50={_fmt(field.get('abs_potential_p99_over_p50'))} |a|p99/p50={_fmt(field.get('accel_norm_p99_over_p50'))}",
            f"  accel_geometry: cross/total_med={_fmt(geom.get('cross_to_total_median'))} |radial|/total_med={_fmt(geom.get('radial_abs_to_total_median'))}",
        ]
        if warnings:
            lines.append("  warnings: " + "; ".join(str(w) for w in warnings))
        return "\n".join(lines)

    def _on_analysis_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        active = self._active_analysis
        self._active_analysis = None
        ok = exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
        if active is not None:
            _dataset, outdir, label = active
            if ok:
                self.analysis_summary.appendPlainText(
                    self._format_analysis_summary(Path(outdir) / "summary.json", label)
                )
            else:
                self.analysis_summary.appendPlainText(f"[analysis] {label} failed with exit_code={exit_code}")
        if self._analysis_queue:
            self._start_next_analysis_job()

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------
    def set_train_tab(self, train_tab: Any) -> None:
        self._train_tab_ref = train_tab

    def _emit_params_changed(self) -> None:
        self.cloud_params_changed.emit(
            self.alt_min_km.value(),
            self.alt_max_km.value(),
            self.degree_min.value(),
            self.degree_max.value(),
        )

    def _on_mode_changed(self, _idx: int) -> None:
        mode = self._mode_combo.currentData()
        self._stack.setCurrentIndex(int(mode))
        self._sync_banner.setVisible(False)
        btn_label = "Start Suite Generation" if mode == self._MODE_SUITE else "Start Cloud Generation"
        self.runner.btn_start.setText(btn_label)
        if hasattr(self, "_btn_generate_now"):
            self._btn_generate_now.setText(btn_label)

    def _update_suite_total_label(self) -> None:
        total = (
            self.s_train_su_n.value()
            + self.s_train_ir2_n.value()
            + self.s_train_rm_n.value()
            + self.s_train_bb_n.value()
        )
        self._suite_total_lbl.setText(f"Toplam train: {total:,}")

    # ------------------------------------------------------------------
    # Suite preset application
    # ------------------------------------------------------------------
    def _apply_suite_preset(self) -> None:
        preset_key = self.s_preset_combo.currentData()
        if not preset_key:
            return
        ssot_preset = SUITE_PRESETS.get(str(preset_key)) if isinstance(SUITE_PRESETS, dict) else None
        if ssot_preset is not None:
            self.s_degree_min.setValue(int(_cfg_value(ssot_preset, "degree_min", self.s_degree_min.value())))
            self.s_degree_max.setValue(int(_cfg_value(ssot_preset, "degree_max", self.s_degree_max.value())))
            self.s_train_alt_min_km.setValue(float(_cfg_value(ssot_preset, "train_alt_min_km", self.s_train_alt_min_km.value())))
            self.s_train_alt_max_km.setValue(float(_cfg_value(ssot_preset, "train_alt_max_km", self.s_train_alt_max_km.value())))
            self.s_ood_margin_km.setValue(float(_cfg_value(ssot_preset, "ood_margin_km", self.s_ood_margin_km.value())))
            self.s_train_su_n.setValue(int(_cfg_value(ssot_preset, "train_stratified_uniform_n", self.s_train_su_n.value())))
            self.s_train_ir2_n.setValue(int(_cfg_value(ssot_preset, "train_inverse_r2_n", self.s_train_ir2_n.value())))
            self.s_train_rm_n.setValue(int(_cfg_value(ssot_preset, "train_residual_mag_n", self.s_train_rm_n.value())))
            self.s_train_bb_n.setValue(int(_cfg_value(ssot_preset, "train_boundary_n", self.s_train_bb_n.value())))
            self.s_val_n.setValue(int(_cfg_value(ssot_preset, "val_n", self.s_val_n.value())))
            self.s_test_n.setValue(int(_cfg_value(ssot_preset, "test_n", self.s_test_n.value())))
            self.s_ood_low_n.setValue(int(_cfg_value(ssot_preset, "ood_low_n", self.s_ood_low_n.value())))
            self.s_ood_high_n.setValue(int(_cfg_value(ssot_preset, "ood_high_n", self.s_ood_high_n.value())))
            self.s_residual_mag_candidate_multiplier.setValue(
                int(_cfg_value(ssot_preset, "residual_mag_candidate_multiplier", self.s_residual_mag_candidate_multiplier.value()))
            )
            self.s_residual_mag_weight_power.setValue(
                float(_cfg_value(ssot_preset, "residual_mag_weight_power", self.s_residual_mag_weight_power.value()))
            )
            boundary_idx = self.s_boundary_mode.findData(str(_cfg_value(ssot_preset, "boundary_mode", self.s_boundary_mode.currentData())))
            if boundary_idx >= 0:
                self.s_boundary_mode.setCurrentIndex(boundary_idx)
            self.s_boundary_width_km.setValue(float(_cfg_value(ssot_preset, "boundary_width_km", self.s_boundary_width_km.value())))
            self.s_seed_base.setValue(int(_cfg_value(ssot_preset, "base_seed", self.s_seed_base.value())))
            self.s_seed_train_uniform.setValue(int(_cfg_value(ssot_preset, "train_uniform_seed", self.s_seed_train_uniform.value())))
            self.s_seed_train_ir2.setValue(int(_cfg_value(ssot_preset, "train_inverse_r2_seed", self.s_seed_train_ir2.value())))
            self.s_seed_train_rm.setValue(int(_cfg_value(ssot_preset, "train_residual_mag_seed", self.s_seed_train_rm.value())))
            self.s_seed_train_bb.setValue(int(_cfg_value(ssot_preset, "train_boundary_seed", self.s_seed_train_bb.value())))
            self.s_seed_val.setValue(int(_cfg_value(ssot_preset, "val_seed", self.s_seed_val.value())))
            self.s_seed_test.setValue(int(_cfg_value(ssot_preset, "test_seed", self.s_seed_test.value())))
            self.s_seed_ood_low.setValue(int(_cfg_value(ssot_preset, "ood_low_seed", self.s_seed_ood_low.value())))
            self.s_seed_ood_high.setValue(int(_cfg_value(ssot_preset, "ood_high_seed", self.s_seed_ood_high.value())))
            self.s_chunk_size.setValue(int(_cfg_value(ssot_preset, "chunk_size", self.s_chunk_size.value())))
            dtype_idx = self.s_dtype.findData(str(_cfg_value(ssot_preset, "dtype", self.s_dtype.currentData())))
            if dtype_idx >= 0:
                self.s_dtype.setCurrentIndex(dtype_idx)
            self._update_suite_total_label()
            return
        presets: Dict[str, Dict[str, int]] = {
            "debug_suite": {
                "su": 50_000, "ir2": 20_000, "rm": 20_000, "bb": 10_000,
                "val": 20_000, "test": 20_000, "ood_lo": 10_000, "ood_hi": 10_000,
            },
            "baseline_uniform_suite": {
                "su": 2_000_000, "ir2": 0, "rm": 0, "bb": 0,
                "val": 500_000, "test": 1_000_000, "ood_lo": 250_000, "ood_hi": 250_000,
            },
            "recommended_hybrid_5M": {
                "su": 2_000_000, "ir2": 1_000_000, "rm": 1_000_000, "bb": 1_000_000,
                "val": 1_000_000, "test": 1_000_000, "ood_lo": 250_000, "ood_hi": 250_000,
            },
            "high_accuracy_10M": {
                "su": 4_000_000, "ir2": 2_000_000, "rm": 2_000_000, "bb": 2_000_000,
                "val": 2_000_000, "test": 2_000_000, "ood_lo": 500_000, "ood_hi": 500_000,
            },
        }
        p = presets.get(preset_key)
        if p is None:
            return
        self.s_train_su_n.setValue(p["su"])
        self.s_train_ir2_n.setValue(p["ir2"])
        self.s_train_rm_n.setValue(p["rm"])
        self.s_train_bb_n.setValue(p["bb"])
        self.s_val_n.setValue(p["val"])
        self.s_test_n.setValue(p["test"])
        self.s_ood_low_n.setValue(p["ood_lo"])
        self.s_ood_high_n.setValue(p["ood_hi"])
        self._update_suite_total_label()

    def _auto_assign_seeds(self) -> None:
        base = int(self.s_seed_base.value())
        self.s_seed_train_uniform.setValue(base)
        self.s_seed_train_ir2.setValue(base + 100)
        self.s_seed_train_rm.setValue(base + 200)
        self.s_seed_train_bb.setValue(base + 300)
        self.s_seed_val.setValue(base + 1000)
        self.s_seed_test.setValue(base + 2000)
        self.s_seed_ood_low.setValue(base + 3000)
        self.s_seed_ood_high.setValue(base + 4000)

    # ------------------------------------------------------------------
    # Suite post-completion
    # ------------------------------------------------------------------
    def _open_suite_folder(self) -> None:
        d = self._last_suite_dir
        if not d or not Path(d).is_dir():
            QMessageBox.information(self, "Suite Folder", "No generated suite folder was found.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _apply_suite_to_train(self) -> None:
        d = self._last_suite_dir
        if not d or not Path(d).is_dir():
            QMessageBox.information(self, "Suite Folder", "Generate a suite first.")
            return
        self._fill_train_tab_from_manifest(Path(d))

    def _fill_train_tab_from_manifest(self, suite_dir: Path) -> None:
        suite_dir = Path(suite_dir)
        if not suite_dir.is_absolute():
            suite_dir = (SCRIPT_DIR / suite_dir).resolve()
        manifest_path = suite_dir / "manifest.json"
        if not manifest_path.exists():
            return
        try:
            import json as _json
            manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.warning(self, "Manifest Hatasi", f"manifest.json okunamadi: {e}")
            return
        files = manifest.get("output_files", {})

        def _suite_file(value: object) -> str:
            """Resolve manifest file paths relative to the suite/script directory."""
            if not value:
                return ""
            p = Path(str(value))
            if not p.is_absolute():
                candidates = [
                    (SCRIPT_DIR / p).resolve(),
                    (suite_dir / p).resolve(),
                    (suite_dir / p.name).resolve(),
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return str(candidate)
                return str(candidates[0])
            return str(p)

        train_path = _suite_file(files.get("train", ""))
        val_path = _suite_file(files.get("val", ""))
        test_path = _suite_file(files.get("test", ""))
        ood_path = _suite_file(files.get("ood_combined", "") or files.get("ood_high", ""))
        t = self._train_tab_ref
        if t is None:
            return
        try:
            # -- File paths --
            if train_path and hasattr(t, "train_data"):
                t.train_data.setText(str(train_path))
            if val_path and hasattr(t, "val_data"):
                t.val_data.setText(str(val_path))
            if test_path and hasattr(t, "test_data"):
                t.test_data.setText(str(test_path))
            if ood_path and hasattr(t, "ood_data"):
                t.ood_data.setText(str(ood_path))
            # -- Dataset name --
            if hasattr(t, "dataset_name"):
                t.dataset_name.setText("data")
            # -- Force independent train/val/test/OOD mode --
            if hasattr(t, "dataset_mode"):
                idx = t.dataset_mode.findData("independent")
                if idx >= 0:
                    t.dataset_mode.setCurrentIndex(idx)
            # -- Force Train then evaluate workflow --
            if hasattr(t, "workflow_mode"):
                idx = t.workflow_mode.findData("train_then_eval")
                if idx >= 0:
                    t.workflow_mode.setCurrentIndex(idx)
            # -- Altitude range from manifest --
            alt_min = manifest.get("train_alt_min_km")
            alt_max = manifest.get("train_alt_max_km")
            if alt_min is not None and hasattr(t, "altitude_min_km"):
                t.altitude_min_km.setValue(float(alt_min))
            if alt_max is not None and hasattr(t, "altitude_max_km"):
                t.altitude_max_km.setValue(float(alt_max))
            # -- Suite manifest provenance --
            if hasattr(t, "applied_suite_manifest_path"):
                t.applied_suite_manifest_path = str(manifest_path.resolve())
            if hasattr(t, "_suite_manifest_label"):
                t._suite_manifest_label.setText(str(manifest_path.resolve()))
                t._suite_manifest_label.setStyleSheet("color: #6ee7b7; font-size: 10px;")
            # -- Trigger dependent UI updates --
            if hasattr(t, "_on_dataset_mode_changed"):
                t._on_dataset_mode_changed()
            if hasattr(t, "_on_workflow_mode_changed"):
                t._on_workflow_mode_changed()
            if hasattr(t, "_refresh_command_preview"):
                t._refresh_command_preview()
            if hasattr(t, "_refresh_checklist"):
                t._refresh_checklist()
        except Exception:
            pass
        # Show confirmation in the banner
        self._sync_banner.setText(
            "Dataset suite applied: independent train/val/test/OOD mode enabled."
        )
        self._sync_banner.setVisible(True)

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------
    def _pick_gfc_path(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self, "Choose Gravity Model",
            self.gfc_path.text() or str(SCRIPT_DIR),
            "GFC/TAB (*.gfc *.tab *.txt);;All (*.*)",
        )
        if fn:
            self.gfc_path.setText(_norm_path(fn))

    def _pick_out_path(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "Output File",
            self.out_path.text() or str(SCRIPT_DIR / "data"),
            "HDF5 (*.h5);;PyTorch (*.pt);;All (*.*)",
        )
        if fn:
            self.out_path.setText(_norm_path(fn))

    def _pick_suite_gfc_path(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self, "Choose Gravity Model",
            self.s_gfc_path.text() or str(SCRIPT_DIR),
            "GFC/TAB (*.gfc *.tab *.txt);;All (*.*)",
        )
        if fn:
            self.s_gfc_path.setText(_norm_path(fn))

    def _pick_suite_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Suite Output Folder",
            self.s_suite_out_dir.text() or str(DATASET_SUITE_OUTPUT_ROOT),
        )
        if d:
            self.s_suite_out_dir.setText(_norm_path(d))

    # ------------------------------------------------------------------
    # CLI arg builders
    # ------------------------------------------------------------------
    def _build_single_args(self, show_errors: bool = True) -> Optional[List[str]]:
        script = _STLRPS_DATA_MODULE_DIR / "spatial_cloud_generator.py"
        if not script.exists():
            if show_errors:
                QMessageBox.critical(self, "Missing Script", "spatial_cloud_generator.py was not found.")
            return None
        deg_min = self.degree_min.value()
        deg_max = self.degree_max.value()
        if deg_max <= deg_min and deg_min != -1:
            if show_errors:
                QMessageBox.critical(self, "Invalid Degree", f"degree_max ({deg_max}) must be greater than degree_min ({deg_min}).")
            return None
        alt_min = self.alt_min_km.value()
        alt_max = self.alt_max_km.value()
        if alt_max <= alt_min:
            if show_errors:
                QMessageBox.critical(self, "Invalid Altitude", f"alt_max ({alt_max}) must be greater than alt_min ({alt_min}).")
            return None
        args: List[str] = ["-u", str(script)]
        args += ["--degree-max", str(deg_max), "--degree-min", str(deg_min)]
        args += ["--n-samples", str(self.n_samples.value())]
        args += ["--alt-range", str(alt_min), str(alt_max)]
        args += ["--sampling-strategy", self.sampling_strategy.currentData() or "mixed"]
        args += ["--surface-bias-ratio", str(self.surface_bias_ratio.value())]
        args += ["--chunk-size", str(self.chunk_size.value())]
        args += ["--workers", str(self.workers.value())]
        args += ["--format", self.out_format.currentData() or "h5"]
        out = self.out_path.text().strip()
        if out:
            args += ["--out", out]
        args += ["--dtype", self.dtype.currentData() or "float32"]
        args += ["--canonical" if self.canonical.isChecked() else "--si"]
        args += ["--seed", str(self.seed.value())]
        gfc = self.gfc_path.text().strip()
        if gfc:
            args += ["--gfc-path", gfc]
        if self.no_multiprocessing.isChecked():
            args += ["--no-multiprocessing"]
        return args

    def _build_suite_args(self, show_errors: bool = True) -> Optional[List[str]]:
        script = _STLRPS_DATA_MODULE_DIR / "spatial_cloud_generator.py"
        if not script.exists():
            if show_errors:
                QMessageBox.critical(self, "Missing Script", "spatial_cloud_generator.py was not found.")
            return None
        deg_min = self.s_degree_min.value()
        deg_max = self.s_degree_max.value()
        if deg_max <= deg_min:
            if show_errors:
                QMessageBox.critical(self, "Invalid Degree", f"degree_max ({deg_max}) must be greater than degree_min ({deg_min}).")
            return None
        alt_min = self.s_train_alt_min_km.value()
        alt_max = self.s_train_alt_max_km.value()
        if alt_max <= alt_min:
            if show_errors:
                QMessageBox.critical(self, "Invalid Altitude", f"alt_max ({alt_max}) must be greater than alt_min ({alt_min}).")
            return None

        args: List[str] = ["-u", str(script), "--generate-suite"]
        args += ["--degree-min", str(deg_min), "--degree-max", str(deg_max)]
        args += ["--train-alt-min-km", str(alt_min), "--train-alt-max-km", str(alt_max)]
        args += ["--ood-margin-km", str(self.s_ood_margin_km.value())]
        args += ["--train-stratified-uniform-n", str(self.s_train_su_n.value())]
        args += ["--train-inverse-r2-n", str(self.s_train_ir2_n.value())]
        args += ["--train-residual-mag-n", str(self.s_train_rm_n.value())]
        args += ["--train-boundary-n", str(self.s_train_bb_n.value())]
        args += ["--val-n", str(self.s_val_n.value())]
        args += ["--test-n", str(self.s_test_n.value())]
        args += ["--ood-low-n", str(self.s_ood_low_n.value())]
        args += ["--ood-high-n", str(self.s_ood_high_n.value())]
        args += ["--base-seed", str(self.s_seed_base.value())]
        args += ["--train-uniform-seed", str(self.s_seed_train_uniform.value())]
        args += ["--train-inverse-r2-seed", str(self.s_seed_train_ir2.value())]
        args += ["--train-residual-mag-seed", str(self.s_seed_train_rm.value())]
        args += ["--train-boundary-seed", str(self.s_seed_train_bb.value())]
        args += ["--val-seed", str(self.s_seed_val.value())]
        args += ["--test-seed", str(self.s_seed_test.value())]
        args += ["--ood-low-seed", str(self.s_seed_ood_low.value())]
        args += ["--ood-high-seed", str(self.s_seed_ood_high.value())]
        args += ["--residual-mag-candidate-multiplier", str(self.s_residual_mag_candidate_multiplier.value())]
        args += ["--residual-mag-weight-power", str(self.s_residual_mag_weight_power.value())]
        args += ["--boundary-mode", self.s_boundary_mode.currentData() or "strict"]
        args += ["--boundary-width-km", str(self.s_boundary_width_km.value())]
        args += ["--chunk-size", str(self.s_chunk_size.value())]
        args += ["--dtype", self.s_dtype.currentData() or "float32"]
        if self.s_combine_ood.isChecked():
            args += ["--combine-ood"]
        else:
            args += ["--no-combine-ood"]
        name = self.s_suite_name.text().strip()
        if name:
            args += ["--suite-name", name]
        out_dir = self.s_suite_out_dir.text().strip()
        if out_dir:
            args += ["--suite-out-dir", out_dir]
        gfc = self.s_gfc_path.text().strip()
        if gfc:
            args += ["--gfc-path", gfc]
        return args

    def _build_args(self, show_errors: bool = True) -> Optional[List[str]]:
        if self._mode_combo.currentData() == self._MODE_SUITE:
            return self._build_suite_args(show_errors=show_errors)
        return self._build_single_args(show_errors=show_errors)

    def _refresh_preview(self) -> None:
        args = self._build_args(show_errors=False)
        if args:
            self.command_preview.setPlainText(_format_command(sys.executable, args))

    def _show_command_preview(self) -> None:
        self._refresh_preview()
        self.command_preview.setVisible(True)

    # ------------------------------------------------------------------
    # Start / finish
    # ------------------------------------------------------------------
    def _start(self) -> None:
        args = self._build_args(show_errors=True)
        if args is None:
            return
        self._sync_banner.setVisible(False)
        self._save_settings()
        self.runner.progress.setRange(0, 0)
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        ok = exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
        if self._mode_combo.currentData() == self._MODE_SUITE:
            self._on_suite_finished(ok)
        else:
            self._on_single_finished(ok)

    def _on_single_finished(self, ok: bool) -> None:
        if ok:
            self._emit_params_changed()
            msg = (
                "Cloud generation complete! "
                f"Altitude: {self.alt_min_km.value():.0f}-{self.alt_max_km.value():.0f} km  |  "
                f"Degree: {self.degree_min.value()}->{self.degree_max.value()}  |  "
                "Training tab synchronized."
            )
            self._sync_banner.setText(msg)
            self._sync_banner.setVisible(True)

    def _on_suite_finished(self, ok: bool) -> None:
        if not ok:
            return
        # Try to find the suite dir from the runner output
        log_text = ""
        try:
            log_text = self.runner.log.toPlainText()
        except Exception:
            pass
        suite_dir: Optional[str] = None
        for line in reversed(log_text.splitlines()):
            m = re.search(r"suite dir\s*[:\->]+\s*(.+)", line, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                candidate_path = Path(candidate)
                if not candidate_path.is_absolute():
                    candidate_path = (SCRIPT_DIR / candidate_path).resolve()
                if candidate_path.is_dir():
                    suite_dir = str(candidate_path)
                    break
        # Fallback: look for manifest.json in recent suite dirs
        if suite_dir is None:
            out_dir_text = self.s_suite_out_dir.text().strip()
            base = Path(out_dir_text) if out_dir_text else DATASET_SUITE_OUTPUT_ROOT
            if base.is_dir():
                dirs = sorted(base.iterdir(), key=lambda d: d.stat().st_mtime if d.is_dir() else 0, reverse=True)
                for d in dirs[:5]:
                    if d.is_dir() and (d / "manifest.json").exists():
                        suite_dir = str(d)
                        break

        self._last_suite_dir = suite_dir
        msg = "Suite generation complete!"
        if suite_dir:
            msg += f"  Folder: {suite_dir}"
        self._sync_banner.setText(msg)
        self._sync_banner.setVisible(True)

        if suite_dir and self.s_auto_apply.isChecked():
            self._fill_train_tab_from_manifest(Path(suite_dir))
        if suite_dir and self.analysis_auto_after_suite.isChecked():
            self._run_suite_analysis(Path(suite_dir))

    def _parse_progress(self, line: str) -> None:
        m = re.search(r"(?i)chunk\s*(\d+)\s*/\s*(\d+)", line)
        if m:
            cur = int(m.group(1))
            tot = int(m.group(2))
            self.runner.progress.setRange(0, max(1, tot))
            self.runner.progress.setValue(min(cur, tot))
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
        if m2:
            pct = min(100, int(float(m2.group(1))))
            self.runner.progress.setRange(0, 100)
            self.runner.progress.setValue(pct)

    # ------------------------------------------------------------------
    # QSettings persistence
    # ------------------------------------------------------------------
    def _save_settings(self) -> None:
        s = _settings()
        s.beginGroup("cloudgen")
        # Single cloud
        s.setValue("degree_max", self.degree_max.value())
        s.setValue("degree_min", self.degree_min.value())
        s.setValue("n_samples", self.n_samples.value())
        s.setValue("alt_min_km", self.alt_min_km.value())
        s.setValue("alt_max_km", self.alt_max_km.value())
        s.setValue("sampling_strategy", self.sampling_strategy.currentData())
        s.setValue("surface_bias_ratio", self.surface_bias_ratio.value())
        s.setValue("chunk_size", self.chunk_size.value())
        s.setValue("workers", self.workers.value())
        s.setValue("out_format", self.out_format.currentData())
        s.setValue("out_path", self.out_path.text())
        s.setValue("dtype", self.dtype.currentData())
        s.setValue("canonical", self.canonical.isChecked())
        s.setValue("seed", self.seed.value())
        s.setValue("gfc_path", self.gfc_path.text())
        s.setValue("no_multiprocessing", self.no_multiprocessing.isChecked())
        # Mode
        s.setValue("mode", self._mode_combo.currentData())
        # Suite
        s.setValue("s_degree_min", self.s_degree_min.value())
        s.setValue("s_degree_max", self.s_degree_max.value())
        s.setValue("s_train_alt_min_km", self.s_train_alt_min_km.value())
        s.setValue("s_train_alt_max_km", self.s_train_alt_max_km.value())
        s.setValue("s_ood_margin_km", self.s_ood_margin_km.value())
        s.setValue("s_train_su_n", self.s_train_su_n.value())
        s.setValue("s_train_ir2_n", self.s_train_ir2_n.value())
        s.setValue("s_train_rm_n", self.s_train_rm_n.value())
        s.setValue("s_train_bb_n", self.s_train_bb_n.value())
        s.setValue("s_val_n", self.s_val_n.value())
        s.setValue("s_test_n", self.s_test_n.value())
        s.setValue("s_ood_low_n", self.s_ood_low_n.value())
        s.setValue("s_ood_high_n", self.s_ood_high_n.value())
        s.setValue("s_seed_base", self.s_seed_base.value())
        s.setValue("s_seed_train_uniform", self.s_seed_train_uniform.value())
        s.setValue("s_seed_train_ir2", self.s_seed_train_ir2.value())
        s.setValue("s_seed_train_rm", self.s_seed_train_rm.value())
        s.setValue("s_seed_train_bb", self.s_seed_train_bb.value())
        s.setValue("s_seed_val", self.s_seed_val.value())
        s.setValue("s_seed_test", self.s_seed_test.value())
        s.setValue("s_seed_ood_low", self.s_seed_ood_low.value())
        s.setValue("s_seed_ood_high", self.s_seed_ood_high.value())
        s.setValue("s_residual_mag_candidate_multiplier", self.s_residual_mag_candidate_multiplier.value())
        s.setValue("s_residual_mag_weight_power", self.s_residual_mag_weight_power.value())
        s.setValue("s_boundary_mode", self.s_boundary_mode.currentData())
        s.setValue("s_boundary_width_km", self.s_boundary_width_km.value())
        s.setValue("s_chunk_size", self.s_chunk_size.value())
        s.setValue("s_dtype", self.s_dtype.currentData())
        s.setValue("s_combine_ood", self.s_combine_ood.isChecked())
        s.setValue("s_suite_name", self.s_suite_name.text())
        s.setValue("s_suite_out_dir", self.s_suite_out_dir.text())
        s.setValue("s_auto_apply", self.s_auto_apply.isChecked())
        s.setValue("s_gfc_path", self.s_gfc_path.text())
        s.endGroup()
        s.sync()

    def _restore_settings(self) -> None:
        s = _settings()
        s.beginGroup("cloudgen")

        def _i(k: str, d: int) -> int:
            return int(s.value(k, d)) if s.contains(k) else d

        def _f(k: str, d: float) -> float:
            return float(s.value(k, d)) if s.contains(k) else d

        def _b(k: str, d: bool) -> bool:
            if s.contains(k):
                v = s.value(k, d)
                return str(v).lower() == "true" if isinstance(v, str) else bool(v)
            return d

        def _st(k: str, d: str) -> str:
            return str(s.value(k, d)) if s.contains(k) else d

        # Single cloud
        self.degree_max.setValue(_i("degree_max", 100))
        self.degree_min.setValue(_i("degree_min", 20))
        self.n_samples.setValue(_i("n_samples", 2_000_000))
        self.alt_min_km.setValue(_f("alt_min_km", 200.0))
        self.alt_max_km.setValue(_f("alt_max_km", 600.0))
        strategy = _st("sampling_strategy", "mixed")
        idx = self.sampling_strategy.findData(strategy)
        if idx >= 0:
            self.sampling_strategy.setCurrentIndex(idx)
        self.surface_bias_ratio.setValue(_f("surface_bias_ratio", 0.70))
        self.chunk_size.setValue(_i("chunk_size", 50_000))
        self.workers.setValue(_i("workers", self.workers.value()))
        fmt = _st("out_format", "h5")
        idx = self.out_format.findData(fmt)
        if idx >= 0:
            self.out_format.setCurrentIndex(idx)
        self.out_path.setText(_st("out_path", ""))
        dtype = _st("dtype", "float32")
        idx = self.dtype.findData(dtype)
        if idx >= 0:
            self.dtype.setCurrentIndex(idx)
        self.canonical.setChecked(_b("canonical", False))
        self.seed.setValue(_i("seed", 12345))
        self.gfc_path.setText(_st("gfc_path", ""))
        self.no_multiprocessing.setChecked(_b("no_multiprocessing", False))
        # Mode
        saved_mode = _i("mode", self._MODE_SINGLE)
        idx = self._mode_combo.findData(saved_mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
            self._stack.setCurrentIndex(saved_mode)
        # Suite
        self.s_degree_min.setValue(_i("s_degree_min", 20))
        self.s_degree_max.setValue(_i("s_degree_max", 100))
        self.s_train_alt_min_km.setValue(_f("s_train_alt_min_km", 200.0))
        self.s_train_alt_max_km.setValue(_f("s_train_alt_max_km", 600.0))
        self.s_ood_margin_km.setValue(_f("s_ood_margin_km", 40.0))
        self.s_train_su_n.setValue(_i("s_train_su_n", 2_000_000))
        self.s_train_ir2_n.setValue(_i("s_train_ir2_n", 1_000_000))
        self.s_train_rm_n.setValue(_i("s_train_rm_n", 1_000_000))
        self.s_train_bb_n.setValue(_i("s_train_bb_n", 1_000_000))
        self.s_val_n.setValue(_i("s_val_n", 1_000_000))
        self.s_test_n.setValue(_i("s_test_n", 1_000_000))
        self.s_ood_low_n.setValue(_i("s_ood_low_n", 250_000))
        self.s_ood_high_n.setValue(_i("s_ood_high_n", 250_000))
        self.s_seed_base.setValue(_i("s_seed_base", 42))
        self.s_seed_train_uniform.setValue(_i("s_seed_train_uniform", 42))
        self.s_seed_train_ir2.setValue(_i("s_seed_train_ir2", 142))
        self.s_seed_train_rm.setValue(_i("s_seed_train_rm", 242))
        self.s_seed_train_bb.setValue(_i("s_seed_train_bb", 342))
        self.s_seed_val.setValue(_i("s_seed_val", 1042))
        self.s_seed_test.setValue(_i("s_seed_test", 2042))
        self.s_seed_ood_low.setValue(_i("s_seed_ood_low", 3042))
        self.s_seed_ood_high.setValue(_i("s_seed_ood_high", 4042))
        self.s_residual_mag_candidate_multiplier.setValue(_i("s_residual_mag_candidate_multiplier", 5))
        self.s_residual_mag_weight_power.setValue(_f("s_residual_mag_weight_power", 0.5))
        bm = _st("s_boundary_mode", "strict")
        idx = self.s_boundary_mode.findData(bm)
        if idx >= 0:
            self.s_boundary_mode.setCurrentIndex(idx)
        self.s_boundary_width_km.setValue(_f("s_boundary_width_km", 20.0))
        self.s_chunk_size.setValue(_i("s_chunk_size", 50_000))
        s_dtype_val = _st("s_dtype", "float32")
        idx = self.s_dtype.findData(s_dtype_val)
        if idx >= 0:
            self.s_dtype.setCurrentIndex(idx)
        self.s_combine_ood.setChecked(_b("s_combine_ood", True))
        self.s_suite_name.setText(_st("s_suite_name", ""))
        self.s_suite_out_dir.setText(_st("s_suite_out_dir", ""))
        self.s_auto_apply.setChecked(_b("s_auto_apply", True))
        self.s_gfc_path.setText(_st("s_gfc_path", ""))
        self._update_suite_total_label()
        s.endGroup()


class CloudAnalysisTab(QWidget):
    """Run spatial_cloud_analysis.py on a dataset and display the resulting plots."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.input_file = ValidatedPathEdit(
            placeholder="Dataset (.h5 or .pt)", check_file=True
        )
        self.input_file.setVisible(False)
        self._analysis_file_label = _compact_path_label("No dataset selected")
        self.input_file.textChanged.connect(
            lambda text: _set_path_label(
                self._analysis_file_label,
                text,
                empty_text="No dataset selected",
            )
        )
        btn_input = QPushButton("Choose Dataset")
        btn_input.clicked.connect(self._pick_input)
        self._btn_start_analysis_top = QPushButton("Start Analysis")
        self._btn_start_analysis_top.clicked.connect(self._start)
        btn_path = QPushButton("Path")
        btn_path.setCheckable(True)
        btn_path.toggled.connect(self.input_file.setVisible)

        file_detail = QWidget()
        file_detail_l = QVBoxLayout(file_detail)
        file_detail_l.setContentsMargins(0, 0, 0, 0)
        file_detail_l.setSpacing(8)
        file_detail_l.addWidget(self._analysis_file_label)
        file_detail_l.addWidget(self.input_file)
        input_card = _data_action_card(
            "Analyze Dataset",
            "Choose a generated cloud and create the quality report.",
            btn_input,
            secondary_buttons=[self._btn_start_analysis_top, btn_path],
            detail=file_detail,
        )

        # --- Analysis parameters ---
        grp_params = QGroupBox("Analysis Settings")
        form_params = QFormLayout()
        _tune_form(form_params)

        self.sample_n = QSpinBox()
        self.sample_n.setRange(1_000, 10_000_000)
        self.sample_n.setValue(200_000)
        self.sample_n.setSingleStep(10_000)
        self.sample_n.setToolTip("Rows sampled for analysis.")

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed_val = 123
        self.seed.setValue(123)
        self.seed.setToolTip("Sampling seed (reproducible)")

        self.alt_min_km = QDoubleSpinBox()
        self.alt_min_km.setRange(-1.0, 100_000.0)
        self.alt_min_km.setValue(-1.0)
        self.alt_min_km.setDecimals(1)
        self.alt_min_km.setSpecialValueText("Filtre yok")
        self.alt_min_km.setToolTip("Minimum irtifa filtresi (-1 = devre disi)")

        self.alt_max_km = QDoubleSpinBox()
        self.alt_max_km.setRange(-1.0, 100_000.0)
        self.alt_max_km.setValue(-1.0)
        self.alt_max_km.setDecimals(1)
        self.alt_max_km.setSpecialValueText("Filtre yok")
        self.alt_max_km.setToolTip("Maksimum irtifa filtresi (-1 = devre disi)")

        self.scatter_n = QSpinBox()
        self.scatter_n.setRange(100, 1_000_000)
        self.scatter_n.setValue(50_000)
        self.scatter_n.setSingleStep(5_000)
        self.scatter_n.setToolTip("3B scatter grafikde kullanilacak nokta sayisi")

        self.no_plots = QCheckBox("Skip plots")
        self.no_plots.setChecked(False)
        self.dump_json = QCheckBox("summary.json kaydet")
        self.dump_json.setChecked(True)

        form_params.addRow("Sample Count", self.sample_n)
        form_params.addRow("Seed", self.seed)
        form_params.addRow("Min Altitude (km)", self.alt_min_km)
        form_params.addRow("Max Altitude (km)", self.alt_max_km)
        form_params.addRow("Scatter Point Count", self.scatter_n)
        form_params.addRow(self.no_plots)
        form_params.addRow(self.dump_json)
        grp_params.setLayout(form_params)

        # --- Output ---
        grp_out = QGroupBox("Output")
        form_out = QFormLayout()
        _tune_form(form_out)

        self.out_dir = ValidatedPathEdit(
            placeholder=f"Empty -> {DATASET_REPORTS_OUTPUT_ROOT}/<dataset>_<timestamp>", check_file=False
        )
        btn_out = QPushButton("Choose...")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)
        form_out.addRow("Output Folder", out_row)
        grp_out.setLayout(form_out)

        self.extra_args = QLineEdit("")
        self.extra_args.setPlaceholderText("Ek CLI argumanlari")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(input_card, 0, 0, 1, 2)
        grid.addWidget(grp_params, 1, 0)
        grid.addWidget(grp_out, 1, 1)
        extra_f = QFormLayout()
        _tune_form(extra_f)
        extra_f.addRow("Ek CLI", self.extra_args)
        extra_w = QWidget()
        extra_w.setLayout(extra_f)
        grid.addWidget(extra_w, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for g in (grp_params, grp_out):
            _tune_inputs(g)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Start Analysis")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_progress_parser(self._parse_progress)
        self.runner.set_finished_hook(self._on_finished)
        self._gallery = ImageGallery()

        top = QWidget()
        top_l = QVBoxLayout()
        top_l.setContentsMargins(8, 8, 8, 8)
        top_l.addLayout(grid)
        top.setLayout(top_l)

        # Phase 8: Tabbed process log and visual analysis to prevent thumbnail compression
        bottom_tabs = QTabWidget()
        bottom_tabs.setDocumentMode(True)
        bottom_tabs.addTab(self.runner.raw_log_widget(), "Analysis Output Log")
        bottom_tabs.addTab(self._gallery, "Visual Gravity Cloud Plots")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(_scroll_wrap(top))
        splitter.addWidget(bottom_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 620])

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)
        self._effective_out_dir = ""
        self._restore_settings()

    # --- file pickers ---

    def _pick_input(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Dataset",
            self.input_file.text() or str(SCRIPT_DIR),
            "HDF5 (*.h5 *.hdf5);;PT (*.pt);;All (*.*)",
        )
        if fn:
            self.input_file.setText(_norm_path(fn))

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Output Folder", self.out_dir.text() or str(DATASET_REPORTS_OUTPUT_ROOT)
        )
        if d:
            self.out_dir.setText(_norm_path(d))

    # --- settings ---

    def _save_settings(self):
        s = QSettings("LunarSurrogate", "CloudAnalysis")
        s.setValue("input_file", self.input_file.text())
        s.setValue("out_dir", self.out_dir.text())
        s.setValue("sample_n", self.sample_n.value())
        s.setValue("seed", self.seed.value())
        s.setValue("scatter_n", self.scatter_n.value())
        s.setValue("alt_min_km", self.alt_min_km.value())
        s.setValue("alt_max_km", self.alt_max_km.value())
        s.setValue("no_plots", self.no_plots.isChecked())
        s.setValue("dump_json", self.dump_json.isChecked())
        s.setValue("extra_args", self.extra_args.text())

    def _restore_settings(self):
        s = QSettings("LunarSurrogate", "CloudAnalysis")
        for attr, key, cast in [
            ("input_file", "input_file", str),
            ("out_dir", "out_dir", str),
            ("extra_args", "extra_args", str),
        ]:
            v = s.value(key)
            if v is not None:
                getattr(self, attr).setText(str(v))
        for attr, key, cast, default in [
            ("sample_n", "sample_n", int, 200_000),
            ("seed", "seed", int, 123),
            ("scatter_n", "scatter_n", int, 50_000),
        ]:
            v = s.value(key)
            if v is not None:
                try:
                    getattr(self, attr).setValue(cast(v))
                except Exception:
                    pass
        for attr, key, cast, default in [
            ("alt_min_km", "alt_min_km", float, -1.0),
            ("alt_max_km", "alt_max_km", float, -1.0),
        ]:
            v = s.value(key)
            if v is not None:
                try:
                    getattr(self, attr).setValue(cast(v))
                except Exception:
                    pass
        for attr, key, default in [
            ("no_plots", "no_plots", False),
            ("dump_json", "dump_json", True),
        ]:
            v = s.value(key)
            if v is not None:
                getattr(self, attr).setChecked(str(v).lower() in ("true", "1"))

    # --- run ---

    def _build_args(self) -> list:
        script = _STLRPS_DATA_MODULE_DIR / "spatial_cloud_analysis.py"
        args = [str(script)]
        inp = self.input_file.text().strip()
        if inp:
            args.append(inp)
        args += ["--sample", str(self.sample_n.value())]
        args += ["--seed", str(self.seed.value())]
        args += ["--scatter-n", str(self.scatter_n.value())]
        if self.alt_min_km.value() >= 0.0:
            args += ["--alt-min-km", str(self.alt_min_km.value())]
        if self.alt_max_km.value() >= 0.0:
            args += ["--alt-max-km", str(self.alt_max_km.value())]
        if self.no_plots.isChecked():
            args.append("--no-plots")
        if self.dump_json.isChecked():
            args.append("--dump-json")
        out = self.out_dir.text().strip()
        if out:
            args += ["--outdir", out]
        extra = self.extra_args.text().strip()
        if extra:
            args += extra.split()
        return args

    def _start(self):
        inp = self.input_file.text().strip()
        if not inp:
            QMessageBox.warning(self, "Missing Input", "Choose a dataset first.")
            return
        if not Path(inp).is_file():
            QMessageBox.warning(self, "File Not Found", f"File does not exist:\n{inp}")
            return
        if not self.out_dir.text().strip():
            self.out_dir.setText(str(_default_dataset_report_dir(inp)))
        self._effective_out_dir = ""
        self._gallery.clear_gallery()
        self._save_settings()
        args = self._build_args()
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _on_finished(self, exit_code: int):
        if exit_code == 0 and self._effective_out_dir:
            self._gallery.load_from_directory(self._effective_out_dir)
        elif exit_code == 0:
            # Try to find output next to input file
            inp = self.input_file.text().strip()
            if inp:
                candidate = _default_dataset_report_dir(inp)
                if candidate.is_dir():
                    self._gallery.load_from_directory(str(candidate))

    def _parse_progress(self, line: str):
        if not self._effective_out_dir:
            m = re.search(r"(?:outdir|Output dir|Saving to|Writing to|saved to)\s*[:=]\s*(.+)", line, re.IGNORECASE)
            if m:
                c = m.group(1).strip().strip("'\"")
                if Path(c).is_dir():
                    self._effective_out_dir = c
                    self.runner.set_output_dir(c)


class DatasetInspectionPanel(QWidget):
    """Dataset readiness panel: pick an HDF5 dataset, inspect metadata, validate."""

    send_to_training = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.path_edit = ValidatedPathEdit(
            placeholder="Paste a dataset path or choose one above", check_file=True
        )
        self.path_edit.setVisible(False)
        self.path_edit.textChanged.connect(self._on_path_text_changed)
        self._selected_file = _compact_path_label("No dataset selected")

        btn_browse = QPushButton("Choose Dataset")
        btn_browse.clicked.connect(self._pick)
        self.btn_validate = QPushButton("Validate Dataset")
        self.btn_validate.setProperty("kind", "ghost")
        self.btn_validate.clicked.connect(self._validate)
        self.btn_send = QPushButton("Send to Training")
        self.btn_send.setProperty("kind", "ghost")
        self.btn_send.clicked.connect(self._send)
        btn_path = QPushButton("Path")
        btn_path.setCheckable(True)
        btn_path.toggled.connect(self.path_edit.setVisible)

        file_detail = QWidget()
        file_detail_l = QVBoxLayout(file_detail)
        file_detail_l.setContentsMargins(0, 0, 0, 0)
        file_detail_l.setSpacing(8)
        file_detail_l.addWidget(self._selected_file)
        file_detail_l.addWidget(self.path_edit)

        action_card = _data_action_card(
            "Inspect Dataset",
            "Choose an HDF5 cloud, validate it, then send it to Training.",
            btn_browse,
            secondary_buttons=[self.btn_validate, self.btn_send, btn_path],
            detail=file_detail,
        )

        self.status_label = QLabel("UNKNOWN")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_status("unknown", "Select a dataset to inspect metadata.")

        # Metadata summary card
        self._summary = QLabel("Metadata appears after validation.")
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._summary.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._summary.setStyleSheet(
            "background: rgba(13, 22, 38, 0.72); border: 1px solid rgba(185, 194, 221, 0.12);"
            " border-radius: 10px; padding: 12px; color: #cdd9ee; font-size: 12px;"
        )
        self._summary.setMinimumHeight(140)

        # Raw metadata panel
        self._raw = QPlainTextEdit()
        _style_command_preview(self._raw, min_h=140)
        self._raw.setPlaceholderText("Raw metadata/attributes will appear here.")

        meta_tabs = QTabWidget()
        meta_tabs.setDocumentMode(True)
        meta_tabs.addTab(self._summary, "Summary")
        meta_tabs.addTab(self._raw, "Raw")

        lo = QVBoxLayout()
        lo.setContentsMargins(8, 8, 8, 8)
        lo.setSpacing(12)
        lo.addWidget(action_card)
        lo.addWidget(self.status_label)
        lo.addWidget(meta_tabs, 1)
        self.setLayout(lo)

    # -- helpers --
    def _on_path_text_changed(self, text: str) -> None:
        _set_path_label(
            self._selected_file,
            text,
            empty_text="No dataset selected",
        )

    def _pick(self) -> None:
        start = self.path_edit.text().strip() or str(_REPO_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dataset", start, "HDF5 (*.h5 *.hdf5);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)
            self._validate()

    def _set_status(self, level: str, text: str) -> None:
        colors = {
            "ready":   ("#2dd4bf", "rgba(45, 212, 191, 0.12)", "Ready"),
            "warning": ("#f6c177", "rgba(246, 193, 119, 0.12)", "Warning"),
            "error":   ("#ff6b7a", "rgba(255, 107, 122, 0.14)", "Error"),
            "unknown": ("#7f91ac", "rgba(127, 145, 172, 0.12)", "Unknown"),
        }
        color, bg, label = colors.get(level, colors["unknown"])
        self.status_label.setText(f"{label}: {text}")
        self.status_label.setStyleSheet(
            f"color: {color}; background: {bg}; border: 1px solid {color};"
            " border-radius: 8px; padding: 6px 10px; font-weight: 600; font-size: 12px;"
        )

    def _send(self) -> None:
        path = self.path_edit.text().strip()
        if path and Path(path).exists():
            self.send_to_training.emit(path)
        else:
            QMessageBox.information(self, "No dataset", "Select a valid dataset file first.")

    def _validate(self) -> None:
        path = self.path_edit.text().strip()
        if not path or not Path(path).exists():
            self._set_status("error", "File not found.")
            self._summary.setText("Select a dataset to inspect metadata.")
            self._raw.clear()
            return
        if not _HAS_H5PY:
            self._set_status(
                "unknown",
                "h5py is not installed; metadata preview is unavailable.",
            )
            return
        info = _introspect_h5(path)
        if info is None:
            self._set_status("error", "Could not read the HDF5 file.")
            self._summary.setText("Could not read the HDF5 file.")
            self._raw.clear()
            return

        attrs = info.get("attrs", {})
        rows = info.get("rows")
        unit_system = _attr_lookup(attrs, "unit_system", "units")
        degree_max = _attr_lookup(attrs, "degree_max", "requested_degree", "max_degree")
        degree_min = _attr_lookup(attrs, "degree_min", "min_degree")
        alt_min = _attr_lookup(attrs, "alt_min_km", "altitude_min_km", "alt_min")
        alt_max = _attr_lookup(attrs, "alt_max_km", "altitude_max_km", "alt_max")

        fields = [
            ("Rows", f"{rows:,}" if isinstance(rows, int) else rows),
            ("Columns", info.get("cols")),
            ("Dataset", info.get("dataset_name")),
            ("Units", unit_system),
            ("Degree", f"{degree_min} -> {degree_max}" if degree_min is not None or degree_max is not None else None),
            ("Altitude", f"{alt_min} -> {alt_max} km" if alt_min is not None or alt_max is not None else None),
            ("Gravity model", _attr_lookup(attrs, "gravity_model_path", "gfc_path", "gravity_model")),
        ]
        html_rows = []
        for label, value in fields:
            shown = "-" if value is None else str(value)
            html_rows.append(
                f"<tr><td style='color:#7f91ac;padding:2px 14px 2px 0;'>{label}</td>"
                f"<td style='color:#e6edf7;font-family:Consolas,monospace;'>{shown}</td></tr>"
            )
        self._summary.setText("<table>" + "".join(html_rows) + "</table>")

        import json as _json
        try:
            self._raw.setPlainText(_json.dumps(attrs, indent=2, default=str))
        except Exception:
            self._raw.setPlainText(str(attrs))

        # Validation verdict
        if not isinstance(rows, int) or rows <= 0:
            self._set_status("error", "Dataset has no rows.")
        elif unit_system and (degree_max is not None):
            self._set_status(
                "ready",
                "Metadata present. Backend performs full convention checks at launch.",
            )
        else:
            self._set_status(
                "warning",
                "Some expected metadata is missing; backend will re-validate at launch.",
            )


class DataPage(QWidget):
    """Data workspace: dataset readiness, generation, and analysis."""

    def __init__(self, cloud_tab: QWidget, analysis_tab: QWidget,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.inspect_panel = DatasetInspectionPanel()
        self._stack = QStackedWidget()
        # Wrap DatasetInspectionPanel in scroll area to prevent metadata and log clipping
        self._stack.addWidget(_scroll_wrap(self.inspect_panel))
        self._stack.addWidget(cloud_tab)
        self._stack.addWidget(analysis_tab)
        self._section_buttons: List[QPushButton] = []

        nav = QFrame()
        nav.setObjectName("dataSectionNav")
        nav.setMaximumWidth(260)
        nav.setStyleSheet(
            "QFrame#dataSectionNav {"
            "  background: rgba(8, 13, 26, 0.74);"
            "  border: 1px solid rgba(185, 194, 221, 0.12);"
            "  border-radius: 14px;"
            "}"
        )
        nav_l = QVBoxLayout()
        nav_l.setContentsMargins(12, 12, 12, 12)
        nav_l.setSpacing(8)

        def _nav_btn(label: str, hint: str, idx: int) -> QPushButton:
            btn = QPushButton(label)
            btn.setToolTip(hint)
            btn.setCheckable(True)
            btn.setMinimumHeight(46)
            btn.setStyleSheet(
                "QPushButton {"
                "  text-align: left; padding: 0 14px;"
                "  border: 1px solid rgba(185, 194, 221, 0.10);"
                "  border-radius: 10px; background: rgba(255,255,255,0.025);"
                "  color: #a8b5d0; font-weight: 750; font-size: 13px;"
                "}"
                "QPushButton:hover { background: rgba(53, 208, 255, 0.06); color: #e8ecf8; }"
                "QPushButton:checked {"
                "  background: rgba(53, 208, 255, 0.12);"
                "  border-color: rgba(53, 208, 255, 0.35);"
                "  color: #f2f8ff;"
                "}"
            )
            btn.clicked.connect(lambda _c=False, i=idx: self._show_section(i))
            self._section_buttons.append(btn)
            return btn

        nav_title = QLabel("Data")
        nav_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #e8ecf8;")
        nav_l.addWidget(nav_title)
        nav_l.addWidget(_nav_btn("Inspect", "Readiness and metadata", 0))
        nav_l.addWidget(_nav_btn("Generate", "Single cloud or train/val/test/OOD suite", 1))
        nav_l.addWidget(_nav_btn("Analyze", "Coverage and field reports", 2))
        nav_l.addStretch(1)
        nav.setLayout(nav_l)
        _style_surface(nav, object_name="dataSectionNav")

        workspace = QHBoxLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(14)
        workspace.addWidget(nav)
        workspace.addWidget(self._stack, 1)

        lo = QVBoxLayout()
        lo.setContentsMargins(22, 20, 22, 20)
        lo.setSpacing(14)
        lo.addWidget(_make_page_header(
            "Data Workspace",
            "Choose a data task, then act from the primary card on that page.",
            "Dataset Pipeline",
        ))
        lo.addLayout(workspace, 1)
        self.setLayout(lo)
        self._show_section(0)

    def _show_section(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._section_buttons):
            btn.setChecked(i == idx)
