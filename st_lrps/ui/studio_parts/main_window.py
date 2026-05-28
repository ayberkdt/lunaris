# -*- coding: utf-8 -*-
"""
st_lrps.ui.studio  -  v3

PyQt6 dashboard for the lunar scalar potential surrogate codebase.

The model predicts residual potential dU(x); residual acceleration da is
computed from the gradient of that scalar field. ST-LRPS is a Sobolev-trained
lunar residual potential surrogate, not a classical q,p state-space model.

What you can do from the UI
---------------------------
- Train a potential surrogate   (runs `python -m st_lrps.training.cli` as a subprocess)
- Resume an interrupted training run
- Evaluate a surrogate run      (runs `python -m st_lrps.evaluation.cli` as a subprocess)
- Profile ST-LRPS runtime inference (runs `python -m st_lrps.runtime.profiling`)
- Browse evaluation plots inline (post-processing dashboard)
- Inspect runtime profiling summaries and plots
- Watch live loss curves during training (pyqtgraph)
- Queue multiple training runs for overnight execution

UX Architecture (v1 → retained)
---------------------------------
1–6.  Groups, Grid, Tooltips, Collapsible, QSettings, Path validation.

UX Architecture (v2 → retained)
---------------------------------
7–12. Image gallery, Log highlight, Auto-scroll, Presets, Post-run, Dependent params.

UX Architecture (v3 → new)
-----------------------------
13. Live Loss Plotting: real-time pyqtgraph chart of train/val loss parsed from logs.
14. Dataset Introspection: auto-read HDF5 metadata (row count, attrs) on path selection.
15. Training Queue: enqueue multiple configs and run them sequentially overnight.

Run
---
  python -m st_lrps.ui.studio
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys

from .qt_common import *
from .qt_common import _USE_PYSIDE

from .data_pages import CloudGenTab, CloudAnalysisTab, DataPage
from .training_pages import STLRPSTrainTab
from .evaluation_pages import STLRPSEvalTab, EvaluationPage
from .runtime_pages import STLRPSProfilingTab, RuntimePerformancePage
from .orbit_benchmark_pages import (
    OrbitBenchmarkTab, OrbitBenchmarkPage,
    OrbitBenchmarkPlotsTab, OrbitBenchmarkPlotsPage,
)


# pyqtgraph — optional, graceful fallback
try:
    import pyqtgraph as pg

    _HAS_PYQTGRAPH = True
except ImportError:
    _HAS_PYQTGRAPH = False

# h5py — optional, for dataset introspection
try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    from st_lrps.artifacts.manager import (
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
    from st_lrps.ui.dashboard_widgets import (
        ExperimentHeader,
        KPIStrip,
        MetricCard,
        StructuredLogView,
        TimeMetricsStrip,
    )
    from st_lrps.ui.training_metrics import (
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
_REPO_ROOT = SCRIPT_DIR.parents[2]
TRAIN_CLI_MODULE = "st_lrps.training.cli"
EVAL_CLI_MODULE = "st_lrps.evaluation.cli"

# Short, header-friendly labels for the model-representation presets.
_PRESET_SHORT = {
    "baseline_raw": "baseline",
    "recommended_physical_radial_decay": "phys-radial",
    "ablation_radial_separation": "abl:radial-sep",
    "ablation_radial_decay_scaled": "abl:radial-decay",
    "ablation_real_sh_low_degree": "abl:real-sh",
    "custom": "custom",
}
PROFILE_CLI_MODULE = "st_lrps.runtime.profiling"
# Filesystem locations are used only for preflight existence checks; launching
# always goes through ``-m`` so package-relative imports resolve correctly.
TRAIN_CLI_PATH = _REPO_ROOT / "st_lrps" / "training" / "cli.py"
EVAL_CLI_PATH = _REPO_ROOT / "st_lrps" / "evaluation" / "cli.py"
PROFILE_CLI_PATH = _REPO_ROOT / "st_lrps" / "runtime" / "profiling.py"

OUTPUT_ROOT = _REPO_ROOT / "outputs"
TRAINING_OUTPUT_ROOT = OUTPUT_ROOT / "training"
DATASET_REPORTS_OUTPUT_ROOT = OUTPUT_ROOT / "dataset_reports"
RUNTIME_PERFORMANCE_OUTPUT_ROOT = OUTPUT_ROOT / "runtime_performance"
EVALUATION_OUTPUT_ROOT = OUTPUT_ROOT / "evaluations"
DATASET_SUITE_OUTPUT_ROOT = OUTPUT_ROOT / "datasets" / "cloud_suites"

# UI defaults are intentionally read from the generator configuration module.
# This keeps the dashboard from drifting away from the command-line SSOT when
# dataset-suite sizes, altitude ranges, seeds, or sampling knobs are tuned.
try:
    from st_lrps.data.spatial_cloud_parameters import (
        DEFAULT_CLOUD_SUITE_CONFIG,
        DEFAULT_SPATIAL_CLOUD_CONFIG,
        SUITE_PRESETS,
    )
except Exception:  # pragma: no cover - UI remains usable without generator deps
    DEFAULT_CLOUD_SUITE_CONFIG = None  # type: ignore[assignment]
    DEFAULT_SPATIAL_CLOUD_CONFIG = None  # type: ignore[assignment]
    SUITE_PRESETS = {}  # type: ignore[assignment]


from .common_widgets import *
from .common_widgets import _tune_form, _tune_inputs, _row_lineedit_with_button, _scroll_wrap, _settings, _read_json_if_exists, _split_cli_args, _format_command, _send_os_notification, _apply_status_tips, _cfg_value, _norm_path, _timestamp_slug, _safe_slug, _default_training_output_dir, _default_runtime_output_dir, _default_dataset_report_dir, _output_standard_text, _mono_font, _inspect_run_artifacts, _NoWheelOnSpinFilter


from .data_pages import *
from .data_pages import _introspect_h5


def _attr_lookup(attrs: Dict[str, Any], *keys: str) -> Any:
    """Return the first present attribute among ``keys`` (case-insensitive)."""
    if not isinstance(attrs, dict):
        return None
    lower = {str(k).lower(): v for k, v in attrs.items()}
    for k in keys:
        if k in attrs:
            return attrs[k]
        if k.lower() in lower:
            return lower[k.lower()]
    return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ST-LRPS Studio")
        self.resize(1320, 860)
        self.setMinimumSize(1024, 680)

        # --- Underlying workflow widgets (logic preserved, pages re-homed) ---
        self._cloud_tab    = CloudGenTab()
        self._train_tab    = STLRPSTrainTab()
        self._profile_tab  = STLRPSProfilingTab()
        self._eval_tab     = STLRPSEvalTab()
        self._analysis_tab = CloudAnalysisTab()
        self._orbit_benchmark_tab = OrbitBenchmarkTab()
        self._orbit_plots_tab = OrbitBenchmarkPlotsTab()

        self._cloud_tab.set_train_tab(self._train_tab)
        self._cloud_tab.cloud_params_changed.connect(self._train_tab.sync_from_cloud)

        # --- Top-level workspace pages ---
        self._data_page = DataPage(self._cloud_tab, self._analysis_tab)
        self._train_setup_page = self._train_tab.setup_page
        self._train_monitor_page = self._train_tab.monitor_page
        self._eval_page = EvaluationPage(self._eval_tab)
        self._runtime_page = RuntimePerformancePage(self._profile_tab)
        self._orbit_benchmark_page = OrbitBenchmarkPage(self._orbit_benchmark_tab)
        self._orbit_plots_page = OrbitBenchmarkPlotsPage(self._orbit_plots_tab)
        self._data_page.inspect_panel.send_to_training.connect(self._on_dataset_to_training)
        self._train_tab.navigate_monitor_requested.connect(lambda: self._navigate(2))

        self._stack = QStackedWidget()
        self._stack.addWidget(self._data_page)              # index 0: Data
        self._stack.addWidget(self._train_setup_page)       # index 1: Training Setup
        self._stack.addWidget(self._train_monitor_page)     # index 2: Training Monitor
        self._stack.addWidget(self._eval_page)              # index 3: Evaluation
        self._stack.addWidget(self._runtime_page)           # index 4: Runtime Performance
        self._stack.addWidget(self._orbit_benchmark_page)   # index 5: Orbit-Level Benchmark
        self._stack.addWidget(self._orbit_plots_page)       # index 6: Gravity Plots
        self._page_titles = [
            "Data",
            "Training Setup",
            "Training Monitor",
            "Evaluation",
            "Runtime Performance",
            "Orbit-Level Benchmark",
            "Gravity Plots",
        ]

        dep_info = []
        if not _HAS_PYQTGRAPH:
            dep_info.append("pyqtgraph yüklü değil (canlı grafik devre dışı)")
        if not _HAS_H5PY:
            dep_info.append("h5py yüklü değil (dataset ön izleme devre dışı)")

        # --- Header card (Phase 2: professional experiment header) ---
        if _HAS_DASHBOARD_V2:
            self._experiment_header = ExperimentHeader()
            header_card = self._experiment_header
            # Detect device
            try:
                import torch
                if torch.cuda.is_available():
                    dev_name = torch.cuda.get_device_name(0)
                    mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                    self._experiment_header.set_device(f"CUDA \u00b7 {mem_total:.1f} GB")
                else:
                    self._experiment_header.set_device("CPU")
            except Exception:
                self._experiment_header.set_device("CPU")
        else:
            header_card = QFrame()
            header_card.setObjectName("appHeaderCard")
            header_card.setStyleSheet(
                "QFrame#appHeaderCard {"
                "  background: #101A2B;"
                "  border: 1px solid #26364F;"
                "  border-radius: 10px;"
                "}"
            )
            header_lo = QHBoxLayout()
            header_lo.setContentsMargins(18, 10, 18, 10)
            header_lo.setSpacing(16)

            title_col = QVBoxLayout()
            title_col.setContentsMargins(0, 0, 0, 0)
            title_col.setSpacing(3)
            lbl_title = QLabel("ST-LRPS Studio")
            lbl_title.setStyleSheet(
                "color: #e8ecf8; font-size: 15px; font-weight: 700;"
                " letter-spacing: 0.3px; background: transparent; border: none;"
            )
            lbl_subtitle = QLabel(
                "Lunar residual-potential surrogate training and evaluation"
            )
            lbl_subtitle.setStyleSheet(
                "color: #8892b0; font-size: 12px; background: transparent; border: none;"
            )
            title_col.addWidget(lbl_title)
            title_col.addWidget(lbl_subtitle)
            header_lo.addLayout(title_col, 1)
            header_card.setLayout(header_lo)


        # The train tab is not in the widget tree (only its pages are), so give
        # it a direct reference to the header for lifecycle/ETA updates.
        if hasattr(self._train_tab, "set_experiment_header"):
            self._train_tab.set_experiment_header(getattr(self, "_experiment_header", None))

        # --- Sidebar navigation ---
        self._nav_buttons: List[QPushButton] = []
        sidebar = self._build_sidebar()

        # --- Main content area: sidebar + page stack ---
        content_area = QWidget()
        content_lo = QHBoxLayout()
        content_lo.setContentsMargins(0, 0, 0, 0)
        content_lo.setSpacing(10)
        content_lo.addWidget(sidebar)
        content_lo.addWidget(self._stack, 1)
        content_area.setLayout(content_lo)

        root = QWidget()
        root_lo = QVBoxLayout()
        root_lo.setContentsMargins(12, 10, 12, 10)
        root_lo.setSpacing(10)
        root_lo.addWidget(header_card)
        root_lo.addWidget(content_area, 1)
        root.setLayout(root_lo)
        self.setCentralWidget(root)

        # --- Status bar: parametre açıklamaları hover'da gösterilir ---
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.setStyleSheet(
            "QStatusBar {"
            "  background: rgba(6, 9, 18, 0.95);"
            "  border-top: 1px solid rgba(185, 194, 221, 0.09);"
            "  color: #5a6480; font-size: 11px; padding: 0 12px;"
            "  min-height: 22px;"
            "}"
            "QStatusBar::item { border: none; }"
        )
        sb.showMessage("Bir parametrenin üzerine gelin — açıklama burada görünür.")

        # Tüm sayfalardaki input widget'larının tooltip'ini status bar'a bağla
        for tab in (
            self._cloud_tab, self._analysis_tab,
            self._train_tab, self._profile_tab, self._eval_tab,
            self._orbit_benchmark_tab, self._orbit_plots_tab,
        ):
            _apply_status_tips(tab)

        # --- Header context badges: keep preset/dataset in sync while idle ---
        hdr = getattr(self, "_experiment_header", None)
        if hdr is not None and hasattr(hdr, "set_preset"):
            def _sync_preset():
                _p = self._train_tab.model_preset.currentData() or "custom"
                hdr.set_preset(_PRESET_SHORT.get(_p, _p))
            self._train_tab.model_preset.currentIndexChanged.connect(lambda *_: _sync_preset())
            _sync_preset()
            self._train_tab.data.textChanged.connect(
                lambda *_: hdr.set_dataset(Path(self._train_tab.data.text().strip()).name
                                           if self._train_tab.data.text().strip() else "—")
            )

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("navSidebar")
        sidebar.setFixedWidth(238)
        sidebar.setStyleSheet(
            "QFrame#navSidebar {"
            "  background: #0b1220;"
            "  border: none;"
            "  border-right: 1px solid rgba(185, 194, 221, 0.12);"
            "  border-radius: 0;"
            "}"
        )

        _NAV_BTN_STYLE = (
            "QPushButton {"
            "  text-align: left; padding: 12px 14px 12px 18px;"
            "  border: none; border-left: 3px solid transparent;"
            "  border-radius: 0; font-size: 13px; font-weight: 600;"
            "  min-height: 40px;"
            "  color: #8a98b8; background: transparent;"
            "}"
            "QPushButton:hover {"
            "  color: #d7e1f7; background: rgba(53, 208, 255, 0.06);"
            "}"
            "QPushButton:checked {"
            "  color: #f2f6ff; font-weight: 700;"
            "  background: rgba(53, 208, 255, 0.12);"
            "  border-left: 3px solid rgba(53, 208, 255, 0.85);"
            "}"
        )

        def _section_lbl(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "color: rgba(185, 194, 221, 0.32); font-size: 10px; font-weight: 700;"
                " letter-spacing: 1.8px; padding: 12px 12px 4px 16px;"
                " background: transparent; border: none;"
            )
            return lbl

        def _nav_btn(label: str, page_idx: int) -> QPushButton:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(_NAV_BTN_STYLE)
            btn.clicked.connect(lambda _c, i=page_idx: self._navigate(i))
            self._nav_buttons.append(btn)
            return btn

        def _group_box() -> "QFrame":
            box = QFrame()
            box.setObjectName("navGroup")
            box.setStyleSheet(
                "QFrame#navGroup {"
                "  background: rgba(53, 208, 255, 0.045);"
                "  border: 1px solid rgba(53, 208, 255, 0.16);"
                "  border-radius: 10px;"
                "}"
            )
            gl = QVBoxLayout(box)
            gl.setContentsMargins(4, 4, 4, 4)
            gl.setSpacing(2)
            return box, gl

        lo = QVBoxLayout()
        lo.setContentsMargins(10, 16, 10, 16)
        lo.setSpacing(4)

        # ── DATA ──
        lo.addWidget(_section_lbl("DATA"))
        lo.addWidget(_nav_btn("Data", 0))

        # ── TRAINING (Setup + Monitor are one category, boxed together) ──
        lo.addWidget(_section_lbl("TRAINING"))
        train_box, train_l = _group_box()
        train_l.addWidget(_nav_btn("Training Setup", 1))
        train_l.addWidget(_nav_btn("Training Monitor", 2))
        lo.addWidget(train_box)

        # ── ANALYSIS ──
        lo.addWidget(_section_lbl("ANALYSIS"))
        analysis_box, analysis_l = _group_box()
        analysis_l.addWidget(_nav_btn("Evaluation", 3))
        analysis_l.addWidget(_nav_btn("Runtime Performance", 4))
        analysis_l.addWidget(_nav_btn("Orbit-Level Benchmark", 5))
        analysis_l.addWidget(_nav_btn("Gravity Plots", 6))
        lo.addWidget(analysis_box)

        lo.addStretch(1)
        sidebar.setLayout(lo)

        self._navigate(0)
        return sidebar

    def _navigate(self, page_idx: int) -> None:
        self._stack.setCurrentIndex(page_idx)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == page_idx)
        # Reflect the active page in the header.
        hdr = getattr(self, "_experiment_header", None)
        if hdr is not None and hasattr(hdr, "set_page"):
            titles = getattr(self, "_page_titles", [])
            if 0 <= page_idx < len(titles):
                hdr.set_page(titles[page_idx])
                
        # Phase 10: Dynamically manage badges visibility on small screen scopes
        if hdr is not None:
            # Hide Preset and Dataset badges on pages where they aren't relevant to save header space
            has_preset = hasattr(hdr, "_preset")
            has_dataset = hasattr(hdr, "_dataset")
            if has_preset and has_dataset:
                is_train = page_idx in (1, 2)
                hdr._preset.setVisible(is_train)
                hdr._dataset.setVisible(is_train)

    def _on_dataset_to_training(self, path: str) -> None:
        """Data page → Training: load the chosen dataset and switch pages."""
        try:
            idx = self._train_tab.dataset_mode.findData("single")
            if idx >= 0:
                self._train_tab.dataset_mode.setCurrentIndex(idx)
            self._train_tab.data.setText(path)
        except Exception:
            pass
        self._navigate(1)


