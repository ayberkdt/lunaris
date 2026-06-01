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

if _HAS_PYQTGRAPH:
    class _CleanLogAxis(pg.AxisItem):
        """Log axis that shows only decade (power-of-ten) ticks.

        pyqtgraph's default log axis draws cramped sub-decade minor ticks
        (…2,3,4,…9) that overlap badly on short plots. Keeping only the major
        decade level makes the value range readable (1e-1, 1e-2, 1e-3, …)."""

        def tickValues(self, minVal, maxVal, size):
            ticks = super().tickValues(minVal, maxVal, size)
            if getattr(self, "logMode", False) and len(ticks) > 1:
                return ticks[:1]
            return ticks

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


def _cfg_value(cfg: Any, name: str, fallback: Any) -> Any:
    """Read a default from the SSOT config object with a safe UI fallback."""

    return getattr(cfg, name, fallback) if cfg is not None else fallback


def _norm_path(p: str) -> str:
    return str(Path(p).expanduser().resolve()) if p else ""


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_slug(text: str, fallback: str = "run") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", (text or "").strip()).strip("._-")
    return slug[:64] if slug else fallback


def _default_training_output_dir() -> Path:
    return TRAINING_OUTPUT_ROOT / f"st_lrps_train_{_timestamp_slug()}"


def _default_runtime_output_dir(model_dir: str = "") -> Path:
    model_name = _safe_slug(Path(model_dir).stem if model_dir else "st_lrps_runtime", "st_lrps_runtime")
    return RUNTIME_PERFORMANCE_OUTPUT_ROOT / f"{model_name}_{_timestamp_slug()}"


def _default_dataset_report_dir(dataset_path: str = "") -> Path:
    dataset_name = _safe_slug(Path(dataset_path).stem if dataset_path else "dataset", "dataset")
    return DATASET_REPORTS_OUTPUT_ROOT / f"{dataset_name}_{_timestamp_slug()}"


def _output_standard_text() -> str:
    return (
        "Generated-output standard:\n"
        f"  training runs       -> {TRAINING_OUTPUT_ROOT}\n"
        "  run-local evals     -> <training-run>/evals/\n"
        f"  standalone evals    -> {EVALUATION_OUTPUT_ROOT}\n"
        f"  runtime profiles    -> {RUNTIME_PERFORMANCE_OUTPUT_ROOT}\n"
        f"  dataset reports     -> {DATASET_REPORTS_OUTPUT_ROOT}\n"
        f"  dataset suites      -> {DATASET_SUITE_OUTPUT_ROOT}"
    )


def _mono_font() -> QFont:
    f = QFont("Consolas")
    if not f.exactMatch():
        f = QFont("Courier New")
    f.setPointSize(10)
    return f


def _tune_form(form: QFormLayout) -> None:
    form.setContentsMargins(14, 12, 14, 12)
    form.setHorizontalSpacing(14)
    form.setVerticalSpacing(10)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)


def _tune_inputs(root: QWidget, h: int = 38) -> None:
    # PySide6 does not support passing a tuple of types to findChildren and prints
    # a warning (FIXME qt_isinstance...) to standard error if attempted.
    if _USE_PYSIDE:
        inputs = []
        for cls in (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox):
            inputs.extend(root.findChildren(cls))
    else:
        try:
            inputs = root.findChildren((QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox))
        except TypeError:
            inputs = []
            for cls in (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox):
                inputs.extend(root.findChildren(cls))

    for w in inputs:
        w.setMinimumHeight(h)
        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    if _USE_PYSIDE:
        spinboxes = []
        for cls in (QSpinBox, QDoubleSpinBox):
            spinboxes.extend(root.findChildren(cls))
    else:
        try:
            spinboxes = root.findChildren((QSpinBox, QDoubleSpinBox))
        except TypeError:
            spinboxes = []
            for cls in (QSpinBox, QDoubleSpinBox):
                spinboxes.extend(root.findChildren(cls))

    for sb in spinboxes:
        sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        sb.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)


def _row_lineedit_with_button(edit: QLineEdit, button: QPushButton) -> QWidget:
    edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setMinimumHeight(edit.minimumHeight())
    wrap = QWidget()
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)
    h.addWidget(edit, 1)
    h.addWidget(button, 0)
    wrap.setLayout(h)
    return wrap


def _scroll_wrap(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setWidget(widget)
    return area


_SETTINGS_ORG = "ST_LRPS_Project"
_SETTINGS_APP = "ST_LRPS_Dashboard"

def _settings() -> QSettings:
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _split_cli_args(text: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Split an advanced CLI text field exactly like a shell would."""
    if not text.strip():
        return [], None
    try:
        return shlex.split(text, posix=(os.name != "nt")), None
    except ValueError as exc:
        return None, str(exc)


def _format_command(program: str, args: List[str]) -> str:
    """Return a copy/paste friendly command line for the generated subprocess."""
    return subprocess.list2cmdline([program] + args)


def _send_os_notification(title: str, message: str) -> None:
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "{title}"',
                ]
            )
        elif system == "Linux":
            subprocess.Popen(["notify-send", title, message])
    except Exception:
        pass


def _apply_status_tips(root: QWidget) -> None:
    """Copy each input widget's toolTip() to statusTip() so the status bar shows it on hover.

    Qt's built-in StatusTip mechanism bubbles the event up to QMainWindow's status bar
    automatically — no signal connections needed.
    """
    for cls in (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox):
        for w in root.findChildren(cls):
            tip = w.toolTip()
            if tip and not w.statusTip():
                w.setStatusTip(tip.replace("\n", "  ·  "))


class _NoWheelOnSpinFilter(QObject):
    """App-level event filter: prevents accidental spinbox value changes via scroll wheel."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.Wheel
            and isinstance(obj, (QSpinBox, QDoubleSpinBox))
        ):
            event.ignore()
            return True
        return False


class ValidatedPathEdit(QLineEdit):
    """QLineEdit with live file/dir validation and optional introspection signal."""

    path_validated = pyqtSignal(str, bool)  # (path, exists)

    _STYLE_VALID = "border: 1px solid rgba(52, 211, 153, 0.7);"
    _STYLE_INVALID = "border: 1px solid rgba(248, 113, 113, 0.75); background-color: rgba(248, 113, 113, 0.08);"
    _STYLE_NEUTRAL = ""

    def __init__(
        self,
        placeholder: str = "",
        check_file: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._check_file = check_file
        if placeholder:
            self.setPlaceholderText(placeholder)
        self.textChanged.connect(self._validate)

    def _validate(self, text: str) -> None:
        path_str = text.strip()
        if not path_str:
            self.setStyleSheet(self._STYLE_NEUTRAL)
            self.path_validated.emit("", False)
            return
        p = Path(path_str)
        exists = p.is_file() if self._check_file else p.is_dir()
        self.setStyleSheet(self._STYLE_VALID if exists else self._STYLE_INVALID)
        self.path_validated.emit(path_str, exists)


class CollapsibleSection(QWidget):
    def __init__(
        self, title: str = "Advanced Settings", parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._title = title
        self._toggle_btn = QPushButton(f"▸  {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setProperty("kind", "ghost")
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 8px 14px; font-weight: 600; "
            "color: #8fb9d4; border: 1px solid transparent; border-radius: 8px; "
            "background: rgba(53, 208, 255, 0.04); }"
            "QPushButton:hover { color: #d7e1f7; background: rgba(53, 208, 255, 0.08); "
            "border-color: rgba(53, 208, 255, 0.18); }"
            "QPushButton:checked { color: #e8ecf8; background: rgba(53, 208, 255, 0.10); "
            "border-color: rgba(53, 208, 255, 0.22); }"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        self._content = QWidget()
        self._content.setVisible(False)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle_btn)
        layout.addWidget(self._content)
        self.setLayout(layout)

    def set_content_layout(self, content_layout) -> None:
        self._content.setLayout(content_layout)

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically expand/collapse the section."""
        self._toggle_btn.setChecked(bool(expanded))
        self._on_toggle(bool(expanded))

    def is_expanded(self) -> bool:
        """True when the section is currently expanded."""
        return self._toggle_btn.isChecked()

    def _on_toggle(self, checked: bool) -> None:
        self._content.setVisible(checked)
        arrow = "▾" if checked else "▸"
        self._toggle_btn.setText(f"{arrow}  {self._title}")


class DatasetInfoLabel(QLabel):
    """
    Feature #14: Small info label that displays HDF5 introspection results
    beneath the dataset path field.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setStyleSheet(
            "QLabel { color: #7c8dc7; font-size: 11px; padding: 3px 10px;"
            " background: rgba(124, 92, 255, 0.06);"
            " border-left: 2px solid rgba(124, 92, 255, 0.35);"
            " border-radius: 0 6px 6px 0; }"
        )
        self.setVisible(False)

    def show_info(self, info: Dict[str, Any]) -> None:
        parts = []
        attrs = info.get("attrs", {}) if isinstance(info.get("attrs"), dict) else {}
        if "rows" in info:
            parts.append(f"Rows: {info['rows']:,}")
        if "cols" in info:
            parts.append(f"Cols: {info['cols']}")
        if "dtype" in info:
            parts.append(f"Dtype: {info['dtype']}")
        if info.get("is_si") is True:
            parts.append("Units: SI")
        elif info.get("is_si") is False:
            parts.append("Units: canonical")
        for key, label in (
            ("unit_system", "Unit system"),
            ("central_body", "Body"),
            ("degree_min", "deg min"),
            ("degree_max", "deg max"),
            ("requested_degree", "deg max"),
            ("alt_min_km", "alt min"),
            ("alt_max_km", "alt max"),
        ):
            if key in attrs and attrs[key] not in ("", None):
                value = attrs[key]
                suffix = " km" if key in {"alt_min_km", "alt_max_km"} else ""
                parts.append(f"{label}: {value}{suffix}")
        if "dataset_name" in info:
            parts.append(f"Dataset: '{info['dataset_name']}'")
        if parts:
            self.setText("Dataset info: " + "  |  ".join(parts))
            self.setVisible(True)
        else:
            self.setVisible(False)

    def clear_info(self) -> None:
        self.setText("")
        self.setVisible(False)


class LogHighlighter(QSyntaxHighlighter):
    def __init__(self, parent: Optional[QTextDocument] = None):
        super().__init__(parent)
        self._rules: List[Tuple[re.Pattern, QTextCharFormat]] = []

        fmt_err = QTextCharFormat()
        fmt_err.setForeground(QColor("#f87171"))
        fmt_err.setFontWeight(QFont.Weight.Bold)
        for p in [
            r"(?i)\[ERROR\]",
            r"(?i)\bError\b",
            r"(?i)\bException\b",
            r"(?i)\bTraceback\b",
            r"(?i)\bFailed\b",
            r"(?i)\bCritical\b",
        ]:
            self._rules.append((re.compile(p), fmt_err))

        fmt_warn = QTextCharFormat()
        fmt_warn.setForeground(QColor("#fbbf24"))
        for p in [
            r"(?i)\[WARNING\]",
            r"(?i)\bWarning\b",
            r"(?i)\bUserWarning\b",
            r"(?i)\bDeprecat\w*\b",
        ]:
            self._rules.append((re.compile(p), fmt_warn))

        fmt_epoch = QTextCharFormat()
        fmt_epoch.setForeground(QColor("#c084fc"))
        fmt_epoch.setFontWeight(QFont.Weight.Bold)
        self._rules.append((re.compile(r"Epoch\s*\[\s*\d+\s*/\s*\d+\s*\]"), fmt_epoch))

        fmt_metric = QTextCharFormat()
        fmt_metric.setForeground(QColor("#34d399"))
        for p in [
            r"(?:Loss|loss|RMSE|rmse|MAE|mae|R²|r2|accuracy|acc)\s*[:=]\s*[\d.eE+\-]+",
            r"(?:Val|val|Train|train)[\s_](?:Loss|loss)\s*[:=]\s*[\d.eE+\-]+",
            r"(?:lr|LR)\s*[:=]\s*[\d.eE+\-]+",
        ]:
            self._rules.append((re.compile(p), fmt_metric))

        fmt_time = QTextCharFormat()
        fmt_time.setForeground(QColor("#22d3ee"))
        for p in [
            r"[\d.]+\s*s/epoch",
            r"[\d,.]+\s*(?:pts|points|samples)/s",
            r"[\d.]+\s*(?:ms|sec|seconds|minutes|min)\b",
        ]:
            self._rules.append((re.compile(p), fmt_time))

        fmt_ui = QTextCharFormat()
        fmt_ui.setForeground(QColor("#7c8dc7"))
        self._rules.append((re.compile(r"^\[UI\].*", re.MULTILINE), fmt_ui))

    def highlightBlock(self, text: str) -> None:
        for regex, fmt in self._rules:
            for m in regex.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


class LiveLossPlot(QWidget):
    """
    Premium real-time loss dashboard using pyqtgraph.

    The widget keeps the old public API intact:
      - parse_line(line)
      - clear()
      - get_final_losses()

    Improvements over the previous compact plot:
      - card-like visual container
      - live metric chips for train/val/best/lr
      - smoother grid/axis styling
      - optional log-y toggle
      - explicit auto-fit button
      - robust duplicate-epoch handling
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        from pyqtgraph.Qt import QtCore as pg_QtCore

        self._epochs: List[int] = []
        self._train_loss: List[float] = []
        self._val_loss: List[float] = []
        self._train_opt_loss: List[float] = []
        self._train_loss_u: List[float] = []
        self._val_loss_u: List[float] = []
        self._train_loss_a: List[float] = []
        self._val_base_loss: List[float] = []
        self._train_loss_dir: List[float] = []
        self._val_dir_loss: List[float] = []
        self._val_loss_a: List[float] = []
        self._train_cos_sim: List[float] = []
        self._val_angular_mean_deg: List[float] = []
        self._val_cos_sim: List[float] = []
        self._checkpoint_scores: List[float] = []
        self._best_scores: List[float] = []
        self._lr_values: List[float] = []
        self._best_val: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._latest_epoch: Optional[int] = None
        self._latest_train_opt: Optional[float] = None
        self._latest_train_ref: Optional[float] = None
        self._latest_val_ref: Optional[float] = None
        self._latest_lam_dir: Optional[float] = None
        self._latest_checkpoint_score: Optional[float] = None
        self._best_metric_name: str = "best metric"
        self._best_formula: str = "N/A"
        self._epochs_since_improvement: Optional[int] = None
        self._checkpoint_status: str = "Waiting for training"
        self._paused: bool = False

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._card = QFrame()
        self._card.setObjectName("liveLossCard")
        self._card.setStyleSheet(
            """
            QFrame#liveLossCard {
                background-color: rgba(10, 16, 31, 0.96);
                border: 1px solid rgba(124, 92, 255, 0.30);
                border-radius: 18px;
            }
            QLabel#lossTitle {
                color: #eef2ff;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }
            QLabel#lossSubtitle {
                color: #7480a8;
                font-size: 11px;
            }
            QLabel[metric="true"] {
                color: #dbe4ff;
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(185, 194, 221, 0.13);
                border-radius: 10px;
                padding: 5px 8px;
                min-width: 82px;
                max-height: 46px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 11px;
            }
            QPushButton[plotControl="true"] {
                color: #b9c2dd;
                background-color: rgba(255, 255, 255, 0.045);
                border: 1px solid rgba(185, 194, 221, 0.16);
                border-radius: 10px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton[plotControl="true"]:hover {
                color: #ffffff;
                background-color: rgba(124, 92, 255, 0.22);
                border: 1px solid rgba(124, 92, 255, 0.55);
            }
            QCheckBox {
                color: #9aa7c7;
                font-size: 11px;
                spacing: 6px;
            }
            """
        )

        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(16, 14, 16, 16)
        card_layout.setSpacing(12)
        self._card.setLayout(card_layout)

        # ----------------------------
        # Row 1: title  |  controls
        # ----------------------------
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(3)
        title = QLabel("Live Training Monitor")
        title.setObjectName("lossTitle")
        subtitle = QLabel("Training / validation loss curve  ·  logarithmic scale recommended")
        subtitle.setObjectName("lossSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        top_row.addLayout(title_col, 1)

        self._chk_log_y = QCheckBox("Log Y")
        self._chk_log_y.setChecked(True)
        self._chk_log_y.setToolTip(
            "Shows the Y axis on a logarithmic scale.\n"
            "Recommended because loss values span several orders of magnitude."
        )
        self._chk_log_y.toggled.connect(self._on_log_toggle)
        top_row.addWidget(self._chk_log_y)

        self._chk_smooth = QCheckBox("Smooth")
        self._chk_smooth.setChecked(False)
        self._chk_smooth.setToolTip("Display-only moving-average smoothing. History files and metrics are unchanged.")
        self._chk_smooth.toggled.connect(lambda _checked: self._update_plot())
        top_row.addWidget(self._chk_smooth)

        self._smooth_window = QSpinBox()
        self._smooth_window.setRange(2, 101)
        self._smooth_window.setValue(5)
        self._smooth_window.setMaximumWidth(70)
        self._smooth_window.setToolTip("Smoothing window in plotted points.")
        self._smooth_window.valueChanged.connect(lambda _value: self._update_plot())
        top_row.addWidget(self._smooth_window)

        self._btn_fit = QPushButton("Auto Scale")
        self._btn_fit.setProperty("plotControl", True)
        self._btn_fit.setToolTip("Automatically refits the plot to the current data.")
        self._btn_fit.clicked.connect(self._auto_range)
        top_row.addWidget(self._btn_fit)

        self._btn_clear = QPushButton("Reset")
        self._btn_clear.setProperty("plotControl", True)
        self._btn_clear.setToolTip("Resets all loss history and metrics in the live plot.")
        self._btn_clear.clicked.connect(self.clear)
        top_row.addWidget(self._btn_clear)

        card_layout.addLayout(top_row)

        # ----------------------------
        # Row 2: metric chips (7 equal widths)
        # ----------------------------
        self._lbl_train = self._metric_label("Train opt/ref", "—")
        self._lbl_val   = self._metric_label("Validation",    "—")
        self._lbl_best  = self._metric_label("Best Val",      "—")
        self._lbl_best_epoch  = self._metric_label("Best Epoch",      "—")
        self._lbl_no_improve  = self._metric_label("No Improvement",  "—")
        self._lbl_lam_dir     = self._metric_label("λ Dir Weight",    "—")
        self._lbl_lr          = self._metric_label("Learning Rate",   "—")

        self._lbl_score = self._metric_label("Checkpoint Score", "...")
        self._lbl_formula = self._metric_label("Formula", "N/A")

        # Metric chips are wrapped in containers so they can be hidden in
        # compact mode (when an external KPI strip already shows these values).
        self._metrics_row1 = QWidget()
        metrics_row = QHBoxLayout(self._metrics_row1)
        metrics_row.setContentsMargins(0, 0, 0, 0)
        metrics_row.setSpacing(6)
        for w in (
            self._lbl_train, self._lbl_val, self._lbl_best,
            self._lbl_best_epoch, self._lbl_no_improve, self._lbl_lam_dir, self._lbl_lr,
        ):
            metrics_row.addWidget(w, 1)
        card_layout.addWidget(self._metrics_row1)

        self._metrics_row2 = QWidget()
        metrics_row2 = QHBoxLayout(self._metrics_row2)
        metrics_row2.setContentsMargins(0, 0, 0, 0)
        metrics_row2.setSpacing(6)
        metrics_row2.addWidget(self._lbl_score, 1)
        metrics_row2.addWidget(self._lbl_formula, 3)
        card_layout.addWidget(self._metrics_row2)

        self._help_label = QLabel(
            "Best metric selects ckpt_best.pt. Hybrid: score = val_base_loss + alpha * val_loss_dir. Lower is better."
        )
        self._help_label.setWordWrap(True)
        self._help_label.setStyleSheet("color: #7f8ab0; font-size: 10px;")
        self._help_label.setToolTip(
            "Best metric is the scalar score used to select ckpt_best.pt. "
            "For hybrid: score = val_base_loss + alpha * val_loss_dir. Lower is better."
        )
        card_layout.addWidget(self._help_label)

        # status label (bottom-aligned)
        self._lbl_status = QLabel("Waiting for training…")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_status.setStyleSheet("color: #5a647a; font-size: 11px;")

        # ----------------------------
        # Plot body
        # ----------------------------
        if _HAS_PYQTGRAPH:
            # NOTE: do NOT set the global 'background' to None here. It is a
            # process-wide pyqtgraph option, and pyqtgraph's GLViewWidget reads
            # it in its constructor and raises ("make a color from (None,)") when
            # it is None — which crashed the Lunar Propagation orbit preview once
            # the Studio had been touched. This widget sets its own background
            # per-instance below, so a global background override is unnecessary.
            pg.setConfigOptions(antialias=True, foreground="#b9c2dd")

            self._plot_widget = pg.PlotWidget(axisItems={"left": _CleanLogAxis(orientation="left")})
            self._plot_widget.setMinimumHeight(360)
            self._plot_widget.setBackground("#0a1120")
            self._plot_widget.setMenuEnabled(False)
            self._plot_widget.showGrid(x=True, y=True, alpha=0.10)
            # Cell titles above each plot describe the y-axis, so keep axes clean.
            self._plot_widget.setLabel("bottom", "Epoch", color="#6f7d9c", size="9pt")
            self._plot_widget.setLogMode(x=False, y=True)

            plot_item = self._plot_widget.getPlotItem()
            plot_item.setContentsMargins(6, 6, 10, 6)
            plot_item.hideButtons()
            for axis_name in ("left", "bottom"):
                axis = self._plot_widget.getAxis(axis_name)
                axis.setTextPen(pg.mkPen("#7f8ca8"))
                axis.setPen(pg.mkPen("#2a3550"))
                axis.setStyle(
                    tickFont=QFont("Consolas", 8),
                    autoExpandTextSpace=True,
                    tickTextOffset=6,
                    tickLength=4,
                )

            self._pen_train = pg.mkPen(color="#8b5cf6", width=2.6)
            self._pen_val = pg.mkPen(color="#22d3ee", width=2.6)
            self._pen_train_shadow = pg.mkPen(color=(139, 92, 246, 70), width=7)
            self._pen_val_shadow = pg.mkPen(color=(34, 211, 238, 70), width=7)

            self._curve_train_shadow = self._plot_widget.plot([], [], pen=self._pen_train_shadow)
            self._curve_val_shadow = self._plot_widget.plot([], [], pen=self._pen_val_shadow)
            self._curve_train = self._plot_widget.plot(
                [], [],
                pen=self._pen_train,
                symbol="o",
                symbolSize=5,
                symbolBrush=pg.mkBrush("#8b5cf6"),
                symbolPen=pg.mkPen("#1b1035"),
                name="train_total",
            )
            self._curve_val = self._plot_widget.plot(
                [], [],
                pen=self._pen_val,
                symbol="o",
                symbolSize=5,
                symbolBrush=pg.mkBrush("#22d3ee"),
                symbolPen=pg.mkPen("#06202a"),
                name="val_total",
            )
            self._curve_train_opt = self._plot_widget.plot(
                [], [],
                pen=pg.mkPen(color="#a78bfa", width=1.8, style=pg_QtCore.Qt.PenStyle.DashLine),
                name="train_objective",
            )
            self._curve_val_base = self._plot_widget.plot(
                [], [],
                pen=pg.mkPen(color="#67e8f9", width=1.8, style=pg_QtCore.Qt.PenStyle.DashLine),
                name="val_base",
            )

            self._best_line = pg.InfiniteLine(
                angle=0,
                movable=False,
                pen=pg.mkPen(color=(52, 211, 153, 130), width=1.2, style=pg_QtCore.Qt.PenStyle.DashLine),
            )
            self._best_line.setVisible(False)
            self._plot_widget.addItem(self._best_line)

            try:
                self._legend = plot_item.addLegend(
                    offset=(12, 12),
                    labelTextSize="9pt",
                    labelTextColor="#dbe4ff",
                    brush=pg.mkBrush(8, 12, 26, 185),
                    pen=pg.mkPen(124, 92, 255, 90),
                )
            except TypeError:
                # Older pyqtgraph versions do not support all styling kwargs.
                self._legend = plot_item.addLegend(offset=(12, 12))

            self._direction_plot = pg.PlotWidget(axisItems={"left": _CleanLogAxis(orientation="left")})
            self._direction_plot.setMinimumHeight(230)
            self._direction_plot.setBackground("#0a1120")
            self._direction_plot.setMenuEnabled(False)
            self._direction_plot.showGrid(x=True, y=True, alpha=0.10)
            self._direction_plot.setLabel("bottom", "Epoch", color="#6f7d9c", size="9pt")
            self._direction_plot.setLogMode(x=False, y=True)
            self._curve_train_loss_a = self._direction_plot.plot([], [], pen=pg.mkPen(color="#34d399", width=2.1), name="train a")
            self._curve_val_loss_a = self._direction_plot.plot([], [], pen=pg.mkPen(color="#10b981", width=2.4), name="val a")
            self._curve_train_dir = self._direction_plot.plot([], [], pen=pg.mkPen(color="#fbbf24", width=2.1), name="train dir")
            self._curve_val_dir = self._direction_plot.plot([], [], pen=pg.mkPen(color="#f59e0b", width=2.4), name="val dir")

            self._direction_quality_plot = pg.PlotWidget()
            self._direction_quality_plot.setMinimumHeight(180)
            self._direction_quality_plot.setBackground("#0a1120")
            self._direction_quality_plot.setMenuEnabled(False)
            self._direction_quality_plot.showGrid(x=True, y=True, alpha=0.10)
            self._direction_quality_plot.setLabel("bottom", "Epoch", color="#6f7d9c", size="9pt")
            self._curve_val_angular = self._direction_quality_plot.plot([], [], pen=pg.mkPen(color="#3b82f6", width=2.4), name="val ang°")
            self._curve_train_cossim = self._direction_quality_plot.plot([], [], pen=pg.mkPen(color="#c084fc", width=2.1), name="train cos")
            self._curve_val_cossim = self._direction_quality_plot.plot([], [], pen=pg.mkPen(color="#8b5cf6", width=2.4), name="val cos")

            self._direction_tab = QWidget()
            direction_layout = QVBoxLayout()
            direction_layout.setContentsMargins(0, 0, 0, 0)
            direction_layout.setSpacing(6)
            direction_layout.addWidget(self._direction_plot, 3)
            direction_layout.addWidget(self._direction_quality_plot, 2)
            self._direction_tab.setLayout(direction_layout)

            self._checkpoint_plot = pg.PlotWidget(axisItems={"left": _CleanLogAxis(orientation="left")})
            self._checkpoint_plot.setMinimumHeight(360)
            self._checkpoint_plot.setBackground("#0a1120")
            self._checkpoint_plot.setMenuEnabled(False)
            self._checkpoint_plot.showGrid(x=True, y=True, alpha=0.10)
            self._checkpoint_plot.setLabel("bottom", "Epoch", color="#6f7d9c", size="9pt")
            self._checkpoint_plot.setLogMode(x=False, y=True)
            self._curve_score = self._checkpoint_plot.plot([], [], pen=pg.mkPen(color="#34d399", width=2.6), name="score")
            self._curve_best_score = self._checkpoint_plot.plot([], [], pen=pg.mkPen(color="#f472b6", width=2.2, style=pg_QtCore.Qt.PenStyle.DashLine), name="best")

            # Consistent clean axes + a compact legend on every companion plot.
            for _p in (self._direction_plot, self._direction_quality_plot, self._checkpoint_plot):
                _pi = _p.getPlotItem()
                _pi.setContentsMargins(6, 6, 10, 6)
                _pi.hideButtons()
                for _axis_name in ("left", "bottom"):
                    _ax = _p.getAxis(_axis_name)
                    _ax.setTextPen(pg.mkPen("#7f8ca8"))
                    _ax.setPen(pg.mkPen("#2a3550"))
                    _ax.setStyle(tickFont=QFont("Consolas", 8), tickTextOffset=6, tickLength=4)
                try:
                    _pi.addLegend(offset=(8, 8), labelTextSize="8pt",
                                  labelTextColor="#cdd9ee",
                                  brush=pg.mkBrush(8, 12, 26, 170),
                                  pen=pg.mkPen(124, 92, 255, 70))
                except TypeError:
                    _pi.addLegend(offset=(8, 8))

            # ── Dashboard grid: all charts visible at once (no hidden tabs) ──
            # A dominant Loss panel on top, with Acceleration/Direction and
            # Direction-quality side by side, and Checkpoint score full width.
            def _titled(title: str, widget, minh: int):
                box = QFrame()
                box.setObjectName("plotCell")
                box.setStyleSheet(
                    "QFrame#plotCell { background: transparent; border: none; }"
                )
                v = QVBoxLayout(box)
                v.setContentsMargins(0, 0, 0, 0)
                v.setSpacing(3)
                lbl = QLabel(title)
                lbl.setStyleSheet(
                    "color: #8ea3c8; font-size: 11px; font-weight: 700;"
                    " letter-spacing: 0.4px; background: transparent; border: none;"
                    " padding-left: 4px;"
                )
                widget.setMinimumHeight(minh)
                v.addWidget(lbl)
                v.addWidget(widget, 1)
                return box

            # The direction-quality plot used to live inside _direction_tab; it is
            # now a standalone cell, so reparent it cleanly.
            self._direction_quality_plot.setParent(None)

            # 2×2 dashboard: Loss (primary, top-left) plus three companions.
            # A 2×2 layout is far more compact than stacked rows, leaving room
            # for the per-epoch History table below the chart card.
            self._plots_container = QWidget()
            plots_grid = QGridLayout(self._plots_container)
            plots_grid.setContentsMargins(0, 0, 0, 0)
            plots_grid.setSpacing(10)
            plots_grid.addWidget(_titled("Loss · train / validation", self._plot_widget, 150), 0, 0)
            plots_grid.addWidget(_titled("Acceleration / direction loss", self._direction_plot, 150), 0, 1)
            plots_grid.addWidget(_titled("Direction quality · cos / angular", self._direction_quality_plot, 150), 1, 0)
            plots_grid.addWidget(_titled("Checkpoint score", self._checkpoint_plot, 150), 1, 1)
            plots_grid.setRowStretch(0, 1)
            plots_grid.setRowStretch(1, 1)
            plots_grid.setColumnStretch(0, 1)
            plots_grid.setColumnStretch(1, 1)
            self._plots_container.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            # Kept for backwards compatibility with set_compact()/references.
            self._plot_tabs = None
            card_layout.addWidget(self._plots_container, 1)
        else:
            self._plot_widget = None
            self._direction_plot = None
            self._checkpoint_plot = None
            placeholder = QLabel(
                "pyqtgraph not installed — live plotting disabled.\n"
                "To install:  pip install pyqtgraph"
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                "color: #7f8ab0; background-color: rgba(4, 8, 18, 0.72); "
                "border: 1px solid rgba(185, 194, 221, 0.12); border-radius: 14px; "
                "padding: 24px; font-style: italic;"
            )
            card_layout.addWidget(placeholder)

        card_layout.addWidget(self._lbl_status)
        outer.addWidget(self._card)
        self.setLayout(outer)

        # Regex patterns for parsing. Supports both old pretty logs and compact logger lines.
        _num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        self._re_epoch = re.compile(r"Epoch\s*\[?\s*(\d+)\s*/\s*(\d+)\s*\]?", re.IGNORECASE)
        self._re_epoch_kv = re.compile(r"\bepoch\s*=\s*(\d+)\b", re.IGNORECASE)
        self._re_train_opt_ref = re.compile(rf"\bTrain\s+opt\s*[:=]\s*({_num})\s+ref\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_val_ref = re.compile(rf"\bVal\s+ref\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_val_total = re.compile(rf"\bval\s+total\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_val_base = re.compile(rf"\bbase\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_val_dir = re.compile(rf"\bdir\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_loss_u = re.compile(rf"\bU\s*[:=]\s*({_num})")
        self._re_loss_a = re.compile(rf"\ba\s*[:=]\s*({_num})")
        self._re_cossim = re.compile(rf"\bcossim\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_angular = re.compile(rf"\bang\s*[:=]\s*({_num})\s*deg", re.IGNORECASE)
        self._re_score = re.compile(rf"\bscore\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_best_score = re.compile(rf"\bbest\s*=\s*(?:YES|no).*?\bscore\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_best_formula = re.compile(r"\[([^:\]]+):\s*([^\]]+)\]")
        self._re_loss_opt = re.compile(rf"\bloss_opt\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_loss_ref = re.compile(rf"\bloss_ref\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_train_loss = re.compile(
            rf"(?:Train|train)[\s_]*(?:Loss|loss)\s*[:=]\s*({_num})"
        )
        self._re_val_loss = re.compile(
            rf"(?:Val|val|Validation|validation)[\s_]*(?:Loss|loss)\s*[:=]\s*({_num})"
        )
        self._re_lr = re.compile(rf"\b(?:lr|LR)\s*[:=]\s*({_num})")
        self._re_lam_dir = re.compile(rf"\b(?:lam_dir|lambda_dir_eff|lam)\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_loss_generic = re.compile(rf"\bloss\s*[:=]\s*({_num})", re.IGNORECASE)
        self._re_ckpt_start = re.compile(r"(?:\[checkpoint\].*)?(?:best[- ]checkpoint.*)?tracking\s+starts?\s+at\s+epoch\s+(\d+)", re.IGNORECASE)
        self._re_ckpt_wait = re.compile(r"\[checkpoint\].*waiting.*epoch\s+(\d+)\s*<\s*start\s+epoch\s+(\d+)", re.IGNORECASE)
        self._re_ckpt_best = re.compile(rf"\[checkpoint\].*best updated.*val_ref\s*[:=]\s*({_num}).*epoch\s*[:=]\s*(\d+)", re.IGNORECASE)
        self._re_ckpt_last = re.compile(r"\[checkpoint\].*last saved.*epoch\s*[:=]\s*(\d+)", re.IGNORECASE)

        self._refresh_metric_labels()

    def set_compact(self, compact: bool = True) -> None:
        """Compact mode hides the in-card metric chips/help (shown elsewhere by
        the KPI/time strips) so the plot grid itself gets the vertical space."""
        for w in (self._metrics_row1, self._metrics_row2, self._help_label):
            w.setVisible(not compact)
        if compact:
            # Uniform, small per-plot minimums so the 2×2 dashboard stays
            # compact and leaves vertical room for the History table below.
            for attr in (
                "_plot_widget", "_direction_plot",
                "_direction_quality_plot", "_checkpoint_plot",
            ):
                plot = getattr(self, attr, None)
                if plot is not None:
                    plot.setMinimumHeight(140)

    def _metric_label(self, name: str, value: str = "—") -> QLabel:
        lbl = QLabel(f"<span style='color:#7480a8;font-size:10px'>{name}</span><br>"
                     f"<span style='font-family:Consolas,monospace;font-size:11px'>{value}</span>")
        lbl.setProperty("metric", True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl._metric_name = name
        return lbl

    @staticmethod
    def _fmt_metric(v: Optional[float]) -> str:
        if v is None:
            return "—"
        try:
            if not math.isfinite(v):
                return "—"
        except Exception:
            return "—"
        return f"{float(v):.3e}"

    def parse_line(self, line: str) -> None:
        """Parse a log line and update live loss/status fields when possible."""
        self._parse_checkpoint_status(line)

        m_epoch = self._re_epoch.search(line)
        epoch = int(m_epoch.group(1)) if m_epoch else None

        # Fallback for compact logs like: epoch=116 ... loss=6.07e-03 lr=...
        if epoch is None:
            m_epoch_kv = self._re_epoch_kv.search(line)
            if m_epoch_kv:
                epoch = int(m_epoch_kv.group(1))
        if epoch is None:
            self._refresh_metric_labels()
            self._lbl_status.setText(self._checkpoint_status or "Waiting")
            return

        self._latest_epoch = int(epoch)
        lower = line.lower()
        is_train_phase = "[train]" in lower or lower.startswith("train ") or " train opt" in f" {lower}"
        is_val_phase = "[val" in lower or " validation " in f" {lower} " or " val ref" in f" {lower}"

        train_opt: Optional[float] = None
        train_ref: Optional[float] = None
        train_u: Optional[float] = None
        train_a: Optional[float] = None
        train_dir: Optional[float] = None
        train_cos: Optional[float] = None
        val_ref: Optional[float] = None
        val_u: Optional[float] = None
        val_a: Optional[float] = None
        val_base: Optional[float] = None
        val_dir: Optional[float] = None
        val_cos: Optional[float] = None
        val_angular: Optional[float] = None
        checkpoint_score: Optional[float] = None
        best_score: Optional[float] = None

        m_train_opt_ref = self._re_train_opt_ref.search(line)
        if m_train_opt_ref:
            train_opt = float(m_train_opt_ref.group(1))
            train_ref = float(m_train_opt_ref.group(2))

        m_val_ref = self._re_val_ref.search(line)
        if m_val_ref:
            val_ref = float(m_val_ref.group(1))
        m_val_total = self._re_val_total.search(line)
        if m_val_total:
            val_ref = float(m_val_total.group(1))
        m_val_base = self._re_val_base.search(line)
        if m_val_base:
            val_base = float(m_val_base.group(1))
        m_val_dir = self._re_val_dir.search(line)
        if m_val_dir:
            if is_train_phase:
                train_dir = float(m_val_dir.group(1))
            else:
                val_dir = float(m_val_dir.group(1))
        m_loss_u = self._re_loss_u.search(line)
        if m_loss_u:
            if is_train_phase:
                train_u = float(m_loss_u.group(1))
            elif is_val_phase:
                val_u = float(m_loss_u.group(1))
        m_loss_a = self._re_loss_a.search(line)
        if m_loss_a:
            if is_train_phase:
                train_a = float(m_loss_a.group(1))
            elif is_val_phase:
                val_a = float(m_loss_a.group(1))
        m_cos = self._re_cossim.search(line)
        if m_cos:
            if is_train_phase:
                train_cos = float(m_cos.group(1))
            elif is_val_phase:
                val_cos = float(m_cos.group(1))
        m_ang = self._re_angular.search(line)
        if m_ang and is_val_phase:
            val_angular = float(m_ang.group(1))
        m_score = self._re_score.search(line)
        if m_score:
            checkpoint_score = float(m_score.group(1))
        m_best_score = self._re_best_score.search(line)
        if m_best_score:
            best_score = float(m_best_score.group(1))
        m_formula = self._re_best_formula.search(line)
        if m_formula:
            self._best_metric_name = m_formula.group(1).strip()
            self._best_formula = m_formula.group(2).strip()

        if train_ref is None and is_train_phase:
            m_loss_ref = self._re_loss_ref.search(line)
            if m_loss_ref:
                train_ref = float(m_loss_ref.group(1))
            m_loss_opt = self._re_loss_opt.search(line)
            if m_loss_opt:
                train_opt = float(m_loss_opt.group(1))

        if val_ref is None and is_val_phase:
            m_loss_ref = self._re_loss_ref.search(line)
            if m_loss_ref:
                val_ref = float(m_loss_ref.group(1))

        # Backward-compatible fallbacks for older "Train Loss" / "Val Loss" logs.
        m_train = self._re_train_loss.search(line)
        if train_ref is None and m_train:
            train_ref = float(m_train.group(1))
        m_val = self._re_val_loss.search(line)
        if val_ref is None and m_val:
            val_ref = float(m_val.group(1))
        if train_ref is None and val_ref is None and train_opt is None:
            m_generic = self._re_loss_generic.search(line)
            if m_generic and is_train_phase:
                train_ref = float(m_generic.group(1))
            elif m_generic and is_val_phase:
                val_ref = float(m_generic.group(1))

        m_lr = self._re_lr.search(line)
        lr_val = float(m_lr.group(1)) if m_lr else float("nan")
        m_lam = self._re_lam_dir.search(line)
        if m_lam:
            self._latest_lam_dir = float(m_lam.group(1))

        if (
            train_ref is None and val_ref is None and train_opt is None
            and train_u is None and train_a is None and train_dir is None and train_cos is None
            and val_u is None and val_a is None and val_dir is None and val_cos is None and val_angular is None
            and checkpoint_score is None and val_base is None and val_dir is None
            and not m_lr and not m_lam
        ):
            self._refresh_metric_labels()
            self._lbl_status.setText(f"Epoch {self._latest_epoch or epoch} | {self._checkpoint_status}")
            return

        # Avoid duplicate epoch points when a logger emits repeated summaries.
        if epoch in self._epochs:
            idx = self._epochs.index(epoch)
            if train_ref is not None:
                self._train_loss[idx] = train_ref
            if train_opt is not None:
                self._train_opt_loss[idx] = train_opt
            if train_u is not None:
                self._train_loss_u[idx] = train_u
            if train_a is not None:
                self._train_loss_a[idx] = train_a
            if train_dir is not None:
                self._train_loss_dir[idx] = train_dir
            if train_cos is not None:
                self._train_cos_sim[idx] = train_cos
            if val_ref is not None:
                self._val_loss[idx] = val_ref
            if val_u is not None:
                self._val_loss_u[idx] = val_u
            if val_a is not None:
                self._val_loss_a[idx] = val_a
            if val_base is not None:
                self._val_base_loss[idx] = val_base
            if val_dir is not None:
                self._val_dir_loss[idx] = val_dir
            if val_cos is not None:
                self._val_cos_sim[idx] = val_cos
            if val_angular is not None:
                self._val_angular_mean_deg[idx] = val_angular
            if checkpoint_score is not None:
                self._checkpoint_scores[idx] = checkpoint_score
            if best_score is not None:
                self._best_scores[idx] = best_score
            if m_lr:
                self._lr_values[idx] = lr_val
        else:
            self._epochs.append(epoch)
            self._train_loss.append(train_ref if train_ref is not None else float("nan"))
            self._train_opt_loss.append(train_opt if train_opt is not None else float("nan"))
            self._train_loss_u.append(train_u if train_u is not None else float("nan"))
            self._train_loss_a.append(train_a if train_a is not None else float("nan"))
            self._train_loss_dir.append(train_dir if train_dir is not None else float("nan"))
            self._train_cos_sim.append(train_cos if train_cos is not None else float("nan"))
            self._val_loss.append(val_ref if val_ref is not None else float("nan"))
            self._val_loss_u.append(val_u if val_u is not None else float("nan"))
            self._val_base_loss.append(val_base if val_base is not None else float("nan"))
            self._val_dir_loss.append(val_dir if val_dir is not None else float("nan"))
            self._val_loss_a.append(val_a if val_a is not None else float("nan"))
            self._val_angular_mean_deg.append(val_angular if val_angular is not None else float("nan"))
            self._val_cos_sim.append(val_cos if val_cos is not None else float("nan"))
            self._checkpoint_scores.append(checkpoint_score if checkpoint_score is not None else float("nan"))
            self._best_scores.append(best_score if best_score is not None else float("nan"))
            self._lr_values.append(lr_val)

        if train_opt is not None and math.isfinite(train_opt):
            self._latest_train_opt = float(train_opt)
        if train_ref is not None and math.isfinite(train_ref):
            self._latest_train_ref = float(train_ref)
        if val_ref is not None and math.isfinite(val_ref):
            self._latest_val_ref = float(val_ref)
            if self._best_val is None or val_ref < self._best_val:
                self._best_val = float(val_ref)
                self._best_epoch = int(epoch)
                self._epochs_since_improvement = 0
                self._checkpoint_status = "Best updated"
            elif self._best_epoch is not None:
                self._epochs_since_improvement = max(0, int(epoch) - int(self._best_epoch))
        elif self._best_epoch is not None:
            self._epochs_since_improvement = max(0, int(epoch) - int(self._best_epoch))
        if checkpoint_score is not None and math.isfinite(checkpoint_score):
            self._latest_checkpoint_score = float(checkpoint_score)
        if best_score is not None and math.isfinite(best_score):
            self._best_val = float(best_score)

        self._update_plot()

    def load_history_file(self, path: str) -> None:
        """Load flat history JSONL/CSV rows without blocking the launcher path."""
        p = Path(path)
        if not p.exists():
            return
        try:
            rows: List[Dict[str, Any]] = []
            if p.suffix.lower() == ".jsonl":
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            elif p.suffix.lower() == ".csv":
                import csv as _csv
                with p.open("r", newline="", encoding="utf-8") as handle:
                    rows.extend(dict(row) for row in _csv.DictReader(handle))
            else:
                return
            rows_by_epoch: Dict[int, Dict[str, Any]] = {}
            for row in rows:
                try:
                    epoch = int(float(row.get("epoch_display") or (float(row.get("epoch", 0)) + 1)))
                except Exception:
                    continue
                rows_by_epoch[epoch] = row

            def _row_float(row: Dict[str, Any], key: str, default: str = "nan") -> float:
                try:
                    value = float(row.get(key, default))
                except Exception:
                    value = float("nan")
                return value if math.isfinite(value) else float("nan")

            self.clear()
            for epoch, row in sorted(rows_by_epoch.items()):
                self._epochs.append(epoch)
                self._train_loss.append(_row_float(row, "train_loss_total"))
                self._train_opt_loss.append(_row_float(row, "train_loss_objective"))
                self._train_loss_u.append(_row_float(row, "train_loss_u"))
                self._train_loss_a.append(_row_float(row, "train_loss_a"))
                self._train_loss_dir.append(_row_float(row, "train_loss_dir"))
                self._train_cos_sim.append(_row_float(row, "train_cos_sim", row.get("train_mean_cossim", "nan")))
                self._val_loss.append(_row_float(row, "val_loss_total"))
                self._val_loss_u.append(_row_float(row, "val_loss_u"))
                self._val_base_loss.append(_row_float(row, "val_loss_base"))
                self._val_dir_loss.append(_row_float(row, "val_loss_dir"))
                self._val_loss_a.append(_row_float(row, "val_loss_a"))
                self._val_angular_mean_deg.append(_row_float(row, "val_angular_mean_deg"))
                self._val_cos_sim.append(_row_float(row, "val_cos_sim", row.get("val_mean_cossim", "nan")))
                self._checkpoint_scores.append(_row_float(row, "checkpoint_score", row.get("val_checkpoint_score", "nan")))
                self._best_scores.append(_row_float(row, "best_score"))
                self._lr_values.append(_row_float(row, "lr"))
                if row.get("best_metric"):
                    self._best_metric_name = str(row.get("best_metric"))
                if row.get("checkpoint_formula"):
                    self._best_formula = str(row.get("checkpoint_formula"))
                try:
                    best_epoch = row.get("best_epoch")
                    if best_epoch not in (None, ""):
                        self._best_epoch = int(float(best_epoch))
                    best_score = row.get("best_score")
                    if best_score not in (None, ""):
                        best_score_f = float(best_score)
                        if math.isfinite(best_score_f):
                            self._best_val = best_score_f
                except Exception:
                    pass
            self._update_plot()
        except Exception:
            self._lbl_status.setText("History unavailable")

    def _parse_checkpoint_status(self, line: str) -> None:
        """Update checkpoint status chips from engine checkpoint log lines."""
        m = self._re_ckpt_start.search(line)
        if m:
            self._checkpoint_status = f"Waiting for direction ramp (start {m.group(1)})"
            return
        m = self._re_ckpt_wait.search(line)
        if m:
            self._checkpoint_status = f"Waiting for direction ramp ({m.group(1)}/{m.group(2)})"
            return
        m = self._re_ckpt_best.search(line)
        if m:
            self._best_val = float(m.group(1))
            self._best_epoch = int(m.group(2))
            self._epochs_since_improvement = 0
            self._checkpoint_status = "Best updated"
            return
        m = self._re_ckpt_last.search(line)
        if m:
            if self._checkpoint_status not in ("Best updated",):
                self._checkpoint_status = "Last checkpoint saved"
            return
        if "[checkpoint]" in line.lower() and "tracking" in line.lower():
            self._checkpoint_status = "Tracking best model"

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * pct / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ordered[lo])
        weight = pos - lo
        return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)

    def _smooth_plot_values(self, values: List[float]) -> List[float]:
        if not self._chk_smooth.isChecked() or len(values) < 3:
            return values
        window = max(2, int(self._smooth_window.value()))
        smoothed: List[float] = []
        for idx in range(len(values)):
            start = max(0, idx - window + 1)
            chunk = values[start : idx + 1]
            smoothed.append(sum(chunk) / max(1, len(chunk)))
        return smoothed

    def _valid_xy(
        self,
        values: List[float],
        *,
        log_y: Optional[bool] = None,
        smooth: bool = True,
    ) -> Tuple[List[int], List[float]]:
        if log_y is None:
            log_y = self._chk_log_y.isChecked()
        xs: List[int] = []
        ys: List[float] = []
        for e, v in zip(self._epochs, values):
            try:
                finite = math.isfinite(v)
            except Exception:
                finite = False
            if finite and (not log_y or float(v) > 0.0):
                xs.append(int(e))
                ys.append(float(v))
        if smooth:
            ys = self._smooth_plot_values(ys)
        return xs, ys

    def _set_group_title(self, plot: Any, title: str, has_data: bool) -> None:
        if not plot:
            return
        if has_data:
            plot.setTitle(title, color="#dbe4ff", size="10pt")
        else:
            message = "Waiting for history/log data..." if not self._epochs else "No data for this metric yet."
            plot.setTitle(message, color="#7f8ab0", size="10pt")

    def _range_for_values(
        self,
        series_values: List[List[float]],
        *,
        log_y: bool,
        y_bounds: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        if y_bounds is not None:
            return y_bounds
        valid: List[float] = []
        for values in series_values:
            for value in values:
                try:
                    v = float(value)
                except Exception:
                    continue
                if not math.isfinite(v):
                    continue
                if log_y and v <= 0.0:
                    continue
                valid.append(v)
        if not valid:
            return None
        lo = self._percentile(valid, 1.0)
        hi = self._percentile(valid, 99.0)
        if not math.isfinite(lo) or not math.isfinite(hi):
            return None
        if log_y:
            lo = max(lo, 1e-30)
            hi = max(hi, lo * 1.01)
            return lo / 1.35, hi * 1.35
        if hi <= lo:
            margin = abs(hi) * 0.1 + 1e-12
            return lo - margin, hi + margin
        margin = (hi - lo) * 0.08
        return lo - margin, hi + margin

    def _set_plot_range(
        self,
        plot: Any,
        series_values: List[List[float]],
        *,
        log_y: bool,
        y_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        if not plot:
            return
        if self._epochs:
            xmin = max(0, min(self._epochs) - 1)
            xmax = max(self._epochs) + 1
            plot.setXRange(xmin, xmax, padding=0.02)
        yr = self._range_for_values(series_values, log_y=log_y, y_bounds=y_bounds)
        if yr is None:
            return
        ymin, ymax = yr
        if log_y:
            ymin = math.log10(max(ymin, 1e-30))
            ymax = math.log10(max(ymax, 1e-30))
        plot.setYRange(ymin, ymax, padding=0.02)

    def _apply_plot_ranges(self) -> None:
        loss_log = self._chk_log_y.isChecked()
        self._set_plot_range(
            self._plot_widget,
            [self._train_loss, self._train_opt_loss, self._val_loss, self._val_base_loss],
            log_y=loss_log,
        )
        self._set_plot_range(
            self._direction_plot,
            [self._train_loss_a, self._val_loss_a, self._train_loss_dir, self._val_dir_loss],
            log_y=loss_log,
        )
        if getattr(self, "_direction_quality_plot", None) is not None:
            self._set_plot_range(
                self._direction_quality_plot,
                [self._train_cos_sim, self._val_cos_sim, self._val_angular_mean_deg],
                log_y=False,
            )
        self._set_plot_range(
            self._checkpoint_plot,
            [self._checkpoint_scores, self._best_scores],
            log_y=loss_log,
        )

    def _update_plot(self) -> None:
        self._refresh_metric_labels()
        if not self._plot_widget or not _HAS_PYQTGRAPH:
            return

        loss_log = self._chk_log_y.isChecked()
        t_ep, t_val = self._valid_xy(self._train_loss, log_y=loss_log)
        to_ep, to_val = self._valid_xy(self._train_opt_loss, log_y=loss_log)
        v_ep, v_val = self._valid_xy(self._val_loss, log_y=loss_log)
        vb_ep, vb_val = self._valid_xy(self._val_base_loss, log_y=loss_log)
        tr_a_ep, tr_a_val = self._valid_xy(self._train_loss_a, log_y=loss_log)
        a_ep, a_val = self._valid_xy(self._val_loss_a, log_y=loss_log)
        tr_dir_ep, tr_dir_val = self._valid_xy(self._train_loss_dir, log_y=loss_log)
        dir_ep, dir_val = self._valid_xy(self._val_dir_loss, log_y=loss_log)
        tr_cos_ep, tr_cos_val = self._valid_xy(self._train_cos_sim, log_y=False)
        ang_ep, ang_val = self._valid_xy(self._val_angular_mean_deg, log_y=False)
        cos_ep, cos_val = self._valid_xy(self._val_cos_sim, log_y=False)
        score_ep, score_val = self._valid_xy(self._checkpoint_scores, log_y=loss_log)
        best_ep, best_val = self._valid_xy(self._best_scores, log_y=loss_log)

        self._curve_train.setData(t_ep, t_val)
        self._curve_train_shadow.setData(t_ep, t_val)
        self._curve_val.setData(v_ep, v_val)
        self._curve_val_shadow.setData(v_ep, v_val)
        if getattr(self, "_curve_train_opt", None) is not None:
            self._curve_train_opt.setData(to_ep, to_val)
        if getattr(self, "_curve_val_base", None) is not None:
            self._curve_val_base.setData(vb_ep, vb_val)
        if getattr(self, "_curve_train_loss_a", None) is not None:
            self._curve_train_loss_a.setData(tr_a_ep, tr_a_val)
        if getattr(self, "_curve_val_dir", None) is not None:
            self._curve_val_dir.setData(dir_ep, dir_val)
        if getattr(self, "_curve_val_loss_a", None) is not None:
            self._curve_val_loss_a.setData(a_ep, a_val)
        if getattr(self, "_curve_train_dir", None) is not None:
            self._curve_train_dir.setData(tr_dir_ep, tr_dir_val)
        if getattr(self, "_curve_val_angular", None) is not None:
            self._curve_val_angular.setData(ang_ep, ang_val)
        if getattr(self, "_curve_train_cossim", None) is not None:
            self._curve_train_cossim.setData(tr_cos_ep, tr_cos_val)
        if getattr(self, "_curve_val_cossim", None) is not None:
            self._curve_val_cossim.setData(cos_ep, cos_val)
        if getattr(self, "_curve_score", None) is not None:
            self._curve_score.setData(score_ep, score_val)
        if getattr(self, "_curve_best_score", None) is not None:
            self._curve_best_score.setData(best_ep, best_val)

        if self._best_val is not None and math.isfinite(self._best_val) and self._best_val > 0:
            line_value = math.log10(float(self._best_val)) if loss_log else float(self._best_val)
            self._best_line.setValue(line_value)
            self._best_line.setVisible(True)
        else:
            self._best_line.setVisible(False)

        self._set_group_title(self._plot_widget, "Loss overview", bool(t_val or to_val or v_val or vb_val))
        self._set_group_title(self._direction_plot, "Acceleration and direction losses", bool(tr_a_val or a_val or tr_dir_val or dir_val))
        if getattr(self, "_direction_quality_plot", None) is not None:
            self._set_group_title(self._direction_quality_plot, "Direction quality", bool(tr_cos_val or cos_val or ang_val))
        self._set_group_title(self._checkpoint_plot, "Checkpoint score", bool(score_val or best_val))
        self._apply_plot_ranges()
        if self._epochs:
            self._lbl_status.setText(
                f"Epoch {self._latest_epoch or self._epochs[-1]}  ·  {self._checkpoint_status}"
            )
        else:
            self._lbl_status.setText("Waiting for history/log data…")

    def _refresh_metric_labels(self) -> None:
        latest_train = next((v for v in reversed(self._train_loss) if math.isfinite(v)), None)
        latest_train_opt = next((v for v in reversed(self._train_opt_loss) if math.isfinite(v)), None)
        latest_val = next((v for v in reversed(self._val_loss) if math.isfinite(v)), None)
        latest_lr = next((v for v in reversed(self._lr_values) if math.isfinite(v)), None)

        self._latest_train_opt = latest_train_opt if latest_train_opt is not None else self._latest_train_opt
        self._latest_train_ref = latest_train if latest_train is not None else self._latest_train_ref
        self._latest_val_ref = latest_val if latest_val is not None else self._latest_val_ref
        def _chip(name: str, value: str) -> str:
            return (
                f"<span style='color:#7480a8;font-size:10px'>{name}</span><br>"
                f"<span style='font-family:Consolas,monospace;font-size:11px'>{value}</span>"
            )

        opt_s  = self._fmt_metric(self._latest_train_opt)
        ref_s  = self._fmt_metric(self._latest_train_ref)
        self._lbl_train.setText(_chip("Train opt/ref", f"{opt_s} / {ref_s}"))
        self._lbl_val.setText(_chip("Validation", self._fmt_metric(self._latest_val_ref)))
        self._lbl_best.setText(_chip("Best Val", self._fmt_metric(self._best_val)))
        self._lbl_best_epoch.setText(_chip(
            "Best Epoch",
            str(self._best_epoch) if self._best_epoch is not None else "—",
        ))
        self._lbl_no_improve.setText(_chip(
            "No Improvement",
            str(self._epochs_since_improvement) if self._epochs_since_improvement is not None else "—",
        ))
        self._lbl_lam_dir.setText(_chip("λ Dir Weight", self._fmt_metric(self._latest_lam_dir)))
        self._lbl_lr.setText(_chip("Learning Rate", self._fmt_metric(latest_lr)))

        self._lbl_score.setText(_chip("Checkpoint Score", self._fmt_metric(self._latest_checkpoint_score)))
        formula = self._best_formula if self._best_formula else "N/A"
        if len(formula) > 56:
            formula = formula[:53] + "..."
        self._lbl_formula.setText(_chip(self._best_metric_name or "Formula", formula))

    def _on_log_toggle(self, checked: bool) -> None:
        if self._plot_widget and _HAS_PYQTGRAPH:
            self._plot_widget.setLogMode(x=False, y=checked)
            if getattr(self, "_direction_plot", None) is not None:
                self._direction_plot.setLogMode(x=False, y=checked)
            if getattr(self, "_direction_quality_plot", None) is not None:
                self._direction_quality_plot.setLogMode(x=False, y=False)
            if getattr(self, "_checkpoint_plot", None) is not None:
                self._checkpoint_plot.setLogMode(x=False, y=checked)
            self._plot_widget.setLabel(
                "left",
                "Loss (log)" if checked else "Loss",
                color="#aeb8d8",
                size="10pt",
            )
            if getattr(self, "_direction_plot", None) is not None:
                self._direction_plot.setLabel(
                    "left",
                    "Loss (log)" if checked else "Loss",
                    color="#aeb8d8",
                    size="10pt",
                )
            self._update_plot()
            self._auto_range()

    def _auto_range(self) -> None:
        if self._plot_widget and _HAS_PYQTGRAPH:
            self._apply_plot_ranges()

    def clear(self) -> None:
        self._epochs.clear()
        self._train_loss.clear()
        self._val_loss.clear()
        self._train_opt_loss.clear()
        self._train_loss_u.clear()
        self._val_loss_u.clear()
        self._train_loss_a.clear()
        self._val_base_loss.clear()
        self._train_loss_dir.clear()
        self._val_dir_loss.clear()
        self._val_loss_a.clear()
        self._train_cos_sim.clear()
        self._val_angular_mean_deg.clear()
        self._val_cos_sim.clear()
        self._checkpoint_scores.clear()
        self._best_scores.clear()
        self._lr_values.clear()
        self._best_val = None
        self._best_epoch = None
        self._latest_epoch = None
        self._latest_train_opt = None
        self._latest_train_ref = None
        self._latest_val_ref = None
        self._latest_lam_dir = None
        self._latest_checkpoint_score = None
        self._best_metric_name = "best metric"
        self._best_formula = "N/A"
        self._epochs_since_improvement = None
        self._checkpoint_status = "Waiting for training"
        if self._plot_widget and _HAS_PYQTGRAPH:
            self._curve_train.setData([], [])
            self._curve_train_shadow.setData([], [])
            self._curve_val.setData([], [])
            self._curve_val_shadow.setData([], [])
            if getattr(self, "_curve_train_opt", None) is not None:
                self._curve_train_opt.setData([], [])
            if getattr(self, "_curve_val_base", None) is not None:
                self._curve_val_base.setData([], [])
            if getattr(self, "_curve_train_loss_a", None) is not None:
                self._curve_train_loss_a.setData([], [])
            if getattr(self, "_curve_train_dir", None) is not None:
                self._curve_train_dir.setData([], [])
            if getattr(self, "_curve_val_dir", None) is not None:
                self._curve_val_dir.setData([], [])
            if getattr(self, "_curve_val_loss_a", None) is not None:
                self._curve_val_loss_a.setData([], [])
            if getattr(self, "_curve_val_angular", None) is not None:
                self._curve_val_angular.setData([], [])
            if getattr(self, "_curve_train_cossim", None) is not None:
                self._curve_train_cossim.setData([], [])
            if getattr(self, "_curve_val_cossim", None) is not None:
                self._curve_val_cossim.setData([], [])
            if getattr(self, "_curve_score", None) is not None:
                self._curve_score.setData([], [])
            if getattr(self, "_curve_best_score", None) is not None:
                self._curve_best_score.setData([], [])
            self._best_line.setVisible(False)
            self._set_group_title(self._plot_widget, "Loss overview", False)
            self._set_group_title(self._direction_plot, "Acceleration and direction losses", False)
            if getattr(self, "_direction_quality_plot", None) is not None:
                self._set_group_title(self._direction_quality_plot, "Direction quality", False)
            self._set_group_title(self._checkpoint_plot, "Checkpoint score", False)
        self._lbl_status.setText("Waiting for history/log data…")
        self._refresh_metric_labels()

    def get_final_losses(self) -> Dict[str, Any]:
        """Return summary for queue status display."""
        result = {}
        valid_train = [v for v in self._train_loss if math.isfinite(v)]
        valid_train_opt = [v for v in self._train_opt_loss if math.isfinite(v)]
        valid_val = [v for v in self._val_loss if math.isfinite(v)]
        if valid_train:
            result["final_train_loss"] = valid_train[-1]
            result["final_train_ref_loss"] = valid_train[-1]
        if valid_train_opt:
            result["final_train_opt_loss"] = valid_train_opt[-1]
        if valid_val:
            result["final_val_loss"] = valid_val[-1]
            result["final_val_ref_loss"] = valid_val[-1]
        if self._best_val is not None:
            result["best_val_loss"] = self._best_val
            result["best_val_ref_loss"] = self._best_val
        if self._best_epoch is not None:
            result["best_epoch"] = self._best_epoch
        if self._latest_lam_dir is not None:
            result["lambda_dir_eff"] = self._latest_lam_dir
        if self._latest_checkpoint_score is not None:
            result["checkpoint_score"] = self._latest_checkpoint_score
            result["best_metric_formula"] = self._best_formula
        if self._epochs_since_improvement is not None:
            result["epochs_since_improvement"] = self._epochs_since_improvement
        return result


class ImageGallery(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._header = QLabel("Result Plots")
        self._header.setStyleSheet(
            "font-weight: 600; color: #c4ccff; font-size: 13px; padding: 4px 2px;"
        )
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.setStyleSheet(
            "QTabBar::tab { padding: 5px 10px; font-size: 10px; max-width: 160px; "
            "white-space: nowrap; text-overflow: ellipsis; overflow: hidden; }"
            "QTabBar::scroller { width: 22px; }"
        )
        self._placeholder = QLabel(
            "Plots will appear here when evaluation completes."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #6b7394; padding: 32px; font-style: italic; font-size: 12px;"
        )
        lo = QVBoxLayout()
        lo.setContentsMargins(0, 6, 0, 0)
        lo.setSpacing(6)
        lo.addWidget(self._header)
        lo.addWidget(self._placeholder)
        lo.addWidget(self._tabs)
        self._tabs.setVisible(False)
        self.setLayout(lo)

    def load_from_directory(self, directory: str) -> int:
        self._tabs.clear()
        d = Path(directory)
        if not d.is_dir():
            self._placeholder.setText(f"Folder not found: {directory}")
            self._placeholder.setVisible(True)
            self._tabs.setVisible(False)
            return 0
        pngs = sorted(d.glob("*.png"), key=lambda p: p.name.lower())
        if not pngs:
            self._placeholder.setText("No .png files found in the output folder.")
            self._placeholder.setVisible(True)
            self._tabs.setVisible(False)
            return 0
        for img_path in pngs:
            pixmap = QPixmap(str(img_path))
            if pixmap.isNull():
                continue
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scaled = pixmap.scaled(
                QSize(900, 600),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl.setPixmap(scaled)
            lbl.setToolTip(str(img_path))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(lbl)
            tab_name = img_path.stem.replace("_", " ").title()
            if len(tab_name) > 22:
                tab_name = tab_name[:20] + "…"
            self._tabs.addTab(scroll, tab_name)
        self._placeholder.setVisible(False)
        self._tabs.setVisible(True)
        return len(pngs)

    def load_images(self, img_paths: List[Path]) -> int:
        """Load an ordered list of image paths (pre-sorted by caller)."""
        self._tabs.clear()
        loaded = 0
        for img_path in img_paths:
            pixmap = QPixmap(str(img_path))
            if pixmap.isNull():
                continue
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scaled = pixmap.scaled(
                QSize(900, 600),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl.setPixmap(scaled)
            lbl.setToolTip(str(img_path))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(lbl)
            tab_name = img_path.stem.replace("_", " ").title()
            sub = img_path.parent.name
            prefix = f"[{sub}] " if sub not in ("", "eval_results") else ""
            tab_name = prefix + (tab_name[:18] + "…" if len(tab_name) > 20 else tab_name)
            self._tabs.addTab(scroll, tab_name)
            loaded += 1
        if loaded:
            self._placeholder.setVisible(False)
            self._tabs.setVisible(True)
        else:
            self._placeholder.setText("No displayable images found in eval output.")
            self._placeholder.setVisible(True)
            self._tabs.setVisible(False)
        return loaded

    def clear_gallery(self) -> None:
        self._tabs.clear()
        self._placeholder.setText(
            "Plots will appear here when evaluation completes."
        )
        self._placeholder.setVisible(True)
        self._tabs.setVisible(False)


class ProcessPane(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self._on_parse_progress: Optional[Callable[[str], None]] = None
        self._display_filter: Optional[Callable[[str], bool]] = None
        self._on_finished_hook: Optional[Callable[[int, QProcess.ExitStatus], None]] = (
            None
        )
        self._stop_hint: str = ""
        self._raw_log_container: Optional[QWidget] = None

        self.status = QLabel("Ready")
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(_mono_font())
        self._highlighter = LogHighlighter(self.log.document())

        self._auto_scroll = QCheckBox("Auto-scroll")
        self._auto_scroll.setChecked(True)
        self._auto_scroll.setToolTip("When enabled, scrolls to the bottom as new lines arrive.")
        self._auto_scroll.setStyleSheet(
            "QCheckBox { font-size: 11px; color: #7480a8; }"
        )

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_clear = QPushButton("Clear Log")
        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.setProperty("kind", "ghost")
        self.btn_open_folder.setVisible(False)
        self._output_dir: str = ""

        self.btn_start.setProperty("kind", "primary")
        self.btn_stop.setProperty("kind", "danger")
        self.btn_clear.setProperty("kind", "ghost")
        self.btn_stop.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(10)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_open_folder)
        btn_row.addStretch(1)
        btn_row.addWidget(self._auto_scroll)
        btn_row.addWidget(self.btn_clear)

        self.status.setStyleSheet(
            "QLabel { color: #9aa7c7; font-size: 12px; padding: 2px 0; }"
        )

        _log_sep = QFrame()
        _log_sep.setFrameShape(QFrame.Shape.HLine)
        _log_sep.setFixedHeight(1)
        _log_sep.setStyleSheet(
            "background: rgba(185, 194, 221, 0.10); border: none; margin: 2px 0;"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addLayout(btn_row)
        layout.addWidget(_log_sep)
        layout.addWidget(self.log, 1)
        self.setLayout(layout)

        self.btn_clear.clicked.connect(self.log.clear)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_open_folder.clicked.connect(self._open_output_folder)

    def raw_log_widget(self) -> QWidget:
        """Return a standalone widget holding ONLY the raw log text plus a
        minimal toolbar (status + auto-scroll toggle + clear).

        Process controls (start/stop/progress/open-folder) are intentionally
        excluded so they can live in a dedicated training control bar. The
        widgets are re-parented out of this ProcessPane's own layout, so the
        pane itself should not be displayed once this is used."""
        if self._raw_log_container is not None:
            return self._raw_log_container
        container = QWidget()
        lo = QVBoxLayout()
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(6)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        bar.addWidget(self.status)
        bar.addStretch(1)
        bar.addWidget(self._auto_scroll)
        bar.addWidget(self.btn_clear)
        lo.addLayout(bar)
        lo.addWidget(self.log, 1)
        container.setLayout(lo)
        self._raw_log_container = container
        return container

    def set_output_dir(self, path: str) -> None:
        self._output_dir = path

    def set_progress_parser(self, fn: Optional[Callable[[str], None]]) -> None:
        self._on_parse_progress = fn

    def set_display_filter(self, fn: Optional[Callable[[str], bool]]) -> None:
        """Install a predicate deciding whether a line is shown in the log.

        The progress parser still receives every line; only the visible log is
        filtered. ``fn`` returning False hides the line. A None/raising filter
        shows everything (fail-open).
        """
        self._display_filter = fn

    def set_finished_hook(
        self, fn: Optional[Callable[[int, QProcess.ExitStatus], None]]
    ) -> None:
        self._on_finished_hook = fn

    def set_stop_hint(self, text: str = "") -> None:
        self._stop_hint = text.strip()

    def append(self, text: str) -> None:
        show = True
        if self._display_filter is not None:
            try:
                show = bool(self._display_filter(text))
            except Exception:
                show = True  # fail-open: never hide on filter error
        if show:
            self.log.appendPlainText(text.rstrip("\n"))
            if self._auto_scroll.isChecked():
                sb = self.log.verticalScrollBar()
                sb.setValue(sb.maximum())
        if self._on_parse_progress:
            try:
                self._on_parse_progress(text)
            except Exception:
                pass

    def start(
        self, program: str, args: list[str], workdir: Optional[str] = None
    ) -> None:
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Running", "A process is already running.")
            return
        self.log.clear()
        self.progress.setValue(0)
        self.btn_open_folder.setVisible(False)
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        if workdir:
            self.proc.setWorkingDirectory(workdir)
        self.append("> " + " ".join([program] + args) + "\n")
        self.status.setText("Running...")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.proc.readyReadStandardOutput.connect(self._on_ready_read)
        self.proc.finished.connect(self._on_finished)
        self.proc.setProgram(program)
        self.proc.setArguments(args)
        self.proc.start()
        if not self.proc.waitForStarted(3000):
            self.append("[ERROR] Failed to start process.")
            self.status.setText("Error")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def stop(self) -> None:
        if not self.proc or self.proc.state() == QProcess.ProcessState.NotRunning:
            return
        self.append("\n[UI] Stop requested...\n")
        if self._stop_hint:
            self.append(self._stop_hint + "\n")
        self.status.setText("Stopping...")

        # On Windows, kill the entire process tree (includes grandchild workers)
        # to prevent orphan subprocesses.
        pid = self.proc.processId()
        if platform.system() == "Windows" and pid:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, check=False,
                )
            except Exception:
                pass

        self.proc.terminate()

        def kill_if_needed():
            if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
                self.append("[UI] Force-killing process.\n")
                self.proc.kill()

        QTimer.singleShot(2000, kill_if_needed)

    def _on_ready_read(self) -> None:
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if data:
            for line in data.splitlines():
                self.append(line)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        st = "Done" if exit_status == QProcess.ExitStatus.NormalExit else "Crashed"
        self.status.setText(f"{st} | exit_code={exit_code}")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if exit_status == QProcess.ExitStatus.NormalExit:
            try:
                self.progress.setValue(self.progress.maximum())
            except Exception:
                pass
            if self._output_dir and Path(self._output_dir).is_dir():
                self.btn_open_folder.setVisible(True)
            _send_os_notification(
                "Lunar Potential Surrogate", f"Process finished (exit={exit_code})."
            )
        if self._on_finished_hook:
            try:
                self._on_finished_hook(exit_code, exit_status)
            except Exception:
                pass

    def _open_output_folder(self) -> None:
        if self._output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._output_dir))


def _inspect_run_artifacts(run_dir: str) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "run_dir": "",
        "manifest_path": None,
        "config_path": None,
        "scaler_path": None,
        "checkpoint_path": None,
        "best_epoch": None,
        "best_score": None,
        "architecture_signature": None,
        "w0_bands": None,
        "checkpoint_schema_version": None,
        "scaler_hash": None,
        "scaler_status": "unknown",
        "warnings": [],
        "source": "fallback",
    }
    if not run_dir or make_run_layout is None:
        return status
    try:
        resolved = (
            resolve_artifact_run_dir(run_dir)
            if resolve_artifact_run_dir is not None
            else Path(run_dir).expanduser().resolve()
        )
        layout = make_run_layout(Path(resolved))
    except Exception as exc:
        status["warnings"].append(f"run_dir_unusable: {exc}")
        return status

    status["run_dir"] = str(layout.run_dir)
    status["manifest_path"] = str(layout.run_manifest_json)
    status["config_path"] = str(layout.config_json)
    status["scaler_path"] = str(layout.scaler_json)

    manifest = read_run_manifest(layout) if read_run_manifest is not None else {}
    if manifest:
        status["source"] = "run_manifest"
    config_payload = _read_json_if_exists(layout.config_json)
    scaler_payload = _read_json_if_exists(layout.scaler_json)
    scaler_hash = manifest.get("scaler_hash")
    if not scaler_hash and scaler_payload and compute_payload_sha256 is not None:
        try:
            scaler_hash = compute_payload_sha256(scaler_payload)
        except Exception:
            scaler_hash = None
    status["scaler_hash"] = scaler_hash

    ckpt_path: Optional[Path] = None
    if layout.ckpt_best.exists():
        ckpt_path = layout.ckpt_best
    elif layout.ckpt_last.exists():
        ckpt_path = layout.ckpt_last
    if ckpt_path is None:
        status["warnings"].append("missing_checkpoint")
    else:
        status["checkpoint_path"] = str(ckpt_path)

    if not layout.scaler_json.exists():
        status["warnings"].append("missing_scaler")
        status["scaler_status"] = "missing"

    ckpt: Dict[str, Any] = {}
    if ckpt_path is not None and load_artifact_checkpoint is not None:
        try:
            import torch

            ckpt = load_artifact_checkpoint(ckpt_path, torch.device("cpu"))
        except Exception as exc:
            status["warnings"].append(f"checkpoint_load_failed: {exc}")
    status["checkpoint_schema_version"] = ckpt.get("schema_version") if ckpt else None
    status["best_epoch"] = (
        manifest.get("best_epoch")
        or ckpt.get("epoch_display")
        or ckpt.get("epoch")
    )
    status["best_score"] = manifest.get("best_score") or (ckpt.get("scoring") or {}).get("score")
    status["architecture_signature"] = (
        manifest.get("architecture_signature")
        or (ckpt.get("architecture") or {}).get("signature")
        or (ckpt.get("config") or {}).get("architecture_signature")
        or config_payload.get("architecture_signature")
    )
    status["w0_bands"] = (
        manifest.get("w0_bands")
        or (ckpt.get("architecture") or {}).get("w0_bands")
        or (ckpt.get("config") or {}).get("w0_bands")
        or config_payload.get("w0_bands")
    )

    if ckpt and ckpt.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        status["warnings"].append("legacy_checkpoint_schema")

    if scaler_payload and (ckpt.get("scaler") or None) and compute_payload_sha256 is not None:
        try:
            scaler_file_hash = compute_payload_sha256(scaler_payload)
            ckpt_scaler_hash = compute_payload_sha256(ckpt["scaler"])
            if scaler_file_hash == ckpt_scaler_hash:
                status["scaler_status"] = "match"
            else:
                status["scaler_status"] = "mismatch"
                status["warnings"].append("scaler_mismatch")
        except Exception:
            status["scaler_status"] = "unknown"
    elif layout.scaler_json.exists():
        status["scaler_status"] = "present"

    mismatch_fields: List[str] = []
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else {}
    if isinstance(config_payload, dict) and isinstance(ckpt_cfg, dict):
        for field in CRITICAL_CONFIG_FIELDS:
            if field in config_payload and field in ckpt_cfg and config_payload.get(field) != ckpt_cfg.get(field):
                mismatch_fields.append(field)
    if mismatch_fields:
        status["warnings"].append(
            "config_checkpoint_mismatch:" + ", ".join(mismatch_fields[:8])
        )

    if ckpt_path is None and not layout.ckpt_last.exists():
        status["warnings"].append("missing_best_and_last_checkpoint")

    return status

