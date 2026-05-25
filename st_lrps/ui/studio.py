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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys

_USE_PYSIDE = "PySide6" in sys.modules or ("PyQt6" not in sys.modules and ("PySide6" in sys.modules or True))

try:
    if _USE_PYSIDE:
        from PySide6.QtCore import (
            QEasingCurve,
            QEvent,
            QObject,
            QProcess,
            QProcessEnvironment,
            QPropertyAnimation,
            QSettings,
            QSize,
            Qt,
            QTimer,
            QUrl,
            Signal as pyqtSignal,
        )
        from PySide6.QtGui import (
            QColor,
            QDesktopServices,
            QFont,
            QGuiApplication,
            QIcon,
            QPalette,
            QPixmap,
            QSyntaxHighlighter,
            QTextCharFormat,
            QTextDocument,
        )
        from PySide6.QtWidgets import (
            QAbstractSpinBox,
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSpinBox,
            QSplitter,
            QStackedWidget,
            QSystemTrayIcon,
            QTabWidget,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )
    else:
        raise ImportError
except ImportError:
    from PyQt6.QtCore import (
        QEasingCurve,
        QEvent,
        QObject,
        QProcess,
        QProcessEnvironment,
        QPropertyAnimation,
        QSettings,
        QSize,
        Qt,
        QTimer,
        QUrl,
        pyqtSignal,
    )
    from PyQt6.QtGui import (
        QColor,
        QDesktopServices,
        QFont,
        QGuiApplication,
        QIcon,
        QPalette,
        QPixmap,
        QSyntaxHighlighter,
        QTextCharFormat,
        QTextDocument,
    )
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QSystemTrayIcon,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
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
_REPO_ROOT = SCRIPT_DIR.parents[1]
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


def _cfg_value(cfg: Any, name: str, fallback: Any) -> Any:
    """Read a default from the SSOT config object with a safe UI fallback."""

    return getattr(cfg, name, fallback) if cfg is not None else fallback

# QSettings organization identifiers
_SETTINGS_ORG = "ST_LRPS_Project"
_SETTINGS_APP = "ST_LRPS_Dashboard"


# =============================================================================
# 1. HELPERS
# =============================================================================


def _norm_path(p: str) -> str:
    return str(Path(p).expanduser().resolve()) if p else ""


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


# =============================================================================
# 2. DATASET INTROSPECTION (Feature #14)
# =============================================================================


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


# =============================================================================
# 3. REUSABLE UI COMPONENTS
# =============================================================================


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
        self, title: str = "Gelişmiş Ayarlar", parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._title = title
        self._toggle_btn = QPushButton(f"▸  {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setProperty("kind", "ghost")
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 8px 14px; font-weight: 600; "
            "color: #9aa7ff; border: 1px solid transparent; border-radius: 8px; "
            "background: rgba(124, 92, 255, 0.05); }"
            "QPushButton:hover { color: #c4ccff; background: rgba(124, 92, 255, 0.10); "
            "border-color: rgba(124, 92, 255, 0.20); }"
            "QPushButton:checked { color: #c4ccff; background: rgba(124, 92, 255, 0.12); "
            "border-color: rgba(124, 92, 255, 0.22); }"
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


# =============================================================================
# 4. LOG SYNTAX HIGHLIGHTER
# =============================================================================


class LogHighlighter(QSyntaxHighlighter):
    def __init__(self, parent: Optional[QTextDocument] = None):
        super().__init__(parent)
        self._rules: List[Tuple[re.Pattern, QTextCharFormat]] = []

        fmt_err = QTextCharFormat()
        fmt_err.setForeground(QColor("#f87171"))
        fmt_err.setFontWeight(QFont.Weight.Bold)
        for p in [
            r"(?i)\[HATA\]",
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


# =============================================================================
# 5. LIVE LOSS PLOT (Feature #13)
# =============================================================================


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
        # Row 1: başlık  |  kontroller
        # ----------------------------
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(3)
        title = QLabel("Canlı Eğitim İzleme")
        title.setObjectName("lossTitle")
        subtitle = QLabel("Eğitim / doğrulama kayıp eğrisi  ·  logaritmik ölçek önerilir")
        subtitle.setObjectName("lossSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        top_row.addLayout(title_col, 1)

        self._chk_log_y = QCheckBox("Log Y")
        self._chk_log_y.setChecked(True)
        self._chk_log_y.setToolTip(
            "Y eksenini logaritmik ölçekte gösterir.\n"
            "Kayıp değerleri birkaç büyüklük mertebesinde değiştiğinden bu görünüm önerilir."
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

        self._btn_fit = QPushButton("Otomatik Ölçek")
        self._btn_fit.setProperty("plotControl", True)
        self._btn_fit.setToolTip("Grafiği mevcut veriye otomatik olarak yeniden sığdırır.")
        self._btn_fit.clicked.connect(self._auto_range)
        top_row.addWidget(self._btn_fit)

        self._btn_clear = QPushButton("Sıfırla")
        self._btn_clear.setProperty("plotControl", True)
        self._btn_clear.setToolTip("Canlı grafikteki tüm kayıp geçmişini ve metrikleri sıfırlar.")
        self._btn_clear.clicked.connect(self.clear)
        top_row.addWidget(self._btn_clear)

        card_layout.addLayout(top_row)

        # ----------------------------
        # Row 2: metrik chip'leri (7 eşit genişlik)
        # ----------------------------
        self._lbl_train = self._metric_label("Eğitim opt/ref", "—")
        self._lbl_val   = self._metric_label("Doğrulama",      "—")
        self._lbl_best  = self._metric_label("En İyi Val",     "—")
        self._lbl_best_epoch  = self._metric_label("En İyi Epoch",   "—")
        self._lbl_no_improve  = self._metric_label("İyileşme Yok",   "—")
        self._lbl_lam_dir     = self._metric_label("λ Yön Ağrl.",    "—")
        self._lbl_lr          = self._metric_label("Öğr. Hızı",      "—")

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

        # durum etiketi (altta ortalanmış)
        self._lbl_status = QLabel("Eğitim bekleniyor…")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_status.setStyleSheet("color: #5a647a; font-size: 11px;")

        # ----------------------------
        # Plot body
        # ----------------------------
        if _HAS_PYQTGRAPH:
            pg.setConfigOptions(antialias=True, background=None, foreground="#b9c2dd")

            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setMinimumHeight(360)
            self._plot_widget.setStyleSheet(
                """
                PlotWidget {
                    background-color: rgba(4, 8, 18, 0.72);
                    border: 1px solid rgba(185, 194, 221, 0.11);
                    border-radius: 14px;
                }
                """
            )
            self._plot_widget.setBackground("#050915")
            self._plot_widget.setMenuEnabled(False)
            self._plot_widget.showGrid(x=True, y=True, alpha=0.18)
            self._plot_widget.setLabel("left", "Loss", color="#aeb8d8", size="10pt")
            self._plot_widget.setLabel("bottom", "Epoch", color="#aeb8d8", size="10pt")
            self._plot_widget.setLogMode(x=False, y=True)

            plot_item = self._plot_widget.getPlotItem()
            plot_item.setContentsMargins(8, 10, 12, 8)
            plot_item.hideButtons()
            for axis_name in ("left", "bottom"):
                axis = self._plot_widget.getAxis(axis_name)
                axis.setTextPen(pg.mkPen("#aeb8d8"))
                axis.setPen(pg.mkPen("#3c4664"))
                axis.setStyle(
                    tickFont=QFont("Consolas", 8),
                    autoExpandTextSpace=True,
                    tickTextOffset=8,
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

            self._direction_plot = pg.PlotWidget()
            self._direction_plot.setMinimumHeight(230)
            self._direction_plot.setBackground("#050915")
            self._direction_plot.setMenuEnabled(False)
            self._direction_plot.showGrid(x=True, y=True, alpha=0.18)
            self._direction_plot.setLabel("left", "Loss", color="#aeb8d8", size="10pt")
            self._direction_plot.setLabel("bottom", "Epoch", color="#aeb8d8", size="10pt")
            self._direction_plot.setLogMode(x=False, y=True)
            self._curve_train_loss_a = self._direction_plot.plot([], [], pen=pg.mkPen(color="#34d399", width=2.1), name="train_a")
            self._curve_val_loss_a = self._direction_plot.plot([], [], pen=pg.mkPen(color="#10b981", width=2.4), name="val_a")
            self._curve_train_dir = self._direction_plot.plot([], [], pen=pg.mkPen(color="#fbbf24", width=2.1), name="train_direction")
            self._curve_val_dir = self._direction_plot.plot([], [], pen=pg.mkPen(color="#f59e0b", width=2.4), name="val_direction")

            self._direction_quality_plot = pg.PlotWidget()
            self._direction_quality_plot.setMinimumHeight(180)
            self._direction_quality_plot.setBackground("#050915")
            self._direction_quality_plot.setMenuEnabled(False)
            self._direction_quality_plot.showGrid(x=True, y=True, alpha=0.18)
            self._direction_quality_plot.setLabel("left", "Quality", color="#aeb8d8", size="10pt")
            self._direction_quality_plot.setLabel("bottom", "Epoch", color="#aeb8d8", size="10pt")
            self._curve_val_angular = self._direction_quality_plot.plot([], [], pen=pg.mkPen(color="#3b82f6", width=2.4), name="val_angular_deg")
            self._curve_train_cossim = self._direction_quality_plot.plot([], [], pen=pg.mkPen(color="#c084fc", width=2.1), name="train_cos_sim")
            self._curve_val_cossim = self._direction_quality_plot.plot([], [], pen=pg.mkPen(color="#8b5cf6", width=2.4), name="val_cos_sim")

            self._direction_tab = QWidget()
            direction_layout = QVBoxLayout()
            direction_layout.setContentsMargins(0, 0, 0, 0)
            direction_layout.setSpacing(6)
            direction_layout.addWidget(self._direction_plot, 3)
            direction_layout.addWidget(self._direction_quality_plot, 2)
            self._direction_tab.setLayout(direction_layout)

            self._checkpoint_plot = pg.PlotWidget()
            self._checkpoint_plot.setMinimumHeight(360)
            self._checkpoint_plot.setBackground("#050915")
            self._checkpoint_plot.setMenuEnabled(False)
            self._checkpoint_plot.showGrid(x=True, y=True, alpha=0.18)
            self._checkpoint_plot.setLabel("left", "Checkpoint score", color="#aeb8d8", size="10pt")
            self._checkpoint_plot.setLabel("bottom", "Epoch", color="#aeb8d8", size="10pt")
            self._checkpoint_plot.setLogMode(x=False, y=True)
            self._curve_score = self._checkpoint_plot.plot([], [], pen=pg.mkPen(color="#34d399", width=2.6), name="Score")
            self._curve_best_score = self._checkpoint_plot.plot([], [], pen=pg.mkPen(color="#f472b6", width=2.2, style=pg_QtCore.Qt.PenStyle.DashLine), name="Best")

            self._plot_tabs = QTabWidget()
            self._plot_tabs.setDocumentMode(True)
            self._plot_tabs.setMinimumHeight(430)
            self._plot_tabs.addTab(self._plot_widget, "Loss overview")
            self._plot_tabs.addTab(self._direction_tab, "Acceleration / direction")
            self._plot_tabs.addTab(self._checkpoint_plot, "Checkpoint score")
            card_layout.addWidget(self._plot_tabs, 1)
        else:
            self._plot_widget = None
            self._direction_plot = None
            self._checkpoint_plot = None
            placeholder = QLabel(
                "pyqtgraph yüklü değil — canlı grafik devre dışı.\n"
                "Yüklemek için:  pip install pyqtgraph"
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
        the KPI/time strips) so the plot itself gets the vertical space."""
        for w in (self._metrics_row1, self._metrics_row2, self._help_label):
            w.setVisible(not compact)
        if compact:
            # Let the chart shrink/grow freely instead of forcing a tall card.
            if getattr(self, "_plot_tabs", None) is not None:
                self._plot_tabs.setMinimumHeight(260)
            for plot in (
                getattr(self, "_plot_widget", None),
                getattr(self, "_direction_plot", None),
                getattr(self, "_direction_quality_plot", None),
                getattr(self, "_checkpoint_plot", None),
            ):
                if plot is not None:
                    plot.setMinimumHeight(180)

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
            self._lbl_status.setText(self._checkpoint_status or "Bekleniyor")
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
        self._lbl_train.setText(_chip("Eğitim opt/ref", f"{opt_s} / {ref_s}"))
        self._lbl_val.setText(_chip("Doğrulama", self._fmt_metric(self._latest_val_ref)))
        self._lbl_best.setText(_chip("En İyi Val", self._fmt_metric(self._best_val)))
        self._lbl_best_epoch.setText(_chip(
            "En İyi Epoch",
            str(self._best_epoch) if self._best_epoch is not None else "—",
        ))
        self._lbl_no_improve.setText(_chip(
            "İyileşme Yok",
            str(self._epochs_since_improvement) if self._epochs_since_improvement is not None else "—",
        ))
        self._lbl_lam_dir.setText(_chip("λ Yön Ağrl.", self._fmt_metric(self._latest_lam_dir)))
        self._lbl_lr.setText(_chip("Öğr. Hızı", self._fmt_metric(latest_lr)))

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


# =============================================================================
# 6. IMAGE GALLERY (retained from v2)
# =============================================================================


class ImageGallery(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._header = QLabel("Sonuç Grafikleri")
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
            "Değerlendirme tamamlandığında grafikler burada görünecek."
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
            self._placeholder.setText(f"Klasör bulunamadı: {directory}")
            self._placeholder.setVisible(True)
            self._tabs.setVisible(False)
            return 0
        pngs = sorted(d.glob("*.png"), key=lambda p: p.name.lower())
        if not pngs:
            self._placeholder.setText("Çıktı klasöründe .png dosyası bulunamadı.")
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
            "Değerlendirme tamamlandığında grafikler burada görünecek."
        )
        self._placeholder.setVisible(True)
        self._tabs.setVisible(False)


# =============================================================================
# 7. PROCESS PANE (enhanced)
# =============================================================================


class ProcessPane(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self._on_parse_progress: Optional[Callable[[str], None]] = None
        self._on_finished_hook: Optional[Callable[[int, QProcess.ExitStatus], None]] = (
            None
        )
        self._stop_hint: str = ""
        self._raw_log_container: Optional[QWidget] = None

        self.status = QLabel("Hazır")
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

        self._auto_scroll = QCheckBox("Otomatik Kaydırma")
        self._auto_scroll.setChecked(True)
        self._auto_scroll.setToolTip("Aktifken yeni satırlarda otomatik alta kayar.")
        self._auto_scroll.setStyleSheet(
            "QCheckBox { font-size: 11px; color: #7480a8; }"
        )

        self.btn_start = QPushButton("Başlat")
        self.btn_stop = QPushButton("Durdur")
        self.btn_clear = QPushButton("Log Temizle")
        self.btn_open_folder = QPushButton("Çıktı Klasörünü Aç")
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

    def set_finished_hook(
        self, fn: Optional[Callable[[int, QProcess.ExitStatus], None]]
    ) -> None:
        self._on_finished_hook = fn

    def set_stop_hint(self, text: str = "") -> None:
        self._stop_hint = text.strip()

    def append(self, text: str) -> None:
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
            QMessageBox.warning(self, "Çalışıyor", "Zaten bir süreç çalışıyor.")
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
        self.status.setText("Çalışıyor...")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.proc.readyReadStandardOutput.connect(self._on_ready_read)
        self.proc.finished.connect(self._on_finished)
        self.proc.setProgram(program)
        self.proc.setArguments(args)
        self.proc.start()
        if not self.proc.waitForStarted(3000):
            self.append("[HATA] Süreç başlatılamadı.")
            self.status.setText("Hata")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def stop(self) -> None:
        if not self.proc or self.proc.state() == QProcess.ProcessState.NotRunning:
            return
        self.append("\n[UI] Durdurma istendi...\n")
        if self._stop_hint:
            self.append(self._stop_hint + "\n")
        self.status.setText("Durduruluyor...")

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
                self.append("[UI] Zorla sonlandırılıyor (kill).\n")
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
        st = "Bitti" if exit_status == QProcess.ExitStatus.NormalExit else "Çöktü"
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
                "Lunar Potential Surrogate", f"İşlem tamamlandı (exit={exit_code})."
            )
        if self._on_finished_hook:
            try:
                self._on_finished_hook(exit_code, exit_status)
            except Exception:
                pass

    def _open_output_folder(self) -> None:
        if self._output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._output_dir))


# =============================================================================
# 8. PRESET / PROFILE SYSTEM (retained from v2)
# =============================================================================

def _base_preset(**overrides) -> Dict[str, Any]:
    base = {
        "dataset_mode": "single",
        "hidden": 512, "depth": 5, "activation": "sine",
        "w0_first": 30.0, "w0_hidden": 30.0, "dropout": 0.0,
        "use_fourier": False, "fourier_n": 256, "fourier_sigma": 1.0,
        "fourier_append_raw": True,
        "epochs": 200, "batch_size": 8192,
        "lr": 1e-4, "weight_decay": 1e-6, "output_head_lr_mult": 1.0,
        "t_max": 200, "warmup_epochs": 5, "min_lr_ratio": 0.05,
        "patience": 30, "no_amp": False,
        "w_u": 1.0, "w_a": 1.0, "gradnorm_mode": "ntk_init",
        "gradnorm_w_a_min": 0.05, "gradnorm_w_a_max": 2.0,
        "potential_only_epochs": 0, "accel_ramp_epochs": 80,
        "accel_min_factor": 0.05,
        "a_sign": "auto", "use_si_index": 0,
        "direction_loss_weight": 0.10, "direction_loss_start_epoch": 30,
        "direction_loss_ramp_epochs": 50, "direction_loss_floor_abs": 3e-6,
        "best_ckpt_start_epoch": -1, "checkpoint_settle_epochs": 5,
        "use_altitude_balanced_loss": False,
        "altitude_bin_width_km": 50.0,
        "altitude_min_km": _cfg_value(DEFAULT_SPATIAL_CLOUD_CONFIG, "alt_min_km", 200.0),
        "altitude_max_km": _cfg_value(DEFAULT_SPATIAL_CLOUD_CONFIG, "alt_max_km", 600.0),
        "resume_enabled": False, "resume_from": "",
        "resume_checkpoint": "last", "resume_nonstrict": False,
        "resume_history_mode": "append",
        "use_radial_cross_loss": False,
        "radial_loss_weight": 0.0,
        "cross_loss_weight": 0.0,
        "use_laplacian_regularization": False,
        "laplacian_weight": 0.0,
        "laplacian_every_n_batches": 5,
        "laplacian_subset_size": 512,
        "max_grad_norm": 0.5, "num_workers": 2, "cache_rows": 65536,
        "fit_rows": 500_000, "seed": 42, "split_seed": 42,
        "log_every": 50, "preload_data": False, "auto_preload_mb": 256.0,
        "pin_memory": True, "quick_check": False, "extra_args": "",
    }
    base.update(overrides)
    return base


_BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "Quick Debug": _base_preset(
        hidden=64, depth=2, epochs=5, batch_size=1024,
        lr=1e-3, weight_decay=0.0, t_max=5, warmup_epochs=0,
        patience=5, num_workers=0, cache_rows=8192, fit_rows=50_000,
        direction_loss_weight=0.0, quick_check=True,
    ),
    "Default SIREN": _base_preset(),
    "Physics-Strong SIREN": _base_preset(
        use_altitude_balanced_loss=True,
        use_radial_cross_loss=True,
        radial_loss_weight=0.05,
        cross_loss_weight=0.05,
        direction_loss_weight=0.10,
        direction_loss_start_epoch=40,
        direction_loss_ramp_epochs=50,
    ),
    "Laplacian Experiment": _base_preset(
        use_laplacian_regularization=True,
        laplacian_weight=1e-5,
        laplacian_every_n_batches=5,
        laplacian_subset_size=512,
    ),
    "SiLU + Fourier": _base_preset(
        activation="silu", use_fourier=True, fourier_n=256, fourier_sigma=1.0,
        hidden=512, depth=4, lr=2e-4, weight_decay=1e-6, batch_size=8192,
    ),
    "Stable SIREN": _base_preset(
        hidden=512, depth=5, activation="sine", w0_first=30.0, w0_hidden=30.0,
        lr=1e-4, weight_decay=1e-6, output_head_lr_mult=1.0, max_grad_norm=0.5,
        warmup_epochs=5, potential_only_epochs=0, accel_ramp_epochs=80,
        accel_min_factor=0.05,
        gradnorm_mode="ntk_init", gradnorm_w_a_min=0.05, gradnorm_w_a_max=2.0,
        laplacian_weight=0.0, radial_loss_weight=0.0, cross_loss_weight=0.0,
        direction_loss_weight=0.05, direction_loss_start_epoch=30,
        direction_loss_ramp_epochs=50, batch_size=8192,
    ),
}


def _load_user_presets() -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    if not _PRESETS_DIR.is_dir():
        return presets
    for fp in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                presets[fp.stem] = json.load(f)
        except Exception:
            pass
    return presets


def _save_user_preset(name: str, data: Dict[str, Any]) -> None:
    _PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_PRESETS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _delete_user_preset(name: str) -> bool:
    fp = _PRESETS_DIR / f"{name}.json"
    if fp.is_file():
        fp.unlink()
        return True
    return False


# =============================================================================
# 9. TRAINING QUEUE (Feature #15)
# =============================================================================


class TrainingQueue(QWidget):
    """
    Sequential training queue — users can enqueue multiple configurations
    and the queue executes them one-by-one.  Each item stores a full
    argument list (List[str]) that will be passed to QProcess.

    Workflow:
    1. User configures parameters, clicks "Kuyruğa Ekle".
    2. Item appears in the list with a short description.
    3. User clicks "Kuyruğu Başlat" — queue runs jobs sequentially.
    4. On each job finish, the next one starts automatically.
    """

    # Emitted when a job from the queue starts: (job_index, args_list)
    job_started = pyqtSignal(int, list)
    # Emitted when entire queue is done
    queue_finished = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._queue: List[Dict[str, Any]] = []  # [{label, args, out_dir, config}, ...]
        self._current_index: int = -1
        self._running: bool = False

        # --- Header ---
        lbl = QLabel("Eğitim Kuyruğu")
        lbl.setStyleSheet("font-weight: 600; color: #c4ccff; font-size: 13px; padding: 2px 0;")

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #7480a8; font-size: 11px;")

        # --- List ---
        self._list = QListWidget()
        self._list.setMinimumHeight(80)

        # --- Buttons ---
        self.btn_start_queue = QPushButton("Kuyruğu Başlat")
        self.btn_start_queue.setProperty("kind", "primary")
        self.btn_start_queue.setToolTip("Kuyruktaki tüm eğitimleri sırasıyla çalıştır.")
        self.btn_start_queue.clicked.connect(self._start_queue)

        self.btn_stop_queue = QPushButton("Kuyruğu Durdur")
        self.btn_stop_queue.setProperty("kind", "danger")
        self.btn_stop_queue.setEnabled(False)
        self.btn_stop_queue.clicked.connect(self._stop_queue)

        btn_remove = QPushButton("Seçiliyi Kaldır")
        btn_remove.setProperty("kind", "ghost")
        btn_remove.clicked.connect(self._remove_selected)

        btn_clear_q = QPushButton("Kuyruğu Temizle")
        btn_clear_q.setProperty("kind", "ghost")
        btn_clear_q.clicked.connect(self._clear_queue)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_start_queue)
        btn_row.addWidget(self.btn_stop_queue)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_remove)
        btn_row.addWidget(btn_clear_q)

        lo = QVBoxLayout()
        lo.setContentsMargins(0, 8, 0, 0)
        lo.setSpacing(6)
        lo.addWidget(lbl)
        lo.addWidget(self._status_lbl)
        lo.addWidget(self._list, 1)
        lo.addLayout(btn_row)
        self.setLayout(lo)
        self._update_status()

    # --- Public API (called by STLRPSTrainTab) ---

    def enqueue(
        self,
        label: str,
        args: List[str],
        out_dir: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a training job to the queue."""
        item_data = {
            "label": label,
            "args": args,
            "out_dir": out_dir,
            "config": config or {},
            "status": "Bekliyor",
        }
        self._queue.append(item_data)
        self._refresh_list()
        self._update_status()

    def is_running(self) -> bool:
        return self._running

    def current_args(self) -> Optional[List[str]]:
        if 0 <= self._current_index < len(self._queue):
            return self._queue[self._current_index]["args"]
        return None

    def on_job_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Called by STLRPSTrainTab when the current subprocess finishes."""
        if not self._running or self._current_index < 0:
            return

        job_ok = exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
        if job_ok:
            self._queue[self._current_index]["status"] = f"Tamamlandı (exit={exit_code})"
        else:
            self._queue[self._current_index]["status"] = f"Hata (exit={exit_code})"

        self._refresh_list()

        if not job_ok:
            reply = QMessageBox.question(
                self,
                "Eğitim Başarısız",
                f"İş #{self._current_index + 1} başarısız bitti (exit={exit_code}).\n"
                "Kuyruğa devam edilsin mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._running = False
                self._current_index = -1
                self.btn_start_queue.setEnabled(True)
                self.btn_stop_queue.setEnabled(False)
                self._update_status()
                return

        self._advance_queue()

    # --- Internal ---

    def _start_queue(self) -> None:
        pending = [i for i, q in enumerate(self._queue) if q["status"] == "Bekliyor"]
        if not pending:
            QMessageBox.information(self, "Boş Kuyruk", "Bekleyen iş yok.")
            return
        self._running = True
        self._current_index = pending[0] - 1  # _advance will increment
        self.btn_start_queue.setEnabled(False)
        self.btn_stop_queue.setEnabled(True)
        self._advance_queue()

    def _advance_queue(self) -> None:
        # Find next pending job
        next_idx = None
        for i in range(self._current_index + 1, len(self._queue)):
            if self._queue[i]["status"] == "Bekliyor":
                next_idx = i
                break

        if next_idx is None:
            # Queue done
            self._running = False
            self._current_index = -1
            self.btn_start_queue.setEnabled(True)
            self.btn_stop_queue.setEnabled(False)
            self._update_status()
            _send_os_notification("Lunar Potential Surrogate", "Eğitim kuyruğu tamamlandı!")
            self.queue_finished.emit()
            return

        self._current_index = next_idx
        self._queue[next_idx]["status"] = "Çalışıyor…"
        self._refresh_list()
        self._update_status()
        self.job_started.emit(next_idx, self._queue[next_idx]["args"])

    def _stop_queue(self) -> None:
        self._running = False
        if 0 <= self._current_index < len(self._queue):
            self._queue[self._current_index]["status"] = "Durduruldu"
        self._current_index = -1
        self.btn_start_queue.setEnabled(True)
        self.btn_stop_queue.setEnabled(False)
        self._refresh_list()
        self._update_status()

    def _remove_selected(self) -> None:
        row = self._list.currentRow()
        if 0 <= row < len(self._queue):
            if self._running and row == self._current_index:
                QMessageBox.warning(
                    self, "Kaldırılamaz", "Çalışmakta olan iş kaldırılamaz."
                )
                return
            self._queue.pop(row)
            if self._current_index > row:
                self._current_index -= 1
            self._refresh_list()
            self._update_status()

    def _clear_queue(self) -> None:
        if self._running:
            QMessageBox.warning(self, "Temizlenemez", "Kuyruk çalışırken temizlenemez.")
            return
        self._queue.clear()
        self._current_index = -1
        self._refresh_list()
        self._update_status()

    def _refresh_list(self) -> None:
        self._list.clear()
        for i, job in enumerate(self._queue):
            icon = {"Bekliyor": "⏳", "Çalışıyor…": "▶️", "Durduruldu": "⏹️"}.get(
                job["status"], "✅" if "Tamamlandı" in job["status"] else "❌"
            )
            text = f"{icon}  [{i + 1}] {job['label']}  —  {job['status']}"
            item = QListWidgetItem(text)
            if "Çalışıyor" in job["status"]:
                item.setForeground(QColor("#c084fc"))
            elif "Tamamlandı" in job["status"]:
                item.setForeground(QColor("#34d399"))
            elif "Hata" in job["status"] or "Durduruldu" in job["status"]:
                item.setForeground(QColor("#f87171"))
            self._list.addItem(item)

    def _update_status(self) -> None:
        pending = sum(1 for q in self._queue if q["status"] == "Bekliyor")
        done = sum(1 for q in self._queue if "Tamamlandı" in q["status"])
        total = len(self._queue)
        if self._running:
            self._status_lbl.setText(
                f"Çalışıyor: {self._current_index + 1}/{total}  |  Bekleyen: {pending}"
            )
        elif total > 0:
            self._status_lbl.setText(
                f"Toplam: {total}  |  Tamamlanan: {done}  |  Bekleyen: {pending}"
            )
        else:
            self._status_lbl.setText(
                "No queued runs — add the current profile to batch-train multiple experiments."
            )


# =============================================================================
# 10. ST-LRPS TRAIN TAB
# =============================================================================


class STLRPSTrainTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # =====================================================================
        # PRESET BAR
        # =====================================================================
        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(200)
        self._preset_combo.setToolTip("Kaydedilmiş hiperparametre profilleri.")
        self._refresh_preset_list()

        btn_load_preset = QPushButton("Yükle")
        btn_load_preset.clicked.connect(self._load_preset)
        btn_save_preset = QPushButton("Kaydet")
        btn_save_preset.clicked.connect(self._save_preset)
        btn_del_preset = QPushButton("Sil")
        btn_del_preset.setProperty("kind", "danger")
        btn_del_preset.clicked.connect(self._delete_preset)

        preset_bar = QHBoxLayout()
        preset_bar.setContentsMargins(4, 0, 4, 0)
        preset_bar.setSpacing(8)
        preset_lbl = QLabel("Profil:")
        preset_lbl.setStyleSheet("font-weight: 600; color: #c4ccff; font-size: 13px;")
        preset_bar.addWidget(preset_lbl)
        preset_bar.addWidget(self._preset_combo, 1)
        preset_bar.addWidget(btn_load_preset)
        preset_bar.addWidget(btn_save_preset)
        preset_bar.addWidget(btn_del_preset)

        # ── Workflow Mode ──────────────────────────────────────────────
        self.workflow_mode = QComboBox()
        self.workflow_mode.addItem("🚂  Train only",              "train_only")
        self.workflow_mode.addItem("📊  Evaluate only",           "eval_only")
        self.workflow_mode.addItem("🚀  Train then evaluate",     "train_then_eval")
        self.workflow_mode.addItem("📋  Queue training runs",     "queue")
        self.workflow_mode.setCurrentIndex(2)  # default: Train then evaluate
        self.workflow_mode.setToolTip(
            f"Train only:         Runs python -m {TRAIN_CLI_MODULE}.\n"
            f"Evaluate only:      Runs python -m {EVAL_CLI_MODULE} (existing model folder required).\n"
            "Train then eval:    Eğitim biter bitmez otomatik olarak değerlendirme başlatılır.\n"
            "Queue:              Kuyruktaki tüm işler sırayla çalıştırılır."
        )
        self.workflow_mode.currentIndexChanged.connect(self._on_workflow_mode_changed)
        wf_lbl = QLabel("Workflow:")
        wf_lbl.setStyleSheet("font-weight: 600; color: #fbbf24; font-size: 13px;")
        workflow_bar = QHBoxLayout()
        workflow_bar.setContentsMargins(4, 2, 4, 2)
        workflow_bar.setSpacing(8)
        workflow_bar.addWidget(wf_lbl)
        workflow_bar.addWidget(self.workflow_mode, 1)

        # Readiness checklist (compact, shown above Start)
        self._checklist_label = QLabel("")
        self._checklist_label.setWordWrap(True)
        self._checklist_label.setStyleSheet(
            "QLabel { font-size: 11px; color: #9aa7c7; "
            "background: rgba(10,16,31,0.7); border-radius: 6px; padding: 4px 8px; }"
        )
        self._checklist_label.setVisible(False)

        # =====================================================================
        # GROUP 1: Data & I/O
        # =====================================================================
        grp_data = QGroupBox("Veri ve Giriş/Çıkış")
        form_data = QFormLayout()
        _tune_form(form_data)

        self.dataset_mode = QComboBox()
        self.dataset_mode.addItem("Single dataset + internal split", "single")
        self.dataset_mode.addItem("Independent train/val/test/OOD datasets", "independent")
        self.dataset_mode.setToolTip(
            f"Single mode passes --data and lets python -m {TRAIN_CLI_MODULE} split train/val. "
            "Independent mode passes --train-data and --val-data explicitly."
        )

        self.data = ValidatedPathEdit(
            placeholder="Empty -> latest .h5 is discovered automatically", check_file=True
        )
        self.data.setToolTip("Single-dataset HDF5 path passed as --data.")
        btn_data = QPushButton("Select...")
        btn_data.clicked.connect(self._pick_data)
        data_row = _row_lineedit_with_button(self.data, btn_data)

        # Feature #14: dataset info label
        self._ds_info = DatasetInfoLabel()
        self.data.path_validated.connect(self._on_data_path_validated)

        self.train_data = ValidatedPathEdit(placeholder="Required in independent mode", check_file=True)
        self.val_data = ValidatedPathEdit(placeholder="Required in independent mode", check_file=True)
        self.test_data = ValidatedPathEdit(placeholder="Optional independent in-band test cloud", check_file=True)
        self.ood_data = ValidatedPathEdit(placeholder="Optional OOD/extrapolation cloud", check_file=True)
        btn_train_data = QPushButton("Select...")
        btn_val_data = QPushButton("Select...")
        btn_test_data = QPushButton("Select...")
        btn_ood_data = QPushButton("Select...")
        btn_train_data.clicked.connect(lambda: self._pick_dataset_path(self.train_data, "Select train dataset"))
        btn_val_data.clicked.connect(lambda: self._pick_dataset_path(self.val_data, "Select validation dataset"))
        btn_test_data.clicked.connect(lambda: self._pick_dataset_path(self.test_data, "Select test dataset"))
        btn_ood_data.clicked.connect(lambda: self._pick_dataset_path(self.ood_data, "Select OOD dataset"))
        self._train_ds_info = DatasetInfoLabel()
        self._val_ds_info = DatasetInfoLabel()
        self._test_ds_info = DatasetInfoLabel()
        self._ood_ds_info = DatasetInfoLabel()
        self.train_data.path_validated.connect(
            lambda path, exists: self._on_dataset_path_validated(path, exists, self._train_ds_info, update_primary=True)
        )
        self.val_data.path_validated.connect(
            lambda path, exists: self._on_dataset_path_validated(path, exists, self._val_ds_info)
        )
        self.test_data.path_validated.connect(
            lambda path, exists: self._on_dataset_path_validated(path, exists, self._test_ds_info)
        )
        self.ood_data.path_validated.connect(
            lambda path, exists: self._on_dataset_path_validated(path, exists, self._ood_ds_info)
        )

        self._single_data_widget = QWidget()
        single_form = QFormLayout()
        _tune_form(single_form)
        single_form.addRow("Dataset (.h5)", data_row)
        single_form.addRow("", self._ds_info)
        self._single_data_widget.setLayout(single_form)

        self._independent_data_widget = QWidget()
        independent_form = QFormLayout()
        _tune_form(independent_form)
        independent_form.addRow("Train Dataset", _row_lineedit_with_button(self.train_data, btn_train_data))
        independent_form.addRow("", self._train_ds_info)
        independent_form.addRow("Validation Dataset", _row_lineedit_with_button(self.val_data, btn_val_data))
        independent_form.addRow("", self._val_ds_info)
        independent_form.addRow("Test Dataset", _row_lineedit_with_button(self.test_data, btn_test_data))
        independent_form.addRow("", self._test_ds_info)
        independent_form.addRow("OOD Dataset", _row_lineedit_with_button(self.ood_data, btn_ood_data))
        independent_form.addRow("", self._ood_ds_info)
        self._independent_data_widget.setLayout(independent_form)

        self.out_dir = ValidatedPathEdit(
            placeholder="Empty -> automatic timestamped run folder", check_file=False
        )
        self.out_dir.setToolTip("Output folder. Empty -> runs/st_lrps_train_<timestamp>.")
        btn_out = QPushButton("Select...")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)

        self.dataset_name = QLineEdit("data")
        self.dataset_name.setToolTip("HDF5 dataset adı.")

        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setDecimals(4)
        self.val_ratio.setRange(0.0, 0.5)
        self.val_ratio.setValue(0.1)
        self.val_ratio.setSingleStep(0.01)
        self.val_ratio.setToolTip("Validation fraction in single-dataset mode. 0.1 -> 10% val, 90% train.")

        self._suite_manifest_label = QLabel("(no suite applied)")
        self._suite_manifest_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self._suite_manifest_label.setWordWrap(True)

        form_data.addRow("Dataset Mode", self.dataset_mode)
        form_data.addRow(self._single_data_widget)
        form_data.addRow(self._independent_data_widget)
        form_data.addRow("Output Folder", out_row)
        form_data.addRow("HDF5 Dataset Name", self.dataset_name)
        form_data.addRow("Validation Fraction", self.val_ratio)
        form_data.addRow("Suite Manifest", self._suite_manifest_label)
        grp_data.setLayout(form_data)
        self.dataset_mode.currentIndexChanged.connect(self._on_dataset_mode_changed)

        # =====================================================================
        # GROUP 1B: Resume Training
        # =====================================================================
        self.resume_section = CollapsibleSection("Resume Training")
        form_resume = QFormLayout()
        _tune_form(form_resume)

        self.resume_enabled = QCheckBox("Resume existing run")
        self.resume_from = ValidatedPathEdit(
            placeholder="Run directory, checkpoints/ directory, or .pt checkpoint",
            check_file=False,
        )
        btn_resume_run = QPushButton("Select Run...")
        btn_resume_run.clicked.connect(self._pick_resume_run)
        btn_resume_ckpt = QPushButton("Select Checkpoint...")
        btn_resume_ckpt.clicked.connect(self._pick_resume_checkpoint)
        resume_path_row = QHBoxLayout()
        resume_path_row.setContentsMargins(0, 0, 0, 0)
        resume_path_row.setSpacing(6)
        resume_path_row.addWidget(self.resume_from, 1)
        resume_path_row.addWidget(btn_resume_run)
        resume_path_row.addWidget(btn_resume_ckpt)
        resume_path_widget = QWidget()
        resume_path_widget.setLayout(resume_path_row)
        self._resume_path_buttons = (btn_resume_run, btn_resume_ckpt)

        self.resume_checkpoint = QComboBox()
        self.resume_checkpoint.addItem("last", "last")
        self.resume_checkpoint.addItem("best", "best")
        self.resume_checkpoint.setCurrentIndex(0)

        self.resume_nonstrict = QCheckBox("Allow non-critical config differences")
        self.resume_nonstrict.setChecked(False)

        self.resume_history_mode = QComboBox()
        self.resume_history_mode.addItem("append previous history", "append")
        self.resume_history_mode.addItem("overwrite history", "overwrite")
        self.resume_history_mode.setCurrentIndex(0)

        resume_help = QLabel(
            "Resume defaults to ckpt_last.pt. --epochs is the total target epoch count, "
            "not additional epochs. Resume is epoch-level; if interrupted mid-epoch, "
            "training resumes from the last completed checkpoint."
        )
        resume_help.setWordWrap(True)
        resume_help.setStyleSheet("color: #94a3b8; font-size: 11px;")

        form_resume.addRow(self.resume_enabled)
        form_resume.addRow("Resume From", resume_path_widget)
        form_resume.addRow("Checkpoint", self.resume_checkpoint)
        form_resume.addRow(self.resume_nonstrict)
        form_resume.addRow("History", self.resume_history_mode)
        form_resume.addRow("", resume_help)
        resume_inner = QWidget()
        resume_inner.setLayout(form_resume)
        _tune_inputs(resume_inner)
        resume_vbox = QVBoxLayout()
        resume_vbox.setContentsMargins(0, 0, 0, 0)
        resume_vbox.addWidget(resume_inner)
        self.resume_section.set_content_layout(resume_vbox)
        self.resume_enabled.toggled.connect(self._on_resume_toggled)
        self.resume_from.textChanged.connect(self._refresh_checklist)

        # =====================================================================
        # GROUP 2: Model Architecture
        # =====================================================================
        grp_arch = QGroupBox("Model Mimarisi")
        form_arch = QFormLayout()
        _tune_form(form_arch)

        self.hidden = QSpinBox()
        self.hidden.setRange(1, 8192)
        self.hidden.setValue(512)
        self.hidden.setToolTip(
            "Her gizli katmandaki nöron (birim) sayısı.\n"
            "Daha yüksek değer → daha güçlü ağ, daha yavaş eğitim.\n"
            "SIREN için önerilen aralık: 256–1024."
        )
        self.depth = QSpinBox()
        self.depth.setRange(1, 64)
        self.depth.setValue(5)
        self.depth.setToolTip(
            "Gizli katman sayısı (derinlik).\n"
            "Çok derin ağlar (>6) SIREN'de gradyan kaybolmasına yol açabilir.\n"
            "Önerilen: 3–5."
        )
        self.activation = QComboBox()
        self.activation.addItems(["sine", "silu", "tanh", "softplus"])
        self.activation.setCurrentText("sine")
        self.activation.setToolTip(
            "Aktivasyon fonksiyonu:\n"
            "  sine   — SIREN ağı; sürekli türevli, yüksek frekanslı alanlar için önerilir.\n"
            "  silu   — SiLU (Swish); genel amaçlı, daha hızlı yakınsama.\n"
            "  tanh   — Klasik MLP; küçük ağlarda dengeli.\n"
            "  softplus — Pozitif çıktı gerektiren durumlarda kullanılır.\n"
            "Not: SIREN aktivasyonu ile Fourier gömme birlikte kullanılamaz."
        )
        self.dropout = QDoubleSpinBox()
        self.dropout.setDecimals(4)
        self.dropout.setRange(0.0, 0.99)
        self.dropout.setValue(0.0)
        self.dropout.setSingleStep(0.01)
        self.dropout.setToolTip(
            "Dropout oranı (0 = kapalı).\n"
            "Eğitim sırasında bu oranda nöron rastgele devre dışı bırakılır.\n"
            "Fizik tabanlı ağlarda genellikle 0 önerilir — aşırı regularizasyon hassasiyeti bozar."
        )

        # SIREN w0 controls
        self.w0_first = QDoubleSpinBox()
        self.w0_first.setDecimals(1)
        self.w0_first.setRange(1.0, 200.0)
        self.w0_first.setValue(30.0)
        self.w0_first.setSingleStep(1.0)
        self.w0_first.setToolTip(
            "SIREN ilk katman frekans çarpanı (ω₀).\n"
            "İlk katmanın sinüs fonksiyonu bu değerle ölçeklenir: sin(ω₀·Wx+b).\n"
            f"Varsayılan: 30. Eğer boş bırakılırsa python -m {TRAIN_CLI_MODULE}, veri setinden otomatik hesaplar."
        )
        self.w0_hidden = QDoubleSpinBox()
        self.w0_hidden.setDecimals(1)
        self.w0_hidden.setRange(1.0, 200.0)
        self.w0_hidden.setValue(30.0)
        self.w0_hidden.setSingleStep(1.0)
        self.w0_hidden.setToolTip(
            "SIREN gizli katman frekans çarpanı (ω₀).\n"
            "İlk katman dışındaki tüm katmanlara uygulanır.\n"
            "Genellikle ilk katman değeriyle aynı tutulur: 30."
        )

        # Fourier/RFF section (only valid for non-sine activations)
        self._fourier_section = CollapsibleSection("Fourier/RFF Gömme (sine dışı aktivasyon)")
        form_fourier = QFormLayout()
        _tune_form(form_fourier)
        self.use_fourier = QCheckBox("Fourier/RFF Gömmeyi Etkinleştir")
        self.use_fourier.setChecked(False)
        self.use_fourier.setToolTip(
            "Random Fourier Feature gömme. SIREN ile KULLANILMAZ."
        )
        self.fourier_info = QLabel("SIREN and Fourier/RFF are mutually exclusive.")
        self.fourier_info.setWordWrap(True)
        self.fourier_info.setStyleSheet("color: #fbbf24; font-size: 11px;")
        self.fourier_n = QSpinBox()
        self.fourier_n.setRange(16, 4096)
        self.fourier_n.setValue(256)
        self.fourier_n.setToolTip("Fourier özellik sayısı (n → 2n boyutlu gömme).")
        self.fourier_sigma = QDoubleSpinBox()
        self.fourier_sigma.setDecimals(3)
        self.fourier_sigma.setRange(0.001, 100.0)
        self.fourier_sigma.setValue(1.0)
        self.fourier_sigma.setToolTip("Frekans matrisinin std'si σ.")
        self.fourier_append_raw = QCheckBox("Ham Koordinatları Ekle")
        self.fourier_append_raw.setChecked(True)
        self.fourier_append_raw.setToolTip("Fourier özelliklerine orijinal xyz'yi de ekle.")
        form_fourier.addRow("", self.fourier_info)
        form_fourier.addRow(self.use_fourier)
        form_fourier.addRow("Özellik Sayısı (n)", self.fourier_n)
        form_fourier.addRow("Sigma (σ)", self.fourier_sigma)
        form_fourier.addRow(self.fourier_append_raw)
        _fourier_inner = QWidget()
        _fourier_inner.setLayout(form_fourier)
        _tune_inputs(_fourier_inner)
        _fourier_vbox = QVBoxLayout()
        _fourier_vbox.setContentsMargins(0, 0, 0, 0)
        _fourier_vbox.addWidget(_fourier_inner)
        self._fourier_section.set_content_layout(_fourier_vbox)

        self._w0_row_first = ("SIREN w0 (ilk katman)", self.w0_first)
        self._w0_row_hidden = ("SIREN w0 (gizli katman)", self.w0_hidden)

        form_arch.addRow("Gizli Katman Boyutu", self.hidden)
        form_arch.addRow("Katman Derinliği", self.depth)
        form_arch.addRow("Aktivasyon Fonksiyonu", self.activation)
        self._w0_first_row_idx = form_arch.rowCount()
        form_arch.addRow("SIREN w0 (ilk katman)", self.w0_first)
        self._w0_hidden_row_idx = form_arch.rowCount()
        form_arch.addRow("SIREN w0 (gizli katman)", self.w0_hidden)
        form_arch.addRow("Dropout Oranı", self.dropout)
        grp_arch.setLayout(form_arch)

        self.activation.currentTextChanged.connect(self._on_activation_changed)

        # =====================================================================
        # GROUP 3: Optimization
        # =====================================================================
        grp_optim = QGroupBox("Optimizasyon")
        form_optim = QFormLayout()
        _tune_form(form_optim)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5_000_000)
        self.epochs.setValue(200)
        self.epochs.setToolTip(
            "Toplam eğitim turu (epoch) sayısı.\n"
            "Her epoch tüm eğitim verisi üzerinden bir geçiş anlamına gelir."
        )
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 10_000_000)
        self.batch_size.setValue(8192)
        self.batch_size.setToolTip(
            "Her güncelleme adımında kullanılan örnek (satır) sayısı.\n"
            "Büyük batch → kararlı gradyanlar, yüksek GPU bellek kullanımı.\n"
            "SIREN için önerilen: 4096–16384."
        )
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(8)
        self.lr.setRange(1e-8, 10.0)
        self.lr.setValue(1e-4)
        self.lr.setSingleStep(1e-5)
        self.lr.setToolTip(
            "Başlangıç öğrenme hızı (Learning Rate).\n"
            "AdamW optimizer bu değerden başlar; cosine decay ile azaltılır.\n"
            "SIREN için önerilen: 1e-4. Çok yüksek değerler ağın ıraksamas  ına yol açabilir."
        )
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setDecimals(8)
        self.weight_decay.setRange(0.0, 10.0)
        self.weight_decay.setValue(1e-6)
        self.weight_decay.setSingleStep(1e-7)
        self.weight_decay.setToolTip(
            "AdamW L2 ağırlık düzenlemesi (weight decay).\n"
            "Ağırlıkları sıfıra çekerek aşırı öğrenmeyi azaltır.\n"
            "Önerilen: 1e-6 ila 1e-4 arası."
        )
        self.output_head_lr_mult = QDoubleSpinBox()
        self.output_head_lr_mult.setDecimals(2)
        self.output_head_lr_mult.setRange(0.1, 100.0)
        self.output_head_lr_mult.setValue(1.0)
        self.output_head_lr_mult.setSingleStep(0.5)
        self.output_head_lr_mult.setToolTip(
            "Çıkış kafası LR çarpanı. Çıkış başının ağırlıklarını daha hızlı günceller."
        )
        self.t_max = QSpinBox()
        self.t_max.setRange(1, 1_000_000)
        self.t_max.setValue(200)
        self.t_max.setToolTip("Cosine LR T_max. Boş bırakılırsa epochs ile aynı.")
        self.warmup_epochs = QSpinBox()
        self.warmup_epochs.setRange(0, 100)
        self.warmup_epochs.setValue(5)
        self.warmup_epochs.setToolTip("Doğrusal LR ısınma epoch sayısı.")
        self.min_lr_ratio = QDoubleSpinBox()
        self.min_lr_ratio.setDecimals(4)
        self.min_lr_ratio.setRange(0.0, 1.0)
        self.min_lr_ratio.setValue(0.05)
        self.min_lr_ratio.setSingleStep(0.01)
        self.min_lr_ratio.setToolTip("Cosine sonundaki minimum LR / başlangıç LR oranı.")
        self.patience = QSpinBox()
        self.patience.setRange(1, 10000)
        self.patience.setValue(30)
        self.patience.setToolTip("Erken durdurma sabrı (val loss iyileşmezse).")
        self.no_amp = QCheckBox("AMP Kapalı")
        self.no_amp.setChecked(False)
        self.no_amp.setToolTip("AMP (Mixed Precision) devre dışı bırak.")

        form_optim.addRow("Epoch Sayısı", self.epochs)
        form_optim.addRow("Batch Boyutu", self.batch_size)
        form_optim.addRow("Öğrenme Hızı (LR)", self.lr)
        form_optim.addRow("Ağırlık Azaltma", self.weight_decay)
        form_optim.addRow("Çıkış Kafası LR Çarpanı", self.output_head_lr_mult)
        form_optim.addRow("Cosine LR Periyodu (T_max)", self.t_max)
        form_optim.addRow("Isınma Epoch'u", self.warmup_epochs)
        form_optim.addRow("Min LR Oranı", self.min_lr_ratio)
        form_optim.addRow("Erken Durdurma Sabrı", self.patience)
        form_optim.addRow(self.no_amp)
        grp_optim.setLayout(form_optim)

        # =====================================================================
        # GROUP 4: Physics & Sobolev Loss
        # =====================================================================
        grp_phys = QGroupBox("Fizik ve Sobolev Kayıp")
        form_phys = QFormLayout()
        _tune_form(form_phys)

        self.w_u = QDoubleSpinBox()
        self.w_u.setDecimals(6)
        self.w_u.setRange(0.0, 1e6)
        self.w_u.setValue(1.0)
        self.w_u.setSingleStep(0.1)
        self.w_u.setToolTip("Potansiyel (U) kayıp ağırlığı (fixed/ntk_init modlarında).")
        self.w_a = QDoubleSpinBox()
        self.w_a.setDecimals(6)
        self.w_a.setRange(0.0, 1e6)
        self.w_a.setValue(1.0)
        self.w_a.setSingleStep(0.1)
        self.w_a.setToolTip("İvme (a) kayıp ağırlığı (fixed/ntk_init modlarında).")

        self.gradnorm_mode = QComboBox()
        self.gradnorm_mode.addItems(["ntk_init", "fixed", "dynamic"])
        self.gradnorm_mode.setCurrentText("ntk_init")
        self.gradnorm_mode.setToolTip(
            "ntk_init: ilk adımda gradyan normlarını dengeler, sonra sabit tutar (önerilen).\n"
            "fixed: w_u ve w_a sabit kalır.\n"
            "dynamic: her adımda gradyan normuna göre dinamik güncelleme."
        )
        self.gradnorm_mode.currentTextChanged.connect(self._on_gradnorm_mode_changed)

        self.gradnorm_w_a_min = QDoubleSpinBox()
        self.gradnorm_w_a_min.setDecimals(4)
        self.gradnorm_w_a_min.setRange(0.0, 100.0)
        self.gradnorm_w_a_min.setValue(0.05)
        self.gradnorm_w_a_min.setToolTip("NTK/dinamik w_a için alt sınır.")
        self.gradnorm_w_a_max = QDoubleSpinBox()
        self.gradnorm_w_a_max.setDecimals(4)
        self.gradnorm_w_a_max.setRange(0.01, 1000.0)
        self.gradnorm_w_a_max.setValue(2.0)
        self.gradnorm_w_a_max.setToolTip("NTK/dinamik w_a için üst sınır.")

        self.potential_only_epochs = QSpinBox()
        self.potential_only_epochs.setRange(0, 1000)
        self.potential_only_epochs.setValue(0)
        self.potential_only_epochs.setToolTip(
            "Başlangıç warm-up epoch sayısı. İvme accel_min_factor tabanıyla aktif kalır."
        )
        self.accel_ramp_epochs = QSpinBox()
        self.accel_ramp_epochs.setRange(0, 1000)
        self.accel_ramp_epochs.setValue(80)
        self.accel_ramp_epochs.setToolTip(
            "İvme kaybını accel_min_factor'dan tam ağırlığa doğrusal olarak artırma epoch sayısı."
        )

        self.accel_min_factor = QDoubleSpinBox()
        self.accel_min_factor.setDecimals(4)
        self.accel_min_factor.setRange(0.0, 1.0)
        self.accel_min_factor.setSingleStep(0.01)
        self.accel_min_factor.setValue(0.05)
        self.accel_min_factor.setToolTip(
            "İvme kaybı için minimum çarpan. 0=tam sıfır (önerilmez), 0.05=küçük taban. "
            "Türev alanının sürüklenmesini önler."
        )

        self.a_sign = QComboBox()
        self.a_sign.addItems(["auto", "+1", "-1"])
        self.a_sign.setCurrentText("auto")
        self.a_sign.setToolTip("İvme işareti: auto | +1 Jeodezi | -1 Newton")

        self.use_si = QComboBox()
        self.use_si.addItems(["SI Birimleri (Önerilen)", "Orijinal (Dönüşüm Yok)"])
        self.use_si.setCurrentIndex(0)
        self.use_si.setToolTip("Kanonik → SI dönüşümü.")

        form_phys.addRow("Potansiyel (U) Kayıp Ağırlığı", self.w_u)
        form_phys.addRow("İvme (a) Kayıp Ağırlığı", self.w_a)
        form_phys.addRow("GradNorm Modu", self.gradnorm_mode)
        form_phys.addRow("GradNorm w_a Minimum", self.gradnorm_w_a_min)
        form_phys.addRow("GradNorm w_a Maksimum", self.gradnorm_w_a_max)
        form_phys.addRow("Yalnızca Potansiyel Epoch", self.potential_only_epochs)
        form_phys.addRow("İvme Ramp Epoch", self.accel_ramp_epochs)
        form_phys.addRow("İvme Min Faktör", self.accel_min_factor)
        form_phys.addRow("İvme İşareti (a_sign)", self.a_sign)
        form_phys.addRow("Birim Sistemi", self.use_si)
        grp_phys.setLayout(form_phys)

        # =====================================================================
        # GROUP 5: Direction Loss (Collapsible)
        # =====================================================================
        self._dir_loss_section = CollapsibleSection("Yön Kaybı (Direction Loss)")
        form_dir = QFormLayout()
        _tune_form(form_dir)

        self.direction_loss_weight = QDoubleSpinBox()
        self.direction_loss_weight.setDecimals(4)
        self.direction_loss_weight.setRange(0.0, 10.0)
        self.direction_loss_weight.setValue(0.10)
        self.direction_loss_weight.setSingleStep(0.01)
        self.direction_loss_weight.setToolTip(
            "Yön kaybı zirve ağırlığı λ_dir.\n"
            "L_dir = mean(1 - cos_sim(a_pred, a_true))"
        )
        self.direction_loss_start_epoch = QSpinBox()
        self.direction_loss_start_epoch.setRange(0, 10000)
        self.direction_loss_start_epoch.setValue(30)
        self.direction_loss_start_epoch.setToolTip(
            "Yön kaybının başlamaya başladığı epoch."
        )
        self.direction_loss_ramp_epochs = QSpinBox()
        self.direction_loss_ramp_epochs.setRange(1, 10000)
        self.direction_loss_ramp_epochs.setValue(50)
        self.direction_loss_ramp_epochs.setToolTip(
            "Yön kaybını 0'dan tam ağırlığa çıkarma epoch sayısı."
        )
        self.direction_loss_floor_abs = QDoubleSpinBox()
        self.direction_loss_floor_abs.setDecimals(8)
        self.direction_loss_floor_abs.setRange(0.0, 1.0)
        self.direction_loss_floor_abs.setValue(3e-6)
        self.direction_loss_floor_abs.setSingleStep(1e-7)
        self.direction_loss_floor_abs.setToolTip(
            "||a_true|| eşik değeri. Bunun altındaki noktalar yön kaybı maskesinde çıkarılır."
        )

        self.best_ckpt_start_epoch = QSpinBox()
        self.best_ckpt_start_epoch.setRange(-1, 10000)
        self.best_ckpt_start_epoch.setValue(-1)
        self.best_ckpt_start_epoch.setToolTip(
            "En iyi checkpoint takibinin başladığı epoch (-1=otomatik).\n"
            "-1: direction loss aktifse start_epoch + ramp_epochs + settle_epochs sonrasına kadar bekler.\n"
            "Böylece direction hesabına oturmamış erken epoch'lar best checkpoint olmaz.\n"
            "0: epoch 0'dan itibaren takip eder; >0: ilk N epoch burn-in olarak atlanır."
        )
        self.checkpoint_settle_epochs = QSpinBox()
        self.checkpoint_settle_epochs.setRange(0, 10000)
        self.checkpoint_settle_epochs.setValue(5)
        self.checkpoint_settle_epochs.setToolTip(
            "Otomatik best checkpoint takibi başlamadan önce direction ramp sonrası beklenecek ek epoch sayısı."
        )

        form_dir.addRow("Zirve Ağırlık (λ)", self.direction_loss_weight)
        form_dir.addRow("Başlangıç Epoch'u", self.direction_loss_start_epoch)
        form_dir.addRow("Ramp Epoch Sayısı", self.direction_loss_ramp_epochs)
        form_dir.addRow("||a|| Eşiği (m/s²)", self.direction_loss_floor_abs)
        form_dir.addRow("Best Ckpt Başlangıç", self.best_ckpt_start_epoch)
        form_dir.addRow("Checkpoint Settle Epoch", self.checkpoint_settle_epochs)
        _dir_inner = QWidget()
        _dir_inner.setLayout(form_dir)
        _tune_inputs(_dir_inner)
        _dir_vbox = QVBoxLayout()
        _dir_vbox.setContentsMargins(0, 0, 0, 0)
        _dir_vbox.addWidget(_dir_inner)
        self._dir_loss_section.set_content_layout(_dir_vbox)

        # =====================================================================
        # GROUP 6: Field-Structure Losses (Collapsible)
        # =====================================================================
        self._field_loss_section = CollapsibleSection("Field-Structure Losses")
        form_field = QFormLayout()
        _tune_form(form_field)

        self.use_altitude_balanced_loss = QCheckBox("Use altitude-balanced loss")
        self.use_altitude_balanced_loss.setChecked(False)
        self.use_altitude_balanced_loss.setToolTip(
            "Average loss over altitude bins so dense/easy shells do not dominate the fit."
        )
        self.altitude_bin_width_km = QDoubleSpinBox()
        self.altitude_bin_width_km.setDecimals(2)
        self.altitude_bin_width_km.setRange(1.0, 10_000.0)
        self.altitude_bin_width_km.setValue(50.0)
        self.altitude_min_km = QDoubleSpinBox()
        self.altitude_min_km.setDecimals(2)
        self.altitude_min_km.setRange(0.0, 1_000_000.0)
        self.altitude_min_km.setValue(float(_cfg_value(DEFAULT_SPATIAL_CLOUD_CONFIG, "alt_min_km", 200.0)))
        self.altitude_max_km = QDoubleSpinBox()
        self.altitude_max_km.setDecimals(2)
        self.altitude_max_km.setRange(0.0, 1_000_000.0)
        self.altitude_max_km.setValue(float(_cfg_value(DEFAULT_SPATIAL_CLOUD_CONFIG, "alt_max_km", 600.0)))

        self.use_radial_cross_loss = QCheckBox("Use radial / cross-radial loss")
        self.use_radial_cross_loss.setChecked(False)
        self.use_radial_cross_loss.setToolTip(
            "Adds lightweight radial and cross-radial acceleration error penalties."
        )
        self.radial_loss_weight = QDoubleSpinBox()
        self.radial_loss_weight.setDecimals(6)
        self.radial_loss_weight.setRange(0.0, 1000.0)
        self.radial_loss_weight.setValue(0.05)
        self.cross_loss_weight = QDoubleSpinBox()
        self.cross_loss_weight.setDecimals(6)
        self.cross_loss_weight.setRange(0.0, 1000.0)
        self.cross_loss_weight.setValue(0.05)

        self.use_laplacian_regularization = QCheckBox("Use sparse Laplacian regularization")
        self.use_laplacian_regularization.setChecked(False)
        self.use_laplacian_regularization.setToolTip(
            "Computes second derivatives on a sparse subset. Expensive; keep subset size small."
        )
        lap_warn = QLabel(
            "This computes second derivatives and can be expensive. Keep subset size small."
        )
        lap_warn.setWordWrap(True)
        lap_warn.setStyleSheet("color: #fbbf24; font-size: 11px;")
        self.laplacian_weight = QDoubleSpinBox()
        self.laplacian_weight.setDecimals(10)
        self.laplacian_weight.setRange(0.0, 1.0)
        self.laplacian_weight.setValue(0.0)
        self.laplacian_weight.setSingleStep(1e-5)
        self.laplacian_every_n_batches = QSpinBox()
        self.laplacian_every_n_batches.setRange(1, 100000)
        self.laplacian_every_n_batches.setValue(5)
        self.laplacian_subset_size = QSpinBox()
        self.laplacian_subset_size.setRange(1, 1_000_000)
        self.laplacian_subset_size.setValue(512)

        form_field.addRow(self.use_altitude_balanced_loss)
        form_field.addRow("Altitude bin width (km)", self.altitude_bin_width_km)
        form_field.addRow("Altitude min (km)", self.altitude_min_km)
        form_field.addRow("Altitude max (km)", self.altitude_max_km)
        form_field.addRow(self.use_radial_cross_loss)
        form_field.addRow("Radial weight", self.radial_loss_weight)
        form_field.addRow("Cross-radial weight", self.cross_loss_weight)
        form_field.addRow(self.use_laplacian_regularization)
        form_field.addRow("", lap_warn)
        form_field.addRow("Laplacian weight", self.laplacian_weight)
        form_field.addRow("Every N batches", self.laplacian_every_n_batches)
        form_field.addRow("Subset size", self.laplacian_subset_size)
        field_inner = QWidget()
        field_inner.setLayout(form_field)
        _tune_inputs(field_inner)
        field_vbox = QVBoxLayout()
        field_vbox.setContentsMargins(0, 0, 0, 0)
        field_vbox.addWidget(field_inner)
        self._field_loss_section.set_content_layout(field_vbox)
        self.use_altitude_balanced_loss.toggled.connect(self._on_loss_feature_toggled)
        self.use_radial_cross_loss.toggled.connect(self._on_loss_feature_toggled)
        self.use_laplacian_regularization.toggled.connect(self._on_loss_feature_toggled)

        # =====================================================================
        # GROUP 6: Advanced (Collapsible)
        # =====================================================================
        self.advanced_section = CollapsibleSection(
            "Gelişmiş Ayarlar (Donanım & Performans)"
        )
        form_adv = QFormLayout()
        _tune_form(form_adv)

        self.max_grad_norm = QDoubleSpinBox()
        self.max_grad_norm.setDecimals(6)
        self.max_grad_norm.setRange(0.0, 1e6)
        self.max_grad_norm.setValue(0.5)
        self.max_grad_norm.setToolTip("Gradyan kırpma normu. 0 → Kırpma yok.")
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 64)
        self.num_workers.setValue(2)
        self.num_workers.setToolTip("DataLoader işçi sayısı.")
        self.cache_rows = QSpinBox()
        self.cache_rows.setRange(1024, 10_000_000)
        self.cache_rows.setValue(65536)
        self.cache_rows.setToolTip("HDF5 RAM önbellek satır sayısı.")
        self.fit_rows = QSpinBox()
        self.fit_rows.setRange(10_000, 50_000_000)
        self.fit_rows.setValue(500_000)
        self.fit_rows.setToolTip("Z-score örneklem satır sayısı.")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(42)
        self.seed.setToolTip("Rastgele tohum.")
        self.split_seed = QSpinBox()
        self.split_seed.setRange(0, 2_147_483_647)
        self.split_seed.setValue(42)
        self.split_seed.setToolTip("Train/val ayrımı için ayrı tohum (None → seed ile aynı).")
        self.device_hint = QComboBox()
        self.device_hint.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_hint.setCurrentText("auto")
        self.device_hint.setToolTip("Cihaz ipucu. cpu/mps → AMP otomatik kapatılır.")
        self.device_hint.currentTextChanged.connect(self._on_device_hint_changed)
        self.log_every_mode = QComboBox()
        self.log_every_mode.addItem("auto", "auto")
        self.log_every_mode.addItem("fixed", "fixed")
        self.log_every_mode.setCurrentIndex(0)  # auto by default
        self.log_every_mode.setToolTip(
            "Auto logs roughly 10 progress updates per epoch (always including the "
            "first and last batch). Fixed uses the batch interval below."
        )
        self.log_every_mode.currentIndexChanged.connect(self._on_log_every_mode_changed)
        self.log_every = QSpinBox()
        self.log_every.setRange(0, 10000)
        self.log_every.setValue(10)
        self.log_every.setToolTip("Her N batch'te bir ilerleme yaz. 0 → devre dışı. (Fixed modunda kullanılır.)")
        self.preload_data = QCheckBox("RAM'e Önceden Yükle")
        self.preload_data.setChecked(False)
        self.preload_data.setToolTip(
            "Küçük dataset'leri (≤auto_preload_mb) CPU RAM'e yükler.\n"
            "Windows'ta HDF5 çok-işçi sorununu çözer."
        )
        self.auto_preload_mb = QDoubleSpinBox()
        self.auto_preload_mb.setDecimals(0)
        self.auto_preload_mb.setRange(0.0, 102400.0)
        self.auto_preload_mb.setValue(256.0)
        self.auto_preload_mb.setSingleStep(64.0)
        self.auto_preload_mb.setToolTip(
            "Bu MB değerinden küçük dataset'ler otomatik RAM'e alınır (0 → devre dışı)."
        )
        self.pin_memory = QCheckBox("Pin Memory")
        self.pin_memory.setChecked(True)
        self.pin_memory.setToolTip("CUDA transferlerini hızlandırmak için CPU belleği sabitle.")
        self.quick_check = QCheckBox("Quick Check Modu")
        self.quick_check.setChecked(False)
        self.quick_check.setToolTip(
            "1 epoch, 5 train + 2 val batch ile pipeline doğrulaması. Gerçek eğitim değil."
        )

        # PINN Architecture (new)
        self.use_residual_blocks = QCheckBox("Residual SIREN Blokları (SirenResBlock)")
        self.use_residual_blocks.setChecked(False)
        self.use_residual_blocks.setToolTip(
            "Gizli katmanları pre-norm + zero-init skip (SirenResBlock) ile sarar.\n"
            "Derinlik >= 6 için önerilir. Ek parametre eklemez."
        )
        self.n_bands = QSpinBox()
        self.n_bands.setRange(1, 16)
        self.n_bands.setValue(1)
        self.n_bands.setToolTip(
            "Multi-scale SIREN bant sayısı. >1 → MultiScaleSirenMLP.\n"
            "1 = standart SirenMLP. Bant w0 değerleri degree_min/max'tan otomatik türetilir."
        )
        self.grad_accumulation_steps = QSpinBox()
        self.grad_accumulation_steps.setRange(1, 128)
        self.grad_accumulation_steps.setValue(1)
        self.grad_accumulation_steps.setToolTip(
            "Optimizer adımından önce gradyanları N batch boyunca biriktirir.\n"
            "Efektif batch = batch_size × N. VRAM kısıtlı durumlar için."
        )
        self.n_hutchinson_samples = QSpinBox()
        self.n_hutchinson_samples.setRange(1, 32)
        self.n_hutchinson_samples.setValue(4)
        self.n_hutchinson_samples.setToolTip(
            "Hutchinson Laplacian tahmini için Rademacher örneklem sayısı K.\n"
            "K=4 → ~%50 göreli hata; sadece Laplacian reg. etkinken kullanılır."
        )

        # Performance extras (new)
        self.prefetch_factor = QSpinBox()
        self.prefetch_factor.setSpecialValueText("auto")
        self.prefetch_factor.setRange(0, 32)
        self.prefetch_factor.setValue(0)
        self.prefetch_factor.setToolTip(
            "DataLoader prefetch_factor (0 = otomatik; yalnızca num_workers > 0 ise geçerli)."
        )
        self.max_train_batches = QSpinBox()
        self.max_train_batches.setSpecialValueText("unlimited")
        self.max_train_batches.setRange(0, 1_000_000)
        self.max_train_batches.setValue(0)
        self.max_train_batches.setToolTip(
            "Epoch başına maksimum eğitim batch sayısı. 0 = sınırsız (tam epoch).\n"
            "Hızlı test için kullanılır; quick_check ile birlikte çalışır."
        )
        self.max_val_batches = QSpinBox()
        self.max_val_batches.setSpecialValueText("unlimited")
        self.max_val_batches.setRange(0, 1_000_000)
        self.max_val_batches.setValue(0)
        self.max_val_batches.setToolTip(
            "Epoch başına maksimum validasyon batch sayısı. 0 = sınırsız."
        )

        form_adv.addRow("Cihaz İpucu (Device)", self.device_hint)
        form_adv.addRow("Gradyan Kırpma Normu", self.max_grad_norm)
        form_adv.addRow("DataLoader İşçi Sayısı", self.num_workers)
        form_adv.addRow("Prefetch Factor", self.prefetch_factor)
        form_adv.addRow("HDF5 Önbellek Satırı", self.cache_rows)
        form_adv.addRow("Scaler Örneklem Satırı", self.fit_rows)
        form_adv.addRow("Rastgele Tohum (Seed)", self.seed)
        form_adv.addRow("Bölme Tohumu (Split Seed)", self.split_seed)
        form_adv.addRow("Log Frekans Modu", self.log_every_mode)
        form_adv.addRow("Batch Log Sıklığı", self.log_every)
        form_adv.addRow(self.preload_data)
        form_adv.addRow("Oto-RAM Yükleme Limiti (MB)", self.auto_preload_mb)
        form_adv.addRow(self.pin_memory)
        # quick_check / max_train_batches / max_val_batches are dev-only debug
        # controls — kept as hidden widgets (for config/profile compatibility)
        # but intentionally NOT shown in the normal Studio workflow.
        self.quick_check.setVisible(False)
        self.max_train_batches.setVisible(False)
        self.max_val_batches.setVisible(False)
        _adv_sep = QLabel("  PINN Mimarisi")
        _adv_sep.setStyleSheet(
            "color: #9aa7ff; font-size: 11px; font-weight: 600;"
            " padding: 4px 10px; margin-top: 4px;"
            " background: rgba(124, 92, 255, 0.08);"
            " border-left: 2px solid rgba(124, 92, 255, 0.40);"
            " border-radius: 0 6px 6px 0;"
        )
        form_adv.addRow(_adv_sep)
        form_adv.addRow(self.use_residual_blocks)
        form_adv.addRow("Frekans Bandı Sayısı (n_bands)", self.n_bands)
        form_adv.addRow("Grad. Birikim Adımı", self.grad_accumulation_steps)
        form_adv.addRow("Hutchinson Örneklem (K)", self.n_hutchinson_samples)

        adv_inner = QWidget()
        adv_inner.setLayout(form_adv)
        _tune_inputs(adv_inner)
        adv_wrapper = QVBoxLayout()
        adv_wrapper.setContentsMargins(0, 0, 0, 0)
        adv_wrapper.addWidget(adv_inner)
        self.advanced_section.set_content_layout(adv_wrapper)

        # =====================================================================
        # MODEL REPRESENTATION (input encoding preset + manual ablation flags)
        # =====================================================================
        # The model_preset combo itself lives in the top toolbar (compact).
        # This group exposes the manual encoding flags and physical-radial-decay
        # options that only apply when model_preset == "custom".
        self.model_preset = QComboBox()
        for _val, _label in (
            ("baseline_raw", "Baseline · raw coordinates"),
            ("recommended_physical_radial_decay", "Recommended · physical radial decay"),
            ("ablation_radial_separation", "Ablation · radial separation"),
            ("ablation_radial_decay_scaled", "Ablation · radial decay (scaled)"),
            ("ablation_real_sh_low_degree", "Ablation · real SH (low degree)"),
            ("custom", "Custom · manual encoding flags"),
        ):
            self.model_preset.addItem(_label, _val)
        _mp_default = self.model_preset.findData("recommended_physical_radial_decay")
        self.model_preset.setCurrentIndex(_mp_default if _mp_default >= 0 else 0)
        self.model_preset.setToolTip(
            "Input-encoding representation preset.\n"
            "Non-custom presets fully control the representation; the manual flags\n"
            "below are only used when 'Custom' is selected."
        )
        self.model_preset.currentIndexChanged.connect(self._on_model_preset_changed)

        grp_model_repr = QGroupBox("Model Representation (manual encoding — Custom only)")
        form_model_repr = QFormLayout()
        _tune_form(form_model_repr)

        self.model_preset_note = QLabel("")
        self.model_preset_note.setWordWrap(True)
        self.model_preset_note.setStyleSheet("color: #fbbf24; font-size: 11px;")

        self.use_radial_separation = QCheckBox("Radial separation encoding [r, ux, uy, uz]")
        self.use_radial_decay_encoding = QCheckBox("Radial decay encoding (scaled inverse-radius)")
        self.use_physical_radial_decay_encoding = QCheckBox("Physical radial decay encoding (R_ref/r)")
        self.use_real_sh_basis = QCheckBox("Real spherical-harmonic basis")
        for _cb in (
            self.use_radial_separation, self.use_radial_decay_encoding,
            self.use_physical_radial_decay_encoding, self.use_real_sh_basis,
        ):
            _cb.toggled.connect(self._refresh_command_preview)

        self.physical_radial_decay_max_power = QSpinBox()
        self.physical_radial_decay_max_power.setRange(1, 16)
        self.physical_radial_decay_max_power.setValue(4)
        self.physical_radial_decay_max_power.setToolTip("Highest power of R_ref/r used in the physical decay encoding.")
        self.physical_radial_decay_append_raw = QCheckBox("Append raw coordinates")
        self.physical_radial_decay_append_raw.setChecked(True)
        self.physical_radial_decay_include_unit = QCheckBox("Include unit direction vector")
        self.physical_radial_decay_include_unit.setChecked(True)
        self.physical_radial_decay_include_r_scaled = QCheckBox("Include scaled radius (r/R_ref)")
        self.physical_radial_decay_include_r_scaled.setChecked(True)

        form_model_repr.addRow("", self.model_preset_note)
        form_model_repr.addRow(self.use_radial_separation)
        form_model_repr.addRow(self.use_radial_decay_encoding)
        form_model_repr.addRow(self.use_physical_radial_decay_encoding)
        form_model_repr.addRow("Phys. decay max power", self.physical_radial_decay_max_power)
        form_model_repr.addRow(self.physical_radial_decay_append_raw)
        form_model_repr.addRow(self.physical_radial_decay_include_unit)
        form_model_repr.addRow(self.physical_radial_decay_include_r_scaled)
        form_model_repr.addRow(self.use_real_sh_basis)
        grp_model_repr.setLayout(form_model_repr)
        self._grp_model_repr = grp_model_repr

        # =====================================================================
        # EXTRA CLI ARGS
        # =====================================================================
        self.extra_args = QLineEdit("")
        self.extra_args.setPlaceholderText("Ek CLI argümanları (opsiyonel)")
        self.extra_args.setToolTip(f"Doğrudan python -m {TRAIN_CLI_MODULE} komutuna iletilecek ek argümanlar.")

        # =====================================================================
        # LAYOUT ASSEMBLY
        # =====================================================================
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(grp_data, 0, 0)
        grid.addWidget(self.resume_section, 0, 1)
        grid.addWidget(grp_arch, 1, 0)
        grid.addWidget(grp_optim, 1, 1)
        grid.addWidget(grp_phys, 2, 0, 1, 2)
        grid.addWidget(self._fourier_section, 3, 0, 1, 2)
        grid.addWidget(self._dir_loss_section, 4, 0, 1, 2)
        grid.addWidget(self._field_loss_section, 5, 0, 1, 2)
        grid.addWidget(self.advanced_section, 6, 0, 1, 2)
        grid.addWidget(grp_model_repr, 7, 0, 1, 2)

        extra_row_layout = QFormLayout()
        _tune_form(extra_row_layout)
        extra_row_layout.addRow("Ek CLI Argümanları", self.extra_args)
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setFont(_mono_font())
        self.command_preview.setMinimumHeight(78)
        self.command_preview.setPlaceholderText(
            f"Click Preview Command to see the exact python -m {TRAIN_CLI_MODULE} command."
        )
        self.command_warning = QLabel("")
        self.command_warning.setWordWrap(True)
        self.command_warning.setStyleSheet("color: #fbbf24; font-size: 11px;")
        btn_preview = QPushButton("Preview Command")
        btn_preview.clicked.connect(self._refresh_command_preview)
        btn_copy = QPushButton("Copy Command")
        btn_copy.clicked.connect(self._copy_command_preview)
        preview_buttons = QHBoxLayout()
        preview_buttons.setContentsMargins(0, 0, 0, 0)
        preview_buttons.addWidget(btn_preview)
        preview_buttons.addWidget(btn_copy)
        preview_buttons.addStretch(1)
        preview_wrap = QWidget()
        preview_wrap.setLayout(preview_buttons)
        extra_row_layout.addRow("", preview_wrap)
        extra_row_layout.addRow("Generated Command", self.command_preview)
        extra_row_layout.addRow("", self.command_warning)
        extra_w = QWidget()
        extra_w.setLayout(extra_row_layout)
        grid.addWidget(extra_w, 8, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for grp in (grp_data, grp_arch, grp_optim, grp_phys):
            _tune_inputs(grp)
        _tune_inputs(self.resume_section)
        _tune_inputs(self._dir_loss_section)
        _tune_inputs(self._field_loss_section)

        # --- ProcessPane ---
        self.runner = ProcessPane()
        self.runner.btn_start.setText("Eğitimi Başlat")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_progress_parser(self._parse_progress)
        self.runner.set_finished_hook(self._on_train_finished)
        self._user_stopped = False  # set when the user clicks Stop (→ INTERRUPTED)
        self._device_badge: Optional[str] = None

        # --- Live Loss Plot (Feature #13) ---
        self._live_plot = LiveLossPlot()
        self._history_poll_timer = QTimer(self)
        self._history_poll_timer.setInterval(2000)
        self._history_poll_timer.timeout.connect(self._poll_training_history)
        self._history_poll_path: Optional[Path] = None
        self._history_poll_mtime: float = 0.0

        # --- Dashboard v2: KPI strip, structured log, ETA, parser ---
        if _HAS_DASHBOARD_V2:
            self._kpi_strip = KPIStrip()
            self._time_strip = TimeMetricsStrip()
            self._structured_log = StructuredLogView()
            self._eta_estimator = ETAEstimator()
            self._epoch_guard = EpochGuard()
            self._log_parser = TrainingLogParser()
            self._metrics_store = TrainingMetricsStore()
            self._eta_update_timer = QTimer(self)
            self._eta_update_timer.setInterval(1000)
            self._eta_update_timer.timeout.connect(self._update_eta_display)
        else:
            self._kpi_strip = None
            self._time_strip = None
            self._structured_log = None
            self._eta_estimator = None
            self._epoch_guard = None
            self._log_parser = None
            self._metrics_store = None


        # --- "Add to Queue" button (Feature #15) ---
        self.btn_enqueue = QPushButton("Kuyruğa Ekle")
        self.btn_enqueue.setToolTip("Mevcut ayarları eğitim kuyruğuna ekler.")
        self.btn_enqueue.setProperty("kind", "ghost")
        self.btn_enqueue.clicked.connect(self._enqueue_current)

        # --- Training Queue (Feature #15) ---
        self._queue = TrainingQueue()
        self._queue.job_started.connect(self._on_queue_job_started)

        # =====================================================================
        # PHASE 3: professional training-console layout.
        #
        #   1. compact experiment toolbar (workflow · profile · model preset)
        #   2. training control bar (Start/Stop/Queue + progress + run info)
        #   3. KPI status strip
        #   4. main workspace splitter (charts dominant | structured progress)
        #   5. lower tabs (Configuration form · Queue)
        # =====================================================================

        # ── 1. Compact experiment toolbar (pinned) ───────────────────────
        model_repr_bar = QHBoxLayout()
        model_repr_bar.setContentsMargins(4, 2, 4, 2)
        model_repr_bar.setSpacing(8)
        mdl_lbl = QLabel("Model:")
        mdl_lbl.setStyleSheet("font-weight: 600; color: #6ee7b7; font-size: 13px;")
        model_repr_bar.addWidget(mdl_lbl)
        model_repr_bar.addWidget(self.model_preset, 1)

        controls_bar = QFrame()
        controls_bar.setObjectName("trainControlsBar")
        controls_bar.setStyleSheet(
            "QFrame#trainControlsBar {"
            "  background: rgba(11, 16, 32, 0.80);"
            "  border: 1px solid rgba(185, 194, 221, 0.12);"
            "  border-radius: 10px;"
            "}"
        )
        ctrl_lo = QVBoxLayout()
        ctrl_lo.setContentsMargins(10, 8, 10, 8)
        ctrl_lo.setSpacing(6)
        ctrl_lo.addLayout(workflow_bar)
        ctrl_lo.addLayout(model_repr_bar)
        ctrl_lo.addLayout(preset_bar)
        controls_bar.setLayout(ctrl_lo)

        # ── 2. Training control bar (Start/Stop/Progress always visible) ──
        train_ctrl_bar = self._build_training_control_bar()

        # ── PARAMETERS tab: the scrollable configuration form ────────────
        top = QWidget()
        top_l = QVBoxLayout()
        top_l.setContentsMargins(6, 6, 6, 6)
        top_l.setSpacing(8)
        top_l.addWidget(self._checklist_label)
        top_l.addLayout(grid)
        top.setLayout(top_l)
        params_page = _scroll_wrap(top)

        # ── LIVE MONITOR tab: charts (dominant) + structured progress ────
        # The chart card runs in compact mode: its redundant in-card metric
        # chips are hidden (the KPI/time strips above already show them), so the
        # actual plot fills the available height.
        if _HAS_DASHBOARD_V2:
            self._live_plot.set_compact(True)

        workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        if _HAS_DASHBOARD_V2 and self._structured_log is not None:
            # Phase 1: the Raw Log tab receives ONLY the raw log widget,
            # never the whole ProcessPane (process controls live in the bar).
            self._structured_log.set_raw_log_widget(self.runner.raw_log_widget())
            workspace_splitter.addWidget(self._live_plot)
            workspace_splitter.addWidget(self._structured_log)
        else:
            workspace_splitter.addWidget(self._live_plot)
            workspace_splitter.addWidget(self.runner)
        workspace_splitter.setStretchFactor(0, 7)   # charts dominant (~70%)
        workspace_splitter.setStretchFactor(1, 3)   # structured progress (~30%)
        workspace_splitter.setSizes([820, 360])
        self._live_plot.setMinimumHeight(300)
        self._live_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Keep the status strips compact so the workspace gets the vertical room.
        if _HAS_DASHBOARD_V2 and self._kpi_strip is not None:
            self._kpi_strip.setMaximumHeight(92)
            self._kpi_strip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if _HAS_DASHBOARD_V2 and self._time_strip is not None:
            self._time_strip.setMaximumHeight(92)
            self._time_strip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        monitor_page = QWidget()
        monitor_l = QVBoxLayout()
        monitor_l.setContentsMargins(4, 4, 4, 4)
        monitor_l.setSpacing(8)
        if _HAS_DASHBOARD_V2 and self._kpi_strip is not None:
            monitor_l.addWidget(self._kpi_strip)        # Phase 4: KPI status strip
        if _HAS_DASHBOARD_V2 and self._time_strip is not None:
            monitor_l.addWidget(self._time_strip)       # Phase 7: time metrics
        monitor_l.addWidget(workspace_splitter, 1)      # charts/log fill the page
        monitor_page.setLayout(monitor_l)

        # ── Queue tab: its own roomy page (no longer crammed into Monitor) ──
        queue_page = QWidget()
        queue_l = QVBoxLayout()
        queue_l.setContentsMargins(4, 4, 4, 4)
        queue_l.setSpacing(8)
        queue_l.addWidget(self._queue, 1)
        queue_page.setLayout(queue_l)

        # ── Phase 3: Parameters | Live Monitor | Queue sub-tabs ──────────
        self._page_tabs = QTabWidget()
        self._page_tabs.setDocumentMode(True)
        self._params_tab_idx = self._page_tabs.addTab(params_page, "Parameters")
        self._monitor_tab_idx = self._page_tabs.addTab(monitor_page, "Live Monitor")
        self._queue_tab_idx = self._page_tabs.addTab(queue_page, "Queue")
        self._page_tabs.setCurrentIndex(self._params_tab_idx)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(controls_bar)
        layout.addWidget(train_ctrl_bar)
        layout.addWidget(self._page_tabs, 1)
        self.setLayout(layout)

        self._epochs_max = int(self.epochs.value())
        self.runner.progress.setRange(0, self._epochs_max)
        self.runner.progress.setFormat("Epoch %v / %m")

        # Suite manifest applied from CloudGenTab
        self.applied_suite_manifest_path: str = ""

        self._restore_settings()
        self._on_dataset_mode_changed()
        self._on_loss_feature_toggled()
        self._on_activation_changed(self.activation.currentText())
        self._on_gradnorm_mode_changed(self.gradnorm_mode.currentText())
        self._on_device_hint_changed(self.device_hint.currentText())
        self._on_resume_toggled(self.resume_enabled.isChecked())
        self._on_workflow_mode_changed()
        self._on_model_preset_changed()
        self._on_log_every_mode_changed()
        self._refresh_command_preview()
        self._refresh_checklist()

    # -----------------------------------------------------------------
    # Dataset Introspection (Feature #14)
    # -----------------------------------------------------------------
    def _on_data_path_validated(self, path: str, exists: bool) -> None:
        if not exists or not path:
            self._ds_info.clear_info()
            return
        info = _introspect_h5(path)
        if info:
            self._ds_info.show_info(info)
            # Auto-set dataset name if found
            if "dataset_name" in info:
                self.dataset_name.setText(info["dataset_name"])
            # Auto-set unit system based on metadata
            if info.get("is_si") is True:
                self.use_si.setCurrentIndex(0)  # SI
            elif info.get("is_si") is False:
                self.use_si.setCurrentIndex(1)  # Canonical
        else:
            self._ds_info.clear_info()

    def _on_dataset_path_validated(
        self,
        path: str,
        exists: bool,
        label: DatasetInfoLabel,
        update_primary: bool = False,
    ) -> None:
        if not exists or not path:
            label.clear_info()
            return
        info = _introspect_h5(path)
        if not info:
            label.clear_info()
            return
        label.show_info(info)
        if update_primary and "dataset_name" in info:
            self.dataset_name.setText(str(info["dataset_name"]))
        if update_primary:
            if info.get("is_si") is True:
                self.use_si.setCurrentIndex(0)
            elif info.get("is_si") is False:
                self.use_si.setCurrentIndex(1)

    def _on_dataset_mode_changed(self, *_args) -> None:
        independent = self.dataset_mode.currentData() == "independent"
        self._single_data_widget.setVisible(not independent)
        self._independent_data_widget.setVisible(independent)
        self.val_ratio.setEnabled(not independent)
        self.val_ratio.setToolTip(
            "Disabled in independent mode because --val-data is supplied directly."
            if independent
            else "Validation fraction in single-dataset mode. 0.1 -> 10% val, 90% train."
        )
        self._refresh_checklist()

    def _on_resume_toggled(self, enabled: bool) -> None:
        widgets = [
            self.resume_from,
            self.resume_checkpoint,
            self.resume_nonstrict,
            self.resume_history_mode,
            *self._resume_path_buttons,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)
        self._refresh_command_preview()
        self._refresh_checklist()

    def _on_loss_feature_toggled(self, *_args) -> None:
        altitude_enabled = self.use_altitude_balanced_loss.isChecked()
        for w in (self.altitude_bin_width_km, self.altitude_min_km, self.altitude_max_km):
            w.setEnabled(altitude_enabled)
        radial_enabled = self.use_radial_cross_loss.isChecked()
        for w in (self.radial_loss_weight, self.cross_loss_weight):
            w.setEnabled(radial_enabled)
        lap_enabled = self.use_laplacian_regularization.isChecked()
        for w in (self.laplacian_weight, self.laplacian_every_n_batches, self.laplacian_subset_size):
            w.setEnabled(lap_enabled)

    # -----------------------------------------------------------------
    # Dependent Parameters
    # -----------------------------------------------------------------
    def _on_activation_changed(self, act: str) -> None:
        is_siren = act.lower() == "sine"
        self.w0_first.setEnabled(is_siren)
        self.w0_hidden.setEnabled(is_siren)
        # Fourier section only makes sense for non-SIREN activations
        self._fourier_section.setEnabled(not is_siren)
        self.fourier_info.setVisible(is_siren)
        if is_siren:
            self.use_fourier.setChecked(False)

    def _on_gradnorm_mode_changed(self, mode: str) -> None:
        is_fixed = mode == "fixed"
        # w_u/w_a spinboxes are only meaningful in fixed mode; ntk_init uses them as seed
        self.w_u.setEnabled(True)  # always show — ntk_init uses them as initial values
        self.w_a.setEnabled(True)
        uses_grad_norms = mode in ("dynamic", "ntk_init")
        self.gradnorm_w_a_min.setEnabled(uses_grad_norms)
        self.gradnorm_w_a_max.setEnabled(uses_grad_norms)

    def _on_device_hint_changed(self, device: str) -> None:
        if device.lower() in ("cpu", "mps"):
            self.no_amp.setChecked(True)
            self.no_amp.setEnabled(False)
        else:
            self.no_amp.setEnabled(True)

    # -----------------------------------------------------------------
    # Phase 2: Training control bar (Start/Stop/Progress always visible)
    # -----------------------------------------------------------------
    def _build_training_control_bar(self) -> QFrame:
        """Compose the always-visible training control bar.

        Re-parents the ProcessPane's primary controls (start/stop/progress)
        into this bar so the existing enable/disable + subprocess logic keeps
        working, and adds secondary/ghost actions next to them."""
        bar = QFrame()
        bar.setObjectName("trainRunBar")
        bar.setStyleSheet(
            "QFrame#trainRunBar {"
            "  background: rgba(11, 16, 32, 0.80);"
            "  border: 1px solid rgba(124, 92, 255, 0.22);"
            "  border-radius: 10px;"
            "}"
        )

        # Flag a user-requested stop so the finish hook can show INTERRUPTED.
        self.runner.btn_stop.clicked.connect(
            lambda: setattr(self, "_user_stopped", True)
        )

        # Compact progress bar (re-parented from the ProcessPane).
        self.runner.progress.setMaximumHeight(22)
        self.runner.progress.setMinimumWidth(160)
        self.runner.progress.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        # Ghost / secondary actions that drive the same slots.
        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.setProperty("kind", "ghost")
        self.btn_clear_log.clicked.connect(self._clear_logs)
        self.btn_open_run = QPushButton("Open Run Folder")
        self.btn_open_run.setProperty("kind", "ghost")
        self.btn_open_run.clicked.connect(self._open_run_folder)
        self.btn_preview_cmd = QPushButton("Preview Command")
        self.btn_preview_cmd.setProperty("kind", "ghost")
        self.btn_preview_cmd.clicked.connect(self._preview_command_popup)
        self.btn_copy_cmd = QPushButton("Copy Command")
        self.btn_copy_cmd.setProperty("kind", "ghost")
        self.btn_copy_cmd.clicked.connect(self._copy_command_preview)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.addWidget(self.runner.btn_start)      # primary
        top_row.addWidget(self.runner.btn_stop)       # danger
        top_row.addWidget(self.runner.progress, 1)    # compact, expanding
        top_row.addWidget(self.btn_enqueue)           # secondary (Add to Queue)
        top_row.addWidget(self.btn_clear_log)
        top_row.addWidget(self.btn_open_run)
        top_row.addWidget(self.btn_preview_cmd)
        top_row.addWidget(self.btn_copy_cmd)

        # Run/output info row.
        self._run_dir_label = QLabel("Output: (auto-timestamped run folder)")
        self._run_dir_label.setStyleSheet("color: #7f91ac; font-size: 11px;")
        self._run_dir_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._workflow_label = QLabel("")
        self._workflow_label.setStyleSheet(
            "color: #8b7cff; font-size: 11px; font-weight: 600;"
        )
        info_row = QHBoxLayout()
        info_row.setContentsMargins(2, 0, 2, 0)
        info_row.setSpacing(12)
        info_row.addWidget(self._run_dir_label, 1)
        info_row.addWidget(self._workflow_label)

        lo = QVBoxLayout()
        lo.setContentsMargins(10, 8, 10, 8)
        lo.setSpacing(6)
        lo.addLayout(top_row)
        lo.addLayout(info_row)
        bar.setLayout(lo)
        return bar

    def _on_model_preset_changed(self, *_args) -> None:
        """Enable manual encoding controls only in Custom preset mode."""
        preset = self.model_preset.currentData() or "custom"
        is_custom = preset == "custom"
        if hasattr(self, "_grp_model_repr"):
            for w in (
                self.use_radial_separation, self.use_radial_decay_encoding,
                self.use_physical_radial_decay_encoding, self.use_real_sh_basis,
                self.physical_radial_decay_max_power,
                self.physical_radial_decay_append_raw,
                self.physical_radial_decay_include_unit,
                self.physical_radial_decay_include_r_scaled,
            ):
                w.setEnabled(is_custom)
            if is_custom:
                self.model_preset_note.setText(
                    "Custom mode: input encoding is controlled by the manual flags below."
                )
            else:
                self.model_preset_note.setText(
                    f"Preset '{preset}' controls the input representation. "
                    "Switch to Custom to edit the manual encoding flags."
                )
        self._refresh_command_preview()

    def _on_log_every_mode_changed(self, *_args) -> None:
        """Disable the fixed-interval spinbox when auto logging is selected."""
        mode = self.log_every_mode.currentData() or "auto"
        if hasattr(self, "log_every"):
            self.log_every.setEnabled(mode == "fixed")
        self._refresh_command_preview()

    def _clear_logs(self) -> None:
        """Clear the raw log text and the structured progress table."""
        if hasattr(self, "runner"):
            self.runner.log.clear()
        if _HAS_DASHBOARD_V2 and self._structured_log is not None:
            self._structured_log.clear()

    def _open_run_folder(self) -> None:
        """Open the most recent output/run directory in the file browser."""
        run_dir = (
            (self.runner._output_dir if hasattr(self, "runner") else "")
            or self.out_dir.text().strip()
        )
        if run_dir and Path(run_dir).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(run_dir))
        else:
            QMessageBox.information(
                self, "No run folder",
                "No output folder is available yet. It is created when training starts "
                "or can be set explicitly in the Configuration tab.",
            )

    def _preview_command_popup(self) -> None:
        """Build the command and show it (also fills the Configuration preview)."""
        self._refresh_command_preview()
        text = self.command_preview.toPlainText().strip()
        if not text:
            QMessageBox.warning(
                self, "Command unavailable",
                self.command_warning.text() or "The current configuration is incomplete.",
            )
            return
        box = QMessageBox(self)
        box.setWindowTitle("Generated Command")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("The training command for the current configuration:")
        box.setDetailedText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _on_workflow_mode_changed(self) -> None:
        mode = self.workflow_mode.currentData() or "train_then_eval"
        labels = {
            "train_only":      "Eğitimi Başlat",
            "eval_only":       "Değerlendirmeyi Başlat",
            "train_then_eval": "Eğit + Değerlendir",
            "queue":           "Kuyruğu Başlat",
        }
        # Update start button label
        if hasattr(self, "runner"):
            self.runner.btn_start.setText(labels.get(mode, "Başlat"))
        # Update the control-bar workflow label
        if hasattr(self, "_workflow_label"):
            self._workflow_label.setText(f"Mode: {labels.get(mode, mode)}")
        self._refresh_checklist()

    def sync_from_cloud(
        self, alt_min: float, alt_max: float, deg_min: int, deg_max: int
    ) -> None:
        """Called by MainWindow when CloudGenTab emits cloud_params_changed.

        Silently updates the altitude range spinboxes (always), so the
        altitude-balanced loss and eval binning are consistent with the dataset.
        The degree values are informational only (no dedicated TrainConfig field).
        """
        self.altitude_min_km.setValue(alt_min)
        self.altitude_max_km.setValue(alt_max)
        if hasattr(self, "runner"):
            self.runner.append(
                f"[UI] Cloud config synced → irtifa: {alt_min:.0f}–{alt_max:.0f} km  |  "
                f"derece: {deg_min}→{deg_max}"
            )

    def _refresh_checklist(self) -> None:
        """Build a compact readiness checklist and update the label."""
        items = []
        ok = True

        script_train = TRAIN_CLI_PATH
        script_eval  = EVAL_CLI_PATH
        mode = self.workflow_mode.currentData() if hasattr(self, "workflow_mode") else "train_then_eval"

        def check(cond: bool, text: str, hard: bool = True) -> None:
            nonlocal ok
            icon = "✓" if cond else ("✗" if hard else "⚠")
            color = "#34d399" if cond else ("#f87171" if hard else "#fbbf24")
            items.append(f'<span style="color:{color}">{icon} {text}</span>')
            if not cond and hard:
                ok = False

        if mode in ("train_only", "train_then_eval", "quick_check", "queue"):
            check(script_train.exists(), "st_lrps.training.cli found")
        if mode in ("eval_only", "train_then_eval"):
            check(script_eval.exists(), "st_lrps.evaluation.cli found")

        dataset_mode = self.dataset_mode.currentData() if hasattr(self, "dataset_mode") else "single"
        if mode != "eval_only":
            resume_active = bool(
                hasattr(self, "resume_enabled") and self.resume_enabled.isChecked()
            )
            if resume_active:
                rp = self.resume_from.text().strip() if hasattr(self, "resume_from") else ""
                if rp:
                    check(Path(rp).exists(), f"Resume source exists: {Path(rp).name}", hard=True)
                else:
                    check(False, "Resume source is required", hard=True)
                items.append(
                    '<span style="color:#7c8dc7">ℹ Resume mode: dataset/output may be inferred from previous run config</span>'
                )
            elif dataset_mode == "single":
                dp = self.data.text().strip() if hasattr(self, "data") else ""
                if dp:
                    check(Path(dp).is_file(), f"Dataset exists: {Path(dp).name}")
                else:
                    items.append('<span style="color:#7c8dc7">ℹ No dataset path → auto-discover</span>')
            else:
                tp = self.train_data.text().strip() if hasattr(self, "train_data") else ""
                vp = self.val_data.text().strip() if hasattr(self, "val_data") else ""
                check(bool(tp) and Path(tp).is_file(), "Train dataset exists", hard=True)
                check(bool(vp) and Path(vp).is_file(), "Val dataset exists", hard=True)

        if mode in ("eval_only",):
            md = self.out_dir.text().strip() if hasattr(self, "out_dir") else ""
            check(bool(md) and Path(md).is_dir(), "Model dir exists (eval only)", hard=True)

        if not hasattr(self, "_checklist_label"):
            return

        if not items:
            self._checklist_label.setVisible(False)
            return

        self._checklist_label.setText("  ".join(items))
        self._checklist_label.setVisible(True)

        # Enable/disable start button based on hard requirements
        if hasattr(self, "runner"):
            self.runner.btn_start.setEnabled(ok)

    # -----------------------------------------------------------------
    # Preset System
    # -----------------------------------------------------------------
    def _refresh_preset_list(self) -> None:
        self._preset_combo.clear()
        for name in _BUILTIN_PRESETS:
            self._preset_combo.addItem(f"⚙  {name}", name)
        user = _load_user_presets()
        if user:
            self._preset_combo.insertSeparator(self._preset_combo.count())
            for name in user:
                self._preset_combo.addItem(f"👤  {name}", name)

    def _current_preset_key(self) -> str:
        return self._preset_combo.currentData() or ""

    def _collect_config(self) -> Dict[str, Any]:
        return {
            # Dataset routing
            "dataset_mode": self.dataset_mode.currentData() or "single",
            "data": self.data.text(),
            "train_data": self.train_data.text(),
            "val_data": self.val_data.text(),
            "test_data": self.test_data.text(),
            "ood_data": self.ood_data.text(),
            "suite_manifest": getattr(self, "applied_suite_manifest_path", ""),
            "out_dir": self.out_dir.text(),
            "dataset_name": self.dataset_name.text(),
            "val_ratio": self.val_ratio.value(),
            # Resume
            "resume_enabled": self.resume_enabled.isChecked(),
            "resume_from": self.resume_from.text(),
            "resume_checkpoint": self.resume_checkpoint.currentData() or "last",
            "resume_nonstrict": self.resume_nonstrict.isChecked(),
            "resume_history_mode": self.resume_history_mode.currentData() or "append",
            # Architecture
            "hidden": self.hidden.value(),
            "depth": self.depth.value(),
            "activation": self.activation.currentText(),
            "w0_first": self.w0_first.value(),
            "w0_hidden": self.w0_hidden.value(),
            "dropout": self.dropout.value(),
            "use_fourier": self.use_fourier.isChecked(),
            "fourier_n": self.fourier_n.value(),
            "fourier_sigma": self.fourier_sigma.value(),
            "fourier_append_raw": self.fourier_append_raw.isChecked(),
            # Optimization
            "epochs": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "lr": self.lr.value(),
            "weight_decay": self.weight_decay.value(),
            "output_head_lr_mult": self.output_head_lr_mult.value(),
            "t_max": self.t_max.value(),
            "warmup_epochs": self.warmup_epochs.value(),
            "min_lr_ratio": self.min_lr_ratio.value(),
            "patience": self.patience.value(),
            "no_amp": self.no_amp.isChecked(),
            # Physics
            "w_u": self.w_u.value(),
            "w_a": self.w_a.value(),
            "gradnorm_mode": self.gradnorm_mode.currentText(),
            "gradnorm_w_a_min": self.gradnorm_w_a_min.value(),
            "gradnorm_w_a_max": self.gradnorm_w_a_max.value(),
            "potential_only_epochs": self.potential_only_epochs.value(),
            "accel_ramp_epochs": self.accel_ramp_epochs.value(),
            "accel_min_factor": self.accel_min_factor.value(),
            "a_sign": self.a_sign.currentText(),
            "use_si_index": self.use_si.currentIndex(),
            # Direction loss
            "direction_loss_weight": self.direction_loss_weight.value(),
            "direction_loss_start_epoch": self.direction_loss_start_epoch.value(),
            "direction_loss_ramp_epochs": self.direction_loss_ramp_epochs.value(),
            "direction_loss_floor_abs": self.direction_loss_floor_abs.value(),
            "best_ckpt_start_epoch": self.best_ckpt_start_epoch.value(),
            "checkpoint_settle_epochs": self.checkpoint_settle_epochs.value(),
            # Field-structure losses
            "use_altitude_balanced_loss": self.use_altitude_balanced_loss.isChecked(),
            "altitude_bin_width_km": self.altitude_bin_width_km.value(),
            "altitude_min_km": self.altitude_min_km.value(),
            "altitude_max_km": self.altitude_max_km.value(),
            "use_radial_cross_loss": self.use_radial_cross_loss.isChecked(),
            "radial_loss_weight": self.radial_loss_weight.value(),
            "cross_loss_weight": self.cross_loss_weight.value(),
            "use_laplacian_regularization": self.use_laplacian_regularization.isChecked(),
            "laplacian_weight": self.laplacian_weight.value(),
            "laplacian_every_n_batches": self.laplacian_every_n_batches.value(),
            "laplacian_subset_size": self.laplacian_subset_size.value(),
            # Advanced / perf
            "max_grad_norm": self.max_grad_norm.value(),
            "num_workers": self.num_workers.value(),
            "prefetch_factor": self.prefetch_factor.value(),
            "cache_rows": self.cache_rows.value(),
            "fit_rows": self.fit_rows.value(),
            "seed": self.seed.value(),
            "split_seed": self.split_seed.value(),
            "log_every": self.log_every.value(),
            "log_every_mode": self.log_every_mode.currentData() or "auto",
            "preload_data": self.preload_data.isChecked(),
            "auto_preload_mb": self.auto_preload_mb.value(),
            "pin_memory": self.pin_memory.isChecked(),
            "quick_check": self.quick_check.isChecked(),
            "max_train_batches": self.max_train_batches.value(),
            "max_val_batches": self.max_val_batches.value(),
            # PINN architecture
            "use_residual_blocks": self.use_residual_blocks.isChecked(),
            "n_bands": self.n_bands.value(),
            "grad_accumulation_steps": self.grad_accumulation_steps.value(),
            "n_hutchinson_samples": self.n_hutchinson_samples.value(),
            # Model representation (input encoding)
            "model_preset": self.model_preset.currentData() or "custom",
            "use_radial_separation": self.use_radial_separation.isChecked(),
            "use_radial_decay_encoding": self.use_radial_decay_encoding.isChecked(),
            "use_physical_radial_decay_encoding": self.use_physical_radial_decay_encoding.isChecked(),
            "use_real_sh_basis": self.use_real_sh_basis.isChecked(),
            "physical_radial_decay_max_power": self.physical_radial_decay_max_power.value(),
            "physical_radial_decay_append_raw": self.physical_radial_decay_append_raw.isChecked(),
            "physical_radial_decay_include_unit": self.physical_radial_decay_include_unit.isChecked(),
            "physical_radial_decay_include_r_scaled": self.physical_radial_decay_include_r_scaled.isChecked(),
            # Workflow
            "workflow_mode": self.workflow_mode.currentData() or "train_then_eval",
            "extra_args": self.extra_args.text(),
        }

    def _apply_config(self, cfg: Dict[str, Any]) -> None:
        _map_int = {
            "hidden": self.hidden, "depth": self.depth, "epochs": self.epochs,
            "batch_size": self.batch_size, "t_max": self.t_max,
            "warmup_epochs": self.warmup_epochs, "patience": self.patience,
            "num_workers": self.num_workers, "prefetch_factor": self.prefetch_factor,
            "cache_rows": self.cache_rows,
            "fit_rows": self.fit_rows, "seed": self.seed, "split_seed": self.split_seed,
            "log_every": self.log_every,
            "potential_only_epochs": self.potential_only_epochs,
            "accel_ramp_epochs": self.accel_ramp_epochs,
            "direction_loss_start_epoch": self.direction_loss_start_epoch,
            "direction_loss_ramp_epochs": self.direction_loss_ramp_epochs,
            "best_ckpt_start_epoch": self.best_ckpt_start_epoch,
            "checkpoint_settle_epochs": self.checkpoint_settle_epochs,
            "fourier_n": self.fourier_n,
            "laplacian_every_n_batches": self.laplacian_every_n_batches,
            "laplacian_subset_size": self.laplacian_subset_size,
            "n_bands": self.n_bands,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "n_hutchinson_samples": self.n_hutchinson_samples,
            "max_train_batches": self.max_train_batches,
            "max_val_batches": self.max_val_batches,
        }
        _map_float = {
            "dropout": self.dropout, "lr": self.lr, "weight_decay": self.weight_decay,
            "output_head_lr_mult": self.output_head_lr_mult,
            "min_lr_ratio": self.min_lr_ratio,
            "w_u": self.w_u, "w_a": self.w_a,
            "gradnorm_w_a_min": self.gradnorm_w_a_min,
            "gradnorm_w_a_max": self.gradnorm_w_a_max,
            "accel_min_factor": self.accel_min_factor,
            "direction_loss_weight": self.direction_loss_weight,
            "direction_loss_floor_abs": self.direction_loss_floor_abs,
            "max_grad_norm": self.max_grad_norm,
            "auto_preload_mb": self.auto_preload_mb,
            "fourier_sigma": self.fourier_sigma,
            "w0_first": self.w0_first, "w0_hidden": self.w0_hidden,
            "altitude_bin_width_km": self.altitude_bin_width_km,
            "altitude_min_km": self.altitude_min_km,
            "altitude_max_km": self.altitude_max_km,
            "radial_loss_weight": self.radial_loss_weight,
            "cross_loss_weight": self.cross_loss_weight,
            "laplacian_weight": self.laplacian_weight,
            "val_ratio": self.val_ratio,
        }
        for key, widget in _map_int.items():
            if key in cfg:
                try:
                    widget.setValue(int(cfg[key]))
                except Exception:
                    pass
        for key, widget in _map_float.items():
            if key in cfg:
                try:
                    widget.setValue(float(cfg[key]))
                except Exception:
                    pass
        if "activation" in cfg:
            self.activation.setCurrentText(str(cfg["activation"]))
        if "gradnorm_mode" in cfg:
            self.gradnorm_mode.setCurrentText(str(cfg["gradnorm_mode"]))
        if "no_amp" in cfg:
            self.no_amp.setChecked(bool(cfg["no_amp"]))
        if "use_fourier" in cfg:
            self.use_fourier.setChecked(bool(cfg["use_fourier"]))
        if "fourier_append_raw" in cfg:
            self.fourier_append_raw.setChecked(bool(cfg["fourier_append_raw"]))
        if "preload_data" in cfg:
            self.preload_data.setChecked(bool(cfg["preload_data"]))
        if "pin_memory" in cfg:
            self.pin_memory.setChecked(bool(cfg["pin_memory"]))
        if "quick_check" in cfg:
            self.quick_check.setChecked(bool(cfg["quick_check"]))
        if "use_altitude_balanced_loss" in cfg:
            self.use_altitude_balanced_loss.setChecked(bool(cfg["use_altitude_balanced_loss"]))
        if "use_radial_cross_loss" in cfg:
            self.use_radial_cross_loss.setChecked(bool(cfg["use_radial_cross_loss"]))
        if "use_laplacian_regularization" in cfg:
            self.use_laplacian_regularization.setChecked(bool(cfg["use_laplacian_regularization"]))
        if "use_residual_blocks" in cfg:
            self.use_residual_blocks.setChecked(bool(cfg["use_residual_blocks"]))
        # Model representation (input encoding) — backward compatible: old
        # profiles without these keys keep the current widget values.
        for key, widget in (
            ("use_radial_separation", self.use_radial_separation),
            ("use_radial_decay_encoding", self.use_radial_decay_encoding),
            ("use_physical_radial_decay_encoding", self.use_physical_radial_decay_encoding),
            ("use_real_sh_basis", self.use_real_sh_basis),
            ("physical_radial_decay_append_raw", self.physical_radial_decay_append_raw),
            ("physical_radial_decay_include_unit", self.physical_radial_decay_include_unit),
            ("physical_radial_decay_include_r_scaled", self.physical_radial_decay_include_r_scaled),
        ):
            if key in cfg:
                widget.setChecked(bool(cfg[key]))
        if "physical_radial_decay_max_power" in cfg:
            try:
                self.physical_radial_decay_max_power.setValue(int(cfg["physical_radial_decay_max_power"]))
            except Exception:
                pass
        if "model_preset" in cfg:
            idx = self.model_preset.findData(str(cfg["model_preset"]))
            if idx >= 0:
                self.model_preset.setCurrentIndex(idx)
        if "log_every_mode" in cfg:
            idx = self.log_every_mode.findData(str(cfg["log_every_mode"]))
            if idx >= 0:
                self.log_every_mode.setCurrentIndex(idx)
        if "resume_enabled" in cfg:
            self.resume_enabled.setChecked(bool(cfg["resume_enabled"]))
        if "resume_nonstrict" in cfg:
            self.resume_nonstrict.setChecked(bool(cfg["resume_nonstrict"]))
        if "a_sign" in cfg:
            self.a_sign.setCurrentText(str(cfg["a_sign"]))
        if "use_si_index" in cfg:
            self.use_si.setCurrentIndex(int(cfg["use_si_index"]))
        if "resume_checkpoint" in cfg:
            idx = self.resume_checkpoint.findData(str(cfg["resume_checkpoint"]))
            if idx >= 0:
                self.resume_checkpoint.setCurrentIndex(idx)
        if "resume_history_mode" in cfg:
            idx = self.resume_history_mode.findData(str(cfg["resume_history_mode"]))
            if idx >= 0:
                self.resume_history_mode.setCurrentIndex(idx)
        if "workflow_mode" in cfg:
            idx = self.workflow_mode.findData(str(cfg["workflow_mode"]))
            if idx >= 0:
                self.workflow_mode.setCurrentIndex(idx)
        if "dataset_mode" in cfg:
            mode = str(cfg["dataset_mode"])
            idx = self.dataset_mode.findData(mode)
            if idx >= 0:
                self.dataset_mode.setCurrentIndex(idx)
        for key, widget in (
            ("data", self.data),
            ("train_data", self.train_data),
            ("val_data", self.val_data),
            ("test_data", self.test_data),
            ("ood_data", self.ood_data),
            ("out_dir", self.out_dir),
            ("dataset_name", self.dataset_name),
            ("resume_from", self.resume_from),
            ("extra_args", self.extra_args),
        ):
            if key in cfg:
                widget.setText(str(cfg[key]))
        if "suite_manifest" in cfg:
            self.applied_suite_manifest_path = str(cfg["suite_manifest"] or "")
            if self.applied_suite_manifest_path:
                self._suite_manifest_label.setText(self.applied_suite_manifest_path)
                self._suite_manifest_label.setStyleSheet("color: #6ee7b7; font-size: 10px;")
            else:
                self._suite_manifest_label.setText("(no suite applied)")
                self._suite_manifest_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self._on_dataset_mode_changed()
        self._on_resume_toggled(self.resume_enabled.isChecked())
        self._on_loss_feature_toggled()
        if hasattr(self, "model_preset"):
            self._on_model_preset_changed()
        if hasattr(self, "log_every_mode"):
            self._on_log_every_mode_changed()

    def _load_preset(self) -> None:
        key = self._current_preset_key()
        if not key:
            return
        cfg = _BUILTIN_PRESETS.get(key) or _load_user_presets().get(key)
        if cfg:
            self._apply_config(cfg)
            self.runner.append(f"[UI] Profil yüklendi: {key}")

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Profil Kaydet", "Profil adı:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in _BUILTIN_PRESETS:
            QMessageBox.warning(
                self, "Engellendi", f"'{name}' yerleşik profil, değiştirilemez."
            )
            return
        _save_user_preset(name, self._collect_config())
        self._refresh_preset_list()
        for i in range(self._preset_combo.count()):
            if self._preset_combo.itemData(i) == name:
                self._preset_combo.setCurrentIndex(i)
                break
        self.runner.append(f"[UI] Profil kaydedildi: {name}")

    def _delete_preset(self) -> None:
        key = self._current_preset_key()
        if not key:
            return
        if key in _BUILTIN_PRESETS:
            QMessageBox.information(self, "Silinemez", f"'{key}' yerleşik profil.")
            return
        reply = QMessageBox.question(
            self,
            "Sil",
            f"'{key}' silinsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            _delete_user_preset(key)
            self._refresh_preset_list()

    # -----------------------------------------------------------------
    # File Dialogs
    # -----------------------------------------------------------------
    def _pick_data(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Dataset Seç",
            self.data.text() or str(SCRIPT_DIR),
            "HDF5 (*.h5 *.hdf5);;All (*.*)",
        )
        if fn:
            self.data.setText(_norm_path(fn))

    def _pick_dataset_path(self, target: ValidatedPathEdit, title: str) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            title,
            target.text() or str(SCRIPT_DIR),
            "HDF5 (*.h5 *.hdf5);;All (*.*)",
        )
        if fn:
            target.setText(_norm_path(fn))

    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Çıktı Klasörü", self.out_dir.text() or str(SCRIPT_DIR)
        )
        if d:
            self.out_dir.setText(_norm_path(d))

    def _pick_resume_run(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self,
            "Resume Run Directory",
            self.resume_from.text() or self.out_dir.text() or str(SCRIPT_DIR),
        )
        if d:
            self.resume_from.setText(_norm_path(d))

    def _pick_resume_checkpoint(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Resume Checkpoint",
            self.resume_from.text() or self.out_dir.text() or str(SCRIPT_DIR),
            "PyTorch checkpoints (*.pt);;All (*.*)",
        )
        if fn:
            self.resume_from.setText(_norm_path(fn))

    # -----------------------------------------------------------------
    # QSettings
    # -----------------------------------------------------------------
    def _save_settings(self) -> None:
        s = _settings()
        s.beginGroup("train")
        s.setValue("data_path", self.data.text())
        s.setValue("out_dir", self.out_dir.text())
        s.setValue("dataset_name", self.dataset_name.text())
        s.setValue("val_ratio", self.val_ratio.value())
        for k, v in self._collect_config().items():
            s.setValue(k, v)
        s.setValue("device_hint", self.device_hint.currentText())
        s.endGroup()
        s.sync()

    def _restore_settings(self) -> None:
        s = _settings()
        s.beginGroup("train")
        if s.contains("data_path"):
            self.data.setText(str(s.value("data_path", "")))
        if s.contains("out_dir"):
            self.out_dir.setText(str(s.value("out_dir", "")))
        if s.contains("dataset_name"):
            self.dataset_name.setText(str(s.value("dataset_name", "data")))
        if s.contains("val_ratio"):
            self.val_ratio.setValue(float(s.value("val_ratio", 0.1)))
        cfg = {}
        for key in self._collect_config():
            if s.contains(key):
                cfg[key] = s.value(key)
        if cfg:
            _bool_keys = {
                "no_amp", "use_fourier", "fourier_append_raw",
                "preload_data", "pin_memory", "quick_check",
                "use_altitude_balanced_loss", "use_radial_cross_loss",
                "use_laplacian_regularization", "use_residual_blocks",
                "resume_enabled", "resume_nonstrict",
                "use_radial_separation", "use_radial_decay_encoding",
                "use_physical_radial_decay_encoding", "use_real_sh_basis",
                "physical_radial_decay_append_raw",
                "physical_radial_decay_include_unit",
                "physical_radial_decay_include_r_scaled",
            }
            _int_keys = {
                "hidden", "depth", "epochs", "batch_size", "t_max",
                "warmup_epochs", "patience", "num_workers", "prefetch_factor",
                "cache_rows",
                "fit_rows", "seed", "split_seed", "log_every", "use_si_index",
                "potential_only_epochs", "accel_ramp_epochs",
                "direction_loss_start_epoch", "direction_loss_ramp_epochs",
                "best_ckpt_start_epoch", "checkpoint_settle_epochs",
                "fourier_n", "laplacian_every_n_batches",
                "laplacian_subset_size",
                "n_bands", "grad_accumulation_steps", "n_hutchinson_samples",
                "max_train_batches", "max_val_batches",
                "physical_radial_decay_max_power",
            }
            _float_keys = {
                "dropout", "lr", "weight_decay", "output_head_lr_mult",
                "min_lr_ratio", "w_u", "w_a", "max_grad_norm",
                "gradnorm_w_a_min", "gradnorm_w_a_max", "accel_min_factor",
                "direction_loss_weight", "direction_loss_floor_abs",
                "auto_preload_mb", "fourier_sigma", "w0_first", "w0_hidden",
                "altitude_bin_width_km", "altitude_min_km",
                "altitude_max_km", "radial_loss_weight",
                "cross_loss_weight", "laplacian_weight", "val_ratio",
            }
            for k in _bool_keys:
                if k in cfg:
                    cfg[k] = str(cfg[k]).lower() == "true"
            for k in _int_keys:
                if k in cfg:
                    try:
                        cfg[k] = int(cfg[k])
                    except Exception:
                        pass
            for k in _float_keys:
                if k in cfg:
                    try:
                        cfg[k] = float(cfg[k])
                    except Exception:
                        pass
            self._apply_config(cfg)
        if s.contains("device_hint"):
            self.device_hint.setCurrentText(str(s.value("device_hint", "auto")))
        s.endGroup()

    # -----------------------------------------------------------------
    # Build CLI args from current widgets
    # -----------------------------------------------------------------
    def _build_args(self, show_errors: bool = True) -> Optional[List[str]]:
        """Build the CLI argument list. Returns None on validation error."""
        def fail(title: str, message: str) -> Optional[List[str]]:
            if show_errors:
                QMessageBox.critical(self, title, message)
            else:
                self.command_warning.setText(message)
            return None

        if not show_errors:
            self.command_warning.setText("")

        if not TRAIN_CLI_PATH.exists():
            return fail("Missing script", "st_lrps/training/cli.py not found in the repository.")

        args = ["-u", "-m", TRAIN_CLI_MODULE]
        resume_active = self.resume_enabled.isChecked()
        if resume_active:
            resume_path = self.resume_from.text().strip()
            if not resume_path:
                return fail(
                    "Missing resume source",
                    "Resume mode requires --resume-from. Select a run directory, checkpoints directory, or .pt checkpoint.",
                )
            if not Path(resume_path).exists():
                return fail("Missing resume source", f"Resume source not found:\n{resume_path}")
            args += ["--resume-from", resume_path]
            args += ["--resume-checkpoint", self.resume_checkpoint.currentData() or "last"]
            if self.resume_nonstrict.isChecked():
                args += ["--resume-nonstrict"]
            history_mode = self.resume_history_mode.currentData() or "append"
            if history_mode == "overwrite":
                args += ["--resume-overwrite-history"]
            else:
                args += ["--resume-append-history"]

        dataset_mode = self.dataset_mode.currentData() or "single"
        if dataset_mode == "independent":
            train_path = self.train_data.text().strip()
            val_path = self.val_data.text().strip()
            if not resume_active and (not train_path or not val_path):
                return fail(
                    "Missing dataset",
                    "Independent mode requires both --train-data and --val-data.",
                )
            for label, path in (("Train dataset", train_path), ("Validation dataset", val_path)):
                if path and not Path(path).exists():
                    return fail("Missing dataset", f"{label} not found:\n{path}")
            if train_path:
                args += ["--train-data", train_path]
            if val_path:
                args += ["--val-data", val_path]
            for flag, path in (
                ("--test-data", self.test_data.text().strip()),
                ("--ood-data", self.ood_data.text().strip()),
            ):
                if path:
                    if not Path(path).exists():
                        return fail("Missing dataset", f"{flag} path not found:\n{path}")
                    args += [flag, path]
            args += ["--split-seed", str(self.split_seed.value())]
        else:
            data_path = self.data.text().strip()
            if data_path:
                if not Path(data_path).exists():
                    return fail("Missing dataset", f"Dataset not found:\n{data_path}")
                args += ["--data", data_path]
            args += ["--val-fraction", str(self.val_ratio.value())]
            args += ["--split-seed", str(self.split_seed.value())]

        out_dir = self.out_dir.text().strip()
        if out_dir:
            args += ["--out", out_dir]

        args += ["--dataset-name", self.dataset_name.text().strip() or "data"]

        # Architecture
        args += ["--hidden", str(self.hidden.value())]
        args += ["--depth", str(self.depth.value())]
        act = self.activation.currentText().strip()
        args += ["--activation", act]
        args += ["--w0-first", str(self.w0_first.value())]
        args += ["--w0-hidden", str(self.w0_hidden.value())]
        args += ["--dropout", str(self.dropout.value())]

        # Model representation / input encoding (Phase 10).
        # Non-custom presets fully define the representation: emit only
        # --model-preset and force fourier off (the backend's apply_model_preset
        # raises if a non-custom preset is combined with active manual encodings).
        preset = self.model_preset.currentData() or "custom"
        args += ["--model-preset", preset]
        if preset == "custom":
            if act != "sine" and self.use_fourier.isChecked():
                args += ["--use-fourier"]
                args += ["--fourier-n", str(self.fourier_n.value())]
                args += ["--fourier-sigma", str(self.fourier_sigma.value())]
                if self.fourier_append_raw.isChecked():
                    args += ["--fourier-append-raw"]
                else:
                    args += ["--no-fourier-append-raw"]
            else:
                args += ["--no-fourier"]
            args += (
                ["--use-radial-separation"] if self.use_radial_separation.isChecked()
                else ["--no-radial-separation"]
            )
            args += (
                ["--use-radial-decay-encoding"] if self.use_radial_decay_encoding.isChecked()
                else ["--no-radial-decay-encoding"]
            )
            if self.use_physical_radial_decay_encoding.isChecked():
                args += ["--use-physical-radial-decay-encoding"]
                args += ["--physical-radial-decay-max-power",
                         str(self.physical_radial_decay_max_power.value())]
                args += (
                    ["--physical-radial-decay-append-raw"]
                    if self.physical_radial_decay_append_raw.isChecked()
                    else ["--no-physical-radial-decay-append-raw"]
                )
                args += (
                    ["--physical-radial-decay-include-unit"]
                    if self.physical_radial_decay_include_unit.isChecked()
                    else ["--no-physical-radial-decay-include-unit"]
                )
                args += (
                    ["--physical-radial-decay-include-r-scaled"]
                    if self.physical_radial_decay_include_r_scaled.isChecked()
                    else ["--no-physical-radial-decay-include-r-scaled"]
                )
            else:
                args += ["--no-physical-radial-decay-encoding"]
            args += (
                ["--use-real-sh-basis"] if self.use_real_sh_basis.isChecked()
                else ["--no-real-sh-basis"]
            )
        else:
            args += ["--no-fourier"]

        # Optimization
        args += ["--epochs", str(self.epochs.value())]
        args += ["--batch-size", str(self.batch_size.value())]
        args += ["--lr", str(self.lr.value())]
        args += ["--weight-decay", str(self.weight_decay.value())]
        args += ["--output-head-lr-mult", str(self.output_head_lr_mult.value())]
        args += ["--max-grad-norm", str(self.max_grad_norm.value())]
        args += ["--t-max", str(self.t_max.value())]
        args += ["--warmup-epochs", str(self.warmup_epochs.value())]
        args += ["--min-lr-ratio", str(self.min_lr_ratio.value())]
        args += ["--patience", str(self.patience.value())]
        if self.no_amp.isChecked():
            args += ["--no-amp"]

        # Physics & Sobolev
        args += ["--w-u", str(self.w_u.value())]
        args += ["--w-a", str(self.w_a.value())]
        args += ["--gradnorm-mode", self.gradnorm_mode.currentText()]
        args += ["--gradnorm-w-a-min", str(self.gradnorm_w_a_min.value())]
        args += ["--gradnorm-w-a-max", str(self.gradnorm_w_a_max.value())]
        args += ["--potential-only-epochs", str(self.potential_only_epochs.value())]
        args += ["--accel-ramp-epochs", str(self.accel_ramp_epochs.value())]
        args += ["--accel-min-factor", str(self.accel_min_factor.value())]
        a_sign_text = self.a_sign.currentText().strip()
        args += ["--a-sign", a_sign_text if a_sign_text in ("+1", "-1") else "auto"]
        if self.use_si.currentIndex() == 0:
            args += ["--use-si"]
        else:
            args += ["--no-si"]

        # Direction loss
        args += ["--direction-loss-weight", str(self.direction_loss_weight.value())]
        args += ["--direction-loss-start-epoch", str(self.direction_loss_start_epoch.value())]
        args += ["--direction-loss-ramp-epochs", str(self.direction_loss_ramp_epochs.value())]
        args += ["--direction-loss-floor-abs", str(self.direction_loss_floor_abs.value())]
        args += ["--best-ckpt-start-epoch", str(self.best_ckpt_start_epoch.value())]
        args += ["--checkpoint-settle-epochs", str(self.checkpoint_settle_epochs.value())]

        # Field-structure losses
        if self.use_altitude_balanced_loss.isChecked():
            args += ["--use-altitude-balanced-loss"]
        args += ["--altitude-bin-width-km", str(self.altitude_bin_width_km.value())]
        args += ["--altitude-min-km", str(self.altitude_min_km.value())]
        args += ["--altitude-max-km", str(self.altitude_max_km.value())]
        if self.use_radial_cross_loss.isChecked():
            args += ["--use-radial-cross-loss"]
        args += ["--radial-loss-weight", str(self.radial_loss_weight.value())]
        args += ["--cross-loss-weight", str(self.cross_loss_weight.value())]
        if self.use_laplacian_regularization.isChecked():
            args += ["--use-laplacian-regularization"]
        args += ["--laplacian-weight", str(self.laplacian_weight.value())]
        args += ["--laplacian-every-n-batches", str(self.laplacian_every_n_batches.value())]
        args += ["--laplacian-subset-size", str(self.laplacian_subset_size.value())]

        # Performance
        args += ["--num-workers", str(self.num_workers.value())]
        pf = self.prefetch_factor.value()
        if pf > 0:
            args += ["--prefetch-factor", str(pf)]
        args += ["--cache-rows", str(self.cache_rows.value())]
        args += ["--fit-rows", str(self.fit_rows.value())]
        args += ["--seed", str(self.seed.value())]
        log_mode = self.log_every_mode.currentData() or "auto"
        args += ["--log-every-mode", log_mode]
        args += ["--log-every", str(self.log_every.value())]
        if self.preload_data.isChecked():
            args += ["--preload-data"]
        args += ["--auto-preload-mb", str(self.auto_preload_mb.value())]
        if self.pin_memory.isChecked():
            args += ["--pin-memory"]
        else:
            args += ["--no-pin-memory"]
        # NOTE: --quick-check / --max-train-batches / --max-val-batches are
        # developer-only debug flags and are intentionally NOT emitted by the
        # normal Studio workflow.

        # PINN architecture
        if self.use_residual_blocks.isChecked():
            args += ["--use-residual-blocks"]
        else:
            args += ["--no-residual-blocks"]
        args += ["--n-bands", str(self.n_bands.value())]
        args += ["--grad-accumulation-steps", str(self.grad_accumulation_steps.value())]
        args += ["--n-hutchinson-samples", str(self.n_hutchinson_samples.value())]

        # Suite manifest provenance (set when a dataset suite is applied)
        _sm = getattr(self, "applied_suite_manifest_path", "") or ""
        if _sm and Path(_sm).is_file():
            args += ["--suite-manifest", _sm]

        extra = self.extra_args.text().strip()
        if extra:
            extra_args, err = _split_cli_args(extra)
            if err:
                return fail("Invalid extra CLI arguments", err)
            if resume_active:
                resume_flags = {
                    "--resume-from",
                    "--resume-checkpoint",
                    "--resume-nonstrict",
                    "--resume-append-history",
                    "--resume-overwrite-history",
                }
                if any(flag in resume_flags for flag in (extra_args or [])):
                    self.command_warning.setText(
                        "Extra args include resume flags; they are appended last and may override UI resume settings."
                    )
            args += extra_args or []
        return args

    def _build_eval_args(
        self,
        model_dir: str,
        *,
        data_path: Optional[str] = None,
        test_data: Optional[str] = None,
        ood_data: Optional[str] = None,
        use_config_datasets: bool = False,
        out_dir: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Build CLI argument list for the evaluation module (st_lrps.evaluation.cli)."""
        if not EVAL_CLI_PATH.exists():
            return None
        args = ["-u", "-m", EVAL_CLI_MODULE]
        if model_dir:
            args += ["--model-dir", model_dir]
        primary_data = data_path or test_data
        if primary_data and Path(primary_data).exists():
            args += ["--data", primary_data]
        if test_data and Path(test_data).exists():
            same_as_primary = bool(primary_data) and Path(test_data).resolve() == Path(primary_data).resolve()
            if not same_as_primary:
                args += ["--test-data", test_data]
        if ood_data and Path(ood_data).exists():
            args += ["--ood-data", ood_data]
        if use_config_datasets:
            args += ["--use-config-datasets"]
        args += ["--dataset-name", self.dataset_name.text().strip() or "data"]
        if out_dir:
            args += ["--out", out_dir]
        # Hardware
        dev = self.device_hint.currentText()
        args += ["--device", dev if dev != "auto" else "auto"]
        args += ["--batch-size", str(self.batch_size.value())]
        a_sign_text = self.a_sign.currentText().strip()
        args += ["--a-sign", "1.0" if a_sign_text not in ("+1", "-1") else a_sign_text.replace("+", "")]
        # Spatial
        args += ["--alt-bin-km", str(self.altitude_bin_width_km.value())]
        args += ["--start", "0"]
        args += ["--max-points-for-plots", "500000"]
        return args

    def _refresh_command_preview(self) -> None:
        args = self._build_args(show_errors=False)
        if args is None:
            self.command_preview.clear()
            return
        self.command_preview.setPlainText(_format_command(sys.executable, args))
        if not self.command_warning.text():
            self.command_warning.setText("Command is valid for the current UI fields.")

    def _copy_command_preview(self) -> None:
        if not self.command_preview.toPlainText().strip():
            self._refresh_command_preview()
        QGuiApplication.clipboard().setText(self.command_preview.toPlainText())

    # -----------------------------------------------------------------
    # Start Training (single run)
    # -----------------------------------------------------------------
    def _start(self) -> None:
        mode = self.workflow_mode.currentData() or "train_then_eval"

        # Evaluate-only: delegate directly to the canonical evaluation module.
        if mode == "eval_only":
            run_dir = self.out_dir.text().strip()
            if not run_dir or not Path(run_dir).is_dir():
                QMessageBox.critical(
                    self,
                    "Model directory missing",
                    "Evaluate only mode requires a valid Output Folder / model directory.\n"
                    "Set the Output Folder to an existing training run directory.",
                )
                return
            eval_args = self._build_eval_args(
                run_dir,
                test_data=self.test_data.text().strip() or None,
                ood_data=self.ood_data.text().strip() or None,
                use_config_datasets=True,
            )
            if eval_args is None:
                QMessageBox.critical(self, "Missing script", "st_lrps/evaluation/cli.py not found.")
                return
            self._save_settings()
            self.runner.progress.setRange(0, 0)
            self.runner.set_output_dir(run_dir)
            self.runner.set_stop_hint("")
            self.runner.start(sys.executable, eval_args, workdir=str(_REPO_ROOT))
            return

        # Queue mode: run queue
        if mode == "queue":
            if not self._queue.is_running():
                self._queue._start_queue()
            return

        args = self._build_args()
        if args is None:
            return

        out_dir = self.out_dir.text().strip()
        resume_source = out_dir or self.resume_from.text().strip()
        self.runner.set_stop_hint(self._resume_stop_hint(resume_source))
        self._epochs_max = int(self.epochs.value())
        self.runner.progress.setRange(0, self._epochs_max)
        self.runner.progress.setValue(0)
        self.runner.progress.setFormat("Epoch %v / %m")

        self._live_plot.clear()
        self.runner.set_output_dir(out_dir if out_dir else "")
        self._set_history_poll_dir(out_dir)
        self._update_run_dir_label(out_dir)
        self._save_settings()
        # Dashboard v2: start ETA tracking and reset dashboard
        if _HAS_DASHBOARD_V2:
            if self._eta_estimator is not None:
                self._eta_estimator.set_total_epochs(int(self.epochs.value()))
                self._eta_estimator.on_training_start()
            if self._epoch_guard is not None:
                self._epoch_guard.reset()        # Phase 8: reset ETA epoch guards
            if self._eta_update_timer is not None:
                self._eta_update_timer.start()
            if self._metrics_store is not None:
                self._metrics_store.clear()
            if self._structured_log is not None:
                self._structured_log.clear()
            if self._kpi_strip is not None:
                self._kpi_strip.reset()
                self._kpi_strip.epoch.set_value(f"0 / {int(self.epochs.value())}")
                self._kpi_strip.phase.set_value("Starting", state="normal")
            if self._time_strip is not None:
                self._time_strip.reset()
            self._update_header_lifecycle("TRAINING")

        # Phase 3: jump to Live Monitor so the user sees charts immediately.
        if hasattr(self, "_page_tabs"):
            self._page_tabs.setCurrentIndex(self._monitor_tab_idx)

        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    # -----------------------------------------------------------------
    # Queue System (Feature #15)
    # -----------------------------------------------------------------
    def _enqueue_current(self) -> None:
        """Add the current parameter configuration to the training queue."""
        args = self._build_args()
        if args is None:
            return
        cfg = self._collect_config()
        label = (
            f"H={cfg['hidden']} D={cfg['depth']} "
            f"E={cfg['epochs']} BS={cfg['batch_size']} "
            f"LR={cfg['lr']:.1e}"
        )
        out_dir = self.out_dir.text().strip()
        self._queue.enqueue(label, args, out_dir, cfg)
        self.runner.append(f"[UI] Kuyruğa eklendi: {label}")

    def _on_queue_job_started(self, job_index: int, args: List[str]) -> None:
        """Called by the queue when it's time to start the next job."""
        self._live_plot.clear()
        self._epochs_max = int(self.epochs.value())
        self.runner.progress.setRange(0, self._epochs_max)
        self.runner.progress.setValue(0)
        self.runner.progress.setFormat("Epoch %v / %m  [Kuyruk]")
        queue_out = self._arg_value(args, "--out") or ""
        self.runner.set_output_dir("")
        self._set_history_poll_dir(queue_out)
        self._update_run_dir_label(queue_out)
        self.runner.set_stop_hint(self._resume_stop_hint(queue_out or self._arg_value(args, "--resume-from") or ""))
        # Reset the live dashboard for each queued job.
        if _HAS_DASHBOARD_V2:
            if self._eta_estimator is not None:
                self._eta_estimator.set_total_epochs(int(self.epochs.value()))
                self._eta_estimator.on_training_start()
            if self._epoch_guard is not None:
                self._epoch_guard.reset()
            if self._eta_update_timer is not None:
                self._eta_update_timer.start()
            if self._metrics_store is not None:
                self._metrics_store.clear()
            if self._structured_log is not None:
                self._structured_log.clear()
            if self._kpi_strip is not None:
                self._kpi_strip.reset()
                self._kpi_strip.phase.set_value("Starting", state="normal")
            if self._time_strip is not None:
                self._time_strip.reset()
            self._update_header_lifecycle("TRAINING")
        self._user_stopped = False
        if hasattr(self, "_page_tabs"):
            self._page_tabs.setCurrentIndex(self._monitor_tab_idx)
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _arg_value(self, args: List[str], flag: str) -> Optional[str]:
        try:
            idx = args.index(flag)
            if idx + 1 < len(args):
                return str(args[idx + 1])
        except ValueError:
            pass
        return None

    def _resume_stop_hint(self, run_dir: str) -> str:
        lines = [
            "[UI] Training can usually be resumed from the last completed epoch.",
            "[UI] Resume is epoch-level; an interrupted mid-epoch batch restarts from the last checkpoint.",
        ]
        run_dir = str(run_dir or "").strip()
        if run_dir:
            lines.append(
                f"[UI] Suggested resume command: {sys.executable} -m {TRAIN_CLI_MODULE} "
                f"--resume-from {run_dir} --epochs {self.epochs.value()}"
            )
        return "\n".join(lines)

    def _set_history_poll_dir(self, run_dir: str) -> None:
        run_dir = str(run_dir or "").strip()
        self._history_poll_path = None
        self._history_poll_mtime = 0.0
        if not run_dir:
            self._history_poll_timer.stop()
            return
        root = Path(run_dir)
        self._history_poll_path = root / "history.jsonl"
        self._history_poll_timer.start()

    def _poll_training_history(self) -> None:
        run_dir = self.runner._output_dir or self.out_dir.text().strip()
        if self._history_poll_path is None and run_dir:
            self._set_history_poll_dir(run_dir)
        if self._history_poll_path is None:
            return
        candidates = [self._history_poll_path]
        if self._history_poll_path.suffix.lower() == ".jsonl":
            candidates.append(self._history_poll_path.with_suffix(".csv"))
        else:
            candidates.append(self._history_poll_path.with_suffix(".jsonl"))
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if path == self._history_poll_path and mtime <= self._history_poll_mtime:
            return
        self._history_poll_path = path
        self._history_poll_mtime = mtime
        self._live_plot.load_history_file(str(path))

    # -----------------------------------------------------------------
    # Post-run hook
    # -----------------------------------------------------------------
    def _on_train_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ) -> None:
        # Determine lifecycle status (distinguish a user-requested stop).
        ok = exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
        if ok:
            status = "COMPLETED"
        elif getattr(self, "_user_stopped", False):
            status = "INTERRUPTED"
        else:
            status = "FAILED"

        # Dashboard v2: stop ETA timer and update status
        if _HAS_DASHBOARD_V2:
            if self._eta_update_timer is not None:
                self._eta_update_timer.stop()
            self._update_header_lifecycle(status)
            if self._kpi_strip is not None:
                phase_state = {
                    "COMPLETED": "success", "FAILED": "danger", "INTERRUPTED": "warning",
                }.get(status, "normal")
                self._kpi_strip.phase.set_value(status.capitalize(), state=phase_state)
            # Phase 7: finalize time metrics — ETA = Done, finish = actual time.
            if self._time_strip is not None:
                from datetime import datetime as _dt
                self._time_strip.set_done(_dt.now().strftime("%H:%M"))
                if self._eta_estimator is not None:
                    self._time_strip.elapsed.set_value(self._eta_estimator.format_elapsed())
        self._user_stopped = False

        self._poll_training_history()
        self._history_poll_timer.stop()
        training_ok = (
            exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
        )

        # --- Discover output dir from log if not already set ---
        if exit_status == QProcess.ExitStatus.NormalExit and not self.runner._output_dir:
            text = self.runner.log.toPlainText()
            for pat in [
                r"(?:out_dir|Output dir|Run dir|Saving to)\s*[:=]\s*(.+)",
                r"Using default output directory:\s*(.+)",
            ]:
                m = re.search(pat, text)
                if m:
                    candidate = m.group(1).strip().strip("'\"")
                    if Path(candidate).is_dir():
                        self.runner.set_output_dir(candidate)
                        self._set_history_poll_dir(candidate)
                        self._poll_training_history()
                        self._history_poll_timer.stop()
                        self.runner.btn_open_folder.setVisible(True)
                        break

        run_dir = self.runner._output_dir or self.out_dir.text().strip()
        self._update_run_dir_label(run_dir)

        # --- Notify the queue so it can advance ---
        if self._queue.is_running():
            self._queue.on_job_finished(exit_code, exit_status)
            return  # Queue mode handles its own chaining

        # --- Auto-evaluate when workflow = "train_then_eval" ---
        mode = self.workflow_mode.currentData() or "train_then_eval"
        if mode != "train_then_eval" or not training_ok:
            if not training_ok and mode == "train_then_eval":
                self.runner.append(
                    "[UI] Training failed — evaluation skipped. "
                    f"(exit_code={exit_code})"
                )
            return

        # Verify checkpoint exists before launching eval
        if run_dir:
            layout = make_run_layout(Path(run_dir)) if make_run_layout is not None else None
            missing: List[str] = []
            if layout is not None:
                if not layout.config_json.exists():
                    missing.append(str(layout.config_json))
                if not layout.scaler_json.exists():
                    missing.append(str(layout.scaler_json))
                if not layout.ckpt_best.exists() and not layout.ckpt_last.exists():
                    missing.append(f"{layout.ckpt_best} or {layout.ckpt_last}")
            if missing:
                self.runner.append(
                    "[UI] Cannot auto-evaluate — missing files:\n  " + "\n  ".join(missing)
                )
                return
            status = _inspect_run_artifacts(run_dir)
            if status.get("warnings"):
                self.runner.append(
                    "[UI] Artifact warnings:\n  " + "\n  ".join(str(item) for item in status["warnings"])
                )
            if layout is not None and not layout.ckpt_best.exists() and layout.ckpt_last.exists():
                self.runner.append(
                    "[UI] ckpt_best.pt was not written yet; evaluator will fall back to ckpt_last.pt. "
                    "This usually means the run ended before direction-aware best-checkpoint tracking began."
                )

        eval_args = self._build_eval_args(
            run_dir,
            test_data=self.test_data.text().strip() or None,
            ood_data=self.ood_data.text().strip() or None,
            use_config_datasets=(
                not self.test_data.text().strip()
                and not self.ood_data.text().strip()
            ),
        )
        if eval_args is None:
            self.runner.append("[UI] st_lrps.evaluation.cli not available — skipping auto-eval.")
            return

        self.runner.append(
            "\n[UI] ─── Training complete ─── launching evaluation …\n"
            f"[UI] Model dir: {run_dir}"
        )
        self._eval_runner = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        self._eval_runner.setProcessEnvironment(env)
        self._eval_runner.setWorkingDirectory(str(_REPO_ROOT))
        self._eval_runner.readyReadStandardOutput.connect(
            lambda: self._on_eval_stdout(self._eval_runner)
        )
        self._eval_runner.readyReadStandardError.connect(
            lambda: self._on_eval_stdout(self._eval_runner)
        )
        self._eval_runner.finished.connect(self._on_auto_eval_finished)
        self._eval_run_dir = run_dir
        self._eval_runner.start(sys.executable, eval_args)

    def _on_eval_stdout(self, proc: "QProcess") -> None:
        raw = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        raw += bytes(proc.readAllStandardError()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            self.runner.append(f"[EVAL] {line}")

    def _on_auto_eval_finished(self, exit_code: int, exit_status: "QProcess.ExitStatus") -> None:
        if exit_code == 0:
            self.runner.append("[UI] ─── Auto-evaluation complete ───")
        else:
            self.runner.append(
                f"[UI] Auto-evaluation exited with code {exit_code}."
            )


    # -----------------------------------------------------------------
    # Dashboard v2: KPI update and ETA display
    # -----------------------------------------------------------------
    def _update_kpi_from_store(self) -> None:
        """Push latest metrics from TrainingMetricsStore into KPI cards."""
        if not _HAS_DASHBOARD_V2 or self._kpi_strip is None:
            return
        store = self._metrics_store
        kpi = self._kpi_strip

        epoch = store.latest_epoch()
        total = int(self.epochs.value())
        kpi.epoch.set_value(f"{epoch} / {total}")

        tl = store.latest_train_loss()
        if tl is not None:
            kpi.train_loss.set_value(f"{tl:.3e}")

        vl = store.latest_val_loss()
        if vl is not None:
            kpi.val_loss.set_value(f"{vl:.3e}")

        lr = store.latest_lr()
        if lr is not None:
            kpi.lr.set_value(f"{lr:.2e}")

        best = store.latest_best_score()
        if best is not None:
            best_ep = store.latest_best_epoch()
            kpi.best_score.set_value(
                f"{best:.3e}",
                subtitle=f"epoch {best_ep}" if best_ep else None,
                state="success",
            )

        # Direction metric
        cos = store.latest("train_cos_sim")
        if cos is not None:
            kpi.direction.set_value(f"cos={cos:.4f}")

    def _update_eta_display(self) -> None:
        """Timer-driven update of time metrics in KPI strip, time strip, header."""
        if not _HAS_DASHBOARD_V2 or self._eta_estimator is None:
            return
        est = self._eta_estimator
        elapsed = est.format_elapsed()
        remaining = est.format_remaining()
        finish = est.format_finish()

        kpi = self._kpi_strip
        if kpi is not None:
            kpi.eta.set_value(remaining)

        # Phase 7: full time-metric cards on the Live Monitor.
        ts = self._time_strip
        if ts is not None:
            ts.elapsed.set_value(elapsed)
            ts.eta.set_value(remaining)
            ts.finish.set_value(finish)
            ts.epoch_duration.set_value(est.format_current_epoch())
            ts.avg_epoch.set_value(est.format_avg_epoch())
            sps = self._metrics_store.latest("samples_per_s") if self._metrics_store else None
            if sps is not None:
                ts.samples_per_s.set_value(f"{sps:,.0f}")

        # Compact time badges on the experiment header.
        main_win = self.window()
        if hasattr(main_win, '_experiment_header'):
            hdr = main_win._experiment_header
            hdr.set_elapsed(elapsed)
            hdr.set_remaining(remaining)
            hdr.set_finish(finish)

    def _update_kpi_phase(self, rec) -> None:
        """Phase 4: reflect the live record's phase in the KPI strip."""
        if not _HAS_DASHBOARD_V2 or self._kpi_strip is None:
            return
        if rec.event == "batch" and rec.phase == "train":
            if rec.total_batches > 0:
                self._kpi_strip.phase.set_value(
                    f"Train {rec.batch}/{rec.total_batches}", state="normal"
                )
            else:
                self._kpi_strip.phase.set_value("Training", state="normal")
        elif rec.event == "val_summary":
            self._kpi_strip.phase.set_value("Validating", state="normal")
        elif rec.event == "best_updated":
            self._kpi_strip.phase.set_value("Best checkpoint", state="success")

    def _update_run_dir_label(self, path: str) -> None:
        """Update the control-bar output/run directory label."""
        if not hasattr(self, "_run_dir_label"):
            return
        path = (path or "").strip()
        self._run_dir_label.setText(
            f"Output: {path}" if path else "Output: (auto-timestamped run folder)"
        )

    def _detect_device_badge(self) -> str:
        """Detect the training device once and cache it (avoid repeated torch imports)."""
        if getattr(self, "_device_badge", None):
            return self._device_badge
        badge = "CPU"
        hint = self.device_hint.currentText() if hasattr(self, "device_hint") else "auto"
        if hint != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    badge = f"CUDA · {mem:.1f} GB"
            except Exception:
                badge = "CPU"
        self._device_badge = badge
        return badge

    def _update_header_lifecycle(self, status: str) -> None:
        """Phase 6/11: reflect real training state in the experiment header."""
        main_win = self.window()
        hdr = getattr(main_win, "_experiment_header", None)
        if hdr is None or not hasattr(hdr, "set_status"):
            return
        hdr.set_status(status)
        if status == "TRAINING":
            hdr.set_elapsed("00:00:00")
            hdr.set_remaining("Estimating…")
            hdr.set_finish("Estimating…")
            hdr.set_device(self._detect_device_badge())
            # Context badges from the current configuration.
            run = (self.runner._output_dir or self.out_dir.text().strip())
            if hasattr(hdr, "set_run"):
                hdr.set_run(Path(run).name if run else "auto")
            ds = self.data.text().strip() or self.train_data.text().strip()
            if hasattr(hdr, "set_dataset"):
                hdr.set_dataset(Path(ds).name if ds else "auto")
            if hasattr(hdr, "set_preset"):
                _p = self.model_preset.currentData() or "custom"
                hdr.set_preset(_PRESET_SHORT.get(_p, _p))

    # -----------------------------------------------------------------
    # Progress parsing (+ live plot feeding)
    # -----------------------------------------------------------------
    def _parse_progress(self, line: str) -> None:
        # Accept both "Epoch [N/M]" banners and the engine's "epoch=N" kv form.
        ep: Optional[int] = None
        total: Optional[int] = None
        m = re.search(r"Epoch\s*(?:\[\s*)?(\d+)\s*/\s*(\d+)(?:\s*\])?", line)
        if m:
            ep = int(m.group(1))
            total = int(m.group(2))
        else:
            m_kv = re.search(r"\bepoch\s*=\s*(\d+)", line, re.IGNORECASE)
            if m_kv:
                ep = int(m_kv.group(1))
        if ep is not None:
            if total is not None:
                self.runner.progress.setRange(0, max(1, total))
                self.runner.progress.setValue(min(ep if ep >= 1 else ep + 1, total))
            else:
                self.runner.progress.setValue(min(ep, self.runner.progress.maximum()))
            # Phase 8: only (re)start the epoch timer the FIRST time a new epoch
            # number appears — many lines repeat the epoch within one epoch.
            if _HAS_DASHBOARD_V2 and self._eta_estimator is not None:
                if total is not None:
                    self._eta_estimator.set_total_epochs(total)
                if self._epoch_guard is not None and self._epoch_guard.should_start(ep):
                    self._eta_estimator.on_epoch_start(ep)

        # Phase 8: feed line to structured log parser + metrics store
        if _HAS_DASHBOARD_V2 and self._log_parser is not None:
            rec = self._log_parser.parse_line(line)
            if rec is not None:
                self._metrics_store.append(rec)
                if self._structured_log is not None:
                    self._structured_log.append_record(rec)
                self._update_kpi_from_store()
                self._update_kpi_phase(rec)
                # ETA: count an epoch's end only once, on its first val summary.
                if (
                    rec.event == "val_summary"
                    and self._eta_estimator is not None
                    and self._epoch_guard is not None
                    and self._epoch_guard.should_end(rec.epoch)
                ):
                    self._eta_estimator.on_epoch_end(rec.epoch)
                # ETA: track batch progress
                if rec.batch > 0 and rec.total_batches > 0 and self._eta_estimator is not None:
                    self._eta_estimator.on_batch_progress(rec.batch, rec.total_batches)

        # Feature #13: feed line to the live loss plot (existing)
        self._live_plot.parse_line(line)

        # Try to capture output dir
        if not self.runner._output_dir:
            m2 = re.search(r"(?:out_dir|Output dir|Saving to)\s*[:=]\s*(.+)", line)
            if m2:
                candidate = m2.group(1).strip().strip("'\"")
                if Path(candidate).is_dir():
                    self.runner.set_output_dir(candidate)

                    self._set_history_poll_dir(candidate)


# =============================================================================
# 11. CLOUD GENERATION TAB
# =============================================================================


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
        mode_lbl = QLabel("Mod:")
        mode_lbl.setStyleSheet("font-weight: 600; color: #c4ccff;")
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Tek Bulut (Single Cloud)", self._MODE_SINGLE)
        self._mode_combo.addItem("Dataset Suite", self._MODE_SUITE)
        self._mode_combo.setToolTip(
            "Single Cloud: tek bir .h5/.pt dosyasi uretir.\n"
            "Dataset Suite: train/val/test/ood + manifest.json seti uretir."
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
        self.command_preview.setReadOnly(True)
        self.command_preview.setFont(_mono_font())
        self.command_preview.setMaximumHeight(82)
        self.command_preview.setPlaceholderText(
            "Onizleme icin 'Preview' butonuna tiklayin."
        )
        btn_preview = QPushButton("Preview Command")
        btn_preview.clicked.connect(self._refresh_preview)
        btn_copy_cmd = QPushButton("Copy")
        btn_copy_cmd.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.command_preview.toPlainText())
        )
        preview_btns = QHBoxLayout()
        preview_btns.setContentsMargins(0, 0, 0, 0)
        preview_btns.setSpacing(8)
        preview_btns.addWidget(btn_preview)
        preview_btns.addWidget(btn_copy_cmd)
        preview_btns.addStretch(1)

        preview_w = QWidget()
        preview_vbox = QVBoxLayout()
        preview_vbox.setContentsMargins(0, 0, 0, 0)
        preview_vbox.setSpacing(6)
        preview_vbox.addLayout(preview_btns)
        preview_vbox.addWidget(self.command_preview)
        preview_w.setLayout(preview_vbox)
        analysis_panel = self._build_analysis_panel()

        # ── ProcessPane ──────────────────────────────────────────────────────
        self.runner = ProcessPane()
        self.runner.btn_start.setText("Uretimi Baslatı")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_finished_hook(self._on_finished)
        self.runner.set_progress_parser(self._parse_progress)

        top = QWidget()
        top_l = QVBoxLayout()
        top_l.setContentsMargins(8, 8, 8, 4)
        top_l.setSpacing(6)
        top_l.addLayout(mode_bar)
        top_l.addWidget(self._sync_banner)
        top_l.addWidget(self._stack, 1)
        top_l.addWidget(preview_w)
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
        grp_grav = QGroupBox("Yercekimi Alani")
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
            placeholder="Bos -> varsayilan Ay yercekimi modeli (jggrx_1800f)", check_file=True
        )
        btn_gfc = QPushButton("Sec...")
        btn_gfc.clicked.connect(self._pick_gfc_path)
        gfc_row = _row_lineedit_with_button(self.gfc_path, btn_gfc)

        form_grav.addRow("Maks. SH Derecesi", self.degree_max)
        form_grav.addRow("Min. SH Derecesi (taban)", self.degree_min)
        form_grav.addRow("GFC Dosyasi", gfc_row)
        grp_grav.setLayout(form_grav)

        # ── Group 2: Spatial Sampling ──────────────────────────────────────
        grp_spatial = QGroupBox("Uzamsal Ornekleme")
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
        self.sampling_strategy.addItem("mixed - karma (onerilen)", "mixed")
        self.sampling_strategy.addItem("uniform - hacimce homojen", "uniform")
        self.sampling_strategy.addItem("inverse_r2 - yuzey odakli", "inverse_r2")
        _strategy_idx = self.sampling_strategy.findData(str(_cfg_value(cloud_cfg, "sampling_strategy", "mixed")))
        if _strategy_idx >= 0:
            self.sampling_strategy.setCurrentIndex(_strategy_idx)

        self.surface_bias_ratio = QDoubleSpinBox()
        self.surface_bias_ratio.setDecimals(2)
        self.surface_bias_ratio.setRange(0.0, 1.0)
        self.surface_bias_ratio.setValue(float(_cfg_value(cloud_cfg, "surface_bias_ratio", 0.70)))
        self.surface_bias_ratio.setSingleStep(0.05)

        form_spatial.addRow("Ornek Sayisi", self.n_samples)
        form_spatial.addRow("Min. Irtifa", self.alt_min_km)
        form_spatial.addRow("Maks. Irtifa", self.alt_max_km)
        form_spatial.addRow("Ornekleme Stratejisi", self.sampling_strategy)
        form_spatial.addRow("Yuzey Agirlik Orani", self.surface_bias_ratio)
        grp_spatial.setLayout(form_spatial)

        # ── Group 3: Output ────────────────────────────────────────────────
        grp_out = QGroupBox("Cikti Ayarlari")
        form_out = QFormLayout()
        _tune_form(form_out)

        self.out_format = QComboBox()
        self.out_format.addItem("HDF5 (.h5) - onerilen", "h5")
        self.out_format.addItem("PyTorch (.pt)", "pt")

        self.out_path = QLineEdit("")
        self.out_path.setPlaceholderText("Bos -> otomatik (data/ klasorunde)")
        btn_out_save = QPushButton("Sec...")
        btn_out_save.clicked.connect(self._pick_out_path)
        out_row = _row_lineedit_with_button(self.out_path, btn_out_save)

        self.dtype = QComboBox()
        self.dtype.addItem("float32 - onerilen", "float32")
        self.dtype.addItem("float64 - yuksek hassasiyet", "float64")

        self.canonical = QCheckBox("Kanonik birimler (boyutsuz)")
        self.canonical.setChecked(bool(_cfg_value(cloud_cfg, "canonical", False)))

        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setValue(int(_cfg_value(cloud_cfg, "seed", 12345)))

        form_out.addRow("Cikti Formati", self.out_format)
        form_out.addRow("Cikti Dosyasi", out_row)
        form_out.addRow("Veri Tipi", self.dtype)
        form_out.addRow(self.canonical)
        form_out.addRow("Rastgele Tohum", self.seed)
        grp_out.setLayout(form_out)

        # ── Group 4: Performance ───────────────────────────────────────────
        grp_perf = QGroupBox("Performans")
        form_perf = QFormLayout()
        _tune_form(form_perf)

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1_000, 10_000_000)
        self.chunk_size.setValue(int(_cfg_value(cloud_cfg, "chunk_size", 50_000)))
        self.chunk_size.setSingleStep(10_000)

        self.workers = QSpinBox()
        self.workers.setRange(1, 256)
        self.workers.setValue(min(cpu_count, int(_cfg_value(cloud_cfg, "workers", 8))))

        self.no_multiprocessing = QCheckBox("Coklu islem devre disi (tek is parcacigi)")
        self.no_multiprocessing.setChecked(bool(_cfg_value(cloud_cfg, "no_multiprocessing", False)))

        form_perf.addRow("Yigin Boyutu (Chunk)", self.chunk_size)
        form_perf.addRow(f"Islemci Sayisi (sistem: {cpu_count})", self.workers)
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
        grp_phys = QGroupBox("Fizik / Derece / Irtifa")
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
        self.s_train_alt_min_km.setToolTip("Egitim irtifa araliginin alt siniri.")

        self.s_train_alt_max_km = QDoubleSpinBox()
        self.s_train_alt_max_km.setDecimals(1)
        self.s_train_alt_max_km.setRange(1.0, 50_000.0)
        self.s_train_alt_max_km.setValue(float(_cfg_value(suite_cfg, "train_alt_max_km", 600.0)))
        self.s_train_alt_max_km.setSuffix(" km")
        self.s_train_alt_max_km.setToolTip("Egitim irtifa araliginin ust siniri.")

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
            placeholder="Bos -> varsayilan Ay yercekimi modeli (jggrx_1800f)", check_file=True
        )
        btn_s_gfc = QPushButton("Sec...")
        btn_s_gfc.clicked.connect(self._pick_suite_gfc_path)
        s_gfc_row = _row_lineedit_with_button(self.s_gfc_path, btn_s_gfc)

        form_phys.addRow("Derece Min (taban)", self.s_degree_min)
        form_phys.addRow("Derece Max (hedef)", self.s_degree_max)
        form_phys.addRow("Egitim Alt Min", self.s_train_alt_min_km)
        form_phys.addRow("Egitim Alt Max", self.s_train_alt_max_km)
        form_phys.addRow("OOD Marjin", self.s_ood_margin_km)
        form_phys.addRow("GFC Dosyasi", s_gfc_row)
        grp_phys.setLayout(form_phys)

        # B) Train hybrid allocation
        grp_train = QGroupBox("Train Hybrid Dagilimi")
        form_train = QFormLayout()
        _tune_form(form_train)

        self.s_train_su_n = QSpinBox()
        self.s_train_su_n.setRange(0, 100_000_000)
        self.s_train_su_n.setValue(int(_cfg_value(suite_cfg, "train_stratified_uniform_n", 2_000_000)))
        self.s_train_su_n.setSingleStep(100_000)
        self.s_train_su_n.setToolTip("Katmanlı düzgün dağılım (stratified uniform) nokta sayısı.")

        self.s_train_ir2_n = QSpinBox()
        self.s_train_ir2_n.setRange(0, 100_000_000)
        self.s_train_ir2_n.setValue(int(_cfg_value(suite_cfg, "train_inverse_r2_n", 1_000_000)))
        self.s_train_ir2_n.setSingleStep(100_000)
        self.s_train_ir2_n.setToolTip("Ters-r2 (yüzeye yakın odaklı) nokta sayısı.")

        self.s_train_rm_n = QSpinBox()
        self.s_train_rm_n.setRange(0, 100_000_000)
        self.s_train_rm_n.setValue(int(_cfg_value(suite_cfg, "train_residual_mag_n", 1_000_000)))
        self.s_train_rm_n.setSingleStep(100_000)
        self.s_train_rm_n.setToolTip("Artık ivme büyüklüğüne göre ağırlıklı örnekleme nokta sayısı.")

        self.s_train_bb_n = QSpinBox()
        self.s_train_bb_n.setRange(0, 100_000_000)
        self.s_train_bb_n.setValue(int(_cfg_value(suite_cfg, "train_boundary_n", 1_000_000)))
        self.s_train_bb_n.setSingleStep(100_000)
        self.s_train_bb_n.setToolTip("Sınır tamponu (alt/üst irtifa kenarları) nokta sayısı.")

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
        self.s_boundary_mode.addItem("strict (egitim araliginda)", "strict")
        self.s_boundary_mode.addItem("soft (sinir uzerinde)", "soft")
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

        self.s_combine_ood = QCheckBox("OOD low + high birlestirilsin (ood_combined.h5)")
        self.s_combine_ood.setChecked(bool(_cfg_value(suite_cfg, "combine_ood", True)))

        form_vto.addRow("Validation Noktalari", self.s_val_n)
        form_vto.addRow("Test Noktalari", self.s_test_n)
        form_vto.addRow("OOD Low Noktalari", self.s_ood_low_n)
        form_vto.addRow("OOD High Noktalari", self.s_ood_high_n)
        form_vto.addRow(self.s_combine_ood)
        grp_vto.setLayout(form_vto)

        # D) Seeds
        grp_seeds = QGroupBox("Tohumlar (Seeds)")
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

        btn_auto_seeds = QPushButton("Bagimsiz Tohumlar Ata")
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
        grp_presets = QGroupBox("Suite Onayarlari (Presets)")
        form_presets = QFormLayout()
        _tune_form(form_presets)

        self.s_preset_combo = QComboBox()
        self.s_preset_combo.addItem("-- Seciniz --", "")
        self.s_preset_combo.addItem("Debug Suite (100k)", "debug_suite")
        self.s_preset_combo.addItem("Baseline Uniform (2M train)", "baseline_uniform_suite")
        self.s_preset_combo.addItem("Recommended Hybrid 5M (onerilen)", "recommended_hybrid_5M")
        self.s_preset_combo.addItem("High Accuracy 10M", "high_accuracy_10M")

        btn_apply_preset = QPushButton("Onayari Uygula")
        btn_apply_preset.clicked.connect(self._apply_suite_preset)

        form_presets.addRow("Preset", self.s_preset_combo)
        form_presets.addRow("", btn_apply_preset)
        grp_presets.setLayout(form_presets)

        # F) Suite output
        grp_suite_out = QGroupBox("Suite Ciktisi")
        form_suite_out = QFormLayout()
        _tune_form(form_suite_out)

        self.s_suite_name = QLineEdit("")
        self.s_suite_name.setPlaceholderText("Bos -> otomatik adlandirilir")

        self.s_suite_out_dir = QLineEdit("")
        self.s_suite_out_dir.setPlaceholderText(
            "Bos -> <script_dir>/data/cloud_suites/"
        )
        btn_suite_out = QPushButton("Sec...")
        btn_suite_out.clicked.connect(self._pick_suite_out_dir)
        suite_out_row = _row_lineedit_with_button(self.s_suite_out_dir, btn_suite_out)

        self.s_auto_apply = QCheckBox("Suite tamamlaninca egitim sekmesini otomatik doldur")
        self.s_auto_apply.setChecked(True)
        self.s_auto_apply.setToolTip(
            "Isaretlenirse: manifest.json'dan train/val/test/ood yollarini\n"
            "otomatik olarak egitim sekmesine yazar."
        )

        self.s_chunk_size = QSpinBox()
        self.s_chunk_size.setRange(1_000, 10_000_000)
        self.s_chunk_size.setValue(int(_cfg_value(suite_cfg, "chunk_size", 50_000)))
        self.s_chunk_size.setSingleStep(10_000)

        self.s_dtype = QComboBox()
        self.s_dtype.addItem("float32 - onerilen", "float32")
        self.s_dtype.addItem("float64 - yuksek hassasiyet", "float64")

        form_suite_out.addRow("Suite Adi", self.s_suite_name)
        form_suite_out.addRow("Cikti Klasoru", suite_out_row)
        form_suite_out.addRow(self.s_auto_apply)
        form_suite_out.addRow("Chunk Boyutu", self.s_chunk_size)
        form_suite_out.addRow("Veri Tipi", self.s_dtype)
        grp_suite_out.setLayout(form_suite_out)

        # G) Suite actions
        btn_open_suite_folder = QPushButton("Suite Klasorunu Ac")
        btn_open_suite_folder.clicked.connect(self._open_suite_folder)
        btn_apply_to_train = QPushButton("Egitim Sekmesine Uygula")
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
        grp = QGroupBox("Cloud Analysis")
        grp.setToolTip(
            "Generated HDF5/PT cloud datasets can be analyzed without leaving the UI. "
            "The analysis computes altitude balance, directional coverage, field dynamic range, "
            "and acceleration radial/cross-radial diagnostics."
        )

        self.analysis_input = ValidatedPathEdit(
            placeholder="Analyze a generated .h5/.hdf5/.pt dataset", check_file=True
        )
        btn_pick = QPushButton("Select...")
        btn_pick.clicked.connect(self._pick_analysis_input)
        input_row = _row_lineedit_with_button(self.analysis_input, btn_pick)

        self.analysis_outdir = QLineEdit("")
        self.analysis_outdir.setPlaceholderText("Empty -> <dataset_folder>/analysis")
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

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addWidget(btn_use_latest)
        action_row.addWidget(btn_run)
        action_row.addWidget(btn_suite)
        action_row.addStretch(1)
        action_row.addWidget(self.analysis_make_plots)
        action_row.addWidget(self.analysis_auto_after_suite)

        self.analysis_summary = QPlainTextEdit()
        self.analysis_summary.setReadOnly(True)
        self.analysis_summary.setFont(_mono_font())
        self.analysis_summary.setMaximumHeight(150)
        self.analysis_summary.setPlaceholderText(
            "Analysis summary will appear here: altitude balance, octant coverage, "
            "finite checks, field dynamic range, and acceleration geometry."
        )

        form = QFormLayout()
        _tune_form(form)
        form.addRow("Dataset", input_row)
        form.addRow("Analysis Output", out_row)
        form.addRow("Sample Rows", self.analysis_sample)
        form.addRow("Scatter Points", self.analysis_scatter_n)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(form)
        layout.addLayout(action_row)
        layout.addWidget(self.analysis_summary)
        grp.setLayout(layout)
        _tune_inputs(grp)

        self._analysis_proc = QProcess(self)
        self._analysis_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._analysis_proc.readyReadStandardOutput.connect(self._on_analysis_output)
        self._analysis_proc.finished.connect(self._on_analysis_finished)
        return grp

    def _pick_analysis_input(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Cloud Dataset Sec",
            self.analysis_input.text() or str(SCRIPT_DIR),
            "Cloud datasets (*.h5 *.hdf5 *.pt);;All (*.*)",
        )
        if fn:
            self.analysis_input.setText(_norm_path(fn))

    def _pick_analysis_outdir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self,
            "Analysis Output Klasoru",
            self.analysis_outdir.text() or str(SCRIPT_DIR),
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
            out = dataset_path.parent / "analysis"
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
            QMessageBox.information(self, "Cloud Analysis", "Analiz edilecek mevcut dataset bulunamadi.")
            return
        self.analysis_input.setText(candidates[0])

    def _build_analysis_args(self, dataset_path: Path, outdir: Path) -> List[str]:
        script = SCRIPT_DIR / "spatial_cloud_analysis.py"
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
            QMessageBox.information(self, "Cloud Analysis", "Analiz zaten calisiyor.")
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
            QMessageBox.warning(self, "Cloud Analysis", f"Dataset bulunamadi:\n{p}")
            return
        outdir = self._analysis_default_outdir(p, label if label != "dataset" else "")
        self._active_analysis = (str(p), str(outdir), label)
        self.analysis_summary.appendPlainText(f"\n[analysis] starting {label}: {p.name}")
        self._analysis_proc.setWorkingDirectory(str(SCRIPT_DIR))
        self._analysis_proc.start(sys.executable, self._build_analysis_args(p, outdir))

    def _run_suite_analysis_from_last_dir(self) -> None:
        if not self._last_suite_dir:
            QMessageBox.information(self, "Cloud Analysis", "Once bir dataset suite uretin veya secin.")
            return
        self._run_suite_analysis(Path(self._last_suite_dir))

    def _run_suite_analysis(self, suite_dir: Path) -> None:
        if self._analysis_proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Cloud Analysis", "Analiz zaten calisiyor.")
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
            QMessageBox.warning(self, "Cloud Analysis", "Suite icinde analiz edilecek HDF5 dosyasi bulunamadi.")
            return
        self.analysis_summary.setPlainText("[analysis] suite analysis queued...")
        self._start_next_analysis_job()

    def _start_next_analysis_job(self) -> None:
        if not self._analysis_queue:
            return
        dataset_path, outdir, label = self._analysis_queue.popleft()
        self._active_analysis = (dataset_path, outdir, label)
        self.analysis_summary.appendPlainText(f"\n[analysis] starting {label}: {Path(dataset_path).name}")
        self._analysis_proc.setWorkingDirectory(str(SCRIPT_DIR))
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
        btn_label = "Suite Uretimini Baslatı" if mode == self._MODE_SUITE else "Bulut Uretimini Baslatı"
        self.runner.btn_start.setText(btn_label)

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
            QMessageBox.information(self, "Suite Klasoru", "Suite henuz uretilmedi veya klasor bulunamadi.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _apply_suite_to_train(self) -> None:
        d = self._last_suite_dir
        if not d or not Path(d).is_dir():
            QMessageBox.information(self, "Suite Klasoru", "Once suite uretin.")
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
            self, "GFC Dosyasi Sec",
            self.gfc_path.text() or str(SCRIPT_DIR),
            "GFC/TAB (*.gfc *.tab *.txt);;All (*.*)",
        )
        if fn:
            self.gfc_path.setText(_norm_path(fn))

    def _pick_out_path(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "Cikti Dosyasi",
            self.out_path.text() or str(SCRIPT_DIR / "data"),
            "HDF5 (*.h5);;PyTorch (*.pt);;All (*.*)",
        )
        if fn:
            self.out_path.setText(_norm_path(fn))

    def _pick_suite_gfc_path(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self, "GFC Dosyasi Sec",
            self.s_gfc_path.text() or str(SCRIPT_DIR),
            "GFC/TAB (*.gfc *.tab *.txt);;All (*.*)",
        )
        if fn:
            self.s_gfc_path.setText(_norm_path(fn))

    def _pick_suite_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Suite Cikti Klasoru",
            self.s_suite_out_dir.text() or str(SCRIPT_DIR),
        )
        if d:
            self.s_suite_out_dir.setText(_norm_path(d))

    # ------------------------------------------------------------------
    # CLI arg builders
    # ------------------------------------------------------------------
    def _build_single_args(self, show_errors: bool = True) -> Optional[List[str]]:
        script = SCRIPT_DIR / "spatial_cloud_generator.py"
        if not script.exists():
            if show_errors:
                QMessageBox.critical(self, "Eksik script", "spatial_cloud_generator.py bulunamadi.")
            return None
        deg_min = self.degree_min.value()
        deg_max = self.degree_max.value()
        if deg_max <= deg_min and deg_min != -1:
            if show_errors:
                QMessageBox.critical(self, "Gecersiz derece", f"degree_max ({deg_max}) > degree_min ({deg_min}) olmali.")
            return None
        alt_min = self.alt_min_km.value()
        alt_max = self.alt_max_km.value()
        if alt_max <= alt_min:
            if show_errors:
                QMessageBox.critical(self, "Gecersiz irtifa", f"alt_max ({alt_max}) > alt_min ({alt_min}) olmali.")
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
        script = SCRIPT_DIR / "spatial_cloud_generator.py"
        if not script.exists():
            if show_errors:
                QMessageBox.critical(self, "Eksik script", "spatial_cloud_generator.py bulunamadi.")
            return None
        deg_min = self.s_degree_min.value()
        deg_max = self.s_degree_max.value()
        if deg_max <= deg_min:
            if show_errors:
                QMessageBox.critical(self, "Gecersiz derece", f"degree_max ({deg_max}) > degree_min ({deg_min}) olmali.")
            return None
        alt_min = self.s_train_alt_min_km.value()
        alt_max = self.s_train_alt_max_km.value()
        if alt_max <= alt_min:
            if show_errors:
                QMessageBox.critical(self, "Gecersiz irtifa", f"alt_max ({alt_max}) > alt_min ({alt_min}) olmali.")
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
                "Bulut uretimi tamamlandi! "
                f"Irtifa: {self.alt_min_km.value():.0f}-{self.alt_max_km.value():.0f} km  |  "
                f"Derece: {self.degree_min.value()}->{self.degree_max.value()}  |  "
                "Egitim sekmesi senkronize edildi."
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
            base = Path(out_dir_text) if out_dir_text else SCRIPT_DIR / "data" / "cloud_suites"
            if base.is_dir():
                dirs = sorted(base.iterdir(), key=lambda d: d.stat().st_mtime if d.is_dir() else 0, reverse=True)
                for d in dirs[:5]:
                    if d.is_dir() and (d / "manifest.json").exists():
                        suite_dir = str(d)
                        break

        self._last_suite_dir = suite_dir
        msg = "Suite uretimi tamamlandi!"
        if suite_dir:
            msg += f"  Klasor: {suite_dir}"
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


# =============================================================================
# 12. ST-LRPS RUNTIME PROFILING TAB
# =============================================================================


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
            placeholder="results/profiling/st_lrps_runtime", check_file=False
        )
        self.profile_out_dir.setText("results/profiling/st_lrps_runtime")
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
        grid.addWidget(grp_model, 0, 0, 1, 2)
        grid.addWidget(grp_runtime, 1, 0)
        grid.addWidget(grp_input, 1, 1)
        grid.addWidget(grp_output, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for group in (grp_model, grp_runtime, grp_input, grp_output):
            _tune_inputs(group)

        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setFont(_mono_font())
        self.command_preview.setMinimumHeight(76)
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
        preview_form.addRow("Generated Command", self.command_preview)
        preview_form.addRow("", self.command_warning)

        top = QWidget()
        top_l = QVBoxLayout()
        top_l.setContentsMargins(8, 8, 8, 8)
        top_l.setSpacing(8)
        top_l.addLayout(grid)
        top_l.addLayout(preview_form)
        top.setLayout(top_l)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Run Profiling")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_finished_hook(self._on_profile_finished)
        self.runner.set_stop_hint("")

        self.profile_summary = QPlainTextEdit()
        self.profile_summary.setReadOnly(True)
        self.profile_summary.setFont(_mono_font())
        self.profile_summary.setPlaceholderText("runtime_profile_summary.md will appear here after profiling.")
        self._gallery = ImageGallery()
        self._gallery._placeholder.setText("Runtime profile plots will appear here.")

        output_splitter = QSplitter(Qt.Orientation.Horizontal)
        output_splitter.addWidget(self.profile_summary)
        output_splitter.addWidget(self._gallery)
        output_splitter.setStretchFactor(0, 1)
        output_splitter.setStretchFactor(1, 1)

        bottom = QWidget()
        bottom_l = QVBoxLayout()
        bottom_l.setContentsMargins(0, 0, 0, 0)
        bottom_l.setSpacing(8)
        bottom_l.addWidget(self.runner, 2)
        bottom_l.addWidget(output_splitter, 1)
        bottom.setLayout(bottom_l)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(_scroll_wrap(top))
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 560])

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter, 1)
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
            self.profile_out_dir.text() or str(_REPO_ROOT / "results" / "profiling"),
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
        self.profile_out_dir.setText(_st("profile_out_dir", "results/profiling/st_lrps_runtime"))
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
        out_dir = self.profile_out_dir.text().strip() or "results/profiling/st_lrps_runtime"

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
        out_text = self._effective_out_dir or self.profile_out_dir.text().strip() or "results/profiling/st_lrps_runtime"
        p = Path(out_text)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _start(self) -> None:
        args = self._build_profile_args()
        if args is None:
            return
        self._save_settings()
        self._effective_out_dir = self.profile_out_dir.text().strip() or "results/profiling/st_lrps_runtime"
        out_path = self._resolved_out_dir()
        self.profile_summary.clear()
        self._gallery.clear_gallery()
        self._gallery._placeholder.setText("Runtime profile plots will appear here.")
        self.runner.progress.setRange(0, 0)
        self.runner.set_output_dir(str(out_path))
        self.runner.set_stop_hint("")
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _on_profile_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
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


# =============================================================================
# 13. ST-LRPS EVALUATE TAB (retained from v2 with minor polish)
# =============================================================================


class STLRPSEvalTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        grp_input = QGroupBox("Girdi Dosyaları")
        form_input = QFormLayout()
        _tune_form(form_input)

        self.model_dir = ValidatedPathEdit(
            placeholder="Boş → en yeni run klasörü", check_file=False
        )
        btn_model = QPushButton("Seç…")
        btn_model.clicked.connect(self._pick_model_dir)
        model_row = _row_lineedit_with_button(self.model_dir, btn_model)

        self.data = ValidatedPathEdit(
            placeholder="Boş → otomatik aranır", check_file=True
        )
        btn_data = QPushButton("Seç…")
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
            placeholder="Boş → <run-dir>/evals/eval_<dataset>_<timestamp>", check_file=False
        )
        btn_out = QPushButton("Seç…")
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

        form_input.addRow("Model Klasörü", model_row)
        form_input.addRow("Test Dataseti", data_row)
        form_input.addRow("Independent Test Dataset", test_data_row)
        form_input.addRow("OOD Dataset", ood_data_row)
        form_input.addRow(self.use_config_datasets)
        form_input.addRow(self.export_hard_samples)
        form_input.addRow("Hard sample count", self.hard_sample_count)
        form_input.addRow("Hard sample metric", self.hard_sample_metric)
        form_input.addRow("HDF5 Dataset Adı", self.dataset_name)
        form_input.addRow("Çıktı Klasörü", out_row)
        form_input.addRow("Artifact Status", self.run_artifact_badge)
        form_input.addRow("Artifact Summary", self.run_artifact_summary)
        grp_input.setLayout(form_input)

        grp_hw = QGroupBox("Donanım ve İşlem")
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
        form_hw.addRow("Cihaz", self.device)
        form_hw.addRow("Batch Boyutu", self.batch_size)
        form_hw.addRow("İvme İşareti", self.a_sign)
        grp_hw.setLayout(form_hw)

        grp_spatial = QGroupBox("Mekansal Analiz")
        form_spatial = QFormLayout()
        _tune_form(form_spatial)
        self.r_ref_m = QLineEdit("")
        self.r_ref_m.setPlaceholderText("Boş → Ay yarıçapı")
        self.alt_bin_km = QDoubleSpinBox()
        self.alt_bin_km.setDecimals(2)
        self.alt_bin_km.setRange(1.0, 10_000.0)
        self.alt_bin_km.setValue(50.0)
        self.start = QSpinBox()
        self.start.setRange(0, 2_147_483_647)
        self.start.setValue(0)
        self.end = QLineEdit("")
        self.end.setPlaceholderText("Boş → EOF")
        self.max_points = QSpinBox()
        self.max_points.setRange(10_000, 50_000_000)
        self.max_points.setValue(500_000)
        form_spatial.addRow("Referans Yarıçap (m)", self.r_ref_m)
        form_spatial.addRow("İrtifa Bin (km)", self.alt_bin_km)
        form_spatial.addRow("Başlangıç", self.start)
        form_spatial.addRow("Bitiş", self.end)
        form_spatial.addRow("Nokta Limiti", self.max_points)
        grp_spatial.setLayout(form_spatial)

        self.extra_args = QLineEdit("")
        self.extra_args.setPlaceholderText("Ek CLI argümanları")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(grp_input, 0, 0, 1, 2)
        grid.addWidget(grp_hw, 1, 0)
        grid.addWidget(grp_spatial, 1, 1)
        extra_f = QFormLayout()
        _tune_form(extra_f)
        extra_f.addRow("Ek CLI", self.extra_args)
        extra_w = QWidget()
        extra_w.setLayout(extra_f)
        grid.addWidget(extra_w, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for g in (grp_input, grp_hw, grp_spatial):
            _tune_inputs(g)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Değerlendirmeyi Başlat")
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
            self, "Model Klasörü", self.model_dir.text() or str(SCRIPT_DIR)
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
            self, "Çıktı", self.out_dir.text() or str(SCRIPT_DIR)
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
            QMessageBox.critical(self, "Bulunamadı", "st_lrps/evaluation/cli.py gerekli.")
            return
        args = ["-u", "-m", EVAL_CLI_MODULE]
        md = self.model_dir.text().strip()
        if md:
            if not Path(md).exists():
                QMessageBox.critical(self, "Bulunamadı", f"Model:\n{md}")
                return
            args += ["--model-dir", md]
        dp = self.data.text().strip()
        if dp:
            if not Path(dp).exists():
                QMessageBox.critical(self, "Bulunamadı", f"Dataset:\n{dp}")
                return
            args += ["--data", dp]
        for flag, path in (
            ("--test-data", self.test_data.text().strip()),
            ("--ood-data", self.ood_data.text().strip()),
        ):
            if path:
                if not Path(path).exists():
                    QMessageBox.critical(self, "Bulunamadı", f"{flag}:\n{path}")
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
                self.runner.append(f"\n[UI] {cnt} grafik yüklendi: {out_dir}")
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


# =============================================================================
# 13. CLOUD ANALYSIS TAB
# =============================================================================


class CloudAnalysisTab(QWidget):
    """Run spatial_cloud_analysis.py on a dataset and display the resulting plots."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # --- Input ---
        grp_input = QGroupBox("Veri Dosyasi")
        form_input = QFormLayout()
        _tune_form(form_input)

        self.input_file = ValidatedPathEdit(
            placeholder="Dataset (.h5 or .pt)", check_file=True
        )
        btn_input = QPushButton("Sec...")
        btn_input.clicked.connect(self._pick_input)
        input_row = _row_lineedit_with_button(self.input_file, btn_input)
        form_input.addRow("Dataset Dosyasi", input_row)
        grp_input.setLayout(form_input)

        # --- Analysis parameters ---
        grp_params = QGroupBox("Analiz Parametreleri")
        form_params = QFormLayout()
        _tune_form(form_params)

        self.sample_n = QSpinBox()
        self.sample_n.setRange(1_000, 10_000_000)
        self.sample_n.setValue(200_000)
        self.sample_n.setSingleStep(10_000)
        self.sample_n.setToolTip("Analiz edilecek satir sayisi (ornekleme ile)")

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed_val = 123
        self.seed.setValue(123)
        self.seed.setToolTip("Ornekleme tohumu (reproducible)")

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

        self.no_plots = QCheckBox("Grafik olusturma")
        self.no_plots.setChecked(False)
        self.dump_json = QCheckBox("summary.json kaydet")
        self.dump_json.setChecked(True)

        form_params.addRow("Ornek Sayisi", self.sample_n)
        form_params.addRow("Tohum", self.seed)
        form_params.addRow("Min Irtifa (km)", self.alt_min_km)
        form_params.addRow("Max Irtifa (km)", self.alt_max_km)
        form_params.addRow("Scatter Nokta Sayisi", self.scatter_n)
        form_params.addRow(self.no_plots)
        form_params.addRow(self.dump_json)
        grp_params.setLayout(form_params)

        # --- Output ---
        grp_out = QGroupBox("Cikti")
        form_out = QFormLayout()
        _tune_form(form_out)

        self.out_dir = ValidatedPathEdit(
            placeholder="Bos -> <dataset_dir>/analysis_out", check_file=False
        )
        btn_out = QPushButton("Sec...")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)
        form_out.addRow("Cikti Klasoru", out_row)
        grp_out.setLayout(form_out)

        self.extra_args = QLineEdit("")
        self.extra_args.setPlaceholderText("Ek CLI argumanlari")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(grp_input, 0, 0, 1, 2)
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
        for g in (grp_input, grp_params, grp_out):
            _tune_inputs(g)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Analizi Baslat")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_progress_parser(self._parse_progress)
        self.runner.set_finished_hook(self._on_finished)
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
            "Dataset Sec",
            self.input_file.text() or str(SCRIPT_DIR),
            "HDF5 (*.h5 *.hdf5);;PT (*.pt);;All (*.*)",
        )
        if fn:
            self.input_file.setText(_norm_path(fn))

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Cikti Klasoru", self.out_dir.text() or str(SCRIPT_DIR)
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
        script = SCRIPT_DIR / "spatial_cloud_analysis.py"
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
            QMessageBox.warning(self, "Eksik Girdi", "Lutfen bir dataset dosyasi secin.")
            return
        if not Path(inp).is_file():
            QMessageBox.warning(self, "Dosya Bulunamadi", f"Dosya mevcut degil:\n{inp}")
            return
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
                candidate = Path(inp).parent / "analysis_out"
                if candidate.is_dir():
                    self._gallery.load_from_directory(str(candidate))

    def _parse_progress(self, line: str):
        if not self._effective_out_dir:
            m = re.search(r"(?:outdir|Output dir|Saving to|Writing to)\s*[:=]\s*(.+)", line, re.IGNORECASE)
            if m:
                c = m.group(1).strip().strip("'\"")
                if Path(c).is_dir():
                    self._effective_out_dir = c
                    self.runner.set_output_dir(c)


# =============================================================================
# 13b. DATA PAGE  (dataset readiness)
# =============================================================================


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


class DatasetInspectionPanel(QWidget):
    """Dataset readiness panel: pick an HDF5 dataset, inspect metadata, validate."""

    send_to_training = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        title = QLabel("Dataset Readiness")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #e6edf7;")
        subtitle = QLabel(
            "Inspect an HDF5 cloud and confirm it is suitable for ST-LRPS training."
        )
        subtitle.setStyleSheet("color: #7f91ac; font-size: 12px;")

        self.path_edit = ValidatedPathEdit(
            placeholder="Select an HDF5 dataset (.h5) to inspect", check_file=True
        )
        btn_browse = QPushButton("Select...")
        btn_browse.clicked.connect(self._pick)
        self.btn_validate = QPushButton("Validate Dataset")
        self.btn_validate.setProperty("kind", "primary")
        self.btn_validate.clicked.connect(self._validate)
        self.btn_send = QPushButton("Send to Training")
        self.btn_send.setProperty("kind", "ghost")
        self.btn_send.clicked.connect(self._send)
        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(8)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(btn_browse)
        path_row.addWidget(self.btn_validate)
        path_row.addWidget(self.btn_send)

        self.status_label = QLabel("UNKNOWN")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_status("unknown", "Select a dataset to inspect metadata.")

        # Metadata summary card
        self._summary = QLabel("Select a dataset to inspect metadata.")
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._summary.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._summary.setStyleSheet(
            "background: rgba(13, 22, 38, 0.85); border: 1px solid #26364f;"
            " border-radius: 10px; padding: 12px; color: #cdd9ee; font-size: 12px;"
        )
        self._summary.setMinimumHeight(220)

        # Raw metadata panel
        self._raw = QPlainTextEdit()
        self._raw.setReadOnly(True)
        self._raw.setFont(_mono_font())
        self._raw.setPlaceholderText("Raw metadata/attributes will appear here.")
        self._raw.setMinimumHeight(180)

        meta_split = QSplitter(Qt.Orientation.Horizontal)
        sl = QWidget(); slo = QVBoxLayout(); slo.setContentsMargins(0, 0, 0, 0)
        slo.addWidget(QLabel("Metadata summary")); slo.addWidget(self._summary, 1); sl.setLayout(slo)
        sr = QWidget(); sro = QVBoxLayout(); sro.setContentsMargins(0, 0, 0, 0)
        sro.addWidget(QLabel("Raw attributes")); sro.addWidget(self._raw, 1); sr.setLayout(sro)
        meta_split.addWidget(sl)
        meta_split.addWidget(sr)
        meta_split.setSizes([520, 420])

        backend_note = QLabel(
            "Full dataset convention validation is performed by the training backend "
            "before training starts."
        )
        backend_note.setWordWrap(True)
        backend_note.setStyleSheet("color: #7f91ac; font-size: 11px;")

        lo = QVBoxLayout()
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setSpacing(10)
        lo.addWidget(title)
        lo.addWidget(subtitle)
        lo.addLayout(path_row)
        lo.addWidget(self.status_label)
        lo.addWidget(meta_split, 1)
        lo.addWidget(backend_note)
        self.setLayout(lo)

    # -- helpers --
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
        self.status_label.setText(f"{label} — {text}")
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

        fields = [
            ("File", Path(path).name),
            ("Path", str(path)),
            ("Rows", f"{rows:,}" if isinstance(rows, int) else rows),
            ("Columns", info.get("cols")),
            ("Dataset name", info.get("dataset_name")),
            ("Unit system", unit_system),
            ("Central body", _attr_lookup(attrs, "central_body", "body")),
            ("Target mode", _attr_lookup(attrs, "target_mode")),
            ("Degree min", degree_min),
            ("Degree max", degree_max),
            ("Altitude min (km)", _attr_lookup(attrs, "alt_min_km", "altitude_min_km", "alt_min")),
            ("Altitude max (km)", _attr_lookup(attrs, "alt_max_km", "altitude_max_km", "alt_max")),
            ("Derivative convention", _attr_lookup(attrs, "derivative_convention_version", "derivative_convention")),
            ("Gravity model", _attr_lookup(attrs, "gravity_model_path", "gfc_path", "gravity_model")),
            ("Include potential", _attr_lookup(attrs, "include_potential")),
            ("DU_m", _attr_lookup(attrs, "DU_m", "du_m")),
            ("TU_s", _attr_lookup(attrs, "TU_s", "tu_s")),
            ("VU_m_s", _attr_lookup(attrs, "VU_m_s", "vu_m_s")),
        ]
        html_rows = []
        for label, value in fields:
            shown = "—" if value is None else str(value)
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
    """Stage 1 — Data: dataset readiness, generation, and analysis."""

    def __init__(self, cloud_tab: QWidget, analysis_tab: QWidget,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.inspect_panel = DatasetInspectionPanel()
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self.inspect_panel, "Dataset")
        tabs.addTab(cloud_tab, "Generate")
        tabs.addTab(analysis_tab, "Analyze")
        self._tabs = tabs
        lo = QVBoxLayout()
        lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(tabs)
        self.setLayout(lo)


# =============================================================================
# 13c. EVALUATION PAGE  (model report + runtime performance + accuracy)
# =============================================================================


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


class EvaluationPage(QWidget):
    """Stage 3 — Evaluation: model report, accuracy, runtime performance."""

    def __init__(self, eval_tab: QWidget, profile_tab: QWidget,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.report_panel = ModelReportPanel()
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self.report_panel, "Model Report")
        tabs.addTab(eval_tab, "Accuracy Evaluation")
        tabs.addTab(profile_tab, "Performance Analysis")
        self._tabs = tabs
        lo = QVBoxLayout()
        lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(tabs)
        self.setLayout(lo)


# =============================================================================
# 14. MAIN WINDOW
# =============================================================================


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ST-LRPS Studio")
        self.resize(1320, 860)
        self.setMinimumSize(1024, 680)

        # --- Underlying tab widgets (preserved, re-homed into the 3 stages) ---
        self._cloud_tab    = CloudGenTab()
        self._train_tab    = STLRPSTrainTab()
        self._profile_tab  = STLRPSProfilingTab()
        self._eval_tab     = STLRPSEvalTab()
        self._analysis_tab = CloudAnalysisTab()

        self._cloud_tab.set_train_tab(self._train_tab)
        self._cloud_tab.cloud_params_changed.connect(self._train_tab.sync_from_cloud)

        # --- Three workflow stages: Data → Training → Evaluation ---
        self._data_page = DataPage(self._cloud_tab, self._analysis_tab)
        self._eval_page = EvaluationPage(self._eval_tab, self._profile_tab)
        self._data_page.inspect_panel.send_to_training.connect(self._on_dataset_to_training)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._data_page)   # index 0: Data
        self._stack.addWidget(self._train_tab)   # index 1: Training
        self._stack.addWidget(self._eval_page)   # index 2: Evaluation
        self._page_titles = ["Data", "Training", "Evaluation"]

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
                "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                "    stop:0 rgba(30, 20, 72, 0.80), stop:0.5 rgba(16, 24, 52, 0.75),"
                "    stop:1 rgba(10, 18, 46, 0.80));"
                "  border: 1px solid rgba(124, 92, 255, 0.24);"
                "  border-radius: 14px;"
                "}"
            )
            header_lo = QHBoxLayout()
            header_lo.setContentsMargins(18, 10, 18, 10)
            header_lo.setSpacing(16)

            title_col = QVBoxLayout()
            title_col.setContentsMargins(0, 0, 0, 0)
            title_col.setSpacing(3)
            lbl_title = QLabel("ST-LRPS Surrogate Console")
            lbl_title.setStyleSheet(
                "color: #e8ecf8; font-size: 15px; font-weight: 700;"
                " letter-spacing: 0.3px; background: transparent; border: none;"
            )
            lbl_subtitle = QLabel(
                "Sobolev-trained lunar residual potential models"
            )
            lbl_subtitle.setStyleSheet(
                "color: #8892b0; font-size: 12px; background: transparent; border: none;"
            )
            title_col.addWidget(lbl_title)
            title_col.addWidget(lbl_subtitle)
            header_lo.addLayout(title_col, 1)
            header_card.setLayout(header_lo)


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
        sidebar.setFixedWidth(192)
        sidebar.setStyleSheet(
            "QFrame#navSidebar {"
            "  background: rgba(9, 13, 26, 0.88);"
            "  border: 1px solid rgba(185, 194, 221, 0.11);"
            "  border-radius: 14px;"
            "}"
        )

        _NAV_BTN_STYLE = (
            "QPushButton {"
            "  text-align: left; padding: 9px 12px 9px 16px;"
            "  border: none; border-left: 3px solid transparent;"
            "  border-radius: 0; font-size: 13px; font-weight: 500;"
            "  color: #7480a8; background: transparent;"
            "}"
            "QPushButton:hover {"
            "  color: #c4ccff; background: rgba(124, 92, 255, 0.09);"
            "}"
            "QPushButton:checked {"
            "  color: #e8ecf8; font-weight: 600;"
            "  background: rgba(124, 92, 255, 0.16);"
            "  border-left: 3px solid rgba(124, 92, 255, 0.90);"
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

        lo = QVBoxLayout()
        lo.setContentsMargins(0, 8, 0, 14)
        lo.setSpacing(1)
        lo.addWidget(_section_lbl("WORKFLOW"))
        lo.addWidget(_nav_btn("1 · Data", 0))
        lo.addWidget(_nav_btn("2 · Training", 1))
        lo.addWidget(_nav_btn("3 · Evaluation", 2))
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


# =============================================================================
# 13. DARK THEME
# =============================================================================


def apply_premium_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#0b1020"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#070b14"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#0f1830"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#121a33"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#141e3a"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#7c5cff"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor("#9aa7ff"))
    app.setPalette(pal)

    app.setStyleSheet("""
        QWidget { font-size: 13px; color: #e8ecf8; }
        QMainWindow, QWidget {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0b1020, stop:1 #070a12);
        }
        QToolTip {
            background-color: #141e3a; color: #e8ecf8;
            border: 1px solid rgba(124, 92, 255, 0.45);
            border-radius: 8px; padding: 8px 10px; font-size: 12px;
        }
        QGroupBox {
            background-color: rgba(16, 24, 48, 0.72);
            border: 1px solid rgba(120, 92, 255, 0.22);
            border-radius: 14px; margin-top: 18px; padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 14px; padding: 3px 14px;
            color: #c4ccff; font-weight: 600; font-size: 13px;
            background-color: rgba(16, 24, 58, 0.98);
            border: 1px solid rgba(120, 92, 255, 0.26);
            border-radius: 8px;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 10px; padding: 0px 12px;
            min-height: 38px; selection-background-color: #7c5cff;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid rgba(124, 92, 255, 0.85);
        }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            color: rgba(185, 194, 221, 0.35); background-color: rgba(12, 16, 30, 0.6);
        }
        QSpinBox, QDoubleSpinBox { padding-right: 40px; }
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
            subcontrol-origin: border; width: 30px;
            background: rgba(18, 26, 51, 0.9);
            border-left: 1px solid rgba(185, 194, 221, 0.18);
        }
        QAbstractSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 10px; }
        QAbstractSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 10px; }
        QPlainTextEdit {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 12px; padding: 10px 12px;
            selection-background-color: #7c5cff;
        }
        QTabWidget::pane {
            border: 1px solid rgba(185, 194, 221, 0.18);
            border-radius: 14px; background-color: rgba(15, 24, 48, 0.35); top: -1px;
        }
        QTabBar {
            alignment: left;
        }
        QTabBar::tab {
            background: rgba(16, 24, 48, 0.65);
            border: 1px solid rgba(185, 194, 221, 0.14); border-bottom: none;
            padding: 9px 18px; margin-right: 4px;
            border-top-left-radius: 12px; border-top-right-radius: 12px;
            color: #8892b0; font-weight: 500; font-size: 13px;
            min-width: 80px; max-width: 240px;
        }
        QTabBar::tab:selected {
            background: rgba(15, 24, 48, 0.95);
            border-color: rgba(124, 92, 255, 0.50);
            border-top: 2px solid rgba(124, 92, 255, 0.80);
            color: #e8ecf8; font-weight: 600;
        }
        QTabBar::tab:hover:!selected { color: #c4ccff; background: rgba(20, 30, 58, 0.8); }
        QTabBar::scroller { width: 24px; }
        QTabBar QToolButton {
            background: rgba(16, 24, 48, 0.8);
            border: 1px solid rgba(185, 194, 221, 0.18);
            border-radius: 6px;
        }
        QTabBar QToolButton:hover { background: rgba(26, 36, 70, 0.95); }
        QProgressBar {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 9px; height: 18px; text-align: center; font-size: 11px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c5cff, stop:1 #38bdf8);
            border-radius: 9px;
        }
        QPushButton {
            border-radius: 10px; padding: 8px 16px;
            border: 1px solid rgba(185, 194, 221, 0.18);
            background-color: rgba(18, 26, 51, 0.9); font-weight: 500;
        }
        QPushButton:hover { background-color: rgba(26, 36, 70, 0.95); }
        QPushButton:pressed { background-color: rgba(14, 20, 40, 0.95); }
        QPushButton:disabled { color: rgba(232, 236, 248, 0.35); background-color: rgba(16, 24, 48, 0.35); }
        QPushButton[kind="primary"] {
            border: 1px solid rgba(124, 92, 255, 0.55);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(124, 92, 255, 0.95), stop:1 rgba(56, 189, 248, 0.85));
            color: #fff; font-weight: 600;
        }
        QPushButton[kind="primary"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(134, 102, 255, 1.0), stop:1 rgba(66, 199, 255, 0.95));
        }
        QPushButton[kind="danger"] {
            border: 1px solid rgba(248, 113, 113, 0.50);
            background-color: rgba(248, 113, 113, 0.14);
            color: #fca5a5;
        }
        QPushButton[kind="danger"]:hover {
            background-color: rgba(248, 113, 113, 0.26);
            border-color: rgba(248, 113, 113, 0.70);
        }
        QPushButton[kind="ghost"] {
            background-color: rgba(16, 24, 48, 0.30);
            border-color: rgba(185, 194, 221, 0.12);
            color: #9aa7c7;
        }
        QPushButton[kind="ghost"]:hover {
            background-color: rgba(26, 36, 70, 0.55);
            color: #c4ccff;
            border-color: rgba(185, 194, 221, 0.22);
        }
        QCheckBox { spacing: 10px; }
        QCheckBox::indicator {
            width: 17px; height: 17px; border-radius: 5px;
            border: 1px solid rgba(185, 194, 221, 0.22);
            background: rgba(7, 11, 20, 0.92);
        }
        QCheckBox::indicator:hover { border-color: rgba(124, 92, 255, 0.55); }
        QCheckBox::indicator:checked {
            background: rgba(124, 92, 255, 0.9);
            border-color: rgba(124, 92, 255, 0.92);
        }
        QCheckBox:disabled { color: rgba(185, 194, 221, 0.35); }
        QLabel { color: #b9c2dd; font-size: 12px; }
        QScrollBar:vertical { background: transparent; width: 10px; }
        QScrollBar::handle:vertical { background: rgba(185, 194, 221, 0.2); min-height: 28px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: rgba(185, 194, 221, 0.35); }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar:horizontal { background: transparent; height: 10px; }
        QScrollBar::handle:horizontal { background: rgba(185, 194, 221, 0.2); min-width: 28px; border-radius: 5px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QSplitter::handle { background: rgba(185, 194, 221, 0.07); }
        QSplitter::handle:horizontal { width: 5px; }
        QSplitter::handle:vertical   { height: 5px; }
        QSplitter::handle:hover      { background: rgba(124, 92, 255, 0.20); }
        QListWidget {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.18);
            border-radius: 12px; padding: 6px; font-size: 12px;
        }
        QListWidget::item { padding: 7px 10px; border-radius: 7px; }
        QListWidget::item:selected {
            background-color: rgba(124, 92, 255, 0.26); color: #ffffff;
        }
        QListWidget::item:hover:!selected { background-color: rgba(124, 92, 255, 0.10); }
        QStatusBar {
            background: rgba(7, 11, 20, 0.95);
            border-top: 1px solid rgba(185, 194, 221, 0.10);
            color: #6f7ca8; font-size: 11px;
        }
        QStatusBar::item { border: none; }
        QFrame#navSidebar QPushButton {
            border-radius: 0;
            border-left: 3px solid transparent;
        }
        QInputDialog { background-color: #0f1830; }
    """)


# =============================================================================
# 14. ENTRY POINT
# =============================================================================


def main() -> None:
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.chdir(str(SCRIPT_DIR))
    app = QApplication(sys.argv)
    apply_premium_dark_theme(app)
    _wheel_guard = _NoWheelOnSpinFilter(app)
    app.installEventFilter(_wheel_guard)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
