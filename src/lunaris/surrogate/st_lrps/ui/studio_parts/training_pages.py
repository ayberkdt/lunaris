"""
ST-LRPS Studio.

PySide6 dashboard for the lunar scalar potential surrogate codebase.

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
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lunaris.common.paths import project_root_from_file

try:
    from lunaris.surrogate.st_lrps.training.checkpoint_manager import resolve_best_ckpt_start_epoch
except Exception:  # pragma: no cover - optional training dependencies
    resolve_best_ckpt_start_epoch = None

from .qt_common import (
    THEME,
    NoScrollComboBox,
    QCheckBox,
    QColor,
    QDesktopServices,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QGuiApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProcess,
    QProcessEnvironment,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    Qt,
    QTimer,
    QUrl,
    QVBoxLayout,
    QWidget,
    pyqtgraph_matches_qt,
    pyqtSignal,
    with_alpha,
)

# pyqtgraph — optional, graceful fallback
try:
    import pyqtgraph as pg

    _HAS_PYQTGRAPH = pyqtgraph_matches_qt(pg)
except ImportError:
    _HAS_PYQTGRAPH = False

# h5py — optional, for dataset introspection
try:
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        CHECKPOINT_SCHEMA_VERSION,
        CRITICAL_CONFIG_FIELDS,
        compute_payload_sha256,
        make_run_layout,
        read_run_manifest,
    )
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        load_checkpoint as load_artifact_checkpoint,
    )
    from lunaris.surrogate.st_lrps.artifacts.manager import (
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
        KPIStrip,
        StructuredLogView,
        TimeMetricsStrip,
    )
    from lunaris.surrogate.st_lrps.ui.training_metrics import (
        EpochGuard,
        ETAEstimator,
        TrainingLogParser,
        TrainingMetricsStore,
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


from lunaris.ui.components.primitives import CompactSearchField, KeyValueList

from .common_widgets import (
    CollapsibleSection,
    DatasetInfoLabel,
    LiveLossPlot,
    ProcessPane,
    ValidatedPathEdit,
    _cfg_value,
    _format_command,
    _inspect_run_artifacts,
    _make_page_header,
    _mono_font,
    _norm_path,
    _read_json_if_exists,
    _row_lineedit_with_button,
    _scroll_wrap,
    _send_os_notification,
    _settings,
    _split_cli_args,
    _style_command_preview,
    _style_surface,
    _tune_form,
    _tune_inputs,
)
from .dataset_introspection import inspect_h5_metadata as _introspect_h5
from .monitor_widgets import PeriodicEvalTable, RunProvenancePanel
from .workspace_widgets import StudioNotice, StudioStatusBadge


def _base_preset(**overrides) -> dict[str, Any]:
    base = {
        "dataset_mode": "independent",
        "dataset_suite_dir": "",
        "hidden": 512, "depth": 6, "activation": "sine",
        "w0_first": 30.0, "w0_hidden": 30.0, "dropout": 0.0,
        "model_preset": "recommended_physical_radial_decay",
        "use_fourier": False, "fourier_n": 256, "fourier_sigma": 1.0,
        "fourier_append_raw": True,
        "epochs": 400, "batch_size": 8192,
        "lr": 1e-4, "weight_decay": 1e-6, "output_head_lr_mult": 1.0,
        "t_max": 390, "warmup_epochs": 5, "min_lr_ratio": 0.05,
        "patience": 30, "no_amp": False,
        "w_u": 1.0, "w_a": 1.0, "gradnorm_mode": "ntk_init",
        "gradnorm_w_a_min": 0.05, "gradnorm_w_a_max": 2.0,
        "potential_only_epochs": 10, "accel_ramp_epochs": 80,
        "accel_min_factor": 0.05,
        "a_sign": "auto", "use_si_index": 0,
        "direction_loss_weight": 0.10, "direction_loss_start_epoch": 30,
        "direction_loss_ramp_epochs": 50, "direction_loss_floor_abs": 3e-6,
        "best_ckpt_start_epoch": -1, "checkpoint_settle_epochs": 5,
        "use_altitude_balanced_loss": True,
        "altitude_bin_width_km": 50.0,
        "altitude_min_km": _cfg_value(DEFAULT_SPATIAL_CLOUD_CONFIG, "alt_min_km", 100.0),
        "altitude_max_km": _cfg_value(DEFAULT_SPATIAL_CLOUD_CONFIG, "alt_max_km", 1000.0),
        "resume_enabled": False, "resume_from": "",
        "resume_checkpoint": "last", "resume_nonstrict": False,
        "resume_history_mode": "append",
        "use_radial_cross_loss": True,
        "radial_loss_weight": 0.05,
        "cross_loss_weight": 0.05,
        "use_laplacian_regularization": True,
        "laplacian_weight": 2e-9,
        "laplacian_every_n_batches": 100,
        "laplacian_subset_size": 512,
        "max_grad_norm": 0.5, "num_workers": 2, "cache_rows": 65536,
        "fit_rows": 500_000, "seed": 42, "split_seed": 42,
        "log_every": 10, "log_every_mode": "auto",
        "preload_data": False, "auto_preload_mb": 2048.0,
        "pin_memory": True, "quick_check": False, "extra_args": "",
        "use_residual_blocks": True,
        "n_bands": 3,
        "grad_accumulation_steps": 1,
        "n_hutchinson_samples": 4,
    }
    base.update(overrides)
    return base


def _legacy_reference_preset(**overrides) -> dict[str, Any]:
    legacy = {
        "depth": 5,
        "auto_preload_mb": 256.0,
        "use_residual_blocks": False,
        "n_bands": 1,
    }
    legacy.update(overrides)
    return _base_preset(**legacy)


_BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "Quick Debug": _base_preset(
        hidden=64, depth=2, epochs=5, batch_size=1024,
        lr=1e-3, weight_decay=0.0, t_max=5, warmup_epochs=0,
        patience=5, num_workers=0, cache_rows=8192, fit_rows=50_000,
        direction_loss_weight=0.0, quick_check=True,
    ),
    "Strong Benchmark": _base_preset(),
    "Default SIREN": _base_preset(),
    "Legacy Reference": _legacy_reference_preset(),
    "Ablation: Raw Input": _base_preset(model_preset="baseline_raw"),
    "Ablation: Single Band": _base_preset(n_bands=1),
    "Ablation: No Aux Losses": _base_preset(
        direction_loss_weight=0.0,
        use_altitude_balanced_loss=False,
        use_radial_cross_loss=False,
        use_laplacian_regularization=False,
        radial_loss_weight=0.0,
        cross_loss_weight=0.0,
        laplacian_weight=0.0,
    ),
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
        model_preset="custom",
        activation="silu", use_fourier=True, fourier_n=256, fourier_sigma=1.0,
        hidden=512, depth=4, lr=2e-4, weight_decay=1e-6, batch_size=8192,
    ),
    "Stable SIREN": _base_preset(
        hidden=512, depth=5, activation="sine", w0_first=30.0, w0_hidden=30.0,
        lr=1e-4, weight_decay=1e-6, output_head_lr_mult=1.0, max_grad_norm=0.5,
        warmup_epochs=5, potential_only_epochs=0, accel_ramp_epochs=80,
        accel_min_factor=0.05,
        gradnorm_mode="ntk_init", gradnorm_w_a_min=0.05, gradnorm_w_a_max=2.0,
        use_laplacian_regularization=False,
        laplacian_weight=0.0, radial_loss_weight=0.0, cross_loss_weight=0.0,
        direction_loss_weight=0.05, direction_loss_start_epoch=30,
        direction_loss_ramp_epochs=50, batch_size=8192,
    ),
}

def _load_user_presets() -> dict[str, dict[str, Any]]:
    presets: dict[str, dict[str, Any]] = {}
    if not _PRESETS_DIR.is_dir():
        return presets
    for fp in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                presets[fp.stem] = json.load(f)
        except Exception:
            pass
    return presets


def _save_user_preset(name: str, data: dict[str, Any]) -> None:
    _PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_PRESETS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _delete_user_preset(name: str) -> bool:
    fp = _PRESETS_DIR / f"{name}.json"
    if fp.is_file():
        fp.unlink()
        return True
    return False


class _ReorderableQueueList(QListWidget):
    order_changed = pyqtSignal()

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        self.order_changed.emit()


class TrainingQueue(QWidget):
    """
    Sequential training queue — users can enqueue multiple configurations
    and the queue executes them one-by-one.  Each item stores a full
    argument list (List[str]) that will be passed to QProcess.

    Workflow:
    1. User configures parameters, clicks "Add to Queue".
    2. Item appears in the list with a short description.
    3. User clicks "Start Queue" — queue runs jobs sequentially.
    4. On each job finish, the next one starts automatically.
    """

    # Emitted when a job from the queue starts: (job_index, args_list)
    job_started = pyqtSignal(int, list)
    # Emitted when entire queue is done
    queue_finished = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._queue: list[dict[str, Any]] = []  # [{label, args, out_dir, config}, ...]
        self._current_index: int = -1
        self._running: bool = False

        # --- Header ---
        lbl = QLabel("Training Queue")
        lbl.setStyleSheet(
            f"font-weight: 600; color: {THEME['accent_hov']}; font-size: 13px; padding: 2px 0;"
        )

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 11px;")

        # --- List ---
        self._list = _ReorderableQueueList()
        self._list.setMinimumHeight(80)
        self._list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.order_changed.connect(self._on_reordered)

        policy_lbl = QLabel("On failure:")
        policy_lbl.setObjectName("fieldLabel")
        self.failure_policy = NoScrollComboBox()
        self.failure_policy.addItem("Stop queue", "stop")
        self.failure_policy.addItem("Continue queue", "continue")
        self.failure_policy.setAccessibleName("Queue failure policy")
        self.failure_policy.setToolTip("Choose whether a failed job stops or advances the queue.")
        self.failure_policy.currentIndexChanged.connect(lambda *_: self._save_persisted())

        # --- Buttons ---
        self.btn_start_queue = QPushButton("Start Queue")
        self.btn_start_queue.setProperty("kind", "primary")
        self.btn_start_queue.setToolTip("Run all queued training jobs sequentially.")
        self.btn_start_queue.clicked.connect(self._start_queue)

        self.btn_stop_queue = QPushButton("Stop Queue")
        self.btn_stop_queue.setProperty("kind", "danger")
        self.btn_stop_queue.setEnabled(False)
        self.btn_stop_queue.clicked.connect(self._stop_queue)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.setProperty("kind", "ghost")
        btn_remove.clicked.connect(self._remove_selected)

        btn_clear_q = QPushButton("Clear Queue")
        btn_clear_q.setProperty("kind", "ghost")
        btn_clear_q.clicked.connect(self._clear_queue)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_start_queue)
        btn_row.addWidget(self.btn_stop_queue)
        btn_row.addWidget(policy_lbl)
        btn_row.addWidget(self.failure_policy)
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
        self._restore_persisted()
        self._update_status()

    # --- Public API (called by STLRPSTrainTab) ---

    def enqueue(
        self,
        label: str,
        args: list[str],
        out_dir: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Add a training job to the queue."""
        item_data = {
            "label": label,
            "args": args,
            "out_dir": out_dir,
            "config": config or {},
            "status": "Pending",
        }
        self._queue.append(item_data)
        self._refresh_list()
        self._update_status()
        self._save_persisted()

    def is_running(self) -> bool:
        return self._running

    def current_args(self) -> list[str] | None:
        if 0 <= self._current_index < len(self._queue):
            return self._queue[self._current_index]["args"]
        return None

    def on_job_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Called by STLRPSTrainTab when the current subprocess finishes."""
        if not self._running or self._current_index < 0:
            return

        job_ok = exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
        if job_ok:
            self._queue[self._current_index]["status"] = f"Completed (exit={exit_code})"
        else:
            self._queue[self._current_index]["status"] = f"Error (exit={exit_code})"

        self._refresh_list()
        self._save_persisted()

        if not job_ok and self.failure_policy.currentData() == "stop":
            self._queue[self._current_index]["status"] = f"Stopped after failure (exit={exit_code})"
            if self._current_index >= 0:
                self._running = False
                self._current_index = -1
                self.btn_start_queue.setEnabled(True)
                self.btn_stop_queue.setEnabled(False)
                self._save_persisted()
                self._update_status()
                return

        self._advance_queue()

    # --- Internal ---

    def _start_queue(self) -> None:
        pending = [i for i, q in enumerate(self._queue) if q["status"] == "Pending"]
        if not pending:
            QMessageBox.information(self, "Empty Queue", "No pending jobs.")
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
            if self._queue[i]["status"] == "Pending":
                next_idx = i
                break

        if next_idx is None:
            # Queue done
            self._running = False
            self._current_index = -1
            self.btn_start_queue.setEnabled(True)
            self.btn_stop_queue.setEnabled(False)
            self._update_status()
            self._save_persisted()
            _send_os_notification("Lunar Potential Surrogate", "Training queue completed!")
            self.queue_finished.emit()
            return

        self._current_index = next_idx
        self._queue[next_idx]["status"] = "Running…"
        self._refresh_list()
        self._update_status()
        self._save_persisted()
        self.job_started.emit(next_idx, self._queue[next_idx]["args"])

    def _stop_queue(self) -> None:
        self._running = False
        if 0 <= self._current_index < len(self._queue):
            self._queue[self._current_index]["status"] = "Stopped"
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
                    self, "Cannot Remove", "The running job cannot be removed."
                )
                return
            self._queue.pop(row)
            if self._current_index > row:
                self._current_index -= 1
            self._refresh_list()
            self._update_status()
            self._save_persisted()

    def _clear_queue(self) -> None:
        if self._running:
            QMessageBox.warning(self, "Cannot Clear", "The queue cannot be cleared while running.")
            return
        self._queue.clear()
        self._current_index = -1
        self._refresh_list()
        self._update_status()
        self._save_persisted()

    def _refresh_list(self) -> None:
        self._list.clear()
        for i, job in enumerate(self._queue):
            icon = {"Pending": "⏳", "Running…": "▶️", "Stopped": "⏹️"}.get(
                job["status"], "✅" if "Completed" in job["status"] else "❌"
            )
            text = f"{icon}  [{i + 1}] {job['label']}  —  {job['status']}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            if "Running" in job["status"]:
                item.setForeground(QColor(THEME["accent"]))
            elif "Completed" in job["status"]:
                item.setForeground(QColor(THEME["success"]))
            elif "Error" in job["status"] or "Stopped" in job["status"]:
                item.setForeground(QColor(THEME["error"]))
            self._list.addItem(item)

    def _on_reordered(self) -> None:
        if self._running:
            self._refresh_list()
            return
        old = list(self._queue)
        ordered: list[dict[str, Any]] = []
        for row in range(self._list.count()):
            old_index = self._list.item(row).data(Qt.ItemDataRole.UserRole)
            if isinstance(old_index, int) and 0 <= old_index < len(old):
                ordered.append(old[old_index])
        if len(ordered) == len(old):
            self._queue = ordered
            self._refresh_list()
            self._save_persisted()

    def _save_persisted(self) -> None:
        payload = [
            {key: job.get(key) for key in ("label", "args", "out_dir", "config", "status")}
            for job in self._queue
        ]
        settings = _settings()
        settings.setValue("training_queue/jobs", json.dumps(payload, ensure_ascii=False))
        settings.setValue("training_queue/failure_policy", self.failure_policy.currentData() or "stop")
        settings.sync()

    def _restore_persisted(self) -> None:
        settings = _settings()
        raw = settings.value("training_queue/jobs", "")
        try:
            payload = json.loads(str(raw)) if raw else []
        except (TypeError, ValueError):
            payload = []
        if isinstance(payload, list):
            for job in payload:
                if not isinstance(job, dict) or not isinstance(job.get("args"), list):
                    continue
                restored = dict(job)
                if str(restored.get("status", "Pending")).startswith("Running"):
                    restored["status"] = "Pending"
                self._queue.append(restored)
        policy = settings.value("training_queue/failure_policy", "stop")
        index = self.failure_policy.findData(str(policy))
        if index >= 0:
            self.failure_policy.setCurrentIndex(index)
        self._refresh_list()

    def _update_status(self) -> None:
        pending = sum(1 for q in self._queue if q["status"] == "Pending")
        done = sum(1 for q in self._queue if "Completed" in q["status"])
        total = len(self._queue)
        if self._running:
            self._status_lbl.setText(
                f"Running: {self._current_index + 1}/{total}  |  Pending: {pending}"
            )
        elif total > 0:
            self._status_lbl.setText(
                f"Total: {total}  |  Completed: {done}  |  Pending: {pending}"
            )
        else:
            self._status_lbl.setText(
                "No queued runs — add the current profile to batch-train multiple experiments."
            )


def _find_resumable_run() -> tuple[str, str] | None:
    import json
    import os
    from pathlib import Path

    from lunaris.utils.env import get_output_dir

    training_dir = Path(get_output_dir()) / "training"
    if not training_dir.exists():
        return None
    try:
        manifests = sorted(training_dir.glob("*/run_manifest.json"), key=os.path.getmtime, reverse=True)
    except Exception:
        manifests = []
    for manifest_path in manifests:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
                if manifest.get("status") in ("interrupted", "running"):
                    return manifest_path.parent.name, str(manifest_path.parent)
        except Exception:
            pass
    return None

class STLRPSTrainTab(QWidget):
    navigate_monitor_requested = pyqtSignal()
    evaluate_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        # =====================================================================
        # PRESET BAR
        # =====================================================================
        self._preset_combo = NoScrollComboBox()
        self._preset_combo.setMinimumWidth(200)
        self._preset_combo.setToolTip("Saved hyperparameter profiles.")
        self._refresh_preset_list()

        btn_load_preset = QPushButton("Load")
        btn_load_preset.clicked.connect(self._load_preset)
        btn_save_preset = QPushButton("Save")
        btn_save_preset.clicked.connect(self._save_preset)
        btn_del_preset = QPushButton("Delete")
        btn_del_preset.setProperty("kind", "danger")
        btn_del_preset.clicked.connect(self._delete_preset)

        preset_bar = QHBoxLayout()
        preset_bar.setContentsMargins(4, 0, 4, 0)
        preset_bar.setSpacing(8)
        preset_lbl = QLabel("Profile:")
        preset_lbl.setStyleSheet(
            f"font-weight: 600; color: {THEME['fg_soft']}; font-size: 13px;"
        )
        preset_bar.addWidget(preset_lbl)
        preset_bar.addWidget(self._preset_combo, 1)
        preset_bar.addWidget(btn_load_preset)
        preset_bar.addWidget(btn_save_preset)
        preset_bar.addWidget(btn_del_preset)

        # ── Workflow Mode ──────────────────────────────────────────────
        self.workflow_mode = NoScrollComboBox()
        self.workflow_mode.addItem("Train only",              "train_only")
        self.workflow_mode.addItem("Evaluate only",           "eval_only")
        self.workflow_mode.addItem("Train then evaluate",     "train_then_eval")
        self.workflow_mode.addItem("Queue training runs",     "queue")
        self.workflow_mode.setCurrentIndex(2)  # default: Train then evaluate
        self.workflow_mode.setToolTip(
            f"Train only:         Runs python -m {TRAIN_CLI_MODULE}.\n"
            f"Evaluate only:      Runs python -m {EVAL_CLI_MODULE} (existing model folder required).\n"
            "Train then eval:    Evaluation starts automatically as soon as training finishes.\n"
            "Queue:              All queued jobs run sequentially."
        )
        self.workflow_mode.currentIndexChanged.connect(self._on_workflow_mode_changed)
        wf_lbl = QLabel("Workflow:")
        wf_lbl.setStyleSheet(
            f"font-weight: 600; color: {THEME['fg_soft']}; font-size: 13px;"
        )
        workflow_bar = QHBoxLayout()
        workflow_bar.setContentsMargins(4, 2, 4, 2)
        workflow_bar.setSpacing(8)
        workflow_bar.addWidget(wf_lbl)
        workflow_bar.addWidget(self.workflow_mode, 1)

        # Readiness checklist (compact, shown above Start)
        self._checklist_label = QLabel("")
        self._checklist_label.setWordWrap(True)
        self._checklist_label.setStyleSheet(
            f"QLabel {{ font-size: 11px; color: {THEME['fg_muted']}; "
            f"background: {with_alpha(THEME['bg_inset'], 0.70)}; "
            "border-radius: 6px; padding: 4px 8px; }"
        )
        self._checklist_label.setVisible(False)
        self._launch_badge = StudioStatusBadge("Checking", "info")
        self._launch_summary = QLabel("Readiness checks will appear here.")
        self._launch_summary.setAccessibleName("Launch readiness details")
        self._launch_summary.setWordWrap(True)
        self._launch_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._launch_summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._launch_summary.setStyleSheet(
            f"color: {THEME['fg_soft']}; font-size: 11px;"
        )

        # =====================================================================
        # GROUP 1: Data & I/O
        # =====================================================================
        grp_data = QGroupBox("Data and Input/Output")
        form_data = QFormLayout()
        _tune_form(form_data)

        self.dataset_mode = NoScrollComboBox()
        self.dataset_mode.addItem("Dataset suite / independent files", "independent")
        self.dataset_mode.addItem("Single dataset + internal split", "single")
        self.dataset_mode.setToolTip(
            "Suite/independent mode passes --train-data and --val-data explicitly. "
            f"Single mode passes --data and lets python -m {TRAIN_CLI_MODULE} split train/val."
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
        self.dataset_suite_dir = ValidatedPathEdit(
            placeholder="Select generated suite folder", check_file=False
        )
        self._last_dataset_dir: Path | None = None
        btn_suite_dir = QPushButton("Select...")
        btn_train_data = QPushButton("Select...")
        btn_val_data = QPushButton("Select...")
        btn_test_data = QPushButton("Select...")
        btn_ood_data = QPushButton("Select...")
        btn_apply_suite = QPushButton("Apply")
        btn_apply_suite.setProperty("kind", "ghost")
        btn_apply_suite.setToolTip(
            "Read the selected suite folder and fill train/validation/test/OOD paths."
        )
        btn_suite_dir.clicked.connect(self._pick_dataset_suite)
        btn_apply_suite.clicked.connect(self._apply_dataset_suite_from_field)
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
        self.dataset_suite_dir.textChanged.connect(self._refresh_checklist)
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
        suite_row = QHBoxLayout()
        suite_row.setContentsMargins(0, 0, 0, 0)
        suite_row.setSpacing(8)
        suite_row.addWidget(self.dataset_suite_dir, 1)
        suite_row.addWidget(btn_suite_dir)
        suite_row.addWidget(btn_apply_suite)
        suite_row_widget = QWidget()
        suite_row_widget.setLayout(suite_row)
        independent_form.addRow("Dataset Suite Folder", suite_row_widget)
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
            placeholder=f"Empty -> {TRAINING_OUTPUT_ROOT}/st_lrps_train_<timestamp>", check_file=False
        )
        self.out_dir.setToolTip(
            f"Training run directory. Empty lets the CLI create {TRAINING_OUTPUT_ROOT}/st_lrps_train_<timestamp>."
        )
        btn_out = QPushButton("Select...")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)

        self.dataset_name = QLineEdit("data")
        self.dataset_name.setToolTip("HDF5 dataset name.")

        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setDecimals(4)
        self.val_ratio.setRange(0.0, 0.5)
        self.val_ratio.setValue(0.1)
        self.val_ratio.setSingleStep(0.01)
        self.val_ratio.setToolTip("Validation fraction in single-dataset mode. 0.1 -> 10% val, 90% train.")

        self._suite_manifest_label = QLabel("(no suite applied)")
        self._suite_manifest_label.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 10px;")
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

        self.resume_checkpoint = NoScrollComboBox()
        self.resume_checkpoint.addItem("last", "last")
        self.resume_checkpoint.addItem("best", "best")
        self.resume_checkpoint.setCurrentIndex(0)

        self.resume_nonstrict = QCheckBox("Allow non-critical config differences")
        self.resume_nonstrict.setChecked(False)

        self.resume_history_mode = NoScrollComboBox()
        self.resume_history_mode.addItem("append previous history", "append")
        self.resume_history_mode.addItem("overwrite history", "overwrite")
        self.resume_history_mode.setCurrentIndex(0)

        resume_help = QLabel(
            "Resume defaults to ckpt_last.pt. --epochs is the total target epoch count, "
            "not additional epochs. Resume is epoch-level; if interrupted mid-epoch, "
            "training resumes from the last completed checkpoint."
        )
        resume_help.setWordWrap(True)
        resume_help.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 11px;")

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

        resumable = _find_resumable_run()

        from lunaris.ui.components.primitives import InlineNotice
        self.resume_notice = InlineNotice(
            f"Resumable run found: {resumable[0]}" if resumable else "",
            kind="info"
        )
        self.resume_notice.setVisible(bool(resumable))
        resume_vbox.addWidget(self.resume_notice)

        resume_vbox.addWidget(resume_inner)
        self.resume_section.set_content_layout(resume_vbox)

        if resumable:
            self.resume_section.set_expanded(True)
            # Pre-fill the path without triggering autoload early
            self.resume_from.setText(resumable[1])
        else:
            self.resume_section.set_expanded(False)
        self.resume_enabled.toggled.connect(self._on_resume_toggled)
        # Autoload only on the genuine user toggle signal (NOT when _apply_config
        # re-invokes _on_resume_toggled directly, which would clobber a profile).
        self.resume_enabled.toggled.connect(
            lambda checked: self._autoload_resume_config() if checked else None
        )
        self.resume_from.textChanged.connect(self._refresh_checklist)
        # When a resume source is typed/pasted, mirror its architecture into the UI.
        self.resume_from.editingFinished.connect(self._autoload_resume_config)

        # =====================================================================
        # GROUP 2: Model Architecture
        # =====================================================================
        grp_arch = QGroupBox("Model Architecture")
        form_arch = QFormLayout()
        _tune_form(form_arch)

        self.hidden = QSpinBox()
        self.hidden.setRange(1, 8192)
        self.hidden.setValue(512)
        self.hidden.setToolTip(
            "Number of neurons (units) in each hidden layer.\n"
            "Higher value → more expressive network, slower training.\n"
            "Recommended range for SIREN: 256–1024."
        )
        self.depth = QSpinBox()
        self.depth.setRange(1, 64)
        self.depth.setValue(6)
        self.depth.setToolTip(
            "Number of hidden layers (depth).\n"
            "Residual SIREN blocks keep the benchmark default stable at depth 6.\n"
            "Recommended benchmark: 6."
        )
        self.activation = NoScrollComboBox()
        self.activation.addItems(["sine", "silu", "tanh", "softplus"])
        self.activation.setCurrentText("sine")
        self.activation.setToolTip(
            "Activation function:\n"
            "  sine   — SIREN network; smooth derivatives, recommended for high-frequency fields.\n"
            "  silu   — SiLU (Swish); general-purpose, faster convergence.\n"
            "  tanh   — Classic MLP; balanced for small networks.\n"
            "  softplus — Used when a positive output is required.\n"
            "Note: SIREN activation cannot be combined with Fourier embedding."
        )
        self.dropout = QDoubleSpinBox()
        self.dropout.setDecimals(4)
        self.dropout.setRange(0.0, 0.99)
        self.dropout.setValue(0.0)
        self.dropout.setSingleStep(0.01)
        self.dropout.setToolTip(
            "Dropout rate (0 = disabled).\n"
            "During training this fraction of neurons is randomly deactivated.\n"
            "Usually 0 is recommended for physics-based networks — excessive regularization hurts accuracy."
        )

        # SIREN w0 controls
        self.w0_first = QDoubleSpinBox()
        self.w0_first.setDecimals(1)
        self.w0_first.setRange(1.0, 200.0)
        self.w0_first.setValue(30.0)
        self.w0_first.setSingleStep(1.0)
        self.w0_first.setToolTip(
            "SIREN first-layer frequency multiplier (ω₀).\n"
            "The first layer's sine is scaled by this value: sin(ω₀·Wx+b).\n"
            f"Default: 30. If left empty, python -m {TRAIN_CLI_MODULE} computes it automatically from the dataset."
        )
        self.w0_hidden = QDoubleSpinBox()
        self.w0_hidden.setDecimals(1)
        self.w0_hidden.setRange(1.0, 200.0)
        self.w0_hidden.setValue(30.0)
        self.w0_hidden.setSingleStep(1.0)
        self.w0_hidden.setToolTip(
            "SIREN hidden-layer frequency multiplier (ω₀).\n"
            "Applied to all layers except the first.\n"
            "Usually kept equal to the first-layer value: 30."
        )

        # Fourier/RFF section (only valid for non-sine activations)
        self._fourier_section = CollapsibleSection("Fourier/RFF Embedding (non-sine activation)")
        form_fourier = QFormLayout()
        _tune_form(form_fourier)
        self.use_fourier = QCheckBox("Enable Fourier/RFF Embedding")
        self.use_fourier.setChecked(False)
        self.use_fourier.setToolTip(
            "Random Fourier Feature embedding. NOT used with SIREN."
        )
        self.fourier_info = QLabel("SIREN and Fourier/RFF are mutually exclusive.")
        self.fourier_info.setWordWrap(True)
        self.fourier_info.setStyleSheet(f"color: {THEME['warning']}; font-size: 11px;")
        self.fourier_n = QSpinBox()
        self.fourier_n.setRange(16, 4096)
        self.fourier_n.setValue(256)
        self.fourier_n.setToolTip("Number of Fourier features (n → 2n-dimensional embedding).")
        self.fourier_sigma = QDoubleSpinBox()
        self.fourier_sigma.setDecimals(3)
        self.fourier_sigma.setRange(0.001, 100.0)
        self.fourier_sigma.setValue(1.0)
        self.fourier_sigma.setToolTip("Standard deviation σ of the frequency matrix.")
        self.fourier_append_raw = QCheckBox("Append Raw Coordinates")
        self.fourier_append_raw.setChecked(True)
        self.fourier_append_raw.setToolTip("Also append the original xyz to the Fourier features.")
        form_fourier.addRow("", self.fourier_info)
        form_fourier.addRow(self.use_fourier)
        form_fourier.addRow("Feature Count (n)", self.fourier_n)
        form_fourier.addRow("Sigma (σ)", self.fourier_sigma)
        form_fourier.addRow(self.fourier_append_raw)
        _fourier_inner = QWidget()
        _fourier_inner.setLayout(form_fourier)
        _tune_inputs(_fourier_inner)
        _fourier_vbox = QVBoxLayout()
        _fourier_vbox.setContentsMargins(0, 0, 0, 0)
        _fourier_vbox.addWidget(_fourier_inner)
        self._fourier_section.set_content_layout(_fourier_vbox)

        self._w0_row_first = ("SIREN w0 (first layer)", self.w0_first)
        self._w0_row_hidden = ("SIREN w0 (hidden layer)", self.w0_hidden)

        form_arch.addRow("Hidden Layer Size", self.hidden)
        form_arch.addRow("Layer Depth", self.depth)
        form_arch.addRow("Activation Function", self.activation)
        self._w0_first_row_idx = form_arch.rowCount()
        form_arch.addRow("SIREN w0 (first layer)", self.w0_first)
        self._w0_hidden_row_idx = form_arch.rowCount()
        form_arch.addRow("SIREN w0 (hidden layer)", self.w0_hidden)
        form_arch.addRow("Dropout Rate", self.dropout)
        grp_arch.setLayout(form_arch)

        self.activation.currentTextChanged.connect(self._on_activation_changed)

        # =====================================================================
        # GROUP 3: Optimization
        # =====================================================================
        grp_optim = QGroupBox("Optimization")
        form_optim = QFormLayout()
        _tune_form(form_optim)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5_000_000)
        self.epochs.setValue(400)
        self.epochs.setToolTip(
            "Total number of training epochs.\n"
            "Each epoch is one pass over the entire training dataset."
        )
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 10_000_000)
        self.batch_size.setValue(8192)
        self.batch_size.setToolTip(
            "Number of samples (rows) used per update step.\n"
            "Large batch → stable gradients, higher GPU memory use.\n"
            "Recommended for SIREN: 4096–16384."
        )
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(8)
        self.lr.setRange(1e-8, 10.0)
        self.lr.setValue(1e-4)
        self.lr.setSingleStep(1e-5)
        self.lr.setToolTip(
            "Initial learning rate.\n"
            "The AdamW optimizer starts here and decays with a cosine schedule.\n"
            "Recommended for SIREN: 1e-4. Values that are too high can cause the network to diverge."
        )
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setDecimals(8)
        self.weight_decay.setRange(0.0, 10.0)
        self.weight_decay.setValue(1e-6)
        self.weight_decay.setSingleStep(1e-7)
        self.weight_decay.setToolTip(
            "AdamW L2 weight regularization (weight decay).\n"
            "Reduces overfitting by pulling weights toward zero.\n"
            "Recommended: between 1e-6 and 1e-4."
        )
        self.output_head_lr_mult = QDoubleSpinBox()
        self.output_head_lr_mult.setDecimals(2)
        self.output_head_lr_mult.setRange(0.1, 100.0)
        self.output_head_lr_mult.setValue(1.0)
        self.output_head_lr_mult.setSingleStep(0.5)
        self.output_head_lr_mult.setToolTip(
            "Output-head LR multiplier. Updates the output head's weights faster."
        )
        self.t_max = QSpinBox()
        self.t_max.setRange(1, 1_000_000)
        self.t_max.setValue(390)
        self.t_max.setToolTip("Cosine LR T_max. If left empty, same as epochs.")
        self.warmup_epochs = QSpinBox()
        self.warmup_epochs.setRange(0, 100)
        self.warmup_epochs.setValue(5)
        self.warmup_epochs.setToolTip("Number of linear LR warmup epochs.")
        self.min_lr_ratio = QDoubleSpinBox()
        self.min_lr_ratio.setDecimals(4)
        self.min_lr_ratio.setRange(0.0, 1.0)
        self.min_lr_ratio.setValue(0.05)
        self.min_lr_ratio.setSingleStep(0.01)
        self.min_lr_ratio.setToolTip("Ratio of minimum LR at the end of cosine to the initial LR.")
        self.patience = QSpinBox()
        self.patience.setRange(1, 10000)
        self.patience.setValue(30)
        self.patience.setToolTip("Early-stopping patience (when val loss does not improve).")
        self.no_amp = QCheckBox("AMP Off")
        self.no_amp.setChecked(False)
        self.no_amp.setToolTip("Disable AMP (mixed precision).")

        form_optim.addRow("Epoch Count", self.epochs)
        form_optim.addRow("Batch Size", self.batch_size)
        form_optim.addRow("Learning Rate (LR)", self.lr)
        form_optim.addRow("Weight Decay", self.weight_decay)
        form_optim.addRow("Output Head LR Multiplier", self.output_head_lr_mult)
        form_optim.addRow("Cosine LR Period (T_max)", self.t_max)
        form_optim.addRow("Warmup Epochs", self.warmup_epochs)
        form_optim.addRow("Min LR Ratio", self.min_lr_ratio)
        form_optim.addRow("Early-Stopping Patience", self.patience)
        form_optim.addRow(self.no_amp)
        grp_optim.setLayout(form_optim)

        # =====================================================================
        # GROUP 4: Physics & Sobolev Loss
        # =====================================================================
        grp_phys = QGroupBox("Physics and Sobolev Loss")
        form_phys = QFormLayout()
        _tune_form(form_phys)

        self.w_u = QDoubleSpinBox()
        self.w_u.setDecimals(6)
        self.w_u.setRange(0.0, 1e6)
        self.w_u.setValue(1.0)
        self.w_u.setSingleStep(0.1)
        self.w_u.setToolTip("Potential (U) loss weight (in fixed/ntk_init modes).")
        self.w_a = QDoubleSpinBox()
        self.w_a.setDecimals(6)
        self.w_a.setRange(0.0, 1e6)
        self.w_a.setValue(1.0)
        self.w_a.setSingleStep(0.1)
        self.w_a.setToolTip("Acceleration (a) loss weight (in fixed/ntk_init modes).")

        self.gradnorm_mode = NoScrollComboBox()
        self.gradnorm_mode.addItems(["ntk_init", "fixed", "dynamic"])
        self.gradnorm_mode.setCurrentText("ntk_init")
        self.gradnorm_mode.setToolTip(
            "ntk_init: balances gradient norms on the first step, then keeps them fixed (recommended).\n"
            "fixed: w_u and w_a stay constant.\n"
            "dynamic: updates dynamically each step based on the gradient norm."
        )
        self.gradnorm_mode.currentTextChanged.connect(self._on_gradnorm_mode_changed)

        self.gradnorm_w_a_min = QDoubleSpinBox()
        self.gradnorm_w_a_min.setDecimals(4)
        self.gradnorm_w_a_min.setRange(0.0, 100.0)
        self.gradnorm_w_a_min.setValue(0.05)
        self.gradnorm_w_a_min.setToolTip("Lower bound for NTK/dynamic w_a.")
        self.gradnorm_w_a_max = QDoubleSpinBox()
        self.gradnorm_w_a_max.setDecimals(4)
        self.gradnorm_w_a_max.setRange(0.01, 1000.0)
        self.gradnorm_w_a_max.setValue(2.0)
        self.gradnorm_w_a_max.setToolTip("Upper bound for NTK/dynamic w_a.")

        self.potential_only_epochs = QSpinBox()
        self.potential_only_epochs.setRange(0, 1000)
        self.potential_only_epochs.setValue(10)
        self.potential_only_epochs.setToolTip(
            "Initial warm-up epoch count. Acceleration stays active at the accel_min_factor floor."
        )
        self.accel_ramp_epochs = QSpinBox()
        self.accel_ramp_epochs.setRange(0, 1000)
        self.accel_ramp_epochs.setValue(80)
        self.accel_ramp_epochs.setToolTip(
            "Number of epochs to ramp acceleration loss linearly from accel_min_factor to full weight."
        )

        self.accel_min_factor = QDoubleSpinBox()
        self.accel_min_factor.setDecimals(4)
        self.accel_min_factor.setRange(0.0, 1.0)
        self.accel_min_factor.setSingleStep(0.01)
        self.accel_min_factor.setValue(0.05)
        self.accel_min_factor.setToolTip(
            "Minimum multiplier for acceleration loss. 0=exactly zero (not recommended), 0.05=small floor. "
            "Prevents the derivative field from drifting."
        )

        self.a_sign = NoScrollComboBox()
        self.a_sign.addItems(["auto", "+1", "-1"])
        self.a_sign.setCurrentText("auto")
        self.a_sign.setToolTip("Acceleration sign: auto | +1 geodesy | -1 Newton")

        self.use_si = NoScrollComboBox()
        self.use_si.addItems(["SI Units (Recommended)", "Original (No Conversion)"])
        self.use_si.setCurrentIndex(0)
        self.use_si.setToolTip("Canonical → SI conversion.")

        form_phys.addRow("Potential (U) Loss Weight", self.w_u)
        form_phys.addRow("Acceleration (a) Loss Weight", self.w_a)
        form_phys.addRow("GradNorm Mode", self.gradnorm_mode)
        form_phys.addRow("GradNorm w_a Minimum", self.gradnorm_w_a_min)
        form_phys.addRow("GradNorm w_a Maximum", self.gradnorm_w_a_max)
        form_phys.addRow("Potential-Only Epochs", self.potential_only_epochs)
        form_phys.addRow("Acceleration Ramp Epochs", self.accel_ramp_epochs)
        form_phys.addRow("Acceleration Min Factor", self.accel_min_factor)
        form_phys.addRow("Acceleration Sign (a_sign)", self.a_sign)
        form_phys.addRow("Unit System", self.use_si)
        grp_phys.setLayout(form_phys)

        # =====================================================================
        # GROUP 5: Direction Loss (Collapsible)
        # =====================================================================
        self._dir_loss_section = CollapsibleSection("Direction Loss")
        form_dir = QFormLayout()
        _tune_form(form_dir)

        self.direction_loss_weight = QDoubleSpinBox()
        self.direction_loss_weight.setDecimals(4)
        self.direction_loss_weight.setRange(0.0, 10.0)
        self.direction_loss_weight.setValue(0.10)
        self.direction_loss_weight.setSingleStep(0.01)
        self.direction_loss_weight.setToolTip(
            "Direction-loss peak weight λ_dir.\n"
            "L_dir = mean(1 - cos_sim(a_pred, a_true))"
        )
        self.direction_loss_start_epoch = QSpinBox()
        self.direction_loss_start_epoch.setRange(0, 10000)
        self.direction_loss_start_epoch.setValue(30)
        self.direction_loss_start_epoch.setToolTip(
            "Epoch at which the direction loss starts."
        )
        self.direction_loss_ramp_epochs = QSpinBox()
        self.direction_loss_ramp_epochs.setRange(1, 10000)
        self.direction_loss_ramp_epochs.setValue(50)
        self.direction_loss_ramp_epochs.setToolTip(
            "Number of epochs to ramp the direction loss from 0 to full weight."
        )
        self.direction_loss_floor_abs = QDoubleSpinBox()
        self.direction_loss_floor_abs.setDecimals(8)
        self.direction_loss_floor_abs.setRange(0.0, 1.0)
        self.direction_loss_floor_abs.setValue(3e-6)
        self.direction_loss_floor_abs.setSingleStep(1e-7)
        self.direction_loss_floor_abs.setToolTip(
            "||a_true|| threshold. Points below it are removed by the direction-loss mask."
        )

        self.best_ckpt_start_epoch = QSpinBox()
        self.best_ckpt_start_epoch.setRange(-1, 10000)
        self.best_ckpt_start_epoch.setValue(-1)
        self.best_ckpt_start_epoch.setToolTip(
            "Epoch at which best-checkpoint tracking starts (-1=automatic).\n"
            "-1: if direction loss is active, waits until start_epoch + ramp_epochs + settle_epochs.\n"
            "This prevents early epochs (before direction has settled) from becoming the best checkpoint.\n"
            "0: tracks from epoch 0; >0: the first N epochs are skipped as burn-in."
        )
        self.checkpoint_settle_epochs = QSpinBox()
        self.checkpoint_settle_epochs.setRange(0, 10000)
        self.checkpoint_settle_epochs.setValue(5)
        self.checkpoint_settle_epochs.setToolTip(
            "Extra epochs to wait after the direction ramp before automatic best-checkpoint tracking starts."
        )

        form_dir.addRow("Peak Weight (λ)", self.direction_loss_weight)
        form_dir.addRow("Start Epoch", self.direction_loss_start_epoch)
        form_dir.addRow("Ramp Epoch Count", self.direction_loss_ramp_epochs)
        form_dir.addRow("||a|| Threshold (m/s²)", self.direction_loss_floor_abs)
        form_dir.addRow("Best Ckpt Start", self.best_ckpt_start_epoch)
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
        self.use_altitude_balanced_loss.setChecked(True)
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
        self.altitude_min_km.setValue(float(_cfg_value(DEFAULT_SPATIAL_CLOUD_CONFIG, "alt_min_km", 100.0)))
        self.altitude_max_km = QDoubleSpinBox()
        self.altitude_max_km.setDecimals(2)
        self.altitude_max_km.setRange(0.0, 1_000_000.0)
        self.altitude_max_km.setValue(float(_cfg_value(DEFAULT_SPATIAL_CLOUD_CONFIG, "alt_max_km", 1000.0)))

        self.use_radial_cross_loss = QCheckBox("Use radial / cross-radial loss")
        self.use_radial_cross_loss.setChecked(True)
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
        self.use_laplacian_regularization.setChecked(True)
        self.use_laplacian_regularization.setToolTip(
            "Computes second derivatives on a sparse subset. Expensive; keep subset size small."
        )
        lap_warn = QLabel(
            "This computes second derivatives and can be expensive. Keep subset size small."
        )
        lap_warn.setWordWrap(True)
        lap_warn.setStyleSheet(f"color: {THEME['warning']}; font-size: 11px;")
        self.laplacian_weight = QDoubleSpinBox()
        self.laplacian_weight.setDecimals(10)
        self.laplacian_weight.setRange(0.0, 1.0)
        self.laplacian_weight.setValue(2e-9)
        self.laplacian_weight.setSingleStep(1e-5)
        self.laplacian_every_n_batches = QSpinBox()
        self.laplacian_every_n_batches.setRange(1, 100000)
        self.laplacian_every_n_batches.setValue(100)
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
            "Advanced Settings (Hardware & Performance)"
        )
        form_adv = QFormLayout()
        _tune_form(form_adv)

        self.max_grad_norm = QDoubleSpinBox()
        self.max_grad_norm.setDecimals(6)
        self.max_grad_norm.setRange(0.0, 1e6)
        self.max_grad_norm.setValue(0.5)
        self.max_grad_norm.setToolTip("Gradient clipping norm. 0 → no clipping.")
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 64)
        self.num_workers.setValue(2)
        self.num_workers.setToolTip("Number of DataLoader workers.")
        self.cache_rows = QSpinBox()
        self.cache_rows.setRange(1024, 10_000_000)
        self.cache_rows.setValue(65536)
        self.cache_rows.setToolTip("HDF5 RAM cache row count.")
        self.fit_rows = QSpinBox()
        self.fit_rows.setRange(10_000, 50_000_000)
        self.fit_rows.setValue(500_000)
        self.fit_rows.setToolTip("Z-score sample row count.")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(42)
        self.seed.setToolTip("Random seed.")
        self.split_seed = QSpinBox()
        self.split_seed.setRange(0, 2_147_483_647)
        self.split_seed.setValue(42)
        self.split_seed.setToolTip("Separate seed for the train/val split (None → same as seed).")
        self.device_hint = NoScrollComboBox()
        self.device_hint.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_hint.setCurrentText("auto")
        self.device_hint.setToolTip("Device hint. cpu/mps → AMP is disabled automatically.")
        self.device_hint.currentTextChanged.connect(self._on_device_hint_changed)
        self.log_every_mode = NoScrollComboBox()
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
        self.log_every.setToolTip("Write progress every N batches. 0 → disabled. (Used in Fixed mode.)")
        self.preload_data = QCheckBox("Preload to RAM")
        self.preload_data.setChecked(False)
        self.preload_data.setToolTip(
            "Loads small datasets (≤auto_preload_mb) into CPU RAM.\n"
            "Fixes the HDF5 multi-worker issue on Windows."
        )
        self.auto_preload_mb = QDoubleSpinBox()
        self.auto_preload_mb.setDecimals(0)
        self.auto_preload_mb.setRange(0.0, 102400.0)
        self.auto_preload_mb.setValue(2048.0)
        self.auto_preload_mb.setSingleStep(64.0)
        self.auto_preload_mb.setToolTip(
            "Datasets smaller than this MB value are loaded into RAM automatically (0 → disabled)."
        )
        self.pin_memory = QCheckBox("Pin Memory")
        self.pin_memory.setChecked(True)
        self.pin_memory.setToolTip("Pin CPU memory to speed up CUDA transfers.")
        self.quick_check = QCheckBox("Quick Check Mode")
        self.quick_check.setChecked(False)
        self.quick_check.setToolTip(
            "Pipeline validation with 1 epoch, 5 train + 2 val batches. Not real training."
        )

        # PINN Architecture (new)
        self.use_residual_blocks = QCheckBox("Residual SIREN Blocks (SirenResBlock)")
        self.use_residual_blocks.setChecked(True)
        self.use_residual_blocks.setToolTip(
            "Wraps hidden layers with pre-norm + zero-init skip (SirenResBlock).\n"
            "Recommended for depth >= 6. Adds no extra parameters."
        )
        self.n_bands = QSpinBox()
        self.n_bands.setRange(1, 16)
        self.n_bands.setValue(3)
        self.n_bands.setToolTip(
            "Multi-scale SIREN band count. >1 → MultiScaleSirenMLP.\n"
            "1 = standard SirenMLP. Band w0 values are derived automatically from degree_min/max."
        )
        self.grad_accumulation_steps = QSpinBox()
        self.grad_accumulation_steps.setRange(1, 128)
        self.grad_accumulation_steps.setValue(1)
        self.grad_accumulation_steps.setToolTip(
            "Accumulates gradients over N batches before the optimizer step.\n"
            "Effective batch = batch_size × N. For VRAM-constrained cases."
        )
        self.n_hutchinson_samples = QSpinBox()
        self.n_hutchinson_samples.setRange(1, 32)
        self.n_hutchinson_samples.setValue(4)
        self.n_hutchinson_samples.setToolTip(
            "Number of Rademacher samples K for the Hutchinson Laplacian estimate.\n"
            "K=4 → ~50% relative error; used only when Laplacian regularization is active."
        )

        # Performance extras (new)
        self.prefetch_factor = QSpinBox()
        self.prefetch_factor.setSpecialValueText("auto")
        self.prefetch_factor.setRange(0, 32)
        self.prefetch_factor.setValue(0)
        self.prefetch_factor.setToolTip(
            "DataLoader prefetch_factor (0 = automatic; only valid when num_workers > 0)."
        )
        self.max_train_batches = QSpinBox()
        self.max_train_batches.setSpecialValueText("unlimited")
        self.max_train_batches.setRange(0, 1_000_000)
        self.max_train_batches.setValue(0)
        self.max_train_batches.setToolTip(
            "Maximum training batches per epoch. 0 = unlimited (full epoch).\n"
            "Used for quick tests; works together with quick_check."
        )
        self.max_val_batches = QSpinBox()
        self.max_val_batches.setSpecialValueText("unlimited")
        self.max_val_batches.setRange(0, 1_000_000)
        self.max_val_batches.setValue(0)
        self.max_val_batches.setToolTip(
            "Maximum validation batches per epoch. 0 = unlimited."
        )

        form_adv.addRow("Device Hint", self.device_hint)
        form_adv.addRow("Gradient Clipping Norm", self.max_grad_norm)
        form_adv.addRow("DataLoader Worker Count", self.num_workers)
        form_adv.addRow("Prefetch Factor", self.prefetch_factor)
        form_adv.addRow("HDF5 Cache Rows", self.cache_rows)
        form_adv.addRow("Scaler Sample Rows", self.fit_rows)
        form_adv.addRow("Random Seed", self.seed)
        form_adv.addRow("Split Seed", self.split_seed)
        form_adv.addRow("Log Frequency Mode", self.log_every_mode)
        form_adv.addRow("Batch Log Interval", self.log_every)
        form_adv.addRow(self.preload_data)
        form_adv.addRow("Auto-RAM Load Limit (MB)", self.auto_preload_mb)
        form_adv.addRow(self.pin_memory)
        # quick_check / max_train_batches / max_val_batches are dev-only debug
        # controls — kept as hidden widgets (for config/profile compatibility)
        # but intentionally NOT shown in the normal Studio workflow.
        self.quick_check.setVisible(False)
        self.max_train_batches.setVisible(False)
        self.max_val_batches.setVisible(False)
        _adv_sep = QLabel("  PINN Architecture")
        _adv_sep.setStyleSheet(
            f"color: {THEME['accent_hov']}; font-size: 11px; font-weight: 600;"
            " padding: 4px 10px; margin-top: 4px;"
            f" background: {with_alpha(THEME['accent'], 0.08)};"
            f" border-left: 2px solid {with_alpha(THEME['accent'], 0.40)};"
            " border-radius: 0 6px 6px 0;"
        )
        form_adv.addRow(_adv_sep)
        form_adv.addRow(self.use_residual_blocks)
        form_adv.addRow("Frequency Band Count (n_bands)", self.n_bands)
        form_adv.addRow("Grad. Accumulation Steps", self.grad_accumulation_steps)
        form_adv.addRow("Hutchinson Samples (K)", self.n_hutchinson_samples)

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
        self.model_preset = NoScrollComboBox()
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
        self.model_preset_note.setStyleSheet(f"color: {THEME['warning']}; font-size: 11px;")

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

        self._loss_physics_section = CollapsibleSection("Loss / Physics Options")
        loss_physics_wrap = QVBoxLayout()
        loss_physics_wrap.setContentsMargins(0, 0, 0, 0)
        loss_physics_wrap.addWidget(grp_phys)
        self._loss_physics_section.set_content_layout(loss_physics_wrap)

        self._model_repr_section = CollapsibleSection("Manual Encoding / Ablations")
        model_repr_wrap = QVBoxLayout()
        model_repr_wrap.setContentsMargins(0, 0, 0, 0)
        model_repr_wrap.addWidget(grp_model_repr)
        self._model_repr_section.set_content_layout(model_repr_wrap)

        # =====================================================================
        # GROUP: Periodic Evaluation During Training (monitoring only)
        # Optional, collapsed, and disabled by default. Runs the evaluation CLI
        # as a subprocess at selected epochs to watch field-level diagnostics.
        # Never affects optimizer / scheduler / GradNorm / checkpoint selection.
        # =====================================================================
        grp_periodic = QGroupBox("Periodic Evaluation (monitoring)")
        form_periodic = QFormLayout()
        _tune_form(form_periodic)

        self.periodic_eval_enabled = QCheckBox("Enable periodic evaluation")
        self.periodic_eval_enabled.setChecked(False)
        self.periodic_eval_enabled.setToolTip(
            "Run the evaluation pipeline (parity, acceleration/potential/angular metrics) "
            "at selected epochs during training, on ckpt_last. Monitoring only — it does "
            "NOT change training, the optimizer, the scheduler, or checkpoint selection."
        )

        self.periodic_eval_mode = NoScrollComboBox()
        self.periodic_eval_mode.addItem("Count over full training", "count")
        self.periodic_eval_mode.addItem("Every K epochs", "every")
        self.periodic_eval_mode.setCurrentIndex(0)
        self.periodic_eval_mode.setToolTip(
            "Count: spread N evaluations evenly across --epochs (e.g. 10 over 400 -> "
            "40,80,...,400). Every: run every K epochs."
        )

        self.periodic_eval_count = QSpinBox()
        self.periodic_eval_count.setRange(1, 100000)
        self.periodic_eval_count.setValue(10)
        self.periodic_eval_count.setToolTip("How many evaluations to run across the full training horizon.")

        self.periodic_eval_every = QSpinBox()
        self.periodic_eval_every.setRange(1, 100000)
        self.periodic_eval_every.setValue(25)
        self.periodic_eval_every.setToolTip("Run a periodic evaluation every K epochs.")

        self.periodic_eval_dataset = NoScrollComboBox()
        self.periodic_eval_dataset.addItem("val", "val")
        self.periodic_eval_dataset.addItem("test", "test")
        self.periodic_eval_dataset.addItem("ood", "ood")
        self.periodic_eval_dataset.setCurrentIndex(0)
        self.periodic_eval_dataset.setToolTip(
            "Dataset used for monitoring evaluation. val falls back to --data for "
            "single-dataset runs. Missing datasets are skipped with a warning."
        )

        self.periodic_eval_max_samples = QSpinBox()
        self.periodic_eval_max_samples.setRange(1, 1_000_000_000)
        self.periodic_eval_max_samples.setSingleStep(10000)
        self.periodic_eval_max_samples.setValue(200000)
        self.periodic_eval_max_samples.setToolTip(
            "Cap rows evaluated per run to keep monitoring lightweight (default 200000). "
            "Do not run full validation/OOD here unless you intend to."
        )

        self.periodic_eval_batch_size = QSpinBox()
        self.periodic_eval_batch_size.setRange(0, 4_000_000)
        self.periodic_eval_batch_size.setSingleStep(1024)
        self.periodic_eval_batch_size.setValue(0)
        self.periodic_eval_batch_size.setSpecialValueText("auto (training batch size)")
        self.periodic_eval_batch_size.setToolTip("Evaluation batch size. 0 = reuse the training batch size.")

        self.periodic_eval_device = NoScrollComboBox()
        self.periodic_eval_device.addItems(["auto", "cpu", "cuda", "mps"])
        self.periodic_eval_device.setCurrentText("auto")
        self.periodic_eval_device.setToolTip("Device for the evaluation subprocess.")

        self.periodic_eval_continue_on_fail = QCheckBox("Continue training if an evaluation fails")
        self.periodic_eval_continue_on_fail.setChecked(True)
        self.periodic_eval_continue_on_fail.setToolTip(
            "When checked (default), a failed periodic evaluation is logged and recorded but "
            "does not stop training. Uncheck to abort training on evaluation failure."
        )

        periodic_help = QLabel(
            "Outputs: <run_dir>/periodic_evals/epoch_XXXX/  (history: periodic_eval_history.jsonl). "
            "Monitoring only — does not affect weights, optimizer, scheduler, GradNorm, RNG, or "
            "best-checkpoint selection. On resume, already-completed evaluations are skipped."
        )
        periodic_help.setWordWrap(True)
        periodic_help.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 11px;")

        form_periodic.addRow(self.periodic_eval_enabled)
        form_periodic.addRow("Mode", self.periodic_eval_mode)
        form_periodic.addRow("Count (N)", self.periodic_eval_count)
        form_periodic.addRow("Every (K epochs)", self.periodic_eval_every)
        form_periodic.addRow("Dataset", self.periodic_eval_dataset)
        form_periodic.addRow("Max samples", self.periodic_eval_max_samples)
        form_periodic.addRow("Batch size", self.periodic_eval_batch_size)
        form_periodic.addRow("Device", self.periodic_eval_device)
        form_periodic.addRow(self.periodic_eval_continue_on_fail)
        form_periodic.addRow("", periodic_help)
        grp_periodic.setLayout(form_periodic)
        self._grp_periodic = grp_periodic

        self._periodic_eval_section = CollapsibleSection("Periodic Evaluation")
        periodic_wrap = QVBoxLayout()
        periodic_wrap.setContentsMargins(0, 0, 0, 0)
        periodic_wrap.addWidget(grp_periodic)
        self._periodic_eval_section.set_content_layout(periodic_wrap)

        # Enable-state + command preview wiring.
        self.periodic_eval_enabled.toggled.connect(self._on_periodic_eval_toggled)
        self.periodic_eval_mode.currentIndexChanged.connect(self._on_periodic_eval_toggled)
        self.periodic_eval_enabled.toggled.connect(self._refresh_command_preview)
        self.periodic_eval_mode.currentIndexChanged.connect(self._refresh_command_preview)
        self.periodic_eval_continue_on_fail.toggled.connect(self._refresh_command_preview)
        for _sp in (
            self.periodic_eval_count, self.periodic_eval_every,
            self.periodic_eval_max_samples, self.periodic_eval_batch_size,
        ):
            _sp.valueChanged.connect(self._refresh_command_preview)
        self.periodic_eval_dataset.currentIndexChanged.connect(self._refresh_command_preview)
        self.periodic_eval_device.currentIndexChanged.connect(self._refresh_command_preview)
        self._on_periodic_eval_toggled()

        # =====================================================================
        # EXTRA CLI ARGS
        # =====================================================================
        self.extra_args = QLineEdit("")
        self.extra_args.setPlaceholderText("Extra CLI arguments (optional)")
        self.extra_args.setToolTip(f"Extra arguments passed directly to the python -m {TRAIN_CLI_MODULE} command.")

        # Command preview controls are mounted later in the visible command
        # section. Keep them unparented here so Qt does not silently move them
        # between an obsolete layout and the current setup workspace.
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setFont(_mono_font())
        self.command_preview.setMinimumHeight(78)
        self.command_preview.setPlaceholderText(
            f"Click Preview Command to see the exact python -m {TRAIN_CLI_MODULE} command."
        )
        self.command_warning = QLabel("")
        self.command_warning.setWordWrap(True)
        self.command_warning.setStyleSheet(f"color: {THEME['warning']}; font-size: 11px;")

        for grp in (grp_data, grp_arch, grp_optim, grp_phys):
            _tune_inputs(grp)
        _tune_inputs(self.resume_section)
        _tune_inputs(self._loss_physics_section)
        _tune_inputs(self._model_repr_section)
        _tune_inputs(self._dir_loss_section)
        _tune_inputs(self._field_loss_section)
        _tune_inputs(self._periodic_eval_section)

        # --- ProcessPane ---
        self.runner = ProcessPane()
        self.runner.btn_start.setText("Start Training")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_progress_parser(self._parse_progress)
        self.runner.set_finished_hook(self._on_train_finished)
        self._user_stopped = False  # set when the user clicks Stop (→ INTERRUPTED)
        self._device_badge: str | None = None
        # The experiment header lives on the MainWindow. Because this tab's
        # setup_page/monitor_page are re-homed into the page stack (this tab
        # itself is not in the widget tree), self.window() does NOT reach it —
        # MainWindow injects a direct reference via set_experiment_header().
        self._hdr_ref = None
        self._train_proc = None

        # --- Live Loss Plot (Feature #13) ---
        self._live_plot = LiveLossPlot()
        self._live_plot.epoch_parsed.connect(self._update_window_title_progress)
        self._history_poll_timer = QTimer(self)
        self._history_poll_timer.setInterval(2000)
        self._history_poll_timer.timeout.connect(self._poll_training_history)
        self._history_poll_path: Path | None = None
        self._history_poll_mtime: float = 0.0
        self._provenance_panel = RunProvenancePanel()
        self._periodic_eval_table = PeriodicEvalTable()
        self._monitor_artifact_timer = QTimer(self)
        self._monitor_artifact_timer.setInterval(2500)
        self._monitor_artifact_timer.timeout.connect(self._refresh_monitor_artifacts)

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
        # Separate Setup buttons to fix duplicate parenting
        self.btn_enqueue_setup = QPushButton("Add to Queue")
        self.btn_enqueue_setup.setToolTip("Adds the current settings to the training queue.")
        self.btn_enqueue_setup.setProperty("kind", "ghost")
        self.btn_enqueue_setup.clicked.connect(self._enqueue_current)

        self.btn_start_setup = QPushButton("Start Training")
        self.btn_start_setup.setToolTip(
            "Start the current training workflow and open the Training Monitor page."
        )
        self.btn_start_setup.setProperty("kind", "primary")
        self.btn_start_setup.clicked.connect(self._start)

        self.btn_preview_cmd_setup = QPushButton("Preview Command")
        self.btn_preview_cmd_setup.setProperty("kind", "ghost")
        self.btn_preview_cmd_setup.clicked.connect(self._preview_command_popup)

        self.btn_copy_cmd_setup = QPushButton("Copy Command")
        self.btn_copy_cmd_setup.setProperty("kind", "ghost")
        self.btn_copy_cmd_setup.clicked.connect(self._copy_command_preview)

        # --- Training Queue (Feature #15) ---
        self._queue = TrainingQueue()
        self._queue.job_started.connect(self._on_queue_job_started)

        # =====================================================================
        # LAYOUT ASSEMBLY RE-WRITE (Phases 1-10)
        # =====================================================================

        # ── 1. Saved Profiles — prominent, always-visible card ──
        # Profiles are the fastest path to a configured run, so the card sits
        # at the top of the page (it was previously collapsed and buried at the
        # very bottom of the parameter grid, where it was easy to miss).
        saved_profiles_card = QFrame()
        saved_profiles_card.setObjectName("savedProfilesCard")
        _style_surface(saved_profiles_card, object_name="savedProfilesCard")
        saved_profiles_l = QVBoxLayout()
        saved_profiles_l.setContentsMargins(16, 12, 16, 12)
        saved_profiles_l.setSpacing(8)
        _profiles_heading = QLabel("Saved Profiles")
        _profiles_heading.setStyleSheet(
            f"font-weight: 700; color: {THEME['fg_main']}; font-size: 13px;"
        )
        _profiles_hint = QLabel("Load a saved hyperparameter set, or save the current configuration.")
        _profiles_hint.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 11px;")
        saved_profiles_l.addWidget(_profiles_heading)
        saved_profiles_l.addLayout(preset_bar)
        saved_profiles_l.addWidget(_profiles_hint)
        saved_profiles_card.setLayout(saved_profiles_l)

        # ── 2. Add Model Preset to grp_arch ──
        form_arch.insertRow(0, "Model Preset", self.model_preset)
        grp_data.setTitle("Dataset and Output")

        # ── 3. Horizontal Launch Plan Strip (Phase 1) ──
        launch_strip = QFrame()
        launch_strip.setObjectName("setupLaunchStrip")
        _style_surface(launch_strip, object_name="setupLaunchStrip")
        launch_l = QHBoxLayout()
        launch_l.setContentsMargins(16, 12, 16, 12)
        launch_l.setSpacing(16)

        launch_l.addLayout(workflow_bar)

        output_mode_box = QHBoxLayout()
        output_mode_box.setSpacing(6)
        out_mode_lbl = QLabel("Output:")
        out_mode_lbl.setStyleSheet(
            f"font-weight: 600; color: {THEME['fg_muted']}; font-size: 13px;"
        )
        self._output_mode_short = QLabel("auto")
        self._output_mode_short.setStyleSheet(f"color: {THEME['fg_soft']}; font-size: 13px;")
        output_mode_box.addWidget(out_mode_lbl)
        output_mode_box.addWidget(self._output_mode_short)

        launch_l.addLayout(output_mode_box)
        launch_l.addWidget(self._launch_badge)
        launch_l.addWidget(self._launch_summary, 1)
        launch_l.addStretch(1)
        launch_l.addWidget(self.btn_enqueue_setup)
        launch_l.addWidget(self.btn_start_setup)
        launch_strip.setLayout(launch_l)

        # Resolved values are visible before launch; the panel is fed from the
        # current widgets and the engine's resolver in _refresh_launch_plan().
        launch_plan_card = QFrame()
        _style_surface(launch_plan_card, object_name="launchPlanCard")
        launch_plan_l = QVBoxLayout(launch_plan_card)
        launch_plan_l.setContentsMargins(16, 12, 16, 12)
        launch_plan_header = QHBoxLayout()
        launch_plan_title = QLabel("Launch Plan")
        launch_plan_title.setObjectName("panelTitle")
        launch_plan_header.addWidget(launch_plan_title)
        self._deviation_label = QLabel("0 deviations")
        self._deviation_label.setObjectName("fieldHint")
        launch_plan_header.addWidget(self._deviation_label)
        launch_plan_header.addStretch(1)
        reset_defaults = QPushButton("Reset deviations")
        reset_defaults.setProperty("kind", "ghost")
        reset_defaults.clicked.connect(self._reset_to_defaults)
        launch_plan_header.addWidget(reset_defaults)
        launch_plan_l.addLayout(launch_plan_header)
        self._launch_plan_values = KeyValueList()
        launch_plan_l.addWidget(self._launch_plan_values)

        # ── 4. Main Configuration Workspace (Phase 2 & 5) ──
        workspace_grid = QGridLayout()
        workspace_grid.setContentsMargins(0, 0, 0, 0)
        workspace_grid.setSpacing(16)
        workspace_grid.setColumnStretch(0, 1)

        # Resume / continue-from-checkpoint sits at the very top of the config
        # so it is the first thing visible (it was previously attached to an
        # unused layout and never displayed at all).
        workspace_grid.addWidget(self.resume_section, 0, 0)
        workspace_grid.addWidget(grp_data, 1, 0)
        workspace_grid.addWidget(grp_arch, 2, 0)
        workspace_grid.addWidget(grp_optim, 3, 0)
        workspace_grid.addWidget(self._loss_physics_section, 4, 0)
        workspace_grid.addWidget(self._fourier_section, 5, 0)
        workspace_grid.addWidget(self._dir_loss_section, 6, 0)
        workspace_grid.addWidget(self._field_loss_section, 7, 0)
        workspace_grid.addWidget(self.advanced_section, 8, 0)
        workspace_grid.addWidget(self._model_repr_section, 9, 0)
        workspace_grid.addWidget(self._periodic_eval_section, 10, 0)

        # ── 5. Command & Launch collapsible section ──
        cmd_section = CollapsibleSection("Command Preview & CLI Arguments")
        cmd_l = QVBoxLayout()
        cmd_l.setContentsMargins(0, 0, 0, 0)
        cmd_l.setSpacing(8)

        cmd_header = QHBoxLayout()
        cmd_header.addWidget(self.btn_preview_cmd_setup)
        cmd_header.addWidget(self.btn_copy_cmd_setup)
        cmd_header.addStretch(1)

        _style_command_preview(self.command_preview, min_h=72, max_h=96)

        extra_row_layout = QHBoxLayout()
        extra_lbl = QLabel("Extra CLI Arguments:")
        extra_lbl.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 12px;")
        extra_row_layout.addWidget(extra_lbl)
        extra_row_layout.addWidget(self.extra_args, 1)

        cmd_l.addLayout(extra_row_layout)
        cmd_l.addWidget(self.command_preview)
        cmd_l.addWidget(self.command_warning)
        cmd_l.addLayout(cmd_header)

        cmd_inner = QWidget()
        cmd_inner.setLayout(cmd_l)
        cmd_vbox = QVBoxLayout()
        cmd_vbox.setContentsMargins(0,0,0,0)
        cmd_vbox.addWidget(cmd_inner)
        cmd_section.set_content_layout(cmd_vbox)

        # Command preview stays at the bottom of the grid; Saved Profiles now
        # lives in its own prominent card above the launch strip.
        workspace_grid.addWidget(cmd_section, 11, 0)

        workspace_inner = QWidget()
        workspace_inner.setLayout(workspace_grid)
        params_page = _scroll_wrap(workspace_inner)

        # ── 6. Setup Page Assembly ──
        self.setup_page = QWidget()
        setup_l = QVBoxLayout()
        setup_l.setContentsMargins(22, 20, 22, 20)
        setup_l.setSpacing(16)

        setup_l.addWidget(_make_page_header(
            "Training Setup",
            "Shape the dataset, architecture, optimization, resume behavior, and launch plan before starting a long run.",
            "Experiment Design",
        ))
        setup_l.addWidget(saved_profiles_card)
        setup_l.addWidget(launch_strip)
        setup_l.addWidget(launch_plan_card)
        self._setup_search = CompactSearchField("Search setup fields")
        self._setup_search.setToolTip("Filter configuration groups by name or field label. Press Escape to clear.")
        self._setup_search.textChanged.connect(self._filter_setup_sections)
        setup_l.addWidget(self._setup_search)
        setup_l.addWidget(params_page, 1)

        self._setup_sections = [
            self.resume_section, grp_data, grp_arch, grp_optim,
            self._loss_physics_section, self._fourier_section,
            self._dir_loss_section, self._field_loss_section,
            self.advanced_section, self._model_repr_section,
            self._periodic_eval_section, cmd_section,
        ]

        self.setup_page.setLayout(setup_l)

        # ── 7. Training Monitor (single scrollable column — no nested
        #       splitters, so the History/Progress/Raw table can no longer be
        #       clipped behind the chart card). ──
        train_ctrl_bar = self._build_training_control_bar()

        if _HAS_DASHBOARD_V2:
            self._live_plot.set_compact(True)
            if self._structured_log is not None:
                self._structured_log.set_raw_log_widget(self.runner.raw_log_widget())

        # Compact status strips.
        if _HAS_DASHBOARD_V2 and self._kpi_strip is not None:
            self._kpi_strip.setMaximumHeight(176)
            self._kpi_strip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if _HAS_DASHBOARD_V2 and self._time_strip is not None:
            self._time_strip.setMaximumHeight(176)
            self._time_strip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Chart card + structured log get generous fixed heights in the column.
        self._live_plot.setMinimumHeight(440)
        self._live_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if _HAS_DASHBOARD_V2 and self._structured_log is not None:
            self._structured_log.setMinimumHeight(280)
            self._structured_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Queue (collapsible, secondary).
        queue_box = QGroupBox("Queue Status")
        queue_box_l = QVBoxLayout()
        queue_box_l.setContentsMargins(10, 10, 10, 10)
        queue_box_l.addWidget(self._queue, 1)
        queue_box.setLayout(queue_box_l)
        queue_collapsible = CollapsibleSection("Execution Queue Status")
        queue_collapsible_l = QVBoxLayout()
        queue_collapsible_l.setContentsMargins(0, 0, 0, 0)
        queue_collapsible_l.addWidget(queue_box)
        queue_collapsible.set_content_layout(queue_collapsible_l)

        # Fixed monitor workspace: metrics/plots on the left, process console on
        # the right. This avoids the old bottom-opening terminal feel and keeps
        # logs visible without stealing vertical space from the chart.
        monitor_content = QWidget()
        monitor_layout = QVBoxLayout(monitor_content)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        monitor_layout.setSpacing(12)
        monitor_layout.addWidget(train_ctrl_bar)

        # Phase UX-1: Finish notice & actions panel
        self._finish_panel = QWidget()
        finish_l = QVBoxLayout(self._finish_panel)
        finish_l.setContentsMargins(0, 0, 0, 0)
        finish_l.setSpacing(8)
        self._finish_notice = StudioNotice("Run Finished", "", kind="info")
        finish_actions_l = QHBoxLayout()
        self._btn_finish_eval = QPushButton("Evaluate Model")
        self._btn_finish_eval.setProperty("kind", "primary")
        self._btn_finish_folder = QPushButton("Open Folder")
        self._btn_finish_report = QPushButton("View Report")
        self._btn_finish_queue = QPushButton("Queue Next")
        for btn in (self._btn_finish_eval, self._btn_finish_folder, self._btn_finish_report, self._btn_finish_queue):
            finish_actions_l.addWidget(btn)
        finish_actions_l.addStretch()
        finish_l.addWidget(self._finish_notice)
        finish_l.addLayout(finish_actions_l)
        self._finish_panel.setVisible(False)
        monitor_layout.addWidget(self._finish_panel)

        self._btn_finish_eval.clicked.connect(lambda: self.evaluate_requested.emit(self.runner._output_dir))
        self._btn_finish_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.runner._output_dir)))
        self._btn_finish_report.clicked.connect(self._open_finish_report)
        self._btn_finish_queue.clicked.connect(lambda: self._scroll_area.ensureWidgetVisible(self._queue))

        if _HAS_DASHBOARD_V2 and self._kpi_strip is not None:
            monitor_layout.addWidget(self._kpi_strip)
        if _HAS_DASHBOARD_V2 and self._time_strip is not None:
            monitor_layout.addWidget(self._time_strip)
        monitor_layout.addWidget(self._provenance_panel)
        monitor_layout.addWidget(self._periodic_eval_table)
        monitor_layout.addWidget(self._live_plot)
        monitor_layout.addWidget(queue_collapsible)
        monitor_layout.addStretch(0)

        console_panel = QFrame()
        _style_surface(console_panel, object_name="trainingConsolePanel")
        console_panel.setMinimumWidth(360)
        console_l = QVBoxLayout(console_panel)
        console_l.setContentsMargins(12, 10, 12, 12)
        console_l.setSpacing(8)

        console_title = QLabel("Run Console")
        console_title.setObjectName("sectionTitle")
        console_hint = QLabel("Process output, parsed events, warnings, and raw log context stay fixed here.")
        console_hint.setObjectName("pageDescription")
        console_hint.setWordWrap(True)
        console_l.addWidget(console_title)
        console_l.addWidget(console_hint)
        if _HAS_DASHBOARD_V2 and self._structured_log is not None:
            console_l.addWidget(self._structured_log, 1)
        else:
            console_l.addWidget(self.runner, 1)

        monitor_splitter = QSplitter(Qt.Orientation.Horizontal)
        monitor_splitter.addWidget(_scroll_wrap(monitor_content))
        monitor_splitter.addWidget(console_panel)
        monitor_splitter.setStretchFactor(0, 3)
        monitor_splitter.setStretchFactor(1, 2)
        monitor_splitter.setSizes([880, 420])

        self.monitor_page = QWidget()
        monitor_shell_l = QVBoxLayout(self.monitor_page)
        monitor_shell_l.setContentsMargins(18, 16, 18, 16)
        monitor_shell_l.setSpacing(12)
        monitor_shell_l.addWidget(_make_page_header(
            "Training Monitor",
            "Track lifecycle, loss curves, phase timing, checkpoints, and process output while the experiment runs.",
            "Live Experiment",
        ))
        monitor_shell_l.addWidget(monitor_splitter, 1)

        # ── 9. Final Setup ──
        self._page_tabs = None
        self._params_tab_idx = 0
        self._monitor_tab_idx = 0
        self._queue_tab_idx = 0

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        layout.addWidget(self.setup_page, 1)

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
        self._connect_launch_plan_signals()
        self._refresh_launch_plan()

    def _connect_launch_plan_signals(self) -> None:
        """Keep the resolved launch summary and deviation count live."""
        for cls in (QLineEdit, QCheckBox):
            for widget in self.findChildren(cls):
                signal = widget.textChanged if cls is QLineEdit else widget.toggled
                signal.connect(lambda *_: self._refresh_launch_plan())
        for cls in (QSpinBox, QDoubleSpinBox):
            for widget in self.findChildren(cls):
                widget.valueChanged.connect(lambda *_: self._refresh_launch_plan())

    def _filter_setup_sections(self, query: str) -> None:
        query = query.strip().casefold()
        for section in getattr(self, "_setup_sections", []):
            title = getattr(section, "_title", "")
            if hasattr(section, "title"):
                title = f"{title} {section.title()}"
            labels = " ".join(label.text() for label in section.findChildren(QLabel))
            section.setVisible(not query or query in f"{title} {labels}".casefold())

    def _reset_to_defaults(self) -> None:
        self._apply_config(_base_preset())
        self._refresh_command_preview()
        self._refresh_checklist()
        self._refresh_launch_plan()

    def _refresh_launch_plan(self) -> None:
        if not hasattr(self, "_launch_plan_values"):
            return
        cfg = self._collect_config()
        defaults = _base_preset()
        deviations = 0
        for key, default in defaults.items():
            if key not in cfg:
                continue
            if str(cfg[key]) != str(default):
                deviations += 1
        self._deviation_label.setText(f"{deviations} deviation{'s' if deviations != 1 else ''} from defaults")
        direction_ready = int(cfg.get("direction_loss_start_epoch", 0)) + int(cfg.get("direction_loss_ramp_epochs", 0)) + int(cfg.get("checkpoint_settle_epochs", 5))
        best_start = int(cfg.get("best_ckpt_start_epoch", -1))
        if resolve_best_ckpt_start_epoch is not None:
            try:
                best_start = resolve_best_ckpt_start_epoch(SimpleNamespace(**cfg)).start_epoch
            except Exception:
                if best_start < 0:
                    best_start = direction_ready if float(cfg.get("direction_loss_weight", 0.0)) > 0.0 else 0
        elif best_start < 0:
            best_start = direction_ready if float(cfg.get("direction_loss_weight", 0.0)) > 0.0 else 0
        self._replace_key_values(
            [
                ("Workflow", str(self.workflow_mode.currentText())),
                ("Effective batch", f"{int(cfg.get('batch_size', 0)) * int(cfg.get('grad_accumulation_steps', 1)):,} samples"),
                ("Best checkpoint from", f"epoch {best_start}"),
                ("Direction-ready epoch", f"epoch {direction_ready}" if float(cfg.get("direction_loss_weight", 0.0)) > 0.0 else "disabled"),
                ("Model preset", str(cfg.get("model_preset", "custom"))),
                ("Device", str(self.device_hint.currentText())),
                ("Workers", str(cfg.get("num_workers", "—"))),
                ("Periodic eval", "enabled" if bool(cfg.get("periodic_eval_enabled")) else "disabled"),
            ]
        )

    def _replace_key_values(self, items: list[tuple[str, str]]) -> None:
        while self._launch_plan_values.layout_grid.count():
            item = self._launch_plan_values.layout_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for key, value in items:
            self._launch_plan_values.add_item(key, value)

    def _open_finish_report(self) -> None:
        run_dir = Path(self.runner._output_dir or self.out_dir.text().strip())
        for candidate in (run_dir / "report.md", run_dir / "reports" / "report.md", run_dir / "evals" / "report.md"):
            if candidate.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(candidate)))
                return

    def trigger_resume_from(self, path: str) -> None:
        """Pre-fill and enable resume from a selected Runs record."""
        self.resume_section.set_expanded(True)
        self.resume_from.setText(path)
        self.resume_enabled.setChecked(True)

    def focus_log(self) -> None:
        """Ctrl+L target: focus the parsed log when it exists."""
        target = self._structured_log if self._structured_log is not None else self.runner.raw_log_widget()
        if target is not None:
            target.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.navigate_monitor_requested.emit()

    def start_from_shortcut(self) -> None:
        """Ctrl+Enter starts through the same validation path as the button."""
        if getattr(self, "btn_start_setup", None) is not None and self.btn_start_setup.isEnabled():
            self._start()

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

    def _on_periodic_eval_toggled(self, *_args) -> None:
        """Enable/disable periodic-eval controls based on the enable checkbox and mode."""
        enabled = self.periodic_eval_enabled.isChecked()
        mode = self.periodic_eval_mode.currentData() or "count"
        for widget in (
            self.periodic_eval_mode,
            self.periodic_eval_count,
            self.periodic_eval_every,
            self.periodic_eval_dataset,
            self.periodic_eval_max_samples,
            self.periodic_eval_batch_size,
            self.periodic_eval_device,
            self.periodic_eval_continue_on_fail,
        ):
            widget.setEnabled(enabled)
        # Only the active mode's spin box is editable.
        self.periodic_eval_count.setEnabled(enabled and mode == "count")
        self.periodic_eval_every.setEnabled(enabled and mode == "every")

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
            f"QFrame#trainRunBar {{"
            f"  background: {with_alpha(THEME['bg_card'], 0.88)};"
            f"  border: 1px solid {with_alpha(THEME['accent'], 0.22)};"
            f"  border-radius: 10px;"
            f"}}"
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

        # Ghost / secondary actions that drive the same slots (Monitor page specific instances).
        self.btn_clear_log_monitor = QPushButton("Clear Log")
        self.btn_clear_log_monitor.setProperty("kind", "ghost")
        self.btn_clear_log_monitor.clicked.connect(self._clear_logs)

        self.btn_open_run_monitor = QPushButton("Open Run Folder")
        self.btn_open_run_monitor.setProperty("kind", "ghost")
        self.btn_open_run_monitor.clicked.connect(self._open_run_folder)

        self.btn_preview_cmd_monitor = QPushButton("Preview Command")
        self.btn_preview_cmd_monitor.setProperty("kind", "ghost")
        self.btn_preview_cmd_monitor.clicked.connect(self._preview_command_popup)

        self.btn_copy_cmd_monitor = QPushButton("Copy Command")
        self.btn_copy_cmd_monitor.setProperty("kind", "ghost")
        self.btn_copy_cmd_monitor.clicked.connect(self._copy_command_preview)

        self.btn_enqueue_monitor = QPushButton("Add to Queue")
        self.btn_enqueue_monitor.setToolTip("Adds the current settings to the training queue.")
        self.btn_enqueue_monitor.setProperty("kind", "ghost")
        self.btn_enqueue_monitor.clicked.connect(self._enqueue_current)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.addWidget(self.runner.btn_start)      # primary
        top_row.addWidget(self.runner.btn_stop)       # danger
        top_row.addWidget(self.runner.progress, 1)    # compact, expanding

        secondary_row = QGridLayout()
        secondary_row.setContentsMargins(0, 0, 0, 0)
        secondary_row.setSpacing(8)
        for index, button in enumerate(
            (
                self.btn_enqueue_monitor,
                self.btn_clear_log_monitor,
                self.btn_open_run_monitor,
                self.btn_preview_cmd_monitor,
                self.btn_copy_cmd_monitor,
            )
        ):
            row, column = divmod(index, 3)
            secondary_row.addWidget(button, row, column)
        secondary_row.setColumnStretch(3, 1)

        # Run/output info row.
        self._run_dir_label = QLabel(f"Output: {TRAINING_OUTPUT_ROOT}/st_lrps_train_<timestamp>")
        self._run_dir_label.setStyleSheet(
            f"color: {THEME['fg_muted']}; font-size: 11px;"
        )
        self._run_dir_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._run_dir_label.setWordWrap(True)
        self._run_dir_label.setMinimumWidth(0)
        self._run_dir_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._workflow_label = QLabel("")
        self._workflow_label.setStyleSheet(
            f"color: {THEME['accent']}; font-size: 11px; font-weight: 600;"
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
        lo.addLayout(secondary_row)
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
            "train_only":      "Start Training",
            "eval_only":       "Start Evaluation",
            "train_then_eval": "Train then Evaluate",
            "queue":           "Start Queue",
        }
        # Update start button label
        if hasattr(self, "runner"):
            self.runner.btn_start.setText(labels.get(mode, "Start"))
        if hasattr(self, "btn_start_setup"):
            self.btn_start_setup.setText(labels.get(mode, "Start Training"))
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
                f"[UI] Cloud config synced → altitude: {alt_min:.0f}–{alt_max:.0f} km  |  "
                f"degree: {deg_min}→{deg_max}"
            )

    def _refresh_checklist(self) -> None:
        """Build a compact readiness checklist and update the label."""
        items = []
        readiness_items: list[tuple[str, str]] = []
        ok = True

        script_train = TRAIN_CLI_PATH
        script_eval  = EVAL_CLI_PATH
        mode = self.workflow_mode.currentData() if hasattr(self, "workflow_mode") else "train_then_eval"

        def check(cond: bool, text: str, hard: bool = True, fail_text: str | None = None) -> None:
            nonlocal ok
            marker = "OK" if cond else ("BLOCK" if hard else "WARN")
            color = THEME["success"] if cond else (THEME["error"] if hard else THEME["warning"])
            shown_text = text if cond else (fail_text or text)
            items.append(f'<span style="color:{color}">{marker}: {shown_text}</span>')
            readiness_items.append(("success" if cond else ("error" if hard else "warning"), shown_text))
            if not cond and hard:
                ok = False

        if mode in ("train_only", "train_then_eval", "queue"):
            check(
                script_train.exists(),
                "Training CLI found",
                fail_text=f"Training CLI is missing: {script_train}",
            )
        if mode in ("eval_only", "train_then_eval"):
            check(
                script_eval.exists(),
                "Evaluation CLI found",
                fail_text=f"Evaluation CLI is missing: {script_eval}",
            )

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
                readiness_items.append(("info", "Resume mode: dataset/output may be inferred from previous run config"))
                items.append(f'<span style="color:{THEME["info"]}">INFO: Resume mode: dataset/output may be inferred from previous run config</span>')
            elif dataset_mode == "single":
                dp = self.data.text().strip() if hasattr(self, "data") else ""
                if dp:
                    check(Path(dp).is_file(), f"Dataset exists: {Path(dp).name}")
                else:
                    readiness_items.append(("info", "No dataset path: latest HDF5 will be auto-discovered"))
                    items.append(f'<span style="color:{THEME["info"]}">INFO: No dataset path: latest HDF5 will be auto-discovered</span>')
            else:
                def check_dataset(label: str, path_text: str, *, required: bool) -> None:
                    if path_text:
                        check(
                            Path(path_text).is_file(),
                            f"{label} found: {Path(path_text).name}",
                            hard=True,
                            fail_text=f"{label} path not found: {path_text}",
                        )
                    elif required:
                        check(
                            False,
                            f"{label} selected",
                            hard=True,
                            fail_text=f"{label} missing - select a Dataset Suite Folder or choose the file.",
                        )
                    else:
                        readiness_items.append(("info", f"{label} not selected (optional)."))
                        items.append(
                            f'<span style="color:{THEME["info"]}">INFO: {label} not selected (optional).</span>'
                        )

                tp = self.train_data.text().strip() if hasattr(self, "train_data") else ""
                vp = self.val_data.text().strip() if hasattr(self, "val_data") else ""
                xp = self.test_data.text().strip() if hasattr(self, "test_data") else ""
                op = self.ood_data.text().strip() if hasattr(self, "ood_data") else ""
                check_dataset("Train dataset", tp, required=True)
                check_dataset("Validation dataset", vp, required=True)
                check_dataset("Test dataset", xp, required=False)
                check_dataset("OOD dataset", op, required=False)

        if mode in ("eval_only",):
            md = self.out_dir.text().strip() if hasattr(self, "out_dir") else ""
            check(
                bool(md) and Path(md).is_dir(),
                "Model directory exists",
                hard=True,
                fail_text="Model directory missing - choose an existing run/output folder for evaluation.",
            )

        if not hasattr(self, "_checklist_label"):
            return

        if hasattr(self, "_launch_badge"):
            blocking = sum(1 for kind, _text in readiness_items if kind in {"error", "blocked"})
            warnings = sum(1 for kind, _text in readiness_items if kind == "warning")
            if blocking:
                self._launch_badge.set_status("error", f"{blocking} blocking")
            elif warnings:
                self._launch_badge.set_status("warning", f"{warnings} warning")
            else:
                self._launch_badge.set_status("success", "Ready")
        blocking_items = [text for kind, text in readiness_items if kind in {"error", "blocked"}]
        warning_items = [text for kind, text in readiness_items if kind == "warning"]
        info_items = [text for kind, text in readiness_items if kind == "info"]
        ok_items = [text for kind, text in readiness_items if kind == "success"]

        def compact(prefix: str, values: list[str]) -> str:
            shown = values[:2]
            suffix = f" (+{len(values) - len(shown)} more)" if len(values) > len(shown) else ""
            return f"{prefix}: " + " | ".join(shown) + suffix

        if hasattr(self, "_launch_summary"):
            if blocking_items:
                summary = compact("Fix before launch", blocking_items)
            elif warning_items:
                summary = compact("Review warning", warning_items)
            elif ok_items:
                summary = compact("Ready", ok_items[:3])
            elif info_items:
                summary = compact("Info", info_items)
            else:
                summary = "Ready to review parameters."
            self._launch_summary.setText(summary)
            tooltip_lines = []
            for heading, values in (
                ("Blocking", blocking_items),
                ("Warnings", warning_items),
                ("Info", info_items),
                ("Passed", ok_items),
            ):
                if values:
                    tooltip_lines.append(f"{heading}:")
                    tooltip_lines.extend(f"  - {value}" for value in values)
            tooltip = "\n".join(tooltip_lines)
            self._launch_summary.setToolTip(tooltip)
            if hasattr(self, "_launch_badge"):
                self._launch_badge.setToolTip(tooltip)

        if not items:
            self._checklist_label.setVisible(False)
            return

        self._checklist_label.setText("  ".join(items))
        self._checklist_label.setVisible(False)

        # Enable/disable start button based on hard requirements
        if hasattr(self, "runner"):
            self.runner.btn_start.setEnabled(ok)
            if not ok and blocking_items:
                self.runner.btn_start.setToolTip(compact("Cannot start", blocking_items))
            else:
                self.runner.btn_start.setToolTip("Start the current training workflow.")
        if hasattr(self, "btn_start_setup"):
            self.btn_start_setup.setEnabled(ok)
            if not ok and blocking_items:
                self.btn_start_setup.setToolTip(compact("Cannot start", blocking_items))
            else:
                self.btn_start_setup.setToolTip(
                    "Start the current training workflow and open the Training Monitor page."
                )

    # -----------------------------------------------------------------
    # Preset System
    # -----------------------------------------------------------------
    def _refresh_preset_list(self) -> None:
        self._preset_combo.clear()
        for name in _BUILTIN_PRESETS:
            if str(name).lower().startswith("quick"):
                continue
            self._preset_combo.addItem(f"⚙  {name}", name)
        user = _load_user_presets()
        if user:
            self._preset_combo.insertSeparator(self._preset_combo.count())
            for name in user:
                self._preset_combo.addItem(f"👤  {name}", name)

    def _current_preset_key(self) -> str:
        return self._preset_combo.currentData() or ""

    def _collect_config(self) -> dict[str, Any]:
        return {
            # Dataset routing
            "dataset_mode": self.dataset_mode.currentData() or "single",
            "dataset_suite_dir": self.dataset_suite_dir.text(),
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
            # Periodic evaluation (monitoring only)
            "periodic_eval_enabled": self.periodic_eval_enabled.isChecked(),
            "periodic_eval_mode": self.periodic_eval_mode.currentData() or "count",
            "periodic_eval_count": self.periodic_eval_count.value(),
            "periodic_eval_every": self.periodic_eval_every.value(),
            "periodic_eval_dataset": self.periodic_eval_dataset.currentData() or "val",
            "periodic_eval_max_samples": self.periodic_eval_max_samples.value(),
            "periodic_eval_batch_size": self.periodic_eval_batch_size.value(),
            "periodic_eval_device": self.periodic_eval_device.currentText(),
            "periodic_eval_continue_on_fail": self.periodic_eval_continue_on_fail.isChecked(),
            # Workflow
            "workflow_mode": self.workflow_mode.currentData() or "train_then_eval",
            "extra_args": self.extra_args.text(),
        }

    def _apply_config(self, cfg: dict[str, Any]) -> None:
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
            ("dataset_suite_dir", self.dataset_suite_dir),
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
                self._suite_manifest_label.setStyleSheet(f"color: {THEME['success']}; font-size: 10px;")
                if not self.dataset_suite_dir.text().strip():
                    self.dataset_suite_dir.setText(str(Path(self.applied_suite_manifest_path).parent))
            else:
                self._suite_manifest_label.setText("(no suite applied)")
                self._suite_manifest_label.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 10px;")
        # Periodic evaluation (monitoring only) — backward compatible: old profiles
        # without these keys keep the current (disabled) widget values.
        for _pk, _spin in (
            ("periodic_eval_count", self.periodic_eval_count),
            ("periodic_eval_every", self.periodic_eval_every),
            ("periodic_eval_max_samples", self.periodic_eval_max_samples),
            ("periodic_eval_batch_size", self.periodic_eval_batch_size),
        ):
            if _pk in cfg:
                try:
                    _spin.setValue(int(cfg[_pk]))
                except Exception:
                    pass
        if "periodic_eval_enabled" in cfg:
            self.periodic_eval_enabled.setChecked(bool(cfg["periodic_eval_enabled"]))
        if "periodic_eval_continue_on_fail" in cfg:
            self.periodic_eval_continue_on_fail.setChecked(bool(cfg["periodic_eval_continue_on_fail"]))
        if "periodic_eval_mode" in cfg:
            idx = self.periodic_eval_mode.findData(str(cfg["periodic_eval_mode"]))
            if idx >= 0:
                self.periodic_eval_mode.setCurrentIndex(idx)
        if "periodic_eval_dataset" in cfg:
            idx = self.periodic_eval_dataset.findData(str(cfg["periodic_eval_dataset"]))
            if idx >= 0:
                self.periodic_eval_dataset.setCurrentIndex(idx)
        if "periodic_eval_device" in cfg:
            self.periodic_eval_device.setCurrentText(str(cfg["periodic_eval_device"]))

        self._on_dataset_mode_changed()
        self._on_resume_toggled(self.resume_enabled.isChecked())
        self._on_loss_feature_toggled()
        self._on_periodic_eval_toggled()
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
            self.runner.append(f"[UI] Profile loaded: {key}")

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Profile", "Profile name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in _BUILTIN_PRESETS:
            QMessageBox.warning(
                self, "Blocked", f"'{name}' is a built-in profile and cannot be modified."
            )
            return
        _save_user_preset(name, self._collect_config())
        self._refresh_preset_list()
        for i in range(self._preset_combo.count()):
            if self._preset_combo.itemData(i) == name:
                self._preset_combo.setCurrentIndex(i)
                break
        self.runner.append(f"[UI] Profile saved: {name}")

    def _delete_preset(self) -> None:
        key = self._current_preset_key()
        if not key:
            return
        if key in _BUILTIN_PRESETS:
            QMessageBox.information(self, "Cannot Delete", f"'{key}' is a built-in profile.")
            return
        reply = QMessageBox.question(
            self,
            "Delete",
            f"Delete '{key}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            _delete_user_preset(key)
            self._refresh_preset_list()

    # -----------------------------------------------------------------
    # File Dialogs
    # -----------------------------------------------------------------
    def _remember_dataset_dir(self, path: str | Path) -> None:
        p = Path(path)
        self._last_dataset_dir = p if p.is_dir() else p.parent

    def _dataset_dialog_start_dir(self, target: ValidatedPathEdit | None = None) -> str:
        candidates: list[Path] = []
        if target is not None and target.text().strip():
            p = Path(target.text().strip())
            candidates.append(p if p.is_dir() else p.parent)
        if getattr(self, "_last_dataset_dir", None) is not None:
            candidates.append(self._last_dataset_dir)
        for edit in (self.train_data, self.val_data, self.test_data, self.ood_data, self.data):
            text = edit.text().strip()
            if text:
                p = Path(text)
                candidates.append(p if p.is_dir() else p.parent)
        candidates.extend((DATASET_SUITE_OUTPUT_ROOT, SCRIPT_DIR))
        for candidate in candidates:
            try:
                if candidate.exists():
                    return str(candidate)
            except OSError:
                continue
        return str(SCRIPT_DIR)

    def _pick_data(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            self._dataset_dialog_start_dir(self.data),
            "HDF5 (*.h5 *.hdf5);;All (*.*)",
        )
        if fn:
            self._remember_dataset_dir(fn)
            self.data.setText(_norm_path(fn))

    def _pick_dataset_path(self, target: ValidatedPathEdit, title: str) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            title,
            self._dataset_dialog_start_dir(target),
            "HDF5 (*.h5 *.hdf5);;All (*.*)",
        )
        if fn:
            self._remember_dataset_dir(fn)
            target.setText(_norm_path(fn))

    def _pick_dataset_suite(self) -> None:
        suite_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Suite Folder",
            self._dataset_dialog_start_dir(),
        )
        if suite_dir:
            self.dataset_suite_dir.setText(_norm_path(suite_dir))
            self._apply_dataset_suite(Path(suite_dir))

    def _apply_dataset_suite_from_field(self) -> None:
        suite_text = self.dataset_suite_dir.text().strip()
        if not suite_text:
            QMessageBox.warning(self, "Dataset Suite", "Select a dataset suite folder first.")
            return
        self._apply_dataset_suite(Path(suite_text))

    def _apply_dataset_suite(self, suite_dir: Path) -> None:
        suite_dir = Path(suite_dir).expanduser()
        if not suite_dir.exists() or not suite_dir.is_dir():
            QMessageBox.warning(self, "Dataset Suite", f"Folder not found:\n{suite_dir}")
            return
        if self.dataset_suite_dir.text().strip() != str(suite_dir):
            self.dataset_suite_dir.setText(_norm_path(suite_dir))
        self._remember_dataset_dir(suite_dir)

        manifest_path = suite_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        files: dict[str, Any] = {}
        source = "folder scan"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                files = dict(manifest.get("output_files", {}) or {})
                source = "manifest.json"
            except Exception as exc:
                QMessageBox.warning(self, "Dataset Suite", f"manifest.json could not be read:\n{exc}")
                return

        def resolve_suite_file(value: object) -> str:
            if not value:
                return ""
            p = Path(str(value)).expanduser()
            if p.is_absolute():
                return str(p)
            for candidate in (
                (suite_dir / p).resolve(),
                (suite_dir / p.name).resolve(),
                (SCRIPT_DIR / p).resolve(),
            ):
                if candidate.exists():
                    return str(candidate)
            return str((suite_dir / p).resolve())

        def first_match(patterns: tuple[str, ...]) -> str:
            for pattern in patterns:
                matches = sorted(suite_dir.glob(pattern))
                if matches:
                    return str(matches[0].resolve())
            return ""

        train_path = resolve_suite_file(files.get("train", "")) or first_match(("train_hybrid*.h5", "train*.h5", "train*.hdf5"))
        val_path = resolve_suite_file(files.get("val", "")) or first_match(("val_uniform*.h5", "val*.h5", "val*.hdf5"))
        test_path = resolve_suite_file(files.get("test", "")) or first_match(("test_uniform*.h5", "test*.h5", "test*.hdf5"))
        ood_path = (
            resolve_suite_file(files.get("ood_combined", "") or files.get("ood_high", "") or files.get("ood_low", ""))
            or first_match(("ood_combined*.h5", "ood_high*.h5", "ood*.h5", "ood*.hdf5"))
        )

        if not train_path or not val_path:
            QMessageBox.warning(
                self,
                "Dataset Suite",
                "Could not find both train and validation datasets in the selected folder.\n"
                "Expected manifest.json or files like train_hybrid*.h5 and val_uniform*.h5.",
            )
            return

        self.train_data.setText(train_path)
        self.val_data.setText(val_path)
        if test_path:
            self.test_data.setText(test_path)
        if ood_path:
            self.ood_data.setText(ood_path)

        idx = self.dataset_mode.findData("independent")
        if idx >= 0:
            self.dataset_mode.setCurrentIndex(idx)
        self.dataset_name.setText("data")

        alt_min = manifest.get("train_alt_min_km")
        alt_max = manifest.get("train_alt_max_km")
        if alt_min is not None:
            self.altitude_min_km.setValue(float(alt_min))
        if alt_max is not None:
            self.altitude_max_km.setValue(float(alt_max))

        if manifest_path.exists():
            self.applied_suite_manifest_path = str(manifest_path.resolve())
            self._suite_manifest_label.setText(self.applied_suite_manifest_path)
            self._suite_manifest_label.setStyleSheet(f"color: {THEME['success']}; font-size: 10px;")
        else:
            self.applied_suite_manifest_path = ""
            self._suite_manifest_label.setText(f"(folder scan: {suite_dir})")
            self._suite_manifest_label.setStyleSheet(f"color: {THEME['info']}; font-size: 10px;")
        self._suite_manifest_label.setToolTip(
            f"Source: {source}\nTrain: {train_path}\nValidation: {val_path}\n"
            f"Test: {test_path or '(not selected)'}\nOOD: {ood_path or '(not selected)'}"
        )

        self._on_dataset_mode_changed()
        self._refresh_command_preview()
        self._refresh_checklist()

    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Output Folder", self.out_dir.text() or str(TRAINING_OUTPUT_ROOT)
        )
        if d:
            self.out_dir.setText(_norm_path(d))

    def _resolve_resume_run_dir(self) -> Path | None:
        """Resolve the run directory from the resume source (run dir / checkpoints/ / .pt)."""
        src = self.resume_from.text().strip()
        if not src:
            return None
        try:
            if resolve_artifact_run_dir is not None:
                return Path(resolve_artifact_run_dir(Path(src).expanduser()))
            p = Path(src).expanduser()
            if p.is_file():               # .../checkpoints/ckpt_last.pt
                return p.parent.parent if p.parent.name == "checkpoints" else p.parent
            if p.name == "checkpoints":
                return p.parent
            return p
        except Exception:
            return None

    def _resume_baseline_epoch(self) -> int:
        """Best-effort last-completed epoch of the resumed run.

        Used to seed the ETA estimator so remaining time = total - resumed
        epoch (instead of treating resume as starting from epoch 0)."""
        run_dir = self._resolve_resume_run_dir()
        if run_dir is None:
            return 0
        best = 0
        for name in ("history.jsonl", "history.csv"):
            p = Path(run_dir) / name
            if not p.exists():
                continue
            try:
                if name.endswith(".jsonl"):
                    with open(p, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                e = int(json.loads(line).get("epoch", -1))
                                best = max(best, e + 1)  # 0-based → completed count
                            except Exception:
                                continue
                else:
                    import csv as _csv
                    with open(p, encoding="utf-8", newline="") as fh:
                        for row in _csv.DictReader(fh):
                            try:
                                e = int(float(row.get("epoch", -1)))
                                best = max(best, e + 1)
                            except Exception:
                                continue
            except Exception:
                pass
            if best:
                break
        total = int(self.epochs.value())
        return max(0, min(best, max(0, total - 1)))

    def _autoload_resume_config(self) -> None:
        """Mirror the checkpoint run's architecture into the UI on resume.

        Resume requires the rebuilt network to match the checkpoint exactly;
        loading the previous run's config.json into the architecture/encoding
        fields prevents strict-resume mismatches."""
        if not self.resume_enabled.isChecked():
            return
        run_dir = self._resolve_resume_run_dir()
        if run_dir is None:
            return
        cfg = _read_json_if_exists(Path(run_dir) / "config.json")
        if not cfg:
            return

        def _set_int(widget, *keys):
            for k in keys:
                if cfg.get(k) is not None:
                    try:
                        widget.setValue(int(cfg[k]))
                        return
                    except Exception:
                        pass

        def _set_float(widget, *keys):
            for k in keys:
                if cfg.get(k) is not None:
                    try:
                        widget.setValue(float(cfg[k]))
                        return
                    except Exception:
                        pass

        def _set_bool(widget, *keys):
            for k in keys:
                if cfg.get(k) is not None:
                    widget.setChecked(bool(cfg[k]))
                    return

        # Core architecture (config.json uses TrainConfig field names).
        _set_int(self.hidden, "hidden")
        _set_int(self.depth, "depth")
        _set_float(self.dropout, "dropout")
        _set_float(self.w0_first, "w0_first")
        _set_float(self.w0_hidden, "w0_hidden")
        _set_int(self.n_bands, "n_bands")
        if hasattr(self, "use_residual_blocks"):
            _set_bool(self.use_residual_blocks, "use_residual_blocks")
        if cfg.get("activation"):
            self.activation.setCurrentText(str(cfg["activation"]))
        if cfg.get("model_preset") and hasattr(self, "model_preset"):
            idx = self.model_preset.findData(str(cfg["model_preset"]))
            if idx >= 0:
                self.model_preset.setCurrentIndex(idx)

        # Input encoding / Fourier.
        if hasattr(self, "use_fourier"):
            _set_bool(self.use_fourier, "use_fourier")
        _set_int(self.fourier_n, "fourier_n_features", "fourier_n")
        _set_float(self.fourier_sigma, "fourier_sigma")
        if hasattr(self, "fourier_append_raw"):
            _set_bool(self.fourier_append_raw, "fourier_append_raw")
        for attr, key in (
            ("use_radial_separation", "use_radial_separation"),
            ("use_radial_decay_encoding", "use_radial_decay_encoding"),
            ("use_physical_radial_decay_encoding", "use_physical_radial_decay_encoding"),
            ("use_real_sh_basis", "use_real_sh_basis"),
            ("physical_radial_decay_append_raw", "physical_radial_decay_append_raw"),
            ("physical_radial_decay_include_unit", "physical_radial_decay_include_unit"),
            ("physical_radial_decay_include_r_scaled", "physical_radial_decay_include_r_scaled"),
        ):
            if hasattr(self, attr):
                _set_bool(getattr(self, attr), key)
        if hasattr(self, "physical_radial_decay_max_power"):
            _set_int(self.physical_radial_decay_max_power, "physical_radial_decay_max_power")

        # Refresh dependent enable-states + command preview.
        if hasattr(self, "_on_model_preset_changed"):
            self._on_model_preset_changed()
        self._on_activation_changed(self.activation.currentText())
        self._refresh_command_preview()
        if hasattr(self, "runner"):
            self.runner.append(
                f"[UI] Resume: rebuilt network architecture from {Path(run_dir).name}/config.json "
                "(hidden/depth/activation/encoding locked to the checkpoint)."
            )

    def _pick_resume_run(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self,
            "Resume Run Directory",
            self.resume_from.text() or self.out_dir.text() or str(SCRIPT_DIR),
        )
        if d:
            self.resume_from.setText(_norm_path(d))
            self._autoload_resume_config()

    def _pick_resume_checkpoint(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Resume Checkpoint",
            self.resume_from.text() or self.out_dir.text() or str(SCRIPT_DIR),
            "PyTorch checkpoints (*.pt);;All (*.*)",
        )
        if fn:
            self.resume_from.setText(_norm_path(fn))
            self._autoload_resume_config()

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
                "preload_data", "pin_memory",
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
    def _build_args(self, show_errors: bool = True) -> list[str] | None:
        """Build the CLI argument list. Returns None on validation error."""
        def fail(title: str, message: str) -> list[str] | None:
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
        else:
            args += ["--no-altitude-balanced-loss"]
        args += ["--altitude-bin-width-km", str(self.altitude_bin_width_km.value())]
        args += ["--altitude-min-km", str(self.altitude_min_km.value())]
        args += ["--altitude-max-km", str(self.altitude_max_km.value())]
        if self.use_radial_cross_loss.isChecked():
            args += ["--use-radial-cross-loss"]
        else:
            args += ["--no-radial-cross-loss"]
        args += ["--radial-loss-weight", str(self.radial_loss_weight.value())]
        args += ["--cross-loss-weight", str(self.cross_loss_weight.value())]
        if self.use_laplacian_regularization.isChecked():
            args += ["--use-laplacian-regularization"]
        else:
            args += ["--no-laplacian-regularization"]
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

        # Periodic Evaluation During Training (monitoring only).
        # Emit flags ONLY when explicitly enabled, so default commands are unchanged.
        if self.periodic_eval_enabled.isChecked():
            mode = self.periodic_eval_mode.currentData() or "count"
            if mode == "every":
                args += ["--periodic-eval-every-epochs", str(self.periodic_eval_every.value())]
            else:
                args += ["--periodic-eval-count", str(self.periodic_eval_count.value())]
            args += ["--periodic-eval-dataset", self.periodic_eval_dataset.currentData() or "val"]
            args += ["--periodic-eval-max-samples", str(self.periodic_eval_max_samples.value())]
            _pe_bs = self.periodic_eval_batch_size.value()
            if _pe_bs > 0:
                args += ["--periodic-eval-batch-size", str(_pe_bs)]
            args += ["--periodic-eval-device", self.periodic_eval_device.currentText()]
            if not self.periodic_eval_continue_on_fail.isChecked():
                args += ["--periodic-eval-fail-fast"]

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
        data_path: str | None = None,
        test_data: str | None = None,
        ood_data: str | None = None,
        use_config_datasets: bool = False,
        out_dir: str | None = None,
    ) -> list[str] | None:
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

    def _show_monitor_page(self) -> None:
        """Open the full Training Monitor page in the main Studio shell."""
        try:
            if hasattr(self, "_page_tabs"):
                self._page_tabs.setCurrentIndex(self._monitor_tab_idx)
        except Exception:
            pass
        self.navigate_monitor_requested.emit()

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
                self._show_monitor_page()
            return

        args = self._build_args()
        if args is None:
            return

        self._user_stopped = False
        self._finish_panel.setVisible(False)
        out_dir = self.out_dir.text().strip()
        resume_source = out_dir or self.resume_from.text().strip()
        self.runner.set_stop_hint(self._resume_stop_hint(resume_source))
        self._epochs_max = int(self.epochs.value())
        self.runner.progress.setRange(0, self._epochs_max)
        self.runner.progress.setValue(0)
        self.runner.progress.setFormat("Epoch %v / %m")

        self._live_plot.clear()
        self._live_plot.set_patience(self.patience.value())
        self.runner.set_output_dir(out_dir if out_dir else "")
        self._set_history_poll_dir(out_dir)
        self._update_run_dir_label(out_dir)
        self._save_settings()
        # Dashboard v2: start ETA tracking and reset dashboard
        if _HAS_DASHBOARD_V2:
            if self._eta_estimator is not None:
                self._eta_estimator.set_total_epochs(int(self.epochs.value()))
                # On resume, seed the last completed epoch so the ETA reflects
                # that we are continuing mid-run (remaining = total - resumed).
                resume_baseline = (
                    self._resume_baseline_epoch()
                    if self.resume_enabled.isChecked() else 0
                )
                self._eta_estimator.on_training_start(start_epoch=resume_baseline)
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

        # Jump to the full Training Monitor page so the user sees charts immediately.
        self._show_monitor_page()
        if hasattr(self, "btn_start_setup"):
            self.btn_start_setup.setEnabled(False)

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
        self.runner.append(f"[UI] Added to queue: {label}")

    def _on_queue_job_started(self, job_index: int, args: list[str]) -> None:
        """Called by the queue when it's time to start the next job."""
        self._live_plot.clear()
        self._live_plot.set_patience(self.patience.value())
        self._epochs_max = int(self.epochs.value())
        self.runner.progress.setRange(0, self._epochs_max)
        self.runner.progress.setValue(0)
        self.runner.progress.setFormat("Epoch %v / %m  [Queue]")
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
        self._show_monitor_page()
        if hasattr(self, "btn_start_setup"):
            self.btn_start_setup.setEnabled(False)
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _arg_value(self, args: list[str], flag: str) -> str | None:
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
            self._monitor_artifact_timer.stop()
            self._provenance_panel.set_run_dir(None)
            self._periodic_eval_table.set_run_dir(None)
            return
        root = Path(run_dir)
        self._history_poll_path = root / "history.jsonl"
        self._history_poll_timer.start()
        self._monitor_artifact_timer.start()
        self._provenance_panel.set_run_dir(root)
        self._periodic_eval_table.set_run_dir(root)

    def _refresh_monitor_artifacts(self) -> None:
        run_dir = self.runner._output_dir or self.out_dir.text().strip()
        if not run_dir:
            return
        self._provenance_panel.set_run_dir(run_dir)
        self._periodic_eval_table.set_run_dir(run_dir)

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
        # Feed the same per-epoch history into the structured History table.
        if _HAS_DASHBOARD_V2 and self._structured_log is not None:
            try:
                self._structured_log.load_history_file(str(path))
            except Exception:
                pass

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
        if hasattr(self, "btn_start_setup"):
            self.btn_start_setup.setEnabled(True)
        w = None
        if h := self._header():
            w = h.window()
        if not w:
            w = self.window()
        if w:
            w.setWindowTitle("ST-LRPS Studio")

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
        user_stopped = bool(getattr(self, "_user_stopped", False))
        self._user_stopped = False

        self._poll_training_history()
        self._history_poll_timer.stop()
        self._monitor_artifact_timer.stop()
        training_ok = (
            exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
        )

        if training_ok:
            self._finish_notice.set_notice("Run Complete", "Training finished successfully.", kind="success")
        elif user_stopped:
            self._finish_notice.set_notice("Run Interrupted", "Training was stopped by the user.", kind="warning")
        else:
            self._finish_notice.set_notice("Run Failed", "Training process encountered an error.", kind="danger")
        self._finish_panel.setVisible(True)

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
        self._btn_finish_report.setEnabled(
            any((Path(run_dir) / relative).is_file() for relative in ("report.md", "reports/report.md", "evals/report.md"))
        )

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
            missing: list[str] = []
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

    def _on_eval_stdout(self, proc: QProcess) -> None:
        raw = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        raw += bytes(proc.readAllStandardError()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            self.runner.append(f"[EVAL] {line}")

    def _on_auto_eval_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
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

        # Direction metric (cosine similarity of predicted vs. reference accel)
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
        # ETA lives on the TimeMetricsStrip now; keep KPI device badge fresh.
        if kpi is not None and hasattr(kpi, "device"):
            kpi.device.set_value(self._detect_device_badge())

        # Phase 7: full time-metric cards on the Live Monitor.
        ts = self._time_strip
        if ts is not None:
            ts.elapsed.set_value(elapsed)
            ts.eta.set_value(remaining)
            if hasattr(ts, "finish"):
                ts.finish.set_value(finish)
            ts.epoch_duration.set_value(est.format_current_epoch())
            ts.avg_epoch.set_value(est.format_avg_epoch())
            sps = self._metrics_store.latest("samples_per_s") if self._metrics_store else None
            if sps is not None:
                ts.samples_per_s.set_value(f"{sps:,.0f}")

        # Compact time badges on the experiment header.
        hdr = self._header()
        if hdr is not None and hasattr(hdr, "set_elapsed"):
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
            f"Output: {path}" if path else f"Output: {TRAINING_OUTPUT_ROOT}/st_lrps_train_<timestamp>"
        )

    def _update_window_title_progress(self, current_epoch: int) -> None:
        target_epochs = self.epochs.value()
        if target_epochs > 0:
            pct = int(min(100, max(0, (current_epoch / target_epochs) * 100)))
            title = f"[{pct}% · ep {current_epoch}/{target_epochs}] ST-LRPS Studio"
            w = None
            if h := self._header():
                w = h.window()
            if not w:
                w = self.window()
            if w:
                w.setWindowTitle(title)

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

    def set_experiment_header(self, header) -> None:
        """Wire the MainWindow's experiment header so lifecycle/ETA updates reach it."""
        self._hdr_ref = header

    def _header(self):
        """Return the experiment header (injected ref, else via window())."""
        h = getattr(self, "_hdr_ref", None)
        if h is not None:
            return h
        win = self.window()
        return getattr(win, "_experiment_header", None)

    def _update_header_lifecycle(self, status: str) -> None:
        """Phase 6/11: reflect real training state in the experiment header."""
        if status == "TRAINING":
            kpi = getattr(self, "_kpi_strip", None)
            if kpi is not None and hasattr(kpi, "device"):
                kpi.device.set_value(self._detect_device_badge())
            ts = getattr(self, "_time_strip", None)
            if ts is not None and hasattr(ts, "started"):
                ts.started.set_value(datetime.now().strftime("%H:%M:%S"))

        hdr = self._header()
        if hdr is None or not hasattr(hdr, "set_status"):
            return
        hdr.set_status(status)
        if status == "TRAINING":
            hdr.set_elapsed("00:00:00")
            hdr.set_remaining("Estimating...")
            hdr.set_finish("Estimating...")
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

    def _parse_progress(self, line: str) -> None:
        # Accept both "Epoch [N/M]" banners and the engine's "epoch=N" kv form.
        ep: int | None = None
        total: int | None = None
        m = re.search(r"Epoch\s*(?:\[\s*)?(\d+)\s*/\s*(\d+)(?:\s*\])?", line)
        if m:
            ep = int(m.group(1))
            total = int(m.group(2))
        else:
            m_kv = re.search(r"\b(?:epoch|ep)\s*[=:]\s*(\d+)", line, re.IGNORECASE)
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

