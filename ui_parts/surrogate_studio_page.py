# ST_LRPS/ui_parts/surrogate_studio_page.py
# -*- coding: utf-8 -*-
"""
Surrogate Studio Page (UI Part)
================================

Filesystem-only management page for trained ST-LRPS surrogate gravity models.

Responsibilities
----------------
1. Discover and list trained runs under a configurable runs root.
2. Render parsed metadata (config.json) and evaluation artifacts
   (eval_report.json + plot files) for the currently selected run.
3. Provide a training command preview builder against
   ``python -m st_lrps.training.cli``.
4. Emit a ``model_selected`` signal so the host window can apply the
   currently selected run as the active ST-LRPS surrogate gravity model.

Design rules
------------
- No heavy ML dependencies (no torch, no h5py). JSON + os only.
- Every filesystem read is wrapped in try/except so the page never crashes
  when directories or files are missing or malformed.
- The training section is preview-only — this module never launches a
  training subprocess on its own.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from .ui_commons import (
        THEME,
        StatusBadge,
        find_project_root,
        get_icon,
    )
except ImportError:
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        print(
            "[ERROR] Run as: python -m ui_parts.surrogate_studio_page",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise


PROJECT_ROOT = find_project_root()
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "st_lrps" / "runs"
TRAIN_MODULE = "st_lrps.training.cli"


# =============================================================================
# Helper data structures
# =============================================================================


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file safely, returning None on any error."""
    try:
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _format_mtime(path: Path) -> str:
    """Return a short human-readable last-modified timestamp."""
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def _short_summary(cfg: Optional[Dict[str, Any]]) -> str:
    """Build a 1-line summary for the run list."""
    if not isinstance(cfg, dict):
        return ""
    parts: List[str] = []
    try:
        deg_max = cfg.get("degree_max")
        deg_min = cfg.get("degree_min")
        dataset_meta = cfg.get("dataset_meta") or {}
        if deg_max is None and isinstance(dataset_meta, dict):
            deg_max = dataset_meta.get("degree_max") or dataset_meta.get("requested_degree")
        if deg_min is None and isinstance(dataset_meta, dict):
            deg_min = dataset_meta.get("degree_min")
        if deg_max is not None:
            if deg_min is not None and int(deg_min) > 0:
                parts.append(f"deg {int(deg_min)}->{int(deg_max)}")
            else:
                parts.append(f"deg {int(deg_max)}")
    except Exception:
        pass
    try:
        alt_min = cfg.get("altitude_min_km")
        alt_max = cfg.get("altitude_max_km")
        if alt_min is not None and alt_max is not None:
            parts.append(f"{float(alt_min):.0f}-{float(alt_max):.0f} km")
    except Exception:
        pass
    return "  |  ".join(parts)


# =============================================================================
# Main Page Widget
# =============================================================================


class SurrogateStudioPage(QtWidgets.QWidget):
    """
    Surrogate Studio: manage trained ST-LRPS runs and assemble training commands.

    Signals
    -------
    model_selected:
        Emitted with the absolute run-directory path when the user clicks
        "Use This Model".  The host window is expected to update the active
        ST-LRPS gravity configuration accordingly.
    """

    model_selected = QtCore.Signal(str)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._runs_root: Path = DEFAULT_RUNS_ROOT
        self._selected_run: Optional[Path] = None
        self._build_ui()
        # Initial discovery is best-effort; never fatal.
        QtCore.QTimer.singleShot(0, self._refresh_runs_list)
        QtCore.QTimer.singleShot(0, self._refresh_command_preview)

    # ------------------------------------------------------------------
    # Public state API
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return a small serializable state dict for session persistence."""
        try:
            return {
                "runs_root": str(self._runs_root) if self._runs_root else "",
                "selected_run": str(self._selected_run) if self._selected_run else "",
            }
        except Exception:
            return {"runs_root": "", "selected_run": ""}

    def apply_state(self, state: Dict[str, Any]) -> None:
        """Restore page state captured by :py:meth:`get_state`."""
        if not isinstance(state, dict):
            return
        try:
            runs_root = str(state.get("runs_root", "") or "").strip()
            if runs_root:
                self._runs_root = Path(runs_root)
                self.ent_runs_root.setText(runs_root)
            self._refresh_runs_list()
            wanted = str(state.get("selected_run", "") or "").strip()
            if wanted:
                self._select_run_by_path(Path(wanted))
        except Exception:
            pass

    def get_selected_run_dir(self) -> str:
        """Return the currently selected run directory as a string (or '')."""
        return str(self._selected_run) if self._selected_run else ""

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        layout.addWidget(self._build_browser_card())
        layout.addWidget(self._build_details_card())
        layout.addWidget(self._build_eval_card())
        layout.addWidget(self._build_training_card())
        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Section 1: Run directory browser
    # ------------------------------------------------------------------

    def _build_browser_card(self) -> QtWidgets.QGroupBox:
        gb = self._make_card("Run Directory Browser")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 22, 16, 16)
        layout.setSpacing(10)

        # Runs root row
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Runs root:"))
        self.ent_runs_root = QtWidgets.QLineEdit(str(self._runs_root))
        self.ent_runs_root.setPlaceholderText("Path to a directory containing ST-LRPS runs")
        row.addWidget(self.ent_runs_root, 1)

        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        btn_browse.clicked.connect(self._on_browse_runs_root)
        row.addWidget(btn_browse)

        btn_auto = QtWidgets.QPushButton("Auto-detect")
        btn_auto.setIcon(get_icon("fa6s.wand-magic-sparkles", THEME["fg_main"]))
        btn_auto.clicked.connect(self._on_auto_detect_runs)
        row.addWidget(btn_auto)

        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.setIcon(get_icon("fa6s.rotate", THEME["fg_main"]))
        btn_refresh.clicked.connect(self._refresh_runs_list)
        row.addWidget(btn_refresh)

        layout.addLayout(row)

        # Run list
        self.list_runs = QtWidgets.QListWidget()
        self.list_runs.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_runs.setMinimumHeight(160)
        self.list_runs.itemSelectionChanged.connect(self._on_run_selected)
        layout.addWidget(self.list_runs)

        self.lbl_runs_status = QtWidgets.QLabel("No runs scanned yet.")
        self.lbl_runs_status.setStyleSheet(
            f"color: {THEME['fg_muted']}; font-size: 9pt;"
        )
        layout.addWidget(self.lbl_runs_status)

        return gb

    # ------------------------------------------------------------------
    # Section 2: Selected run details
    # ------------------------------------------------------------------

    def _build_details_card(self) -> QtWidgets.QGroupBox:
        gb = self._make_card("Selected Run Details")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 22, 16, 16)
        layout.setSpacing(10)

        self.lbl_selected_name = QtWidgets.QLabel("No run selected.")
        self.lbl_selected_name.setStyleSheet(
            f"color: {THEME['fg_soft']}; font-weight: 700; font-size: 11pt;"
        )
        layout.addWidget(self.lbl_selected_name)

        # Badge row
        self._badges: Dict[str, StatusBadge] = {}
        badges_row = QtWidgets.QHBoxLayout()
        badges_row.setSpacing(8)
        for key, label in (
            ("config", "config.json"),
            ("scaler", "scaler.json"),
            ("ckpt_best", "ckpt_best"),
            ("ckpt_last", "ckpt_last"),
            ("lunar", "lunar"),
        ):
            badge = StatusBadge(label.upper(), kind="error", parent=self)
            badge.setMinimumWidth(110)
            self._badges[key] = badge
            badges_row.addWidget(badge)
        badges_row.addStretch(1)
        layout.addLayout(badges_row)

        # Metadata grid
        meta_card = QtWidgets.QFrame()
        meta_card.setStyleSheet(
            f"background: {THEME['bg_card_alt']}; border: 1px solid {THEME['border_soft']};"
            f" border-radius: 8px;"
        )
        self.meta_grid = QtWidgets.QGridLayout(meta_card)
        self.meta_grid.setContentsMargins(12, 10, 12, 10)
        self.meta_grid.setHorizontalSpacing(20)
        self.meta_grid.setVerticalSpacing(6)
        layout.addWidget(meta_card)

        self._meta_labels: Dict[str, QtWidgets.QLabel] = {}
        meta_keys = [
            ("degree_min_max", "Degree (min -> max)"),
            ("alt_range", "Training altitude range"),
            ("activation", "Activation"),
            ("hidden_depth", "Hidden / depth"),
            ("residual_blocks", "Residual blocks"),
            ("n_bands", "Frequency bands"),
            ("best_epoch", "Best epoch"),
            ("best_metric", "Best metric"),
            ("dir_loss_weight", "Direction loss weight"),
        ]
        for i, (key, label) in enumerate(meta_keys):
            r = i // 2
            c = (i % 2) * 2
            lbl = QtWidgets.QLabel(label + ":")
            lbl.setStyleSheet(f"color: {THEME['fg_muted']};")
            self.meta_grid.addWidget(lbl, r, c)
            val = QtWidgets.QLabel("—")
            val.setStyleSheet(f"color: {THEME['fg_soft']}; font-weight: 600;")
            self._meta_labels[key] = val
            self.meta_grid.addWidget(val, r, c + 1)

        # Action buttons
        action_row = QtWidgets.QHBoxLayout()
        self.btn_use_model = QtWidgets.QPushButton("Use This Model")
        self.btn_use_model.setIcon(get_icon("fa6s.circle-check", "#FFFFFF"))
        self.btn_use_model.setObjectName("primaryBtn")
        self.btn_use_model.clicked.connect(self._on_use_model)
        action_row.addWidget(self.btn_use_model)

        self.btn_open_folder = QtWidgets.QPushButton("Open Run Folder")
        self.btn_open_folder.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        self.btn_open_folder.clicked.connect(self._on_open_run_folder)
        action_row.addWidget(self.btn_open_folder)

        self.btn_open_config = QtWidgets.QPushButton("Open config.json")
        self.btn_open_config.setIcon(get_icon("fa6s.file-code", THEME["fg_main"]))
        self.btn_open_config.clicked.connect(self._on_open_config)
        action_row.addWidget(self.btn_open_config)

        self.btn_open_eval_report = QtWidgets.QPushButton("Open Evaluation Report")
        self.btn_open_eval_report.setIcon(get_icon("fa6s.chart-simple", THEME["fg_main"]))
        self.btn_open_eval_report.clicked.connect(self._on_open_eval_report)
        action_row.addWidget(self.btn_open_eval_report)

        action_row.addStretch(1)
        layout.addLayout(action_row)
        self._set_actions_enabled(False)
        return gb

    # ------------------------------------------------------------------
    # Section 3: Evaluation artifacts
    # ------------------------------------------------------------------

    def _build_eval_card(self) -> QtWidgets.QGroupBox:
        gb = self._make_card("Evaluation Artifacts")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 22, 16, 16)
        layout.setSpacing(8)

        # Metric grid
        metric_card = QtWidgets.QFrame()
        metric_card.setStyleSheet(
            f"background: {THEME['bg_card_alt']}; border: 1px solid {THEME['border_soft']};"
            f" border-radius: 8px;"
        )
        self.metric_grid = QtWidgets.QGridLayout(metric_card)
        self.metric_grid.setContentsMargins(12, 10, 12, 10)
        self.metric_grid.setHorizontalSpacing(20)
        self.metric_grid.setVerticalSpacing(6)
        layout.addWidget(metric_card)

        self._metric_labels: Dict[str, QtWidgets.QLabel] = {}
        rows = [
            ("acc_rmse", "Acceleration RMSE"),
            ("acc_mae", "Acceleration MAE"),
            ("ang_mean_deg", "Angular mean (deg)"),
            ("ang_p90_deg", "Angular p90 (deg)"),
            ("cossim_mean", "Mean cosine similarity"),
            ("masked_ang_mean", "Masked angular mean"),
        ]
        for i, (key, label) in enumerate(rows):
            r = i // 2
            c = (i % 2) * 2
            lbl = QtWidgets.QLabel(label + ":")
            lbl.setStyleSheet(f"color: {THEME['fg_muted']};")
            self.metric_grid.addWidget(lbl, r, c)
            val = QtWidgets.QLabel("—")
            val.setStyleSheet(f"color: {THEME['fg_soft']}; font-weight: 600;")
            self._metric_labels[key] = val
            self.metric_grid.addWidget(val, r, c + 1)

        # Timestamp
        self.lbl_eval_timestamp = QtWidgets.QLabel("Last Evaluated: —")
        self.lbl_eval_timestamp.setStyleSheet(
            f"color: {THEME['fg_muted']}; font-size: 9pt;"
        )
        layout.addWidget(self.lbl_eval_timestamp)

        # Plot list
        self.list_eval_plots = QtWidgets.QListWidget()
        self.list_eval_plots.setMaximumHeight(110)
        layout.addWidget(self.list_eval_plots)

        btn_row = QtWidgets.QHBoxLayout()
        btn_open_plot = QtWidgets.QPushButton("Open Selected Plot")
        btn_open_plot.setIcon(get_icon("fa6s.up-right-from-square", THEME["fg_main"]))
        btn_open_plot.clicked.connect(self._on_open_eval_plot)
        btn_row.addWidget(btn_open_plot)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.lbl_eval_empty = QtWidgets.QLabel(
            "No evaluation artifacts found. Run st_lrps.evaluation.cli first."
        )
        self.lbl_eval_empty.setStyleSheet(
            f"color: {THEME['fg_muted']}; font-style: italic;"
        )
        layout.addWidget(self.lbl_eval_empty)

        return gb

    # ------------------------------------------------------------------
    # Section 4: Training command builder
    # ------------------------------------------------------------------

    def _build_training_card(self) -> QtWidgets.QGroupBox:
        gb = self._make_card("Training Command Builder")
        outer = QtWidgets.QVBoxLayout(gb)
        outer.setContentsMargins(16, 22, 16, 16)
        outer.setSpacing(10)

        # Paths
        path_grid = QtWidgets.QGridLayout()
        path_grid.setHorizontalSpacing(8)
        path_grid.setVerticalSpacing(6)

        path_grid.addWidget(QtWidgets.QLabel("Dataset path:"), 0, 0)
        self.ent_train_data = QtWidgets.QLineEdit()
        self.ent_train_data.setPlaceholderText("Path to dataset .h5 (optional, auto-detected)")
        path_grid.addWidget(self.ent_train_data, 0, 1)
        btn_dataset = QtWidgets.QPushButton("Browse")
        btn_dataset.clicked.connect(self._on_browse_dataset)
        path_grid.addWidget(btn_dataset, 0, 2)

        path_grid.addWidget(QtWidgets.QLabel("Output directory:"), 1, 0)
        self.ent_train_out = QtWidgets.QLineEdit()
        self.ent_train_out.setPlaceholderText("Output run directory (optional)")
        path_grid.addWidget(self.ent_train_out, 1, 1)
        btn_out = QtWidgets.QPushButton("Browse")
        btn_out.clicked.connect(self._on_browse_train_out)
        path_grid.addWidget(btn_out, 1, 2)

        outer.addLayout(path_grid)

        # Parameter grid
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        def add_int(row: int, col: int, label: str, default: int, lo: int = 0, hi: int = 10_000_000) -> QtWidgets.QSpinBox:
            grid.addWidget(QtWidgets.QLabel(label), row, col)
            sp = QtWidgets.QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(default)
            grid.addWidget(sp, row, col + 1)
            return sp

        def add_double(row: int, col: int, label: str, default: float, lo: float, hi: float, step: float, decimals: int = 3) -> QtWidgets.QDoubleSpinBox:
            grid.addWidget(QtWidgets.QLabel(label), row, col)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(default)
            grid.addWidget(sp, row, col + 1)
            return sp

        self.sp_epochs = add_int(0, 0, "Epochs:", 250, 1, 100_000)
        self.sp_batch = add_int(0, 2, "Batch size:", 16384, 1, 1_000_000)
        self.sp_hidden = add_int(1, 0, "Hidden neurons:", 512, 8, 8192)
        self.sp_depth = add_int(1, 2, "Depth:", 6, 1, 64)

        grid.addWidget(QtWidgets.QLabel("Activation:"), 2, 0)
        self.cb_activation = QtWidgets.QComboBox()
        self.cb_activation.addItems(["sine", "silu", "tanh", "softplus"])
        grid.addWidget(self.cb_activation, 2, 1)

        self.sp_dir_loss_weight = add_double(
            2, 2, "Dir. loss weight:", 0.20, 0.0, 1.0, 0.05, decimals=2
        )

        self.sp_dir_loss_start = add_int(3, 0, "Dir. loss start epoch:", 10, 0, 100_000)
        self.sp_dir_loss_ramp = add_int(3, 2, "Dir. loss ramp epochs:", 40, 1, 100_000)

        grid.addWidget(QtWidgets.QLabel("Dir. loss floor abs:"), 4, 0)
        self.ent_dir_loss_floor = QtWidgets.QLineEdit("1e-7")
        grid.addWidget(self.ent_dir_loss_floor, 4, 1)

        self.chk_alt_balanced = QtWidgets.QCheckBox("Altitude balanced loss")
        self.chk_alt_balanced.setChecked(True)
        grid.addWidget(self.chk_alt_balanced, 4, 2, 1, 2)

        self.chk_radial_cross = QtWidgets.QCheckBox("Radial / cross loss")
        self.chk_radial_cross.setChecked(True)
        grid.addWidget(self.chk_radial_cross, 5, 0, 1, 2)

        self.sp_radial_w = add_double(5, 2, "Radial weight:", 0.05, 0.0, 0.5, 0.01, decimals=3)
        self.sp_cross_w = add_double(6, 0, "Cross weight:", 0.10, 0.0, 0.5, 0.01, decimals=3)

        self.chk_residual = QtWidgets.QCheckBox("Residual blocks")
        self.chk_residual.setChecked(True)
        grid.addWidget(self.chk_residual, 6, 2, 1, 2)

        self.sp_n_bands = add_int(7, 0, "N bands:", 3, 1, 8)
        self.sp_grad_acc = add_int(7, 2, "Grad accum steps:", 2, 1, 64)

        grid.addWidget(QtWidgets.QLabel("Best metric:"), 8, 0)
        self.cb_best_metric = QtWidgets.QComboBox()
        self.cb_best_metric.addItems(["total_loss", "hybrid", "direction_loss"])
        self.cb_best_metric.setCurrentText("hybrid")
        grid.addWidget(self.cb_best_metric, 8, 1)

        self.sp_hybrid_alpha = add_double(
            8, 2, "Hybrid dir. alpha:", 0.30, 0.0, 1.0, 0.05, decimals=2
        )

        self.chk_preload = QtWidgets.QCheckBox("Preload data into RAM")
        self.chk_preload.setChecked(True)
        grid.addWidget(self.chk_preload, 9, 0, 1, 2)

        outer.addLayout(grid)

        # Buttons (preset / refresh / copy)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_preset = QtWidgets.QPushButton("Apply Angular Drift Mitigation Preset")
        self.btn_preset.setIcon(get_icon("fa6s.wand-magic-sparkles", THEME["fg_main"]))
        self.btn_preset.clicked.connect(self._apply_drift_preset)
        btn_row.addWidget(self.btn_preset)

        self.btn_refresh_cmd = QtWidgets.QPushButton("Refresh Preview")
        self.btn_refresh_cmd.setIcon(get_icon("fa6s.rotate", THEME["fg_main"]))
        self.btn_refresh_cmd.clicked.connect(self._refresh_command_preview)
        btn_row.addWidget(self.btn_refresh_cmd)

        self.btn_copy_cmd = QtWidgets.QPushButton("Copy Command")
        self.btn_copy_cmd.setIcon(get_icon("fa6s.copy", THEME["fg_main"]))
        self.btn_copy_cmd.clicked.connect(self._copy_command_preview)
        btn_row.addWidget(self.btn_copy_cmd)

        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        # Command preview
        self.txt_train_cmd = QtWidgets.QPlainTextEdit()
        self.txt_train_cmd.setReadOnly(True)
        self.txt_train_cmd.setMinimumHeight(110)
        self.txt_train_cmd.setStyleSheet(
            f"background-color: {THEME['bg_log']}; color: {THEME['fg_main']};"
            f" font-family: Consolas, monospace; border-radius: 6px;"
        )
        outer.addWidget(self.txt_train_cmd)

        note = QtWidgets.QLabel(
            "Preview only. This page never launches training automatically — "
            "copy the command to a terminal to start a training run."
        )
        note.setStyleSheet(f"color: {THEME['fg_muted']}; font-style: italic; font-size: 9pt;")
        note.setWordWrap(True)
        outer.addWidget(note)

        return gb

    # ------------------------------------------------------------------
    # Cosmetic helpers
    # ------------------------------------------------------------------

    def _make_card(self, title: str) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox(title)
        gb.setStyleSheet(
            f"""
            QGroupBox {{
                border: 1px solid {THEME['border']};
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 6px;
                background: {THEME['bg_card']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                color: {THEME['fg_soft']};
                font-weight: 700;
                font-size: 10pt;
            }}
            """
        )
        return gb

    def _set_actions_enabled(self, enabled: bool) -> None:
        for btn in (
            self.btn_use_model,
            self.btn_open_folder,
            self.btn_open_config,
            self.btn_open_eval_report,
        ):
            btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Section 1 logic: runs list
    # ------------------------------------------------------------------

    def _on_browse_runs_root(self) -> None:
        try:
            current = self.ent_runs_root.text().strip() or str(self._runs_root)
        except Exception:
            current = str(PROJECT_ROOT)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select ST-LRPS Runs Root", current
        )
        if path:
            self.ent_runs_root.setText(path)
            self._runs_root = Path(path)
            self._refresh_runs_list()

    def _on_auto_detect_runs(self) -> None:
        """Search the project for any directories named `runs/` under a folder containing `st_lrps_*`."""
        candidate_roots: List[Path] = []
        try:
            base = PROJECT_ROOT
            for sub in [
                base / "st_lrps" / "runs",
                base / "runs",
                base,
            ]:
                if sub.is_dir():
                    candidate_roots.append(sub)
            # Also walk shallowly looking for runs/ directories containing st_lrps_*
            for root_dir in [base / "st_lrps", base]:
                if not root_dir.is_dir():
                    continue
                try:
                    for entry in root_dir.iterdir():
                        if not entry.is_dir():
                            continue
                        if entry.name == "runs":
                            candidate_roots.append(entry)
                except Exception:
                    pass
        except Exception:
            pass

        chosen: Optional[Path] = None
        for candidate in candidate_roots:
            try:
                if any(p.is_dir() and p.name.startswith("st_lrps_") for p in candidate.iterdir()):
                    chosen = candidate
                    break
            except Exception:
                continue
        if chosen is None and candidate_roots:
            chosen = candidate_roots[0]
        if chosen is None:
            chosen = DEFAULT_RUNS_ROOT

        self._runs_root = chosen
        self.ent_runs_root.setText(str(chosen))
        self._refresh_runs_list()

    def _refresh_runs_list(self) -> None:
        self.list_runs.clear()
        try:
            text = self.ent_runs_root.text().strip()
            if text:
                self._runs_root = Path(text)
        except Exception:
            pass

        if not self._runs_root or not self._runs_root.is_dir():
            self.lbl_runs_status.setText(
                f"Runs directory does not exist: {self._runs_root}"
            )
            self._clear_details()
            return

        try:
            entries = [p for p in self._runs_root.iterdir() if p.is_dir()]
        except Exception as exc:
            self.lbl_runs_status.setText(f"Could not list directory: {exc}")
            self._clear_details()
            return

        # Filter to st_lrps_* directories first, fall back to all
        filtered = [p for p in entries if p.name.startswith("st_lrps_")]
        if not filtered:
            filtered = entries
        try:
            filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            filtered.sort(key=lambda p: p.name, reverse=True)

        if not filtered:
            self.lbl_runs_status.setText("No runs found in this directory.")
            self._clear_details()
            return

        for run_dir in filtered:
            cfg = _safe_json_load(run_dir / "config.json")
            summary = _short_summary(cfg)
            modified = _format_mtime(run_dir)
            display = f"{run_dir.name}   ·   {modified}"
            if summary:
                display += f"   ·   {summary}"
            item = QtWidgets.QListWidgetItem(display)
            item.setData(QtCore.Qt.UserRole, str(run_dir))
            self.list_runs.addItem(item)

        self.lbl_runs_status.setText(f"{len(filtered)} run(s) discovered.")

    def _select_run_by_path(self, path: Path) -> None:
        target = str(path.resolve()) if path.exists() else str(path)
        for i in range(self.list_runs.count()):
            item = self.list_runs.item(i)
            data = str(item.data(QtCore.Qt.UserRole) or "")
            if Path(data).resolve() == Path(target).resolve() if Path(data).exists() else data == target:
                self.list_runs.setCurrentItem(item)
                return

    def _on_run_selected(self) -> None:
        item = self.list_runs.currentItem()
        if item is None:
            self._clear_details()
            return
        run_dir_str = str(item.data(QtCore.Qt.UserRole) or "")
        if not run_dir_str:
            self._clear_details()
            return
        run_dir = Path(run_dir_str)
        self._selected_run = run_dir
        self._refresh_details(run_dir)
        self._refresh_eval_artifacts(run_dir)
        self._set_actions_enabled(True)

    def _clear_details(self) -> None:
        self._selected_run = None
        self.lbl_selected_name.setText("No run selected.")
        for badge in self._badges.values():
            badge.set_status("error", "MISSING")
        for lbl in self._meta_labels.values():
            lbl.setText("—")
        for lbl in self._metric_labels.values():
            lbl.setText("—")
        self.lbl_eval_timestamp.setText("Last Evaluated: —")
        self.list_eval_plots.clear()
        self.lbl_eval_empty.setVisible(True)
        self._set_actions_enabled(False)

    # ------------------------------------------------------------------
    # Section 2 logic: details
    # ------------------------------------------------------------------

    def _refresh_details(self, run_dir: Path) -> None:
        self.lbl_selected_name.setText(run_dir.name)

        # Badges
        cfg_path = run_dir / "config.json"
        scaler_path = run_dir / "scaler.json"
        ckpt_best = run_dir / "checkpoints" / "ckpt_best.pt"
        ckpt_last = run_dir / "checkpoints" / "ckpt_last.pt"

        self._badges["config"].set_status(
            "success" if cfg_path.is_file() else "error",
            "CONFIG OK" if cfg_path.is_file() else "CONFIG MISSING",
        )
        self._badges["scaler"].set_status(
            "success" if scaler_path.is_file() else "warning",
            "SCALER OK" if scaler_path.is_file() else "NO SCALER",
        )
        self._badges["ckpt_best"].set_status(
            "success" if ckpt_best.is_file() else "warning",
            "BEST OK" if ckpt_best.is_file() else "NO BEST",
        )
        self._badges["ckpt_last"].set_status(
            "success" if ckpt_last.is_file() else "warning",
            "LAST OK" if ckpt_last.is_file() else "NO LAST",
        )

        cfg = _safe_json_load(cfg_path) or {}
        # Lunar check
        lunar_ok: Optional[bool] = None
        try:
            from .surrogate_artifacts import looks_like_lunar_surrogate_run
            lunar_ok = bool(looks_like_lunar_surrogate_run(run_dir))
        except Exception:
            lunar_ok = None
        if lunar_ok is True:
            self._badges["lunar"].set_status("success", "LUNAR OK")
        elif lunar_ok is False:
            self._badges["lunar"].set_status("warning", "LUNAR ?")
        else:
            self._badges["lunar"].set_status("warning", "LUNAR ?")

        # Meta fields
        try:
            deg_max = cfg.get("degree_max")
            deg_min = cfg.get("degree_min")
            dataset_meta = cfg.get("dataset_meta") or {}
            if deg_max is None and isinstance(dataset_meta, dict):
                deg_max = dataset_meta.get("degree_max") or dataset_meta.get("requested_degree")
            if deg_min is None and isinstance(dataset_meta, dict):
                deg_min = dataset_meta.get("degree_min")
            self._meta_labels["degree_min_max"].setText(
                f"{deg_min if deg_min is not None else '?'} -> {deg_max if deg_max is not None else '?'}"
            )
        except Exception:
            self._meta_labels["degree_min_max"].setText("—")

        try:
            alt_min = cfg.get("altitude_min_km")
            alt_max = cfg.get("altitude_max_km")
            if alt_min is not None and alt_max is not None:
                self._meta_labels["alt_range"].setText(
                    f"{float(alt_min):.1f} to {float(alt_max):.1f} km"
                )
            else:
                self._meta_labels["alt_range"].setText("—")
        except Exception:
            self._meta_labels["alt_range"].setText("—")

        for key, cfg_key, fmt in [
            ("activation", "activation", lambda v: str(v)),
            ("hidden_depth", None, None),  # handled below
            ("residual_blocks", "use_residual_blocks", lambda v: "Yes" if bool(v) else "No"),
            ("n_bands", "n_bands", lambda v: str(int(v))),
            ("best_epoch", "best_epoch", lambda v: str(int(v))),
            ("best_metric", "best_metric", lambda v: str(v)),
            ("dir_loss_weight", "direction_loss_weight", lambda v: f"{float(v):.3g}"),
        ]:
            if key == "hidden_depth":
                try:
                    hid = cfg.get("hidden")
                    dep = cfg.get("depth")
                    if hid is not None and dep is not None:
                        self._meta_labels["hidden_depth"].setText(
                            f"{int(hid)} / {int(dep)}"
                        )
                    else:
                        self._meta_labels["hidden_depth"].setText("—")
                except Exception:
                    self._meta_labels["hidden_depth"].setText("—")
                continue
            try:
                v = cfg.get(cfg_key)
                self._meta_labels[key].setText(fmt(v) if v is not None else "—")
            except Exception:
                self._meta_labels[key].setText("—")

    # ------------------------------------------------------------------
    # Section 3 logic: evaluation artifacts
    # ------------------------------------------------------------------

    def _eval_dir(self, run_dir: Path) -> Optional[Path]:
        """Return the directory that contains evaluation artifacts, if any."""
        candidates: List[Path] = []
        try:
            if (run_dir / "eval_report.json").is_file():
                candidates.append(run_dir)
        except Exception:
            pass
        try:
            sub_eval = run_dir / "eval"
            if sub_eval.is_dir() and (sub_eval / "eval_report.json").is_file():
                candidates.append(sub_eval)
        except Exception:
            pass
        return candidates[0] if candidates else None

    def _refresh_eval_artifacts(self, run_dir: Path) -> None:
        for lbl in self._metric_labels.values():
            lbl.setText("—")
        self.list_eval_plots.clear()
        self.lbl_eval_timestamp.setText("Last Evaluated: —")

        eval_dir = self._eval_dir(run_dir)
        # Even when eval_report.json is missing, search for plots in run_dir/eval
        plots_dir = eval_dir or (run_dir / "eval" if (run_dir / "eval").is_dir() else run_dir)
        report = _safe_json_load(eval_dir / "eval_report.json") if eval_dir else None

        has_any = False
        if isinstance(report, dict):
            has_any = True
            metrics = report.get("metrics") or report
            def _set(key: str, candidates: List[str], fmt: str = "{:.4g}") -> None:
                for c in candidates:
                    if isinstance(metrics, dict) and c in metrics and metrics[c] is not None:
                        try:
                            self._metric_labels[key].setText(fmt.format(float(metrics[c])))
                            return
                        except Exception:
                            try:
                                self._metric_labels[key].setText(str(metrics[c]))
                                return
                            except Exception:
                                continue
                self._metric_labels[key].setText("—")

            _set("acc_rmse", ["acc_rmse", "acceleration_rmse", "a_rmse"])
            _set("acc_mae", ["acc_mae", "acceleration_mae", "a_mae"])
            _set("ang_mean_deg", ["angular_mean_deg", "ang_mean_deg", "angular_mean"])
            _set("ang_p90_deg", ["angular_p90_deg", "ang_p90_deg", "angular_p90"])
            _set("cossim_mean", ["mean_cosine_similarity", "cossim_mean", "cosine_similarity_mean"])
            _set("masked_ang_mean", ["masked_angular_mean", "masked_ang_mean_deg"])

            try:
                ts = (eval_dir / "eval_report.json").stat().st_mtime
                self.lbl_eval_timestamp.setText(
                    "Last Evaluated: " + datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                )
            except Exception:
                pass

        # Plot list
        try:
            if plots_dir and plots_dir.is_dir():
                plots = sorted(
                    [p for p in plots_dir.iterdir() if p.suffix.lower() == ".png"]
                )
                for png in plots:
                    item = QtWidgets.QListWidgetItem(png.name)
                    item.setData(QtCore.Qt.UserRole, str(png))
                    self.list_eval_plots.addItem(item)
                if plots:
                    has_any = True
        except Exception:
            pass

        self.lbl_eval_empty.setVisible(not has_any)

    # ------------------------------------------------------------------
    # Actions: opening folders / files
    # ------------------------------------------------------------------

    def _open_with_os(self, path: Path) -> bool:
        try:
            url = QtCore.QUrl.fromLocalFile(str(path))
            return bool(QtGui.QDesktopServices.openUrl(url))
        except Exception:
            try:
                if sys.platform == "win32":
                    os.startfile(str(path))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(path)])
                else:
                    subprocess.Popen(["xdg-open", str(path)])
                return True
            except Exception:
                return False

    def _on_use_model(self) -> None:
        if self._selected_run is None:
            return
        try:
            self.model_selected.emit(str(self._selected_run))
        except Exception:
            pass

    def _on_open_run_folder(self) -> None:
        if self._selected_run and self._selected_run.is_dir():
            self._open_with_os(self._selected_run)

    def _on_open_config(self) -> None:
        if self._selected_run is None:
            return
        cfg_path = self._selected_run / "config.json"
        if cfg_path.is_file():
            self._open_with_os(cfg_path)

    def _on_open_eval_report(self) -> None:
        if self._selected_run is None:
            return
        eval_dir = self._eval_dir(self._selected_run)
        if eval_dir is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Evaluation Report",
                "eval_report.json was not found for this run.",
            )
            return
        report_path = eval_dir / "eval_report.json"
        if report_path.is_file():
            self._open_with_os(report_path)

    def _on_open_eval_plot(self) -> None:
        item = self.list_eval_plots.currentItem()
        if item is None:
            return
        path_str = str(item.data(QtCore.Qt.UserRole) or "")
        if not path_str:
            return
        p = Path(path_str)
        if p.is_file():
            self._open_with_os(p)

    # ------------------------------------------------------------------
    # Section 4 logic: training command preview
    # ------------------------------------------------------------------

    def _on_browse_dataset(self) -> None:
        current = self.ent_train_data.text().strip() or str(PROJECT_ROOT)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ST-LRPS Training Dataset",
            current,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)",
        )
        if path:
            self.ent_train_data.setText(path)
            self._refresh_command_preview()

    def _on_browse_train_out(self) -> None:
        current = self.ent_train_out.text().strip() or str(DEFAULT_RUNS_ROOT)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Training Output Directory", current
        )
        if path:
            self.ent_train_out.setText(path)
            self._refresh_command_preview()

    def _apply_drift_preset(self) -> None:
        """Apply the angular-drift mitigation preset to the form."""
        try:
            self.sp_epochs.setValue(250)
            self.sp_batch.setValue(16384)
            self.sp_hidden.setValue(512)
            self.sp_depth.setValue(6)
            self.cb_activation.setCurrentText("sine")
            self.sp_dir_loss_weight.setValue(0.20)
            self.sp_dir_loss_start.setValue(10)
            self.sp_dir_loss_ramp.setValue(40)
            self.ent_dir_loss_floor.setText("1e-7")
            self.chk_alt_balanced.setChecked(True)
            self.chk_radial_cross.setChecked(True)
            self.sp_radial_w.setValue(0.05)
            self.sp_cross_w.setValue(0.10)
            self.chk_residual.setChecked(True)
            self.sp_n_bands.setValue(3)
            self.sp_grad_acc.setValue(2)
            self.cb_best_metric.setCurrentText("hybrid")
            self.sp_hybrid_alpha.setValue(0.30)
            self.chk_preload.setChecked(True)
            self._refresh_command_preview()
        except Exception:
            pass

    def _build_training_command(self) -> List[str]:
        cmd: List[str] = [sys.executable, "-m", TRAIN_MODULE]

        data_path = self.ent_train_data.text().strip()
        if data_path:
            cmd.extend(["--data", data_path])

        out_dir = self.ent_train_out.text().strip()
        if out_dir:
            cmd.extend(["--out", out_dir])

        cmd.extend(["--epochs", str(int(self.sp_epochs.value()))])
        cmd.extend(["--batch-size", str(int(self.sp_batch.value()))])
        cmd.extend(["--hidden", str(int(self.sp_hidden.value()))])
        cmd.extend(["--depth", str(int(self.sp_depth.value()))])
        cmd.extend(["--activation", str(self.cb_activation.currentText())])

        cmd.extend(["--direction-loss-weight", f"{float(self.sp_dir_loss_weight.value()):g}"])
        cmd.extend(["--direction-loss-start-epoch", str(int(self.sp_dir_loss_start.value()))])
        cmd.extend(["--direction-loss-ramp-epochs", str(int(self.sp_dir_loss_ramp.value()))])

        floor_raw = self.ent_dir_loss_floor.text().strip()
        if floor_raw:
            try:
                _ = float(floor_raw)
                cmd.extend(["--direction-loss-floor-abs", floor_raw])
            except Exception:
                # Skip invalid value; user can correct it
                pass

        if self.chk_alt_balanced.isChecked():
            cmd.append("--use-altitude-balanced-loss")
            cmd.extend(["--altitude-bin-width-km", "50"])

        if self.chk_radial_cross.isChecked():
            cmd.append("--use-radial-cross-loss")
            cmd.extend(["--radial-loss-weight", f"{float(self.sp_radial_w.value()):g}"])
            cmd.extend(["--cross-loss-weight", f"{float(self.sp_cross_w.value()):g}"])

        if self.chk_residual.isChecked():
            cmd.append("--use-residual-blocks")
        cmd.extend(["--n-bands", str(int(self.sp_n_bands.value()))])
        cmd.extend(["--grad-accumulation-steps", str(int(self.sp_grad_acc.value()))])

        cmd.extend(["--best-metric", str(self.cb_best_metric.currentText())])
        cmd.extend(["--hybrid-direction-alpha", f"{float(self.sp_hybrid_alpha.value()):g}"])

        if self.chk_preload.isChecked():
            cmd.append("--preload-data")

        return [str(x) for x in cmd]

    def _refresh_command_preview(self) -> None:
        try:
            cmd = self._build_training_command()
        except Exception as exc:
            self.txt_train_cmd.setPlainText(f"# Could not build command: {exc}")
            return
        try:
            if os.name == "nt":
                rendered = subprocess.list2cmdline(cmd)
            else:
                rendered = shlex.join(cmd)
        except Exception:
            rendered = " ".join(cmd)
        self.txt_train_cmd.setPlainText(rendered)

    def _copy_command_preview(self) -> None:
        text = self.txt_train_cmd.toPlainText()
        if text.strip():
            try:
                QtWidgets.QApplication.clipboard().setText(text)
            except Exception:
                pass


# =============================================================================
# Manual test entry point
# =============================================================================

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    page = SurrogateStudioPage()
    page.resize(1100, 900)
    page.show()
    sys.exit(app.exec())
