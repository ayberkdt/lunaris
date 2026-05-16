# LUNAR_SIMULATION/ui_parts/monte_carlo_page.py
# -*- coding: utf-8 -*-
"""
Monte Carlo Analysis Page (Page 7)
====================================

Provides a dedicated PySide6 page for configuring, launching, and monitoring
Monte Carlo orbital uncertainty propagation runs.

Layout
------
The page is split into two workspace tabs:

1. **Setup & Run**
   Left column  (60%) — scrollable configuration cards:
     - Ensemble
     - State uncertainty
     - Spacecraft uncertainty
     - Backend / integration
     - Output / impact settings
   Right column (40%) — run controls + live metrics

2. **Result Analysis**
   A dedicated post-processing workspace for loading Monte Carlo archives,
   computing statistics, previewing plots, and exporting a PDF report.

Integration with the rest of the application
---------------------------------------------
- ``get_data()``   → dict fed to ``build_mc_command()`` in command_builder.py
- ``load_data()``  → called by session_persistence to restore a saved profile
- ``update_results()`` → called by MainWindow after the MC subprocess finishes
  and the output file has been read back
- ``update_progress()`` → called by MainWindow for human-readable MC log lines
- ``update_progress_payload()`` → called by MainWindow for structured MC
  progress updates containing percent, stage, scenario counts, and ETA
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from .ui_commons import THEME, NumericDragLineEdit, ToggleSwitch, get_icon
    from .monte_carlo_analysis_panel import MonteCarloAnalysisPanel
    from .force_models_page import ST_LRPS_RUNS_DIR, list_st_lrps_model_dirs
except ImportError:
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("Run as:  python -m ui_parts.monte_carlo_page", file=sys.stderr)
        raise SystemExit(2)
    raise


# =============================================================================
# 1.                       STATE DATACLASS
# =============================================================================

@dataclass
class UIMonteCarloConfig:
    """
    Mutable mirror of ``common.montecarlo_defs.MonteCarloConfig`` for the UI.

    All values are kept as plain Python types so the page can safely serialize
    them to JSON (for session persistence) and pass them to the CLI argument
    builder without importing the heavy backend modules.
    """
    # Ensemble
    n_samples: int = 500
    seed: int = 42

    # State uncertainty
    sigma_r_m: float = 500.0    # position 1-sigma [m]
    sigma_v_m_s: float = 0.5    # velocity 1-sigma [m/s]

    # Spacecraft uncertainty (0 = deterministic)
    sigma_mass_kg: float = 0.0
    sigma_area_m2: float = 0.0
    sigma_cd: float = 0.0
    sigma_cr: float = 0.0

    # Backend
    use_gpu: bool = True
    gpu_device_id: int = 0
    gpu_sh_degree: int = 10        # 0-24
    gpu_threads_per_block: int = 128
    gravity_mode_override: str = "follow_mission"
    st_lrps_model_dir: str = ""

    # Integration (GPU RK4 fixed-step)
    dt_s: float = 60.0             # RK4 step [s]
    max_vram_gb: float = 4.0

    # Output
    output_format: str = "hdf5"    # "hdf5" or "npz"
    output_path: str = "mc_results/mc_output.h5"

    # Impact detection
    impact_alt_km: float = 0.0


# =============================================================================
# 2.                       HELPER WIDGETS
# =============================================================================

def _detect_cuda_available() -> bool:
    """
    Best-effort CUDA availability probe for the MC page defaults.

    The page should open safely even on machines without CUDA.  We therefore
    treat any import/probe failure as "GPU unavailable" and simply default the
    backend toggle to CPU.
    """

    try:
        from numba import cuda  # type: ignore

        return bool(cuda.is_available())
    except Exception:
        return False


def _preferred_output_suffix(fmt: str) -> str:
    """Return the canonical filename suffix for the selected MC archive format."""

    return ".npz" if str(fmt).strip().lower() == "npz" else ".h5"


def _normalize_output_path_for_format(path_text: str, fmt: str) -> str:
    """
    Keep the visible output path aligned with the chosen archive format.

    Users often switch the combo box after accepting the default path.  Updating
    the suffix here avoids confusing cases like selecting ``npz`` while still
    writing to an ``.h5``-looking filename.
    """

    raw = str(path_text).strip()
    suffix = _preferred_output_suffix(fmt)
    if not raw:
        return f"mc_results/mc_output{suffix}"

    current = Path(raw)
    lower_name = current.name.lower()
    if lower_name.endswith((".h5", ".hdf5", ".npz")):
        if lower_name.endswith(".hdf5"):
            return str(current.with_name(current.name[: -len(".hdf5")] + suffix))
        return str(current.with_suffix(suffix))
    return raw

def _card(title: str) -> QtWidgets.QGroupBox:
    """Styled card GroupBox following the project-wide QSS pattern."""
    gb = QtWidgets.QGroupBox(title)
    gb.setStyleSheet(f"""
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
            color: {THEME['fg_main']};
            font-weight: 700;
            font-size: 10pt;
        }}
    """)
    return gb


def _label(text: str, muted: bool = False) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    if muted:
        lbl.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
    return lbl


def _metric_row(key: str, value: str = "—") -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    row.addWidget(_label(key, muted=True))
    row.addStretch(1)
    val_lbl = QtWidgets.QLabel(value)
    val_lbl.setAlignment(QtCore.Qt.AlignRight)
    val_lbl.setObjectName("metricValue")
    row.addWidget(val_lbl)
    return row, val_lbl


def _format_clock_span(seconds: Optional[float]) -> str:
    """
    Convert a duration in seconds to a compact human-readable clock string.

    The progress panel needs short, scan-friendly time stamps rather than the
    verbose natural-language durations usually used in message boxes.  Values
    are therefore rendered as ``MM:SS`` or ``H:MM:SS`` depending on span.
    """

    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0.0:
        return "—"

    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# =============================================================================
# 3.                       MAIN PAGE WIDGET
# =============================================================================

class MonteCarloPage(QtWidgets.QWidget):
    """
    Page 7: Monte Carlo Analysis — configuration + live metrics.

    Signals
    -------
    run_requested :
        Emitted when the user clicks "Run Monte Carlo".  The main window
        collects all page states, builds the CLI command, and spawns the
        backend process.
    """

    run_requested = QtCore.Signal()

    def __init__(
        self,
        mc_cfg: Optional[UIMonteCarloConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if mc_cfg is None:
            mc_cfg = UIMonteCarloConfig(use_gpu=_detect_cuda_available())
        self.mc_cfg = mc_cfg
        self._last_progress_payload: Dict[str, Any] = {}
        self._build_ui()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: 1px solid {THEME['border']};
                border-radius: 10px;
                background: transparent;
                margin-top: 6px;
            }}
            QTabBar::tab {{
                background: {THEME['bg_card']};
                color: {THEME['fg_muted']};
                padding: 9px 16px;
                margin-right: 4px;
                border: 1px solid {THEME['border']};
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
            QTabBar::tab:hover {{
                background: {THEME['bg_entry']};
                color: {THEME['fg_main']};
            }}
            QTabBar::tab:selected {{
                background: {THEME['bg_entry']};
                color: {THEME['fg_main']};
                border: 1px solid {THEME['border']};
                border-bottom-color: {THEME['bg_space']};
                font-weight: 600;
            }}
            """
        )
        root.addWidget(self.tabs, 1)

        run_tab = QtWidgets.QWidget()
        run_root = QtWidgets.QHBoxLayout(run_tab)
        run_root.setContentsMargins(0, 0, 0, 0)
        run_root.setSpacing(16)

        # ----- Left: scrollable configuration --------------------------------
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)

        left_layout.addWidget(self._card_ensemble())
        left_layout.addWidget(self._card_state_uncertainty())
        left_layout.addWidget(self._card_spacecraft_uncertainty())
        left_layout.addWidget(self._card_backend())
        left_layout.addWidget(self._card_integration())
        left_layout.addWidget(self._card_output())
        left_layout.addWidget(self._card_impact())
        left_layout.addStretch(1)

        left_scroll.setWidget(left_container)
        run_root.addWidget(left_scroll, 6)

        # ----- Right: run controls + metrics ---------------------------------
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        right_layout.addWidget(self._card_run_controls())
        right_layout.addWidget(self._card_metrics())
        right_layout.addStretch(1)

        run_root.addWidget(right_widget, 4)

        self.analysis_panel = MonteCarloAnalysisPanel(parent=self)

        self.tabs.addTab(run_tab, "Setup & Run")
        self.tabs.addTab(self.analysis_panel, "Result Analysis")

    # -------------------------------------------------------------------------
    # Configuration cards
    # -------------------------------------------------------------------------

    def _card_ensemble(self) -> QtWidgets.QGroupBox:
        gb = _card("Ensemble")
        grid = QtWidgets.QGridLayout(gb)
        grid.setContentsMargins(16, 20, 16, 16)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("Number of Samples:"), 0, 0)
        self.ent_n_samples = NumericDragLineEdit(
            str(self.mc_cfg.n_samples),
            step=50, min_value=2, max_value=100_000, decimals=0,
        )
        self.ent_n_samples.setToolTip("Total number of Monte Carlo trajectories (N ≥ 2)")
        grid.addWidget(self.ent_n_samples, 0, 1)

        grid.addWidget(_label("Random Seed:"), 1, 0)
        self.ent_seed = NumericDragLineEdit(
            str(self.mc_cfg.seed),
            step=1, min_value=0, max_value=2**31 - 1, decimals=0,
        )
        self.ent_seed.setToolTip("Seed for numpy.random.default_rng — ensures reproducibility")
        grid.addWidget(self.ent_seed, 1, 1)

        return gb

    def _card_state_uncertainty(self) -> QtWidgets.QGroupBox:
        gb = _card("Initial State Uncertainty  (1-σ, Isotropic)")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(10)

        desc = _label(
            "Gaussian perturbations are applied to the nominal initial state\n"
            "via Y₀ = nominal + L·z,  z~N(0,I),  L = chol(P₀).",
            muted=True,
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("Position σᵣ  [m]:"), 0, 0)
        self.ent_sigma_r = NumericDragLineEdit(
            str(self.mc_cfg.sigma_r_m),
            step=100, min_value=0, max_value=1e7, decimals=1,
        )
        self.ent_sigma_r.setToolTip("1-sigma position uncertainty (isotropic, all axes)")
        grid.addWidget(self.ent_sigma_r, 0, 1)

        grid.addWidget(_label("Velocity σ_v  [m/s]:"), 1, 0)
        self.ent_sigma_v = NumericDragLineEdit(
            str(self.mc_cfg.sigma_v_m_s),
            step=0.1, min_value=0, max_value=1e4, decimals=3,
        )
        self.ent_sigma_v.setToolTip("1-sigma velocity uncertainty (isotropic, all axes)")
        grid.addWidget(self.ent_sigma_v, 1, 1)

        layout.addLayout(grid)

        self.lbl_sigma_summary = _label("", muted=True)
        layout.addWidget(self.lbl_sigma_summary)
        self._update_sigma_summary()

        self.ent_sigma_r.value_changed.connect(lambda _: self._update_sigma_summary())
        self.ent_sigma_v.value_changed.connect(lambda _: self._update_sigma_summary())

        return gb

    def _update_sigma_summary(self) -> None:
        try:
            r = float(self.ent_sigma_r.text())
            v = float(self.ent_sigma_v.text())
            self.lbl_sigma_summary.setText(
                f"Δr ≈ {r/1000:.3f} km   Δv ≈ {v:.3f} m/s"
            )
        except Exception:
            pass

    def _card_spacecraft_uncertainty(self) -> QtWidgets.QGroupBox:
        gb = _card("Spacecraft Property Uncertainty  (optional)")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(10)

        desc = _label("Zero σ = deterministic (no perturbation). Sampling uses truncated-normal (positive values only).", muted=True)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("σ Mass  [kg]:"), 0, 0)
        self.ent_sigma_mass = NumericDragLineEdit(str(self.mc_cfg.sigma_mass_kg), step=1, min_value=0, decimals=2)
        grid.addWidget(self.ent_sigma_mass, 0, 1)

        grid.addWidget(_label("σ Area  [m²]:"), 1, 0)
        self.ent_sigma_area = NumericDragLineEdit(str(self.mc_cfg.sigma_area_m2), step=0.01, min_value=0, decimals=3)
        grid.addWidget(self.ent_sigma_area, 1, 1)

        grid.addWidget(_label("σ C_D  [-]:"), 2, 0)
        self.ent_sigma_cd = NumericDragLineEdit(str(self.mc_cfg.sigma_cd), step=0.01, min_value=0, decimals=3)
        grid.addWidget(self.ent_sigma_cd, 2, 1)

        grid.addWidget(_label("σ C_R  [-]:"), 3, 0)
        self.ent_sigma_cr = NumericDragLineEdit(str(self.mc_cfg.sigma_cr), step=0.01, min_value=0, decimals=3)
        grid.addWidget(self.ent_sigma_cr, 3, 1)

        layout.addLayout(grid)
        return gb

    def _card_backend(self) -> QtWidgets.QGroupBox:
        gb = _card("Physics Backend")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(12)

        # GPU / CPU toggle row
        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.addWidget(_label("Use GPU Acceleration:"))
        self.toggle_gpu = ToggleSwitch()
        self.toggle_gpu.setChecked(self.mc_cfg.use_gpu)
        self.toggle_gpu.toggled.connect(self._on_backend_changed)
        toggle_row.addWidget(self.toggle_gpu)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)

        gravity_row = QtWidgets.QHBoxLayout()
        gravity_row.addWidget(_label("Central Gravity Source:"))
        self.cb_mc_gravity_mode = QtWidgets.QComboBox()
        self.cb_mc_gravity_mode.addItem("Follow Mission Setup", "follow_mission")
        self.cb_mc_gravity_mode.addItem("Force Classical Gravity", "classic_sh")
        self.cb_mc_gravity_mode.addItem("Force ST-LRPS Gravity", "st_lrps")
        self.cb_mc_gravity_mode.currentIndexChanged.connect(self._on_gravity_mode_changed)
        gravity_row.addWidget(self.cb_mc_gravity_mode, 1)
        layout.addLayout(gravity_row)

        gravity_hint = _label(
            "Use this when you want Monte Carlo to reuse the mission gravity setup "
            "or explicitly force the classical SH model versus the ST-LRPS model.",
            muted=True,
        )
        gravity_hint.setWordWrap(True)
        layout.addWidget(gravity_hint)

        # ST-LRPS surrogate selection is only relevant when MC explicitly forces
        # the surrogate backend.  Leaving it blank intentionally falls back to
        # the global Force Models page setting.
        self.st_lrps_config_frame = QtWidgets.QFrame()
        self.st_lrps_config_frame.setStyleSheet(f"""
            QFrame {{
                border: 1px solid rgba(245, 158, 11, 0.35);
                border-radius: 10px;
                background: rgba(245, 158, 11, 0.055);
            }}
            QLineEdit {{
                background: {THEME['bg_entry']};
                border: 1px solid {THEME['border']};
                border-radius: 6px;
                padding: 5px 8px;
                color: {THEME['fg_main']};
            }}
            QLineEdit:focus {{
                border-color: #f59e0b;
            }}
        """)
        st_lrps_layout = QtWidgets.QVBoxLayout(self.st_lrps_config_frame)
        st_lrps_layout.setContentsMargins(12, 10, 12, 12)
        st_lrps_layout.setSpacing(8)

        st_lrps_title_row = QtWidgets.QHBoxLayout()
        st_lrps_title = _label("ST-LRPS Model Run")
        st_lrps_title.setStyleSheet("color: #f8d48a; font-weight: 800;")
        st_lrps_title_row.addWidget(st_lrps_title)
        st_lrps_title_row.addStretch(1)
        st_lrps_layout.addLayout(st_lrps_title_row)

        st_lrps_help = _label(
            "Select the trained ST-LRPS run used only by this Monte Carlo run. "
            "If left empty, MC falls back to the main Force Models ST-LRPS directory.",
            muted=True,
        )
        st_lrps_help.setWordWrap(True)
        st_lrps_layout.addWidget(st_lrps_help)

        st_lrps_path_row = QtWidgets.QHBoxLayout()
        self.ent_mc_st_lrps_model_dir = QtWidgets.QLineEdit(self.mc_cfg.st_lrps_model_dir)
        self.ent_mc_st_lrps_model_dir.setPlaceholderText(
            str(ST_LRPS_RUNS_DIR / "<trained_run>")
        )
        self.ent_mc_st_lrps_model_dir.setToolTip(
            "Path to a trained ST-LRPS run directory. Usually one folder under "
            "surrogate_gravity_model/runs."
        )
        btn_st_lrps_browse = QtWidgets.QPushButton("Browse...")
        btn_st_lrps_browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_muted"]))
        btn_st_lrps_browse.setCursor(QtCore.Qt.PointingHandCursor)
        btn_st_lrps_browse.clicked.connect(self._browse_st_lrps_model_dir)
        btn_st_lrps_latest = QtWidgets.QPushButton("Use Latest")
        btn_st_lrps_latest.setIcon(get_icon("fa6s.clock-rotate-left", THEME["fg_muted"]))
        btn_st_lrps_latest.setCursor(QtCore.Qt.PointingHandCursor)
        btn_st_lrps_latest.clicked.connect(self._use_latest_st_lrps_model_dir)
        st_lrps_path_row.addWidget(self.ent_mc_st_lrps_model_dir, 1)
        st_lrps_path_row.addWidget(btn_st_lrps_browse)
        st_lrps_path_row.addWidget(btn_st_lrps_latest)
        st_lrps_layout.addLayout(st_lrps_path_row)

        layout.addWidget(self.st_lrps_config_frame)

        # GPU-specific frame (hidden when CPU)
        self.gpu_frame = QtWidgets.QFrame()
        self.gpu_frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {THEME['border']}; border-radius: 8px; padding: 4px; }}"
        )
        gpu_grid = QtWidgets.QGridLayout(self.gpu_frame)
        gpu_grid.setContentsMargins(12, 12, 12, 12)
        gpu_grid.setVerticalSpacing(8)
        gpu_grid.setHorizontalSpacing(12)

        gpu_grid.addWidget(_label("SH Degree on GPU  (0-24):"), 0, 0)
        self.ent_gpu_sh = NumericDragLineEdit(
            str(self.mc_cfg.gpu_sh_degree),
            step=1, min_value=0, max_value=24, decimals=0,
        )
        self.ent_gpu_sh.setToolTip(
            "Spherical-harmonic degree evaluated per CUDA thread.\n"
            "GPU kernel workspace is fixed at 26×26 → max degree 24.\n"
            "0 = point-mass only (fastest)."
        )
        gpu_grid.addWidget(self.ent_gpu_sh, 0, 1)

        gpu_grid.addWidget(_label("Threads/Block:"), 1, 0)
        self.ent_tpb = NumericDragLineEdit(
            str(self.mc_cfg.gpu_threads_per_block),
            step=32, min_value=32, max_value=1024, decimals=0,
        )
        self.ent_tpb.setToolTip(
            "CUDA launch width hint.\n"
            "The runtime aligns this value to the active device warp size and hardware limits."
        )
        gpu_grid.addWidget(self.ent_tpb, 1, 1)

        gpu_grid.addWidget(_label("GPU Device ID:"), 2, 0)
        self.ent_gpu_dev = NumericDragLineEdit(
            str(self.mc_cfg.gpu_device_id),
            step=1, min_value=0, max_value=7, decimals=0,
        )
        gpu_grid.addWidget(self.ent_gpu_dev, 2, 1)

        # GPU-only warning banner
        warn_lbl = _label(
            "GPU path supports SH <= 24, Sun/Earth third-body, Earth J2, SRP, and relativity. "
            "Surface-lighting, tides, and surrogate gravity automatically switch the run to the CPU full-fidelity backend.",
            muted=True,
        )
        warn_lbl.setWordWrap(True)
        gpu_grid.addWidget(warn_lbl, 3, 0, 1, 2)

        layout.addWidget(self.gpu_frame)
        gravity_mode_index = self.cb_mc_gravity_mode.findData(self.mc_cfg.gravity_mode_override)
        if gravity_mode_index < 0:
            gravity_mode_index = 0
        self.cb_mc_gravity_mode.setCurrentIndex(gravity_mode_index)
        self._on_gravity_mode_changed()
        self._on_backend_changed(self.mc_cfg.use_gpu)

        # CPU hint
        self.cpu_hint = _label(
            "CPU path uses the full-fidelity propagator (all physics flags).\n"
            "It mirrors the main mission-analysis physics contract for accuracy-first runs.",
            muted=True,
        )
        self.cpu_hint.setWordWrap(True)
        layout.addWidget(self.cpu_hint)
        self.cpu_hint.setVisible(not self.mc_cfg.use_gpu)

        return gb

    def _on_backend_changed(self, gpu_on: bool) -> None:
        self.gpu_frame.setVisible(gpu_on)
        if hasattr(self, "cpu_hint"):
            self.cpu_hint.setVisible(not gpu_on)

    def _on_gravity_mode_changed(self, *_args: Any) -> None:
        """
        Show MC-specific ST-LRPS controls only when the surrogate backend is forced.

        The global Force Models page remains the default source of truth.  This
        panel is an explicit per-Monte-Carlo override for experiments where the
        operator wants to compare different trained surrogate runs.
        """

        is_st_lrps = str(self.cb_mc_gravity_mode.currentData() or "") == "st_lrps"
        if hasattr(self, "st_lrps_config_frame"):
            self.st_lrps_config_frame.setVisible(is_st_lrps)

    def _browse_st_lrps_model_dir(self) -> None:
        """Open a folder chooser rooted at the surrogate run directory."""

        current = self.ent_mc_st_lrps_model_dir.text().strip()
        if current:
            start_path = Path(current).expanduser()
            if start_path.is_file():
                start_path = start_path.parent
            if not start_path.exists():
                start_path = ST_LRPS_RUNS_DIR
        else:
            start_path = ST_LRPS_RUNS_DIR

        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select ST-LRPS Run Directory",
            str(start_path),
        )
        if path:
            self.ent_mc_st_lrps_model_dir.setText(str(Path(path).expanduser().resolve()))

    def _use_latest_st_lrps_model_dir(self) -> None:
        """Fill the MC ST-LRPS directory with the newest valid trained ST-LRPS run."""

        runs = list_st_lrps_model_dirs(ST_LRPS_RUNS_DIR)
        if not runs:
            QtWidgets.QMessageBox.information(
                self,
                "No ST-LRPS Runs Found",
                "No valid lunar ST-LRPS run directory was found under:\n"
                f"{ST_LRPS_RUNS_DIR}",
            )
            return
        self.ent_mc_st_lrps_model_dir.setText(str(runs[0]))

    def _card_integration(self) -> QtWidgets.QGroupBox:
        gb = _card("Integration  (GPU RK4 and batching)")
        grid = QtWidgets.QGridLayout(gb)
        grid.setContentsMargins(16, 20, 16, 16)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("RK4 Step  dt [s]:"), 0, 0)
        self.ent_dt = NumericDragLineEdit(
            str(self.mc_cfg.dt_s),
            step=10, min_value=0.1, max_value=3600, decimals=1,
        )
        self.ent_dt.setToolTip(
            "Fixed time-step for the GPU RK4 integrator.\n"
            "60 s is adequate for LEO/LLO; reduce for high-eccentricity orbits."
        )
        grid.addWidget(self.ent_dt, 0, 1)
        grid.addWidget(_label("s"), 0, 2)

        grid.addWidget(_label("VRAM Budget  [GB]:"), 1, 0)
        self.ent_vram = NumericDragLineEdit(
            str(self.mc_cfg.max_vram_gb),
            step=0.5, min_value=0.5, max_value=80.0, decimals=1,
        )
        self.ent_vram.setToolTip(
            "Maximum GPU memory used per sub-batch.\n"
            "Large ensembles are automatically tiled to stay within this budget."
        )
        grid.addWidget(self.ent_vram, 1, 1)
        grid.addWidget(_label("GB"), 1, 2)

        return gb

    def _card_output(self) -> QtWidgets.QGroupBox:
        gb = _card("Output")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(10)

        # Format
        fmt_row = QtWidgets.QHBoxLayout()
        fmt_row.addWidget(_label("Format:"))
        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.addItems(["hdf5", "npz"])
        self.cb_format.setCurrentText(self.mc_cfg.output_format)
        self.cb_format.currentTextChanged.connect(self._on_output_format_changed)
        self.cb_format.setStyleSheet(f"""
            QComboBox {{
                background: {THEME['bg_entry']};
                border: 1px solid {THEME['border']};
                border-radius: 6px;
                padding: 4px 8px;
                color: {THEME['fg_main']};
            }}
        """)
        fmt_row.addWidget(self.cb_format)
        fmt_row.addStretch(1)
        layout.addLayout(fmt_row)

        # Output path
        path_row = QtWidgets.QHBoxLayout()
        self.ent_output = QtWidgets.QLineEdit(self.mc_cfg.output_path)
        self.ent_output.setStyleSheet(f"""
            QLineEdit {{
                background: {THEME['bg_entry']};
                border: 1px solid {THEME['border']};
                border-radius: 6px;
                padding: 5px 8px;
                color: {THEME['fg_main']};
            }}
            QLineEdit:focus {{ border-color: {THEME['accent']}; }}
        """)
        self.ent_output.setPlaceholderText("mc_results/mc_output.h5")
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.setFixedHeight(30)
        btn_browse.clicked.connect(self._browse_output)
        path_row.addWidget(self.ent_output, 1)
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)

        self._on_output_format_changed(self.cb_format.currentText())

        return gb

    def _browse_output(self) -> None:
        fmt = self.cb_format.currentText()
        ext_filter = "HDF5 Files (*.h5 *.hdf5)" if fmt == "hdf5" else "NumPy Files (*.npz)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save MC Output", self.ent_output.text(), ext_filter
        )
        if path:
            self.ent_output.setText(path)

    def _on_output_format_changed(self, fmt: str) -> None:
        """
        Keep the output placeholder and default path consistent with format changes.

        This is intentionally lightweight: user-chosen custom basenames are
        preserved, but legacy/default extensions are rewritten to the newly
        selected archive format.
        """

        normalized = _normalize_output_path_for_format(self.ent_output.text(), fmt)
        self.ent_output.setPlaceholderText(_normalize_output_path_for_format("", fmt))
        if normalized != self.ent_output.text().strip():
            self.ent_output.setText(normalized)

    def _card_impact(self) -> QtWidgets.QGroupBox:
        gb = _card("Impact Detection")
        grid = QtWidgets.QGridLayout(gb)
        grid.setContentsMargins(16, 20, 16, 16)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("Impact Altitude Threshold:"), 0, 0)
        self.ent_impact_alt = NumericDragLineEdit(
            str(self.mc_cfg.impact_alt_km),
            step=1, min_value=0, max_value=100, decimals=1,
        )
        self.ent_impact_alt.setToolTip(
            "Samples crossing below this altitude above the mean lunar surface\n"
            "are flagged as impacted and removed from further propagation."
        )
        grid.addWidget(self.ent_impact_alt, 0, 1)
        grid.addWidget(_label("km"), 0, 2)

        return gb

    # -------------------------------------------------------------------------
    # Run controls + metrics panels
    # -------------------------------------------------------------------------

    def _card_run_controls(self) -> QtWidgets.QGroupBox:
        gb = _card("Run Monte Carlo")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(10)

        # Status badge row
        status_row = QtWidgets.QHBoxLayout()
        self.badge_mc = QtWidgets.QLabel("IDLE")
        self.badge_mc.setObjectName("statusBadge")
        self.badge_mc.setAlignment(QtCore.Qt.AlignCenter)
        self.badge_mc.setFixedHeight(24)
        self.badge_mc.setContentsMargins(10, 4, 10, 4)
        self.badge_mc.setProperty("kind", "info")
        self.badge_mc.setStyleSheet(
            f"border-radius: 10px; border: 1px solid {THEME['accent']};"
            f" background: rgba(59,130,246,0.1); color: {THEME['accent']};"
            f" font-weight: 700; padding: 0 8px;"
        )
        status_row.addWidget(self.badge_mc)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        # Progress bar
        self.progress_mc = QtWidgets.QProgressBar()
        self.progress_mc.setRange(0, 100)
        self.progress_mc.setValue(0)
        self.progress_mc.setTextVisible(True)
        self.progress_mc.setFormat("Waiting…")
        self.progress_mc.setFixedHeight(16)
        self.progress_mc.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                background: {THEME['bg_entry']};
                color: {THEME['fg_main']};
                font-size: 8pt;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {THEME['accent']};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_mc)

        self.lbl_progress_summary = _label("Waiting for run", muted=False)
        self.lbl_progress_summary.setStyleSheet(
            f"color: {THEME['fg_main']}; font-size: 9.5pt; font-weight: 600;"
        )
        self.lbl_progress_summary.setWordWrap(True)
        layout.addWidget(self.lbl_progress_summary)

        self.lbl_progress_meta = _label("No active Monte Carlo run", muted=True)
        self.lbl_progress_meta.setWordWrap(True)
        self.lbl_progress_meta.setStyleSheet(
            f"color: {THEME['fg_muted']}; font-size: 9pt;"
        )
        layout.addWidget(self.lbl_progress_meta)

        # Live log (last few lines)
        self.txt_progress = QtWidgets.QPlainTextEdit()
        self.txt_progress.setReadOnly(True)
        self.txt_progress.setFixedHeight(80)
        self.txt_progress.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {THEME['bg_log']};
                color: {THEME['fg_muted']};
                border: 1px solid {THEME['border']};
                border-radius: 6px;
                font-family: Consolas, monospace;
                font-size: 9pt;
                padding: 4px;
            }}
        """)
        self.txt_progress.setPlaceholderText("MC engine output appears here…")
        layout.addWidget(self.txt_progress)

        # Buttons row
        btn_row = QtWidgets.QHBoxLayout()

        self.btn_run_mc = QtWidgets.QPushButton("  Run Monte Carlo")
        self.btn_run_mc.setObjectName("primaryBtn")
        self.btn_run_mc.setIcon(get_icon("fa6s.dice", THEME["fg_main"]))
        self.btn_run_mc.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_run_mc.setFixedHeight(36)
        self.btn_run_mc.clicked.connect(self._on_run_clicked)

        self.btn_open_folder = QtWidgets.QPushButton("  Open Folder")
        self.btn_open_folder.setIcon(get_icon("fa6s.folder-open", THEME["fg_muted"]))
        self.btn_open_folder.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_open_folder.setFixedHeight(36)
        self.btn_open_folder.clicked.connect(self._open_output_folder)

        btn_row.addWidget(self.btn_run_mc, 2)
        btn_row.addWidget(self.btn_open_folder, 1)
        layout.addLayout(btn_row)

        return gb

    def _on_run_clicked(self) -> None:
        self._set_running(True)
        self.clear_results()
        self.txt_progress.clear()
        self.txt_progress.appendPlainText("[MC] Queuing run…")
        self.run_requested.emit()

    def _open_output_folder(self) -> None:
        path = Path(self.ent_output.text()).expanduser().resolve().parent
        if path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        else:
            QtWidgets.QMessageBox.information(
                self, "Folder Not Found",
                f"Output directory does not exist yet:\n{path}"
            )

    def _card_metrics(self) -> QtWidgets.QGroupBox:
        gb = _card("Results  —  Last Run")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(6)

        self._metric_labels: Dict[str, QtWidgets.QLabel] = {}

        def _add(key: str, label: str) -> None:
            row_layout, val_lbl = _metric_row(label)
            layout.addLayout(row_layout)
            self._metric_labels[key] = val_lbl

        _add("n_samples",      "N Samples")
        _add("n_impacts",      "N Impacts")
        _add("p_impact",       "Impact Probability")
        _add("p_impact_ci95",  "95% CI")
        _add("t_impact_mean",  "Mean Impact Time")
        _add("alt_mean_0",     "Initial Mean Altitude")
        _add("alt_std_0",      "Initial Alt 1-σ")
        _add("alt_mean_f",     "Final Mean Altitude")
        _add("alt_std_f",      "Final Alt 1-σ")
        _add("wall_time",      "Wall Time")
        _add("backend",        "Backend")

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet(f"color: {THEME['border']};")
        layout.addWidget(sep)

        self.btn_open_report = QtWidgets.QPushButton("  Open PDF Report")
        self.btn_open_report.setIcon(get_icon("fa6s.file-pdf", THEME["fg_muted"]))
        self.btn_open_report.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_open_report.setFixedHeight(32)
        self.btn_open_report.setEnabled(False)
        self.btn_open_report.clicked.connect(self._open_report)
        layout.addWidget(self.btn_open_report)

        self._last_report_path: Optional[str] = None
        return gb

    def _open_report(self) -> None:
        if self._last_report_path and Path(self._last_report_path).exists():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(self._last_report_path)
            )

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        """
        Toggle the page between idle and active-run states.

        The structured progress payload is the authoritative source once the
        backend starts emitting it.  Until then, the page shows a restrained
        warm-up state so the user immediately sees that the request was queued.
        """

        self.btn_run_mc.setEnabled(not running)
        if running:
            total = max(1, self._parse_int(self.ent_n_samples.text(), self.mc_cfg.n_samples))
            self._last_progress_payload = {}
            self._set_badge("RUNNING", accent=THEME["success"])
            self.progress_mc.setRange(0, 0)   # indeterminate until first structured payload
            self.progress_mc.setValue(0)
            self.progress_mc.setFormat("Preparing…")
            self.lbl_progress_summary.setText("Preparing Monte Carlo ensemble")
            self.lbl_progress_meta.setText(f"0 / {total} scenarios | Waiting for backend")
        else:
            if self.progress_mc.maximum() == 0:
                self.progress_mc.setRange(0, 1000)

    def _set_badge(self, text: str, accent: str = "") -> None:
        self.badge_mc.setText(text)
        c = accent or THEME["accent"]
        self.badge_mc.setStyleSheet(
            f"border-radius: 10px; border: 1px solid {c};"
            f" background: transparent; color: {c};"
            f" font-weight: 700; padding: 0 8px;"
        )

    def update_progress(self, line: str) -> None:
        """
        Append a human-readable MC log line to the page-local mini log.

        Structured progress payloads are handled by ``update_progress_payload``.
        This method only keeps the operator-facing narrative lines visible and
        retains a lightweight fallback progress parser for legacy output.
        """

        stripped = line.rstrip()
        if stripped.startswith("[MC_PROGRESS]") or stripped.startswith("[MC_METRICS]"):
            return

        self.txt_progress.appendPlainText(stripped)
        sb = self.txt_progress.verticalScrollBar()
        sb.setValue(sb.maximum())

        # Keep a minimal legacy fallback so older batch-only output still moves
        # the progress bar in a sensible way during development/debug runs.
        low = stripped.lower()
        if "batch" in low and "/" in stripped:
            try:
                parts = stripped.split()[1].split("/")
                done, total = int(parts[0]), int(parts[1])
                pct = float(done) / float(max(total, 1))
                self.progress_mc.setRange(0, 1000)
                self.progress_mc.setValue(int(round(pct * 1000.0)))
                self.progress_mc.setFormat(f"{pct * 100.0:.1f}%")
                self.lbl_progress_summary.setText("Propagating scenarios")
                self.lbl_progress_meta.setText(f"Batch {done}/{total}")
            except Exception:
                pass

    def update_progress_payload(self, payload: Dict[str, Any]) -> None:
        """
        Render a structured backend progress payload in the MC control card.

        The backend emits machine-readable progress updates so the page can show
        a professional progress experience: phase label, overall percent,
        scenario-total context, batch position, and ETA.  This avoids brittle
        parsing of free-form log text.
        """

        self._last_progress_payload = dict(payload)

        stage = str(payload.get("stage", "propagating")).strip().lower()
        percent = max(0.0, min(100.0, float(payload.get("percent", 0.0) or 0.0)))
        fraction = max(0.0, min(1.0, float(payload.get("fraction", percent / 100.0) or 0.0)))
        total_samples = max(1, int(payload.get("total_samples", self._parse_int(self.ent_n_samples.text(), 1)) or 1))
        done_samples_raw = float(payload.get("done_samples", 0.0) or 0.0)
        done_samples_raw = max(0.0, min(float(total_samples), done_samples_raw))
        done_samples = int(math.floor(done_samples_raw + 1.0e-9))
        approx_done = abs(done_samples_raw - round(done_samples_raw)) > 1.0e-6
        batch_index = int(payload["batch_index"]) if "batch_index" in payload and payload.get("batch_index") is not None else None
        batch_count = int(payload["batch_count"]) if "batch_count" in payload and payload.get("batch_count") is not None else None
        eta_s = payload.get("eta_s")
        elapsed_s = payload.get("elapsed_s")
        backend = str(payload.get("backend", "") or "").strip().upper()
        detail = str(payload.get("detail", "") or "").strip()

        stage_summary = {
            "sampling": "Preparing Monte Carlo ensemble",
            "propagating": "Propagating scenarios",
            "writing": "Writing ensemble results",
            "finalizing": "Finalizing Monte Carlo archive",
        }.get(stage, "Running Monte Carlo")

        badge_text = {
            "sampling": "PREPARING",
            "propagating": "RUNNING",
            "writing": "WRITING",
            "finalizing": "FINALIZING",
        }.get(stage, "RUNNING")
        self._set_badge(badge_text, accent=THEME["success"] if stage == "propagating" else THEME["accent"])

        self.progress_mc.setRange(0, 1000)
        self.progress_mc.setValue(int(round(fraction * 1000.0)))
        self.progress_mc.setFormat(f"{percent:.1f}%")

        summary_suffix = f" ({backend})" if backend and backend != "PENDING" else ""
        self.lbl_progress_summary.setText(stage_summary + summary_suffix)

        scenario_prefix = "~" if approx_done and stage == "propagating" else ""
        scenario_text = f"{scenario_prefix}{done_samples} / {total_samples} scenarios"
        meta_parts: List[str] = [scenario_text]
        if batch_index is not None and batch_count is not None and batch_index >= 1:
            meta_parts.append(f"Batch {batch_index}/{batch_count}")
        if eta_s is not None:
            meta_parts.append(f"ETA {_format_clock_span(float(eta_s))}")
        elif elapsed_s is not None:
            meta_parts.append(f"Elapsed {_format_clock_span(float(elapsed_s))}")
        if detail and stage != "propagating":
            meta_parts.append(detail)
        self.lbl_progress_meta.setText(" | ".join(meta_parts))

    def on_run_finished(
        self,
        exit_code: int,
        output_path: str,
        report_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Called by MainWindow when the MC subprocess exits.

        Parameters
        ----------
        exit_code : int
            Process exit code (0 = success).
        output_path : str
            Path to the HDF5/NPZ output file.
        report_path : str, optional
            Path to a generated PDF report (if available).
        metrics : dict, optional
            Pre-computed summary metrics to display.
        """
        self._set_running(False)
        if exit_code == 0:
            self._set_badge("DONE", accent=THEME["success"])
            total_samples = int((metrics or {}).get("n_samples", self._parse_int(self.ent_n_samples.text(), self.mc_cfg.n_samples)))
            self.progress_mc.setRange(0, 1000)
            self.progress_mc.setValue(1000)
            self.progress_mc.setFormat("100.0%")
            self.lbl_progress_summary.setText("Monte Carlo run completed")
            self.lbl_progress_meta.setText(f"{total_samples} / {total_samples} scenarios | Results ready")
            if metrics:
                self.update_results(metrics)
            if output_path:
                self.analysis_panel.set_result_path(output_path, auto_analyze=True)
                self.tabs.setCurrentWidget(self.analysis_panel)
            if report_path and Path(report_path).exists():
                self._last_report_path = report_path
                self.btn_open_report.setEnabled(True)
        else:
            self._set_badge("FAILED", accent=THEME["error"])
            if self.progress_mc.maximum() == 0:
                self.progress_mc.setRange(0, 1000)
            self.progress_mc.setFormat("Failed")
            self.lbl_progress_summary.setText("Monte Carlo run failed")
            if self._last_progress_payload:
                total_samples = max(1, int(self._last_progress_payload.get("total_samples", 1)))
                done_samples = int(min(total_samples, max(0.0, float(self._last_progress_payload.get("done_samples", 0.0) or 0.0))))
                self.lbl_progress_meta.setText(f"{done_samples} / {total_samples} scenarios | Review the execution log")
            else:
                self.lbl_progress_meta.setText("Review the execution log for the failure cause")

    def shutdown(self) -> None:
        """
        Stop background sub-components owned by the Monte Carlo page.

        The main window calls this during shutdown so the analysis workspace
        does not keep a background worker alive while the application exits.
        """

        if hasattr(self, "analysis_panel"):
            self.analysis_panel.shutdown()

    def update_results(self, metrics: Dict[str, Any]) -> None:
        """
        Populate the metrics panel from a dict returned by the MC engine.

        Expected keys (all optional — missing keys show '—'):
            n_samples, n_impacts, p_impact, p_impact_ci95,
            t_impact_mean_days, alt_mean_0_km, alt_std_0_km,
            alt_mean_f_km, alt_std_f_km, wall_time_s, backend
        """
        def _set(key: str, value: str) -> None:
            lbl = self._metric_labels.get(key)
            if lbl is not None:
                lbl.setText(value)

        _set("n_samples",     str(metrics.get("n_samples", "—")))
        _set("n_impacts",     str(metrics.get("n_impacts", "—")))

        p = metrics.get("p_impact")
        _set("p_impact", f"{p:.4f}" if p is not None else "—")

        ci = metrics.get("p_impact_ci95")
        if ci and len(ci) == 2:
            _set("p_impact_ci95", f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            _set("p_impact_ci95", "—")

        t_d = metrics.get("t_impact_mean_days")
        _set("t_impact_mean", f"{t_d:.3f} d" if t_d is not None else "—")

        _set("alt_mean_0", f"{metrics.get('alt_mean_0_km', '—'):.2f} km" if "alt_mean_0_km" in metrics else "—")
        _set("alt_std_0",  f"{metrics.get('alt_std_0_km',  '—'):.2f} km" if "alt_std_0_km"  in metrics else "—")
        _set("alt_mean_f", f"{metrics.get('alt_mean_f_km', '—'):.2f} km" if "alt_mean_f_km" in metrics else "—")
        _set("alt_std_f",  f"{metrics.get('alt_std_f_km',  '—'):.2f} km" if "alt_std_f_km"  in metrics else "—")

        wt = metrics.get("wall_time_s")
        _set("wall_time", f"{wt:.1f} s" if wt is not None else "—")
        _set("backend",   str(metrics.get("backend", "—")))

    def clear_results(self) -> None:
        """Reset all metric labels to '—'."""
        for lbl in self._metric_labels.values():
            lbl.setText("—")
        self.btn_open_report.setEnabled(False)
        self._last_report_path = None

    # -------------------------------------------------------------------------
    # Serialization (session persistence + command builder)
    # -------------------------------------------------------------------------

    def get_data(self) -> Dict[str, Any]:
        """Return current UI state as a plain dict (JSON-serializable)."""
        return {
            "n_samples":             self._parse_int(self.ent_n_samples.text(), 500),
            "seed":                  self._parse_int(self.ent_seed.text(), 42),
            "sigma_r_m":             self._parse_float(self.ent_sigma_r.text(), 500.0),
            "sigma_v_m_s":           self._parse_float(self.ent_sigma_v.text(), 0.5),
            "sigma_mass_kg":         self._parse_float(self.ent_sigma_mass.text(), 0.0),
            "sigma_area_m2":         self._parse_float(self.ent_sigma_area.text(), 0.0),
            "sigma_cd":              self._parse_float(self.ent_sigma_cd.text(), 0.0),
            "sigma_cr":              self._parse_float(self.ent_sigma_cr.text(), 0.0),
            "use_gpu":               bool(self.toggle_gpu.isChecked()),
            "gpu_device_id":         self._parse_int(self.ent_gpu_dev.text(), 0),
            "gpu_sh_degree":         self._parse_int(self.ent_gpu_sh.text(), 10),
            "gpu_threads_per_block": self._parse_int(self.ent_tpb.text(), 128),
            "gravity_mode_override": str(self.cb_mc_gravity_mode.currentData() or "follow_mission"),
            "st_lrps_model_dir":     self.ent_mc_st_lrps_model_dir.text().strip(),
            "dt_s":                  self._parse_float(self.ent_dt.text(), 60.0),
            "max_vram_gb":           self._parse_float(self.ent_vram.text(), 4.0),
            "output_format":         self.cb_format.currentText(),
            "output_path":           _normalize_output_path_for_format(
                self.ent_output.text(),
                self.cb_format.currentText(),
            ),
            "impact_alt_km":         self._parse_float(self.ent_impact_alt.text(), 0.0),
        }

    def load_data(self, data: Dict[str, Any]) -> None:
        """Restore UI state from a plain dict (e.g., loaded from JSON session)."""
        def _s(key: str, default) -> str:
            return str(data.get(key, default))

        self.ent_n_samples.setText(_s("n_samples", 500))
        self.ent_seed.setText(_s("seed", 42))
        self.ent_sigma_r.setText(_s("sigma_r_m", 500.0))
        self.ent_sigma_v.setText(_s("sigma_v_m_s", 0.5))
        self.ent_sigma_mass.setText(_s("sigma_mass_kg", 0.0))
        self.ent_sigma_area.setText(_s("sigma_area_m2", 0.0))
        self.ent_sigma_cd.setText(_s("sigma_cd", 0.0))
        self.ent_sigma_cr.setText(_s("sigma_cr", 0.0))
        self.toggle_gpu.setChecked(bool(data.get("use_gpu", True)))
        self.ent_gpu_dev.setText(_s("gpu_device_id", 0))
        self.ent_gpu_sh.setText(_s("gpu_sh_degree", 10))
        self.ent_tpb.setText(_s("gpu_threads_per_block", 128))
        gravity_mode = str(data.get("gravity_mode_override", "follow_mission") or "follow_mission")
        gravity_idx = self.cb_mc_gravity_mode.findData(gravity_mode)
        if gravity_idx < 0:
            gravity_idx = 0
        self.cb_mc_gravity_mode.setCurrentIndex(gravity_idx)
        self.ent_mc_st_lrps_model_dir.setText(str(data.get("st_lrps_model_dir", "") or ""))
        self._on_gravity_mode_changed()
        self.ent_dt.setText(_s("dt_s", 60.0))
        self.ent_vram.setText(_s("max_vram_gb", 4.0))
        fmt = str(data.get("output_format", "hdf5"))
        idx = self.cb_format.findText(fmt)
        if idx >= 0:
            self.cb_format.setCurrentIndex(idx)
        self.ent_output.setText(
            _normalize_output_path_for_format(
                str(data.get("output_path", "mc_results/mc_output.h5")),
                fmt,
            )
        )
        self.ent_impact_alt.setText(_s("impact_alt_km", 0.0))
        self._update_sigma_summary()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_float(text: str, default: float) -> float:
        try:
            return float(text)
        except Exception:
            return default

    @staticmethod
    def _parse_int(text: str, default: int) -> int:
        try:
            return int(float(text))
        except Exception:
            return default


# =============================================================================
# 4.                      STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Monte Carlo Page — Test")
    win.resize(1100, 750)
    win.setStyleSheet(f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};")

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    page = MonteCarloPage()

    def _on_run():
        print("[Test] run_requested signal received")
        print("[Test] get_data() =", page.get_data())

    page.run_requested.connect(_on_run)
    scroll.setWidget(page)
    win.setCentralWidget(scroll)
    win.show()
    sys.exit(app.exec())
