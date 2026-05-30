# ST_LRPS/ui_parts/force_models_page.py
# -*- coding: utf-8 -*-

"""
Force Models Page (UI Part) for ST-LRPS Studio.

This module defines the Force Models configuration page that lives inside the
MainWindow's page stack (e.g., Page 2). The page owns all force-model widgets
(toggles, indicators, and settings buttons) and exposes a small, explicit API
for the host window to read/write state.

Scope
-----
- Central body gravity (spherical harmonics toggle + settings entry point)
- Third-body perturbations (Sun, Earth, Earth J2)
- Non-gravitational perturbations (SRP, Albedo, Thermal)
- Tides (k2 / k3)
- Relativity (1PN)

Design rules
------------
- No backward-compat access via MainWindow attributes.
  The host must access widgets through the page instance:
      forces_ui = self.page_forces
      forces_ui.sw_gravity.isChecked()

- The page is responsible for creating and owning:
  sw_* toggles, CostIndicator widgets, and settings buttons.

- The host (MainWindow) remains responsible for:
  - Opening dialogs (e.g., gravity/albedo settings) if those dialogs depend on
    global app state, configs, or file-system paths.
  - Command building and preflight data collection, by reading state from
    this page.

Public API (expected)
---------------------
Class: ForceModelsPage(QtWidgets.QWidget)

Attributes created by the page (minimum contract):
- sw_gravity
- sw_sun
- sw_earth
- sw_earth_j2
- sw_srp
- sw_albedo
- sw_thermal
- sw_tides_k2
- sw_tides_k3
- sw_relativity_1pn

Optional helpers (recommended):
- get_data() -> dict
- load_data(data: dict) -> None

Typical usage (MainWindow)
--------------------------
    from lunaris.ui.widgets.force_models_page import ForceModelsPage

    self.page_forces = ForceModelsPage(
        on_gravity_settings=self._on_gravity_settings,
        on_albedo_settings=self._on_albedo_settings,
        parent=self,
    )

    # Reading:
    forces_ui = self.page_forces
    grav_on = forces_ui.sw_gravity.isChecked()

    # Restoring:
    forces_ui.sw_sun.setChecked(True)

Dependencies
------------
- PySide6
- ui_commons: THEME, ToggleSwitch, CostIndicator, get_icon (and any shared styles)

Notes
-----
If you move legacy code out of MainWindow, also move any helper callbacks that
are purely UI-local into this page (e.g., toggle dependency sync such as
"SRP requires Sun").
"""


# =============================================================================
# 0.                                    IMPORTS 
# =============================================================================
from __future__ import annotations

import os
import re
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PySide6 import QtCore, QtWidgets


try:
    from .ui_commons import normalize_path, THEME, QuickChip, ToggleSwitch, get_icon, find_project_root, CostIndicator
    from .surrogate_artifacts import is_valid_surrogate_run, looks_like_lunar_surrogate_run
    from .gravity_artifact_utils import (
        GRAVITY_EXTENSIONS,
        extract_sh_degree,
        find_best_gravity_file,
        list_st_lrps_model_dirs,
        ST_LRPS_RUNS_DIR as _ST_LRPS_RUNS_DIR_UTIL,
    )
except ImportError:
        # Only handle the "ran as a script" case; don't mask real import errors.
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  When executed directly, relative imports like '.constants' fail.", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m lunaris.ui.widgets.force_models_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2)
    raise




PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"


# =============================================================================
# 1.                             GRAVITY
# =============================================================================

# Quick selection for Spherical Harmonic (SH) degree
QUICK_DEGREES = ("20", "70", "140", "300", "660", "1200")

# -----------------------------------------------------------------------------
# Adaptive Gravity Profiles
# -----------------------------------------------------------------------------
ADAPTIVE_GRAVITY_PROFILES = {
    "Balanced (Scientific)": {
        "description": "Targeting 1e-12 m/s² precision. Optimized for LLO.",
        "interp": "smoothstep",
        "blend_km": 5.0,
        "table_km": [(10.0, 1000), (50.0, 660), (200.0, 140), (1000.0, 20)],
    },
    "High Fidelity (Science-Op)": {
        "description": "Max resolution for low periselene orbits (<20km).",
        "interp": "smoothstep",
        "blend_km": 10.0,
        "table_km": [(5.0, 1200), (30.0, 1000), (100.0, 660), (500.0, 70)],
    },
    "Fast Preview": {
        "description": "Rapid integration. Suitable for initial mission design.",
        "interp": "linear",
        "blend_km": 0.0,
        "table_km": [(50.0, 180), (200.0, 70), (1000.0, 10)],
    },
}

ADAPTIVE_CUSTOM_ID = "Custom Configuration..."

# -----------------------------------------------------------------------------
# File System Constants  (canonical definitions live in gravity_artifact_utils)
# -----------------------------------------------------------------------------
ST_LRPS_RUNS_DIR = _ST_LRPS_RUNS_DIR_UTIL


def _is_valid_st_lrps_model_dir(path: Path) -> bool:
    """Thin wrapper — delegates to shared resolver (accepts ckpt_last fallback)."""
    return is_valid_surrogate_run(path)


def _looks_like_lunar_st_lrps_model_dir(path: Path) -> bool:
    """Thin wrapper — delegates to shared resolver (5 % tolerance)."""
    return looks_like_lunar_surrogate_run(path)




# =============================================================================
# 1A.                        GRAVITY CONFIGURATION
# =============================================================================

@dataclass
class UIGravityConfig:
    """
    Mutable gravity configuration container for the UI dialog.
    Separates gravity settings from main UI.
    """
    enabled: bool = True
    degree: int = 100
    file_path: str = ""
    backend: str = "classic_sh"
    st_lrps_model_dir: str = ""
    adaptive_enabled: bool = False
    adaptive_preset: str = "Balanced (Scientific)"
    adaptive_table: List[Tuple[float, int]] = field(default_factory=lambda: [
        (10.0, 1000), (50.0, 660), (200.0, 140), (1000.0, 20)
    ])
    
    def sort_and_validate(self):
        """
        Normalize the adaptive table into a backend-safe altitude schedule.

        Two guard rails matter here:
        - rows must stay sorted so the CLI/backend interpret them deterministically
        - requested adaptive degrees must never exceed the selected base degree

        Without the degree clamp, the UI could happily emit rules like "10 km ->
        degree 1000" while the active gravity model is loaded only to degree 100.
        The backend can clamp those values later, but normalizing them here keeps
        the preview honest and prevents confusing "adaptive enabled" behavior.
        """

        cleaned = []
        max_degree = max(0, int(self.degree or 0))
        min_degree = 0 if max_degree == 0 else 1
        for alt, deg in self.adaptive_table:
            try:
                a = max(0.0, float(alt))
                d = max(min_degree, min(max_degree, int(deg)))
                cleaned.append((a, d))
            except (ValueError, TypeError):
                continue
        cleaned.sort(key=lambda x: x[0])
        self.adaptive_table = cleaned
    
    def apply_preset(self, preset_name: str):
        """Apply predefined adaptive gravity profile."""
        if preset_name not in ADAPTIVE_GRAVITY_PROFILES:
            return
        profile = ADAPTIVE_GRAVITY_PROFILES[preset_name]
        self.adaptive_preset = preset_name
        self.adaptive_table = [tuple(row) for row in profile.get("table_km", [])]
        self.sort_and_validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return dataclasses.asdict(self)

    def from_dict(self, data: Dict[str, Any]):
        """Load from dictionary."""
        if not data:
            return
        self.enabled = data.get("enabled", True)
        self.degree = data.get("degree", 100)
        self.file_path = data.get("file_path", "")
        self.backend = str(data.get("backend", "classic_sh") or "classic_sh")
        self.st_lrps_model_dir = str(data.get("st_lrps_model_dir", "") or "")
        self.adaptive_enabled = data.get("adaptive_enabled", False)
        self.adaptive_preset = data.get("adaptive_preset", "Balanced (Scientific)")
        raw_table = data.get("adaptive_table", [])
        if raw_table:
            self.adaptive_table = [tuple(x) for x in raw_table]
        if self.backend not in {"classic_sh", "st_lrps"}:
            self.backend = "classic_sh"
        self.sort_and_validate()


class GravitySettingsDialog(QtWidgets.QDialog):
    """
    Advanced configuration dialog for Lunar Gravity models.
    Consolidates all gravity settings in one place.
    """
    
    def __init__(self, parent: QtWidgets.QWidget, cfg: UIGravityConfig):
        super().__init__(parent)
        self.setWindowTitle("Gravity Field Configuration")
        self.setModal(True)
        self.resize(750, 600)
        self._cfg = cfg  # Reference to mutable config object
        
        # Main Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Style the dialog
        self.setStyleSheet(f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};")
        
        # --- HEADER ---
        header = QtWidgets.QLabel("Lunar Gravity Field Configuration")
        header.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {THEME['accent']};")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Choose either the classical spherical-harmonics field or a trained "
            "trained surrogate model for the Moon's central gravity model."
        )
        desc.setStyleSheet(f"color: {THEME['fg_muted']};")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # --- TABS ---
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self._apply_tab_style()
        
        # Tab 1: Basic Settings
        self.tab_basic = self._create_basic_tab()
        self.tabs.addTab(self.tab_basic, "Basic")
        
        # Tab 2: Adaptive Optimization
        self.tab_adaptive = self._create_adaptive_tab()
        self.tabs.addTab(self.tab_adaptive, "Adaptive")
        
        layout.addWidget(self.tabs, 1)
        
        # --- BUTTONS ---
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_save = QtWidgets.QPushButton("Apply Settings")
        
        for btn in (self.btn_cancel, self.btn_save):
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFixedHeight(32)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME['bg_entry']};
                    border: 1px solid {THEME['border']};
                    border-radius: 6px;
                    color: {THEME['fg_main']};
                    padding: 0 16px;
                }}
                QPushButton:hover {{
                    background-color: {THEME['border']};
                }}
            """)
        
        # Highlight Save button
        self.btn_save.setStyleSheet(self.btn_save.styleSheet() + f"""
            QPushButton {{
                background-color: {THEME['accent']};
                border: 1px solid {THEME['accent']};
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_hov']};
            }}
        """)
        
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        
        # Signals
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)
        
        # Initialize UI
        self._load_current_config()
    
    def _apply_tab_style(self):
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {THEME['border']};
                border-radius: 8px;
                background: {THEME['bg_card']};
            }}
            QTabBar::tab {{
                background: {THEME['bg_space']};
                border: 1px solid {THEME['border']};
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                color: {THEME['fg_muted']};
            }}
            QTabBar::tab:selected {{
                background: {THEME['bg_card']};
                color: {THEME['fg_main']};
                border-bottom: 1px solid {THEME['bg_card']};
            }}
        """)
    
    def _create_basic_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Enable/Disable Toggle
        self.chk_enabled = QtWidgets.QCheckBox("Enable Gravity Model")
        self.chk_enabled.setChecked(self._cfg.enabled)
        layout.addWidget(self.chk_enabled)

        # Backend Selection
        backend_group = QtWidgets.QGroupBox("Gravity Computation Mode")
        backend_group.setStyleSheet(f"border: 1px solid {THEME['border']}; border-radius: 8px;")
        backend_layout = QtWidgets.QVBoxLayout(backend_group)

        self.cb_backend = QtWidgets.QComboBox()
        self.cb_backend.addItem("Classical Spherical Harmonics", "classic_sh")
        self.cb_backend.addItem("ST-LRPS Gravity Surrogate", "st_lrps")
        self.cb_backend.currentIndexChanged.connect(self._on_backend_changed)
        backend_layout.addWidget(self.cb_backend)

        self.lbl_backend_hint = QtWidgets.QLabel()
        self.lbl_backend_hint.setWordWrap(True)
        self.lbl_backend_hint.setStyleSheet(f"color: {THEME['fg_muted']};")
        backend_layout.addWidget(self.lbl_backend_hint)

        layout.addWidget(backend_group)

        # Degree Selection
        self.degree_group = QtWidgets.QGroupBox("Maximum Spherical Harmonic Degree")
        self.degree_group.setStyleSheet(f"border: 1px solid {THEME['border']}; border-radius: 8px;")
        deg_layout = QtWidgets.QHBoxLayout(self.degree_group)
        
        self.sp_degree = QtWidgets.QSpinBox()
        self.sp_degree.setRange(0, 2000)
        self.sp_degree.setValue(self._cfg.degree)
        self.sp_degree.setFixedWidth(100)
        self.sp_degree.valueChanged.connect(self._on_degree_changed)
        deg_layout.addWidget(self.sp_degree)
        
        # Quick chips
        chip_container = QtWidgets.QHBoxLayout()
        chip_container.setSpacing(6)
        for d in QUICK_DEGREES:
            btn = QuickChip(str(d))
            btn.clicked.connect(lambda _, x=int(d): self.sp_degree.setValue(x))
            chip_container.addWidget(btn)
        
        chip_container.addStretch()
        deg_layout.addLayout(chip_container, 1)
        layout.addWidget(self.degree_group)

        # File Selection
        self.file_group = QtWidgets.QGroupBox("Gravity Model File")
        self.file_group.setStyleSheet(f"border: 1px solid {THEME['border']}; border-radius: 8px;")
        file_layout = QtWidgets.QVBoxLayout(self.file_group)
        
        self.ent_file = QtWidgets.QLineEdit(self._cfg.file_path)
        self.ent_file.setPlaceholderText("Path to .shbdr / .tab file...")
        file_layout.addWidget(self.ent_file)
        
        btn_row = QtWidgets.QHBoxLayout()
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.setIcon(get_icon("fa6s.folder-open", THEME['fg_main']))
        btn_browse.clicked.connect(self._browse_gravity_file)
        btn_auto = QtWidgets.QPushButton("Auto-Detect")
        btn_auto.setIcon(get_icon("fa6s.wand-magic-sparkles", THEME['accent']))
        btn_auto.clicked.connect(self._auto_detect_file)
        
        btn_row.addWidget(btn_browse)
        btn_row.addWidget(btn_auto)
        btn_row.addStretch()
        file_layout.addLayout(btn_row)
        
        layout.addWidget(self.file_group)

        # Surrogate Run Selection
        self.surrogate_group = QtWidgets.QGroupBox("Surrogate Gravity Run")
        self.surrogate_group.setStyleSheet(f"border: 1px solid {THEME['border']}; border-radius: 8px;")
        surrogate_layout = QtWidgets.QVBoxLayout(self.surrogate_group)

        self.ent_surrogate_dir = QtWidgets.QLineEdit(self._cfg.st_lrps_model_dir)
        self.ent_surrogate_dir.setPlaceholderText(
            "Path to a trained run directory containing config.json and ckpt_best.pt..."
        )
        surrogate_layout.addWidget(self.ent_surrogate_dir)

        surrogate_btn_row = QtWidgets.QHBoxLayout()
        btn_surrogate_browse = QtWidgets.QPushButton("Browse Run")
        btn_surrogate_browse.setIcon(get_icon("fa6s.folder-open", THEME['fg_main']))
        btn_surrogate_browse.clicked.connect(self._browse_surrogate_dir)

        btn_surrogate_auto = QtWidgets.QPushButton("Use Latest Run")
        btn_surrogate_auto.setIcon(get_icon("fa6s.wand-magic-sparkles", THEME['accent']))
        btn_surrogate_auto.clicked.connect(self._auto_detect_surrogate_dir)

        surrogate_btn_row.addWidget(btn_surrogate_browse)
        surrogate_btn_row.addWidget(btn_surrogate_auto)
        surrogate_btn_row.addStretch()
        surrogate_layout.addLayout(surrogate_btn_row)

        self.lbl_surrogate_hint = QtWidgets.QLabel(
            "The selected run must be a Moon-trained surrogate run and include config.json plus a checkpoint (ckpt_best.pt or ckpt_last.pt)."
        )
        self.lbl_surrogate_hint.setWordWrap(True)
        self.lbl_surrogate_hint.setStyleSheet(f"color: {THEME['fg_muted']};")
        surrogate_layout.addWidget(self.lbl_surrogate_hint)

        layout.addWidget(self.surrogate_group)
        layout.addStretch(1)

        return page
    
    def _create_adaptive_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Enable Adaptive
        adaptive_header = QtWidgets.QHBoxLayout()
        
        self.toggle_adaptive = ToggleSwitch()
        self.toggle_adaptive.setChecked(self._cfg.adaptive_enabled)
        adaptive_header.addWidget(self.toggle_adaptive)
        
        lbl_adaptive = QtWidgets.QLabel("Enable Adaptive Degree Optimization")
        lbl_adaptive.setStyleSheet(f"font-weight: bold; color: {THEME['fg_main']};")
        adaptive_header.addWidget(lbl_adaptive)
        adaptive_header.addStretch()
        
        layout.addLayout(adaptive_header)
        
        # Preset Selection
        presets_group = QtWidgets.QGroupBox("Optimization Profile")
        presets_group.setStyleSheet(f"border: 1px solid {THEME['border']}; border-radius: 8px;")
        presets_layout = QtWidgets.QVBoxLayout(presets_group)
        
        self.cb_preset = QtWidgets.QComboBox()
        self.cb_preset.addItems(list(ADAPTIVE_GRAVITY_PROFILES.keys()) + [ADAPTIVE_CUSTOM_ID])
        self.cb_preset.currentTextChanged.connect(self._on_preset_change)
        presets_layout.addWidget(self.cb_preset)
        
        layout.addWidget(presets_group)
        
        # Table Preview
        table_group = QtWidgets.QGroupBox("Altitude vs Degree Rules")
        table_group.setStyleSheet(f"border: 1px solid {THEME['border']}; border-radius: 8px;")
        table_layout = QtWidgets.QVBoxLayout(table_group)
        
        self.table_preview = QtWidgets.QTableWidget()
        self.table_preview.setColumnCount(2)
        self.table_preview.setHorizontalHeaderLabels(["Altitude (km)", "Max Degree"])
        self.table_preview.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_preview.verticalHeader().setVisible(False)
        self.table_preview.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        
        table_layout.addWidget(self.table_preview)
        
        btn_edit = QtWidgets.QPushButton("Edit Rules Table")
        btn_edit.setIcon(get_icon("fa6s.pen-to-square", THEME['fg_main']))
        btn_edit.clicked.connect(self._edit_adaptive_table)
        table_layout.addWidget(btn_edit)
        
        layout.addWidget(table_group, 1)
        
        return page
    
    def _load_current_config(self):
        """Initialize UI with current config values."""
        self.chk_enabled.setChecked(self._cfg.enabled)
        self.sp_degree.setValue(self._cfg.degree)
        self.ent_file.setText(self._cfg.file_path)
        self.ent_surrogate_dir.setText(self._cfg.st_lrps_model_dir)
        self.toggle_adaptive.setChecked(self._cfg.adaptive_enabled)

        backend_index = self.cb_backend.findData(self._cfg.backend)
        if backend_index < 0:
            backend_index = 0
        self.cb_backend.setCurrentIndex(backend_index)

        if self._cfg.adaptive_preset in ADAPTIVE_GRAVITY_PROFILES:
            self.cb_preset.setCurrentText(self._cfg.adaptive_preset)
        else:
            self.cb_preset.setCurrentText(ADAPTIVE_CUSTOM_ID)

        self._update_table_preview()
        self._sync_backend_mode_ui()
    
    def _update_table_preview(self):
        """Update the table preview with current adaptive rules."""
        self.table_preview.setRowCount(0)
        for alt, deg in self._cfg.adaptive_table:
            row = self.table_preview.rowCount()
            self.table_preview.insertRow(row)
            self.table_preview.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{alt:.1f}"))
            self.table_preview.setItem(row, 1, QtWidgets.QTableWidgetItem(str(deg)))

    def _on_degree_changed(self, value: int) -> None:
        """
        Keep adaptive preview rows aligned with the currently selected base degree.

        Users often select a lower base degree after choosing a preset. Re-clamping
        the preview immediately prevents the dialog from showing impossible rule
        values that the backend would later have to trim.
        """

        self._cfg.degree = int(value)
        self._cfg.sort_and_validate()
        self._update_table_preview()

    def _on_backend_changed(self, _index: int) -> None:
        """Mirror the selected backend into the working config object."""

        self._cfg.backend = str(self.cb_backend.currentData() or "classic_sh")
        self._sync_backend_mode_ui()

    def _sync_backend_mode_ui(self) -> None:
        """
        Show only the controls that matter for the active gravity backend.

        Classical SH runs need file / degree / adaptive controls. The surrogate
        path needs only the trained run directory and should not expose SH-only
        tuning that the backend will ignore anyway.
        """

        backend = str(self.cb_backend.currentData() or self._cfg.backend or "classic_sh")
        is_surrogate = backend == "st_lrps"

        self.degree_group.setVisible(not is_surrogate)
        self.file_group.setVisible(not is_surrogate)
        self.surrogate_group.setVisible(is_surrogate)
        self.tabs.setTabEnabled(1, not is_surrogate)
        if is_surrogate and self.tabs.currentWidget() is self.tab_adaptive:
            self.tabs.setCurrentWidget(self.tab_basic)

        if is_surrogate:
            self.lbl_backend_hint.setText(
                "Uses the trained surrogate model as the central gravity model. "
                "Adaptive SH degree rules are not used in this mode."
            )
        else:
            self.lbl_backend_hint.setText(
                "Uses the classical spherical-harmonic gravity field with the selected "
                "coefficient file and optional adaptive degree schedule."
            )
    
    def _browse_gravity_file(self, _checked: bool = False):
        """Open file dialog for gravity model."""
        current = self.ent_file.text() or str(PROJECT_ROOT)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Gravity Model", current,
            "Gravity Files (*.shbdr *.tab *.dat *.gfc);;All Files (*.*)"
        )
        if path:
            self.ent_file.setText(normalize_path(path))
    
    def _auto_detect_file(self, _checked: bool = False):
        """Auto-detect gravity file based on selected degree."""
        target_degree = self.sp_degree.value()
        found = find_best_gravity_file(PROJECT_ROOT, target_degree)
        if found:
            self.ent_file.setText(found)
            QtWidgets.QMessageBox.information(
                self, "Auto-Detect", 
                f"Found: {Path(found).name}\nDetected degree: {extract_sh_degree(found) or 'Unknown'}"
            )
        else:
            QtWidgets.QMessageBox.warning(
                self, "Auto-Detect", 
                "No suitable gravity model files found in project directories."
            )

    def _browse_surrogate_dir(self, _checked: bool = False):
        """Select a trained surrogate gravity run directory."""

        current = self.ent_surrogate_dir.text() or str(ST_LRPS_RUNS_DIR)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Surrogate Gravity Run",
            current,
        )
        if path:
            self.ent_surrogate_dir.setText(normalize_path(path))

    def _auto_detect_surrogate_dir(self, _checked: bool = False):
        """Pick the newest valid surrogate gravity run from the repository."""

        runs = list_st_lrps_model_dirs()
        if not runs:
            QtWidgets.QMessageBox.warning(
                self,
                "Surrogate Gravity",
                "No lunar-compatible surrogate gravity run was found under st_lrps/runs.",
            )
            return

        picked = runs[0]
        self.ent_surrogate_dir.setText(normalize_path(str(picked)))
        QtWidgets.QMessageBox.information(
            self,
            "Surrogate Gravity",
            f"Selected latest run: {picked.name}",
        )
    
    def _on_preset_change(self, text: str):
        """Handle preset selection change."""
        if text == ADAPTIVE_CUSTOM_ID:
            return
        if text in ADAPTIVE_GRAVITY_PROFILES:
            self._cfg.apply_preset(text)
            self._update_table_preview()
    
    def _edit_adaptive_table(self, _checked: bool = False):
        """
        Open the detailed adaptive-rule editor using an isolated working copy.

        `dataclasses.replace()` is not enough on its own here because the table is
        list-backed. Copying the list explicitly prevents a cancelled dialog from
        mutating the live config through shared list references.
        """

        from dataclasses import replace
        temp_cfg = replace(self._cfg, adaptive_table=[tuple(row) for row in self._cfg.adaptive_table])
        
        # Create and execute adaptive dialog
        dlg = AdaptiveDegreeDialog(self, temp_cfg)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            temp_cfg.sort_and_validate()
            self._cfg.adaptive_table = temp_cfg.adaptive_table
            self._cfg.adaptive_preset = ADAPTIVE_CUSTOM_ID
            self.cb_preset.setCurrentText(ADAPTIVE_CUSTOM_ID)
            self._update_table_preview()
    
    def _on_save(self, _checked: bool = False):
        """Validate and commit gravity settings back to the shared config object."""

        backend = str(self.cb_backend.currentData() or "classic_sh")
        file_path = normalize_path(self.ent_file.text())
        surrogate_dir = normalize_path(self.ent_surrogate_dir.text())

        if self.chk_enabled.isChecked():
            if backend == "classic_sh" and not file_path:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing Gravity File",
                    "Please choose a spherical-harmonic gravity model file.",
                )
                return
            if backend == "st_lrps" and not surrogate_dir:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing Surrogate Run",
                    "Please choose a trained surrogate gravity run directory.",
                )
                return

        self._cfg.enabled = self.chk_enabled.isChecked()
        self._cfg.degree = self.sp_degree.value()
        self._cfg.file_path = file_path
        self._cfg.backend = backend
        self._cfg.st_lrps_model_dir = surrogate_dir
        self._cfg.adaptive_enabled = self.toggle_adaptive.isChecked()
        self._cfg.adaptive_preset = self.cb_preset.currentText()
        self._cfg.sort_and_validate()
        
        self.accept()


# =============================================================================
# 1B.                      ADAPTIVE GRAVITY CONFIG (DIALOG)
# =============================================================================

@dataclass
class UIAdaptiveConfig:
    enabled: bool = False
    preset_name: str = "Balanced (Scientific)"
    interp_method: str = "smoothstep"
    blend_width_km: float = 5.0
    table_km: List[Tuple[float, int]] = field(default_factory=lambda: [
        (10.0, 1000), (50.0, 660), (200.0, 140), (1000.0, 20)
    ])
    
    def sort_and_validate(self):
        """Ensures the table is sorted by altitude and contains valid numbers."""
        cleaned = []
        for alt, deg in self.table_km:
            try:
                a = max(0.0, float(alt))
                d = max(1, int(deg))
                cleaned.append((a, d))
            except (ValueError, TypeError):
                continue
        cleaned.sort(key=lambda x: x[0])
        self.table_km = cleaned
    
    def apply_preset(self, preset_name: str):
        if preset_name not in ADAPTIVE_GRAVITY_PROFILES:
            return
        profile = ADAPTIVE_GRAVITY_PROFILES[preset_name]
        self.preset_name = preset_name
        self.interp_method = profile.get("interp", "smoothstep")
        self.blend_width_km = profile.get("blend_km", 5.0)
        self.table_km = [tuple(row) for row in profile.get("table_km", [])]
        self.sort_and_validate()


class AdaptiveDegreeDialog(QtWidgets.QDialog):
    """
    Editor for the Altitude-vs-Degree lookup table.
    Allows users to define performance/accuracy trade-offs.
    """
    
    def __init__(self, parent: QtWidgets.QWidget, cfg: UIAdaptiveConfig):
        super().__init__(parent)
        self.setWindowTitle("Adaptive Gravity Configuration")
        self.setModal(True)
        self.resize(700, 500)
        self._cfg = cfg
        
        # Main Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Apply Theme
        self.setStyleSheet(f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};")
        
        # --- Header ---
        header = QtWidgets.QLabel("Adaptive Gravity Logic")
        header.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {THEME['accent']};")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Automatically reduce Spherical Harmonic degree at higher altitudes to save computation time.\n"
            "Define thresholds below. The engine interpolates between steps."
        )
        desc.setStyleSheet(f"color: {THEME['fg_muted']};")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # --- Settings Form ---
        form_frame = QtWidgets.QFrame()
        form_frame.setStyleSheet(f"background-color: {THEME['bg_card']}; border-radius: 8px; border: 1px solid {THEME['border']};")
        form_layout = QtWidgets.QGridLayout(form_frame)
        form_layout.setContentsMargins(15, 15, 15, 15)
        form_layout.setVerticalSpacing(12)
        
        # Preset Selector
        self.cb_preset = QtWidgets.QComboBox()
        self.cb_preset.addItems(list(ADAPTIVE_GRAVITY_PROFILES.keys()) + [ADAPTIVE_CUSTOM_ID])
        self.cb_preset.setCurrentText(cfg.preset_name if cfg.preset_name in ADAPTIVE_GRAVITY_PROFILES else ADAPTIVE_CUSTOM_ID)
        self.cb_preset.currentTextChanged.connect(self._on_preset_change)
        
        # Interpolation Method
        self.cb_interp = QtWidgets.QComboBox()
        self.cb_interp.addItems(["linear", "smoothstep"])
        self.cb_interp.setCurrentText(cfg.interp_method)
        
        # Blend Width
        self.sp_blend = QtWidgets.QDoubleSpinBox()
        self.sp_blend.setRange(0.0, 500.0)
        self.sp_blend.setValue(cfg.blend_width_km)
        self.sp_blend.setSuffix(" km")
        
        self._add_form_row(form_layout, 0, "Load Profile:", self.cb_preset)
        self._add_form_row(form_layout, 1, "Interpolation:", self.cb_interp)
        self._add_form_row(form_layout, 2, "Blend Width:", self.sp_blend)
        
        layout.addWidget(form_frame)
        
        # --- Table Editor ---
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Altitude Threshold [km]", "Max SH Degree"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        
        # Table Styling
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {THEME['bg_entry']};
                gridline-color: {THEME['border']};
                border: 1px solid {THEME['border']};
                selection-background-color: {THEME['accent']};
            }}
            QHeaderView::section {{
                background-color: {THEME['bg_card']};
                padding: 6px;
                border: 1px solid {THEME['border']};
                color: {THEME['fg_muted']};
            }}
        """)
        
        layout.addWidget(self.table, 1)
        
        # --- Table Actions ---
        action_layout = QtWidgets.QHBoxLayout()
        
        btn_add = self._create_btn("Add Step", self._add_row)
        btn_remove = self._create_btn("Remove Selected", self._remove_row)
        
        action_layout.addWidget(btn_add)
        action_layout.addWidget(btn_remove)
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
        
        # --- Dialog Buttons ---
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch()
        
        btn_cancel = self._create_btn("Cancel", self.reject, primary=False)
        btn_save = self._create_btn("Save Configuration", self._save_and_close, primary=True)
        
        footer.addWidget(btn_cancel)
        footer.addWidget(btn_save)
        layout.addLayout(footer)
        
        # Initialize Data
        self._load_table_data()
    
    def _add_form_row(self, layout, row, label, widget):
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet(f"color: {THEME['fg_main']};")
        layout.addWidget(lbl, row, 0)
        layout.addWidget(widget, row, 1)
    
    def _create_btn(self, text, callback, primary=False):
        btn = QtWidgets.QPushButton(text)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        
        bg = THEME['accent'] if primary else THEME['bg_entry']
        border = THEME['accent'] if primary else THEME['border']
        
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 6px 12px;
                color: {THEME['fg_main']};
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_hov'] if primary else THEME['border']};
            }}
        """)
        return btn
    
    def _on_preset_change(self, text):
        """Auto-fill form when a preset is selected."""
        if text == ADAPTIVE_CUSTOM_ID:
            return
            
        if text in ADAPTIVE_GRAVITY_PROFILES:
            profile = ADAPTIVE_GRAVITY_PROFILES[text]
            self.cb_interp.setCurrentText(profile.get("interp", "smoothstep"))
            self.sp_blend.setValue(profile.get("blend_km", 5.0))
            
            # Update internal config temporarily to load table
            self._cfg.table_km = [tuple(r) for r in profile.get("table_km", [])]
            self._load_table_data()
    
    def _load_table_data(self):
        """Populates the QTableWidget from the config object."""
        self.table.setRowCount(0)
        for alt, deg in self._cfg.table_km:
            self._insert_table_row(alt, deg)
    
    def _insert_table_row(self, alt: float, deg: int):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        item_alt = QtWidgets.QTableWidgetItem(f"{alt:.1f}")
        item_deg = QtWidgets.QTableWidgetItem(str(deg))
        
        item_alt.setTextAlignment(QtCore.Qt.AlignCenter)
        item_deg.setTextAlignment(QtCore.Qt.AlignCenter)
        
        self.table.setItem(row, 0, item_alt)
        self.table.setItem(row, 1, item_deg)
    
    def _add_row(self):
        """Adds a default row and switches preset to Custom."""
        self._insert_table_row(0.0, 100)
        self.cb_preset.setCurrentText(ADAPTIVE_CUSTOM_ID)
        self.table.scrollToBottom()
    
    def _remove_row(self):
        """Removes selected rows."""
        rows = sorted(set(index.row() for index in self.table.selectedIndexes()), reverse=True)
        if rows:
            for r in rows:
                self.table.removeRow(r)
            self.cb_preset.setCurrentText(ADAPTIVE_CUSTOM_ID)
    
    def _read_table(self) -> List[Tuple[float, int]]:
        """Parses table content into a list of tuples."""
        data = []
        for r in range(self.table.rowCount()):
            try:
                t_alt = self.table.item(r, 0).text()
                t_deg = self.table.item(r, 1).text()
                alt = float(t_alt)
                deg = int(float(t_deg)) # Handle inputs like "100.0"
                data.append((alt, deg))
            except ValueError:
                continue
        return data
    
    def _save_and_close(self):
        """Validates input, updates config, and closes dialog."""
        raw_data = self._read_table()
        
        if len(raw_data) < 1:
            QtWidgets.QMessageBox.warning(self, "Invalid Config", "Please define at least one altitude step.")
            return
        
        # Commit changes to config object
        self._cfg.preset_name = self.cb_preset.currentText()
        self._cfg.interp_method = self.cb_interp.currentText()
        self._cfg.blend_width_km = self.sp_blend.value()
        self._cfg.table_km = raw_data
        
        # Auto-sort and cleanup
        self._cfg.sort_and_validate()
        
        self.accept()



# =============================================================================
# 2.                                ALBEDO
# =============================================================================

@dataclass
class UIAlbedoConfig:
    """
    Mutable configuration container for the UI dialog.
    This acts as a buffer before saving to the main immutable SimConfig.
    """
    label_path: str = ""
    img_path: str = ""
    model: str = "Lambertian"  # Options: Lambertian, Lommel-Seeliger
    use_ls: bool = False
    sampling: str = "bilinear" # Options: bilinear, nearest
    normal_mult: float = 1.0
    update_interval: float = 60.0


class AlbedoSettingsDialog(QtWidgets.QDialog):
    """
    Advanced configuration dialog for Lunar Albedo models.
    Handles file selection for PDS labels/images and numeric solver tunings.
    """
    def __init__(self, parent: QtWidgets.QWidget, cfg: UIAlbedoConfig):
        super().__init__(parent)
        self.setWindowTitle("Albedo Model Configuration")
        self.setModal(True)
        self.resize(720, 500)
        self._cfg = cfg
        
        # Main Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Style the dialog background
        self.setStyleSheet(f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};")
        
        # Tabs Container
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self._apply_tab_style()
        
        # --- TAB 1: DATA SOURCES (MAPS) ---
        self.tab_maps = self._create_maps_tab()
        self.tabs.addTab(self.tab_maps, "Data Sources")
        
        # --- TAB 2: PHYSICAL MODEL ---
        self.tab_model = self._create_model_tab()
        self.tabs.addTab(self.tab_model, "Physics Model")
        
        layout.addWidget(self.tabs, 1)
        
        # Action Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_save = QtWidgets.QPushButton("Apply Settings")
        
        for btn in (self.btn_cancel, self.btn_save):
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFixedHeight(32)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME['bg_entry']};
                    border: 1px solid {THEME['border']};
                    border-radius: 6px;
                    color: {THEME['fg_main']};
                    padding: 0 16px;
                }}
                QPushButton:hover {{
                    background-color: {THEME['border']};
                }}
            """)
        
        # Highlight Save button
        self.btn_save.setStyleSheet(self.btn_save.styleSheet() + f"""
            QPushButton {{
                background-color: {THEME['accent']};
                border: 1px solid {THEME['accent']};
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_hov']};
            }}
        """)
        
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        
        # Signals
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)
    
    def _apply_tab_style(self):
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {THEME['border']};
                border-radius: 8px;
                background: {THEME['bg_card']};
            }}
            QTabBar::tab {{
                background: {THEME['bg_space']};
                border: 1px solid {THEME['border']};
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                color: {THEME['fg_muted']};
            }}
            QTabBar::tab:selected {{
                background: {THEME['bg_card']};
                color: {THEME['fg_main']};
                border-bottom: 1px solid {THEME['bg_card']};
            }}
        """)
    
    def _create_maps_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setVerticalSpacing(15)
        
        # Label Path Input
        self.ent_label = self._create_file_input(layout, 0, "Albedo Label (.lbl)", self._cfg.label_path,
                                               "PDS Label (*.lbl *.txt);;All Files (*.*)")
        
        # Image Path Input
        self.ent_img = self._create_file_input(layout, 1, "Albedo Image (.img) [Optional]", self._cfg.img_path,
                                             "Binary Image (*.img);;All Files (*.*)")
        
        # Info Note
        note = QtWidgets.QLabel("ℹ️ If a PDS3 label is provided, the engine will attempt to load the Albedo grid. Otherwise, a constant default albedo model is used.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {THEME['fg_muted']}; font-style: italic; margin-top: 10px;")
        layout.addWidget(note, 2, 0, 1, 3)
        
        layout.setRowStretch(3, 1)
        return page
    
    def _create_file_input(self, layout, row, label_text, default_val, filters):
        lbl = QtWidgets.QLabel(label_text)
        ent = QtWidgets.QLineEdit(default_val)
        ent.setStyleSheet(f"background: {THEME['bg_entry']}; border: 1px solid {THEME['border']}; padding: 6px; border-radius: 4px; color: {THEME['fg_main']};")
        
        btn = QtWidgets.QPushButton("Browse")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setStyleSheet(f"background: {THEME['bg_entry']}; border: 1px solid {THEME['border']}; padding: 6px 12px; border-radius: 4px; color: {THEME['fg_main']};")
        
        def _browse(_checked: bool = False):
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, label_text, ent.text(), filters)
            if f: ent.setText(normalize_path(f))
        
        btn.clicked.connect(_browse)
        
        layout.addWidget(lbl, row, 0)
        layout.addWidget(ent, row, 1)
        layout.addWidget(btn, row, 2)
        return ent
    
    def _create_model_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # BRDF Model
        self.cb_model = QtWidgets.QComboBox()
        self.cb_model.addItems(["Lambertian", "Lommel-Seeliger"])
        self.cb_model.setCurrentText(self._cfg.model)
        self.cb_model.setStyleSheet(f"background: {THEME['bg_entry']}; border: 1px solid {THEME['border']}; padding: 4px;")
        layout.addRow("Reflectance Model:", self.cb_model)
        
        # Lommel-Seeliger Toggle
        self.chk_ls = QtWidgets.QCheckBox("Enable LS Phase Function")
        self.chk_ls.setChecked(self._cfg.use_ls)
        self.chk_ls.setStyleSheet(f"color: {THEME['fg_main']};")
        layout.addRow("", self.chk_ls)
        
        # Sampling Method
        self.cb_sample = QtWidgets.QComboBox()
        self.cb_sample.addItems(["bilinear", "nearest"])
        self.cb_sample.setCurrentText(self._cfg.sampling)
        self.cb_sample.setStyleSheet(f"background: {THEME['bg_entry']}; border: 1px solid {THEME['border']}; padding: 4px;")
        layout.addRow("Grid Sampling:", self.cb_sample)
        
        # Numeric Spinners
        self.sp_normal = QtWidgets.QDoubleSpinBox()
        self.sp_normal.setRange(0.1, 10.0)
        self.sp_normal.setValue(self._cfg.normal_mult)
        self.sp_normal.setStyleSheet(f"background: {THEME['bg_entry']}; border: 1px solid {THEME['border']};")
        layout.addRow("Normal Step Multiplier:", self.sp_normal)
        
        self.sp_interval = QtWidgets.QDoubleSpinBox()
        self.sp_interval.setRange(0.0, 3600.0)
        self.sp_interval.setValue(self._cfg.update_interval)
        self.sp_interval.setSuffix(" s")
        self.sp_interval.setStyleSheet(f"background: {THEME['bg_entry']}; border: 1px solid {THEME['border']};")
        layout.addRow("LS Update Interval:", self.sp_interval)
        
        return page
    
    def _on_save(self, _checked: bool = False):
        # Commit UI state back to the config object
        self._cfg.label_path = normalize_path(self.ent_label.text())
        self._cfg.img_path = normalize_path(self.ent_img.text())
        self._cfg.model = self.cb_model.currentText()
        self._cfg.use_ls = self.chk_ls.isChecked()
        self._cfg.sampling = self.cb_sample.currentText()
        self._cfg.normal_mult = self.sp_normal.value()
        self._cfg.update_interval = self.sp_interval.value()
        self.accept()


# =============================================================================
# 3.                             FORCE MODEL
# =============================================================================

class ForceModelsPage(QtWidgets.QWidget):
    """
    Page 2: Force Model Settings.
    Encapsulates all widgets (sw_gravity, sw_sun, sw_earth, etc.) inside this page.
    """

    def __init__(
        self,
        gravity_cfg: Optional["UIGravityConfig"] = None,
        albedo_cfg: Optional["UIAlbedoConfig"] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)

        # Keep references so dialogs update the SAME objects MainWindow uses
        self.gravity_cfg: "UIGravityConfig" = gravity_cfg if gravity_cfg is not None else UIGravityConfig()
        self.albedo_cfg: "UIAlbedoConfig" = albedo_cfg if albedo_cfg is not None else UIAlbedoConfig()

        # Build UI into self
        self._build_page_forces()

        # Post-wiring consistency
        self._sync_albedo_settings_button()
        self._sync_force_dependencies()

    def get_data(self) -> Dict[str, Any]:
        """
        Return a host-friendly snapshot of the currently selected force models.

        This mirrors the page-level API used by the other UI parts
        (`OrbitPage.get_data`, `MissionPropagationPage.to_dict`, etc.) and lets
        the main window build commands or save sessions without reaching into
        individual widgets unless it truly has to.
        """

        return {
            "gravity": {
                "enabled": bool(self.sw_gravity.isChecked()),
                "config": self.gravity_cfg.to_dict(),
            },
            "sun": bool(self.sw_sun.isChecked()),
            "earth": bool(self.sw_earth.isChecked()),
            "earth_j2": bool(self.sw_earth_j2.isChecked()),
            "srp": bool(self.sw_srp.isChecked()),
            "albedo": bool(self.sw_albedo.isChecked()),
            "thermal": bool(self.sw_thermal.isChecked()),
            "tides_k2": bool(self.sw_tides_k2.isChecked()),
            "tides_k3": bool(self.sw_tides_k3.isChecked()),
            "relativity_1pn": bool(self.sw_relativity_1pn.isChecked()),
        }

    def load_data(self, data: Dict[str, Any]) -> None:
        """
        Restore a previously saved force-model snapshot onto the page.

        Parameters
        ----------
        data:
            Dictionary previously produced by `get_data()`. Missing keys are
            tolerated so older session files can still be restored.
        """

        if not data:
            return

        gravity_payload = data.get("gravity", {}) or {}
        gravity_config = gravity_payload.get("config", {}) or {}
        if gravity_config:
            self.gravity_cfg.from_dict(gravity_config)

        self.sw_gravity.setChecked(bool(gravity_payload.get("enabled", True)))
        self.sw_sun.setChecked(bool(data.get("sun", True)))
        self.sw_earth.setChecked(bool(data.get("earth", True)))
        self.sw_earth_j2.setChecked(bool(data.get("earth_j2", False)))
        self.sw_srp.setChecked(bool(data.get("srp", False)))
        self.sw_albedo.setChecked(bool(data.get("albedo", False)))
        self.sw_thermal.setChecked(bool(data.get("thermal", False)))
        self.sw_tides_k2.setChecked(bool(data.get("tides_k2", True)))
        self.sw_tides_k3.setChecked(bool(data.get("tides_k3", False)))
        self.sw_relativity_1pn.setChecked(bool(data.get("relativity_1pn", False)))

        self._update_gravity_summary_ui()
        self._sync_force_dependencies()


    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _create_card(self, title: str) -> QtWidgets.QGroupBox:
        """
        Create a force-page card whose title stays fully visible inside scroll areas.

        The old negative title offset occasionally clipped the first row of
        cards. Keeping the title inside the card margin is less flashy but much
        more robust across different window sizes and font metrics.
        """

        gb = QtWidgets.QGroupBox(title)
        gb.setStyleSheet(f"""
            QGroupBox {{
                background-color: {THEME['bg_card']};
                border: 1px solid {THEME['border']};
                border-radius: 10px;
                margin-top: 18px;
                padding-top: 10px;
                color: {THEME['fg_main']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                left: 12px;
                top: 0px;
                color: {THEME['fg_main']};
                font-weight: 700;
            }}
        """)
        return gb

    def _estimate_gravity_cost_level(self) -> str:
        """
        Heuristic cost estimate for display only.
        - adaptive enabled: usually medium (work per step varies)
        - surrogate model: high (Python + neural inference in-loop)
        - degree >= 800: high
        - degree >= 200: medium
        - else: low
        """
        try:
            if str(getattr(self.gravity_cfg, "backend", "classic_sh") or "classic_sh") == "st_lrps":
                return "high"
            if bool(getattr(self.gravity_cfg, "adaptive_enabled", False)):
                return "medium"
            deg = int(getattr(self.gravity_cfg, "degree", 0) or 0)
            if deg >= 800:
                return "high"
            if deg >= 200:
                return "medium"
            return "low"
        except Exception:
            return "medium"

    def _update_gravity_summary_ui(self):
        # status label
        try:
            backend = str(getattr(self.gravity_cfg, "backend", "classic_sh") or "classic_sh")
            if backend == "st_lrps":
                model_dir = str(getattr(self.gravity_cfg, "st_lrps_model_dir", "") or "").strip()
                msg = "Surrogate model selected"
                if model_dir:
                    msg += f" | {QtCore.QFileInfo(model_dir).fileName()}"
            else:
                deg = int(getattr(self.gravity_cfg, "degree", 0) or 0)
                path = str(getattr(self.gravity_cfg, "file_path", "") or "").strip()
                adaptive = bool(getattr(self.gravity_cfg, "adaptive_enabled", False))
                if adaptive:
                    msg = f"Adaptive enabled (base degree {deg})"
                else:
                    msg = f"Degree {deg}"
                if path:
                    msg += f" | {QtCore.QFileInfo(path).fileName()}"
            self.lbl_gravity_status.setText(msg)
        except Exception:
            self.lbl_gravity_status.setText("Gravity config updated")

        # cost indicator
        try:
            self.ind_gravity_cost.set_level(self._estimate_gravity_cost_level())
        except Exception:
            pass

    def _sync_force_dependencies(self):
        """
        Show/hide dependency warnings (non-blocking) and enforce hard deps.
        - SRP/Albedo/Thermal: warn when Sun is not enabled
        - Earth J2: warn when Earth third-body is not enabled
        """
        sun_on = hasattr(self, "sw_sun") and self.sw_sun.isChecked()
        srp_on = hasattr(self, "sw_srp") and self.sw_srp.isChecked()
        albedo_on = hasattr(self, "sw_albedo") and self.sw_albedo.isChecked()
        thermal_on = hasattr(self, "sw_thermal") and self.sw_thermal.isChecked()
        earth_on = hasattr(self, "sw_earth") and self.sw_earth.isChecked()
        earth_j2_on = hasattr(self, "sw_earth_j2") and self.sw_earth_j2.isChecked()

        # Warning: SRP/Albedo/Thermal without Sun
        if hasattr(self, "lbl_warn_srp_sun"):
            needs_sun = srp_on or albedo_on or thermal_on
            self.lbl_warn_srp_sun.setVisible(needs_sun and not sun_on)

        # Warning: Earth J2 without Earth
        if hasattr(self, "lbl_warn_earth_j2"):
            self.lbl_warn_earth_j2.setVisible(earth_j2_on and not earth_on)

        # Disable settings buttons when toggles are off
        if hasattr(self, "btn_gravity_settings") and hasattr(self, "sw_gravity"):
            self.btn_gravity_settings.setEnabled(bool(self.sw_gravity.isChecked()))
        self._sync_albedo_settings_button()


    # -------------------------------------------------------------------------
    # 20. PAGE BUILDERS: FORCE MODELS (PAGE 2)
    # -------------------------------------------------------------------------
    def _build_page_forces(self) -> QtWidgets.QWidget:
        """Page 2: Force Model Settings."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # Top Row: Gravity + Third-Body
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(20)
        top_row.addWidget(self._group_gravity_force())
        top_row.addWidget(self._group_thirdbody_force())
        top_row.setStretch(0, 1)
        top_row.setStretch(1, 1)
        layout.addLayout(top_row)

        # Middle Row: Non-Gravitational
        layout.addWidget(self._group_nongrav_force())

        # Bottom Row: Tides & Relativity
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(20)
        bottom_row.addWidget(self._group_tides_force())
        bottom_row.addWidget(self._group_relativity_force())
        bottom_row.setStretch(0, 1)
        bottom_row.setStretch(1, 1)
        layout.addLayout(bottom_row)

        layout.addStretch(1)
        return self

    def _group_gravity_force(self) -> QtWidgets.QGroupBox:
        """Gravity force card with settings dialog button."""
        gb = self._create_card("Central Body Gravity")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(15)

        # Header with toggle and settings button
        header = QtWidgets.QHBoxLayout()

        self.sw_gravity = ToggleSwitch()
        self.sw_gravity.setChecked(True)
        header.addWidget(self.sw_gravity)

        lbl = QtWidgets.QLabel("Lunar Gravity Field")
        lbl.setStyleSheet(f"font-weight: bold; color: {THEME['fg_main']};")
        header.addWidget(lbl)

        header.addStretch()

        # Settings button
        self.btn_gravity_settings = QtWidgets.QPushButton()
        self.btn_gravity_settings.setFixedSize(32, 32)
        self.btn_gravity_settings.setIcon(get_icon("fa6s.gear", THEME["fg_main"]))
        self.btn_gravity_settings.setToolTip("Configure Gravity Model")
        self.btn_gravity_settings.clicked.connect(self._on_gravity_settings)
        header.addWidget(self.btn_gravity_settings)

        layout.addLayout(header)

        # Status indicator
        self.lbl_gravity_status = QtWidgets.QLabel("Default model loaded")
        self.lbl_gravity_status.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
        self.lbl_gravity_status.setWordWrap(True)
        layout.addWidget(self.lbl_gravity_status)

        # Cost indicator
        cost_row = QtWidgets.QHBoxLayout()
        cost_row.addWidget(QtWidgets.QLabel("CPU Cost:"))

        self.ind_gravity_cost = CostIndicator("high")
        cost_row.addWidget(self.ind_gravity_cost)
        cost_row.addStretch()

        layout.addLayout(cost_row)

        # Wiring
        self.sw_gravity.toggled.connect(lambda _v: self._sync_force_dependencies())

        # Initial UI from config
        self._update_gravity_summary_ui()

        return gb

    def _group_thirdbody_force(self) -> QtWidgets.QGroupBox:
        """Third-body perturbations card."""
        gb = self._create_card("Third-Body Perturbations")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QtWidgets.QLabel("Solar System Perturbers")
        header.setStyleSheet(f"font-weight: bold; color: {THEME['fg_main']}; margin-bottom: 10px;")
        layout.addWidget(header)

        # Sun perturbation
        sun_row = QtWidgets.QHBoxLayout()
        self.sw_sun = ToggleSwitch()
        self.sw_sun.setChecked(True)
        sun_row.addWidget(self.sw_sun)

        sun_lbl = QtWidgets.QLabel("Sun (Point Mass)")
        sun_lbl.setStyleSheet(f"color: {THEME['fg_main']};")
        sun_row.addWidget(sun_lbl)
        sun_row.addStretch()

        self.ind_sun_cost = CostIndicator("medium")
        sun_row.addWidget(self.ind_sun_cost)
        layout.addLayout(sun_row)

        # Earth point mass perturbation
        earth_row = QtWidgets.QHBoxLayout()
        self.sw_earth = ToggleSwitch()
        self.sw_earth.setChecked(True)
        earth_row.addWidget(self.sw_earth)

        earth_lbl = QtWidgets.QLabel("Earth (Point Mass)")
        earth_lbl.setStyleSheet(f"color: {THEME['fg_main']};")
        earth_row.addWidget(earth_lbl)
        earth_row.addStretch()

        self.ind_earth_cost = CostIndicator("medium")
        earth_row.addWidget(self.ind_earth_cost)
        layout.addLayout(earth_row)

        # Earth J2 perturbation
        earth_j2_row = QtWidgets.QHBoxLayout()
        self.sw_earth_j2 = ToggleSwitch()
        self.sw_earth_j2.setChecked(False)
        earth_j2_row.addWidget(self.sw_earth_j2)

        earth_j2_lbl = QtWidgets.QLabel("Earth J2 (Oblateness)")
        earth_j2_lbl.setStyleSheet(f"color: {THEME['fg_main']};")
        earth_j2_row.addWidget(earth_j2_lbl)
        earth_j2_row.addStretch()

        self.ind_earth_j2_cost = CostIndicator("low")
        earth_j2_row.addWidget(self.ind_earth_j2_cost)
        layout.addLayout(earth_j2_row)

        # Warning label: Earth J2 requires Earth third-body
        self.lbl_warn_earth_j2 = QtWidgets.QLabel("Warning: Earth J2 requires Earth third-body to be enabled.")
        self.lbl_warn_earth_j2.setStyleSheet(
            f"color: {THEME['warning']}; font-size: 9pt; font-style: italic;"
        )
        self.lbl_warn_earth_j2.setWordWrap(True)
        self.lbl_warn_earth_j2.setMinimumHeight(22)
        self.lbl_warn_earth_j2.setVisible(False)
        layout.addWidget(self.lbl_warn_earth_j2)

        # Wiring dependencies
        self.sw_sun.toggled.connect(lambda _v: self._sync_force_dependencies())
        self.sw_earth.toggled.connect(lambda _v: self._sync_force_dependencies())
        self.sw_earth_j2.toggled.connect(lambda _v: self._sync_force_dependencies())

        return gb

    def _group_nongrav_force(self) -> QtWidgets.QGroupBox:
        """Non-gravitational forces card."""
        gb = self._create_card("Non-Gravitational Forces")
        layout = QtWidgets.QGridLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(15)

        # SRP
        self.sw_srp = ToggleSwitch()
        self.sw_srp.setChecked(False)
        layout.addWidget(self.sw_srp, 0, 0)

        lbl_srp = QtWidgets.QLabel("Solar Radiation Pressure")
        lbl_srp.setStyleSheet(f"color: {THEME['fg_main']};")
        layout.addWidget(lbl_srp, 0, 1)

        self.ind_srp_cost = CostIndicator("medium")
        layout.addWidget(self.ind_srp_cost, 0, 2)

        # Albedo
        self.sw_albedo = ToggleSwitch()
        self.sw_albedo.setChecked(False)
        layout.addWidget(self.sw_albedo, 1, 0)

        lbl_albedo = QtWidgets.QLabel("Lunar Albedo")
        lbl_albedo.setStyleSheet(f"color: {THEME['fg_main']};")
        layout.addWidget(lbl_albedo, 1, 1)

        self.ind_albedo_cost = CostIndicator("high")
        layout.addWidget(self.ind_albedo_cost, 1, 2)

        # Albedo settings button
        self.btn_albedo_settings = QtWidgets.QPushButton()
        self.btn_albedo_settings.setFixedSize(28, 28)
        self.btn_albedo_settings.setIcon(get_icon("fa6s.gear", THEME["fg_main"]))
        self.btn_albedo_settings.setToolTip("Configure Albedo Model")
        self.btn_albedo_settings.clicked.connect(self._on_albedo_settings)
        layout.addWidget(self.btn_albedo_settings, 1, 3)

        # Thermal
        self.sw_thermal = ToggleSwitch()
        self.sw_thermal.setChecked(False)
        layout.addWidget(self.sw_thermal, 2, 0)

        lbl_thermal = QtWidgets.QLabel("Thermal Re-radiation")
        lbl_thermal.setStyleSheet(f"color: {THEME['fg_main']};")
        layout.addWidget(lbl_thermal, 2, 1)

        self.ind_thermal_cost = CostIndicator("medium")
        layout.addWidget(self.ind_thermal_cost, 2, 2)

        # Warning label: SRP/Albedo/Thermal require Sun position
        self.lbl_warn_srp_sun = QtWidgets.QLabel(
            "Warning: SRP/Albedo/Thermal require Sun position (enable Sun perturbation)."
        )
        self.lbl_warn_srp_sun.setStyleSheet(
            f"color: {THEME['warning']}; font-size: 9pt; font-style: italic;"
        )
        self.lbl_warn_srp_sun.setWordWrap(True)
        self.lbl_warn_srp_sun.setVisible(False)
        layout.addWidget(self.lbl_warn_srp_sun, 3, 0, 1, 4)

        # Connect SRP/Albedo to require Sun perturbation
        self.sw_srp.toggled.connect(self._sync_srp_requirement)
        self.sw_albedo.toggled.connect(self._sync_srp_requirement)
        self.sw_albedo.toggled.connect(self._sync_albedo_settings_button)
        self.sw_thermal.toggled.connect(self._sync_srp_requirement)

        return gb

    def _group_tides_force(self) -> QtWidgets.QGroupBox:
        """Solid tides force card."""
        gb = self._create_card("Solid Body Tides")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QtWidgets.QLabel("Lunar Solid Tides")
        header.setStyleSheet(f"font-weight: bold; color: {THEME['fg_main']}; margin-bottom: 10px;")
        layout.addWidget(header)

        # k2 Love number
        k2_row = QtWidgets.QHBoxLayout()
        self.sw_tides_k2 = ToggleSwitch()
        self.sw_tides_k2.setChecked(True)
        k2_row.addWidget(self.sw_tides_k2)

        k2_lbl = QtWidgets.QLabel("k2 Love Number (Degree 2)")
        k2_lbl.setStyleSheet(f"color: {THEME['fg_main']};")
        k2_row.addWidget(k2_lbl)
        k2_row.addStretch()

        self.ind_tides_k2_cost = CostIndicator("low")
        k2_row.addWidget(self.ind_tides_k2_cost)
        layout.addLayout(k2_row)

        # k3 Love number
        k3_row = QtWidgets.QHBoxLayout()
        self.sw_tides_k3 = ToggleSwitch()
        self.sw_tides_k3.setChecked(False)
        k3_row.addWidget(self.sw_tides_k3)

        k3_lbl = QtWidgets.QLabel("k3 Love Number (Degree 3)")
        k3_lbl.setStyleSheet(f"color: {THEME['fg_main']};")
        k3_row.addWidget(k3_lbl)
        k3_row.addStretch()

        self.ind_tides_k3_cost = CostIndicator("low")
        k3_row.addWidget(self.ind_tides_k3_cost)
        layout.addLayout(k3_row)

        # K3 implies K2 in the backend
        self.sw_tides_k3.toggled.connect(
            lambda on: self.sw_tides_k2.setChecked(True) if on else None
        )

        # Info note
        note = QtWidgets.QLabel("Note: Love numbers represent the Moon's elastic response to tidal forces. K3 implies K2.")
        note.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt; font-style: italic;")
        note.setWordWrap(True)
        note.setMinimumHeight(36)
        layout.addWidget(note)

        return gb

    def _group_relativity_force(self) -> QtWidgets.QGroupBox:
        """General Relativity force card."""
        gb = self._create_card("General Relativity")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QtWidgets.QLabel("Post-Newtonian Corrections")
        header.setStyleSheet(f"font-weight: bold; color: {THEME['fg_main']}; margin-bottom: 10px;")
        layout.addWidget(header)

        # 1PN correction
        pn_row = QtWidgets.QHBoxLayout()
        self.sw_relativity_1pn = ToggleSwitch()
        self.sw_relativity_1pn.setChecked(False)
        pn_row.addWidget(self.sw_relativity_1pn)

        pn_lbl = QtWidgets.QLabel("1PN Force (Post-Newtonian)")
        pn_lbl.setStyleSheet(f"color: {THEME['fg_main']};")
        pn_row.addWidget(pn_lbl)
        pn_row.addStretch()

        self.ind_relativity_cost = CostIndicator("low")
        pn_row.addWidget(self.ind_relativity_cost)
        layout.addLayout(pn_row)

        # Info note
        note = QtWidgets.QLabel("Note: First-order post-Newtonian correction for high-precision lunar orbits.")
        note.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt; font-style: italic;")
        note.setWordWrap(True)
        note.setMinimumHeight(36)
        layout.addWidget(note)

        return gb


    # -------------------------------------------------------------------------
    # Slots / callbacks referenced by the UI above
    # -------------------------------------------------------------------------
    def _on_gravity_settings(self, _checked: bool = False):
        """Open GravitySettingsDialog and apply changes to bound gravity_cfg."""
        try:
            dlg = GravitySettingsDialog(self.window() or self, self.gravity_cfg)
            if dlg.exec() == QtWidgets.QDialog.Accepted:
                self._update_gravity_summary_ui()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Gravity Settings", f"Could not open gravity settings:\n\n{e}")

    def _on_albedo_settings(self, _checked: bool = False):
        """Open AlbedoSettingsDialog and apply changes to bound albedo_cfg."""
        try:
            dlg = AlbedoSettingsDialog(self.window() or self, self.albedo_cfg)
            dlg.exec()  # config object is updated in-place by dialog on save
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Albedo Settings", f"Could not open albedo settings:\n\n{e}")

    def _sync_srp_requirement(self, _checked: bool = False):
        """Ensure Sun is enabled if SRP or Albedo is enabled."""
        self._sync_force_dependencies()

    def _sync_albedo_settings_button(self, _checked: bool = False):
        """Enable/disable Albedo settings button based on Albedo toggle."""
        try:
            if hasattr(self, "btn_albedo_settings") and hasattr(self, "sw_albedo"):
                self.btn_albedo_settings.setEnabled(bool(self.sw_albedo.isChecked()))
        except Exception:
            pass



# =============================================================================
# 4.                      TESTING FORCE MODELS PAGE
# =============================================================================

if __name__ == "__main__":
    import sys

    # Start the application
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create the test window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Force Models Page Test")
    window.resize(1000, 700)

    # Set the background color (to simulate a dark theme)
    window.setStyleSheet(
        f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};"
    )

    # Load the page
    page = ForceModelsPage()
    window.setCentralWidget(page)

    window.show()

    def dump_force_state(p: "ForceModelsPage") -> dict:
        return {
            "toggles": {
                "gravity": p.sw_gravity.isChecked(),
                "sun": p.sw_sun.isChecked(),
                "earth": p.sw_earth.isChecked(),
                "earth_j2": p.sw_earth_j2.isChecked(),
                "srp": p.sw_srp.isChecked(),
                "albedo": p.sw_albedo.isChecked(),
                "thermal": p.sw_thermal.isChecked(),
                "tides_k2": p.sw_tides_k2.isChecked(),
                "tides_k3": p.sw_tides_k3.isChecked(),
                "relativity_1pn": p.sw_relativity_1pn.isChecked(),
            },
            # configs owned by the page (dataclasses)
            "gravity_cfg": dataclasses.asdict(p.gravity_cfg),
            "albedo_cfg": dataclasses.asdict(p.albedo_cfg),
        }

    print("Test started...")
    print("Initial State:", dump_force_state(page))

    sys.exit(app.exec())
