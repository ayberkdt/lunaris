# ST_LRPS/ui_parts/force_models_page.py

"""
Force Models Page (UI Part) for Lunaris Mission Studio.

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
    from lunaris.ui.pages.force_models_page import ForceModelsPage

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

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

try:
    from lunaris.ui.components.primitives import DataTable, InlineNotice, Section
    from lunaris.ui.core.gravity_artifact_utils import (
        ST_LRPS_RUNS_DIR as _ST_LRPS_RUNS_DIR_UTIL,
    )
    from lunaris.ui.core.gravity_artifact_utils import (
        extract_sh_degree,
        find_best_gravity_file,
        list_st_lrps_model_dirs,
    )
    from lunaris.ui.core.surrogate_artifacts import (
        is_valid_surrogate_run,
        looks_like_lunar_surrogate_run,
    )
    from lunaris.ui.core.ui_commons import (
        THEME,
        CostIndicator,
        NoWheelComboBox,
        NoWheelDoubleSpinBox,
        NoWheelSpinBox,
        QuickChip,
        ToggleSwitch,
        find_project_root,
        get_icon,
        normalize_path,
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
        print("\n      python -m lunaris.ui.pages.force_models_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2) from None
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
    adaptive_table: list[tuple[float, int]] = field(default_factory=lambda: [
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return dataclasses.asdict(self)

    def from_dict(self, data: dict[str, Any]):
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
        self.setObjectName("settingsDialog")
        self.setModal(True)
        self.resize(750, 600)
        self.setMinimumSize(640, 500)
        self._cfg = cfg  # Reference to mutable config object

        # Main Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # --- HEADER ---
        header = QtWidgets.QLabel("Lunar Gravity Field Configuration")
        header.setObjectName("dialogTitle")
        layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Choose either the classical spherical-harmonics field or a trained "
            "surrogate model for the Moon's central gravity model."
        )
        desc.setObjectName("dialogDescription")
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
            btn.setMinimumHeight(34)
        self.btn_cancel.setProperty("kind", "ghost")
        self.btn_save.setProperty("kind", "primary")

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        # Signals
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)

        # Initialize UI
        self._load_current_config()

    def _apply_tab_style(self):
        # Tab styling comes from the global stylesheet (QTabWidget / QTabBar).
        pass

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
        backend_layout = QtWidgets.QVBoxLayout(backend_group)

        self.cb_backend = NoWheelComboBox()
        self.cb_backend.addItem("Classical Spherical Harmonics", "classic_sh")
        self.cb_backend.addItem("ST-LRPS Gravity Surrogate", "st_lrps")
        self.cb_backend.currentIndexChanged.connect(self._on_backend_changed)
        backend_layout.addWidget(self.cb_backend)

        self.lbl_backend_hint = QtWidgets.QLabel()
        self.lbl_backend_hint.setWordWrap(True)
        self.lbl_backend_hint.setObjectName("fieldHint")
        backend_layout.addWidget(self.lbl_backend_hint)

        layout.addWidget(backend_group)

        # Degree Selection
        self.degree_group = QtWidgets.QGroupBox("Maximum Spherical Harmonic Degree")
        deg_layout = QtWidgets.QHBoxLayout(self.degree_group)

        self.sp_degree = NoWheelSpinBox()
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
        self.lbl_surrogate_hint.setObjectName("fieldHint")
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
        lbl_adaptive.setObjectName("valueLabel")
        self.toggle_adaptive.setAccessibleName("Enable adaptive degree optimization")
        adaptive_header.addWidget(lbl_adaptive)
        adaptive_header.addStretch()

        layout.addLayout(adaptive_header)

        # Preset Selection
        presets_group = QtWidgets.QGroupBox("Optimization Profile")
        presets_layout = QtWidgets.QVBoxLayout(presets_group)

        self.cb_preset = NoWheelComboBox()
        self.cb_preset.addItems(list(ADAPTIVE_GRAVITY_PROFILES.keys()) + [ADAPTIVE_CUSTOM_ID])
        self.cb_preset.currentTextChanged.connect(self._on_preset_change)
        presets_layout.addWidget(self.cb_preset)

        layout.addWidget(presets_group)

        # Table Preview
        table_group = QtWidgets.QGroupBox("Altitude vs Degree Rules")
        table_layout = QtWidgets.QVBoxLayout(table_group)

        # Read-only rule preview uses the shared DataTable so it inherits
        # sorting, Ctrl+C copy, and monospace right-aligned numerics.
        self.table_preview = DataTable(
            [("Altitude", "km"), "Max Degree"],
            numeric_columns=(0, 1),
        )
        self.table_preview.setAccessibleName("Adaptive gravity rules preview")
        self.table_preview.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

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
        backend_index = max(backend_index, 0)
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
            self.table_preview.append_row([f"{alt:.1f}", str(deg)])

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
    table_km: list[tuple[float, int]] = field(default_factory=lambda: [
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
        self.setObjectName("settingsDialog")
        self.setModal(True)
        self.resize(700, 500)
        self.setMinimumSize(600, 460)
        self._cfg = cfg

        # Main Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # --- Header ---
        header = QtWidgets.QLabel("Adaptive Gravity Logic")
        header.setObjectName("dialogTitle")
        layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Reduce Spherical Harmonic degree at higher altitudes to save computation time.\n"
            "Define thresholds below. The engine interpolates between steps."
        )
        desc.setObjectName("dialogDescription")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Settings Form ---
        form_frame = QtWidgets.QFrame()
        form_frame.setObjectName("section")
        form_layout = QtWidgets.QGridLayout(form_frame)
        form_layout.setContentsMargins(16, 16, 16, 16)
        form_layout.setVerticalSpacing(12)

        # Preset Selector
        self.cb_preset = NoWheelComboBox()
        self.cb_preset.addItems(list(ADAPTIVE_GRAVITY_PROFILES.keys()) + [ADAPTIVE_CUSTOM_ID])
        self.cb_preset.setCurrentText(cfg.preset_name if cfg.preset_name in ADAPTIVE_GRAVITY_PROFILES else ADAPTIVE_CUSTOM_ID)
        self.cb_preset.currentTextChanged.connect(self._on_preset_change)

        # Interpolation Method
        self.cb_interp = NoWheelComboBox()
        self.cb_interp.addItems(["linear", "smoothstep"])
        self.cb_interp.setCurrentText(cfg.interp_method)

        # Blend Width
        self.sp_blend = NoWheelDoubleSpinBox()
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

        # Surface styling comes from the global QTableWidget#dataTable rule.
        self.table.setObjectName("dataTable")

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
        layout.addWidget(lbl, row, 0)
        layout.addWidget(widget, row, 1)

    def _create_btn(self, text, callback, primary=False):
        btn = QtWidgets.QPushButton(text)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setMinimumHeight(34)
        btn.setProperty("kind", "primary" if primary else "ghost")
        btn.clicked.connect(callback)
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

    def _read_table(self) -> list[tuple[float, int]]:
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
    """UI buffer for the lunar albedo (reflected-solar) physics model.

    Mirrors the backend ``lunaris.physics.surface_effects.AlbedoConfig`` knobs
    exposed through the CLI. The albedo *grid raster* (LOLA LDAM) itself is
    selected on the Data Files page via the Albedo Root path; this dialog owns
    only the physics-model parameters, which ``command_builder`` translates into
    ``--albedo-*`` flags.
    """

    # Backend: "lambert_facets" (facet Lambertian, default) | "simple" (legacy cannonball)
    model: str = "lambert_facets"
    # Per-facet albedo source: "constant_albedo" | "scaled_dn_grid" (samples Albedo Root)
    source: str = "constant_albedo"
    albedo_const: float = 0.12          # constant lunar albedo in [0, 1]
    pressure_coefficient: float = 1.0   # C_R_albedo (facet model; distinct from SRP cr)
    facet_lat_count: int = 18
    facet_lon_count: int = 36
    enable_eclipse: bool = True         # lunar-eclipse (Earth-umbra) dimming
    require_provider: bool = False      # fail-closed: demand a real surface provider


class AlbedoSettingsDialog(QtWidgets.QDialog):
    """Configuration dialog for the lunar albedo (reflected-solar) model.

    Exposes the backend ``AlbedoConfig`` knobs the CLI honors: backend
    (lambert_facets / simple), albedo source (constant / surface grid), constant
    albedo, radiation-pressure coefficient, facet resolution, and lunar-eclipse
    dimming. The albedo grid raster itself is selected on the Data Files page.
    """

    def __init__(self, parent: QtWidgets.QWidget, cfg: UIAlbedoConfig):
        super().__init__(parent)
        self.setWindowTitle("Albedo Model Configuration")
        self.setObjectName("settingsDialog")
        self.setModal(True)
        self.resize(620, 540)
        self.setMinimumSize(560, 500)
        self._cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        header = QtWidgets.QLabel("Lunar Albedo (Reflected Solar) Model")
        header.setObjectName("dialogTitle")
        layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Radiation pressure from sunlight reflected off the lunar surface. The "
            "facet model sums Lambertian contributions from facets that are both "
            "sunlit and visible. This is reflected solar radiation, not gravity."
        )
        desc.setWordWrap(True)
        desc.setObjectName("dialogDescription")
        layout.addWidget(desc)

        form_frame = QtWidgets.QFrame()
        form_frame.setObjectName("section")
        form = QtWidgets.QFormLayout(form_frame)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(12)

        self.cb_model = NoWheelComboBox()
        self.cb_model.addItem("Lambertian facets (recommended)", "lambert_facets")
        self.cb_model.addItem("Simple cannonball (legacy)", "simple")
        form.addRow("Backend:", self.cb_model)

        self.cb_source = NoWheelComboBox()
        self.cb_source.addItem("Constant albedo", "constant_albedo")
        self.cb_source.addItem("Surface grid (LOLA Albedo Root)", "scaled_dn_grid")
        form.addRow("Albedo source:", self.cb_source)

        self.sp_const = NoWheelDoubleSpinBox()
        self.sp_const.setRange(0.0, 1.0)
        self.sp_const.setSingleStep(0.01)
        self.sp_const.setDecimals(3)
        form.addRow("Constant albedo:", self.sp_const)

        self.sp_pcoef = NoWheelDoubleSpinBox()
        self.sp_pcoef.setRange(0.0, 5.0)
        self.sp_pcoef.setSingleStep(0.1)
        self.sp_pcoef.setDecimals(2)
        form.addRow("Pressure coefficient (C_R):", self.sp_pcoef)

        facet_row = QtWidgets.QHBoxLayout()
        self.sp_lat = NoWheelSpinBox()
        self.sp_lat.setRange(1, 180)
        self.sp_lon = NoWheelSpinBox()
        self.sp_lon.setRange(1, 360)
        facet_row.addWidget(self.sp_lat)
        facet_row.addWidget(QtWidgets.QLabel("x"))
        facet_row.addWidget(self.sp_lon)
        facet_row.addWidget(QtWidgets.QLabel("(lat x lon)"))
        facet_row.addStretch()
        facet_holder = QtWidgets.QWidget()
        facet_holder.setLayout(facet_row)
        form.addRow("Facet resolution:", facet_holder)

        self.chk_eclipse = QtWidgets.QCheckBox("Apply lunar-eclipse (Earth-umbra) dimming")
        form.addRow("", self.chk_eclipse)

        self.chk_require_provider = QtWidgets.QCheckBox("Require real provider (fail-closed)")
        self.chk_require_provider.setToolTip(
            "Refuse to run with the constant-albedo fallback: the run fails "
            "unless a real surface provider (Albedo Root grid) is available."
        )
        form.addRow("", self.chk_require_provider)

        layout.addWidget(form_frame)

        self.lbl_note = QtWidgets.QLabel()
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setObjectName("fieldHint")
        layout.addWidget(self.lbl_note)

        layout.addStretch(1)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_save = QtWidgets.QPushButton("Apply Settings")
        for btn in (self.btn_cancel, self.btn_save):
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setMinimumHeight(34)
        self.btn_cancel.setProperty("kind", "ghost")
        self.btn_save.setProperty("kind", "primary")
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)
        self.cb_model.currentIndexChanged.connect(self._sync_enabled)
        self.cb_source.currentIndexChanged.connect(self._sync_enabled)

        self._load_current_config()

    def _load_current_config(self) -> None:
        idx = self.cb_model.findData(str(getattr(self._cfg, "model", "lambert_facets")))
        self.cb_model.setCurrentIndex(max(idx, 0))
        sdx = self.cb_source.findData(str(getattr(self._cfg, "source", "constant_albedo")))
        self.cb_source.setCurrentIndex(max(sdx, 0))
        self.sp_const.setValue(float(getattr(self._cfg, "albedo_const", 0.12)))
        self.sp_pcoef.setValue(float(getattr(self._cfg, "pressure_coefficient", 1.0)))
        self.sp_lat.setValue(int(getattr(self._cfg, "facet_lat_count", 18)))
        self.sp_lon.setValue(int(getattr(self._cfg, "facet_lon_count", 36)))
        self.chk_eclipse.setChecked(bool(getattr(self._cfg, "enable_eclipse", True)))
        self.chk_require_provider.setChecked(bool(getattr(self._cfg, "require_provider", False)))
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        is_facet = self.cb_model.currentData() == "lambert_facets"
        is_grid = self.cb_source.currentData() == "scaled_dn_grid"
        # Facet-only knobs are meaningless for the legacy cannonball backend.
        self.sp_pcoef.setEnabled(is_facet)
        self.sp_lat.setEnabled(is_facet)
        self.sp_lon.setEnabled(is_facet)
        self.chk_eclipse.setEnabled(is_facet)
        if is_grid:
            self.lbl_note.setText(
                "Surface-grid mode samples the Albedo Root raster configured on the "
                "Data Files page (constant albedo is the nodata fallback)."
            )
        else:
            self.lbl_note.setText(
                "Constant mode applies a single albedo everywhere; no surface grid required."
            )

    def _on_save(self, _checked: bool = False) -> None:
        self._cfg.model = str(self.cb_model.currentData() or "lambert_facets")
        self._cfg.source = str(self.cb_source.currentData() or "constant_albedo")
        self._cfg.albedo_const = float(self.sp_const.value())
        self._cfg.pressure_coefficient = float(self.sp_pcoef.value())
        self._cfg.facet_lat_count = int(self.sp_lat.value())
        self._cfg.facet_lon_count = int(self.sp_lon.value())
        self._cfg.enable_eclipse = bool(self.chk_eclipse.isChecked())
        self._cfg.require_provider = bool(self.chk_require_provider.isChecked())
        self.accept()


# =============================================================================
# 2b.                              THERMAL IR
# =============================================================================

@dataclass
class UIThermalConfig:
    """UI buffer for the lunar thermal-IR (re-radiation) physics model.

    Mirrors the backend ``lunaris.physics.surface_effects.ThermalConfig`` knobs
    exposed through the CLI ``--thermal-*`` flags. Defaults match the backend
    dataclass so an untouched dialog reproduces engine behavior exactly.
    """

    mode: str = "constant_temperature"  # constant_temperature | equilibrium_temperature | temperature_grid
    temperature_k: float = 250.0        # constant-mode surface temperature [K]
    night_temperature_k: float = 100.0  # equilibrium-mode night/floor temperature [K]
    emissivity: float = 0.95            # surface emissivity [0,1]
    surface_albedo: float = 0.12        # equilibrium-mode absorbed-solar albedo [0,1]
    ir_coefficient: float = 1.0         # spacecraft IR pressure coefficient [-]
    floor_flux_w_m2: float = 0.0        # minimum thermal exitance floor [W/m^2]
    facet_lat_count: int = 18
    facet_lon_count: int = 36


class ThermalSettingsDialog(QtWidgets.QDialog):
    """Configuration dialog for the lunar thermal-IR re-radiation model.

    Follows the Albedo dialog pattern: the dialog edits the shared
    ``UIThermalConfig`` in place on Apply, and ``command_builder`` translates
    the fields into ``--thermal-*`` CLI flags.
    """

    def __init__(self, parent: QtWidgets.QWidget, cfg: UIThermalConfig):
        super().__init__(parent)
        self.setWindowTitle("Thermal IR Model Configuration")
        self.setObjectName("settingsDialog")
        self.setModal(True)
        self.resize(620, 560)
        self.setMinimumSize(560, 520)
        self._cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        header = QtWidgets.QLabel("Lunar Thermal IR (Re-radiation) Model")
        header.setObjectName("dialogTitle")
        layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Radiation pressure from the Moon's own thermal emission. Facets "
            "radiate by their surface temperature: a constant value, a "
            "day/night equilibrium model, or a temperature grid."
        )
        desc.setWordWrap(True)
        desc.setObjectName("dialogDescription")
        layout.addWidget(desc)

        form_frame = QtWidgets.QFrame()
        form_frame.setObjectName("section")
        form = QtWidgets.QFormLayout(form_frame)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(12)

        self.cb_mode = NoWheelComboBox()
        self.cb_mode.addItem("Constant temperature", "constant_temperature")
        self.cb_mode.addItem("Day/night equilibrium", "equilibrium_temperature")
        self.cb_mode.addItem("Temperature grid", "temperature_grid")
        form.addRow("Mode:", self.cb_mode)

        self.sp_temperature = NoWheelDoubleSpinBox()
        self.sp_temperature.setRange(0.0, 500.0)
        self.sp_temperature.setDecimals(1)
        self.sp_temperature.setSuffix(" K")
        form.addRow("Surface temperature:", self.sp_temperature)

        self.sp_night_temperature = NoWheelDoubleSpinBox()
        self.sp_night_temperature.setRange(0.0, 500.0)
        self.sp_night_temperature.setDecimals(1)
        self.sp_night_temperature.setSuffix(" K")
        form.addRow("Night temperature:", self.sp_night_temperature)

        self.sp_emissivity = NoWheelDoubleSpinBox()
        self.sp_emissivity.setRange(0.0, 1.0)
        self.sp_emissivity.setSingleStep(0.01)
        self.sp_emissivity.setDecimals(3)
        form.addRow("Emissivity:", self.sp_emissivity)

        self.sp_surface_albedo = NoWheelDoubleSpinBox()
        self.sp_surface_albedo.setRange(0.0, 1.0)
        self.sp_surface_albedo.setSingleStep(0.01)
        self.sp_surface_albedo.setDecimals(3)
        form.addRow("Absorbed-solar albedo:", self.sp_surface_albedo)

        self.sp_ir_coefficient = NoWheelDoubleSpinBox()
        self.sp_ir_coefficient.setRange(0.0, 5.0)
        self.sp_ir_coefficient.setSingleStep(0.1)
        self.sp_ir_coefficient.setDecimals(2)
        form.addRow("IR pressure coefficient:", self.sp_ir_coefficient)

        self.sp_floor_flux = NoWheelDoubleSpinBox()
        self.sp_floor_flux.setRange(0.0, 2000.0)
        self.sp_floor_flux.setDecimals(1)
        self.sp_floor_flux.setSuffix(" W/m²")
        form.addRow("Exitance floor:", self.sp_floor_flux)

        facet_row = QtWidgets.QHBoxLayout()
        self.sp_lat = NoWheelSpinBox()
        self.sp_lat.setRange(1, 180)
        self.sp_lon = NoWheelSpinBox()
        self.sp_lon.setRange(1, 360)
        facet_row.addWidget(self.sp_lat)
        facet_row.addWidget(QtWidgets.QLabel("x"))
        facet_row.addWidget(self.sp_lon)
        facet_row.addWidget(QtWidgets.QLabel("(lat x lon)"))
        facet_row.addStretch()
        facet_holder = QtWidgets.QWidget()
        facet_holder.setLayout(facet_row)
        form.addRow("Facet resolution:", facet_holder)

        layout.addWidget(form_frame)
        layout.addStretch(1)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_save = QtWidgets.QPushButton("Apply Settings")
        for btn in (self.btn_cancel, self.btn_save):
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setMinimumHeight(34)
        self.btn_cancel.setProperty("kind", "ghost")
        self.btn_save.setProperty("kind", "primary")
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)
        self.cb_mode.currentIndexChanged.connect(self._sync_enabled)

        self._load_current_config()

    def _load_current_config(self) -> None:
        idx = self.cb_mode.findData(str(getattr(self._cfg, "mode", "constant_temperature")))
        self.cb_mode.setCurrentIndex(max(idx, 0))
        self.sp_temperature.setValue(float(getattr(self._cfg, "temperature_k", 250.0)))
        self.sp_night_temperature.setValue(float(getattr(self._cfg, "night_temperature_k", 100.0)))
        self.sp_emissivity.setValue(float(getattr(self._cfg, "emissivity", 0.95)))
        self.sp_surface_albedo.setValue(float(getattr(self._cfg, "surface_albedo", 0.12)))
        self.sp_ir_coefficient.setValue(float(getattr(self._cfg, "ir_coefficient", 1.0)))
        self.sp_floor_flux.setValue(float(getattr(self._cfg, "floor_flux_w_m2", 0.0)))
        self.sp_lat.setValue(int(getattr(self._cfg, "facet_lat_count", 18)))
        self.sp_lon.setValue(int(getattr(self._cfg, "facet_lon_count", 36)))
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        mode = str(self.cb_mode.currentData() or "constant_temperature")
        # Constant temperature only matters in constant mode; night floor and
        # absorbed-solar albedo only matter in equilibrium mode.
        self.sp_temperature.setEnabled(mode == "constant_temperature")
        self.sp_night_temperature.setEnabled(mode == "equilibrium_temperature")
        self.sp_surface_albedo.setEnabled(mode == "equilibrium_temperature")

    def _on_save(self, _checked: bool = False) -> None:
        self._cfg.mode = str(self.cb_mode.currentData() or "constant_temperature")
        self._cfg.temperature_k = float(self.sp_temperature.value())
        self._cfg.night_temperature_k = float(self.sp_night_temperature.value())
        self._cfg.emissivity = float(self.sp_emissivity.value())
        self._cfg.surface_albedo = float(self.sp_surface_albedo.value())
        self._cfg.ir_coefficient = float(self.sp_ir_coefficient.value())
        self._cfg.floor_flux_w_m2 = float(self.sp_floor_flux.value())
        self._cfg.facet_lat_count = int(self.sp_lat.value())
        self._cfg.facet_lon_count = int(self.sp_lon.value())
        self.accept()


# =============================================================================
# 3.                             FORCE MODEL
# =============================================================================

class ForceModelsPage(QtWidgets.QWidget):
    """
    Page 2: Force Model Settings.
    Encapsulates all widgets (sw_gravity, sw_sun, sw_earth, etc.) inside this page.
    """

    # Emitted by the Spacecraft Bus shortcut in the Non-Gravitational Forces card.
    # The spacecraft's mass/area/reflectivity are what scale SRP, albedo, and
    # thermal, so the editor now lives next to those forces (it used to sit on the
    # Propagation page, away from the forces it actually drives).
    spacecraft_settings_requested = QtCore.Signal()

    def __init__(
        self,
        gravity_cfg: UIGravityConfig | None = None,
        albedo_cfg: UIAlbedoConfig | None = None,
        thermal_cfg: UIThermalConfig | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        # Keep references so dialogs update the SAME objects MainWindow uses
        self.gravity_cfg: UIGravityConfig = gravity_cfg if gravity_cfg is not None else UIGravityConfig()
        self.albedo_cfg: UIAlbedoConfig = albedo_cfg if albedo_cfg is not None else UIAlbedoConfig()
        self.thermal_cfg: UIThermalConfig = thermal_cfg if thermal_cfg is not None else UIThermalConfig()

        # Build UI into self
        self._build_page_forces()

        # Post-wiring consistency
        self._sync_albedo_settings_button()
        self._sync_force_dependencies()

    def get_data(self) -> dict[str, Any]:
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
            "tide_k2_value": self.ent_tide_k2.text().strip(),
            "tide_k3_value": self.ent_tide_k3.text().strip(),
            "tide_r_ref_m": self.ent_tide_r_ref.text().strip(),
            "tide_bodies": str(self.cb_tide_bodies.currentData() or ""),
            "relativity_1pn": bool(self.sw_relativity_1pn.isChecked()),
        }

    def load_data(self, data: dict[str, Any]) -> None:
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
        self.ent_tide_k2.setText(str(data.get("tide_k2_value", "") or ""))
        self.ent_tide_k3.setText(str(data.get("tide_k3_value", "") or ""))
        self.ent_tide_r_ref.setText(str(data.get("tide_r_ref_m", "") or ""))
        bodies_idx = self.cb_tide_bodies.findData(str(data.get("tide_bodies", "") or ""))
        self.cb_tide_bodies.setCurrentIndex(max(bodies_idx, 0))
        self.sw_relativity_1pn.setChecked(bool(data.get("relativity_1pn", False)))

        self._update_gravity_summary_ui()
        self._sync_force_dependencies()


    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _create_card(self, title: str) -> Section:
        """
        Create a force-page card using the shared ``Section`` primitive.

        Surface, border, radius, and title typography come from the global
        stylesheet (``QFrame#section`` / ``QLabel#sectionTitle``) instead of
        per-card inline QSS, so every page card stays visually consistent and
        token-driven.
        """

        return Section(title)

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
        - SRP/Albedo: warn when Sun is not enabled
        - Earth J2: warn when Earth third-body is not enabled
        """
        sun_on = hasattr(self, "sw_sun") and self.sw_sun.isChecked()
        srp_on = hasattr(self, "sw_srp") and self.sw_srp.isChecked()
        albedo_on = hasattr(self, "sw_albedo") and self.sw_albedo.isChecked()
        earth_on = hasattr(self, "sw_earth") and self.sw_earth.isChecked()
        earth_j2_on = hasattr(self, "sw_earth_j2") and self.sw_earth_j2.isChecked()

        # Warning: SRP/Albedo without Sun. The UI thermal toggle uses the
        # constant-temperature thermal IR mode, which only needs Moon attitude.
        if hasattr(self, "lbl_warn_srp_sun"):
            needs_sun = srp_on or albedo_on
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

        # Accessibility: each ToggleSwitch is icon-only paint, so give it the
        # adjacent control's meaning explicitly for keyboard/screen-reader users.
        for switch, name in (
            (self.sw_gravity, "Lunar gravity field"),
            (self.sw_sun, "Sun point-mass perturbation"),
            (self.sw_earth, "Earth point-mass perturbation"),
            (self.sw_earth_j2, "Earth J2 oblateness perturbation"),
            (self.sw_srp, "Solar radiation pressure"),
            (self.sw_albedo, "Lunar albedo"),
            (self.sw_thermal, "Thermal re-radiation"),
            (self.sw_tides_k2, "Solid tides k2 Love number"),
            (self.sw_tides_k3, "Solid tides k3 Love number"),
            (self.sw_relativity_1pn, "1PN relativistic correction"),
        ):
            switch.setAccessibleName(name)

        layout.addStretch(1)
        return self

    def _group_gravity_force(self) -> Section:
        """Gravity force card with settings dialog button."""
        gb = self._create_card("Central Body Gravity")
        layout = gb.content_layout

        # Header with toggle and settings button
        header = QtWidgets.QHBoxLayout()

        self.sw_gravity = ToggleSwitch()
        self.sw_gravity.setChecked(True)
        header.addWidget(self.sw_gravity)

        header.addWidget(QtWidgets.QLabel("Lunar Gravity Field"))

        header.addStretch()

        # Settings button
        self.btn_gravity_settings = QtWidgets.QPushButton()
        self.btn_gravity_settings.setObjectName("iconButton")
        self.btn_gravity_settings.setIcon(get_icon("fa6s.gear", THEME["fg_main"]))
        self.btn_gravity_settings.setToolTip("Configure Gravity Model")
        self.btn_gravity_settings.setAccessibleName("Configure gravity model")
        self.btn_gravity_settings.clicked.connect(self._on_gravity_settings)
        header.addWidget(self.btn_gravity_settings)

        layout.addLayout(header)

        # Status indicator
        self.lbl_gravity_status = QtWidgets.QLabel("Default model loaded")
        self.lbl_gravity_status.setObjectName("statusLabel")
        self.lbl_gravity_status.setWordWrap(True)
        layout.addWidget(self.lbl_gravity_status)

        # Push the cost summary to the bottom so this sparser card fills its
        # (stretched) height intentionally and aligns with the denser
        # third-body card beside it, rather than leaving a dead gap below.
        layout.addStretch(1)

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

    def _group_thirdbody_force(self) -> Section:
        """Third-body perturbations card."""
        gb = self._create_card("Third-Body Perturbations")
        layout = gb.content_layout

        # Sun perturbation
        sun_row = QtWidgets.QHBoxLayout()
        self.sw_sun = ToggleSwitch()
        self.sw_sun.setChecked(True)
        sun_row.addWidget(self.sw_sun)

        sun_row.addWidget(QtWidgets.QLabel("Sun (Point Mass)"))
        sun_row.addStretch()

        self.ind_sun_cost = CostIndicator("medium")
        sun_row.addWidget(self.ind_sun_cost)
        layout.addLayout(sun_row)

        # Earth point mass perturbation
        earth_row = QtWidgets.QHBoxLayout()
        self.sw_earth = ToggleSwitch()
        self.sw_earth.setChecked(True)
        earth_row.addWidget(self.sw_earth)

        earth_row.addWidget(QtWidgets.QLabel("Earth (Point Mass)"))
        earth_row.addStretch()

        self.ind_earth_cost = CostIndicator("medium")
        earth_row.addWidget(self.ind_earth_cost)
        layout.addLayout(earth_row)

        # Earth J2 perturbation
        earth_j2_row = QtWidgets.QHBoxLayout()
        self.sw_earth_j2 = ToggleSwitch()
        self.sw_earth_j2.setChecked(False)
        earth_j2_row.addWidget(self.sw_earth_j2)

        earth_j2_row.addWidget(QtWidgets.QLabel("Earth J2 (Oblateness)"))
        earth_j2_row.addStretch()

        self.ind_earth_j2_cost = CostIndicator("low")
        earth_j2_row.addWidget(self.ind_earth_j2_cost)
        layout.addLayout(earth_j2_row)

        # Warning: Earth J2 requires Earth third-body
        self.lbl_warn_earth_j2 = InlineNotice(
            "Earth J2 requires the Earth third-body perturbation to be enabled.",
            "warning",
        )
        self.lbl_warn_earth_j2.setVisible(False)
        layout.addWidget(self.lbl_warn_earth_j2)

        # Wiring dependencies
        self.sw_sun.toggled.connect(lambda _v: self._sync_force_dependencies())
        self.sw_earth.toggled.connect(lambda _v: self._sync_force_dependencies())
        self.sw_earth_j2.toggled.connect(lambda _v: self._sync_force_dependencies())

        return gb

    def _group_nongrav_force(self) -> Section:
        """Non-gravitational forces card."""
        gb = self._create_card("Non-Gravitational Forces")
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(16)
        # Let the label column absorb the slack so the cost indicator and the
        # per-row settings button stay grouped at the right edge instead of the
        # four columns spreading apart (which left the lone gear button adrift).
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 0)

        # SRP
        self.sw_srp = ToggleSwitch()
        self.sw_srp.setChecked(False)
        grid.addWidget(self.sw_srp, 0, 0)

        grid.addWidget(QtWidgets.QLabel("Solar Radiation Pressure"), 0, 1)

        self.ind_srp_cost = CostIndicator("medium")
        grid.addWidget(self.ind_srp_cost, 0, 2)

        # Albedo
        self.sw_albedo = ToggleSwitch()
        self.sw_albedo.setChecked(False)
        grid.addWidget(self.sw_albedo, 1, 0)

        grid.addWidget(QtWidgets.QLabel("Lunar Albedo"), 1, 1)

        # Albedo uses the same lat-lon facet machinery as thermal IR (default
        # 18x36 facets), so its per-step cost sits alongside thermal, not above it.
        self.ind_albedo_cost = CostIndicator("medium")
        grid.addWidget(self.ind_albedo_cost, 1, 2)

        # Albedo settings button
        self.btn_albedo_settings = QtWidgets.QPushButton()
        self.btn_albedo_settings.setObjectName("iconButton")
        self.btn_albedo_settings.setIcon(get_icon("fa6s.gear", THEME["fg_main"]))
        self.btn_albedo_settings.setToolTip("Configure Albedo Model")
        self.btn_albedo_settings.setAccessibleName("Configure albedo model")
        self.btn_albedo_settings.clicked.connect(self._on_albedo_settings)
        grid.addWidget(self.btn_albedo_settings, 1, 3)

        # Thermal
        self.sw_thermal = ToggleSwitch()
        self.sw_thermal.setChecked(False)
        grid.addWidget(self.sw_thermal, 2, 0)

        grid.addWidget(QtWidgets.QLabel("Thermal Re-radiation"), 2, 1)

        self.ind_thermal_cost = CostIndicator("medium")
        grid.addWidget(self.ind_thermal_cost, 2, 2)

        # Thermal settings button (Albedo dialog pattern)
        self.btn_thermal_settings = QtWidgets.QPushButton()
        self.btn_thermal_settings.setObjectName("iconButton")
        self.btn_thermal_settings.setIcon(get_icon("fa6s.gear", THEME["fg_main"]))
        self.btn_thermal_settings.setToolTip("Configure Thermal IR Model")
        self.btn_thermal_settings.setAccessibleName("Configure thermal IR model")
        self.btn_thermal_settings.clicked.connect(self._on_thermal_settings)
        grid.addWidget(self.btn_thermal_settings, 2, 3)
        self.sw_thermal.toggled.connect(
            lambda on: self.btn_thermal_settings.setEnabled(bool(on))
        )
        self.btn_thermal_settings.setEnabled(self.sw_thermal.isChecked())

        gb.content_layout.addLayout(grid)

        # Warning: SRP/Albedo/Thermal require Sun position
        self.lbl_warn_srp_sun = InlineNotice(
            "SRP, albedo, and thermal re-radiation require the Sun perturbation to be enabled.",
            "warning",
        )
        self.lbl_warn_srp_sun.setVisible(False)
        gb.content_layout.addWidget(self.lbl_warn_srp_sun)

        # Connect SRP/Albedo to require Sun perturbation
        self.sw_srp.toggled.connect(self._sync_srp_requirement)
        self.sw_albedo.toggled.connect(self._sync_srp_requirement)
        self.sw_albedo.toggled.connect(self._sync_albedo_settings_button)
        self.sw_thermal.toggled.connect(self._sync_srp_requirement)

        # Spacecraft bus shortcut. Mass, area, and reflectivity are exactly what
        # scale SRP / albedo / thermal, so the bus editor belongs here next to the
        # forces it drives — and it fills this card's otherwise empty lower band.
        divider = QtWidgets.QFrame()
        divider.setObjectName("formDivider")
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        divider.setFixedHeight(1)
        gb.content_layout.addWidget(divider)

        sc_row = QtWidgets.QHBoxLayout()
        sc_row.setContentsMargins(0, 0, 0, 0)
        sc_caption = QtWidgets.QLabel(
            "Spacecraft mass, area & reflectivity scale these forces."
        )
        sc_caption.setObjectName("sectionDescription")
        sc_caption.setWordWrap(True)
        sc_row.addWidget(sc_caption, 1)

        self.btn_spacecraft_settings = QtWidgets.QPushButton("Spacecraft Bus…")
        self.btn_spacecraft_settings.setProperty("kind", "ghost")
        self.btn_spacecraft_settings.setIcon(get_icon("fa6s.rocket", THEME["fg_main"]))
        self.btn_spacecraft_settings.setToolTip(
            "Open spacecraft mass, area, drag, and reflectivity settings."
        )
        self.btn_spacecraft_settings.setAccessibleName("Spacecraft bus settings")
        self.btn_spacecraft_settings.clicked.connect(self.spacecraft_settings_requested.emit)
        sc_row.addWidget(self.btn_spacecraft_settings, 0)
        gb.content_layout.addLayout(sc_row)

        return gb

    def _group_tides_force(self) -> Section:
        """Solid tides force card."""
        gb = self._create_card("Solid Body Tides")
        layout = gb.content_layout

        # k2 Love number
        k2_row = QtWidgets.QHBoxLayout()
        self.sw_tides_k2 = ToggleSwitch()
        self.sw_tides_k2.setChecked(True)
        k2_row.addWidget(self.sw_tides_k2)

        k2_row.addWidget(QtWidgets.QLabel("k2 Love Number (Degree 2)"))
        k2_row.addStretch()

        self.ind_tides_k2_cost = CostIndicator("low")
        k2_row.addWidget(self.ind_tides_k2_cost)
        layout.addLayout(k2_row)

        # k3 Love number
        k3_row = QtWidgets.QHBoxLayout()
        self.sw_tides_k3 = ToggleSwitch()
        self.sw_tides_k3.setChecked(False)
        k3_row.addWidget(self.sw_tides_k3)

        k3_row.addWidget(QtWidgets.QLabel("k3 Love Number (Degree 3)"))
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
        note.setObjectName("fieldHint")
        note.setWordWrap(True)
        layout.addWidget(note)

        # Advanced values (CLI parity: --tide-k2/--tide-k3/--tide-bodies/
        # --tide-r-ref-m). Blank fields keep the engine defaults; the command
        # builder emits a flag only when a value is entered.
        adv_grid = QtWidgets.QGridLayout()
        adv_grid.setHorizontalSpacing(12)
        adv_grid.setVerticalSpacing(8)

        def _tide_value_row(row: int, caption: str, placeholder: str, tooltip: str) -> QtWidgets.QLineEdit:
            lbl = QtWidgets.QLabel(caption)
            lbl.setObjectName("fieldLabel")
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setAccessibleName(caption)
            edit.setToolTip(tooltip)
            edit.setMaximumWidth(220)
            adv_grid.addWidget(lbl, row, 0)
            adv_grid.addWidget(edit, row, 1)
            return edit

        self.ent_tide_k2 = _tide_value_row(
            0, "k2 value", "engine default (0.02416)",
            "Degree-2 lunar potential Love number. Blank keeps the engine default.",
        )
        self.ent_tide_k3 = _tide_value_row(
            1, "k3 value", "engine default",
            "Degree-3 lunar potential Love number (used when the k3 row is on).",
        )
        self.ent_tide_r_ref = _tide_value_row(
            2, "Reference radius [m]", "engine default",
            "Lunar tide reference radius in meters. Blank keeps the engine default.",
        )

        bodies_lbl = QtWidgets.QLabel("Tide bodies")
        bodies_lbl.setObjectName("fieldLabel")
        self.cb_tide_bodies = NoWheelComboBox()
        self.cb_tide_bodies.setAccessibleName("Tide-raising bodies")
        self.cb_tide_bodies.addItem("Engine default", "")
        self.cb_tide_bodies.addItem("Earth", "earth")
        self.cb_tide_bodies.addItem("Earth + Sun", "earth,sun")
        self.cb_tide_bodies.addItem("Sun", "sun")
        adv_grid.addWidget(bodies_lbl, 3, 0)
        adv_grid.addWidget(self.cb_tide_bodies, 3, 1)
        adv_grid.setColumnStretch(2, 1)
        layout.addLayout(adv_grid)

        return gb

    def _group_relativity_force(self) -> Section:
        """General Relativity force card."""
        gb = self._create_card("General Relativity")
        layout = gb.content_layout

        # 1PN correction
        pn_row = QtWidgets.QHBoxLayout()
        self.sw_relativity_1pn = ToggleSwitch()
        self.sw_relativity_1pn.setChecked(False)
        pn_row.addWidget(self.sw_relativity_1pn)

        pn_row.addWidget(QtWidgets.QLabel("1PN Force (Post-Newtonian)"))
        pn_row.addStretch()

        self.ind_relativity_cost = CostIndicator("low")
        pn_row.addWidget(self.ind_relativity_cost)
        layout.addLayout(pn_row)

        # Info note
        note = QtWidgets.QLabel("Note: First-order post-Newtonian correction for high-precision lunar orbits.")
        note.setObjectName("fieldHint")
        note.setWordWrap(True)
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

    def _on_thermal_settings(self, _checked: bool = False):
        """Open ThermalSettingsDialog and apply changes to bound thermal_cfg."""
        try:
            dlg = ThermalSettingsDialog(self.window() or self, self.thermal_cfg)
            dlg.exec()  # config object is updated in-place by dialog on save
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Thermal Settings", f"Could not open thermal settings:\n\n{e}"
            )

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

    def dump_force_state(p: ForceModelsPage) -> dict:
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
