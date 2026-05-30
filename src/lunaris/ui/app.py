# -*- coding: utf-8 -*-
"""
PySide6 desktop interface for the general Lunaris orbit simulator.

This module hosts the main application window and wires the modular page widgets
from `lunaris.ui.widgets` into a single desktop workflow.
"""



# =============================================================================
# 0.                                    IMPORTS 
# =============================================================================
from __future__ import annotations

import json
import os
import re
import sys
import ast
import subprocess
import time
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

from PySide6 import QtCore, QtGui, QtWidgets




# =============================================================================
# 1.                            UI CONFIGURATION
# =============================================================================
from lunaris.ui.widgets.ui_commons import (
    APP_NAME,
    APP_VERSION,
    ASSETS_DIR as UI_ASSETS_DIR,
    LOG_COLORS,
    THEME,
    WINDOW_SETTINGS,
)

# Updated navigation for 6 specialized pages (added Data & Files)
NAV_PAGES = [
    ("Orbit",       "Orbit Setup",       "fa6s.rocket"),
    ("Forces",      "Force Models",      "fa6s.atom"),
    ("Propagation", "Propagation",       "fa6s.hourglass-half"),
    ("Output",      "Results & Export",  "fa6s.folder-open"),
    ("Telemetry",   "Live Telemetry",    "fa6s.chart-line"),
    ("Data",        "Data & Files",      "fa6s.database"),
    ("MonteCarlo",  "Monte Carlo",       "fa6s.dice"),
    ("Surrogate",   "ST-LRPS Studio",    "fa6s.brain"),
]

# Default UI values (Internal SI units convention)
DEFAULT_UI_STATE = {
    "hp_km": 50.0,
    "ha_km": 50.0,
    "mass_kg": 1000.0,
    "area_m2": 5.0,
    "cr": 1.5,
    "dt_out_s": 60.0,
}


# =============================================================================
# 3.                          FONT LOADING
# =============================================================================
from lunaris.ui.widgets.ui_commons import find_project_root, load_fonts

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = UI_ASSETS_DIR


# =============================================================================
# 4.                          ICON UTILITIES
# =============================================================================
from lunaris.ui.widgets.ui_commons import get_icon


# =============================================================================
# 5.                          UTILITY HELPERS
# =============================================================================
from lunaris.ui.widgets.ui_commons import normalize_path

from lunaris.ui.widgets.force_models_page import find_best_gravity_file
from lunaris.ui.widgets.command_builder import build_command, build_command_preview, build_preflight_snapshot, build_mc_command
from lunaris.ui.widgets.preflight_validation import PreFlightWorker
from lunaris.ui.widgets.result_exports_page import OutputPageState, ResultsExportPage
from lunaris.ui.widgets.solver_policy import normalize_solver_config_object
from lunaris.ui.widgets.session_persistence import (
    apply_session_snapshot,
    apply_visual_state,
    autodetect_data_state,
    collect_session_snapshot,
    collect_visual_state,
)
from lunaris.ui.widgets.surrogate_studio_page import SurrogateStudioPage



# =============================================================================
# 6.                        DATACLASSES (main glue)
# =============================================================================
from lunaris.ui.widgets.mission_propagation_page import UISolverConfig, UISpacecraftConfig

@dataclass
class SimulationState:
    """Tracks the current engine status for UI synchronization."""
    status: str = "idle"
    message: str = ""
    progress: int = 0
    start_time: float = 0.0
    total_duration: float = 0.0  # In seconds



# =============================================================================
# 7.                        CUSTOM UI PRIMITIVES
# =============================================================================
from lunaris.ui.widgets.ui_commons import StatusBadge


# =============================================================================
# 10.                       GRAVITY CONFIGURATION
# =============================================================================
from lunaris.ui.widgets.force_models_page import UIGravityConfig



# =============================================================================
# 11.                       SOLVER SETTINGS DIALOG
# =============================================================================
from lunaris.ui.widgets.mission_propagation_page import SolverSettingsDialog



# =============================================================================
# 12.                       SPACECRAFT BUILDER DIALOG
# =============================================================================
from lunaris.ui.widgets.mission_propagation_page import SpacecraftBusDialog



# =============================================================================
# 15.                       ALBEDO CONFIGURATION
# =============================================================================
from lunaris.ui.widgets.force_models_page import UIAlbedoConfig



# =============================================================================
# 16.                       UI HELPERS
# =============================================================================

def _make_lbl(text: str, style: str = "") -> QtWidgets.QLabel:
    """Small helper to create a plain styled QLabel without importing Qt in tests."""
    lbl = QtWidgets.QLabel(text)
    if style:
        lbl.setStyleSheet(style)
    return lbl


# =============================================================================
# 17.                       MAIN WINDOW APPLICATION
# =============================================================================

class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the modular ST-LRPS Studio UI.

    The window now acts primarily as an orchestration layer: individual pages
    own their widgets and page-local state, while the main window coordinates
    cross-page workflows such as session persistence, command building,
    pre-flight validation, and backend process management.
    """
    
    def __init__(self):
        super().__init__()
        
        # ---------------------------------------------------------------------
        # 1. Window Configuration
        # ---------------------------------------------------------------------
        self.setWindowTitle(WINDOW_SETTINGS["title"])
        self.resize(*WINDOW_SETTINGS["size"])
        self.setMinimumSize(*WINDOW_SETTINGS["min_size"])
        
        # Icon setup
        icon_path = ASSETS_DIR / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        
        # ---------------------------------------------------------------------
        # 2. Path & Session Management
        # ---------------------------------------------------------------------
        # The backend is launched as a subprocess via the installed `lunaris`
        # package modules (`python <module file>`), not via root-level launcher
        # scripts. Resolve the module files from the package directory so this
        # works regardless of where `lunaris` is installed.
        _lunaris_pkg = Path(__file__).resolve().parents[1]
        self.main_script_path = _lunaris_pkg / "cli" / "main.py"
        
        # Session Persistence
        app_data_loc = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)
        self.app_data_dir = Path(app_data_loc) / "STLRPSStudio" if app_data_loc else Path.home() / ".stlrps_studio"
        self.app_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_path = self.app_data_dir / "studio_session.json"
        
        # ---------------------------------------------------------------------
        # 3. Application State & Sub-Configs
        # ---------------------------------------------------------------------
        self.process: Optional[QtCore.QProcess] = None
        self.mc_process: Optional[QtCore.QProcess] = None
        self.preflight_worker: Optional[PreFlightWorker] = None
        self.mc_script_path   = _lunaris_pkg / "core" / "mc_runner.py"
        self._mc_stdout_buf: str = ""
        
        # UI State Containers (Mutable)
        self.sim_state = SimulationState()
        self.gravity_cfg = UIGravityConfig()
        self.albedo_cfg = UIAlbedoConfig()
        self.solver_cfg = UISolverConfig()
        normalize_solver_config_object(self.solver_cfg)
        self.spacecraft_cfg = UISpacecraftConfig()
        
        # Mission Timeline
        self.mission_epoch = QtCore.QDateTime.fromString("2025-10-01 18:00:00", "yyyy-MM-dd HH:mm:ss")
        
        # Data & Files Configuration
        self.ldem_root_path = ""  # LDEM root directory
        self.albedo_root_path = ""  # Albedo root directory
        self.kernel_dir_path = ""  # SPICE kernels directory
        self.ldem_ppd = 4  # Pixels per degree resolution
        
        # Runtime Flags
        self.recent_presets: List[str] = []
        self.last_cmd_preview: str = ""
        self.is_log_collapsed: bool = False
        self._stdout_buf = ""
        # Reset impact monitoring for this run
        self._collision_triggered = False
        self._collision_reason = ""
        # Progress tracking
        self._run_wall_t0: Optional[float] = None
        self._last_telem_t_s: Optional[float] = None
        self._progress_is_determinate: bool = False
        
        # ---------------------------------------------------------------------
        # 4. UI Construction
        # ---------------------------------------------------------------------
        self._build_ui()
        self._apply_theme()
        
        # ---------------------------------------------------------------------
        # 5. Initialization & Bootstrapping
        # ---------------------------------------------------------------------
        self._try_prefill_topography_from_config()
        self._try_load_last_session()
        self._bootstrap()
        
        # ---------------------------------------------------------------------
        # 6. Watchdog Timer
        # ---------------------------------------------------------------------
        self.tick_timer = QtCore.QTimer(self)
        self.tick_timer.setInterval(250)  # 250ms refresh rate
        self.tick_timer.timeout.connect(self._ui_tick)
        self.tick_timer.start()
    
    # =========================================================================
    # 18. UI CONSTRUCTION & STYLING
    # =========================================================================
    
    def _build_ui(self):
        """Constructs the visual hierarchy of the main window."""
        
        # Central Container
        central = QtWidgets.QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        
        # ---------------------------------------------------------------------
        # A. Header Bar (Title, Status, Actions)
        # ---------------------------------------------------------------------
        header_frame = QtWidgets.QFrame()
        header_frame.setObjectName("header")
        h_layout = QtWidgets.QHBoxLayout(header_frame)
        h_layout.setContentsMargins(16, 10, 16, 10)
        h_layout.setSpacing(12)
        
        # App Title
        title_lbl = QtWidgets.QLabel(APP_NAME)
        title_lbl.setObjectName("title")
        h_layout.addWidget(title_lbl)
        
        # Page Indicator (StatusBadge)
        self.badge_page = StatusBadge("Orbit", "info")
        h_layout.addWidget(self.badge_page)
        
        h_layout.addStretch(1)
        
        # Progress Bar in Header
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedWidth(165)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setTextVisible(False)
        # Start as idle (determinate). Switch to indeterminate while the backend warms up.
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setFormat("")
        self.progress_bar.setValue(0)
        self._progress_is_determinate = False
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {THEME['border']};
                border-radius: 3px;
                background: {THEME['bg_entry']};
                text-align: center;
                color: {THEME['fg_main']};
                font-size: 9pt;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {THEME['accent']},
                    stop: 1 {THEME['secondary']}
                );
                border-radius: 2px;
            }}
        """)
        h_layout.addWidget(self.progress_bar)
        # Extra progress text (t/T + ETA)
        self.lbl_progress = QtWidgets.QLabel("")
        self.lbl_progress.setObjectName("progressText")
        self.lbl_progress.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
        self.lbl_progress.setMinimumWidth(155)
        h_layout.addWidget(self.lbl_progress)
        h_layout.addSpacing(8)
        
        # Execution State (Dot + Label)
        self.dot_run = QtWidgets.QFrame()
        self.dot_run.setObjectName("runDot")
        self.dot_run.setFixedSize(12, 12)
        self.dot_run.setProperty("kind", "idle")
        
        self.lbl_run_state = QtWidgets.QLabel("")
        self.lbl_run_state.setObjectName("runState")
        
        state_container = QtWidgets.QHBoxLayout()
        state_container.setSpacing(6)
        state_container.addWidget(self.dot_run)
        state_container.addWidget(self.lbl_run_state)
        
        self.state_frame = QtWidgets.QFrame()
        self.state_frame.setObjectName("stateFrame")
        self.state_frame.setLayout(state_container)
        h_layout.addWidget(self.state_frame)
        
        h_layout.addSpacing(16)
        
        # Action Buttons
        self.btn_stop = QtWidgets.QPushButton("  Stop")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_stop.setIcon(get_icon('fa6s.stop', THEME['fg_main']))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_process)
        
        self.btn_run = QtWidgets.QPushButton("  Run Mission Analysis")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_run.setIcon(get_icon('fa6s.play', THEME['fg_main']))
        self.btn_run.clicked.connect(self._start_preflight_validation)
        
        h_layout.addWidget(self.btn_stop)
        h_layout.addWidget(self.btn_run)

        # The header stays quieter when the app is idle. Progress and transient
        # execution state indicators are only shown while something actionable
        # is happening.
        self.progress_bar.hide()
        self.lbl_progress.hide()
        self.state_frame.hide()

        root.addWidget(header_frame)

        # ---------------------------------------------------------------------
        # A2. Mission Status Summary Bar
        # ---------------------------------------------------------------------
        status_bar_frame = QtWidgets.QFrame()
        status_bar_frame.setObjectName("missionStatusBar")
        status_bar_frame.setStyleSheet(f"""
            QFrame#missionStatusBar {{
                background: {THEME['bg_card_alt']};
                border: 1px solid {THEME['border_soft']};
                border-radius: 8px;
            }}
        """)
        sb_layout = QtWidgets.QHBoxLayout(status_bar_frame)
        sb_layout.setContentsMargins(12, 6, 12, 6)
        sb_layout.setSpacing(16)

        _label_style = f"color: {THEME['fg_muted']}; font-size: 9pt;"
        _value_style = f"color: {THEME['fg_soft']}; font-size: 9pt; font-weight: 600;"

        sb_layout.addWidget(_make_lbl("Gravity:", _label_style))
        self.lbl_gravity_status = QtWidgets.QLabel("SH [100]")
        self.lbl_gravity_status.setStyleSheet(_value_style)
        sb_layout.addWidget(self.lbl_gravity_status)

        sb_layout.addWidget(_make_lbl("|", f"color: {THEME['border']}; font-size: 9pt;"))

        sb_layout.addWidget(_make_lbl("Output:", _label_style))
        self.lbl_output_status = QtWidgets.QLabel("Not set")
        self.lbl_output_status.setStyleSheet(_value_style)
        self.lbl_output_status.setMaximumWidth(200)
        sb_layout.addWidget(self.lbl_output_status)

        sb_layout.addWidget(_make_lbl("|", f"color: {THEME['border']}; font-size: 9pt;"))

        sb_layout.addWidget(_make_lbl("Preflight:", _label_style))
        self.lbl_preflight_status = StatusBadge("IDLE", "info")
        self.lbl_preflight_status.setFixedWidth(70)
        sb_layout.addWidget(self.lbl_preflight_status)

        sb_layout.addWidget(_make_lbl("|", f"color: {THEME['border']}; font-size: 9pt;"))

        sb_layout.addWidget(_make_lbl("Run:", _label_style))
        self.lbl_run_status = StatusBadge("IDLE", "info")
        self.lbl_run_status.setFixedWidth(70)
        sb_layout.addWidget(self.lbl_run_status)

        sb_layout.addStretch(1)
        root.addWidget(status_bar_frame)

        # ---------------------------------------------------------------------
        # B. Main Content Area (Splitter: Nav+Pages | Log)
        # ---------------------------------------------------------------------
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.main_splitter.setObjectName("mainSplit")
        root.addWidget(self.main_splitter, 1)
        
        # Top Section: Navigation Sidebar + Stacked Pages
        content_container = QtWidgets.QWidget()
        content_container.setObjectName("contentRoot")
        content_container.setMinimumHeight(0)
        content_container.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Ignored,
        )
        content_layout = QtWidgets.QHBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)
        
        # 1. Navigation Drawer
        self.nav_list = QtWidgets.QListWidget()
        self.nav_list.setObjectName("navDrawer")
        self.nav_list.setFixedWidth(246)
        self.nav_list.setMinimumHeight(0)
        self.nav_list.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Ignored,
        )
        self.nav_list.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.nav_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.nav_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.nav_list.setSpacing(4)
        
        # Populate Nav Items with icons
        self._page_map = {}
        
        for i, (key, label, icon_name) in enumerate(NAV_PAGES):
            item = QtWidgets.QListWidgetItem(label)
            item.setSizeHint(QtCore.QSize(226, 40))
            item.setData(QtCore.Qt.UserRole, key)
            item.setIcon(get_icon(icon_name, THEME['fg_muted']))
            self.nav_list.addItem(item)
            self._page_map[key] = i
        
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        content_layout.addWidget(self.nav_list)
        
        # 2. Page Stack
        self.stack_pages = QtWidgets.QStackedWidget()
        self.stack_pages.setObjectName("pages")
        self.stack_pages.setMinimumHeight(0)
        self.stack_pages.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Ignored,
        )
        
        # Build individual page widgets
        self.page_orbit = self._build_page_orbit()
        self.page_forces = self._build_page_forces()
        self.page_propagation = self._build_page_propagation()
        self.page_output = self._build_page_output()
        self.page_telemetry = self._build_page_telemetry()
        self.page_data = self._build_page_data()
        self.page_mc = self._build_page_mc()
        self.page_surrogate = self._build_page_surrogate()

        # Wrap pages in scroll areas (except telemetry, MC, and Surrogate which have their own)
        self.stack_pages.addWidget(self._wrap_scroll(self.page_orbit))
        self.stack_pages.addWidget(self._wrap_scroll(self.page_forces))
        self.stack_pages.addWidget(self._wrap_scroll(self.page_propagation))
        self.stack_pages.addWidget(self._wrap_scroll(self.page_output))
        self.stack_pages.addWidget(self.page_telemetry)  # Telemetry doesn't need scroll
        self.stack_pages.addWidget(self._wrap_scroll(self.page_data))
        self.stack_pages.addWidget(self.page_mc)         # MC has internal scrolls
        self.stack_pages.addWidget(self.page_surrogate)  # Surrogate has its own scroll area
        
        content_layout.addWidget(self.stack_pages, 1)
        self.main_splitter.addWidget(content_container)
        
        # ---------------------------------------------------------------------
        # C. Log Panel
        # ---------------------------------------------------------------------
        self.log_panel = self._build_log_panel()
        self.log_panel.setMinimumHeight(140)
        self.log_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.main_splitter.addWidget(self.log_panel)
        
        # Initial Splitter Sizes
        self.main_splitter.setHandleWidth(8)   # wide enough to grab reliably
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)  # prevent log from disappearing
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        
        # Build Menu & Status
        self._build_menubar()
        self._build_statusbar()
        
        # Set Initial State
        self._switch_page("Orbit")
        self._update_run_visuals("idle")

        # Impact / collision monitoring
        self._collision_triggered = False
        self._collision_reason = ""

    
    def _build_menubar(self):
        """Constructs the native window menu."""
        mb = self.menuBar()
        mb.setObjectName("menuBar")
        
        # FILE MENU
        m_file = mb.addMenu("&File")
        
        a_load = m_file.addAction("Load Mission Profile...")
        a_load.setShortcut("Ctrl+O")
        a_load.triggered.connect(self._action_load_session)
        
        a_save = m_file.addAction("Save Mission Profile")
        a_save.setShortcut("Ctrl+S")
        a_save.triggered.connect(self._action_save_session)
        
        m_file.addSeparator()
        
        a_open_dir = m_file.addAction("Open Results Folder")
        a_open_dir.setShortcut("Ctrl+Shift+O")
        a_open_dir.triggered.connect(self._action_open_out_dir)
        
        m_file.addSeparator()
        a_exit = m_file.addAction("Exit")
        a_exit.setShortcut("Alt+F4")
        a_exit.triggered.connect(self.close)
        
        # ANALYSIS MENU
        m_run = mb.addMenu("&Analysis")
        
        a_run = m_run.addAction("Start Propagation")
        a_run.setShortcut("F5")
        a_run.triggered.connect(self._start_preflight_validation)
        
        a_stop = m_run.addAction("Abort Propagation")
        a_stop.setShortcut("Shift+F5")
        a_stop.triggered.connect(self._stop_process)
        
        # SETTINGS MENU
        m_settings = mb.addMenu("&Settings")
        
        a_solver = m_settings.addAction("Solver Configuration...")
        a_solver.triggered.connect(self._on_solver_settings)
        
        a_spacecraft = m_settings.addAction("Spacecraft Properties...")
        a_spacecraft.triggered.connect(self._on_spacecraft_settings)
        
        m_settings.addSeparator()
        
        a_gravity = m_settings.addAction("Gravity Model...")
        a_gravity.triggered.connect(self._on_gravity_settings)
        
        a_albedo = m_settings.addAction("Albedo Model...")
        a_albedo.triggered.connect(self._on_albedo_settings)
        
        # VIEW MENU
        m_view = mb.addMenu("&View")
        
        a_log = m_view.addAction("Toggle Log Panel")
        a_log.setShortcut("Ctrl+L")
        a_log.triggered.connect(self._toggle_log_collapsed)
        
        a_clear = m_view.addAction("Clear Log")
        a_clear.setShortcut("Ctrl+K")
        a_clear.triggered.connect(self._clear_log)
    
    def _build_statusbar(self):
        """Create a hidden status bar so idle text does not clutter the footer."""
        sb = QtWidgets.QStatusBar()
        sb.setObjectName("statusBar")
        self.setStatusBar(sb)
        sb.clearMessage()
        sb.setSizeGripEnabled(False)
        sb.hide()
    
    def _apply_theme(self):
        """
        Applies the global QSS stylesheet using the predefined THEME dictionary.
        """
        # 1. Set Base Palette
        app = QtWidgets.QApplication.instance()
        if app:
            pal = QtGui.QPalette()
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor(THEME['bg_space']))
            pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(THEME['fg_main']))
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor(THEME['bg_entry']))
            pal.setColor(QtGui.QPalette.Text, QtGui.QColor(THEME['fg_main']))
            pal.setColor(QtGui.QPalette.Button, QtGui.QColor(THEME['bg_card']))
            pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(THEME['fg_main']))
            pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(THEME['accent']))
            app.setPalette(pal)
        
        # 2. Generate CSS from THEME dict — Lunar Aurora palette
        qss = f"""
        /* GLOBAL FOUNDATION */
        QMainWindow,
        QWidget#centralRoot {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 {THEME['bg_space']},
                stop: 0.48 {THEME['bg_shell']},
                stop: 1 #0F1A2E
            );
            color: {THEME['fg_main']};
        }}
        QWidget {{
            background: transparent;
            color: {THEME['fg_main']};
            font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif;
            font-size: 10pt;
        }}
        QLabel {{
            background: transparent;
        }}
        QWidget#contentRoot,
        QStackedWidget#pages,
        QScrollArea,
        QScrollArea > QWidget > QWidget {{
            background: transparent;
            border: none;
        }}
        QToolTip {{
            background: {THEME['bg_card_alt']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            padding: 6px 8px;
        }}

        /* MENUS */
        QMenuBar {{
            background: {THEME['bg_shell']};
            color: {THEME['fg_main']};
            border-bottom: 1px solid {THEME['border_soft']};
            padding: 4px 8px;
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 6px 12px;
            border-radius: 7px;
        }}
        QMenuBar::item:selected {{
            background: {THEME['accent_dim']};
            color: {THEME['fg_soft']};
        }}
        QMenu {{
            background: {THEME['bg_card']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            padding: 6px;
        }}
        QMenu::item {{
            padding: 7px 22px 7px 12px;
            border-radius: 7px;
        }}
        QMenu::item:selected {{
            background: {THEME['accent_dim']};
            color: {THEME['fg_soft']};
        }}
        QMenu::separator {{
            height: 1px;
            background: {THEME['border_soft']};
            margin: 6px 8px;
        }}

        /* CONTAINERS */
        QFrame#header, QFrame#logHeader {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 {THEME['bg_card']},
                stop: 1 {THEME['bg_card_alt']}
            );
            border: 1px solid {THEME['border']};
            border-radius: 14px;
        }}
        QFrame#stateFrame {{
            background: {THEME['accent_dim']};
            border: 1px solid rgba(53,208,255,0.20);
            border-radius: 8px;
            padding: 2px 8px;
        }}

        /* TEXT */
        QLabel#title {{
            font-size: 15pt;
            font-weight: 700;
            color: {THEME['fg_soft']};
            letter-spacing: 0.3px;
        }}
        QLabel#runState {{
            color: {THEME['fg_soft']};
            font-weight: 600;
        }}
        QLabel#progressText {{
            color: {THEME['fg_muted']};
            font-size: 9pt;
        }}

        /* NAVIGATION */
        QListWidget#navDrawer {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 {THEME['bg_card']},
                stop: 1 {THEME['bg_shell']}
            );
            border: 1px solid {THEME['border']};
            border-radius: 14px;
            padding: 10px;
            outline: none;
        }}
        QListWidget#navDrawer::item {{
            background: transparent;
            padding: 11px 14px;
            margin-bottom: 6px;
            border-radius: 10px;
            color: {THEME['fg_muted']};
        }}
        QListWidget#navDrawer::item:hover {{
            background: rgba(53,208,255,0.06);
            color: {THEME['fg_main']};
        }}
        QListWidget#navDrawer::item:selected {{
            background: {THEME['secondary_dim']};
            color: {THEME['fg_soft']};
            font-weight: 700;
            border-left: 3px solid {THEME['accent']};
        }}

        /* INPUTS */
        QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox, QDateTimeEdit, QSpinBox {{
            background: {THEME['bg_entry']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            border-radius: 9px;
            padding: 7px 10px;
            selection-background-color: {THEME['secondary']};
        }}
        QLineEdit:hover, QComboBox:hover, QPlainTextEdit:hover,
        QDoubleSpinBox:hover, QDateTimeEdit:hover, QSpinBox:hover {{
            border: 1px solid rgba(53,208,255,0.45);
        }}
        QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus,
        QDoubleSpinBox:focus, QDateTimeEdit:focus, QSpinBox:focus {{
            border: 1px solid {THEME['accent']};
            background: {THEME['bg_card_alt']};
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
        QDoubleSpinBox:disabled, QDateTimeEdit:disabled {{
            color: {THEME['text_disabled']};
            background: {THEME['bg_card']};
            border-color: {THEME['border_soft']};
        }}
        QComboBox::drop-down,
        QDateTimeEdit::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background: {THEME['bg_card_alt']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            selection-background-color: {THEME['secondary_dim']};
        }}

        /* CARDS */
        QGroupBox {{
            background: {THEME['bg_card']};
            border: 1px solid {THEME['border']};
            border-radius: 14px;
            margin-top: 24px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            margin-left: 12px;
            color: {THEME['fg_soft']};
            font-size: 10.4pt;
        }}

        /* BUTTONS (default / secondary) */
        QPushButton {{
            background: {THEME['bg_card_alt']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            border-radius: 9px;
            padding: 7px 16px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            border-color: rgba(53,208,255,0.45);
            background: {THEME['bg_entry']};
            color: {THEME['fg_main']};
        }}
        QPushButton:pressed {{
            background: {THEME['border_soft']};
        }}
        QPushButton:disabled {{
            background: {THEME['bg_card']};
            border-color: {THEME['border_soft']};
            color: {THEME['text_disabled']};
        }}
        QPushButton#quickChip {{
            background: {THEME['accent_dim']};
            border: 1px solid rgba(53,208,255,0.20);
            color: {THEME['fg_soft']};
            border-radius: 9px;
            padding: 5px 12px;
        }}
        QPushButton#quickChip:hover {{
            background: rgba(53,208,255,0.18);
            border-color: {THEME['accent']};
        }}

        /* PRIMARY BUTTON (RUN) */
        QPushButton#primaryBtn {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 {THEME['accent']},
                stop: 1 {THEME['secondary']}
            );
            border: 1px solid {THEME['accent']};
            color: #05090F;
            font-weight: 700;
        }}
        QPushButton#primaryBtn:hover {{
            background: {THEME['accent_hov']};
            border-color: {THEME['accent_hov']};
            color: #05090F;
        }}
        QPushButton#primaryBtn:disabled {{
            background: {THEME['bg_entry']};
            border-color: {THEME['border']};
            color: {THEME['text_disabled']};
        }}

        /* DANGER BUTTON (STOP) */
        QPushButton#dangerBtn {{
            background: rgba(255,107,122,0.10);
            border: 1px solid rgba(255,107,122,0.26);
            color: {THEME['fg_main']};
        }}
        QPushButton#dangerBtn:hover {{
            background: rgba(255,107,122,0.20);
            border-color: {THEME['error']};
        }}
        QPushButton#dangerBtn:disabled {{
            background: {THEME['bg_entry']};
            border-color: {THEME['border']};
            color: {THEME['text_disabled']};
        }}

        /* TOOL + CHECK CONTROLS */
        QToolButton {{
            background: transparent;
            border: none;
            padding: 4px;
        }}
        QToolButton:hover {{
            background: {THEME['accent_dim']};
            border-radius: 7px;
        }}
        QCheckBox {{
            spacing: 8px;
            color: {THEME['fg_muted']};
        }}
        QCheckBox::indicator {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
            border: 1px solid {THEME['border']};
            background: {THEME['bg_entry']};
        }}
        QCheckBox::indicator:checked {{
            background: {THEME['accent']};
            border-color: {THEME['accent']};
        }}
        QCheckBox::indicator:hover {{
            border-color: rgba(53,208,255,0.55);
        }}

        /* STATUS BADGE */
        QLabel#statusBadge {{
            border-radius: 10px;
            border: 1px solid {THEME['border']};
            font-weight: 700;
            padding: 0 8px;
        }}
        QLabel#statusBadge[kind="info"] {{ background: {THEME['accent_dim']}; color: {THEME['accent_hov']}; border-color: rgba(53,208,255,0.35); }}
        QLabel#statusBadge[kind="success"] {{ background: rgba(45,212,191,0.12); color: {THEME['success']}; border-color: rgba(45,212,191,0.32); }}
        QLabel#statusBadge[kind="error"] {{ background: rgba(255,107,122,0.12); color: {THEME['error']}; border-color: rgba(255,107,122,0.30); }}
        QLabel#statusBadge[kind="warning"] {{ background: rgba(246,193,119,0.12); color: {THEME['warning']}; border-color: rgba(246,193,119,0.32); }}

        /* RUN DOT */
        QFrame#runDot {{ border-radius: 6px; }}
        QFrame#runDot[kind="idle"] {{ background: {THEME['fg_muted']}; }}
        QFrame#runDot[kind="running"] {{ background: {THEME['success']}; }}
        QFrame#runDot[kind="error"] {{ background: {THEME['error']}; }}
        QFrame#runDot[kind="warning"] {{ background: {THEME['warning']}; }}

        /* TABS */
        QTabWidget::pane {{
            border: 1px solid {THEME['border']};
            background: {THEME['bg_card']};
            border-radius: 12px;
            top: -1px;
        }}
        QTabBar::tab {{
            background: rgba(255,255,255,0.03);
            color: {THEME['fg_muted']};
            border: 1px solid {THEME['border_soft']};
            padding: 8px 16px;
            margin-right: 6px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }}
        QTabBar::tab:selected {{
            background: {THEME['accent_dim']};
            color: {THEME['fg_soft']};
            border-color: rgba(53,208,255,0.30);
            border-bottom: 2px solid {THEME['accent']};
        }}
        QTabBar::tab:hover:!selected {{
            color: {THEME['fg_main']};
            background: rgba(255,255,255,0.04);
        }}

        /* LOGGING */
        QTextEdit#log {{
            background: {THEME['bg_log']};
            color: {LOG_COLORS['default']};
            font-family: "Consolas", "Courier New", monospace;
            font-size: 10pt;
            border: 1px solid {THEME['border']};
            border-radius: 10px;
            padding: 8px;
        }}

        /* SCROLLBARS */
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: transparent;
            width: 8px;
            height: 8px;
            margin: 0;
        }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
            background: {THEME['border']};
            border-radius: 4px;
            min-height: 20px;
            min-width: 20px;
        }}
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
            background: rgba(53,208,255,0.40);
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ height: 0px; width: 0px; }}

        /* SPLITTER */
        /* SPLITTER */
        QSplitter::handle {{
            background: {THEME['border_soft']};
        }}
        QSplitter::handle:vertical {{
            height: 8px;
            margin: 0 4px;
            border-radius: 3px;
            image: none;
        }}
        QSplitter::handle:vertical:hover {{
            background: rgba(53,208,255,0.35);
        }}
        QSplitter#mainSplit::handle:vertical {{
            background: {THEME['border']};
            height: 8px;
            border-radius: 3px;
        }}
        QSplitter#mainSplit::handle:vertical:hover {{
            background: rgba(53,208,255,0.40);
        }}

        /* TREE / LIST WIDGETS */
        QTreeWidget, QListWidget {{
            background: {THEME['bg_entry']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            border-radius: 10px;
            outline: none;
            alternate-background-color: {THEME['bg_card_alt']};
        }}
        QTreeWidget::item, QListWidget::item {{
            padding: 4px 6px;
            border-radius: 5px;
        }}
        QTreeWidget::item:selected, QListWidget::item:selected {{
            background: {THEME['secondary_dim']};
            color: {THEME['fg_main']};
        }}
        QTreeWidget::item:hover, QListWidget::item:hover {{
            background: {THEME['accent_dim']};
        }}
        QHeaderView::section {{
            background: {THEME['bg_card']};
            color: {THEME['fg_soft']};
            border: none;
            border-bottom: 1px solid {THEME['border']};
            padding: 5px 8px;
            font-weight: 600;
        }}
        """
        self.setStyleSheet(qss)
    
    # =========================================================================
    # 19. PAGE BUILDERS: ORBIT PAGE (PAGE 1)
    # =========================================================================

    def _build_page_orbit(self) -> QtWidgets.QWidget:
        """
        Page 1: Orbit Configuration.
        Delegated to ui_parts.orbit_config_page.OrbitPage
        """
        from lunaris.ui.widgets.orbit_config_page import OrbitPage  # local import to avoid circulars
        self.page_orbit = OrbitPage()
        return self.page_orbit
    

    # =========================================================================
    # 20. PAGE BUILDERS: FORCES PAGE (PAGE 2)
    # =========================================================================

    def _build_page_forces(self) -> QtWidgets.QWidget:
        """
        Page 2: Force Model Settings.
        Delegated to ui_parts.force_models_page.ForceModelsPage
        """
        from lunaris.ui.widgets.force_models_page import ForceModelsPage  # local import to avoid circulars

        # IMPORTANT: pass shared config objects so dialogs mutate the same instances
        self.page_forces = ForceModelsPage(
            gravity_cfg=self.gravity_cfg,
            albedo_cfg=self.albedo_cfg,
        )
        return self.page_forces

    
    # =========================================================================
    # 21. PAGE BUILDERS: PROPAGATION CONFIGURATION (PAGE 3)
    # =========================================================================
    
    def _build_page_propagation(self) -> QtWidgets.QWidget:
        from lunaris.ui.widgets.mission_propagation_page import MissionPropagationPage
        self.page_propagation = MissionPropagationPage(
            parent=self,
            mission_epoch=self.mission_epoch,
            solver_cfg=self.solver_cfg,
            spacecraft_cfg=self.spacecraft_cfg,
        )
        return self.page_propagation

    
    # =========================================================================
    # 22. PAGE BUILDERS: OUTPUT (PAGE 4)
    # =========================================================================
    def _build_page_output(self) -> QtWidgets.QWidget:
        """
        Build the dedicated results/export page.

        The page owns its widgets and exposes them through page-level helpers.
        A few legacy aliases are still mirrored onto `MainWindow` so the rest of
        the existing orchestration code can be migrated incrementally without
        breaking behavior.
        """

        page = ResultsExportPage(
            project_root=PROJECT_ROOT,
            create_card=self._create_card,
            initial_state=OutputPageState(
                output_dir=str(PROJECT_ROOT / "outputs" / "missions"),
                generate_3d_plots=False,
                downsample_3d=1,
            ),
            parent=self,
        )
        page.browse_output_dir_requested.connect(self._browse_out_dir)
        page.open_output_dir_requested.connect(self._action_open_out_dir)
        page.refresh_preview_requested.connect(self._update_command_preview)
        page.copy_preview_requested.connect(self._copy_command_preview)

        self.page_output = page
        return page
    
    def _build_page_telemetry(self) -> QtWidgets.QWidget:
        from lunaris.ui.widgets.live_telemetry_page import TelemetryPage
        self.page_telemetry = TelemetryPage()
        return self.page_telemetry

    
    # =========================================================================
    # 24. PAGE BUILDERS: DATA & FILES (PAGE 6) 
    # =========================================================================
    def _build_page_data(self) -> QtWidgets.QWidget:
        from lunaris.ui.widgets.data_files_page import DataPage, DataFilesState

        # Initial state comes from the values held on MainWindow:
        init = DataFilesState(
            ldem_root=getattr(self, "ldem_root_path", "") or "",
            albedo_root=getattr(self, "albedo_root_path", "") or "",
            kernel_dir=getattr(self, "kernel_dir_path", "") or "",
            ldem_ppd=int(getattr(self, "ldem_ppd", 4) or 4),
            use_ldem_for_albedo=True,
        )

        self.page_data = DataPage(
            project_root=PROJECT_ROOT,
            normalize_path=normalize_path,
            log_message=lambda msg: self._log_message(msg, severity="system"),
            create_card=self._create_card,
            initial_state=init,
        )
        return self.page_data

    
    # =========================================================================
    # 24b. PAGE BUILDERS: MONTE CARLO (PAGE 7)
    # =========================================================================

    def _build_page_mc(self) -> QtWidgets.QWidget:
        """Page 7: Monte Carlo Analysis — configuration + live metrics."""
        from lunaris.ui.widgets.monte_carlo_page import MonteCarloPage
        self.page_mc = MonteCarloPage(parent=self)
        self.page_mc.run_requested.connect(self._on_mc_run_requested)
        return self.page_mc

    # =========================================================================
    # 24c. PAGE BUILDERS: SURROGATE STUDIO (PAGE 8)
    # =========================================================================

    def _build_page_surrogate(self) -> QtWidgets.QWidget:
        """Page 8: ST-LRPS Surrogate Studio — runs browser, eval artifacts, training preview."""
        self.page_surrogate = SurrogateStudioPage(parent=self)
        try:
            self.page_surrogate.model_selected.connect(self._on_surrogate_model_selected)
        except Exception:
            pass
        return self.page_surrogate

    def _on_surrogate_model_selected(self, run_dir: str) -> None:
        """
        Slot: invoked when the Surrogate Studio user clicks "Use This Model".

        Sets the gravity backend to ``st_lrps`` and wires the run directory into
        the shared ``UIGravityConfig`` so subsequent command builds and
        preflight validation pick the correct surrogate model.
        """
        if not run_dir:
            return
        try:
            if hasattr(self.gravity_cfg, "backend"):
                self.gravity_cfg.backend = "st_lrps"
            if hasattr(self.gravity_cfg, "st_lrps_model_dir"):
                self.gravity_cfg.st_lrps_model_dir = run_dir
            # Best-effort: keep the force-models page summary visually aligned.
            try:
                if hasattr(self, "page_forces") and hasattr(self.page_forces, "_update_gravity_summary_ui"):
                    self.page_forces._update_gravity_summary_ui()
            except Exception:
                pass
            self._log_message(f"[UI] ST-LRPS model selected: {run_dir}", severity="success")
            try:
                if self.statusBar() is not None:
                    self.statusBar().showMessage(
                        f"ST-LRPS model active: {Path(run_dir).name}", 4000
                    )
            except Exception:
                pass
            # Refresh command preview so the new surrogate path is reflected.
            try:
                self._update_command_preview_silent()
            except Exception:
                pass
        except Exception as exc:
            self._log_message(
                f"[Warning] Failed to apply surrogate model: {exc}", severity="warning"
            )

    def _on_mc_run_requested(self) -> None:
        """Slot: user clicked 'Run Monte Carlo' on the MC page."""
        if self.mc_process is not None:
            try:
                state = self.mc_process.state()
                if state != QtCore.QProcess.NotRunning:
                    self._log_message("[MC] A run is already in progress.", severity="warning")
                    return
            except RuntimeError:
                self.mc_process = None

        mc_data = self.page_mc.get_data()

        try:
            cmd = build_mc_command(
                python_executable=sys.executable,
                mc_runner_path=self.mc_script_path,
                orbit=self.page_orbit.get_data(),
                forces=self.page_forces.get_data(),
                propagation=self.page_propagation.to_dict(),
                mc_data=mc_data,
                data_files=self.page_data.get_state(),
                gravity_cfg=self.gravity_cfg,
                solver_cfg=self.solver_cfg,
                spacecraft_cfg=self.spacecraft_cfg,
                log_warning=lambda m: self._log_message(m, severity="warning"),
            )
        except Exception as exc:
            self._log_message(f"[MC][Error] Failed to build command: {exc}", severity="error")
            self.page_mc.on_run_finished(1, "", None, None)
            return

        self._log_separator()
        self._log_message("[MC] Starting Monte Carlo run…", severity="system")

        self.mc_process = QtCore.QProcess(self)
        self.mc_process.readyReadStandardOutput.connect(self._on_mc_stdout)
        self.mc_process.readyReadStandardError.connect(self._on_mc_stderr)
        self.mc_process.finished.connect(self._on_mc_finished)

        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.mc_process.setProcessEnvironment(env)

        self._mc_stdout_buf = ""
        self._mc_metrics: dict = {}
        self._mc_output_path: str = mc_data.get("output_path", "")

        self.mc_process.start(cmd[0], cmd[1:])
        if not self.mc_process.waitForStarted(2000):
            self._log_message("[MC][Error] Failed to start MC process.", severity="error")
            self.page_mc.on_run_finished(1, "", None, None)
            self.mc_process = None

    def _on_mc_stdout(self) -> None:
        """
        Stream stdout from the MC subprocess to the page progress log.

        Monte Carlo runs now emit two kinds of structured control lines:
        ``[MC_PROGRESS]`` for live progress payloads and ``[MC_METRICS]`` for
        the final summary blob.  These are consumed directly by the page rather
        than dumped into the human-readable log stream.
        """
        if self.mc_process is None:
            return
        try:
            raw = bytes(self.mc_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            return

        self._mc_stdout_buf += raw
        while "\n" in self._mc_stdout_buf:
            line, self._mc_stdout_buf = self._mc_stdout_buf.split("\n", 1)
            line = line.rstrip()
            if not line:
                continue

            if line.startswith("[MC_PROGRESS]"):
                try:
                    payload = json.loads(line[len("[MC_PROGRESS]"):].strip())
                except Exception:
                    self._log_message("[MC] Ignored malformed progress payload.", severity="warning")
                else:
                    if hasattr(self, "page_mc"):
                        self.page_mc.update_progress_payload(payload)
                continue

            if line.startswith("[MC_METRICS]"):
                try:
                    payload = line[len("[MC_METRICS]"):].strip()
                    self._mc_metrics = json.loads(payload)
                except Exception:
                    self._log_message("[MC] Ignored malformed metrics payload.", severity="warning")
                continue

            self._log_message(line, severity="info")

            # Forward to the page's mini-log + progress bar
            if hasattr(self, "page_mc"):
                self.page_mc.update_progress(line)

    def _on_mc_stderr(self) -> None:
        """Route MC stderr to the main log as warnings."""
        if self.mc_process is None:
            return
        try:
            raw = bytes(self.mc_process.readAllStandardError()).decode("utf-8", errors="replace")
        except Exception:
            return
        for line in raw.splitlines():
            if line.strip():
                self._log_message(f"[MC] {line.strip()}", severity="warning")

    def _on_mc_finished(self, exit_code: int, _exit_status) -> None:
        """Handle MC subprocess completion and update the page."""
        metrics = getattr(self, "_mc_metrics", {}) or {}
        output_path = metrics.get("output_path", getattr(self, "_mc_output_path", ""))

        if exit_code == 0:
            wt = metrics.get("wall_time_s")
            wt_str = f"{wt:.1f}s" if isinstance(wt, (int, float)) else "?"
            self._log_message(
                f"[MC] Run complete — {metrics.get('n_impacts', '?')}/{metrics.get('n_samples', '?')} "
                f"impacts  wall={wt_str}",
                severity="success",
            )
        else:
            self._log_message(f"[MC] Run failed (exit code {exit_code}).", severity="error")

        if hasattr(self, "page_mc"):
            self.page_mc.on_run_finished(
                exit_code=exit_code,
                output_path=str(output_path),
                report_path=None,
                metrics=metrics if exit_code == 0 else None,
            )

        try:
            if self.mc_process is not None:
                self.mc_process.deleteLater()
        except Exception:
            pass
        self.mc_process = None

    # =========================================================================
    # 25. LOG PANEL BUILDER
    # =========================================================================
    
    def _build_log_panel(self) -> QtWidgets.QWidget:
        """Constructs the collapsible logging panel with rich text support and Copy button."""
        container = QtWidgets.QWidget()
        container.setMinimumHeight(0)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # A. Header Bar
        header = QtWidgets.QFrame()
        header.setObjectName("logHeader")
        header.setFixedHeight(48)
        self.log_header = header
        
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(12, 4, 12, 4)
        h_layout.setSpacing(12)
        
        # Collapse Button
        self.btn_log_toggle = QtWidgets.QToolButton()
        self.btn_log_toggle.setIcon(get_icon("fa6s.chevron-down", THEME['fg_main']))
        self.btn_log_toggle.setToolTip("Hide Log Panel")
        self.btn_log_toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_log_toggle.setStyleSheet("border: none; background: transparent;")
        self.btn_log_toggle.clicked.connect(self._toggle_log_collapsed)
        h_layout.addWidget(self.btn_log_toggle)
        
        label = QtWidgets.QLabel("Execution Log")
        label.setStyleSheet("font-weight: 600;")
        h_layout.addWidget(label)
        
        h_layout.addStretch()
        
        # Tools
        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        h_layout.addWidget(self.chk_autoscroll)
        
        # Copy button
        btn_copy = QtWidgets.QPushButton("Copy")
        btn_copy.setFixedSize(60, 24)
        btn_copy.setStyleSheet("padding: 2px;")
        btn_copy.setIcon(get_icon("fa6s.copy", THEME['fg_main']))
        btn_copy.clicked.connect(self._copy_log_to_clipboard)
        h_layout.addWidget(btn_copy)
        self.btn_log_copy = btn_copy
        
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.setFixedSize(60, 24)
        btn_clear.setStyleSheet("padding: 2px;")
        btn_clear.clicked.connect(self._clear_log)
        h_layout.addWidget(btn_clear)
        self.btn_log_clear = btn_clear
        
        layout.addWidget(header)
        
        # B. Text Area (QTextEdit for HTML support)
        self.txt_log = QtWidgets.QTextEdit()
        self.txt_log.setObjectName("log")
        self.txt_log.setMinimumHeight(0)
        self.txt_log.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.txt_log.setReadOnly(True)
        self.txt_log.setAcceptRichText(True)
        self.txt_log.document().setDefaultStyleSheet(
            "p { margin: 0 0 6px 0; }"
        )
        layout.addWidget(self.txt_log)
        
        return container

    def _apply_default_log_splitter_sizes(self, *, top_ratio: float = 0.72) -> None:
        """
        Rebalance the main vertical splitter using the live window geometry.

        Without an explicit size pass, Qt tends to honor child size hints from
        the page stack, which makes the lower terminal/log area feel stuck and
        difficult to drag upward. This helper gives the splitter a practical
        starting ratio after the window has a real size.
        """

        total = sum(max(0, size) for size in self.main_splitter.sizes())
        if total <= 0:
            total = max(self.main_splitter.height(), 480)

        min_top = 220
        min_bottom = 140
        if total < (min_top + min_bottom):
            min_top = max(120, int(total * 0.55))
            min_bottom = max(80, total - min_top)

        top_size = max(min_top, int(total * float(top_ratio)))
        bottom_size = max(min_bottom, total - top_size)
        if (top_size + bottom_size) > total:
            top_size = max(min_top, total - min_bottom)
            bottom_size = max(min_bottom, total - top_size)

        self.main_splitter.setSizes([top_size, bottom_size])
    
    def _copy_log_to_clipboard(self, _checked: bool = False):
        """Copy the entire log content to clipboard."""
        plain_text = self.txt_log.toPlainText()
        if plain_text.strip():
            QtWidgets.QApplication.clipboard().setText(plain_text)
            self._log_message("[UI] Log copied to clipboard", severity="system")
        else:
            self._log_message("[UI] Log is empty, nothing to copy", severity="warning")
    
    # =========================================================================
    # 26. RICH TEXT LOGGING IMPLEMENTATION
    # =========================================================================
    
    def _parse_log_severity(self, text: str) -> Tuple[str, str]:
        """
        Parse log message to determine severity and accent color.

        The returned color is used for a compact severity tag rather than for
        coloring the whole line. That keeps the console easier to scan during
        long runs.
        """
        text_lower = text.lower()
        
        # Error patterns
        if any(pattern in text_lower for pattern in ["[err]", "error:", "failed", "exception", "traceback", "critical"]):
            return "error", LOG_COLORS["error"]
        
        # Warning patterns
        if any(pattern in text_lower for pattern in ["[warning]", "[warn]", "warning:", "caution", "deprecated"]):
            return "warning", LOG_COLORS["warning"]
        
        # Success patterns
        if any(pattern in text_lower for pattern in ["success", "finished", "completed", "✓", "passed"]):
            return "success", LOG_COLORS["success"]
        
        # System/Info patterns
        if any(pattern in text_lower for pattern in ["[system]", "[ui]", "initializing", "loading", "validating"]):
            return "system", LOG_COLORS["system"]
        
        # Default/Info
        return "info", LOG_COLORS["info"]
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters to prevent injection."""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#39;"))
    
    def _format_timestamp(self) -> str:
        """Generate a subtle timestamp block for the log panel."""
        timestamp = QtCore.QDateTime.currentDateTime().toString("HH:mm:ss")
        return f'<span style="color: {LOG_COLORS["timestamp"]}; font-weight: 600;">[{timestamp}]</span>'

    def _format_log_level(self, severity: str, color: str) -> str:
        """
        Render a compact severity tag for rich-text log output.

        The message body stays neutral and readable while the tag carries the
        color cue for the severity level.
        """

        labels = {
            "error": "ERROR",
            "warning": "WARN",
            "success": "OK",
            "system": "SYSTEM",
            "info": "INFO",
            "debug": "DEBUG",
        }
        label = labels.get(severity, "INFO")
        return f'<span style="color: {color}; font-weight: 700;">[{label}]</span>'
    
    def _log_message(self, text: str, is_error: bool = False, severity: str = None):
        """
        Append rich text message to log panel.
        
        Args:
            text: The message text
            is_error: Legacy error flag (overrides auto-detection)
            severity: Optional explicit severity ("error", "warning", "success", "system", "info")
        """
        if not text.strip():
            return
        
        # Determine severity and color
        if severity:
            log_severity = severity
            color = LOG_COLORS.get(severity, LOG_COLORS["default"])
        elif is_error:
            log_severity = "error"
            color = LOG_COLORS["error"]
        else:
            log_severity, color = self._parse_log_severity(text)
        
        # Escape HTML in the message
        escaped_text = self._escape_html(text.strip())
        
        # Format the message with a muted timestamp, a colored severity tag,
        # and a neutral body color for readability.
        timestamp = self._format_timestamp()
        level_tag = self._format_log_level(log_severity, color)
        formatted_message = (
            f'{timestamp} {level_tag} '
            f'<span style="color: {LOG_COLORS["default"]};">{escaped_text}</span><br>'
        )
        
        # Append to log (thread-safe)
        QtCore.QMetaObject.invokeMethod(
            self.txt_log,
            "append",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, formatted_message)
        )
        
        # Auto-scroll if enabled
        if self.chk_autoscroll.isChecked():
            QtCore.QMetaObject.invokeMethod(
                self.txt_log.verticalScrollBar(),
                "setValue",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, self.txt_log.verticalScrollBar().maximum())
            )
    
    def _log_separator(self):
        """Add a visual separator to the log."""
        separator = f'<hr style="border: none; border-top: 1px solid {THEME["border"]}; margin: 10px 0;">'
        QtCore.QMetaObject.invokeMethod(
            self.txt_log,
            "append",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, separator)
        )
    
    def _clear_log(self, _checked: bool = False):
        """Clear the log panel."""
        self.txt_log.clear()
        self._log_message("[UI] Log cleared.", severity="system")
    
    # =========================================================================
    # 27. ASYNCHRONOUS PRE-FLIGHT VALIDATION
    # =========================================================================

    def _collect_preflight_data(self) -> Dict[str, Any]:
        """
        Collect the minimal UI snapshot required for pre-flight validation.

        The heavy lifting is delegated to `ui_parts.command_builder` so the
        window only coordinates which page/config objects provide the source
        state.
        """

        try:
            return build_preflight_snapshot(
                orbit=self.page_orbit.get_data(),
                forces=self.page_forces.get_data(),
                propagation=self.page_propagation.to_dict(),
                output=self.page_output.get_state(),
                data_files=self.page_data.get_state(),
                spacecraft_cfg=self.spacecraft_cfg,
                solver_cfg=self.solver_cfg,
                gravity_cfg=self.gravity_cfg,
                albedo_cfg=self.albedo_cfg,
            )
        except ValueError as e:
            self._log_message(f"[Error] Invalid input values: {e}", severity="error")
            return {}

    def _process_state(self) -> QtCore.QProcess.ProcessState:
        """
        Return the current QProcess state while tolerating stale Qt wrappers.

        Slot exceptions or late object deletion can leave `self.process`
        pointing at an object that no longer has a valid C++ backing instance.
        Treating those cases as `NotRunning` keeps the Run button recoverable.
        """

        if self.process is None:
            return QtCore.QProcess.NotRunning
        try:
            return self.process.state()
        except RuntimeError:
            self.process = None
            return QtCore.QProcess.NotRunning

    def _has_running_process(self) -> bool:
        """True only while the backend process is actively starting or running."""

        return self._process_state() != QtCore.QProcess.NotRunning

    def _dispose_process(self) -> None:
        """
        Release the current QProcess wrapper after a run has fully ended.

        Recreating the wrapper per run is slightly more verbose than reusing a
        single object, but it avoids stale-state edge cases after backend crashes,
        forced kills, or Python exceptions inside finish handlers.
        """

        proc = self.process
        if proc is None:
            return
        try:
            proc.close()
        except Exception:
            pass
        try:
            proc.deleteLater()
        except Exception:
            pass
        self.process = None

    def _start_preflight_validation(self, _checked: bool = False):
        """Start asynchronous pre-flight validation with visual feedback."""
        # Check if already running
        if self._has_running_process():
            self._log_message("[Warning] Simulation already running", severity="warning")
            return

        # Check if preflight already running
        if self.preflight_worker and self.preflight_worker.isRunning():
            self._log_message("[Warning] Pre-flight validation already in progress", severity="warning")
            return

        # Collect data for validation
        cmd_data = self._collect_preflight_data()
        if not cmd_data:
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Error",
                "Invalid input values detected. Please check your inputs."
            )
            return

        # Update UI for validation state
        self._set_preflight_state("validating")
        self._log_separator()
        self._log_message("[System] Starting pre-flight validation...", severity="system")

        # Create and start preflight worker
        self.preflight_worker = PreFlightWorker(cmd_data, self.main_script_path)

        # Connect signals
        self.preflight_worker.validation_complete.connect(self._on_preflight_complete)
        self.preflight_worker.validation_progress.connect(
            lambda msg: self._log_message(f"[Validation] {msg}", severity="system")
        )
        self.preflight_worker.validation_warning.connect(
            lambda msg: self._log_message(f"[Warning] {msg}", severity="warning")
        )
        self.preflight_worker.validation_error.connect(
            lambda msg: self._log_message(f"[Error] {msg}", severity="error")
        )

        # Start worker
        self.preflight_worker.start()

    def _set_preflight_state(self, state: str):
        """Update UI for pre-flight validation state."""
        if state == "validating":
            self.btn_run.setText("  Validating...")
            self.btn_run.setIcon(get_icon('fa6s.spinner', THEME['fg_main']))
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.progress_bar.show()
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("")
            self.lbl_progress.show()
            self.lbl_progress.setText("Validating inputs...")
            self.state_frame.show()
            self.dot_run.setProperty("kind", "warning")
            self.dot_run.style().unpolish(self.dot_run)
            self.dot_run.style().polish(self.dot_run)
            self.lbl_run_state.setText("Validating")
            self.badge_page.set_status("warning", "VALIDATING")
        elif state == "idle":
            # Restore button to original state
            self.btn_run.setText("  Run Mission Analysis")
            self.btn_run.setIcon(get_icon('fa6s.play', THEME['fg_main']))
            self._update_run_visuals("idle")

    def _on_preflight_complete(self, success: bool, message: str):
        """Handle pre-flight validation completion."""
        # Clean up worker
        self.preflight_worker = None

        # Restore button state first
        self.btn_run.setText("  Run Mission Analysis")
        self.btn_run.setIcon(get_icon('fa6s.play', THEME['fg_main']))

        if success:
            self._log_message(f"[System] {message}", severity="success")
            # Proceed with actual simulation
            QtCore.QTimer.singleShot(100, self._run_process)
        else:
            self._log_message(f"[System] {message}", severity="error")
            self._set_preflight_state("idle")

            # Show error dialog
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Failed",
                f"Pre-flight validation failed:\n\n{message}\n\nPlease check your configuration and try again."
            )

    
    # =========================================================================
    # 28. NAVIGATION & LOGIC UPDATES
    # =========================================================================
    
    def _on_nav_changed(self, row: int):
        """Handle sidebar navigation."""
        item = self.nav_list.item(row)
        if item:
            key = item.data(QtCore.Qt.UserRole)
            self._switch_page(key)
    
    def _switch_page(self, key: str):
        """Switch between main pages."""
        if key not in self._page_map:
            return
            
        idx = self._page_map[key]
        self.stack_pages.setCurrentIndex(idx)
        
        if self.nav_list.currentRow() != idx:
            self.nav_list.blockSignals(True)
            self.nav_list.setCurrentRow(idx)
            self.nav_list.blockSignals(False)
            
        labels = {item[0]: item[1] for item in NAV_PAGES}
        display_name = labels.get(key, key)
        
        if hasattr(self, "badge_page"):
            self.badge_page.set_status("info", display_name)
    
    def _toggle_log_collapsed(self, _checked: bool = False):
        """
        Collapse or restore the terminal panel without leaving dead vertical space.

        The earlier implementation only hid the text widget, but the splitter
        still respected the log panel's expanded minimum height.  That made the
        terminal feel half-closed.  The current implementation collapses the
        panel into a slim dock rail and restores the expanded geometry on
        demand.
        """

        self.is_log_collapsed = not self.is_log_collapsed

        icon_name = "fa6s.chevron-up" if self.is_log_collapsed else "fa6s.chevron-down"
        self.btn_log_toggle.setIcon(get_icon(icon_name, THEME['fg_main']))
        self.btn_log_toggle.setToolTip("Show Log Panel" if self.is_log_collapsed else "Hide Log Panel")

        if self.is_log_collapsed:
            self._log_expanded_sizes = self.main_splitter.sizes()
            self.txt_log.hide()
            self.chk_autoscroll.hide()
            self.btn_log_copy.hide()
            self.btn_log_clear.hide()
            self.log_header.setFixedHeight(34)
            self.log_panel.setMinimumHeight(34)
            self.main_splitter.setSizes([10000, 34])
        else:
            self.txt_log.show()
            self.chk_autoscroll.show()
            self.btn_log_copy.show()
            self.btn_log_clear.show()
            self.log_header.setFixedHeight(48)
            self.log_panel.setMinimumHeight(140)
            sizes = getattr(self, "_log_expanded_sizes", None)
            if sizes and sizes[1] > 60:
                self.main_splitter.setSizes(sizes)
            else:
                self._apply_default_log_splitter_sizes()
    
    # =========================================================================
    # 29. ORBIT & FORCE MODEL LOGIC
    # =========================================================================

    def _on_gravity_settings(self, _checked: bool = False):
        """
        Forward the gravity-settings action to the force-model page.

        The menu bar still exposes a top-level shortcut, but the page now owns
        the dialog implementation and the related shared config updates.
        """

        try:
            self.page_forces._on_gravity_settings(_checked)
            self._log_message("[UI] Gravity settings updated.", severity="system")
        except Exception as e:
            self._log_message(f"[Warning] Could not open gravity settings: {e}", severity="warning")

    def _on_albedo_settings(self, _checked: bool = False):
        """
        Forward the albedo-settings action to the force-model page.

        This keeps legacy menu hooks stable while preserving page ownership of
        the underlying UI widgets and config-edit workflow.
        """

        try:
            self.page_forces._on_albedo_settings(_checked)
            self._log_message("[UI] Albedo settings updated.", severity="system")
        except Exception as e:
            self._log_message(f"[Warning] Could not open albedo settings: {e}", severity="warning")
    
    def _on_solver_settings(self, _checked: bool = False):
        """Open Solver Settings Dialog."""
        dlg = SolverSettingsDialog(self, self.solver_cfg)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._log_message("[UI] Solver settings updated.", severity="system")

            # Keep the compact propagation-page widgets visually aligned with
            # the richer shared solver config edited in the dialog.
            prop_ui = getattr(self, "page_propagation", None)
            if prop_ui is not None:
                try:
                    prop_ui.sync_solver_widgets_from_config()
                except Exception:
                    pass

    def _on_spacecraft_settings(self, _checked: bool = False):
        """Open Spacecraft Properties Dialog."""
        dlg = SpacecraftBusDialog(self, self.spacecraft_cfg)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._log_message("[UI] Spacecraft properties updated.", severity="system")
    
    def _update_gravity_status(self):
        """Delegate gravity summary refresh to the dedicated force-model page."""
        try:
            self.page_forces._update_gravity_summary_ui()
        except Exception:
            pass
    

    # =========================================================================
    # 31. COMMAND BUILDING & PROCESS MANAGEMENT
    # =========================================================================
    def _build_command(self) -> List[str]:
        """
        Build the backend CLI command from the modular page/config state.

        Command translation is centralized in `ui_parts.command_builder` so the
        main window does not have to mirror the backend flag schema inline.
        """
        return build_command(
            python_executable=sys.executable,
            main_script_path=self.main_script_path,
            orbit=self.page_orbit.get_data(),
            forces=self.page_forces.get_data(),
            propagation=self.page_propagation.to_dict(),
            output=self.page_output.get_state(),
            data_files=self.page_data.get_state(),
            gravity_cfg=self.gravity_cfg,
            solver_cfg=self.solver_cfg,
            spacecraft_cfg=self.spacecraft_cfg,
            log_warning=lambda msg: self._log_message(msg, severity="warning"),
        )

    def _run_process(self):
        """Launch the mission propagation."""
        # Check if already running
        if self._has_running_process():
            return
        self._dispose_process()

        # Build command
        try:
            cmd_list = self._build_command()
        except Exception as e:
            self._log_message(f"[Error] Failed to build command: {e}", severity="error")
            return

        # Prep UI
        self._set_run_state("running")
        self.txt_log.clear()

        # Telemetry reset (TelemetryPage owns the plot)
        try:
            self.page_telemetry.telemetry_multiplot.clear_all()
        except Exception:
            pass

        self.progress_bar.setValue(0)
        self._stdout_buf = ""
        self._run_wall_t0 = time.time()
        self._last_telem_t_s = None
        self._progress_is_determinate = False
        self.progress_bar.show()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Starting...")
        if hasattr(self, "lbl_progress"):
            self.lbl_progress.show()
            self.lbl_progress.setText("Starting simulation...")
        if hasattr(self, "state_frame"):
            self.state_frame.show()

        # Calculate total duration for progress bar (from propagation page)
        self.sim_state.total_duration = 0.0
        prop_ui = getattr(self, "page_propagation", None)
        if prop_ui is not None:
            dur_txt = prop_ui.ent_duration.text().strip()
            if dur_txt:
                try:
                    dur_val = float(dur_txt)
                    if dur_val > 0:
                        unit = prop_ui.cb_duration_unit.currentText().strip().lower()
                        if unit.startswith("day"):
                            self.sim_state.total_duration = dur_val * 86400.0
                        else:
                            self.sim_state.total_duration = dur_val * 3600.0
                except ValueError:
                    self.sim_state.total_duration = 0.0

        # Prepare output directory
        out_dir_txt = self.page_output.get_state().output_dir.strip()
        try:
            out_dir = Path(out_dir_txt)
            out_dir.mkdir(parents=True, exist_ok=True)
            stop_path = out_dir / ".stlrps_stop"
            if stop_path.exists():
                stop_path.unlink()
        except Exception as e:
            self._log_message(f"[Warning] Could not prepare output dir: {e}", severity="warning")

        # Start process
        self._log_separator()
        self._log_message(f"[System] Launching mission analysis...", severity="system")

        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.readyReadStandardError.connect(self._handle_stderr)
        self.process.finished.connect(self._on_process_finished)

        # Ensure python process uses unbuffered output for streaming telemetry
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)

        self.process.start(cmd_list[0], cmd_list[1:])
        if not self.process.waitForStarted(1500):
            self._log_message("[Error] Failed to start backend process.", severity="error")
            self._set_run_state("idle")
            self._dispose_process()
            return

        self._log_message("[System] Backend process started.", severity="system")

    def _stop_process(self, _checked: bool = False):
        """Stop the running propagation."""
        if not self._has_running_process():
            return

        proc = self.process
        if proc is None:
            return
        
        self._log_message("[System] Sending stop signal...", severity="system")
        
        # Create stop file (signal for backend)
        try:
            out_dir = Path(self.page_output.get_state().output_dir.strip())
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / ".stlrps_stop").touch()
        except Exception as e:
            self._log_message(f"[Warning] Could not create stop file: {e}", severity="warning")
        
        # Step 1: Try graceful termination
        proc.terminate()
        
        # Wait for graceful termination
        if not proc.waitForFinished(2000):  # Wait 2 seconds
            self._log_message("[System] Graceful termination failed -> forcing kill...", severity="error")
            
            # Step 2: Force kill
            proc.kill()
            
            # Wait for kill to complete
            if not proc.waitForFinished(1000):  # Wait 1 more second
                self._log_message("[System] Kill command may have failed", severity="error")
        
        # Update UI
        self._set_run_state("idle")
        self._dispose_process()
    

    # -------------------------------------------------------------------------
    # Collision / impact monitoring
    # -------------------------------------------------------------------------
    @staticmethod
    def _try_float(v):
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if not s:
                return None
            return float(s)
        except Exception:
            return None

    def _check_collision(self, telem: dict) -> None:
        """Best-effort collision detection.

        Priority:
          1) If telemetry provides terrain clearance directly, use that.
          2) If telemetry provides a surface radius (km) and spacecraft radius (km), compare those.
          3) If telemetry provides altitude above mean radius (km) and surface elevation above mean (km), compare those.
          4) Fallback: altitude <= 0 km (mean-radius impact).
        """
        if getattr(self, "_collision_triggered", False):
            return

        def first_float(*keys: str) -> Optional[float]:
            for key in keys:
                value = self._try_float(telem.get(key))
                if value is not None:
                    return value
            return None

        r_km = first_float("r_km", "radius_km", "r_norm_km")
        alt_km = first_float("alt_km", "altitude_km", "alt")
        terrain_clearance_km = first_float(
            "terrain_clearance_km",
            "surface_clearance_km",
            "clearance_km",
        )
        surface_r_km = first_float("surface_r_km", "terrain_r_km", "ldem_r_km")
        surface_alt_km = first_float("surface_alt_km", "terrain_alt_km", "topo_km", "elev_km")

        if surface_r_km is None and surface_alt_km is not None and surface_alt_km > 500.0:
            surface_r_km = surface_alt_km
            surface_alt_km = None

        hit = False
        reason = ""

        if terrain_clearance_km is not None:
            if terrain_clearance_km <= 0.0:
                hit = True
                reason = (
                    "Impact detected "
                    f"(terrain_clearance_km={terrain_clearance_km:.3f} <= 0.000)."
                )
        elif (r_km is not None) and (surface_r_km is not None):
            if r_km <= surface_r_km:
                hit = True
                reason = f"Impact detected (r_km={r_km:.3f} <= surface_r_km={surface_r_km:.3f})."
        elif (alt_km is not None) and (surface_alt_km is not None):
            if alt_km <= surface_alt_km:
                hit = True
                reason = f"Impact detected (alt_km={alt_km:.3f} <= surface_alt_km={surface_alt_km:.3f})."
        elif alt_km is not None:
            if alt_km <= 0.0:
                hit = True
                reason = f"Impact detected (alt_km={alt_km:.3f} <= 0.000)."

        if not hit:
            return

        self._collision_triggered = True
        self._collision_reason = reason

        try:
            self._stop_process()
        except Exception:
            try:
                if getattr(self, "process", None) is not None:
                    self.process.kill()
            except Exception:
                pass

        try:
            QtWidgets.QMessageBox.warning(self, "Collision / Impact", reason)
        except Exception:
            try:
                self._log_message("[WARN] " + reason, severity="warning")
            except Exception:
                pass

    def _handle_stdout(self):
        """Handle stdout from process."""
        if self.process is None:
            return

        chunk = self.process.readAllStandardOutput().data().decode("utf-8", errors="ignore")
        if not chunk:
            return

        self._stdout_buf += chunk

        # Process complete lines
        while "\n" in self._stdout_buf:
            line, self._stdout_buf = self._stdout_buf.split("\n", 1)
            clean_line = line.rstrip("\r").strip()
            if not clean_line:
                continue

            # Check for telemetry JSON
            if clean_line.startswith("{") and ("\"t\"" in clean_line or "\"t_s\"" in clean_line):
                try:
                    telem = json.loads(clean_line)
                except json.JSONDecodeError:
                    try:
                        telem = ast.literal_eval(clean_line)
                    except (ValueError, SyntaxError):
                        telem = None

                if isinstance(telem, dict):
                    # Pass to enhanced telemetry system (TelemetryPage owns the plot)
                    try:
                        self.page_telemetry.telemetry_multiplot.add_datapoint(telem)
                    except Exception:
                        pass

                    # impact monitoring
                    self._check_collision(telem)

                    # Update progress based on time
                    t_s = None
                    for t_key in ["t_s", "t_sec", "t"]:
                        if t_key in telem:
                            t_s = float(telem[t_key])
                            # Handle unit conversion if 't' is used without explicit unit
                            if t_key == "t":
                                unit = str(telem.get("t_unit", "")).strip().lower()
                                if unit.startswith("h"):
                                    t_s *= 3600.0
                                elif unit.startswith("d"):
                                    t_s *= 86400.0
                            break

                    if t_s is not None:
                        self._last_telem_t_s = float(t_s)

                    if t_s is not None and self.sim_state.total_duration > 0:
                        total = float(self.sim_state.total_duration)
                        frac = 0.0 if total <= 0 else (float(t_s) / total)
                        frac = max(0.0, min(1.0, frac))

                        # determinate range is always 0..1000 for stability
                        if not self._progress_is_determinate:
                            self.progress_bar.setRange(0, 1000)
                            self.progress_bar.setTextVisible(True)
                            self._progress_is_determinate = True

                        self.progress_bar.setValue(int(frac * 1000.0))
                        self.progress_bar.setFormat(f"{(frac*100.0):4.1f}%")

                        # Extra text: t/T and ETA
                        if hasattr(self, "lbl_progress"):
                            t_days = float(t_s) / 86400.0
                            T_days = total / 86400.0
                            eta_txt = ""
                            if self._run_wall_t0 is not None and frac > 1e-6:
                                elapsed = max(0.0, time.time() - float(self._run_wall_t0))
                                eta_s = elapsed * (1.0 - frac) / max(frac, 1e-6)
                                if eta_s >= 3600:
                                    eta_txt = f" | ETA {eta_s/3600.0:.1f} h"
                                elif eta_s >= 60:
                                    eta_txt = f" | ETA {eta_s/60.0:.1f} min"
                                else:
                                    eta_txt = f" | ETA {eta_s:.0f} s"
                            self.lbl_progress.setText(f"{t_days:.2f}/{T_days:.2f} d{eta_txt}")

                    # Skip logging telemetry JSON to prevent log spam
                    continue

            # Regular log message
            self._log_message(clean_line)

    def _handle_stderr(self):
        """Handle stderr from process with warning/error classification."""
        if self.process is None:
            return
        data = self.process.readAllStandardError().data().decode('utf-8', errors='ignore')
        if not data.strip():
            return

        lowered = data.lower()
        severity = "warning" if (
            "warning" in lowered
            and "traceback" not in lowered
            and "[fatal]" not in lowered
            and "fatal error" not in lowered
        ) else "error"
        self._log_message(data, severity=severity)
    
    def _on_process_finished(self, exit_code, exit_status):
        """
        Handle process completion and always return the UI to a restart-ready state.

        Even failed runs should leave the window immediately runnable again. The
        log retains the error details, so there is little value in keeping the
        header latched in a pseudo-running state after the backend has exited.
        """

        if exit_code == 0:
            status_msg = "Mission analysis completed successfully"
            self._log_message(f"[System] {status_msg}", severity="success")
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setFormat("Done")
            self.progress_bar.setValue(1000)
            if hasattr(self, "lbl_progress"):
                self.lbl_progress.setText("Completed")
        else:
            status_msg = f"Mission analysis failed (Exit Code: {exit_code})"
            self._log_message(f"[System] {status_msg}", severity="error")

            # Restore a readable percentage format if the bar was left in its
            # indeterminate warm-up mode when the backend failed early.
            if not self.progress_bar.isTextVisible():
                self.progress_bar.setFormat("%p%")
                self.progress_bar.setTextVisible(True)
            if hasattr(self, "lbl_progress"):
                self.lbl_progress.setText("Run stopped with an error")

        self._run_wall_t0 = None
        self._last_telem_t_s = None
        self._progress_is_determinate = False
        self._set_run_state("idle")
        self._dispose_process()
    
    def _set_run_state(self, state: str):
        """Update UI based on execution state."""
        self.sim_state.status = state
        is_running = (state == "running")
        
        # Buttons
        self.btn_run.setEnabled(not is_running)
        self.btn_stop.setEnabled(is_running)
        
        self._update_run_visuals(state)
    
    # =========================================================================
    # 33. STATE MANAGEMENT & SERIALIZATION
    # =========================================================================

    def _collect_preset_dict(self) -> Dict[str, Any]:
        """Collect all modular page/config state into a serializable snapshot."""

        snapshot = collect_session_snapshot(
            orbit_page=self.page_orbit,
            propagation_page=self.page_propagation,
            force_page=self.page_forces,
            output_page=self.page_output,
            data_page=self.page_data,
            gravity_cfg=self.gravity_cfg,
            albedo_cfg=self.albedo_cfg,
            solver_cfg=self.solver_cfg,
            spacecraft_cfg=self.spacecraft_cfg,
            app_version=APP_VERSION,
            mc_page=getattr(self, "page_mc", None),
            surrogate_page=getattr(self, "page_surrogate", None),
        )

        # Collect visual workspace state (Task 11)
        active_key = ""
        try:
            row = self.nav_list.currentRow()
            if 0 <= row < len(NAV_PAGES):
                active_key = NAV_PAGES[row][0]
        except Exception:
            pass

        splitter_sizes: list[int] = []
        try:
            splitter_sizes = list(self.main_splitter.sizes())
        except Exception:
            pass

        telemetry_plot_type = ""
        telemetry_time_unit = ""
        try:
            mp = getattr(getattr(self, "page_telemetry", None), "multi_plot", None)
            if mp is None:
                mp = getattr(self, "page_telemetry", None)
            if mp is not None:
                combo = getattr(mp, "plot_type_combo", None)
                if combo:
                    telemetry_plot_type = combo.currentText()
                tu = getattr(mp, "time_axis_combo", None)
                if tu:
                    telemetry_time_unit = tu.currentText()
        except Exception:
            pass

        artifact_filter = ""
        artifact_recursive = False
        try:
            cb = getattr(getattr(self, "page_output", None), "cb_artifact_filter", None)
            if cb:
                artifact_filter = cb.currentText()
            chk = getattr(getattr(self, "page_output", None), "chk_recursive_scan", None)
            if chk:
                artifact_recursive = chk.isChecked()
        except Exception:
            pass

        mc_active_tab = 0
        try:
            mc_tabs = getattr(getattr(self, "page_mc", None), "tabs", None)
            if mc_tabs:
                mc_active_tab = mc_tabs.currentIndex()
        except Exception:
            pass

        snapshot["visual_state"] = collect_visual_state(
            active_page_key=active_key,
            splitter_sizes=splitter_sizes,
            log_collapsed=bool(self.is_log_collapsed),
            telemetry_plot_type=telemetry_plot_type,
            telemetry_time_unit=telemetry_time_unit,
            artifact_filter=artifact_filter,
            artifact_recursive=artifact_recursive,
            mc_active_tab=mc_active_tab,
        )
        return snapshot

    def _apply_preset_dict(self, data: Dict[str, Any]):
        """Apply a saved session payload through the modular restore helpers."""
        try:
            apply_session_snapshot(
                data,
                orbit_page=self.page_orbit,
                propagation_page=self.page_propagation,
                force_page=self.page_forces,
                output_page=self.page_output,
                data_page=self.page_data,
                gravity_cfg=self.gravity_cfg,
                albedo_cfg=self.albedo_cfg,
                solver_cfg=self.solver_cfg,
                spacecraft_cfg=self.spacecraft_cfg,
                project_root=PROJECT_ROOT,
                log_warning=lambda msg: self._log_message(msg, severity="warning"),
                mc_page=getattr(self, "page_mc", None),
                surrogate_page=getattr(self, "page_surrogate", None),
            )

            state = self.page_data.get_state()
            self.ldem_root_path = state.ldem_root
            self.albedo_root_path = state.albedo_root
            self.kernel_dir_path = state.kernel_dir
            self.ldem_ppd = state.ldem_ppd
        except Exception as e:
            self._log_message(f"[Warning] Could not fully restore session: {e}", severity="warning")

        # Restore visual workspace state — tolerant of missing key (old sessions)
        try:
            visual = data.get("visual_state", {}) or {}
            if visual:
                apply_visual_state(visual, main_window=self)
        except Exception as e:
            self._log_message(f"[Warning] Could not restore visual state: {e}", severity="warning")

    def _browse_out_dir(self, _checked: bool = False):
        """Open directory dialog for output directory."""
        current = self.page_output.get_state().output_dir or str(PROJECT_ROOT)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", current
        )
        if path:
            self.page_output.set_output_dir(normalize_path(path))
            self._log_message(f"[UI] Output directory set to: {Path(path).name}", severity="system")
    
    def _action_open_out_dir(self, _checked: bool = False):
        """Open output directory in file explorer."""
        out_dir = self.page_output.get_state().output_dir.strip()
        if not out_dir:
            return
        
        path = Path(out_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
            self._log_message(f"[UI] Opened output directory: {path}", severity="system")
        except Exception as e:
            self._log_message(f"[Error] Could not open directory: {e}", severity="error")
    
    def _action_load_session(self, _checked: bool = False):
        """Load session from file."""
        current = str(self.app_data_dir) if self.app_data_dir.exists() else str(PROJECT_ROOT)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Mission Profile", current,
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not path:
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._apply_preset_dict(data)
            self._log_message(f"[UI] Session loaded from: {Path(path).name}", severity="success")
        except Exception as e:
            self._log_message(f"[Error] Failed to load session: {e}", severity="error")
            QtWidgets.QMessageBox.warning(
                self, "Load Error",
                f"Failed to load session file:\n\n{str(e)}"
            )
    
    def _action_save_session(self, _checked: bool = False):
        """Save session to file."""
        current = str(self.app_data_dir / "mission_profile.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Mission Profile", current,
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not path:
            return
        
        try:
            data = self._collect_preset_dict()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._log_message(f"[UI] Session saved to: {Path(path).name}", severity="success")
        except Exception as e:
            self._log_message(f"[Error] Failed to save session: {e}", severity="error")
            QtWidgets.QMessageBox.warning(
                self, "Save Error",
                f"Failed to save session file:\n\n{str(e)}"
            )
    
    def _try_prefill_topography_from_config(self):
        """Auto-detect data roots using the repository-aware persistence helper."""
        try:
            new_state, messages = autodetect_data_state(PROJECT_ROOT, self.page_data.get_state())
            self.page_data.apply_state(new_state)
            self.ldem_root_path = new_state.ldem_root
            self.albedo_root_path = new_state.albedo_root
            self.kernel_dir_path = new_state.kernel_dir
            self.ldem_ppd = new_state.ldem_ppd
            for message in messages:
                self._log_message(message, severity="system")
        except Exception:
            pass

    def _try_load_last_session(self):
        """Attempt to load last session from app data."""
        if not self.session_path.exists():
            return
        
        try:
            with open(self.session_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._apply_preset_dict(data)
            self._log_message("[UI] Previous session restored.", severity="system")
        except Exception as e:
            self._log_message(f"[Warning] Could not restore last session: {e}", severity="warning")
    
    def _bootstrap(self):
        """Initial bootstrapping tasks."""
        self._log_message(f"[System] {APP_NAME} initialized", severity="system")
        self._log_message(f"[System] Project Root: {PROJECT_ROOT}", severity="system")
        self._log_message(f"[System] Main Script: {self.main_script_path.name}", severity="system")

        # Check for main script
        if not self.main_script_path.exists():
            self._log_message(f"[Error] Main simulation script not found at: {self.main_script_path}", severity="error")
            QtWidgets.QMessageBox.critical(
                self, "Critical Error",
                f"Main simulation script not found:\n{self.main_script_path}\n\n"
                "Please ensure the backend is properly installed."
            )

        # Wire page signals (no aliases: pages own their widgets)
        try:
            self.page_propagation.solver_settings_requested.connect(self._on_solver_settings)
            self.page_propagation.spacecraft_settings_requested.connect(self._on_spacecraft_settings)
        except Exception:
            pass

        # Let the window reach a real geometry before forcing the initial
        # splitter ratio; this prevents page size hints from trapping the log
        # panel at an awkwardly small height.
        QtCore.QTimer.singleShot(0, self._apply_default_log_splitter_sizes)

        # Update command preview (delayed: lets UI settle)
        QtCore.QTimer.singleShot(500, self._update_command_preview_silent)

        # Auto-detect gravity file if none set
        if not self.gravity_cfg.file_path:
            QtCore.QTimer.singleShot(1000, self._auto_detect_gravity)

    def _auto_detect_gravity(self):
        """Auto-detect gravity file in background."""
        if not hasattr(self, "gravity_cfg"):
            return
        
        if not self.gravity_cfg.file_path:
            found = find_best_gravity_file(PROJECT_ROOT, self.gravity_cfg.degree)
            if found:
                self.gravity_cfg.file_path = found
                self._log_message(f"[UI] Auto-detected gravity model: {Path(found).name}", severity="system")
                try:
                    self.page_forces._update_gravity_summary_ui()
                except Exception:
                    pass
    
    def _update_run_visuals(self, state: str):
        """Update run state visuals."""
        self.dot_run.setProperty("kind", state)
        self.dot_run.style().unpolish(self.dot_run)
        self.dot_run.style().polish(self.dot_run)
        
        status_map = {
            "idle": "",
            "running": "Propagation active",
            "error": "Run error",
            "warning": "Validating"
        }
        label_text = status_map.get(state, "")
        self.lbl_run_state.setText(label_text)
        self.sim_state.message = label_text

        if hasattr(self, "state_frame"):
            self.state_frame.setVisible(bool(label_text))

        if state == "idle":
            self.progress_bar.hide()
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setFormat("")
            self.lbl_progress.clear()
            self.lbl_progress.hide()
    
    def _update_status_bar(self) -> None:
        """
        Refresh the mission status summary bar from current UI state.
        Called from _ui_tick every 250ms.
        """
        try:
            # Gravity backend label
            if hasattr(self, "gravity_cfg"):
                backend = str(getattr(self.gravity_cfg, "backend", "classic_sh") or "classic_sh")
                if backend == "st_lrps":
                    model_dir = str(getattr(self.gravity_cfg, "st_lrps_model_dir", "") or "").strip()
                    model_name = model_dir.split("/")[-1].split("\\")[-1] if model_dir else "?"
                    grav_text = f"ST-LRPS [{model_name}]"
                else:
                    deg = int(getattr(self.gravity_cfg, "degree", 100) or 100)
                    grav_text = f"SH [{deg}]"
                if hasattr(self, "lbl_gravity_status"):
                    self.lbl_gravity_status.setText(grav_text)
        except Exception:
            pass

        try:
            # Output directory (shortened)
            if hasattr(self, "page_output"):
                out_dir = self.page_output.get_state().output_dir.strip()
                if not out_dir:
                    out_text = "Not set"
                elif len(out_dir) > 30:
                    out_text = "..." + out_dir[-27:]
                else:
                    out_text = out_dir
                if hasattr(self, "lbl_output_status"):
                    self.lbl_output_status.setText(out_text)
        except Exception:
            pass

        try:
            # Preflight state badge
            if hasattr(self, "lbl_preflight_status") and hasattr(self, "preflight_worker"):
                if self.preflight_worker is not None and self.preflight_worker.isRunning():
                    self.lbl_preflight_status.set_status("warning", "CHECK")
        except Exception:
            pass

        try:
            # Run state badge
            if hasattr(self, "lbl_run_status") and hasattr(self, "sim_state"):
                status = self.sim_state.status if hasattr(self.sim_state, "status") else "idle"
                status_map = {
                    "idle": ("IDLE", "info"),
                    "running": ("RUN", "success"),
                    "done": ("DONE", "success"),
                    "error": ("ERROR", "error"),
                    "stopped": ("STOP", "warning"),
                    "preflight": ("CHECK", "warning"),
                }
                kind_text, kind = status_map.get(status, ("IDLE", "info"))
                self.lbl_run_status.set_status(kind, kind_text)
        except Exception:
            pass

    def _ui_tick(self):
        """Periodic UI updates."""
        # Update command preview if needed
        self._update_command_preview_silent()

        # Update mission status bar
        self._update_status_bar()

        # Update session auto-save (every 30 seconds)
        current_time = QtCore.QDateTime.currentSecsSinceEpoch()
        if hasattr(self, "_last_save_time"):
            if current_time - self._last_save_time > 30:
                self._auto_save_session()
                self._last_save_time = current_time
        else:
            self._last_save_time = current_time
    
    def _auto_save_session(self, *, notify_on_failure: bool = False) -> bool:
        """
        Auto-save session state and surface failures instead of swallowing them.

        Periodic saves should stay quiet on success, but repeated failures should
        still be diagnosable. The method therefore logs each distinct failure
        once and can optionally show a dialog when called from a user-driven
        lifecycle event such as window close.
        """
        try:
            data = self._collect_preset_dict()
            with open(self.session_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._last_autosave_error = None
            return True
        except Exception as exc:
            message = f"[Warning] Session auto-save failed: {exc}"
            if getattr(self, "_last_autosave_error", None) != message:
                self._last_autosave_error = message
                try:
                    self._log_message(message, severity="warning")
                except Exception:
                    pass
            if notify_on_failure:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Session Save Warning",
                    "Could not save the current session automatically.\n\n"
                    f"{exc}",
                )
            return False
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any running processes
        if self._has_running_process():
            reply = QtWidgets.QMessageBox.question(
                self, "Confirm Exit",
                "A simulation is currently running. Are you sure you want to exit?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.No:
                event.ignore()
                return
            
            # Try to stop process
            self._stop_process()
            if self.process is not None:
                try:
                    if not self.process.waitForFinished(2000):
                        self.process.kill()
                except RuntimeError:
                    self.process = None
        
        # Stop preflight worker if running
        if self.preflight_worker and self.preflight_worker.isRunning():
            self.preflight_worker.stop()
            self.preflight_worker.wait(1000)

        # Stop any MC subprocess
        if self.mc_process is not None:
            try:
                if self.mc_process.state() != QtCore.QProcess.NotRunning:
                    self.mc_process.kill()
                    self.mc_process.waitForFinished(1000)
            except Exception:
                pass

        # Stop any background MC analysis work owned by the Monte Carlo page.
        try:
            if hasattr(self, "page_mc"):
                self.page_mc.shutdown()
        except Exception:
            pass

        # Save the latest state after shutdown prompts/process cleanup so the
        # persisted snapshot reflects the final visible UI values.
        self._auto_save_session(notify_on_failure=True)

        event.accept()

    # =========================================================================
    # UI HELPERS (used by multiple pages: forces, output, etc.)
    # =========================================================================

    def _create_card(self, title: str) -> QtWidgets.QGroupBox:
        """Factory for standard titled group boxes (Cards)."""
        gb = QtWidgets.QGroupBox(title)
        return gb

    def _wrap_scroll(self, content_widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        """Wraps content in a responsive, frameless scroll area."""
        container = QtWidgets.QWidget()
        container.setObjectName("scrollBody")
        container.setMinimumHeight(0)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)
        layout.addWidget(content_widget)
        layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Ignored,
        )
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.viewport().setAutoFillBackground(False)
        scroll.setWidget(container)
        return scroll
    
    # =========================================================================
    # COMMAND PREVIEW (used by Output page)
    # =========================================================================

    def _build_command_preview_safe(self) -> Tuple[str, str]:
        """Generate shell-safe command string for preview."""
        try:
            return (build_command_preview(self._build_command()), "")
        except Exception as e:
            return ("", f"Error building command: {type(e).__name__}: {e}")


    def _update_command_preview_silent(self):
        """Update command preview without logging."""
        if not hasattr(self, "page_output"):
            return

        cmd_str, err = self._build_command_preview_safe()

        if err:
            self.page_output.set_command_preview(f"# PREVIEW ERROR\n{err}", is_error=True)
            return

        if cmd_str != getattr(self, "last_cmd_preview", ""):
            self.last_cmd_preview = cmd_str
            self.page_output.set_command_preview(cmd_str, is_error=False)


    def _update_command_preview(self, _checked: bool = False):
        """Update command preview with logging."""
        self._update_command_preview_silent()
        self._log_message("[UI] Command preview refreshed.", severity="system")


    def _copy_command_preview(self, _checked: bool = False):
        """Copy command to clipboard."""
        cmd_str = getattr(self, "last_cmd_preview", "")
        if cmd_str:
            QtWidgets.QApplication.clipboard().setText(cmd_str)
            self._log_message("[UI] Command copied to clipboard.", severity="system")
        else:
            self._log_message("[UI] Nothing to copy (Command is empty).", severity="warning")




# =============================================================================
# 35. APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Application entry point."""
    
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("ST_LRPS")
    
    # Load fonts
    font = load_fonts()
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
