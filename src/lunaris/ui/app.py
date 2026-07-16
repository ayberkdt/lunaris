"""
PySide6 desktop interface for the general Lunaris orbit simulator.

This module hosts the main application window and wires the modular page widgets
from `lunaris.ui.widgets` into a single desktop workflow.
"""



# =============================================================================
# 0.                                    IMPORTS
# =============================================================================
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.common.constants import DAY_S
from lunaris.ui.components import ElidedLabel, PageShell

# =============================================================================
# 1.                            UI CONFIGURATION
# =============================================================================
from lunaris.ui.core.ui_commons import (
    APP_NAME,
    APP_VERSION,
    LOG_COLORS,
    THEME,
    WINDOW_SETTINGS,
    prefers_reduced_motion,
    stepper_arrow_icons,
)
from lunaris.ui.core.ui_commons import (
    ASSETS_DIR as UI_ASSETS_DIR,
)
from lunaris.ui.theme import build_app_stylesheet
from lunaris.ui.theme.tokens import DESIGN_TOKENS
from lunaris.ui.widgets.log_panel import (
    COLLAPSED_HEIGHT as LOG_COLLAPSED_HEIGHT,
)
from lunaris.ui.widgets.log_panel import (
    EXPANDED_MIN_HEIGHT as EXPANDED_MIN_LOG_HEIGHT,
)
from lunaris.ui.widgets.log_panel import (
    ExecutionConsoleDock,
)

# Navigation entries for the specialized mission-analysis pages.
NAV_PAGES = [
    ("Orbit",       "Orbit Setup",       "fa6s.rocket"),
    ("Forces",      "Force Models",      "fa6s.atom"),
    ("Propagation", "Propagation",       "fa6s.hourglass-half"),
    ("Output",      "Results & Export",  "fa6s.folder-open"),
    ("Telemetry",   "Live Telemetry",    "fa6s.chart-line"),
    ("Monitor",     "Mission Monitor",   "fa6s.gauge-high"),
    ("Data",        "Data & Files",      "fa6s.database"),
    ("BatchPropagation",  "Batch Propagation", "fa6s.dice"),
    ("FrozenSearch", "Frozen Search",    "fa6s.snowflake"),
]

PAGE_DESCRIPTIONS = {
    "Orbit": "Define the initial lunar orbit and review its derived geometry.",
    "Forces": "Select and configure the physical models used by the propagator.",
    "Propagation": "Set the mission timeline, integrator, and output cadence.",
    "Output": "Choose result destinations and inspect generated artifacts.",
    "Telemetry": "Signals from the active run.",
    "Monitor": "Mission observation console: live telemetry widgets with provenance.",
    "Data": "Locate, validate, and manage mission data sources.",
    "BatchPropagation": "Configure ensemble sampling, execute batch propagation, and inspect distributions.",
    "FrozenSearch": "Run the staged frozen-orbit search and inspect its output contract.",
}

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
from lunaris.ui.core.ui_commons import find_project_root, load_fonts

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = UI_ASSETS_DIR


# =============================================================================
# 4.                          ICON UTILITIES
# =============================================================================
from lunaris.ui.core import shortcuts
from lunaris.ui.core.command_builder import (
    build_batch_command,
    build_command,
    build_command_preview,
    build_preflight_snapshot,
)
from lunaris.ui.core.log_stream import LineAssembler
from lunaris.ui.core.preflight_validation import PreFlightWorker
from lunaris.ui.core.session_persistence import (
    apply_session_snapshot,
    apply_visual_state,
    autodetect_data_state,
    collect_session_snapshot,
    collect_visual_state,
)
from lunaris.ui.core.solver_policy import normalize_solver_config_object

# =============================================================================
# 5.                          UTILITY HELPERS
# =============================================================================
from lunaris.ui.core.ui_commons import get_icon, normalize_path
from lunaris.ui.monitor.protocol import MetaMessage, ProtocolProblem, SampleMessage
from lunaris.ui.monitor.workspace import MonitorController, MonitorPage
from lunaris.ui.pages.force_models_page import find_best_gravity_file

# =============================================================================
# 6.                        DATACLASSES (main glue)
# =============================================================================
from lunaris.ui.pages.mission_propagation_page import UISolverConfig, UISpacecraftConfig
from lunaris.ui.pages.result_exports_page import OutputPageState, ResultsExportPage


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
import contextlib

from lunaris.ui.core.ui_commons import StatusBadge

# =============================================================================
# 10.                       GRAVITY CONFIGURATION
# =============================================================================
# =============================================================================
# 15.                       ALBEDO CONFIGURATION
# =============================================================================
from lunaris.ui.pages.force_models_page import (
    UIAlbedoConfig,
    UIGravityConfig,
    UIThermalConfig,
)

# =============================================================================
# 11.                       SOLVER SETTINGS DIALOG
# =============================================================================
# =============================================================================
# 12.                       SPACECRAFT BUILDER DIALOG
# =============================================================================
from lunaris.ui.pages.mission_propagation_page import SolverSettingsDialog, SpacecraftBusDialog

# =============================================================================
# 16.                       UI HELPERS
# =============================================================================

# =============================================================================
# 17.                       MAIN WINDOW APPLICATION
# =============================================================================

class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the modular Lunaris Mission Studio UI.

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
        # Restore the last window geometry (size/position) if one was saved.
        try:
            saved_geometry = self._density_settings().value("ui/geometry")
            if saved_geometry:
                self.restoreGeometry(saved_geometry)
        except Exception:
            pass

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

        # Session Persistence.
        #
        # The app shipped historically as "ST-LRPS Studio" and stored its data
        # under an "STLRPSStudio" folder. The visible app is now "Lunaris Mission
        # Studio", so new data lives under "LunarisMissionStudio". We still check
        # the legacy folder once (read-only) so a user's previously saved mission
        # profile survives the rename instead of silently disappearing.
        app_data_loc = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)
        _base_dir = Path(app_data_loc) if app_data_loc else Path.home()
        _app_data_override = os.environ.get("LUNARIS_APP_DATA_DIR", "").strip()
        self.app_data_dir = (
            Path(_app_data_override).expanduser()
            if _app_data_override
            else (
                _base_dir / "LunarisMissionStudio"
                if app_data_loc
                else _base_dir / ".lunaris_studio"
            )
        )
        self.app_data_dir.mkdir(parents=True, exist_ok=True)

        self.session_path = self.app_data_dir / "studio_session.json"

        # Legacy app-data location, kept only for one-time backward-compatible
        # session migration (see _try_load_last_session).
        _legacy_dir = (
            self.app_data_dir
            if _app_data_override
            else (
                _base_dir / "STLRPSStudio"
                if app_data_loc
                else _base_dir / ".stlrps_studio"
            )
        )
        self._legacy_session_path = _legacy_dir / "studio_session.json"

        # ---------------------------------------------------------------------
        # 3. Application State & Sub-Configs
        # ---------------------------------------------------------------------
        self.process: QtCore.QProcess | None = None
        self.batch_process: QtCore.QProcess | None = None
        self.preflight_worker: PreFlightWorker | None = None
        self.batch_runner_script_path   = _lunaris_pkg / "cli" / "batch_runner.py"
        self._batch_stdout_buf: str = ""

        # UI State Containers (Mutable)
        self.sim_state = SimulationState()
        self.gravity_cfg = UIGravityConfig()
        self.albedo_cfg = UIAlbedoConfig()
        self.thermal_cfg = UIThermalConfig()
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
        self.recent_presets: list[str] = []
        self.last_cmd_preview: str = ""
        self.is_log_collapsed: bool = True
        self._visual_state_restored: bool = False
        # Independent partial-line buffers for the two backend streams. Mixing
        # stdout and stderr fragments would interleave half-written lines, so
        # each stream owns a dedicated assembler.
        self._stdout_assembler = LineAssembler()
        self._stderr_assembler = LineAssembler()
        # Reset impact monitoring for this run
        self._collision_triggered = False
        self._collision_reason = ""
        # Progress tracking
        self._run_wall_t0: float | None = None
        self._last_telem_t_s: float | None = None
        self._progress_is_determinate: bool = False

        # UI density (comfortable / compact) — restored from user settings so the
        # choice persists across sessions; applied by _apply_theme().
        self._density: str = self._load_density_pref()

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
        root.setContentsMargins(
            DESIGN_TOKENS.layout.shell_margin,
            DESIGN_TOKENS.layout.shell_margin,
            DESIGN_TOKENS.layout.shell_margin,
            DESIGN_TOKENS.layout.shell_margin,
        )
        root.setSpacing(DESIGN_TOKENS.layout.shell_gap)

        # ---------------------------------------------------------------------
        # A. Header Bar (Title, Status, Actions)
        # ---------------------------------------------------------------------
        header_frame = QtWidgets.QFrame()
        header_frame.setObjectName("header")
        h_layout = QtWidgets.QHBoxLayout(header_frame)
        # Shell chrome shares one horizontal margin (spacing.lg) so the header and
        # the status summary bar below it align to a common left edge; vertical
        # padding and gaps snap to the 4/8 spacing scale.
        _sp = DESIGN_TOKENS.spacing
        h_layout.setContentsMargins(_sp.lg, _sp.sm, _sp.lg, _sp.sm)
        h_layout.setSpacing(_sp.md)

        # App Title. Elided rather than plain: when a run starts the header
        # gains ~450px of progress/stop chrome, and a plain QLabel answers that
        # squeeze by clipping mid-glyph ("Lunaris Mis") instead of eliding.
        title_lbl = ElidedLabel(APP_NAME)
        title_lbl.setObjectName("title")
        self.lbl_app_title = title_lbl
        h_layout.addWidget(title_lbl)

        # Page Indicator (StatusBadge). Fixed horizontally: a centred badge
        # answers a squeeze by clipping *both* ends ("PROPAGATION" ->
        # "ROPAGATIO"), which is unreadable. It must never shrink; the header
        # sheds lower-priority items instead (see _apply_header_breakpoint).
        self.badge_page = StatusBadge("Orbit", "info")
        self.badge_page.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        h_layout.addWidget(self.badge_page)

        # Header context chips. The separate mission-status ribbon (which showed
        # idle "Preflight/Run IDLE" badges carrying no information) is gone; the
        # two settings genuinely worth seeing from every page — the gravity model
        # and the output destination — live here as quiet, clickable chips. Each
        # opens its owning control (gravity dialog / output picker). Execution
        # readiness is reported by the run dot + progress on the right, only while
        # a run is active, so nothing here shouts when the app is idle.
        def _header_chip(icon_name: str, initial: str,
                         on_click) -> QtWidgets.QPushButton:
            chip = QtWidgets.QPushButton(initial)
            chip.setObjectName("headerContextChip")
            chip.setCursor(QtCore.Qt.PointingHandCursor)
            chip.setIcon(get_icon(icon_name, THEME['fg_muted']))
            chip.clicked.connect(lambda _=False: on_click())
            # QPushButton has no elide mode, so a squeezed chip clips its label.
            # Chips hold their size and are hidden wholesale at the narrow
            # breakpoint instead.
            chip.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
            )
            return chip

        h_layout.addSpacing(DESIGN_TOKENS.spacing.md)
        self.lbl_gravity_status = _header_chip(
            "fa6s.atom", "SH [100]", self._on_gravity_settings)
        self.lbl_gravity_status.setToolTip("Gravity model — click to configure")
        h_layout.addWidget(self.lbl_gravity_status)

        self.lbl_output_status = _header_chip(
            "fa6s.folder-open", "Not set", self._browse_out_dir)
        self.lbl_output_status.setToolTip("Output directory — click to choose")
        self.lbl_output_status.setMaximumWidth(300)
        h_layout.addWidget(self.lbl_output_status)

        # Working reference frame — informational (not clickable): the engine
        # propagates in one frame and every number on every page is expressed
        # in it. The label resolves from the engine SSOT constant; if that
        # import is unavailable the chip is omitted rather than guessed.
        self.lbl_frame_status: QtWidgets.QLabel | None = None
        try:
            from lunaris.physics.ephemeris import DEFAULT_INERTIAL_FRAME
        except Exception:
            DEFAULT_INERTIAL_FRAME = ""
        if DEFAULT_INERTIAL_FRAME:
            frame_chip = QtWidgets.QLabel(f"Moon-centered {DEFAULT_INERTIAL_FRAME}")
            frame_chip.setObjectName("headerContextChip")
            frame_chip.setToolTip(
                "Working reference frame: Moon-centered inertial, "
                f"{DEFAULT_INERTIAL_FRAME} axes. All state vectors and "
                "orbital elements shown in the app use this frame."
            )
            frame_chip.setAccessibleName("Working reference frame")
            frame_chip.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
            )
            self.lbl_frame_status = frame_chip
            h_layout.addWidget(frame_chip)

        # Resolved-backend provenance for the most recent completed run.
        # Hidden until a run that reports backend metadata finishes; on a
        # requested!=actual fallback it switches to the warning style and
        # carries the reported reason, so a silent GPU->CPU downgrade is
        # visible from every page, not only inside the Batch workspace.
        self.lbl_backend_status = QtWidgets.QLabel("")
        self.lbl_backend_status.setObjectName("headerContextChip")
        self.lbl_backend_status.setAccessibleName("Last run backend")
        self.lbl_backend_status.hide()
        h_layout.addWidget(self.lbl_backend_status)

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
        h_layout.addWidget(self.progress_bar)
        # Extra progress text (t/T + ETA). A *maximum* rather than the previous
        # 155px minimum: as a minimum it forced the label to hold width it could
        # not fill, and its unelided text painted over the run-state chip to its
        # right. Elided + capped, it degrades to "…" and never overlaps.
        self.lbl_progress = ElidedLabel("")
        self.lbl_progress.setObjectName("progressText")
        self.lbl_progress.setMaximumWidth(200)
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
        self.state_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        h_layout.addWidget(self.state_frame)

        h_layout.addSpacing(16)

        # Action Buttons
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_stop.setIcon(get_icon('fa6s.stop', THEME['fg_main']))
        self.btn_stop.setEnabled(False)
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self._stop_process)

        self.btn_run = QtWidgets.QPushButton("Run Analysis")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_run.setIcon(get_icon('fa6s.play', THEME['fg_main']))
        self.btn_run.clicked.connect(self._start_preflight_validation)

        # Both run actions live in the global header so they are reachable from
        # every page. Previously Run Analysis was mounted only on the Orbit page
        # header (via the PageShell action slot), so it disappeared on every
        # other workspace. Run stays primary and always visible; Stop sits to
        # its right and only appears while a run is active.
        # The run actions are the header's highest-priority items: they hold
        # their full size at every width and everything else yields around them.
        for _btn in (self.btn_run, self.btn_stop):
            _btn.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
            )
        h_layout.addWidget(self.btn_run)
        h_layout.addWidget(self.btn_stop)

        # The header stays quieter when the app is idle. Progress and transient
        # execution state indicators are only shown while something actionable
        # is happening.
        self.progress_bar.hide()
        self.lbl_progress.hide()
        self.state_frame.hide()

        self._header_frame = header_frame
        self._in_header_breakpoint = False
        # Re-run the width budget whenever the header's contents change, not
        # only on resize. Showing the run chrome adds ~450px but emits no
        # resizeEvent, and callers show it *after* flipping the run state, so
        # any hook hung off the state transition would measure the old header.
        # LayoutRequest fires on child show/hide/sizeHint changes, which is
        # exactly the condition the budget depends on.
        header_frame.installEventFilter(self)
        self._apply_header_breakpoint()

        root.addWidget(header_frame)

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
        self.nav_list.setAccessibleName("Workspace navigation")
        self.nav_list.setFixedWidth(DESIGN_TOKENS.layout.nav_width)
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
            item.setSizeHint(QtCore.QSize(DESIGN_TOKENS.layout.nav_width - 20, 40))
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
        self.page_monitor = self._build_page_monitor()
        self.page_data = self._build_page_data()
        self.page_batch = self._build_page_batch()
        self.page_frozen_search = self._build_page_frozen_search()

        self.page_shells = {
            "Orbit": PageShell(
                "Orbit Setup",
                PAGE_DESCRIPTIONS["Orbit"],
                content=self.page_orbit,
                # The Orbit page owns an internal split (scrollable form + fixed
                # preview), so it must NOT sit inside the shell's own scroll/max-
                # width column — that is what pushed it into a narrow centred band
                # and forced scrolling just to see the preview update.
                scrollable=False,
            ),
            "Forces": PageShell(
                "Force Models",
                PAGE_DESCRIPTIONS["Forces"],
                content=self.page_forces,
            ),
            "Propagation": PageShell(
                "Propagation",
                PAGE_DESCRIPTIONS["Propagation"],
                content=self.page_propagation,
            ),
            "Output": PageShell(
                "Results & Export",
                PAGE_DESCRIPTIONS["Output"],
                content=self.page_output,
            ),
            "Telemetry": PageShell(
                "Live Telemetry",
                PAGE_DESCRIPTIONS["Telemetry"],
                content=self.page_telemetry,
                scrollable=False,
            ),
            "Monitor": PageShell(
                "Mission Monitor",
                PAGE_DESCRIPTIONS["Monitor"],
                content=self.page_monitor,
                # The monitor owns a dockable workspace with internal geometry
                # management; wrapping it in the shell's scroll/max-width column
                # would fight the dock layout.
                scrollable=False,
            ),
            "Data": PageShell(
                "Data & Files",
                PAGE_DESCRIPTIONS["Data"],
                content=self.page_data,
            ),
            "BatchPropagation": PageShell(
                "Batch Propagation",
                PAGE_DESCRIPTIONS["BatchPropagation"],
                content=self.page_batch,
                scrollable=False,
            ),
            "FrozenSearch": PageShell(
                "Frozen Search",
                PAGE_DESCRIPTIONS["FrozenSearch"],
                content=self.page_frozen_search,
                scrollable=False,
            ),
        }
        for key, _, _ in NAV_PAGES:
            self.stack_pages.addWidget(self.page_shells[key])

        content_layout.addWidget(self.stack_pages, 1)
        self.main_splitter.addWidget(content_container)

        # ---------------------------------------------------------------------
        # C. Log Panel
        # ---------------------------------------------------------------------
        self.log_panel = self._build_log_panel()
        self.log_panel.setMinimumHeight(LOG_COLLAPSED_HEIGHT)
        self.log_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.main_splitter.addWidget(self.log_panel)

        # Initial Splitter Sizes — content dominant (~68%), console ~32%.
        self.main_splitter.setHandleWidth(8)   # wide enough to grab reliably
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)  # prevent log from disappearing
        self.main_splitter.setStretchFactor(0, 88)
        self.main_splitter.setStretchFactor(1, 12)
        self.log_panel.set_collapsed(True)

        # Build Menu & Status
        self._build_menubar()
        self._build_statusbar()
        self._build_global_shortcuts()

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

        a_load = m_file.addAction(shortcuts.spec("file.load_profile").label)
        a_load.setShortcut(shortcuts.primary_key("file.load_profile"))
        a_load.triggered.connect(self._action_load_session)

        a_save = m_file.addAction(shortcuts.spec("file.save_profile").label)
        a_save.setShortcut(shortcuts.primary_key("file.save_profile"))
        a_save.triggered.connect(self._action_save_session)

        m_file.addSeparator()

        a_open_dir = m_file.addAction(shortcuts.spec("file.open_results").label)
        a_open_dir.setShortcut(shortcuts.primary_key("file.open_results"))
        a_open_dir.triggered.connect(self._action_open_out_dir)

        m_file.addSeparator()
        a_exit = m_file.addAction(shortcuts.spec("file.exit").label)
        a_exit.setShortcut(shortcuts.primary_key("file.exit"))
        a_exit.triggered.connect(self.close)

        # ANALYSIS MENU
        m_run = mb.addMenu("&Analysis")

        a_run = m_run.addAction(shortcuts.spec("analysis.run").label)
        a_run.setShortcut(shortcuts.primary_key("analysis.run"))
        a_run.triggered.connect(self._start_preflight_validation)

        a_stop = m_run.addAction(shortcuts.spec("analysis.stop").label)
        a_stop.setShortcut(shortcuts.primary_key("analysis.stop"))
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

        a_log = m_view.addAction(shortcuts.spec("view.toggle_log").label)
        a_log.setShortcuts(
            [QtGui.QKeySequence(k) for k in shortcuts.keys("view.toggle_log")]
        )
        a_log.triggered.connect(self._toggle_log_collapsed)

        a_clear = m_view.addAction(shortcuts.spec("view.clear_log").label)
        a_clear.setShortcut(shortcuts.primary_key("view.clear_log"))
        a_clear.triggered.connect(self._clear_log)

        m_view.addSeparator()

        a_density = m_view.addAction(shortcuts.spec("view.compact_density").label)
        a_density.setCheckable(True)
        a_density.setChecked(self._density == "compact")
        a_density.setShortcut(shortcuts.primary_key("view.compact_density"))
        a_density.toggled.connect(self._toggle_density)

        a_reduce_motion = m_view.addAction("Reduce Motion")
        a_reduce_motion.setCheckable(True)
        a_reduce_motion.setChecked(prefers_reduced_motion())
        a_reduce_motion.toggled.connect(self._toggle_reduce_motion)

    def _build_global_shortcuts(self) -> None:
        """Keyboard shortcuts for fast, mouse-free workspace navigation.

        Ctrl+1..Ctrl+9 jump directly to a workspace page (in nav order); Ctrl+Shift+F
        focuses the execution-console search field. Run/Stop/Save/Open already have
        menu shortcuts (F5 / Shift+F5 / Ctrl+S / Ctrl+O).
        """
        page_keys = shortcuts.keys("nav.page")
        self._page_shortcuts: list[QtGui.QShortcut] = []
        # zip truncates to the shorter side: at most nine pages get a chord.
        for (key, _label, _icon), seq in zip(NAV_PAGES, page_keys, strict=False):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.WindowShortcut)
            sc.activated.connect(lambda k=key: self._switch_page(k))
            self._page_shortcuts.append(sc)

        self._console_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(shortcuts.primary_key("console.focus_search")), self
        )
        self._console_shortcut.setContext(QtCore.Qt.WindowShortcut)
        self._console_shortcut.activated.connect(self._focus_console_search)

    def _focus_console_search(self) -> None:
        """Move keyboard focus to the execution-console search field."""
        if hasattr(self, "log_panel"):
            self.log_panel.focus_search()

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
            pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(THEME['fg_inverse']))
            pal.setColor(QtGui.QPalette.Link, QtGui.QColor(THEME['fg_link']))
            app.setPalette(pal)

        # 2. Build the global QSS from the Lunar Graphite palette.
        #    The large stylesheet now lives in lunaris.ui.theme.stylesheet
        #    so app.py stays an orchestration layer, not a design-token dump.
        #    Chevron images for the spin/combo steppers are rasterized here
        #    (needs a running QApplication) and passed to the binding-neutral
        #    QSS builder as plain paths.
        arrows = stepper_arrow_icons()
        self.setStyleSheet(
            build_app_stylesheet(
                THEME,
                LOG_COLORS,
                getattr(self, "_density", "comfortable"),
                spin_up_icon=arrows.get("up"),
                spin_down_icon=arrows.get("down"),
            )
        )

    # -------------------------------------------------------------------------
    # UI density (comfortable / compact)
    # -------------------------------------------------------------------------
    @staticmethod
    def _density_settings() -> QtCore.QSettings:
        return QtCore.QSettings("Lunaris", "MissionStudio")

    def _load_density_pref(self) -> str:
        value = str(self._density_settings().value("ui/density", "comfortable") or "comfortable")
        return "compact" if value.lower() == "compact" else "comfortable"

    def _toggle_density(self, compact: bool) -> None:
        """Switch between comfortable and compact density and persist the choice."""
        self._density = "compact" if compact else "comfortable"
        self._density_settings().setValue("ui/density", self._density)
        self._apply_theme()

    def _toggle_reduce_motion(self, enabled: bool) -> None:
        """Persist the reduced-motion preference (read live by progress widgets)."""
        self._density_settings().setValue("ui/reduce_motion", bool(enabled))

    # =========================================================================
    # 19. PAGE BUILDERS: ORBIT PAGE (PAGE 1)
    # =========================================================================

    def _build_page_orbit(self) -> QtWidgets.QWidget:
        """
        Page 1: Orbit Configuration.
        Delegated to ui_parts.orbit_config_page.OrbitPage
        """
        from lunaris.ui.pages.orbit_config_page import OrbitPage  # local import to avoid circulars
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
        from lunaris.ui.pages.force_models_page import (
            ForceModelsPage,  # local import to avoid circulars
        )

        # IMPORTANT: pass shared config objects so dialogs mutate the same instances
        self.page_forces = ForceModelsPage(
            gravity_cfg=self.gravity_cfg,
            albedo_cfg=self.albedo_cfg,
            thermal_cfg=self.thermal_cfg,
        )
        return self.page_forces


    # =========================================================================
    # 21. PAGE BUILDERS: PROPAGATION CONFIGURATION (PAGE 3)
    # =========================================================================

    def _build_page_propagation(self) -> QtWidgets.QWidget:
        from lunaris.ui.pages.mission_propagation_page import MissionPropagationPage
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
        from lunaris.ui.pages.live_telemetry_page import TelemetryPage
        self.page_telemetry = TelemetryPage()
        return self.page_telemetry

    def _build_page_monitor(self) -> QtWidgets.QWidget:
        """Mission Monitor: dockable live/replay observation workspace."""
        self.monitor_controller = MonitorController(self)
        page = MonitorPage(self.monitor_controller)
        self._restore_monitor_layout(page)
        return page

    def _monitor_layout_path(self) -> Path:
        from lunaris.ui.monitor.persistence import MONITOR_LAYOUT_FILENAME

        return self.app_data_dir / MONITOR_LAYOUT_FILENAME

    def _restore_monitor_layout(self, page: QtWidgets.QWidget) -> None:
        """Apply the saved monitor layout; a broken file is quarantined and
        the default preset opens instead (startup can never fail here)."""
        try:
            from lunaris.ui.monitor.persistence import load_layout_or_quarantine

            layout = load_layout_or_quarantine(
                self._monitor_layout_path(),
                log_warning=lambda msg: self._log_message(
                    f"[Monitor] {msg}", severity="warning"
                ),
            )
            if layout is not None:
                page.restore_layout(layout)
        except Exception:
            # Layout restoration is strictly best-effort.
            pass

    def _save_monitor_layout(self) -> None:
        try:
            from lunaris.ui.monitor.persistence import save_layout

            page = getattr(self, "page_monitor", None)
            if page is not None:
                save_layout(self._monitor_layout_path(), page.capture_layout())
        except Exception:
            pass


    # =========================================================================
    # 24. PAGE BUILDERS: DATA & FILES (PAGE 6)
    # =========================================================================
    def _build_page_data(self) -> QtWidgets.QWidget:
        from lunaris.ui.pages.data_files_page import DataFilesState, DataPage

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
    # 24b. PAGE BUILDERS: BATCH PROPAGATION (PAGE 7)
    # =========================================================================

    def _build_page_batch(self) -> QtWidgets.QWidget:
        """Page 7: Batch propagation analysis - configuration + live metrics."""
        from lunaris.ui.pages.batch_propagation_page import BatchPropagationPage
        self.page_batch = BatchPropagationPage(parent=self)
        self.page_batch.run_requested.connect(self._on_batch_run_requested)
        return self.page_batch

    def _build_page_frozen_search(self) -> QtWidgets.QWidget:
        """Page 8: Frozen-orbit staged search launcher."""
        from lunaris.ui.pages.frozen_search_page import FrozenSearchPage
        self.page_frozen_search = FrozenSearchPage(parent=self)
        return self.page_frozen_search

    def _on_batch_run_requested(self) -> None:
        """Slot: user clicked 'Run Batch' on the batch page."""
        if self.batch_process is not None:
            try:
                state = self.batch_process.state()
                if state != QtCore.QProcess.NotRunning:
                    self._log_message("[BATCH] A run is already in progress.", severity="warning")
                    return
            except RuntimeError:
                self.batch_process = None

        batch_data = self.page_batch.get_data()

        try:
            cmd = build_batch_command(
                python_executable=sys.executable,
                batch_runner_path=self.batch_runner_script_path,
                orbit=self.page_orbit.get_data(),
                forces=self.page_forces.get_data(),
                propagation=self.page_propagation.to_dict(),
                batch_data=batch_data,
                data_files=self.page_data.get_state(),
                gravity_cfg=self.gravity_cfg,
                solver_cfg=self.solver_cfg,
                spacecraft_cfg=self.spacecraft_cfg,
                log_warning=lambda m: self._log_message(m, severity="warning"),
            )
        except Exception as exc:
            self._log_message(f"[BATCH][Error] Failed to build command: {exc}", severity="error")
            self.page_batch.on_run_finished(1, "", None, None)
            return

        self._log_separator()
        self._log_message("[BATCH] Starting batch propagation run…", severity="system")

        # A stale backend chip from the previous run must not be readable as
        # this run's provenance.
        if getattr(self, "lbl_backend_status", None) is not None:
            self.lbl_backend_status.hide()

        # A fresh batch subprocess supersedes the previous batch observability
        # in the Mission Monitor (its own store/run state is untouched).
        with contextlib.suppress(Exception):
            self.monitor_controller.begin_batch_run()

        self.batch_process = QtCore.QProcess(self)
        self.batch_process.readyReadStandardOutput.connect(self._on_batch_stdout)
        self.batch_process.readyReadStandardError.connect(self._on_batch_stderr)
        self.batch_process.finished.connect(self._on_batch_finished)

        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.batch_process.setProcessEnvironment(env)

        self._batch_stdout_buf = ""
        self._batch_metrics: dict = {}
        self._batch_output_path: str = batch_data.get("output_path", "")

        self.batch_process.start(cmd[0], cmd[1:])
        if not self.batch_process.waitForStarted(2000):
            self._log_message("[BATCH][Error] Failed to start batch process.", severity="error")
            self.page_batch.on_run_finished(1, "", None, None)
            self.batch_process = None

    def _on_batch_stdout(self) -> None:
        """
        Stream stdout from the batch subprocess to the page progress log.

        Batch runs now emit two kinds of structured control lines:
        ``[BATCH_PROGRESS]`` for live progress payloads and ``[BATCH_METRICS]`` for
        the final summary blob.  These are consumed directly by the page rather
        than dumped into the human-readable log stream.
        """
        if self.batch_process is None:
            return
        try:
            raw = bytes(self.batch_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            return

        self._batch_stdout_buf += raw
        while "\n" in self._batch_stdout_buf:
            line, self._batch_stdout_buf = self._batch_stdout_buf.split("\n", 1)
            line = line.rstrip()
            if not line:
                continue

            if line.startswith("[BATCH_PROGRESS]"):
                try:
                    payload = json.loads(line[len("[BATCH_PROGRESS]"):].strip())
                except Exception:
                    self._log_message("[BATCH] Ignored malformed progress payload.", severity="warning")
                else:
                    if hasattr(self, "page_batch"):
                        self.page_batch.update_progress_payload(payload)
                    # Mirror into the Mission Monitor's batch observability.
                    with contextlib.suppress(Exception):
                        self.monitor_controller.set_batch_progress(payload)
                continue

            if line.startswith("[BATCH_METRICS]"):
                try:
                    payload = line[len("[BATCH_METRICS]"):].strip()
                    self._batch_metrics = json.loads(payload)
                except Exception:
                    self._log_message("[BATCH] Ignored malformed metrics payload.", severity="warning")
                else:
                    with contextlib.suppress(Exception):
                        self.monitor_controller.set_batch_metrics(self._batch_metrics)
                continue

            self._log_message(line, severity="info", source="batch")

            # Forward to the page's mini-log + progress bar
            if hasattr(self, "page_batch"):
                self.page_batch.update_progress(line)

    def _on_batch_stderr(self) -> None:
        """Route batch stderr to the main log as warnings."""
        if self.batch_process is None:
            return
        try:
            raw = bytes(self.batch_process.readAllStandardError()).decode("utf-8", errors="replace")
        except Exception:
            return
        for line in raw.splitlines():
            if line.strip():
                self._log_message(f"[BATCH] {line.strip()}", severity="warning")

    def _on_batch_finished(self, exit_code: int, _exit_status) -> None:
        """Handle batch subprocess completion and update the page."""
        metrics = getattr(self, "_batch_metrics", {}) or {}
        output_path = metrics.get("output_path", getattr(self, "_batch_output_path", ""))

        if exit_code == 0:
            wt = metrics.get("wall_time_s")
            wt_str = f"{wt:.1f}s" if isinstance(wt, int | float) else "?"
            self._log_message(
                f"[BATCH] Run complete — {metrics.get('n_impacts', '?')}/{metrics.get('n_samples', '?')} "
                f"impacts  wall={wt_str}",
                severity="success",
            )
        else:
            self._log_message(f"[BATCH] Run failed (exit code {exit_code}).", severity="error")

        if hasattr(self, "page_batch"):
            self.page_batch.on_run_finished(
                exit_code=exit_code,
                output_path=str(output_path),
                report_path=None,
                metrics=metrics if exit_code == 0 else None,
            )

        if exit_code == 0:
            self._update_backend_chip(metrics)

        try:
            if self.batch_process is not None:
                self.batch_process.deleteLater()
        except Exception:
            pass
        self.batch_process = None

    def _update_backend_chip(self, metrics: dict) -> None:
        """
        Surface the resolved run backend in the header chip row.

        Reads the same run metadata keys the Batch metrics panel uses
        (``actual_batch_backend`` / ``requested_batch_backend`` /
        ``fallback_reason``); when no backend was reported the chip stays
        hidden instead of inventing a value.
        """
        chip = getattr(self, "lbl_backend_status", None)
        if chip is None:
            return
        actual = str(
            metrics.get("actual_batch_backend") or metrics.get("backend") or ""
        ).strip()
        if not actual:
            chip.hide()
            return
        requested = str(metrics.get("requested_batch_backend") or "").strip()
        fell_back = bool(requested) and requested.lower() != actual.lower()
        if fell_back:
            reason = str(
                metrics.get("fallback_reason") or metrics.get("backend_note") or ""
            ).strip()
            chip.setText(f"Backend {actual} (requested {requested})")
            chip.setToolTip(
                reason or f"Requested {requested} but the run executed on {actual}."
            )
            chip.setProperty("kind", "warning")
        else:
            chip.setText(f"Backend {actual}")
            chip.setToolTip("Backend that executed the last completed run.")
            chip.setProperty("kind", "")
        chip.style().unpolish(chip)
        chip.style().polish(chip)
        chip.show()

    # =========================================================================
    # 25. LOG PANEL BUILDER
    # =========================================================================

    def _build_log_panel(self) -> QtWidgets.QWidget:
        """Construct the buffered Execution Console (see widgets.log_panel)."""
        panel = ExecutionConsoleDock(output_dir_provider=self._current_output_dir)
        panel.setMinimumHeight(EXPANDED_MIN_LOG_HEIGHT)
        panel.collapsed_changed.connect(self._on_log_collapsed_changed)
        # The header button routes through the animated host toggle so every
        # collapse path (button, menu, shortcut) shares the same drawer motion.
        panel.set_toggle_handler(self._toggle_log_collapsed)
        return panel

    def _current_output_dir(self) -> str:
        """Best-effort current mission output directory (used by the Save button)."""
        try:
            return self.page_output.get_state().output_dir
        except Exception:
            return ""

    def _apply_default_log_splitter_sizes(self, *, top_ratio: float = 0.68) -> None:
        """
        Rebalance the main vertical splitter using the live window geometry.

        Without an explicit size pass, Qt honors child size hints from the page
        stack, which makes the lower console feel stuck and hard to drag. This
        gives the splitter a practical starting ratio once the window has a
        real size. Collapsed terminal state stays compact and is not expanded by
        this bootstrap pass.
        """
        if self.is_log_collapsed or self.log_panel.is_collapsed:
            self.main_splitter.setSizes([max(1, self.main_splitter.height()), LOG_COLLAPSED_HEIGHT])
            return

        total = sum(max(0, size) for size in self.main_splitter.sizes())
        if total <= 0:
            total = max(self.main_splitter.height(), 480)

        min_top = 240
        min_bottom = EXPANDED_MIN_LOG_HEIGHT
        if total < (min_top + min_bottom):
            min_top = max(120, int(total * 0.55))
            min_bottom = max(80, total - min_top)

        top_size = max(min_top, int(total * float(top_ratio)))
        bottom_size = max(min_bottom, total - top_size)
        if (top_size + bottom_size) > total:
            top_size = max(min_top, total - min_bottom)
            bottom_size = max(min_bottom, total - top_size)

        self.main_splitter.setSizes([top_size, bottom_size])

    def _log_drawer_animations_enabled(self) -> bool:
        """Console drawer motion is skipped for reduced motion and pre-show layout."""
        return not prefers_reduced_motion() and self.isVisible()

    def _animate_log_splitter(self, target_bottom: int, on_finished=None) -> None:
        """Slide the console band of the main splitter to *target_bottom* px.

        One interruptible animation at a time: starting a new slide silently
        drops the previous one (including its pending ``on_finished`` commit),
        so a mid-flight toggle simply changes direction instead of queueing
        conflicting state changes.
        """
        previous = getattr(self, "_log_splitter_anim", None)
        if previous is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                previous.finished.disconnect()
            previous.stop()

        sizes = self.main_splitter.sizes()
        if len(sizes) != 2:
            self.main_splitter.setSizes([max(1, self.height()), target_bottom])
            if on_finished is not None:
                on_finished()
            return

        total = sizes[0] + sizes[1]
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(float(sizes[1]))
        anim.setEndValue(float(target_bottom))
        anim.setDuration(DESIGN_TOKENS.motion.duration_standard_ms)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.valueChanged.connect(
            lambda value: self.main_splitter.setSizes(
                [max(1, total - int(value)), int(value)]
            )
        )
        if on_finished is not None:
            anim.finished.connect(on_finished)
        self._log_splitter_anim = anim
        anim.start()

    def _expanded_log_target(self) -> int:
        """Bottom-band height an expanded console should land on."""
        sizes = getattr(self, "_log_expanded_sizes", None)
        if sizes and len(sizes) == 2 and sizes[1] >= EXPANDED_MIN_LOG_HEIGHT:
            return sizes[1]
        total = sum(max(0, size) for size in self.main_splitter.sizes())
        if total <= 0:
            total = max(self.main_splitter.height(), 480)
        return max(EXPANDED_MIN_LOG_HEIGHT, int(total * 0.32))

    def _on_log_collapsed_changed(self, collapsed: bool) -> None:
        """React to the console collapse toggle by resizing the splitter.

        The last genuinely-expanded geometry is remembered so expanding restores
        it; a window resize while collapsed cannot clobber that memory because we
        only capture sizes that look expanded. Expanding always lands on a usable
        height, falling back to a sensible default when no memory exists.

        Expanding animates the drawer open (the widget has already shown its
        body, so the content is progressively revealed); the collapse direction
        is animated *before* the state commit in :meth:`_toggle_log_collapsed`,
        so the collapsed branch here just asserts the final geometry.
        """
        self.is_log_collapsed = bool(collapsed)
        if collapsed:
            sizes = self.main_splitter.sizes()
            # Only remember a genuinely expanded layout (ignore not-yet-laid-out
            # or already-collapsed geometry), so the saved height survives resizes.
            if len(sizes) == 2 and sizes[1] > LOG_COLLAPSED_HEIGHT + 4:
                self._log_expanded_sizes = sizes
            self.log_panel.setMinimumHeight(LOG_COLLAPSED_HEIGHT)
            # A large top value is clamped by Qt to the available space, leaving
            # the console at exactly the collapsed header height (never zero).
            self.main_splitter.setSizes([max(1, self.height()), LOG_COLLAPSED_HEIGHT])
        elif self._log_drawer_animations_enabled():
            # Keep the minimum height low while the drawer slides open (a 210 px
            # minimum applied up front would snap the splitter past the motion),
            # then restore it so manual splitter drags cannot crush the console.
            target = self._expanded_log_target()
            self._animate_log_splitter(
                target,
                on_finished=lambda: self.log_panel.setMinimumHeight(
                    EXPANDED_MIN_LOG_HEIGHT
                ),
            )
        else:
            self.log_panel.setMinimumHeight(EXPANDED_MIN_LOG_HEIGHT)
            sizes = getattr(self, "_log_expanded_sizes", None)
            if sizes and len(sizes) == 2 and sizes[1] >= EXPANDED_MIN_LOG_HEIGHT:
                self.main_splitter.setSizes(sizes)
            else:
                self._apply_default_log_splitter_sizes()

    # =========================================================================
    # 26. EXECUTION CONSOLE LOGGING (delegates to ExecutionLogPanel)
    # =========================================================================

    def _log_message(self, text: str, is_error: bool = False, severity: str | None = None, source: str = ""):
        """
        Route a message to the Execution Console.

        Kept as a thin MainWindow delegate so the many existing call sites keep
        working. Severity defaults to auto-detection from the message text.
        """
        if text is None:
            return
        sev = severity or ("error" if is_error else "auto")
        self.log_panel.append(text, severity=sev, source=source)

    def _log_separator(self):
        """Add a visual separator to the console."""
        self.log_panel.append_separator()

    def _clear_log(self, _checked: bool = False):
        """Clear the console."""
        self.log_panel.clear()
        self._log_message("[UI] Console cleared.", severity="system")

    def _copy_log_to_clipboard(self, _checked: bool = False):
        """Copy the console contents to the clipboard."""
        self.log_panel.copy_to_clipboard()

    # =========================================================================
    # 27. ASYNCHRONOUS PRE-FLIGHT VALIDATION
    # =========================================================================

    def _collect_preflight_data(self) -> dict[str, Any]:
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
        with contextlib.suppress(Exception):
            proc.close()
        with contextlib.suppress(Exception):
            proc.deleteLater()
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

        # Field-level gate: surface every invalid propagation field inline and
        # move focus to the first one instead of failing later in preflight.
        page = getattr(self, "page_propagation", None)
        if page is not None and hasattr(page, "validate_inputs") and not page.validate_inputs():
            self._switch_page("Propagation")
            self._log_message(
                "[Error] Fix the highlighted propagation fields before running.",
                severity="error",
            )
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
            # Reduced motion: avoid the marquee (indeterminate) animation.
            if prefers_reduced_motion():
                self.progress_bar.setRange(0, 1000)
                self.progress_bar.setTextVisible(True)
                self.progress_bar.setFormat("Validating…")
            else:
                self.progress_bar.setTextVisible(False)
                self.progress_bar.setRange(0, 0)
                self.progress_bar.setFormat("")
            self.progress_bar.setValue(0)
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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Keep shell chrome compact without changing the navigation model."""
        super().resizeEvent(event)
        if hasattr(self, "nav_list"):
            nav_width = (
                DESIGN_TOKENS.layout.nav_compact_width
                if event.size().width() < 1160
                else DESIGN_TOKENS.layout.nav_width
            )
            self.nav_list.setFixedWidth(nav_width)
        # The header re-budgets itself from its own Resize/LayoutRequest events
        # (see eventFilter); doing it here would read the header's pre-resize
        # width and shed chips that actually fit.

    def eventFilter(  # noqa: N802 - Qt override
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        """Re-budget the header whenever its contents change size or visibility."""
        if watched is getattr(self, "_header_frame", None) and event.type() in (
            # LayoutRequest: a child was shown/hidden or changed its hint (the
            # run chrome appearing). Resize: the header itself got wider or
            # narrower. Both change the width budget, and neither is reliably
            # reported by MainWindow.resizeEvent — that fires before the header
            # child has been resized, so reading its width there is stale.
            QtCore.QEvent.LayoutRequest,
            QtCore.QEvent.Resize,
        ):
            self._apply_header_breakpoint()
        return super().eventFilter(watched, event)

    def _header_optional_chips(self) -> list[QtWidgets.QWidget]:
        """Header chips in shed order (first to go, first in the list).

        These three are informational and each is shown in full on the page
        that owns it, so dropping them costs the user nothing they cannot get
        elsewhere. Everything not in this list — Run/Stop, the progress
        cluster, the page badge — is load-bearing and always stays.
        """
        ordered = (self.lbl_frame_status, self.lbl_output_status,
                   self.lbl_gravity_status)
        return [c for c in ordered if c is not None]

    def _apply_header_breakpoint(self) -> None:
        """Shed low-priority header chips before anything is forced to clip.

        The header is a priority list, not a row of equals. Every item except
        the app title is horizontally fixed, so when a run starts and the
        header gains ~450px of progress/stop chrome there is no give: Qt
        squeezes the fixed widgets below their size hint and their labels clip
        (the page badge, being centred, loses characters from *both* ends).

        The budget is measured rather than guessed from a magic width, because
        the required width depends on the run state, the density setting and
        the user's font — a single hard-coded breakpoint is wrong in most of
        those combinations. Chips are restored in reverse order as soon as the
        space comes back, so widening the window is not a one-way door.
        """
        if not hasattr(self, "_header_frame") or self._in_header_breakpoint:
            return
        available = self._header_frame.width()
        if available <= 0:
            # Freshly built: no geometry yet. Keep the roomy layout; the first
            # real resizeEvent decides.
            return

        self._in_header_breakpoint = True
        try:
            self._pin_header_minimums()
            chips = self._header_optional_chips()
            # Show everything, then drop chips one at a time until the fixed
            # items fit. The title is elided (size policy Ignored) and so is
            # not part of the budget: it absorbs whatever is left over.
            for chip in chips:
                chip.setVisible(True)
            for chip in chips:
                if self._header_required_width() <= available:
                    break
                chip.setVisible(False)
        finally:
            self._in_header_breakpoint = False

    def _header_compressible(self) -> tuple[QtWidgets.QWidget, ...]:
        """Header items allowed to shrink instead of forcing a shed.

        Both are :class:`ElidedLabel`: they report a minimum width of 0 and
        degrade to an ellipsis, so they are the header's slack. Everything else
        either renders at its full size or is hidden outright.
        """
        return (self.lbl_app_title, self.lbl_progress)

    def _pin_header_minimums(self) -> None:
        """Make the load-bearing header items genuinely incompressible.

        ``QSizePolicy.Fixed`` is not enough on its own: when a layout cannot
        fit its items it shrinks them toward ``minimumSizeHint``, and a
        QPushButton's minimum sits well below its label width — it is willing
        to clip itself. Pinning each item's minimum width to its size hint
        removes that willingness, so the deficit lands entirely on the two
        elided labels (title, progress text), which report a minimum of 0 and
        degrade to "…" as designed.

        Recomputed on every pass rather than pinned once at construction: the
        hints move with the UI font and the density setting.
        """
        layout = self._header_frame.layout()
        compressible = self._header_compressible()
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget is None or widget in compressible or widget.isHidden():
                continue
            widget.setMinimumWidth(widget.sizeHint().width())

    def _header_required_width(self) -> int:
        """Width the header's non-elastic items need to render without clipping.

        Deliberately summed from ``sizeHint`` rather than read from the
        layout's ``minimumSize``: QPushButton reports a minimum well below its
        size hint (it is willing to clip its own label), so the layout happily
        certifies a header as "fitting" at a width where every button is
        chopped. The size hint is the width at which the text actually renders.

        ``isHidden`` rather than ``isVisible``: the latter is False whenever an
        ancestor is not yet shown, which is exactly the case during
        construction and in offscreen rendering, and would make this return
        near-zero and never shed anything.
        """
        layout = self._header_frame.layout()
        compressible = self._header_compressible()
        margins = layout.contentsMargins()
        total = margins.left() + margins.right()
        gaps = max(0, layout.count() - 1)
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                # The elided labels are the header's slack, not part of the
                # budget: counting their full text width would make the header
                # shed chips to buy room for text that was going to elide
                # anyway.
                if widget.isHidden() or widget in compressible:
                    continue
                total += widget.sizeHint().width()
            elif item.spacerItem() is not None:
                # The stretch collapses to nothing first; fixed spacers do not.
                total += item.spacerItem().sizeHint().width()
        return total + layout.spacing() * gaps

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
        """Collapse or expand the Execution Console.

        Expanding commits the widget state immediately (its body must be
        visible for the drawer-open animation to reveal anything); collapsing
        animates the drawer shut with the content still visible and commits the
        collapsed state only when the slide finishes. A toggle during the
        collapse slide reverses direction without committing (the drop of the
        pending ``on_finished`` in ``_animate_log_splitter`` cancels the
        commit), keeping the animation interruptible.
        """
        panel = self.log_panel
        if panel.is_collapsed or not self._log_drawer_animations_enabled():
            panel.toggle_collapsed()
            return

        anim = getattr(self, "_log_splitter_anim", None)
        closing = (
            anim is not None
            and anim.state() == QtCore.QAbstractAnimation.Running
            and int(anim.endValue()) == LOG_COLLAPSED_HEIGHT
        )
        if closing:
            # Mid-collapse reversal: the collapsed state was never committed,
            # so just slide back open and restore the minimum height after.
            self._animate_log_splitter(
                self._expanded_log_target(),
                on_finished=lambda: panel.setMinimumHeight(EXPANDED_MIN_LOG_HEIGHT),
            )
            return

        sizes = self.main_splitter.sizes()
        if len(sizes) == 2 and sizes[1] > LOG_COLLAPSED_HEIGHT + 4:
            self._log_expanded_sizes = sizes
        panel.setMinimumHeight(LOG_COLLAPSED_HEIGHT)
        self._animate_log_splitter(
            LOG_COLLAPSED_HEIGHT, on_finished=panel.toggle_collapsed
        )

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
                with contextlib.suppress(Exception):
                    prop_ui.sync_solver_widgets_from_config()

    def _on_spacecraft_settings(self, _checked: bool = False):
        """Open Spacecraft Properties Dialog."""
        dlg = SpacecraftBusDialog(self, self.spacecraft_cfg)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._log_message("[UI] Spacecraft properties updated.", severity="system")

    def _update_gravity_status(self):
        """Delegate gravity summary refresh to the dedicated force-model page."""
        with contextlib.suppress(Exception):
            self.page_forces._update_gravity_summary_ui()


    # =========================================================================
    # 31. COMMAND BUILDING & PROCESS MANAGEMENT
    # =========================================================================
    def _build_command(self) -> list[str]:
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
            albedo_cfg=self.albedo_cfg,
            thermal_cfg=self.thermal_cfg,
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

        # Prep UI. A new run inserts a labeled separator instead of wiping the
        # console, so the pre-flight output and prior context stay visible. The
        # user can still clear explicitly (toolbar / Ctrl+K).
        self._set_run_state("running")
        self.log_panel.append_run_separator()

        # Telemetry reset (TelemetryPage owns the plot)
        with contextlib.suppress(Exception):
            self.page_telemetry.telemetry_multiplot.clear_all()

        self.progress_bar.setValue(0)
        self._stdout_assembler.clear()
        self._stderr_assembler.clear()
        self._run_wall_t0 = time.time()
        self._last_telem_t_s = None
        self._progress_is_determinate = False
        self.progress_bar.show()
        if prefers_reduced_motion():
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
        else:
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
                            self.sim_state.total_duration = dur_val * DAY_S
                        else:
                            self.sim_state.total_duration = dur_val * 3600.0
                except ValueError:
                    self.sim_state.total_duration = 0.0

        # Reset the Mission Monitor for the new run (fresh store + sequence
        # space; the run's own [TELEMETRY_META] line pins the real run id).
        with contextlib.suppress(Exception):
            self.monitor_controller.begin_live_run(
                expected_duration_s=self.sim_state.total_duration or None
            )

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

        # A new run supersedes the previous run's diagnostics panel; reset it
        # to the explicit empty state so stale numbers can never be read as
        # belonging to the run that is about to start.
        with contextlib.suppress(Exception):
            self.page_output.set_run_diagnostics(None)

        # Start process. The run separator above already provides the visual
        # break, so we go straight to the launch line.
        self._log_message("[System] Launching mission analysis...", severity="system")

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
            if isinstance(v, int | float):
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

        def first_float(*keys: str) -> float | None:
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
            with contextlib.suppress(Exception):
                self._log_message("[WARN] " + reason, severity="warning")

    def _handle_stdout(self):
        """Assemble complete stdout lines and route them (telemetry or log)."""
        if self.process is None:
            return

        chunk = self.process.readAllStandardOutput().data().decode("utf-8", errors="ignore")
        if not chunk:
            return

        for line in self._stdout_assembler.push(chunk):
            self._consume_stdout_line(line)

    def _consume_stdout_line(self, line: str) -> None:
        """Route one complete stdout line to telemetry parsing or the log.

        A line that *looks* like a telemetry payload is parsed in isolation: any
        malformed payload is surfaced as a single warning rather than being
        allowed to break the log stream or crash the UI.
        """
        clean_line = line.strip()
        if not clean_line:
            return

        # Mission Monitor structured telemetry ([TELEMETRY] / [TELEMETRY_META]
        # v1 lines and legacy bare-JSON telemetry). The classifier returns None
        # for every ordinary log line, so the existing routing below stays
        # authoritative for anything that is not telemetry.
        monitor_msg = None
        controller = getattr(self, "monitor_controller", None)
        if controller is not None:
            try:
                monitor_msg = controller.feed_line(clean_line)
            except Exception:
                monitor_msg = None
        if isinstance(monitor_msg, SampleMessage):
            # Keep the legacy telemetry surfaces (Live Telemetry plots,
            # collision watchdog, progress bar) fed from the same sample.
            self._apply_telemetry_sample(monitor_msg.sample)
            return
        if isinstance(monitor_msg, MetaMessage):
            self._log_message(
                "[Monitor] Run provenance received — see Mission Monitor.",
                severity="system",
            )
            return
        if isinstance(monitor_msg, ProtocolProblem):
            # Fail-closed: the payload claimed to be telemetry but could not be
            # decoded; surface it in the console instead of guessing.
            self._log_message(clean_line, severity="warning")
            return

        # Structured engine diagnostics emitted once at the end of a run.
        # Routed to the Results page panel; a malformed payload falls through
        # to the plain log so nothing is silently dropped.
        if clean_line.startswith("[DIAG]"):
            try:
                payload = json.loads(clean_line[len("[DIAG]"):].strip())
                if isinstance(payload, dict):
                    if controller is not None:
                        with contextlib.suppress(Exception):
                            controller.set_run_diagnostics(payload)
                    self.page_output.set_run_diagnostics(payload)
                    wall = payload.get("wall_time_s")
                    if isinstance(wall, int | float):
                        self._log_message(
                            f"[Run] Engine diagnostics received (wall {wall:.2f} s) — see Results & Export.",
                            severity="system",
                        )
                    return
            except Exception:
                pass

        telemetry_line = clean_line
        for prefix in ("JSON_TELEM:", "TELEMETRY:"):
            if clean_line.startswith(prefix):
                telemetry_line = clean_line[len(prefix):].strip()
                break

        time_keys = ('"t"', '"t_s"', '"time_s"', "'t'", "'t_s'", "'time_s'")
        if telemetry_line.startswith("{") and any(key in telemetry_line for key in time_keys):
            try:
                if self._handle_telemetry_line(telemetry_line):
                    return
            except Exception:
                self._log_message(clean_line, severity="warning")
                return

        self._log_message(clean_line)

    def _handle_telemetry_line(self, clean_line: str) -> bool:
        """Parse and apply a telemetry line. Returns True if it was telemetry.

        Returning False means the line only resembled telemetry but did not
        parse into a dict, so the caller should log it as an ordinary message.
        """
        try:
            telem = json.loads(clean_line)
        except json.JSONDecodeError:
            try:
                telem = ast.literal_eval(clean_line)
            except (ValueError, SyntaxError):
                telem = None

        if not isinstance(telem, dict):
            return False

        # Mirror python-repr / prefixed legacy payloads into the Mission
        # Monitor store (JSON legacy lines are consumed upstream by the
        # monitor classifier and never reach this method).
        monitor = getattr(self, "monitor_controller", None)
        if monitor is not None:
            with contextlib.suppress(Exception):
                monitor.feed_legacy_mapping(telem)

        # Pass to the telemetry plot (TelemetryPage owns the plot).
        with contextlib.suppress(Exception):
            self.page_telemetry.telemetry_multiplot.add_datapoint(telem)

        # impact monitoring
        self._check_collision(telem)

        # Update progress based on time
        t_s = None
        for t_key in ["t_s", "time_s", "t_sec", "t"]:
            if t_key in telem:
                t_s = float(telem[t_key])
                # Handle unit conversion if 't' is used without explicit unit
                if t_key == "t":
                    unit = str(telem.get("t_unit", "")).strip().lower()
                    if unit.startswith("h"):
                        t_s *= 3600.0
                    elif unit.startswith("d"):
                        t_s *= DAY_S
                break

        self._update_run_progress(t_s)

        # Telemetry payloads are not echoed to the log to prevent spam.
        return True

    def _apply_telemetry_sample(self, sample) -> None:
        """Mirror a structured monitor sample into the legacy telemetry surfaces.

        Live Telemetry plots, the collision watchdog and the progress bar all
        predate the v1 contract and consume km-based dicts; deriving that dict
        from the typed sample keeps a single producer emission feeding both the
        Mission Monitor and the legacy pipeline.
        """
        legacy: dict[str, float] = {"t_s": float(sample.simulation_time_s)}
        if sample.altitude_m is not None:
            legacy["alt_km"] = sample.altitude_m / 1000.0
        if sample.speed_m_s is not None:
            legacy["v_km_s"] = sample.speed_m_s / 1000.0
        ecc = sample.orbital_elements.get("ecc")
        if ecc is not None:
            legacy["ecc"] = float(ecc)
        if sample.surface_radius_m is not None:
            legacy["surface_r_km"] = sample.surface_radius_m / 1000.0
        if sample.terrain_clearance_m is not None:
            legacy["terrain_clearance_km"] = sample.terrain_clearance_m / 1000.0
            if sample.altitude_m is not None:
                # surface_alt = altitude - clearance (both relative to R_ref).
                legacy["surface_alt_km"] = (
                    sample.altitude_m - sample.terrain_clearance_m
                ) / 1000.0

        with contextlib.suppress(Exception):
            self.page_telemetry.telemetry_multiplot.add_datapoint(legacy)
        self._check_collision(legacy)
        self._update_run_progress(sample.simulation_time_s)

    def _update_run_progress(self, t_s: float | None) -> None:
        """Progress bar / ETA update from the latest telemetry time."""
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
                t_days = float(t_s) / DAY_S
                T_days = total / DAY_S
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

    @staticmethod
    def _classify_stderr_line(line: str) -> str:
        """Classify a single stderr line as 'warning' or 'error'."""
        lowered = line.lower()
        is_warning = (
            "warning" in lowered
            and "traceback" not in lowered
            and "[fatal]" not in lowered
            and "fatal error" not in lowered
        )
        return "warning" if is_warning else "error"

    def _handle_stderr(self):
        """Assemble complete stderr lines with warning/error classification.

        Per-stream buffering means a chunk boundary never splits a message
        mid-line during high-volume output; only complete lines are logged.
        """
        if self.process is None:
            return
        data = self.process.readAllStandardError().data().decode('utf-8', errors='ignore')
        if not data:
            return

        for line in self._stderr_assembler.push(data):
            if line.strip():
                self._log_message(line, severity=self._classify_stderr_line(line))

    def _flush_stream_buffers(self):
        """Flush trailing partial stdout/stderr lines left without a newline.

        Called when the process exits so the final unterminated line of each
        stream is not lost. The stdout tail still passes through telemetry
        parsing in case the run ended mid-payload.
        """
        for line in self._stderr_assembler.flush():
            if line.strip():
                self._log_message(line, severity=self._classify_stderr_line(line))
        for line in self._stdout_assembler.flush():
            self._consume_stdout_line(line)

    def _on_process_finished(self, exit_code, exit_status):
        """
        Handle process completion and always return the UI to a restart-ready state.

        Even failed runs should leave the window immediately runnable again. The
        log retains the error details, so there is little value in keeping the
        header latched in a pseudo-running state after the backend has exited.
        """

        # Emit any trailing partial stderr line the stream buffering held back.
        self._flush_stream_buffers()

        # Close out the Mission Monitor run (after the flush, so a trailing
        # partial telemetry line still lands in the store first).
        with contextlib.suppress(Exception):
            self.monitor_controller.finish_live_run(exit_code=int(exit_code))

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
        self.btn_stop.setVisible(is_running)

        self._update_run_visuals(state)

    # =========================================================================
    # 33. STATE MANAGEMENT & SERIALIZATION
    # =========================================================================

    def _collect_preset_dict(self) -> dict[str, Any]:
        """Collect all modular page/config state into a serializable snapshot."""

        snapshot = collect_session_snapshot(
            orbit_page=self.page_orbit,
            propagation_page=self.page_propagation,
            force_page=self.page_forces,
            output_page=self.page_output,
            data_page=self.page_data,
            gravity_cfg=self.gravity_cfg,
            albedo_cfg=self.albedo_cfg,
            thermal_cfg=self.thermal_cfg,
            solver_cfg=self.solver_cfg,
            spacecraft_cfg=self.spacecraft_cfg,
            app_version=APP_VERSION,
            batch_page=getattr(self, "page_batch", None),
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
        with contextlib.suppress(Exception):
            splitter_sizes = list(self.main_splitter.sizes())

        telemetry_plot_type = ""
        telemetry_time_unit = ""
        try:
            mp = getattr(getattr(self, "page_telemetry", None), "telemetry_multiplot", None)
            if mp is not None:
                if hasattr(mp, "current_plot_name"):
                    telemetry_plot_type = mp.current_plot_name()
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

        batch_active_tab = 0
        try:
            batch_tabs = getattr(getattr(self, "page_batch", None), "tabs", None)
            if batch_tabs:
                batch_active_tab = batch_tabs.currentIndex()
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
            batch_active_tab=batch_active_tab,
        )
        return snapshot

    def _apply_preset_dict(self, data: dict[str, Any]):
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
                thermal_cfg=self.thermal_cfg,
                solver_cfg=self.solver_cfg,
                spacecraft_cfg=self.spacecraft_cfg,
                project_root=PROJECT_ROOT,
                log_warning=lambda msg: self._log_message(msg, severity="warning"),
                batch_page=getattr(self, "page_batch", None),
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
                self._visual_state_restored = True
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
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
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
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            self._apply_preset_dict(data)
            self._log_message(f"[UI] Session loaded from: {Path(path).name}", severity="success")
        except Exception as e:
            self._log_message(f"[Error] Failed to load session: {e}", severity="error")
            QtWidgets.QMessageBox.warning(
                self, "Load Error",
                f"Failed to load session file:\n\n{e!s}"
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
                f"Failed to save session file:\n\n{e!s}"
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
        """Attempt to load the last session, migrating a legacy profile if needed."""
        # Prefer the current session file; fall back once to the pre-rename
        # ("ST-LRPS Studio") location so existing users keep their saved profile.
        source_path = self.session_path
        migrating_legacy = False
        if not source_path.exists():
            legacy = getattr(self, "_legacy_session_path", None)
            if legacy is not None and legacy.exists():
                source_path = legacy
                migrating_legacy = True
            else:
                return

        try:
            with open(source_path, encoding='utf-8') as f:
                data = json.load(f)
            self._apply_preset_dict(data)
            if migrating_legacy:
                # Copy the migrated profile forward so future launches use the
                # new location; never delete the legacy file.
                try:
                    with open(self.session_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                self._log_message(
                    "[UI] Migrated previous session from legacy app-data folder.",
                    severity="system",
                )
            else:
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
            # Spacecraft Bus now lives on the Force Models page (it scales the
            # non-gravitational forces), so the dialog opens from there.
            self.page_forces.spacecraft_settings_requested.connect(self._on_spacecraft_settings)
        except Exception:
            pass

        # Let the window reach a real geometry before forcing the initial
        # splitter ratio; this prevents page size hints from trapping the log
        # panel at an awkwardly small height.
        if not self._visual_state_restored:
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
                with contextlib.suppress(Exception):
                    self.page_forces._update_gravity_summary_ui()

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

        # The console's status chip reads as execution status to users, so it
        # must not say "Idle" while the header is showing run progress.
        if hasattr(self, "log_panel"):
            self.log_panel.set_run_status(state)

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
        Refresh the two header context chips (gravity model, output directory)
        from current UI state. Called from _ui_tick every 250ms. Execution
        readiness (preflight/run) is no longer summarised here — it is reported
        by the header run dot + progress while a run is active.
        """
        try:
            # Gravity backend chip
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
            # Output directory chip (shortened tail, full path in tooltip)
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
                    self.lbl_output_status.setToolTip(
                        out_dir or "Output directory — click to choose"
                    )
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
                with contextlib.suppress(Exception):
                    self._log_message(message, severity="warning")
            if notify_on_failure:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Session Save Warning",
                    "Autosave failed; the current session was not saved.\n\n"
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

        # Stop any batch subprocess
        if self.batch_process is not None:
            try:
                if self.batch_process.state() != QtCore.QProcess.NotRunning:
                    self.batch_process.kill()
                    self.batch_process.waitForFinished(1000)
            except Exception:
                pass

        # Stop any background analysis work owned by the batch propagation page.
        try:
            if hasattr(self, "page_batch"):
                self.page_batch.shutdown()
        except Exception:
            pass

        # Stop any frozen-search subprocess owned by the dedicated page.
        try:
            if hasattr(self, "page_frozen_search"):
                self.page_frozen_search.shutdown()
        except Exception:
            pass

        # Persist window geometry so the workspace reopens where the user left it.
        with contextlib.suppress(Exception):
            self._density_settings().setValue("ui/geometry", self.saveGeometry())

        # Persist the Mission Monitor dashboards (tabs, widgets, dock layout).
        self._save_monitor_layout()

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

    def _build_command_preview_safe(self) -> tuple[str, str]:
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
