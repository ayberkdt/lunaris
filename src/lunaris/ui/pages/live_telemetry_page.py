# ST_LRPS/ui_parts/live_telemetry_page.py
"""
Live Telemetry Page (UI)

This module defines the **TelemetryPage** used by the Lunaris Mission Studio UI to display
runtime telemetry in a compact, mission-control style view.

What it provides
- A Qt widget page (TelemetryPage) that can be embedded into the main stacked UI.
- A multi-plot telemetry panel (TelemetryMultiPlot) that subscribes to incoming telemetry
  dictionaries and visualizes key signals (e.g., time, altitude, velocity, eccentricity).
- A small API surface designed for the main process runner:
  - feed datapoints as dicts (typically parsed from stdout lines like `JSON_TELEM:{...}`)
  - update progress / status indicators as needed

Expected telemetry format
Telemetry is expected as a `dict` with a time field and any number of scalar fields.
Recommended keys (examples):
- "t_s"      : float   simulation time in seconds
- "alt_km"   : float   altitude in km
- "v_km_s"   : float   speed in km/s
- "ecc"      : float   orbital eccentricity
- "lat_deg"  : float   latitude in deg (optional)
- "lon_deg"  : float   longitude in deg (optional)

Notes
- Plotting uses `pyqtgraph` if available. If `pyqtgraph` is not installed, the page can
  still be constructed (plots degrade gracefully), so the rest of the UI remains usable.
- The module intentionally contains no process-launching logic; it only visualizes data.
  Process execution and stdout parsing should be handled in the main UI controller.

Author / Project
ST_LRPS Core - UI components.
"""


# =============================================================================
# 0.                                    IMPORTS
# =============================================================================
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from PySide6 import QtCore, QtWidgets

from lunaris.common.constants import DAY_S

# Modern Icon Library
try:
    import qtawesome as qta  # noqa: F401  # availability probe for HAS_QTAWESOME
    HAS_QTAWESOME = True
except ImportError:
    HAS_QTAWESOME = False
    print("[Warning] qtawesome not installed. Icons will be disabled.")

# Live Plotting & 3D Visualization
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl  # noqa: F401  # availability probe for HAS_OPENGL
    HAS_PYQTGRAPH = True
    HAS_OPENGL = True
    # Enable antialiasing globally
    pg.setConfigOptions(antialias=True)
except ImportError as e:
    HAS_PYQTGRAPH = False
    HAS_OPENGL = False
    print(f"[Warning] PyQtGraph/OpenGL not installed. Advanced visualization disabled: {e}")


try:
    from lunaris.ui.components.primitives import EmptyState, SegmentedControl
    from lunaris.ui.core.ui_commons import (
        THEME,
        NoWheelComboBox,
        NoWheelDoubleSpinBox,
        get_icon,
    )
    from lunaris.ui.theme.tokens import DESIGN_TOKENS
except ImportError:
        # Only handle the "ran as a script" case; don't mask real import errors.
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  When executed directly, relative imports like '.constants' fail.", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m lunaris.ui.pages.live_telemetry_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2) from None
    raise


import logging as _logging

_log = _logging.getLogger(__name__)
_warned_keys: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """Log each unique warning key only once to avoid log spam."""
    if key not in _warned_keys:
        _warned_keys.add(key)
        _log.warning("[telemetry] %s", msg)


# =============================================================================
# 1.                        ADVANCED TELEMETRY PLOT
# =============================================================================

class MultiTelemetryPlot(QtWidgets.QWidget):
    """
    Enhanced telemetry widget with multiple plot types.
    - Altitude vs Time
    - Velocity vs Time
    - Eccentricity vs Time
    - Ground Track (Latitude vs Longitude)
    """

    # Emitted after each buffer flush with the latest scalar sample so an owner
    # (the page's KPI strip) can show live mission-control readouts. An empty
    # dict means "reset to placeholders" (e.g. after Clear All).
    sample_updated = QtCore.Signal(dict)

    # Canonical plot names, in segmented-control order. Session persistence
    # stores/restores by name, so these strings are a stable contract.
    _PLOT_NAMES = (
        "Altitude vs Time",
        "Velocity vs Time",
        "Eccentricity vs Time",
        "Ground Track",
    )

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data storage (ring buffers)
        self.max_points = 5000
        self.time_data = deque(maxlen=self.max_points)
        self.alt_data = deque(maxlen=self.max_points)
        self.vel_data = deque(maxlen=self.max_points)
        self.ecc_data = deque(maxlen=self.max_points)
        self.lat_data = deque(maxlen=self.max_points)
        self.lon_data = deque(maxlen=self.max_points)

        # Buffers for incoming data
        self._buffer_lock = QtCore.QMutex()
        self._time_buffer = []
        self._alt_buffer = []
        self._vel_buffer = []
        self._ecc_buffer = []
        self._lat_buffer = []
        self._lon_buffer = []


        # Time axis presentation (for Live Telemetry)
        # Many engines emit absolute seconds (e.g., ET or Unix); we display relative time by default.
        self._t0_raw = None
        self._last_time_unit = "s"

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_PYQTGRAPH:
            # Designed empty-state when the plotting backend is unavailable.
            layout.addWidget(
                EmptyState(
                    "Live telemetry unavailable",
                    "Install PyQtGraph to enable real-time telemetry plots.",
                )
            )
            return

        # Toolbar.
        #
        # A single control bar keeps the vertical space for the plot itself: the
        # plot selector is a segmented control (4 options, scannable), the time
        # unit + relative toggle stay inline (adjusted often), and the expert
        # scaling controls (mouse-zoom axes + manual Y range) fold into a "Scale"
        # popover so they are one click away without crowding the bar.
        ctrl_h = DESIGN_TOKENS.controls.compact_height

        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("telemetryToolbar")
        bar = QtWidgets.QHBoxLayout(toolbar)
        bar.setContentsMargins(12, 8, 12, 8)
        bar.setSpacing(DESIGN_TOKENS.spacing.md)

        self.plot_segment = SegmentedControl(
            ["Altitude", "Velocity", "Eccentricity", "Ground Track"]
        )
        self.plot_segment.setAccessibleName("Plot type")
        self.plot_segment.current_changed.connect(self._on_plot_segment_changed)
        bar.addWidget(self.plot_segment)

        bar.addStretch(1)

        bar.addWidget(QtWidgets.QLabel("Time:"))
        self.time_axis_combo = NoWheelComboBox()
        self.time_axis_combo.addItems(["Auto", "s", "min", "h", "d"])
        self.time_axis_combo.setFixedHeight(ctrl_h)
        self.time_axis_combo.setMinimumWidth(70)
        self.time_axis_combo.setAccessibleName("Time axis unit")
        self.time_axis_combo.currentTextChanged.connect(self._on_axis_controls_changed)
        bar.addWidget(self.time_axis_combo)

        self.chk_time_relative = QtWidgets.QCheckBox("T+")
        self.chk_time_relative.setChecked(True)
        self.chk_time_relative.setToolTip("Display time relative to first received sample.")
        self.chk_time_relative.toggled.connect(self._on_axis_controls_changed)
        bar.addWidget(self.chk_time_relative)

        self.btn_scale = self._build_scale_popover(ctrl_h)
        bar.addWidget(self.btn_scale)

        self.btn_clear = QtWidgets.QPushButton("Clear All")
        self.btn_clear.setFixedHeight(ctrl_h)
        self.btn_clear.clicked.connect(self.clear_all)
        bar.addWidget(self.btn_clear)

        layout.addWidget(toolbar)

        # Stacked widget for different plots. A generous minimum height keeps the
        # plot the dominant element on the page instead of a thin strip squeezed
        # between the KPI row and the console.
        self.plot_stack = QtWidgets.QStackedWidget()
        self.plot_stack.setMinimumHeight(320)
        layout.addWidget(self.plot_stack, 1)

        # Create individual plot widgets
        self._create_altitude_plot()
        self._create_velocity_plot()
        self._create_eccentricity_plot()
        self._create_ground_track_plot()

        # Empty-state overlay: an idle telemetry plot with empty axes reads as
        # "broken / unfinished". This centered overlay tells the operator the
        # view is waiting for a run; it is hidden the moment samples arrive and
        # shown again on Clear All.
        self._empty_overlay = self._build_empty_overlay(self.plot_stack)
        self._has_telemetry = False
        self._position_empty_overlay()

        # Timer for buffered updates (30 FPS)
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setInterval(33)  # ~30 Hz
        self.update_timer.timeout.connect(self._flush_buffer)
        self.update_timer.start()

        # Set initial plot
        self._switch_plot("Altitude vs Time")

    # ------------------------------------------------------------------
    # Toolbar helpers
    # ------------------------------------------------------------------
    def _on_plot_segment_changed(self, index: int) -> None:
        """Map a segmented-control index to the corresponding plot."""
        names = self._PLOT_NAMES
        self._switch_plot(names[index] if 0 <= index < len(names) else names[0])

    def current_plot_name(self) -> str:
        """Return the canonical name of the currently selected plot (for session save)."""
        segment = getattr(self, "plot_segment", None)
        if segment is None:
            return ""
        idx = segment.current_index()
        return self._PLOT_NAMES[idx] if 0 <= idx < len(self._PLOT_NAMES) else ""

    def set_plot_by_name(self, name: str) -> None:
        """Select a plot by its canonical name (for session restore). No-op if unknown."""
        if name in self._PLOT_NAMES:
            self._switch_plot(name)

    def _build_scale_popover(self, ctrl_h: int) -> QtWidgets.QToolButton:
        """Build the "Scale" popover holding the expert axis-scaling controls.

        Mouse-zoom axes and the manual Y range are rarely touched during a run,
        so they live behind one button instead of occupying a second toolbar row.
        The controls keep their original attribute names and signal wiring; only
        their location changes.
        """
        button = QtWidgets.QToolButton()
        button.setObjectName("telemetryScaleButton")
        button.setText("Scale")
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        button.setIcon(get_icon("fa6s.sliders", THEME["fg_soft"]))
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.setFixedHeight(ctrl_h)
        button.setToolTip("Axis zoom and manual Y range")

        menu = QtWidgets.QMenu(button)
        panel = QtWidgets.QWidget()
        panel.setObjectName("telemetryScalePanel")
        grid = QtWidgets.QGridLayout(panel)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(DESIGN_TOKENS.spacing.md)
        grid.setVerticalSpacing(DESIGN_TOKENS.spacing.sm)

        # Mouse-zoom axes.
        grid.addWidget(QtWidgets.QLabel("Mouse zoom"), 0, 0)
        self.chk_mouse_x = QtWidgets.QCheckBox("X")
        self.chk_mouse_y = QtWidgets.QCheckBox("Y")
        self.chk_mouse_x.setChecked(False)  # keep time axis stable by default
        self.chk_mouse_y.setChecked(True)
        self.chk_mouse_x.toggled.connect(self._apply_mouse_zoom_settings)
        self.chk_mouse_y.toggled.connect(self._apply_mouse_zoom_settings)
        zoom_row = QtWidgets.QHBoxLayout()
        zoom_row.setContentsMargins(0, 0, 0, 0)
        zoom_row.addWidget(self.chk_mouse_x)
        zoom_row.addWidget(self.chk_mouse_y)
        zoom_row.addStretch(1)
        zoom_wrap = QtWidgets.QWidget()
        zoom_wrap.setLayout(zoom_row)
        grid.addWidget(zoom_wrap, 0, 1, 1, 3)

        # Auto vs manual Y range.
        self.chk_auto_y = QtWidgets.QCheckBox("Auto Y-range")
        self.chk_auto_y.setChecked(True)
        self.chk_auto_y.toggled.connect(self._on_y_mode_changed)
        grid.addWidget(self.chk_auto_y, 1, 0, 1, 4)

        grid.addWidget(QtWidgets.QLabel("Y min"), 2, 0)
        self.ed_ymin = QtWidgets.QLineEdit()
        self.ed_ymin.setPlaceholderText("min")
        self.ed_ymin.setFixedHeight(ctrl_h)
        self.ed_ymin.setAccessibleName("Y-axis minimum")
        grid.addWidget(self.ed_ymin, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Y max"), 2, 2)
        self.ed_ymax = QtWidgets.QLineEdit()
        self.ed_ymax.setPlaceholderText("max")
        self.ed_ymax.setFixedHeight(ctrl_h)
        self.ed_ymax.setAccessibleName("Y-axis maximum")
        grid.addWidget(self.ed_ymax, 2, 3)

        grid.addWidget(QtWidgets.QLabel("Margin"), 3, 0)
        self.spin_y_pad = NoWheelDoubleSpinBox()
        self.spin_y_pad.setSuffix("%")
        self.spin_y_pad.setDecimals(1)
        self.spin_y_pad.setRange(0.0, 50.0)
        self.spin_y_pad.setSingleStep(1.0)
        self.spin_y_pad.setValue(5.0)
        self.spin_y_pad.setFixedHeight(ctrl_h)
        self.spin_y_pad.setAccessibleName("Y-axis padding percent")
        self.spin_y_pad.valueChanged.connect(self._on_axis_controls_changed)
        grid.addWidget(self.spin_y_pad, 3, 1)

        self.btn_y_apply = QtWidgets.QPushButton("Apply")
        self.btn_y_apply.setFixedHeight(ctrl_h)
        self.btn_y_apply.clicked.connect(self._apply_manual_y_range)
        self.btn_y_fit = QtWidgets.QPushButton("Fit to data")
        self.btn_y_fit.setFixedHeight(ctrl_h)
        self.btn_y_fit.clicked.connect(self._fit_y_range_to_data)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self.btn_y_apply)
        btn_row.addWidget(self.btn_y_fit)
        btn_row.addStretch(1)
        btn_wrap = QtWidgets.QWidget()
        btn_wrap.setLayout(btn_row)
        grid.addWidget(btn_wrap, 4, 0, 1, 4)

        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)
        button.setMenu(menu)
        return button

    # ------------------------------------------------------------------
    # Empty-state overlay
    # ------------------------------------------------------------------
    def _build_empty_overlay(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Build the 'waiting for telemetry' overlay shown over an idle plot."""
        overlay = QtWidgets.QWidget(parent)
        overlay.setObjectName("telemetryEmpty")
        overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        box = QtWidgets.QVBoxLayout(overlay)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.setAlignment(QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("Waiting for telemetry")
        title.setObjectName("telemetryEmptyTitle")
        title.setAlignment(QtCore.Qt.AlignCenter)
        sub = QtWidgets.QLabel("No active run.")
        sub.setObjectName("telemetryEmptyText")
        sub.setAlignment(QtCore.Qt.AlignCenter)
        sub.setWordWrap(True)

        box.addStretch(1)
        box.addWidget(title, 0, QtCore.Qt.AlignCenter)
        box.addWidget(sub, 0, QtCore.Qt.AlignCenter)
        box.addStretch(1)
        overlay.raise_()
        return overlay

    def _position_empty_overlay(self) -> None:
        """Keep the overlay covering the plot stack as the page resizes."""
        overlay = getattr(self, "_empty_overlay", None)
        if overlay is not None and getattr(self, "plot_stack", None) is not None:
            overlay.setGeometry(self.plot_stack.rect())
            if overlay.isVisible():
                overlay.raise_()

    def _set_empty_visible(self, visible: bool) -> None:
        """Show/hide the empty-state overlay and keep it correctly stacked."""
        overlay = getattr(self, "_empty_overlay", None)
        if overlay is None:
            return
        overlay.setVisible(visible)
        if visible:
            self._position_empty_overlay()
            overlay.raise_()

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._position_empty_overlay()

    def _create_altitude_plot(self):
        """Altitude vs Time plot (Lunar Graphite styling)."""
        widget = pg.PlotWidget()
        widget.setBackground(THEME['bg_space'])
        widget.setTitle("Orbital Altitude", color=THEME['fg_main'], size='14pt', bold=True)
        widget.setLabel('left', 'Altitude [km]', color=THEME['fg_soft'], size='11pt')
        widget.setLabel('bottom', 'Time [s]', color=THEME['fg_soft'], size='11pt')
        widget.showGrid(x=True, y=True, alpha=0.3)

        # Enhanced axis styling
        axis_pen = pg.mkPen(color=THEME['fg_muted'], width=1.5)
        widget.getAxis('left').setPen(axis_pen)
        widget.getAxis('left').setTextPen(THEME['fg_soft'])
        widget.getAxis('bottom').setPen(axis_pen)
        widget.getAxis('bottom').setTextPen(THEME['fg_soft'])


        # Store widgets / viewbox for axis controls
        self.alt_plot_widget = widget
        self.alt_viewbox = widget.getViewBox()
        try:
            # Keep time axis stable by default; allow vertical zoom only
            self.alt_viewbox.setMouseEnabled(x=False, y=True)
        except Exception as exc:
            _warn_once("alt_viewbox_mouse", str(exc))

        # Enhanced plot line with glow effect
        self.alt_curve = widget.plot(
            pen=pg.mkPen(color=THEME['accent'], width=2.5),
            name='Altitude',
            shadowPen=pg.mkPen(color=THEME['accent'], width=4, alpha=0.3)
        )

        self.plot_stack.addWidget(widget)

    def _create_velocity_plot(self):
        """Velocity vs Time plot (Lunar Graphite styling)."""
        widget = pg.PlotWidget()
        widget.setBackground(THEME['bg_space'])
        widget.setTitle("Orbital Velocity", color=THEME['fg_main'], size='14pt', bold=True)
        widget.setLabel('left', 'Velocity [km/s]', color=THEME['fg_soft'], size='11pt')
        widget.setLabel('bottom', 'Time [s]', color=THEME['fg_soft'], size='11pt')
        widget.showGrid(x=True, y=True, alpha=0.3)

        axis_pen = pg.mkPen(color=THEME['fg_muted'], width=1.5)
        widget.getAxis('left').setPen(axis_pen)
        widget.getAxis('left').setTextPen(THEME['fg_soft'])
        widget.getAxis('bottom').setPen(axis_pen)
        widget.getAxis('bottom').setTextPen(THEME['fg_soft'])


        # Store widgets / viewbox for axis controls
        self.vel_plot_widget = widget
        self.vel_viewbox = widget.getViewBox()
        try:
            self.vel_viewbox.setMouseEnabled(x=False, y=True)
        except Exception as exc:
            _warn_once("vel_viewbox_mouse", str(exc))

        self.vel_curve = widget.plot(
            pen=pg.mkPen(color=THEME['secondary'], width=2.5),
            name='Velocity',
            shadowPen=pg.mkPen(color=THEME['secondary'], width=4, alpha=0.3)
        )

        self.plot_stack.addWidget(widget)

    def _create_eccentricity_plot(self):
        """Eccentricity vs Time plot (Lunar Graphite styling)."""
        widget = pg.PlotWidget()
        widget.setBackground(THEME['bg_space'])
        widget.setTitle("Orbit Eccentricity", color=THEME['fg_main'], size='14pt', bold=True)
        widget.setLabel('left', 'Eccentricity', color=THEME['fg_soft'], size='11pt')
        widget.setLabel('bottom', 'Time [s]', color=THEME['fg_soft'], size='11pt')
        widget.showGrid(x=True, y=True, alpha=0.3)

        axis_pen = pg.mkPen(color=THEME['fg_muted'], width=1.5)
        widget.getAxis('left').setPen(axis_pen)
        widget.getAxis('left').setTextPen(THEME['fg_soft'])
        widget.getAxis('bottom').setPen(axis_pen)
        widget.getAxis('bottom').setTextPen(THEME['fg_soft'])


        # Store widgets / viewbox for axis controls
        self.ecc_plot_widget = widget
        self.ecc_viewbox = widget.getViewBox()
        try:
            self.ecc_viewbox.setMouseEnabled(x=False, y=True)
        except Exception as exc:
            _warn_once("ecc_viewbox_mouse", str(exc))

        self.ecc_curve = widget.plot(
            pen=pg.mkPen(color=THEME['warning'], width=2.5),
            name='Eccentricity',
            shadowPen=pg.mkPen(color=THEME['warning'], width=4, alpha=0.3)
        )

        self.plot_stack.addWidget(widget)

    def _create_ground_track_plot(self):
        """Ground Track (Latitude vs Longitude) plot (Lunar Graphite styling)."""
        widget = pg.PlotWidget()
        widget.setBackground(THEME['bg_space'])
        widget.setTitle("Ground Track", color=THEME['fg_main'], size='14pt', bold=True)
        widget.setLabel('left', 'Latitude [deg]', color=THEME['fg_soft'], size='11pt')
        widget.setLabel('bottom', 'Longitude [deg]', color=THEME['fg_soft'], size='11pt')
        widget.showGrid(x=True, y=True, alpha=0.3)

        # Set axis ranges for Moon
        widget.setXRange(-180, 180)
        widget.setYRange(-90, 90)

        axis_pen = pg.mkPen(color=THEME['info'], width=1.5)
        widget.getAxis('left').setPen(axis_pen)
        widget.getAxis('left').setTextPen(THEME['fg_soft'])
        widget.getAxis('bottom').setPen(axis_pen)
        widget.getAxis('bottom').setTextPen(THEME['fg_soft'])

        self.ground_track_curve = widget.plot(
            pen=pg.mkPen(color=THEME['info'], width=2.0, style=QtCore.Qt.DashLine),
            symbol='o',
            symbolSize=5,
            symbolBrush=THEME['info'],
            symbolPen=pg.mkPen(color=THEME['info'], width=1),
            name='Ground Track'
        )

        self.plot_stack.addWidget(widget)

    def _switch_plot(self, plot_name):
        """Switch between different plot types."""
        plot_map = {name: i for i, name in enumerate(self._PLOT_NAMES)}

        idx = plot_map.get(plot_name, 0)
        self.plot_stack.setCurrentIndex(idx)

        # Keep the segmented control in sync when the plot is switched
        # programmatically (e.g. initial state); emit=False avoids a signal loop.
        segment = getattr(self, "plot_segment", None)
        if segment is not None and segment.current_index() != idx:
            segment.set_current_index(idx)

        # Disable axis controls for Ground Track
        is_timeseries = idx in (0, 1, 2)
        for w in (getattr(self, "time_axis_combo", None),
                  getattr(self, "chk_time_relative", None),
                  getattr(self, "chk_mouse_x", None),
                  getattr(self, "chk_mouse_y", None),
                  getattr(self, "chk_auto_y", None),
                  getattr(self, "ed_ymin", None),
                  getattr(self, "ed_ymax", None),
                  getattr(self, "spin_y_pad", None),
                  getattr(self, "btn_y_apply", None),
                  getattr(self, "btn_y_fit", None),
                  getattr(self, "btn_scale", None)):
            if w is None:
                continue
            w.setEnabled(is_timeseries)

        if is_timeseries:
            self._apply_mouse_zoom_settings()
            self._on_y_mode_changed()
            # Force label refresh
            try:
                unit = str(self.time_axis_combo.currentText()).strip()
                if unit.lower() == "auto":
                    unit = getattr(self, "_last_time_unit", "s")
                self._set_time_axis_label(unit)
            except Exception as exc:
                _warn_once("time_axis_label", str(exc))

    # ------------------------------------------------------------------
    # Axis / scaling helpers (Live Telemetry)
    # ------------------------------------------------------------------
    def _get_plot_time_and_unit(self, t_raw: list) -> tuple[list, str]:
        """Return (time_values_for_plot, unit_label)."""
        if not t_raw:
            return [], "s"

        # Establish reference time (first sample) if using relative time
        if getattr(self, "chk_time_relative", None) is not None and self.chk_time_relative.isChecked():
            if self._t0_raw is None:
                self._t0_raw = float(t_raw[0])
            t0 = self._t0_raw
        else:
            t0 = 0.0

        # Work in seconds for unit selection
        t_sec = [float(x) - t0 for x in t_raw]
        if not t_sec:
            return [], "s"

        span = max(t_sec) - min(t_sec)

        # Choose unit
        choice = "Auto"
        if getattr(self, "time_axis_combo", None) is not None:
            choice = str(self.time_axis_combo.currentText()).strip()

        if choice.lower() == "auto":
            if span >= 5.0 * DAY_S:
                unit = "d"
            elif span >= 3.0 * 3600.0:
                unit = "h"
            elif span >= 5.0 * 60.0:
                unit = "min"
            else:
                unit = "s"
        else:
            unit = choice

        factor = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": DAY_S}.get(unit, 1.0)
        t_plot = [x / factor for x in t_sec]

        self._last_time_unit = unit
        return t_plot, unit

    def _set_time_axis_label(self, unit: str):
        """Update the X label on time-series plots."""
        label = f"Time [{unit}]"
        for w in (getattr(self, "alt_plot_widget", None),
                  getattr(self, "vel_plot_widget", None),
                  getattr(self, "ecc_plot_widget", None)):
            if w is None:
                continue
            try:
                w.setLabel('bottom', label, color=THEME['fg_soft'], size='11pt')
            except Exception as exc:
                _warn_once("axis_bottom_label", str(exc))

    def _current_timeseries_key(self) -> str | None:
        idx = int(self.plot_stack.currentIndex())
        if idx == 0:
            return "alt"
        if idx == 1:
            return "vel"
        if idx == 2:
            return "ecc"
        return None

    def _current_timeseries_values(self) -> list:
        key = self._current_timeseries_key()
        if key == "alt":
            return list(self.alt_data)
        if key == "vel":
            return list(self.vel_data)
        if key == "ecc":
            return list(self.ecc_data)
        return []

    def _current_viewbox(self):
        key = self._current_timeseries_key()
        if key == "alt":
            return getattr(self, "alt_viewbox", None)
        if key == "vel":
            return getattr(self, "vel_viewbox", None)
        if key == "ecc":
            return getattr(self, "ecc_viewbox", None)
        return None

    def _apply_mouse_zoom_settings(self):
        """Allow independent mouse zoom per-axis."""
        x_en = bool(getattr(self, "chk_mouse_x", None) and self.chk_mouse_x.isChecked())
        y_en = bool(getattr(self, "chk_mouse_y", None) and self.chk_mouse_y.isChecked())

        for vb in (getattr(self, "alt_viewbox", None),
                   getattr(self, "vel_viewbox", None),
                   getattr(self, "ecc_viewbox", None)):
            if vb is None:
                continue
            try:
                vb.setMouseEnabled(x=x_en, y=y_en)
            except Exception as exc:
                _warn_once("vb_mouse_enable", str(exc))

    def _on_y_mode_changed(self, _checked: bool = False):
        """Toggle auto/manual Y scaling."""
        auto = bool(getattr(self, "chk_auto_y", None) and self.chk_auto_y.isChecked())

        for w in (getattr(self, "ed_ymin", None), getattr(self, "ed_ymax", None),
                  getattr(self, "btn_y_apply", None)):
            if w is None:
                continue
            w.setEnabled(not auto)

        vb = self._current_viewbox()
        if vb is None:
            return

        try:
            vb.enableAutoRange(axis='y', enable=auto)
        except Exception as exc:
            _warn_once("vb_auto_range", str(exc))

        if auto:
            self._apply_live_auto_y_range()

    def _apply_manual_y_range(self):
        """Apply manual y-range for the currently selected time-series plot."""
        vb = self._current_viewbox()
        if vb is None:
            return

        try:
            y0 = float(self.ed_ymin.text().strip())
            y1 = float(self.ed_ymax.text().strip())
        except Exception:
            return

        if not (math.isfinite(y0) and math.isfinite(y1)) or y1 <= y0:
            return

        try:
            vb.enableAutoRange(axis='y', enable=False)
        except Exception as exc:
            _warn_once("vb_disable_auto_range", str(exc))
        try:
            vb.setYRange(y0, y1, padding=0.0)
        except Exception as exc:
            _warn_once("vb_set_y_range", str(exc))

    def _fit_y_range_to_data(self):
        """Fit Y range to the data of the current plot with configurable padding."""
        vb = self._current_viewbox()
        if vb is None:
            return

        y = np.asarray(self._current_timeseries_values(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if not (math.isfinite(y_min) and math.isfinite(y_max)):
            return
        if y_max <= y_min:
            # Flat line; widen a little
            eps = 1e-12 if abs(y_min) < 1e-6 else abs(y_min) * 0.01
            y_min -= eps
            y_max += eps

        try:
            pad_frac = float(self.spin_y_pad.value()) / 100.0 if getattr(self, "spin_y_pad", None) else 0.05
        except Exception:
            pad_frac = 0.05

        span = y_max - y_min
        y_min2 = y_min - span * pad_frac
        y_max2 = y_max + span * pad_frac

        # Update input boxes for visibility (also supports sci notation)
        try:
            self.ed_ymin.setText(f"{y_min2:.12g}")
            self.ed_ymax.setText(f"{y_max2:.12g}")
        except Exception as exc:
            _warn_once("fit_y_text_set", str(exc))

        try:
            vb.enableAutoRange(axis='y', enable=False)
        except Exception as exc:
            _warn_once("fit_y_disable_auto", str(exc))
        try:
            vb.setYRange(y_min2, y_max2, padding=0.0)
        except Exception as exc:
            _warn_once("fit_y_set_range", str(exc))

    def _apply_live_auto_y_range(self) -> None:
        """
        Keep the active time-series Y axis following incoming telemetry.

        The desktop operator expects the `Auto` toggle to remain live for the
        whole run.  PyQtGraph's built-in autorange can settle after the first
        fit depending on interaction state, so we explicitly recompute the
        visible Y envelope from the latest finite samples.
        """

        if not (getattr(self, "chk_auto_y", None) and self.chk_auto_y.isChecked()):
            return

        vb = self._current_viewbox()
        if vb is None:
            return

        y = np.asarray(self._current_timeseries_values(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if not (math.isfinite(y_min) and math.isfinite(y_max)):
            return

        if y_max <= y_min:
            eps = 1e-12 if abs(y_min) < 1e-6 else abs(y_min) * 0.01
            y_min -= eps
            y_max += eps

        try:
            pad_frac = float(self.spin_y_pad.value()) / 100.0 if getattr(self, "spin_y_pad", None) else 0.05
        except Exception:
            pad_frac = 0.05

        span = y_max - y_min
        y_min2 = y_min - span * pad_frac
        y_max2 = y_max + span * pad_frac

        try:
            self.ed_ymin.setText(f"{y_min2:.12g}")
            self.ed_ymax.setText(f"{y_max2:.12g}")
        except Exception as exc:
            _warn_once("live_y_text_set", str(exc))

        try:
            vb.enableAutoRange(axis='y', enable=False)
        except Exception as exc:
            _warn_once("live_y_disable_auto", str(exc))
        try:
            vb.setYRange(y_min2, y_max2, padding=0.0)
        except Exception as exc:
            _warn_once("live_y_set_range", str(exc))

    def _on_axis_controls_changed(self, *_args):
        """Apply time unit changes and Y padding immediately (without waiting for new telemetry)."""
        if not HAS_PYQTGRAPH:
            return

        t_raw = list(self.time_data)
        t_list, unit = self._get_plot_time_and_unit(t_raw)
        self._set_time_axis_label(unit)

        # Redraw time-series curves with the new X axis
        try:
            self.alt_curve.setData(t_list, list(self.alt_data))
        except Exception as exc:
            _warn_once("axis_redraw_alt", str(exc))
        try:
            self.vel_curve.setData(t_list, list(self.vel_data))
        except Exception as exc:
            _warn_once("axis_redraw_vel", str(exc))
        try:
            self.ecc_curve.setData(t_list, list(self.ecc_data))
        except Exception as exc:
            _warn_once("axis_redraw_ecc", str(exc))

        # If auto Y is enabled, re-fit with the new padding
        if getattr(self, "chk_auto_y", None) is not None and self.chk_auto_y.isChecked():
            self._apply_live_auto_y_range()


    def add_datapoint(self, telem_data: dict[str, Any]):
        """
        Add one telemetry sample in a strictly synchronized way.

        Contract
        --------
        - A point is only accepted if a valid time stamp can be parsed.
        - For each accepted time stamp, *all* series (alt/vel/ecc/lat/lon) receive
          exactly one value; missing values are stored as NaN.
        This prevents X/Y length mismatches during plotting.
        """
        if not HAS_PYQTGRAPH:
            return

        def _sf(x) -> float:
            try:
                if x is None:
                    return float("nan")
                return float(x)
            except (ValueError, TypeError):
                return float("nan")

        # --- Parse time first (mandatory) ---
        # NOTE: do NOT use `or` chaining here; valid timestamps like 0.0 are falsy.
        t_val = telem_data.get("t_s")
        if t_val is None:
            t_val = telem_data.get("t")
        if t_val is None:
            t_val = telem_data.get("time_s")
        if t_val is None:
            t_val = telem_data.get("time")
        if t_val is None:
            return

        try:
            t_s = float(t_val)
        except (ValueError, TypeError):
            return

        # Handle unit conversion if 't' is used without explicit unit
        if ("t" in telem_data) and ("t_s" not in telem_data) and ("time_s" not in telem_data):
            unit = str(telem_data.get("t_unit", "s")).lower()
            if unit.startswith("h"):
                t_s *= 3600.0
            elif unit.startswith("d"):
                t_s *= DAY_S

        # --- Extract remaining fields (optional, NaN if missing) ---
        alt_keys = ("alt_km", "altitude_km", "h_km", "alt")
        vel_keys = ("v_km_s", "velocity_km_s", "v")

        alt_val = float("nan")
        for k in alt_keys:
            if k in telem_data:
                alt_val = _sf(telem_data.get(k))
                break

        vel_val = float("nan")
        for k in vel_keys:
            if k in telem_data:
                vel_val = _sf(telem_data.get(k))
                break

        ecc_val = _sf(telem_data.get("ecc") if "ecc" in telem_data else telem_data.get("e"))

        lat_val = _sf(telem_data.get("lat_deg") if "lat_deg" in telem_data else telem_data.get("lat"))
        lon_val = _sf(telem_data.get("lon_deg") if "lon_deg" in telem_data else telem_data.get("lon"))

        # --- Thread-safe buffer append (all series stay aligned) ---
        self._buffer_lock.lock()
        try:
            self._time_buffer.append(t_s)
            self._alt_buffer.append(alt_val)
            self._vel_buffer.append(vel_val)
            self._ecc_buffer.append(ecc_val)
            self._lat_buffer.append(lat_val)
            self._lon_buffer.append(lon_val)
        finally:
            self._buffer_lock.unlock()

    def _flush_buffer(self):
        """Transfer buffered data to main storage and update plots (shape-safe)."""
        if not HAS_PYQTGRAPH:
            return

        # Get buffered data
        self._buffer_lock.lock()
        try:
            if not self._time_buffer:
                return

            # Copy buffers (do NOT assume they're aligned)
            time_chunk = list(self._time_buffer)
            alt_chunk = list(self._alt_buffer)
            vel_chunk = list(self._vel_buffer)
            ecc_chunk = list(self._ecc_buffer)
            lat_chunk = list(self._lat_buffer)
            lon_chunk = list(self._lon_buffer)

            # Clear buffers
            self._time_buffer.clear()
            self._alt_buffer.clear()
            self._vel_buffer.clear()
            self._ecc_buffer.clear()
            self._lat_buffer.clear()
            self._lon_buffer.clear()

        finally:
            self._buffer_lock.unlock()

        # Enforce equal chunk lengths (defensive; should already match)
        n = min(len(time_chunk), len(alt_chunk), len(vel_chunk), len(ecc_chunk), len(lat_chunk), len(lon_chunk))
        if n <= 0:
            return

        if not (len(time_chunk) == len(alt_chunk) == len(vel_chunk) == len(ecc_chunk) == len(lat_chunk) == len(lon_chunk)):
            # Drop trailing unmatched samples to keep internal state consistent.
            time_chunk = time_chunk[:n]
            alt_chunk = alt_chunk[:n]
            vel_chunk = vel_chunk[:n]
            ecc_chunk = ecc_chunk[:n]
            lat_chunk = lat_chunk[:n]
            lon_chunk = lon_chunk[:n]
            print(
                f"[Telemetry] Warning: buffer length mismatch; truncated to n={n} "
                f"(t={len(time_chunk)}, alt={len(alt_chunk)}, vel={len(vel_chunk)}, ecc={len(ecc_chunk)}, lat={len(lat_chunk)}, lon={len(lon_chunk)})"
            )

        # First real samples — retire the empty-state overlay.
        if not getattr(self, "_has_telemetry", False):
            self._has_telemetry = True
            self._set_empty_visible(False)

        # Append to main storage (aligned, includes NaNs)
        self.time_data.extend(time_chunk)
        self.alt_data.extend(alt_chunk)
        self.vel_data.extend(vel_chunk)
        self.ecc_data.extend(ecc_chunk)
        self.lat_data.extend(lat_chunk)
        self.lon_data.extend(lon_chunk)

        # Hard-align deques in case older runs left them mismatched
        def _align_to_time(series: deque, fill_nan: bool = True):
            lt = len(self.time_data)
            while len(series) > lt:
                series.popleft()
            while fill_nan and len(series) < lt:
                series.append(float("nan"))

        _align_to_time(self.alt_data, fill_nan=True)
        _align_to_time(self.vel_data, fill_nan=True)
        _align_to_time(self.ecc_data, fill_nan=True)
        _align_to_time(self.lat_data, fill_nan=True)
        _align_to_time(self.lon_data, fill_nan=True)

        # --- Update plots safely ---
        t_raw = list(self.time_data)
        t_list, unit = self._get_plot_time_and_unit(t_raw)
        self._set_time_axis_label(unit)

        try:
            self.alt_curve.setData(t_list, list(self.alt_data))
        except Exception:
            # UI redraw race / curve deleted; ignore to keep stream alive
            pass
        try:
            self.vel_curve.setData(t_list, list(self.vel_data))
        except Exception:
            # UI redraw race / curve deleted; ignore to keep stream alive
            pass
        try:
            self.ecc_curve.setData(t_list, list(self.ecc_data))
        except Exception:
            # UI redraw race / curve deleted; ignore to keep stream alive
            pass

        self._apply_live_auto_y_range()

        # Ground track: only plot finite lat/lon pairs
        lon_arr = np.asarray(self.lon_data, dtype=float)
        lat_arr = np.asarray(self.lat_data, dtype=float)
        mask = np.isfinite(lon_arr) & np.isfinite(lat_arr)
        if mask.any():
            try:
                self.ground_track_curve.setData(lon_arr[mask].tolist(), lat_arr[mask].tolist())
            except Exception:
                # UI redraw race / curve deleted; ignore to keep stream alive
                pass
        else:
            try:
                self.ground_track_curve.setData([], [])
            except Exception:
                # UI redraw race / curve deleted; ignore to keep stream alive
                pass

        # Publish the latest scalar sample for the KPI strip.
        if self.time_data:
            def _last(dq):
                return float(dq[-1]) if dq else float("nan")
            try:
                self.sample_updated.emit({
                    "t_s": _last(self.time_data),
                    "alt_km": _last(self.alt_data),
                    "v_km_s": _last(self.vel_data),
                    "ecc": _last(self.ecc_data),
                    "lat_deg": _last(self.lat_data),
                    "lon_deg": _last(self.lon_data),
                })
            except Exception:
                pass

    def clear_all(self, _checked: bool = False):
        """Clear all telemetry data."""
        self._buffer_lock.lock()
        try:
            self._time_buffer.clear()
            self._alt_buffer.clear()
            self._vel_buffer.clear()
            self._ecc_buffer.clear()
            self._lat_buffer.clear()
            self._lon_buffer.clear()
        finally:
            self._buffer_lock.unlock()

        self.time_data.clear()
        self.alt_data.clear()
        self.vel_data.clear()
        self.ecc_data.clear()
        self.lat_data.clear()
        self.lon_data.clear()

        # Reset relative time origin
        self._t0_raw = None

        if HAS_PYQTGRAPH:
            self.alt_curve.setData([], [])
            self.vel_curve.setData([], [])
            self.ecc_curve.setData([], [])
            self.ground_track_curve.setData([], [])

        # Back to the idle empty-state until the next run streams data.
        self._has_telemetry = False
        self._set_empty_visible(True)
        # Reset the KPI strip to placeholders.
        self.sample_updated.emit({})



# =============================================================================
# 2.                           TELEMETRY PAGE
# =============================================================================

class TelemetryPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._kpis: dict = {}
        self._build_ui()

    def _kpi_chip(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
        """A compact live readout cell (label + value) for the KPI strip."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("telemetryKpiCell")
        v = QtWidgets.QVBoxLayout(frame)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(2)
        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("telemetryKpiLabel")
        v.addWidget(title_lbl)
        value_lbl = QtWidgets.QLabel("--")
        value_lbl.setObjectName("telemetryKpiValue")
        v.addWidget(value_lbl)
        return frame, value_lbl

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.layout.page_gap)

        # Live KPI strip — current mission-control readouts. Replaces the
        # redundant "Real-time Mission Telemetry" header (the shell already
        # titles the page) and turns an empty single-plot view into a dashboard.
        kpi_frame = QtWidgets.QFrame()
        kpi_frame.setObjectName("telemetryKpiStrip")
        kpi_row = QtWidgets.QHBoxLayout(kpi_frame)
        kpi_row.setContentsMargins(0, 0, 0, 0)
        kpi_row.setSpacing(DESIGN_TOKENS.spacing.sm)
        for key, title in (
            ("t", "Mission Elapsed"),
            ("alt", "Altitude (km)"),
            ("vel", "Speed (km/s)"),
            ("ecc", "Eccentricity"),
            ("lat", "Latitude (°)"),
            ("lon", "Longitude (°)"),
        ):
            chip, value_lbl = self._kpi_chip(title)
            kpi_row.addWidget(chip, 1)
            self._kpis[key] = value_lbl
        layout.addWidget(kpi_frame)

        # Enhanced Telemetry Widget
        self.telemetry_multiplot = MultiTelemetryPlot()
        self.telemetry_multiplot.sample_updated.connect(self._update_kpis)
        layout.addWidget(self.telemetry_multiplot, 1)

    @staticmethod
    def _fmt_elapsed(t_s: float) -> str:
        """Human-readable mission-elapsed time from seconds."""
        if not math.isfinite(t_s):
            return "--"
        t = abs(float(t_s))
        if t < 60.0:
            return f"{t:.1f} s"
        if t < 3600.0:
            return f"{int(t // 60)}m {int(t % 60):02d}s"
        if t < DAY_S:
            h = int(t // 3600)
            return f"{h}h {int((t % 3600) // 60):02d}m"
        d = int(t // DAY_S)
        return f"{d}d {int((t % DAY_S) // 3600):02d}h"

    def _update_kpis(self, sample: dict) -> None:
        """Refresh the KPI strip from the latest telemetry sample (or reset)."""
        if not sample:
            for value_lbl in self._kpis.values():
                value_lbl.setText("--")
            return

        def _fmt(x, fmt: str) -> str:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return "--"
            return fmt.format(xf) if math.isfinite(xf) else "--"

        self._kpis["t"].setText(self._fmt_elapsed(sample.get("t_s", float("nan"))))
        self._kpis["alt"].setText(_fmt(sample.get("alt_km"), "{:,.1f}"))
        self._kpis["vel"].setText(_fmt(sample.get("v_km_s"), "{:.3f}"))
        self._kpis["ecc"].setText(_fmt(sample.get("ecc"), "{:.4f}"))
        self._kpis["lat"].setText(_fmt(sample.get("lat_deg"), "{:.2f}"))
        self._kpis["lon"].setText(_fmt(sample.get("lon_deg"), "{:.2f}"))



# =============================================================================
# 3.                         TESTING TELEMETRY PAGE
# =============================================================================

if __name__ == "__main__":
    import math
    import sys

    # Start the application
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create the test window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Live Telemetry Page Test")
    window.resize(1000, 700)

    # Set the background color (to simulate a dark theme)
    window.setStyleSheet(
        f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};"
    )

    # Load the page
    page = TelemetryPage()
    window.setCentralWidget(page)
    window.show()

    print("Test started...")

    # Optional: feed fake telemetry periodically (works only if pyqtgraph is installed;
    # otherwise add_datapoint() is a no-op and the UI still opens)
    t_s = [0.0]

    def push_fake_telem():
        t = t_s[0]
        telem = {
            "t_s": t,
            "alt_km": 100.0 + 10.0 * math.sin(t / 15.0),
            "v_km_s": 1.6 + 0.05 * math.cos(t / 20.0),
            "ecc": 0.01 + 0.002 * math.sin(t / 40.0),
            "lat_deg": 10.0 * math.sin(t / 30.0),
            "lon_deg": (t * 0.5) % 360.0,
        }
        page.telemetry_multiplot.add_datapoint(telem)
        t_s[0] += 1.0

    timer = QtCore.QTimer()
    timer.timeout.connect(push_fake_telem)
    timer.start(200)  # ms

    sys.exit(app.exec())
