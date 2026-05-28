# ST_LRPS/ui_parts/live_telemetry_page.py
"""
Live Telemetry Page (UI)

This module defines the **TelemetryPage** used by the ST-LRPS Studio UI to display
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

import numpy as np
from typing import Dict, Any
from collections import deque

from PySide6 import QtCore, QtWidgets

# Modern Icon Library
try:
    import qtawesome as qta
    HAS_QTAWESOME = True
except ImportError:
    HAS_QTAWESOME = False
    print("[Warning] qtawesome not installed. Icons will be disabled.")

# Live Plotting & 3D Visualization
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    HAS_PYQTGRAPH = True
    HAS_OPENGL = True
    # Enable antialiasing globally
    pg.setConfigOptions(antialias=True)
except ImportError as e:
    HAS_PYQTGRAPH = False
    HAS_OPENGL = False
    print(f"[Warning] PyQtGraph/OpenGL not installed. Advanced visualization disabled: {e}")


try:
    from .ui_commons import THEME
except ImportError:
        # Only handle the "ran as a script" case; don't mask real import errors.
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  When executed directly, relative imports like '.constants' fail.", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m lunaris.ui.widgets.live_telemetry_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2)
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
    
    Updated v12: Sci-Fi/Mission Control aesthetic with enhanced styling
    """
    
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
            # Fallback message
            placeholder = QtWidgets.QLabel("PyQtGraph not installed.\nLive telemetry unavailable.")
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            placeholder.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 14px;")
            layout.addWidget(placeholder)
            return
        
        # Plot Type Selector
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.setContentsMargins(10, 5, 10, 5)
        
        selector_layout.addWidget(QtWidgets.QLabel("Plot Type:"))
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems([
            "Altitude vs Time",
            "Velocity vs Time",
            "Eccentricity vs Time",
            "Ground Track"
        ])
        self.plot_type_combo.currentTextChanged.connect(self._switch_plot)
        selector_layout.addWidget(self.plot_type_combo)
        
        selector_layout.addStretch()

        # ------------------------------
        # Live Telemetry Axis Controls
        # ------------------------------
        selector_layout.addWidget(QtWidgets.QLabel("Time:"))
        self.time_axis_combo = QtWidgets.QComboBox()
        self.time_axis_combo.addItems(["Auto", "s", "min", "h", "d"])
        self.time_axis_combo.setFixedHeight(24)
        self.time_axis_combo.setFixedWidth(70)
        self.time_axis_combo.currentTextChanged.connect(self._on_axis_controls_changed)
        selector_layout.addWidget(self.time_axis_combo)

        self.chk_time_relative = QtWidgets.QCheckBox("T+")
        self.chk_time_relative.setChecked(True)
        self.chk_time_relative.setToolTip("Display time relative to first received sample.")
        self.chk_time_relative.toggled.connect(self._on_axis_controls_changed)
        selector_layout.addWidget(self.chk_time_relative)

        selector_layout.addWidget(QtWidgets.QLabel("Mouse zoom:"))
        self.chk_mouse_x = QtWidgets.QCheckBox("X")
        self.chk_mouse_y = QtWidgets.QCheckBox("Y")
        self.chk_mouse_x.setChecked(False)  # keep time axis stable by default
        self.chk_mouse_y.setChecked(True)
        self.chk_mouse_x.toggled.connect(self._apply_mouse_zoom_settings)
        self.chk_mouse_y.toggled.connect(self._apply_mouse_zoom_settings)
        selector_layout.addWidget(self.chk_mouse_x)
        selector_layout.addWidget(self.chk_mouse_y)

        selector_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.chk_auto_y = QtWidgets.QCheckBox("Auto")
        self.chk_auto_y.setChecked(True)
        self.chk_auto_y.toggled.connect(self._on_y_mode_changed)
        selector_layout.addWidget(self.chk_auto_y)

        self.ed_ymin = QtWidgets.QLineEdit()
        self.ed_ymin.setPlaceholderText("min")
        self.ed_ymin.setFixedWidth(70)
        self.ed_ymax = QtWidgets.QLineEdit()
        self.ed_ymax.setPlaceholderText("max")
        self.ed_ymax.setFixedWidth(70)
        selector_layout.addWidget(self.ed_ymin)
        selector_layout.addWidget(self.ed_ymax)

        self.spin_y_pad = QtWidgets.QDoubleSpinBox()
        self.spin_y_pad.setSuffix("%")
        self.spin_y_pad.setDecimals(1)
        self.spin_y_pad.setRange(0.0, 50.0)
        self.spin_y_pad.setSingleStep(1.0)
        self.spin_y_pad.setValue(5.0)
        self.spin_y_pad.setFixedWidth(75)
        self.spin_y_pad.valueChanged.connect(self._on_axis_controls_changed)
        selector_layout.addWidget(self.spin_y_pad)

        self.btn_y_apply = QtWidgets.QPushButton("Apply")
        self.btn_y_apply.setFixedHeight(24)
        self.btn_y_apply.clicked.connect(self._apply_manual_y_range)
        selector_layout.addWidget(self.btn_y_apply)

        self.btn_y_fit = QtWidgets.QPushButton("Fit")
        self.btn_y_fit.setFixedHeight(24)
        self.btn_y_fit.clicked.connect(self._fit_y_range_to_data)
        selector_layout.addWidget(self.btn_y_fit)

        self.btn_clear = QtWidgets.QPushButton("Clear All")
        self.btn_clear.setFixedHeight(24)
        self.btn_clear.clicked.connect(self.clear_all)
        selector_layout.addWidget(self.btn_clear)
        
        layout.addLayout(selector_layout)
        
        # Stacked widget for different plots
        self.plot_stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.plot_stack, 1)
        
        # Create individual plot widgets
        self._create_altitude_plot()
        self._create_velocity_plot()
        self._create_eccentricity_plot()
        self._create_ground_track_plot()
        
        # Timer for buffered updates (30 FPS)
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setInterval(33)  # ~30 Hz
        self.update_timer.timeout.connect(self._flush_buffer)
        self.update_timer.start()
        
        # Set initial plot
        self._switch_plot("Altitude vs Time")
    
    def _create_altitude_plot(self):
        """Altitude vs Time plot with Sci-Fi styling."""
        widget = pg.PlotWidget()
        widget.setBackground(THEME['bg_space'])
        widget.setTitle("Orbital Altitude", color='#00FFFF', size='14pt', bold=True)  # Cyan
        widget.setLabel('left', 'Altitude [km]', color='#FFFFFF', size='11pt')
        widget.setLabel('bottom', 'Time [s]', color='#FFFFFF', size='11pt')
        widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Enhanced axis styling
        axis_pen = pg.mkPen(color='#8BE9FD', width=1.5)  # Bright cyan
        widget.getAxis('left').setPen(axis_pen)
        widget.getAxis('left').setTextPen('#FFFFFF')
        widget.getAxis('bottom').setPen(axis_pen)
        widget.getAxis('bottom').setTextPen('#FFFFFF')
        

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
            pen=pg.mkPen(color='#00FFFF', width=2.5),  # Bright cyan
            name='Altitude',
            shadowPen=pg.mkPen(color='#00FFFF', width=4, alpha=0.3)
        )
        
        self.plot_stack.addWidget(widget)
    
    def _create_velocity_plot(self):
        """Velocity vs Time plot with Sci-Fi styling."""
        widget = pg.PlotWidget()
        widget.setBackground(THEME['bg_space'])
        widget.setTitle("Orbital Velocity", color='#FF00FF', size='14pt', bold=True)  # Magenta
        widget.setLabel('left', 'Velocity [km/s]', color='#FFFFFF', size='11pt')
        widget.setLabel('bottom', 'Time [s]', color='#FFFFFF', size='11pt')
        widget.showGrid(x=True, y=True, alpha=0.3)
        
        axis_pen = pg.mkPen(color='#FF79C6', width=1.5)  # Bright pink
        widget.getAxis('left').setPen(axis_pen)
        widget.getAxis('left').setTextPen('#FFFFFF')
        widget.getAxis('bottom').setPen(axis_pen)
        widget.getAxis('bottom').setTextPen('#FFFFFF')
        

        # Store widgets / viewbox for axis controls
        self.vel_plot_widget = widget
        self.vel_viewbox = widget.getViewBox()
        try:
            self.vel_viewbox.setMouseEnabled(x=False, y=True)
        except Exception as exc:
            _warn_once("vel_viewbox_mouse", str(exc))

        self.vel_curve = widget.plot(
            pen=pg.mkPen(color='#FF00FF', width=2.5),  # Magenta
            name='Velocity',
            shadowPen=pg.mkPen(color='#FF00FF', width=4, alpha=0.3)
        )
        
        self.plot_stack.addWidget(widget)
    
    def _create_eccentricity_plot(self):
        """Eccentricity vs Time plot with Sci-Fi styling."""
        widget = pg.PlotWidget()
        widget.setBackground(THEME['bg_space'])
        widget.setTitle("Orbit Eccentricity", color='#FFFF00', size='14pt', bold=True)  # Yellow
        widget.setLabel('left', 'Eccentricity', color='#FFFFFF', size='11pt')
        widget.setLabel('bottom', 'Time [s]', color='#FFFFFF', size='11pt')
        widget.showGrid(x=True, y=True, alpha=0.3)
        
        axis_pen = pg.mkPen(color='#F1FA8C', width=1.5)  # Light yellow
        widget.getAxis('left').setPen(axis_pen)
        widget.getAxis('left').setTextPen('#FFFFFF')
        widget.getAxis('bottom').setPen(axis_pen)
        widget.getAxis('bottom').setTextPen('#FFFFFF')
        

        # Store widgets / viewbox for axis controls
        self.ecc_plot_widget = widget
        self.ecc_viewbox = widget.getViewBox()
        try:
            self.ecc_viewbox.setMouseEnabled(x=False, y=True)
        except Exception as exc:
            _warn_once("ecc_viewbox_mouse", str(exc))

        self.ecc_curve = widget.plot(
            pen=pg.mkPen(color='#FFFF00', width=2.5),  # Yellow
            name='Eccentricity',
            shadowPen=pg.mkPen(color='#FFFF00', width=4, alpha=0.3)
        )
        
        self.plot_stack.addWidget(widget)
    
    def _create_ground_track_plot(self):
        """Ground Track (Latitude vs Longitude) plot with Sci-Fi styling."""
        widget = pg.PlotWidget()
        widget.setBackground(THEME['bg_space'])
        widget.setTitle("Ground Track", color='#50FA7B', size='14pt', bold=True)  # Green
        widget.setLabel('left', 'Latitude [deg]', color='#FFFFFF', size='11pt')
        widget.setLabel('bottom', 'Longitude [deg]', color='#FFFFFF', size='11pt')
        widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Set axis ranges for Moon
        widget.setXRange(-180, 180)
        widget.setYRange(-90, 90)
        
        axis_pen = pg.mkPen(color='#50FA7B', width=1.5)  # Green
        widget.getAxis('left').setPen(axis_pen)
        widget.getAxis('left').setTextPen('#FFFFFF')
        widget.getAxis('bottom').setPen(axis_pen)
        widget.getAxis('bottom').setTextPen('#FFFFFF')
        
        self.ground_track_curve = widget.plot(
            pen=pg.mkPen(color='#50FA7B', width=2.0, style=QtCore.Qt.DashLine),
            symbol='o',
            symbolSize=5,
            symbolBrush=(80, 250, 123, 0.9),  # Green with alpha
            symbolPen=pg.mkPen(color='#50FA7B', width=1),
            name='Ground Track'
        )
        
        self.plot_stack.addWidget(widget)
    
    def _switch_plot(self, plot_name):
        """Switch between different plot types."""
        plot_map = {
            "Altitude vs Time": 0,
            "Velocity vs Time": 1,
            "Eccentricity vs Time": 2,
            "Ground Track": 3
        }

        idx = plot_map.get(plot_name, 0)
        self.plot_stack.setCurrentIndex(idx)

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
                  getattr(self, "btn_y_fit", None)):
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
            if span >= 5.0 * 86400.0:
                unit = "d"
            elif span >= 3.0 * 3600.0:
                unit = "h"
            elif span >= 5.0 * 60.0:
                unit = "min"
            else:
                unit = "s"
        else:
            unit = choice

        factor = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}.get(unit, 1.0)
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
                w.setLabel('bottom', label, color='#FFFFFF', size='11pt')
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


    def add_datapoint(self, telem_data: Dict[str, Any]):
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
                t_s *= 86400.0

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



# =============================================================================
# 2.                           TELEMETRY PAGE
# =============================================================================

class TelemetryPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QtWidgets.QLabel("Real-time Mission Telemetry")
        header.setStyleSheet(f"""
            color: {THEME['fg_main']};
            font-weight: bold;
            font-size: 14pt;
            margin: 20px;
        """)
        layout.addWidget(header)

        # Enhanced Telemetry Widget
        self.telemetry_multiplot = MultiTelemetryPlot()
        layout.addWidget(self.telemetry_multiplot, 1)



# =============================================================================
# 3.                         TESTING TELEMETRY PAGE
# =============================================================================

if __name__ == "__main__":
    import sys
    import math

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
