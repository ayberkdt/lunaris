# LUNAR_SIMULATION/ui_parts/orbit_config_page.py
# -*- coding: utf-8 -*-

"""
Orbit Configuration & Visualization Module for Lunar Mission Studio.

This module manages the user interface for defining the Initial State Vector 
of the spacecraft using Keplerian elements. It integrates real-time mathematical 
conversions with an interactive 3D OpenGL visualization.

Key Features:
-----------------
1. Dual Input Logic (Altitude vs. Classical):
   Allows the user to define the orbit using two distinct paradigms:
   - Altitude-based: Periselene (hp) and Aposelene (ha) altitudes relative to R_MOON.
   - Classical Keplerian: Semi-major axis (a) and Eccentricity (e).
   
2. Bi-Directional Synchronization ("Ghosting"):
   Implements a live "shadowing" mechanism where:
   - If 'Altitude' mode is active, classical elements (a, e) are automatically 
     calculated and displayed as read-only "ghost" text.
   - If 'Classical' mode is active, altitudes (hp, ha) are back-calculated.
   This provides immediate feedback on the relationship between altitude and orbital geometry.

3. Interactive 3D Visualization:
   Uses pyqtgraph.opengl to render:
   - The Moon (scaled sphere).
   - The orbital trajectory (calculated via True Anomaly propagation).
   - Periapsis markers and coordinate axes (ECI Frame).
   
4. Input Validation:
   Enforces physical constraints (e.g., 0 <= e < 1.0, non-negative altitudes) 
   before data is passed to the simulation engine.

Dependencies:
    - PySide6 (UI Widgets)
    - pyqtgraph.opengl (3D Rendering)
    - ui_parts.ui_commons (Shared styling and custom widgets)
"""

# =============================================================================
# 0.                                    IMPORTS 
# =============================================================================
from __future__ import annotations

import numpy as np
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
    from .ui_commons import THEME, NumericDragLineEdit, get_icon, R_MOON_KM, MU_MOON_KM3_S2, StatusBadge
    R_MOON = R_MOON_KM  # local alias used throughout this module
except ImportError:
        # Only handle the "ran as a script" case; don't mask real import errors.
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  When executed directly, relative imports like '.constants' fail.", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m ui_parts.orbit_config_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2)
    raise



# =============================================================================
# 1.                        3D ORBIT VISUALIZER
# =============================================================================

class OrbitViz3D(QtWidgets.QWidget):
    """
    3D Orbit Visualizer using PyQtGraph OpenGL.
    Displays Moon as central sphere and orbit as elliptical line.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 280)
        
        # Keplerian elements storage
        self._a_km = 2000.0
        self._e = 0.0
        self._inc_deg = 90.0
        self._raan_deg = 0.0
        self._argp_deg = 0.0
        self._ta_deg = 0.0
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if not HAS_OPENGL:
            # Fallback: Show error message
            error_label = QtWidgets.QLabel(
                "OpenGL visualization unavailable.\n"
                "Install pyqtgraph with OpenGL support:\n"
                "pip install pyqtgraph pyopengl"
            )
            error_label.setAlignment(QtCore.Qt.AlignCenter)
            error_label.setStyleSheet(f"color: {THEME['error']}; padding: 20px;")
            layout.addWidget(error_label)
            return
        
        # Create GL View Widget
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor(THEME['bg_space'])
        self.gl_widget.opts['distance'] = 8000  # Initial camera distance (km)
        self.gl_widget.opts['elevation'] = 30
        self.gl_widget.opts['azimuth'] = 45
        
        # Add coordinate axes
        self._add_axes()
        
        # Create Moon sphere
        self._create_moon()
        
        # Orbit line (will be updated)
        self.orbit_line = None
        
        layout.addWidget(self.gl_widget)
        
        # Add view controls
        control_bar = QtWidgets.QHBoxLayout()
        control_bar.setContentsMargins(8, 4, 8, 8)
        
        self.btn_reset_view = QtWidgets.QPushButton("Reset View")
        self.btn_reset_view.setFixedHeight(28)
        self.btn_reset_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        # Reserve a little extra width so the label never looks clipped when the
        # global UI stylesheet increases horizontal padding or the active font
        # metrics differ slightly across Windows machines.
        reset_text_w = self.btn_reset_view.fontMetrics().horizontalAdvance("Reset View")
        self.btn_reset_view.setMinimumWidth(max(108, reset_text_w + 28))
        self.btn_reset_view.clicked.connect(self.reset_view)
        
        control_bar.addWidget(self.btn_reset_view)
        control_bar.addStretch()
        
        layout.addLayout(control_bar)
        
        # Initial draw
        QtCore.QTimer.singleShot(100, lambda: self.update_orbit())
    
    def _add_axes(self):
        """Add XYZ axes for orientation reference."""
        # X axis (red)
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [5000, 0, 0]]),
            color=(1, 0, 0, 0.7), width=2, antialias=True
        )
        self.gl_widget.addItem(x_axis)
        
        # Y axis (green)
        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 5000, 0]]),
            color=(0, 1, 0, 0.7), width=2, antialias=True
        )
        self.gl_widget.addItem(y_axis)
        
        # Z axis (blue)
        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 5000]]),
            color=(0, 0.5, 1, 0.7), width=2, antialias=True
        )
        self.gl_widget.addItem(z_axis)
    
    def _create_moon(self):
        """Create the Moon as a grey sphere."""
        # Create sphere mesh
        md = gl.MeshData.sphere(rows=20, cols=40, radius=R_MOON)
        
        # Color the sphere grey
        colors = np.ones((md.faceCount(), 4), dtype=float)
        colors[:, 0] = 0.5  # R
        colors[:, 1] = 0.5  # G
        colors[:, 2] = 0.5  # B
        colors[:, 3] = 1.0  # A
        
        md.setFaceColors(colors)
        
        self.moon_mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            shader='shaded',
            glOptions='opaque'
        )
        self.gl_widget.addItem(self.moon_mesh)
    
    def _kepler_to_cartesian(self, a_km, e, inc_rad, raan_rad, argp_rad, ta_rad):
        """
        Convert Keplerian elements to 3D Cartesian coordinates.
        Returns array of points for one orbit.
        """
        mu = MU_MOON_KM3_S2  # km³/s², derived from common.constants

        # Generate true anomalies from 0 to 2π
        n_points = 200
        true_anomalies = np.linspace(0, 2 * np.pi, n_points)
        
        # Calculate radius for each true anomaly
        r = a_km * (1 - e**2) / (1 + e * np.cos(true_anomalies))
        
        # Position in perifocal frame (PQW)
        x_pqw = r * np.cos(true_anomalies)
        y_pqw = r * np.sin(true_anomalies)
        z_pqw = np.zeros_like(true_anomalies)
        
        # Rotation matrix from perifocal to ECI
        cos_raan = np.cos(raan_rad)
        sin_raan = np.sin(raan_rad)
        cos_inc = np.cos(inc_rad)
        sin_inc = np.sin(inc_rad)
        cos_argp = np.cos(argp_rad)
        sin_argp = np.sin(argp_rad)
        
        # Combined rotation matrix: R_z(Ω) * R_x(i) * R_z(ω)
        R = np.array([
            [cos_raan*cos_argp - sin_raan*cos_inc*sin_argp, -cos_raan*sin_argp - sin_raan*cos_inc*cos_argp, sin_raan*sin_inc],
            [sin_raan*cos_argp + cos_raan*cos_inc*sin_argp, -sin_raan*sin_argp + cos_raan*cos_inc*cos_argp, -cos_raan*sin_inc],
            [sin_inc*sin_argp, sin_inc*cos_argp, cos_inc]
        ])
        
        # Transform to ECI
        points_pqw = np.stack([x_pqw, y_pqw, z_pqw], axis=1)
        points_eci = points_pqw @ R.T
        
        return points_eci
    
    def set_orbit_params(self, a_km: float, e: float, inc_deg: float, 
                         raan_deg: float, argp_deg: float, ta_deg: float):
        """Update the orbit parameters and redraw."""
        self._a_km = max(1.0, float(a_km))
        self._e = max(0.0, min(0.99, float(e)))
        self._inc_deg = float(inc_deg)
        self._raan_deg = float(raan_deg)
        self._argp_deg = float(argp_deg)
        self._ta_deg = float(ta_deg)
        
        self.update_orbit()
    
    def update_orbit(self):
        """Update the 3D orbit visualization."""
        if not HAS_OPENGL or not hasattr(self, 'gl_widget'):
            return
        
        # Convert degrees to radians
        inc_rad = np.deg2rad(self._inc_deg)
        raan_rad = np.deg2rad(self._raan_deg)
        argp_rad = np.deg2rad(self._argp_deg)
        ta_rad = np.deg2rad(self._ta_deg)
        
        try:
            # Generate orbit points
            points = self._kepler_to_cartesian(
                self._a_km, self._e, inc_rad, raan_rad, argp_rad, ta_rad
            )
            
            # Remove old orbit line if exists
            if self.orbit_line is not None:
                self.gl_widget.removeItem(self.orbit_line)
            
            # Create new orbit line
            self.orbit_line = gl.GLLinePlotItem(
                pos=points,
                color=(0.76, 0.63, 0.39, 1.0),
                width=3,
                antialias=True,
                glOptions='translucent'
            )
            self.gl_widget.addItem(self.orbit_line)
            
            # Add periapsis marker
            self._add_periapsis_marker(points[0])
            
        except Exception as e:
            print(f"[3D Viz] Error updating orbit: {e}")
    
    def _add_periapsis_marker(self, periapsis_point):
        """Add a marker at periapsis."""
        if hasattr(self, 'periapsis_marker'):
            self.gl_widget.removeItem(self.periapsis_marker)
        
        # Create a small sphere at periapsis
        md = gl.MeshData.sphere(rows=10, cols=20, radius=50)  # 50km radius marker
        
        colors = np.ones((md.faceCount(), 4), dtype=float)
        colors[:, 0] = 0.84
        colors[:, 1] = 0.70
        colors[:, 2] = 0.43
        md.setFaceColors(colors)
        
        self.periapsis_marker = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            shader='shaded',
            glOptions='translucent'
        )
        self.periapsis_marker.translate(*periapsis_point)
        self.gl_widget.addItem(self.periapsis_marker)
    
    def reset_view(self, _checked: bool = False):
        """Reset camera to default position."""
        if HAS_OPENGL and hasattr(self, 'gl_widget'):
            self.gl_widget.setCameraPosition(
                distance=8000,
                elevation=30,
                azimuth=45
            )



# =============================================================================
# 2.                        MAIN ORBIT PAGE CLASS
# =============================================================================

class OrbitPage(QtWidgets.QWidget):
    """
    The main widget page for configuring the orbit.
    Contains inputs for orbit elements and the 3D visualization.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating_ghost = False # Flag to prevent recursive signal loops
        self._build_ui()
        
    def _create_card(self, title: str) -> QtWidgets.QGroupBox:
        """Factory for standard titled group boxes (Cards)."""
        gb = QtWidgets.QGroupBox(title)
        return gb
    
    def _build_ui(self):
        """Constructs the layout of the Orbit Configuration Page."""
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(20)
        
        # Left Column: Orbit Parameters
        self.group_params = self._create_params_group()
        layout.addWidget(self.group_params, 0, 0, 2, 1)
        
        # Right Column: 3D Visualization
        self.group_viz = self._create_viz_group()
        layout.addWidget(self.group_viz, 0, 1, 2, 1)
        
        # Adjust column ratios (Left slightly wider for inputs, Right for Viz)
        layout.setColumnStretch(0, 11)
        layout.setColumnStretch(1, 9)

    def _create_params_group(self) -> QtWidgets.QGroupBox:
        """Orbit parameters card with Modern Segmented Control."""
        gb = self._create_card("Initial Orbit State")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(22, 24, 22, 22)
        layout.setSpacing(16)
        
        # A. Modern Segmented Control for Input Mode
        mode_container = QtWidgets.QWidget()
        mode_container.setFixedHeight(46)
        mode_container.setStyleSheet(f"""
            background-color: {THEME['bg_card_alt']};
            border-radius: 10px;
            border: 1px solid {THEME['border_soft']};
        """)
        
        mode_layout = QtWidgets.QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(4, 4, 4, 4)
        mode_layout.setSpacing(0)
        
        # Create three styled buttons as segments
        self.btn_mode_altitude = QtWidgets.QPushButton("Altitude (hp/ha)")
        self.btn_mode_classical = QtWidgets.QPushButton("Classical (a/e)")
        self.btn_mode_circular = QtWidgets.QPushButton("Circular (alt)")

        for btn in (self.btn_mode_altitude, self.btn_mode_classical, self.btn_mode_circular):
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {THEME['fg_muted']};
                    border: none;
                    border-radius: 8px;
                    padding: 7px 16px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 255, 255, 0.04);
                }}
                QPushButton:checked {{
                    background-color: rgba(185, 151, 91, 0.22);
                    border: 1px solid rgba(185, 151, 91, 0.34);
                    color: {THEME['fg_main']};
                }}
            """)
            mode_layout.addWidget(btn)

        # Create button group for exclusive selection
        self.mode_button_group = QtWidgets.QButtonGroup(self)
        self.mode_button_group.addButton(self.btn_mode_altitude, 0)
        self.mode_button_group.addButton(self.btn_mode_classical, 1)
        self.mode_button_group.addButton(self.btn_mode_circular, 2)
        self.mode_button_group.setExclusive(True)
        self.btn_mode_altitude.setChecked(True)

        # Connect signal
        self.btn_mode_altitude.toggled.connect(self._sync_orbit_mode_ghosting)
        self.btn_mode_circular.toggled.connect(self._sync_orbit_mode_ghosting)
        
        layout.addWidget(mode_container)

        intro = QtWidgets.QLabel(
            "Choose the orbit entry style you prefer. Related values stay "
            "synchronized automatically so you can review the full orbit shape at a glance."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {THEME['fg_muted']};")
        layout.addWidget(intro)
        
        # B. Parameter Form with Ghosting
        form_layout = QtWidgets.QGridLayout()
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(12)
        
        def add_param(row, label, widget, unit=""):
            lbl = QtWidgets.QLabel(label)
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lbl.setStyleSheet(f"color: {THEME['fg_soft']}; font-weight: 600;")
            form_layout.addWidget(lbl, row, 0)
            form_layout.addWidget(widget, row, 1)
            if unit:
                lbl_unit = QtWidgets.QLabel(unit)
                lbl_unit.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
                form_layout.addWidget(lbl_unit, row, 2)
        
        # Input Fields - All are NumericDragLineEdit
        self.ent_hp = NumericDragLineEdit("100.0", step=5.0, min_value=0.0, decimals=1)
        self.ent_ha = NumericDragLineEdit("", step=5.0, min_value=0.0, decimals=1)
        self.ent_ha.setPlaceholderText("Circular (same as hp)")
        
        self.ent_a = NumericDragLineEdit("", step=10.0, min_value=1.0, decimals=2)
        self.ent_e = NumericDragLineEdit("0.0", step=0.01, min_value=0.0, max_value=0.999, decimals=4)
        
        self.ent_inc = NumericDragLineEdit("90.0", step=1.0, min_value=0.0, max_value=180.0, decimals=2)
        self.ent_raan = NumericDragLineEdit("0.0", step=5.0, min_value=0.0, max_value=360.0, decimals=2)
        self.ent_argp = NumericDragLineEdit("0.0", step=5.0, min_value=0.0, max_value=360.0, decimals=2)
        self.ent_ta = NumericDragLineEdit("0.0", step=5.0, min_value=0.0, max_value=360.0, decimals=2)
        
        # Circular altitude mode input (shown only in "circular" mode)
        self.ent_alt_circular = NumericDragLineEdit("100.0", step=10.0, min_value=0.0, max_value=10000.0, decimals=1)

        orbit_shape_lbl = QtWidgets.QLabel("Orbit size and shape")
        orbit_shape_lbl.setStyleSheet(f"color: {THEME['fg_soft']}; font-weight: 700;")
        form_layout.addWidget(orbit_shape_lbl, 0, 0, 1, 3)

        # Add to Form
        add_param(1, "Periselene Altitude (hp)", self.ent_hp, "km")
        add_param(2, "Aposelene Altitude (ha)", self.ent_ha, "km")
        add_param(3, "Semi-major Axis (a)", self.ent_a, "km")
        add_param(4, "Eccentricity (e)", self.ent_e, "")
        add_param(5, "Circular Altitude", self.ent_alt_circular, "km")
        
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet(f"color: {THEME['border_soft']};")
        form_layout.addWidget(sep, 6, 0, 1, 3)

        orientation_lbl = QtWidgets.QLabel("Plane and orientation")
        orientation_lbl.setStyleSheet(f"color: {THEME['fg_soft']}; font-weight: 700;")
        form_layout.addWidget(orientation_lbl, 7, 0, 1, 3)

        add_param(8, "Inclination (i)", self.ent_inc, "deg")
        add_param(9, "RAAN (Omega)", self.ent_raan, "deg")
        add_param(10, "Argument of Periapsis (omega)", self.ent_argp, "deg")
        add_param(11, "True Anomaly (nu)", self.ent_ta, "deg")
        
        layout.addLayout(form_layout)
        
        # C. Quick Actions
        action_bar = QtWidgets.QHBoxLayout()
        
        btn_zero = QtWidgets.QPushButton("Reset Orientation")
        btn_zero.setIcon(get_icon("fa6s.rotate-left", THEME['fg_main']))
        btn_zero.clicked.connect(self._zero_angles)
        btn_zero.setFixedHeight(32)
        
        btn_circular = QtWidgets.QPushButton("Set Circular Orbit")
        btn_circular.setIcon(get_icon("fa6s.circle", THEME['fg_main']))
        btn_circular.clicked.connect(self._make_circular)
        btn_circular.setFixedHeight(32)
        
        action_bar.addWidget(btn_zero)
        action_bar.addWidget(btn_circular)
        action_bar.addStretch()
        
        layout.addLayout(action_bar)
        
        # Connect Signals for Bidirectional Ghosting
        self.ent_hp.value_changed.connect(lambda _: self._update_ghost_orbit())
        self.ent_ha.value_changed.connect(lambda _: self._update_ghost_orbit())
        self.ent_a.value_changed.connect(lambda _: self._update_ghost_orbit())
        self.ent_e.value_changed.connect(lambda _: self._update_ghost_orbit())

        # Connect for 3D Visualization
        for w in (self.ent_hp, self.ent_ha, self.ent_a, self.ent_e,
                  self.ent_alt_circular, self.ent_inc, self.ent_raan, self.ent_argp, self.ent_ta):
            w.value_changed.connect(lambda _: self._update_orbit_3d())
        self.btn_mode_altitude.toggled.connect(self._update_orbit_3d)
        self.btn_mode_circular.toggled.connect(self._update_orbit_3d)
        
        # Initial Ghosting State
        self._sync_orbit_mode_ghosting()
        
        return gb

    def _create_viz_group(self) -> QtWidgets.QGroupBox:
        """3D orbit preview card."""
        gb = self._create_card("Orbit Preview")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(16, 24, 16, 16)
        layout.setSpacing(14)

        intro = QtWidgets.QLabel(
            "Review the orbit geometry, viewing angle, and estimated period before you launch the mission run."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {THEME['fg_muted']};")
        layout.addWidget(intro)

        self.orbit_viz_3d = OrbitViz3D()
        self.orbit_viz_3d.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.orbit_viz_3d)

        info_frame = QtWidgets.QFrame()
        info_frame.setStyleSheet(
            f"background: {THEME['bg_card_alt']}; border: 1px solid {THEME['border_soft']}; border-radius: 8px;"
        )
        info_layout = QtWidgets.QHBoxLayout(info_frame)

        self.lbl_period = QtWidgets.QLabel("Estimated Period: --")
        self.lbl_period.setStyleSheet(f"color: {THEME['fg_muted']};")

        self.lbl_energy = QtWidgets.QLabel("Orbit Energy: --")
        self.lbl_energy.setStyleSheet(f"color: {THEME['fg_muted']};")

        info_layout.addWidget(self.lbl_period)
        info_layout.addWidget(self.lbl_energy)
        info_layout.addStretch()

        layout.addWidget(info_frame)

        QtCore.QTimer.singleShot(100, self._update_orbit_3d)
        return gb

    # =========================================================================
    # 3.                        LOGIC & MATH
    # =========================================================================

    def _sync_orbit_mode_ghosting(self, enabled: bool = False):
        """Toggle between altitude, classical, and circular orbit input modes with ghosting."""
        is_alt_mode = self.btn_mode_altitude.isChecked()
        is_circular_mode = self.btn_mode_circular.isChecked()

        ghost_style = (
            f"background: rgba(255, 255, 255, 0.02);"
            f"border: 1px dashed {THEME['border_soft']};"
            f"color: {THEME['fg_muted']};"
            "font-style: italic;"
        )

        if is_circular_mode:
            # Circular mode: only alt_circular is active; hp/ha/a/e are ghost
            active_fields = [self.ent_alt_circular]
            ghost_fields = [self.ent_hp, self.ent_ha, self.ent_a, self.ent_e]
        elif is_alt_mode:
            # Altitude mode active: hp/ha are active, a/e and circular are ghost
            active_fields = [self.ent_hp, self.ent_ha]
            ghost_fields = [self.ent_a, self.ent_e, self.ent_alt_circular]
        else:
            # Classical mode active: a/e are active, hp/ha and circular are ghost
            active_fields = [self.ent_a, self.ent_e]
            ghost_fields = [self.ent_hp, self.ent_ha, self.ent_alt_circular]

        # Set active fields
        for field in active_fields:
            field.setReadOnly(False)
            field.setStyleSheet("")
            field.setEnabled(True)

        # Set ghost fields
        for field in ghost_fields:
            field.setReadOnly(True)
            field.setStyleSheet(ghost_style)
            field.setEnabled(False)

        # Trigger initial ghost calculation
        self._update_ghost_orbit()

    def _update_ghost_orbit(self):
        """Bidirectional calculation between altitude and classical parameters."""
        # Prevent infinite recursion
        if self._updating_ghost:
            return
        
        self._updating_ghost = True
        
        try:
            is_alt_mode = self.btn_mode_altitude.isChecked()
            
            if is_alt_mode:
                # Altitude mode active: calculate a/e from hp/ha
                try:
                    hp_text = self.ent_hp.text().strip()
                    ha_text = self.ent_ha.text().strip()
                    
                    if hp_text:
                        hp = float(hp_text)
                        ha = float(ha_text) if ha_text else hp
                        
                        # Formulas: From Altitude to Classical
                        rp = R_MOON + hp
                        ra = R_MOON + ha
                        
                        # Ensure periapsis <= apoapsis
                        if rp > ra:
                            rp, ra = ra, rp
                        
                        a = (rp + ra) / 2.0
                        e = (ra - rp) / (ra + rp) if (ra + rp) > 0 else 0.0
                        
                        # Update ghost fields (block signals to prevent recursion)
                        self.ent_a.blockSignals(True)
                        self.ent_e.blockSignals(True)
                        self.ent_a.setText(f"{a:.2f}")
                        self.ent_e.setText(f"{e:.5f}")
                        self.ent_a.blockSignals(False)
                        self.ent_e.blockSignals(False)
                except ValueError:
                    pass
            else:
                # Classical mode active: calculate hp/ha from a/e
                try:
                    a_text = self.ent_a.text().strip()
                    e_text = self.ent_e.text().strip()
                    
                    if a_text:
                        a = float(a_text)
                        e = float(e_text) if e_text else 0.0
                        
                        # Clamp eccentricity
                        e = max(0.0, min(0.999, e))
                        
                        # Formulas: From Classical to Altitude
                        rp = a * (1 - e)
                        ra = a * (1 + e)
                        
                        hp = rp - R_MOON
                        ha = ra - R_MOON
                        
                        # Ensure non-negative altitudes
                        hp = max(0.0, hp)
                        ha = max(0.0, ha)
                        
                        # Update ghost fields (block signals to prevent recursion)
                        self.ent_hp.blockSignals(True)
                        self.ent_ha.blockSignals(True)
                        self.ent_hp.setText(f"{hp:.1f}")
                        self.ent_ha.setText(f"{ha:.1f}")
                        self.ent_hp.blockSignals(False)
                        self.ent_ha.blockSignals(False)
                except ValueError:
                    pass
        finally:
            self._updating_ghost = False

    def _update_orbit_3d(self, _=None):
        """Update the 3D orbit visualizer."""
        if not hasattr(self, "orbit_viz_3d"):
            return

        try:
            # Get parameters based on current mode
            if self.btn_mode_circular.isChecked():
                alt_text = self.ent_alt_circular.text().strip()
                if alt_text:
                    alt = float(alt_text)
                    a_km = R_MOON + alt
                    e = 0.0
                else:
                    return
            elif self.btn_mode_altitude.isChecked():
                hp_text = self.ent_hp.text().strip()
                ha_text = self.ent_ha.text().strip()

                if hp_text:
                    hp = float(hp_text)
                    ha = float(ha_text) if ha_text else hp

                    R_body = R_MOON
                    rp = R_body + hp
                    ra = R_body + ha

                    if rp > ra:
                        rp, ra = ra, rp

                    a_km = (rp + ra) / 2.0
                    e = (ra - rp) / (ra + rp) if (ra + rp) > 0 else 0.0
                else:
                    return
            else:
                a_text = self.ent_a.text().strip()
                e_text = self.ent_e.text().strip()

                if a_text:
                    a_km = float(a_text)
                    e = float(e_text) if e_text else 0.0
                else:
                    return
            
            # Get angular elements
            inc_deg = float(self.ent_inc.text() or 90.0)
            raan_deg = float(self.ent_raan.text() or 0.0)
            argp_deg = float(self.ent_argp.text() or 0.0)
            ta_deg = float(self.ent_ta.text() or 0.0)
            
            # Update 3D visualizer
            self.orbit_viz_3d.set_orbit_params(a_km, e, inc_deg, raan_deg, argp_deg, ta_deg)
            
            # Update orbital period estimate (Kepler's third law)
            mu = MU_MOON_KM3_S2  # km³/s², derived from common.constants
            if a_km > 0:
                period_s = 2 * 3.1415926535 * (a_km ** 3 / mu) ** 0.5
                period_h = period_s / 3600
                self.lbl_period.setText(f"Estimated Period: {period_h:.1f} h")
                
                # Specific orbital energy
                energy = -mu / (2 * a_km)
                self.lbl_energy.setText(f"Orbit Energy: {energy:.3f} km^2/s^2")
        
        except ValueError:
            pass

    def _zero_angles(self, _checked: bool = False):
        """Reset orbital angles to zero."""
        for w in (self.ent_inc, self.ent_raan, self.ent_argp, self.ent_ta):
            w.setText("0.0")
        self._update_orbit_3d()

    def _make_circular(self, _checked: bool = False):
        """Make orbit circular."""
        if self.btn_mode_classical.isChecked():
            self.ent_e.setText("0.0")
        else:
            hp = self.ent_hp.text().strip()
            if hp:
                self.ent_ha.setText(hp)
        
        self._update_ghost_orbit()
        self._update_orbit_3d()

    # =========================================================================
    # 4.                        DATA ACCESS (Interface)
    # =========================================================================

    def get_data(self) -> dict:
        """
        Retrieves the current orbital configuration.
        Returns a dictionary suitable for the main simulation logic.
        """
        # Determine active mode
        if self.btn_mode_circular.isChecked():
            mode = "circular"
        elif self.btn_mode_altitude.isChecked():
            mode = "hp_ha"
        else:
            mode = "a_e"

        data = {
            "mode": mode,
            # Angular params are always the same
            "inc_deg": float(self.ent_inc.text() or "90"),
            "raan_deg": float(self.ent_raan.text() or "0"),
            "argp_deg": float(self.ent_argp.text() or "0"),
            "ta_deg": float(self.ent_ta.text() or "0"),
        }

        # Add mode-specific params
        if mode == "circular":
            data["alt_km"] = float(self.ent_alt_circular.text() or "100")
        elif mode == "hp_ha":
            data["hp_km"] = float(self.ent_hp.text() or "100")
            # If ha is empty, assume circular (ha=hp)
            data["ha_km"] = float(self.ent_ha.text() or self.ent_hp.text() or "100")
        else:
            data["a_km"] = float(self.ent_a.text() or "2000")
            data["e"] = float(self.ent_e.text() or "0")

        return data

    def load_data(self, data: dict):
        """
        Populate the UI from a dictionary (e.g. from a saved session).
        """
        if not data:
            return

        mode = data.get("mode", "hp_ha")
        self.btn_mode_altitude.setChecked(mode == "hp_ha")
        self.btn_mode_classical.setChecked(mode == "a_e")
        self.btn_mode_circular.setChecked(mode == "circular")

        # Set text fields
        self.ent_hp.setText(str(data.get("hp_km", "100.0")))
        self.ent_ha.setText(str(data.get("ha_km", "")))
        self.ent_a.setText(str(data.get("a_km", "")))
        self.ent_e.setText(str(data.get("e", "0.0")))
        self.ent_alt_circular.setText(str(data.get("alt_km", "100.0")))

        self.ent_inc.setText(str(data.get("inc_deg", "90.0")))
        self.ent_raan.setText(str(data.get("raan_deg", "0.0")))
        self.ent_argp.setText(str(data.get("argp_deg", "0.0")))
        self.ent_ta.setText(str(data.get("ta_deg", "0.0")))

        # Force update logic
        self._sync_orbit_mode_ghosting()
        self._update_orbit_3d()



# =============================================================================
# 3.                        TESTING ORBIT PAGE
# =============================================================================

if __name__ == "__main__":
    import sys

    # Start the application
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create the test window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Orbit Page Test")
    window.resize(1000, 700)

    # Set the background color (to simulate a dark theme)
    window.setStyleSheet(
        f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};"
    )

    # Load the page
    page = OrbitPage()
    window.setCentralWidget(page)

    window.show()

    print("Test started...")
    print("Initial Data:", page.get_data())

    sys.exit(app.exec())
