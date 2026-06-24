# ST_LRPS/ui_parts/orbit_config_page.py

"""
Orbit Configuration & Visualization Module for Lunaris Mission Studio.

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

import math
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

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
    from lunaris.ui.components.primitives import EmptyState, Section
    from lunaris.ui.core.ui_commons import (
        MU_MOON_KM3_S2,
        ORBIT_THEME,
        R_MOON_KM,
        THEME,
        NumericDragLineEdit,
        get_icon,
        hex_to_rgba_float,
        rgba_css_to_tuple,
    )
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
        print("\n      python -m lunaris.ui.pages.orbit_config_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2) from None
    raise



# =============================================================================
# 0b.                       VALIDATED ORBIT STATE
# =============================================================================

@dataclass(frozen=True)
class _OrbitState:
    """A fully validated orbit, derived once and shared by preview and metrics.

    Keeping this separate from any OpenGL call means the validation/derivation
    logic can be unit-tested without a rendering backend.
    """

    a_km: float
    e: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    ta_deg: float


# =============================================================================
# 1.                        3D ORBIT VISUALIZER
# =============================================================================

class OrbitViz3D(QtWidgets.QWidget):
    """
    3D orbit visualizer using pyqtgraph's OpenGL backend.

    Renders the Moon as a softly shaded regolith sphere with faint lat/long
    guide rings, the orbital trajectory as a clean translucent arc with an
    optional glow underlay, and labelled markers for periapsis, apoapsis, and
    the current spacecraft position (true anomaly).  All colors come from
    ``ORBIT_THEME`` so the preview matches the Lunar Graphite palette.
    """


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 340)

        # Keplerian element storage
        self._a_km = 2000.0
        self._e = 0.0
        self._inc_deg = 90.0
        self._raan_deg = 0.0
        self._argp_deg = 0.0
        self._ta_deg = 0.0

        # GL item handles (created lazily / refreshed on each update)
        self.gl_widget = None
        self.orbit_line = None
        self.orbit_glow = None
        self.periapsis_marker = None
        self.apoapsis_marker = None
        self.spacecraft_marker = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if not HAS_OPENGL:
            self._install_fallback(layout)
            return

        try:
            # pyqtgraph's GLViewWidget reads the *global* 'background' config
            # option in its constructor and raises if it is None. Another
            # pyqtgraph-using subsystem in the same process can set that
            # process-wide, so pin a valid value before constructing the widget.
            if pg.getConfigOption('background') is None:
                pg.setConfigOption('background', ORBIT_THEME['space_bg'])

            self.gl_widget = gl.GLViewWidget()
            self.gl_widget.setBackgroundColor(ORBIT_THEME['space_bg'])
            self.gl_widget.opts['distance'] = 8000  # Initial camera distance (km)
            self.gl_widget.opts['elevation'] = 28
            self.gl_widget.opts['azimuth'] = 45

            self._add_axes()
            self._create_moon()
        except Exception as exc:
            # A 3D preview must never prevent the mission window from opening.
            print(f"[3D Viz] GL initialization failed, using fallback: {exc}")
            self.gl_widget = None
            self._install_fallback(layout)
            return

        layout.addWidget(self.gl_widget, 1)
        layout.addLayout(self._build_camera_bar())

        # Initial draw
        QtCore.QTimer.singleShot(100, self.update_orbit)

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

    def _build_camera_bar(self) -> QtWidgets.QHBoxLayout:
        """Compact camera-preset row: Reset | Top | Side | Iso | Fit."""
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 4)
        bar.setSpacing(6)

        presets = (
            ("Reset", "Default mission view", self.reset_view),
            ("Top", "Look straight down +Z", self._view_top),
            ("Side", "Equatorial side view", self._view_side),
            ("Iso", "Isometric mission view", self._view_iso),
            ("Fit", "Frame the whole orbit", self._view_fit),
        )
        self._cam_buttons = []
        for label, tip, handler in presets:
            btn = QtWidgets.QPushButton(label)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFixedHeight(26)
            btn.setMinimumWidth(46)
            btn.setToolTip(tip)
            btn.clicked.connect(handler)
            self._cam_buttons.append(btn)
            bar.addWidget(btn)
        bar.addStretch(1)
        return bar

    def _install_fallback(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Designed empty-state shown when OpenGL is unavailable."""
        card = EmptyState(
            "3D preview unavailable",
            "The orbit can still be configured normally — only the interactive "
            "3D preview needs OpenGL, which is not available on this system.\n\n"
            "Install support:   pip install pyqtgraph PyOpenGL",
        )
        layout.addWidget(card)

    def _add_axes(self):
        """Add muted ECI reference axes — orientation hints, not the focus.

        Colors come from ``ORBIT_THEME`` (desaturated, translucent red/teal/blue)
        so the axes read as quiet guides against the dark lunar scene.
        """
        axis_len = 3600.0
        specs = (
            (ORBIT_THEME['axis_x'], (axis_len, 0.0, 0.0), 'X'),
            (ORBIT_THEME['axis_y'], (0.0, axis_len, 0.0), 'Y'),
            (ORBIT_THEME['axis_z'], (0.0, 0.0, axis_len), 'Z'),
        )
        for css, end, name in specs:
            color = rgba_css_to_tuple(css)
            axis = gl.GLLinePlotItem(
                pos=np.array([[0.0, 0.0, 0.0], list(end)]),
                color=color, width=2.2, antialias=True,
            )
            axis.setGLOptions('translucent')
            self.gl_widget.addItem(axis)
            # Crisp, fully-opaque X/Y/Z label so orientation reads clearly.
            try:
                label_color = (
                    int(round(color[0] * 255)),
                    int(round(color[1] * 255)),
                    int(round(color[2] * 255)),
                    255,
                )
                label = gl.GLTextItem(
                    pos=np.array(end, dtype=float) * 1.05, text=name, color=label_color
                )
                self.gl_widget.addItem(label)
            except Exception:
                pass

    def _create_moon(self):
        """Create the Moon as a softly shaded regolith sphere.

        A higher-resolution smooth mesh removes the faceted look and the
        'shaded' lighting model gives a natural lit/terminator gradient instead
        of the previous flat toy-grey. Faint guide rings add a scientific feel.
        """
        md = gl.MeshData.sphere(rows=48, cols=96, radius=R_MOON)

        base = hex_to_rgba_float(ORBIT_THEME['moon_mid'])
        colors = np.empty((md.faceCount(), 4), dtype=float)
        colors[:] = base
        md.setFaceColors(colors)

        self.moon_mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            shader='shaded',
            glOptions='opaque',
        )
        self.gl_widget.addItem(self.moon_mesh)
        self._add_moon_grid()

    def _add_moon_grid(self):
        """Add very faint latitude/longitude guide rings for a scientific look."""
        ring_color = hex_to_rgba_float(ORBIT_THEME['moon_light'], 0.10)
        r = R_MOON * 1.001
        n = 120
        t = np.linspace(0.0, 2.0 * np.pi, n)

        # Latitude rings
        for lat_deg in (-60.0, -30.0, 0.0, 30.0, 60.0):
            lat = np.deg2rad(lat_deg)
            rr = r * np.cos(lat)
            z = r * np.sin(lat)
            pts = np.column_stack([rr * np.cos(t), rr * np.sin(t), np.full_like(t, z)])
            ring = gl.GLLinePlotItem(pos=pts, color=ring_color, width=1.0, antialias=True)
            ring.setGLOptions('translucent')
            self.gl_widget.addItem(ring)

        # Meridians
        phi = np.linspace(-np.pi, np.pi, n)
        for lon_deg in (0.0, 45.0, 90.0, 135.0):
            lon = np.deg2rad(lon_deg)
            pts = np.column_stack([
                r * np.cos(phi) * np.cos(lon),
                r * np.cos(phi) * np.sin(lon),
                r * np.sin(phi),
            ])
            mer = gl.GLLinePlotItem(pos=pts, color=ring_color, width=1.0, antialias=True)
            mer.setGLOptions('translucent')
            self.gl_widget.addItem(mer)

    # -------------------------------------------------------------------------
    # Orbital geometry
    # -------------------------------------------------------------------------

    @staticmethod
    def _rotation_matrix(inc_rad, raan_rad, argp_rad):
        """Perifocal (PQW) -> ECI rotation: R_z(raan) R_x(inc) R_z(argp)."""
        cr, sr = np.cos(raan_rad), np.sin(raan_rad)
        ci, si = np.cos(inc_rad), np.sin(inc_rad)
        cw, sw = np.cos(argp_rad), np.sin(argp_rad)
        return np.array([
            [cr * cw - sr * ci * sw, -cr * sw - sr * ci * cw, sr * si],
            [sr * cw + cr * ci * sw, -sr * sw + cr * ci * cw, -cr * si],
            [si * sw, si * cw, ci],
        ])

    def _kepler_to_cartesian(self, a_km, e, inc_rad, raan_rad, argp_rad, ta_rad=0.0):
        """Convert Keplerian elements to a full-orbit array of ECI points."""
        n_points = 360
        true_anomalies = np.linspace(0.0, 2.0 * np.pi, n_points)

        r = a_km * (1 - e ** 2) / (1 + e * np.cos(true_anomalies))
        pqw = np.stack([
            r * np.cos(true_anomalies),
            r * np.sin(true_anomalies),
            np.zeros_like(true_anomalies),
        ], axis=1)

        R = self._rotation_matrix(inc_rad, raan_rad, argp_rad)
        return pqw @ R.T

    def _eci_point_at_ta(self, ta_deg):
        """ECI position (km) for a single true anomaly using current elements."""
        ta = np.deg2rad(ta_deg)
        denom = 1 + self._e * np.cos(ta)
        if abs(denom) < 1e-9:
            denom = 1e-9
        r = self._a_km * (1 - self._e ** 2) / denom
        pqw = np.array([r * np.cos(ta), r * np.sin(ta), 0.0])
        R = self._rotation_matrix(
            np.deg2rad(self._inc_deg),
            np.deg2rad(self._raan_deg),
            np.deg2rad(self._argp_deg),
        )
        return R @ pqw

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
        """Refresh the orbit line, glow underlay, and the three markers.

        GL items are created once and then updated in place — the line/glow via
        ``setData`` and the markers via a transform — so dragging a parameter
        reuses existing OpenGL objects instead of churning new ones every frame.
        The camera is never touched here, so the view stays put as values change.
        """
        if not HAS_OPENGL or getattr(self, 'gl_widget', None) is None:
            return

        inc_rad = np.deg2rad(self._inc_deg)
        raan_rad = np.deg2rad(self._raan_deg)
        argp_rad = np.deg2rad(self._argp_deg)

        try:
            points = self._kepler_to_cartesian(
                self._a_km, self._e, inc_rad, raan_rad, argp_rad
            )

            # Glow underlay first, crisp line on top — reuse via setData().
            if self.orbit_glow is None:
                self.orbit_glow = gl.GLLinePlotItem(
                    pos=points,
                    color=hex_to_rgba_float(ORBIT_THEME['orbit_glow'], 0.35),
                    width=8.0,
                    antialias=True,
                    glOptions='translucent',
                )
                self.gl_widget.addItem(self.orbit_glow)
            else:
                self.orbit_glow.setData(pos=points)

            if self.orbit_line is None:
                self.orbit_line = gl.GLLinePlotItem(
                    pos=points,
                    color=hex_to_rgba_float(ORBIT_THEME['orbit_line'], 1.0),
                    width=3.2,
                    antialias=True,
                    glOptions='translucent',
                )
                self.gl_widget.addItem(self.orbit_line)
            else:
                self.orbit_line.setData(pos=points)

            # Markers at periapsis (nu=0), apoapsis (nu=180), and current nu.
            r_marker = self._marker_radius_km()
            self._update_marker('periapsis_marker', self._eci_point_at_ta(0.0),
                                ORBIT_THEME['periapsis'], r_marker)
            self._update_marker('apoapsis_marker', self._eci_point_at_ta(180.0),
                                ORBIT_THEME['apoapsis'], r_marker)
            self._update_marker('spacecraft_marker', self._eci_point_at_ta(self._ta_deg),
                                ORBIT_THEME['spacecraft'], r_marker * 0.72)

        except Exception as exc:
            print(f"[3D Viz] Error updating orbit: {exc}")

    def _marker_radius_km(self) -> float:
        """Marker radius (km) scaled to the orbit so it never looks cartoonish.

        Scales with the orbit extent (aposelene radius, floored at the Moon
        radius) and is clamped so markers stay visible on tight orbits without
        becoming huge balls on very large ones.
        """
        extent = max(self._a_km * (1.0 + self._e), R_MOON)
        return float(np.clip(0.006 * extent, 12.0, 40.0))

    def _update_marker(self, attr: str, point, color_token: str, radius: float):
        """Position/size a marker sphere stored under *attr*, reusing the mesh.

        The sphere mesh is built once as a unit sphere; subsequent updates only
        replace the item transform (scale + translate), so we never rebuild
        marker geometry just because a parameter changed.
        """
        marker = getattr(self, attr, None)
        if marker is None:
            md = gl.MeshData.sphere(rows=12, cols=24, radius=1.0)
            rgba = hex_to_rgba_float(color_token)
            colors = np.empty((md.faceCount(), 4), dtype=float)
            colors[:] = rgba
            md.setFaceColors(colors)
            marker = gl.GLMeshItem(
                meshdata=md, smooth=True, shader='shaded', glOptions='translucent'
            )
            self.gl_widget.addItem(marker)
            setattr(self, attr, marker)

        # Compose M = T * S so the unit sphere is scaled to *radius* then moved
        # to *point* (QMatrix4x4 post-multiplies, so translate is applied first).
        transform = QtGui.QMatrix4x4()
        transform.translate(float(point[0]), float(point[1]), float(point[2]))
        transform.scale(float(radius), float(radius), float(radius))
        marker.setTransform(transform)

    # -------------------------------------------------------------------------
    # Camera presets
    # -------------------------------------------------------------------------

    def _set_camera(self, **kwargs):
        if HAS_OPENGL and getattr(self, 'gl_widget', None) is not None:
            self.gl_widget.setCameraPosition(**kwargs)

    def reset_view(self, _checked: bool = False):
        """Reset camera to the default mission view."""
        self._set_camera(distance=8000, elevation=28, azimuth=45)

    def _view_top(self, _checked: bool = False):
        """Look straight down the +Z axis."""
        self._set_camera(elevation=90, azimuth=0)

    def _view_side(self, _checked: bool = False):
        """Equatorial side view."""
        self._set_camera(elevation=0, azimuth=0)

    def _view_iso(self, _checked: bool = False):
        """Isometric, mission-style view."""
        self._set_camera(distance=8000, elevation=26, azimuth=135)

    def _view_fit(self, _checked: bool = False):
        """Frame the whole orbit by setting distance from its aposelene radius."""
        ra = self._a_km * (1.0 + self._e)  # aposelene radius (km)
        self._set_camera(distance=max(3000.0, 2.6 * ra))



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
        self._compact_layout = None

        # The last orbit state that validated cleanly. Invalid or partial input
        # never overwrites it, so the preview keeps showing the last good orbit.
        self._last_valid_orbit: _OrbitState | None = None

        # Debounce timer: many rapid parameter changes (especially dragging)
        # collapse into a single preview/metric refresh, avoiding redraw stutter.
        self._orbit_update_timer = QtCore.QTimer(self)
        self._orbit_update_timer.setSingleShot(True)
        self._orbit_update_timer.setInterval(40)  # ms
        self._orbit_update_timer.timeout.connect(self._apply_orbit_update)

        self._build_ui()

    def _create_card(self, title: str, description: str = "") -> Section:
        """Factory for standard titled cards built on the shared ``Section`` primitive."""
        return Section(title, description)

    def _metric_chip(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
        """Return a compact ``(frame, value_label)`` metric chip for the info strip."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("orbitMetric")
        v = QtWidgets.QVBoxLayout(frame)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(1)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("orbitMetricLabel")
        v.addWidget(title_lbl)

        value_lbl = QtWidgets.QLabel("--")
        value_lbl.setObjectName("orbitMetricValue")
        v.addWidget(value_lbl)
        return frame, value_lbl

    def _build_ui(self):
        """Constructs the layout of the Orbit Configuration Page."""
        layout = QtWidgets.QGridLayout(self)
        self._page_layout = layout
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
        QtCore.QTimer.singleShot(0, self._update_responsive_layout)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _update_responsive_layout(self) -> None:
        """Stack the configuration and preview when the workspace is narrow."""
        available_width = self.width()
        ancestor = self.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QtWidgets.QAbstractScrollArea):
                available_width = ancestor.viewport().width()
                break
            ancestor = ancestor.parentWidget()
        compact = available_width < 1050
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        layout = self._page_layout
        layout.removeWidget(self.group_params)
        layout.removeWidget(self.group_viz)
        if compact:
            layout.addWidget(self.group_params, 0, 0)
            layout.addWidget(self.group_viz, 1, 0)
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(1, 0)
        else:
            layout.addWidget(self.group_params, 0, 0, 2, 1)
            layout.addWidget(self.group_viz, 0, 1, 2, 1)
            layout.setColumnStretch(0, 11)
            layout.setColumnStretch(1, 9)

    def _create_params_group(self) -> Section:
        """Orbit parameters card with Modern Segmented Control."""
        gb = self._create_card(
            "Initial Orbit State",
            "Choose the orbit entry style you prefer. Related values stay "
            "synchronized automatically so you can review the full orbit shape at a glance.",
        )
        layout = gb.content_layout
        layout.setSpacing(16)

        # A. Modern Segmented Control for Input Mode
        mode_container = QtWidgets.QWidget()
        mode_container.setObjectName("segmentedControl")
        mode_container.setFixedHeight(46)

        mode_layout = QtWidgets.QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(4, 4, 4, 4)
        mode_layout.setSpacing(0)

        # Create three styled buttons as segments
        self.btn_mode_altitude = QtWidgets.QPushButton("Altitude (hp/ha)")
        self.btn_mode_classical = QtWidgets.QPushButton("Classical (a/e)")
        self.btn_mode_circular = QtWidgets.QPushButton("Circular (alt)")

        for btn in (self.btn_mode_altitude, self.btn_mode_classical, self.btn_mode_circular):
            btn.setObjectName("segmentButton")
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
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

        # B. Parameter Form with Ghosting
        form_layout = QtWidgets.QGridLayout()
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(12)

        def add_param(row, label, widget, unit=""):
            lbl = QtWidgets.QLabel(label)
            lbl.setObjectName("fieldLabel")
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lbl.setBuddy(widget)
            if label and not widget.accessibleName():
                widget.setAccessibleName(label.rstrip(": ").strip())
            form_layout.addWidget(lbl, row, 0)
            form_layout.addWidget(widget, row, 1)
            if unit:
                lbl_unit = QtWidgets.QLabel(unit)
                lbl_unit.setObjectName("fieldUnit")
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
        orbit_shape_lbl.setObjectName("sectionTitle")
        form_layout.addWidget(orbit_shape_lbl, 0, 0, 1, 3)

        # Add to Form
        add_param(1, "Periselene Altitude (hp)", self.ent_hp, "km")
        add_param(2, "Aposelene Altitude (ha)", self.ent_ha, "km")
        add_param(3, "Semi-major Axis (a)", self.ent_a, "km")
        add_param(4, "Eccentricity (e)", self.ent_e, "")
        add_param(5, "Circular Altitude", self.ent_alt_circular, "km")

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Plain)
        form_layout.addWidget(sep, 6, 0, 1, 3)

        orientation_lbl = QtWidgets.QLabel("Plane and orientation")
        orientation_lbl.setObjectName("sectionTitle")
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

    def _create_viz_group(self) -> Section:
        """3D orbit preview card."""
        gb = self._create_card(
            "Orbit Preview",
            "Two-body Keplerian preview for geometry, viewing angle, and period — "
            "the mission run adds the selected perturbations, so the propagated "
            "trajectory will differ from this idealized orbit.",
        )
        layout = gb.content_layout
        layout.setSpacing(14)

        self.orbit_viz_3d = OrbitViz3D()
        self.orbit_viz_3d.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.orbit_viz_3d)

        # Compact metric strip — period, periselene, aposelene, e, i, energy.
        info_frame = QtWidgets.QFrame()
        info_frame.setObjectName("orbitInfoStrip")
        info_grid = QtWidgets.QGridLayout(info_frame)
        info_grid.setContentsMargins(0, 0, 0, 0)
        info_grid.setHorizontalSpacing(8)
        info_grid.setVerticalSpacing(8)

        chip_period, self.lbl_period = self._metric_chip("Period (h)")
        chip_hp, self.lbl_hp = self._metric_chip("Periselene (km)")
        chip_ha, self.lbl_ha = self._metric_chip("Aposelene (km)")
        chip_e, self.lbl_ecc = self._metric_chip("Eccentricity")
        chip_i, self.lbl_inc = self._metric_chip("Inclination (°)")
        chip_energy, self.lbl_energy = self._metric_chip("Energy (km²/s²)")

        for col, chip in enumerate((chip_period, chip_hp, chip_ha)):
            info_grid.addWidget(chip, 0, col)
            info_grid.setColumnStretch(col, 1)
        for col, chip in enumerate((chip_e, chip_i, chip_energy)):
            info_grid.addWidget(chip, 1, col)

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
            field.setProperty("ghost", False)
            field.setEnabled(True)
            field.style().unpolish(field)
            field.style().polish(field)

        # Set ghost fields
        for field in ghost_fields:
            field.setReadOnly(True)
            field.setStyleSheet("")
            field.setProperty("ghost", True)
            field.setEnabled(True)
            field.style().unpolish(field)
            field.style().polish(field)

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
        """Schedule a debounced preview/metric refresh.

        Rapid parameter changes (dragging, ghost back-fills, session loads) all
        funnel through one short single-shot timer so the heavy redraw happens
        once per burst rather than on every individual ``value_changed``.
        """
        if not hasattr(self, "orbit_viz_3d"):
            return
        self._orbit_update_timer.start()

    def _compute_orbit_state(self) -> _OrbitState | None:
        """Validate the current inputs into an :class:`_OrbitState`.

        Returns ``None`` when the active-mode inputs are empty, malformed, or
        physically impossible. Pure (no rendering) so it can be tested directly
        and so both the 3D preview and the metric strip consume identical state.
        """
        try:
            if self.btn_mode_circular.isChecked():
                alt_text = self.ent_alt_circular.text().strip()
                if not alt_text:
                    return None
                a_km = R_MOON + float(alt_text)
                e = 0.0
            elif self.btn_mode_altitude.isChecked():
                hp_text = self.ent_hp.text().strip()
                if not hp_text:
                    return None
                ha_text = self.ent_ha.text().strip()
                hp = float(hp_text)
                ha = float(ha_text) if ha_text else hp
                rp = R_MOON + hp
                ra = R_MOON + ha
                if rp > ra:
                    rp, ra = ra, rp
                a_km = (rp + ra) / 2.0
                e = (ra - rp) / (ra + rp) if (ra + rp) > 0 else 0.0
            else:
                a_text = self.ent_a.text().strip()
                if not a_text:
                    return None
                a_km = float(a_text)
                e_text = self.ent_e.text().strip()
                e = float(e_text) if e_text else 0.0

            inc_deg = float(self.ent_inc.text() or 90.0)
            raan_deg = float(self.ent_raan.text() or 0.0)
            argp_deg = float(self.ent_argp.text() or 0.0)
            ta_deg = float(self.ent_ta.text() or 0.0)
        except (ValueError, TypeError):
            return None

        if not math.isfinite(a_km) or a_km <= 0.0:
            return None
        e = max(0.0, min(0.999, e))
        return _OrbitState(a_km, e, inc_deg, raan_deg, argp_deg, ta_deg)

    def _apply_orbit_update(self) -> None:
        """Apply the latest validated orbit to the preview and metric strip."""
        if not hasattr(self, "orbit_viz_3d"):
            return
        state = self._compute_orbit_state()
        if state is None:
            # Keep the last valid preview rather than blanking on partial input.
            return
        self._last_valid_orbit = state
        self.orbit_viz_3d.set_orbit_params(
            state.a_km, state.e, state.inc_deg,
            state.raan_deg, state.argp_deg, state.ta_deg,
        )
        self._update_metric_strip(state)

    def _update_metric_strip(self, state: _OrbitState) -> None:
        """Refresh the period/geometry/energy chips from a validated state."""
        mu = MU_MOON_KM3_S2  # km³/s², derived from lunaris.common.constants
        a_km, e = state.a_km, state.e
        period_h = (2.0 * math.pi * (a_km ** 3 / mu) ** 0.5) / 3600.0
        self.lbl_period.setText(f"{period_h:.2f}")
        self.lbl_hp.setText(f"{a_km * (1.0 - e) - R_MOON:,.0f}")
        self.lbl_ha.setText(f"{a_km * (1.0 + e) - R_MOON:,.0f}")
        self.lbl_ecc.setText(f"{e:.4f}")
        self.lbl_inc.setText(f"{state.inc_deg:.1f}")
        self.lbl_energy.setText(f"{-mu / (2.0 * a_km):.3f}")

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
