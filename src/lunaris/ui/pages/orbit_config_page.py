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

import contextlib
import math
import os
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# Qt's public C++ QWIDGETSIZE_MAX constant is not exported by PySide6.
_QWIDGETSIZE_MAX = 16_777_215

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
    from lunaris.ui.components.primitives import InlineNotice, Section
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

class OrbitSchematic2D(QtWidgets.QWidget):
    """Token-driven 2D orbit schematic for offscreen or no-OpenGL environments."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self._a_km = 2000.0
        self._e = 0.0
        self._inc_deg = 90.0
        self._raan_deg = 0.0
        self._argp_deg = 0.0
        self._ta_deg = 0.0

    @staticmethod
    def _qcolor(token: str, alpha: float | None = None) -> QtGui.QColor:
        r, g, b, a = rgba_css_to_tuple(token)
        return QtGui.QColor(
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
            int(round((a if alpha is None else alpha) * 255)),
        )

    def set_orbit_params(
        self,
        a_km: float,
        e: float,
        inc_deg: float,
        raan_deg: float,
        argp_deg: float,
        ta_deg: float,
    ) -> None:
        self._a_km = max(1.0, float(a_km))
        self._e = max(0.0, min(0.99, float(e)))
        self._inc_deg = float(inc_deg)
        self._raan_deg = float(raan_deg)
        self._argp_deg = float(argp_deg)
        self._ta_deg = float(ta_deg)
        self.update()

    def _orbit_xy(self, anomalies: np.ndarray) -> np.ndarray:
        denom = 1.0 + self._e * np.cos(anomalies)
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        radius = self._a_km * (1.0 - self._e**2) / denom
        xy = np.column_stack([radius * np.cos(anomalies), radius * np.sin(anomalies)])

        # Inclination is a visual cue in this fallback, not a true camera
        # projection. Keep polar orbits legible instead of collapsing them.
        inc = math.radians(self._inc_deg)
        xy[:, 1] *= 0.58 + 0.42 * abs(math.cos(inc))
        theta = math.radians(self._raan_deg + self._argp_deg)
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        return xy @ rot.T

    def _point_xy(self, ta_deg: float) -> np.ndarray:
        return self._orbit_xy(np.array([math.radians(ta_deg)]))[0]

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, self._qcolor(ORBIT_THEME["space_bg"]))

        anomalies = np.linspace(0.0, 2.0 * math.pi, 361)
        xy = self._orbit_xy(anomalies)
        extent = max(float(np.max(np.linalg.norm(xy, axis=1))), R_MOON * 1.08)
        scale = min(
            max(0.01, (rect.width() - 112.0) / (2.0 * extent)),
            max(0.01, (rect.height() - 92.0) / (2.0 * extent)),
        )
        center = QtCore.QPointF(rect.center().x(), rect.center().y() + 4)

        def to_screen(point: np.ndarray) -> QtCore.QPointF:
            return QtCore.QPointF(
                center.x() + float(point[0]) * scale,
                center.y() - float(point[1]) * scale,
            )

        painter.setPen(QtGui.QPen(self._qcolor(THEME["grid_color"], 0.42), 1.0))
        for frac in (0.35, 0.60, 0.85):
            grid_radius = extent * frac * scale
            painter.drawEllipse(center, grid_radius, grid_radius)
        painter.drawLine(rect.left() + 18, center.y(), rect.right() - 18, center.y())
        painter.drawLine(center.x(), rect.top() + 34, center.x(), rect.bottom() - 18)

        moon_radius = max(13.0, R_MOON * scale)
        moon_grad = QtGui.QRadialGradient(
            center.x() - moon_radius * 0.35,
            center.y() - moon_radius * 0.35,
            moon_radius * 1.25,
        )
        moon_grad.setColorAt(0.0, self._qcolor(ORBIT_THEME["moon_light"], 0.96))
        moon_grad.setColorAt(0.55, self._qcolor(ORBIT_THEME["moon_mid"], 0.94))
        moon_grad.setColorAt(1.0, self._qcolor(ORBIT_THEME["moon_dark"], 0.98))
        painter.setPen(QtGui.QPen(self._qcolor(ORBIT_THEME["moon_light"], 0.34), 1.0))
        painter.setBrush(QtGui.QBrush(moon_grad))
        painter.drawEllipse(center, moon_radius, moon_radius)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(self._qcolor(ORBIT_THEME["moon_light"], 0.16), 1.0))
        for factor in (0.35, 0.65):
            painter.drawEllipse(center, moon_radius * factor, moon_radius * factor)
        painter.drawLine(
            QtCore.QPointF(center.x() - moon_radius, center.y()),
            QtCore.QPointF(center.x() + moon_radius, center.y()),
        )

        orbit_path = QtGui.QPainterPath(to_screen(xy[0]))
        for point in xy[1:]:
            orbit_path.lineTo(to_screen(point))
        painter.setPen(QtGui.QPen(self._qcolor(ORBIT_THEME["orbit_glow"], 0.30), 7.0))
        painter.drawPath(orbit_path)
        painter.setPen(QtGui.QPen(self._qcolor(ORBIT_THEME["orbit_line"]), 2.4))
        painter.drawPath(orbit_path)

        self._draw_marker(painter, to_screen(self._point_xy(0.0)), "P", ORBIT_THEME["periapsis"])
        self._draw_marker(painter, to_screen(self._point_xy(180.0)), "A", ORBIT_THEME["apoapsis"])
        # Spacecraft label goes below its marker: at ta=0/180 the spacecraft
        # coincides with an apsis marker and an above-right label would
        # overprint "P"/"A" into an illegible glyph.
        self._draw_marker(
            painter,
            to_screen(self._point_xy(self._ta_deg)),
            "SC",
            ORBIT_THEME["spacecraft"],
            below=True,
        )

        self._draw_axis_glyph(painter, rect)
        self._draw_readout(painter, rect)

    def _draw_marker(
        self,
        painter: QtGui.QPainter,
        point: QtCore.QPointF,
        label: str,
        token: str,
        below: bool = False,
    ) -> None:
        color = self._qcolor(token)
        painter.setBrush(color)
        painter.setPen(QtGui.QPen(self._qcolor(ORBIT_THEME["space_bg"], 0.85), 1.0))
        painter.drawEllipse(point, 4.5, 4.5)
        painter.setPen(QtGui.QPen(color, 1.0))
        offset = QtCore.QPointF(8, 16) if below else QtCore.QPointF(8, -8)
        painter.drawText(point + offset, label)

    def _draw_axis_glyph(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        origin = QtCore.QPointF(rect.left() + 34, rect.bottom() - 30)
        painter.setPen(QtGui.QPen(self._qcolor(ORBIT_THEME["axis_x"]), 1.8))
        painter.drawLine(origin, origin + QtCore.QPointF(36, 0))
        painter.drawText(origin + QtCore.QPointF(41, 4), "X")
        painter.setPen(QtGui.QPen(self._qcolor(ORBIT_THEME["axis_y"]), 1.8))
        painter.drawLine(origin, origin + QtCore.QPointF(0, -30))
        painter.drawText(origin + QtCore.QPointF(-4, -36), "Y")

    def _draw_readout(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        readout = (
            f"ECI schematic   a={self._a_km:,.0f} km   "
            f"e={self._e:.4f}   i={self._inc_deg:.1f} deg"
        )
        text_rect = QtCore.QRectF(rect.left() + 14, rect.top() + 10, rect.width() - 28, 24)
        painter.setPen(QtGui.QPen(self._qcolor(THEME["fg_soft"]), 1.0))
        painter.drawText(text_rect, int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter), readout)


class OrbitViz3D(QtWidgets.QWidget):
    """
    3D orbit visualizer using pyqtgraph's OpenGL backend.

    Renders the Moon as a softly shaded regolith sphere with faint lat/long
    guide rings, the orbital trajectory as a clean translucent arc with an
    optional glow underlay, and labelled markers for periapsis, apoapsis, and
    the current spacecraft position (true anomaly).  All colors come from
    ``ORBIT_THEME`` so the preview matches the Lunar Graphite palette.
    """


    focus_mode_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # A hard minimum the compact (stacked) layout cannot honor makes the
        # section layout place the metric chips *over* the scene, so keep the
        # floor low enough for a 900px-tall window with the form above.
        self.setMinimumSize(360, 260)

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

        # Mission-control annotation layers (apsides/nodes lines, node markers,
        # orbital-plane disk, velocity arrow, and floating text labels). Created
        # on demand inside ``_update_annotations`` and toggled via the layer
        # checkboxes built in ``_build_controls``.
        self.plane_disk = None
        self.apsides_line = None
        self.nodes_line = None
        self.asc_node_marker = None
        self.desc_node_marker = None
        self.vel_arrow_line = None
        self.vel_arrow_cone = None
        self.starfield = None
        self._labels: dict = {}
        # Layer visibility (mirrors the checkbox states; defaults all on).
        self._layers = {"labels": True, "nodes": True, "velocity": True, "plane": True}
        # True once the camera has framed the first valid orbit (so the scene
        # fills the viewport instead of floating as a small moon in a dark void),
        # after which the user's manual camera moves are respected.
        self._did_autofit = False
        self._camera_user_modified = False
        self._fit_resize_pending = False
        self._compact_controls: bool | None = None
        self._focus_mode_active = False
        # Set when the line of nodes collapses onto the line of apsides
        # (argp ~ 0 or 180): the node markers/labels are then redundant and are
        # suppressed so "Apoapsis" and "AN" stop colliding into illegible mush.
        self._nodes_coincident = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
        if platform in {"offscreen", "minimal", "vnc"}:
            self._install_fallback(layout)
            return

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

            self._add_starfield()
            self._add_axes()
            self._create_moon()
        except Exception as exc:
            # A 3D preview must never prevent the mission window from opening.
            print(f"[3D Viz] GL initialization failed, using fallback: {exc}")
            self.gl_widget = None
            self._install_fallback(layout)
            return

        self._install_scene_interactions(self.gl_widget)
        layout.addWidget(self.gl_widget, 1)
        layout.addWidget(self._build_controls())

        # Initial draw
        QtCore.QTimer.singleShot(100, self.update_orbit)

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QFrame:
        """Two compact control rows below the scene.

        Row 1 — camera presets (Reset | Top | Side | Iso | Fit).
        Row 2 — mission-control layer toggles (Labels | Nodes | Velocity | Plane)
        so the operator can declutter the schematic to taste, the way STK/GMAT
        let you switch annotation layers on and off.
        """
        from lunaris.ui.theme.tokens import DESIGN_TOKENS

        panel = QtWidgets.QFrame()
        panel.setObjectName("orbitControlBar")
        col = QtWidgets.QVBoxLayout(panel)
        col.setContentsMargins(10, 8, 10, 8)
        col.setSpacing(6)

        # --- Row 1: camera presets ---
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(6)
        view_lbl = QtWidgets.QLabel("View")
        view_lbl.setObjectName("orbitControlLabel")
        bar.addWidget(view_lbl)

        presets = (
            ("Reset", "Default mission view", self.reset_view),
            ("Top", "Look straight down +Z", self._view_top),
            ("Side", "Equatorial side view", self._view_side),
            ("Iso", "Isometric mission view", self._view_iso),
        )
        self._cam_buttons = []
        self._camera_button_group = QtWidgets.QButtonGroup(self)
        self._camera_button_group.setExclusive(True)
        for label, tip, handler in presets:
            btn = QtWidgets.QToolButton()
            btn.setText(label)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            btn.setObjectName("orbitPresetBtn")
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFixedHeight(DESIGN_TOKENS.controls.compact_height)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.clicked.connect(handler)
            self._camera_button_group.addButton(btn)
            self._cam_buttons.append(btn)
            bar.addWidget(btn)

        self._cam_buttons[0].setChecked(True)
        bar.addSpacing(DESIGN_TOKENS.spacing.sm)

        self._fit_button = QtWidgets.QToolButton()
        self._fit_button.setText("Fit orbit")
        self._fit_button.setObjectName("orbitFitBtn")
        self._fit_button.setIcon(get_icon("fa6s.expand", THEME["fg_soft"]))
        self._fit_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._fit_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._fit_button.setFixedHeight(DESIGN_TOKENS.controls.compact_height)
        self._fit_button.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        self._fit_button.setToolTip("Frame the whole orbit (F)")
        self._fit_button.clicked.connect(self._view_fit)
        bar.addWidget(self._fit_button)

        bar.addStretch(1)
        col.addLayout(bar)

        # --- Row 2: annotation layer toggles ---
        layers = QtWidgets.QHBoxLayout()
        layers.setSpacing(DESIGN_TOKENS.spacing.md)
        layer_lbl = QtWidgets.QLabel("Layers")
        layer_lbl.setObjectName("orbitControlLabel")
        layers.addWidget(layer_lbl)

        self._layer_checks = {}
        layer_specs = (
            ("labels", "Labels", "Floating periapsis / apoapsis / node / S-C labels"),
            ("nodes", "Nodes", "Ascending and descending node markers and line of nodes"),
            ("velocity", "Velocity", "Direction-of-motion arrow at the spacecraft"),
            ("plane", "Plane", "Translucent orbital-plane disk and line of apsides"),
        )
        for key, text, tip in layer_specs:
            chk = QtWidgets.QCheckBox(text)
            chk.setChecked(self._layers.get(key, True))
            chk.setToolTip(tip)
            chk.setCursor(QtCore.Qt.PointingHandCursor)
            chk.toggled.connect(lambda on, k=key: self._on_layer_toggled(k, on))
            self._layer_checks[key] = chk
            layers.addWidget(chk)
        layers.addStretch(1)

        self._focus_button = QtWidgets.QToolButton()
        self._focus_button.setText("Focus")
        self._focus_button.setObjectName("orbitFocusBtn")
        self._focus_button.setIcon(get_icon("fa6s.expand", THEME["fg_soft"]))
        self._focus_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._focus_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._focus_button.setFixedHeight(DESIGN_TOKENS.controls.compact_height)
        self._focus_button.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        self._focus_button.setToolTip("Expand the orbit preview; double-click also toggles")
        self._focus_button.clicked.connect(
            lambda _checked=False: self.focus_mode_requested.emit()
        )
        layers.addWidget(self._focus_button)
        col.addLayout(layers)

        QtCore.QTimer.singleShot(0, self._update_control_density)

        return panel

    def _update_control_density(self) -> None:
        """Shorten secondary actions before a narrow toolbar can clip them."""
        if not hasattr(self, "_focus_button"):
            return
        compact = self.width() < 620
        if compact == self._compact_controls:
            return
        self._compact_controls = compact
        self._fit_button.setText("Fit" if compact else "Fit orbit")
        self._focus_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonIconOnly
            if compact
            else QtCore.Qt.ToolButtonTextBesideIcon
        )
        self._refresh_focus_button()

    def _refresh_focus_button(self) -> None:
        active = self._focus_mode_active
        self._focus_button.setText("Exit focus" if active else "Focus")
        self._focus_button.setIcon(
            get_icon(
                "fa6s.compress" if active else "fa6s.expand",
                THEME["fg_soft"],
            )
        )
        self._focus_button.setAccessibleName(
            "Exit orbit preview focus mode" if active else "Enter orbit preview focus mode"
        )
        self._focus_button.setToolTip(
            "Restore the orbit form (Esc or double-click)"
            if active
            else "Expand the orbit preview; double-click also toggles"
        )

    def _install_scene_interactions(self, scene: QtWidgets.QWidget) -> None:
        """Expose useful power-user actions without hiding the primary controls."""
        self._scene_widget = scene
        scene.setFocusPolicy(QtCore.Qt.StrongFocus)
        scene.setAccessibleName("Interactive orbit preview")
        scene.setAccessibleDescription(
            "Drag to orbit the camera, use the wheel to zoom, press F to fit, "
            "or double-click to toggle preview focus mode."
        )
        scene.setToolTip(
            "Drag to orbit · Wheel to zoom · F to fit · Double-click to focus"
        )
        scene.installEventFilter(self)

        self._fit_shortcut = QtGui.QShortcut(QtGui.QKeySequence("F"), scene)
        self._fit_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._fit_shortcut.activated.connect(self._view_fit)

    def eventFilter(self, watched, event):
        if watched is getattr(self, "_scene_widget", None):
            if event.type() in (
                QtCore.QEvent.MouseButtonPress,
                QtCore.QEvent.Wheel,
            ):
                self._camera_user_modified = True
            elif event.type() == QtCore.QEvent.MouseButtonDblClick:
                self.focus_mode_requested.emit()
            elif event.type() == QtCore.QEvent.Resize:
                self._update_control_density()
                self._schedule_auto_refit()
        return super().eventFilter(watched, event)

    def _schedule_auto_refit(self) -> None:
        if (
            not self._did_autofit
            or self._camera_user_modified
            or self._fit_resize_pending
        ):
            return
        self._fit_resize_pending = True

        def apply_fit() -> None:
            self._fit_resize_pending = False
            if not self._camera_user_modified:
                self._fit_camera(mark_user=False)

        QtCore.QTimer.singleShot(0, apply_fit)

    def set_focus_mode(self, active: bool) -> None:
        """Keep the reversible focus-mode control honest and discoverable."""
        self._focus_mode_active = bool(active)
        self._refresh_focus_button()

    def _on_layer_toggled(self, key: str, on: bool) -> None:
        """Persist a layer toggle and refresh the scene's annotation visibility."""
        self._layers[key] = bool(on)
        self._refresh_layer_visibility()

    def _add_starfield(self):
        """Scatter a faint, fixed starfield on a far shell for depth.

        A flat near-black void reads as an empty CAD viewport; a subtle starfield
        gives the scene a sense of space and scale without competing with the
        orbit. The points sit far enough out that they never intersect the orbit
        and a fixed RNG seed keeps the pattern stable across redraws.
        """
        rng = np.random.default_rng(42)
        n = 520
        # Uniform directions on a sphere, pushed to a far radius.
        vecs = rng.normal(size=(n, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        radius = 60000.0
        pos = vecs * radius
        base = hex_to_rgba_float(ORBIT_THEME['moon_light'], 1.0)
        colors = np.empty((n, 4), dtype=float)
        colors[:] = base
        # Vary brightness so the field looks natural rather than a flat dot grid.
        colors[:, 3] = rng.uniform(0.18, 0.7, size=n)
        sizes = rng.uniform(1.0, 2.6, size=n)
        self.starfield = gl.GLScatterPlotItem(
            pos=pos, color=colors, size=sizes, pxMode=True
        )
        self.starfield.setGLOptions('additive')
        self.gl_widget.addItem(self.starfield)

    def _install_fallback(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Install a deterministic engineering schematic when GL cannot paint."""
        self.schematic_widget = OrbitSchematic2D()
        self._install_scene_interactions(self.schematic_widget)
        layout.addWidget(self.schematic_widget, 1)
        controls = self._build_controls()
        for button in (*self._cam_buttons, self._fit_button):
            button.setEnabled(False)
            button.setToolTip("Camera presets require the interactive 3D renderer")
        for checkbox in self._layer_checks.values():
            checkbox.setEnabled(False)
            checkbox.setToolTip("Scene layers require the interactive 3D renderer")
        layout.addWidget(controls)

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

        # A darker regolith base (was the lighter ``moon_mid``) keeps the sphere
        # from reading as a flat bright-grey ball that drowns out the orbit. The
        # 'shaded' shader then gives a natural lit/terminator gradient and the
        # crisp blue orbit line reads clearly against it.
        base = hex_to_rgba_float(ORBIT_THEME['moon_dark'])
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
        schematic = getattr(self, "schematic_widget", None)
        if schematic is not None:
            schematic.set_orbit_params(
                self._a_km,
                self._e,
                self._inc_deg,
                self._raan_deg,
                self._argp_deg,
                self._ta_deg,
            )
            return

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

            # Mission-control annotations: apsides/nodes lines, node markers,
            # orbital-plane disk, velocity arrow, and floating labels.
            self._update_annotations(r_marker)

            # Frame the first valid orbit so it fills the viewport; later edits
            # leave the camera where the user put it.
            if not self._did_autofit:
                self._did_autofit = True
                self._fit_camera(mark_user=False)

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
    # Mission-control annotations
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_disk_meshdata(n: int = 96):
        """Build a unit-radius disk in the local XY plane (triangle fan)."""
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])
        verts = np.vstack([[0.0, 0.0, 0.0], ring])
        # Ring vertices are 1..n (0 is the centre); the modulo wraps the last
        # face back to ring vertex 1 so no face can index past the ring.
        faces = np.array(
            [[0, i + 1, ((i + 1) % n) + 1] for i in range(n)], dtype=int
        )
        return gl.MeshData(vertexes=verts, faces=faces)

    def _perifocal_velocity_dir(self, ta_deg: float) -> np.ndarray:
        """Unit velocity direction (ECI) at the given true anomaly.

        In the perifocal frame the velocity is parallel to
        ``(-sin nu, e + cos nu, 0)``; rotating that by the orbit's PQW->ECI
        matrix gives the in-plane direction of motion shown by the arrow.
        """
        ta = np.deg2rad(ta_deg)
        v_pqw = np.array([-np.sin(ta), self._e + np.cos(ta), 0.0])
        norm = np.linalg.norm(v_pqw)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        v_pqw /= norm
        R = self._rotation_matrix(
            np.deg2rad(self._inc_deg),
            np.deg2rad(self._raan_deg),
            np.deg2rad(self._argp_deg),
        )
        return R @ v_pqw

    def _ensure_line(self, attr: str, pos, color_token: str, alpha: float, width: float):
        """Create (once) or update a translucent annotation line stored at *attr*."""
        line = getattr(self, attr, None)
        color = hex_to_rgba_float(color_token, alpha)
        if line is None:
            line = gl.GLLinePlotItem(
                pos=pos, color=color, width=width, antialias=True,
                glOptions='translucent',
            )
            self.gl_widget.addItem(line)
            setattr(self, attr, line)
        else:
            line.setData(pos=pos, color=color, width=width)
        return line

    def _ensure_label(self, key: str, point, text: str, color_token: str):
        """Create (once) or reposition a floating text label."""
        rgba = hex_to_rgba_float(color_token, 1.0)
        col = (
            int(round(rgba[0] * 255)),
            int(round(rgba[1] * 255)),
            int(round(rgba[2] * 255)),
            255,
        )
        label = self._labels.get(key)
        if label is None:
            try:
                font = QtGui.QFont()
                font.setPointSize(9)
                label = gl.GLTextItem(
                    pos=np.asarray(point, dtype=float), text=text, color=col, font=font
                )
                self.gl_widget.addItem(label)
                self._labels[key] = label
            except Exception:
                return None
        else:
            with contextlib.suppress(Exception):
                label.setData(pos=np.asarray(point, dtype=float), text=text, color=col)
        return label

    def _update_annotations(self, r_marker: float) -> None:
        """Refresh apsides/nodes lines, node markers, plane disk, velocity arrow.

        Everything is derived from the same validated elements the orbit line
        uses, so the schematic stays self-consistent. Items are created once and
        then updated in place; per-layer visibility is applied at the end.
        """
        if not HAS_OPENGL or getattr(self, 'gl_widget', None) is None:
            return

        peri = self._eci_point_at_ta(0.0)
        apo = self._eci_point_at_ta(180.0)
        # Ascending node crosses the reference plane at argument of latitude 0
        # (nu = -argp); descending node at 180 deg further along.
        asc = self._eci_point_at_ta(-self._argp_deg)
        desc = self._eci_point_at_ta(180.0 - self._argp_deg)
        sc = self._eci_point_at_ta(self._ta_deg)

        # Line of apsides (major axis) — quiet, periapsis-tinted.
        self._ensure_line(
            'apsides_line', np.array([peri, apo]),
            ORBIT_THEME['periapsis'], 0.55, 1.6,
        )

        # Orbital-plane disk — a unit disk rotated into the orbit plane and
        # scaled to the aposelene radius, kept very translucent for context.
        ra = self._a_km * (1.0 + self._e)
        if self.plane_disk is None:
            md = self._make_disk_meshdata()
            self.plane_disk = gl.GLMeshItem(
                meshdata=md,
                color=hex_to_rgba_float(ORBIT_THEME['orbit_glow'], 0.07),
                glOptions='translucent', smooth=False, drawEdges=False,
            )
            self.gl_widget.addItem(self.plane_disk)
        R = self._rotation_matrix(
            np.deg2rad(self._inc_deg),
            np.deg2rad(self._raan_deg),
            np.deg2rad(self._argp_deg),
        )
        disk_tf = QtGui.QMatrix4x4()
        disk_tf.setColumn(0, QtGui.QVector4D(R[0, 0] * ra, R[1, 0] * ra, R[2, 0] * ra, 0.0))
        disk_tf.setColumn(1, QtGui.QVector4D(R[0, 1] * ra, R[1, 1] * ra, R[2, 1] * ra, 0.0))
        disk_tf.setColumn(2, QtGui.QVector4D(R[0, 2] * ra, R[1, 2] * ra, R[2, 2] * ra, 0.0))
        disk_tf.setColumn(3, QtGui.QVector4D(0.0, 0.0, 0.0, 1.0))
        self.plane_disk.setTransform(disk_tf)

        # Line of nodes (through the focus, ascending <-> descending).
        self._ensure_line(
            'nodes_line', np.array([desc, asc]),
            ORBIT_THEME['orbit_node'], 0.5, 1.6,
        )
        self._update_marker('asc_node_marker', asc, ORBIT_THEME['orbit_node'], r_marker * 0.6)
        self._update_marker('desc_node_marker', desc, ORBIT_THEME['orbit_node'], r_marker * 0.6)

        # Velocity arrow at the spacecraft — line + cone tip in the direction of
        # motion, scaled to the current orbital radius.
        v_dir = self._perifocal_velocity_dir(self._ta_deg)
        arrow_len = float(np.clip(0.45 * np.linalg.norm(sc), 200.0, 2400.0))
        cone_len = 0.26 * arrow_len
        base = np.asarray(sc, dtype=float) + v_dir * (arrow_len - cone_len)
        self._ensure_line(
            'vel_arrow_line', np.array([sc, base]),
            ORBIT_THEME['spacecraft'], 0.9, 2.4,
        )
        if self.vel_arrow_cone is None:
            cmd = gl.MeshData.cylinder(rows=2, cols=16, radius=[0.32 * cone_len, 0.0], length=cone_len)
            self.vel_arrow_cone = gl.GLMeshItem(
                meshdata=cmd, smooth=True, shader='shaded',
                color=hex_to_rgba_float(ORBIT_THEME['spacecraft'], 1.0),
                glOptions='opaque',
            )
            self.gl_widget.addItem(self.vel_arrow_cone)
        self.vel_arrow_cone.setTransform(self._cone_transform(base, v_dir))

        # When argp ~ 0 or 180 the nodes sit exactly on the apsides; labelling
        # both there produces the "Apoapsis"+"AN" overlap. Detect that and let
        # _refresh_layer_visibility suppress the redundant node annotations.
        argp_mod = self._argp_deg % 180.0
        self._nodes_coincident = min(argp_mod, 180.0 - argp_mod) < 6.0

        # Floating labels — pushed off their markers so the text never sits on
        # top of the sphere or another label. Apsis labels float radially
        # outward; node labels lift along the orbit normal; the S/C label rides
        # ahead of the velocity arrow.
        off = max(R_MOON * 0.05, 0.10 * self._a_km)

        def _unit(vec):
            v = np.asarray(vec, dtype=float)
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else v

        h_hat = _unit(R @ np.array([0.0, 0.0, 1.0]))
        self._ensure_label('peri', np.asarray(peri, float) + _unit(peri) * off,
                           "Periapsis", ORBIT_THEME['periapsis'])
        self._ensure_label('apo', np.asarray(apo, float) + _unit(apo) * off,
                           "Apoapsis", ORBIT_THEME['apoapsis'])
        self._ensure_label('asc', np.asarray(asc, float) + h_hat * off,
                           "AN", ORBIT_THEME['orbit_node'])
        self._ensure_label('desc', np.asarray(desc, float) - h_hat * off,
                           "DN", ORBIT_THEME['orbit_node'])
        self._ensure_label('sc', np.asarray(sc, dtype=float) + v_dir * (arrow_len * 1.15),
                           "S/C", ORBIT_THEME['spacecraft'])

        self._refresh_layer_visibility()

    @staticmethod
    def _cone_transform(base, direction) -> QtGui.QMatrix4x4:
        """Transform placing a +Z cone's base at *base*, apex along *direction*."""
        tf = QtGui.QMatrix4x4()
        tf.translate(float(base[0]), float(base[1]), float(base[2]))
        z = np.array([0.0, 0.0, 1.0])
        d = np.asarray(direction, dtype=float)
        nd = np.linalg.norm(d)
        if nd > 1e-12:
            d = d / nd
            axis = np.cross(z, d)
            axis_n = np.linalg.norm(axis)
            dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
            if axis_n > 1e-9:
                angle = math.degrees(math.acos(dot))
                tf.rotate(angle, float(axis[0]), float(axis[1]), float(axis[2]))
            elif dot < 0.0:
                # Antiparallel: flip 180 deg about any perpendicular axis.
                tf.rotate(180.0, 1.0, 0.0, 0.0)
        return tf

    def _refresh_layer_visibility(self) -> None:
        """Apply the current per-layer toggle states to the GL items."""
        show_labels = self._layers.get("labels", True)
        # Hide the node layer entirely when it has collapsed onto the apsides,
        # so we never draw a doubled marker or a colliding "AN"/"Apoapsis" pair.
        show_nodes = self._layers.get("nodes", True) and not self._nodes_coincident
        show_vel = self._layers.get("velocity", True)
        show_plane = self._layers.get("plane", True)

        for item in (self.nodes_line, self.asc_node_marker, self.desc_node_marker):
            if item is not None:
                item.setVisible(show_nodes)
        for item in (self.vel_arrow_line, self.vel_arrow_cone):
            if item is not None:
                item.setVisible(show_vel)
        for item in (self.plane_disk, self.apsides_line):
            if item is not None:
                item.setVisible(show_plane)
        for key, label in self._labels.items():
            if label is None:
                continue
            if key in ("asc", "desc"):
                label.setVisible(show_labels and show_nodes)
            else:
                label.setVisible(show_labels)

    # -------------------------------------------------------------------------
    # Camera presets
    # -------------------------------------------------------------------------

    def _set_camera(self, **kwargs):
        if HAS_OPENGL and getattr(self, 'gl_widget', None) is not None:
            self.gl_widget.setCameraPosition(**kwargs)

    def reset_view(self, _checked: bool = False):
        """Reset camera to the default mission view."""
        self._camera_user_modified = True
        self._set_camera(elevation=28, azimuth=45)
        self._fit_camera(mark_user=True)

    def _view_top(self, _checked: bool = False):
        """Look straight down the +Z axis."""
        self._camera_user_modified = True
        self._set_camera(elevation=90, azimuth=0)

    def _view_side(self, _checked: bool = False):
        """Equatorial side view."""
        self._camera_user_modified = True
        self._set_camera(elevation=0, azimuth=0)

    def _view_iso(self, _checked: bool = False):
        """Isometric, mission-style view."""
        self._camera_user_modified = True
        self._set_camera(elevation=26, azimuth=135)
        self._fit_camera(mark_user=True)

    def _view_fit(self, _checked: bool = False):
        self._fit_camera(mark_user=True)

    def _fit_camera(self, *, mark_user: bool) -> None:
        """Frame the whole orbit with margin, also keeping the Moon in view.

        The camera distance is driven by whichever is larger — the aposelene
        radius or the Moon — and uses a generous multiplier so the sphere and
        orbit sit comfortably inside the viewport instead of being clipped at
        the edges (the previous 2.6x framing cropped the Moon on low orbits).
        """
        if mark_user:
            self._camera_user_modified = True
        ra = self._a_km * (1.0 + self._e)  # aposelene radius (km)
        extent = max(ra, R_MOON)

        # GLViewWidget treats ``fov`` as horizontal. A wide, shallow preview
        # therefore has a much smaller vertical field of view than a square one;
        # a constant 3.4x multiplier clips the Moon at precisely the desktop
        # aspect ratios where the preview is most useful. Derive the vertical
        # half-angle from the live scene geometry and leave 18% annotation room.
        scene = getattr(self, "gl_widget", None)
        if scene is None:
            return
        width = max(scene.width(), 1)
        height = max(scene.height(), 1)
        aspect = width / height
        horizontal_fov = float(scene.opts.get("fov", 60.0))
        half_h = math.radians(horizontal_fov / 2.0)
        half_v = math.atan(math.tan(half_h) / max(aspect, 0.1))
        distance = 1.18 * extent / max(math.sin(half_v), 0.08)
        self._set_camera(distance=max(4200.0, distance))



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
        self._preview_focus_active = False
        self._focus_restore_sizes: list[int] | None = None

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

    def _create_card(
        self, title: str, description: str = "", *, elevated: bool = False
    ) -> Section:
        """Factory for standard titled cards built on the shared ``Section`` primitive."""
        return Section(title, description, elevated=elevated)

    def _metric_chip(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
        """Return a compact ``(frame, value_label)`` metric chip for the info strip."""
        from lunaris.ui.theme.tokens import DESIGN_TOKENS

        frame = QtWidgets.QFrame()
        frame.setObjectName("orbitMetric")
        v = QtWidgets.QVBoxLayout(frame)
        v.setContentsMargins(
            DESIGN_TOKENS.spacing.md, DESIGN_TOKENS.spacing.xs,
            DESIGN_TOKENS.spacing.md, DESIGN_TOKENS.spacing.xs,
        )
        # 1px inner gap keeps the value tight under its label (deliberate, off-scale).
        v.setSpacing(1)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("orbitMetricLabel")
        v.addWidget(title_lbl)

        value_lbl = QtWidgets.QLabel("--")
        value_lbl.setObjectName("orbitMetricValue")
        v.addWidget(value_lbl)
        return frame, value_lbl

    def _build_ui(self):
        """Two-pane workspace filling the full page width.

        Left: a *scrollable* parameter form. Right: a *fixed*, always-visible 3D
        preview. The preview no longer scrolls away with the form, so editing any
        element updates the orbit in place while it stays in view — and the
        splitter spans the whole width instead of a centred narrow column, so the
        old left/right dead space is gone. A draggable handle lets the operator
        rebalance form vs. preview.
        """
        root = QtWidgets.QVBoxLayout(self)
        self._page_layout = root
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left pane: the parameter form in its own scroll area.
        self.group_params = self._create_params_group()
        self._params_scroll = QtWidgets.QScrollArea()
        self._params_scroll.setObjectName("orbitParamsScroll")
        self._params_scroll.setWidgetResizable(True)
        self._params_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._params_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._params_scroll.setWidget(self.group_params)
        self._params_scroll.setMinimumWidth(440)
        # A scroll area's own minimum-size hint is tiny (~2 scrollbar widths),
        # so in the stacked compact layout the splitter could crush the whole
        # form into an unusable sliver. Guarantee roughly six visible form
        # rows in either orientation.
        from lunaris.ui.theme.tokens import DESIGN_TOKENS
        self._params_scroll.setMinimumHeight(
            DESIGN_TOKENS.controls.minimum_height * 7
        )

        # Right pane: the fixed 3D preview.
        self.group_viz = self._create_viz_group()
        self.group_viz.setMinimumWidth(520)
        self.group_viz.installEventFilter(self)

        self._split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._split.setObjectName("orbitSplit")
        self._split.setChildrenCollapsible(False)
        self._split.setHandleWidth(8)
        self._split.addWidget(self._params_scroll)
        self._split.addWidget(self.group_viz)
        self._split.setStretchFactor(0, 2)
        self._split.setStretchFactor(1, 3)
        self._split.splitterMoved.connect(lambda *_: self._update_metric_layout())
        root.addWidget(self._split)

        self._exit_focus_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        self._exit_focus_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._exit_focus_shortcut.activated.connect(self._exit_preview_focus)

        QtCore.QTimer.singleShot(0, self._update_responsive_layout)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout()

    def eventFilter(self, watched, event):
        if (
            watched is getattr(self, "group_viz", None)
            and event.type() == QtCore.QEvent.Resize
        ):
            QtCore.QTimer.singleShot(0, self._update_metric_layout)
        return super().eventFilter(watched, event)

    def _update_responsive_layout(self) -> None:
        """Stack the form above the preview only when the workspace is narrow."""
        if not hasattr(self, "_split"):
            return
        # The measured comfortable budget is 440px for the form, 520px for the
        # preview and an 8px handle. Below that, stacking preserves both surfaces.
        compact = self.width() < 968
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        if compact:
            self._split.setOrientation(QtCore.Qt.Vertical)
            self._params_scroll.setMaximumWidth(_QWIDGETSIZE_MAX)
            # Explicit sizes: without them the splitter leaves the form pane
            # at its (tiny) hint and the preview swallows the page.
            total = max(self.height(), 1)
            self._split.setSizes([int(total * 0.55), int(total * 0.45)])
        else:
            self._split.setOrientation(QtCore.Qt.Horizontal)
            # Scientific form rows stop becoming easier to read beyond this
            # width; donate the remaining workspace to the visual result.
            self._params_scroll.setMaximumWidth(620)
            total = max(self.width(), 1)
            self._split.setSizes([min(620, int(total * 0.40)), int(total * 0.60)])
        self._update_metric_layout()

    def _toggle_preview_focus(self) -> None:
        self._set_preview_focus(not self._preview_focus_active)

    def _exit_preview_focus(self) -> None:
        if self._preview_focus_active:
            self._set_preview_focus(False)

    def _set_preview_focus(self, active: bool) -> None:
        """Temporarily give the visual workspace the full canvas, reversibly."""
        active = bool(active)
        if active == self._preview_focus_active:
            return
        if active:
            self._focus_restore_sizes = self._split.sizes()
            self._params_scroll.hide()
        else:
            self._params_scroll.show()
            if self._focus_restore_sizes:
                self._split.setSizes(self._focus_restore_sizes)
        self._preview_focus_active = active
        self.orbit_viz_3d.set_focus_mode(active)
        QtCore.QTimer.singleShot(0, self._update_metric_layout)

    def _create_params_group(self) -> Section:
        """Orbit parameters card with Modern Segmented Control."""
        gb = self._create_card(
            "Initial Orbit State",
            "Entry mode determines which elements are editable; the rest are derived.",
        )
        layout = gb.content_layout
        layout.setSpacing(16)

        # A. Modern Segmented Control for Input Mode
        mode_container = QtWidgets.QWidget()
        mode_container.setObjectName("segmentedControl")
        mode_container.setFixedHeight(40)

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

        # Cap the input width so a short value like "100.0" doesn't float in a
        # huge box now that the form pane fills the full width; the slack becomes
        # a right margin (stretch column 3) instead of an over-wide field.
        form_layout.setColumnStretch(3, 1)

        self._param_labels: dict[QtWidgets.QWidget, tuple[QtWidgets.QLabel, str]] = {}

        def add_param(row, label, widget, unit=""):
            lbl = QtWidgets.QLabel(label)
            lbl.setObjectName("fieldLabel")
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lbl.setBuddy(widget)
            if label and not widget.accessibleName():
                widget.setAccessibleName(label.rstrip(": ").strip())
            from lunaris.ui.theme.tokens import DESIGN_TOKENS
            widget.setMaximumWidth(DESIGN_TOKENS.controls.input_width_standard)
            self._param_labels[widget] = (lbl, label)
            form_layout.addWidget(lbl, row, 0)
            form_layout.addWidget(widget, row, 1)
            if unit:
                lbl_unit = QtWidgets.QLabel(unit)
                lbl_unit.setObjectName("fieldUnit")
                form_layout.addWidget(lbl_unit, row, 2)

        # Input Fields - All are NumericDragLineEdit
        self.ent_hp = NumericDragLineEdit("100.0", step=5.0, min_value=0.0, decimals=1)
        # Seeded to hp, not "": NumericDragLineEdit coerces an empty initial
        # value to 0.0 (it has no blank state), so "" did not mean "circular at
        # hp" as intended — it booted the page at hp=100 / ha=0. That is an
        # inverted orbit, which _update_ghost_orbit then silently swapped, so
        # the form read "Periselene 100 / Aposelene 0" while the preview beside
        # it read "Periselene 0 / Aposelene 100". Defaulting to a circular
        # 100 km orbit keeps the two panels telling the same story on boot.
        self.ent_ha = NumericDragLineEdit("100.0", step=5.0, min_value=0.0, decimals=1)
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

        self.orbit_validation_notice = InlineNotice("", "error")
        self.orbit_validation_notice.setAccessibleName("Orbit input error")
        self.btn_swap_apsides = QtWidgets.QPushButton("Swap values")
        self.btn_swap_apsides.setObjectName("ghostBtn")
        self.btn_swap_apsides.setAccessibleName("Swap periselene and aposelene values")
        self.btn_swap_apsides.clicked.connect(self._swap_apsides)
        notice_layout = self.orbit_validation_notice.layout()
        if notice_layout is not None:
            notice_layout.addWidget(self.btn_swap_apsides, 0, QtCore.Qt.AlignVCenter)
        self.orbit_validation_notice.hide()
        form_layout.addWidget(self.orbit_validation_notice, 3, 0, 1, 4)

        add_param(4, "Semi-major Axis (a)", self.ent_a, "km")
        add_param(5, "Eccentricity (e)", self.ent_e, "")
        add_param(6, "Circular Altitude", self.ent_alt_circular, "km")

        # A flat 1px rule styled from the theme — the old beveled QFrame.HLine
        # read as a dated Win-Forms divider. ``#formDivider`` is themed in QSS.
        sep = QtWidgets.QFrame()
        sep.setObjectName("formDivider")
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Plain)
        sep.setFixedHeight(1)
        form_layout.addWidget(sep, 7, 0, 1, 3)

        orientation_lbl = QtWidgets.QLabel("Plane and orientation")
        orientation_lbl.setObjectName("sectionTitle")
        form_layout.addWidget(orientation_lbl, 8, 0, 1, 3)

        add_param(9, "Inclination (i)", self.ent_inc, "deg")
        add_param(10, "RAAN (Omega)", self.ent_raan, "deg")
        add_param(11, "Argument of Periapsis (omega)", self.ent_argp, "deg")
        add_param(12, "True Anomaly (nu)", self.ent_ta, "deg")

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
        from lunaris.ui.theme.tokens import DESIGN_TOKENS

        gb = self._create_card(
            "Orbit Preview",
            "Two-body preview. The mission run adds the selected perturbations.",
            elevated=True,
        )
        layout = gb.content_layout
        layout.setSpacing(DESIGN_TOKENS.spacing.md)

        self.preview_validation_notice = InlineNotice(
            "Last valid preview — correct orbit inputs.",
            "warning",
        )
        self.preview_validation_notice.setAccessibleName("Orbit preview status")
        self.preview_validation_notice.hide()
        layout.addWidget(self.preview_validation_notice)

        self.orbit_viz_3d = OrbitViz3D()
        self.orbit_viz_3d.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.orbit_viz_3d.focus_mode_requested.connect(self._toggle_preview_focus)
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

        self._metric_grid = info_grid
        self._metric_chips = (
            chip_period, chip_hp, chip_ha, chip_e, chip_i, chip_energy,
        )
        self._metric_columns: int | None = None
        self._update_metric_layout()

        layout.addWidget(info_frame)

        QtCore.QTimer.singleShot(100, self._update_orbit_3d)
        return gb

    def _update_metric_layout(self) -> None:
        """Use one metric row when the preview earns enough horizontal space."""
        if not hasattr(self, "_metric_grid"):
            return
        preview = getattr(self, "group_viz", self._metric_grid.parentWidget())
        columns = 6 if preview.width() >= 700 else 3
        if columns == self._metric_columns:
            return
        self._metric_columns = columns
        for chip in self._metric_chips:
            self._metric_grid.removeWidget(chip)
        for index, chip in enumerate(self._metric_chips):
            self._metric_grid.addWidget(chip, index // columns, index % columns)
        for column in range(6):
            self._metric_grid.setColumnStretch(column, 1 if column < columns else 0)

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

        param_labels = getattr(self, "_param_labels", {})

        # Set active fields
        for field in active_fields:
            field.setReadOnly(False)
            field.setStyleSheet("")
            field.setProperty("ghost", False)
            field.setEnabled(True)
            field.setFocusPolicy(QtCore.Qt.StrongFocus)
            field.style().unpolish(field)
            field.style().polish(field)
            if field in param_labels:
                lbl, base_text = param_labels[field]
                lbl.setText(base_text)
                lbl.setProperty("derived", False)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)

        # Set ghost fields — display-only: they leave the tab order so a
        # keyboard walk visits editable inputs only.
        for field in ghost_fields:
            field.setReadOnly(True)
            field.setStyleSheet("")
            field.setProperty("ghost", True)
            field.setEnabled(True)
            field.setFocusPolicy(QtCore.Qt.NoFocus)
            field.style().unpolish(field)
            field.style().polish(field)
            if field in param_labels:
                lbl, base_text = param_labels[field]
                lbl.setText(f"{base_text} · derived")
                lbl.setProperty("derived", True)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)

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

                        error = self._altitude_order_error(hp, ha)
                        self._set_altitude_validation(error)
                        if error:
                            self._set_derived_values_unavailable()
                            return

                        # Formulas: From Altitude to Classical
                        rp = R_MOON + hp
                        ra = R_MOON + ha

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
                self._set_altitude_validation(None)
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

    @staticmethod
    def _altitude_order_error(hp: float, ha: float) -> str | None:
        if hp > ha:
            return "Periselene altitude cannot exceed aposelene altitude."
        return None

    def _set_altitude_validation(self, message: str | None) -> None:
        """Expose apsis ordering errors without silently reinterpreting values."""
        is_error = bool(message)
        for field in (self.ent_hp, self.ent_ha):
            field.setProperty("fieldError", is_error)
            field.setAccessibleDescription(message or "")
            field.setToolTip(message or "")
            field.style().unpolish(field)
            field.style().polish(field)
        self.orbit_validation_notice.label.setText(message or "")
        self.orbit_validation_notice.setVisible(is_error)
        if hasattr(self, "preview_validation_notice"):
            self.preview_validation_notice.setVisible(is_error)

    def _set_derived_values_unavailable(self) -> None:
        for field in (self.ent_a, self.ent_e):
            field.blockSignals(True)
            field.setText("—")
            field.blockSignals(False)
        for label in (
            self.lbl_period,
            self.lbl_hp,
            self.lbl_ha,
            self.lbl_ecc,
            self.lbl_inc,
            self.lbl_energy,
        ):
            label.setText("—")

    def _swap_apsides(self, _checked: bool = False) -> None:
        """Apply the user's explicit correction and refresh derived state."""
        hp_text = self.ent_hp.text()
        ha_text = self.ent_ha.text()
        self.ent_hp.blockSignals(True)
        self.ent_ha.blockSignals(True)
        self.ent_hp.setText(ha_text)
        self.ent_ha.setText(hp_text)
        self.ent_hp.blockSignals(False)
        self.ent_ha.blockSignals(False)
        self._update_ghost_orbit()
        self._update_orbit_3d()
        self.ent_hp.setFocus(QtCore.Qt.OtherFocusReason)

    def validate_inputs(self) -> bool:
        """Validate desktop-only orbit semantics and focus the first error."""
        if not self.btn_mode_altitude.isChecked():
            self._set_altitude_validation(None)
            return True
        try:
            hp = float(self.ent_hp.text().strip())
            ha_text = self.ent_ha.text().strip()
            ha = float(ha_text) if ha_text else hp
        except (TypeError, ValueError):
            return True
        message = self._altitude_order_error(hp, ha)
        self._set_altitude_validation(message)
        if message:
            self._set_derived_values_unavailable()
            self.ent_hp.setFocus(QtCore.Qt.OtherFocusReason)
            self.ent_hp.selectAll()
            self._params_scroll.ensureWidgetVisible(self.ent_hp)
            return False
        return True

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
                if self._altitude_order_error(hp, ha):
                    return None
                rp = R_MOON + hp
                ra = R_MOON + ha
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
