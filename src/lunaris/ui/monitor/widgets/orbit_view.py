"""3D orbit view widget (Mission Monitor 11.1).

Reuses the Orbit Setup page's GL scene pattern (same Lunar Graphite orbit
theme, same GLViewWidget background trap handling) rather than introducing a
second 3D stack. Shows the Moon, the accumulated trajectory trace, the current
spacecraft position (timeline cursor in replay), and impact/stop markers.

Rendering honesty and cost:

* The trace renders a display-decimated copy (uniform index decimation with
  endpoints kept — a min/max envelope has no meaning for a 3-D path) capped at
  a few thousand vertices; the full-resolution trajectory stays in the run
  artifacts.
* One persistent ``GLLinePlotItem`` is updated via ``setData`` per UI tick —
  items are never rebuilt per sample.
* Offscreen platforms and missing-OpenGL installs get an explicit fallback
  note (numeric widgets still carry the data); the 3D view must never take
  down the workspace.
"""

from __future__ import annotations

import os

import numpy as np
from PySide6 import QtWidgets

from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

_DISPLAY_VERTICES = 4000
_M_TO_KM = 1.0 / 1000.0


def _gl_available() -> bool:
    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform in {"offscreen", "minimal", "vnc"}:
        # pyqtgraph GL renders a blank 0x0 framebuffer offscreen; be explicit
        # instead of silently showing nothing.
        return False
    return HAS_OPENGL


class OrbitViewWidget(MonitorWidgetFrame):
    def build_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.gl_widget = None
        self._trace_item = None
        self._marker_item = None
        self._impact_item = None
        self._fitted = False

        if not _gl_available():
            note = QtWidgets.QLabel(
                "3D rendering is unavailable on this platform (no OpenGL "
                "surface). Trajectory data is still collected — use the "
                "State Vector and Altitude widgets."
            )
            note.setObjectName("sectionDescription")
            note.setWordWrap(True)
            layout.addWidget(note, 1)
            return panel

        try:
            from lunaris.ui.core.ui_commons import ORBIT_THEME, hex_to_rgba_float

            # GLViewWidget reads the *global* pyqtgraph 'background' option in
            # its constructor and raises on None (same trap the Orbit page
            # documents); pin it before construction.
            if pg.getConfigOption("background") is None:
                pg.setConfigOption("background", ORBIT_THEME["space_bg"])
            self.gl_widget = gl.GLViewWidget()
            self.gl_widget.setBackgroundColor(ORBIT_THEME["space_bg"])
            self.gl_widget.opts["distance"] = 8000
            self.gl_widget.opts["elevation"] = 28
            self.gl_widget.opts["azimuth"] = 45

            self._build_moon(hex_to_rgba_float, ORBIT_THEME)

            self._trace_item = gl.GLLinePlotItem(
                pos=np.zeros((2, 3)), width=2.0, antialias=True,
                color=hex_to_rgba_float(ORBIT_THEME["orbit_line"], 1.0),
            )
            self._trace_item.setVisible(False)
            self.gl_widget.addItem(self._trace_item)

            self._marker_item = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3)), size=9.0, pxMode=True,
                color=hex_to_rgba_float(ORBIT_THEME["orbit_glow"], 1.0),
            )
            self._marker_item.setVisible(False)
            self.gl_widget.addItem(self._marker_item)

            self._impact_item = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3)), size=11.0, pxMode=True,
                color=(0.95, 0.35, 0.30, 1.0),
            )
            self._impact_item.setVisible(False)
            self.gl_widget.addItem(self._impact_item)
        except Exception as exc:
            # A 3D widget failure must never take down the monitor workspace.
            self.gl_widget = None
            note = QtWidgets.QLabel(f"3D initialization failed: {exc}")
            note.setObjectName("sectionDescription")
            note.setWordWrap(True)
            layout.addWidget(note, 1)
            return panel

        layout.addWidget(self.gl_widget, 1)

        controls = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset View")
        reset_btn.setToolTip("Return the camera to the default mission view")
        reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(reset_btn)
        fit_btn = QtWidgets.QPushButton("Fit Orbit")
        fit_btn.setToolTip("Frame the whole accumulated trajectory")
        fit_btn.clicked.connect(lambda: self._fit_camera(force=True))
        controls.addWidget(fit_btn)
        controls.addStretch(1)
        layout.addLayout(controls)
        return panel

    # ---------------------------------------------------------------- scene
    def _build_moon(self, hex_to_rgba_float, orbit_theme) -> None:
        from lunaris.ui.core.ui_commons import R_MOON_KM

        md = gl.MeshData.sphere(rows=32, cols=64, radius=R_MOON_KM)
        base = hex_to_rgba_float(orbit_theme["moon_dark"])
        colors = np.empty((md.faceCount(), 4), dtype=float)
        colors[:] = base
        md.setFaceColors(colors)
        self._moon_item = gl.GLMeshItem(
            meshdata=md, smooth=True, shader="shaded", glOptions="opaque"
        )
        self.gl_widget.addItem(self._moon_item)

    def _reset_view(self) -> None:
        if self.gl_widget is not None:
            self.gl_widget.opts["distance"] = 8000
            self.gl_widget.opts["elevation"] = 28
            self.gl_widget.opts["azimuth"] = 45
            self.gl_widget.update()

    def _fit_camera(self, *, force: bool = False) -> None:
        if self.gl_widget is None:
            return
        if self._fitted and not force:
            return
        t, y = self.controller.store.snapshot_state("state_inertial", max_points=512)
        if t.shape[0] == 0:
            return
        max_r_km = float(np.max(np.linalg.norm(y[:, 0:3], axis=1))) * _M_TO_KM
        self.gl_widget.opts["distance"] = max(max_r_km * 2.6, 4500.0)
        self.gl_widget.update()
        self._fitted = True

    # -------------------------------------------------------------- refresh
    def refresh(self, store: TelemetryStore) -> None:
        frame = None
        if store.latest_sample is not None:
            frame = store.latest_sample.frame_inertial
        self.set_badges(
            f"km · {frame or 'frame unknown'} · display-decimated · {store.mode}",
            tooltip="Scene units are km; the trace is a display-decimated copy "
                    "of the sampled trajectory (full resolution lives in the "
                    "run artifacts).",
        )
        if self.gl_widget is None:
            return  # fallback note is the content on GL-less platforms

        t, y = store.snapshot_state("state_inertial", max_points=_DISPLAY_VERTICES)
        if t.shape[0] < 2:
            return
        trace_km = np.ascontiguousarray(y[:, 0:3] * _M_TO_KM)
        self._trace_item.setData(pos=trace_km)
        self._trace_item.setVisible(True)

        cursor = self.controller.cursor_time_s
        if cursor is not None:
            hit = store.state_at_or_before(cursor)
            current = hit[1][0:3] * _M_TO_KM if hit is not None else trace_km[-1]
        else:
            current = trace_km[-1]
        self._marker_item.setData(pos=np.asarray(current, dtype=float).reshape(1, 3))
        self._marker_item.setVisible(True)

        impact_pos = self._impact_position(store)
        if impact_pos is not None:
            self._impact_item.setData(pos=impact_pos.reshape(1, 3))
            self._impact_item.setVisible(True)
        else:
            self._impact_item.setVisible(False)

        self._fit_camera()

    def _impact_position(self, store: TelemetryStore) -> np.ndarray | None:
        """Impact/terminal marker at the state nearest the event time."""
        for event in store.events():
            if event.event_type in ("impact", "terminal_event"):
                hit = store.state_at_or_before(event.simulation_time_s)
                if hit is not None:
                    return hit[1][0:3] * _M_TO_KM
        return None


ORBIT_VIEW_SPEC = MonitorWidgetSpec(
    widget_id="orbit_view",
    title="3D Orbit View",
    category="Trajectory",
    description="Moon, trajectory trace, current position and impact markers.",
    required_channels=("state_inertial",),
    factory=OrbitViewWidget,
)

__all__ = ["ORBIT_VIEW_SPEC", "OrbitViewWidget"]
