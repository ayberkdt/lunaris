"""Altitude / radius history widget (Mission Monitor 11.2).

Shows the mean-radius altitude (definition visible on the widget: altitude =
r − R_ref, with R_ref derived exactly from the samples), the raw radius, and —
when the run carries topography telemetry — the terrain clearance. Missing
metrics are absent from the selector, never zero-filled. The plot renders a
display-resolution min/max envelope snapshot, so spikes survive and the UI
never draws millions of points.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from lunaris.ui.monitor.formatting import UNAVAILABLE, format_length
from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

#: (channel_id, selector label, y-axis label)
_METRICS = (
    ("altitude_m", "Altitude", "Altitude [km]"),
    ("radius_m", "Radius", "Radius [km]"),
    ("terrain_clearance_m", "Terrain clearance", "Terrain clearance [km]"),
)
_DISPLAY_POINTS = 2000
_DAY_S = 86_400.0


class AltitudeWidget(MonitorWidgetFrame):
    def build_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        selector_row = QtWidgets.QHBoxLayout()
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.setToolTip("Metric shown in the history plot")
        self.metric_combo.currentIndexChanged.connect(lambda _i: self._on_update())
        selector_row.addWidget(self.metric_combo)
        selector_row.addStretch(1)
        layout.addLayout(selector_row)

        stats_row = QtWidgets.QHBoxLayout()
        self.value_labels: dict[str, QtWidgets.QLabel] = {}
        for key, caption in (("current", "Current"), ("min", "Min"), ("max", "Max")):
            cell = QtWidgets.QVBoxLayout()
            caption_label = QtWidgets.QLabel(caption)
            caption_label.setObjectName("metricLabel")
            value_label = QtWidgets.QLabel(UNAVAILABLE)
            value_label.setObjectName("metricValue")
            cell.addWidget(caption_label)
            cell.addWidget(value_label)
            stats_row.addLayout(cell, 1)
            self.value_labels[key] = value_label
        layout.addLayout(stats_row)

        if HAS_PYQTGRAPH:
            from lunaris.ui.core.plot_style import series_pen

            self.plot = pg.PlotWidget()
            self.plot.showGrid(x=True, y=True, alpha=0.25)
            self.plot.setLabel("bottom", "Simulation time [h]")
            self.curve = self.plot.plot([], [], pen=series_pen(0))
            layout.addWidget(self.plot, 1)
        else:
            self.plot = None
            self.curve = None
            fallback = QtWidgets.QLabel(
                "pyqtgraph is not installed — showing numeric values only."
            )
            fallback.setObjectName("sectionDescription")
            fallback.setWordWrap(True)
            layout.addWidget(fallback, 1)

        self.definition_label = QtWidgets.QLabel("")
        self.definition_label.setObjectName("sectionDescription")
        self.definition_label.setWordWrap(True)
        layout.addWidget(self.definition_label)
        return panel

    # ------------------------------------------------------------------ data
    def _sync_metric_choices(self, store: TelemetryStore) -> str | None:
        available = [(cid, label, axis) for cid, label, axis in _METRICS
                     if store.has_channel(cid)]
        current = self.metric_combo.currentData()
        wanted_ids = [cid for cid, _l, _a in available]
        existing_ids = [self.metric_combo.itemData(i) for i in range(self.metric_combo.count())]
        if wanted_ids != existing_ids:
            self.metric_combo.blockSignals(True)
            self.metric_combo.clear()
            for cid, label, _axis in available:
                self.metric_combo.addItem(label, cid)
            if current in wanted_ids:
                self.metric_combo.setCurrentIndex(wanted_ids.index(current))
            self.metric_combo.blockSignals(False)
        data = self.metric_combo.currentData()
        return str(data) if data else None

    def refresh(self, store: TelemetryStore) -> None:
        channel = self._sync_metric_choices(store)
        if channel is None:
            return

        cursor = self.controller.cursor_time_s
        t, v = store.snapshot(channel, max_points=_DISPLAY_POINTS)
        if t.shape[0] == 0:
            return

        if cursor is not None:
            current = store.value_at_or_before(channel, cursor)
        else:
            current = float(v[-1])
        self.value_labels["current"].setText(format_length(current))
        self.value_labels["min"].setText(format_length(float(v.min())))
        self.value_labels["max"].setText(format_length(float(v.max())))

        if self.curve is not None and self.plot is not None:
            span_s = float(t[-1] - t[0]) if t.shape[0] > 1 else 0.0
            if span_s >= 2.0 * _DAY_S:
                t_disp, unit = t / _DAY_S, "d"
            else:
                t_disp, unit = t / 3600.0, "h"
            axis_label = next(a for cid, _l, a in _METRICS if cid == channel)
            self.plot.setLabel("left", axis_label)
            self.plot.setLabel("bottom", f"Simulation time [{unit}]")
            self.curve.setData(t_disp, v / 1000.0)

        self._update_definition(store, channel)
        sampling = self._sampling_note(store)
        self.set_badges(
            f"km · {store.mode}{sampling}",
            tooltip="Values are stored in SI (m) and displayed in km. "
                    "Series is display-downsampled with a min/max envelope.",
        )

    def _update_definition(self, store: TelemetryStore, channel: str) -> None:
        sample = store.latest_sample
        if channel == "altitude_m" and sample is not None and \
                sample.altitude_m is not None and sample.radius_m is not None:
            r_ref_km = (sample.radius_m - sample.altitude_m) / 1000.0
            self.definition_label.setText(
                f"Altitude = r − R_ref (mean reference radius, R_ref = {r_ref_km:,.1f} km). "
                "Impact threshold is altitude 0 at the mean radius, not local terrain."
            )
        elif channel == "terrain_clearance_m":
            self.definition_label.setText(
                "Terrain clearance = r − local topographic surface radius "
                "(sampled under the spacecraft)."
            )
        else:
            self.definition_label.setText("Radius = |r|, Moon-centered distance.")

    @staticmethod
    def _sampling_note(store: TelemetryStore) -> str:
        prov = store.provenance
        if prov is not None and prov.telemetry_cadence_s:
            return f" · every {prov.telemetry_cadence_s:g} s"
        return ""


ALTITUDE_SPEC = MonitorWidgetSpec(
    widget_id="altitude",
    title="Altitude / Radius",
    category="Trajectory",
    description="Mean-radius altitude, radius and terrain-clearance history.",
    required_channels=("altitude_m", "radius_m", "terrain_clearance_m"),
    factory=AltitudeWidget,
)

__all__ = ["ALTITUDE_SPEC", "AltitudeWidget"]
