"""Event timeline widget (Mission Monitor 11.5).

Chronological table of discrete mission events (periapsis passes, impact,
terminal events, user stop, warnings, backend fallback). Events come from the
telemetry stream, from run-provenance (fallback reason), and from the
end-of-run diagnostics; the store deduplicates repeated emissions. In replay
mode, double-clicking a row asks the timeline controller to jump there.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from lunaris.ui.monitor.formatting import format_duration
from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame

_SEVERITY_LABEL = {"info": "info", "warning": "warning", "critical": "critical"}


class EventTimelineWidget(MonitorWidgetFrame):
    def build_content(self) -> QtWidgets.QWidget:
        self.table = QtWidgets.QTreeWidget()
        self.table.setObjectName("monitorEventTable")
        self.table.setColumnCount(4)
        self.table.setHeaderLabels(["Sim time", "Event", "Severity", "Detail"])
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setUniformRowHeights(True)
        self.table.itemDoubleClicked.connect(self._on_item_activated)
        header = self.table.header()
        header.setStretchLastSection(True)
        self._rendered_count = -1
        return self.table

    def refresh(self, store: TelemetryStore) -> None:
        events = store.events()
        if len(events) != self._rendered_count:
            self.table.clear()
            for event in events:
                item = QtWidgets.QTreeWidgetItem([
                    format_duration(event.simulation_time_s),
                    event.event_type,
                    _SEVERITY_LABEL.get(event.severity, event.severity),
                    event.message,
                ])
                item.setData(0, 0x0100, event.simulation_time_s)  # Qt.UserRole
                item.setToolTip(3, event.message)
                self.table.addTopLevelItem(item)
            for column in range(3):
                self.table.resizeColumnToContents(column)
            self._rendered_count = len(events)
        self.set_badges(
            f"{len(events)} events · {store.mode}",
            tooltip="Events are deduplicated by (type, time, message). "
                    "Double-click a row in replay mode to jump the timeline.",
        )

    def _on_item_activated(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        t_s = item.data(0, 0x0100)
        if t_s is None:
            return
        jump = getattr(self.controller, "jump_to_time", None)
        if callable(jump):
            jump(float(t_s))

    def has_data(self, store: TelemetryStore) -> bool:
        # An empty-but-live run is a legitimate "no events yet" state; show the
        # (empty) table only once telemetry is flowing at all.
        return store.n_samples > 0 or bool(store.events())


EVENT_TIMELINE_SPEC = MonitorWidgetSpec(
    widget_id="event_timeline",
    title="Event Timeline",
    category="Events",
    description="Chronological mission events with replay jump support.",
    required_channels=(),
    factory=EventTimelineWidget,
)

__all__ = ["EVENT_TIMELINE_SPEC", "EventTimelineWidget"]
