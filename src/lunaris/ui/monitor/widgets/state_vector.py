"""State vector inspector widget (Mission Monitor 11.6).

Cartesian position/velocity with norms, frame selection and the sample epoch.
Source data stays SI (m, m/s) end to end; km presentation happens only here,
in the render path. The body-fixed frame option is offered only when the run
actually carries a ``state_fixed`` channel — otherwise the segment is disabled
with an explicit "unavailable" reason, never silently mapped to inertial.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from lunaris.ui.components.primitives import SegmentedControl
from lunaris.ui.monitor.formatting import UNAVAILABLE, format_duration
from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame

_ROWS: tuple[tuple[str, str], ...] = (
    ("x", "x"), ("y", "y"), ("z", "z"),
    ("vx", "vx"), ("vy", "vy"), ("vz", "vz"),
    ("r_norm", "|r|"), ("v_norm", "|v|"),
    ("epoch", "Sample time"),
)


def _km(value_m: float) -> str:
    return f"{value_m / 1000.0:,.6f} km"


def _km_s(value_m_s: float) -> str:
    return f"{value_m_s / 1000.0:,.6f} km/s"


class StateVectorWidget(MonitorWidgetFrame):
    def build_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.frame_control = SegmentedControl(["Inertial", "Body-fixed"])
        self.frame_control.setAccessibleName("State vector frame")
        self.frame_control.current_changed.connect(lambda _i: self._on_update())
        layout.addWidget(self.frame_control)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        self.value_labels: dict[str, QtWidgets.QLabel] = {}
        for row, (key, label) in enumerate(_ROWS):
            key_label = QtWidgets.QLabel(label)
            key_label.setObjectName("keyLabel")
            value_label = QtWidgets.QLabel(UNAVAILABLE)
            value_label.setObjectName("valueLabel")
            grid.addWidget(key_label, row, 0)
            grid.addWidget(value_label, row, 1)
            self.value_labels[key] = value_label
        layout.addLayout(grid)
        layout.addStretch(1)

        note = QtWidgets.QLabel(
            "Source values are SI (m, m/s); km conversion happens at display "
            "time only."
        )
        note.setObjectName("sectionDescription")
        note.setWordWrap(True)
        layout.addWidget(note)
        return panel

    def _selected_channel(self, store: TelemetryStore) -> str:
        fixed_available = store.has_channel("state_fixed")
        fixed_button = self.frame_control.buttons[1]
        fixed_button.setEnabled(fixed_available)
        fixed_button.setToolTip(
            "" if fixed_available else
            "Body-fixed state is unavailable for this run/backend."
        )
        if self.frame_control.current_index() == 1 and fixed_available:
            return "state_fixed"
        return "state_inertial"

    def refresh(self, store: TelemetryStore) -> None:
        channel = self._selected_channel(store)
        cursor = self.controller.cursor_time_s
        if cursor is not None:
            hit = store.state_at_or_before(cursor, channel)
        else:
            t, y = store.snapshot_state(channel)
            hit = (float(t[-1]), y[-1]) if t.shape[0] else None
        if hit is None:
            for label in self.value_labels.values():
                label.setText(UNAVAILABLE)
            return
        t_s, state = hit
        r = state[0:3]
        v = state[3:6]
        for idx, key in enumerate(("x", "y", "z")):
            self.value_labels[key].setText(_km(float(r[idx])))
        for idx, key in enumerate(("vx", "vy", "vz")):
            self.value_labels[key].setText(_km_s(float(v[idx])))
        self.value_labels["r_norm"].setText(_km(float(np.linalg.norm(r))))
        self.value_labels["v_norm"].setText(_km_s(float(np.linalg.norm(v))))
        self.value_labels["epoch"].setText(format_duration(t_s))

        sample = store.latest_sample
        if channel == "state_fixed":
            frame = (sample.frame_fixed if sample is not None else None) or "body-fixed"
        else:
            frame = (sample.frame_inertial if sample is not None else None) or "inertial"
        self.set_badges(
            f"km, km/s · {frame} · {store.mode}",
            tooltip="Integrated state at the shown sample time, in the named "
                    "frame. Stored in SI; displayed in km.",
        )


STATE_VECTOR_SPEC = MonitorWidgetSpec(
    widget_id="state_vector",
    title="State Vector",
    category="Trajectory",
    description="Cartesian position/velocity inspector with frame selection.",
    required_channels=("state_inertial", "state_fixed"),
    factory=StateVectorWidget,
)

__all__ = ["STATE_VECTOR_SPEC", "StateVectorWidget"]
