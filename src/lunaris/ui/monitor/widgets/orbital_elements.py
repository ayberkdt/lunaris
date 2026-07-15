"""Osculating orbital elements widget (Mission Monitor 11.3).

Shows the classical 2-body osculating elements with honest singularity
handling: for near-circular orbits the argument of periapsis is *undefined*
(the producer omits the channel and this widget says so), and for
near-equatorial orbits the RAAN is undefined. Nothing is rendered as a fake
zero. The frame and element convention are visible on the widget.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from lunaris.ui.monitor.formatting import (
    UNAVAILABLE,
    format_angle_from_rad,
    format_dimensionless,
    format_length,
)
from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame

#: Presentation thresholds for declaring a *displayed* element singular.
_CIRCULAR_ECC = 1e-8
_EQUATORIAL_INC_RAD = 1e-7

_ROWS: tuple[tuple[str, str], ...] = (
    ("elements.sma_m", "Semi-major axis"),
    ("elements.ecc", "Eccentricity"),
    ("elements.inc_rad", "Inclination"),
    ("elements.raan_rad", "RAAN"),
    ("elements.argp_rad", "Arg. of periapsis"),
    ("elements.nu_rad", "True anomaly"),
)


class OrbitalElementsWidget(MonitorWidgetFrame):
    def build_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        self.value_labels: dict[str, QtWidgets.QLabel] = {}
        for row, (channel_id, label) in enumerate(_ROWS):
            key_label = QtWidgets.QLabel(label)
            key_label.setObjectName("keyLabel")
            value_label = QtWidgets.QLabel(UNAVAILABLE)
            value_label.setObjectName("valueLabel")
            grid.addWidget(key_label, row, 0)
            grid.addWidget(value_label, row, 1)
            self.value_labels[channel_id] = value_label
        layout.addLayout(grid)
        layout.addStretch(1)

        convention = QtWidgets.QLabel(
            "Osculating 2-body elements (Vallado convention), derived from the "
            "integrated state. Angles displayed in degrees; source values in radians."
        )
        convention.setObjectName("sectionDescription")
        convention.setWordWrap(True)
        layout.addWidget(convention)
        return panel

    def refresh(self, store: TelemetryStore) -> None:
        cursor = self.controller.cursor_time_s

        def value(channel_id: str) -> float | None:
            if cursor is not None:
                return store.value_at_or_before(channel_id, cursor)
            t, v = store.snapshot(channel_id)
            return float(v[-1]) if v.shape[0] else None

        ecc = value("elements.ecc")
        inc = value("elements.inc_rad")
        near_circular = ecc is not None and ecc < _CIRCULAR_ECC
        near_equatorial = inc is not None and (
            inc < _EQUATORIAL_INC_RAD or inc > (3.141592653589793 - _EQUATORIAL_INC_RAD)
        )

        for channel_id, _label in _ROWS:
            label = self.value_labels[channel_id]
            raw = value(channel_id)
            if channel_id == "elements.argp_rad" and raw is None and near_circular:
                label.setText("undefined (circular orbit)")
                label.setToolTip("e ≈ 0: the periapsis direction is singular; "
                                 "no substitute value is shown.")
                continue
            if channel_id == "elements.raan_rad" and raw is None and near_equatorial:
                label.setText("undefined (equatorial orbit)")
                label.setToolTip("i ≈ 0 or 180°: the node line is singular; "
                                 "no substitute value is shown.")
                continue
            label.setToolTip("")
            if channel_id == "elements.sma_m":
                label.setText(format_length(raw))
            elif channel_id == "elements.ecc":
                label.setText(format_dimensionless(raw))
            else:
                label.setText(format_angle_from_rad(raw))

        frame = None
        if store.latest_sample is not None:
            frame = store.latest_sample.frame_inertial
        if frame is None and store.provenance is not None:
            frame = store.provenance.frame_inertial
        self.set_badges(
            f"{frame or 'frame unknown'} · derived · {store.mode}",
            tooltip="Elements are derived (osculating, 2-body) from the propagated "
                    "state in the named inertial frame — not directly measured.",
        )


ORBITAL_ELEMENTS_SPEC = MonitorWidgetSpec(
    widget_id="orbital_elements",
    title="Orbital Elements",
    category="Trajectory",
    description="Osculating 2-body elements with singular-orbit honesty.",
    required_channels=("elements.sma_m", "elements.ecc"),
    factory=OrbitalElementsWidget,
)

__all__ = ["ORBITAL_ELEMENTS_SPEC", "OrbitalElementsWidget"]
