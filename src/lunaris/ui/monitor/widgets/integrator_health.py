"""Integrator health widget (Mission Monitor 11.4).

Renders only what the active backend actually reports: live rows come from the
telemetry stream itself (simulation/wall time, throughput, sample counters,
sequence health) and end-of-run rows from the engine's [DIAG] diagnostics
payload (nfev, integration backend, stop reason). A field the backend cannot
provide simply does not appear — no fake values for fixed-step or SciPy gaps.
"""

from __future__ import annotations

from typing import Any

from PySide6 import QtWidgets

from lunaris.ui.monitor.formatting import format_count, format_duration
from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame

#: End-of-run [DIAG] keys worth surfacing, with display labels and formatting.
_DIAG_ROWS: tuple[tuple[str, str, str], ...] = (
    ("integrator", "Integrator", "text"),
    ("integration_backend", "Integration backend", "text"),
    ("rhs_path", "RHS path", "text"),
    ("nfev", "RHS evaluations (nfev)", "count"),
    ("n_points", "Output samples", "count"),
    ("wall_time_s", "Engine wall time", "duration"),
    ("max_step_s", "Max step", "duration"),
    ("max_step_limiting_reason", "Max-step limited by", "text"),
    ("stop_reason", "Stop reason", "text"),
)


class IntegratorHealthWidget(MonitorWidgetFrame):
    def build_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.grid = QtWidgets.QGridLayout()
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnStretch(1, 1)
        layout.addLayout(self.grid)
        layout.addStretch(1)
        note = QtWidgets.QLabel(
            "Only backend-reported quantities are listed; fields the active "
            "integrator does not measure are omitted entirely."
        )
        note.setObjectName("sectionDescription")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._row_labels: dict[str, QtWidgets.QLabel] = {}
        self._row_order: list[str] = []
        return panel

    # ---------------------------------------------------------------- rows
    def _set_row(self, key: str, label: str, value: str) -> None:
        existing = self._row_labels.get(key)
        if existing is not None:
            existing.setText(value)
            return
        row = len(self._row_order)
        key_label = QtWidgets.QLabel(label)
        key_label.setObjectName("keyLabel")
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("valueLabel")
        self.grid.addWidget(key_label, row, 0)
        self.grid.addWidget(value_label, row, 1)
        self._row_labels[key] = value_label
        self._row_order.append(key)

    def refresh(self, store: TelemetryStore) -> None:
        sample = store.latest_sample
        counters = store.counters

        if sample is not None:
            self._set_row("sim_time", "Simulation time",
                          format_duration(sample.simulation_time_s))
            if sample.wall_time_s is not None:
                self._set_row("wall_time", "Wall time (telemetry)",
                              format_duration(sample.wall_time_s))
                if sample.wall_time_s > 0.0:
                    ratio = sample.simulation_time_s / sample.wall_time_s
                    self._set_row("throughput", "Throughput (sim/wall)", f"{ratio:,.0f}×")
            if store.expected_duration_s and store.expected_duration_s > 0:
                frac = max(0.0, min(1.0, sample.simulation_time_s / store.expected_duration_s))
                self._set_row("progress", "Progress", f"{frac * 100.0:.1f} %")
                if sample.wall_time_s and frac > 1e-6 and store.outcome is None:
                    eta = sample.wall_time_s * (1.0 - frac) / frac
                    self._set_row("eta", "Estimated remaining", format_duration(eta))

        self._set_row("samples", "Telemetry samples received", format_count(counters.accepted))
        if counters.gap_samples:
            self._set_row("gaps", "Missing samples (sequence gaps)",
                          format_count(counters.gap_samples))
        if counters.duplicates or counters.out_of_order:
            self._set_row("dupes", "Duplicate / out-of-order samples",
                          f"{counters.duplicates} / {counters.out_of_order}")

        for key, value in store.latest_diagnostics.items():
            self._set_row(f"live.{key}", key, self._format_diag(value))

        diag = store.run_diagnostics or {}
        for key, label, kind in _DIAG_ROWS:
            if key not in diag:
                continue
            raw: Any = diag[key]
            if kind == "count":
                self._set_row(f"diag.{key}", label, format_count(raw))
            elif kind == "duration":
                self._set_row(f"diag.{key}", label,
                              format_duration(float(raw)) if isinstance(raw, int | float)
                              else str(raw))
            else:
                self._set_row(f"diag.{key}", label, str(raw))

        if store.outcome is not None and store.outcome.reason:
            self._set_row("outcome", "Run outcome", store.outcome.reason)

        source = "live telemetry"
        if diag:
            source += " + engine diagnostics"
        self.set_badges(
            f"{source} · {store.mode}",
            tooltip="Live rows update at telemetry cadence; engine rows arrive once "
                    "with the end-of-run [DIAG] payload.",
        )

    @staticmethod
    def _format_diag(value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int | float):
            return f"{value:,g}"
        return str(value)


INTEGRATOR_HEALTH_SPEC = MonitorWidgetSpec(
    widget_id="integrator_health",
    title="Integrator Health",
    category="Numerics",
    description="Backend-reported timing, step and evaluation diagnostics.",
    required_channels=(),  # renders as soon as any telemetry/diagnostics exist
    factory=IntegratorHealthWidget,
)

__all__ = ["INTEGRATOR_HEALTH_SPEC", "IntegratorHealthWidget"]
