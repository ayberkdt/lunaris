"""Backend & provenance widget (Mission Monitor 11.7).

Answers "what actually ran": requested configuration from the run's
[TELEMETRY_META] line, effective runtime facts (RHS path, integration backend)
merged from the end-of-run [DIAG] payload, plus config hash / git commit.
Unknown fields render as "Unavailable" — never guessed. A backend fallback is
surfaced prominently, not hidden.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from lunaris.ui.monitor.formatting import UNAVAILABLE
from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame


class BackendProvenanceWidget(MonitorWidgetFrame):
    def build_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.fallback_notice = QtWidgets.QLabel("")
        self.fallback_notice.setObjectName("fieldHint")
        self.fallback_notice.setProperty("kind", "warning")
        self.fallback_notice.setWordWrap(True)
        self.fallback_notice.setVisible(False)
        layout.addWidget(self.fallback_notice)

        self.grid = QtWidgets.QGridLayout()
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnStretch(1, 1)
        layout.addLayout(self.grid)
        layout.addStretch(1)
        self._row_labels: dict[str, QtWidgets.QLabel] = {}
        self._row_order: list[str] = []
        return panel

    def _set_row(self, key: str, label: str, value: str | None) -> None:
        text = value if value else UNAVAILABLE
        existing = self._row_labels.get(key)
        if existing is not None:
            existing.setText(text)
            return
        row = len(self._row_order)
        key_label = QtWidgets.QLabel(label)
        key_label.setObjectName("keyLabel")
        value_label = QtWidgets.QLabel(text)
        value_label.setObjectName("valueLabel")
        value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        value_label.setWordWrap(True)
        self.grid.addWidget(key_label, row, 0)
        self.grid.addWidget(value_label, row, 1)
        self._row_labels[key] = value_label
        self._row_order.append(key)

    def refresh(self, store: TelemetryStore) -> None:
        prov = store.provenance
        diag = store.run_diagnostics or {}

        self._set_row("run_id", "Run ID", store.run_id)
        if prov is not None:
            self._set_row("requested_backend", "Requested backend", prov.requested_backend)
            self._set_row("gravity_backend", "Gravity backend", prov.gravity_backend)
            self._set_row("gravity_model", "Gravity model", prov.gravity_model)
            degree = str(prov.sh_degree) if prov.sh_degree is not None else None
            if prov.adaptive_degree:
                degree = f"{degree or '?'} (adaptive)"
            self._set_row("sh_degree", "SH degree", degree)
            self._set_row("integrator", "Integrator", prov.integrator)
            self._set_row("device", "Device", prov.device)
            self._set_row("st_lrps", "ST-LRPS artifact", prov.st_lrps_artifact)
            self._set_row("config_hash", "Config hash", prov.config_sha256)
            self._set_row("git_commit", "Git commit", prov.git_commit)
        effective = diag.get("rhs_path") or diag.get("integration_backend")
        self._set_row("effective_backend", "Effective backend (measured)",
                      str(effective) if effective else None)
        if "integration_backend" in diag:
            self._set_row("integration_backend", "Integration backend",
                          str(diag["integration_backend"]))

        fallback = prov.fallback_reason if prov is not None else None
        if fallback:
            self.fallback_notice.setText(f"Backend fallback: {fallback}")
            self.fallback_notice.setVisible(True)
        else:
            self.fallback_notice.setVisible(False)

        source = "meta" if prov is not None else "no meta received"
        if diag:
            source += " + [DIAG]"
        self.set_badges(
            f"{source} · {store.mode}",
            tooltip="Requested facts come from the run's provenance meta line; "
                    "effective facts are measured by the engine and arrive with "
                    "the end-of-run diagnostics.",
        )

    def has_data(self, store: TelemetryStore) -> bool:
        return store.provenance is not None or bool(store.run_diagnostics) \
            or store.n_samples > 0


BACKEND_PROVENANCE_SPEC = MonitorWidgetSpec(
    widget_id="backend_provenance",
    title="Backend & Provenance",
    category="Provenance",
    description="Requested vs effective backend, models, hashes, fallback state.",
    required_channels=(),
    factory=BackendProvenanceWidget,
)

__all__ = ["BACKEND_PROVENANCE_SPEC", "BackendProvenanceWidget"]
