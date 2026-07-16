"""Batch / ensemble progress widget (Mission Monitor phase 6).

Observes the ``[BATCH_PROGRESS]`` / ``[BATCH_METRICS]`` protocol the batch
runner already emits — no new producer wiring. Live rows come from the latest
progress payload (stage, completed samples, ETA, backend); final rows from the
metrics payload (impact counts with the runner's own 95% CI, wall time,
requested vs actual backend, fallback reason).

Batch observability is process-scoped, not run-scoped: it lives on the
controller (``batch_progress`` / ``batch_metrics``), so a batch running next
to a live single-run session never contaminates the run's provenance/store.
"""

from __future__ import annotations

from typing import Any

from PySide6 import QtWidgets

from lunaris.ui.monitor.formatting import format_count, format_duration
from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame


class BatchProgressWidget(MonitorWidgetFrame):
    def build_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setAccessibleName("Batch progress")
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

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

    # ------------------------------------------------------------------ rows
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
        value_label.setWordWrap(True)
        self.grid.addWidget(key_label, row, 0)
        self.grid.addWidget(value_label, row, 1)
        self._row_labels[key] = value_label
        self._row_order.append(key)

    def has_data(self, store: TelemetryStore) -> bool:
        # Batch payloads live on the controller, not in the run store.
        return (
            self.controller.batch_progress is not None
            or self.controller.batch_metrics is not None
        )

    def _show_unavailable_state(self, store: TelemetryStore) -> None:
        self._show_waiting_state()

    def _show_waiting_state(self) -> None:
        self._empty.set_message(
            "No batch run observed",
            "Start a batch propagation; its progress stream appears here.",
        )
        self._stack.setCurrentWidget(self._empty)

    # -------------------------------------------------------------- refresh
    def refresh(self, store: TelemetryStore) -> None:
        progress: dict[str, Any] = self.controller.batch_progress or {}
        metrics: dict[str, Any] = self.controller.batch_metrics or {}

        fraction = progress.get("fraction")
        if metrics:
            self.progress_bar.setValue(1000)
            self.progress_bar.setFormat("Completed")
        elif isinstance(fraction, int | float):
            self.progress_bar.setValue(int(max(0.0, min(1.0, float(fraction))) * 1000))
            self.progress_bar.setFormat(f"{float(fraction) * 100.0:.1f}%")

        stage = progress.get("stage")
        if isinstance(stage, str) and stage:
            detail = progress.get("detail")
            text = stage if not (isinstance(detail, str) and detail) else f"{stage} — {detail}"
            self._set_row("stage", "Stage", text)
        done = progress.get("done_samples")
        total = progress.get("total_samples")
        if isinstance(done, int | float) and isinstance(total, int | float) and total:
            self._set_row("samples", "Samples", f"{int(done):,} / {int(total):,}")
        elapsed = progress.get("elapsed_s")
        if isinstance(elapsed, int | float):
            self._set_row("elapsed", "Elapsed", format_duration(float(elapsed)))
        eta = progress.get("eta_s")
        if isinstance(eta, int | float) and not metrics:
            self._set_row("eta", "Estimated remaining", format_duration(float(eta)))

        if metrics:
            self._refresh_metrics(metrics)

        backend = metrics.get("actual_batch_backend") or progress.get("backend")
        self.set_badges(
            f"{backend or 'backend unknown'} · batch",
            tooltip="Live rows from [BATCH_PROGRESS]; final statistics from "
                    "[BATCH_METRICS] as reported by the batch runner.",
        )

    def _refresh_metrics(self, metrics: dict[str, Any]) -> None:
        n_samples = metrics.get("n_samples")
        n_impacts = metrics.get("n_impacts")
        if isinstance(n_impacts, int | float) and isinstance(n_samples, int | float):
            self._set_row("impacts", "Impacts",
                          f"{format_count(n_impacts)} / {format_count(n_samples)}")
        p_impact = metrics.get("p_impact")
        ci = metrics.get("p_impact_ci95")
        if isinstance(p_impact, int | float):
            text = f"{float(p_impact) * 100.0:.2f} %"
            if isinstance(ci, list) and len(ci) == 2:
                text += f"  (95% CI {float(ci[0]) * 100.0:.2f}–{float(ci[1]) * 100.0:.2f} %)"
            self._set_row("p_impact", "Impact probability", text)
        wall = metrics.get("wall_time_s")
        if isinstance(wall, int | float):
            self._set_row("wall", "Wall time", format_duration(float(wall)))

        requested = metrics.get("requested_batch_backend")
        actual = metrics.get("actual_batch_backend")
        if requested or actual:
            self._set_row("backend", "Backend (requested → actual)",
                          f"{requested or '—'} → {actual or '—'}")
        device = metrics.get("device_name")
        if isinstance(device, str) and device:
            self._set_row("device", "Device", device)
        kind = metrics.get("runtime_model_kind")
        if isinstance(kind, str) and kind:
            self._set_row("model_kind", "ST-LRPS runtime kind", kind)
        req_deg = metrics.get("requested_sh_degree")
        act_deg = metrics.get("actual_sh_degree")
        if req_deg is not None or act_deg is not None:
            self._set_row("degree", "SH degree (requested → actual)",
                          f"{req_deg if req_deg is not None else '—'} → "
                          f"{act_deg if act_deg is not None else '—'}")
        output_path = metrics.get("output_path")
        if isinstance(output_path, str) and output_path:
            self._set_row("output", "Output", output_path)

        fallback = metrics.get("fallback_reason")
        if isinstance(fallback, str) and fallback:
            self.fallback_notice.setText(f"Backend fallback: {fallback}")
            self.fallback_notice.setVisible(True)
        else:
            self.fallback_notice.setVisible(False)


BATCH_PROGRESS_SPEC = MonitorWidgetSpec(
    widget_id="batch_progress",
    title="Batch Progress",
    category="Batch",
    description="Ensemble completion, impact statistics and backend provenance.",
    required_channels=(),  # availability comes from controller batch payloads
    supports_replay=False,
    factory=BatchProgressWidget,
)

__all__ = ["BATCH_PROGRESS_SPEC", "BatchProgressWidget"]
