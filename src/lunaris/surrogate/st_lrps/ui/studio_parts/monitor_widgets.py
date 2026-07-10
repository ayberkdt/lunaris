"""Monitor-only provenance and periodic-evaluation widgets."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

from lunaris.ui.components.primitives import DataTable, EmptyState, KeyValueList, Section

from .run_inspection import provenance_items, read_periodic_evals


class RunProvenancePanel(Section):
    """Compact paper-safety context sourced from run artifacts."""

    def __init__(self, parent=None):
        super().__init__("Run Provenance", "The artifact context used to judge whether this run is reproducible and defensible.", parent=parent)
        self._empty = EmptyState("Provenance is not available", "Start or select a run with a run_manifest.json to populate this panel.")
        self._values = KeyValueList()
        self._notice = QLabel("")
        self._notice.setObjectName("fieldHint")
        self._notice.setWordWrap(True)
        self.content_layout.addWidget(self._empty)
        self.content_layout.addWidget(self._values)
        self.content_layout.addWidget(self._notice)
        self._values.setVisible(False)
        self._notice.setVisible(False)

    def set_run_dir(self, run_dir: str | Path | None) -> None:
        items = provenance_items(Path(run_dir)) if run_dir else []
        self._empty.setVisible(not bool(items))
        self._values.setVisible(bool(items))
        self._notice.setVisible(False)
        while self._values.layout_grid.count():
            item = self._values.layout_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for label, value, kind in items:
            value_label = self._values.add_item(label, value)
            value_label.setProperty("kind", kind)
            if kind == "warning":
                value_label.setToolTip("This provenance value needs review before using the run as evidence.")
        bad_scope = next((value for label, value, kind in items if label == "Scaler fit scope" and kind == "warning"), None)
        if bad_scope:
            self._notice.setText(f"Review required: scaler fit scope is {bad_scope}; it is not train_only.")
            self._notice.setVisible(True)


class PeriodicEvalTable(Section):
    """Live periodic-evaluation table with explicit units in headers."""

    def __init__(self, parent=None):
        super().__init__("Periodic Evaluations", "Monitoring-only evaluations written during training; they do not change model weights.", parent=parent)
        self._empty = EmptyState("No periodic evaluations yet", "Enable periodic evaluation in Setup to populate this table.")
        self.table = DataTable(
            [("Epoch", ""), ("RMSE U", ""), ("RMSE a", ""), ("Angle", "deg")],
            numeric_columns=(0, 1, 2, 3),
        )
        self.content_layout.addWidget(self._empty)
        self.content_layout.addWidget(self.table)
        self.table.setVisible(False)
        self._run_dir: Path | None = None
        self._last_signature: tuple[int, int] | None = None

    def set_run_dir(self, run_dir: str | Path | None) -> None:
        self._run_dir = Path(run_dir) if run_dir else None
        self.refresh()

    def refresh(self) -> None:
        if self._run_dir is None:
            self._empty.setVisible(True)
            self.table.setVisible(False)
            return
        rows = read_periodic_evals(self._run_dir)
        self.table.setRowCount(0)
        for row in rows:
            self.table.append_row([row.get("epoch", "—"), row.get("rmse_u", "—"), row.get("rmse_a", "—"), row.get("angle", "—")])
        visible = bool(rows)
        self._empty.setVisible(not visible)
        self.table.setVisible(visible)
        self.table.resizeColumnsToContents()
