"""First-class ST-LRPS run browser and comparison surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QFileSystemWatcher, QModelIndex, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lunaris.common.paths import project_root_from_file
from lunaris.surrogate.st_lrps.ui.studio_parts.local_primitives import (
    ActionBar,
    CompactSearchField,
    DataTable,
    EmptyState,
    MetricRow,
    Section,
)
from lunaris.ui_foundation import DESIGN_TOKENS

from .qt_common import THEME, with_alpha
from .run_inspection import (
    config_diff,
    format_dataset,
    format_device,
    format_preset,
    load_run_records,
    read_history,
)

_TRAINING_OUTPUT_ROOT = project_root_from_file(__file__) / "outputs" / "training"
_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "running": ("accent", "Running"),
    "completed": ("success", "Completed"),
    "failed": ("error", "Failed"),
    "interrupted": ("warning", "Interrupted"),
}


def _fmt_score(manifest: dict[str, Any]) -> str:
    score = manifest.get("best_score")
    metric = manifest.get("best_metric") or manifest.get("best_score_name")
    if score in (None, ""):
        return "—"
    try:
        value = f"{float(score):.4e}"
    except (TypeError, ValueError):
        return "—"
    return f"{value} ({metric})" if metric else value


def _fmt_epoch(manifest: dict[str, Any]) -> str:
    best = manifest.get("best_epoch")
    latest = manifest.get("latest_epoch")
    return f"{best if best not in (None, '') else '—'} / {latest if latest not in (None, '') else '—'}"


class RunsTableModel(QAbstractTableModel):
    """Read-only model with safe manifest handling and a searchable view."""

    COLUMNS = ["Name", "Status", "Dataset", "Best Score", "Epoch (best/latest)", "Preset", "Date", "Device"]

    def __init__(self, parent=None, *, training_dir: Path | None = None):
        super().__init__(parent)
        self._training_dir = training_dir or _TRAINING_OUTPUT_ROOT
        self._runs: list[dict[str, Any]] = []
        self._visible: list[dict[str, Any]] = []
        self._query = ""

    @property
    def training_dir(self) -> Path:
        return self._training_dir

    @property
    def runs(self) -> list[dict[str, Any]]:
        return self._visible

    def refresh(self) -> None:
        self.beginResetModel()
        self._runs = load_run_records(self._training_dir)
        self._visible = self._filtered()
        self.endResetModel()

    def set_query(self, query: str) -> None:
        self._query = query.strip().casefold()
        self.beginResetModel()
        self._visible = self._filtered()
        self.endResetModel()

    def _filtered(self) -> list[dict[str, Any]]:
        if not self._query:
            return list(self._runs)
        fields = ("name", "status", "dataset", "preset", "device", "path")
        return [run for run in self._runs if any(self._query in str(run.get(field, "")).casefold() for field in fields)]

    def run_at(self, index: QModelIndex) -> dict[str, Any] | None:
        return self._visible[index.row()] if index.isValid() and 0 <= index.row() < len(self._visible) else None

    def rowCount(self, parent=None):
        parent = parent if parent is not None else QModelIndex()
        return 0 if parent.isValid() else len(self._visible)

    def columnCount(self, parent=None):
        parent = parent if parent is not None else QModelIndex()
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        run = self._visible[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return run
        if role == Qt.ItemDataRole.ToolTipRole:
            return run.get("path") if index.column() == 0 else None
        if role == Qt.ItemDataRole.DisplayRole:
            values = [
                run.get("name", "—"),
                run.get("status", "unknown").capitalize(),
                run.get("dataset") or format_dataset(run),
                _fmt_score(run),
                _fmt_epoch(run),
                run.get("preset") or format_preset(run),
                run.get("date", "—"),
                run.get("device") or format_device(run),
            ]
            return values[index.column()]
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return None


class StatusBadgeDelegate(QStyledItemDelegate):
    """Status text and semantic tint; meaning remains available without color."""

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        status = str(index.data(Qt.ItemDataRole.DisplayRole) or "Unknown")
        key, label = _STATUS_STYLE.get(status.casefold(), ("fg_muted", status))
        fg = THEME[key]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor(with_alpha(THEME["accent"], 0.24)))
        rect = option.rect.adjusted(6, 6, -6, -6)
        font = QFont(option.font)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        badge = rect
        badge.setWidth(min(painter.fontMetrics().horizontalAdvance(label) + 18, rect.width()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(with_alpha(fg, 0.16)))
        painter.drawRoundedRect(badge, min(DESIGN_TOKENS.radii.pill, badge.height() / 2), min(DESIGN_TOKENS.radii.pill, badge.height() / 2))
        painter.setPen(QColor(fg))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()


class RunCompareDialog(QDialog):
    """Compare loss histories and configuration differences for 2–4 runs."""

    def __init__(self, runs: list[dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setObjectName("runCompareDialog")
        self.setWindowTitle("Compare training runs")
        self.resize(980, 680)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(DESIGN_TOKENS.spacing.md)
        names = ", ".join(str(run.get("name", "run")) for run in runs)
        title = QLabel(f"Run comparison · {names}")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        metrics = MetricRow([("Runs", str(len(runs))), ("Best score", _fmt_score(min(runs, key=lambda item: float(item.get("best_score_value") or float("inf")))) if runs else "—")])
        root.addWidget(metrics)
        tabs = QTabWidget()
        tabs.addTab(self._history_panel(runs), "History")
        tabs.addTab(self._config_panel(runs), "Config diff")
        root.addWidget(tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _history_panel(self, runs: list[dict[str, Any]]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        plot = None
        try:
            import pyqtgraph as pg

            from .qt_common import pyqtgraph_matches_qt
            if pyqtgraph_matches_qt(pg):
                plot = pg.PlotWidget()
                plot.setBackground(THEME["bg_card"])
                plot.showGrid(x=True, y=True, alpha=0.2)
                plot.setLabel("bottom", "Epoch")
                plot.setLabel("left", "Validation loss")
        except ImportError:
            plot = None
        if plot is None:
            empty = EmptyState("Plotting is unavailable", "Install the optional pyqtgraph dependency to inspect history curves.")
            layout.addWidget(empty, 1)
            return page
        colors = [THEME["accent"], THEME["success"], THEME["warning"], THEME["tertiary"]]
        styles = [None, "DashLine", "DotLine", "DashDotLine"]
        for index, run in enumerate(runs):
            rows = read_history(Path(str(run["path"])))
            points: list[tuple[float, float]] = []
            for row in rows:
                epoch = row.get("epoch")
                loss = row.get("val_loss_total", row.get("val_loss", row.get("val_total")))
                try:
                    points.append((float(epoch), float(loss)))
                except (TypeError, ValueError):
                    continue
            if not points:
                continue
            pen = pg.mkPen(colors[index % len(colors)], width=2, style=getattr(Qt.PenStyle, styles[index], Qt.PenStyle.SolidLine) if index else Qt.PenStyle.SolidLine)
            plot.plot([p[0] for p in points], [p[1] for p in points], pen=pen, name=str(run.get("name", "run")))
        plot.addLegend(offset=(10, 10))
        layout.addWidget(plot, 1)
        return page

    def _config_panel(self, runs: list[dict[str, Any]]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        names = [str(run.get("name", "run")) for run in runs]
        diff = config_diff([Path(str(run["path"])) for run in runs])
        if not diff:
            layout.addWidget(EmptyState("Configurations match", "No differing config fields were found."), 1)
            return page
        table = DataTable(["Field", *names])
        for row in diff:
            table.append_row([row.get("field", "—"), *(row.get(name, "—") for name in names)])
        table.resizeColumnsToContents()
        layout.addWidget(Section("Changed fields", "Only fields that differ between the selected runs are shown."), 0)
        layout.addWidget(table, 1)
        return page


class RunsBrowserPage(QWidget):
    """Browse, filter, hand off, and compare training runs."""

    resume_requested = Signal(str)
    evaluate_requested = Signal(str)
    benchmark_requested = Signal(str)
    monitor_requested = Signal()

    def __init__(self, parent=None, *, training_dir: Path | None = None):
        super().__init__(parent)
        self.setObjectName("runsBrowserPage")
        layout = QVBoxLayout(self)
        margin = DESIGN_TOKENS.layout.shell_margin
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(DESIGN_TOKENS.spacing.md)
        page_title = QLabel("Runs")
        page_title.setObjectName("pageTitle")
        layout.addWidget(page_title)
        self.search = CompactSearchField("Search runs by name, status, dataset, or preset")
        self.search.textChanged.connect(self._on_search)
        layout.addWidget(self.search)
        self.action_bar = ActionBar(self)
        self.btn_resume = QPushButton("Resume")
        self.btn_evaluate = QPushButton("Evaluate")
        self.btn_benchmark = QPushButton("Benchmark")
        self.btn_compare = QPushButton("Compare")
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_resume.setProperty("kind", "primary")
        for button in (self.btn_resume, self.btn_evaluate, self.btn_benchmark, self.btn_compare, self.btn_open_folder):
            self.action_bar.add_action(button)
        layout.addWidget(self.action_bar)
        self.model = RunsTableModel(self, training_dir=training_dir)
        self.table = QTableView()
        self.table.setObjectName("runsTable")
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setItemDelegateForColumn(1, StatusBadgeDelegate(self.table))
        layout.addWidget(self.table, 1)
        self.empty_state = EmptyState("No training runs found", "Runs appear here after a manifest is written to the training output folder.")
        self.empty_state.setVisible(False)
        layout.addWidget(self.empty_state)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._update_actions())
        self.table.doubleClicked.connect(self._on_double_click)
        self.btn_resume.clicked.connect(lambda: self._emit_single(self.resume_requested))
        self.btn_evaluate.clicked.connect(lambda: self._emit_single(self.evaluate_requested))
        self.btn_benchmark.clicked.connect(lambda: self._emit_single(self.benchmark_requested))
        self.btn_compare.clicked.connect(self._compare)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(lambda *_: self.refresh())
        self._watcher.fileChanged.connect(lambda *_: self.refresh())
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(3000)
        self._poll_timer.timeout.connect(self.refresh)
        self._poll_timer.start()
        self.refresh()

    def refresh(self) -> None:
        self.model.refresh()
        paths = [str(self.model.training_dir)] + [str(Path(run["manifest_path"])) for run in self.model.runs]
        current = set(self._watcher.files()) | set(self._watcher.directories())
        remove = [path for path in current if path not in paths]
        if remove:
            self._watcher.removePaths(remove)
        add = [path for path in paths if Path(path).exists() and path not in current]
        if add:
            self._watcher.addPaths(add)
        self.empty_state.setVisible(self.model.rowCount() == 0)
        self._update_actions()

    def _on_search(self, query: str) -> None:
        self.model.set_query(query)
        self.empty_state.setVisible(self.model.rowCount() == 0)
        self._update_actions()

    def _selected_runs(self) -> list[dict[str, Any]]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [self.model.runs[row] for row in rows if 0 <= row < len(self.model.runs)]

    def _selected_run(self) -> dict[str, Any] | None:
        selected = self._selected_runs()
        return selected[0] if selected else None

    def _update_actions(self) -> None:
        selected = self._selected_runs()
        run = selected[0] if len(selected) == 1 else None
        status = str(run.get("status", "")).lower() if run else ""
        self.btn_resume.setEnabled(run is not None and status in {"interrupted", "failed", "running"})
        self.btn_evaluate.setEnabled(run is not None and status in {"completed", "interrupted"})
        self.btn_benchmark.setEnabled(run is not None and status in {"completed", "interrupted"})
        self.btn_open_folder.setEnabled(run is not None)
        self.btn_compare.setEnabled(2 <= len(selected) <= 4)
        self.btn_compare.setToolTip("Select 2–4 runs to compare their curves and config differences.")

    def _emit_single(self, signal: Signal) -> None:
        run = self._selected_run()
        if run:
            signal.emit(str(run["path"]))

    def _on_double_click(self, index: QModelIndex) -> None:
        run = self.model.run_at(index)
        if run and str(run.get("status", "")).lower() == "running":
            self.monitor_requested.emit()

    def _compare(self) -> None:
        selected = self._selected_runs()
        if 2 <= len(selected) <= 4:
            RunCompareDialog(selected, self).exec()

    def _on_open_folder(self) -> None:
        run = self._selected_run()
        if not run:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(run["path"])))
