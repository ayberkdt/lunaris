"""
Dashboard widgets for the ST-LRPS training console.

New PySide6 components that enhance the existing studio.py architecture
without replacing existing functionality.

Components:
- ExperimentHeader    — professional header with status pill and live metrics
- StatusPill          — colored status badge (IDLE/TRAINING/COMPLETED/FAILED)
- HeaderMetric        — small label+value widget for the header bar
- MetricCard          — larger KPI card for the status strip
- ProgressTableModel  — QAbstractTableModel for structured training progress
- StructuredLogView   — tabbed widget with progress table + raw log
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from PyQt6.QtCore import (
        QAbstractTableModel,
        QModelIndex,
        QTimer,
        Qt,
    )
    from PyQt6.QtGui import QColor, QFont
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPlainTextEdit,
        QSizePolicy,
        QTabWidget,
        QTableView,
        QVBoxLayout,
        QWidget,
    )

    _HAS_QT = True
except ImportError:
    _HAS_QT = False

if _HAS_QT:
    from st_lrps.ui.training_metrics import TrainingRecord

# ═══════════════════════════════════════════════════════════════════════════
# Design tokens (consistent with studio.py's apply_premium_dark_theme)
# ═══════════════════════════════════════════════════════════════════════════

_COLORS = {
    "app_bg": "#070B14",
    "panel_bg": "#101A2B",
    "panel_bg_alt": "#0A1020",
    "input_bg": "#0D1626",
    "border": "#26364F",
    "border_soft": "#1A2940",
    "text_main": "#E6EDF7",
    "text_secondary": "#BFD2EA",
    "text_muted": "#7F91AC",
    "cyan": "#35D0FF",
    "violet": "#8B7CFF",
    "purple": "#7c5cff",
    "success": "#2DD4BF",
    "warning": "#F6C177",
    "danger": "#FF6B7A",
    "info_bg": "rgba(53, 208, 255, 0.08)",
    "success_bg": "rgba(45, 212, 191, 0.08)",
    "warning_bg": "rgba(246, 193, 119, 0.08)",
    "danger_bg": "rgba(255, 107, 122, 0.08)",
}

_STATUS_STYLES = {
    "IDLE": {"color": _COLORS["text_muted"], "bg": "rgba(127, 145, 172, 0.12)", "border": "rgba(127, 145, 172, 0.25)"},
    "TRAINING": {"color": _COLORS["cyan"], "bg": "rgba(53, 208, 255, 0.12)", "border": "rgba(53, 208, 255, 0.35)"},
    "COMPLETED": {"color": _COLORS["success"], "bg": "rgba(45, 212, 191, 0.12)", "border": "rgba(45, 212, 191, 0.35)"},
    "FAILED": {"color": _COLORS["danger"], "bg": "rgba(255, 107, 122, 0.14)", "border": "rgba(255, 107, 122, 0.40)"},
    "INTERRUPTED": {"color": _COLORS["warning"], "bg": "rgba(246, 193, 119, 0.12)", "border": "rgba(246, 193, 119, 0.35)"},
}


if _HAS_QT:

    # ═══════════════════════════════════════════════════════════════════════
    # StatusPill
    # ═══════════════════════════════════════════════════════════════════════

    class StatusPill(QLabel):
        """Small colored status badge."""

        def __init__(self, initial: str = "IDLE", parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.set_status(initial)

        def set_status(self, status: str) -> None:
            status = status.upper()
            style = _STATUS_STYLES.get(status, _STATUS_STYLES["IDLE"])
            self.setText(status)
            self.setStyleSheet(
                f"QLabel {{"
                f"  color: {style['color']};"
                f"  background: {style['bg']};"
                f"  border: 1px solid {style['border']};"
                f"  border-radius: 9px;"
                f"  padding: 3px 12px;"
                f"  font-size: 11px;"
                f"  font-weight: 700;"
                f"  letter-spacing: 1.2px;"
                f"}}"
            )

    # ═══════════════════════════════════════════════════════════════════════
    # HeaderMetric
    # ═══════════════════════════════════════════════════════════════════════

    class HeaderMetric(QWidget):
        """Compact label+value widget for the experiment header."""

        def __init__(
            self,
            label: str,
            initial_value: str = "—",
            parent: Optional[QWidget] = None,
        ):
            super().__init__(parent)
            layout = QVBoxLayout()
            layout.setContentsMargins(8, 2, 8, 2)
            layout.setSpacing(1)

            self._label = QLabel(label.upper())
            self._label.setStyleSheet(
                f"color: {_COLORS['text_muted']}; font-size: 10px; font-weight: 600;"
                " letter-spacing: 1px; background: transparent; border: none;"
            )
            self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self._value = QLabel(initial_value)
            self._value.setStyleSheet(
                f"color: {_COLORS['text_main']}; font-size: 14px; font-weight: 600;"
                " font-family: Consolas, 'Courier New', monospace;"
                " background: transparent; border: none;"
            )
            self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)

            layout.addWidget(self._label)
            layout.addWidget(self._value)
            self.setLayout(layout)

        def set_value(self, value: str) -> None:
            self._value.setText(value)

    # ═══════════════════════════════════════════════════════════════════════
    # ExperimentHeader
    # ═══════════════════════════════════════════════════════════════════════

    class ExperimentHeader(QFrame):
        """Professional experiment header bar with status and live metrics."""

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.setObjectName("experimentHeader")
            self.setStyleSheet(
                "QFrame#experimentHeader {"
                "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                "    stop:0 rgba(30, 20, 72, 0.82), stop:0.5 rgba(16, 24, 52, 0.78),"
                "    stop:1 rgba(10, 18, 46, 0.82));"
                "  border: 1px solid rgba(124, 92, 255, 0.24);"
                "  border-radius: 14px;"
                "}"
            )

            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(20, 10, 20, 10)
            main_layout.setSpacing(16)

            # ── Left: title + subtitle + status pill ──
            left_col = QVBoxLayout()
            left_col.setContentsMargins(0, 0, 0, 0)
            left_col.setSpacing(3)

            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(12)

            self._title = QLabel("ST-LRPS Surrogate Console")
            self._title.setStyleSheet(
                f"color: {_COLORS['text_main']}; font-size: 15px; font-weight: 700;"
                " letter-spacing: 0.3px; background: transparent; border: none;"
            )
            title_row.addWidget(self._title)

            self._status_pill = StatusPill("IDLE")
            title_row.addWidget(self._status_pill)
            title_row.addStretch(1)

            self._subtitle = QLabel(
                "Sobolev-trained lunar residual potential models  ·  "
                "<span style='color:#6f7ca8; font-size:11px;'>"
                "dU residual potential; acceleration via grad(dU)</span>"
            )
            self._subtitle.setStyleSheet(
                f"color: {_COLORS['text_secondary']}; font-size: 12px;"
                " background: transparent; border: none;"
            )

            left_col.addLayout(title_row)
            left_col.addWidget(self._subtitle)
            main_layout.addLayout(left_col, 1)

            # ── Separator ──
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setStyleSheet("color: rgba(124, 92, 255, 0.18);")
            main_layout.addWidget(sep)

            # ── Right: live metrics ──
            self._elapsed = HeaderMetric("ELAPSED", "--:--:--")
            self._remaining = HeaderMetric("REMAINING", "—")
            self._finish = HeaderMetric("FINISH", "—")
            self._device = HeaderMetric("DEVICE", "CPU")

            metrics_layout = QHBoxLayout()
            metrics_layout.setContentsMargins(0, 0, 0, 0)
            metrics_layout.setSpacing(4)
            for m in (self._elapsed, self._remaining, self._finish, self._device):
                metrics_layout.addWidget(m)

            main_layout.addLayout(metrics_layout)
            self.setLayout(main_layout)

        # ── Public API ──

        def set_status(self, status: str) -> None:
            self._status_pill.set_status(status)

        def set_elapsed(self, text: str) -> None:
            self._elapsed.set_value(text)

        def set_remaining(self, text: str) -> None:
            self._remaining.set_value(text)

        def set_finish(self, text: str) -> None:
            self._finish.set_value(text)

        def set_device(self, text: str) -> None:
            self._device.set_value(text)

    # ═══════════════════════════════════════════════════════════════════════
    # MetricCard
    # ═══════════════════════════════════════════════════════════════════════

    class MetricCard(QFrame):
        """KPI card widget for the training status strip."""

        def __init__(
            self,
            label: str,
            initial_value: str = "—",
            parent: Optional[QWidget] = None,
        ):
            super().__init__(parent)
            self.setObjectName("metricCard")
            self._state = "normal"  # normal, success, warning, danger

            layout = QVBoxLayout()
            layout.setContentsMargins(12, 8, 12, 8)
            layout.setSpacing(2)

            self._label = QLabel(label.upper())
            self._label.setAlignment(Qt.AlignmentFlag.AlignLeft)

            self._value = QLabel(initial_value)
            self._value.setAlignment(Qt.AlignmentFlag.AlignLeft)

            self._subtitle = QLabel("")
            self._subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self._subtitle.setVisible(False)

            layout.addWidget(self._label)
            layout.addWidget(self._value)
            layout.addWidget(self._subtitle)
            self.setLayout(layout)

            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.setMinimumWidth(100)
            self._apply_style()

        def set_value(
            self,
            value: str,
            subtitle: Optional[str] = None,
            state: Optional[str] = None,
        ) -> None:
            self._value.setText(value)
            if subtitle:
                self._subtitle.setText(subtitle)
                self._subtitle.setVisible(True)
            else:
                self._subtitle.setVisible(False)
            if state and state != self._state:
                self._state = state
                self._apply_style()

        def _apply_style(self) -> None:
            state_colors = {
                "normal": (_COLORS["text_muted"], "rgba(255, 255, 255, 0.04)", "rgba(185, 194, 221, 0.12)"),
                "success": (_COLORS["success"], _COLORS["success_bg"], "rgba(45, 212, 191, 0.22)"),
                "warning": (_COLORS["warning"], _COLORS["warning_bg"], "rgba(246, 193, 119, 0.22)"),
                "danger": (_COLORS["danger"], _COLORS["danger_bg"], "rgba(255, 107, 122, 0.22)"),
            }
            label_color, bg, border = state_colors.get(
                self._state, state_colors["normal"]
            )

            self.setStyleSheet(
                f"QFrame#metricCard {{"
                f"  background: {bg};"
                f"  border: 1px solid {border};"
                f"  border-radius: 10px;"
                f"}}"
            )
            self._label.setStyleSheet(
                f"color: {label_color}; font-size: 10px; font-weight: 600;"
                " letter-spacing: 1px; background: transparent; border: none;"
            )
            self._value.setStyleSheet(
                f"color: {_COLORS['text_main']}; font-size: 15px; font-weight: 600;"
                " font-family: Consolas, 'Courier New', monospace;"
                " background: transparent; border: none;"
            )
            self._subtitle.setStyleSheet(
                f"color: {_COLORS['text_muted']}; font-size: 10px;"
                " background: transparent; border: none;"
            )

    # ═══════════════════════════════════════════════════════════════════════
    # KPIStrip
    # ═══════════════════════════════════════════════════════════════════════

    class KPIStrip(QWidget):
        """Horizontal strip of MetricCard widgets for training KPIs."""

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)

            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)

            self.epoch = MetricCard("Epoch", "— / —")
            self.phase = MetricCard("Phase", "Waiting")
            self.train_loss = MetricCard("Train Loss", "—")
            self.val_loss = MetricCard("Val Loss", "—")
            self.best_score = MetricCard("Best Score", "—")
            self.lr = MetricCard("Learning Rate", "—")
            self.direction = MetricCard("Direction", "—")
            self.eta = MetricCard("ETA", "—")

            for card in (
                self.epoch, self.phase, self.train_loss, self.val_loss,
                self.best_score, self.lr, self.direction, self.eta,
            ):
                layout.addWidget(card, 1)

            self.setLayout(layout)

        def reset(self) -> None:
            """Reset all cards to default values."""
            self.epoch.set_value("— / —")
            self.phase.set_value("Waiting")
            self.train_loss.set_value("—")
            self.val_loss.set_value("—")
            self.best_score.set_value("—", state="normal")
            self.lr.set_value("—")
            self.direction.set_value("—")
            self.eta.set_value("—")

    # ═══════════════════════════════════════════════════════════════════════
    # ProgressTableModel
    # ═══════════════════════════════════════════════════════════════════════

    _PROGRESS_COLUMNS = [
        "Time", "Epoch", "Phase", "Loss", "Val Loss",
        "U Loss", "Accel Loss", "LR", "Event",
    ]

    _SEVERITY_COLORS = {
        "info": None,
        "success": QColor(45, 212, 191, 30),
        "warning": QColor(246, 193, 119, 30),
        "error": QColor(255, 107, 122, 35),
    }

    class ProgressTableModel(QAbstractTableModel):
        """Table model for structured training progress rows."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._records: List[TrainingRecord] = []

        def rowCount(self, parent=QModelIndex()):
            return len(self._records)

        def columnCount(self, parent=QModelIndex()):
            return len(_PROGRESS_COLUMNS)

        def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
            if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
                if 0 <= section < len(_PROGRESS_COLUMNS):
                    return _PROGRESS_COLUMNS[section]
            return None

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid():
                return None
            row = index.row()
            col = index.column()
            if row < 0 or row >= len(self._records):
                return None

            rec = self._records[row]

            if role == Qt.ItemDataRole.DisplayRole:
                return self._display_value(rec, col)

            if role == Qt.ItemDataRole.BackgroundRole:
                severity_color = _SEVERITY_COLORS.get(rec.severity)
                if severity_color:
                    return severity_color
                # Highlight validation rows
                if rec.phase == "val":
                    return QColor(124, 92, 255, 18)
                if rec.phase == "checkpoint":
                    return QColor(45, 212, 191, 18)
                return None

            if role == Qt.ItemDataRole.ForegroundRole:
                if rec.severity == "error":
                    return QColor(_COLORS["danger"])
                if rec.severity == "warning":
                    return QColor(_COLORS["warning"])
                if rec.severity == "success":
                    return QColor(_COLORS["success"])
                return None

            return None

        def append_record(self, record: TrainingRecord) -> None:
            row = len(self._records)
            self.beginInsertRows(QModelIndex(), row, row)
            self._records.append(record)
            self.endInsertRows()

        def clear_records(self) -> None:
            self.beginResetModel()
            self._records.clear()
            self.endResetModel()

        def _display_value(self, rec: TrainingRecord, col: int) -> str:
            if col == 0:  # Time
                return rec.timestamp
            elif col == 1:  # Epoch
                return str(rec.epoch) if rec.epoch > 0 else ""
            elif col == 2:  # Phase
                return rec.phase
            elif col == 3:  # Loss
                return _fmt_loss(rec.loss_opt) or _fmt_loss(rec.loss_ref) or ""
            elif col == 4:  # Val Loss
                if rec.phase == "val":
                    return _fmt_loss(rec.loss_ref) or ""
                return ""
            elif col == 5:  # U Loss
                return _fmt_loss(rec.loss_u) or ""
            elif col == 6:  # Accel Loss
                return _fmt_loss(rec.loss_a) or ""
            elif col == 7:  # LR
                return _fmt_loss(rec.lr) or ""
            elif col == 8:  # Event
                if rec.event in ("batch", "val_summary"):
                    return rec.phase
                return rec.event.replace("_", " ")
            return ""

    def _fmt_loss(v: Optional[float]) -> Optional[str]:
        if v is None:
            return None
        if abs(v) < 1e-2:
            return f"{v:.3e}"
        return f"{v:.5f}"


    # ═══════════════════════════════════════════════════════════════════════
    # StructuredLogView
    # ═══════════════════════════════════════════════════════════════════════

    class StructuredLogView(QWidget):
        """Tabbed widget: structured progress table + raw log text."""

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)

            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._tabs = QTabWidget()

            # ── Tab 1: Structured Progress ──
            self._model = ProgressTableModel()
            self._table = QTableView()
            self._table.setModel(self._model)
            self._table.setAlternatingRowColors(True)
            self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
            self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
            self._table.verticalHeader().setVisible(False)
            self._table.setShowGrid(False)
            self._table.setStyleSheet(
                "QTableView {"
                f"  background-color: {_COLORS['panel_bg_alt']};"
                f"  alternate-background-color: {_COLORS['panel_bg']};"
                f"  border: 1px solid {_COLORS['border_soft']};"
                "  border-radius: 10px;"
                f"  color: {_COLORS['text_main']};"
                "  font-size: 12px;"
                "  gridline-color: transparent;"
                "}"
                "QTableView::item { padding: 4px 8px; }"
                "QTableView::item:selected {"
                "  background-color: rgba(124, 92, 255, 0.22);"
                "}"
                "QHeaderView::section {"
                f"  background-color: {_COLORS['panel_bg']};"
                f"  color: {_COLORS['text_muted']};"
                f"  border: none;"
                f"  border-bottom: 1px solid {_COLORS['border_soft']};"
                "  padding: 6px 8px;"
                "  font-size: 11px;"
                "  font-weight: 600;"
                "  letter-spacing: 0.5px;"
                "}"
            )

            header = self._table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self._table.verticalHeader().setDefaultSectionSize(28)

            self._auto_scroll = True

            self._tabs.addTab(self._table, "Progress")

            # ── Tab 2: Raw Log (placeholder - the actual QPlainTextEdit
            #    is injected by STLRPSTrainTab via set_raw_log_widget) ──
            self._raw_log_placeholder = QLabel(
                "Raw log will appear here during training."
            )
            self._raw_log_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._raw_log_placeholder.setStyleSheet(
                f"color: {_COLORS['text_muted']}; font-size: 12px;"
            )
            self._raw_tab_idx = self._tabs.addTab(self._raw_log_placeholder, "Raw Log")

            layout.addWidget(self._tabs)
            self.setLayout(layout)

        @property
        def model(self) -> ProgressTableModel:
            return self._model

        def set_raw_log_widget(self, widget: QWidget) -> None:
            """Replace the raw log placeholder with the actual log widget."""
            self._tabs.removeTab(self._raw_tab_idx)
            self._raw_tab_idx = self._tabs.addTab(widget, "Raw Log")

        def append_record(self, record: TrainingRecord) -> None:
            """Add a record and auto-scroll if needed."""
            self._model.append_record(record)
            if self._auto_scroll:
                self._table.scrollToBottom()

        def clear(self) -> None:
            self._model.clear_records()

        def set_auto_scroll(self, enabled: bool) -> None:
            self._auto_scroll = enabled
