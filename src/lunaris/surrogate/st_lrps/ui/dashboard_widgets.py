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

import csv
import json
from typing import Any

from lunaris.ui_foundation import DESIGN_TOKENS, with_alpha

# Lunaris standardizes on PySide6 (Phase 3 consolidation; PyQt6 dropped). The
# try/except keeps graceful degradation: when PySide6 is absent the module still
# imports with _HAS_QT = False and the widgets are simply unavailable.
try:
    from PySide6.QtCore import (
        QAbstractTableModel,
        QModelIndex,
        Qt,
    )
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPushButton,
        QSizePolicy,
        QTableView,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    _HAS_QT = True
except ImportError:
    _HAS_QT = False

if _HAS_QT:
    from lunaris.surrogate.st_lrps.ui.training_metrics import TrainingRecord

    def _qt_color_alpha(color: str, alpha: int) -> QColor:
        value = QColor(color)
        value.setAlpha(alpha)
        return value

# ═══════════════════════════════════════════════════════════════════════════
# Design tokens (consistent with studio.py's apply_premium_dark_theme)
# ═══════════════════════════════════════════════════════════════════════════

def _alpha(color: str, alpha: float) -> str:
    return with_alpha(color, alpha)


_C = DESIGN_TOKENS.colors
_COLORS = {
    "app_bg": _C.bg_space,
    "panel_bg": _C.bg_card,
    "panel_bg_alt": _C.bg_shell,
    "input_bg": _C.bg_entry,
    "border": _C.border,
    "border_soft": _C.border_soft,
    "text_main": _C.fg_main,
    "text_secondary": _C.fg_soft,
    "text_muted": _C.fg_muted,
    "accent": _C.accent,
    "success": _C.success,
    "warning": _C.warning,
    "danger": _C.error,
    "info_bg": _alpha(_C.accent, 0.08),
    "success_bg": _alpha(_C.success, 0.08),
    "warning_bg": _alpha(_C.warning, 0.08),
    "danger_bg": _alpha(_C.error, 0.08),
}

_STATUS_STYLES = {
    "IDLE": {"color": _COLORS["text_muted"], "bg": _alpha(_C.fg_muted, 0.12), "border": _alpha(_C.fg_muted, 0.25)},
    "TRAINING": {"color": _COLORS["accent"], "bg": _alpha(_C.accent, 0.12), "border": _alpha(_C.accent, 0.35)},
    "COMPLETED": {"color": _COLORS["success"], "bg": _alpha(_C.success, 0.12), "border": _alpha(_C.success, 0.35)},
    "FAILED": {"color": _COLORS["danger"], "bg": _alpha(_C.error, 0.14), "border": _alpha(_C.error, 0.40)},
    "INTERRUPTED": {"color": _COLORS["warning"], "bg": _alpha(_C.warning, 0.12), "border": _alpha(_C.warning, 0.35)},
}


if _HAS_QT:

    # ═══════════════════════════════════════════════════════════════════════
    # StatusPill
    # ═══════════════════════════════════════════════════════════════════════

    class StatusPill(QLabel):
        """Small colored status badge."""

        def __init__(self, initial: str = "IDLE", parent: QWidget | None = None):
            super().__init__(parent)
            self.setObjectName("statusBadge")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setAccessibleName("Training status")
            self.set_status(initial)

        def set_status(self, status: str) -> None:
            status = status.upper()
            self.setText(status)
            kind_map = {
                "IDLE": "idle",
                "TRAINING": "running",
                "COMPLETED": "completed",
                "FAILED": "failed",
                "INTERRUPTED": "paused",
            }
            self.setProperty("kind", kind_map.get(status, "idle"))
            self.style().unpolish(self)
            self.style().polish(self)

    # ═══════════════════════════════════════════════════════════════════════
    # HeaderMetric
    # ═══════════════════════════════════════════════════════════════════════

    class HeaderMetric(QWidget):
        """Compact label+value widget for the experiment header."""

        def __init__(
            self,
            label: str,
            initial_value: str = "—",
            parent: QWidget | None = None,
        ):
            super().__init__(parent)
            self.setObjectName("headerMetric")
            layout = QHBoxLayout()
            layout.setContentsMargins(9, 3, 9, 3)
            layout.setSpacing(6)

            self._label = QLabel(label.upper())
            self._label.setObjectName("headerMetricLabel")
            self._label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

            self._value = QLabel(initial_value)
            self._value.setObjectName("headerMetricValue")
            self._value.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

            layout.addWidget(self._label)
            layout.addWidget(self._value)
            self.setLayout(layout)
            self.setMaximumHeight(28)
            self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        def set_value(self, value: str) -> None:
            self._value.setText(value)

    # ═══════════════════════════════════════════════════════════════════════
    # ExperimentHeader
    # ═══════════════════════════════════════════════════════════════════════

    def _short(text: str, n: int = 20) -> str:
        """Compact a label for a header badge (keeps the trailing part of paths)."""
        text = (text or "").strip()
        if not text:
            return "—"
        if len(text) <= n:
            return text
        return "…" + text[-(n - 1):]

    class ExperimentHeader(QFrame):
        """Compact, restrained application header for ST-LRPS Studio."""

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setObjectName("experimentHeader")
            self.setAccessibleName("ST-LRPS workspace header")
            self.setMinimumHeight(56)
            self.setMaximumHeight(64)

            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(14, 6, 14, 6)
            main_layout.setSpacing(12)

            # ── Left: title + status pill + subtitle ──
            left_col = QVBoxLayout()
            left_col.setContentsMargins(0, 0, 0, 0)
            left_col.setSpacing(2)

            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(10)

            self._title = QLabel("ST-LRPS Studio")
            self._title.setObjectName("title")
            title_row.addWidget(self._title)
            self._status_pill = StatusPill("IDLE")
            title_row.addWidget(self._status_pill)
            title_row.addStretch(1)

            self._subtitle = QLabel(
                "Lunar residual-potential surrogate training and evaluation"
            )
            self._subtitle.setObjectName("pageDescription")

            left_col.addLayout(title_row)
            left_col.addWidget(self._subtitle)
            main_layout.addLayout(left_col)
            main_layout.addStretch(1)

            # ── Right: compact context + time badges ──
            self._page = HeaderMetric("PAGE", "—")
            self._run = HeaderMetric("RUN", "—")
            self._dataset = HeaderMetric("DATASET", "—")
            self._preset = HeaderMetric("PRESET", "—")
            self._device = HeaderMetric("DEVICE", "CPU")
            self._checkpoint = HeaderMetric("CKPT", "—")

            self._elapsed = HeaderMetric("ELAPSED", "--:--:--")
            self._remaining = HeaderMetric("ETA", "—")
            self._finish = HeaderMetric("FINISH", "—")

            for hidden_metric in (
                self._run,
                self._dataset,
                self._preset,
                self._checkpoint,
                self._elapsed,
                self._remaining,
                self._finish,
            ):
                hidden_metric.setVisible(False)

            badges = QHBoxLayout()
            badges.setContentsMargins(0, 0, 0, 0)
            badges.setSpacing(6)
            for m in (
                self._page, self._run, self._dataset, self._preset,
                self._device, self._checkpoint, self._elapsed,
                self._remaining, self._finish
            ):
                badges.addWidget(m)
            main_layout.addLayout(badges)
            self.setLayout(main_layout)

        # ── Public API ──

        def set_status(self, status: str) -> None:
            self._status_pill.set_status(status)

        def set_page(self, text: str) -> None:
            self._page.set_value(text or "—")

        def set_run(self, text: str) -> None:
            value = _short(text)
            self._run.set_value(value)
            self._run.setVisible(value != "—")

        def set_dataset(self, text: str) -> None:
            self._dataset.set_value(_short(text, 16))

        def set_preset(self, text: str) -> None:
            # Preset names are most recognizable by their leading words.
            text = (text or "").strip() or "—"
            if len(text) > 18:
                text = text[:17] + "…"
            self._preset.set_value(text)

        def set_checkpoint(self, text: str) -> None:
            self._checkpoint.set_value(text or "—")

        def set_elapsed(self, text: str) -> None:
            self._elapsed.set_value(text)

        def set_remaining(self, text: str) -> None:
            self._remaining.set_value(text)
            self._remaining.setVisible(bool(text and text != "—"))

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
            parent: QWidget | None = None,
        ):
            super().__init__(parent)
            self.setObjectName("metricCard")
            self._state = "normal"  # normal, success, warning, danger

            layout = QVBoxLayout()
            layout.setContentsMargins(12, 8, 12, 8)
            layout.setSpacing(2)

            self._label = QLabel(label.upper())
            self._label.setObjectName("metricCardLabel")
            self._label.setAlignment(Qt.AlignmentFlag.AlignLeft)

            self._value = QLabel(initial_value)
            self._value.setObjectName("metricCardValue")
            self._value.setAlignment(Qt.AlignmentFlag.AlignLeft)

            self._subtitle = QLabel("")
            self._subtitle.setObjectName("metricCardSubtitle")
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
            subtitle: str | None = None,
            state: str | None = None,
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
            self.setProperty("state", self._state)
            self.style().unpolish(self)
            self.style().polish(self)

    # ═══════════════════════════════════════════════════════════════════════
    # KPIStrip
    # ═══════════════════════════════════════════════════════════════════════

    class KPIStrip(QWidget):
        """Responsive grid of MetricCard widgets for training KPIs."""

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)

            self._layout = QGridLayout()
            self._layout.setContentsMargins(0, 0, 0, 0)
            self._layout.setSpacing(8)

            self.epoch = MetricCard("Epoch", "— / —")
            self.phase = MetricCard("Phase", "Waiting")
            self.train_loss = MetricCard("Train Loss", "—")
            self.val_loss = MetricCard("Val Loss", "—")
            self.best_score = MetricCard("Best Score", "—")
            self.lr = MetricCard("Learning Rate", "—")
            self.direction = MetricCard("Direction", "—")
            self.device = MetricCard("Device", "CPU")

            self._cards = (
                self.epoch, self.phase, self.train_loss, self.val_loss,
                self.best_score, self.lr, self.direction, self.device,
            )
            self._columns = 0
            self.setLayout(self._layout)
            self._relayout(4)

        def _relayout(self, columns: int) -> None:
            if columns == self._columns:
                return
            self._columns = columns
            for card in self._cards:
                self._layout.removeWidget(card)
            for index, card in enumerate(self._cards):
                row, column = divmod(index, columns)
                self._layout.addWidget(card, row, column)
            for column in range(columns):
                self._layout.setColumnStretch(column, 1)

        def reset(self) -> None:
            """Reset all cards to default values."""
            self.epoch.set_value("— / —")
            self.phase.set_value("Waiting")
            self.train_loss.set_value("—")
            self.val_loss.set_value("—")
            self.best_score.set_value("—", state="normal")
            self.lr.set_value("—")
            self.direction.set_value("—")
            self.device.set_value("—")

    # ═══════════════════════════════════════════════════════════════════════
    # TimeMetricsStrip (Phase 7: training time observability)
    # ═══════════════════════════════════════════════════════════════════════

    class TimeMetricsStrip(QWidget):
        """Responsive grid of time-oriented MetricCards for the Live Monitor."""

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self._layout = QGridLayout()
            self._layout.setContentsMargins(0, 0, 0, 0)
            self._layout.setSpacing(8)

            self.elapsed = MetricCard("Elapsed", "--:--:--")
            self.eta = MetricCard("ETA Remaining", "—")
            self.finish = MetricCard("Est. Finish", "—")
            self.started = MetricCard("Started At", "—")
            self.epoch_duration = MetricCard("Epoch Duration", "--:--:--")
            self.avg_epoch = MetricCard("Avg Epoch", "—")
            self.samples_per_s = MetricCard("Samples/s", "—")

            self._cards = (
                self.elapsed, self.eta, self.finish, self.started,
                self.epoch_duration, self.avg_epoch, self.samples_per_s,
            )
            self._columns = 0
            self.setLayout(self._layout)
            self._relayout(4)

        def _relayout(self, columns: int) -> None:
            if columns == self._columns:
                return
            self._columns = columns
            for card in self._cards:
                self._layout.removeWidget(card)
            for index, card in enumerate(self._cards):
                row, column = divmod(index, columns)
                self._layout.addWidget(card, row, column)
            for column in range(columns):
                self._layout.setColumnStretch(column, 1)

        def reset(self) -> None:
            self.elapsed.set_value("--:--:--")
            self.eta.set_value("Estimating…")
            self.finish.set_value("Estimating…")
            self.started.set_value("—")
            self.epoch_duration.set_value("--:--:--")
            self.avg_epoch.set_value("Estimating…")
            self.samples_per_s.set_value("—")

        def set_done(self, finish_text: str = "") -> None:
            """Final state when training stops: ETA = Done, finish = actual time."""
            self.eta.set_value("Done", state="success")
            if finish_text:
                self.finish.set_value(finish_text, state="success")

    # ═══════════════════════════════════════════════════════════════════════
    # ProgressTableModel
    # ═══════════════════════════════════════════════════════════════════════

    _PROGRESS_COLUMNS = [
        "Time", "Epoch", "Phase", "Batch", "Progress %",
        "Loss Opt", "Loss Ref", "U Loss", "Accel Loss",
        "Direction Loss", "Cos Sim", "LR", "Samples/s",
        "ETA", "Memory", "Event",
    ]

    _SEVERITY_COLORS = {
        "info": None,
        "success": _qt_color_alpha(_C.success, 30),
        "warning": _qt_color_alpha(_C.warning, 30),
        "error": _qt_color_alpha(_C.error, 35),
    }

    class ProgressTableModel(QAbstractTableModel):
        """Table model for structured training progress rows."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._records: list[TrainingRecord] = []

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
                    return _qt_color_alpha(_C.accent, 18)
                if rec.phase == "checkpoint":
                    return _qt_color_alpha(_C.success, 18)
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
            if col == 1:  # Epoch
                return str(rec.epoch) if rec.epoch > 0 else ""
            if col == 2:  # Phase
                return rec.phase
            if col == 3:  # Batch
                if rec.total_batches > 0:
                    return f"{rec.batch}/{rec.total_batches}"
                return str(rec.batch) if rec.batch > 0 else ""
            if col == 4:  # Progress %
                if rec.progress_pct is not None:
                    return f"{rec.progress_pct:.0f}%"
                return ""
            if col == 5:  # Loss Opt
                return _fmt_loss(rec.loss_opt) or ""
            if col == 6:  # Loss Ref
                return _fmt_loss(rec.loss_ref) or ""
            if col == 7:  # U Loss
                return _fmt_loss(rec.loss_u) or ""
            if col == 8:  # Accel Loss
                return _fmt_loss(rec.loss_a) or ""
            if col == 9:  # Direction Loss
                return _fmt_loss(rec.direction_loss) or ""
            if col == 10:  # Cos Sim
                return f"{rec.cos_sim:.4f}" if rec.cos_sim is not None else ""
            if col == 11:  # LR
                return _fmt_loss(rec.lr) or ""
            if col == 12:  # Samples/s
                return f"{rec.samples_per_s:,.0f}" if rec.samples_per_s is not None else ""
            if col == 13:  # ETA
                return _fmt_eta(rec.eta_s)
            if col == 14:  # Memory
                return rec.memory or ""
            if col == 15:  # Event
                if rec.event in ("batch", "val_summary"):
                    return rec.phase
                return rec.event.replace("_", " ")
            return ""

    def _fmt_loss(v: float | None) -> str | None:
        if v is None:
            return None
        if abs(v) < 1e-2:
            return f"{v:.3e}"
        return f"{v:.5f}"

    def _fmt_eta(seconds: float | None) -> str:
        if seconds is None:
            return ""
        seconds = max(0, int(round(seconds)))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"


    # ═══════════════════════════════════════════════════════════════════════
    # EpochHistoryModel — per-epoch training history as an aligned table
    # ═══════════════════════════════════════════════════════════════════════

    # (display label, candidate source keys, format kind). Candidates cover both
    # the flat CSV schema and the (flattened) nested JSONL schema.
    _HISTORY_COLUMNS = [
        ("Epoch",     ["epoch_display", "epoch"],                          "int"),
        ("Train Loss", ["train_loss_total"],                               "sci"),
        ("Val Loss",  ["val_loss_total"],                                  "sci"),
        ("Train U",   ["train_loss_u"],                                    "sci"),
        ("Val U",     ["val_loss_u"],                                      "sci"),
        ("Train a",   ["train_loss_a"],                                    "sci"),
        ("Val a",     ["val_loss_a"],                                      "sci"),
        ("Train dir", ["train_loss_dir"],                                  "sci"),
        ("Val dir",   ["val_loss_dir"],                                    "sci"),
        ("Val score", ["val_checkpoint_score", "checkpoint_score"],        "sci"),
        ("Train cos", ["train_cos_sim", "train_mean_cossim"],              "f4"),
        ("Val cos",   ["val_cos_sim", "val_mean_cossim"],                  "f4"),
        ("Val ang°",  ["val_angular_mean_deg", "val_ang_deg"],             "f2"),
        ("LR",        ["lr"],                                              "sci"),
    ]

    def _flatten_history_row(d: dict[str, Any]) -> dict[str, Any]:
        """Flatten one level of nested dicts into ``parent_child`` keys.

        Flat CSV rows pass through unchanged; nested JSONL rows
        (``{"train": {"loss_total": ...}}``) become ``train_loss_total``.
        """
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if not isinstance(v2, dict | list):
                        out[f"{k}_{k2}"] = v2
            elif not isinstance(v, list):
                out[k] = v
        return out

    def _hist_fmt(value: Any, kind: str) -> str:
        if value is None or value == "":
            return ""
        try:
            f = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not (f == f):  # NaN
            return ""
        if kind == "int":
            return str(int(round(f)))
        if kind == "f4":
            return f"{f:.4f}"
        if kind == "f2":
            return f"{f:.2f}"
        # scientific for losses
        return f"{f:.3e}"

    def load_history_rows(path: str) -> list[dict[str, Any]]:
        """Load per-epoch history rows from a .csv or .jsonl history file."""
        rows: list[dict[str, Any]] = []
        try:
            p = str(path)
            if p.lower().endswith(".jsonl"):
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                rows.append(_flatten_history_row(obj))
                        except Exception:
                            continue
            else:  # treat as CSV
                with open(p, encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        rows.append(_flatten_history_row(dict(r)))
        except Exception:
            return rows
        return rows

    class EpochHistoryModel(QAbstractTableModel):
        """Table model rendering the curated per-epoch training history."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._rows: list[dict[str, Any]] = []
            self._cols = _HISTORY_COLUMNS

        def rowCount(self, parent=QModelIndex()):
            return len(self._rows)

        def columnCount(self, parent=QModelIndex()):
            return len(self._cols)

        def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
            if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
                if 0 <= section < len(self._cols):
                    return self._cols[section][0]
            return None

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid():
                return None
            row, col = index.row(), index.column()
            if row < 0 or row >= len(self._rows):
                return None
            label, keys, kind = self._cols[col]
            rec = self._rows[row]
            value = None
            for k in keys:
                if k in rec and rec[k] not in (None, ""):
                    value = rec[k]
                    break
            if role == Qt.ItemDataRole.DisplayRole:
                return _hist_fmt(value, kind)
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return None

        def set_rows(self, rows: list[dict[str, Any]]) -> None:
            self.beginResetModel()
            self._rows = list(rows)
            self.endResetModel()

        def row_count(self) -> int:
            return len(self._rows)


    # ═══════════════════════════════════════════════════════════════════════
    # StructuredLogView
    # ═══════════════════════════════════════════════════════════════════════

    class StructuredLogView(QWidget):
        """Tabbed widget: structured progress table + raw log text."""

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)

            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._tabs = QTabWidget()

            # ── Toolbar: Severity Counters ──
            self._warn_count = 0
            self._error_count = 0
            self._severity_filter: str | None = None

            self._filter_bar = QWidget()
            filter_layout = QHBoxLayout(self._filter_bar)
            filter_layout.setContentsMargins(8, 4, 8, 4)
            filter_layout.setSpacing(8)

            self._lbl_title = QLabel("Progress & Logs")
            self._lbl_title.setObjectName("dashCaption")
            filter_layout.addWidget(self._lbl_title)
            filter_layout.addStretch()

            self.btn_filter_warn = QPushButton("0 Warnings")
            self.btn_filter_warn.setCheckable(True)
            self.btn_filter_warn.setCursor(Qt.CursorShape.PointingHandCursor)

            self.btn_filter_err = QPushButton("0 Errors")
            self.btn_filter_err.setCheckable(True)
            self.btn_filter_err.setCursor(Qt.CursorShape.PointingHandCursor)

            self.btn_filter_warn.setObjectName("severityFilter")
            self.btn_filter_err.setObjectName("severityFilter")
            self.btn_filter_err.setProperty("kind", "error")

            filter_layout.addWidget(self.btn_filter_warn)
            filter_layout.addWidget(self.btn_filter_err)

            self.btn_filter_warn.toggled.connect(self._on_filter_warn)
            self.btn_filter_err.toggled.connect(self._on_filter_err)

            layout.addWidget(self._filter_bar)

            # ── Tab 1: Structured Progress ──
            self._model = ProgressTableModel()

            from PySide6.QtCore import QSortFilterProxyModel
            class SeverityProxyModel(QSortFilterProxyModel):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.severity_filter = None

                def filterAcceptsRow(self, source_row, source_parent):
                    if not self.severity_filter:
                        return True
                    model = self.sourceModel()
                    if not hasattr(model, "_records") or source_row >= len(model._records):
                        return True
                    return model._records[source_row].severity == self.severity_filter

            self._proxy = SeverityProxyModel()
            self._proxy.setSourceModel(self._model)

            self._table = QTableView()
            self._table.setModel(self._proxy)
            self._table.setAlternatingRowColors(True)
            self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
            self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
            self._table.verticalHeader().setVisible(False)
            self._table.setShowGrid(False)
            self._table.setObjectName("logTable")

            header = self._table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self._table.verticalHeader().setDefaultSectionSize(28)

            self._auto_scroll = True

            # ── Tab 0: Training History (per-epoch, pandas-style aligned) ──
            self._history_model = EpochHistoryModel()
            self._history_table = QTableView()
            self._history_table.setModel(self._history_model)
            self._history_table.setAlternatingRowColors(True)
            self._history_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
            self._history_table.verticalHeader().setVisible(False)
            self._history_table.setShowGrid(False)
            self._history_table.setObjectName("logTable")
            h_header = self._history_table.horizontalHeader()
            h_header.setStretchLastSection(True)
            h_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self._history_table.verticalHeader().setDefaultSectionSize(26)
            self._history_auto_scroll = True

            # Empty-state placeholder shown until the first history rows arrive.
            self._history_placeholder = QLabel(
                "Training history will appear here per epoch,\n"
                "aligned like a table — as soon as the first epoch completes."
            )
            self._history_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._history_placeholder.setObjectName("fieldHint")
            self._history_stack = QWidget()
            _hist_l = QVBoxLayout(self._history_stack)
            _hist_l.setContentsMargins(0, 0, 0, 0)
            _hist_l.addWidget(self._history_placeholder)
            _hist_l.addWidget(self._history_table)
            self._history_table.setVisible(False)

            self._tabs.addTab(self._history_stack, "History")
            self._tabs.addTab(self._table, "Progress")

            # ── Tab 2: Raw Log (placeholder - the actual QPlainTextEdit
            #    is injected by STLRPSTrainTab via set_raw_log_widget) ──
            self._raw_log_placeholder = QLabel(
                "Waiting for training logs...\nMetrics will appear after the first parsed training line."
            )
            self._raw_log_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._raw_log_placeholder.setObjectName("fieldHint")
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
            """Add a record and auto-scroll only if the user is at the bottom."""
            sb = self._table.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 2
            self._model.append_record(record)

            if record.severity == "warning":
                self._warn_count += 1
                self.btn_filter_warn.setText(f"{self._warn_count} Warnings")
            elif record.severity == "error":
                self._error_count += 1
                self.btn_filter_err.setText(f"{self._error_count} Errors")

            if self._auto_scroll and at_bottom:
                self._table.scrollToBottom()

        def _on_filter_warn(self, checked: bool) -> None:
            if checked:
                self.btn_filter_err.setChecked(False)
                self._proxy.severity_filter = "warning"
            else:
                self._proxy.severity_filter = None
            self._proxy.invalidateFilter()

        def _on_filter_err(self, checked: bool) -> None:
            if checked:
                self.btn_filter_warn.setChecked(False)
                self._proxy.severity_filter = "error"
            else:
                self._proxy.severity_filter = None
            self._proxy.invalidateFilter()

        def load_history_file(self, path: str) -> None:
            """Load the per-epoch training history table from a csv/jsonl file."""
            rows = load_history_rows(path)
            if not rows:
                return
            sb = self._history_table.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 2
            self._history_model.set_rows(rows)
            self._history_placeholder.setVisible(False)
            self._history_table.setVisible(True)
            if self._history_auto_scroll and at_bottom:
                self._history_table.scrollToBottom()

        def clear(self) -> None:
            self._model.clear_records()
            self._warn_count = 0
            self._error_count = 0
            self.btn_filter_warn.setText("0 Warnings")
            self.btn_filter_err.setText("0 Errors")
            self.btn_filter_warn.setChecked(False)
            self.btn_filter_err.setChecked(False)

            if hasattr(self, "_history_model"):
                self._history_model.set_rows([])
                self._history_table.setVisible(False)
                self._history_placeholder.setVisible(True)

        def set_auto_scroll(self, enabled: bool) -> None:
            self._auto_scroll = enabled
