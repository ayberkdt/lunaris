# -*- coding: utf-8 -*-
"""
Execution Console — the buffered log panel for Lunaris Mission Studio.

This replaces the previous fragile ``QTextEdit.append(<html>)`` approach, which
caused extra paragraph spacing, autoscroll races, and poor performance during
high-volume subprocess streaming. The new design:

* keeps an internal :class:`LogEntry` model and a small **pending queue**,
* flushes the queue to the widget on a short timer (batched inserts), so a busy
  propagation run never triggers thousands of individual Qt operations,
* renders into a :class:`QPlainTextEdit` with controlled, colored ``QTextCursor``
  insertion (stable line spacing, clean copy, no HTML layout bugs),
* bounds retained output to :data:`MAX_LOG_LINES`,
* gives the user Pause / Copy / Clear / Save / Collapse controls plus
  Auto-scroll / Wrap / Show-timestamps toggles.

The widget is self-contained (its own QSS) so it can be constructed and tested
in isolation, and exposes a small public API used by ``MainWindow``:
``append``, ``append_separator``, ``clear``, ``copy_to_clipboard``,
``save_to_file``, ``set_collapsed`` / ``toggle_collapsed`` and the
``collapsed_changed`` signal.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui.core.ui_commons import LOG_COLORS, THEME, find_project_root, get_icon

# Maximum number of retained log lines. A long propagation run must never make
# the UI unusable, so older lines are trimmed once this bound is exceeded.
MAX_LOG_LINES = 10000

# How often the pending queue is flushed to the widget (milliseconds). Batching
# keeps the UI responsive under high-volume subprocess output.
_FLUSH_INTERVAL_MS = 80

# Header-only height when collapsed, and the minimum height when expanded.
COLLAPSED_HEIGHT = 46
EXPANDED_MIN_HEIGHT = 200

# Short, scannable severity labels for the tag column.
_SEVERITY_LABELS = {
    "error": "ERROR",
    "warning": "WARN",
    "success": "OK",
    "system": "SYS",
    "info": "INFO",
    "debug": "DBG",
}

# Width the bracketed tag is padded to so the message column stays aligned.
_TAG_WIDTH = 9


@dataclass
class LogEntry:
    """One console line: a timestamp, severity, optional source, and message."""

    timestamp: str
    severity: str
    source: str
    message: str


def detect_severity(text: str) -> str:
    """Best-effort severity classification from a raw log line.

    Mirrors the previous ``MainWindow._parse_log_severity`` heuristics so log
    lines that arrive without an explicit severity still get a sensible tag.
    """
    t = text.lower()
    if any(p in t for p in ("[err]", "error:", "failed", "exception", "traceback", "critical")):
        return "error"
    if any(p in t for p in ("[warning]", "[warn]", "warning:", "caution", "deprecated")):
        return "warning"
    if any(p in t for p in ("success", "finished", "completed", "✓", "passed")):
        return "success"
    if any(p in t for p in ("[system]", "[ui]", "initializing", "loading", "validating")):
        return "system"
    return "info"


class ExecutionLogPanel(QtWidgets.QWidget):
    """A batched, themed execution-log console widget."""

    collapsed_changed = QtCore.Signal(bool)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        output_dir_provider: Optional[Callable[[], str]] = None,
    ):
        super().__init__(parent)
        self.setObjectName("logPanel")

        self._output_dir_provider = output_dir_provider

        # Model + pending queue (both bounded). The queue is filled by append()
        # — possibly from a worker thread — and drained on the GUI thread.
        self._entries: "deque[LogEntry]" = deque(maxlen=MAX_LOG_LINES)
        self._pending: "deque[LogEntry]" = deque(maxlen=MAX_LOG_LINES)
        self._lock = threading.Lock()

        self._collapsed = False
        self._paused = False
        self._show_timestamps = True
        self._status_revert_timer: Optional[QtCore.QTimer] = None

        self._build_ui()
        self._build_formats()
        self._apply_style()

        # Flush timer: batches inserts so streaming output stays smooth.
        self._flush_timer = QtCore.QTimer(self)
        self._flush_timer.setInterval(_FLUSH_INTERVAL_MS)
        self._flush_timer.timeout.connect(self._flush)
        self._flush_timer.start()

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # --- Header ("Execution Console") -----------------------------------
        self.header = QtWidgets.QFrame()
        self.header.setObjectName("logHeader")
        self.header.setMinimumHeight(COLLAPSED_HEIGHT)
        hl = QtWidgets.QHBoxLayout(self.header)
        hl.setContentsMargins(12, 6, 12, 6)
        hl.setSpacing(10)

        self.btn_collapse = QtWidgets.QToolButton()
        self.btn_collapse.setObjectName("logCollapseButton")
        self.btn_collapse.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_collapse.setIcon(get_icon("fa6s.chevron-down", THEME["fg_soft"]))
        self.btn_collapse.setToolTip("Collapse console")
        self.btn_collapse.setStyleSheet("border: none; background: transparent;")
        self.btn_collapse.clicked.connect(self.toggle_collapsed)
        hl.addWidget(self.btn_collapse)

        title_box = QtWidgets.QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(0)
        self.lbl_title = QtWidgets.QLabel("Execution Console")
        self.lbl_title.setObjectName("logTitle")
        self.lbl_subtitle = QtWidgets.QLabel("Live process output")
        self.lbl_subtitle.setObjectName("logSubtitle")
        title_box.addWidget(self.lbl_title)
        title_box.addWidget(self.lbl_subtitle)
        hl.addLayout(title_box)

        self.status_chip = QtWidgets.QLabel("Idle")
        self.status_chip.setObjectName("logStatusChip")
        self.status_chip.setAlignment(QtCore.Qt.AlignCenter)
        hl.addWidget(self.status_chip)

        hl.addStretch(1)

        # Toggles
        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setToolTip("Keep the latest output in view")
        hl.addWidget(self.chk_autoscroll)

        self.chk_wrap = QtWidgets.QCheckBox("Wrap")
        self.chk_wrap.setChecked(False)
        self.chk_wrap.setToolTip("Wrap long lines to the console width")
        self.chk_wrap.toggled.connect(self._on_wrap_toggled)
        hl.addWidget(self.chk_wrap)

        self.chk_timestamps = QtWidgets.QCheckBox("Timestamps")
        self.chk_timestamps.setChecked(True)
        self.chk_timestamps.setToolTip("Show the [HH:MM:SS] prefix")
        self.chk_timestamps.toggled.connect(self._on_timestamps_toggled)
        hl.addWidget(self.chk_timestamps)

        # Action buttons
        self.btn_pause = self._toolbar_button("Pause", "Pause live output (buffered)")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        hl.addWidget(self.btn_pause)

        self.btn_copy = self._toolbar_button("Copy", "Copy console text to clipboard")
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        hl.addWidget(self.btn_copy)

        self.btn_clear = self._toolbar_button("Clear", "Clear the console")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        hl.addWidget(self.btn_clear)

        self.btn_save = self._toolbar_button("Save", "Save console text to a file")
        self.btn_save.clicked.connect(self._on_save_clicked)
        hl.addWidget(self.btn_save)

        outer.addWidget(self.header)

        # --- Console body ----------------------------------------------------
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setObjectName("logConsole")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(MAX_LOG_LINES)
        self.console.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.console.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.console.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.console.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        outer.addWidget(self.console, 1)

    def _toolbar_button(self, text: str, tooltip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setObjectName("logToolbarButton")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setToolTip(tooltip)
        btn.setMinimumHeight(28)
        # Compact but readable — no cramped 60 px fixed widths.
        fm = btn.fontMetrics()
        btn.setMinimumWidth(max(64, fm.horizontalAdvance(text) + 28))
        return btn

    def _build_formats(self) -> None:
        """Pre-build the QTextCharFormats used for colored cursor insertion."""
        def fmt(color: str) -> QtGui.QTextCharFormat:
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            return f

        self._fmt_ts = fmt(LOG_COLORS["timestamp"])
        self._fmt_body = fmt(LOG_COLORS["default"])
        self._fmt_sep = fmt(THEME["border"])
        self._fmt_sev = {
            sev: fmt(LOG_COLORS.get(sev, LOG_COLORS["default"]))
            for sev in ("error", "warning", "success", "system", "info", "debug")
        }

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QFrame#logHeader {{
                background: {THEME['bg_shell']};
                border: 1px solid {THEME['border_soft']};
                border-radius: 10px;
            }}
            QLabel#logTitle {{
                color: {THEME['fg_soft']};
                font-size: 10.5pt;
                font-weight: 700;
            }}
            QLabel#logSubtitle {{
                color: {THEME['fg_muted']};
                font-size: 8pt;
            }}
            QLabel#logStatusChip {{
                color: {THEME['fg_muted']};
                background: {THEME['bg_card_alt']};
                border: 1px solid {THEME['border_soft']};
                border-radius: 8px;
                padding: 2px 10px;
                font-size: 8.5pt;
                font-weight: 600;
            }}
            QPushButton#logToolbarButton {{
                background: {THEME['bg_card_alt']};
                color: {THEME['fg_soft']};
                border: 1px solid {THEME['border']};
                border-radius: 7px;
                padding: 4px 12px;
                font-size: 9pt;
                font-weight: 600;
            }}
            QPushButton#logToolbarButton:hover {{
                border-color: {THEME['accent_deep']};
                color: {THEME['fg_main']};
            }}
            QPushButton#logToolbarButton:checked {{
                background: {THEME['accent_dim']};
                border-color: {THEME['accent']};
                color: {THEME['fg_main']};
            }}
            QCheckBox {{
                color: {THEME['fg_muted']};
                font-size: 9pt;
                spacing: 6px;
            }}
            QPlainTextEdit#logConsole {{
                background: {THEME['bg_log']};
                color: {THEME['fg_main']};
                border: 1px solid {THEME['border']};
                border-radius: 10px;
                padding: 10px;
                font-family: "Cascadia Mono", "Consolas", "Courier New", monospace;
                font-size: 9.5pt;
            }}
            """
        )

    # ---------------------------------------------------------------- public
    def append(self, message: str, severity: str = "auto", source: str = "") -> None:
        """Queue a log line. Safe to call from a worker thread.

        ``severity`` may be an explicit level or ``"auto"`` to classify from the
        message text. Empty messages are ignored.
        """
        if message is None:
            return
        text = str(message).rstrip("\n")
        if not text.strip():
            return
        sev = severity if severity and severity != "auto" else detect_severity(text)
        entry = LogEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            severity=sev,
            source=str(source or ""),
            message=text,
        )
        with self._lock:
            self._pending.append(entry)

    def append_separator(self) -> None:
        """Queue a muted divider rule."""
        with self._lock:
            self._pending.append(
                LogEntry(
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    severity="separator",
                    source="",
                    message="",
                )
            )

    def clear(self) -> None:
        """Clear the console and the retained model."""
        with self._lock:
            self._pending.clear()
        self._entries.clear()
        self.console.clear()

    def copy_to_clipboard(self) -> bool:
        text = self._plain_text(with_timestamps=self._show_timestamps)
        if not text.strip():
            self._flash_status("Nothing to copy")
            return False
        QtWidgets.QApplication.clipboard().setText(text)
        self._flash_status("Copied to clipboard")
        return True

    def save_to_file(self, path: str) -> bool:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._plain_text(with_timestamps=True))
            self._flash_status("Saved")
            return True
        except Exception:
            self._flash_status("Save failed")
            return False

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """Show/hide the console body; the host adjusts the splitter via the signal."""
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            # Still emit so a host can re-assert splitter sizing if needed.
            self.collapsed_changed.emit(collapsed)
            return
        self._collapsed = collapsed

        self.console.setVisible(not collapsed)
        for w in (
            self.chk_autoscroll, self.chk_wrap, self.chk_timestamps,
            self.btn_pause, self.btn_copy, self.btn_clear, self.btn_save,
            self.status_chip,
        ):
            w.setVisible(not collapsed)

        icon = "fa6s.chevron-up" if collapsed else "fa6s.chevron-down"
        self.btn_collapse.setIcon(get_icon(icon, THEME["fg_soft"]))
        self.btn_collapse.setToolTip("Expand console" if collapsed else "Collapse console")
        self.collapsed_changed.emit(collapsed)

    # --------------------------------------------------------------- internal
    def _on_pause_toggled(self, checked: bool) -> None:
        self._paused = bool(checked)
        self.btn_pause.setText("Resume" if checked else "Pause")
        if checked:
            self._set_status("Paused")
        else:
            self._set_status("Live process output")
            self._flush()  # catch up on anything buffered while paused

    def _on_wrap_toggled(self, checked: bool) -> None:
        self.console.setLineWrapMode(
            QtWidgets.QPlainTextEdit.WidgetWidth if checked else QtWidgets.QPlainTextEdit.NoWrap
        )

    def _on_timestamps_toggled(self, checked: bool) -> None:
        self._show_timestamps = bool(checked)
        self._rebuild_console()

    def _on_clear_clicked(self) -> None:
        self.clear()
        self.append("[UI] Console cleared.", severity="system")

    def _on_save_clicked(self) -> None:
        default_dir: Optional[Path] = None
        if self._output_dir_provider is not None:
            try:
                provided = self._output_dir_provider()
                if provided and Path(provided).is_dir():
                    default_dir = Path(provided)
            except Exception:
                default_dir = None
        if default_dir is None:
            default_dir = find_project_root() / "outputs" / "ui" / "logs"
            try:
                default_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                default_dir = Path.home()

        fname = f"lunaris_execution_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Execution Log", str(default_dir / fname),
            "Text Files (*.txt);;All Files (*.*)",
        )
        if path:
            self.save_to_file(path)

    def _flush(self) -> None:
        """Drain the pending queue into the console (GUI thread)."""
        if self._paused:
            return
        with self._lock:
            if not self._pending:
                return
            batch = list(self._pending)
            self._pending.clear()

        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.beginEditBlock()
        for entry in batch:
            self._entries.append(entry)
            self._insert_entry(cursor, entry)
        cursor.endEditBlock()

        if self.chk_autoscroll.isChecked():
            self.console.moveCursor(QtGui.QTextCursor.End)
            self.console.ensureCursorVisible()

    def _insert_entry(self, cursor: QtGui.QTextCursor, entry: LogEntry) -> None:
        if entry.severity == "separator":
            cursor.insertText("─" * 56 + "\n", self._fmt_sep)
            return
        if self._show_timestamps:
            cursor.insertText(f"[{entry.timestamp}] ", self._fmt_ts)
        tag = entry.source.upper() if entry.source else _SEVERITY_LABELS.get(entry.severity, "INFO")
        sev_fmt = self._fmt_sev.get(entry.severity, self._fmt_body)
        cursor.insertText(f"[{tag}]".ljust(_TAG_WIDTH), sev_fmt)
        cursor.insertText(entry.message + "\n", self._fmt_body)

    def _rebuild_console(self) -> None:
        """Re-render the whole model (used when the timestamp toggle changes)."""
        self.console.clear()
        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.beginEditBlock()
        for entry in list(self._entries):
            self._insert_entry(cursor, entry)
        cursor.endEditBlock()
        if self.chk_autoscroll.isChecked():
            self.console.moveCursor(QtGui.QTextCursor.End)
            self.console.ensureCursorVisible()

    def _plain_text(self, *, with_timestamps: bool) -> str:
        lines = []
        for entry in list(self._entries):
            if entry.severity == "separator":
                lines.append("─" * 56)
                continue
            prefix = f"[{entry.timestamp}] " if with_timestamps else ""
            tag = entry.source.upper() if entry.source else _SEVERITY_LABELS.get(entry.severity, "INFO")
            lines.append(f"{prefix}{('[' + tag + ']').ljust(_TAG_WIDTH)}{entry.message}")
        return "\n".join(lines)

    # -- status chip helpers --
    def _set_status(self, text: str) -> None:
        self.status_chip.setText(text)

    def _flash_status(self, text: str, revert_ms: int = 1600) -> None:
        self._set_status(text)
        if self._status_revert_timer is not None:
            self._status_revert_timer.stop()
        self._status_revert_timer = QtCore.QTimer(self)
        self._status_revert_timer.setSingleShot(True)
        self._status_revert_timer.timeout.connect(
            lambda: self._set_status("Paused" if self._paused else "Live process output")
        )
        self._status_revert_timer.start(revert_ms)
