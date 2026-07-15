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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui.components.primitives import CompactSearchField, OverflowMenuButton
from lunaris.ui.core.log_stream import is_near_bottom
from lunaris.ui.core.ui_commons import (
    LOG_COLORS,
    THEME,
    NoWheelComboBox,
    find_project_root,
    get_icon,
)
from lunaris.ui.theme.tokens import DESIGN_TOKENS

# Maximum number of retained log lines. A long propagation run must never make
# the UI unusable, so older lines are trimmed once this bound is exceeded.
MAX_LOG_LINES = 10000

# How often the pending queue is flushed to the widget (milliseconds). Batching
# keeps the UI responsive under high-volume subprocess output.
_FLUSH_INTERVAL_MS = 33

# Header-only height when collapsed, and the minimum height when expanded.
COLLAPSED_HEIGHT = DESIGN_TOKENS.layout.console_collapsed_height
EXPANDED_MIN_HEIGHT = DESIGN_TOKENS.layout.console_expanded_min_height

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

# Total character width of a console divider rule.
_SEPARATOR_WIDTH = 56


def _separator_text(label: str = "") -> str:
    """Return a divider rule, optionally with a centered *label*.

    Plain rule: ``────────…────────``.
    Labeled:    ``──────── New Mission Run ────────``.
    """
    if not label:
        return "─" * _SEPARATOR_WIDTH
    tag = f" {label.strip()} "
    fill = max(2, _SEPARATOR_WIDTH - len(tag))
    left = fill // 2
    right = fill - left
    return "─" * left + tag + "─" * right


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


class ExecutionConsoleDock(QtWidgets.QWidget):
    """A batched, themed execution-log console widget."""

    collapsed_changed = QtCore.Signal(bool)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        output_dir_provider: Callable[[], str] | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("logPanel")

        self._output_dir_provider = output_dir_provider

        # Model + pending queue (both bounded). The queue is filled by append()
        # — possibly from a worker thread — and drained on the GUI thread.
        self._entries: deque[LogEntry] = deque(maxlen=MAX_LOG_LINES)
        self._pending: deque[LogEntry] = deque(maxlen=MAX_LOG_LINES)
        self._lock = threading.Lock()

        self._collapsed = False
        self._paused = False
        self._toggle_handler: Callable[[], None] | None = None
        self._show_timestamps = True
        self._status_revert_timer: QtCore.QTimer | None = None
        self._severity_counts = {"warning": 0, "error": 0}

        self._build_ui()
        self._build_formats()

        # Flush timer: batches inserts so streaming output stays smooth.
        self._flush_timer = QtCore.QTimer(self)
        self._flush_timer.setInterval(_FLUSH_INTERVAL_MS)
        self._flush_timer.timeout.connect(self._flush)
        self._flush_timer.start()

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(DESIGN_TOKENS.spacing.sm)

        # The compact strip remains visible when the console body is collapsed.
        self.header = QtWidgets.QFrame()
        self.header.setObjectName("logHeader")
        self.header.setFixedHeight(COLLAPSED_HEIGHT)
        hl = QtWidgets.QHBoxLayout(self.header)
        hl.setContentsMargins(
            DESIGN_TOKENS.spacing.md, DESIGN_TOKENS.spacing.xs,
            DESIGN_TOKENS.spacing.md, DESIGN_TOKENS.spacing.xs,
        )
        hl.setSpacing(DESIGN_TOKENS.spacing.sm)

        self.btn_collapse = QtWidgets.QToolButton()
        self.btn_collapse.setObjectName("logCollapseButton")
        self.btn_collapse.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_collapse.setIcon(get_icon("fa6s.chevron-down", THEME["fg_soft"]))
        self.btn_collapse.setText("v")
        self.btn_collapse.setFixedSize(28, 28)
        self.btn_collapse.setAccessibleName("Collapse execution console")
        self.btn_collapse.setToolTip("Collapse console")
        self.btn_collapse.clicked.connect(self._on_collapse_clicked)
        hl.addWidget(self.btn_collapse)

        self.lbl_title = QtWidgets.QLabel("Execution Console")
        self.lbl_title.setObjectName("logTitle")
        hl.addWidget(self.lbl_title)

        self.lbl_subtitle = QtWidgets.QLabel("Live process output")
        self.lbl_subtitle.setObjectName("logSubtitle")
        self.lbl_subtitle.hide()

        self.lbl_latest = QtWidgets.QLabel("No output yet")
        self.lbl_latest.setObjectName("logLatestMessage")
        self.lbl_latest.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_latest.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        hl.addWidget(self.lbl_latest, 1)

        self.status_chip = QtWidgets.QLabel("Idle")
        self.status_chip.setObjectName("logStatusChip")
        self.status_chip.setAlignment(QtCore.Qt.AlignCenter)
        hl.addWidget(self.status_chip)

        self.lbl_warning_count = QtWidgets.QLabel("W 0")
        self.lbl_warning_count.setObjectName("logCounter")
        self.lbl_warning_count.setToolTip("Warning count")
        hl.addWidget(self.lbl_warning_count)
        self.lbl_error_count = QtWidgets.QLabel("E 0")
        self.lbl_error_count.setObjectName("logCounter")
        self.lbl_error_count.setToolTip("Error count")
        hl.addWidget(self.lbl_error_count)
        outer.addWidget(self.header)

        self.body = QtWidgets.QWidget()
        self.body.setObjectName("logBody")
        body_layout = QtWidgets.QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(DESIGN_TOKENS.spacing.sm)

        self.toolbar = QtWidgets.QFrame()
        self.toolbar.setObjectName("toolbar")
        tl = QtWidgets.QHBoxLayout(self.toolbar)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(DESIGN_TOKENS.spacing.sm)

        self.search_field = CompactSearchField("Search console")
        self.search_field.textChanged.connect(self._rebuild_console)
        tl.addWidget(self.search_field, 1)

        self.severity_filter = NoWheelComboBox()
        self.severity_filter.setAccessibleName("Severity filter")
        self.severity_filter.addItems(
            ["All levels", "Errors", "Warnings", "Info", "System", "Success", "Debug"]
        )
        self.severity_filter.currentIndexChanged.connect(self._rebuild_console)
        tl.addWidget(self.severity_filter)

        self.chk_autoscroll = QtWidgets.QCheckBox("Follow output")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setToolTip("Keep the latest output in view")
        tl.addWidget(self.chk_autoscroll)

        self.chk_wrap = QtWidgets.QCheckBox("Wrap")
        self.chk_wrap.setChecked(False)
        self.chk_wrap.setToolTip("Wrap long lines to the console width")
        self.chk_wrap.toggled.connect(self._on_wrap_toggled)

        self.chk_timestamps = QtWidgets.QCheckBox("Timestamps")
        self.chk_timestamps.setChecked(True)
        self.chk_timestamps.setToolTip("Show the [HH:MM:SS] prefix")
        self.chk_timestamps.toggled.connect(self._on_timestamps_toggled)

        self.btn_pause = self._toolbar_button("Pause", "Pause live output (buffered)")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        self.btn_pause.hide()

        self.btn_copy = self._toolbar_button("Copy", "Copy console text to clipboard")
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        tl.addWidget(self.btn_copy)

        self.btn_clear = self._toolbar_button("Clear", "Clear the console")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        tl.addWidget(self.btn_clear)

        self.btn_save = self._toolbar_button("Save", "Save console text to a file")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_save.hide()

        menu = QtWidgets.QMenu(self)
        self.action_pause = menu.addAction("Pause output")
        self.action_pause.setCheckable(True)
        self.action_pause.toggled.connect(self.btn_pause.setChecked)
        self.btn_pause.toggled.connect(self.action_pause.setChecked)
        self.action_wrap = menu.addAction("Wrap long lines")
        self.action_wrap.setCheckable(True)
        self.action_wrap.toggled.connect(self.chk_wrap.setChecked)
        self.chk_wrap.toggled.connect(self.action_wrap.setChecked)
        self.action_timestamps = menu.addAction("Show timestamps")
        self.action_timestamps.setCheckable(True)
        self.action_timestamps.setChecked(True)
        self.action_timestamps.toggled.connect(self.chk_timestamps.setChecked)
        self.chk_timestamps.toggled.connect(self.action_timestamps.setChecked)
        menu.addSeparator()
        menu.addAction("Save log...", self._on_save_clicked)
        self.btn_more = OverflowMenuButton(menu)
        tl.addWidget(self.btn_more)
        body_layout.addWidget(self.toolbar)

        self.console = QtWidgets.QPlainTextEdit()
        self.console.setObjectName("logConsole")
        self.console.setAccessibleName("Execution console log")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(MAX_LOG_LINES)
        self.console.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.console.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.console.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.console.setUndoRedoEnabled(False)
        self.console.setCenterOnScroll(False)
        self.console.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        body_layout.addWidget(self.console, 1)
        outer.addWidget(self.body, 1)

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

    def append_separator(self, label: str = "") -> None:
        """Queue a muted divider rule, optionally with a centered *label*."""
        with self._lock:
            self._pending.append(
                LogEntry(
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    severity="separator",
                    source="",
                    message=str(label or ""),
                )
            )

    def append_run_separator(self, label: str = "New Mission Run") -> None:
        """Queue a labeled separator marking the start of a new run.

        Starting a run inserts this divider instead of clearing the console, so
        useful pre-flight and prior context is preserved. Explicit ``clear`` is
        still available as a deliberate user action.
        """
        self.append_separator(label)

    def clear(self) -> None:
        """Clear the console: pending queue, retained model, and the view.

        After a clear, no previously queued entry can reappear — the pending
        queue is emptied under the same lock that ``append`` uses, so anything
        enqueued *before* the clear is discarded while anything enqueued *after*
        renders normally on the next flush.
        """
        with self._lock:
            self._pending.clear()
        self._entries.clear()
        self.console.clear()
        self._severity_counts = {"warning": 0, "error": 0}
        self.lbl_warning_count.setText("W 0")
        self.lbl_error_count.setText("E 0")
        self.lbl_latest.setText("No output yet")

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

    def set_toggle_handler(self, handler: Callable[[], None] | None) -> None:
        """Route the header collapse button through *handler*.

        A host that animates the surrounding splitter registers its own toggle
        here so the header button and any external shortcut share one code
        path; without a handler the button toggles the widget directly.
        """
        self._toggle_handler = handler

    def _on_collapse_clicked(self) -> None:
        if self._toggle_handler is not None:
            self._toggle_handler()
        else:
            self.toggle_collapsed()

    def focus_search(self) -> None:
        """Expand the console (if collapsed) and move keyboard focus to its search field.

        Wired to a global shortcut so users can jump straight to filtering the
        execution log without reaching for the mouse.
        """
        if self._collapsed:
            self.set_collapsed(False)
        self.search_field.setFocus(QtCore.Qt.ShortcutFocusReason)
        self.search_field.selectAll()

    def set_collapsed(self, collapsed: bool) -> None:
        """Show/hide the console body; the host adjusts the splitter via the signal."""
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            # Still emit so a host can re-assert splitter sizing if needed.
            self.collapsed_changed.emit(collapsed)
            return
        self._collapsed = collapsed

        self.body.setVisible(not collapsed)
        self.console.setVisible(not collapsed)
        if collapsed:
            self.setMaximumHeight(COLLAPSED_HEIGHT)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Fixed,
            )
        else:
            self.setMaximumHeight(16_777_215)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Expanding,
            )

        icon = "fa6s.chevron-up" if collapsed else "fa6s.chevron-down"
        self.btn_collapse.setIcon(get_icon(icon, THEME["fg_soft"]))
        self.btn_collapse.setText("^" if collapsed else "v")
        self.btn_collapse.setAccessibleName(
            "Expand execution console" if collapsed else "Collapse execution console"
        )
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
            # Resume: drain everything buffered while paused, committing and
            # rendering it exactly once, in order.
            self._flush()

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
        default_dir: Path | None = None
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
        """Drain the pending queue into the model and append it to the view.

        ``_entries`` is the single source of truth for *retained* lines and is
        bounded by :data:`MAX_LOG_LINES`. While paused the drain is held back
        entirely, so buffered lines stay queued in order and are committed +
        rendered exactly once on resume — never duplicated, never reordered. The
        live path appends only the new batch (no full re-render per tick), which
        keeps high-volume streaming smooth.
        """
        if self._paused:
            return
        with self._lock:
            if not self._pending:
                return
            batch = list(self._pending)
            self._pending.clear()

        for entry in batch:
            self._entries.append(entry)
            if entry.severity in self._severity_counts:
                self._severity_counts[entry.severity] += 1
            if entry.severity != "separator":
                self.lbl_latest.setText(entry.message)

        self.lbl_warning_count.setText(f"W {self._severity_counts['warning']}")
        self.lbl_error_count.setText(f"E {self._severity_counts['error']}")

        self._append_entries_to_view(batch)

    def _is_near_bottom(self) -> bool:
        """True when the console viewport is already at (or near) the bottom."""
        sb = self.console.verticalScrollBar()
        return is_near_bottom(sb.value(), sb.maximum())

    def _should_autoscroll(self, was_at_bottom: bool) -> bool:
        """Auto-scroll only when following is enabled *and* the user was at end.

        Combining the explicit toggle with the live scroll position means manual
        upward scrolling is respected — new output never yanks the viewport back
        down — while returning to the bottom transparently resumes following.
        """
        return bool(self.chk_autoscroll.isChecked()) and bool(was_at_bottom)

    def _scroll_to_bottom(self) -> None:
        self.console.moveCursor(QtGui.QTextCursor.End)
        self.console.ensureCursorVisible()

    def _insert_entry(self, cursor: QtGui.QTextCursor, entry: LogEntry, *, leading_newline: bool) -> None:
        """Insert one entry as a single block (no trailing blank line)."""
        if leading_newline:
            cursor.insertText("\n", self._fmt_body)
        if entry.severity == "separator":
            cursor.insertText(_separator_text(entry.message), self._fmt_sep)
            return
        if self._show_timestamps:
            cursor.insertText(f"[{entry.timestamp}] ", self._fmt_ts)
        tag = entry.source.upper() if entry.source else _SEVERITY_LABELS.get(entry.severity, "INFO")
        sev_fmt = self._fmt_sev.get(entry.severity, self._fmt_body)
        cursor.insertText(f"[{tag}]".ljust(_TAG_WIDTH), sev_fmt)
        cursor.insertText(entry.message, self._fmt_body)

    def _append_entries_to_view(self, entries) -> None:
        """Append already-modelled *entries* to the widget, respecting filters."""
        visible = [e for e in entries if self._entry_matches_filters(e)]
        if not visible:
            return
        was_at_bottom = self._is_near_bottom()
        doc_nonempty = not self.console.document().isEmpty()
        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.beginEditBlock()
        for index, entry in enumerate(visible):
            self._insert_entry(cursor, entry, leading_newline=doc_nonempty or index > 0)
        cursor.endEditBlock()
        if self._should_autoscroll(was_at_bottom):
            self._scroll_to_bottom()

    def _rebuild_console(self, *_args) -> None:
        """Re-render the retained model using the active query and severity."""
        was_at_bottom = self._is_near_bottom()
        self.console.clear()
        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.beginEditBlock()
        first = True
        for entry in list(self._entries):
            if self._entry_matches_filters(entry):
                self._insert_entry(cursor, entry, leading_newline=not first)
                first = False
        cursor.endEditBlock()
        if self._should_autoscroll(was_at_bottom):
            self._scroll_to_bottom()

    def _entry_matches_filters(self, entry: LogEntry) -> bool:
        query = self.search_field.text().strip().casefold()
        if query and query not in f"{entry.source} {entry.message}".casefold():
            return False
        selected = self.severity_filter.currentText()
        severity_map = {
            "Errors": "error",
            "Warnings": "warning",
            "Info": "info",
            "System": "system",
            "Success": "success",
            "Debug": "debug",
        }
        expected = severity_map.get(selected)
        return expected is None or entry.severity == expected

    def _plain_text(self, *, with_timestamps: bool) -> str:
        lines = []
        for entry in list(self._entries):
            if entry.severity == "separator":
                lines.append(_separator_text(entry.message))
                continue
            prefix = f"[{entry.timestamp}] " if with_timestamps else ""
            tag = entry.source.upper() if entry.source else _SEVERITY_LABELS.get(entry.severity, "INFO")
            lines.append(f"{prefix}{('[' + tag + ']').ljust(_TAG_WIDTH)}{entry.message}")
        return "\n".join(lines)

    # -- status chip helpers --
    def _set_status(self, text: str) -> None:
        self.status_chip.setText(text)

    def set_run_status(self, state: str) -> None:
        """Reflect the application's run state in the console's status chip.

        The chip sits in a bar labelled "Execution Console", so users read it
        as the execution status. It previously only ever changed on
        pause/resume, which left it reading "Idle" for the whole of a run while
        the header showed a progress bar — two widgets on screen contradicting
        each other about the same fact. Pause is still the user's own explicit
        override of the stream and keeps precedence over the run state.
        """
        if self._paused:
            return
        self._set_status(
            {
                "running": "Running",
                "error": "Run error",
                "warning": "Validating",
            }.get(state, "Idle")
        )

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


# Historical name retained for all existing imports and tests.
ExecutionLogPanel = ExecutionConsoleDock
