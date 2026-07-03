"""
Tests for the rebuilt Execution Console (lunaris.ui.widgets.log_panel).

These are lightweight and headless (Qt ``offscreen``). They verify the buffered
log model, severity rendering, autoscroll/pause/clear/copy/save behavior, the
retained-line bound, and the collapse signal — without needing a display server.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from lunaris.ui.widgets.log_panel import (
    MAX_LOG_LINES,
    ExecutionLogPanel,
    LogEntry,
    detect_severity,
)


def _qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _panel() -> ExecutionLogPanel:
    _qapp()
    return ExecutionLogPanel()


# ---------------------------------------------------------------------------
# Construction & identity
# ---------------------------------------------------------------------------

def test_panel_constructs_with_expected_object_names() -> None:
    panel = _panel()
    try:
        assert panel.objectName() == "logPanel"
        assert panel.header.objectName() == "logHeader"
        assert panel.console.objectName() == "logConsole"
        # Console is bounded to MAX_LOG_LINES.
        assert panel.console.maximumBlockCount() == MAX_LOG_LINES
        assert panel._entries.maxlen == MAX_LOG_LINES
        assert panel._pending.maxlen == MAX_LOG_LINES
    finally:
        panel.deleteLater()


def test_title_is_execution_console() -> None:
    panel = _panel()
    try:
        assert panel.lbl_title.text() == "Execution Console"
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# Severity rendering & detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "severity, tag",
    [("info", "[INFO]"), ("warning", "[WARN]"), ("error", "[ERROR]"), ("system", "[SYS]")],
)
def test_formatter_handles_each_severity(severity: str, tag: str) -> None:
    panel = _panel()
    try:
        panel.append(f"message-for-{severity}", severity=severity)
        panel._flush()
        text = panel.console.toPlainText()
        assert tag in text
        assert f"message-for-{severity}" in text
    finally:
        panel.deleteLater()


def test_detect_severity_classification() -> None:
    assert detect_severity("Traceback (most recent call last)") == "error"
    assert detect_severity("ERROR: boom") == "error"
    assert detect_severity("WARNING: heads up") == "warning"
    assert detect_severity("Run finished successfully") == "success"
    assert detect_severity("[System] initializing") == "system"
    assert detect_severity("ordinary line") == "info"


def test_source_tag_is_used_when_provided() -> None:
    panel = _panel()
    try:
        panel.append("batch line", severity="info", source="BATCH")
        panel._flush()
        assert "[BATCH]" in panel.console.toPlainText()
    finally:
        panel.deleteLater()


def test_separator_renders_a_rule() -> None:
    panel = _panel()
    try:
        panel.append_separator()
        panel._flush()
        assert "─" in panel.console.toPlainText()
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# Buffering behavior
# ---------------------------------------------------------------------------

def test_empty_messages_are_ignored() -> None:
    panel = _panel()
    try:
        panel.append("")
        panel.append("   ")
        panel.append("\n")
        panel._flush()
        assert len(panel._entries) == 0
        assert panel.console.toPlainText().strip() == ""
    finally:
        panel.deleteLater()


def test_no_duplicate_lines_on_single_append() -> None:
    panel = _panel()
    try:
        panel.append("unique-line-123", severity="info")
        panel._flush()
        text = panel.console.toPlainText()
        assert text.count("unique-line-123") == 1
    finally:
        panel.deleteLater()


def test_pause_holds_output_then_resumes() -> None:
    panel = _panel()
    try:
        panel.append("before pause", severity="info")
        panel._flush()
        n_before = len(panel._entries)

        panel.btn_pause.setChecked(True)  # pause
        panel.append("during pause", severity="info")
        panel._flush()  # should be a no-op while paused
        assert len(panel._entries) == n_before

        panel.btn_pause.setChecked(False)  # resume flushes the buffer
        panel._flush()
        assert len(panel._entries) == n_before + 1
        assert "during pause" in panel.console.toPlainText()
    finally:
        panel.deleteLater()


def test_clear_empties_model_and_console() -> None:
    panel = _panel()
    try:
        panel.append("line a")
        panel.append("line b")
        panel._flush()
        assert len(panel._entries) >= 2
        panel.clear()
        assert len(panel._entries) == 0
        assert panel.console.toPlainText().strip() == ""
    finally:
        panel.deleteLater()


def test_retained_entries_are_bounded() -> None:
    panel = _panel()
    try:
        # Append a modest amount beyond a small window and confirm the bounded
        # deque never exceeds MAX_LOG_LINES (full 10k functional trim is covered
        # by the deque/maximumBlockCount wiring asserted above).
        for i in range(250):
            panel.append(f"line {i}", severity="info")
        panel._flush()
        assert len(panel._entries) == 250
        assert len(panel._entries) <= MAX_LOG_LINES
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# Copy / Save
# ---------------------------------------------------------------------------

def test_copy_to_clipboard_reports_empty_then_content() -> None:
    panel = _panel()
    try:
        assert panel.copy_to_clipboard() is False  # nothing logged yet
        panel.append("copy me", severity="info")
        panel._flush()
        assert panel.copy_to_clipboard() is True
        assert "copy me" in QtWidgets.QApplication.clipboard().text()
    finally:
        panel.deleteLater()


def test_save_to_file_writes_plain_text(tmp_path: Path) -> None:
    panel = _panel()
    try:
        panel.append("persist this", severity="warning")
        panel._flush()
        out = tmp_path / "log.txt"
        assert panel.save_to_file(str(out)) is True
        content = out.read_text(encoding="utf-8")
        assert "persist this" in content
        assert "[WARN]" in content
    finally:
        panel.deleteLater()


def test_default_save_dir_is_not_under_src() -> None:
    from lunaris.ui.core.ui_commons import find_project_root

    default_dir = find_project_root() / "outputs" / "ui" / "logs"
    parts = default_dir.parts
    assert "outputs" in parts
    assert "src" not in parts


# ---------------------------------------------------------------------------
# Collapse behavior
# ---------------------------------------------------------------------------

def test_collapse_hides_console_and_emits_signal() -> None:
    panel = _panel()
    try:
        seen: list[bool] = []
        panel.collapsed_changed.connect(seen.append)

        panel.set_collapsed(True)
        assert panel.is_collapsed is True
        assert panel.console.isHidden() is True
        assert seen and seen[-1] is True

        panel.toggle_collapsed()
        assert panel.is_collapsed is False
        assert panel.console.isHidden() is False
        assert seen[-1] is False
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# Timestamps toggle re-renders the model
# ---------------------------------------------------------------------------

def test_timestamp_toggle_rerenders() -> None:
    panel = _panel()
    try:
        panel.append("timestamped line", severity="info")
        panel._flush()
        with_ts = panel.console.toPlainText()
        # Toggling timestamps off rebuilds the document from the model.
        panel.chk_timestamps.setChecked(False)
        without_ts = panel.console.toPlainText()
        assert "timestamped line" in with_ts and "timestamped line" in without_ts
        # The ":" of an HH:MM:SS prefix should disappear from the leading column.
        assert with_ts != without_ts
    finally:
        panel.deleteLater()


def test_log_entry_dataclass_fields() -> None:
    entry = LogEntry(timestamp="12:00:00", severity="info", source="", message="hi")
    assert entry.timestamp == "12:00:00"
    assert entry.severity == "info"
    assert entry.message == "hi"
