"""
Behavioral tests for the Execution Console data flow (objective 1).

These exercise the deterministic model -> render pipeline directly on the
``ExecutionConsoleDock`` widget: pause/resume ordering, clear semantics, the
line limit at the model level, scroll-aware auto-scroll, copy/save consistency,
and the labeled run separator.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

from lunaris.ui.widgets.log_panel import MAX_LOG_LINES, ExecutionConsoleDock


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _panel() -> ExecutionConsoleDock:
    _app()
    return ExecutionConsoleDock()


def test_append_one_complete_line_reaches_model_and_view() -> None:
    p = _panel()
    try:
        p.append("nominal", severity="info")
        p._flush()
        assert len(p._entries) == 1
        assert "nominal" in p.console.toPlainText()
    finally:
        p.deleteLater()


def test_pause_buffers_then_resume_renders_once_in_order() -> None:
    p = _panel()
    try:
        p.append("before pause", severity="info")
        p._flush()
        base_text = p.console.toPlainText()
        assert "before pause" in base_text

        # Pause: buffered lines stay queued; neither the model nor the view
        # advances until resume.
        p.btn_pause.setChecked(True)
        p.append("line 1", severity="info")
        p.append("line 2", severity="info")
        p.append("line 3", severity="info")
        p._flush()  # no-op while paused
        assert len(p._entries) == 1
        assert p.console.toPlainText() == base_text

        # Resume: commit + render exactly once, in order, no duplicates.
        p.btn_pause.setChecked(False)
        assert len(p._entries) == 4
        text = p.console.toPlainText()
        for token in ("before pause", "line 1", "line 2", "line 3"):
            assert text.count(token) == 1, (token, text)
        # Ordering preserved.
        assert text.index("line 1") < text.index("line 2") < text.index("line 3")
    finally:
        p.deleteLater()


def test_clear_while_pending_then_new_logs_do_not_resurrect() -> None:
    p = _panel()
    try:
        # Queue without flushing, then clear: queued entries must be discarded.
        p.append("ghost 1", severity="info")
        p.append("ghost 2", severity="info")
        p.clear()
        p._flush()
        assert p.console.toPlainText().strip() == ""
        assert len(p._entries) == 0

        # New logs after clear render normally and the old ones never reappear.
        p.append("fresh", severity="info")
        p._flush()
        text = p.console.toPlainText()
        assert "fresh" in text
        assert "ghost" not in text
    finally:
        p.deleteLater()


def test_clear_resets_counters_and_latest() -> None:
    p = _panel()
    try:
        p.append("a warning", severity="warning")
        p.append("an error", severity="error")
        p._flush()
        assert p.lbl_warning_count.text() == "W 1"
        assert p.lbl_error_count.text() == "E 1"
        p.clear()
        assert p.lbl_warning_count.text() == "W 0"
        assert p.lbl_error_count.text() == "E 0"
        assert p.lbl_latest.text() == "No output yet"
    finally:
        p.deleteLater()


def test_line_limit_enforced_at_model_level() -> None:
    p = _panel()
    try:
        overflow = 25
        for i in range(MAX_LOG_LINES + overflow):
            p.append(f"line {i}", severity="info")
        p._flush()
        assert len(p._entries) == MAX_LOG_LINES
        # The oldest were trimmed; the newest are retained.
        assert p._entries[-1].message == f"line {MAX_LOG_LINES + overflow - 1}"
    finally:
        p.deleteLater()


def test_autoscroll_decision_respects_toggle_and_position() -> None:
    p = _panel()
    try:
        p.chk_autoscroll.setChecked(True)
        assert p._should_autoscroll(was_at_bottom=True) is True
        assert p._should_autoscroll(was_at_bottom=False) is False
        p.chk_autoscroll.setChecked(False)
        assert p._should_autoscroll(was_at_bottom=True) is False
    finally:
        p.deleteLater()


def test_near_bottom_true_when_nothing_to_scroll() -> None:
    # An empty / short console has no scroll range and therefore is, by
    # definition, already showing the bottom. (The value/maximum arithmetic is
    # covered exhaustively by the pure is_near_bottom tests.)
    p = _panel()
    try:
        assert p._is_near_bottom() is True
    finally:
        p.deleteLater()


def test_wrap_toggle_changes_mode_without_error() -> None:
    p = _panel()
    try:
        p.append("a very long line " * 20, severity="info")
        p._flush()
        p.chk_wrap.setChecked(True)
        assert p.console.lineWrapMode() == QtWidgets.QPlainTextEdit.WidgetWidth
        p.chk_wrap.setChecked(False)
        assert p.console.lineWrapMode() == QtWidgets.QPlainTextEdit.NoWrap
        # Content survives the toggle.
        assert "a very long line" in p.console.toPlainText()
    finally:
        p.deleteLater()


def test_collapse_and_expand_toggles_body_visibility() -> None:
    p = _panel()
    try:
        p.set_collapsed(True)
        assert p.is_collapsed is True
        assert p.body.isHidden()
        p.set_collapsed(False)
        assert p.is_collapsed is False
        assert not p.body.isHidden()
    finally:
        p.deleteLater()


def test_copy_output_reflects_model() -> None:
    p = _panel()
    try:
        p.append("copy me", severity="info")
        p.append("and me", severity="info")
        p._flush()
        assert p.copy_to_clipboard() is True
        clip = QtWidgets.QApplication.clipboard().text()
        assert "copy me" in clip and "and me" in clip
    finally:
        p.deleteLater()


def test_save_output_reflects_model(tmp_path: Path) -> None:
    p = _panel()
    try:
        p.append("persisted line", severity="success")
        p._flush()
        out = tmp_path / "log.txt"
        assert p.save_to_file(str(out)) is True
        content = out.read_text(encoding="utf-8")
        assert "persisted line" in content
    finally:
        p.deleteLater()


def test_run_separator_is_labeled_and_preserves_prior_context() -> None:
    p = _panel()
    try:
        p.append("preflight ok", severity="system")
        p.append_run_separator()
        p.append("launching", severity="system")
        p._flush()
        text = p.console.toPlainText()
        # Prior context retained (no implicit clear) and the divider is labeled.
        assert "preflight ok" in text
        assert "New Mission Run" in text
        assert "launching" in text
    finally:
        p.deleteLater()
