"""
Subprocess-output assembly and telemetry routing (objective 2).

Constructs the real ``MainWindow`` (isolated app-data dir) to verify that the
stdout line-assembly path routes telemetry to the telemetry plot, logs ordinary
lines, and survives a malformed telemetry payload without crashing or breaking
the log stream — plus that the final partial line is flushed at process end.
"""

from __future__ import annotations

import pytest

pytest.importorskip('PySide6.QtWidgets')


import os

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def win(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
    app = _app()
    from lunaris.ui.app import MainWindow

    w = MainWindow()
    yield w, app
    w.close()
    w.deleteLater()
    app.processEvents()


def _log_count(w) -> int:
    return len(w.log_panel._pending) + len(w.log_panel._entries)


def test_valid_telemetry_line_is_not_logged(win) -> None:
    w, _ = win
    before = _log_count(w)
    w._consume_stdout_line('{"t_s": 100.0, "alt_km": 50.0}')
    # Telemetry payloads are routed to the plot, not echoed to the console.
    assert _log_count(w) == before


def test_prefixed_time_s_telemetry_updates_progress(win) -> None:
    w, _ = win
    w.sim_state.total_duration = 200.0
    before = _log_count(w)

    w._consume_stdout_line('JSON_TELEM: {"time_s": 100.0, "alt_km": 50.0}')

    assert _log_count(w) == before
    assert w._last_telem_t_s == 100.0
    assert w.progress_bar.value() == 500


def test_ordinary_line_is_logged(win) -> None:
    w, _ = win
    before = _log_count(w)
    w._consume_stdout_line("[System] backend started")
    assert _log_count(w) == before + 1


def test_malformed_telemetry_line_does_not_crash_and_is_surfaced(win) -> None:
    w, _ = win
    before = _log_count(w)
    # Parses as a dict but has a non-numeric time -> must not raise.
    w._consume_stdout_line('{"t": "not-a-number", "alt_km": 5}')
    assert _log_count(w) == before + 1   # surfaced once as a warning


def test_jsonish_but_not_dict_falls_through_to_log(win) -> None:
    w, _ = win
    before = _log_count(w)
    w._consume_stdout_line('{"t" broken payload')
    assert _log_count(w) == before + 1   # logged as an ordinary line, no crash


def test_line_split_across_chunks_is_assembled(win) -> None:
    w, _ = win
    before = _log_count(w)
    # Feed a single logical line in fragments through the stdout assembler.
    assert w._stdout_assembler.push("[System] hel") == []
    for line in w._stdout_assembler.push("lo world\n"):
        w._consume_stdout_line(line)
    assert _log_count(w) == before + 1
    w.log_panel._flush()
    assert "hello world" in w.log_panel.console.toPlainText()


def test_flush_stream_buffers_emits_trailing_partial_lines(win) -> None:
    w, _ = win
    before = _log_count(w)
    # Unterminated fragments on both streams (no trailing newline).
    w._stdout_assembler.push("final stdout tail")
    w._stderr_assembler.push("final stderr tail")
    w._flush_stream_buffers()
    assert _log_count(w) >= before + 2
    assert w._stdout_assembler.pending == ""
    assert w._stderr_assembler.pending == ""
