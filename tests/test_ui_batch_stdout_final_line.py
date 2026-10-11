"""Mission Studio batch runs: the last stdout line counts even without a newline.

Regression tests: ``_on_batch_stdout`` kept the text after the last ``\\n`` in
``_batch_stdout_buf`` and ``_on_batch_finished`` never read it, so a final
``[BATCH_METRICS]`` line written without a trailing newline was dropped and
the run reported ``?/?`` impacts. Batch stderr was split per read chunk, so a
line spread over two reads was logged as two lines.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from lunaris.ui.app import MainWindow  # noqa: E402


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout: list[bytes] = []
        self.stderr: list[bytes] = []

    def readAllStandardOutput(self) -> bytes:
        return self.stdout.pop(0) if self.stdout else b""

    def readAllStandardError(self) -> bytes:
        return self.stderr.pop(0) if self.stderr else b""

    def deleteLater(self) -> None:
        pass


def _window() -> SimpleNamespace:
    win = SimpleNamespace(messages=[], finished=[], progress=[], chips=[])
    win.batch_process = _FakeProcess()
    win._batch_output_path = "out.h5"
    win._log_message = lambda msg, severity="info", source=None: win.messages.append((severity, msg))
    win.page_batch = SimpleNamespace(
        update_progress=lambda line: win.progress.append(line),
        update_progress_payload=lambda payload: None,
        on_run_finished=lambda **kw: win.finished.append(kw),
    )
    win.monitor_controller = SimpleNamespace(
        set_batch_progress=lambda payload: None, set_batch_metrics=lambda payload: None
    )
    win._update_backend_chip = lambda metrics: win.chips.append(metrics)
    MainWindow._reset_batch_streams(win)
    for name in ("_on_batch_stdout", "_on_batch_stderr", "_on_batch_finished",
                 "_consume_batch_stdout_line", "_log_batch_stderr_line"):
        setattr(win, name, getattr(MainWindow, name).__get__(win))
    return win


def test_unterminated_final_metrics_line_is_read_on_finish() -> None:
    win = _window()
    metrics = {"n_impacts": 3, "n_samples": 64, "wall_time_s": 1.5}
    win.batch_process.stdout.append(b"[BATCH] starting\n")
    win._on_batch_stdout()
    win.batch_process.stdout.append(b"[BATCH_METRICS] " + json.dumps(metrics).encode())
    win._on_batch_stdout()

    win._on_batch_finished(0, None)

    assert win.finished[0]["metrics"] == metrics
    assert any("3/64 impacts" in msg for _sev, msg in win.messages)


def test_unterminated_final_log_line_is_logged_on_finish() -> None:
    win = _window()
    win.batch_process.stdout.append(b"Saved archive to out.h5")
    win._on_batch_stdout()
    win._on_batch_finished(0, None)
    assert "Saved archive to out.h5" in win.progress


def test_stderr_line_split_across_reads_is_one_line() -> None:
    win = _window()
    win.batch_process.stderr.extend([b"RuntimeWarning: overflow enc", b"ountered in exp\n"])
    win._on_batch_stderr()
    win._on_batch_stderr()
    stderr_lines = [msg for _sev, msg in win.messages if "overflow" in msg]
    assert stderr_lines == ["[BATCH] RuntimeWarning: overflow encountered in exp"]


def test_multibyte_character_split_across_reads_survives() -> None:
    win = _window()
    text = "Moon radius 1737.4 km ± 0.1\n".encode()
    cut = text.index("±".encode()) + 1  # inside the two-byte sequence
    win.batch_process.stdout.extend([text[:cut], text[cut:]])
    win._on_batch_stdout()
    win._on_batch_stdout()
    assert win.progress == ["Moon radius 1737.4 km ± 0.1"]
