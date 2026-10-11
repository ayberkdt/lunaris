"""Mission Studio child processes write UTF-8 to their pipes.

Regression test: the run and batch children got only ``PYTHONUNBUFFERED``; on
Windows a Python child whose stdout is a pipe encodes with the ANSI code page
(cp1252) and dies with ``UnicodeEncodeError`` on the first character outside
it, while the UI decodes the stream as UTF-8.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

from lunaris.ui.app import MainWindow, _child_process_environment  # noqa: E402


def test_child_environment_sets_utf8_and_unbuffered() -> None:
    env = _child_process_environment()
    assert env.value("PYTHONIOENCODING") == "utf-8"
    assert env.value("PYTHONUNBUFFERED") == "1"


def test_run_and_batch_launchers_use_it() -> None:
    assert "_child_process_environment()" in inspect.getsource(MainWindow._run_process)
    assert "_child_process_environment()" in inspect.getsource(MainWindow._on_batch_run_requested)


def test_child_with_that_environment_prints_non_ansi_text() -> None:
    env = _child_process_environment()
    child_env = {key: env.value(key) for key in env.keys()}
    proc = subprocess.run(
        [sys.executable, "-c", "print('\\u2192 \\u03bc \\u00b0')"],
        capture_output=True, env=child_env, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.decode("utf-8").strip() == "→ μ °"
