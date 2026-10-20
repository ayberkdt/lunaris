"""Mission Studio Stop returns at once; the force-kill follows on a timer.

Regression test: ``_stop_process`` called ``terminate()`` and then blocked
the GUI thread in ``waitForFinished(2000)`` and ``waitForFinished(1000)``. On
Windows ``terminate()`` posts WM_CLOSE, which a console Python child ignores,
so every Stop froze the window for the full 2-3 s. The child here ignores
SIGTERM so the same holds on every platform.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from lunaris.ui.app import MainWindow  # noqa: E402

_CHILD = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)


def _window(tmp_path) -> SimpleNamespace:
    win = SimpleNamespace(messages=[], states=[])
    win._has_running_process = lambda: True
    win._log_message = lambda msg, severity="info", **_k: win.messages.append((severity, msg))
    win.page_output = SimpleNamespace(get_state=lambda: SimpleNamespace(output_dir=str(tmp_path)))
    win._set_run_state = lambda state: win.states.append(state)
    win._dispose_process = lambda: setattr(win, "process", None)
    win._force_kill_if_still_running = (
        lambda proc: MainWindow._force_kill_if_still_running(win, proc)
    )
    win.btn_stop = QtWidgets.QPushButton("Stop")
    return win


def test_stop_does_not_block_and_the_child_is_killed(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    anchor = QtCore.QObject()  # parent of the child process
    win = _window(tmp_path)
    proc = QtCore.QProcess(anchor)
    finished: list[int] = []
    proc.finished.connect(lambda code, _status: finished.append(code))
    proc.start(sys.executable, ["-c", _CHILD])
    assert proc.waitForStarted(5000)
    assert proc.waitForReadyRead(10000)
    win.process = proc

    t0 = time.perf_counter()
    MainWindow._stop_process(win)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"Stop blocked the GUI thread for {elapsed:.2f} s"
    assert not win.btn_stop.isEnabled()

    # The force-kill timer fires after 2 s; the rest of the budget absorbs a
    # fully loaded machine (the full suite once missed an 8 s budget).
    deadline = time.perf_counter() + 30.0
    while not finished and time.perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    assert finished, "the child was never force-killed"
    assert any("forcing kill" in msg for _sev, msg in win.messages)


def test_force_kill_happens_even_if_logging_fails(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    anchor = QtCore.QObject()
    win = _window(tmp_path)

    def broken_log(*_a, **_k):
        raise AttributeError("log pane already torn down")

    win._log_message = broken_log
    proc = QtCore.QProcess(anchor)
    finished: list[int] = []
    proc.finished.connect(lambda code, _status: finished.append(code))
    proc.start(sys.executable, ["-c", _CHILD])
    assert proc.waitForStarted(5000)
    assert proc.waitForReadyRead(10000)
    win.process = proc

    MainWindow._force_kill_if_still_running(win, proc)  # must not raise
    deadline = time.perf_counter() + 30.0
    while not finished and time.perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    assert finished, "a failing log write left the child running"
