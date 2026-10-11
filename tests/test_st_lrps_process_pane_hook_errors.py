"""ST-LRPS Studio: a failing post-run hook is reported, not swallowed.

Regression test: ``ProcessPane._on_finished`` ran the page's finished hook
under ``contextlib.suppress(Exception)``. The hook chains the training queue,
train-then-evaluate and the results refresh, so an exception there silently
stopped those follow-ups while the pane still read "Done".
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402
from PySide6.QtCore import QProcess  # noqa: E402

from lunaris.surrogate.st_lrps.ui.studio_parts import common_widgets  # noqa: E402


@pytest.fixture()
def pane(monkeypatch):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(common_widgets, "_send_os_notification", lambda *a, **k: None)
    p = common_widgets.ProcessPane()
    yield p
    p.deleteLater()


def test_failing_hook_is_shown_in_the_log_and_status(pane, caplog) -> None:
    def hook(_code, _status):
        raise KeyError("queue_item")

    pane.set_finished_hook(hook)
    with caplog.at_level(logging.ERROR):
        pane._on_finished(0, QProcess.ExitStatus.NormalExit)

    log_text = pane.log.toPlainText()
    assert "Post-run step failed" in log_text
    assert "KeyError" in log_text and "queue_item" in log_text
    assert "post-run step failed" in pane.status.text()
    assert any("queue_item" in r.getMessage() or r.exc_info for r in caplog.records)


def test_successful_hook_keeps_plain_status(pane) -> None:
    calls: list[int] = []
    pane.set_finished_hook(lambda code, _status: calls.append(code))
    pane._on_finished(0, QProcess.ExitStatus.NormalExit)
    assert calls == [0]
    assert pane.status.text() == "Done | exit_code=0"
    assert "Post-run step failed" not in pane.log.toPlainText()
