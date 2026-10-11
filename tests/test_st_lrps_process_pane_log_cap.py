"""ST-LRPS Studio: the process log is capped and still finds the run folder.

Regression tests: ``ProcessPane.log`` was a ``QPlainTextEdit`` with no block
limit, fed every stdout line of training, evaluation, benchmark and data runs,
so a long run grew the document (and the UI's memory and repaint cost) without
bound. A plain ``setMaximumBlockCount`` would drop the first lines, which is
where the runs print their output folder; the training, evaluation and suite
pages search the log for it after the run. ``output_text()`` keeps those
first lines.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from lunaris.surrogate.st_lrps.ui.studio_parts import common_widgets  # noqa: E402


@pytest.fixture()
def pane():
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = common_widgets.ProcessPane()
    yield p
    p.deleteLater()


def test_log_document_is_capped(pane) -> None:
    cap = common_widgets._LOG_MAX_BLOCKS
    for i in range(cap + 500):
        pane.append(f"epoch line {i}")
    assert pane.log.document().blockCount() <= cap
    assert pane.log.document().lastBlock().text() == f"epoch line {cap + 499}"


def test_output_text_keeps_the_head_after_trimming(pane) -> None:
    cap = common_widgets._LOG_MAX_BLOCKS
    pane.append("Output dir: C:/runs/run_42")
    for i in range(cap + 10):
        pane.append(f"epoch line {i}")
    text = pane.output_text()
    assert "Output dir: C:/runs/run_42" in text
    assert f"epoch line {cap + 9}" in text
    assert "Output dir" not in pane.log.toPlainText()  # really trimmed from view


def test_output_text_is_the_log_when_nothing_was_trimmed(pane) -> None:
    pane.append("Output dir: C:/runs/run_7")
    pane.append("done")
    assert pane.output_text() == pane.log.toPlainText()


def test_clearing_the_log_forgets_the_head(pane) -> None:
    cap = common_widgets._LOG_MAX_BLOCKS
    pane.append("Output dir: C:/runs/old")
    for i in range(cap + 1):
        pane.append(f"x{i}")
    pane.log.clear()  # the pages' Clear buttons call this directly
    pane.append("Output dir: C:/runs/new")
    assert "old" not in pane.output_text()
    assert "C:/runs/new" in pane.output_text()
