"""Mission Studio: stderr lines get the severity of the message they belong to.

Regression tests: each stderr line was classified on its own ("warning"
anywhere in it -> warning, otherwise error), so

* the source line Python prints under every ``warnings.warn`` record
  (indented, no "warning" in it) was shown as an error;
* traceback frames inside ``warnings.py`` were shown as warnings in the middle
  of an error;
* the final line of a traceback raised from a Warning class (``-W error``) was
  shown as a warning although the run crashed;
* batch stderr was shown as warnings line by line, tracebacks included.
"""

from __future__ import annotations

from lunaris.ui.core.log_stream import StderrClassifier


def _classify(lines: list[str]) -> list[str]:
    classifier = StderrClassifier()
    return [classifier.classify(line) for line in lines]


def test_warning_source_line_follows_its_warning() -> None:
    lines = [
        "C:/x/dynamics.py:88: RuntimeWarning: overflow encountered in exp",
        "  r = np.exp(big)",
    ]
    assert _classify(lines) == ["warning", "warning"]


def test_traceback_is_error_throughout() -> None:
    lines = [
        "Traceback (most recent call last):",
        '  File "C:/py/Lib/warnings.py", line 109, in _showwarnmsg',
        "    sw(msg.message, msg.category)",
        "RuntimeWarning: overflow encountered in exp",
    ]
    assert _classify(lines) == ["error"] * 4


def test_after_a_traceback_the_next_warning_is_a_warning() -> None:
    lines = [
        "Traceback (most recent call last):",
        '  File "run.py", line 3, in <module>',
        "ValueError: bad orbit",
        "C:/x/io.py:12: UserWarning: kernel list is empty",
        "  warnings.warn(msg)",
    ]
    assert _classify(lines) == ["error", "error", "error", "warning", "warning"]


def test_plain_lines_keep_the_previous_rule() -> None:
    assert _classify(["[FATAL] gravity file missing"]) == ["error"]
    assert _classify(["DeprecationWarning: use x"]) == ["warning"]
    assert _classify(["fatal error: warning flag"]) == ["error"]


def test_reset_forgets_the_previous_severity() -> None:
    classifier = StderrClassifier()
    classifier.classify("x.py:1: UserWarning: w")
    classifier.reset()
    assert classifier.classify("  indented first line") == "error"


def test_batch_stderr_traceback_is_shown_as_error() -> None:
    import pytest

    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from lunaris.ui.app import MainWindow

    messages: list[tuple[str, str]] = []
    win = SimpleNamespace(_log_message=lambda msg, severity="info", **_k: messages.append((severity, msg)))
    MainWindow._reset_batch_streams(win)
    for line in ("Traceback (most recent call last):", '  File "b.py", line 1', "KeyError: 'n'",
                 "b.py:9: UserWarning: slow path"):
        MainWindow._log_batch_stderr_line(win, line)
    assert [sev for sev, _msg in messages] == ["error", "error", "error", "warning"]
