"""
Qt-free helpers for the Execution Console data flow.

The desktop UI reads ``stdout`` and ``stderr`` from a backend ``QProcess`` in
arbitrary chunks. A single decoded chunk may contain several complete lines, a
fraction of a line, or split one line across two reads. Rendering raw chunks
produces interleaved, half-written log entries.

This module isolates the two pieces of console logic that must stay correct
under high-volume streaming and that benefit from being unit-testable without a
running application:

* :class:`LineAssembler` — one partial-line buffer per stream, so only complete
  lines are forwarded to the log model and the final unterminated line can be
  flushed explicitly when the process exits.
* :func:`is_near_bottom` — the auto-scroll decision, expressed as a pure function
  of scrollbar geometry so the "follow output only when the user is already at
  the bottom" rule can be verified independently of OpenGL/widget rendering.
"""

from __future__ import annotations

# How close (in scrollbar units, ~pixels) to the bottom still counts as "at the
# bottom" for auto-scroll purposes. A small slack absorbs sub-line rounding so a
# user resting at the end keeps following output.
DEFAULT_SCROLL_SLACK = 8


class LineAssembler:
    """Accumulate decoded stream chunks and emit only complete lines.

    One instance owns the partial-line buffer for exactly one stream. Mixing
    ``stdout`` and ``stderr`` through a single assembler would interleave their
    partial lines, so callers must use a dedicated assembler per stream.
    """

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer = ""

    @property
    def pending(self) -> str:
        """The partial line retained so far (without a trailing newline)."""
        return self._buffer

    def push(self, chunk: str) -> list[str]:
        """Append a decoded *chunk* and return any newly completed lines.

        A line is considered complete once its ``\\n`` terminator has been seen.
        A trailing ``\\r`` (Windows ``\\r\\n``) is stripped. The unterminated
        remainder is retained for the next :meth:`push` / :meth:`flush`.
        """
        if not chunk:
            return []
        self._buffer += chunk
        if "\n" not in self._buffer:
            return []
        parts = self._buffer.split("\n")
        # The final element is the still-incomplete remainder (possibly empty).
        self._buffer = parts.pop()
        return [line.rstrip("\r") for line in parts]

    def flush(self) -> list[str]:
        """Return the final unterminated line (if any) and clear the buffer."""
        tail = self._buffer
        self._buffer = ""
        if not tail:
            return []
        return [tail.rstrip("\r")]

    def clear(self) -> None:
        """Discard any retained partial line without emitting it."""
        self._buffer = ""


class StderrClassifier:
    """Give each complete stderr line the severity of the message it belongs to.

    Classifying lines one by one ("warning" in it -> warning, else error) got
    multi-line messages wrong: the indented source line under a
    ``warnings.warn`` record became an error, and traceback frames in
    ``warnings.py`` (or a final ``RuntimeWarning:`` raised under ``-W error``)
    became warnings in the middle of a crash. Rules, in order:

    * ``Traceback (most recent call last):`` starts an error block that runs
      through the indented frames up to and including the exception line;
    * an indented line continues the previous line's severity;
    * otherwise "warning" in the line (and no traceback / fatal marker) is a
      warning and anything else an error.

    One instance per stream; :meth:`reset` at the start of each run.
    """

    __slots__ = ("_in_traceback", "_last")

    def __init__(self) -> None:
        self._in_traceback = False
        self._last: str | None = None

    def reset(self) -> None:
        self._in_traceback = False
        self._last = None

    def classify(self, line: str) -> str:
        indented = line[:1] in (" ", "\t")
        if line.startswith("Traceback (most recent call last)"):
            self._in_traceback = True
            self._last = "error"
            return "error"
        if self._in_traceback:
            if not indented:  # the exception line closes the traceback
                self._in_traceback = False
            self._last = "error"
            return "error"
        if indented and self._last is not None:
            return self._last
        lowered = line.lower()
        is_warning = (
            "warning" in lowered
            and "traceback" not in lowered
            and "[fatal]" not in lowered
            and "fatal error" not in lowered
        )
        self._last = "warning" if is_warning else "error"
        return self._last


def is_near_bottom(value: int, maximum: int, slack: int = DEFAULT_SCROLL_SLACK) -> bool:
    """Return ``True`` when a scrollbar at *value* is within *slack* of *maximum*.

    Used to decide whether new console output should auto-scroll. When there is
    nothing to scroll (``maximum <= 0``) the viewport is, by definition, already
    showing the bottom.
    """
    if maximum <= 0:
        return True
    return value >= maximum - max(0, int(slack))
