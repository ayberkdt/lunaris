# -*- coding: utf-8 -*-
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

from typing import List

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

    def push(self, chunk: str) -> List[str]:
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

    def flush(self) -> List[str]:
        """Return the final unterminated line (if any) and clear the buffer."""
        tail = self._buffer
        self._buffer = ""
        if not tail:
            return []
        return [tail.rstrip("\r")]

    def clear(self) -> None:
        """Discard any retained partial line without emitting it."""
        self._buffer = ""


def is_near_bottom(value: int, maximum: int, slack: int = DEFAULT_SCROLL_SLACK) -> bool:
    """Return ``True`` when a scrollbar at *value* is within *slack* of *maximum*.

    Used to decide whether new console output should auto-scroll. When there is
    nothing to scroll (``maximum <= 0``) the viewport is, by definition, already
    showing the bottom.
    """
    if maximum <= 0:
        return True
    return value >= maximum - max(0, int(slack))
