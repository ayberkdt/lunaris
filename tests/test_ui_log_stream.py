"""
Unit tests for the Qt-free Execution Console stream helpers.

These cover the line-assembly contract (the heart of objective 2) and the
auto-scroll decision (objective 1) without constructing any widgets, so the
buffering and follow-output rules are pinned independently of rendering.
"""

from __future__ import annotations

from lunaris.ui.core.log_stream import LineAssembler, is_near_bottom


# --------------------------------------------------------------------------- #
# LineAssembler
# --------------------------------------------------------------------------- #
def test_push_emits_one_complete_line() -> None:
    asm = LineAssembler()
    assert asm.push("hello world\n") == ["hello world"]
    assert asm.pending == ""


def test_push_assembles_line_split_across_chunks() -> None:
    asm = LineAssembler()
    assert asm.push("hel") == []
    assert asm.push("lo ") == []
    assert asm.push("world\n") == ["hello world"]


def test_push_returns_multiple_lines_from_one_chunk() -> None:
    asm = LineAssembler()
    assert asm.push("a\nb\nc\n") == ["a", "b", "c"]
    assert asm.pending == ""


def test_push_retains_only_trailing_partial_line() -> None:
    asm = LineAssembler()
    assert asm.push("done\npartial") == ["done"]
    assert asm.pending == "partial"
    assert asm.push(" more\n") == ["partial more"]


def test_flush_emits_final_unterminated_line_and_clears() -> None:
    asm = LineAssembler()
    asm.push("tail without newline")
    assert asm.flush() == ["tail without newline"]
    assert asm.pending == ""
    # A second flush with nothing pending yields nothing.
    assert asm.flush() == []


def test_crlf_terminators_are_normalized() -> None:
    asm = LineAssembler()
    assert asm.push("win\r\nline\r\n") == ["win", "line"]


def test_stdout_and_stderr_buffers_are_independent() -> None:
    out = LineAssembler()
    err = LineAssembler()
    # Interleaved partials on each stream must not bleed into each other.
    assert out.push("OUT par") == []
    assert err.push("ERR par") == []
    assert out.push("t1\n") == ["OUT part1"]
    assert err.push("t2\n") == ["ERR part2"]
    assert out.pending == "" and err.pending == ""


def test_clear_discards_partial_without_emitting() -> None:
    asm = LineAssembler()
    asm.push("half a line")
    asm.clear()
    assert asm.pending == ""
    assert asm.flush() == []


def test_blank_lines_are_preserved_as_empty_strings() -> None:
    asm = LineAssembler()
    assert asm.push("\n\n") == ["", ""]


# --------------------------------------------------------------------------- #
# is_near_bottom
# --------------------------------------------------------------------------- #
def test_is_near_bottom_when_nothing_to_scroll() -> None:
    assert is_near_bottom(0, 0) is True


def test_is_near_bottom_at_exact_bottom() -> None:
    assert is_near_bottom(100, 100) is True


def test_is_near_bottom_within_slack() -> None:
    assert is_near_bottom(95, 100, slack=8) is True


def test_is_not_near_bottom_when_scrolled_up() -> None:
    assert is_near_bottom(40, 100, slack=8) is False
    assert is_near_bottom(90, 100, slack=8) is False
