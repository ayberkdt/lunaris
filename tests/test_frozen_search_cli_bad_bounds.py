"""``lunaris-frozen-search`` refuses impossible bounds with one line and exit 2.

Regression test: ``ElementBounds`` / ``FrozenSearchConfig`` were built outside
any error handling, so crossed bounds or ``--n-samples 1`` printed a
ValueError traceback, and ``--hp-km`` with an eccentricity bound of 1 raised
ZeroDivisionError.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lunaris.cli.frozen_search import main


@pytest.mark.parametrize(
    "extra",
    [
        ["--e", "0.5", "0.1"],  # crossed bounds
        ["--n-samples", "1"],
        ["--hp-km", "10", "20", "--e", "0", "1"],  # e = 1 with perilune altitudes
    ],
)
def test_impossible_bounds_exit_2_with_a_fatal_line(tmp_path: Path, capsys, extra: list[str]) -> None:
    rc = main(["--out", str(tmp_path / "out"), *extra])
    err = capsys.readouterr().err
    assert rc == 2
    assert "[FATAL]" in err
    assert "Traceback" not in err
