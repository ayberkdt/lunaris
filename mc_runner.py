"""Backward-compatible launcher for Lunaris Monte Carlo runs."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import lunaris.core.mc_runner as _mc_runner
from lunaris.core.monte_carlo_engine import mc_entry

main = _mc_runner.main

__all__ = ["main", "mc_entry"]


if __name__ == "__main__":
    raise SystemExit(mc_entry())
