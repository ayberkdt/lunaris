"""Backward-compatible launcher for the Lunaris CLI."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lunaris.cli.main import apply_args_to_config, init_ephemeris, main, main_entry, parse_args

__all__ = ["apply_args_to_config", "init_ephemeris", "main", "main_entry", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main_entry())
