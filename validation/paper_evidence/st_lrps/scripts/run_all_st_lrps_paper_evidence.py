#!/usr/bin/env python3
"""Thin wrapper for the ST-LRPS paper evidence runner (Part 1).

The real logic lives in the installed package so it is importable and tested:
``lunaris.surrogate.st_lrps.paper_evidence.runner``. This script lets the
workspace be driven directly:

    python validation/paper_evidence/st_lrps/scripts/run_all_st_lrps_paper_evidence.py \
        --stage train --config configs/st_lrps/paper/train_full_seed42.json --dry-run

Equivalent console entry point: ``lunaris-st-lrps-paper-evidence``.

Only the ``train`` stage is implemented in Part 1; the other stages print that
they belong to Part 2/3.
"""

from __future__ import annotations

import sys

try:
    from lunaris.surrogate.st_lrps.paper_evidence.runner import main
except ModuleNotFoundError as exc:  # pragma: no cover - install hint only
    sys.stderr.write(
        "Could not import lunaris. Install the project first: `python -m pip install -e .`\n"
        f"Original error: {exc}\n"
    )
    raise SystemExit(2) from exc


if __name__ == "__main__":
    raise SystemExit(main())
