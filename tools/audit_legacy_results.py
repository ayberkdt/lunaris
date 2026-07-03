#!/usr/bin/env python
"""CLI shim: audit legacy batch/ST-LRPS results into a trust manifest (reviewer §10).

The implementation lives in the importable package
``lunaris.analysis.ensemble.result_audit`` so it is testable without putting
``tools/`` on ``sys.path``. This file is only the console entry point.

Usage
-----
    python -m tools.audit_legacy_results [--root outputs] [--output PATH]
    python -m lunaris.analysis.ensemble.result_audit  # (equivalent module)
"""

from __future__ import annotations

from lunaris.analysis.ensemble.result_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
