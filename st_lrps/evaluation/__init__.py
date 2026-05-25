# st_lrps/evaluation/__init__.py
"""ST-LRPS evaluation subpackage: evaluation CLI and ablation matrix.

Import-safe: importing this package must not pull torch or run evaluation.
Canonical entry points:

    python -m st_lrps.evaluation.cli
    python -m st_lrps.evaluation.ablation
"""

from __future__ import annotations

__all__: list[str] = []
