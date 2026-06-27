"""Lazy bridge to the canonical ST-LRPS force runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_force_runtime(run_dir: Path, *, device: str, chunk_size: int) -> Any:
    """Load the canonical force runtime without making it an import-time dependency."""

    from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model

    return load_surrogate_force_model(run_dir, device=device, chunk_size=chunk_size)


__all__ = ["load_force_runtime"]
