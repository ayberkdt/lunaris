"""Lazy bridge to the canonical ST-LRPS force runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_force_runtime(
    run_dir: Path,
    *,
    device: str,
    chunk_size: int,
    strict_domain: bool = False,
    paper_safe: bool = False,
) -> Any:
    """Load the canonical force runtime without making it an import-time dependency."""

    from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model

    return load_surrogate_force_model(
        run_dir,
        device=device,
        chunk_size=chunk_size,
        strict_domain=strict_domain,
        paper_safe=paper_safe,
    )


__all__ = ["load_force_runtime"]
