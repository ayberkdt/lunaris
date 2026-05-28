"""Dependency-light path helpers for editable and installed Lunaris layouts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

_ROOT_MARKERS = ("pyproject.toml", "README.md", ".git")


def find_project_root(start: Path | str | None = None) -> Path:
    """Return the nearest repository/project root without importing GUI code."""
    anchor = Path(start).resolve() if start is not None else Path.cwd().resolve()
    if anchor.is_file():
        anchor = anchor.parent
    for current in (anchor, *anchor.parents):
        if any((current / marker).exists() for marker in _ROOT_MARKERS):
            return current
    return anchor


def project_root_from_file(file: str | os.PathLike[str]) -> Path:
    """Resolve the project root for a module ``__file__`` path."""
    return find_project_root(Path(file).resolve())


def data_dir_from_root(root: Path, env_names: Iterable[str] = ("LUNARIS_DATA_DIR", "STLRPS_DATA_DIR")) -> Path:
    """Return external data directory, honoring environment overrides first."""
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value).expanduser().resolve()
    return root / "data"
