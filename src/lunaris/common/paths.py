"""Dependency-light path helpers for editable and installed Lunaris layouts."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

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


def user_data_dir(app: str = "lunaris") -> Path:
    """Per-user external-data directory for installed (non-checkout) layouts."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip() or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME", "").strip() or str(Path.home() / ".local" / "share")
    return Path(base) / app / "data"


def data_dir_from_root(root: Path, env_names: Iterable[str] = ("LUNARIS_DATA_DIR", "STLRPS_DATA_DIR")) -> Path:
    """Return external data directory, honoring environment overrides first.

    Resolution order: environment override > repository ``data/`` (identified
    by the tracked ``data_sources.json`` manifest) > per-user data directory.
    The last step matters for wheel installs: their root walk-up lands inside
    ``site-packages``, and mission data must never be downloaded into the
    package installation.
    """
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value).expanduser().resolve()
    repo_data = root / "data"
    if (repo_data / "data_sources.json").exists():
        return repo_data
    return user_data_dir()
