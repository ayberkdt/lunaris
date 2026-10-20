"""Dependency-light provenance helpers shared across Lunaris subsystems.

This module is intentionally standard-library only so lower layers and optional
ST-LRPS/reporting tools can share byte-identical file hashes, UTC timestamps and
the repository commit (:func:`resolve_repo_commit`) without importing each other.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso(*, z_suffix: bool = True, timespec: str = "seconds") -> str:
    """Return an ISO-8601 UTC timestamp with second precision by default."""

    text = datetime.now(timezone.utc).isoformat(timespec=timespec)
    return text.replace("+00:00", "Z") if z_suffix else text


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(
    path: str | os.PathLike[str] | None,
    *,
    missing_ok: bool = False,
    suppress_errors: bool = False,
    normalize_text_suffixes: Iterable[str] | None = None,
    chunk_size: int = 1024 * 1024,
) -> str | None:
    """Return the SHA-256 hex digest of a file's exact bytes.

    Parameters preserve the historical call-site behaviors:
    - ``missing_ok=True`` returns ``None`` when the path is unset or not a file.
    - ``suppress_errors=True`` converts any read/hash error to ``None``.
    - ``normalize_text_suffixes`` normalizes CRLF to LF for selected suffixes.
    """

    if path is None:
        if missing_ok or suppress_errors:
            return None
        raise TypeError("path must not be None")

    try:
        p = Path(path)
        if not p.is_file():
            if missing_ok:
                return None
            raise FileNotFoundError(str(p))

        digest = hashlib.sha256()
        suffixes = {s.lower() for s in (normalize_text_suffixes or ())}
        if p.suffix.lower() in suffixes:
            with p.open("r", encoding="utf-8", newline="") as handle:
                for line in handle:
                    digest.update(line.replace("\r\n", "\n").encode("utf-8"))
        else:
            with p.open("rb") as handle:
                for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        if suppress_errors:
            return None
        raise


# ---------------------------------------------------------------------------
# Repository identity
# ---------------------------------------------------------------------------
# Single source of truth for "what commit produced this artifact". Git runs with
# cwd pinned to the installed lunaris package's own directory -- never the
# caller's process cwd, which may sit inside an unrelated checkout.

_GIT_TIMEOUT_S = 5


def package_source_dir() -> Path:
    """Directory of the installed ``lunaris`` package (editable or wheel install).

    Resolved from this file's location (``lunaris/common/provenance.py``), so no
    import of the package root -- and of everything it pulls in -- is needed.
    """
    return Path(__file__).resolve().parents[1]


def resolve_repo_commit(*, env_commit_var: str = "GIT_COMMIT") -> str | None:
    """Best-effort git commit SHA of the checkout holding the ``lunaris`` package.

    ``env_commit_var`` (default ``GIT_COMMIT``) wins when set, so CI can pin
    provenance without a working tree. Otherwise ``git rev-parse HEAD`` runs
    with ``cwd`` fixed to :func:`package_source_dir`. Returns ``None`` when git
    is unavailable or the package is not inside a git checkout (e.g. a wheel
    install) -- an unknown commit, never the caller's repository's commit.
    """
    env_commit = (os.environ.get(env_commit_var) or "").strip()
    if env_commit:
        return env_commit
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(package_source_dir()),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except Exception:  # pragma: no cover - git missing / sandboxed
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


__all__ = [
    "package_source_dir",
    "resolve_repo_commit",
    "sha256_file",
    "sha256_text",
    "utc_now_iso",
]
