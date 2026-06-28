"""Dependency-light provenance helpers shared across Lunaris subsystems.

This module is intentionally standard-library only so lower layers and optional
ST-LRPS/reporting tools can share byte-identical file hashes and UTC timestamps
without importing each other.
"""

from __future__ import annotations

import hashlib
import os
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


__all__ = ["sha256_file", "sha256_text", "utc_now_iso"]
