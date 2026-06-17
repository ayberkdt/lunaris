"""Hashing, path resolution, and atomic writes for reference validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lunaris.common.hashing import canonical_json_text


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def find_repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Find the repository root by walking up to ``pyproject.toml``."""
    here = Path(start or Path.cwd()).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 hex digest of a file's exact bytes.
    For text files (.csv, .json), CRLF is normalized to LF before hashing
    to ensure cross-platform stability (e.g., Windows vs Linux in CI).
    """
    p = Path(path)
    digest = hashlib.sha256()
    
    if p.suffix in {".csv", ".json"}:
        with p.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                digest.update(line.replace("\r\n", "\n").encode("utf-8"))
    else:
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a compact immutable-file record."""
    p = Path(path)
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "size_bytes": int(p.stat().st_size),
    }


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load JSON using UTF-8 and reject duplicate keys."""
    def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicates)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Atomically write UTF-8 text on the same filesystem."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp, dst)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Atomically write canonical JSON."""
    atomic_write_text(path, canonical_json_text(payload))


def resolve_manifest_path(
    path_value: str,
    *,
    manifest_path: str | os.PathLike[str],
    must_exist: bool = False,
) -> Path:
    """Resolve a manifest path without allowing parent-directory traversal."""
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Manifest path values must be non-empty strings.")
    raw = Path(path_value)
    if any(part == ".." for part in raw.parts):
        raise ValueError(f"Parent-directory traversal is not allowed in manifest paths: {path_value}")
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (find_repo_root(manifest_path) / raw).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved

