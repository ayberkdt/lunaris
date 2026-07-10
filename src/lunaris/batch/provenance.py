"""Provenance and metadata helpers for batch propagation archives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.common.provenance import sha256_file


def _sha256_file(path: Any) -> str | None:
    """Return the SHA-256 hex digest of a file, or ``None`` when unavailable.

    Used to stamp artifact / coefficient / kernel provenance into the archive
    manifest. Never raises: missing or unreadable files yield ``None`` so
    provenance capture cannot abort a completed run.
    """
    return sha256_file(path, missing_ok=True, suppress_errors=True) if path else None


def _active_physics_capabilities(sim_cfg: Any) -> list[str]:
    """Return the canonical force models active in ``sim_cfg.flags`` for provenance.

    Recorded in the run manifest so a reader can see exactly which physics the run
    requested - and, paired with ``actual_batch_backend``, whether the chosen backend
    could honor them.
    """
    from lunaris.core.backend_capabilities import FORCE_MODEL_FLAG_ATTR

    flags = getattr(sim_cfg, "flags", None)
    if flags is None:
        return []
    active: list[str] = []
    for canonical, attr in FORCE_MODEL_FLAG_ATTR.items():
        if bool(getattr(flags, attr, False)):
            active.append(canonical)
    return active


def _provenance_text(value: Any) -> str | None:
    """Return a non-empty provenance label, unwrapping enum-like values."""

    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def build_degraded_batch_backend_metadata(
    *,
    requested_backend: Any,
    backend_plan: Any | None,
    backend_diagnostics: Mapping[str, Any] | None,
    requested_sh_degree: int,
    error: Exception,
) -> dict[str, Any]:
    """Build an honest manifest when optional provenance enrichment fails.

    A completed batch run should remain loadable when supplementary provenance
    assembly raises. The fallback must never copy a requested backend into
    ``actual_batch_backend``: that would turn a recorded CPU fallback into a
    false GPU claim. Prefer post-run diagnostics, then the resolved policy plan;
    when neither source is available, make the uncertainty explicit as
    ``"unknown"``.
    """

    diagnostics = backend_diagnostics or {}
    requested = (
        _provenance_text(getattr(backend_plan, "requested_backend", None))
        or _provenance_text(requested_backend)
        or "unknown"
    )
    actual = (
        _provenance_text(diagnostics.get("actual_backend"))
        or _provenance_text(getattr(backend_plan, "actual_backend", None))
        or _provenance_text(diagnostics.get("backend"))
        or "unknown"
    )
    resolved = (
        _provenance_text(getattr(backend_plan, "final_backend", None))
        or _provenance_text(diagnostics.get("backend"))
        or actual
    )
    return {
        "requested_batch_backend": requested,
        "actual_batch_backend": actual,
        "batch_backend": resolved,
        "requested_sh_degree": int(requested_sh_degree),
        "provenance_status": "degraded",
        "provenance_error_type": type(error).__name__,
    }


def _metadata_value_to_jsonable(value: Any) -> Any:
    """
    Convert runtime metadata values into JSON-safe primitives.

    Batch/ensemble archives are often reopened long after the run completed, so
    even lightweight metadata such as seed, cadence, and backend selection is
    worth preserving in a transport-safe form across both HDF5 and NPZ outputs.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_metadata_value_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value_to_jsonable(val) for key, val in value.items()}
    return str(value)


def _decode_archive_metadata(raw: Any) -> dict[str, Any]:
    """
    Decode a metadata payload stored inside an archive.

    The helper is intentionally permissive: missing or malformed metadata
    should not block result loading.
    """

    if raw is None:
        return {}

    try:
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        text = str(raw).strip()
        if not text:
            return {}
        decoded = json.loads(text)
        return decoded if isinstance(decoded, dict) else {}
    except Exception:
        return {}


def _decode_metadata_value(raw: Any) -> Any:
    """Normalize HDF5/NPZ metadata while preserving ordinary strings."""
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text)
            except Exception:
                return raw
        return raw
    if isinstance(raw, np.ndarray):
        return [_decode_metadata_value(item) for item in raw.tolist()]
    return _metadata_value_to_jsonable(raw)


__all__ = [
    "_sha256_file",
    "_active_physics_capabilities",
    "build_degraded_batch_backend_metadata",
    "_metadata_value_to_jsonable",
    "_decode_archive_metadata",
    "_decode_metadata_value",
]
