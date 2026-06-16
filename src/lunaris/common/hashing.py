"""Canonical JSON hashing — the single source of truth for content digests.

Kept dependency-free (standard library only) so both the heavy artifact layer
(``lunaris.surrogate.st_lrps.artifacts``) and the lightweight audit layer
(``lunaris.analysis.monte_carlo.result_audit``) hash *byte-identically* without
either side re-implementing the canonicalization. A digest written by training
can therefore be recomputed and verified by the auditor.

The canonical form is deterministic JSON: keys sorted, two-space indent, ASCII
escaping, a trailing newline, and ``default=str`` for any non-JSON-native value
(so the digest is stable across a JSON round-trip). Callers that may pass
non-JSON-native objects (numpy/torch/Path) should normalize first; for values
that have already been through ``json.loads`` this is a no-op.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_text(payload: Any, *, indent: int = 2) -> str:
    """Return the canonical JSON text used for all content digests."""
    return (
        json.dumps(
            payload,
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        + "\n"
    )


def canonical_json_sha256(payload: Any, *, indent: int = 2) -> str:
    """Return the SHA-256 hex digest of :func:`canonical_json_text`."""
    return hashlib.sha256(canonical_json_text(payload, indent=indent).encode("utf-8")).hexdigest()
