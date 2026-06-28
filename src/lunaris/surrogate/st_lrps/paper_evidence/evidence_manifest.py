"""Evidence manifest + provenance helpers for the ST-LRPS paper pipeline.

The evidence manifest makes a run reproducible and auditable: it records the
environment (git, Python, package versions, torch/CUDA), and for every external
input/output a `{path, sha256, missing_reason}` record so a **missing file is
recorded as missing, not silently ignored**.

The manifest can be written before training completes (dry-run / planning), so
the evidence workspace always documents intent even when artifacts do not exist
yet.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lunaris.common.provenance import sha256_file, sha256_text, utc_now_iso

# Reuse the benchmark provenance collectors so environment/git capture is
# consistent across the whole ST-LRPS evidence stack.
from lunaris.surrogate.st_lrps.evaluation.provenance import (
    artifact_record as _benchmark_artifact_record,
)
from lunaris.surrogate.st_lrps.evaluation.provenance import (
    collect_environment as _collect_env,
)
from lunaris.surrogate.st_lrps.evaluation.provenance import (
    collect_git_info,
)

EVIDENCE_MANIFEST_SCHEMA_VERSION = 1


def compute_file_sha256(path: str | Path | None) -> str | None:
    """SHA-256 of a file, or ``None`` if the path is missing/not a file."""
    return sha256_file(path, missing_ok=True)


def compute_json_hash(obj: Any) -> str:
    """Stable SHA-256 over a JSON-serializable object (sorted keys, canonical)."""
    text = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    return sha256_text(text)


def artifact_record(path: str | Path | None, *, label: str = "file") -> dict[str, Any]:
    """`{path, sha256, missing_reason}` for one artifact. Missing → recorded, not dropped."""
    return _benchmark_artifact_record(path, label=label)


def collect_environment() -> dict[str, Any]:
    """Environment snapshot: git, Python, platform, and package/torch/CUDA versions."""
    env = dict(_collect_env())
    env["git"] = collect_git_info()
    env["collected_at_utc"] = utc_now_iso()
    return env


def build_evidence_manifest(
    *,
    stage: str,
    run_key: str,
    config_path: str | Path | None,
    config: Mapping[str, Any] | None,
    out_dir: str | Path | None,
    artifacts: Mapping[str, str | Path | None] | None = None,
    command: list[str] | None = None,
    dry_run: bool = False,
    environment: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a single-run evidence entry.

    ``artifacts`` maps a logical name (e.g. ``checkpoint``, ``scaler``,
    ``split_manifest``, ``artifact_contract``, ``dataset``) to a path; each is
    turned into a `{path, sha256, missing_reason}` record.
    """
    artifact_records = {
        name: artifact_record(path, label=name) for name, path in (artifacts or {}).items()
    }
    entry: dict[str, Any] = {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "stage": str(stage),
        "run_key": str(run_key),
        "dry_run": bool(dry_run),
        "created_at_utc": utc_now_iso(),
        "config": {
            "path": (str(config_path) if config_path is not None else None),
            "file_sha256": compute_file_sha256(config_path),
            "content_hash": (compute_json_hash(config) if config is not None else None),
        },
        "out_dir": (str(out_dir) if out_dir is not None else None),
        "command": list(command) if command else None,
        "artifacts": artifact_records,
        "environment": dict(environment) if environment is not None else collect_environment(),
    }
    if extra:
        entry["extra"] = dict(extra)
    return entry


def write_evidence_manifest(
    manifest_path: str | Path,
    *,
    run_key: str,
    entry: Mapping[str, Any],
) -> Path:
    """Merge a run entry into the top-level evidence manifest under ``runs[run_key]``.

    Multiple seeds/stages accumulate in one canonical file. Writing is additive,
    so the manifest can be built incrementally as runs complete.
    """
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any]
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    if not isinstance(manifest, dict) or "runs" not in manifest:
        manifest = {
            "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "created_at_utc": utc_now_iso(),
            "runs": {},
        }
    manifest["updated_at_utc"] = utc_now_iso()
    manifest["runs"][str(run_key)] = dict(entry)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "artifact_record",
    "build_evidence_manifest",
    "collect_environment",
    "compute_file_sha256",
    "compute_json_hash",
    "utc_now_iso",
    "write_evidence_manifest",
]
