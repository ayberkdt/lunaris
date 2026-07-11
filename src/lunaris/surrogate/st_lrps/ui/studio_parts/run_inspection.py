"""Pure run-inspection helpers shared by the Studio Runs and Monitor pages.

The helpers in this module deliberately do not import Qt.  Run manifests and
history files are user-owned artifacts, so malformed or partially-written files
must degrade to an empty/partial view instead of taking the UI down.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | int | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _nested(mapping: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value not in (None, ""):
            return value
    return None


def load_run_records(training_dir: Path) -> list[dict[str, Any]]:
    """Return normalized records for valid manifests, newest first."""

    records: list[dict[str, Any]] = []
    try:
        manifests = sorted(
            training_dir.glob("*/run_manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        manifests = []
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        if not manifest:
            continue
        try:
            mtime = manifest_path.stat().st_mtime
            date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            date = "—"
        record = dict(manifest)
        record.update(
            {
                "path": str(manifest_path.parent),
                "manifest_path": str(manifest_path),
                "name": str(manifest.get("run_id") or manifest_path.parent.name),
                "status": str(manifest.get("status", "unknown")).lower(),
                "date": date,
                "best_score_value": _number(
                    _first(manifest, "best_score", "best_metric_value")
                ),
                "best_epoch_value": _number(manifest.get("best_epoch")),
                "latest_epoch_value": _number(manifest.get("latest_epoch")),
                "dataset": format_dataset(manifest),
                "preset": format_preset(manifest),
                "device": format_device(manifest),
            }
        )
        records.append(record)
    return records


def format_dataset(manifest: dict[str, Any]) -> str:
    meta = manifest.get("dataset_meta")
    if isinstance(meta, dict) and meta.get("dataset_name"):
        name = str(meta["dataset_name"])
        if name and name != "data":
            return name
    paths = manifest.get("data_paths")
    if isinstance(paths, dict):
        for key in ("data", "train_data", "train"):
            value = paths.get(key)
            if value:
                return Path(str(value)).name
    if isinstance(meta, dict) and meta.get("dataset_name"):
        return str(meta["dataset_name"])
    return "—"


def format_preset(manifest: dict[str, Any]) -> str:
    summary = manifest.get("resolved_config_summary")
    if isinstance(summary, dict):
        value = _first(summary, "model_preset", "run_preset", "preset")
        if value:
            return str(value)
    return str(_first(manifest, "model_preset", "preset") or "custom")


def format_device(manifest: dict[str, Any]) -> str:
    accounting = manifest.get("compute_accounting")
    if isinstance(accounting, dict):
        value = _first(accounting, "device", "actual_device")
        if value:
            return str(value)
    return str(_first(manifest, "actual_device", "device") or "—")


def read_jsonl(path: Path, *, limit: int = 10_000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= limit:
                    break
                try:
                    value = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def read_history(run_dir: Path, *, limit: int = 10_000) -> list[dict[str, Any]]:
    """Read the canonical history file, accepting jsonl and legacy csv."""

    jsonl = run_dir / "history.jsonl"
    if jsonl.is_file():
        return read_jsonl(jsonl, limit=limit)
    csv_path = run_dir / "history.csv"
    if not csv_path.is_file():
        return []
    import csv

    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in list(csv.DictReader(handle))[:limit]]
    except (OSError, csv.Error):
        return []


def read_periodic_evals(run_dir: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    """Normalize periodic-evaluation rows from known artifact locations."""

    candidates = (
        run_dir / "periodic_evals" / "history.jsonl",
        run_dir / "periodic_evals" / "periodic_eval_history.jsonl",
        run_dir / "periodic_eval_history.jsonl",
    )
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if path.is_file():
            rows = read_jsonl(path, limit=limit)
            if rows:
                break
    normalized: list[dict[str, Any]] = []
    for row in rows:
        epoch = _first(row, "epoch", "eval_epoch", "step")
        u = _first(row, "rmse_u", "val_rmse_u", "potential_rmse", "rmse_potential")
        a = _first(row, "rmse_a", "val_rmse_a", "accel_rmse", "rmse_accel")
        angle = _first(row, "ang", "angle", "angular_error_deg", "mean_angle_deg")
        if epoch is None and u is None and a is None and angle is None:
            continue
        normalized.append(
            {"epoch": _number(epoch), "rmse_u": _number(u), "rmse_a": _number(a), "angle": _number(angle), "raw": row}
        )
    return normalized


def _flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def config_diff(run_dirs: Iterable[Path]) -> list[dict[str, str]]:
    """Create a stable, human-readable diff for two to four run configs."""

    paths = list(run_dirs)
    configs: list[dict[str, Any]] = []
    for path in paths:
        config = read_json(path / "config.json")
        if not config:
            config = read_json(path / "resolved_config.json")
        configs.append(_flatten(config))
    keys = sorted({key for config in configs for key in config})
    result: list[dict[str, str]] = []
    labels = [path.name for path in paths]
    for key in keys:
        values = [config.get(key, "—") for config in configs]
        if len({json.dumps(value, sort_keys=True, default=str) for value in values}) <= 1:
            continue
        row = {"field": key}
        for index, label in enumerate(labels):
            row[label] = _display_value(values[index])
        result.append(row)
    return result


def _display_value(value: Any) -> str:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def provenance_items(run_dir: Path) -> list[tuple[str, str, str]]:
    """Return ``(label, value, semantic_kind)`` provenance rows."""

    manifest = read_json(run_dir / "run_manifest.json")
    preflight = read_json(run_dir / "preflight_report.json")
    if not preflight:
        preflight = read_json(run_dir / "provenance" / "preflight_report.json")
    config = read_json(run_dir / "config.json")
    if not manifest and not preflight and not config:
        return []
    split = _first(manifest, "split_policy") or _nested(
        manifest, ("dataset_meta", "split_policy"), ("resolved_config_summary", "split_policy")
    ) or _first(config, "split_policy") or "—"
    counts = _first(manifest, "split_counts", "dataset_counts")
    if not isinstance(counts, dict):
        counts = _nested(manifest, ("dataset_meta", "split_counts"), ("dataset_meta", "counts"))
    count_text = "—"
    if isinstance(counts, dict):
        count_text = ", ".join(f"{key}={counts[key]}" for key in ("train", "val", "test") if key in counts) or "—"
    scaler_scope = _first(manifest, "scaler_fit_scope") or _nested(
        manifest, ("scaler", "fit_scope"), ("provenance", "scaler_fit_scope")
    ) or _first(config, "scaler_fit_scope") or "—"
    preflight_ok = _first(preflight, "status", "result", "decision") or _first(manifest, "preflight_status") or "—"
    preflight_kind = "success" if str(preflight_ok).lower() in {"ok", "pass", "passed", "go", "ready"} else "warning"
    dataset_hash = _first(manifest, "content_sha256", "dataset_sha256", "dataset_hash") or _nested(
        manifest,
        ("dataset_contract", "content_sha256"),
        ("dataset_meta", "content_sha256"),
        ("dataset_meta", "sha256"),
        ("dataset_meta", "dataset_sha256"),
    ) or "—"
    if dataset_hash != "—":
        dataset_hash = str(dataset_hash)[:16]
    requested = _first(manifest, "requested_device") or _nested(manifest, ("compute_accounting", "requested_device")) or "—"
    actual = _first(manifest, "actual_device") or _nested(manifest, ("compute_accounting", "device")) or format_device(manifest)
    device = str(requested) if requested == actual else f"{requested} → {actual}"
    deterministic = _first(manifest, "deterministic", "determinism") or _nested(manifest, ("provenance", "deterministic")) or "—"
    return [
        ("Split policy", str(split), "info"),
        ("Split counts", count_text, "info"),
        ("Scaler fit scope", str(scaler_scope), "success" if str(scaler_scope).lower() == "train_only" else "warning"),
        ("Preflight", str(preflight_ok), preflight_kind),
        ("Dataset hash", str(dataset_hash), "info"),
        ("Device", device, "info"),
        ("Determinism", str(deterministic), "info"),
    ]
