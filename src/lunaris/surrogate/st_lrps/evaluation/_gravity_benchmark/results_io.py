# -*- coding: utf-8 -*-
"""Internal module of the lunar gravity-model benchmark harness.

Part of :mod:`lunaris.surrogate.st_lrps.evaluation.compare_gravity_models`;
this is an implementation detail, not a public API. See that module's
docstring for CLI usage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lunaris.core.config import SimConfig
from lunaris.physics.ephemeris import EphemerisManager

# --- intra-package wiring (auto-generated split) ---
from .types import (
    BENCHMARK_CACHE_SCHEMA_VERSION,
    CachedTrajectory,
    GravityModelCache,
    SCENARIO_MANIFEST_CSV,
    SCENARIO_MANIFEST_JSON,
    SCENARIO_UNIT_DIM,
    Scenario,
    TruthTrajectorySet,
    _METRICS_FIELDNAMES,
    _cfg_with_integrator,
    _find_st_lrps_weight_file,
)
from .compute import (
    _parse_float_list_csv,
    _sobol_note,
    _state_from_elements,
    generate_validation_scenarios,
    propagate_for_scenario,
)

# =============================================================================
# CSV / JSON helpers
# =============================================================================

def _ensure_dir(path: Path) -> None:
    """Create parent directories for a file path if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_scenarios_csv(scenarios: List[Scenario], out_dir: Path) -> None:
    fieldnames = ["scenario_id", "hp_km", "ha_km", "a_km", "e",
                  "inc_deg", "raan_deg", "argp_deg", "ta_deg"]
    p = out_dir / "scenarios.csv"
    _ensure_dir(p)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in scenarios:
            w.writerow({k: getattr(s, k) for k in fieldnames})


def _scenario_count_for_args(args: argparse.Namespace) -> int:
    count = max(0, int(args.random_scenarios))
    limit = getattr(args, "scenario_limit", None)
    if limit is not None:
        count = min(count, max(0, int(limit)))
    return count


def _sampling_metadata(args: argparse.Namespace, scenario_count: Optional[int] = None) -> Dict[str, Any]:
    count = _scenario_count_for_args(args) if scenario_count is None else int(scenario_count)
    method = str(getattr(args, "sampling_method", "random"))
    note = _sobol_note(method, int(args.random_scenarios))
    return {
        "scenario_count": count,
        "requested_random_scenarios": int(args.random_scenarios),
        "scenario_limit": (
            None if getattr(args, "scenario_limit", None) is None
            else int(args.scenario_limit)
        ),
        "sampling_method": method,
        "scenario_seed": int(args.scenario_seed),
        "scenario_mode": str(args.scenario_mode),
        "inclination_sampling": str(getattr(args, "inclination_sampling", "uniform_deg")),
        "altitude_min_km": float(args.altitude_min_km),
        "altitude_max_km": float(args.altitude_max_km),
        "ecc_min": float(args.ecc_min),
        "ecc_max": float(args.ecc_max),
        "inc_min_deg": float(args.inc_min_deg),
        "inc_max_deg": float(args.inc_max_deg),
        "raan_min_deg": float(args.raan_min_deg),
        "raan_max_deg": float(args.raan_max_deg),
        "argp_min_deg": float(args.argp_min_deg),
        "argp_max_deg": float(args.argp_max_deg),
        "ta_min_deg": float(args.ta_min_deg),
        "ta_max_deg": float(args.ta_max_deg),
        "altitude_bounds_km": {
            "min": float(args.altitude_min_km),
            "max": float(args.altitude_max_km),
        },
        "eccentricity_bounds": {
            "min": float(args.ecc_min),
            "max": float(args.ecc_max),
        },
        "inclination_bounds_deg": {
            "min": float(args.inc_min_deg),
            "max": float(args.inc_max_deg),
        },
        "angular_bounds_deg": {
            "raan": {"min": float(args.raan_min_deg), "max": float(args.raan_max_deg)},
            "argp": {"min": float(args.argp_min_deg), "max": float(args.argp_max_deg)},
            "ta": {"min": float(args.ta_min_deg), "max": float(args.ta_max_deg)},
        },
        "module_name": __name__,
        "code_path": str(Path(__file__).resolve()),
        "sampling_note": note,
        "lhs_append_mode": (
            "blockwise" if method == "lhs" and bool(getattr(args, "allow_lhs_append", False))
            else None
        ),
        "warning": (
            "Blockwise LHS append is not equivalent to a single global LHS design."
            if method == "lhs" and bool(getattr(args, "allow_lhs_append", False))
            else ""
        ),
    }


def _scenario_generation_args(args: argparse.Namespace, n: int) -> argparse.Namespace:
    child = argparse.Namespace(**vars(args))
    child.random_scenarios = int(n)
    child.scenario_limit = None
    return child


def _scenario_numeric_tuple(s: Scenario) -> Tuple[float, ...]:
    return (
        float(s.hp_km), float(s.ha_km), float(s.a_km), float(s.e),
        float(s.inc_deg), float(s.raan_deg), float(s.argp_deg), float(s.ta_deg),
    )


def _scenarios_match(a: Scenario, b: Scenario, atol: float = 1e-9) -> bool:
    if int(a.scenario_id) != int(b.scenario_id):
        return False
    av = _scenario_numeric_tuple(a)
    bv = _scenario_numeric_tuple(b)
    return all(math.isclose(x, y, rel_tol=0.0, abs_tol=atol) for x, y in zip(av, bv))


def _renumber_scenarios(scenarios: List[Scenario], start_id: int) -> List[Scenario]:
    out: List[Scenario] = []
    for offset, scenario in enumerate(scenarios):
        out.append(replace(scenario, scenario_id=int(start_id + offset)))
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _scenario_manifest_row(
    scenario: Scenario,
    args: argparse.Namespace,
    csv_mode: bool = False,
) -> Dict[str, Any]:
    raw = scenario.raw_unit_sample
    raw_list = None if raw is None else [float(x) for x in raw]
    row: Dict[str, Any] = {
        "scenario_id": int(scenario.scenario_id),
        "sampling_method": str(getattr(scenario, "sampling_method", getattr(args, "sampling_method", "random"))),
        "scenario_seed": int(args.scenario_seed),
        "scenario_mode": str(args.scenario_mode),
        "inclination_sampling": str(getattr(args, "inclination_sampling", "uniform_deg")),
        "raw_unit_sample": raw_list,
        "hp_km": float(scenario.hp_km),
        "ha_km": float(scenario.ha_km),
        "a_km": float(scenario.a_km),
        "e": float(scenario.e),
        "inc_deg": float(scenario.inc_deg),
        "raan_deg": float(scenario.raan_deg),
        "argp_deg": float(scenario.argp_deg),
        "ta_deg": float(scenario.ta_deg),
    }
    for i in range(SCENARIO_UNIT_DIM):
        row[f"unit_u{i}"] = "" if raw_list is None or i >= len(raw_list) else float(raw_list[i])
    if csv_mode:
        row["raw_unit_sample"] = "" if raw_list is None else json.dumps(raw_list, separators=(",", ":"))
    return row


def _write_scenario_manifest(
    scenarios: List[Scenario],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    metadata = _sampling_metadata(args, len(scenarios))
    rows = [_scenario_manifest_row(s, args, csv_mode=False) for s in scenarios]
    payload = {
        "metadata": metadata,
        "scenarios": rows,
    }

    json_path = out_dir / SCENARIO_MANIFEST_JSON
    _ensure_dir(json_path)
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=4),
        encoding="utf-8",
    )

    csv_path = out_dir / SCENARIO_MANIFEST_CSV
    csv_rows = [_scenario_manifest_row(s, args, csv_mode=True) for s in scenarios]
    fieldnames = [
        "scenario_id", "sampling_method", "scenario_seed", "scenario_mode",
        "inclination_sampling", "raw_unit_sample",
        "unit_u0", "unit_u1", "unit_u2", "unit_u3", "unit_u4", "unit_u5",
        "hp_km", "ha_km", "a_km", "e",
        "inc_deg", "raan_deg", "argp_deg", "ta_deg",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)


def _manifest_value_text(value: Any) -> str:
    if value is None:
        return "<missing>"
    return str(value)


def _manifest_values_equal(old: Any, new: Any) -> bool:
    if old is None:
        return new is None
    if isinstance(new, float):
        try:
            return math.isclose(float(old), float(new), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return old == new


def _verify_scenario_manifest_matches(
    manifest: Dict[str, Any],
    args: argparse.Namespace,
    *,
    require_count: bool = True,
) -> None:
    if getattr(args, "rebuild_metrics", False):
        return
    metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
    expected = _sampling_metadata(args)
    fields = [
        "sampling_method",
        "scenario_seed",
        "scenario_mode",
        "inclination_sampling",
        "altitude_min_km",
        "altitude_max_km",
        "ecc_min",
        "ecc_max",
        "inc_min_deg",
        "inc_max_deg",
        "raan_min_deg",
        "raan_max_deg",
        "argp_min_deg",
        "argp_max_deg",
        "ta_min_deg",
        "ta_max_deg",
    ]
    if require_count:
        fields[2:2] = ["scenario_count", "requested_random_scenarios", "scenario_limit"]
    for field in fields:
        old = metadata.get(field)
        new = expected.get(field)
        if not _manifest_values_equal(old, new):
            raise ValueError(
                "Existing scenario_manifest uses "
                f"{field}={_manifest_value_text(old)} but current request uses "
                f"{_manifest_value_text(new)}."
            )


def _scenario_from_manifest_row(row: Dict[str, Any]) -> Scenario:
    raw = row.get("raw_unit_sample")
    raw_list = None
    if isinstance(raw, list):
        raw_list = [float(x) for x in raw]
    a_km = float(row["a_km"])
    e = float(row["e"])
    inc_deg = float(row["inc_deg"])
    raan_deg = float(row["raan_deg"])
    argp_deg = float(row["argp_deg"])
    ta_deg = float(row["ta_deg"])
    state = _state_from_elements(a_km * 1_000.0, e, inc_deg, raan_deg, argp_deg, ta_deg)
    return Scenario(
        scenario_id=int(row["scenario_id"]),
        hp_km=float(row["hp_km"]),
        ha_km=float(row["ha_km"]),
        a_km=a_km,
        e=e,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        ta_deg=ta_deg,
        initial_state=state,
        raw_unit_sample=raw_list,
        sampling_method=str(row.get("sampling_method", "random")),
    )


def _load_scenarios_from_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    *,
    require_count: bool = True,
) -> List[Scenario]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_scenario_manifest_matches(manifest, args, require_count=require_count)
    rows = manifest.get("scenarios", [])
    if not isinstance(rows, list):
        raise ValueError("Existing scenario_manifest has invalid scenarios payload.")
    scenarios = [_scenario_from_manifest_row(row) for row in rows]
    if require_count and len(scenarios) != _scenario_count_for_args(args):
        raise ValueError(
            "Existing scenario_manifest uses scenario_count="
            f"{len(scenarios)} but current request uses {_scenario_count_for_args(args)}."
        )
    return scenarios


def prepare_scenarios(args: argparse.Namespace, out_dir: Path) -> List[Scenario]:
    manifest_path = out_dir / SCENARIO_MANIFEST_JSON
    use_existing = (
        bool(getattr(args, "resume", False))
        or bool(getattr(args, "reuse_cache", False))
        or bool(getattr(args, "rebuild_metrics", False))
        or int(getattr(args, "append_scenarios", 0) or 0) > 0
    )
    if use_existing and manifest_path.exists():
        existing = _load_scenarios_from_manifest(manifest_path, args, require_count=False)
        existing_count = len(existing)
        append_count = max(0, int(getattr(args, "append_scenarios", 0) or 0))
        if append_count > 0:
            target_count = existing_count + append_count
        elif bool(getattr(args, "rebuild_metrics", False)):
            target_count = existing_count
        else:
            target_count = int(args.random_scenarios)

        if target_count < existing_count:
            raise ValueError(
                "Existing scenario_manifest uses scenario_count="
                f"{existing_count} but current request targets {target_count}. "
                "Use a new output directory or request at least the existing count."
            )
        if target_count == existing_count:
            scenarios = existing
            _write_scenarios_csv(scenarios, out_dir)
            print(f"[cache] Scenario manifest found: {len(scenarios)} scenarios.", flush=True)
            return scenarios

        method = str(getattr(args, "sampling_method", "random"))
        if method == "lhs" and not bool(getattr(args, "allow_lhs_append", False)):
            raise ValueError(
                f"Existing LHS manifest has {existing_count} scenarios. "
                "LHS is not naturally nested. Use Sobol/Sobol-scrambled for "
                "extendable benchmark sets, or rerun a fresh benchmark. "
                "Use --allow-lhs-append for explicit blockwise LHS append."
            )

        if method == "lhs":
            block_args = _scenario_generation_args(args, append_count or (target_count - existing_count))
            block_args.scenario_seed = int(args.scenario_seed) + existing_count
            new_block = generate_validation_scenarios(block_args)
            scenarios = existing + _renumber_scenarios(new_block, existing_count)
            print("[cache] WARNING: blockwise LHS append is not equivalent to a single "
                  "global LHS design.", flush=True)
        else:
            generated = generate_validation_scenarios(_scenario_generation_args(args, target_count))
            for old, new in zip(existing, generated[:existing_count]):
                if not _scenarios_match(old, new):
                    raise ValueError(
                        "Existing scenario_manifest is incompatible with regenerated "
                        f"{method} sequence at scenario_id={old.scenario_id}."
                    )
            scenarios = generated

        print(f"[cache] Extending scenario manifest: {existing_count} -> {len(scenarios)}.",
              flush=True)
        _write_scenarios_csv(scenarios, out_dir)
        _write_scenario_manifest(scenarios, args, out_dir)
        return scenarios

    if bool(getattr(args, "resume", False)) and manifest_path.exists():
        scenarios = _load_scenarios_from_manifest(manifest_path, args)
        _write_scenarios_csv(scenarios, out_dir)
        print(f"[scenarios] resume: loaded {len(scenarios)} scenarios from {manifest_path}",
              flush=True)
        return scenarios

    scenarios = generate_validation_scenarios(args)
    if getattr(args, "scenario_limit", None) is not None:
        scenarios = scenarios[:int(args.scenario_limit)]
    note = _sobol_note(str(getattr(args, "sampling_method", "random")), int(args.random_scenarios))
    if note:
        print(f"[scenarios] NOTE: {note}", flush=True)
    _write_scenarios_csv(scenarios, out_dir)
    _write_scenario_manifest(scenarios, args, out_dir)
    return scenarios


def _append_metrics_csv(metrics: Dict, path: Path, write_header: bool) -> None:
    _ensure_dir(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METRICS_FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(metrics)


def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _cache_requested(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "cache_trajectories", False)
        or getattr(args, "reuse_cache", False)
        or getattr(args, "rebuild_metrics", False)
        or getattr(args, "resume", False)
        or int(getattr(args, "append_scenarios", 0) or 0) > 0
    )


def _benchmark_cache_dir(args: argparse.Namespace, out_dir: Path) -> Path:
    raw = getattr(args, "cache_dir", None)
    return Path(raw) if raw else out_dir / "benchmark_cache"


def _safe_cache_name(name: str) -> str:
    clean = str(name).strip().lower().replace("gpu_", "").replace("_rk4", "")
    clean = clean.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return clean or "unknown"


def _truth_cache_name(args: argparse.Namespace) -> str:
    return f"{str(args.truth).lower()}_{str(getattr(args, 'truth_integrator', 'DOP853')).lower()}"


def _trajectory_cache_path(
    cache_dir: Path,
    model_type: str,
    model_name: str,
    scenario_id: int,
    args: Optional[argparse.Namespace] = None,
) -> Path:
    if model_type == "truth":
        group = cache_dir / "truth" / _safe_cache_name(model_name)
    else:
        group = cache_dir / "models" / _safe_cache_name(model_name)
    return group / f"scenario_{int(scenario_id):06d}.npz"


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    weight_file = _find_st_lrps_weight_file(getattr(args, "st_lrps_model_dir", None))
    return {
        "cache_schema_version": BENCHMARK_CACHE_SCHEMA_VERSION,
        "truth": str(args.truth).lower(),
        "truth_integrator": str(getattr(args, "truth_integrator", "DOP853")),
        "duration_days": float(args.duration_days),
        "dt_out": float(args.dt_out),
        "rk4_dt_s": (
            None if getattr(args, "rk4_dt_s", None) is None
            else float(args.rk4_dt_s)
        ),
        "gpu_rk4_dt_s_list": _parse_float_list_csv(getattr(args, "gpu_rk4_dt_s_list", None)),
        "st_lrps_rk4_dt": float(getattr(args, "st_lrps_rk4_dt", 30.0)),
        "gpu_integrator": str(getattr(args, "gpu_integrator", "medium")),
        "torch_dtype": str(getattr(args, "torch_dtype", "float64")),
        "batch_frame_mode": str(getattr(args, "batch_frame_mode", "match_dynamics_engine")),
        "st_lrps_model_dir": getattr(args, "st_lrps_model_dir", None),
        "st_lrps_weight_file": weight_file,
        "st_lrps_weight_sha256": _file_sha256(weight_file),
    }


def _write_cache_manifest(
    args: argparse.Namespace,
    cache_dir: Path,
    scenarios: List[Scenario],
    selected_models: Optional[List[str]] = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_schema_version": BENCHMARK_CACHE_SCHEMA_VERSION,
        "metadata": _cache_metadata(args),
        # Top-level deterministic fingerprint (Fix 5). Kept outside ``metadata``
        # so it does not change the cache key or the field-by-field validator.
        "config_fingerprint": _config_fingerprint(args),
        "scenario_count": len(scenarios),
        "scenario_ids": [int(s.scenario_id) for s in scenarios],
        "selected_models": selected_models or [],
        "updated_utc_s": time.time(),
    }
    path = cache_dir / "cache_manifest.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=4), encoding="utf-8")
    os.replace(tmp, path)


def _validate_cache_compatibility(
    args: argparse.Namespace, cache_dir: Path
) -> List[str]:
    """Field-by-field + fingerprint cache-compatibility gate.

    Returns a list of human-readable warnings (possibly empty). By default any
    incompatibility raises ``ValueError`` so stale trajectories are never
    silently reused; ``--allow-stale-cache`` downgrades every refusal to a
    warning instead. A cache written before fingerprints existed is allowed
    with a warning (the explicit field checks above still apply).
    """
    warnings: List[str] = []
    if getattr(args, "rebuild_metrics", False):
        return warnings
    path = cache_dir / "cache_manifest.json"
    if not path.exists():
        return warnings
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Existing benchmark cache manifest is unreadable: {exc}") from exc

    allow_stale = bool(getattr(args, "allow_stale_cache", False))

    def _refuse(message: str) -> None:
        # Default behaviour refuses reuse; --allow-stale-cache downgrades to warn.
        if allow_stale:
            warnings.append("[allow-stale-cache] " + message)
        else:
            raise ValueError(message)

    old_meta = old.get("metadata", {})
    new_meta = _cache_metadata(args)
    fields = [
        "cache_schema_version", "truth", "truth_integrator", "duration_days", "dt_out",
        "rk4_dt_s", "st_lrps_rk4_dt", "gpu_integrator", "torch_dtype",
        "batch_frame_mode",
    ]
    for field in fields:
        old_value = old_meta.get(field)
        new_value = new_meta.get(field)
        if not _manifest_values_equal(old_value, new_value):
            _refuse(
                "Existing benchmark cache uses "
                f"{field}={_manifest_value_text(old_value)} but current request uses "
                f"{_manifest_value_text(new_value)}. Refusing to reuse cached trajectories."
            )
    if "st_lrps" in str(getattr(args, "models", "")) or "st_lrps" in str(getattr(args, "gpu_models", "")):
        for field in ("st_lrps_model_dir", "st_lrps_weight_file", "st_lrps_weight_sha256"):
            old_value = old_meta.get(field)
            new_value = new_meta.get(field)
            if old_value and new_value and old_value != new_value:
                _refuse(
                    "Existing benchmark cache uses "
                    f"{field}={_manifest_value_text(old_value)} but current request uses "
                    f"{_manifest_value_text(new_value)}. Refusing to reuse ST-LRPS trajectories."
                )

    # Deterministic fingerprint guard (Fix 5). Catches drift in cache-relevant
    # fields that are not in the explicit list above (e.g. gpu_rk4_dt_s_list and
    # the ST-LRPS weight hash). Missing fingerprint => allow + warn.
    old_fp = old.get("config_fingerprint")
    new_fp = _config_fingerprint(args)
    if not old_fp:
        warnings.append(
            "Existing benchmark cache predates config fingerprints; reusing without a "
            "fingerprint check. Re-run without --reuse-cache to stamp one."
        )
    elif str(old_fp) != str(new_fp):
        _refuse(
            f"Existing benchmark cache fingerprint {str(old_fp)[:12]} does not match the "
            f"current configuration fingerprint {str(new_fp)[:12]}. Refusing to reuse cached "
            "trajectories (pass --allow-stale-cache to override)."
        )
    return warnings


def _save_cached_trajectory(
    cache_dir: Path,
    scenario: Scenario,
    model_name: str,
    model_type: str,
    t: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    *,
    runtime_s: float = float("nan"),
    integrator: str = "",
    rk4_dt_s: Optional[float] = None,
    dtype: str = "",
    device: str = "",
    backend: str = "",
    truth_model: str = "",
) -> Path:
    path = _trajectory_cache_path(cache_dir, model_type, model_name, scenario.scenario_id, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {
        "cache_schema_version": np.asarray(BENCHMARK_CACHE_SCHEMA_VERSION, dtype=np.int64),
        "scenario_id": np.asarray(int(scenario.scenario_id), dtype=np.int64),
        "model_name": np.asarray(str(model_name)),
        "model_type": np.asarray(str(model_type)),
        "integrator": np.asarray(str(integrator)),
        "t": np.asarray(t, dtype=np.float64),
        "state": np.asarray(y, dtype=np.float64),
        "position": np.asarray(y, dtype=np.float64)[:, :3],
        "velocity": np.asarray(y, dtype=np.float64)[:, 3:],
        "duration_days": np.asarray(float(args.duration_days), dtype=np.float64),
        "dt_out": np.asarray(float(args.dt_out), dtype=np.float64),
        "rk4_dt_s": np.asarray(float("nan") if rk4_dt_s is None else float(rk4_dt_s), dtype=np.float64),
        "dtype": np.asarray(str(dtype)),
        "device": np.asarray(str(device)),
        "backend": np.asarray(str(backend)),
        "truth_model": np.asarray(str(truth_model)),
        "runtime_s": np.asarray(float(runtime_s), dtype=np.float64),
        "hp_km": np.asarray(float(scenario.hp_km), dtype=np.float64),
        "ha_km": np.asarray(float(scenario.ha_km), dtype=np.float64),
        "a_km": np.asarray(float(scenario.a_km), dtype=np.float64),
        "e": np.asarray(float(scenario.e), dtype=np.float64),
        "inc_deg": np.asarray(float(scenario.inc_deg), dtype=np.float64),
        "raan_deg": np.asarray(float(scenario.raan_deg), dtype=np.float64),
        "argp_deg": np.asarray(float(scenario.argp_deg), dtype=np.float64),
        "ta_deg": np.asarray(float(scenario.ta_deg), dtype=np.float64),
        "frame_mode": np.asarray(str(getattr(args, "batch_frame_mode", ""))),
        "gpu_integrator": np.asarray(str(getattr(args, "gpu_integrator", ""))),
    }
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    print(f"[cache] Saved model={model_name} scenario={scenario.scenario_id:06d} file={path}",
          flush=True)
    return path


def _load_cached_trajectory(path: Path) -> Optional[CachedTrajectory]:
    if path.suffix != ".npz" or not path.exists() or path.name.endswith(".tmp"):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["cache_schema_version"]).item())
            if version != BENCHMARK_CACHE_SCHEMA_VERSION:
                return None
            t = np.asarray(data["t"], dtype=np.float64)
            y = np.asarray(data["state"], dtype=np.float64)
            if t.ndim != 1 or y.ndim != 2 or y.shape[1] != 6 or y.shape[0] != t.shape[0]:
                return None
            if not np.isfinite(t).all() or not np.isfinite(y).all():
                return None
            metadata = {k: data[k].tolist() for k in data.files if k not in {"t", "state", "position", "velocity"}}
            runtime = float(np.asarray(data["runtime_s"]).item()) if "runtime_s" in data.files else float("nan")
            return CachedTrajectory(t=t, y=y, runtime_s=runtime, metadata=metadata)
    except Exception:
        return None


def _cached_truth_path(cache_dir: Path, args: argparse.Namespace, scenario_id: int) -> Path:
    return _trajectory_cache_path(cache_dir, "truth", _truth_cache_name(args), scenario_id, args)


def _cached_model_path(cache_dir: Path, model_name: str, scenario_id: int) -> Path:
    return _trajectory_cache_path(cache_dir, "comparison_model", model_name, scenario_id)


def _truth_cache_completion(
    cache_dir: Path,
    args: argparse.Namespace,
    scenarios: List[Scenario],
) -> Tuple[int, List[Scenario]]:
    complete = 0
    missing: List[Scenario] = []
    for scenario in scenarios:
        path = _cached_truth_path(cache_dir, args, scenario.scenario_id)
        if _load_cached_trajectory(path) is not None:
            complete += 1
        else:
            missing.append(scenario)
    return complete, missing


def _model_cache_completion(
    cache_dir: Path,
    model_name: str,
    scenarios: List[Scenario],
) -> Tuple[int, List[Scenario]]:
    complete = 0
    missing: List[Scenario] = []
    for scenario in scenarios:
        path = _cached_model_path(cache_dir, model_name, scenario.scenario_id)
        if _load_cached_trajectory(path) is not None:
            complete += 1
        else:
            missing.append(scenario)
    return complete, missing


# String-valued metric columns that must not be coerced to float on reload.
_METRIC_STRING_KEYS = {"model", "reference", "status", "backend", "device", "failure_reason"}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a metrics CSV into a list of string-valued dict rows (empty if absent)."""
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _coerce_numeric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce CSV string values to float where possible for aggregation.

    Non-numeric / blank values become NaN (so finite-guards skip them); known
    string columns are preserved verbatim.
    """
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if key in _METRIC_STRING_KEYS:
            out[key] = value
            continue
        if value in (None, "", "None", "nan", "NaN"):
            out[key] = float("nan")
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out




# ---------------------------------------------------------------------------
# Reproducibility metadata helpers (bookkeeping only; never touch physics).
# ---------------------------------------------------------------------------

BENCHMARK_METADATA_SCHEMA_VERSION = 2

# Cache-relevant config fields that define a deterministic compatibility key.
# Mirrors the trajectory cache key so a fingerprint change implies the cached
# trajectories were produced under a different configuration.
_FINGERPRINT_FIELDS = (
    "cache_schema_version", "truth", "truth_integrator", "duration_days", "dt_out",
    "rk4_dt_s", "gpu_rk4_dt_s_list", "st_lrps_rk4_dt", "gpu_integrator", "torch_dtype",
    "batch_frame_mode", "st_lrps_weight_sha256",
)


def _ensure_run_id(args: argparse.Namespace) -> str:
    """Stable per-process run id, shared by run_metadata.json and the summary."""
    rid = getattr(args, "_run_id", None)
    if not rid:
        rid = uuid.uuid4().hex[:16]
        setattr(args, "_run_id", rid)
    return str(rid)


def _package_version() -> str:
    """Best-effort lunaris package version; never raises."""
    try:
        from lunaris.surrogate.st_lrps import __version__ as _v
        if _v:
            return str(_v)
    except Exception:
        pass
    try:
        from importlib.metadata import version as _pkg_version
        return str(_pkg_version("lunaris"))
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_mode_label(args: argparse.Namespace) -> str:
    return "gpu_batch" if bool(getattr(args, "gpu_batch_compare", False)) else "cpu_sweep"


def _device_info(args: argparse.Namespace) -> Dict[str, Any]:
    """Compute-device metadata without forcing a torch import on CPU-only runs.

    For GPU-batch requests this performs a guarded lazy ``torch`` import to read
    CUDA availability/device; it never raises. CPU-sweep runs report the CPU
    backend without importing torch.
    """
    if not bool(getattr(args, "gpu_batch_compare", False)):
        return {
            "backend": "cpu_sweep",
            "cuda_available": False,
            "device": "cpu",
            "device_name": None,
            "torch_version": None,
        }
    info: Dict[str, Any] = {
        "backend": "gpu_batch",
        "cuda_available": False,
        "device": None,
        "device_name": None,
        "torch_version": None,
        "requested_fallback": str(getattr(args, "gpu_fallback", "error")),
    }
    # --refresh-metadata must not import torch; report the device as unprobed.
    if getattr(args, "_no_torch_probe", False):
        info["note"] = "torch probe skipped (metadata refresh)"
        return info
    try:
        import torch
        info["torch_version"] = str(getattr(torch, "__version__", ""))
        cuda = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda
        if cuda:
            info["device"] = "cuda:0"
            try:
                info["device_name"] = str(torch.cuda.get_device_name(0))
            except Exception:
                info["device_name"] = None
        elif str(getattr(args, "gpu_fallback", "error")) == "cpu":
            info["device"] = "cpu"
    except Exception:
        pass
    return info


def _config_fingerprint(args: argparse.Namespace) -> str:
    """Deterministic sha256 over the cache-relevant configuration.

    Used only to gate cache reuse and to stamp metadata — never alters physics.
    """
    meta = _cache_metadata(args)
    canon = {k: meta.get(k) for k in _FINGERPRINT_FIELDS}
    blob = json.dumps(canon, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _effective_frame_modes(args: argparse.Namespace) -> Dict[str, str]:
    """Unambiguous frame-mode reporting for the GPU batch path.

    The harness passes ``--batch-frame-mode`` straight to the integrator, so the
    requested and effective modes are identical; the DOP853 truth always runs in
    the rotating dynamics-engine frame.
    """
    requested = str(getattr(args, "batch_frame_mode", "match_dynamics_engine"))
    return {
        "requested_batch_frame_mode": requested,
        "effective_batch_frame_mode": requested,
        "gpu_frame_mode": requested,
        "truth_frame_mode": "match_dynamics_engine",
    }


def _model_status_breakdown(
    requested_display: List[str],
    status_by_model: Dict[str, str],
    aggregate_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Split requested models into completed/failed/partial/skipped buckets.

    ``status_by_model`` maps a display name to ``completed|failed|partial|skipped``.
    ``models_in_metrics`` is the set of identities that contributed aggregate rows.
    """
    in_metrics = sorted({str(r.get("model")) for r in aggregate_rows if r.get("model")})
    completed: List[str] = []
    failed: List[str] = []
    partial: List[str] = []
    skipped: List[str] = []
    for name in requested_display:
        st = str(status_by_model.get(name, "skipped"))
        if st == "completed":
            completed.append(name)
        elif st == "failed":
            failed.append(name)
        elif st == "partial":
            partial.append(name)
        else:
            skipped.append(name)
    return {
        "requested_models": list(requested_display),
        "completed_models": completed,
        "failed_models": failed,
        "partial_models": partial,
        "skipped_models": skipped,
        "models_in_metrics": in_metrics,
    }


def _benchmark_consistency_warnings(
    breakdown: Dict[str, Any], summary: Dict[str, Any]
) -> List[str]:
    """Lightweight end-of-run sanity checks. Returns human warnings (never raises)."""
    warns: List[str] = []
    requested = breakdown.get("requested_models", [])
    in_metrics = set(breakdown.get("models_in_metrics", []))
    failed = set(breakdown.get("failed_models", []))
    for name in breakdown.get("failed_models", []):
        warns.append(f"Model {name} failed and is excluded from accuracy metrics.")
    for name in breakdown.get("partial_models", []):
        warns.append(
            f"Model {name} has partial cache coverage; metrics cover only cached scenarios."
        )
    for name in breakdown.get("skipped_models", []):
        warns.append(f"Model {name} was requested but produced no result (skipped).")
    for name in requested:
        if name not in in_metrics and name not in failed:
            warns.append(f"Requested model {name} produced no aggregate metrics.")
    if requested and not in_metrics:
        warns.append("No requested model produced aggregate metrics.")
    if int(summary.get("n_scenarios_total", 0) or 0) <= 0:
        warns.append("Scenario count is zero; metrics may be empty.")
    return warns


def _cache_provenance(
    args: argparse.Namespace,
    cache_dir: Optional[Path],
    *,
    enabled: bool,
    truth_counts: Dict[str, Any],
    model_entries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the cache-provenance block recorded in the summary (Fix 4)."""
    return {
        "enabled": bool(enabled),
        "cache_dir": (str(cache_dir) if cache_dir is not None else None),
        "config_fingerprint": _config_fingerprint(args),
        "truth": {
            "model": _truth_cache_name(args),
            "integrator": str(getattr(args, "truth_integrator", "DOP853")),
            **truth_counts,
        },
        "models": model_entries,
    }


def _build_gpu_batch_summary(
    args: argparse.Namespace,
    *,
    aggregate_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    gpu_models: List[str],
    requested_display: List[str],
    status_by_model: Dict[str, str],
    n_scenarios_total: int,
    n_scenarios_new_this_run: int,
    truth_total_runtime_s: Optional[float],
    truth_mean_runtime_per_scenario_s: Optional[float],
    equivalent: Dict[str, Any],
    selected: Dict[str, Any],
    cache_provenance: Dict[str, Any],
    rebuilt_from_cache: bool,
    source: str,
    extra_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical ``gpu_batch_summary`` payload for every code path.

    Centralising this guarantees the live run, the cache rebuild, and the
    metadata refresh emit consistent, self-describing bookkeeping (models,
    frame modes, cache provenance, scope, and warnings).
    """
    frame = _effective_frame_modes(args)
    breakdown = _model_status_breakdown(requested_display, status_by_model, aggregate_rows)

    n_requested = len(breakdown["requested_models"])
    n_completed = len(breakdown["completed_models"])
    all_done = (
        n_requested > 0
        and n_completed == n_requested
        and not breakdown["failed_models"]
        and not breakdown["partial_models"]
        and not breakdown["skipped_models"]
    )
    summary_scope = "all_requested_models" if all_done else "completed_models_only"
    if all_done:
        summary_note = f"All {n_requested} requested model(s) completed."
    elif n_requested == 0:
        summary_note = "No models were requested."
    else:
        bits = [f"{n_completed}/{n_requested} requested model(s) completed"]
        if breakdown["failed_models"]:
            bits.append("failed: " + ", ".join(breakdown["failed_models"]))
        if breakdown["partial_models"]:
            bits.append("partial: " + ", ".join(breakdown["partial_models"]))
        if breakdown["skipped_models"]:
            bits.append("skipped: " + ", ".join(breakdown["skipped_models"]))
        summary_note = "; ".join(bits) + "."

    summary: Dict[str, Any] = {
        "schema_version": BENCHMARK_METADATA_SCHEMA_VERSION,
        "run_id": _ensure_run_id(args),
        "timestamp_utc": _utc_now_iso(),
        "source": str(source),
        "truth": _truth_cache_name(args),
        "truth_integrator": str(getattr(args, "truth_integrator", "DOP853")),
        "gpu_models": list(gpu_models),
        "gpu_model_variants": list(requested_display),
        "n_scenarios_total": int(n_scenarios_total),
        "n_scenarios_new_this_run": int(n_scenarios_new_this_run),
        "accumulated": bool(getattr(args, "resume", False)),
        "rebuilt_from_cache": bool(rebuilt_from_cache),
        "sampling": _sampling_metadata(args, n_scenarios_total),
        "truth_workers": int(getattr(args, "workers", 1)),
        "gpu_integrator": str(getattr(args, "gpu_integrator", "medium")),
        "gpu_finite_check_mode": str(getattr(args, "gpu_finite_check_mode", "snapshot")),
        # Frame metadata (Fix 3): explicit, never collapses to a single field.
        "frame_mode": frame["effective_batch_frame_mode"],
        "requested_batch_frame_mode": frame["requested_batch_frame_mode"],
        "effective_batch_frame_mode": frame["effective_batch_frame_mode"],
        "gpu_frame_mode": frame["gpu_frame_mode"],
        "truth_frame_mode": frame["truth_frame_mode"],
        "uses_lunar_rotation": frame["effective_batch_frame_mode"] == "match_dynamics_engine",
        "matches_dynamics_engine_frame":
            frame["effective_batch_frame_mode"] == "match_dynamics_engine",
        # Model accounting (Fix 2).
        "requested_models": breakdown["requested_models"],
        "completed_models": breakdown["completed_models"],
        "failed_models": breakdown["failed_models"],
        "partial_models": breakdown["partial_models"],
        "skipped_models": breakdown["skipped_models"],
        "models_in_metrics": breakdown["models_in_metrics"],
        # Cache provenance (Fix 4) + deterministic fingerprint (Fix 5).
        "cache_provenance": cache_provenance,
        "config_fingerprint": _config_fingerprint(args),
        "truth_total_runtime_s": truth_total_runtime_s,
        "truth_mean_runtime_per_scenario_s": truth_mean_runtime_per_scenario_s,
        "equivalent_sh_degree": equivalent,
        "selected_stlrps_scenarios": selected,
        # Honest scope language (Fix 8).
        "summary_scope": summary_scope,
        "summary_note": summary_note,
        "aggregate": aggregate_rows,
        "runtime": runtime_rows,
    }
    warns = _benchmark_consistency_warnings(breakdown, summary)
    if extra_warnings:
        warns = list(extra_warnings) + warns
    summary["metadata_warnings"] = warns
    return summary


def _write_run_metadata(
    args: argparse.Namespace,
    out_dir: Path,
    scenarios: Optional[List[Scenario]] = None,
) -> None:
    """Persist reproducibility metadata for the validation run."""

    weight_file = _find_st_lrps_weight_file(getattr(args, "st_lrps_model_dir", None))
    scenario_count = len(scenarios) if scenarios is not None else _scenario_count_for_args(args)
    meta = {
        "models": [m.strip().lower() for m in str(args.models).split(",") if m.strip()],
        "truth": str(args.truth).lower(),
        "random_scenarios": int(args.random_scenarios),
        "scenario_count": int(scenario_count),
        "scenario_seed": int(args.scenario_seed),
        "scenario_mode": str(args.scenario_mode),
        "sampling_method": str(getattr(args, "sampling_method", "random")),
        "inclination_sampling": str(getattr(args, "inclination_sampling", "uniform_deg")),
        "altitude_min_km": float(args.altitude_min_km),
        "altitude_max_km": float(args.altitude_max_km),
        "ecc_min": float(args.ecc_min),
        "ecc_max": float(args.ecc_max),
        "inc_min_deg": float(args.inc_min_deg),
        "inc_max_deg": float(args.inc_max_deg),
        "raan_min_deg": float(args.raan_min_deg),
        "raan_max_deg": float(args.raan_max_deg),
        "argp_min_deg": float(args.argp_min_deg),
        "argp_max_deg": float(args.argp_max_deg),
        "ta_min_deg": float(args.ta_min_deg),
        "ta_max_deg": float(args.ta_max_deg),
        "duration_days": float(args.duration_days),
        "dt_out_s": float(args.dt_out),
        "integrator": str(args.integrator),
        "workers": int(getattr(args, "workers", 1)),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_step_s": float(args.max_step),
        "st_lrps_model_dir": getattr(args, "st_lrps_model_dir", None),
        "st_lrps_weight_file": weight_file,
        "st_lrps_mode": str(args.st_lrps_mode),
        "batch_rk4": bool(args.batch_rk4),
        "batch_rk4_reference": str(args.batch_rk4_reference),
        "rk4_dt_s": float(args.rk4_dt_s if args.rk4_dt_s is not None else args.st_lrps_rk4_dt),
        "gpu_rk4_dt_s_list": _parse_float_list_csv(getattr(args, "gpu_rk4_dt_s_list", None)),
        "gpu_fallback": str(args.gpu_fallback),
        "gpu_finite_check_mode": str(getattr(args, "gpu_finite_check_mode", "snapshot")),
        "torch_dtype": str(args.torch_dtype),
        "force_batch_size": int(args.force_batch_size),
        "cache_trajectories": bool(getattr(args, "cache_trajectories", False)),
        "reuse_cache": bool(getattr(args, "reuse_cache", False)),
        "cache_dir": getattr(args, "cache_dir", None),
        "append_scenarios": int(getattr(args, "append_scenarios", 0) or 0),
        "rebuild_metrics": bool(getattr(args, "rebuild_metrics", False)),
        "strict_complete": bool(getattr(args, "strict_complete", False)),
        "allow_lhs_append": bool(getattr(args, "allow_lhs_append", False)),
    }
    # Reproducibility/bookkeeping (Fix 1): identity, environment, and the
    # deterministic config fingerprint. All additive — existing keys unchanged.
    meta["schema_version"] = BENCHMARK_METADATA_SCHEMA_VERSION
    meta["run_id"] = _ensure_run_id(args)
    meta["timestamp_utc"] = _utc_now_iso()
    meta["argv"] = list(sys.argv)
    meta["lunaris_version"] = _package_version()
    meta["mode"] = _run_mode_label(args)
    meta["truth_integrator"] = str(getattr(args, "truth_integrator", "DOP853"))
    meta["gpu_integrator"] = str(getattr(args, "gpu_integrator", "medium"))
    meta.update(_effective_frame_modes(args))
    meta["device"] = _device_info(args)
    meta["cache_enabled"] = bool(_cache_requested(args))
    meta["config_fingerprint"] = _config_fingerprint(args)
    meta["sampling"] = _sampling_metadata(args, scenario_count)
    p = out_dir / "run_metadata.json"
    _ensure_dir(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, default=str)


def _truth_cache_metadata(args: argparse.Namespace, scenarios: List[Scenario]) -> Dict[str, Any]:
    return {
        "truth": str(args.truth).lower(),
        "random_scenarios": int(len(scenarios)),
        "scenario_seed": int(args.scenario_seed),
        "scenario_mode": str(args.scenario_mode),
        "sampling_method": str(getattr(args, "sampling_method", "random")),
        "inclination_sampling": str(getattr(args, "inclination_sampling", "uniform_deg")),
        "altitude_min_km": float(args.altitude_min_km),
        "altitude_max_km": float(args.altitude_max_km),
        "ecc_min": float(args.ecc_min),
        "ecc_max": float(args.ecc_max),
        "inc_min_deg": float(args.inc_min_deg),
        "inc_max_deg": float(args.inc_max_deg),
        "raan_min_deg": float(args.raan_min_deg),
        "raan_max_deg": float(args.raan_max_deg),
        "argp_min_deg": float(args.argp_min_deg),
        "argp_max_deg": float(args.argp_max_deg),
        "ta_min_deg": float(args.ta_min_deg),
        "ta_max_deg": float(args.ta_max_deg),
        "duration_days": float(args.duration_days),
        "dt_out_s": float(args.dt_out),
        "integrator": str(args.integrator),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_step_s": float(args.max_step),
        "scenario_ids": [int(s.scenario_id) for s in scenarios],
    }


def _truth_cache_available(cache_dir: Path, args: argparse.Namespace, scenarios: List[Scenario]) -> bool:
    """Cheap predicate: would ``_load_truth_cache`` produce a valid hit?

    Checks file presence + metadata equality without loading the (large) NPZ.
    Used only to decide the overall-progress weighting (truth weight collapses
    when truth is served from cache).
    """
    meta_path = cache_dir / "truth_metadata.json"
    npz_path = cache_dir / "sh200_dop853_trajectories.npz"
    if not meta_path.exists() or not npz_path.exists():
        return False
    try:
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return old_meta == _truth_cache_metadata(args, scenarios)
    except Exception:
        return False


def _load_truth_cache(cache_dir: Path, args: argparse.Namespace, scenarios: List[Scenario]) -> Optional[TruthTrajectorySet]:
    meta_path = cache_dir / "truth_metadata.json"
    npz_path = cache_dir / "sh200_dop853_trajectories.npz"
    if not meta_path.exists() or not npz_path.exists():
        return None
    try:
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta != _truth_cache_metadata(args, scenarios):
            return None
        data = np.load(npz_path)
        t_common = data["t"]
        y_all = data["y"]
        runtime = data["runtime"]
        t_by = {s.scenario_id: np.asarray(t_common, dtype=np.float64) for s in scenarios}
        y_by = {s.scenario_id: np.asarray(y_all[i], dtype=np.float64) for i, s in enumerate(scenarios)}
        rt_by = {s.scenario_id: float(runtime[i]) for i, s in enumerate(scenarios)}
        print(f"[truth] Reused cache: {npz_path}", flush=True)
        return TruthTrajectorySet("sh200_dop853", t_by, y_by, rt_by)
    except Exception as exc:
        print(f"[truth] Cache ignored: {exc}", flush=True)
        return None


def _save_truth_cache(cache_dir: Path, args: argparse.Namespace, scenarios: List[Scenario], truth: TruthTrajectorySet) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = truth.t_by_scenario[scenarios[0].scenario_id]
    y_all = np.stack([truth.y_by_scenario[s.scenario_id] for s in scenarios], axis=0)
    runtime = np.asarray([truth.runtime_by_scenario[s.scenario_id] for s in scenarios], dtype=np.float64)
    np.savez_compressed(cache_dir / "sh200_dop853_trajectories.npz", t=t0, y=y_all, runtime=runtime)
    (cache_dir / "truth_metadata.json").write_text(
        json.dumps(_truth_cache_metadata(args, scenarios), indent=4),
        encoding="utf-8",
    )


def build_truth_trajectory_set(
    args: argparse.Namespace,
    scenarios: List[Scenario],
    cfg_base: SimConfig,
    ephem: Any,
    model_cache: GravityModelCache,
    truth_dir: Path,
    on_progress: Optional[Any] = None,
) -> TruthTrajectorySet:
    """Generate or load SH200 DOP853 truth trajectories for all scenarios.

    ``on_progress`` (optional) is invoked as ``cb(completed, total, elapsed_s,
    eta_s)`` after each scenario finishes; it is logging-only.
    """

    def _report(completed: int, total: int, elapsed_s: float) -> None:
        if on_progress is None:
            return
        rate = completed / max(elapsed_s, 1e-9)
        eta = (total - completed) / max(rate, 1e-9)
        try:
            on_progress(int(completed), int(total), float(elapsed_s), float(eta))
        except Exception:
            pass

    if args.reuse_truth_cache:
        cached = _load_truth_cache(truth_dir, args, scenarios)
        if cached is not None:
            return cached

    cache_enabled = _cache_requested(args) or bool(getattr(args, "cache_truth", False))
    cache_dir = _benchmark_cache_dir(args, Path(args.output_dir))
    t_by: Dict[int, np.ndarray] = {}
    y_by: Dict[int, np.ndarray] = {}
    rt_by: Dict[int, float] = {}
    truth_model = str(args.truth).lower()
    truth_integrator = str(getattr(args, "truth_integrator", "DOP853"))
    truth_cfg = _cfg_with_integrator(cfg_base, truth_integrator)
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    pending_scenarios = list(scenarios)
    if cache_enabled:
        complete, pending_scenarios = _truth_cache_completion(cache_dir, args, scenarios)
        print(f"[cache] Truth cache {_truth_cache_name(args)}: {complete}/{len(scenarios)} complete.",
              flush=True)
        for scenario in scenarios:
            cached = _load_cached_trajectory(_cached_truth_path(cache_dir, args, scenario.scenario_id))
            if cached is None:
                continue
            t_by[scenario.scenario_id] = cached.t
            y_by[scenario.scenario_id] = cached.y
            rt_by[scenario.scenario_id] = cached.runtime_s
    print(f"[truth] Building {truth_model.upper()} {truth_integrator} reference "
          f"for {len(pending_scenarios)} missing of {len(scenarios)} scenarios.", flush=True)
    t_truth_start = time.perf_counter()
    if workers > 1 and len(pending_scenarios) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        worker_count = min(workers, len(pending_scenarios))
        print(f"[truth] CPU parallel truth generation: {worker_count} workers "
              f"(integrator={truth_integrator}).", flush=True)
        payloads = [(scenario, truth_model) for scenario in pending_scenarios]
        completed = 0
        scenario_by_id = {s.scenario_id: s for s in pending_scenarios}
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_parallel_worker_init,
            initargs=(args, cfg_base),
        ) as executor:
            futures = {executor.submit(_parallel_worker_truth, p): p[0] for p in payloads}
            for future in as_completed(futures):
                scenario = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    if args.fail_fast:
                        raise RuntimeError(
                            f"truth propagation failed for scenario {scenario.scenario_id}: {exc}"
                        ) from exc
                    print(f"[truth] WARNING: scenario {scenario.scenario_id} failed: {exc}",
                          flush=True)
                    continue
                if result.get("truth_failed"):
                    if args.fail_fast:
                        raise RuntimeError(f"truth propagation failed for scenario {scenario.scenario_id}")
                    print(f"[truth] WARNING: scenario {scenario.scenario_id} failed; "
                          "omitted from metrics.", flush=True)
                    continue
                sid = int(result["scenario_id"])
                t_by[sid] = np.asarray(result["t"], dtype=np.float64)
                y_by[sid] = np.asarray(result["y"], dtype=np.float64)
                rt_by[sid] = float(result["truth_rt"])
                if cache_enabled and not result.get("saved_to_cache"):
                    _save_cached_trajectory(
                        cache_dir, scenario_by_id[sid], _truth_cache_name(args), "truth",
                        t_by[sid], y_by[sid], args,
                        runtime_s=rt_by[sid], integrator=truth_integrator,
                        dtype="float64", device="cpu", truth_model=truth_model,
                    )
                elapsed = time.perf_counter() - t_truth_start
                rate = completed / max(elapsed, 1e-9)
                remaining = (len(pending_scenarios) - completed) / max(rate, 1e-9)
                mm, ss = divmod(int(remaining), 60)
                hh, mm = divmod(mm, 60)
                print(f"[truth] Scenario {completed:03d}/{len(pending_scenarios)} done "
                      f"| id={sid} | runtime={rt_by[sid]:.2f}s "
                      f"| ETA {hh:02d}:{mm:02d}:{ss:02d} "
                      f"| elapsed {elapsed/60.0:.1f} min", flush=True)
                _report(completed, len(pending_scenarios), elapsed)
    else:
        for idx, scenario in enumerate(pending_scenarios, 1):
            print(f"\n[truth] Scenario {idx:03d}/{len(pending_scenarios)} | id={scenario.scenario_id} "
                  f"| hp={scenario.hp_km:.0f} km  ha={scenario.ha_km:.0f} km  "
                  f"i={scenario.inc_deg:.1f} deg", flush=True)
            res, runtime = propagate_for_scenario(
                truth_model, scenario.initial_state, args, truth_cfg, ephem, model_cache
            )
            if res is None:
                if args.fail_fast:
                    raise RuntimeError(f"truth propagation failed for scenario {scenario.scenario_id}")
                print(f"[truth] WARNING: scenario {scenario.scenario_id} failed; omitted from metrics.",
                      flush=True)
                continue
            t_by[scenario.scenario_id] = np.asarray(res.t, dtype=np.float64)
            y_by[scenario.scenario_id] = np.asarray(res.y, dtype=np.float64)
            rt_by[scenario.scenario_id] = float(runtime)
            if cache_enabled:
                _save_cached_trajectory(
                    cache_dir, scenario, _truth_cache_name(args), "truth",
                    t_by[scenario.scenario_id], y_by[scenario.scenario_id], args,
                    runtime_s=float(runtime), integrator=truth_integrator,
                    dtype="float64", device="cpu", truth_model=truth_model,
                )
            elapsed = time.perf_counter() - t_truth_start
            rate = idx / max(elapsed, 1e-9)
            remaining = (len(pending_scenarios) - idx) / max(rate, 1e-9)
            mm, ss = divmod(int(remaining), 60)
            hh, mm = divmod(mm, 60)
            print(f"[truth] Scenario {idx:03d}/{len(pending_scenarios)} done in {runtime:.2f}s "
                  f"| ETA {hh:02d}:{mm:02d}:{ss:02d} "
                  f"| elapsed {elapsed/60.0:.1f} min", flush=True)
            _report(idx, len(pending_scenarios), elapsed)

    truth = TruthTrajectorySet(f"{truth_model}_{truth_integrator.lower()}", t_by, y_by, rt_by)
    if args.cache_truth and len(t_by) == len(scenarios):
        _save_truth_cache(truth_dir, args, scenarios, truth)
    return truth


# ---- moved here to keep the import graph acyclic ----
def _parallel_worker_init(args: argparse.Namespace, cfg_base: SimConfig) -> None:
    """ProcessPool initializer: build per-worker ephemeris + gravity caches once."""
    ephem = EphemerisManager.from_time_and_spice(cfg_base.time, cfg_base.spice)
    _PARALLEL_STATE["args"] = args
    _PARALLEL_STATE["cfg_base"] = cfg_base
    _PARALLEL_STATE["truth_cfg"] = _cfg_with_integrator(
        cfg_base, str(getattr(args, "truth_integrator", "DOP853"))
    )
    _PARALLEL_STATE["ephem"] = ephem
    _PARALLEL_STATE["cache"] = GravityModelCache(cfg_base, args)
def _parallel_worker_truth(payload: Tuple[Scenario, str]) -> Dict[str, Any]:
    """Propagate only the adaptive truth trajectory for one scenario."""

    scenario, truth_model = payload
    st = _PARALLEL_STATE
    args = st["args"]
    truth_cfg = st["truth_cfg"]
    ephem = st["ephem"]
    cache = st["cache"]
    truth_res, truth_rt = propagate_for_scenario(
        truth_model, scenario.initial_state, args, truth_cfg, ephem, cache
    )
    if truth_res is None:
        return {
            "scenario_id": scenario.scenario_id,
            "truth_failed": True,
            "truth_rt": None,
        }
    saved = False
    if _cache_requested(args) or bool(getattr(args, "cache_truth", False)):
        try:
            cache_dir = _benchmark_cache_dir(args, Path(args.output_dir))
            _save_cached_trajectory(
                cache_dir, scenario, _truth_cache_name(args), "truth",
                np.asarray(truth_res.t, dtype=np.float64),
                np.asarray(truth_res.y, dtype=np.float64),
                args,
                runtime_s=float(truth_rt),
                integrator=str(getattr(args, "truth_integrator", "DOP853")),
                dtype="float64",
                device="cpu",
                truth_model=truth_model,
            )
            saved = True
        except Exception:
            saved = False
    return {
        "scenario_id": scenario.scenario_id,
        "truth_failed": False,
        "truth_rt": float(truth_rt),
        "t": np.asarray(truth_res.t, dtype=np.float64),
        "y": np.asarray(truth_res.y, dtype=np.float64),
        "saved_to_cache": saved,
    }

_PARALLEL_STATE: Dict[str, Any] = {}
