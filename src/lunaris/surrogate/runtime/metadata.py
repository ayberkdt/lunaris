"""Metadata and central-body resolution helpers for surrogate runtime artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lunaris.common.lunar_data import resolve_lunar_gravity_path

def _extract_degree_metadata(config: dict[str, Any]) -> tuple:
    """
    Resolve ``degree_min`` and ``degree_max`` from a run ``config.json``.

    Resolution order (mirrors ``st_lrps/evaluation/cli.py``):
    1. Top-level ``degree_min`` / ``degree_max`` keys.
    2. ``dataset_meta.degree_min`` / ``dataset_meta.degree_max`` fallback.
    3. ``dataset_meta.requested_degree`` as a last resort for ``degree_max``.
    Raises ``ValueError`` when ``degree_max`` cannot be resolved.
    """

    dm = config.get("dataset_meta") or {}

    deg_min = config.get("degree_min")
    if deg_min is None:
        deg_min = dm.get("degree_min")

    deg_max = config.get("degree_max")
    if deg_max is None:
        deg_max = dm.get("degree_max")
    if deg_max is None:
        deg_max = dm.get("requested_degree")

    if deg_max is None:
        raise ValueError(
            "ST-LRPS model is missing degree metadata. "
            "Expected 'degree_max' in config.json at the top level or under 'dataset_meta'. "
            "Re-generate the dataset with spatial_cloud_generator.py >= v2.0 which writes "
            "degree_max to config.json automatically."
        )

    return int(deg_min if deg_min is not None else 0), int(deg_max)


def _config_path_value(config: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty path-like value from config or dataset_meta."""

    mappings: list[dict[str, Any]] = [config]
    dataset_meta = config.get("dataset_meta")
    if isinstance(dataset_meta, dict):
        mappings.append(dataset_meta)

    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        cloud_json = mapping.get("cloud_config_json")
        if isinstance(cloud_json, str) and cloud_json.strip():
            try:
                nested = json.loads(cloud_json)
            except Exception:
                nested = {}
            if isinstance(nested, dict):
                for key in keys:
                    value = nested.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

    return None


def _resolve_baseline_gravity_path(config: dict[str, Any]) -> Path:
    """
    Resolve the SH coefficient file used for the ST-LRPS baseline.

    Older run configs may not carry the path explicitly.  In that case we use
    the surrogate pipeline SSOT default, which is the repository-local lunar
    JGGRX file used by the generator defaults.
    """

    path_value = _config_path_value(
        config,
        "gravity_model_path",
        "gfc_path",
        "gravity_gfc_path",
        "gravity_file_path",
    )
    if path_value:
        return resolve_lunar_gravity_path(path_value)
    return resolve_lunar_gravity_path()

@dataclass(frozen=True, slots=True)
class SurrogateGravityMetadata:
    """User-facing summary of a loaded surrogate gravity run."""

    model_dir: str
    training_mode: str
    scaler_kind: str
    activation: str
    hidden: int
    depth: int
    a_sign: float
    mu_m3s2: float
    r_ref_m: float
    device: str

__all__ = [
    "SurrogateGravityMetadata",
    "_extract_degree_metadata",
    "_config_path_value",
    "_resolve_baseline_gravity_path",
]
