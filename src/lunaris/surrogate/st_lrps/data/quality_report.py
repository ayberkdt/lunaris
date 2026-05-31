# -*- coding: utf-8 -*-
"""Dataset quality statistics for ST-LRPS HDF5 clouds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"


def _stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _read_all(path: Path, dataset_name: str) -> tuple[np.ndarray, str]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        name = dataset_name if dataset_name in handle else next(
            key for key in handle.keys() if hasattr(handle[key], "shape")
        )
        return np.asarray(handle[name], dtype=np.float64), name


def build_dataset_quality_report(
    data_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    dataset_name: str = "data",
    bins: int = 20,
    split_manifest: Mapping[str, Any] | None = None,
    allow_legacy_dataset_contract: bool = True,
) -> dict[str, Any]:
    """Compute lightweight dataset quality statistics and optionally write files."""

    path = Path(data_path).expanduser().resolve()
    data, resolved_name = _read_all(path, dataset_name)
    contract = DatasetContract.from_hdf5(
        path,
        dataset_name=resolved_name,
        allow_legacy_dataset_contract=allow_legacy_dataset_contract,
        allow_missing_dataset_contract=allow_legacy_dataset_contract,
        allow_legacy_derivative_convention=allow_legacy_dataset_contract,
    )
    xyz = data[:, 0:3]
    r = np.linalg.norm(xyz, axis=1)
    altitude = (r - float(contract.r_ref_m)) / 1000.0
    accel = data[:, 4:7]
    accel_mag = np.linalg.norm(accel, axis=1)
    finite = np.isfinite(data)
    hist_counts, hist_edges = np.histogram(
        altitude[np.isfinite(altitude)],
        bins=max(1, int(bins)),
        range=(float(contract.altitude_min_km), float(contract.altitude_max_km)),
    )
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-30), -1.0, 1.0)))
    rounded_xyz = np.round(xyz, decimals=6)
    duplicate_fraction = 0.0
    if rounded_xyz.shape[0] > 0:
        duplicate_fraction = float(1.0 - np.unique(rounded_xyz, axis=0).shape[0] / float(rounded_xyz.shape[0]))

    report = {
        "schema_version": 1,
        "data_path": str(path),
        "dataset_name": resolved_name,
        "n_samples": int(data.shape[0]),
        "position_norm_m": _stats(r),
        "altitude_km": _stats(altitude),
        "altitude_histogram": {
            "counts": [int(v) for v in hist_counts.tolist()],
            "edges_km": [float(v) for v in hist_edges.tolist()],
        },
        "latitude_deg": _stats(lat),
        "longitude_deg": _stats(lon),
        "residual_potential": _stats(data[:, 3]),
        "residual_acceleration_magnitude": _stats(accel_mag),
        "nan_count": int(np.isnan(data).sum()),
        "inf_count": int(np.isinf(data).sum()),
        "finite_fraction": float(finite.sum() / max(1, data.size)),
        "duplicate_fraction": duplicate_fraction,
        "split_counts": _split_counts(split_manifest),
        "source_gravity_model": contract.source_gravity_model,
        "source_gravity_file_sha256": contract.source_gravity_file_sha256,
        "contract": contract.to_dict(),
        "warnings": [],
    }
    if report["finite_fraction"] < 1.0:
        report["warnings"].append("dataset contains non-finite values")
    if duplicate_fraction > 0.0:
        report["warnings"].append("duplicate positions detected")

    if out_dir is not None:
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "dataset_quality_report.json").write_text(_json_text(report), encoding="utf-8")
        (out / "dataset_quality_summary.md").write_text(_markdown_summary(report), encoding="utf-8")
    return report


def _split_counts(split_manifest: Mapping[str, Any] | None) -> dict[str, int]:
    if not isinstance(split_manifest, Mapping):
        return {}
    return {
        "train": int(split_manifest.get("train_count", 0) or 0),
        "val": int(split_manifest.get("val_count", 0) or 0),
        "test": int(split_manifest.get("test_count", 0) or 0),
        "ood": int(split_manifest.get("ood_count", 0) or 0),
    }


def _markdown_summary(report: Mapping[str, Any]) -> str:
    alt = report.get("altitude_km", {})
    acc = report.get("residual_acceleration_magnitude", {})
    lines = [
        f"# Dataset Quality Summary",
        "",
        f"- Dataset: {report.get('data_path')}",
        f"- Samples: {report.get('n_samples')}",
        f"- Altitude range: {alt.get('min')} to {alt.get('max')} km",
        f"- Residual acceleration magnitude mean: {acc.get('mean')}",
        f"- Finite fraction: {report.get('finite_fraction')}",
        f"- Duplicate fraction: {report.get('duplicate_fraction')}",
        f"- Source gravity model: {report.get('source_gravity_model')}",
        "",
        "## Warnings",
    ]
    warnings = list(report.get("warnings", []) or [])
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    return "\n".join(lines) + "\n"


__all__ = ["build_dataset_quality_report"]
