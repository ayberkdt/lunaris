# -*- coding: utf-8 -*-
"""Lightweight ST-LRPS dataset validation and report writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract, DatasetContractError

GravityFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"


def _sample_indices(n_total: int, n_check: int, seed: int) -> np.ndarray:
    n = min(int(n_total), max(1, int(n_check)))
    if n >= int(n_total):
        return np.arange(int(n_total), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n_total), size=n, replace=False).astype(np.int64))


def _read_subset(path: Path, dataset_name: str, indices: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        name = dataset_name if dataset_name in handle else next(
            key for key in handle.keys() if hasattr(handle[key], "shape")
        )
        ds = handle[name]
        shape = tuple(int(v) for v in ds.shape)
        return np.asarray(ds[indices, :], dtype=np.float64), shape


def validate_dataset_file(
    data_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    dataset_name: str = "data",
    n_check: int = 1024,
    seed: int = 0,
    strict: bool = True,
    truth_fn: Optional[GravityFn] = None,
    baseline_fn: Optional[GravityFn] = None,
    potential_atol: float = 1e-8,
    accel_atol: float = 1e-10,
    allow_legacy_dataset_contract: bool = False,
    allow_missing_dataset_contract: bool = False,
    allow_legacy_derivative_convention: bool = False,
) -> dict[str, Any]:
    """Validate an ST-LRPS HDF5 dataset and optionally write a JSON report."""

    path = Path(data_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []

    try:
        contract = DatasetContract.from_hdf5(
            path,
            dataset_name=dataset_name,
            allow_legacy_dataset_contract=allow_legacy_dataset_contract,
            allow_missing_dataset_contract=allow_missing_dataset_contract,
            allow_legacy_derivative_convention=allow_legacy_derivative_convention,
        )
        checked.append("dataset_contract")
    except Exception as exc:
        contract = None
        errors.append(f"dataset contract invalid: {exc}")

    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        name = dataset_name if dataset_name in handle else None
        if name is None:
            for key in handle.keys():
                if hasattr(handle[key], "shape"):
                    name = key
                    break
        if name is None:
            errors.append("no HDF5 dataset found")
            shape = (0, 0)
        else:
            shape = tuple(int(v) for v in handle[name].shape)
    if len(shape) != 2 or shape[1] != 7:
        errors.append(f"dataset must have shape (N, 7), got {shape}")
        n_total = int(shape[0]) if shape else 0
        data = np.empty((0, 7), dtype=np.float64)
    else:
        n_total = int(shape[0])
        idx = _sample_indices(n_total, n_check, seed)
        data, _ = _read_subset(path, dataset_name, idx)
    checked.append("shape")

    finite_mask = np.isfinite(data)
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())
    if nan_count:
        errors.append(f"dataset contains {nan_count} NaN values in checked subset")
    if inf_count:
        errors.append(f"dataset contains {inf_count} Inf values in checked subset")
    checked.append("finite_values")

    r = np.linalg.norm(data[:, 0:3], axis=1) if data.size else np.asarray([], dtype=float)
    r_ref = float(contract.r_ref_m if contract is not None else 1_737_400.0)
    altitude_km = (r - r_ref) / 1000.0 if r.size else np.asarray([], dtype=float)
    if r.size and np.any(r <= r_ref):
        errors.append("position norm must exceed lunar reference radius for orbital shell samples")
    if contract is not None and altitude_km.size:
        lo = float(contract.altitude_min_km)
        hi = float(contract.altitude_max_km)
        outside = (altitude_km < lo - 1e-9) | (altitude_km > hi + 1e-9)
        if np.any(outside):
            msg = f"{int(outside.sum())} checked samples are outside contract altitude envelope [{lo}, {hi}] km"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
    checked.append("altitude_envelope")

    residual_potential_max_abs_error: Optional[float] = None
    residual_accel_max_abs_error: Optional[float] = None
    if truth_fn is not None:
        u_truth, a_truth = truth_fn(data[:, 0:3])
        if baseline_fn is not None:
            u_base, a_base = baseline_fn(data[:, 0:3])
        else:
            u_base = np.zeros_like(np.asarray(u_truth))
            a_base = np.zeros_like(np.asarray(a_truth))
        expected_u = np.asarray(u_truth).reshape(-1) - np.asarray(u_base).reshape(-1)
        expected_a = np.asarray(a_truth).reshape(-1, 3) - np.asarray(a_base).reshape(-1, 3)
        observed_u = data[:, 3].reshape(-1)
        observed_a = data[:, 4:7].reshape(-1, 3)
        residual_potential_max_abs_error = float(np.max(np.abs(observed_u - expected_u))) if observed_u.size else 0.0
        residual_accel_max_abs_error = float(np.max(np.abs(observed_a - expected_a))) if observed_a.size else 0.0
        if residual_potential_max_abs_error > float(potential_atol):
            errors.append(
                "residual potential label mismatch: "
                f"max_abs_error={residual_potential_max_abs_error:.6e}"
            )
        if residual_accel_max_abs_error > float(accel_atol):
            errors.append(
                "residual acceleration label mismatch: "
                f"max_abs_error={residual_accel_max_abs_error:.6e}"
            )
        checked.append("residual_label_recompute")

    duplicate_fraction = 0.0
    if data.shape[0] > 1:
        rounded = np.round(data[:, 0:3], decimals=6)
        unique = np.unique(rounded, axis=0).shape[0]
        duplicate_fraction = float(1.0 - unique / float(data.shape[0]))
        if duplicate_fraction > 0.0:
            warnings.append(f"duplicate point estimate in checked subset: {duplicate_fraction:.6f}")
    checked.append("duplicate_points")

    accel_mag = np.linalg.norm(data[:, 4:7], axis=1) if data.size else np.asarray([], dtype=float)
    outlier_summary: dict[str, Any] = {}
    if accel_mag.size:
        median = float(np.median(accel_mag))
        p99 = float(np.percentile(accel_mag, 99))
        maxv = float(np.max(accel_mag))
        outlier_summary = {"accel_mag_median": median, "accel_mag_p99": p99, "accel_mag_max": maxv}
        if median > 0.0 and maxv > max(100.0 * median, p99 * 10.0):
            warnings.append("extreme residual acceleration outlier detected")
    checked.append("outliers")

    report = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked": sorted(set(checked)),
        "data_path": str(path),
        "dataset_name": dataset_name,
        "n_samples_total": int(n_total),
        "n_samples_checked": int(data.shape[0]),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "altitude_min_km": float(np.min(altitude_km)) if altitude_km.size else None,
        "altitude_max_km": float(np.max(altitude_km)) if altitude_km.size else None,
        "residual_accel_max_abs_error": residual_accel_max_abs_error,
        "residual_potential_max_abs_error": residual_potential_max_abs_error,
        "duplicate_fraction": duplicate_fraction,
        "outlier_summary": outlier_summary,
        "contract": contract.to_dict() if contract is not None else None,
    }
    if out_dir is not None:
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "dataset_validation_report.json").write_text(_json_text(report), encoding="utf-8")
    return report


def require_dataset_valid(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = validate_dataset_file(*args, **kwargs)
    if not report["passed"]:
        raise DatasetContractError("; ".join(str(item) for item in report["errors"]))
    return report


__all__ = ["GravityFn", "require_dataset_valid", "validate_dataset_file"]
