"""Lightweight ST-LRPS dataset validation and report writing."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract, DatasetContractError

GravityFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"


# Headroom over the first-order storage-error bounds below. Measured on a stored
# float32 residual dataset (SH degree 20..200, 60-100 km altitude, 2000 points):
# potential error <= 0.24 and acceleration error <= 0.04 of the scale each factor
# multiplies, so 4 leaves >= 16x margin while a corrupted label (wrong sign,
# shifted coefficient, wrong degree band) is off by O(1) of its own magnitude.
_RECOMPUTE_TOL_FACTOR = 4.0


def _recompute_tolerances(
    xyz: np.ndarray,
    u: np.ndarray,
    a: np.ndarray,
    storage_dtype: np.dtype,
    degree_max: int | None,
) -> tuple[float, float]:
    """Absolute (potential, acceleration) tolerances for the label recompute check.

    Labels are computed in float64 at float64 positions, then the whole row is
    rounded to the storage dtype (machine epsilon ``eps``). Recomputing at the
    rounded positions therefore differs from the stored labels by, per point and
    to first order in ``eps``:

    * potential: ``0.5*eps*|U|`` (label rounding) + ``|a . dr| <= 0.5*eps*|a|*|r|``
      (position rounding, ``|dr| <= 0.5*eps*|r|``, ``grad U = a``);
    * acceleration: ``0.5*eps*|a|`` + ``|H dr|``; for a field of spherical-harmonic
      degree <= N about the origin ``|H| <~ (N + 2)*|a|/|r|``, so
      ``|H dr| <~ 0.5*eps*(N + 2)*|a|``.

    A fixed 1e-8 / 1e-10 tolerance rejected every float32 dataset. The
    position-rounding term is what a magnitude-only ``k*eps*max|U|`` rule
    misses: for a residual band starting at degree n it is ~(n + 1) times the
    label-rounding term (measured: 8.9 eps*max|U| on the dataset above).
    """
    eps = float(np.finfo(storage_dtype).eps) if np.issubdtype(storage_dtype, np.floating) else float(np.finfo(np.float64).eps)
    if u.size == 0:
        return 0.0, 0.0
    r_norm = np.linalg.norm(xyz, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    pot_scale = float(np.max(np.abs(u) + a_norm * r_norm))
    acc_scale = float((max(int(degree_max or 0), 0) + 2) * np.max(a_norm))
    return _RECOMPUTE_TOL_FACTOR * eps * pot_scale, _RECOMPUTE_TOL_FACTOR * eps * acc_scale


def _sample_indices(n_total: int, n_check: int, seed: int) -> np.ndarray:
    n = min(int(n_total), max(1, int(n_check)))
    if n >= int(n_total):
        return np.arange(int(n_total), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n_total), size=n, replace=False).astype(np.int64))


def _read_subset(path: Path, dataset_name: str, indices: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    import h5py

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
    truth_fn: GravityFn | None = None,
    baseline_fn: GravityFn | None = None,
    potential_atol: float | None = None,
    accel_atol: float | None = None,
) -> dict[str, Any]:
    """Validate an ST-LRPS HDF5 dataset and optionally write a JSON report.

    ``potential_atol``/``accel_atol`` default to ``None``: the label recompute
    tolerance is then derived from the storage dtype (see
    ``_recompute_tolerances``). The stamped ``content_sha256`` is recomputed from
    the payload and compared, so a file edited after stamping fails.
    """

    path = Path(data_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []

    try:
        contract = DatasetContract.from_hdf5(
            path,
            dataset_name=dataset_name,
        )
        checked.append("dataset_contract")
    except Exception as exc:
        contract = None
        errors.append(f"dataset contract invalid: {exc}")

    import h5py

    with h5py.File(path, "r") as handle:
        name = dataset_name if dataset_name in handle else None
        if name is None:
            for key in handle.keys():
                if hasattr(handle[key], "shape"):
                    name = key
                    break
        shape: tuple[int, ...]
        stored_dtype: np.dtype = np.dtype(np.float64)
        if name is None:
            errors.append("no HDF5 dataset found")
            shape = (0, 0)
        else:
            shape = tuple(int(v) for v in handle[name].shape)
            stored_dtype = np.dtype(handle[name].dtype)
    if len(shape) != 2 or shape[1] != 7:
        errors.append(f"dataset must have shape (N, 7), got {shape}")
        n_total = int(shape[0]) if shape else 0
        data = np.empty((0, 7), dtype=np.float64)
    else:
        n_total = int(shape[0])
        if contract is not None and int(contract.n_samples) != n_total:
            errors.append(f"contract n_samples={contract.n_samples} does not match HDF5 rows={n_total}")
        if contract is not None:
            layout_shape = (contract.dataset_layout or {}).get("shape")
            if isinstance(layout_shape, list | tuple) and len(layout_shape) >= 2:
                expected_shape = tuple(int(v) for v in layout_shape[:2])
                if expected_shape != tuple(shape[:2]):
                    errors.append(f"contract dataset_layout.shape={expected_shape} does not match HDF5 shape={shape}")
        idx = _sample_indices(n_total, n_check, seed)
        data, _ = _read_subset(path, dataset_name, idx)
    checked.append("shape")

    # content_sha256 is the tamper check: recompute it from the payload and
    # compare, instead of trusting the value stamped next to it.
    content_sha256_recorded: str | None = None
    content_sha256_observed: str | None = None
    if contract is not None and name is not None and len(shape) == 2 and shape[1] == 7:
        from lunaris.surrogate.st_lrps.data.dataset_contract import content_sha256_for_hdf5_dataset

        content_sha256_recorded = contract.content_sha256 or None
        content_sha256_observed = content_sha256_for_hdf5_dataset(path, dataset_name=name)
        if content_sha256_recorded is None:
            warnings.append("dataset contract records no content_sha256; payload integrity not verified")
        elif content_sha256_observed != content_sha256_recorded:
            errors.append(
                f"content_sha256 mismatch: recorded={content_sha256_recorded} "
                f"recomputed={content_sha256_observed} (payload modified after stamping)"
            )
        checked.append("content_sha256")

    np.isfinite(data)
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
    if (
        contract is not None
        and altitude_km.size
        and contract.altitude_min_km is not None
        and contract.altitude_max_km is not None
    ):
        lo = float(contract.altitude_min_km)
        hi = float(contract.altitude_max_km)
        envelope_tol_km = 1e-3
        outside = (altitude_km < lo - envelope_tol_km) | (altitude_km > hi + envelope_tol_km)
        if np.any(outside):
            msg = f"{int(outside.sum())} checked samples are outside contract altitude envelope [{lo}, {hi}] km"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
    checked.append("altitude_envelope")

    residual_potential_max_abs_error: float | None = None
    residual_accel_max_abs_error: float | None = None
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
        derived_u_atol, derived_a_atol = _recompute_tolerances(
            data[:, 0:3], observed_u, observed_a, stored_dtype,
            contract.degree_max if contract is not None else None,
        )
        resolved_potential_atol = float(potential_atol) if potential_atol is not None else derived_u_atol
        resolved_accel_atol = float(accel_atol) if accel_atol is not None else derived_a_atol
        if residual_potential_max_abs_error > resolved_potential_atol:
            errors.append(
                "residual potential label mismatch: "
                f"max_abs_error={residual_potential_max_abs_error:.6e} (atol={resolved_potential_atol:.6e})"
            )
        if residual_accel_max_abs_error > resolved_accel_atol:
            errors.append(
                "residual acceleration label mismatch: "
                f"max_abs_error={residual_accel_max_abs_error:.6e} (atol={resolved_accel_atol:.6e})"
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
        "content_sha256_recorded": content_sha256_recorded,
        "content_sha256_observed": content_sha256_observed,
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
