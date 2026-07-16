"""ST-LRPS validation suite beyond random interpolation.

Random point-cloud validation only measures *interpolation* inside the training
cloud. A defensible paper needs more axes of evidence, kept clearly separated:

* **interpolation** - random / altitude-stratified splits,
* **spatial generalization** - Moon-fixed lon/lat block holdout,
* **altitude extrapolation** - OOD low / high altitude bands,
* **trajectory** - orbit-level propagation error (optional hook).

This module computes the field-level metrics on each split and writes a report
that labels which kind of generalization each number represents, so a random
interpolation RMSE can never be mislabelled as "generalization".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.data.splits import (
    hash_indices,
    radius_lat_lon_deg,
    split_dataset_indices,
)
from lunaris.surrogate.st_lrps.evaluation.cli import _accel_error_radial_cross_components

logger = logging.getLogger(__name__)


# Split policy -> the kind of generalization it actually measures. Keeping this
# explicit is the whole point of the suite.
SPLIT_KIND = {
    "seeded_random": "interpolation",
    "altitude_stratified": "interpolation",
    "spatial_block": "spatial_generalization",
    "ood_low_altitude": "altitude_extrapolation",
    "ood_high_altitude": "altitude_extrapolation",
    "spatial_plus_altitude_stratified": "spatial_generalization",
}

DEFAULT_FIELD_POLICIES = (
    "seeded_random",
    "altitude_stratified",
    "spatial_block",
    "ood_low_altitude",
    "ood_high_altitude",
)


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else float("nan")


def _binned_error(
    coordinate: np.ndarray,
    err_norm: np.ndarray,
    *,
    edges: np.ndarray,
    label: str,
) -> list[dict[str, Any]]:
    coordinate = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    err_norm = np.asarray(err_norm, dtype=np.float64).reshape(-1)
    out: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        last = i == len(edges) - 2
        mask = (coordinate >= lo) & (coordinate <= hi if last else coordinate < hi)
        seg = err_norm[mask]
        out.append(
            {
                f"{label}_min": lo,
                f"{label}_max": hi,
                "count": int(seg.size),
                "accel_rmse": _rms(seg) if seg.size else None,
            }
        )
    return out


def compute_field_metrics(
    xyz: np.ndarray,
    u_true: np.ndarray,
    a_true: np.ndarray,
    u_pred: np.ndarray,
    a_pred: np.ndarray,
    *,
    r_ref_m: float = R_MOON_SI,
    n_altitude_bins: int = 8,
    n_lat_bins: int = 6,
    n_lon_bins: int = 6,
) -> dict[str, Any]:
    """Field-level residual error metrics for one split.

    All inputs are residual quantities in SI (the model predicts residual
    potential/acceleration in the Moon-fixed frame). Returns the metrics the
    paper needs: potential/accel RMSE, relative & angular accel error, radial /
    cross-radial RMS, altitude/lat/lon-binned error, and worst-tail statistics.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    u_true = np.asarray(u_true, dtype=np.float64).reshape(-1)
    u_pred = np.asarray(u_pred, dtype=np.float64).reshape(-1)
    a_true = np.asarray(a_true, dtype=np.float64).reshape(-1, 3)
    a_pred = np.asarray(a_pred, dtype=np.float64).reshape(-1, 3)
    n = xyz.shape[0]
    if n == 0:
        return {"count": 0}

    err_vec = a_pred - a_true
    err_norm = np.linalg.norm(err_vec, axis=1)
    a_true_norm = np.linalg.norm(a_true, axis=1)

    # Relative acceleration error (%) over points with a meaningful true signal.
    valid_rel = a_true_norm > 1e-15
    rel_pct = np.full(n, np.nan)
    rel_pct[valid_rel] = 100.0 * err_norm[valid_rel] / a_true_norm[valid_rel]

    # Angular error (deg) between predicted and true acceleration directions.
    a_pred_norm = np.linalg.norm(a_pred, axis=1)
    valid_ang = (a_true_norm > 1e-15) & (a_pred_norm > 1e-15)
    cos = np.full(n, np.nan)
    cos[valid_ang] = np.clip(
        np.sum(a_pred[valid_ang] * a_true[valid_ang], axis=1) / (a_pred_norm[valid_ang] * a_true_norm[valid_ang]),
        -1.0,
        1.0,
    )
    angular_deg = np.degrees(np.arccos(cos[valid_ang])) if np.any(valid_ang) else np.asarray([])

    radial_err, cross_err, _t, _nrm = _accel_error_radial_cross_components(err_vec, xyz)

    radius, lat, lon = radius_lat_lon_deg(xyz)
    altitude_km = (radius - float(r_ref_m)) / 1000.0

    # Non-finite predictions (counted before any percentile reduction).
    non_finite_rows = ~np.isfinite(u_pred) | np.any(~np.isfinite(a_pred), axis=1)
    non_finite_count = int(np.sum(non_finite_rows))

    finite_err = err_norm[np.isfinite(err_norm)]
    sorted_err = np.sort(finite_err)
    worst_1 = sorted_err[max(0, int(np.ceil(0.99 * sorted_err.size)) - 1) :] if sorted_err.size else sorted_err
    worst_5 = sorted_err[max(0, int(np.ceil(0.95 * sorted_err.size)) - 1) :] if sorted_err.size else sorted_err

    return {
        "count": int(n),
        "residual_potential_rmse_m2_s2": _rms(u_pred - u_true),
        "residual_accel_rmse_m_s2": _rms(err_norm),
        "residual_accel_mae_m_s2": float(np.mean(finite_err)) if finite_err.size else None,
        "relative_accel_error_pct_median": float(np.nanmedian(rel_pct)) if np.any(valid_rel) else None,
        "relative_accel_error_pct_mean": float(np.nanmean(rel_pct)) if np.any(valid_rel) else None,
        "angular_accel_error_deg_median": float(np.median(angular_deg)) if angular_deg.size else None,
        "angular_accel_error_deg_mean": float(np.mean(angular_deg)) if angular_deg.size else None,
        "radial_error_rms_m_s2": _rms(radial_err),
        "cross_radial_error_rms_m_s2": _rms(cross_err),
        "accel_error_p50_m_s2": float(np.percentile(finite_err, 50)) if finite_err.size else None,
        "accel_error_p90_m_s2": float(np.percentile(finite_err, 90)) if finite_err.size else None,
        "accel_error_p95_m_s2": float(np.percentile(finite_err, 95)) if finite_err.size else None,
        "accel_error_p99_m_s2": float(np.percentile(finite_err, 99)) if finite_err.size else None,
        "accel_error_worst_1pct_mean_m_s2": float(np.mean(worst_1)) if worst_1.size else None,
        "accel_error_worst_5pct_mean_m_s2": float(np.mean(worst_5)) if worst_5.size else None,
        "non_finite_prediction_count": non_finite_count,
        "radius_domain_warning_count": 0,
        "altitude_binned_error": _binned_error(
            altitude_km,
            err_norm,
            edges=np.linspace(float(altitude_km.min()), float(altitude_km.max()), max(2, n_altitude_bins + 1)),
            label="altitude_km",
        ),
        "latitude_binned_error": _binned_error(
            lat, err_norm, edges=np.linspace(-90.0, 90.0, max(2, n_lat_bins + 1)), label="latitude_deg"
        ),
        "longitude_binned_error": _binned_error(
            lon, err_norm, edges=np.linspace(-180.0, 180.0, max(2, n_lon_bins + 1)), label="longitude_deg"
        ),
    }


def _dataset_content_sha(dataset_path: str | Path | None, dataset_name: str = "data") -> str | None:
    """Best-effort content hash of the evaluation dataset, or ``None``.

    Mirrors the hash the training pipeline stamps into the split manifest
    (``content_sha256_for_hdf5_dataset``), so the two can be compared to confirm
    the evaluation dataset *is* the one the model trained on. Returns ``None`` on
    any failure (missing file, unknown dataset name) so the caller treats dataset
    identity as merely unknown rather than asserting a false mismatch.
    """
    if dataset_path is None:
        return None
    try:
        from lunaris.surrogate.st_lrps.data.dataset_contract import content_sha256_for_hdf5_dataset
        from lunaris.surrogate.st_lrps.data.datasets import discover_dataset_name

        path = Path(dataset_path)
        name = dataset_name
        try:
            import h5py

            with h5py.File(path, "r") as f:
                if name not in f:
                    name = discover_dataset_name(path, dataset_name)
        except Exception:
            pass
        return content_sha256_for_hdf5_dataset(path, dataset_name=name)
    except Exception:
        return None


def _resolve_training_heldout_guard(
    model_dir: str | Path,
    n_rows: int,
    xyz: np.ndarray,
    altitude_km: np.ndarray,
    *,
    dataset_path: str | Path | None = None,
    dataset_name: str = "data",
) -> dict[str, Any]:
    """Establish (and verify) the model's TRAINING train-set for leakage filtering.

    Random / altitude-stratified ("interpolation") validation is only honest if
    the evaluated rows were never seen during training. The training run records
    its split in ``provenance/split_manifest.json`` and the *exact* split indices
    in ``provenance/split_indices.npz``. This helper establishes the training
    train-index set with two layers of defence against silent error:

    1. **Dataset identity** — the evaluation dataset's content hash is compared
       to the manifest's ``dataset_content_sha256``. A mismatch means the user
       pointed evaluation at a different (or modified) dataset, where the stored
       indices reference different physical rows; that is reported ``unverified``
       and never filtered, because index values alone cannot detect it.
    2. **Index provenance** — the persisted ``split_indices.npz`` is loaded and
       hash-checked against the manifest (deterministic, exact, robust to any
       split policy and fraction rounding). Only when the sidecar is absent (old
       runs) is the split re-derived from policy/seed and hash-verified.

    The returned ``status`` drives the caller:

    * ``"verified"`` — the training train-set is known and hash-verified;
      ``train_index_array`` lets evaluation exclude every training row regardless
      of the evaluation re-split seed. ``method`` is ``"persisted_indices"`` or
      ``"reconstructed"``; ``dataset_identity`` is ``"verified"``/``"unknown"``.
    * ``"independent_files"`` — training used dedicated train/val files; the
      single-dataset re-split leakage check does not apply.
    * ``"manifest_missing"`` / ``"unverified"`` — the held-out set could not be
      established (no manifest, dataset mismatch, row-count drift, or hash
      disagreement). The caller must NOT silently claim clean metrics.
    """
    try:
        from lunaris.surrogate.st_lrps.artifacts.manager import (
            make_run_layout,
            resolve_run_dir,
        )
        from lunaris.surrogate.st_lrps.data.splits import (
            SPLIT_INDICES_FILENAME,
            load_verified_train_indices,
        )

        run_dir = resolve_run_dir(model_dir)
        provenance_dir = make_run_layout(run_dir).provenance_dir
        manifest_path = provenance_dir / "split_manifest.json"
        if not manifest_path.exists():
            return {
                "status": "manifest_missing",
                "reason": f"no split_manifest.json under {manifest_path.parent}",
            }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - filesystem/parse edge
        return {"status": "manifest_missing", "reason": f"could not read split manifest: {exc}"}

    policy = str(manifest.get("split_policy", "")).strip().lower()
    if policy in ("independent_files", "independent"):
        return {
            "status": "independent_files",
            "training_split_policy": policy,
            "reason": "training used independent train/val files; single-dataset re-split leakage check is not applicable",
        }

    seed = manifest.get("split_seed")
    expected_hash = (manifest.get("index_hashes") or {}).get("train")
    if seed is None or not expected_hash:
        return {
            "status": "unverified",
            "training_split_policy": policy,
            "reason": "split manifest is missing split_seed or the train index hash",
        }

    n_total = sum(
        int(manifest.get(key, 0) or 0)
        for key in ("train_count", "val_count", "test_count", "ood_count")
    )
    if n_total and n_total != int(n_rows):
        return {
            "status": "unverified",
            "training_split_policy": policy,
            "reason": (
                f"evaluation dataset has {int(n_rows)} rows but the training split covered "
                f"{n_total}; the dataset differs from the one the model trained on"
            ),
        }

    # Dataset identity guard: a same-row-count but different/modified dataset
    # would make the stored indices point at the wrong physical rows. Index
    # values alone cannot catch this, so compare dataset content hashes.
    manifest_content_sha = manifest.get("dataset_content_sha256")
    dataset_identity = "unknown"
    if manifest_content_sha:
        actual_sha = _dataset_content_sha(dataset_path, dataset_name)
        if actual_sha is None:
            dataset_identity = "unknown"
        elif str(actual_sha) != str(manifest_content_sha):
            return {
                "status": "unverified",
                "training_split_policy": policy,
                "dataset_identity": "mismatch",
                "reason": (
                    "evaluation dataset content hash does not match the training dataset "
                    f"(eval={str(actual_sha)[:12]}…, train={str(manifest_content_sha)[:12]}…); "
                    "the model was trained on a different dataset, so held-out filtering "
                    "would reference the wrong rows"
                ),
            }
        else:
            dataset_identity = "verified"

    # Preferred path: load the EXACT persisted training indices, hash-verified.
    persisted = load_verified_train_indices(
        provenance_dir / SPLIT_INDICES_FILENAME, expected_train_hash=str(expected_hash)
    )
    if persisted is not None:
        return {
            "status": "verified",
            "method": "persisted_indices",
            "dataset_identity": dataset_identity,
            "training_split_policy": policy,
            "training_split_seed": int(seed),
            "train_hash_expected": expected_hash,
            "train_count": int(persisted.size),
            "train_index_array": persisted,
            "reason": (
                "training indices loaded from split_indices.npz and hash-verified; "
                "evaluation rows exclude all training rows"
            ),
        }

    # Fallback for older runs without the indices sidecar: re-derive the split
    # from policy/seed (subject to fraction-rounding, hence hash-verified).
    val_frac = (int(manifest.get("val_count", 0) or 0) / n_rows) if n_rows else 0.0
    test_frac = (int(manifest.get("test_count", 0) or 0) / n_rows) if n_rows else 0.0

    opts: dict[str, Any] = {}
    spatial_bins = manifest.get("spatial_bins") or {}
    if spatial_bins:
        if spatial_bins.get("lon_bins") is not None:
            opts["spatial_lon_bins"] = int(spatial_bins["lon_bins"])
        if spatial_bins.get("lat_bins") is not None:
            opts["spatial_lat_bins"] = int(spatial_bins["lat_bins"])
        if spatial_bins.get("altitude_bins") is not None:
            opts["spatial_altitude_bins"] = int(spatial_bins["altitude_bins"])
        opts["spatial_val_block_fraction"] = val_frac
        opts["spatial_test_block_fraction"] = test_frac
    ood_thresholds = manifest.get("ood_thresholds") or {}
    if ood_thresholds:
        side = str(ood_thresholds.get("side", "")).strip().lower()
        thr = ood_thresholds.get("threshold_km")
        if thr is not None and side == "low":
            opts["ood_low_altitude_max_km"] = float(thr)
        elif thr is not None and side == "high":
            opts["ood_high_altitude_min_km"] = float(thr)

    try:
        recon = split_dataset_indices(
            n_rows=int(n_rows),
            split_policy=policy,
            split_seed=int(seed),
            val_fraction=float(val_frac),
            test_fraction=float(test_frac),
            altitude_km=altitude_km,
            xyz=xyz,
            options=opts or None,
        )
        train_recon = np.asarray(recon.get("train", []), dtype=np.int64)
        recon_hash = hash_indices(train_recon)
    except Exception as exc:
        return {
            "status": "unverified",
            "training_split_policy": policy,
            "training_split_seed": int(seed),
            "dataset_identity": dataset_identity,
            "reason": (
                f"could not reconstruct the training split: {exc} "
                "(no split_indices.npz sidecar to fall back on; retrain to record one)"
            ),
        }

    if recon_hash != expected_hash:
        return {
            "status": "unverified",
            "training_split_policy": policy,
            "training_split_seed": int(seed),
            "dataset_identity": dataset_identity,
            "train_hash_expected": expected_hash,
            "train_hash_reconstructed": recon_hash,
            "reason": (
                "reconstructed training train-set hash does not match split_manifest.json "
                "and no split_indices.npz sidecar is present; cannot guarantee a held-out "
                "evaluation (retrain to record the exact indices)"
            ),
        }

    return {
        "status": "verified",
        "method": "reconstructed",
        "dataset_identity": dataset_identity,
        "training_split_policy": policy,
        "training_split_seed": int(seed),
        "train_hash_expected": expected_hash,
        "train_hash_reconstructed": recon_hash,
        "train_count": int(train_recon.size),
        "train_index_array": train_recon,
        "reason": "training train-set reconstructed and hash-verified; evaluation rows exclude all training rows",
    }


def _load_residual_rows(dataset_path: Path, dataset_name: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load (xyz, u, a) residual rows in SI plus the reference radius."""
    import h5py

    from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta, discover_dataset_name

    path = Path(dataset_path)
    meta = DatasetMeta.from_h5(path)
    with h5py.File(path, "r") as f:
        name = dataset_name if dataset_name in f else discover_dataset_name(path, dataset_name)
        arr = np.asarray(f[name][:], dtype=np.float64)
    xyz, u, a = arr[:, 0:3], arr[:, 3:4], arr[:, 4:7]
    if meta.unit_system == "canonical":
        xyz, u, a = meta.convert_xyz_U_a_to_si(xyz, u, a)
    r_ref = float(meta.r_ref_m or R_MOON_SI)
    return xyz, u[:, 0], a, r_ref


def run_field_validation(
    model_dir: str | Path,
    dataset_path: str | Path,
    *,
    policies: Sequence[str] = DEFAULT_FIELD_POLICIES,
    split_seed: int = 1234,
    val_fraction: float = 0.15,
    options: Mapping[str, Any] | None = None,
    device: str = "cpu",
    enforce_heldout: bool = True,
    strict_leakage: bool = False,
) -> dict[str, Any]:
    """Evaluate a trained ST-LRPS model on each split policy's validation rows.

    For every policy the same dataset is re-split, the model is evaluated on that
    policy's validation indices, and the field metrics are computed. Results are
    grouped by the *kind* of generalization measured.

    Leakage guard
    -------------
    The evaluation re-split is independent of the training split, so a naive
    "validation" set can include rows the model trained on — silently inflating
    the interpolation numbers. When ``enforce_heldout`` is True (default) the
    model's training train-set is reconstructed from ``split_manifest.json`` and
    hash-verified; every evaluated row that the model trained on is then excluded,
    so the reported metrics are genuinely held-out *regardless of* whether
    ``split_seed`` matches the training seed. If the held-out set cannot be
    verified (no manifest, dataset drift, or hash mismatch) the report is flagged
    accordingly; ``strict_leakage=True`` turns that flag into a hard error.
    """
    from lunaris.surrogate.st_lrps.runtime.force_model import (
        load_surrogate_force_model,
    )

    dataset_path = Path(dataset_path)
    xyz, u_true, a_true, r_ref = _load_residual_rows(dataset_path)
    altitude_km = (np.linalg.norm(xyz, axis=1) - r_ref) / 1000.0
    n_rows = xyz.shape[0]

    fm = load_surrogate_force_model(model_dir, device=device, strict_domain=False)

    # Leakage guard: establish the exact rows the model trained on so they can be
    # excluded from every evaluation split (see the docstring). The dataset path
    # is threaded through so the guard can verify dataset identity (content hash)
    # and catch evaluation being pointed at a different/modified dataset.
    guard = _resolve_training_heldout_guard(
        model_dir, n_rows, xyz, altitude_km, dataset_path=dataset_path
    )
    train_excl = (
        guard.pop("train_index_array", None) if guard.get("status") == "verified" else None
    )
    if enforce_heldout and guard.get("status") not in ("verified", "independent_files"):
        msg = (
            "field validation could not verify a held-out training split "
            f"({guard.get('status')}: {guard.get('reason')}); reported interpolation "
            "metrics may include rows the model trained on."
        )
        if strict_leakage:
            raise RuntimeError(
                msg + " strict_leakage=True; refusing to produce unverified field metrics."
            )
        logger.warning(msg)

    results: dict[str, Any] = {}
    for policy in policies:
        try:
            info: dict[str, Any] = {}
            splits = split_dataset_indices(
                n_rows=n_rows,
                split_policy=policy,
                split_seed=split_seed,
                val_fraction=val_fraction,
                altitude_km=altitude_km,
                xyz=xyz,
                options=options,
                split_info_out=info,
            )
            idx = np.asarray(splits.get("val", []), dtype=np.int64)
            heldout_info: dict[str, int] | None = None
            if enforce_heldout and train_excl is not None and idx.size:
                keep = ~np.isin(idx, train_excl)
                kept = int(np.count_nonzero(keep))
                heldout_info = {
                    "val_before": int(idx.size),
                    "train_rows_excluded": int(idx.size - kept),
                    "evaluated": kept,
                }
                idx = idx[keep]
            if idx.size == 0:
                results[policy] = {
                    "kind": SPLIT_KIND.get(policy, "unknown"),
                    "leakage_guard_status": guard.get("status"),
                    "error": (
                        "empty validation split after held-out filtering"
                        if heldout_info
                        else "empty validation split"
                    ),
                }
                continue
            r_fixed = xyz[idx]
            u_pred = np.asarray(fm.predict_residual_potential_fixed(r_fixed), dtype=np.float64).reshape(-1)
            a_pred = np.asarray(fm.predict_residual_accel_fixed(r_fixed), dtype=np.float64).reshape(-1, 3)
            metrics = compute_field_metrics(
                r_fixed, u_true[idx], a_true[idx], u_pred, a_pred, r_ref_m=r_ref
            )
            metrics["kind"] = SPLIT_KIND.get(policy, "unknown")
            metrics["split_geometry"] = info
            metrics["radius_domain_warning_count"] = _count_domain_warnings(fm, r_fixed, r_ref)
            metrics["leakage_guard_status"] = guard.get("status")
            if heldout_info is not None:
                metrics["heldout_filtering"] = heldout_info
            results[policy] = metrics
        except Exception as exc:  # keep going so one bad policy does not sink the report
            logger.warning("field validation policy %s failed: %s", policy, exc)
            results[policy] = {"kind": SPLIT_KIND.get(policy, "unknown"), "error": str(exc)}

    runtime_kind = str(getattr(fm, "runtime_model_kind", "potential_autograd"))
    # Only the conservative potential_autograd surrogate is loadable on main;
    # its acceleration is the autograd gradient of a scalar potential, so the
    # finite-difference curl check is analytically zero and no longer computed.
    conservativeness = None

    return {
        "schema_version": 1,
        "model_dir": str(Path(model_dir)),
        "dataset_path": str(dataset_path),
        "split_seed": int(split_seed),
        "val_fraction": float(val_fraction),
        "n_rows": int(n_rows),
        "runtime_model_kind": runtime_kind,
        "enforce_heldout": bool(enforce_heldout),
        "leakage_guard": guard,
        "conservativeness": conservativeness,
        "field_validation": results,
        "orbit_validation": None,  # populated by run_orbit_validation when requested
    }


def _count_domain_warnings(fm: Any, r_fixed: np.ndarray, r_ref: float) -> int:
    """Count validation points outside the model's trained altitude / scaler radius."""
    r_norm = np.linalg.norm(np.asarray(r_fixed, dtype=np.float64), axis=1)
    alt_km = (r_norm - float(getattr(fm, "r_ref_m", r_ref))) / 1000.0
    warn = np.zeros(r_norm.shape[0], dtype=bool)
    alt_min = getattr(fm, "_train_alt_min_km", None)
    alt_max = getattr(fm, "_train_alt_max_km", None)
    if alt_min is not None and alt_max is not None:
        warn |= (alt_km < float(alt_min) - 1.0) | (alt_km > float(alt_max) + 1.0)
    try:
        x_scale = float(fm.scaler.x.scale)
        if x_scale > 0:
            warn |= (r_norm / max(x_scale, 1.0)) > 1.05
    except Exception:
        pass
    return int(np.sum(warn))


# Scalar field metrics emitted to field_validation_metrics.csv (in order).
_FIELD_METRIC_COLUMNS = (
    "residual_potential_rmse_m2_s2",
    "residual_accel_rmse_m_s2",
    "residual_accel_mae_m_s2",
    "relative_accel_error_pct_median",
    "angular_accel_error_deg_median",
    "radial_error_rms_m_s2",
    "cross_radial_error_rms_m_s2",
    "accel_error_p50_m_s2",
    "accel_error_p90_m_s2",
    "accel_error_p95_m_s2",
    "accel_error_p99_m_s2",
    "accel_error_worst_1pct_mean_m_s2",
    "accel_error_worst_5pct_mean_m_s2",
    "radius_domain_warning_count",
    "non_finite_prediction_count",
)


def write_field_validation_csvs(report: Mapping[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Write the paper field-validation tables (Task 4 of the evidence pipeline).

    Produces ``field_validation_metrics.csv`` (one row per split policy),
    ``field_validation_by_altitude.csv`` and ``field_validation_by_lat_lon.csv``
    (binned error), and ``field_validation_summary.md``. Random/altitude splits
    are labelled *interpolation*; spatial/OOD are labelled accordingly.
    """
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    field = report.get("field_validation", {}) or {}
    paths: dict[str, Path] = {}

    # 1. Per-policy scalar metrics.
    metrics_path = out / "field_validation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "kind", "count", *_FIELD_METRIC_COLUMNS, "error"])
        for policy, m in field.items():
            if "error" in m:
                writer.writerow([policy, m.get("kind", ""), 0, *([""] * len(_FIELD_METRIC_COLUMNS)), m["error"]])
            else:
                writer.writerow(
                    [policy, m.get("kind", ""), m.get("count", 0)]
                    + [m.get(col) for col in _FIELD_METRIC_COLUMNS]
                    + [""]
                )
    paths["metrics"] = metrics_path

    # 2. Altitude-binned error.
    alt_path = out / "field_validation_by_altitude.csv"
    with alt_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "kind", "altitude_km_min", "altitude_km_max", "count", "accel_rmse"])
        for policy, m in field.items():
            for row in m.get("altitude_binned_error", []) or []:
                writer.writerow([
                    policy, m.get("kind", ""), row.get("altitude_km_min"), row.get("altitude_km_max"),
                    row.get("count"), row.get("accel_rmse"),
                ])
    paths["by_altitude"] = alt_path

    # 3. Latitude/longitude-binned error (one file, a 'dimension' column).
    latlon_path = out / "field_validation_by_lat_lon.csv"
    with latlon_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "kind", "dimension", "bin_min", "bin_max", "count", "accel_rmse"])
        for policy, m in field.items():
            for row in m.get("latitude_binned_error", []) or []:
                writer.writerow([policy, m.get("kind", ""), "latitude_deg",
                                 row.get("latitude_deg_min"), row.get("latitude_deg_max"),
                                 row.get("count"), row.get("accel_rmse")])
            for row in m.get("longitude_binned_error", []) or []:
                writer.writerow([policy, m.get("kind", ""), "longitude_deg",
                                 row.get("longitude_deg_min"), row.get("longitude_deg_max"),
                                 row.get("count"), row.get("accel_rmse")])
    paths["by_lat_lon"] = latlon_path

    # 4. Markdown summary (reuses the kind-grouped report writer).
    paths.update(write_validation_report(report, out))
    summary_path = out / "field_validation_summary.md"
    if (out / "validation_suite.md").exists():
        summary_path.write_text((out / "validation_suite.md").read_text(encoding="utf-8"), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def attach_orbit_validation(report: dict[str, Any], orbit_metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Attach orbit-level (trajectory) metrics under a clearly separate key.

    Orbit propagation validation (final/RMS/max position error, radial/along/
    cross-track, velocity error, domain-warning count, runtime) is produced by
    the orbit benchmark harness; this keeps it distinct from the field metrics so
    the two are never conflated.
    """
    report["orbit_validation"] = dict(orbit_metrics) if orbit_metrics else None
    return report


def write_validation_report(report: Mapping[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Write the suite report as JSON + a clearly-sectioned Markdown summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "validation_suite.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    field = report.get("field_validation", {}) or {}
    by_kind: dict[str, list[str]] = {}
    for policy, metrics in field.items():
        by_kind.setdefault(str(metrics.get("kind", "unknown")), []).append(policy)

    guard = report.get("leakage_guard") or {}
    guard_status = str(guard.get("status", "unknown"))
    guard_ok = guard_status in ("verified", "independent_files")
    lines = [
        "# ST-LRPS Validation Suite",
        "",
        f"- Model: `{report.get('model_dir')}`",
        f"- Dataset: `{report.get('dataset_path')}`",
        f"- Rows: {report.get('n_rows')}  |  split_seed: {report.get('split_seed')}",
        f"- Runtime kind: `{report.get('runtime_model_kind', 'potential_autograd')}`",
        f"- Held-out leakage guard: **{guard_status}**"
        + (
            f" (method: {guard.get('method')}, dataset identity: {guard.get('dataset_identity', 'n/a')})"
            if guard_status == "verified"
            else ""
        )
        + ("" if guard_ok else f" — {guard.get('reason', '')}"),
        "",
        "> Random / altitude splits are **interpolation**. Spatial-block is "
        "**spatial generalization**. OOD low/high are **altitude extrapolation**. "
        "Orbit is **trajectory propagation**. These are NOT interchangeable.",
        "",
    ]
    if not guard_ok:
        lines += [
            "> ⚠ **Leakage guard not verified.** Evaluation rows could not be proven "
            "disjoint from the training set, so the interpolation numbers below may be "
            "optimistic. Re-run with a present, matching `split_manifest.json` or pass "
            "`strict_leakage=True` to hard-fail.",
            "",
        ]
    section_titles = {
        "interpolation": "## Interpolation (random / altitude)",
        "spatial_generalization": "## Spatial generalization (block holdout)",
        "altitude_extrapolation": "## Altitude extrapolation (OOD bands)",
        "unknown": "## Other",
    }
    for kind in ("interpolation", "spatial_generalization", "altitude_extrapolation", "unknown"):
        policies = by_kind.get(kind)
        if not policies:
            continue
        lines.append(section_titles[kind])
        lines.append("")
        lines.append("| policy | accel RMSE [m/s²] | rel err [%] | angular [deg] | P99 [m/s²] | n |")
        lines.append("|---|---|---|---|---|---|")
        for policy in policies:
            m = field[policy]
            if "error" in m:
                lines.append(f"| {policy} | ERROR: {m['error']} | | | | |")
                continue
            lines.append(
                f"| {policy} | {_fmt(m.get('residual_accel_rmse_m_s2'))} | "
                f"{_fmt(m.get('relative_accel_error_pct_median'))} | "
                f"{_fmt(m.get('angular_accel_error_deg_median'))} | "
                f"{_fmt(m.get('accel_error_p99_m_s2'))} | {m.get('count')} |"
            )
        lines.append("")

    orbit = report.get("orbit_validation")
    lines.append("## Trajectory propagation (orbit-level)")
    lines.append("")
    lines.append("Not run in this report." if not orbit else f"```\n{json.dumps(orbit, indent=2, default=str)}\n```")
    lines.append("")

    md_path = out / "validation_suite.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "md": md_path}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4e}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run the ST-LRPS validation suite (field-level, multi-split).")
    ap.add_argument("--model-dir", required=True, help="Trained run directory.")
    ap.add_argument("--data", required=True, help="Residual cloud HDF5 dataset.")
    ap.add_argument("--out", required=True, help="Output directory for the report.")
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args(argv)

    report = run_field_validation(
        args.model_dir,
        args.data,
        split_seed=int(args.split_seed),
        val_fraction=float(args.val_fraction),
        device=str(args.device),
    )
    paths = write_validation_report(report, args.out)
    print(f"[validation-suite] wrote {paths['json']} and {paths['md']}", flush=True)
    return 0


__all__ = [
    "DEFAULT_FIELD_POLICIES",
    "SPLIT_KIND",
    "attach_orbit_validation",
    "compute_field_metrics",
    "run_field_validation",
    "write_field_validation_csvs",
    "write_validation_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
