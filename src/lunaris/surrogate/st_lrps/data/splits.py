"""Explicit split policies and split-manifest writing for ST-LRPS datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract, utc_now_iso


def _hash_indices(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()


def _split_counts(n_rows: int, val_fraction: float, test_fraction: float = 0.0) -> tuple[int, int, int]:
    n_total = int(n_rows)
    n_val = int(round(n_total * float(val_fraction)))
    n_test = int(round(n_total * float(test_fraction)))
    n_val = max(1 if val_fraction > 0 else 0, min(n_val, n_total - 1))
    n_test = max(0, min(n_test, n_total - n_val - 1))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("split fractions leave no training samples")
    return n_train, n_val, n_test


def make_seeded_random_split(
    n_rows: int,
    *,
    val_fraction: float,
    test_fraction: float = 0.0,
    seed: int,
) -> dict[str, np.ndarray]:
    n_train, n_val, n_test = _split_counts(n_rows, val_fraction, test_fraction)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n_rows)).astype(np.int64, copy=False)
    val = np.sort(perm[:n_val])
    test = np.sort(perm[n_val : n_val + n_test])
    train = np.sort(perm[n_val + n_test : n_val + n_test + n_train])
    return {"train": train, "val": val, "test": test, "ood": np.asarray([], dtype=np.int64)}


def make_altitude_stratified_split(
    altitude_km: np.ndarray,
    *,
    val_fraction: float,
    test_fraction: float = 0.0,
    seed: int,
    bins: int = 10,
) -> dict[str, np.ndarray]:
    altitude = np.asarray(altitude_km, dtype=np.float64).reshape(-1)
    if altitude.size == 0:
        raise ValueError("altitude array is empty")
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    edges = np.linspace(float(np.nanmin(altitude)), float(np.nanmax(altitude)), max(2, int(bins) + 1))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (altitude >= lo) & (altitude <= hi if i == len(edges) - 2 else altitude < hi)
        idx = np.nonzero(mask)[0].astype(np.int64, copy=False)
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        _, n_val, n_test = _split_counts(idx.size, val_fraction, test_fraction)
        val_parts.append(idx[:n_val])
        test_parts.append(idx[n_val : n_val + n_test])
        train_parts.append(idx[n_val + n_test :])
    return {
        "train": np.sort(np.concatenate(train_parts) if train_parts else np.asarray([], dtype=np.int64)),
        "val": np.sort(np.concatenate(val_parts) if val_parts else np.asarray([], dtype=np.int64)),
        "test": np.sort(np.concatenate(test_parts) if test_parts else np.asarray([], dtype=np.int64)),
        "ood": np.asarray([], dtype=np.int64),
    }


def radius_lat_lon_deg(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Moon-fixed Cartesian ``(N,3)`` positions to (radius, lat°, lon°).

    Latitude is geocentric in ``[-90, 90]``; longitude is in ``[-180, 180)``.
    These body-fixed angular coordinates are what ``spatial_block`` partitions on
    so train and validation never share a tight local patch.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    radius = np.linalg.norm(pts, axis=1)
    safe_r = np.where(radius > 0.0, radius, 1.0)
    lat = np.degrees(np.arcsin(np.clip(pts[:, 2] / safe_r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return radius, lat, lon


def _empty() -> np.ndarray:
    return np.asarray([], dtype=np.int64)


def make_spatial_block_split(
    xyz: np.ndarray,
    *,
    val_block_fraction: float,
    test_block_fraction: float = 0.0,
    seed: int,
    lon_bins: int = 12,
    lat_bins: int = 6,
) -> dict[str, np.ndarray]:
    """Hold out whole Moon-fixed lon/lat blocks for validation/test.

    Points are binned into a ``lon_bins × lat_bins`` grid; entire cells are
    assigned (seeded) to train/val/test so a held-out validation block shares no
    local neighbourhood with any training point. This measures *spatial*
    generalization, unlike a random point split which interpolates within the
    same cloud.
    """
    _radius, lat, lon = radius_lat_lon_deg(xyz)
    n = lat.size
    if n == 0:
        raise ValueError("spatial_block split requires a non-empty position array")
    lon_bins = max(1, int(lon_bins))
    lat_bins = max(1, int(lat_bins))
    lon_idx = np.clip(((lon + 180.0) / 360.0 * lon_bins).astype(np.int64), 0, lon_bins - 1)
    lat_idx = np.clip(((lat + 90.0) / 180.0 * lat_bins).astype(np.int64), 0, lat_bins - 1)
    cell = lat_idx * lon_bins + lon_idx
    unique_cells = np.unique(cell)
    n_cells = unique_cells.size

    rng = np.random.default_rng(int(seed))
    perm = unique_cells[rng.permutation(n_cells)]
    n_val_cells = int(round(n_cells * float(val_block_fraction)))
    n_test_cells = int(round(n_cells * float(test_block_fraction)))
    if val_block_fraction > 0:
        n_val_cells = max(1, n_val_cells)
    if test_block_fraction > 0:
        n_test_cells = max(1, n_test_cells)
    # Always keep at least one training cell.
    if n_val_cells + n_test_cells >= n_cells:
        n_test_cells = min(n_test_cells, max(0, n_cells - 1 - n_val_cells))
        n_val_cells = min(n_val_cells, max(0, n_cells - 1 - n_test_cells))

    val_cells = set(int(c) for c in perm[:n_val_cells])
    test_cells = set(int(c) for c in perm[n_val_cells : n_val_cells + n_test_cells])

    is_val = np.isin(cell, list(val_cells)) if val_cells else np.zeros(n, dtype=bool)
    is_test = np.isin(cell, list(test_cells)) if test_cells else np.zeros(n, dtype=bool)
    is_train = ~(is_val | is_test)

    return {
        "train": np.sort(np.nonzero(is_train)[0].astype(np.int64)),
        "val": np.sort(np.nonzero(is_val)[0].astype(np.int64)),
        "test": np.sort(np.nonzero(is_test)[0].astype(np.int64)),
        "ood": _empty(),
    }


def make_ood_altitude_split(
    altitude_km: np.ndarray,
    *,
    side: str,
    seed: int,
    threshold_km: float | None = None,
    holdout_fraction: float = 0.2,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
) -> tuple[dict[str, np.ndarray], float]:
    """Hold out the lowest (``side="low"``) or highest (``side="high"``) altitude band.

    The held-out band is the OOD extrapolation region: ``train`` contains the
    in-distribution side of the threshold; ``val``/``test``/``ood`` are all drawn
    from the held-out band so validation reflects altitude *extrapolation*, not
    interpolation. When ``threshold_km`` is ``None`` it is derived from
    ``holdout_fraction`` via the altitude quantile. Returns ``(splits, threshold_km)``.
    """
    altitude = np.asarray(altitude_km, dtype=np.float64).reshape(-1)
    if altitude.size == 0:
        raise ValueError("ood altitude split requires a non-empty altitude array")
    finite = altitude[np.isfinite(altitude)]
    if finite.size == 0:
        raise ValueError("ood altitude split requires finite altitudes")

    side_l = str(side).strip().lower()
    if side_l == "low":
        thr = float(np.quantile(finite, float(holdout_fraction))) if threshold_km is None else float(threshold_km)
        band_mask = altitude <= thr
    elif side_l == "high":
        thr = (
            float(np.quantile(finite, 1.0 - float(holdout_fraction)))
            if threshold_km is None
            else float(threshold_km)
        )
        band_mask = altitude >= thr
    else:
        raise ValueError(f"ood altitude side must be 'low' or 'high', got {side!r}")

    band_idx = np.nonzero(band_mask)[0].astype(np.int64)
    train_idx = np.nonzero(~band_mask)[0].astype(np.int64)
    if train_idx.size == 0 or band_idx.size == 0:
        raise ValueError(
            f"ood_{side_l}_altitude threshold {thr:.3f} km leaves an empty "
            f"train ({train_idx.size}) or holdout ({band_idx.size}) set"
        )

    rng = np.random.default_rng(int(seed))
    rng.shuffle(band_idx)
    n_band = band_idx.size
    n_val = min(int(round(n_band * float(val_fraction))), n_band)
    n_test = min(int(round(n_band * float(test_fraction))), n_band - n_val)
    val_idx = band_idx[:n_val]
    test_idx = band_idx[n_val : n_val + n_test]
    ood_idx = band_idx[n_val + n_test :]
    return (
        {
            "train": np.sort(train_idx),
            "val": np.sort(val_idx),
            "test": np.sort(test_idx),
            "ood": np.sort(ood_idx),
        },
        float(thr),
    )


def make_spatial_plus_altitude_split(
    xyz: np.ndarray,
    altitude_km: np.ndarray,
    *,
    val_block_fraction: float,
    test_block_fraction: float = 0.0,
    seed: int,
    lon_bins: int = 12,
    lat_bins: int = 6,
    altitude_bins: int = 4,
) -> dict[str, np.ndarray]:
    """Spatial-block holdout that keeps altitude ranges balanced across splits.

    Cells are stratified by mean altitude into ``altitude_bins`` strata, and a
    fraction of blocks is held out *within each altitude stratum*. This yields a
    spatial generalization split whose val/test altitude envelope still matches
    train, isolating the spatial axis from the altitude axis.
    """
    _radius, lat, lon = radius_lat_lon_deg(xyz)
    altitude = np.asarray(altitude_km, dtype=np.float64).reshape(-1)
    n = lat.size
    if n == 0 or altitude.size != n:
        raise ValueError("spatial_plus_altitude split requires matching xyz/altitude arrays")
    lon_bins = max(1, int(lon_bins))
    lat_bins = max(1, int(lat_bins))
    lon_idx = np.clip(((lon + 180.0) / 360.0 * lon_bins).astype(np.int64), 0, lon_bins - 1)
    lat_idx = np.clip(((lat + 90.0) / 180.0 * lat_bins).astype(np.int64), 0, lat_bins - 1)
    cell = lat_idx * lon_bins + lon_idx

    unique_cells, inv = np.unique(cell, return_inverse=True)
    # Mean altitude per cell -> altitude stratum per cell.
    cell_alt = np.zeros(unique_cells.size, dtype=np.float64)
    np.add.at(cell_alt, inv, altitude)
    counts = np.bincount(inv, minlength=unique_cells.size).astype(np.float64)
    cell_alt = cell_alt / np.maximum(counts, 1.0)
    edges = np.linspace(float(np.min(cell_alt)), float(np.max(cell_alt)), max(2, int(altitude_bins) + 1))

    rng = np.random.default_rng(int(seed))
    val_cells: set[int] = set()
    test_cells: set[int] = set()
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (cell_alt >= lo) & (cell_alt <= hi if b == len(edges) - 2 else cell_alt < hi)
        stratum = unique_cells[in_bin]
        if stratum.size == 0:
            continue
        stratum = stratum[rng.permutation(stratum.size)]
        n_val_cells = int(round(stratum.size * float(val_block_fraction)))
        n_test_cells = int(round(stratum.size * float(test_block_fraction)))
        if val_block_fraction > 0 and stratum.size > 1:
            n_val_cells = max(1, n_val_cells)
        n_val_cells = min(n_val_cells, max(0, stratum.size - 1))
        n_test_cells = min(n_test_cells, max(0, stratum.size - 1 - n_val_cells))
        val_cells.update(int(c) for c in stratum[:n_val_cells])
        test_cells.update(int(c) for c in stratum[n_val_cells : n_val_cells + n_test_cells])

    is_val = np.isin(cell, list(val_cells)) if val_cells else np.zeros(n, dtype=bool)
    is_test = np.isin(cell, list(test_cells)) if test_cells else np.zeros(n, dtype=bool)
    is_train = ~(is_val | is_test)
    return {
        "train": np.sort(np.nonzero(is_train)[0].astype(np.int64)),
        "val": np.sort(np.nonzero(is_val)[0].astype(np.int64)),
        "test": np.sort(np.nonzero(is_test)[0].astype(np.int64)),
        "ood": _empty(),
    }


def build_split_manifest(
    *,
    dataset_contract: DatasetContract | Mapping[str, Any],
    splits: Mapping[str, np.ndarray],
    split_policy: str,
    split_seed: int,
    altitude_km: np.ndarray | None = None,
    xyz: np.ndarray | None = None,
    spatial_bins: Mapping[str, Any] | None = None,
    ood_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = (
        dataset_contract
        if isinstance(dataset_contract, DatasetContract)
        else DatasetContract.from_dict(
            dataset_contract,
            allow_legacy_dataset_contract=True,
            allow_missing_source_gravity=True,
        )
    )
    # Latitude/longitude are derived from xyz when available; altitude may be
    # passed directly or derived from xyz + the contract reference radius.
    lat = lon = None
    if xyz is not None:
        _radius, lat, lon = radius_lat_lon_deg(xyz)
        if altitude_km is None and contract.r_ref_m:
            altitude_km = (_radius - float(contract.r_ref_m)) / 1000.0

    manifest = {
        "schema_version": 1,
        "dataset_id": contract.dataset_id,
        "dataset_content_sha256": contract.content_sha256,
        "split_policy": str(split_policy),
        "split_seed": int(split_seed),
        "train_count": int(len(splits.get("train", []))),
        "val_count": int(len(splits.get("val", []))),
        "test_count": int(len(splits.get("test", []))),
        "ood_count": int(len(splits.get("ood", []))),
        "index_hashes": {
            name: _hash_indices(np.asarray(indices, dtype=np.int64))
            for name, indices in splits.items()
        },
        "altitude_range_per_split": _range_per_split(splits, altitude_km),
        "latitude_range_per_split": _range_per_split(splits, lat),
        "longitude_range_per_split": _range_per_split(splits, lon),
        "created_at_utc": utc_now_iso(),
    }
    if spatial_bins:
        manifest["spatial_bins"] = dict(spatial_bins)
    if ood_thresholds:
        manifest["ood_thresholds"] = dict(ood_thresholds)
    return manifest


def write_split_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return out


SUPPORTED_SPLIT_POLICIES = (
    "seeded_random",
    "altitude_stratified",
    "spatial_block",
    "ood_low_altitude",
    "ood_high_altitude",
    "spatial_plus_altitude_stratified",
)


def split_dataset_indices(
    *,
    n_rows: int,
    split_policy: str,
    split_seed: int,
    val_fraction: float,
    test_fraction: float = 0.0,
    altitude_km: np.ndarray | None = None,
    xyz: np.ndarray | None = None,
    options: Mapping[str, Any] | None = None,
    split_info_out: dict | None = None,
) -> dict[str, np.ndarray]:
    """Dispatch to a split policy.

    ``options`` carries policy-specific knobs (``spatial_lon_bins``,
    ``spatial_lat_bins``, ``spatial_val_block_fraction``,
    ``spatial_test_block_fraction``, ``ood_low_altitude_max_km``,
    ``ood_high_altitude_min_km``, ``ood_holdout_fraction``,
    ``spatial_altitude_bins``). When ``split_info_out`` is provided it is
    populated with the resolved geometric metadata (bin grid, OOD threshold) so
    the caller can record it in the split manifest.
    """
    policy = str(split_policy or "seeded_random").strip().lower()
    opts = dict(options or {})
    info = split_info_out if split_info_out is not None else {}

    if policy in {"random", "seeded_random"}:
        return make_seeded_random_split(
            n_rows,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=split_seed,
        )
    if policy == "altitude_stratified":
        if altitude_km is None:
            raise ValueError("altitude_stratified split requires altitude_km")
        return make_altitude_stratified_split(
            altitude_km,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=split_seed,
        )
    if policy == "spatial_block":
        if xyz is None:
            raise ValueError("spatial_block split requires xyz positions")
        lon_bins = int(opts.get("spatial_lon_bins", 12))
        lat_bins = int(opts.get("spatial_lat_bins", 6))
        info["spatial_bins"] = {"lon_bins": lon_bins, "lat_bins": lat_bins}
        return make_spatial_block_split(
            xyz,
            val_block_fraction=float(opts.get("spatial_val_block_fraction", val_fraction)),
            test_block_fraction=float(opts.get("spatial_test_block_fraction", test_fraction)),
            seed=split_seed,
            lon_bins=lon_bins,
            lat_bins=lat_bins,
        )
    if policy in {"ood_low_altitude", "ood_high_altitude"}:
        if altitude_km is None:
            raise ValueError(f"{policy} split requires altitude_km")
        side = "low" if policy == "ood_low_altitude" else "high"
        threshold = (
            opts.get("ood_low_altitude_max_km") if side == "low" else opts.get("ood_high_altitude_min_km")
        )
        splits, resolved_thr = make_ood_altitude_split(
            altitude_km,
            side=side,
            seed=split_seed,
            threshold_km=(float(threshold) if threshold is not None else None),
            holdout_fraction=float(opts.get("ood_holdout_fraction", 0.2)),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        info["ood_thresholds"] = {"side": side, "threshold_km": resolved_thr}
        return splits
    if policy == "spatial_plus_altitude_stratified":
        if xyz is None or altitude_km is None:
            raise ValueError("spatial_plus_altitude_stratified split requires xyz and altitude_km")
        lon_bins = int(opts.get("spatial_lon_bins", 12))
        lat_bins = int(opts.get("spatial_lat_bins", 6))
        alt_bins = int(opts.get("spatial_altitude_bins", 4))
        info["spatial_bins"] = {"lon_bins": lon_bins, "lat_bins": lat_bins, "altitude_bins": alt_bins}
        return make_spatial_plus_altitude_split(
            xyz,
            altitude_km,
            val_block_fraction=float(opts.get("spatial_val_block_fraction", val_fraction)),
            test_block_fraction=float(opts.get("spatial_test_block_fraction", test_fraction)),
            seed=split_seed,
            lon_bins=lon_bins,
            lat_bins=lat_bins,
            altitude_bins=alt_bins,
        )
    raise ValueError(f"unknown split_policy={split_policy!r}; supported: {SUPPORTED_SPLIT_POLICIES}")


def _range_per_split(
    splits: Mapping[str, np.ndarray], values: np.ndarray | None
) -> dict[str, dict[str, float | None]]:
    """Per-split ``{min, max}`` of a scalar field (altitude/lat/lon), or ``{}``.

    Makes train/val/test separation auditable: e.g. an OOD-low split shows the
    validation altitude band sitting strictly below the train band.
    """
    if values is None:
        return {}
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out: dict[str, dict[str, float | None]] = {}
    for name, idx in splits.items():
        indices = np.asarray(idx, dtype=np.int64)
        vals = arr[indices] if indices.size else np.asarray([], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[name] = {
            "min": float(np.min(vals)) if vals.size else None,
            "max": float(np.max(vals)) if vals.size else None,
        }
    return out


__all__ = [
    "SUPPORTED_SPLIT_POLICIES",
    "build_split_manifest",
    "make_altitude_stratified_split",
    "make_ood_altitude_split",
    "make_seeded_random_split",
    "make_spatial_block_split",
    "make_spatial_plus_altitude_split",
    "radius_lat_lon_deg",
    "split_dataset_indices",
    "write_split_manifest",
]
