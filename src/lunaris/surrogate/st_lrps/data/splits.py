# -*- coding: utf-8 -*-
"""Explicit split policies and split-manifest writing for ST-LRPS datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

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


def build_split_manifest(
    *,
    dataset_contract: DatasetContract | Mapping[str, Any],
    splits: Mapping[str, np.ndarray],
    split_policy: str,
    split_seed: int,
    altitude_km: Optional[np.ndarray] = None,
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
        "altitude_range_per_split": _altitude_ranges(splits, altitude_km),
        "created_at_utc": utc_now_iso(),
    }
    return manifest


def write_split_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return out


def split_dataset_indices(
    *,
    n_rows: int,
    split_policy: str,
    split_seed: int,
    val_fraction: float,
    test_fraction: float = 0.0,
    altitude_km: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    policy = str(split_policy or "seeded_random").strip().lower()
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
    if policy in {"spatial_block", "ood_low_altitude", "ood_high_altitude"}:
        raise NotImplementedError(f"split_policy={policy!r} metadata is supported but automatic splitting is not implemented")
    raise ValueError(f"unknown split_policy={split_policy!r}")


def _altitude_ranges(splits: Mapping[str, np.ndarray], altitude_km: Optional[np.ndarray]) -> dict[str, dict[str, float | None]]:
    if altitude_km is None:
        return {}
    altitude = np.asarray(altitude_km, dtype=np.float64)
    out: dict[str, dict[str, float | None]] = {}
    for name, idx in splits.items():
        indices = np.asarray(idx, dtype=np.int64)
        vals = altitude[indices] if indices.size else np.asarray([], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[name] = {
            "min_km": float(np.min(vals)) if vals.size else None,
            "max_km": float(np.max(vals)) if vals.size else None,
        }
    return out


__all__ = [
    "build_split_manifest",
    "make_altitude_stratified_split",
    "make_seeded_random_split",
    "split_dataset_indices",
    "write_split_manifest",
]
