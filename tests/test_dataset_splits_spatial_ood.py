"""Task 2 — spatial-block and OOD altitude split policies."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.data.splits import (
    build_split_manifest,
    radius_lat_lon_deg,
    split_dataset_indices,
    write_split_manifest,
)

from dataset_pipeline_test_utils import make_toy_dataset_contract


def _shell_xyz(n: int, *, alt_min_km: float = 50.0, alt_max_km: float = 500.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    alt_km = rng.uniform(alt_min_km, alt_max_km, size=n)
    return dirs * (float(R_MOON_SI) + alt_km[:, None] * 1000.0)


def _altitude_km(xyz: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(xyz, axis=1) - float(R_MOON_SI)) / 1000.0


def _assert_disjoint(splits):
    seen = set()
    for name in ("train", "val", "test", "ood"):
        idx = set(int(i) for i in splits[name])
        assert seen.isdisjoint(idx), f"{name} overlaps another split"
        seen |= idx


def test_radius_lat_lon_ranges():
    xyz = _shell_xyz(2000, seed=3)
    radius, lat, lon = radius_lat_lon_deg(xyz)
    assert np.all(radius > R_MOON_SI)
    assert lat.min() >= -90.0 and lat.max() <= 90.0
    assert lon.min() >= -180.0 and lon.max() < 180.0001


def test_spatial_block_holdout_is_disjoint_and_blockwise():
    xyz = _shell_xyz(3000, seed=7)
    splits = split_dataset_indices(
        n_rows=xyz.shape[0],
        split_policy="spatial_block",
        split_seed=11,
        val_fraction=0.2,
        test_fraction=0.1,
        xyz=xyz,
        options={"spatial_lon_bins": 12, "spatial_lat_bins": 6},
    )
    _assert_disjoint(splits)
    assert splits["train"].size > 0 and splits["val"].size > 0

    # Block-wise property: validation cells must never appear in train. Recompute
    # the cell id for each point and check the train/val cell sets are disjoint.
    _r, lat, lon = radius_lat_lon_deg(xyz)
    lon_idx = np.clip(((lon + 180.0) / 360.0 * 12).astype(int), 0, 11)
    lat_idx = np.clip(((lat + 90.0) / 180.0 * 6).astype(int), 0, 5)
    cell = lat_idx * 12 + lon_idx
    train_cells = set(cell[splits["train"]].tolist())
    val_cells = set(cell[splits["val"]].tolist())
    assert train_cells.isdisjoint(val_cells)


def test_spatial_block_is_deterministic():
    xyz = _shell_xyz(1500, seed=4)
    kw = dict(
        n_rows=xyz.shape[0],
        split_policy="spatial_block",
        split_seed=99,
        val_fraction=0.25,
        xyz=xyz,
    )
    a = split_dataset_indices(**kw)
    b = split_dataset_indices(**kw)
    assert np.array_equal(a["train"], b["train"])
    assert np.array_equal(a["val"], b["val"])


def test_ood_low_altitude_holds_out_low_band():
    xyz = _shell_xyz(2000, alt_min_km=50.0, alt_max_km=500.0, seed=8)
    alt = _altitude_km(xyz)
    info = {}
    splits = split_dataset_indices(
        n_rows=xyz.shape[0],
        split_policy="ood_low_altitude",
        split_seed=5,
        val_fraction=0.2,
        altitude_km=alt,
        options={"ood_low_altitude_max_km": 150.0},
        split_info_out=info,
    )
    _assert_disjoint(splits)
    thr = info["ood_thresholds"]["threshold_km"]
    assert thr == pytest.approx(150.0)
    # Train strictly above threshold; val + ood strictly at/below threshold.
    assert np.all(alt[splits["train"]] > thr)
    if splits["val"].size:
        assert np.all(alt[splits["val"]] <= thr)
    if splits["ood"].size:
        assert np.all(alt[splits["ood"]] <= thr)


def test_ood_high_altitude_holds_out_high_band():
    xyz = _shell_xyz(2000, alt_min_km=50.0, alt_max_km=500.0, seed=9)
    alt = _altitude_km(xyz)
    splits = split_dataset_indices(
        n_rows=xyz.shape[0],
        split_policy="ood_high_altitude",
        split_seed=5,
        val_fraction=0.2,
        altitude_km=alt,
        options={"ood_high_altitude_min_km": 400.0},
    )
    _assert_disjoint(splits)
    assert np.all(alt[splits["train"]] < 400.0)
    if splits["val"].size:
        assert np.all(alt[splits["val"]] >= 400.0)


def test_ood_fraction_based_threshold():
    xyz = _shell_xyz(2000, seed=10)
    alt = _altitude_km(xyz)
    info = {}
    split_dataset_indices(
        n_rows=xyz.shape[0],
        split_policy="ood_low_altitude",
        split_seed=1,
        val_fraction=0.1,
        altitude_km=alt,
        options={"ood_holdout_fraction": 0.25},
        split_info_out=info,
    )
    # ~25% quantile threshold (not an explicit km value).
    assert info["ood_thresholds"]["threshold_km"] == pytest.approx(np.quantile(alt, 0.25), rel=1e-6)


def test_spatial_plus_altitude_keeps_altitude_balanced():
    xyz = _shell_xyz(3000, seed=12)
    alt = _altitude_km(xyz)
    splits = split_dataset_indices(
        n_rows=xyz.shape[0],
        split_policy="spatial_plus_altitude_stratified",
        split_seed=3,
        val_fraction=0.2,
        xyz=xyz,
        altitude_km=alt,
        options={"spatial_altitude_bins": 4},
    )
    _assert_disjoint(splits)
    assert splits["val"].size > 0
    # Val altitude envelope should overlap train (balanced), not be an OOD band.
    train_alt = alt[splits["train"]]
    val_alt = alt[splits["val"]]
    assert val_alt.min() < np.median(train_alt) < val_alt.max()


def test_manifest_records_geometry_and_thresholds(tmp_path):
    xyz = _shell_xyz(1200, seed=6)
    alt = _altitude_km(xyz)
    info = {}
    splits = split_dataset_indices(
        n_rows=xyz.shape[0],
        split_policy="ood_low_altitude",
        split_seed=2,
        val_fraction=0.2,
        altitude_km=alt,
        options={"ood_low_altitude_max_km": 120.0},
        split_info_out=info,
    )
    manifest = build_split_manifest(
        dataset_contract=make_toy_dataset_contract(n=xyz.shape[0]),
        splits=splits,
        split_policy="ood_low_altitude",
        split_seed=2,
        altitude_km=alt,
        xyz=xyz,
        ood_thresholds=info.get("ood_thresholds"),
    )
    assert manifest["ood_thresholds"]["threshold_km"] == pytest.approx(120.0)
    assert "latitude_range_per_split" in manifest
    assert "longitude_range_per_split" in manifest
    assert "train" in manifest["index_hashes"]
    # Train altitude band sits above the validation band -> auditable separation.
    assert manifest["altitude_range_per_split"]["train"]["min"] >= 120.0
    out = write_split_manifest(tmp_path / "manifest.json", manifest)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["ood_thresholds"]["side"] == "low"
