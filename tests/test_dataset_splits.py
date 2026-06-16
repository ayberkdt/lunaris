from __future__ import annotations

import json

import numpy as np
import pytest

from dataset_pipeline_test_utils import make_toy_dataset_contract
from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.data.splits import (
    build_split_manifest,
    make_altitude_stratified_split,
    make_ood_altitude_split,
    make_seeded_random_split,
    make_spatial_block_split,
    make_spatial_plus_altitude_split,
    split_dataset_indices,
    write_split_manifest,
)


def test_seeded_random_split_is_reproducible_and_disjoint():
    first = split_dataset_indices(
        n_rows=40,
        split_policy="seeded_random",
        split_seed=123,
        val_fraction=0.25,
        test_fraction=0.10,
    )
    second = split_dataset_indices(
        n_rows=40,
        split_policy="seeded_random",
        split_seed=123,
        val_fraction=0.25,
        test_fraction=0.10,
    )

    assert np.array_equal(first["train"], second["train"])
    assert len(first["train"]) == 26
    assert len(first["val"]) == 10
    assert len(first["test"]) == 4
    assert set(first["train"]).isdisjoint(set(first["val"]))
    assert set(first["train"]).isdisjoint(set(first["test"]))


def test_altitude_stratified_split_and_manifest(tmp_path):
    altitude = np.linspace(100.0, 500.0, 40)
    splits = split_dataset_indices(
        n_rows=40,
        split_policy="altitude_stratified",
        split_seed=5,
        val_fraction=0.25,
        altitude_km=altitude,
    )
    manifest = build_split_manifest(
        dataset_contract=make_toy_dataset_contract(n=40),
        splits=splits,
        split_policy="altitude_stratified",
        split_seed=5,
        altitude_km=altitude,
    )

    assert manifest["split_policy"] == "altitude_stratified"
    assert manifest["train_count"] + manifest["val_count"] == 40
    assert "train" in manifest["index_hashes"]
    out = write_split_manifest(tmp_path / "split_manifest.json", manifest)
    assert json.loads(out.read_text(encoding="utf-8"))["split_seed"] == 5


def test_unknown_split_policy_is_explicit():
    with pytest.raises(ValueError):
        split_dataset_indices(
            n_rows=20,
            split_policy="totally_made_up",
            split_seed=0,
            val_fraction=0.2,
        )


def test_ood_low_altitude_requires_altitude():
    with pytest.raises(ValueError):
        split_dataset_indices(
            n_rows=20,
            split_policy="ood_low_altitude",
            split_seed=0,
            val_fraction=0.2,
        )


# ---------------------------------------------------------------------------
# Low-level split makers: input-validation (ValueError) branches.
# ---------------------------------------------------------------------------

def test_split_counts_rejects_no_training_samples():
    # A single row with a validation fraction forces n_val=1, leaving no training.
    with pytest.raises(ValueError, match="no training"):
        make_seeded_random_split(1, val_fraction=0.5, seed=0)


def test_altitude_stratified_rejects_empty_altitude():
    with pytest.raises(ValueError, match="altitude array is empty"):
        make_altitude_stratified_split(
            np.asarray([], dtype=np.float64), val_fraction=0.2, seed=0
        )


def test_spatial_block_rejects_empty_positions():
    with pytest.raises(ValueError, match="non-empty position"):
        make_spatial_block_split(np.zeros((0, 3)), val_block_fraction=0.2, seed=0)


def test_spatial_plus_altitude_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="matching xyz/altitude"):
        make_spatial_plus_altitude_split(
            np.zeros((5, 3)), np.zeros(3), val_block_fraction=0.2, seed=0
        )


def test_ood_altitude_validation_branches():
    alt = np.linspace(100.0, 300.0, 50)
    with pytest.raises(ValueError, match="non-empty"):
        make_ood_altitude_split(np.asarray([]), side="low", seed=0)
    with pytest.raises(ValueError, match="finite"):
        make_ood_altitude_split(np.full(5, np.nan), side="low", seed=0)
    with pytest.raises(ValueError, match="low.*high|side"):
        make_ood_altitude_split(alt, side="sideways", seed=0)


def test_split_dataset_indices_spatial_block_requires_xyz():
    with pytest.raises(ValueError, match="spatial_block split requires xyz"):
        split_dataset_indices(
            n_rows=20, split_policy="spatial_block", split_seed=0, val_fraction=0.2
        )


def test_split_dataset_indices_altitude_stratified_requires_altitude():
    with pytest.raises(ValueError, match="altitude_stratified split requires"):
        split_dataset_indices(
            n_rows=20, split_policy="altitude_stratified", split_seed=0, val_fraction=0.2
        )
