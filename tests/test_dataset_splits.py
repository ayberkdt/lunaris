from __future__ import annotations

import json

import numpy as np
import pytest

from lunaris.surrogate.st_lrps.data.splits import (
    build_split_manifest,
    split_dataset_indices,
    write_split_manifest,
)

from dataset_pipeline_test_utils import make_toy_dataset_contract


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


def test_unimplemented_ood_split_policy_is_explicit():
    with pytest.raises(NotImplementedError):
        split_dataset_indices(
            n_rows=20,
            split_policy="ood_low_altitude",
            split_seed=0,
            val_fraction=0.2,
        )
