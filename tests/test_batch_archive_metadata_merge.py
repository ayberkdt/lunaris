"""The batch archive metadata is assembled from several provenance blocks that
may legitimately repeat a key (ST-LRPS artifact block vs backend plan). They
must merge with a defined precedence instead of crashing the writer with
``got multiple values for keyword argument`` after a completed propagation."""

from __future__ import annotations

import logging

import pytest

from lunaris.batch.engine import merge_archive_metadata


def test_duplicate_keys_merge_with_later_block_winning():
    prov = {"kernel_module": "k", "runtime_model_kind": "potential_autograd"}
    st = {"gravity_backend": "st_lrps", "runtime_model_kind": "potential_autograd"}
    plan = {"runtime_model_kind": "potential_autograd", "dtype": "float64"}
    merged = merge_archive_metadata(prov, st, plan)
    assert merged["runtime_model_kind"] == "potential_autograd"
    assert merged["gravity_backend"] == "st_lrps"
    assert merged["dtype"] == "float64"
    assert merged["kernel_module"] == "k"


def test_conflicting_values_are_logged_not_silent(caplog):
    caplog.set_level(logging.WARNING)
    merged = merge_archive_metadata({"a": 1}, {"a": 2})
    assert merged["a"] == 2
    assert "a" in caplog.text and "differ" in caplog.text


def test_result_is_usable_as_kwargs():
    def sink(**kw):
        return kw
    assert sink(**merge_archive_metadata({"x": 1}, {"x": 1, "y": 2})) == {"x": 1, "y": 2}


def test_empty_blocks_are_fine():
    assert merge_archive_metadata({}, {}, {}) == {}
    with pytest.raises(TypeError):
        merge_archive_metadata({"x": 1}, None)  # type: ignore[arg-type]
