from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from lunaris.surrogate.st_lrps.data.datasets import (
    DatasetMeta,
    read_dataset_contract_from_h5,
    validate_dataset_contract,
)
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI


def _write_h5(path: Path, **attrs) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((4, 7), dtype=np.float32))
        base = {
            "unit_system": "si",
            "central_body": "moon",
            "mu_si": MU_MOON_SI,
            "r_ref_m": R_MOON_SI,
            "target_mode": "residual",
            "degree_min": 20,
            "degree_max": 200,
            "alt_min_km": 100.0,
            "alt_max_km": 1000.0,
            "a_sign_convention": "+1",
            "derivative_convention_version": "dP_dphi_corrected_v1",
            "columns": "[x,y,z,dU,dax,day,daz]",
        }
        base.update(attrs)
        for key, value in base.items():
            if value is not None:
                handle.attrs[key] = value
    return path


def test_synthetic_hdf5_metadata_is_read_correctly(tmp_path):
    path = _write_h5(tmp_path / "data.h5")
    contract = read_dataset_contract_from_h5(path)
    assert contract["target_mode"] == "residual"
    assert contract["degree_min"] == 20
    assert contract["degree_max"] == 200
    assert contract["n_samples"] == 4


def test_missing_degree_metadata_fails_in_strict_mode(tmp_path):
    path = _write_h5(tmp_path / "data.h5", degree_min=None, degree_max=None)
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="degree_min"):
        validate_dataset_contract(meta, data_path=path)


def test_legacy_inference_requires_explicit_allow_flag(tmp_path):
    path = _write_h5(tmp_path / "data.h5", target_mode=None)
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="target_mode"):
        validate_dataset_contract(meta, data_path=path)
    contract = validate_dataset_contract(meta, data_path=path, allow_legacy_target_mode_inference=True)
    assert contract["degree_min"] == 20


def test_derivative_convention_mismatch_rejected_unless_allowed(tmp_path):
    path = _write_h5(tmp_path / "data.h5", derivative_convention_version="legacy")
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="derivative_convention"):
        validate_dataset_contract(meta, data_path=path)
    assert validate_dataset_contract(meta, data_path=path, allow_legacy_derivative_convention=True)


def test_altitude_bounds_are_validated(tmp_path):
    path = _write_h5(tmp_path / "data.h5", alt_min_km=1000.0, alt_max_km=100.0)
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="altitude"):
        validate_dataset_contract(meta, data_path=path)


def test_units_are_validated(tmp_path):
    path = _write_h5(tmp_path / "data.h5", unit_system="mystery")
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="unit_system"):
        validate_dataset_contract(meta, data_path=path)
