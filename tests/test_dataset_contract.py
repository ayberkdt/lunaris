from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from lunaris.surrogate.st_lrps.data.dataset_contract import (
    GRAVITY_LABEL_ENGINE_VERSION,
    REQUIRED_DERIVATIVE_CONVENTION,
    REQUIRED_SH_PHASE_CONVENTION,
    DatasetContract,
    DatasetContractError,
    build_contract_payload_for_generator,
    stamp_hdf5_content_hash,
)
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.data.datasets import (
    DatasetMeta,
    read_dataset_contract_from_h5,
    validate_dataset_contract,
)


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
            "spherical_harmonic_convention": "4pi_geodesy_no_condon_shortley_v1",
            "gravity_label_engine_version": "lunaris_sh_v2",
            "columns": "[x,y,z,dU,dax,day,daz]",
        }
        base.update(attrs)
        payload = {
            "schema_version": 1,
            "dataset_id": path.stem,
            "dataset_kind": "st_lrps_spatial_cloud",
            "n_samples": 4,
            "target_mode": base.get("target_mode"),
            "baseline_kind": "spherical_harmonics",
            "degree_min": base.get("degree_min"),
            "degree_max": base.get("degree_max"),
            "mu_si": base.get("mu_si"),
            "r_ref_m": base.get("r_ref_m"),
            "a_sign": 1.0,
            "altitude_min_km": base.get("alt_min_km"),
            "altitude_max_km": base.get("alt_max_km"),
            "source_gravity_model": "toy",
            "source_gravity_file_path": "toy.gfc",
            "source_gravity_file_sha256": "a" * 64,
            "derivative_convention": base.get("derivative_convention_version"),
            "spherical_harmonic_convention": base.get("spherical_harmonic_convention"),
            "gravity_label_engine_version": base.get("gravity_label_engine_version"),
            "columns": ["x", "y", "z", "dU", "dax", "day", "daz"],
            "dataset_layout": {"dataset_name": "data", "shape": [4, 7]},
        }
        handle.attrs["dataset_contract_json"] = json.dumps(payload, sort_keys=True)
        for key, value in base.items():
            if value is not None:
                handle.attrs[key] = value
    return path


def _contract_payload(**overrides) -> dict:
    payload = {
        "dataset_id": "contract-test",
        "n_samples": 4,
        "target_mode": "residual",
        "baseline_kind": "spherical_harmonics",
        "degree_min": 2,
        "degree_max": 4,
        "mu_si": MU_MOON_SI,
        "r_ref_m": R_MOON_SI,
        "altitude_min_km": 100.0,
        "altitude_max_km": 200.0,
        "source_gravity_model": "toy",
        "source_gravity_file_path": "toy.gfc",
        "source_gravity_file_sha256": "a" * 64,
        "derivative_convention": REQUIRED_DERIVATIVE_CONVENTION,
        "spherical_harmonic_convention": REQUIRED_SH_PHASE_CONVENTION,
        "gravity_label_engine_version": GRAVITY_LABEL_ENGINE_VERSION,
        "dataset_layout": {"dataset_name": "data", "shape": [4, 7]},
    }
    payload.update(overrides)
    return payload


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


def test_missing_target_mode_fails(tmp_path):
    path = _write_h5(tmp_path / "data.h5", target_mode=None)
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="target_mode"):
        validate_dataset_contract(meta, data_path=path)


def test_derivative_convention_mismatch_rejected(tmp_path):
    path = _write_h5(tmp_path / "data.h5", derivative_convention_version="legacy")
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="derivative_convention"):
        validate_dataset_contract(meta, data_path=path)


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


def test_dataset_contract_content_hash_can_be_stamped(tmp_path):
    path = tmp_path / "contracted.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.ones((3, 7), dtype=np.float32))
        contract = DatasetContract(
            dataset_id="hash-test",
            n_samples=3,
            target_mode="residual",
            baseline_kind="spherical_harmonics",
            degree_min=2,
            degree_max=4,
            altitude_min_km=100.0,
            altitude_max_km=200.0,
            source_gravity_model="toy",
            source_gravity_file_path="toy.gfc",
            source_gravity_file_sha256="a" * 64,
            spherical_harmonic_convention="4pi_geodesy_no_condon_shortley_v1",
            gravity_label_engine_version="lunaris_sh_v2",
            dataset_layout={"dataset_name": "data", "shape": [3, 7]},
        )
        contract.write_hdf5_attrs(handle)

    updated = stamp_hdf5_content_hash(path)
    reread = read_dataset_contract_from_h5(path)

    assert updated.content_sha256
    assert reread["content_sha256"] == updated.content_sha256


def test_generator_contract_payload_stamps_gravity_label_contract():
    payload = build_contract_payload_for_generator(
        dataset_id="generator-contract",
        n_samples=4,
        degree_min=2,
        degree_max=4,
        target_mode="residual",
        baseline_kind="spherical_harmonics",
        mu_si=MU_MOON_SI,
        r_ref_m=R_MOON_SI,
        altitude_min_km=100.0,
        altitude_max_km=200.0,
        random_seed=7,
        sampling_policy={"name": "toy"},
        source_gravity_model="toy",
        source_gravity_file_path="toy.gfc",
        source_gravity_file_sha256="a" * 64,
        generator_version="test",
        columns=["x", "y", "z", "dU", "dax", "day", "daz"],
    )

    assert payload["spherical_harmonic_convention"] == REQUIRED_SH_PHASE_CONVENTION
    assert payload["gravity_label_engine_version"] == GRAVITY_LABEL_ENGINE_VERSION


def test_gravity_label_engine_version_is_required():
    payload = _contract_payload()
    payload.pop("gravity_label_engine_version")

    with pytest.raises(DatasetContractError, match="gravity_label_engine_version"):
        DatasetContract.from_dict(payload)


def test_from_dict_rejects_unsafe_derivative_contract():
    payload = _contract_payload(derivative_convention="legacy")
    payload.pop("spherical_harmonic_convention")
    payload.pop("gravity_label_engine_version")

    with pytest.raises(DatasetContractError, match="derivative_convention"):
        DatasetContract.from_dict(payload)


def test_compatibility_report_rejects_gravity_label_contract_mismatch():
    current = DatasetContract.from_dict(_contract_payload())
    legacy_payload = _contract_payload(
        spherical_harmonic_convention="legacy_csphase",
        gravity_label_engine_version="lunaris_sh_v1",
    )
    legacy = DatasetContract(**legacy_payload, _skip_initial_validation=True)

    report = current.compatibility_report(legacy)

    assert report["compatible"] is False
    assert any("spherical_harmonic_convention" in error for error in report["errors"])
    assert any("gravity_label_engine_version" in error for error in report["errors"])
