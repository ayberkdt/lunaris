from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
_ = pytest.importorskip("torch.utils")

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
    validate_training_dataset_convention,
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


def test_residual_with_inverted_degree_bounds_fails(tmp_path):
    """degree_max <= degree_min is not a valid residual band."""
    path = _write_h5(tmp_path / "data.h5", degree_min=200, degree_max=20)
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="degree_max"):
        validate_training_dataset_convention(meta, data_path=path)


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


def test_missing_altitude_bounds_fail(tmp_path):
    """A dataset with no altitude envelope cannot bound the training shell."""
    path = _write_h5(tmp_path / "data.h5", alt_min_km=None, alt_max_km=None)
    meta = DatasetMeta.from_h5(path)
    assert meta.alt_min_km is None and meta.alt_max_km is None
    with pytest.raises(ValueError, match="altitude bounds"):
        validate_training_dataset_convention(meta, data_path=path)


@pytest.mark.parametrize("missing", ["alt_min_km", "alt_max_km"])
def test_partially_missing_altitude_bounds_fail(tmp_path, missing):
    path = _write_h5(tmp_path / "data.h5", **{missing: None})
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="altitude bounds"):
        validate_training_dataset_convention(meta, data_path=path)


def test_equal_altitude_bounds_fail(tmp_path):
    """alt_min == alt_max is a zero-thickness shell, not a training envelope."""
    path = _write_h5(tmp_path / "data.h5", alt_min_km=500.0, alt_max_km=500.0)
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="altitude bounds"):
        validate_training_dataset_convention(meta, data_path=path)


def test_invalid_target_mode_value_fails(tmp_path):
    path = _write_h5(tmp_path / "data.h5", target_mode="hybrid")
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="target_mode"):
        validate_training_dataset_convention(meta, data_path=path)


def test_equal_degree_bounds_fail(tmp_path):
    """degree_max == degree_min is an empty residual band."""
    path = _write_h5(tmp_path / "data.h5", degree_min=60, degree_max=60)
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="degree_max"):
        validate_training_dataset_convention(meta, data_path=path)


def test_units_are_validated(tmp_path):
    path = _write_h5(tmp_path / "data.h5", unit_system="mystery")
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="unit_system"):
        validate_dataset_contract(meta, data_path=path)


def test_canonical_units_without_scaling_constants_fail(tmp_path):
    """Phase 7: a canonical (non-dimensional) dataset with no DU/TU/VU cannot be
    converted to SI unambiguously, so preflight must fail before training."""
    path = _write_h5(tmp_path / "data.h5", unit_system="canonical")
    meta = DatasetMeta.from_h5(path)
    assert meta.can_convert_to_si() is False
    with pytest.raises(ValueError, match="canonical"):
        validate_training_dataset_convention(meta, data_path=path)


def test_canonical_units_with_scaling_constants_pass_convention_check(tmp_path):
    path = _write_h5(
        tmp_path / "data.h5",
        unit_system="canonical",
        DU_m=1.7e6,
        TU_s=5.0e3,
        VU_m_s=340.0,
    )
    meta = DatasetMeta.from_h5(path)
    assert meta.can_convert_to_si() is True
    # Must not raise: canonical WITH scaling constants is unambiguous.
    validate_training_dataset_convention(meta, data_path=path)


def test_non_lunar_body_fails(tmp_path):
    path = _write_h5(tmp_path / "data.h5", central_body="earth")
    meta = DatasetMeta.from_h5(path)
    with pytest.raises(ValueError, match="not lunar"):
        validate_training_dataset_convention(meta, data_path=path)


def test_missing_derivative_convention_fails(tmp_path):
    # None override is skipped when writing attrs, so the attr is absent entirely.
    path = _write_h5(tmp_path / "data.h5", derivative_convention_version=None)
    meta = DatasetMeta.from_h5(path)
    assert meta.derivative_convention_version is None
    with pytest.raises(ValueError, match="derivative_convention"):
        validate_training_dataset_convention(meta, data_path=path)


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


def test_npz_writer_finalize_checks_incomplete_stream(tmp_path):
    from lunaris.batch.storage import _NPZWriter
    # R24: _NPZWriter.finalize() must raise ValueError if the stream terminates early
    path = tmp_path / "test.npz"
    writer = _NPZWriter(path, 10, ["Y", "Y_summary"])

    # Write 5 sub-batches of 1
    for _ in range(5):
        writer.write({"Y": np.zeros(1), "Y_summary": np.zeros(1)})

    with pytest.raises(ValueError, match="Expected 10 samples"):
        writer.finalize()

