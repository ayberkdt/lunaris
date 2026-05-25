from __future__ import annotations

from types import SimpleNamespace

import pytest

from st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from st_lrps.shared.contracts import REQUIRED_DERIVATIVE_CONVENTION, TargetContract
from st_lrps.shared.scaling import compute_base_accel_from_contract, compute_base_potential_from_contract
from st_lrps.training.config import TrainConfig
from st_lrps.training.config_summary import build_experiment_feature_summary

torch = pytest.importorskip("torch")


def _meta(**overrides):
    values = {
        "central_body": "moon",
        "target_mode": "residual",
        "degree_min": 20,
        "degree_max": 100,
        "requested_degree": 100,
        "unit_system": "si",
        "derivative_convention_version": REQUIRED_DERIVATIVE_CONVENTION,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_residual_contract_from_metadata() -> None:
    c = TargetContract.from_dataset_meta(_meta(), MU_MOON_SI, R_MOON_SI, 1.0)
    assert c.is_residual
    assert c.baseline_kind == "spherical_harmonics"
    assert c.base_degree == 20
    assert c.target_degree == 100
    assert c.requires_baseline


def test_full_field_point_mass_contract_from_metadata() -> None:
    c = TargetContract.from_dataset_meta(
        _meta(target_mode="full", degree_min=-1, degree_max=100),
        MU_MOON_SI,
        R_MOON_SI,
        -1.0,
    )
    assert not c.is_residual
    assert c.baseline_kind == "point_mass"
    assert c.a_sign == -1.0


def test_invalid_target_mode_raises() -> None:
    with pytest.raises(ValueError, match="target_mode"):
        TargetContract.from_dataset_meta(_meta(target_mode="delta"), MU_MOON_SI, R_MOON_SI, 1.0)


def test_non_lunar_body_raises() -> None:
    with pytest.raises(ValueError, match="not lunar"):
        TargetContract.from_dataset_meta(_meta(central_body="earth"), MU_MOON_SI, R_MOON_SI, 1.0)


def test_residual_degree_order_raises() -> None:
    with pytest.raises(ValueError, match="target_degree > base_degree"):
        TargetContract.from_dataset_meta(_meta(degree_min=50, degree_max=50), MU_MOON_SI, R_MOON_SI, 1.0)


def test_backward_compatible_reconstruction_from_old_config() -> None:
    c = TargetContract.from_legacy_config(
        {
            "target_mode": "residual",
            "degree_min": 10,
            "degree_max": 60,
            "resolved_mu_si": MU_MOON_SI,
            "resolved_r_ref_m": R_MOON_SI,
            "resolved_a_sign": 1.0,
        }
    )
    assert c.target_mode == "residual"
    assert c.base_degree == 10
    assert c.target_degree == 60
    assert TargetContract.from_dict(c.to_dict()) == c


def test_missing_target_mode_requires_explicit_legacy_inference() -> None:
    with pytest.raises(ValueError, match="missing target_mode"):
        TargetContract.from_dataset_meta(_meta(target_mode=None), MU_MOON_SI, R_MOON_SI, 1.0)
    c = TargetContract.from_dataset_meta(
        _meta(target_mode=None),
        MU_MOON_SI,
        R_MOON_SI,
        1.0,
        allow_inferred_target_mode=True,
    )
    assert c.target_mode == "residual"


def test_contract_aware_base_subtraction_modes() -> None:
    x = torch.tensor([[R_MOON_SI + 1000.0, 0.0, 0.0]], dtype=torch.float64)
    residual = TargetContract.from_dataset_meta(_meta(), MU_MOON_SI, R_MOON_SI, 1.0)
    assert torch.count_nonzero(compute_base_potential_from_contract(x, residual)) == 0
    assert torch.count_nonzero(compute_base_accel_from_contract(x, residual)) == 0

    full = TargetContract.from_dataset_meta(
        _meta(target_mode="full", degree_min=-1),
        MU_MOON_SI,
        R_MOON_SI,
        1.0,
    )
    u_base = compute_base_potential_from_contract(x, full)
    a_base = compute_base_accel_from_contract(x, full)
    assert float(u_base[0, 0]) == pytest.approx(MU_MOON_SI / float(x[0, 0]))
    assert float(a_base[0, 0]) == pytest.approx(-MU_MOON_SI / float(x[0, 0]) ** 2)


def test_feature_summary_records_contract_and_active_features() -> None:
    c = TargetContract.from_dataset_meta(_meta(), MU_MOON_SI, R_MOON_SI, 1.0)
    cfg = TrainConfig(data="x.h5", out="run", model_preset="baseline_raw")
    summary = build_experiment_feature_summary(cfg, c, model=SimpleNamespace(
        embedding_type="raw",
        input_feature_dim=3,
    ))
    assert summary["model_preset"] == "baseline_raw"
    assert summary["input_encoding"] == "raw"
    assert summary["target_contract"]["target_mode"] == "residual"
