from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from lunaris.surrogate.st_lrps.artifacts.manager import (
    build_checkpoint_payload,
    build_resolved_config,
)
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.networks.models import (
    build_model_from_config,
    compute_architecture_signature,
)
from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack


def tiny_scaler(*, x_scale: float | None = None) -> ScalerPack:
    return ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=float(x_scale or (R_MOON_SI + 1_200_000.0))),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
        provenance={"fit_rows": 16, "alt_min_km": 100.0, "alt_max_km": 1000.0},
    )


def tiny_training_cfg(
    *,
    degree_min: int = 20,
    degree_max: int = 60,
    alt_min_km: float = 100.0,
    alt_max_km: float = 1000.0,
    **overrides: Any,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "data": "synthetic_contract_fixture.h5",
        "train_data_path": "synthetic_contract_fixture.h5",
        "val_data_path": "synthetic_contract_fixture.h5",
        "dataset_name": "data",
        "central_body": "moon",
        "target_mode": "residual",
        "degree_min": int(degree_min),
        "degree_max": int(degree_max),
        "unit_system": "si",
        "resolved_mu_si": MU_MOON_SI,
        "resolved_r_ref_m": R_MOON_SI,
        "resolved_a_sign": 1.0,
        "activation": "tanh",
        "hidden": 8,
        "depth": 1,
        "dropout": 0.0,
        "use_residual_blocks": False,
        "n_bands": 1,
        "w0_bands": None,
        "w0_first": 30.0,
        "w0_hidden": 30.0,
        "use_fourier": False,
        "fourier_append_raw": True,
        "fourier_n_features": 0,
        "fourier_sigma": 0.0,
        "fourier_seed": 0,
        "use_sh_encoding": False,
        "sh_encoding_degree": 4,
        "sh_append_raw": True,
        "use_radial_separation": False,
        "radial_append_raw": False,
        "use_radial_decay_encoding": False,
        "radial_decay_max_power": 4,
        "radial_decay_append_raw": True,
        "use_physical_radial_decay_encoding": False,
        "physical_radial_decay_max_power": 4,
        "physical_radial_decay_append_raw": True,
        "physical_radial_decay_include_unit": True,
        "physical_radial_decay_include_r_scaled": True,
        "runtime_model_kind": "potential_autograd",
        "model_preset": "custom",
        "best_metric": "val_total_loss",
        "altitude_min_km": float(alt_min_km),
        "altitude_max_km": float(alt_max_km),
        "run_name": "contract_fixture",
    }
    cfg.update(overrides)
    return cfg


def tiny_dataset_meta(
    *,
    degree_min: int = 20,
    degree_max: int = 60,
    alt_min_km: float = 100.0,
    alt_max_km: float = 1000.0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_name": "data",
        "dataset_sha256": "0" * 64,
        "target_mode": "residual",
        "baseline_kind": "spherical_harmonics",
        "degree_min": int(degree_min),
        "degree_max": int(degree_max),
        "mu_si": MU_MOON_SI,
        "r_ref_m": R_MOON_SI,
        "central_body": "moon",
        "unit_system": "si",
        "alt_min_km": float(alt_min_km),
        "alt_max_km": float(alt_max_km),
        "altitude_min_km": float(alt_min_km),
        "altitude_max_km": float(alt_max_km),
        "a_sign": 1.0,
        "a_sign_convention": "+1",
        "derivative_convention_version": "dP_dphi_corrected_v1",
        "coordinate_frame": "moon_fixed_cartesian",
        "units": {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"},
        "dataset_contract": {
            "schema_version": 1,
            "dataset_kind": "st_lrps_spatial_cloud",
            "dataset_sha256": "0" * 64,
            "target_mode": "residual",
            "baseline_kind": "spherical_harmonics",
            "degree_min": int(degree_min),
            "degree_max": int(degree_max),
            "mu_si": MU_MOON_SI,
            "r_ref_m": R_MOON_SI,
            "a_sign": 1.0,
            "altitude_min_km": float(alt_min_km),
            "altitude_max_km": float(alt_max_km),
            "coordinate_frame": "moon_fixed_cartesian",
            "units": {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"},
            "source_gravity_file_sha256": "1" * 64,
        },
    }


def make_contract_run(
    tmp_path: Path,
    *,
    degree_min: int = 20,
    degree_max: int = 60,
    alt_min_km: float = 100.0,
    alt_max_km: float = 1000.0,
    include_contract: bool = True,
    contract_overrides: dict[str, Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = tmp_path / "contract_run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = tiny_training_cfg(
        degree_min=degree_min,
        degree_max=degree_max,
        alt_min_km=alt_min_km,
        alt_max_km=alt_max_km,
        **(cfg_overrides or {}),
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    cfg["input_feature_dim"] = int(getattr(model, "input_feature_dim", 3))
    cfg["embedding_type"] = str(getattr(model, "embedding_type", "raw"))
    cfg["model_builder_version"] = str(getattr(model, "model_builder_version", "unknown"))
    arch_sig = compute_architecture_signature(cfg)
    scaler = tiny_scaler()
    dataset_meta = tiny_dataset_meta(
        degree_min=degree_min,
        degree_max=degree_max,
        alt_min_km=alt_min_km,
        alt_max_km=alt_max_km,
    )
    resolved_cfg = build_resolved_config(cfg, dataset_meta, model, scaler, arch_sig)
    payload = build_checkpoint_payload(
        kind="best",
        epoch=0,
        model=model,
        optimizer=None,
        scheduler=None,
        cfg=resolved_cfg,
        scaler=scaler,
        train_stats={
            "lr": 1e-3,
            "w_u": 1.0,
            "w_a": 1.0,
            "gradnorm_status": "fixed",
            "accel_factor": 1.0,
            "lambda_dir_eff": 0.0,
        },
        val_stats={"loss": 0.0, "val_total_loss": 0.0, "val_checkpoint_score": 0.0},
        dataset_meta=dataset_meta,
        architecture_signature=arch_sig,
        global_step=1,
    )
    if contract_overrides:
        contract = dict(payload["artifact_contract"])
        contract.update(contract_overrides)
        payload["artifact_contract"] = contract
        payload["config"]["artifact_contract"] = contract
        resolved_cfg["artifact_contract"] = contract
    if not include_contract:
        payload.pop("artifact_contract", None)
        payload.pop("schema_version", None)
        payload["config"].pop("artifact_contract", None)
        resolved_cfg.pop("artifact_contract", None)

    (run_dir / "config.json").write_text(json.dumps(resolved_cfg, indent=2, sort_keys=True), encoding="utf-8")
    scaler.save_json(run_dir / "scaler.json")
    torch.save(payload, run_dir / "checkpoints" / "ckpt_best.pt")
    return {
        "run_dir": run_dir,
        "cfg": resolved_cfg,
        "payload": payload,
        "model": model,
        "scaler": scaler,
        "dataset_meta": dataset_meta,
        "architecture_signature": arch_sig,
        "scaler_payload": asdict(scaler),
    }
