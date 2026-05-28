"""Experiment feature summaries for ST-LRPS training runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _contract_dict(target_contract: Any) -> dict[str, Any] | None:
    if target_contract is None:
        return None
    if hasattr(target_contract, "to_dict"):
        return target_contract.to_dict()
    if is_dataclass(target_contract):
        return asdict(target_contract)
    if isinstance(target_contract, Mapping):
        return dict(target_contract)
    return None


def build_experiment_feature_summary(
    cfg: Any,
    target_contract: Any = None,
    model: Any = None,
) -> dict[str, Any]:
    """Return a compact ablation-friendly summary of active features."""

    return {
        "model_preset": str(_cfg_get(cfg, "model_preset", "custom")),
        "input_encoding": str(getattr(model, "embedding_type", _cfg_get(cfg, "embedding_type", "raw"))),
        "input_feature_dim": int(getattr(model, "input_feature_dim", _cfg_get(cfg, "input_feature_dim", 3))),
        "runtime_model_kind": str(_cfg_get(cfg, "runtime_model_kind", "potential_autograd")),
        "residual_blocks": bool(_cfg_get(cfg, "use_residual_blocks", False)),
        "n_bands": int(_cfg_get(cfg, "n_bands", 1) or 1),
        "multiscale_mode": str(_cfg_get(cfg, "multiscale_mode", "concat_shared")),
        "altitude_balanced_loss": bool(_cfg_get(cfg, "use_altitude_balanced_loss", False)),
        "radial_cross_loss": {
            "enabled": bool(_cfg_get(cfg, "use_radial_cross_loss", False)),
            "radial_weight": float(_cfg_get(cfg, "radial_loss_weight", 0.0) or 0.0),
            "cross_weight": float(_cfg_get(cfg, "cross_loss_weight", 0.0) or 0.0),
        },
        "direction_loss": {
            "weight": float(_cfg_get(cfg, "direction_loss_weight", 0.0) or 0.0),
            "start_epoch": int(_cfg_get(cfg, "direction_loss_start_epoch", 0) or 0),
            "ramp_epochs": int(_cfg_get(cfg, "direction_loss_ramp_epochs", 0) or 0),
        },
        "laplacian_mode": str(_cfg_get(cfg, "laplacian_mode", "diagnostic")),
        "laplacian_regularization": bool(_cfg_get(cfg, "use_laplacian_regularization", False)),
        "gradnorm_mode": str(_cfg_get(cfg, "gradnorm_mode", "fixed")),
        "curriculum": {
            "potential_only_epochs": int(_cfg_get(cfg, "potential_only_epochs", 0) or 0),
            "accel_ramp_epochs": int(_cfg_get(cfg, "accel_ramp_epochs", 0) or 0),
            "accel_min_factor": float(_cfg_get(cfg, "accel_min_factor", 0.0) or 0.0),
        },
        "physical_radial_decay": {
            "enabled": bool(_cfg_get(cfg, "use_physical_radial_decay_encoding", False)),
            "max_power": int(_cfg_get(cfg, "physical_radial_decay_max_power", 4) or 4),
            "append_raw": bool(_cfg_get(cfg, "physical_radial_decay_append_raw", True)),
            "include_unit": bool(_cfg_get(cfg, "physical_radial_decay_include_unit", True)),
            "include_r_scaled": bool(_cfg_get(cfg, "physical_radial_decay_include_r_scaled", True)),
        },
        "x_scale_m": _cfg_get(cfg, "x_scale_m"),
        "r_ref_m": _cfg_get(cfg, "resolved_r_ref_m", _cfg_get(cfg, "r_ref_m")),
        "target_contract": _contract_dict(target_contract),
    }


__all__ = ["build_experiment_feature_summary"]
