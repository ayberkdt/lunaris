from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from lunaris.surrogate.st_lrps.networks.models import (
    ARCH_SIGNATURE_FIELDS,
    MODEL_BUILDER_VERSION,
    compute_architecture_signature,
    reconstruct_model_from_artifacts,
)
from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
from lunaris.surrogate.st_lrps.shared.contracts import (
    ArtifactContract,
    ArtifactContractError,
)


logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = "st_lrps_checkpoint_v2"
RUN_MANIFEST_SCHEMA_VERSION = "st_lrps_run_manifest_v1"
CHECKPOINTS_MANIFEST_SCHEMA_VERSION = "st_lrps_checkpoints_manifest_v1"
EVAL_MANIFEST_SCHEMA_VERSION = "st_lrps_eval_manifest_v1"

CRITICAL_CONFIG_FIELDS: Tuple[str, ...] = (
    "activation",
    "hidden",
    "depth",
    "dropout",
    "model_preset",
    "runtime_model_kind",
    "n_bands",
    "w0_bands",
    "use_residual_blocks",
    "use_fourier",
    "fourier_append_raw",
    "fourier_n_features",
    "fourier_sigma",
    "fourier_seed",
    "use_sh_encoding",
    "sh_encoding_degree",
    "sh_append_raw",
    "use_radial_separation",
    "radial_append_raw",
    "use_radial_decay_encoding",
    "radial_decay_max_power",
    "radial_decay_append_raw",
    "use_physical_radial_decay_encoding",
    "physical_radial_decay_max_power",
    "physical_radial_decay_append_raw",
    "physical_radial_decay_include_unit",
    "physical_radial_decay_include_r_scaled",
    "x_scale_m",
    "input_feature_dim",
    "embedding_type",
    "model_builder_version",
    "architecture_signature",
    "degree_min",
    "degree_max",
    "target_mode",
    "resolved_mu_si",
    "resolved_r_ref_m",
    "resolved_a_sign",
    "artifact_contract",
    "dataset_contract",
    "training_config_hash",
)


@dataclass
class RunLayout:
    run_dir: Path
    config_json: Path
    scaler_json: Path
    run_manifest_json: Path
    command_txt: Path
    train_log: Path
    checkpoints_dir: Path
    ckpt_best: Path
    ckpt_last: Path
    ckpt_epoch_pattern: str
    history_csv: Path
    history_jsonl: Path
    plots_dir: Path
    evals_dir: Path
    provenance_dir: Path


def make_run_layout(run_dir: Path) -> RunLayout:
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoints_dir = run_dir / "checkpoints"
    return RunLayout(
        run_dir=run_dir,
        config_json=run_dir / "config.json",
        scaler_json=run_dir / "scaler.json",
        run_manifest_json=run_dir / "run_manifest.json",
        command_txt=run_dir / "command.txt",
        train_log=run_dir / "train.log",
        checkpoints_dir=checkpoints_dir,
        ckpt_best=checkpoints_dir / "ckpt_best.pt",
        ckpt_last=checkpoints_dir / "ckpt_last.pt",
        ckpt_epoch_pattern=str(checkpoints_dir / "ckpt_epoch_{epoch_display:06d}.pt"),
        history_csv=run_dir / "history.csv",
        history_jsonl=run_dir / "history.jsonl",
        plots_dir=run_dir / "plots",
        evals_dir=run_dir / "evals",
        provenance_dir=run_dir / "provenance",
    )


def ensure_run_layout(run_dir: Path) -> RunLayout:
    layout = make_run_layout(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    layout.plots_dir.mkdir(parents=True, exist_ok=True)
    (layout.plots_dir / "training").mkdir(parents=True, exist_ok=True)
    layout.evals_dir.mkdir(parents=True, exist_ok=True)
    layout.provenance_dir.mkdir(parents=True, exist_ok=True)
    return layout


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {k: _json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _canonical_json_text(payload: Mapping[str, Any], *, indent: int = 2) -> str:
    return json.dumps(
        _json_safe(dict(payload)),
        indent=indent,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    ) + "\n"


def atomic_write_json(path: Path, payload: dict, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        text = _canonical_json_text(payload, indent=indent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_text(payload, indent=2).encode("utf-8")).hexdigest()


def compute_payload_sha256(payload: Mapping[str, Any]) -> str:
    return _compute_payload_sha256(payload)


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _infer_checkpoint_kind_from_path(path: Path) -> str:
    name = path.name.lower()
    if "best" in name:
        return "best"
    if "last" in name:
        return "last"
    return "epoch"


def _state_dict_from_checkpoint(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("model_state_dict"), dict):
            return ckpt["model_state_dict"]
        if isinstance(ckpt.get("model"), dict):
            return ckpt["model"]
        if all(isinstance(k, str) for k in ckpt.keys()) and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt  # pragma: no cover - raw state_dict legacy path
    raise ValueError("Checkpoint does not contain a usable model state_dict.")


def _extract_config_block(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    raw_cfg = ckpt.get("config")
    if isinstance(raw_cfg, dict):
        cfg = dict(raw_cfg)
    elif isinstance(ckpt.get("cfg"), dict):
        cfg = dict(ckpt["cfg"])
    else:
        cfg = {}

    # Legacy checkpoints stored architecture and physics fields at the top level.
    alias_keys = set(ARCH_SIGNATURE_FIELDS) | {
        "resolved_mu_si",
        "resolved_r_ref_m",
        "resolved_a_sign",
        "degree_min",
        "degree_max",
        "target_mode",
        "residual_mode",
        "central_body",
        "dataset_name",
        "mu_si",
        "r_ref_m",
        "w0_bands",
        "model_builder_version",
        "input_feature_dim",
        "embedding_type",
        "use_fourier",
        "fourier_append_raw",
        "fourier_n_features",
        "fourier_sigma",
        "fourier_seed",
        "use_sh_encoding",
        "sh_encoding_degree",
        "sh_append_raw",
        "use_radial_separation",
        "radial_append_raw",
        "n_bands",
        "activation",
        "hidden",
        "depth",
        "dropout",
        "use_residual_blocks",
        "architecture_signature",
    }
    for key in alias_keys:
        if cfg.get(key) is None and ckpt.get(key) is not None:
            cfg[key] = ckpt.get(key)

    dataset_meta = cfg.get("dataset_meta")
    if not isinstance(dataset_meta, dict):
        dataset_meta = {}
    for key in (
        "unit_system",
        "central_body",
        "mu_si",
        "r_ref_m",
        "requested_degree",
        "degree_min",
        "degree_max",
        "target_mode",
        "alt_min_km",
        "alt_max_km",
        "derivative_convention_version",
        "a_sign_convention",
    ):
        if dataset_meta.get(key) is None and cfg.get(key) is not None:
            dataset_meta[key] = cfg.get(key)
    if dataset_meta:
        cfg["dataset_meta"] = dataset_meta

    if cfg and cfg.get("architecture_signature") in (None, ""):
        try:
            cfg["architecture_signature"] = compute_architecture_signature(cfg)
        except Exception:
            pass
    if cfg and cfg.get("model_builder_version") in (None, ""):
        cfg["model_builder_version"] = MODEL_BUILDER_VERSION
    return cfg


def _build_architecture_block_from_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "signature": cfg.get("architecture_signature"),
        "model_builder_version": cfg.get("model_builder_version", MODEL_BUILDER_VERSION),
        "activation": cfg.get("activation"),
        "hidden": _coerce_int_or_none(cfg.get("hidden")),
        "depth": _coerce_int_or_none(cfg.get("depth")),
        "n_bands": _coerce_int_or_none(cfg.get("n_bands")),
        "w0_bands": list(cfg.get("w0_bands") or []) or None,
        "input_feature_dim": _coerce_int_or_none(cfg.get("input_feature_dim")),
        "embedding_type": cfg.get("embedding_type"),
        "use_fourier": bool(cfg.get("use_fourier", False)),
        "fourier_append_raw": bool(cfg.get("fourier_append_raw", True)),
        "fourier_n_features": _coerce_int_or_none(cfg.get("fourier_n_features")),
        "fourier_sigma": _coerce_float_or_none(cfg.get("fourier_sigma")),
        "fourier_seed": _coerce_int_or_none(cfg.get("fourier_seed")),
        "use_sh_encoding": bool(cfg.get("use_sh_encoding", False)),
        "sh_encoding_degree": _coerce_int_or_none(cfg.get("sh_encoding_degree")),
        "sh_append_raw": bool(cfg.get("sh_append_raw", True)),
        "use_radial_separation": bool(cfg.get("use_radial_separation", False)),
        "radial_append_raw": bool(cfg.get("radial_append_raw", False)),
        "use_residual_blocks": bool(cfg.get("use_residual_blocks", False)),
        "dropout": _coerce_float_or_none(cfg.get("dropout")),
    }


def _build_dataset_block_from_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_meta = cfg.get("dataset_meta") or {}
    if not isinstance(dataset_meta, dict):
        dataset_meta = {}
    return {
        "train_data": _coerce_str_or_none(cfg.get("train_data_path")),
        "val_data": _coerce_str_or_none(cfg.get("val_data_path")),
        "test_data": _coerce_str_or_none(cfg.get("test_data_path")),
        "ood_data": _coerce_str_or_none(cfg.get("ood_data_path")),
        "dataset_name": _coerce_str_or_none(cfg.get("dataset_name")) or "data",
        "central_body": _coerce_str_or_none(cfg.get("central_body") or dataset_meta.get("central_body")) or "moon",
        "target_mode": _coerce_str_or_none(cfg.get("target_mode") or dataset_meta.get("target_mode")) or "full",
        "degree_min": _coerce_int(cfg.get("degree_min", dataset_meta.get("degree_min", -1)), default=-1),
        "degree_max": _coerce_int(cfg.get("degree_max", dataset_meta.get("degree_max", -1)), default=-1),
        "unit_system": _coerce_str_or_none(cfg.get("unit_system") or dataset_meta.get("unit_system")) or "unknown",
        "mu_si": float(cfg.get("resolved_mu_si", dataset_meta.get("mu_si", 0.0)) or 0.0),
        "r_ref_m": float(cfg.get("resolved_r_ref_m", dataset_meta.get("r_ref_m", 0.0)) or 0.0),
        "alt_min_km": _coerce_float_or_none(dataset_meta.get("alt_min_km")),
        "alt_max_km": _coerce_float_or_none(dataset_meta.get("alt_max_km")),
        "derivative_convention_version": _coerce_str_or_none(dataset_meta.get("derivative_convention_version")),
        "a_sign_convention": _coerce_str_or_none(dataset_meta.get("a_sign_convention")),
    }


def normalize_legacy_checkpoint(ckpt: dict) -> dict:
    if not isinstance(ckpt, dict):
        ckpt = {"model_state_dict": ckpt}

    cfg = _extract_config_block(ckpt)
    state_dict = _state_dict_from_checkpoint(ckpt)
    epoch = _coerce_int(ckpt.get("epoch"), default=0)

    best_metric = (
        _coerce_str_or_none((ckpt.get("scoring") or {}).get("best_metric"))
        or _coerce_str_or_none(ckpt.get("best_score_name"))
        or _coerce_str_or_none(cfg.get("best_score_name"))
        or "val_total_loss"
    )

    score = (
        _coerce_float_or_none((ckpt.get("scoring") or {}).get("score"))
        or _coerce_float_or_none(ckpt.get("best_score"))
        or _coerce_float_or_none(ckpt.get("best_val"))
        or _coerce_float_or_none(cfg.get("best_score"))
        or _coerce_float_or_none(cfg.get("best_val_loss"))
    )

    scoring = {
        "best_metric": best_metric,
        "score": score,
        "val_loss": _coerce_float_or_none(ckpt.get("best_val")) or _coerce_float_or_none(cfg.get("best_val_loss")) or 0.0,
        "val_base_loss": _coerce_float_or_none(ckpt.get("best_val_base_loss")) or _coerce_float_or_none(cfg.get("best_val_base_loss")),
        "val_total_loss": _coerce_float_or_none(ckpt.get("best_val_total_loss")) or _coerce_float_or_none(cfg.get("best_val_total_loss")),
        "val_physics_loss": _coerce_float_or_none(ckpt.get("best_val_physics_loss")) or _coerce_float_or_none(cfg.get("best_val_physics_loss")),
        "loss_dir": _coerce_float_or_none((ckpt.get("val_stats") or {}).get("loss_dir")) or _coerce_float_or_none(cfg.get("loss_dir")),
        "mean_cos_sim": _coerce_float_or_none((ckpt.get("val_stats") or {}).get("cossim_mean")) or _coerce_float_or_none(cfg.get("mean_cos_sim")),
        "mean_ang_deg": _coerce_float_or_none((ckpt.get("val_stats") or {}).get("angular_mean_deg")) or _coerce_float_or_none(cfg.get("mean_ang_deg")),
    }

    training_state = {
        "lr": _coerce_float_or_none((ckpt.get("training_state") or {}).get("lr")) or _coerce_float_or_none(ckpt.get("lr")) or 0.0,
        "w_u": _coerce_float_or_none((ckpt.get("training_state") or {}).get("w_u")) or _coerce_float_or_none(ckpt.get("w_u")) or 0.0,
        "w_a": _coerce_float_or_none((ckpt.get("training_state") or {}).get("w_a")) or _coerce_float_or_none(ckpt.get("w_a")) or 0.0,
        "gradnorm_status": _coerce_str_or_none((ckpt.get("training_state") or {}).get("gradnorm_status")),
        "accel_factor": _coerce_float_or_none((ckpt.get("training_state") or {}).get("accel_factor")) or _coerce_float_or_none(ckpt.get("accel_factor")) or 1.0,
        "lambda_dir_eff": _coerce_float_or_none((ckpt.get("training_state") or {}).get("lambda_dir_eff")) or _coerce_float_or_none(ckpt.get("lambda_dir_eff")) or 0.0,
        "rng_state": (ckpt.get("training_state") or {}).get("rng_state"),
        # Resume-only: loss-weighting (GradNorm) internal state. Optional; older
        # checkpoints simply carry None and resume falls back to a fresh GradNorm.
        "gradnorm_weights": (ckpt.get("training_state") or {}).get("gradnorm_weights"),
    }

    normalized = {
        "schema_version": ckpt.get("schema_version") or "legacy",
        "kind": ckpt.get("kind") or "epoch",
        "epoch": epoch,
        "epoch_display": _coerce_int(ckpt.get("epoch_display"), default=epoch + 1),
        "global_step": _coerce_int_or_none(ckpt.get("global_step")),
        "model_state_dict": state_dict,
        "optimizer_state_dict": ckpt.get("optimizer_state_dict", ckpt.get("optimizer")),
        "scheduler_state_dict": ckpt.get("scheduler_state_dict", ckpt.get("scheduler")),
        "config": cfg,
        "scaler": dict(ckpt.get("scaler") or {}),
        "architecture": _build_architecture_block_from_cfg(cfg),
        "dataset": _build_dataset_block_from_cfg(cfg),
        "scoring": scoring,
        "training_state": training_state,
        "created_at_utc": _coerce_str_or_none(ckpt.get("created_at_utc")) or _coerce_str_or_none(cfg.get("created_at_utc")) or "",
    }
    normalized["architecture"]["signature"] = (
        normalized["architecture"].get("signature")
        or cfg.get("architecture_signature")
        or normalized["architecture"].get("signature")
    )
    if "scaler_hash" in ckpt:
        normalized["scaler_hash"] = ckpt.get("scaler_hash")
    for key in (
        "dataset_contract",
        "resolved_config",
        "training_config_hash",
        "dataset_hash",
        "model_builder_version",
    ):
        if key in ckpt:
            normalized[key] = ckpt.get(key)
        elif key in cfg:
            normalized[key] = cfg.get(key)
    if isinstance(normalized.get("dataset_contract"), dict):
        normalized["config"].setdefault("dataset_contract", dict(normalized["dataset_contract"]))
    if isinstance(ckpt.get("artifact_contract"), dict):
        normalized["artifact_contract"] = dict(ckpt["artifact_contract"])
        normalized["config"].setdefault("artifact_contract", dict(ckpt["artifact_contract"]))
    elif isinstance(normalized["config"].get("artifact_contract"), dict):
        normalized["artifact_contract"] = dict(normalized["config"]["artifact_contract"])
    if "model" in ckpt or "model_state_dict" not in ckpt:
        normalized["model"] = state_dict
    return normalized


def validate_checkpoint_schema(ckpt: dict, *, strict: bool = True) -> dict:
    normalized = normalize_legacy_checkpoint(ckpt)

    required_top = (
        "schema_version",
        "kind",
        "epoch",
        "epoch_display",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "config",
        "scaler",
        "architecture",
        "dataset",
        "scoring",
        "training_state",
        "created_at_utc",
    )
    for key in required_top:
        if key not in normalized:
            raise KeyError(f"Checkpoint is missing required key: {key}")

    if normalized["kind"] not in {"best", "last", "epoch"}:
        if strict:
            raise RuntimeError(f"Unsupported checkpoint kind: {normalized['kind']!r}")
        normalized["kind"] = "epoch"

    if strict and normalized.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint schema mismatch: expected {CHECKPOINT_SCHEMA_VERSION!r}, "
            f"got {normalized.get('schema_version')!r}."
        )

    if not isinstance(normalized["model_state_dict"], dict):
        raise TypeError("checkpoint['model_state_dict'] must be a dict.")
    if not isinstance(normalized["config"], dict):
        raise TypeError("checkpoint['config'] must be a dict.")
    if not isinstance(normalized["scaler"], dict):
        raise TypeError("checkpoint['scaler'] must be a dict.")
    if not isinstance(normalized["architecture"], dict):
        raise TypeError("checkpoint['architecture'] must be a dict.")
    if not isinstance(normalized["dataset"], dict):
        raise TypeError("checkpoint['dataset'] must be a dict.")
    if not isinstance(normalized["scoring"], dict):
        raise TypeError("checkpoint['scoring'] must be a dict.")
    if not isinstance(normalized["training_state"], dict):
        raise TypeError("checkpoint['training_state'] must be a dict.")

    cfg = normalized["config"]
    arch = normalized["architecture"]
    if cfg and cfg.get("architecture_signature") in (None, ""):
        try:
            cfg["architecture_signature"] = compute_architecture_signature(cfg)
        except Exception:
            pass
    if arch.get("signature") in (None, "") and cfg.get("architecture_signature"):
        arch["signature"] = cfg.get("architecture_signature")

    if strict and cfg.get("architecture_signature") and arch.get("signature"):
        if str(cfg["architecture_signature"]) != str(arch["signature"]):
            raise RuntimeError(
                "Checkpoint architecture signature mismatch between config and architecture block: "
                f"{cfg['architecture_signature']!r} != {arch['signature']!r}."
            )

    if strict and not normalized.get("created_at_utc"):
        raise RuntimeError("Canonical checkpoints must record created_at_utc.")
    if strict and not isinstance(normalized.get("artifact_contract"), dict):
        raise RuntimeError("Canonical checkpoints must record artifact_contract.")

    normalized["epoch"] = _coerce_int(normalized.get("epoch"), default=0)
    normalized["epoch_display"] = _coerce_int(
        normalized.get("epoch_display"),
        default=normalized["epoch"] + 1,
    )
    return normalized


def _checkpoint_manifest_path(layout: RunLayout) -> Path:
    return layout.checkpoints_dir / "checkpoints_manifest.json"


def _write_checkpoints_manifest(layout: RunLayout) -> None:
    entries = []
    for path in sorted(layout.checkpoints_dir.glob("ckpt_*.pt")):
        try:
            ckpt = load_checkpoint(path, torch.device("cpu"))
            entries.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "sha256": compute_file_sha256(path),
                    "schema_version": ckpt.get("schema_version"),
                    "kind": ckpt.get("kind"),
                    "epoch": ckpt.get("epoch"),
                    "epoch_display": ckpt.get("epoch_display"),
                    "architecture_signature": (ckpt.get("architecture") or {}).get("signature"),
                }
            )
        except Exception as exc:
            entries.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "error": str(exc),
                }
            )
    atomic_write_json(
        _checkpoint_manifest_path(layout),
        {
            "schema_version": CHECKPOINTS_MANIFEST_SCHEMA_VERSION,
            "updated_at_utc": _utcnow_iso(),
            "checkpoints": entries,
        },
    )


def save_checkpoint(
    layout: RunLayout,
    *,
    kind: Literal["best", "last", "epoch"],
    payload: dict,
    epoch: int,
    write_epoch_snapshot: bool = False,
) -> Path:
    ensure_run_layout(layout.run_dir)
    ckpt = dict(payload)
    ckpt["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    ckpt["kind"] = kind
    ckpt["epoch"] = int(epoch)
    ckpt["epoch_display"] = int(epoch) + 1
    if "model_state_dict" not in ckpt and "model" in ckpt:
        ckpt["model_state_dict"] = ckpt["model"]
    if "model" not in ckpt and "model_state_dict" in ckpt:
        ckpt["model"] = ckpt["model_state_dict"]
    if isinstance(ckpt.get("scaler"), dict) and "scaler_hash" not in ckpt:
        ckpt["scaler_hash"] = _compute_payload_sha256(ckpt["scaler"])
    ckpt = validate_checkpoint_schema(ckpt, strict=True)

    if kind == "best":
        target = layout.ckpt_best
    elif kind == "last":
        target = layout.ckpt_last
    else:
        target = Path(layout.ckpt_epoch_pattern.format(epoch_display=int(epoch) + 1))

    _atomic_torch_save(target, ckpt)

    if write_epoch_snapshot and kind in {"best", "last"}:
        snapshot = dict(ckpt)
        snapshot["kind"] = "epoch"
        snapshot = validate_checkpoint_schema(snapshot, strict=True)
        snapshot_path = Path(layout.ckpt_epoch_pattern.format(epoch_display=int(epoch) + 1))
        _atomic_torch_save(snapshot_path, snapshot)

    _write_checkpoints_manifest(layout)
    return target


def load_checkpoint(path: Path, device: torch.device) -> dict:
    path = Path(path).expanduser().resolve()
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    ckpt = normalize_legacy_checkpoint(obj if isinstance(obj, dict) else {"model_state_dict": obj})
    if ckpt.get("kind") not in {"best", "last", "epoch"}:
        ckpt["kind"] = _infer_checkpoint_kind_from_path(path)
    ckpt = validate_checkpoint_schema(
        ckpt,
        strict=(ckpt.get("schema_version") == CHECKPOINT_SCHEMA_VERSION),
    )
    ckpt["checkpoint_path"] = str(path)
    ckpt["checkpoint_hash"] = compute_file_sha256(path)
    return ckpt


def load_best_or_last(
    layout: RunLayout,
    prefer: str = "best",
    device: Optional[torch.device] = None,
) -> tuple[Path, dict]:
    layout = make_run_layout(layout.run_dir)
    dev = device or torch.device("cpu")
    prefer_l = str(prefer).strip().lower()
    if prefer_l not in {"best", "last"}:
        raise ValueError("prefer must be 'best' or 'last'")
    order = [layout.ckpt_best, layout.ckpt_last] if prefer_l == "best" else [layout.ckpt_last, layout.ckpt_best]
    for candidate in order:
        if candidate.exists():
            return candidate, load_checkpoint(candidate, dev)
    raise FileNotFoundError(
        f"No checkpoint found in {layout.checkpoints_dir}. Expected {layout.ckpt_best.name} or {layout.ckpt_last.name}."
    )


def _deep_merge(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), MutableMapping):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def read_run_manifest(layout: RunLayout) -> Dict[str, Any]:
    path = make_run_layout(layout.run_dir).run_manifest_json
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_manifest(layout: RunLayout, manifest: dict) -> None:
    ensure_run_layout(layout.run_dir)
    payload = dict(manifest)
    payload.setdefault("schema_version", RUN_MANIFEST_SCHEMA_VERSION)
    payload.setdefault("updated_at_utc", _utcnow_iso())
    atomic_write_json(layout.run_manifest_json, payload)


def update_run_manifest(layout: RunLayout, updates: dict) -> None:
    current = read_run_manifest(layout)
    if not current:
        current = {"schema_version": RUN_MANIFEST_SCHEMA_VERSION}
    _deep_merge(current, updates)
    current["updated_at_utc"] = _utcnow_iso()
    atomic_write_json(layout.run_manifest_json, current)


def append_run_evaluation(layout: RunLayout, evaluation: Mapping[str, Any]) -> None:
    manifest = read_run_manifest(layout)
    evaluations = list(manifest.get("evaluations") or [])
    evaluations.append(dict(evaluation))
    manifest["evaluations"] = evaluations
    manifest.setdefault("schema_version", RUN_MANIFEST_SCHEMA_VERSION)
    manifest["updated_at_utc"] = _utcnow_iso()
    atomic_write_json(layout.run_manifest_json, manifest)


def resolve_run_dir(path_like: Path | str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if path.is_file():
        return path.parent.parent
    if path.is_dir() and path.name == "checkpoints":
        return path.parent
    return path


def resolve_resume_checkpoint(
    path_like: Path | str,
    *,
    prefer: str = "last",
    device: Optional[torch.device] = None,
) -> Tuple["RunLayout", Path, dict]:
    """Resolve a resume target into ``(layout, checkpoint_path, checkpoint_payload)``.

    Centralizes all resume path handling so the engine never re-implements it.

    Accepts:
      * a direct ``.pt`` checkpoint file (run_dir inferred via :func:`resolve_run_dir`),
      * a ``checkpoints/`` directory (parent is the run dir),
      * a run directory (``checkpoints/ckpt_last.pt`` by default, or
        ``ckpt_best.pt`` when ``prefer='best'``).

    The checkpoint is loaded and schema-validated through :func:`load_checkpoint`.
    """
    dev = device or torch.device("cpu")
    prefer_l = str(prefer).strip().lower()
    if prefer_l not in {"last", "best"}:
        raise ValueError("resume prefer must be 'last' or 'best'")

    p = Path(path_like).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Resume path does not exist: {p}")

    run_dir = resolve_run_dir(p)
    layout = make_run_layout(run_dir)

    if p.is_file():
        ckpt_path = p
    else:
        order = (
            [layout.ckpt_best, layout.ckpt_last]
            if prefer_l == "best"
            else [layout.ckpt_last, layout.ckpt_best]
        )
        ckpt_path = next((c for c in order if c.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found to resume from in {layout.checkpoints_dir}. "
                f"Expected {layout.ckpt_last.name} or {layout.ckpt_best.name}."
            )

    payload = load_checkpoint(ckpt_path, dev)
    return layout, ckpt_path, payload


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python / NumPy / torch (CPU + CUDA) RNG state for resume.

    Returns a pickle/torch-saveable dict. Restoration is best-effort and
    epoch-level: it does NOT guarantee bitwise-identical DataLoader worker
    ordering, only that the core generators continue from a consistent state.
    """
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    except Exception as exc:  # pragma: no cover - hardware dependent
        logger.warning("capture_rng_state: could not capture CUDA RNG state: %s", exc)
    return state


def restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`.

    Tolerant of ``None`` and of partial/foreign payloads; never raises so a
    resume is not aborted by an RNG-restore hiccup.
    """
    if not state:
        return
    py = state.get("python")
    if py is not None:
        try:
            random.setstate(tuple(py) if isinstance(py, list) else py)
        except Exception as exc:  # pragma: no cover
            logger.warning("restore_rng_state: python RNG restore failed: %s", exc)
    nps = state.get("numpy")
    if nps is not None:
        try:
            np.random.set_state(nps)
        except Exception as exc:  # pragma: no cover
            logger.warning("restore_rng_state: numpy RNG restore failed: %s", exc)
    tc = state.get("torch_cpu")
    if tc is not None:
        try:
            torch.set_rng_state(tc.cpu().to(torch.uint8) if hasattr(tc, "cpu") else tc)
        except Exception as exc:  # pragma: no cover
            logger.warning("restore_rng_state: torch CPU RNG restore failed: %s", exc)
    cuda = state.get("torch_cuda_all")
    if cuda is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("restore_rng_state: torch CUDA RNG restore failed: %s", exc)


def canonical_scaler_payload(scaler: Any) -> Dict[str, Any]:
    if isinstance(scaler, dict):
        return _json_safe(scaler)
    if dataclasses.is_dataclass(scaler):
        return _json_safe(asdict(scaler))
    raise TypeError("Unsupported scaler payload type.")


def write_scaler_json(layout: RunLayout, scaler: Any) -> Dict[str, Any]:
    payload = canonical_scaler_payload(scaler)
    atomic_write_json(layout.scaler_json, payload)
    return {
        "scaler_hash": _compute_payload_sha256(payload),
        "scaler_file_sha256": compute_file_sha256(layout.scaler_json),
        "scaler_payload": payload,
    }


def _scaler_pack_from_payload(payload: Mapping[str, Any], *, device: torch.device, dtype: torch.dtype) -> ScalerPack:
    return ScalerPack(
        x=IsometricScaleParams(**payload["x"]),
        u=IsometricScaleParams(**payload["u"]),
        a=IsometricScaleParams(**payload["a"]),
        provenance=dict(payload.get("provenance") or {}),
    ).to_tensors(device=device, dtype=dtype)


def load_scaler_for_run(
    layout: RunLayout,
    ckpt: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[ScalerPack, Dict[str, Any]]:
    file_payload: Optional[Dict[str, Any]] = None
    if layout.scaler_json.exists():
        file_payload = json.loads(layout.scaler_json.read_text(encoding="utf-8"))

    ckpt_payload = ckpt.get("scaler")
    if not isinstance(ckpt_payload, dict):
        ckpt_payload = None
    elif not {"x", "u", "a"}.issubset(set(ckpt_payload.keys())):
        ckpt_payload = None

    if file_payload is None and ckpt_payload is None:
        raise FileNotFoundError(
            f"No scaler artifact found for run {layout.run_dir}. Expected {layout.scaler_json} or checkpoint['scaler']."
        )

    if file_payload is not None and ckpt_payload is not None:
        file_hash = _compute_payload_sha256(file_payload)
        ckpt_hash = _coerce_str_or_none(ckpt.get("scaler_hash")) or _compute_payload_sha256(ckpt_payload)
        if file_hash != ckpt_hash:
            raise RuntimeError(
                "scaler.json and checkpoint['scaler'] disagree. "
                f"scaler.json hash={file_hash}, checkpoint scaler hash={ckpt_hash}."
            )
        payload = file_payload
        source = str(layout.scaler_json)
        hash_value = file_hash
    elif file_payload is not None:
        payload = file_payload
        source = str(layout.scaler_json)
        hash_value = _compute_payload_sha256(file_payload)
    else:
        payload = ckpt_payload or {}
        source = "checkpoint_embedded"
        hash_value = _compute_payload_sha256(payload)

    scaler = _scaler_pack_from_payload(payload, device=device, dtype=dtype)
    return scaler, {
        "scaler_source": source,
        "scaler_hash": hash_value,
        "scaler_file_path": str(layout.scaler_json) if layout.scaler_json.exists() else None,
    }


def build_resolved_config(
    cfg: Any,
    dataset_meta: Any,
    model: torch.nn.Module,
    scaler: Any,
    architecture_signature: str,
) -> dict:
    cfg_dict = dict(asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg)
    ds_meta = dict(dataset_meta or {})
    scaler_payload = canonical_scaler_payload(scaler)

    cfg_mu = cfg_dict.get("resolved_mu_si")
    cfg_r_ref = cfg_dict.get("resolved_r_ref_m")
    cfg_a_sign = cfg_dict.get("resolved_a_sign")
    cfg_dict["resolved_mu_si"] = float((cfg_mu if cfg_mu is not None else ds_meta.get("mu_si", 0.0)) or 0.0)
    cfg_dict["resolved_r_ref_m"] = float((cfg_r_ref if cfg_r_ref is not None else ds_meta.get("r_ref_m", 0.0)) or 0.0)
    cfg_dict["resolved_a_sign"] = float((cfg_a_sign if cfg_a_sign is not None else ds_meta.get("a_sign", 1.0)) or 1.0)
    cfg_dict["mu_si"] = float(cfg_dict.get("resolved_mu_si"))
    cfg_dict["r_ref_m"] = float(cfg_dict.get("resolved_r_ref_m"))
    cfg_dict["architecture_signature"] = str(architecture_signature)
    cfg_dict["model_builder_version"] = str(getattr(model, "model_builder_version", MODEL_BUILDER_VERSION))
    cfg_dict["input_feature_dim"] = int(getattr(model, "input_feature_dim", cfg_dict.get("input_feature_dim", 3)))
    cfg_dict["embedding_type"] = str(getattr(model, "embedding_type", cfg_dict.get("embedding_type", "raw")))
    cfg_dict["w0_bands"] = list(getattr(model, "w0_bands", cfg_dict.get("w0_bands") or []) or []) or None
    cfg_dict["use_fourier"] = bool(cfg_dict.get("use_fourier", False))
    cfg_dict["fourier_append_raw"] = bool(cfg_dict.get("fourier_append_raw", True))
    cfg_dict["fourier_n_features"] = _coerce_int(cfg_dict.get("fourier_n_features", 0), default=0)
    cfg_dict["fourier_sigma"] = float(cfg_dict.get("fourier_sigma", 0.0))
    cfg_dict["fourier_seed"] = _coerce_int(cfg_dict.get("fourier_seed", 0), default=0)
    cfg_dict["use_sh_encoding"] = bool(cfg_dict.get("use_sh_encoding", False))
    cfg_dict["sh_encoding_degree"] = _coerce_int(cfg_dict.get("sh_encoding_degree", 4), default=4)
    cfg_dict["sh_append_raw"] = bool(cfg_dict.get("sh_append_raw", True))
    cfg_dict["use_radial_separation"] = bool(cfg_dict.get("use_radial_separation", False))
    cfg_dict["radial_append_raw"] = bool(cfg_dict.get("radial_append_raw", False))
    cfg_dict["use_radial_decay_encoding"] = bool(cfg_dict.get("use_radial_decay_encoding", False))
    cfg_dict["radial_decay_max_power"] = _coerce_int(cfg_dict.get("radial_decay_max_power", 4), default=4)
    cfg_dict["radial_decay_append_raw"] = bool(cfg_dict.get("radial_decay_append_raw", True))
    cfg_dict["use_physical_radial_decay_encoding"] = bool(
        cfg_dict.get("use_physical_radial_decay_encoding", False)
    )
    cfg_dict["physical_radial_decay_max_power"] = _coerce_int(
        cfg_dict.get("physical_radial_decay_max_power", 4),
        default=4,
    )
    cfg_dict["physical_radial_decay_append_raw"] = bool(
        cfg_dict.get("physical_radial_decay_append_raw", True)
    )
    cfg_dict["physical_radial_decay_include_unit"] = bool(
        cfg_dict.get("physical_radial_decay_include_unit", True)
    )
    cfg_dict["physical_radial_decay_include_r_scaled"] = bool(
        cfg_dict.get("physical_radial_decay_include_r_scaled", True)
    )
    cfg_dict["x_scale_m"] = float(
        cfg_dict.get("x_scale_m", (scaler_payload.get("x") or {}).get("scale", 0.0)) or 0.0
    )
    cfg_dict["model_preset"] = str(cfg_dict.get("model_preset", "custom"))
    cfg_dict["runtime_model_kind"] = str(cfg_dict.get("runtime_model_kind", "potential_autograd"))
    cfg_dict["n_bands"] = _coerce_int(cfg_dict.get("n_bands", 1), default=1)
    cfg_dict["degree_min"] = _coerce_int(cfg_dict.get("degree_min", ds_meta.get("degree_min", -1)), default=-1)
    cfg_dict["degree_max"] = _coerce_int(cfg_dict.get("degree_max", ds_meta.get("degree_max", -1)), default=-1)
    cfg_dict["target_mode"] = _coerce_str_or_none(cfg_dict.get("target_mode") or ds_meta.get("target_mode")) or "full"
    cfg_dict["unit_system"] = _coerce_str_or_none(cfg_dict.get("unit_system") or ds_meta.get("unit_system")) or "unknown"
    cfg_dict["central_body"] = _coerce_str_or_none(cfg_dict.get("central_body") or ds_meta.get("central_body")) or "moon"
    cfg_dict["scaler_kind"] = "isometric"
    cfg_dict["base_potential_kind"] = "point_mass"
    cfg_dict["st_lrps_version"] = str(cfg_dict.get("st_lrps_version", "st_lrps_lunar_hybrid_fourier_v1"))
    cfg_dict["dataset_meta"] = dict(ds_meta)
    cfg_dict["scaler_summary"] = {
        "x_scale": float((scaler_payload.get("x") or {}).get("scale", 0.0) or 0.0),
        "u_scale": float((scaler_payload.get("u") or {}).get("scale", 0.0) or 0.0),
        "a_scale": float((scaler_payload.get("a") or {}).get("scale", 0.0) or 0.0),
    }
    dataset_contract = (
        cfg_dict.get("dataset_contract")
        or (cfg_dict.get("dataset_meta") or {}).get("dataset_contract")
        or _build_dataset_block_from_cfg(cfg_dict)
    )
    cfg_dict["dataset_contract"] = _json_safe(dataset_contract)
    cfg_dict["training_config_hash"] = _compute_payload_sha256(
        {k: v for k, v in cfg_dict.items() if k not in {"artifact_contract", "training_config_hash"}}
    )
    cfg_dict["artifact_contract"] = ArtifactContract.from_legacy_config(
        cfg_dict,
        scaler_payload=scaler_payload,
        dataset_contract=cfg_dict["dataset_contract"],
        architecture_signature=architecture_signature,
    ).to_dict()
    return _json_safe(cfg_dict)


def build_artifact_contract(
    cfg: Mapping[str, Any],
    *,
    scaler_payload: Optional[Mapping[str, Any]] = None,
    dataset_contract: Optional[Mapping[str, Any]] = None,
    architecture_signature: Optional[str] = None,
) -> ArtifactContract:
    """Build an :class:`ArtifactContract` from resolved training artifacts."""

    return ArtifactContract.from_legacy_config(
        cfg,
        scaler_payload=scaler_payload,
        dataset_contract=dataset_contract,
        architecture_signature=architecture_signature,
    )


def validate_checkpoint_contract(
    ckpt: Mapping[str, Any],
    *,
    cfg: Optional[Mapping[str, Any]] = None,
    scaler_payload: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
    allow_legacy_contract: bool = False,
) -> Dict[str, Any]:
    """Read and validate the artifact contract embedded in a checkpoint."""

    cfg_payload = dict(cfg or ckpt.get("config") or {})
    source = "checkpoint"
    legacy = False
    raw_contract = ckpt.get("artifact_contract")
    if not isinstance(raw_contract, dict):
        raw_contract = cfg_payload.get("artifact_contract")
        source = "config"
    if isinstance(raw_contract, dict):
        contract = ArtifactContract.from_dict(raw_contract)
    else:
        legacy = True
        source = "legacy_inferred"
        if strict and not allow_legacy_contract:
            raise ArtifactContractError(
                "Checkpoint is missing artifact_contract. Strict runtime loading requires a "
                "versioned artifact contract; pass allow_legacy_contract=True only for "
                "compatibility inspection of old runs."
            )
        try:
            contract = ArtifactContract.from_legacy_config(
                cfg_payload,
                scaler_payload=(
                    scaler_payload
                    if isinstance(scaler_payload, Mapping)
                    else ckpt.get("scaler") if isinstance(ckpt.get("scaler"), dict) else None
                ),
                dataset_contract=ckpt.get("dataset") if isinstance(ckpt.get("dataset"), dict) else None,
                architecture_signature=(ckpt.get("architecture") or {}).get("signature"),
            )
        except Exception as exc:
            if strict and not allow_legacy_contract:
                raise ArtifactContractError(
                    "Checkpoint is missing artifact_contract and its legacy metadata could not be "
                    f"promoted to a validated contract: {exc}. Pass allow_legacy_contract=True "
                    "only for inspection of old runs."
                ) from exc
            raise
    if not legacy and strict:
        expected_cfg = dict(cfg_payload)
        expected_cfg.pop("artifact_contract", None)
        expected_dataset = (
            expected_cfg.get("dataset_contract")
            or ckpt.get("dataset_contract")
            or ckpt.get("dataset")
        )
        try:
            expected = ArtifactContract.from_legacy_config(
                expected_cfg,
                scaler_payload=(
                    scaler_payload
                    if isinstance(scaler_payload, Mapping)
                    else ckpt.get("scaler") if isinstance(ckpt.get("scaler"), dict) else None
                ),
                dataset_contract=expected_dataset if isinstance(expected_dataset, Mapping) else None,
                architecture_signature=(ckpt.get("architecture") or {}).get("signature")
                if isinstance(ckpt.get("architecture"), Mapping)
                else expected_cfg.get("architecture_signature"),
            )
            compatibility = contract.compatibility_report(expected, strict_domain=False)
        except Exception as exc:
            raise ArtifactContractError(
                f"artifact_contract could not be cross-checked against checkpoint/config metadata: {exc}"
            ) from exc
        if compatibility["errors"]:
            raise ArtifactContractError(
                "artifact_contract disagrees with checkpoint/config metadata: "
                + "; ".join(str(item) for item in compatibility["errors"])
            )
    return {
        "artifact_contract": contract.to_dict(),
        "contract_source": source,
        "legacy_contract": legacy,
    }


def read_artifact_contract(
    run_dir: Path | str,
    *,
    prefer: str = "best",
    device: Optional[torch.device] = None,
    strict: bool = True,
    allow_legacy_contract: bool = False,
) -> ArtifactContract:
    """Load the preferred checkpoint for a run and return its artifact contract."""

    layout = make_run_layout(resolve_run_dir(run_dir))
    _, ckpt = load_best_or_last(layout, prefer=prefer, device=device or torch.device("cpu"))
    cfg = dict(ckpt.get("config") or {})
    if layout.config_json.exists():
        try:
            cfg = json.loads(layout.config_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    scaler_payload = None
    try:
        scaler, _ = load_scaler_for_run(layout, ckpt, device=device or torch.device("cpu"))
        scaler_payload = canonical_scaler_payload(scaler)
    except Exception:
        scaler_payload = None
    report = validate_checkpoint_contract(
        ckpt,
        cfg=cfg,
        scaler_payload=scaler_payload,
        strict=strict,
        allow_legacy_contract=allow_legacy_contract,
    )
    return ArtifactContract.from_dict(report["artifact_contract"])


def compare_artifact_contracts(
    artifact: ArtifactContract | Mapping[str, Any],
    requested: ArtifactContract | Mapping[str, Any],
    *,
    strict_domain: bool = False,
) -> Dict[str, Any]:
    """Return a machine-readable compatibility report for two contracts."""

    left = artifact if isinstance(artifact, ArtifactContract) else ArtifactContract.from_dict(artifact)
    return left.compatibility_report(requested, strict_domain=strict_domain)


def verify_critical_config_fields_match(config_payload: Mapping[str, Any], ckpt_config: Mapping[str, Any]) -> None:
    mismatches = []
    for key in CRITICAL_CONFIG_FIELDS:
        if key not in config_payload and key not in ckpt_config:
            continue
        if _json_safe(config_payload.get(key)) != _json_safe(ckpt_config.get(key)):
            mismatches.append((key, config_payload.get(key), ckpt_config.get(key)))
    if mismatches:
        details = ", ".join(f"{key}={left!r}!={right!r}" for key, left, right in mismatches)
        raise RuntimeError(
            "config.json and checkpoint['config'] disagree on architecture-critical fields: "
            + details
        )


def build_checkpoint_payload(
    *,
    kind: Literal["best", "last", "epoch"],
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    cfg: Mapping[str, Any],
    scaler: Any,
    train_stats: Mapping[str, Any],
    val_stats: Mapping[str, Any],
    dataset_meta: Mapping[str, Any],
    architecture_signature: str,
    global_step: Optional[int],
) -> dict:
    cfg_dict = dict(cfg)
    scaler_payload = canonical_scaler_payload(scaler)
    architecture = _build_architecture_block_from_cfg(cfg_dict)
    architecture["signature"] = architecture_signature

    dataset = _build_dataset_block_from_cfg(cfg_dict)
    dataset.update(
        {
            "dataset_name": _coerce_str_or_none(cfg_dict.get("dataset_name")) or dataset.get("dataset_name") or "data",
            "central_body": _coerce_str_or_none(cfg_dict.get("central_body")) or dataset.get("central_body") or "moon",
            "target_mode": _coerce_str_or_none(cfg_dict.get("target_mode")) or dataset.get("target_mode") or "full",
            "degree_min": _coerce_int(cfg_dict.get("degree_min", dataset.get("degree_min", -1)), default=-1),
            "degree_max": _coerce_int(cfg_dict.get("degree_max", dataset.get("degree_max", -1)), default=-1),
            "unit_system": _coerce_str_or_none(cfg_dict.get("unit_system")) or dataset.get("unit_system") or "unknown",
            "mu_si": float(cfg_dict.get("resolved_mu_si", dataset.get("mu_si", 0.0)) or 0.0),
            "r_ref_m": float(cfg_dict.get("resolved_r_ref_m", dataset.get("r_ref_m", 0.0)) or 0.0),
            "alt_min_km": _coerce_float_or_none((dataset_meta or {}).get("alt_min_km")),
            "alt_max_km": _coerce_float_or_none((dataset_meta or {}).get("alt_max_km")),
            "derivative_convention_version": _coerce_str_or_none((dataset_meta or {}).get("derivative_convention_version")),
            "a_sign_convention": _coerce_str_or_none((dataset_meta or {}).get("a_sign_convention")),
            "target_contract": cfg_dict.get("target_contract"),
        }
    )

    score = _coerce_float_or_none(val_stats.get("val_checkpoint_score"))
    if score is None:
        score = _coerce_float_or_none(val_stats.get("loss"))
    scoring = {
        "best_metric": _coerce_str_or_none(cfg_dict.get("best_metric")) or "val_total_loss",
        "score": score if score is not None else 0.0,
        "formula": _coerce_str_or_none(val_stats.get("checkpoint_formula"))
                   or _coerce_str_or_none((cfg_dict.get("checkpoint_selection") or {}).get("formula")),
        "lower_is_better": bool((cfg_dict.get("checkpoint_selection") or {}).get("lower_is_better", True)),
        "eligible_for_best": bool(val_stats.get("eligible_for_best", True)),
        "val_loss": float(val_stats.get("loss", 0.0) or 0.0),
        "val_base_loss": _coerce_float_or_none(val_stats.get("val_base_loss")),
        "val_total_loss": _coerce_float_or_none(val_stats.get("val_total_loss")),
        "val_physics_loss": _coerce_float_or_none(val_stats.get("val_physics_loss")),
        "val_loss_u": _coerce_float_or_none(val_stats.get("mse_u")),
        "val_loss_a": _coerce_float_or_none(val_stats.get("mse_a")),
        "val_loss_dir": _coerce_float_or_none(val_stats.get("loss_dir")),
        "loss_dir": _coerce_float_or_none(val_stats.get("loss_dir")),
        "mean_cos_sim": _coerce_float_or_none(val_stats.get("cossim_mean")),
        "mean_ang_deg": _coerce_float_or_none(val_stats.get("angular_mean_deg")),
    }

    training_state = {
        "lr": float(train_stats.get("lr", 0.0) or 0.0),
        "w_u": float(train_stats.get("w_u", 0.0) or 0.0),
        "w_a": float(train_stats.get("w_a", 0.0) or 0.0),
        "gradnorm_status": _coerce_str_or_none(train_stats.get("gradnorm_status")),
        "accel_factor": float(train_stats.get("accel_factor", 1.0) or 1.0),
        "lambda_dir_eff": float(train_stats.get("lambda_dir_eff", 0.0) or 0.0),
        "rng_state": train_stats.get("rng_state"),
        # Resume-only: GradNormWeights.state_dict() so a resumed run continues the
        # frozen NTK / EMA loss-weighting instead of recomputing it.
        "gradnorm_weights": train_stats.get("gradnorm_weights"),
    }

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "epoch": int(epoch),
        "epoch_display": int(epoch) + 1,
        "global_step": _coerce_int_or_none(global_step),
        "model_state_dict": model.state_dict(),
        "model": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": (scheduler.state_dict() if hasattr(scheduler, "state_dict") else scheduler),
        "config": dict(cfg_dict),
        "scaler": scaler_payload,
        "scaler_hash": _compute_payload_sha256(scaler_payload),
        "architecture": architecture,
        "dataset": dataset,
        "dataset_contract": cfg_dict.get("dataset_contract") or dataset,
        "artifact_contract": (
            ArtifactContract.from_legacy_config(
                cfg_dict,
                scaler_payload=scaler_payload,
                dataset_contract=cfg_dict.get("dataset_contract") or dataset,
                architecture_signature=architecture_signature,
            ).to_dict()
        ),
        "resolved_config": dict(cfg_dict),
        "training_config_hash": cfg_dict.get("training_config_hash") or _compute_payload_sha256(cfg_dict),
        "dataset_hash": (cfg_dict.get("dataset_contract") or dataset).get("dataset_sha256")
        if isinstance((cfg_dict.get("dataset_contract") or dataset), Mapping)
        else None,
        "model_builder_version": cfg_dict.get("model_builder_version", MODEL_BUILDER_VERSION),
        "scoring": scoring,
        "training_state": training_state,
        "created_at_utc": _utcnow_iso(),
    }
    return validate_checkpoint_schema(payload, strict=True)


def write_command_txt(layout: RunLayout, argv: Optional[Sequence[str]] = None) -> str:
    argv = list(argv if argv is not None else [sys.executable, *sys.argv])
    command = subprocess.list2cmdline([str(v) for v in argv])
    _atomic_write_text(layout.command_txt, command + "\n")
    return command


def capture_environment_snapshot(layout: RunLayout, *, extra: Optional[Mapping[str, Any]] = None) -> Path:
    payload = {
        "created_at_utc": _utcnow_iso(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if extra:
        payload.update(dict(extra))
    path = layout.provenance_dir / "environment.json"
    atomic_write_json(path, payload)
    return path


def default_eval_output_dir(layout: RunLayout, dataset_path: Path, *, timestamp: Optional[str] = None) -> Path:
    stem = Path(dataset_path).stem or "dataset"
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return layout.evals_dir / f"eval_{stem}_{stamp}"


def write_eval_manifest(out_dir: Path, manifest: Mapping[str, Any]) -> Path:
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval_manifest.json"
    payload = dict(manifest)
    payload.setdefault("schema_version", EVAL_MANIFEST_SCHEMA_VERSION)
    payload.setdefault("created_at_utc", _utcnow_iso())
    atomic_write_json(path, payload)
    return path


def write_evaluate_summary(out_dir: Path, summary_lines: Iterable[str]) -> Path:
    path = Path(out_dir).expanduser().resolve() / "evaluate_summary.txt"
    _atomic_write_text(path, "\n".join(str(line) for line in summary_lines).rstrip() + "\n")
    return path


def reload_model_from_run_dir(
    run_dir: Path | str,
    device: torch.device,
    *,
    prefer: str = "best",
    allow_config_mismatch: bool = False,
) -> Tuple[torch.nn.Module, ScalerPack, Dict[str, Any], Dict[str, Any]]:
    layout = make_run_layout(resolve_run_dir(run_dir))
    ckpt_path, ckpt = load_best_or_last(layout, prefer=prefer, device=device)

    if layout.config_json.exists():
        cfg_json = json.loads(layout.config_json.read_text(encoding="utf-8"))
        config_source = "config_json"
    elif isinstance(ckpt.get("config"), dict):
        cfg_json = dict(ckpt["config"])
        config_source = "checkpoint"
    else:
        raise FileNotFoundError(
            f"Missing config artifact for run {layout.run_dir}. Expected {layout.config_json} or checkpoint['config']."
        )

    model, merged_cfg, report = reconstruct_model_from_artifacts(
        cfg_json,
        ckpt,
        device,
        allow_config_mismatch=allow_config_mismatch,
    )
    scaler, scaler_report = load_scaler_for_run(layout, ckpt, device=device, dtype=torch.float32)
    manifest = read_run_manifest(layout)
    contract_report = validate_checkpoint_contract(
        ckpt,
        cfg=merged_cfg,
        scaler_payload=canonical_scaler_payload(scaler),
        strict=False,
        allow_legacy_contract=True,
    )

    report.update(
        {
            "checkpoint_schema_version": ckpt.get("schema_version"),
            "checkpoint_kind": ckpt.get("kind"),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_epoch_display": ckpt.get("epoch_display"),
            "checkpoint_metric": ((ckpt.get("scoring") or {}).get("score") if isinstance(ckpt.get("scoring"), dict) else None),
            "checkpoint_hash": ckpt.get("checkpoint_hash"),
            "checkpoint_config_source": config_source if config_source == "checkpoint" else report.get("checkpoint_config_source"),
            "architecture_signature": report.get("architecture_signature")
            or (ckpt.get("architecture") or {}).get("signature")
            or merged_cfg.get("architecture_signature"),
            "run_manifest_path": str(layout.run_manifest_json) if layout.run_manifest_json.exists() else None,
            "run_manifest": manifest or None,
            "scaler_source": scaler_report["scaler_source"],
            "scaler_hash": scaler_report["scaler_hash"],
            "w0_bands": merged_cfg.get("w0_bands"),
            "input_feature_dim": merged_cfg.get("input_feature_dim"),
            "embedding_type": merged_cfg.get("embedding_type"),
            "artifact_contract": contract_report.get("artifact_contract"),
            "artifact_contract_source": contract_report.get("contract_source"),
            "legacy_contract": contract_report.get("legacy_contract"),
        }
    )
    return model, scaler, merged_cfg, report


def _critical_config_mismatches(config_payload: Mapping[str, Any], ckpt_config: Mapping[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    mismatches: Dict[str, Tuple[Any, Any]] = {}
    for key in CRITICAL_CONFIG_FIELDS:
        if _json_safe(config_payload.get(key)) != _json_safe(ckpt_config.get(key)):
            mismatches[key] = (config_payload.get(key), ckpt_config.get(key))
    return mismatches


def inspect_run(run_dir: Path | str) -> Dict[str, Any]:
    layout = make_run_layout(resolve_run_dir(run_dir))
    summary: Dict[str, Any] = {
        "run_dir": str(layout.run_dir),
        "config_json": str(layout.config_json),
        "scaler_json": str(layout.scaler_json),
        "run_manifest_json": str(layout.run_manifest_json),
        "classification": "usable_current_schema",
    }

    cfg_json = json.loads(layout.config_json.read_text(encoding="utf-8")) if layout.config_json.exists() else None
    scaler_json = json.loads(layout.scaler_json.read_text(encoding="utf-8")) if layout.scaler_json.exists() else None
    summary["config_present"] = cfg_json is not None
    summary["scaler_present"] = scaler_json is not None

    loaded = {}
    for name, path in (("best", layout.ckpt_best), ("last", layout.ckpt_last)):
        if path.exists():
            loaded[name] = load_checkpoint(path, torch.device("cpu"))
            loaded[name]["path"] = str(path)
        else:
            loaded[name] = None
    summary["checkpoints"] = {
        name: (
            {
                "path": item["path"],
                "schema_version": item.get("schema_version"),
                "kind": item.get("kind"),
                "epoch": item.get("epoch"),
                "epoch_display": item.get("epoch_display"),
                "architecture_signature": (item.get("architecture") or {}).get("signature"),
                "w0_bands": (item.get("architecture") or {}).get("w0_bands"),
                "checkpoint_hash": item.get("checkpoint_hash"),
            }
            if item is not None else None
        )
        for name, item in loaded.items()
    }

    config_mismatches = {}
    if cfg_json is not None:
        for name, item in loaded.items():
            if item is not None:
                config_mismatches[name] = _critical_config_mismatches(cfg_json, item.get("config") or {})
    summary["config_checkpoint_mismatches"] = config_mismatches

    scaler_hash = _compute_payload_sha256(scaler_json) if isinstance(scaler_json, dict) else None
    ckpt_scaler_hashes = {
        name: (
            _coerce_str_or_none((item or {}).get("scaler_hash"))
            or _compute_payload_sha256((item or {}).get("scaler") or {})
            if item is not None and isinstance((item or {}).get("scaler"), dict) else None
        )
        for name, item in loaded.items()
    }
    summary["scaler_hash"] = scaler_hash
    summary["checkpoint_scaler_hashes"] = ckpt_scaler_hashes
    summary["scaler_consistent"] = {
        name: (scaler_hash == hash_value if scaler_hash and hash_value else None)
        for name, hash_value in ckpt_scaler_hashes.items()
    }

    try:
        _, _, _, report = reload_model_from_run_dir(layout.run_dir, torch.device("cpu"))
        summary["canonical_reload"] = {
            "ok": True,
            "checkpoint_path": report.get("checkpoint_path"),
            "checkpoint_kind": report.get("checkpoint_kind"),
            "architecture_signature": report.get("architecture_signature"),
        }
    except Exception as exc:
        summary["canonical_reload"] = {"ok": False, "error": str(exc)}
        if "architecture" in str(exc).lower():
            summary["classification"] = "architecture_mismatch"
        elif cfg_json is None:
            summary["classification"] = "missing_config"
        elif scaler_json is None:
            summary["classification"] = "missing_scaler"
        else:
            summary["classification"] = "broken_checkpoint"

    if summary["classification"] == "usable_current_schema":
        if any(item and item.get("schema_version") != CHECKPOINT_SCHEMA_VERSION for item in loaded.values() if item is not None):
            summary["classification"] = "usable_legacy"
    return summary


def migrate_run(
    run_dir: Path | str,
    *,
    write_normalized_checkpoints: bool = False,
) -> Dict[str, Any]:
    layout = ensure_run_layout(resolve_run_dir(run_dir))
    inspection = inspect_run(layout.run_dir)

    manifest = read_run_manifest(layout)
    if not manifest:
        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": layout.run_dir.name,
            "created_at_utc": _utcnow_iso(),
            "status": "migrated",
            "classification": inspection.get("classification"),
            "notes": ["Generated by st_lrps_artifacts --migrate-run"],
        }
    write_run_manifest(layout, manifest)
    _write_checkpoints_manifest(layout)

    migrated_paths = []
    if write_normalized_checkpoints:
        migrated_dir = layout.checkpoints_dir / "migrated"
        migrated_dir.mkdir(parents=True, exist_ok=True)
        for src in (layout.ckpt_best, layout.ckpt_last):
            if not src.exists():
                continue
            ckpt = load_checkpoint(src, torch.device("cpu"))
            ckpt["schema_version"] = CHECKPOINT_SCHEMA_VERSION
            normalized_path = migrated_dir / f"{src.stem}_v2.pt"
            _atomic_torch_save(normalized_path, validate_checkpoint_schema(ckpt, strict=False))
            migrated_paths.append(str(normalized_path))

    inspection["migrated_checkpoints"] = migrated_paths
    return inspection


def _print_inspection(summary: Mapping[str, Any]) -> None:
    print(f"run_dir: {summary.get('run_dir')}")
    print(f"classification: {summary.get('classification')}")
    print(f"config_present: {summary.get('config_present')}")
    print(f"scaler_present: {summary.get('scaler_present')}")
    print(f"scaler_hash: {summary.get('scaler_hash')}")
    for name, item in (summary.get("checkpoints") or {}).items():
        print(f"{name}: {item}")
    print(f"config_checkpoint_mismatches: {summary.get('config_checkpoint_mismatches')}")
    print(f"scaler_consistent: {summary.get('scaler_consistent')}")
    print(f"canonical_reload: {summary.get('canonical_reload')}")


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect or migrate ST-LRPS run artifacts.")
    ap.add_argument("--inspect-run", default=None, help="Inspect a run directory/checkpoint/checkpoints dir.")
    ap.add_argument("--migrate-run", default=None, help="Create canonical manifests for an old run.")
    ap.add_argument(
        "--write-normalized-checkpoints",
        action="store_true",
        help="When used with --migrate-run, also write migrated/ckpt_*_v2.pt copies.",
    )
    args = ap.parse_args(argv)
    if not args.inspect_run and not args.migrate_run:
        ap.error("Pass --inspect-run <run_dir> or --migrate-run <run_dir>.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_cli(argv)
    if args.inspect_run:
        _print_inspection(inspect_run(args.inspect_run))
    if args.migrate_run:
        _print_inspection(
            migrate_run(
                args.migrate_run,
                write_normalized_checkpoints=bool(args.write_normalized_checkpoints),
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
