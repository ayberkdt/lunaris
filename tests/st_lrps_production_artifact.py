"""Build a small ST-LRPS run directory with the production architecture.

Same writer functions, file layout and architecture family as a real training run
(two-band concat_shared SIREN with residual blocks, physical radial-decay input
encoding, rho**p output envelope), only narrower, with random weights. Used by the
equation and artifact-contract tests so they exercise what the runtime loads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lunaris.common.constants import MU_MOON

R_REF_M = 1.738e6
X_SCALE_M = R_REF_M + 1.0e6
ENVELOPE_POWER = 26.0


def make_production_like_run(
    run_dir: Path,
    *,
    degree_min: int = 1,
    hidden: int = 16,
    seed: int = 7,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write config.json, scaler.json and checkpoints for a production-like run.

    ``degree_min=1`` keeps the runtime baseline a point mass (no coefficient file);
    the production runs use 25 with the SH file from the dataset contract.
    """
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        atomic_write_json,
        build_checkpoint_payload,
        build_resolved_config,
        ensure_run_layout,
        save_checkpoint,
        write_scaler_json,
    )
    from lunaris.surrogate.st_lrps.networks.models import (
        build_model_from_config,
        compute_architecture_signature,
        compute_harmonic_w0_bands,
    )
    from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
    from lunaris.surrogate.st_lrps.training.config import TrainConfig, apply_model_preset

    torch.manual_seed(seed)
    layout = ensure_run_layout(run_dir)
    cfg = TrainConfig(
        data="d.h5", out=str(run_dir), hidden=hidden, depth=5, n_bands=2,
        use_residual_blocks=True, model_preset="recommended_physical_radial_decay",
        output_envelope_power=ENVELOPE_POWER, degree_min=degree_min, degree_max=200,
        x_scale_m=X_SCALE_M, resolved_r_ref_m=R_REF_M,
    )
    cfg = apply_model_preset(cfg)
    cfg.w0_bands = [float(w) for w in compute_harmonic_w0_bands(2, degree_min, 200)]
    for key, value in (config_overrides or {}).items():
        setattr(cfg, key, value)
    model = build_model_from_config(cfg, in_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    sig = compute_architecture_signature(cfg)
    scaler = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=X_SCALE_M),
        u=IsometricScaleParams(mean=[-1.5e-4], scale=8.8),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.1e-4),
    )
    dataset_meta = {
        "mu_si": float(MU_MOON), "r_ref_m": R_REF_M, "degree_min": degree_min,
        "degree_max": 200, "target_mode": "residual", "unit_system": "si",
        "central_body": "moon", "altitude_min_km": 100.0, "altitude_max_km": 1000.0,
    }
    resolved = build_resolved_config(cfg, dataset_meta, model, scaler, sig)
    resolved["best_epoch"] = 3
    resolved["altitude_min_km"] = 100.0
    resolved["altitude_max_km"] = 1000.0
    atomic_write_json(layout.config_json, resolved)
    write_scaler_json(layout, scaler)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for kind, epoch in (("last", 4), ("best", 3)):
        payload = build_checkpoint_payload(
            kind=kind, epoch=epoch, model=model, optimizer=opt, scheduler={"kind": "none"},
            cfg=resolved, scaler=scaler, train_stats={}, val_stats={"loss": 0.5},
            dataset_meta=dataset_meta, architecture_signature=sig, global_step=40,
        )
        save_checkpoint(layout, kind=kind, payload=payload, epoch=epoch)
    return layout.run_dir
