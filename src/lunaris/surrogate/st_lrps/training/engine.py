# -*- coding: utf-8 -*-
"""
Training engine for the lunar scalar potential surrogate.

The engine is the orchestration layer: it receives a validated ``TrainConfig``,
loads lunar residual clouds, fits or restores scalers, builds the neural field,
executes the Sobolev training loop, and writes checkpoints/metrics.

Design notes
------------
* The learned quantity is scalar residual potential ``dU``.
* Residual acceleration ``da`` is computed from the autograd gradient of ``dU``.
* Validation keeps gradients enabled because acceleration metrics require that
  derivative path.
* Best-checkpoint selection can be delayed until direction loss is active, so a
  physically incomplete early epoch does not become the preferred checkpoint.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.training.config import TrainConfig, apply_model_preset
from lunaris.surrogate.st_lrps.training.config_summary import build_experiment_feature_summary
from lunaris.surrogate.st_lrps.data.datasets import (
    DTYPE, BlockShuffleSampler, DatasetMeta, H5BlockDataset, TensorMemoryDataset,
    _discover_dataset_name, _resolve_loader_worker_count,
    build_dataset_contract,
    _resolve_lunar_dataset_contract, collate_xyz_u_a, infer_a_sign_from_data,
    validate_training_dataset_convention,
)
from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract
from lunaris.surrogate.st_lrps.data.dataset_validation import validate_dataset_file
from lunaris.surrogate.st_lrps.data.splits import build_split_manifest, split_dataset_indices, write_split_manifest
from lunaris.surrogate.st_lrps.artifacts.manager import (
    atomic_write_json,
    build_checkpoint_payload,
    build_resolved_config,
    capture_environment_snapshot,
    capture_rng_state,
    compute_file_sha256,
    compute_payload_sha256,
    ensure_run_layout,
    read_run_manifest,
    resolve_resume_checkpoint,
    restore_rng_state,
    update_run_manifest,
    save_checkpoint,
    verify_critical_config_fields_match,
    write_command_txt,
    write_scaler_json,
    write_run_manifest,
)
from lunaris.surrogate.st_lrps.training.losses import (
    GradNormWeights, LossCurriculum, SobolevLoss, _direction_loss_factor,
    collocation_laplacian_loss,
)
from lunaris.surrogate.st_lrps.training.periodic_eval import (
    completed_periodic_eval_epochs,
    resolve_periodic_eval_plan,
    run_periodic_eval,
)
from lunaris.surrogate.st_lrps.training.metrics import (
    HISTORY_FIELDNAMES,
    checkpoint_selection_block,
    compute_checkpoint_score,
    flatten_epoch_metrics,
    format_batch_summary,
    format_epoch_summary,
    normalize_best_metric,
)
from lunaris.surrogate.st_lrps.networks.models import (
    _compute_harmonic_w0_bands, _get_output_head_params, build_model_from_config,
    MODEL_BUILDER_VERSION, compute_architecture_signature,
)
from lunaris.surrogate.st_lrps.shared.scaling import ScalerPack, fit_scaler_streaming
from lunaris.surrogate.st_lrps.shared.contracts import TargetContract

logger = logging.getLogger(__name__)


def _log_section(title: str, values: Mapping[str, Any]) -> None:
    logger.info(f"=== {title} ===")
    for key, value in values.items():
        logger.info(f"{str(key):24s}: {value}")


def set_seed(seed: int = 42, *, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Fixes all random number generator (RNG) seeds for reproducibility.

    ``deterministic`` / ``benchmark`` control cuDNN behavior. Defaults preserve
    the historical deterministic configuration; pass ``deterministic=False`` /
    ``benchmark=True`` for throughput at the cost of run-to-run reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)

def get_device() -> torch.device:
    """
    Selects the best available hardware accelerator.
    Priority: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def safe_mkdir(p: Union[str, Path]) -> Path:
    path_obj = Path(p)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def _human_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    size = float(n)
    for unit in units[:-1]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} {units[-1]}"

def _format_seconds(seconds: float) -> str:
    """Format a duration into a compact human-readable string."""
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"

def _format_cuda_memory_mib(
    allocated_mib: int,
    reserved_mib: int,
    peak_allocated_mib: int,
    peak_reserved_mib: int,
    total_vram_mib: int,
) -> str:
    return (
        f" cuda_mem={allocated_mib}/{reserved_mib}MiB"
        f" peak={peak_allocated_mib}/{peak_reserved_mib}MiB"
        f" total={total_vram_mib}MiB"
    )


def _cuda_memory_string(device: torch.device) -> str:
    """Return compact PyTorch CUDA allocator memory diagnostics.

    ``allocated`` and ``reserved`` are PyTorch allocator values, not total
    physical GPU VRAM and not the same accounting shown by ``nvidia-smi``.
    ``peak`` values are the allocator maxima since peak stats were last reset,
    which makes them more useful for batch-size decisions than the current
    post-batch allocation alone.
    """
    if device.type != "cuda":
        return ""
    try:
        mib = 1024 * 1024
        alloc_mib = int(torch.cuda.memory_allocated(device) // mib)
        reserved_mib = int(torch.cuda.memory_reserved(device) // mib)
        peak_alloc_mib = int(torch.cuda.max_memory_allocated(device) // mib)
        peak_reserved_mib = int(torch.cuda.max_memory_reserved(device) // mib)
        total_mib = int(torch.cuda.get_device_properties(device).total_memory // mib)
    except Exception:
        return ""
    return _format_cuda_memory_mib(
        alloc_mib,
        reserved_mib,
        peak_alloc_mib,
        peak_reserved_mib,
        total_mib,
    )


def _available_ram_mb() -> Optional[float]:
    """Return available system RAM in MB using psutil, or None if unavailable.

    psutil is an optional dependency: when it is missing we simply skip the
    RAM-safety check rather than failing the run.
    """
    try:
        import psutil  # optional
    except Exception:
        return None
    try:
        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        return None


def _estimate_preload_ram_mb(n_rows: int) -> float:
    """Estimate peak RAM (MB) to preload ``n_rows`` of [x,y,z,U,ax,ay,az].

    The preload path reads the whole array as float64 ``(N, 7)`` and then builds
    float32 train/val copies, so the transient peak is roughly the float64 buffer
    plus the float32 copies.
    """
    f64 = float(n_rows) * 7.0 * 8.0
    f32 = float(n_rows) * 7.0 * 4.0
    return (f64 + f32) / (1024.0 * 1024.0)


def _decide_preload(
    policy: str,
    *,
    dataset_mb: float,
    auto_preload_mb: float,
    est_ram_mb: float,
    avail_ram_mb: Optional[float],
) -> Tuple[bool, str]:
    """Resolve whether to RAM-preload the dataset and explain why.

    Returns ``(should_preload, reason)``. The 60%-of-available-RAM guard vetoes
    the ``auto`` decision; under an explicit ``always`` request it does not veto
    but emits a loud warning embedded in the reason string.
    """
    policy = str(policy).strip().lower()
    over_ram = (avail_ram_mb is not None) and (est_ram_mb > 0.60 * avail_ram_mb)

    if policy == "never":
        return False, "policy=never"
    if policy == "always":
        if over_ram:
            return True, (
                f"policy=always (WARNING: estimated {est_ram_mb:.0f} MB exceeds 60% of "
                f"available {avail_ram_mb:.0f} MB - OOM risk; honouring explicit request)"
            )
        return True, "policy=always"
    # auto
    if dataset_mb > auto_preload_mb:
        return False, (
            f"policy=auto: dataset {dataset_mb:.1f} MB > auto_preload_mb {auto_preload_mb:.1f} MB"
        )
    if over_ram:
        return False, (
            f"policy=auto: estimated {est_ram_mb:.0f} MB > 60% of available "
            f"{avail_ram_mb:.0f} MB (RAM safety veto)"
        )
    return True, (
        f"policy=auto: dataset {dataset_mb:.1f} MB <= auto_preload_mb {auto_preload_mb:.1f} MB"
    )


def _warn_batch_size_for_vram(device: torch.device, cfg: TrainConfig) -> None:
    """Advisory-only check: warn if batch_size looks large for the detected GPU.

    Sobolev training holds a second-order autograd graph (a = ∇U), so memory
    scales with batch_size, depth, and the number of multi-scale bands. This
    never changes the batch size — it only suggests using
    ``grad_accumulation_steps`` to keep the effective batch while fitting VRAM.
    """
    if device.type != "cuda":
        return
    try:
        props = torch.cuda.get_device_properties(device)
        total_gb = float(props.total_memory) / (1024.0 ** 3)
        gpu_name = props.name
    except Exception:
        return

    bs = int(cfg.batch_size)
    depth = int(getattr(cfg, "depth", 6))
    n_bands = int(getattr(cfg, "n_bands", 1))
    heavy = depth >= 6 and n_bands >= 3
    logger.info(f"CUDA device: {gpu_name} ({total_gb:.1f} GiB total VRAM)")

    suggestion = (
        "Sobolev autograd memory scales with batch_size×depth×n_bands. "
        "Prefer raising --grad-accumulation-steps over lowering the effective batch."
    )
    if total_gb <= 8.0:
        if bs > 4096:
            logger.warning(
                f"VRAM advisory: batch_size={bs} on a {total_gb:.1f} GiB GPU may OOM. "
                f"Consider --batch-size 4096 with --grad-accumulation-steps 2-4. {suggestion}"
            )
    elif total_gb <= 16.0:
        if heavy and bs >= 8192:
            logger.warning(
                f"VRAM advisory: batch_size={bs} with depth={depth}+n_bands={n_bands} on a "
                f"{total_gb:.1f} GiB GPU is borderline. If you hit OOM, use "
                f"--grad-accumulation-steps 2. {suggestion}"
            )
        elif bs > 16384:
            logger.warning(
                f"VRAM advisory: batch_size={bs} on a {total_gb:.1f} GiB GPU may be tight. {suggestion}"
            )
    else:
        if bs > 65536:
            logger.warning(
                f"VRAM advisory: batch_size={bs} is very large even for {total_gb:.1f} GiB. {suggestion}"
            )

def move_batch_to_device(
    x: torch.Tensor,
    u: torch.Tensor,
    a: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transfer a (x, u, a) batch to device with non_blocking for CUDA."""
    nb = device.type == "cuda"
    return (
        x.to(device, non_blocking=nb),
        u.to(device, non_blocking=nb),
        a.to(device, non_blocking=nb),
    )

def _laplacian_requested(cfg: TrainConfig) -> bool:
    """Return True only if the user explicitly asked for any Laplacian work.

    With the default config (use_laplacian_regularization=False,
    laplacian_mode="diagnostic", collocation_laplacian_weight=0,
    laplacian_weight=0) this is False, so normal training does ZERO Laplacian
    computation — no in-batch penalty, no collocation diagnostics, no autograd
    overhead, and no Laplacian term in the objective.
    """
    return (
        bool(getattr(cfg, "use_laplacian_regularization", False))
        or str(getattr(cfg, "laplacian_mode", "off")).strip().lower() == "train"
        or float(getattr(cfg, "collocation_laplacian_weight", 0.0)) > 0.0
        or float(getattr(cfg, "laplacian_weight", 0.0)) > 0.0
    )


class STLRPSTrainer:
    """
    Encapsulates the training state and execution logic.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        weights: "GradNormWeights",
        device: torch.device,
        cfg: TrainConfig,
        collocation_r_min_m: Optional[float] = None,
        collocation_r_max_m: Optional[float] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.weights = weights
        self.device = device
        self.cfg = cfg
        self.curriculum = LossCurriculum(
            potential_only_epochs=cfg.potential_only_epochs,
            accel_ramp_epochs=cfg.accel_ramp_epochs,
            accel_min_factor=float(getattr(cfg, "accel_min_factor", 0.05)),
        )
        # Collocation Laplacian bounds
        self.collocation_r_min_m: Optional[float] = collocation_r_min_m
        self.collocation_r_max_m: Optional[float] = collocation_r_max_m
        # Whether any Laplacian work is requested at all. When False, the default,
        # all Laplacian paths are skipped (no autograd overhead).
        self.laplacian_requested: bool = _laplacian_requested(cfg)
        _lmode = str(getattr(cfg, "laplacian_mode", "diagnostic")).strip().lower()
        if _lmode not in ("off", "diagnostic", "train"):
            _lmode = "diagnostic"
        if _lmode == "off" and bool(getattr(cfg, "use_laplacian_regularization", False)):
            _lmode = "diagnostic"
        self.laplacian_mode: str = _lmode
        
        # bfloat16 instead of float16: SIREN sin(w0 · x) overflows fp16 mantissa.
        # bfloat16 has fp32 exponent range; disable AMP entirely if unavailable.
        # Laplacian regularization now uses the Hutchinson trace estimator which only
        # requires create_graph=False for its second autodiff pass → AMP-compatible.
        self.use_amp = bool(cfg.amp and device.type == "cuda")
        if self.use_amp:
            if torch.cuda.is_bf16_supported():
                self._amp_dtype = torch.bfloat16
                # bfloat16 does NOT need GradScaler (same exponent as FP32)
                self.scaler_amp = None
            else:
                # GPU lacks bfloat16 → AMP is unsafe for PINN derivatives.
                logger.warning(
                    "GPU does not support bfloat16.  Disabling AMP to prevent "
                    "FP16 underflow/overflow in autograd derivatives (SIREN)."
                )
                self.use_amp = False
                self._amp_dtype = torch.float32
                self.scaler_amp = None
        else:
            self._amp_dtype = torch.float32
            self.scaler_amp = None

    def run_epoch(
        self,
        loader: DataLoader,
        is_train: bool,
        epoch: int,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        if isinstance(loader.sampler, BlockShuffleSampler):
            loader.sampler.set_epoch(epoch)

        self.model.train(is_train)
        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass
        accel_factor = self.curriculum.accel_factor(epoch) if is_train else 1.0

        total_loss = total_opt_loss = total_u = total_a = total_grad_norm = 0.0
        total_dir = total_cossim = total_radial = total_cross = total_lap = 0.0
        total_mask_frac = total_a_norm_mean = total_angular_mean_deg = 0.0
        total_col_lap_diag = total_col_lap_train = 0.0
        col_lap_diag_count = col_lap_train_count = 0
        col_lap_attempt_count = col_lap_fail_count = col_lap_success_count = 0
        a_norm_max = 0.0
        n_batches = 0
        optimizer_steps_done = 0
        samples_done = 0
        last_stats: Dict[str, float] = {}

        if is_train:
            lambda_dir_eff = _direction_loss_factor(epoch, self.cfg)
        else:
            lambda_dir_eff = float(max(0.0, getattr(self.cfg, "direction_loss_weight", 0.0)))
        grad_accum = max(1, int(getattr(self.cfg, "grad_accumulation_steps", 1)))

        phase = "train" if is_train else "val "
        log_every = int(max(0, self.cfg.log_every))
        total_batches_est = len(loader)
        if max_batches is not None:
            total_batches_est = min(total_batches_est, int(max_batches))

        # Auto logging frequency: ~10 progress updates per epoch, derived from the
        # batch count. The first and last batch are always logged below.
        if str(getattr(self.cfg, "log_every_mode", "fixed")).lower() == "auto":
            log_every = max(1, math.ceil(total_batches_est / 10))

        logger.info(f"Starting epoch {epoch + 1} {'train' if is_train else 'validation'} phase...")
        phase_t0 = time.perf_counter()

        with torch.set_grad_enabled(True):  # keep grads for val: a = ∇U
            for batch_idx, (xb, ub, ab) in enumerate(loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break

                xb, ub, ab = move_batch_to_device(xb, ub, ab, self.device)

                # Gradient accumulation bookkeeping
                is_last_batch = (
                    (batch_idx + 1 == len(loader))
                    or (max_batches is not None and batch_idx + 1 >= int(max_batches))
                )
                is_accum_boundary = (batch_idx + 1) % grad_accum == 0 or is_last_batch

                if is_train and batch_idx % grad_accum == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                apply_lap = (
                    is_train
                    and self.laplacian_requested
                    and bool(self.cfg.use_laplacian_regularization)
                    and int(self.cfg.laplacian_every_n_batches) > 0
                    and (batch_idx % int(self.cfg.laplacian_every_n_batches) == 0)
                )

                # GradNorm weight update only happens on optimizer steps
                allow_weight_update = bool(accel_factor > 0.0 and is_accum_boundary)

                with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self.use_amp):
                    loss, stats = self.loss_fn(
                        self.model,
                        xb,
                        ub,
                        ab,
                        self.weights,
                        is_train=is_train,
                        accel_factor=accel_factor,
                        allow_dynamic_weight_update=allow_weight_update,
                        direction_lambda=lambda_dir_eff,
                        direction_floor_abs=self.cfg.direction_loss_floor_abs,
                        use_altitude_balanced_loss=bool(self.cfg.use_altitude_balanced_loss),
                        altitude_bin_width_km=float(self.cfg.altitude_bin_width_km),
                        altitude_min_km=float(self.cfg.altitude_min_km),
                        altitude_max_km=float(self.cfg.altitude_max_km),
                        use_radial_cross_loss=bool(self.cfg.use_radial_cross_loss),
                        radial_lambda=float(self.cfg.radial_loss_weight),
                        cross_lambda=float(self.cfg.cross_loss_weight),
                        apply_laplacian=bool(apply_lap),
                        laplacian_lambda=float(self.cfg.laplacian_weight),
                        laplacian_subset_size=int(self.cfg.laplacian_subset_size),
                        laplacian_n_hutchinson=int(getattr(self.cfg, "n_hutchinson_samples", 4)),
                        laplacian_mode=self.laplacian_mode,
                    )

                # Explosion guard: stop on NaN/Inf immediately to avoid corrupt checkpoints.
                _loss_check = float(stats.get("loss_opt", loss.item()))
                if math.isnan(_loss_check) or math.isinf(_loss_check):
                    logger.error(
                        f"[{phase}] NaN/Inf loss detected at epoch={epoch+1} batch={n_batches}. "
                        "Possible derivative instability. "
                        "Suggestions: lower lr, ensure accel_min_factor>0, lower w0, increase accel_ramp_epochs. "
                        "Stopping epoch early."
                    )
                    # Return a sentinel so the caller can save a failure manifest.
                    return {
                        "loss": float("nan"), "objective_loss": float("nan"),
                        "mse_u": float("nan"), "mse_a": float("nan"),
                        "loss_dir": 0.0, "cossim_mean": 0.0,
                        "loss_radial": 0.0, "loss_cross": 0.0, "loss_laplacian": 0.0,
                        "loss_laplacian_diag": 0.0, "loss_laplacian_train": 0.0,
                        "lambda_laplacian_eff": float(getattr(self.cfg, "collocation_laplacian_weight", 0.0)),
                        "collocation_laplacian_applied": False,
                        "lambda_dir_eff": lambda_dir_eff,
                        "lr": float(self.optimizer.param_groups[0]["lr"]),
                        "w_u": float(last_stats.get("w_u", self.cfg.w_u)),
                        "w_a": float(last_stats.get("w_a", self.cfg.w_a)),
                        "w_a_raw": float(last_stats.get("w_a_raw", self.cfg.w_a)),
                        "accel_factor": float(accel_factor),
                        "grad_norm": 0.0,
                        "nan_detected": True,
                        "val_base_loss": float("nan"),
                        "val_physics_loss": float("nan"),
                        "val_total_loss": float("nan"),
                        "train_base_loss": float("nan"),
                        "train_physics_loss": float("nan"),
                        "optimizer_steps": int(optimizer_steps_done),
                        "samples_seen": int(samples_done),
                    }

                if is_train:
                    # Collocation Laplacian: computed BEFORE backward so it can be added to
                    # the loss in "train" mode, or logged only in "diagnostic" mode.
                    _col_lap_weight = float(getattr(self.cfg, "collocation_laplacian_weight", 0.0))
                    _col_lap_every = max(1, int(getattr(self.cfg, "collocation_laplacian_every", 25)))
                    _col_lap_active = (
                        self.laplacian_requested
                        and self.laplacian_mode in ("diagnostic", "train")
                        and self.collocation_r_min_m is not None
                        and self.collocation_r_max_m is not None
                        and optimizer_steps_done % _col_lap_every == 0
                    )
                    _col_lap_loss_val: Optional[torch.Tensor] = None
                    _col_lap_scalar = 0.0
                    if _col_lap_active:
                        _n_pts = max(1, int(getattr(self.cfg, "collocation_laplacian_samples",
                                                     getattr(self.cfg, "laplacian_subset_size", 512))))
                        _n_hutch = max(1, int(getattr(self.cfg, "collocation_laplacian_hutchinson_samples",
                                                       getattr(self.cfg, "n_hutchinson_samples", 4))))
                        col_lap_attempt_count += 1
                        try:
                            # Always use the ScalerPack from loss_fn for consistent scaling
                            _cl_loss = collocation_laplacian_loss(
                                self.model, self.loss_fn,
                                r_min_m=float(self.collocation_r_min_m),
                                r_max_m=float(self.collocation_r_max_m),
                                n_points=_n_pts,
                                device=self.device,
                                dtype=DTYPE,
                                n_hutchinson=_n_hutch,
                                mode=self.laplacian_mode,
                            )
                            _col_lap_scalar = float(_cl_loss.detach().item())
                            if math.isfinite(_col_lap_scalar):
                                col_lap_success_count += 1
                                if self.laplacian_mode == "train":
                                    total_col_lap_train += _col_lap_scalar
                                    col_lap_train_count += 1
                                    _col_lap_loss_val = _cl_loss
                                else:  # diagnostic
                                    total_col_lap_diag += _col_lap_scalar
                                    col_lap_diag_count += 1
                            else:
                                col_lap_fail_count += 1
                                if self.laplacian_mode == "train":
                                    raise RuntimeError(
                                        f"collocation_laplacian_loss returned non-finite "
                                        f"value {_col_lap_scalar} in train mode."
                                    )
                                logger.warning(
                                    "[train] collocation_laplacian_loss non-finite "
                                    f"({_col_lap_scalar}); skipped this step (diagnostic mode)."
                                )
                        except Exception as _col_e:
                            col_lap_fail_count += 1
                            # In train mode the Laplacian is part of the objective; a
                            # silent skip would disable the physics constraint while the
                            # logs/metrics still claim it is active. Fail loudly instead.
                            if self.laplacian_mode == "train":
                                raise RuntimeError(
                                    "collocation_laplacian_loss failed in train mode "
                                    f"(epoch={epoch+1}, batch={n_batches}): {_col_e}. "
                                    "The physics constraint cannot be silently dropped; "
                                    "fix the cause or switch laplacian_mode to 'diagnostic'/'off'."
                                ) from _col_e
                            logger.warning(f"[train] collocation_laplacian_loss failed: {_col_e}")

                    # Scale loss by accumulation steps so gradients average over the
                    # effective batch rather than summing (preserves LR invariance).
                    scaled_loss = loss / float(grad_accum)

                    # Add collocation laplacian to loss in "train" mode
                    if _col_lap_loss_val is not None and self.laplacian_mode == "train" and _col_lap_weight > 0.0:
                        scaled_loss = scaled_loss + (_col_lap_weight * _col_lap_loss_val) / float(grad_accum)
                        # NaN/Inf guard for collocation Laplacian contribution
                        _cl_check = float(scaled_loss.item())
                        if math.isnan(_cl_check) or math.isinf(_cl_check):
                            logger.error(
                                f"[train] NaN/Inf after adding collocation Laplacian at epoch={epoch+1} "
                                f"batch={n_batches}. Saving failure manifest and stopping."
                            )
                            import json as _json_mod
                            try:
                                _fm_path = Path(self.cfg.out) / "failure_manifest.json"
                                _fm_path.parent.mkdir(parents=True, exist_ok=True)
                                _fm_path.write_text(
                                    _json_mod.dumps({
                                        "epoch": epoch, "batch": n_batches,
                                        "reason": "nan_loss_after_collocation_laplacian",
                                        "collocation_laplacian_scalar": _col_lap_scalar,
                                    }, indent=2, default=str)
                                )
                            except Exception:
                                pass
                            return {
                                "loss": float("nan"), "objective_loss": float("nan"),
                                "mse_u": float("nan"), "mse_a": float("nan"),
                                "loss_dir": 0.0, "cossim_mean": 0.0,
                                "loss_radial": 0.0, "loss_cross": 0.0, "loss_laplacian": 0.0,
                                "loss_laplacian_diag": 0.0, "loss_laplacian_train": _col_lap_scalar,
                                "lambda_laplacian_eff": _col_lap_weight,
                                "collocation_laplacian_applied": True,
                                "lambda_dir_eff": lambda_dir_eff,
                                "lr": float(self.optimizer.param_groups[0]["lr"]),
                                "w_u": float(last_stats.get("w_u", self.cfg.w_u)),
                                "w_a": float(last_stats.get("w_a", self.cfg.w_a)),
                                "w_a_raw": float(last_stats.get("w_a_raw", self.cfg.w_a)),
                                "accel_factor": float(accel_factor),
                                "grad_norm": 0.0,
                                "nan_detected": True,
                                "val_base_loss": float("nan"),
                                "val_physics_loss": float("nan"),
                                "val_total_loss": float("nan"),
                                "train_base_loss": float("nan"),
                                "train_physics_loss": float("nan"),
                                "optimizer_steps": int(optimizer_steps_done),
                                "samples_seen": int(samples_done),
                            }

                    if self.use_amp and self.scaler_amp is not None:
                        self.scaler_amp.scale(scaled_loss).backward()
                        if is_accum_boundary:
                            self.scaler_amp.unscale_(self.optimizer)
                            if self.cfg.max_grad_norm > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=self.cfg.max_grad_norm
                                )
                            else:
                                grad_norm = torch.tensor(0.0, device=self.device)
                            self.scaler_amp.step(self.optimizer)
                            self.scaler_amp.update()
                            optimizer_steps_done += 1
                            total_grad_norm += float(grad_norm)
                            if float(grad_norm) > 50.0:
                                logger.warning(
                                    f"[train] grad_norm={float(grad_norm):.1f} > 50 at epoch={epoch+1} "
                                    "batch={n_batches}: possible derivative explosion. "
                                    "Consider lower lr or max_grad_norm."
                                )
                    else:
                        scaled_loss.backward()
                        if is_accum_boundary:
                            if self.cfg.max_grad_norm > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=self.cfg.max_grad_norm
                                )
                            else:
                                grad_norm = torch.tensor(0.0, device=self.device)
                            self.optimizer.step()
                            optimizer_steps_done += 1
                            total_grad_norm += float(grad_norm)
                            if float(grad_norm) > 50.0:
                                logger.warning(
                                    f"[train] grad_norm={float(grad_norm):.1f} > 50 at epoch={epoch+1} "
                                    f"batch={n_batches}: possible derivative explosion. "
                                    "Consider lower lr or max_grad_norm."
                                )

                samples_done += int(xb.shape[0])
                total_loss += float(stats["loss_ref"])
                total_opt_loss += float(stats["loss_opt"])
                total_u += float(stats["mse_u"])
                total_a += float(stats["mse_a"])
                total_dir += float(stats.get("loss_dir", 0.0))
                total_cossim += float(stats.get("cossim_mean", 1.0))
                total_angular_mean_deg += float(stats.get("angular_mean_deg", 0.0))
                total_mask_frac += float(stats.get("mask_frac", 0.0))
                total_radial += float(stats.get("loss_radial", 0.0))
                total_cross += float(stats.get("loss_cross", 0.0))
                total_lap += float(stats.get("loss_laplacian", 0.0))
                _a_norm_b = float(ab.detach().norm(dim=-1).mean().item())
                total_a_norm_mean += _a_norm_b
                a_norm_max = max(a_norm_max, _a_norm_b)
                n_batches += 1
                last_stats = stats

                if log_every > 0 and (
                    n_batches == 1
                    or n_batches % log_every == 0
                    or n_batches == total_batches_est
                ):
                    elapsed = time.perf_counter() - phase_t0
                    spb = elapsed / max(1, n_batches)
                    eta = max(0.0, spb * (total_batches_est - n_batches))
                    sps = samples_done / max(elapsed, 1e-9)
                    cur_lr = float(self.optimizer.param_groups[0]["lr"])
                    w_a_cur = float(last_stats.get("w_a_eff", last_stats.get("w_a", self.cfg.w_a)))
                    mem_str = _cuda_memory_string(self.device)
                    dir_str = (
                        f" dir={total_dir/n_batches:.3e} cossim={total_cossim/n_batches:.4f}"
                        f" ang={total_angular_mean_deg/n_batches:.2f}deg"
                        f" mask_frac={total_mask_frac/n_batches:.2f} lam_dir={lambda_dir_eff:.3e}"
                        if lambda_dir_eff > 0.0 else ""
                    )
                    extra_terms = ""
                    if bool(self.cfg.use_radial_cross_loss):
                        extra_terms += (
                            f" radial={total_radial/n_batches:.3e}"
                            f" cross={total_cross/n_batches:.3e}"
                        )
                    if bool(self.cfg.use_laplacian_regularization):
                        extra_terms += f" lap={total_lap/n_batches:.3e}"
                    if bool(self.cfg.use_altitude_balanced_loss):
                        extra_terms += " alt-balance=on"
                    # loss_opt = optimizer loss (uses accel_factor); loss_ref = full diagnostic loss
                    logger.info(
                        format_batch_summary(
                            phase=phase.strip(),
                            epoch=epoch + 1,
                            batch=n_batches,
                            total_batches=total_batches_est,
                            loss_opt=total_opt_loss / n_batches,
                            loss_ref=total_loss / n_batches,
                            loss_u=total_u / n_batches,
                            loss_a=total_a / n_batches,
                            loss_dir=(total_dir / n_batches if lambda_dir_eff > 0.0 else None),
                            lr=cur_lr,
                            eta_s=eta,
                            samples_per_s=sps,
                            memory=mem_str,
                        )
                    )

        phase_time = time.perf_counter() - phase_t0
        n_safe = max(1, n_batches)
        dir_summary = (
            f" dir={total_dir/n_safe:.3e} cossim={total_cossim/n_safe:.4f}"
            f" ang={total_angular_mean_deg/n_safe:.2f}deg"
            f" mask_frac={total_mask_frac/n_safe:.2f} lam_dir={lambda_dir_eff:.3e}"
            if lambda_dir_eff > 0.0 else ""
        )
        extra_summary = ""
        if bool(self.cfg.use_radial_cross_loss):
            extra_summary += f" radial={total_radial/n_safe:.3e} cross={total_cross/n_safe:.3e}"
        if bool(self.cfg.use_laplacian_regularization):
            extra_summary += f" lap={total_lap/n_safe:.3e}"
        if bool(self.cfg.use_altitude_balanced_loss):
            extra_summary += " alt-balance=on"
        # loss_ref is always the full reference (val uses full weight; train uses accel_factor)
        logger.info(
            f"[{phase}] epoch={epoch+1} done: {samples_done:,} samples in {_format_seconds(phase_time)}"
            f" ({phase_time / n_safe * 1000:.1f}ms/batch)"
            f" loss_opt={total_opt_loss/n_safe:.5e} loss_ref={total_loss/n_safe:.5e}"
            f" U={total_u/n_safe:.3e} a={total_a/n_safe:.3e}"
            f" a_norm_mean={total_a_norm_mean/n_safe:.3e} a_norm_max={a_norm_max:.3e}"
            f" accel_f={accel_factor:.3f}{dir_summary}{extra_summary}"
        )

        _n_col_diag = max(1, col_lap_diag_count)
        _n_col_train = max(1, col_lap_train_count)
        _col_lap_diag_avg = total_col_lap_diag / _n_col_diag if col_lap_diag_count > 0 else 0.0
        _col_lap_train_avg = total_col_lap_train / _n_col_train if col_lap_train_count > 0 else 0.0
        _col_lap_applied = (col_lap_diag_count > 0 or col_lap_train_count > 0)
        _col_lap_weight_eff = float(getattr(self.cfg, "collocation_laplacian_weight", 0.0))
        return {
            "loss": total_loss / n_safe,
            "objective_loss": total_opt_loss / n_safe,
            "mse_u": total_u / n_safe,
            "mse_a": total_a / n_safe,
            "loss_dir": total_dir / n_safe,
            "cossim_mean": total_cossim / n_safe,
            "angular_mean_deg": total_angular_mean_deg / n_safe,
            "mask_frac": total_mask_frac / n_safe,
            "a_norm_mean": total_a_norm_mean / n_safe,
            "a_norm_max": a_norm_max,
            "loss_radial": total_radial / n_safe,
            "loss_cross": total_cross / n_safe,
            "loss_laplacian": total_lap / n_safe,
            "loss_laplacian_diag": _col_lap_diag_avg,
            "loss_laplacian_train": _col_lap_train_avg,
            "lambda_laplacian_eff": _col_lap_weight_eff,
            "collocation_laplacian_applied": _col_lap_applied,
            "collocation_laplacian_attempt_count": int(col_lap_attempt_count),
            "collocation_laplacian_success_count": int(col_lap_success_count),
            "collocation_laplacian_fail_count": int(col_lap_fail_count),
            "lambda_dir_eff": lambda_dir_eff,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "w_u": float(last_stats.get("w_u", self.cfg.w_u)),
            "w_a": float(last_stats.get("w_a", self.cfg.w_a)),
            "w_a_raw": float(last_stats.get("w_a_raw", self.cfg.w_a)),
            "accel_factor": float(last_stats.get("accel_factor", accel_factor)),
            "grad_norm": total_grad_norm / n_safe,
            "val_base_loss": (total_u + total_a) / n_safe,  # U + accel MSE only
            "val_physics_loss": (total_dir + total_radial + total_cross + total_lap + _col_lap_train_avg) / n_safe,
            "val_total_loss": total_loss / n_safe,   # alias for "loss"
            "train_base_loss": (total_u + total_a) / n_safe,
            "train_physics_loss": (total_dir + total_radial + total_cross + total_lap + _col_lap_train_avg) / n_safe,
            "optimizer_steps": int(optimizer_steps_done),
            "samples_seen": int(samples_done),
        }

def _lr_multiplier_for_epoch(
    epoch: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
    t_max: Optional[int],
) -> float:
    """
    Warm up linearly, then decay with a cosine schedule to ``min_lr_ratio``.
    """

    epoch_i = max(0, int(epoch))
    total_i = max(1, int(total_epochs))
    warmup_i = max(0, int(warmup_epochs))
    min_ratio = float(min(max(float(min_lr_ratio), 0.0), 1.0))

    if warmup_i > 0 and epoch_i < warmup_i:
        return float((epoch_i + 1) / warmup_i)

    decay_total = int(t_max) if t_max is not None else total_i
    decay_total = max(warmup_i + 1, decay_total)
    denom = max(1, decay_total - warmup_i - 1)
    progress = min(1.0, max(0.0, (epoch_i - warmup_i) / float(denom)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cosine)

def _apply_lr_multiplier(optimizer: torch.optim.Optimizer, multiplier: float) -> None:
    for group in optimizer.param_groups:
        base_lr = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = base_lr * float(multiplier)

def _write_training_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    extra_fields = sorted({str(k) for row in history for k in row.keys()} - set(HISTORY_FIELDNAMES))
    fieldnames = list(HISTORY_FIELDNAMES) + extra_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _append_history_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _dataset_meta_snapshot(
    meta: DatasetMeta,
    *,
    dataset_name: str,
    data_path: Optional[Path],
    train_data_path: Optional[Path],
    val_data_path: Optional[Path],
    test_data_path: Optional[str],
    ood_data_path: Optional[str],
    target_mode: str,
    central_body: str,
    resolved_mu_si: float,
    resolved_r_ref_m: float,
) -> Dict[str, Any]:
    snapshot = {
        "schema_version": 1,
        "dataset_sha256": (
            compute_file_sha256(data_path)
            if data_path is not None and Path(data_path).exists()
            else (
                compute_file_sha256(train_data_path)
                if train_data_path is not None and Path(train_data_path).exists()
                else None
            )
        ),
        "dataset_name": str(dataset_name),
        "data_path": (str(data_path) if data_path is not None else None),
        "train_data_path": (str(train_data_path) if train_data_path is not None else None),
        "val_data_path": (str(val_data_path) if val_data_path is not None else None),
        "test_data_path": (str(test_data_path) if test_data_path else None),
        "ood_data_path": (str(ood_data_path) if ood_data_path else None),
        "target_mode": str(target_mode),
        "central_body": str(central_body),
        "mu_si": float(resolved_mu_si),
        "r_ref_m": float(resolved_r_ref_m),
        "unit_system": meta.unit_system,
        "requested_degree": meta.requested_degree,
        "degree_min": meta.degree_min,
        "degree_max": meta.degree_max,
        "alt_min_km": meta.alt_min_km,
        "alt_max_km": meta.alt_max_km,
        "columns": meta.columns,
        "a_sign_convention": meta.a_sign_convention,
        "derivative_convention_version": meta.derivative_convention_version,
        "DU_m": meta.DU_m,
        "TU_s": meta.TU_s,
        "VU_m_s": meta.VU_m_s,
        "include_potential": meta.include_potential,
        "gravity_model_path": meta.gravity_model_path,
        "cloud_config": meta.cloud_config,
        "raw_attrs": dict(meta.raw_attrs),
    }
    snapshot["dataset_contract"] = build_dataset_contract(
        meta,
        data_path=(data_path or train_data_path or Path(".")),
        n_samples=None,
        dataset_sha256=snapshot["dataset_sha256"],
    )
    return snapshot

def _save_training_plots(history: List[Dict[str, float]], outdir: Path) -> None:
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping training-history plots.")
        return

    outdir = Path(outdir) / "plots" / "training"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        epochs = np.asarray([int(item["epoch"]) + 1 for item in history], dtype=float)
    except Exception as exc:
        logger.warning(f"training-history plots skipped because epoch values could not be read: {exc}")
        return

    def _series(key: str, default: float = float("nan")) -> List[float]:
        values: List[float] = []
        for item in history:
            try:
                value = float(item.get(key, default))
            except Exception:
                value = float("nan")
            values.append(value if math.isfinite(value) else float("nan"))
        return values

    def _robust_ylim(series_values: List[np.ndarray], *, logy: bool) -> Optional[Tuple[float, float]]:
        valid: List[np.ndarray] = []
        for arr in series_values:
            finite = arr[np.isfinite(arr)]
            if logy:
                finite = finite[finite > 0.0]
            if finite.size:
                valid.append(finite)
        if not valid:
            return None
        merged = np.concatenate(valid)
        if merged.size == 1:
            value = float(merged[0])
            if logy:
                return max(value / 2.0, 1e-30), value * 2.0
            margin = abs(value) * 0.1 + 1e-12
            return value - margin, value + margin
        lo, hi = np.nanpercentile(merged, [1.0, 99.0])
        lo = float(lo)
        hi = float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi):
            return None
        if logy:
            lo = max(lo, 1e-30)
            hi = max(hi, lo * 1.01)
            return lo / 1.25, hi * 1.25
        if hi <= lo:
            margin = abs(hi) * 0.1 + 1e-12
            return lo - margin, hi + margin
        margin = (hi - lo) * 0.08
        return lo - margin, hi + margin

    def _plot_series(
        path: Path,
        title: str,
        y_label: str,
        series: List[Tuple[str, List[float]]],
        *,
        logy: bool = False,
        y_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        try:
            fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
            plotted_arrays: List[np.ndarray] = []
            for label, values in series:
                arr = np.asarray(values, dtype=float)
                arr[~np.isfinite(arr)] = np.nan
                if logy:
                    arr[arr <= 0.0] = np.nan
                mask = np.isfinite(arr)
                if not np.any(mask):
                    continue
                plotted_arrays.append(arr)
                ax.plot(epochs[mask], arr[mask], label=label, linewidth=2.0)
            if not plotted_arrays:
                plt.close(fig)
                return
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(y_label)
            if logy:
                ax.set_yscale("log")
            ylim = y_bounds or _robust_ylim(plotted_arrays, logy=logy)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best")
            fig.savefig(path, dpi=180)
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"training plot {path.name} could not be written: {exc}")

    _plot_series(
        outdir / "loss_total.png",
        "Total Loss",
        "Loss",
        [
            ("train_total", _series("train_loss_total")),
            ("val_total", _series("val_loss_total")),
        ],
        logy=True,
    )
    _plot_series(
        outdir / "loss_U.png",
        "Potential Loss",
        "MSE",
        [
            ("train_U", _series("train_loss_u")),
            ("val_U", _series("val_loss_u")),
        ],
        logy=True,
    )
    _plot_series(
        outdir / "loss_a.png",
        "Acceleration Loss",
        "MSE",
        [
            ("train_a", _series("train_loss_a")),
            ("val_a", _series("val_loss_a")),
        ],
        logy=True,
    )
    _plot_series(
        outdir / "lr_schedule.png",
        "Learning Rate Schedule",
        "Learning Rate",
        [("lr", _series("lr"))],
        logy=True,
    )
    _plot_series(
        outdir / "weights.png",
        "Sobolev Loss Weights",
        "Weight",
        [
            ("w_U", _series("w_u")),
            ("w_a_raw", _series("w_a_raw")),
            ("w_a_eff", _series("w_a_eff")),
        ],
        logy=False,
    )
    # Only plot direction loss if it was ever non-zero
    if any(
        math.isfinite(v) and v > 0.0
        for v in _series("train_loss_dir", 0.0)
    ):
        _plot_series(
            outdir / "loss_dir.png",
            "Direction Loss (1 - cos_sim)",
            "Loss",
            [
                ("train_direction", _series("train_loss_dir", 0.0)),
                ("val_direction", _series("val_loss_dir", 0.0)),
            ],
            logy=True,
        )
        _plot_series(
            outdir / "cossim.png",
            "Mean Cosine Similarity (a_pred vs a_true)",
            "cos_sim",
            [
                ("train_cos_sim", _series("train_mean_cossim", 1.0)),
                ("val_cos_sim", _series("val_mean_cossim", 1.0)),
            ],
            logy=False,
            y_bounds=(-1.05, 1.05),
        )

def _log_dataset_and_model_summary(N, _effective_target, bytes_est, cfg, data_path, dataset_body_name, dset_name, independent_val, meta, n_train, n_val, resolved_mu_si, resolved_r_ref_m, train_data_path, val_data_path):
    logger.info("=== Dataset ===")
    if independent_val:
        logger.info(f"Train file: {train_data_path.name} ({n_train:,} samples)")
        logger.info(f"Val file  : {val_data_path.name} ({n_val:,} samples)")
        if cfg.test_data:
            logger.info(f"Test file : {cfg.test_data}")
        if cfg.ood_data:
            logger.info(f"OOD file  : {cfg.ood_data}")
    _sm = getattr(cfg, "suite_manifest", None)
    if _sm:
        logger.info(f"Suite manifest: {_sm}")
    else:
        logger.info(f"File: {data_path.name}")
    logger.info(f"Target Dataset: {dset_name} | Total: [{N:,}, 7] | Size: {_human_bytes(bytes_est)}")
    logger.info(f"Train/val split: {n_train:,} / {n_val:,}")
    logger.info("=== Physics Metadata (auto-synced from HDF5) ===")
    logger.info(f"central_body : {dataset_body_name}")
    logger.info(f"unit_system  : {meta.unit_system}")
    logger.info(f"mu_si        : {resolved_mu_si}  |  r_ref_m : {resolved_r_ref_m}")
    logger.info(f"degree_max   : {meta.requested_degree}  |  degree_min : {meta.degree_min}")
    logger.info(f"target_mode  : {meta.target_mode or 'unknown (inferred: ' + _effective_target + ')'}")
    logger.info(f"columns      : {meta.columns or 'unknown'}")
    logger.info(f"a_sign_conv  : {meta.a_sign_convention or 'unknown'}")
    _dcv = getattr(meta, "derivative_convention_version", None)
    if _dcv is None:
        logger.warning(
            "derivative_convention_version: MISSING - dataset may have been generated before "
            "the dP_dphi sign fix. Latitude acceleration labels could be sign-flipped. "
            "Regenerate with the corrected spatial_cloud_generator.py."
        )
    else:
        logger.info(f"deriv_conv   : {_dcv}")
    if meta.alt_min_km is not None and meta.alt_max_km is not None:
        logger.info(f"alt range    : [{meta.alt_min_km}, {meta.alt_max_km}] km")
    logger.info(f"Conversion factors (DU/TU/VU): {meta.DU_m} / {meta.TU_s} / {meta.VU_m_s}")
    logger.info("=== Model ===")
    _n_bands_log = getattr(cfg, "n_bands", 1)
    _use_res_log = getattr(cfg, "use_residual_blocks", False)
    _grad_acc_log = getattr(cfg, "grad_accumulation_steps", 1)
    logger.info(f"{'model.activation':24s}: {cfg.activation}")
    logger.info(f"{'model.hidden':24s}: {cfg.hidden}")
    logger.info(f"{'model.depth':24s}: {cfg.depth}")
    logger.info(f"{'model.preset':24s}: {getattr(cfg, 'model_preset', 'custom')}")
    logger.info(f"{'model.n_bands':24s}: {_n_bands_log}")
    logger.info(f"{'model.w0_first':24s}: {cfg.w0_first}")
    logger.info(f"{'model.w0_hidden':24s}: {cfg.w0_hidden}")
    logger.info(f"{'model.w0_bands':24s}: {getattr(cfg, 'w0_bands', None)}")
    logger.info(f"{'model.residual_blocks':24s}: {_use_res_log}")
    logger.info(f"{'grad_accum':24s}: {_grad_acc_log}")

def _log_training_curriculum(_accel_min_fac, cfg):
    logger.info("=== Training Curriculum ===")
    logger.info(
        f"potential_only_epochs={cfg.potential_only_epochs} | "
        f"accel_ramp_epochs={cfg.accel_ramp_epochs} | "
        f"accel_min_factor={_accel_min_fac}"
    )
    # Derivative training note: acceleration is ∇U, so it must be constrained from epoch 0.
    if cfg.potential_only_epochs > 0:
        logger.warning(
            "potential_only_epochs > 0 detected. "
            "SIREN can fit dU while grad(dU) drifts because acceleration is computed via autograd. "
            f"accel_min_factor={_accel_min_fac} keeps a floor to limit drift. "
            "Set accel_min_factor=0.0 only if you explicitly want pure potential-only behaviour."
        )
    if _accel_min_fac == 0.0:
        logger.info("  Derivative training note: accel_min_factor=0.0 (pure potential-only during warm-up).")
    else:
        logger.info(
            f"  Derivative training note: acceleration is always active (floor={_accel_min_fac}). "
            "This prevents grad(dU) from drifting during curriculum warm-up."
        )

    if cfg.use_altitude_balanced_loss:
        logger.info(f"  Altitude-Balanced Loss: ON (bins={cfg.altitude_bin_width_km}km)")
    if cfg.use_radial_cross_loss:
        logger.info(f"  Radial/Cross Loss: ON (radial_w={cfg.radial_loss_weight}, cross_w={cfg.cross_loss_weight})")
    if cfg.use_laplacian_regularization:
        _lap_mode_log = str(getattr(cfg, "laplacian_mode", "diagnostic")).strip().lower()
        if _lap_mode_log == "train":
            logger.info(
                f"  In-batch Laplacian Reg: ON, mode=train (gradient backpropagates) "
                f"(w={cfg.laplacian_weight}, every={cfg.laplacian_every_n_batches})"
            )
        else:
            logger.info(
                f"  In-batch Laplacian Reg: ON, mode={_lap_mode_log} (DIAGNOSTIC ONLY - logged, "
                f"NOT backpropagated). For a trainable physics constraint set --laplacian-mode train "
                f"(collocation Laplacian is the preferred trainable regulariser)."
            )
    logger.info(f"  Direction Loss: weight={cfg.direction_loss_weight}, start={cfg.direction_loss_start_epoch}, ramp={cfg.direction_loss_ramp_epochs}")
    # NOTE: best-checkpoint-metric logging moved below, after _best_metric_canonical
    # and checkpoint_selection are defined (they were referenced here before
    # assignment, which made train() raise UnboundLocalError on every run).

    # Fail fast on invalid architecture combination
    if cfg.activation.lower() == "sine" and cfg.use_fourier:
        raise ValueError(
            "activation='sine' (SIREN) and use_fourier=True are mutually exclusive. "
            "Stacking RFF on a SIREN creates a sin-of-sin composition that causes "
            "catastrophic out-of-distribution overfitting. "
            "Use one of:\n"
            "  (1) activation='silu'/'tanh' + use_fourier=True\n"
            "  (2) activation='sine' + use_fourier=False  (recommended default)"
        )

def _log_data_loading_policy(N, _avail_ram_mb, _est_ram_mb, _policy, _preload_reason, cfg, dataset_mb, should_preload):
    logger.info("=== Data Loading Policy ===")
    logger.info(f"  dataset estimated size : {dataset_mb:.1f} MB ({N:,} rows)")
    logger.info(f"  preload_policy         : {_policy}")
    logger.info(f"  auto_preload_mb        : {float(getattr(cfg, 'auto_preload_mb', 2048.0)):.1f} MB")
    logger.info(f"  estimated preload RAM  : {_est_ram_mb:.0f} MB")
    if _avail_ram_mb is not None:
        logger.info(f"  available system RAM   : {_avail_ram_mb:.0f} MB (psutil)")
    else:
        logger.info("  available system RAM   : unknown (psutil not installed; RAM safety check skipped)")
    logger.info(f"  decision               : {'RAM preload' if should_preload else 'HDF5 streaming'}")
    logger.info(f"  reason                 : {_preload_reason}")


def train(cfg: TrainConfig) -> None:
    """Main execution pipeline for the Physics-Informed setup and training.

    This is a scalar residual potential surrogate, NOT a classical q,p
    Sobolev-Trained Lunar Residual Potential Surrogate.  The model learns DeltaU(x) and acceleration
    is obtained by differentiating the learned potential via autograd:
        Delta_a = a_sign * grad(DeltaU_scaled) * (u_scale / x_scale)
    """

    # 1. Initialization
    apply_model_preset(cfg)
    if str(getattr(cfg, "runtime_model_kind", "potential_autograd")) == "force_direct":
        raise NotImplementedError(
            "runtime_model_kind='force_direct' is a future distilled acceleration "
            "runtime and is not supported by the current scalar-potential training path."
        )
    set_seed(
        cfg.seed,
        deterministic=bool(getattr(cfg, "deterministic", True)),
        benchmark=bool(getattr(cfg, "benchmark_cudnn", False)),
    )
    device = get_device()
    layout = ensure_run_layout(Path(cfg.out))
    outdir = layout.run_dir
    run_created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(layout.train_log)]
    )

    # -----------------------------------------------------------------------
    # Resume resolution (epoch-level). When cfg.resume_from is set, restore
    # model/optimizer/GradNorm/RNG state (below) and continue from the last
    # completed epoch. cfg.epochs is the TOTAL target epoch count, not extra
    # epochs. Architecture/scaler/dataset-identity fields are locked to the
    # previous run so the rebuilt model + scaler match the checkpoint exactly.
    # -----------------------------------------------------------------------
    _resume_requested = bool(getattr(cfg, "resume_from", None))
    _resume_ckpt: Optional[Dict[str, Any]] = None
    _resume_ckpt_path: Optional[Path] = None
    start_epoch = 0
    _resume_best_val = float("inf")
    _resume_best_epoch = -1
    _resume_epochs_without_improve = 0
    _resume_global_step = 0
    _resume_prev_manifest: Dict[str, Any] = {}
    if _resume_requested:
        _resume_layout, _resume_ckpt_path, _resume_ckpt = resolve_resume_checkpoint(
            cfg.resume_from,
            prefer=str(getattr(cfg, "resume_checkpoint", "last")),
            device=device,
        )
        if _resume_layout.run_dir != layout.run_dir:
            logger.warning(
                f"[resume] --out ({layout.run_dir}) differs from the resumed run "
                f"({_resume_layout.run_dir}); writing to the resumed run directory."
            )
            layout = ensure_run_layout(_resume_layout.run_dir)
            outdir = layout.run_dir
            cfg.out = str(outdir)
        _resume_prev_manifest = read_run_manifest(layout) or {}
        run_created_at = str(_resume_prev_manifest.get("created_at_utc") or run_created_at)

        # Load previous resolved config (prefer config.json; fall back to ckpt).
        _prev_cfg: Dict[str, Any] = {}
        if layout.config_json.is_file():
            try:
                _prev_cfg = json.loads(layout.config_json.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"[resume] could not read previous config.json: {exc}")
        if not _prev_cfg:
            _prev_cfg = dict(_resume_ckpt.get("config") or {})

        # Lock architecture / encoding / scaler-math / dataset-identity fields so
        # the rebuilt model + scaler reproduce the checkpoint exactly. Optimization,
        # LR schedule, and the epoch target are intentionally taken from THIS run.
        _RESUME_LOCKED_FIELDS = (
            "activation", "hidden", "depth", "dropout", "w0_first", "w0_hidden",
            "use_residual_blocks", "n_bands", "multiscale_mode",
            "use_fourier", "fourier_append_raw", "fourier_n_features", "fourier_sigma", "fourier_seed",
            "use_sh_encoding", "sh_encoding_degree", "sh_append_raw",
            "use_radial_separation", "radial_append_raw",
            "use_radial_decay_encoding", "radial_decay_max_power", "radial_decay_append_raw",
            "use_real_sh_basis", "real_sh_degree", "real_sh_append_raw", "real_sh_include_radial",
            "u_scale_mode", "a_scale_mode", "target_scale_multiplier",
            "use_si", "dataset_name", "split_seed", "val_ratio", "seed",
        )
        _locked_changed = []
        for _f in _RESUME_LOCKED_FIELDS:
            if _f in _prev_cfg and hasattr(cfg, _f):
                if getattr(cfg, _f) != _prev_cfg.get(_f):
                    _locked_changed.append(_f)
                setattr(cfg, _f, _prev_cfg.get(_f))
        if _locked_changed:
            logger.info(f"[resume] locked architecture/scaler fields to previous run: {_locked_changed}")

        # Continuation counters from the checkpoint.
        start_epoch = int(_resume_ckpt.get("epoch", -1)) + 1
        _ck_cfg = dict(_resume_ckpt.get("config") or {})
        _bs = _ck_cfg.get("best_score", _ck_cfg.get("best_val_loss"))
        try:
            _resume_best_val = float(_bs) if (_bs is not None and math.isfinite(float(_bs))) else float("inf")
        except (TypeError, ValueError):
            _resume_best_val = float("inf")
        _be = _ck_cfg.get("best_epoch")
        try:
            _resume_best_epoch = (int(_be) - 1) if (_be is not None and int(_be) > 0) else -1
        except (TypeError, ValueError):
            _resume_best_epoch = -1
        try:
            _resume_epochs_without_improve = int(_ck_cfg.get("epochs_since_improvement", 0) or 0)
        except (TypeError, ValueError):
            _resume_epochs_without_improve = 0
        try:
            _resume_global_step = int(_resume_ckpt.get("global_step") or 0)
        except (TypeError, ValueError):
            _resume_global_step = 0

        if int(cfg.epochs) <= int(start_epoch):
            raise ValueError(
                f"--epochs ({cfg.epochs}) must be greater than the last completed epoch "
                f"({start_epoch}) to resume. --epochs is the TOTAL target epoch count; "
                f"pass e.g. --epochs {start_epoch + 100}."
            )
        logger.info(
            f"[resume] checkpoint={_resume_ckpt_path} | last_completed_epoch={start_epoch} "
            f"| resuming at epoch {start_epoch + 1} | target_epochs={cfg.epochs} "
            f"| strict={bool(getattr(cfg, 'resume_strict', True))}"
        )

    _log_section(
        "ST-LRPS Training",
        {
            "run_dir": outdir,
            "created_at_utc": run_created_at,
            "physics_design": "scalar residual potential dU; acceleration from autograd grad(dU)",
        },
    )
    _log_section(
        "Runtime / Device",
        {
            "device": device.type.upper(),
            "seed": cfg.seed,
            "torch_dtype": str(DTYPE).replace("torch.", ""),
        },
    )
    _warn_batch_size_for_vram(device, cfg)
    if device.type == "cuda":
        logger.info(
            "CUDA memory log format: cuda_mem=current_allocated/current_reserved MiB, "
            "peak=peak_allocated/peak_reserved MiB since phase start, total=physical VRAM. "
            "Values are PyTorch allocator memory, not nvidia-smi process memory."
        )

    # Effective configuration summary (so the active feature set is unambiguous in
    # the log, especially now that several features default ON).
    _grad_accum = int(getattr(cfg, "grad_accumulation_steps", 1))
    _log_section(
        "Loss Configuration",
        {
            "optim.lr": f"{cfg.lr:g}",
            "optim.weight_decay": f"{cfg.weight_decay:g}",
            "batch_size": cfg.batch_size,
            "grad_accumulation_steps": _grad_accum,
            "effective_batch": cfg.batch_size * _grad_accum,
            "accel_ramp_epochs": cfg.accel_ramp_epochs,
            "accel_min_factor": getattr(cfg, "accel_min_factor", 0.05),
            "direction.weight": cfg.direction_loss_weight,
            "direction.start_epoch": cfg.direction_loss_start_epoch,
            "direction.ramp_epochs": cfg.direction_loss_ramp_epochs,
            "direction.floor_abs": f"{cfg.direction_loss_floor_abs:g}",
            "altitude_balanced": bool(cfg.use_altitude_balanced_loss),
            "radial_cross": f"{bool(cfg.use_radial_cross_loss)} (radial={cfg.radial_loss_weight}, cross={cfg.cross_loss_weight})",
        },
    )

    command_line = write_command_txt(layout)
    _manifest_dict = {
        "schema_version": "st_lrps_run_manifest_v1",
        "run_id": outdir.name,
        "created_at_utc": run_created_at,
        "status": "running",
        "command_line": command_line,
        "command_path": str(layout.command_txt),
        "script_version": "st_lrps_engine",
        "git_commit": os.environ.get("GIT_COMMIT") or None,
        "data_paths": {
            "data": str(cfg.data),
            "train_data": str(cfg.train_data) if cfg.train_data else None,
            "val_data": str(cfg.val_data) if cfg.val_data else None,
            "test_data": str(cfg.test_data) if cfg.test_data else None,
            "ood_data": str(cfg.ood_data) if cfg.ood_data else None,
            "suite_manifest": str(cfg.suite_manifest) if cfg.suite_manifest else None,
        },
        "config_path": str(layout.config_json),
        "scaler_path": str(layout.scaler_json),
        "best_checkpoint_path": str(layout.ckpt_best),
        "last_checkpoint_path": str(layout.ckpt_last),
        "history_csv_path": str(layout.history_csv),
        "history_jsonl_path": str(layout.history_jsonl),
        "warnings": [],
        "evaluations": [],
    }
    if _resume_requested:
        # Preserve the original run manifest (run_id, created_at, evaluations);
        # record the resume event and set status back to "running".
        _resumed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            _resume_cmd_path = layout.provenance_dir / (
                "resume_command_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + ".txt"
            )
            _resume_cmd_path.write_text(command_line + "\n", encoding="utf-8")
        except Exception as _rc_exc:  # pragma: no cover - defensive
            logger.warning(f"[resume] could not write resume command provenance: {_rc_exc}")
        update_run_manifest(
            layout,
            {
                "status": "running",
                "command_line": command_line,
                "command_path": str(layout.command_txt),
                "resumed": True,
                "resumed_from_checkpoint": str(_resume_ckpt_path),
                "resume_start_epoch": int(start_epoch) + 1,
                "previous_latest_epoch": int(
                    _resume_ckpt.get("epoch_display", start_epoch) or start_epoch
                ),
                "target_epochs": int(cfg.epochs),
                "resumed_at_utc": _resumed_at,
            },
        )
    else:
        write_run_manifest(layout, _manifest_dict)
    _log_section(
        "Output Artifacts",
        {
            "config_json": layout.config_json,
            "scaler_json": layout.scaler_json,
            "train_log": layout.train_log,
            "history_csv": layout.history_csv,
            "history_jsonl": layout.history_jsonl,
            "ckpt_best": layout.ckpt_best,
            "ckpt_last": layout.ckpt_last,
        },
    )

    if cfg.quick_check:
        # Quick-check runs ONE epoch to verify the pipeline. On resume that means
        # one ADDITIONAL epoch past the last completed one (start_epoch), so the
        # quick-check still does real work instead of an empty range.
        _qc_epochs = (int(start_epoch) + 1) if _resume_requested else 1
        logger.info("=" * 62)
        logger.info("QUICK CHECK MODE: this is not a real training run.")
        logger.info(f"  epochs={_qc_epochs}  max_train_batches=5  max_val_batches=2  log_every=1")
        logger.info("=" * 62)
        cfg.epochs = _qc_epochs
        cfg.log_every = 1
        cfg.max_train_batches = cfg.max_train_batches if cfg.max_train_batches is not None else 5
        cfg.max_val_batches   = cfg.max_val_batches   if cfg.max_val_batches   is not None else 2

    # 2. Dataset Discovery & Validation
    data_path = Path(cfg.data)
    independent_val = cfg.train_data is not None and cfg.val_data is not None
    train_data_path: Optional[Path] = None
    val_data_path: Optional[Path] = None

    if independent_val:
        train_data_path = Path(cfg.train_data)
        val_data_path = Path(cfg.val_data)
        if not train_data_path.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_data_path}")
        if not val_data_path.exists():
            raise FileNotFoundError(f"Val dataset not found: {val_data_path}")
        primary_path = train_data_path
    else:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        primary_path = data_path

    dset_name = cfg.dataset_name
    try:
        with h5py.File(primary_path, "r") as f:
            _ = f[dset_name]
    except (KeyError, OSError):
        dset_name = _discover_dataset_name(primary_path, preferred=cfg.dataset_name)

    # 3. Read Metadata (SSOT)
    meta = DatasetMeta.from_h5(primary_path)

    if independent_val:
        with h5py.File(train_data_path, "r", swmr=True) as f:
            N_train_file = int(f[dset_name].shape[0])
            bytes_est_train = N_train_file * 7 * (4 if str(f[dset_name].dtype) == "float32" else 8)
        with h5py.File(val_data_path, "r", swmr=True) as f:
            N_val_file = int(f[dset_name].shape[0])
            bytes_est_val = N_val_file * 7 * (4 if str(f[dset_name].dtype) == "float32" else 8)
        bytes_est = bytes_est_train + bytes_est_val
        N = N_train_file + N_val_file
        meta_val = DatasetMeta.from_h5(val_data_path)
        _resolve_lunar_dataset_contract(meta_val, data_path=val_data_path)

        def _require_meta_match(name: str, left: Any, right: Any) -> None:
            if left is None or right is None:
                return
            if isinstance(left, float) or isinstance(right, float):
                if abs(float(left) - float(right)) <= 1.0:
                    return
            elif left == right:
                return
            raise ValueError(
                f"Train/Val metadata mismatch for {name}: {left!r} vs {right!r}. "
                "Independent train/validation clouds must use the same lunar gravity contract."
            )

        _require_meta_match("central_body", meta.central_body, meta_val.central_body)
        _require_meta_match("mu_si", meta.mu_si, meta_val.mu_si)
        _require_meta_match("r_ref_m", meta.r_ref_m, meta_val.r_ref_m)
        _require_meta_match("unit_system", meta.unit_system, meta_val.unit_system)
        _require_meta_match("degree_min", meta.degree_min, meta_val.degree_min)
        _require_meta_match("requested_degree", meta.requested_degree, meta_val.requested_degree)
        _require_meta_match("target_mode", meta.target_mode, meta_val.target_mode)
        if meta_val.mu_si is not None and meta.mu_si is not None:
            if abs(meta_val.mu_si - meta.mu_si) > 1.0:
                logger.warning(f"Train/Val mu_si mismatch: {meta.mu_si} vs {meta_val.mu_si}")
        if meta_val.r_ref_m is not None and meta.r_ref_m is not None:
            if abs(meta_val.r_ref_m - meta.r_ref_m) > 1.0:
                logger.warning(f"Train/Val r_ref_m mismatch: {meta.r_ref_m} vs {meta_val.r_ref_m}")
        if meta.unit_system != meta_val.unit_system:
            logger.warning(f"Train/Val unit_system mismatch: {meta.unit_system} vs {meta_val.unit_system}")
        if meta.degree_min != meta_val.degree_min:
            logger.warning(f"Train/Val degree_min mismatch: {meta.degree_min} vs {meta_val.degree_min}")
    else:
        with h5py.File(data_path, "r", swmr=True) as f:
            N = int(f[dset_name].shape[0])
            bytes_est = N * 7 * (4 if str(f[dset_name].dtype) == "float32" else 8)

    if cfg.use_si and meta.unit_system == "canonical" and not meta.can_convert_to_si():
        raise ValueError("Configuration demands SI units, but dataset is missing DU_m/TU_s/VU_m_s attributes.")

    allow_legacy_dataset_contract = bool(getattr(cfg, "allow_legacy_dataset_contract", False))
    allow_missing_dataset_contract = bool(getattr(cfg, "allow_missing_dataset_contract", False))
    dataset_contract_obj = DatasetContract.from_hdf5(
        primary_path,
        dataset_name=dset_name,
        allow_legacy_dataset_contract=allow_legacy_dataset_contract,
        allow_missing_dataset_contract=allow_missing_dataset_contract,
        allow_legacy_derivative_convention=bool(getattr(cfg, "allow_legacy_derivative_convention", False)),
    )
    validation_report = validate_dataset_file(
        primary_path,
        out_dir=layout.provenance_dir,
        dataset_name=dset_name,
        n_check=min(1024, int(N)),
        seed=int(cfg.split_seed if cfg.split_seed is not None else cfg.seed),
        allow_legacy_dataset_contract=allow_legacy_dataset_contract,
        allow_missing_dataset_contract=allow_missing_dataset_contract,
        allow_legacy_derivative_convention=bool(getattr(cfg, "allow_legacy_derivative_convention", False)),
    )
    if not validation_report.get("passed") and not bool(getattr(cfg, "allow_dataset_validation_fail", False)):
        raise ValueError(
            "Dataset validation failed before training: "
            + "; ".join(str(item) for item in validation_report.get("errors", []))
        )

    # 4. Data Splitting
    split_seed = int(cfg.split_seed if cfg.split_seed is not None else cfg.seed)
    split_policy = str(getattr(cfg, "split_policy", "seeded_random") or "seeded_random")
    if independent_val:
        train_indices = None
        val_indices = None
        n_train = N_train_file
        n_val = N_val_file
        split_manifest = {
            "schema_version": 1,
            "dataset_id": dataset_contract_obj.dataset_id,
            "split_policy": "independent_files",
            "split_seed": split_seed,
            "train_count": int(n_train),
            "val_count": int(n_val),
            "test_count": 0,
            "ood_count": 0,
            "index_hashes": {},
            "altitude_range_per_split": {},
            "created_at_utc": validation_report.get("created_at_utc"),
        }
    else:
        if split_policy in {"seeded_random", "random"}:
            splits = split_dataset_indices(
                n_rows=N,
                split_policy=split_policy,
                split_seed=split_seed,
                val_fraction=float(cfg.val_ratio),
            )
        elif split_policy == "altitude_stratified":
            with h5py.File(primary_path, "r", swmr=True) as f:
                xyz_all = np.asarray(f[dset_name][:, 0:3], dtype=np.float64)
            altitude_all = (np.linalg.norm(xyz_all, axis=1) - float(meta.r_ref_m or R_MOON_SI)) / 1000.0
            splits = split_dataset_indices(
                n_rows=N,
                split_policy=split_policy,
                split_seed=split_seed,
                val_fraction=float(cfg.val_ratio),
                altitude_km=altitude_all,
            )
        else:
            raise ValueError(f"Unsupported training split_policy={split_policy!r}")
        train_indices = splits["train"]
        val_indices = splits["val"]
        n_train = int(train_indices.size)
        n_val = int(val_indices.size)
        split_manifest = build_split_manifest(
            dataset_contract=dataset_contract_obj,
            splits=splits,
            split_policy=split_policy,
            split_seed=split_seed,
        )
    split_manifest_path = write_split_manifest(layout.provenance_dir / "split_manifest.json", split_manifest)

    # 4b. Validate metadata contract
    degree_min_val = int(meta.degree_min) if meta.degree_min is not None else -1
    _effective_target = meta.target_mode or ("residual" if degree_min_val >= 0 else "full")
    dataset_body_name, resolved_mu_si, resolved_r_ref_m = _resolve_lunar_dataset_contract(
        meta, data_path=primary_path,
    )
    # Hard convention guard: reject sign-flipped legacy datasets and other
    # silently-wrong metadata before any training happens.
    validate_training_dataset_convention(
        meta,
        data_path=primary_path,
        allow_legacy_derivative_convention=bool(
            getattr(cfg, "allow_legacy_derivative_convention", False)
        ),
        allow_legacy_target_mode_inference=bool(
            getattr(cfg, "allow_legacy_target_mode_inference", False)
        ),
        allow_missing_dataset_contract=bool(
            getattr(cfg, "allow_missing_dataset_contract", False)
        ),
    )
    if _effective_target == "residual" and degree_min_val < 0:
        raise ValueError(
            "Residual dataset detected (target_mode='residual') but degree_min is missing or < 0. "
            "Check HDF5 metadata: degree_min must be >= 0 for residual datasets."
        )
    if meta.columns is not None:
        _cols_lower = str(meta.columns).lower()
        _has_residual_cols = ("du" in _cols_lower or "dax" in _cols_lower)
        if degree_min_val >= 0 and not _has_residual_cols and "[x,y,z,u,ax,ay,az]" in _cols_lower:
            logger.warning(
                f"Dataset columns are labeled full-field ({meta.columns!r}) but "
                f"degree_min={degree_min_val} >= 0 suggests a residual dataset. "
                "Verify dataset generation parameters."
            )

    _log_dataset_and_model_summary(N, _effective_target, bytes_est, cfg, data_path, dataset_body_name, dset_name, independent_val, meta, n_train, n_val, resolved_mu_si, resolved_r_ref_m, train_data_path, val_data_path)
    # SIREN derivative-training safety check
    if cfg.activation.lower() == "sine":
        if cfg.lr > 5e-4:
            logger.warning(
                f"SIREN+Sobolev stability: lr={cfg.lr:.2e} is high. "
                "Recommended lr <= 5e-4 (1e-4 is safer) for derivative/Sobolev training."
            )
        if float(cfg.output_head_lr_mult) > 1.0:
            logger.warning(
                f"SIREN+Sobolev: output_head_lr_mult={cfg.output_head_lr_mult} > 1.0 can "
                "destabilize the grad(U) output. Recommended value: 1.0."
            )

    _accel_min_fac = float(getattr(cfg, "accel_min_factor", 0.05))
    _log_training_curriculum(_accel_min_fac, cfg)

    # 5. Resolve mu_si
    mu_val = float(resolved_mu_si)
    logger.info(f"Hierarchical base model: mu_si = {mu_val:.6e}")

    # 6. Infer acceleration sign
    if isinstance(cfg.a_sign, str) and cfg.a_sign.lower() == "auto":
        if meta.a_sign_convention is not None:
            _sgn = str(meta.a_sign_convention).strip()
            if _sgn in ("+1", "1"):
                a_sign = 1.0
                logger.info(f"Acceleration sign from dataset metadata: a_sign=+1.0")
            elif _sgn == "-1":
                a_sign = -1.0
                logger.info(f"Acceleration sign from dataset metadata: a_sign=-1.0")
            else:
                logger.warning(f"Unrecognised a_sign_convention='{_sgn}'; falling back to auto-inference.")
                a_sign = infer_a_sign_from_data(
                    h5_path=primary_path, dset_name=dset_name, meta=meta,
                    use_si=cfg.use_si, n_probe=50_000, seed=cfg.fit_seed + 777
                )
        else:
            a_sign = infer_a_sign_from_data(
                h5_path=primary_path, dset_name=dset_name, meta=meta,
                use_si=cfg.use_si, n_probe=50_000, seed=cfg.fit_seed + 777
            )
    else:
        a_sign = float(cfg.a_sign)

    target_contract = TargetContract.from_dataset_meta(
        meta,
        resolved_mu_si=float(resolved_mu_si),
        resolved_r_ref_m=float(resolved_r_ref_m),
        a_sign=float(a_sign),
        allow_inferred_target_mode=bool(getattr(cfg, "allow_legacy_target_mode_inference", False)),
        allow_legacy_derivative_convention=bool(
            getattr(cfg, "allow_legacy_derivative_convention", False)
        ),
    )
    logger.info(
        "Target contract: "
        f"mode={target_contract.target_mode}, baseline={target_contract.baseline_kind}, "
        f"base_degree={target_contract.base_degree}, target_degree={target_contract.target_degree}, "
        f"frame={target_contract.frame}"
    )

    # 7. Fit isometric scalers on residuals
    scaler_path = layout.scaler_json
    scaler_hash_info: Dict[str, Any]
    if scaler_path.exists():
        logger.info(f"Loading existing scaler from {scaler_path.name}")
        scaler = ScalerPack.load_json(scaler_path)
        scaler_hash_info = {
            "scaler_hash": compute_payload_sha256(asdict(scaler)),
            "scaler_file_sha256": compute_file_sha256(scaler_path),
            "scaler_payload": asdict(scaler),
        }
    else:
        scaler = fit_scaler_streaming(
            h5_path=primary_path, dset_name=dset_name, meta=meta,
            use_si=cfg.use_si, mu_si=mu_val, a_sign=a_sign,
            n_fit=cfg.fit_rows, seed=cfg.fit_seed, chunk_rows=cfg.fit_chunk_rows,
            degree_min=degree_min_val,
            target_mode=_effective_target,
            degree_max=int(meta.degree_max if meta.degree_max is not None else (meta.requested_degree or -1)),
            u_scale_mode=str(getattr(cfg, "u_scale_mode", "hybrid")),
            a_scale_mode=str(getattr(cfg, "a_scale_mode", "hybrid")),
            target_scale_multiplier=float(getattr(cfg, "target_scale_multiplier", 6.0)),
            target_contract=target_contract,
        )
        scaler_hash_info = write_scaler_json(layout, scaler)
    logger.info(f"[artifacts] scaler_hash={scaler_hash_info['scaler_hash']}")
    update_run_manifest(
        layout,
        {
            "scaler_path": str(layout.scaler_json),
            "scaler_hash": scaler_hash_info["scaler_hash"],
            "scaler_file_sha256": scaler_hash_info["scaler_file_sha256"],
        },
    )

    # 8. Construct DataLoaders
    dataset_mb = bytes_est / (1024.0 * 1024.0)
    # Resolve the preload policy. --preload-data is a convenience alias for "always".
    _policy = str(getattr(cfg, "preload_policy", "auto")).strip().lower()
    if bool(getattr(cfg, "preload_data", False)) and _policy != "never":
        _policy = "always"
    _est_ram_mb = _estimate_preload_ram_mb(int(N))
    _avail_ram_mb = _available_ram_mb()
    should_preload, _preload_reason = _decide_preload(
        _policy,
        dataset_mb=dataset_mb,
        auto_preload_mb=float(getattr(cfg, "auto_preload_mb", 2048.0)),
        est_ram_mb=_est_ram_mb,
        avail_ram_mb=_avail_ram_mb,
    )
    _log_data_loading_policy(N, _avail_ram_mb, _est_ram_mb, _policy, _preload_reason, cfg, dataset_mb, should_preload)
    if should_preload and "WARNING" in _preload_reason:
        logger.warning(f"Preload RAM-safety: {_preload_reason}")

    if should_preload:
        logger.info(f"Data mode: RAM preload")
        if independent_val:
            logger.info(f"Loading train ({n_train:,}) from {train_data_path.name}...")
            with h5py.File(train_data_path, "r", libver="latest", swmr=True) as _f:
                _arr_train = np.asarray(_f[dset_name][:], dtype=np.float64)
            logger.info(f"Loading val ({n_val:,}) from {val_data_path.name}...")
            with h5py.File(val_data_path, "r", libver="latest", swmr=True) as _f:
                _arr_val = np.asarray(_f[dset_name][:], dtype=np.float64)

            _xt, _ut, _at = _arr_train[:, 0:3], _arr_train[:, 3:4], _arr_train[:, 4:7]
            _xv, _uv, _av = _arr_val[:, 0:3], _arr_val[:, 3:4], _arr_val[:, 4:7]
            del _arr_train, _arr_val

            if cfg.use_si and meta.unit_system == "canonical":
                _xt, _ut, _at = meta.convert_xyz_U_a_to_si(_xt, _ut, _at)
                _xv, _uv, _av = meta.convert_xyz_U_a_to_si(_xv, _uv, _av)

            train_ds: Dataset = TensorMemoryDataset(
                _xt.astype(np.float32), _ut.astype(np.float32), _at.astype(np.float32)
            )
            val_ds: Dataset = TensorMemoryDataset(
                _xv.astype(np.float32), _uv.astype(np.float32), _av.astype(np.float32)
            )
            del _xt, _ut, _at, _xv, _uv, _av
        else:
            logger.info(f"Loading {N:,} rows into CPU memory (~{dataset_mb:.2f} MB)...")
            with h5py.File(data_path, "r", libver="latest", swmr=True) as _f:
                _arr = np.asarray(_f[dset_name][:], dtype=np.float64)

            _x_mem = _arr[:, 0:3]
            _u_mem = _arr[:, 3:4]
            _a_mem = _arr[:, 4:7]
            del _arr

            if cfg.use_si and meta.unit_system == "canonical":
                _x_mem, _u_mem, _a_mem = meta.convert_xyz_U_a_to_si(_x_mem, _u_mem, _a_mem)

            _x_mem = _x_mem.astype(np.float32)
            _u_mem = _u_mem.astype(np.float32)
            _a_mem = _a_mem.astype(np.float32)

            train_ds = TensorMemoryDataset(
                _x_mem[train_indices], _u_mem[train_indices], _a_mem[train_indices]
            )
            val_ds = TensorMemoryDataset(
                _x_mem[val_indices], _u_mem[val_indices], _a_mem[val_indices]
            )
            del _x_mem, _u_mem, _a_mem

        n_train = len(train_ds)
        n_val   = len(val_ds)
        pin = cfg.pin_memory and device.type == "cuda"
        mem_workers = max(0, cfg.num_workers)
        pf = cfg.prefetch_factor if (mem_workers > 0 and cfg.prefetch_factor is not None) else None
        logger.info(f"Train/val split: {n_train:,} / {n_val:,}")
        logger.info(
            f"pin_memory={pin}, non_blocking={pin}, num_workers={mem_workers} (requested={cfg.num_workers})"
            + (f", prefetch_factor={pf}" if pf is not None else "")
        )

        _dl_kw: Dict[str, Any] = dict(
            batch_size=cfg.batch_size, num_workers=mem_workers, pin_memory=pin,
            persistent_workers=(mem_workers > 0), collate_fn=collate_xyz_u_a,
        )
        if pf is not None:
            _dl_kw["prefetch_factor"] = pf
        train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **_dl_kw)
        val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **_dl_kw)
    else:
        logger.info("Data mode: HDF5 streaming")
        if independent_val:
            train_ds = H5BlockDataset(
                train_data_path, dset_name, 0, n_train, meta, cfg.use_si, cfg.cache_rows, indices=None
            )
            val_ds = H5BlockDataset(
                val_data_path, dset_name, 0, n_val, meta, cfg.use_si, cfg.cache_rows, indices=None
            )
        else:
            train_ds = H5BlockDataset(
                data_path, dset_name, 0, N, meta, cfg.use_si, cfg.cache_rows, indices=train_indices
            )
            val_ds = H5BlockDataset(
                data_path, dset_name, 0, N, meta, cfg.use_si, cfg.cache_rows, indices=val_indices
            )

        train_sampler = BlockShuffleSampler(len(train_ds), cfg.sampler_block_size, cfg.seed + 100)
        val_sampler   = BlockShuffleSampler(len(val_ds),   cfg.sampler_block_size, cfg.seed + 200)
        _streaming_path = train_data_path if independent_val else data_path
        train_workers = _resolve_loader_worker_count(_streaming_path, cfg.num_workers)
        if train_workers == 0 and int(cfg.num_workers) > 0:
            logger.warning(
                "Windows HDF5 safety: num_workers forced to 0 for HDF5 streaming. "
                "Use --preload-data (or --auto-preload-mb) for multi-worker loading."
            )
        val_workers = max(0, train_workers // 2)
        pin = cfg.pin_memory and device.type == "cuda"
        tr_pf = cfg.prefetch_factor if (train_workers > 0 and cfg.prefetch_factor is not None) else None
        va_pf = cfg.prefetch_factor if (val_workers   > 0 and cfg.prefetch_factor is not None) else None
        logger.info(
            f"pin_memory={pin}, train_workers={train_workers} (requested={cfg.num_workers}),"
            f" val_workers={val_workers}"
            + (f", prefetch_factor={tr_pf}" if tr_pf is not None else "")
        )

        _tr_kw: Dict[str, Any] = dict(
            batch_size=cfg.batch_size, sampler=train_sampler,
            num_workers=train_workers, pin_memory=pin,
            persistent_workers=(train_workers > 0), collate_fn=collate_xyz_u_a, drop_last=True,
        )
        _va_kw: Dict[str, Any] = dict(
            batch_size=cfg.batch_size, sampler=val_sampler,
            num_workers=val_workers, pin_memory=pin,
            persistent_workers=(val_workers > 0), collate_fn=collate_xyz_u_a, drop_last=False,
        )
        if tr_pf is not None:
            _tr_kw["prefetch_factor"] = tr_pf
        if va_pf is not None:
            _va_kw["prefetch_factor"] = va_pf
        train_loader = DataLoader(train_ds, **_tr_kw)
        val_loader   = DataLoader(val_ds,   **_va_kw)

    # 9. Build model via the shared factory (build_model_from_config) — authoritative builder
    # used by evaluator and force model. This ensures SH/radial encoding flags are honoured.
    # The SIREN+Fourier mutual exclusion check is inside build_model_from_config().
    #
    # CRITICAL (reload-safety): resolve the dataset's degree range and the
    # multi-scale band frequencies INTO cfg before building, and BEFORE writing
    # config.json. Previously the model was built from cfg (which had no degree
    # fields → silent 0/50 defaults) while config.json recorded the meta-derived
    # degrees. For n_bands>1 that produced a model whose SIREN band frequencies
    # differed from what evaluation reconstructed: the state_dict matched by
    # shape but the functional model was wrong. Resolving here makes training
    # and evaluation build the identical spectrum.
    degree_max_val = int(
        meta.degree_max if meta.degree_max is not None
        else (meta.requested_degree if meta.requested_degree is not None else -1)
    )
    cfg.degree_min = int(degree_min_val)
    cfg.degree_max = int(degree_max_val)
    cfg.resolved_r_ref_m = float(resolved_r_ref_m)
    cfg.x_scale_m = float(scaler.x.scale)
    if cfg.activation.lower() == "sine" and int(getattr(cfg, "n_bands", 1)) > 1:
        if cfg.degree_max <= 0:
            raise ValueError(
                "Multi-scale SIREN (n_bands>1) requires a known degree_max from the "
                f"dataset metadata, but resolved degree_max={cfg.degree_max}. "
                "Regenerate the dataset with degree_max recorded, or use n_bands=1."
            )
        cfg.w0_bands = [
            float(w) for w in _compute_harmonic_w0_bands(
                int(cfg.n_bands), int(cfg.degree_min), int(cfg.degree_max)
            )
        ]
    else:
        cfg.w0_bands = None

    model = build_model_from_config(
        cfg,
        in_dim=3,
        device=device,
        dtype=DTYPE,
    )

    if _resume_requested and _resume_ckpt is not None:
        _msd = _resume_ckpt.get("model_state_dict") or _resume_ckpt.get("model")
        if _msd is None:
            raise RuntimeError("[resume] checkpoint has no model_state_dict.")
        _load_res = model.load_state_dict(_msd, strict=bool(getattr(cfg, "resume_strict", True)))
        _missing = list(getattr(_load_res, "missing_keys", []) or [])
        _unexpected = list(getattr(_load_res, "unexpected_keys", []) or [])
        if _missing or _unexpected:
            logger.warning(f"[resume] non-strict model load: missing={_missing} unexpected={_unexpected}")
        logger.info(f"[resume] model weights restored from {_resume_ckpt_path}")

    # Log architecture details (equivalent to old manual logging, but from the built model)
    _n_bands_built = max(1, int(getattr(cfg, "n_bands", 1)))
    _use_res_built = bool(getattr(cfg, "use_residual_blocks", False))
    if cfg.use_fourier:
        logger.info(
            f"Fourier embedding: n_features={cfg.fourier_n_features}, "
            f"sigma={cfg.fourier_sigma}, append_raw={cfg.fourier_append_raw}"
        )
    if cfg.activation.lower() == "sine":
        if _n_bands_built > 1:
            # Log the EXACT bands the model was built with (resolved into cfg above),
            # not an independently recomputed value that could silently diverge.
            logger.info(
                f"Built Multi-Scale SIREN: n_bands={_n_bands_built}, w0_bands={cfg.w0_bands}, "
                f"degree_min={cfg.degree_min}, degree_max={cfg.degree_max}, "
                f"depth={cfg.depth}, hidden={cfg.hidden}"
            )
            # Defensive cross-check: the model's resolved bands must equal cfg's.
            _model_bands = list(getattr(model, "w0_bands", []) or [])
            if _model_bands and [round(b, 4) for b in _model_bands] != [round(b, 4) for b in (cfg.w0_bands or [])]:
                raise RuntimeError(
                    f"Internal error: model w0_bands {_model_bands} != cfg.w0_bands {cfg.w0_bands}. "
                    "Refusing to train a model whose spectrum cannot be reproduced from config."
                )
        else:
            logger.info(
                f"Built SIREN backbone: depth={cfg.depth}, hidden={cfg.hidden}, "
                f"w0_first={cfg.w0_first}, w0_hidden={cfg.w0_hidden}, "
                f"residual_blocks={_use_res_built}"
            )
    else:
        logger.info(
            f"Built MLP backbone: depth={cfg.depth}, hidden={cfg.hidden}, activation={cfg.activation}"
        )
    _total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {_total_params:,}")

    # Capture model-derived architecture metadata for config.json / checkpoint
    # persistence below. (Previously this block tried to rewrite config.json
    # before it had been created, so the metadata was silently lost every run.)
    _emb_type_built = str(getattr(model, "embedding_type", "raw"))
    _in_fdim_built = int(getattr(model, "input_feature_dim", 3))
    _arch_signature = compute_architecture_signature(cfg)
    logger.info(
        f"Encoding: embedding_type={_emb_type_built}  input_feature_dim={_in_fdim_built}  "
        f"builder={MODEL_BUILDER_VERSION}  arch_signature={_arch_signature}"
    )

    weights = GradNormWeights(
        w_u=cfg.w_u,
        w_a=cfg.w_a,
        mode=cfg.gradnorm_mode,
        w_a_min=cfg.gradnorm_w_a_min,
        w_a_max=cfg.gradnorm_w_a_max,
    )
    logger.info(f"Loss weighting: mode={cfg.gradnorm_mode}  w_u={cfg.w_u:.2f}  w_a_init={cfg.w_a:.2f}")

    if _resume_requested:
        _gnw_state = (_resume_ckpt.get("training_state") or {}).get("gradnorm_weights")
        if _gnw_state:
            weights.load_state_dict(_gnw_state)
            logger.info(
                f"[resume] GradNorm state restored (w_a={weights.w_a:.4f}, ntk_done={weights._ntk_done})."
            )
        else:
            logger.warning(
                "[resume] checkpoint lacks GradNorm state; continuing with a fresh GradNorm "
                "(NTK init may recompute on the first resumed step)."
            )

    loss_fn = SobolevLoss(
        scaler=scaler,
        a_sign=a_sign,
        mu_si=mu_val,
        r_ref_m=resolved_r_ref_m,
        degree_min=degree_min_val,
        degree_max=degree_max_val,
        target_contract=target_contract,
    ).to(device=device, dtype=DTYPE)
    logger.info(f"Residual baseline: {target_contract.baseline_description}")

    head_params = _get_output_head_params(model)
    head_param_ids = {id(param) for param in head_params}
    body_params = [param for param in model.parameters() if id(param) not in head_param_ids]
    param_groups: List[Dict[str, Any]] = []
    if body_params:
        param_groups.append(
            {
                "params": body_params,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
            }
        )
    param_groups.append(
        {
            "params": head_params,
            "lr": cfg.lr * float(cfg.output_head_lr_mult),
            "weight_decay": 0.0,
        }
    )
    opt = AdamW(param_groups)
    logger.info(
        f"Optimizer groups: body_lr={cfg.lr:.2e}, body_wd={cfg.weight_decay:.2e}, "
        f"head_lr={cfg.lr * float(cfg.output_head_lr_mult):.2e}, head_wd=0.00e+00"
    )
    for group in opt.param_groups:
        group["initial_lr"] = float(group["lr"])

    if _resume_requested:
        _osd = _resume_ckpt.get("optimizer_state_dict")
        if _osd is not None:
            try:
                opt.load_state_dict(_osd)
                logger.info("[resume] optimizer state restored.")
            except Exception as _oexc:
                if bool(getattr(cfg, "resume_strict", True)):
                    raise RuntimeError(f"[resume] optimizer state restore failed: {_oexc}") from _oexc
                logger.warning(f"[resume] optimizer restore failed (non-strict); continuing fresh: {_oexc}")
        elif bool(getattr(cfg, "resume_strict", True)):
            raise RuntimeError(
                "[resume] checkpoint has no optimizer_state_dict (strict). "
                "Use --resume-nonstrict to continue with a fresh optimizer."
            )
        else:
            logger.warning("[resume] checkpoint has no optimizer_state_dict; continuing with a fresh optimizer.")
        # Re-assert the manual-cosine base LR per group (load_state_dict may not carry it).
        for group in opt.param_groups:
            group.setdefault("initial_lr", float(group["lr"]))

    # 10. Save canonical config + provenance snapshot
    config_path = layout.config_json
    _is_residual = (_effective_target == "residual")
    dataset_snapshot = _dataset_meta_snapshot(
        meta,
        dataset_name=dset_name,
        data_path=(None if independent_val else data_path),
        train_data_path=(train_data_path if independent_val else None),
        val_data_path=(val_data_path if independent_val else None),
        test_data_path=cfg.test_data,
        ood_data_path=cfg.ood_data,
        target_mode=_effective_target,
        central_body=dataset_body_name,
        resolved_mu_si=resolved_mu_si,
        resolved_r_ref_m=resolved_r_ref_m,
    )
    dataset_snapshot["target_contract"] = target_contract.to_dict()
    dataset_snapshot["dataset_contract"] = dataset_contract_obj.to_dict()
    dataset_snapshot["dataset_validation_report_path"] = str(layout.provenance_dir / "dataset_validation_report.json")
    dataset_snapshot["split_manifest_path"] = str(split_manifest_path)
    dataset_snapshot["split_manifest"] = split_manifest
    dataset_snapshot["dataset_safety_overrides"] = {
        "allow_legacy_dataset_contract": allow_legacy_dataset_contract,
        "allow_missing_dataset_contract": allow_missing_dataset_contract,
        "allow_legacy_derivative_convention": bool(getattr(cfg, "allow_legacy_derivative_convention", False)),
        "allow_dataset_validation_fail": bool(getattr(cfg, "allow_dataset_validation_fail", False)),
    }
    atomic_write_json(layout.provenance_dir / "dataset_meta.json", dataset_snapshot)
    feature_summary = build_experiment_feature_summary(cfg, target_contract, model)
    atomic_write_json(layout.provenance_dir / "feature_summary.json", feature_summary)

    suite_manifest_path = str(getattr(cfg, "suite_manifest", "") or "").strip()
    suite_manifest: Dict[str, Any] = {}
    if suite_manifest_path:
        try:
            manifest_path = Path(suite_manifest_path)
            if manifest_path.exists():
                suite_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                atomic_write_json(layout.provenance_dir / "suite_manifest_snapshot.json", suite_manifest)
            else:
                logger.warning(f"Suite manifest not found while writing config: {suite_manifest_path}")
        except Exception as exc:
            logger.warning(f"Could not read suite manifest while writing config: {exc}")

    capture_environment_snapshot(layout, extra={"device": str(device), "run_id": outdir.name})

    _raw_ckpt_start = int(getattr(cfg, "best_ckpt_start_epoch", -1))
    _direction_ready_epoch = (
        int(cfg.direction_loss_start_epoch)
        + int(cfg.direction_loss_ramp_epochs)
        + int(getattr(cfg, "checkpoint_settle_epochs", 5))
    )
    if _raw_ckpt_start < 0:
        if float(cfg.direction_loss_weight) > 0.0:
            _ckpt_start = _direction_ready_epoch
            _ckpt_start_reason = "waits until direction loss ramp is active and settled"
        else:
            _ckpt_start = 0
            _ckpt_start_reason = "direction loss disabled; checkpoint tracking starts immediately"
    else:
        _ckpt_start = _raw_ckpt_start
        _ckpt_start_reason = "manual best_ckpt_start_epoch"
        if float(cfg.direction_loss_weight) > 0.0 and _ckpt_start < _direction_ready_epoch:
            logger.warning(
                "Manual best_ckpt_start_epoch is earlier than the direction-ready epoch "
                f"({_ckpt_start} < {_direction_ready_epoch}). The run is allowed, but auto "
                "mode is safer for production checkpoints."
            )
    _best_metric_canonical = normalize_best_metric(getattr(cfg, "best_metric", "hybrid"))
    checkpoint_selection = checkpoint_selection_block(
        cfg,
        start_epoch=int(_ckpt_start),
        reason=_ckpt_start_reason,
    )
    _log_section(
        "Checkpoint Selection",
        {
            "best_metric": checkpoint_selection["best_metric"],
            "formula": checkpoint_selection["formula"],
            "start_epoch": checkpoint_selection["start_epoch_display"],
            "reason": checkpoint_selection.get("reason"),
            "lower_is_better": checkpoint_selection["lower_is_better"],
        },
    )
    _bm = str(_best_metric_canonical)
    logger.info(f"  Best-checkpoint metric: {_bm} | formula={checkpoint_selection['formula']}")
    if _bm == "direction_loss":
        logger.warning("best_metric='direction_loss' is experimental. "
                       "Early epochs may select underdeveloped checkpoints. "
                       "Consider 'hybrid' instead.")
    if _bm == "hybrid" and float(getattr(cfg, "direction_loss_weight", 0.0)) == 0.0:
        logger.warning("best_metric='hybrid' selected but direction_loss_weight=0. "
                       "Hybrid score will effectively use val_base_loss. "
                       "Set --direction-loss-weight > 0 to enable hybrid selection.")

    resolved_cfg_source = asdict(cfg)
    resolved_cfg_source.update(
        {
            "resolved_a_sign": float(a_sign),
            "resolved_mu_si": float(mu_val),
            "resolved_r_ref_m": float(resolved_r_ref_m),
            "mu_si": float(mu_val),
            "r_ref_m": float(resolved_r_ref_m),
            "degree_min": int(degree_min_val),
            "degree_max": int(degree_max_val),
            "target_mode": _effective_target,
            "target_contract": target_contract.to_dict(),
            "baseline_kind": target_contract.baseline_kind,
            "base_degree": int(target_contract.base_degree),
            "target_degree": int(target_contract.target_degree),
            "runtime_model_kind": str(getattr(cfg, "runtime_model_kind", "potential_autograd")),
            "residual_mode": _is_residual,
            "unit_system": meta.unit_system,
            "central_body": dataset_body_name,
            "train_data_path": (str(train_data_path) if independent_val else None),
            "val_data_path": (str(val_data_path) if independent_val else None),
            "test_data_path": (str(cfg.test_data) if cfg.test_data else None),
            "ood_data_path": (str(cfg.ood_data) if cfg.ood_data else None),
            "suite_manifest": suite_manifest_path or None,
            "dataset_name": str(dset_name),
            "x_mean_policy": "origin_fixed",
            "x_scale_policy": "max_norm_from_origin",
            "lr_schedule": {
                "kind": "warmup_cosine",
                "warmup_epochs": int(cfg.warmup_epochs),
                "min_lr_ratio": float(cfg.min_lr_ratio),
                "t_max": (int(cfg.t_max) if cfg.t_max is not None else None),
            },
            "loss_features": {
                "altitude_balanced": bool(cfg.use_altitude_balanced_loss),
                "radial_cross": bool(cfg.use_radial_cross_loss),
                "laplacian_regularization": bool(cfg.use_laplacian_regularization),
                "direction_loss_weight": float(cfg.direction_loss_weight),
                "direction_loss_start_epoch": int(cfg.direction_loss_start_epoch),
                "direction_loss_ramp_epochs": int(cfg.direction_loss_ramp_epochs),
                "checkpoint_settle_epochs": int(getattr(cfg, "checkpoint_settle_epochs", 5)),
            },
            "best_val_loss": None,
            "best_epoch": None,
            "best_score": None,
            "best_score_name": str(_best_metric_canonical),
            "best_metric": str(_best_metric_canonical),
            "best_ckpt_start_epoch_resolved": int(_ckpt_start),
            "best_ckpt_start_epoch_resolved_display": int(_ckpt_start + 1),
            "checkpoint_selection": checkpoint_selection,
            "model_builder_version": MODEL_BUILDER_VERSION,
            "embedding_type": _emb_type_built,
            "input_feature_dim": int(_in_fdim_built),
            "x_scale_m": float(scaler.x.scale),
            "total_params": int(_total_params),
            "w0_bands": list(cfg.w0_bands) if cfg.w0_bands is not None else None,
            "dataset_meta": dataset_snapshot,
            "feature_summary": feature_summary,
        }
    )
    if suite_manifest:
        resolved_cfg_source.update(
            {
                "suite_id": suite_manifest.get("suite_id"),
                "suite_name": suite_manifest.get("suite_name"),
                "suite_files": suite_manifest.get("output_files"),
                "suite_component_counts": suite_manifest.get("train_components"),
                "suite_train_total_n": suite_manifest.get("train_total_n"),
                "suite_seeds": {
                    "train_uniform": (suite_manifest.get("train_components") or {}).get("stratified_uniform", {}).get("seed"),
                    "train_inverse_r2": (suite_manifest.get("train_components") or {}).get("inverse_r2", {}).get("seed"),
                    "train_residual_mag": (suite_manifest.get("train_components") or {}).get("residual_mag", {}).get("seed"),
                    "train_boundary": (suite_manifest.get("train_components") or {}).get("boundary", {}).get("seed"),
                    "val": suite_manifest.get("val_seed"),
                    "test": suite_manifest.get("test_seed"),
                    "ood_low": suite_manifest.get("ood_low_seed"),
                    "ood_high": suite_manifest.get("ood_high_seed"),
                },
            }
        )

    payload = build_resolved_config(
        resolved_cfg_source,
        dataset_snapshot,
        model,
        scaler,
        _arch_signature,
    )
    atomic_write_json(config_path, payload)
    payload_readback = json.loads(config_path.read_text(encoding="utf-8"))
    verify_critical_config_fields_match(payload_readback, payload)
    if _resume_requested:
        # Compatibility gate: the rebuilt config must agree with the resumed
        # checkpoint on architecture/dataset/scaler-critical fields.
        try:
            verify_critical_config_fields_match(payload_readback, dict(_resume_ckpt.get("config") or {}))
            logger.info("[resume] checkpoint config is compatible with the rebuilt run config.")
        except RuntimeError as _vexc:
            if bool(getattr(cfg, "resume_strict", True)):
                raise RuntimeError(
                    f"[resume] architecture/dataset-critical config mismatch vs checkpoint: {_vexc}"
                ) from _vexc
            logger.warning(f"[resume] non-strict: ignoring config mismatch vs checkpoint: {_vexc}")
    update_run_manifest(
        layout,
        {
            "config_path": str(layout.config_json),
            "resolved_config_summary": {key: payload.get(key) for key in (
                "activation",
                "hidden",
                "depth",
                "n_bands",
                "w0_bands",
                "embedding_type",
                "input_feature_dim",
                "model_preset",
                "runtime_model_kind",
                "architecture_signature",
                "degree_min",
                "degree_max",
                "target_mode",
                "target_contract",
                "artifact_contract",
                "training_config_hash",
            )},
            "feature_summary_path": str(layout.provenance_dir / "feature_summary.json"),
            "dataset_validation_report_path": str(layout.provenance_dir / "dataset_validation_report.json"),
            "split_manifest_path": str(split_manifest_path),
            "architecture_signature": _arch_signature,
            "artifact_contract": payload.get("artifact_contract"),
            "training_config_hash": payload.get("training_config_hash"),
            "dataset_contract": payload.get("dataset_contract"),
            "split_manifest": split_manifest,
            "dataset_validation_passed": bool(validation_report.get("passed")),
            "dataset_safety_overrides": dataset_snapshot.get("dataset_safety_overrides"),
            "dataset_hash": (payload.get("dataset_contract") or {}).get("dataset_sha256")
            if isinstance(payload.get("dataset_contract"), dict) else None,
            "w0_bands": payload.get("w0_bands"),
            "status": "running",
        },
    )

    # 11. Train
    # Resolve collocation altitude bounds — only when a Laplacian is requested.
    # By default no Laplacian is requested, so these stay None and the collocation
    # path is fully skipped (no overhead).
    _col_r_min_m: Optional[float] = None
    _col_r_max_m: Optional[float] = None
    _lap_requested = _laplacian_requested(cfg)
    _col_lmode = str(getattr(cfg, "laplacian_mode", "diagnostic")).strip().lower()
    if _col_lmode not in ("off", "diagnostic", "train"):
        _col_lmode = "diagnostic"
    if not _lap_requested:
        logger.info("Laplacian: not requested (default) — no Laplacian diagnostics or regularization.")
    if _lap_requested and _col_lmode in ("diagnostic", "train"):
        _r_ref_col = float(resolved_r_ref_m)
        _col_alt_min = getattr(cfg, "collocation_alt_min_km", None)
        _col_alt_max = getattr(cfg, "collocation_alt_max_km", None)
        _col_alt_min_resolved = float(_col_alt_min) if _col_alt_min is not None else float(cfg.altitude_min_km)
        _col_alt_max_resolved = float(_col_alt_max) if _col_alt_max is not None else float(cfg.altitude_max_km)
        if _col_lmode == "train":
            if _col_alt_min_resolved is None or _col_alt_max_resolved is None:
                raise ValueError(
                    "laplacian_mode='train' requires collocation altitude bounds to be resolvable. "
                    "Set --collocation-alt-min-km / --collocation-alt-max-km or "
                    "--altitude-min-km / --altitude-max-km."
                )
        _col_r_min_m = _r_ref_col + _col_alt_min_resolved * 1000.0
        _col_r_max_m = _r_ref_col + _col_alt_max_resolved * 1000.0
        logger.info(
            f"Collocation Laplacian: mode={_col_lmode}, r_min={_col_r_min_m:.3e} m, "
            f"r_max={_col_r_max_m:.3e} m "
            f"(alt [{_col_alt_min_resolved:.1f}, {_col_alt_max_resolved:.1f}] km)"
        )

    trainer = STLRPSTrainer(
        model, loss_fn, opt, weights, device, cfg,
        collocation_r_min_m=_col_r_min_m,
        collocation_r_max_m=_col_r_max_m,
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    _prev_val_cossim = 1.0   # for direction drift detection
    _prev_val_mse_a = float("inf")
    best_path = layout.ckpt_best
    last_path = layout.ckpt_last
    log_path = layout.history_jsonl
    history: List[Dict[str, float]] = []
    _prev_mse_a: Optional[float] = None  # for epoch-level explosion detection
    global_step = 0
    run_status = "completed"

    if _resume_requested:
        best_val = _resume_best_val
        best_epoch = _resume_best_epoch
        epochs_without_improve = _resume_epochs_without_improve
        global_step = _resume_global_step
        logger.info(
            "[resume] restored counters: "
            f"best_epoch={best_epoch + 1 if best_epoch >= 0 else 'none'}, "
            f"best_val={best_val if math.isfinite(best_val) else 'inf'}, "
            f"epochs_without_improve={epochs_without_improve}, global_step={global_step}"
        )

    if _ckpt_start > 0:
        logger.info(
            f"[checkpoint] best tracking starts at epoch {_ckpt_start + 1} "
            "(auto waits until direction-loss training is active, ramped, and settled; "
            "epochs before this are excluded from best-ckpt selection)."
        )
    else:
        logger.info("[checkpoint] best tracking starts at epoch 1.")
    payload["best_ckpt_start_epoch_resolved"] = int(_ckpt_start)
    payload["best_ckpt_start_epoch_resolved_display"] = int(_ckpt_start + 1)
    payload["checkpoint_settle_epochs"] = int(getattr(cfg, "checkpoint_settle_epochs", 5))
    payload["checkpoint_selection"] = dict(checkpoint_selection)
    atomic_write_json(config_path, payload)
    ckpt_config_base = dict(payload)
    update_run_manifest(
        layout,
        {
            "best_ckpt_start_epoch": int(_ckpt_start),
            "best_ckpt_start_epoch_display": int(_ckpt_start + 1),
            "checkpoint_settle_epochs": int(getattr(cfg, "checkpoint_settle_epochs", 5)),
            "checkpoint_selection": dict(checkpoint_selection),
        },
    )
    logger.info(f"[artifacts] schema=st_lrps_checkpoint_v2")
    logger.info(f"[artifacts] architecture_signature={_arch_signature}")
    logger.info("Beginning training loop...")
    # Restore RNG state and decide history append-vs-overwrite for resume.
    _hist_mode = "w"
    if _resume_requested:
        _rng_state = (_resume_ckpt.get("training_state") or {}).get("rng_state")
        if _rng_state:
            restore_rng_state(_rng_state)
            logger.info("[resume] RNG state restored (epoch-level; DataLoader worker order is not bitwise-guaranteed).")
        else:
            logger.warning("[resume] checkpoint lacks RNG state; continuing with the seeded RNG.")
        if bool(getattr(cfg, "resume_append_history", True)) and log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as _hf:
                    for _line in _hf:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _prev_row = json.loads(_line)
                        except Exception:
                            continue
                        # Project to the flat row schema (strip nested train/val blocks).
                        history.append({
                            k: v for k, v in _prev_row.items()
                            if k not in ("train", "val", "checkpoint_report")
                        })
                logger.info(f"[resume] loaded {len(history)} previous history rows (append mode).")
                _hist_mode = "a"
            except Exception as _hexc:
                logger.warning(f"[resume] could not read previous history ({_hexc}); writing fresh rows.")
                _hist_mode = "w"
        elif log_path.exists():
            log_path.unlink()
    else:
        if log_path.exists():
            log_path.unlink()

    # Periodic Evaluation During Training (monitoring only; OFF by default).
    # Schedule is resolved from the first epoch that will actually run, so a
    # resume drops already-passed scheduled epochs. Evaluation runs in a separate
    # process after ckpt_last is saved and cannot affect training state.
    periodic_plan = resolve_periodic_eval_plan(cfg, start_epoch=int(start_epoch) + 1)
    _periodic_completed: set = set()
    if periodic_plan.enabled:
        _periodic_completed = completed_periodic_eval_epochs(outdir)
        logger.info(
            f"[periodic-eval] scheduled epochs: {','.join(str(e) for e in periodic_plan.epochs)} "
            f"| dataset={periodic_plan.dataset} checkpoint={periodic_plan.prefer_checkpoint} "
            f"max_samples={periodic_plan.max_samples} device={periodic_plan.device} "
            f"continue_on_fail={periodic_plan.continue_on_fail}"
        )
        if _periodic_completed:
            _already = sorted(e for e in _periodic_completed if e in periodic_plan.epochs_set)
            if _already:
                logger.info(f"[periodic-eval] already recorded (resume) — will skip: {_already}")

    # Graceful, epoch-level interruption. A single Ctrl+C (SIGINT) sets a flag;
    # the loop finishes the current epoch, saves ckpt_last, marks the manifest
    # "interrupted", and exits. A second Ctrl+C restores default handling for a
    # hard stop. Installing a handler requires the main thread; if unavailable
    # (e.g. tests in a worker thread) we fall back to plain KeyboardInterrupt.
    import signal as _signal
    _interrupt = {"flag": False}
    _orig_sigint = None

    def _on_sigint(_signum, _frame):  # pragma: no cover - signal timing dependent
        if _interrupt["flag"]:
            if _orig_sigint is not None:
                _signal.signal(_signal.SIGINT, _orig_sigint)
            raise KeyboardInterrupt
        _interrupt["flag"] = True
        logger.warning(
            "[interrupt] Stop requested — will finish the current epoch, save, and exit. "
            "Press Ctrl+C again to force-quit."
        )

    try:
        _orig_sigint = _signal.signal(_signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        _orig_sigint = None

    with open(log_path, _hist_mode, encoding="utf-8") as logf:
        for epoch in range(start_epoch, cfg.epochs):
            epoch_t0 = time.perf_counter()
            lr_scale = _lr_multiplier_for_epoch(
                epoch,
                total_epochs=cfg.epochs,
                warmup_epochs=cfg.warmup_epochs,
                min_lr_ratio=cfg.min_lr_ratio,
                t_max=cfg.t_max,
            )
            _apply_lr_multiplier(opt, lr_scale)
            _ldir_log = _direction_loss_factor(epoch, cfg)
            if _ldir_log > 0.0 or epoch == cfg.direction_loss_start_epoch:
                logger.info(f"[epoch {epoch+1}] effective lambda_dir={_ldir_log:.4e}")
            tr = trainer.run_epoch(train_loader, is_train=True,  epoch=epoch, max_batches=cfg.max_train_batches)

            # Epoch-level explosion detection: save failure manifest and stop on NaN.
            if tr.get("nan_detected"):
                run_status = "failed"
                logger.error(
                    f"Training stopped at epoch {epoch+1} due to NaN/Inf loss. "
                    f"Saving failure manifest to {outdir / 'failure_manifest.json'}."
                )
                with open(outdir / "failure_manifest.json", "w", encoding="utf-8") as _fmf:
                    json.dump({"epoch": epoch, "reason": "nan_loss", "config": asdict(cfg)}, _fmf, indent=2, default=str)
                update_run_manifest(
                    layout,
                    {
                        "status": "failed",
                        "latest_epoch": int(epoch + 1),
                        "notes": [f"Training stopped due to NaN/Inf loss at epoch {epoch + 1}."],
                    },
                )
                break

            # Warn if acceleration loss jumped 100x vs previous epoch (early explosion signal).
            _cur_mse_a = float(tr.get("mse_a", 0.0))
            if _prev_mse_a is not None and _prev_mse_a > 1e-12 and _cur_mse_a > 100.0 * _prev_mse_a:
                logger.warning(
                    f"Epoch {epoch+1}: acceleration loss jumped {_cur_mse_a/_prev_mse_a:.0f}x "
                    f"({_prev_mse_a:.3e} -> {_cur_mse_a:.3e}). "
                    "Possible derivative instability. Consider: lower lr, ensure accel_min_factor>0, "
                    "lower w0, increase accel_ramp_epochs."
                )
            _prev_mse_a = _cur_mse_a

            va = trainer.run_epoch(val_loader,   is_train=False, epoch=epoch, max_batches=cfg.max_val_batches)
            epoch_time_s = time.perf_counter() - epoch_t0
            global_step += int(tr.get("optimizer_steps", 0))

            # Direction drift warning: magnitude improving but direction metric worsening.
            _val_cossim_now = float(va.get("cossim_mean", 1.0))
            _val_mse_a_now = float(va.get("mse_a", 0.0))
            if (
                epoch > 0
                and float(getattr(cfg, "direction_loss_weight", 0.0)) > 0.0
                and _val_mse_a_now < _prev_val_mse_a * 0.98
                and _val_cossim_now < _prev_val_cossim - 0.005
            ):
                logger.warning(
                    f"Epoch {epoch+1}: val mse_a improved ({_prev_val_mse_a:.3e} → {_val_mse_a_now:.3e}) "
                    f"but direction metric is drifting "
                    f"(cossim: {_prev_val_cossim:.4f} → {_val_cossim_now:.4f}). "
                    "Consider increasing direction_loss_weight or lowering direction_loss_floor_abs."
                )
            _prev_val_cossim = _val_cossim_now
            _prev_val_mse_a = _val_mse_a_now

            _best_metric_mode = str(_best_metric_canonical)
            va["eligible_for_best"] = bool(epoch >= _ckpt_start)
            _ckpt_score, checkpoint_report = compute_checkpoint_score(va, cfg)
            va["val_checkpoint_score"] = float(_ckpt_score)
            va["checkpoint_formula"] = str(checkpoint_report["formula"])
            va["best_metric"] = str(checkpoint_report["best_metric"])

            ckpt_config = dict(ckpt_config_base)
            ckpt_config["best_val_loss"] = float(best_val) if math.isfinite(best_val) else None
            ckpt_config["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
            ckpt_config["best_score"] = float(best_val) if math.isfinite(best_val) else None
            ckpt_config["best_score_name"] = str(_best_metric_mode)
            ckpt_config["checkpoint_selection"] = dict(checkpoint_selection)
            ckpt_config["checkpoint_report"] = dict(checkpoint_report)
            ckpt_config["best_ckpt_start_epoch_resolved"] = int(_ckpt_start)
            ckpt_config["current_epoch"] = int(epoch + 1)
            ckpt_config["current_val_ref_loss"] = float(va["loss"])
            ckpt_config["epochs_since_improvement"] = int(epochs_without_improve)

            scheduler_state = {
                "kind": "warmup_cosine",
                "epoch": int(epoch),
                "warmup_epochs": int(cfg.warmup_epochs),
                "min_lr_ratio": float(cfg.min_lr_ratio),
                "t_max": (int(cfg.t_max) if cfg.t_max is not None else None),
            }
            checkpoint_train_stats = dict(tr)
            checkpoint_train_stats["gradnorm_status"] = str(getattr(cfg, "gradnorm_mode", "fixed"))
            # Resume state: GradNorm internal state + RNG snapshot so a resumed
            # run continues loss-weighting and randomness from this epoch.
            checkpoint_train_stats["gradnorm_weights"] = weights.state_dict()
            checkpoint_train_stats["rng_state"] = capture_rng_state()
            checkpoint_payload = build_checkpoint_payload(
                kind="last",
                epoch=epoch,
                model=model,
                optimizer=opt,
                scheduler=scheduler_state,
                cfg=ckpt_config,
                scaler=scaler,
                train_stats=checkpoint_train_stats,
                val_stats=va,
                dataset_meta=dataset_snapshot,
                architecture_signature=_arch_signature,
                global_step=global_step,
            )
            verify_critical_config_fields_match(payload_readback, checkpoint_payload["config"])

            checkpoint_info = {
                "kind": "last",
                "score": float(_ckpt_score),
                "formula": str(checkpoint_report["formula"]),
                "best_metric": str(checkpoint_report["best_metric"]),
                "path": str(last_path),
                "best_epoch": int(best_epoch + 1) if best_epoch >= 0 else None,
            }
            checkpoint_report["is_best_update"] = False
            checkpoint_report["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
            checkpoint_report["best_score"] = float(best_val) if math.isfinite(best_val) else None
            if epoch < _ckpt_start:
                # Burn-in phase: save last checkpoint but do not update best or count patience.
                logger.info(f"[checkpoint] waiting: epoch {epoch+1} < start epoch {_ckpt_start + 1}")
                if epoch == _ckpt_start - 1:
                    logger.info(
                        f"[checkpoint] waiting complete: epoch {epoch+1}. "
                        f"Best-checkpoint tracking and patience counter start from next epoch."
                    )
            else:
                if _ckpt_score < best_val:
                    best_val = _ckpt_score
                    best_epoch = int(epoch)
                    epochs_without_improve = 0
                    checkpoint_payload["scoring"]["score"] = float(_ckpt_score)
                    checkpoint_payload["config"]["best_val_loss"] = float(best_val)
                    checkpoint_payload["config"]["best_epoch"] = int(best_epoch + 1)
                    checkpoint_payload["config"]["best_score"] = float(_ckpt_score)
                    checkpoint_payload["config"]["best_score_name"] = str(_best_metric_mode)
                    checkpoint_payload["config"]["checkpoint_selection"] = dict(checkpoint_selection)
                    checkpoint_payload["config"]["best_val_base_loss"] = float(va.get("val_base_loss", va.get("mse_u", 0.0) + va.get("mse_a", 0.0)))
                    checkpoint_payload["config"]["best_val_total_loss"] = float(va.get("val_total_loss", va["loss"]))
                    checkpoint_payload["config"]["best_val_physics_loss"] = float(va.get("val_physics_loss", 0.0))
                    checkpoint_payload["config"]["epochs_since_improvement"] = 0
                    checkpoint_report["is_best_update"] = True
                    checkpoint_report["best_epoch"] = int(best_epoch + 1)
                    checkpoint_report["best_score"] = float(best_val)
                    checkpoint_payload["config"]["checkpoint_report"] = dict(checkpoint_report)
                    checkpoint_payload["scoring"].update(dict(checkpoint_report))
                    save_checkpoint(layout, kind="best", payload=checkpoint_payload, epoch=epoch)
                    best_ckpt_hash = compute_file_sha256(best_path)
                    logger.info(f"[artifacts] checkpoint saved: kind=best epoch={epoch + 1}")
                    logger.info(f"[checkpoint] best updated: val_ref={va['loss']:.6e} score={_ckpt_score:.6e} epoch={best_epoch + 1}")
                    checkpoint_info = {
                        "kind": "best",
                        "score": float(_ckpt_score),
                        "formula": str(checkpoint_report["formula"]),
                        "best_metric": str(checkpoint_report["best_metric"]),
                        "path": str(best_path),
                        "best_epoch": int(best_epoch + 1),
                    }
                    update_run_manifest(
                        layout,
                        {
                            "best_checkpoint_path": str(best_path),
                            "last_checkpoint_path": str(last_path),
                            "best_epoch": int(best_epoch + 1),
                            "best_score": float(_ckpt_score),
                            "checkpoint_selection": dict(checkpoint_selection),
                            "checkpoint_report": dict(checkpoint_report),
                            "latest_epoch": int(epoch + 1),
                            "checkpoint_hashes": {
                                "best": best_ckpt_hash,
                                "last": (compute_file_sha256(last_path) if last_path.exists() else None),
                            },
                        },
                    )
                else:
                    epochs_without_improve += 1

            checkpoint_payload["config"]["best_val_loss"] = float(best_val) if math.isfinite(best_val) else None
            checkpoint_payload["config"]["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
            checkpoint_payload["config"]["best_score"] = float(best_val) if math.isfinite(best_val) else None
            checkpoint_payload["config"]["best_score_name"] = str(_best_metric_mode)
            checkpoint_payload["config"]["checkpoint_selection"] = dict(checkpoint_selection)
            checkpoint_report["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
            checkpoint_report["best_score"] = float(best_val) if math.isfinite(best_val) else None
            checkpoint_payload["config"]["checkpoint_report"] = dict(checkpoint_report)
            checkpoint_payload["scoring"].update(dict(checkpoint_report))
            checkpoint_payload["config"]["epochs_since_improvement"] = int(epochs_without_improve)
            save_checkpoint(
                layout,
                kind="last",
                payload=checkpoint_payload,
                epoch=epoch,
                write_epoch_snapshot=bool(
                    getattr(cfg, "save_epoch_snapshots", False)
                    and ((epoch + 1) % max(1, int(getattr(cfg, "epoch_snapshot_every", 1))) == 0)
                ),
            )
            last_ckpt_hash = compute_file_sha256(last_path)
            logger.info(f"[artifacts] checkpoint saved: kind=last epoch={epoch + 1}")
            logger.info(f"[checkpoint] last saved: epoch={epoch + 1}")
            update_run_manifest(
                layout,
                {
                    "status": "running",
                    "latest_epoch": int(epoch + 1),
                    "checkpoint_report": dict(checkpoint_report),
                    "checkpoint_selection": dict(checkpoint_selection),
                    "last_checkpoint_path": str(last_path),
                    "checkpoint_hashes": {
                        "best": (compute_file_sha256(best_path) if best_path.exists() else None),
                        "last": last_ckpt_hash,
                    },
                },
            )

            tr_with_time = dict(tr)
            tr_with_time["epoch_time_s"] = float(epoch_time_s)
            row = flatten_epoch_metrics(epoch, tr_with_time, va, checkpoint_report, cfg)
            # Extra compatibility / diagnostics not in the stable core schema.
            row["val_mae_a_vec"] = float(va.get("mae_a_vec", 0.0)) if va.get("mae_a_vec") is not None else None
            row["val_rmse_a_vec"] = float(va.get("rmse_a_vec", 0.0)) if va.get("rmse_a_vec") is not None else None
            history.append(row)
            jsonl_row = dict(row)
            jsonl_row["train"] = dict(tr)
            jsonl_row["val"] = dict(va)
            jsonl_row["checkpoint_report"] = dict(checkpoint_report)
            logf.write(json.dumps(jsonl_row, sort_keys=True, default=str) + "\n")
            logf.flush()

            # Periodic evaluation (monitoring only). Runs AFTER ckpt_last is saved
            # and the history row is written. It is a fully isolated subprocess and
            # cannot change optimizer/scheduler/GradNorm/RNG/model/checkpoint state.
            if periodic_plan.enabled:
                _epoch_1based = int(epoch) + 1
                if (
                    _epoch_1based in periodic_plan.epochs_set
                    and _epoch_1based not in _periodic_completed
                ):
                    _peval_ok = True
                    try:
                        _peval_ok = run_periodic_eval(
                            cfg, outdir, _epoch_1based, periodic_plan,
                            log=logger, dataset_name=dset_name,
                        )
                    except Exception as _peval_exc:  # pragma: no cover - defensive
                        _peval_ok = False
                        logger.warning(f"[periodic-eval] epoch={_epoch_1based} unexpected error: {_peval_exc}")
                    _periodic_completed.add(_epoch_1based)
                    if not _peval_ok and not periodic_plan.continue_on_fail:
                        run_status = "failed"
                        logger.error(
                            f"[periodic-eval] aborting training at epoch {_epoch_1based} "
                            "because continue_on_fail is disabled."
                        )
                        update_run_manifest(
                            layout,
                            {
                                "status": "failed",
                                "latest_epoch": int(epoch + 1),
                                "notes": [
                                    f"Training stopped: periodic evaluation failed at epoch {_epoch_1based} "
                                    "with --periodic-eval-fail-fast."
                                ],
                            },
                        )
                        break

            _lde = float(tr.get("lambda_dir_eff", 0.0))
            _dir_log = (
                f" | dir={tr.get('loss_dir',0.0):.2e}/val={va.get('loss_dir',0.0):.2e}"
                f" cossim={tr.get('cossim_mean',1.0):.4f} lam={_lde:.2e}"
                if _lde > 0.0 else ""
            )
            logger.info(format_epoch_summary(row, total_epochs=int(cfg.epochs)))

            if epochs_without_improve >= int(cfg.patience):
                logger.info(
                    f"Early stopping triggered after {epochs_without_improve} epochs without validation improvement. "
                    f"Best epoch: {best_epoch + 1} | best_val_loss={best_val:.6e}"
                )
                break

            if _interrupt["flag"]:
                run_status = "interrupted"
                _interrupt_done = int(epoch) + 1
                _resume_hint = f"python -m lunaris.surrogate.st_lrps.training.cli --resume-from {outdir} --epochs {cfg.epochs}"
                logger.warning(
                    f"[interrupt] Stopping after completed epoch {_interrupt_done}. "
                    f"ckpt_last.pt is saved. Resume with:\n  {_resume_hint}"
                )
                update_run_manifest(
                    layout,
                    {
                        "status": "interrupted",
                        "latest_epoch": _interrupt_done,
                        "interrupted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "resume_hint": _resume_hint,
                    },
                )
                break

    # Restore the original SIGINT handler (best-effort).
    if _orig_sigint is not None:
        try:
            _signal.signal(_signal.SIGINT, _orig_sigint)
        except Exception:  # pragma: no cover
            pass

    _write_training_history_csv(history, layout.history_csv)
    _save_training_plots(history, outdir)

    payload["best_val_loss"] = float(best_val) if math.isfinite(best_val) else None
    payload["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
    payload["best_score"] = float(best_val) if math.isfinite(best_val) else None
    payload["best_score_name"] = str(_best_metric_canonical)
    payload["best_metric"] = str(_best_metric_canonical)
    payload["checkpoint_selection"] = dict(checkpoint_selection)
    atomic_write_json(config_path, payload)
    update_run_manifest(
        layout,
        {
            "status": run_status,
            "best_epoch": int(best_epoch + 1) if best_epoch >= 0 else None,
            "best_score": float(best_val) if math.isfinite(best_val) else None,
            "best_metric": str(_best_metric_canonical),
            "checkpoint_selection": dict(checkpoint_selection),
            "latest_epoch": (int(history[-1]["epoch"]) + 1) if history else 0,
            "checkpoint_hashes": {
                "best": (compute_file_sha256(best_path) if best_path.exists() else None),
                "last": (compute_file_sha256(last_path) if last_path.exists() else None),
            },
        },
    )

    eval_suggestion = (
        f"python -m lunaris.surrogate.st_lrps.evaluation.cli --model-dir {outdir} "
        f"--data {cfg.test_data or cfg.val_data or cfg.data} --out {outdir / 'evals' / 'publication_eval'}"
    )
    _log_section(
        "Training Complete",
        {
            "status": run_status,
            "best_epoch": int(best_epoch + 1) if best_epoch >= 0 else "none",
            "best_score": f"{best_val:.6e}" if math.isfinite(best_val) else "none",
            "best_metric": str(_best_metric_canonical),
            "best_formula": checkpoint_selection["formula"],
            "best_checkpoint": best_path if best_path.exists() else "not selected",
            "last_checkpoint": last_path,
            "history_csv": layout.history_csv,
            "history_jsonl": layout.history_jsonl,
            "eval_suggestion": eval_suggestion,
        },
    )


# ---------------------------------------------------------------------------
# CLI & Auto-Configuration Helpers
# ---------------------------------------------------------------------------


__all__ = ['STLRPSTrainer', 'train', 'set_seed', 'get_device']
