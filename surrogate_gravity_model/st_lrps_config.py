# -*- coding: utf-8 -*-
"""
Configuration and CLI parsing for the lunar potential surrogate trainer.

This module is the single source of truth for training CLI defaults. The PyQt
dashboard builds commands against these names, and ``st_lrps_train.py`` delegates
all argument parsing here. Defaults that describe generated cloud geometry
(altitude range in particular) are pulled from ``spatial_cloud_parameters`` so
the generator and trainer do not drift apart.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

try:
    from .st_lrps_data import DatasetMeta, _find_latest_dataset
except ImportError:  # pragma: no cover
    from st_lrps_data import DatasetMeta, _find_latest_dataset

# Pull altitude defaults from the cloud-generation SSOT so both modules
# always agree on the training envelope without manual synchronisation.
try:
    from .spatial_cloud_parameters import DEFAULT_SPATIAL_CLOUD_CONFIG as _CLOUD_CFG
except ImportError:  # pragma: no cover
    try:
        from spatial_cloud_parameters import DEFAULT_SPATIAL_CLOUD_CONFIG as _CLOUD_CFG  # type: ignore
    except ImportError:
        _CLOUD_CFG = None  # type: ignore

_DEFAULT_ALT_MIN_KM: float = float(getattr(_CLOUD_CFG, "alt_min_km", 200.0))
_DEFAULT_ALT_MAX_KM: float = float(getattr(_CLOUD_CFG, "alt_max_km", 600.0))

@dataclass
class TrainConfig:
    """Hyperparameter configuration for the Physics-Informed Neural Network."""
    data: str
    out: str
    dataset_name: str = "data"
    train_data: Optional[str] = None
    val_data: Optional[str] = None
    test_data: Optional[str] = None
    ood_data: Optional[str] = None
    suite_manifest: Optional[str] = None  # path to suite manifest.json (provenance only)

    seed: int = 42
    epochs: int = 200
    batch_size: int = 8192

    cache_rows: int = 65_536
    sampler_block_size: int = 65_536
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: Optional[int] = None  # only used when num_workers > 0

    val_ratio: float = 0.1
    split_seed: Optional[int] = None

    # Model architecture
    hidden: int = 512
    depth: int = 5
    activation: str = "sine"   # "sine" (SIREN) | "silu" | "tanh" | "softplus"
    dropout: float = 0.0
    w0_first: float = 30.0
    w0_hidden: float = 30.0

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-6
    output_head_lr_mult: float = 1.0
    max_grad_norm: float = 0.5
    t_max: Optional[int] = None   # defaults to epochs for monotonic cosine decay
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.05
    patience: int = 30

    # Loss weighting
    w_u: float = 1.0
    w_a: float = 1.0
    # gradnorm_mode: "ntk_init" (default) | "fixed" | "dynamic"
    # "ntk_init" computes gradient-norm ratio once at training start then freezes w_a.
    # Avoids instability from repeated Hessian-level updates (a_pred = ∇U makes
    # ∂L_a/∂W a second-order quantity, which makes full EMA GradNorm unstable).
    gradnorm_mode: str = "ntk_init"
    dynamic_weights: bool = False  # legacy flag; True overrides gradnorm_mode → "dynamic"
    gradnorm_w_a_min: float = 0.05
    gradnorm_w_a_max: float = 2.0
    potential_only_epochs: int = 0
    accel_ramp_epochs: int = 80
    # Minimum acceleration factor applied even during potential_only phase.
    # Prevents the derivative field from drifting completely unconstrained.
    # Set to 0.0 to restore original full potential-only behaviour.
    accel_min_factor: float = 0.05
    a_sign: Union[float, str] = "auto"

    # SSOT / Physics Meta behavior
    use_si: bool = True
    fit_rows: int = 500_000
    fit_seed: int = 123
    fit_chunk_rows: int = 131_072

    amp: bool = False

    # Fourier/RFF embedding → only for non-sine MLPs (activation="silu"/"tanh"/"softplus").
    # MUST NOT be combined with activation="sine" (SIREN): train() raises ValueError.
    use_fourier: bool = False
    fourier_append_raw: bool = True
    fourier_n_features: int = 256   # n → 2n-dim embedding (sin + cos)
    fourier_sigma: float = 1.0      # std of frequency matrix B
    fourier_seed: int = 42

    # Progress logging → log every N batches; 0 to disable
    log_every: int = 10

    # RAM preload → load whole dataset into CPU tensors for better GPU throughput
    # On Windows, HDF5 forces num_workers=0; RAM mode removes that constraint.
    preload_data: bool = False        # always preload regardless of size
    auto_preload_mb: float = 256.0    # auto-preload when dataset fits in this many MB

    # Quick-check mode: run 1 epoch with 5 train + 2 val batches to verify the
    # full pipeline (CUDA, autograd, checkpoint, metrics) in under a minute.
    quick_check: bool = False
    max_train_batches: Optional[int] = None  # cap training batches (None = full epoch)
    max_val_batches: Optional[int] = None    # cap validation batches (None = full epoch)

    # Acceleration direction loss -> penalises angular error between a_pred and a_true.
    # L_dir = mean(1 - cos_sim(a_pred, a_true)) for points where ||a_true|| > floor.
    # Ramped in after direction_loss_start_epoch to avoid destabilising early training.
    direction_loss_weight: float = 0.10
    direction_loss_start_epoch: int = 30
    direction_loss_ramp_epochs: int = 50
    direction_loss_floor_abs: float = 3e-6   # mask threshold on ||a_true||

    # Best-checkpoint selection burn-in.
    # -1 (default) = auto: if direction_loss_weight > 0, delays to
    # direction_loss_start_epoch + direction_loss_ramp_epochs + checkpoint_settle_epochs.
    # This prevents early epochs from winning the checkpoint race before
    # direction-aware training has started and settled.
    # Set to 0 to disable and start tracking from epoch 0.
    best_ckpt_start_epoch: int = -1
    checkpoint_settle_epochs: int = 5

    # Optional altitude-balanced residual loss.
    # Defaults pulled from spatial_cloud_parameters.DEFAULT_SPATIAL_CLOUD_CONFIG
    # so training envelope always matches the generated dataset without edits.
    use_altitude_balanced_loss: bool = False
    altitude_bin_width_km: float = 50.0
    altitude_min_km: float = _DEFAULT_ALT_MIN_KM
    altitude_max_km: float = _DEFAULT_ALT_MAX_KM

    # Optional radial / cross-radial acceleration penalties.
    use_radial_cross_loss: bool = False
    radial_loss_weight: float = 0.0
    cross_loss_weight: float = 0.0

    # Optional sparse Laplacian regularisation for the residual potential.
    # Uses the Hutchinson stochastic trace estimator (AMP-compatible, O(K) passes).
    use_laplacian_regularization: bool = False
    laplacian_weight: float = 0.0
    laplacian_every_n_batches: int = 5
    laplacian_subset_size: int = 512
    n_hutchinson_samples: int = 4   # Rademacher samples per Laplacian estimate

    # Residual SIREN blocks — wraps hidden layers in SirenResBlock.
    # Recommended for depth >= 6; adds LayerNorm + zero-init skip per block.
    use_residual_blocks: bool = False

    # Multi-scale SIREN — parallel frequency bands matched to the harmonic range.
    # n_bands > 1 uses MultiScaleSirenMLP; requires degree_min/degree_max metadata.
    n_bands: int = 1

    # Gradient accumulation — accumulate gradients over N batches before stepping.
    # Effective batch size = batch_size * grad_accumulation_steps.
    grad_accumulation_steps: int = 1

def _default_outdir(base: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / "runs" / f"st_lrps_train_{ts}"

def parse_args() -> TrainConfig:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description="Sobolev scalar-potential surrogate training for residual lunar gravity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data & Output
    group_data = ap.add_argument_group("Data & Output")
    group_data.add_argument("--data", default=None, help="Path to input HDF5 file (fallback for train/val split).")
    group_data.add_argument("--train-data", default=None, help="Optional independent train dataset path.")
    group_data.add_argument("--val-data", default=None, help="Optional independent validation dataset path.")
    group_data.add_argument("--test-data", default=None, help="Optional independent test dataset path (saved to config).")
    group_data.add_argument("--ood-data", default=None, help="Optional independent OOD dataset path (saved to config).")
    group_data.add_argument("--suite-manifest", default=None, help="Path to dataset suite manifest.json (stored in config for provenance).")
    group_data.add_argument("--out", "--out-dir", dest="out", default=None, help="Output directory for results.")
    group_data.add_argument("--dataset-name", default="data", help="HDF5 internal dataset name.")
    group_data.add_argument("--val-fraction", "--val-ratio", dest="val_fraction", type=float, default=0.1,
                            help="Fraction of data reserved for validation (if using --data).")
    group_data.add_argument("--split-seed", type=int, default=None,
                            help="Seed for the deterministic shuffled train/validation split.")

    # Architecture
    group_arch = ap.add_argument_group("Model Architecture")
    group_arch.add_argument("--hidden", type=int, default=512, help="Neurons per hidden layer (default: 512).")
    group_arch.add_argument("--depth", type=int, default=5, help="Number of hidden layers (default: 5).")
    group_arch.add_argument("--activation", type=str, default="sine",
                            choices=["sine", "silu", "tanh", "softplus"],
                            help="Activation function. 'sine' = SIREN.")
    group_arch.add_argument("--w0-first", type=float, default=None,
                            help="SIREN w0 for first layer (default: auto-derived from dataset degree_max).")
    group_arch.add_argument("--w0-hidden", type=float, default=None,
                            help="SIREN w0 for hidden layers (default: auto-derived from dataset degree_max).")
    group_arch.add_argument("--dropout", type=float, default=0.0)
    fourier_group = group_arch.add_mutually_exclusive_group()
    fourier_group.add_argument("--use-fourier", action="store_true", dest="use_fourier",
                               help="Enable Random Fourier Feature input embedding.")
    fourier_group.add_argument("--no-fourier", action="store_false", dest="use_fourier",
                               help="Disable Random Fourier Feature input embedding.")
    raw_skip_group = group_arch.add_mutually_exclusive_group()
    raw_skip_group.add_argument("--fourier-append-raw", action="store_true", dest="fourier_append_raw",
                                help="Concatenate raw scaled xyz with Fourier features before the backbone.")
    raw_skip_group.add_argument("--no-fourier-append-raw", action="store_false", dest="fourier_append_raw",
                                help="Use Fourier features without the raw-coordinate skip path.")
    group_arch.add_argument("--fourier-n", type=int, default=256,
                            help="Number of Fourier features (embedding dim = 2*n).")
    group_arch.add_argument("--fourier-sigma", type=float, default=1.0,
                            help="Std of frequency matrix B; larger = finer spatial detail.")
    group_arch.add_argument("--fourier-seed", type=int, default=42,
                            help="Seed used to construct the fixed Fourier feature matrix.")

    # Optimization
    group_opt = ap.add_argument_group("Optimization")
    group_opt.add_argument("--epochs", type=int, default=200)
    group_opt.add_argument("--batch-size", type=int, default=8192)
    group_opt.add_argument("--lr", type=float, default=1e-4)
    group_opt.add_argument("--weight-decay", type=float, default=1e-6)
    group_opt.add_argument("--output-head-lr-mult", type=float, default=1.0,
                           help="Learning-rate multiplier applied only to the final scalar output head.")
    group_opt.add_argument("--grad-clip", "--max-grad-norm", dest="grad_clip", type=float, default=0.5,
                           help="Global gradient clipping threshold.")
    group_opt.add_argument("--t-max", type=int, default=None,
                           help="Cosine scheduler T_max (default: equals --epochs for monotonic decay).")
    group_opt.add_argument("--warmup-epochs", type=int, default=5,
                           help="Linear learning-rate warm-up duration before cosine decay.")
    group_opt.add_argument("--min-lr-ratio", type=float, default=0.05,
                           help="Final cosine-decay learning-rate ratio relative to the base LR.")
    group_opt.add_argument("--patience", type=int, default=30,
                           help="Early-stopping patience measured on validation total loss.")
    amp_group = group_opt.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", action="store_true", dest="amp",
                           help="Enable CUDA AMP when the derivative path supports it safely.")
    amp_group.add_argument("--no-amp", action="store_false", dest="amp",
                           help="Disable Automatic Mixed Precision.")

    # Physics & Sobolev Weights
    group_phys = ap.add_argument_group("Physics & Loss Weights")
    group_phys.add_argument("--w-u", type=float, default=1.0, help="Initial weight for Potential (ΔU) loss.")
    group_phys.add_argument("--w-a", type=float, default=1.0, help="Initial weight for Acceleration (Δa) loss.")
    group_phys.add_argument("--gradnorm-mode", choices=["fixed", "ntk_init", "dynamic"], default="ntk_init",
                            help="Loss-weighting policy for the Sobolev objective.")
    group_phys.add_argument("--dynamic-weights", action="store_true", default=False,
                            help="Legacy alias that forces gradnorm_mode='dynamic'.")
    group_phys.add_argument("--no-dynamic-weights", action="store_false", dest="dynamic_weights",
                            help="Disable the legacy dynamic-weight override.")
    group_phys.add_argument("--gradnorm-w-a-min", type=float, default=0.05,
                            help="Lower clamp for NTK/dynamic acceleration-loss weight.")
    group_phys.add_argument("--gradnorm-w-a-max", type=float, default=2.0,
                            help="Upper clamp for NTK/dynamic acceleration-loss weight.")
    group_phys.add_argument("--potential-only-epochs", type=int, default=0,
                            help="Initial epochs that optimise only the residual potential ΔU.")
    group_phys.add_argument("--accel-ramp-epochs", type=int, default=80,
                            help="Epochs used to linearly ramp the acceleration loss from accel_min_factor to full weight.")
    group_phys.add_argument("--accel-min-factor", type=float, default=0.05,
                            help="Minimum acceleration loss factor during curriculum warm-up (floor). "
                                 "0.0 = pure potential-only; 0.05 = small floor to prevent derivative drift.")
    group_phys.add_argument("--a-sign", default="auto", help="Sign of -grad(U). 'auto' or +1/-1.")
    group_phys.add_argument("--use-si", action="store_true", dest="use_si", help="Convert canonical units to SI.")
    group_phys.add_argument("--no-si", action="store_false", dest="use_si", help="Keep dataset units as-is.")
    ap.set_defaults(use_si=True, pin_memory=True)
    ap.set_defaults(use_fourier=False, fourier_append_raw=True, amp=False)
    ap.set_defaults(use_residual_blocks=False)

    # Hardware & Performance
    group_perf = ap.add_argument_group("Performance & Scaler")
    group_perf.add_argument("--num-workers", type=int, default=2)
    group_perf.add_argument("--cache-rows", type=int, default=65536, help="H5BlockDataset cache size.")
    group_perf.add_argument("--fit-rows", type=int, default=500_000, help="Rows for isometric scaler fitting.")
    group_perf.add_argument("--seed", type=int, default=42)
    pin_group = group_perf.add_mutually_exclusive_group()
    pin_group.add_argument("--pin-memory", action="store_true", dest="pin_memory",
                           help="Pin CPU tensors for faster CUDA transfers (default: True on CUDA).")
    pin_group.add_argument("--no-pin-memory", action="store_false", dest="pin_memory",
                           help="Disable pin_memory.")
    group_perf.add_argument("--prefetch-factor", type=int, default=None,
                            help="DataLoader prefetch_factor (only valid when num_workers > 0).")
    preload_group = group_perf.add_mutually_exclusive_group()
    preload_group.add_argument("--preload-data", action="store_true", dest="preload_data",
                               help="Always load the full dataset into CPU RAM before training.")
    preload_group.add_argument("--no-auto-preload", action="store_true", dest="no_auto_preload",
                               help="Disable automatic RAM preload even for small datasets.")
    group_perf.add_argument("--auto-preload-mb", type=float, default=256.0,
                            help="Auto-preload when dataset size is at most this many MB (default: 256).")

    # Direction Loss
    group_dir = ap.add_argument_group("Direction Loss")
    group_dir.add_argument("--direction-loss-weight", type=float, default=0.10,
                           help="Peak weight for the cosine direction loss (lam_dir).")
    group_dir.add_argument("--direction-loss-start-epoch", type=int, default=30,
                           help="Epoch at which direction loss begins to ramp in.")
    group_dir.add_argument("--direction-loss-ramp-epochs", type=int, default=50,
                           help="Epochs over which direction loss ramps from 0 to full weight.")
    group_dir.add_argument("--direction-loss-floor-abs", type=float, default=3e-6,
                           help="||a_true|| threshold below which direction loss is masked out.")
    group_dir.add_argument("--best-ckpt-start-epoch", type=int, default=-1,
                           help="Epoch from which best-checkpoint tracking and patience counting begin. "
                                "-1 = auto (delays to direction_loss_start_epoch + "
                                "direction_loss_ramp_epochs + checkpoint_settle_epochs when direction loss is active).")
    group_dir.add_argument("--checkpoint-settle-epochs", type=int, default=5,
                           help="Additional settled epochs after the direction-loss ramp before auto best-checkpoint tracking starts.")

    # Altitude-Balanced Loss
    group_alt = ap.add_argument_group("Altitude-Balanced Loss")
    group_alt.add_argument("--use-altitude-balanced-loss", action="store_true", default=False,
                           help="Compute acceleration error by altitude bins instead of raw sample mean.")
    group_alt.add_argument("--altitude-bin-width-km", type=float, default=50.0, help="Bin width in km.")
    group_alt.add_argument("--altitude-min-km", type=float, default=_DEFAULT_ALT_MIN_KM, help="Min altitude in km.")
    group_alt.add_argument("--altitude-max-km", type=float, default=_DEFAULT_ALT_MAX_KM, help="Max altitude in km.")

    # Radial / Cross-Radial Loss
    group_rad = ap.add_argument_group("Radial/Cross-Radial Loss")
    group_rad.add_argument("--use-radial-cross-loss", action="store_true", default=False,
                           help="Decompose acceleration error and penalise components.")
    group_rad.add_argument("--radial-loss-weight", type=float, default=0.0, help="Weight for radial loss.")
    group_rad.add_argument("--cross-loss-weight", type=float, default=0.0, help="Weight for cross-radial loss.")

    # Sparse Laplacian Regularization
    group_lap = ap.add_argument_group("Sparse Laplacian Regularization")
    group_lap.add_argument("--use-laplacian-regularization", action="store_true", default=False,
                           help="Apply sparse Laplacian regularization (∇²U=0 physics constraint).")
    group_lap.add_argument("--laplacian-weight", type=float, default=0.0, help="Weight for Laplacian loss.")
    group_lap.add_argument("--laplacian-every-n-batches", type=int, default=5, help="Compute every N batches.")
    group_lap.add_argument("--laplacian-subset-size", type=int, default=512,
                           help="Batch subset size for Hutchinson Laplacian estimator.")
    group_lap.add_argument("--n-hutchinson-samples", type=int, default=4,
                           help="Rademacher samples per Hutchinson trace estimate (K=4 → ~50%% relative error).")

    # PINN architecture
    group_pinn = ap.add_argument_group("PINN Architecture (residual & multi-scale SIREN)")
    res_group = group_pinn.add_mutually_exclusive_group()
    res_group.add_argument("--use-residual-blocks", action="store_true", dest="use_residual_blocks",
                           help="Wrap SIREN hidden layers in SirenResBlock (pre-norm + zero-init skip). "
                                "Recommended for --depth >= 6.")
    res_group.add_argument("--no-residual-blocks", action="store_false", dest="use_residual_blocks",
                           help="Use plain Linear+Sine hidden layers (legacy default).")
    group_pinn.add_argument("--n-bands", type=int, default=1,
                            help="Number of harmonic frequency bands for multi-scale SIREN. "
                                 ">1 uses MultiScaleSirenMLP with band w0s derived from "
                                 "degree_min/degree_max. (default: 1 = standard SirenMLP)")
    group_pinn.add_argument("--grad-accumulation-steps", type=int, default=1,
                            help="Accumulate gradients over N batches before optimizer step. "
                                 "Effective batch = batch_size × N. (default: 1 = no accumulation)")

    # Logging & Quick-check
    group_log = ap.add_argument_group("Logging & Quick-check")
    group_log.add_argument("--log-every", type=int, default=10,
                           help="Print batch-level progress every N batches (0 to disable).")
    group_log.add_argument("--quick-check", action="store_true", default=False,
                           help="Run 1 epoch with 5 train + 2 val batches to verify the full pipeline.")
    group_log.add_argument("--max-train-batches", type=int, default=None,
                           help="Cap the number of training batches per epoch (None = full epoch).")
    group_log.add_argument("--max-val-batches", type=int, default=None,
                           help="Cap the number of validation batches per epoch (None = full epoch).")

    a = ap.parse_args()

    # 1. Resolve Data Path
    script_dir = Path(__file__).resolve().parent
    data_path_raw = a.data or os.environ.get("SPATIAL_CLOUD_INPUT") or os.environ.get("DATASET_PATH")
    
    if data_path_raw is None and a.train_data is None:
        found = _find_latest_dataset(script_dir)
        if found:
            data_path = found
            print(f"[AUTO] No --data provided. Found latest: {data_path}")
        else:
            print("\nError: No input dataset found.")
            print("Please provide --data <file.h5> or --train-data <file.h5>\n")
            sys.exit(1)
    else:
        data_path = Path(data_path_raw) if data_path_raw is not None else Path(a.train_data)

    # 2. Resolve Output Directory
    out_dir = Path(a.out) if a.out else _default_outdir(script_dir)
    if not a.out:
        print(f"[AUTO] Using default output directory: {out_dir}")

    # 3. Auto-sync: read dataset metadata and print auto-detected parameters
    w0_first_val = a.w0_first
    w0_hidden_val = a.w0_hidden

    if data_path.suffix.lower() in (".h5", ".hdf5"):
        try:
            meta_early = DatasetMeta.from_h5(data_path)
            degree_max_meta = meta_early.requested_degree
            degree_min_meta = meta_early.degree_min
            # Also check cloud_config for degree_max
            if degree_max_meta is None and meta_early.cloud_config is not None:
                try:
                    degree_max_meta = int(meta_early.cloud_config.get("degree_max", 0)) or None
                except (TypeError, ValueError):
                    pass

            print("\n" + "=" * 62)
            print("  AUTO-DETECTED DATASET PARAMETERS")
            print("=" * 62)
            print(f"  File         : {data_path.name}")
            print(f"  Unit system  : {meta_early.unit_system}")
            print(f"  degree_max   : {degree_max_meta if degree_max_meta is not None else 'unknown'}")
            print(f"  degree_min   : {degree_min_meta if degree_min_meta is not None else 'unknown (full field)'}")
            print(f"  alt range    : {meta_early.alt_min_km} to {meta_early.alt_max_km} km"
                  if (meta_early.alt_min_km is not None and meta_early.alt_max_km is not None)
                  else "  alt range    : unknown")
            if meta_early.mu_si is not None:
                print(f"  mu_si        : {meta_early.mu_si:.6e} m^3/s^2")
            if meta_early.r_ref_m is not None:
                print(f"  r_ref_m      : {meta_early.r_ref_m:.6e} m")

            # Auto-scale w0 from degree_max if not explicitly set by user
            if degree_max_meta is not None and degree_max_meta > 0:
                auto_w0 = max(10.0, min(100.0, float(degree_max_meta) ** 0.5 * 3.0))
                auto_w0 = round(auto_w0, 1)
                if w0_first_val is None:
                    w0_first_val = auto_w0
                    print(f"  w0_first     : {w0_first_val} [auto from degree_max={degree_max_meta}]")
                else:
                    print(f"  w0_first     : {w0_first_val} [user-specified, auto would be {auto_w0}]")
                if w0_hidden_val is None:
                    w0_hidden_val = auto_w0
                    print(f"  w0_hidden    : {w0_hidden_val} [auto from degree_max={degree_max_meta}]")
                else:
                    print(f"  w0_hidden    : {w0_hidden_val} [user-specified, auto would be {auto_w0}]")
            else:
                if w0_first_val is None:
                    w0_first_val = 30.0
                if w0_hidden_val is None:
                    w0_hidden_val = 30.0
                print(f"  w0_first     : {w0_first_val} [fallback default]")
                print(f"  w0_hidden    : {w0_hidden_val} [fallback default]")

            print("=" * 62 + "\n")
        except Exception as _e:
            print(f"[AUTO] Could not read dataset metadata: {_e}")
            if w0_first_val is None:
                w0_first_val = 30.0
            if w0_hidden_val is None:
                w0_hidden_val = 30.0
    else:
        if w0_first_val is None:
            w0_first_val = 30.0
        if w0_hidden_val is None:
            w0_hidden_val = 30.0

    # 4. Resolve a_sign
    a_sign_val: Union[float, str] = "auto"
    if str(a.a_sign).lower() != "auto":
        try:
            a_sign_val = float(a.a_sign)
        except ValueError:
            print(f"Error: --a-sign must be 'auto', '1.0', or '-1.0'. Got: {a.a_sign}")
            sys.exit(1)

    return TrainConfig(
        data=str(data_path),
        train_data=a.train_data,
        val_data=a.val_data,
        test_data=a.test_data,
        ood_data=a.ood_data,
        suite_manifest=a.suite_manifest,
        out=str(out_dir),
        dataset_name=a.dataset_name,
        seed=a.seed,
        epochs=a.epochs,
        batch_size=a.batch_size,
        val_ratio=a.val_fraction,
        split_seed=(a.split_seed if a.split_seed is not None else a.seed),
        hidden=a.hidden,
        depth=a.depth,
        activation=a.activation,
        dropout=a.dropout,
        w0_first=float(w0_first_val),
        w0_hidden=float(w0_hidden_val),
        lr=a.lr,
        weight_decay=a.weight_decay,
        output_head_lr_mult=float(a.output_head_lr_mult),
        max_grad_norm=a.grad_clip,
        t_max=a.t_max,
        warmup_epochs=max(0, int(a.warmup_epochs)),
        min_lr_ratio=float(a.min_lr_ratio),
        patience=max(1, int(a.patience)),
        w_u=a.w_u,
        w_a=a.w_a,
        gradnorm_mode=str(a.gradnorm_mode),
        dynamic_weights=a.dynamic_weights,
        gradnorm_w_a_min=a.gradnorm_w_a_min,
        gradnorm_w_a_max=a.gradnorm_w_a_max,
        potential_only_epochs=max(0, int(a.potential_only_epochs)),
        accel_ramp_epochs=max(0, int(a.accel_ramp_epochs)),
        accel_min_factor=float(max(0.0, a.accel_min_factor)),
        a_sign=a_sign_val,
        use_si=a.use_si,
        cache_rows=a.cache_rows,
        num_workers=a.num_workers,
        pin_memory=bool(a.pin_memory),
        prefetch_factor=(int(a.prefetch_factor) if a.prefetch_factor is not None else None),
        fit_rows=a.fit_rows,
        amp=bool(a.amp),
        use_fourier=bool(a.use_fourier),
        fourier_append_raw=bool(a.fourier_append_raw),
        fourier_n_features=int(a.fourier_n),
        fourier_sigma=float(a.fourier_sigma),
        fourier_seed=int(a.fourier_seed),
        log_every=max(0, int(a.log_every)),
        preload_data=bool(a.preload_data),
        auto_preload_mb=float(a.auto_preload_mb) if not getattr(a, "no_auto_preload", False) else 0.0,
        quick_check=bool(a.quick_check),
        max_train_batches=(int(a.max_train_batches) if a.max_train_batches is not None else None),
        max_val_batches=(int(a.max_val_batches) if a.max_val_batches is not None else None),
        direction_loss_weight=float(a.direction_loss_weight),
        direction_loss_start_epoch=max(0, int(a.direction_loss_start_epoch)),
        direction_loss_ramp_epochs=max(1, int(a.direction_loss_ramp_epochs)),
        direction_loss_floor_abs=float(a.direction_loss_floor_abs),
        best_ckpt_start_epoch=int(a.best_ckpt_start_epoch),
        checkpoint_settle_epochs=max(0, int(a.checkpoint_settle_epochs)),
        use_altitude_balanced_loss=bool(a.use_altitude_balanced_loss),
        altitude_bin_width_km=float(a.altitude_bin_width_km),
        altitude_min_km=float(a.altitude_min_km),
        altitude_max_km=float(a.altitude_max_km),
        use_radial_cross_loss=bool(a.use_radial_cross_loss),
        radial_loss_weight=float(a.radial_loss_weight),
        cross_loss_weight=float(a.cross_loss_weight),
        use_laplacian_regularization=bool(a.use_laplacian_regularization),
        laplacian_weight=float(a.laplacian_weight),
        laplacian_every_n_batches=max(0, int(a.laplacian_every_n_batches)),
        laplacian_subset_size=max(1, int(a.laplacian_subset_size)),
        n_hutchinson_samples=max(1, int(a.n_hutchinson_samples)),
        use_residual_blocks=bool(a.use_residual_blocks),
        n_bands=max(1, int(a.n_bands)),
        grad_accumulation_steps=max(1, int(a.grad_accumulation_steps)),
    )


# =============================================================================
# DEBUG ENTRY POINT
# =============================================================================
# st_lrps_config.py is a configuration module, NOT a training entry point.
# Launch training via:  python st_lrps_train.py [--data ...] [--out ...]

if __name__ == "__main__":
    import json as _json
    from dataclasses import asdict as _asdict
    _cfg = parse_args()
    print(_json.dumps(_asdict(_cfg), indent=2, default=str))


__all__ = ['TrainConfig', 'parse_args']
