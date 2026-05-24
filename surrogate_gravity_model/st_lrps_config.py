# -*- coding: utf-8 -*-
"""
Configuration and CLI parsing for the lunar potential surrogate trainer.

This module is the single source of truth for training CLI defaults. The PyQt
dashboard builds commands against these names, and ``st_lrps_train.py`` delegates
all argument parsing here. Defaults that describe generated cloud geometry
(altitude range in particular) are pulled from ``spatial_cloud_parameters`` so
the generator and trainer do not drift apart.

Configuration policy
--------------------
* ``TrainConfig`` defaults ARE the recommended production/research configuration.
  There is no hidden "legacy mode". Older configurations are reproduced by passing
  the corresponding CLI flags explicitly (e.g. ``--no-residual-blocks --n-bands 1``)
  or via ``run_ablation_matrix.py``.
* The word "legacy" elsewhere refers only to loading older checkpoints/datasets
  (e.g. ``--allow-legacy-derivative-convention``), not to a default-config mode.
* Experimental input encodings (off by default): ``--use-radial-decay-encoding``
  (physically motivated R/r decay; helps altitude generalization) and
  ``--use-real-sh-basis`` (genuine 4π-normalized real spherical harmonics).
  Treat both as ablation/experimental until benchmarked.
* The Laplacian regulariser is OFF by default and adds no overhead unless
  explicitly requested. JAX migration is out of scope for this stack.
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
    depth: int = 6
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
    accel_ramp_epochs: int = 40
    # Minimum acceleration factor applied even during potential_only phase.
    # Prevents the derivative field from drifting completely unconstrained.
    # Set to 0.0 to restore original full potential-only behaviour.
    accel_min_factor: float = 0.15
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
    preload_data: bool = False        # convenience alias for preload_policy="always"
    auto_preload_mb: float = 2048.0   # auto-preload when dataset fits in this many MB
    # Preload policy: "auto" (preload if estimated size <= auto_preload_mb),
    # "always" (always preload), or "never" (always stream from HDF5).
    preload_policy: str = "auto"

    # Quick-check mode: run 1 epoch with 5 train + 2 val batches to verify the
    # full pipeline (CUDA, autograd, checkpoint, metrics) in under a minute.
    quick_check: bool = False
    max_train_batches: Optional[int] = None  # cap training batches (None = full epoch)
    max_val_batches: Optional[int] = None    # cap validation batches (None = full epoch)

    # Acceleration direction loss -> penalises angular error between a_pred and a_true.
    # L_dir = mean(1 - cos_sim(a_pred, a_true)) for points where ||a_true|| > floor.
    # Ramped in after direction_loss_start_epoch to avoid destabilising early training.
    direction_loss_weight: float = 0.20
    direction_loss_start_epoch: int = 10
    direction_loss_ramp_epochs: int = 40
    direction_loss_floor_abs: float = 1e-7   # mask threshold on ||a_true||

    # Best-checkpoint selection burn-in.
    # -1 (default) = auto: if direction_loss_weight > 0, delays to
    # direction_loss_start_epoch + direction_loss_ramp_epochs + checkpoint_settle_epochs.
    # This prevents early epochs from winning the checkpoint race before
    # direction-aware training has started and settled.
    # Set to 0 to disable and start tracking from epoch 0.
    best_ckpt_start_epoch: int = -1
    checkpoint_settle_epochs: int = 5

    # Best-checkpoint metric selection.
    # "hybrid" (default): val_base_loss + hybrid_direction_alpha * val_direction_loss.
    # "total_loss": val reference loss only.
    # "direction_loss": val direction loss only (experimental, not recommended alone).
    best_metric: str = "hybrid"
    hybrid_direction_alpha: float = 0.30
    save_epoch_snapshots: bool = False
    epoch_snapshot_every: int = 1

    # Optional altitude-balanced residual loss.
    # Defaults pulled from spatial_cloud_parameters.DEFAULT_SPATIAL_CLOUD_CONFIG
    # so training envelope always matches the generated dataset without edits.
    use_altitude_balanced_loss: bool = True
    altitude_bin_width_km: float = 50.0
    altitude_min_km: float = _DEFAULT_ALT_MIN_KM
    altitude_max_km: float = _DEFAULT_ALT_MAX_KM

    # Optional radial / cross-radial acceleration penalties.
    use_radial_cross_loss: bool = True
    radial_loss_weight: float = 0.05
    cross_loss_weight: float = 0.10

    # Optional sparse Laplacian regularisation for the residual potential.
    # Uses the Hutchinson stochastic trace estimator (AMP-compatible, O(K) passes).
    use_laplacian_regularization: bool = False
    laplacian_weight: float = 0.0
    laplacian_every_n_batches: int = 5
    laplacian_subset_size: int = 512
    n_hutchinson_samples: int = 4   # Rademacher samples per Laplacian estimate
    collocation_laplacian_weight: float = 0.0
    laplacian_mode: str = "diagnostic"    # "off" | "diagnostic" | "train"
    collocation_laplacian_every: int = 25  # optimizer steps between collocation Laplacian evaluations
    # Collocation altitude bounds (defaults to altitude_min_km / altitude_max_km when None)
    collocation_alt_min_km: Optional[float] = None
    collocation_alt_max_km: Optional[float] = None
    # Separate control over collocation samples (alias for laplacian_subset_size in collocation call)
    collocation_laplacian_samples: int = 512
    collocation_laplacian_hutchinson_samples: int = 4

    # Input encodings. At most ONE of {use_fourier, use_sh_encoding,
    # use_radial_separation, use_radial_decay_encoding, use_real_sh_basis} may be
    # True. All default to False → raw Cartesian xyz input.
    #   use_sh_encoding         : SHInspiredAngularEncoding (Cartesian angular polynomial).
    #   use_radial_separation   : RadialSeparationEncoding [r, ux, uy, uz].
    #   use_radial_decay_encoding: RadialDecayEncoding (R/r decay powers; experimental).
    #   use_real_sh_basis       : RealSHBasisEncoding (real spherical harmonics; experimental).
    use_sh_encoding: bool = False
    sh_encoding_degree: int = 4          # max polynomial degree (1..8)
    sh_append_raw: bool = True           # always True (required by SHInspiredAngularEncoding)
    use_radial_separation: bool = False
    radial_append_raw: bool = False      # True → 7-dim output, False → 4-dim

    # Radial decay-aware encoding (experimental). Encodes the R/r radial decay of
    # SH residual terms via inverse-radial powers, which is important for altitude
    # generalization. See RadialDecayEncoding.
    use_radial_decay_encoding: bool = False
    radial_decay_max_power: int = 4
    radial_decay_append_raw: bool = True

    # Real spherical-harmonic angular basis (experimental). Genuine real SH up to
    # real_sh_degree (orthonormal recurrence). See RealSHBasisEncoding.
    use_real_sh_basis: bool = False
    real_sh_degree: int = 4
    real_sh_append_raw: bool = True
    real_sh_include_radial: bool = True

    # Residual SIREN blocks — wraps hidden layers in SirenResBlock.
    # Recommended for depth >= 6; adds LayerNorm + zero-init skip per block.
    # Default on (recommended); disable with --no-residual-blocks.
    use_residual_blocks: bool = True

    # Multi-scale SIREN — parallel frequency bands matched to the harmonic range.
    # n_bands > 1 uses a multi-scale SIREN; requires degree_min/degree_max metadata.
    # Default 3 (recommended); set --n-bands 1 for a single-scale SirenMLP.
    n_bands: int = 3
    # Multi-scale composition: "concat_shared" (parallel bands -> concat -> shared
    # trunk, default) or "additive" (per-band trunks summed: dU = sum_k dU_k).
    multiscale_mode: str = "concat_shared"

    # Harmonic degree range of the dataset. Resolved from HDF5 metadata by the
    # engine BEFORE the model is built, then persisted to config.json and the
    # checkpoint so evaluation reconstructs the identical multi-scale spectrum.
    # Leaving these None at build time for n_bands>1 is a hard error (no silent
    # fallback to 0/50, which silently corrupted reloaded MultiScale SIRENs).
    degree_min: Optional[int] = None
    degree_max: Optional[int] = None
    # Resolved per-band SIREN frequencies (filled in by the engine for n_bands>1).
    w0_bands: Optional[list] = None

    # Target scaler robustness. "max" lets a single outlier shrink every
    # normalized residual target; "hybrid" caps the scale at
    # target_scale_multiplier * RMS, which is far more robust. x scaling is
    # always origin-fixed max-radius and is NOT affected by these.
    u_scale_mode: str = "hybrid"   # "max" | "rms" | "hybrid"
    a_scale_mode: str = "hybrid"
    target_scale_multiplier: float = 6.0

    # Dataset convention safety. Datasets generated before the dP_dphi sign fix
    # have sign-flipped latitude acceleration; training on them is silently
    # wrong. By default such datasets are rejected; set True only for inspection.
    allow_legacy_derivative_convention: bool = False

    # Determinism / cuDNN. Defaults preserve prior behavior.
    deterministic: bool = True
    benchmark_cudnn: bool = False

    # Gradient accumulation — accumulate gradients over N batches before stepping.
    # Effective batch size = batch_size * grad_accumulation_steps.
    grad_accumulation_steps: int = 1

import dataclasses as _dataclasses
_TC_DEFAULTS: dict = {
    f.name: f.default
    for f in _dataclasses.fields(TrainConfig)
    if f.default is not _dataclasses.MISSING
}


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
    group_arch.add_argument("--depth", type=int, default=_TC_DEFAULTS["depth"],
                            help="Number of hidden layers (default: 6).")
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
    group_phys.add_argument("--accel-ramp-epochs", type=int, default=_TC_DEFAULTS["accel_ramp_epochs"],
                            help="Epochs used to linearly ramp the acceleration loss from accel_min_factor to full weight (default: 40).")
    group_phys.add_argument("--accel-min-factor", type=float, default=_TC_DEFAULTS["accel_min_factor"],
                            help="Minimum acceleration loss factor during curriculum warm-up (floor). "
                                 "0.0 = pure potential-only; 0.15 = floor to prevent derivative drift (default: 0.15).")
    group_phys.add_argument("--a-sign", default="auto", help="Sign of -grad(U). 'auto' or +1/-1.")
    group_phys.add_argument("--use-si", action="store_true", dest="use_si", help="Convert canonical units to SI.")
    group_phys.add_argument("--no-si", action="store_false", dest="use_si", help="Keep dataset units as-is.")
    ap.set_defaults(use_si=True, pin_memory=True)
    ap.set_defaults(use_fourier=False, fourier_append_raw=True, amp=False)
    ap.set_defaults(use_residual_blocks=_TC_DEFAULTS["use_residual_blocks"])
    ap.set_defaults(
        use_altitude_balanced_loss=_TC_DEFAULTS["use_altitude_balanced_loss"],
        use_radial_cross_loss=_TC_DEFAULTS["use_radial_cross_loss"],
    )

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
                               help="Always load the full dataset into CPU RAM before training "
                                    "(alias for --preload-policy always).")
    preload_group.add_argument("--no-auto-preload", action="store_true", dest="no_auto_preload",
                               help="Disable automatic RAM preload even for small datasets.")
    group_perf.add_argument("--auto-preload-mb", type=float, default=_TC_DEFAULTS["auto_preload_mb"],
                            help="Auto-preload when dataset size is at most this many MB (default: 2048).")
    group_perf.add_argument("--preload-policy", choices=["auto", "always", "never"],
                            default=_TC_DEFAULTS["preload_policy"],
                            help="RAM preload policy: 'auto' (preload if estimated size <= --auto-preload-mb "
                                 "and RAM allows), 'always', or 'never' (default: auto).")

    # Direction Loss
    group_dir = ap.add_argument_group("Direction Loss")
    group_dir.add_argument("--direction-loss-weight", type=float, default=_TC_DEFAULTS['direction_loss_weight'],
                           help="Peak weight for the cosine direction loss (lam_dir).")
    group_dir.add_argument("--direction-loss-start-epoch", type=int, default=_TC_DEFAULTS['direction_loss_start_epoch'],
                           help="Epoch at which direction loss begins to ramp in.")
    group_dir.add_argument("--direction-loss-ramp-epochs", type=int, default=_TC_DEFAULTS['direction_loss_ramp_epochs'],
                           help="Epochs over which direction loss ramps from 0 to full weight.")
    group_dir.add_argument("--direction-loss-floor-abs", type=float, default=_TC_DEFAULTS['direction_loss_floor_abs'],
                           help="||a_true|| threshold below which direction loss is masked out.")
    group_dir.add_argument("--best-ckpt-start-epoch", type=int, default=-1,
                           help="Epoch from which best-checkpoint tracking and patience counting begin. "
                                "-1 = auto (delays to direction_loss_start_epoch + "
                                "direction_loss_ramp_epochs + checkpoint_settle_epochs when direction loss is active).")
    group_dir.add_argument("--checkpoint-settle-epochs", type=int, default=5,
                           help="Additional settled epochs after the direction-loss ramp before auto best-checkpoint tracking starts.")
    group_dir.add_argument("--best-metric",
                           choices=["total_loss", "direction_loss", "hybrid"],
                           default=_TC_DEFAULTS['best_metric'],
                           help="Metric used for best-checkpoint selection. "
                                "'total_loss': val ref loss (default, backward-compatible). "
                                "'hybrid': val_loss + alpha * val_direction_loss. "
                                "'direction_loss': val direction loss only (experimental).")
    group_dir.add_argument("--hybrid-direction-alpha", type=float, default=_TC_DEFAULTS['hybrid_direction_alpha'],
                           help="Weight alpha for direction loss in hybrid best-metric: "
                                "score = val_loss + alpha * val_direction_loss.")
    group_dir.add_argument(
        "--save-epoch-snapshots",
        action="store_true",
        default=_TC_DEFAULTS["save_epoch_snapshots"],
        help="Also write checkpoints/ckpt_epoch_XXXXXX.pt snapshots at the configured interval.",
    )
    group_dir.add_argument(
        "--epoch-snapshot-every",
        type=int,
        default=_TC_DEFAULTS["epoch_snapshot_every"],
        help="Write an epoch snapshot every N epochs when --save-epoch-snapshots is enabled.",
    )

    # Altitude-Balanced Loss
    group_alt = ap.add_argument_group("Altitude-Balanced Loss")
    alt_bal_group = group_alt.add_mutually_exclusive_group()
    alt_bal_group.add_argument("--use-altitude-balanced-loss", action="store_true", dest="use_altitude_balanced_loss",
                               help="Compute acceleration error by altitude bins instead of raw sample mean (default: on).")
    alt_bal_group.add_argument("--no-altitude-balanced-loss", action="store_false", dest="use_altitude_balanced_loss",
                               help="Use the raw per-sample mean instead of altitude-binned balancing.")
    group_alt.add_argument("--altitude-bin-width-km", type=float, default=50.0, help="Bin width in km.")
    group_alt.add_argument("--altitude-min-km", type=float, default=_DEFAULT_ALT_MIN_KM, help="Min altitude in km.")
    group_alt.add_argument("--altitude-max-km", type=float, default=_DEFAULT_ALT_MAX_KM, help="Max altitude in km.")

    # Radial / Cross-Radial Loss
    group_rad = ap.add_argument_group("Radial/Cross-Radial Loss")
    rad_cross_group = group_rad.add_mutually_exclusive_group()
    rad_cross_group.add_argument("--use-radial-cross-loss", action="store_true", dest="use_radial_cross_loss",
                                 help="Decompose acceleration error and penalise radial/cross components (default: on).")
    rad_cross_group.add_argument("--no-radial-cross-loss", action="store_false", dest="use_radial_cross_loss",
                                 help="Disable the radial/cross-radial acceleration penalties.")
    group_rad.add_argument("--radial-loss-weight", type=float, default=_TC_DEFAULTS["radial_loss_weight"],
                           help="Weight for radial loss (default: 0.05).")
    group_rad.add_argument("--cross-loss-weight", type=float, default=_TC_DEFAULTS["cross_loss_weight"],
                           help="Weight for cross-radial loss (default: 0.10).")

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
    group_lap.add_argument("--laplacian-mode",
        choices=["off", "diagnostic", "train"], default=_TC_DEFAULTS.get('laplacian_mode', 'diagnostic'),
        help="Laplacian regularization mode. "
             "diagnostic = logs the physics (Laplace) violation only, no gradient is backpropagated; "
             "train = backpropagates the Laplacian penalty into model weights (create_graph=True); "
             "off = skip entirely. Default: diagnostic.")
    group_lap.add_argument("--collocation-laplacian-every", type=int, default=25,
        help="Optimizer steps between collocation Laplacian evaluations (default: 25).")
    group_lap.add_argument("--collocation-alt-min-km", type=float, default=None,
        help="Min altitude in km for collocation Laplacian points (default: use altitude-min-km).")
    group_lap.add_argument("--collocation-alt-max-km", type=float, default=None,
        help="Max altitude in km for collocation Laplacian points (default: use altitude-max-km).")
    group_lap.add_argument("--collocation-laplacian-weight", type=float,
        default=_TC_DEFAULTS.get('collocation_laplacian_weight', 0.0),
        help="Weight applied to collocation Laplacian loss when mode='train'.")
    group_lap.add_argument("--collocation-laplacian-samples", type=int,
        default=_TC_DEFAULTS.get('collocation_laplacian_samples', 512),
        help="Number of collocation points for the Laplacian estimator (default: 512).")
    group_lap.add_argument("--collocation-laplacian-hutchinson-samples", type=int,
        default=_TC_DEFAULTS.get('collocation_laplacian_hutchinson_samples', 4),
        help="Hutchinson samples per collocation Laplacian estimate (default: 4).")

    # Angular / Radial Input Encoding
    group_enc = ap.add_argument_group("Input Encoding (SH-angular or radial separation)")
    enc_sh_group = group_enc.add_mutually_exclusive_group()
    enc_sh_group.add_argument(
        "--use-sh-encoding", action="store_true", dest="use_sh_encoding",
        help="Use SHInspiredAngularEncoding (Cartesian angular polynomial). "
             "Mutually exclusive with --use-radial-separation.",
    )
    enc_sh_group.add_argument(
        "--no-sh-encoding", action="store_false", dest="use_sh_encoding",
        help="Disable SH angular polynomial encoding (default).",
    )
    group_enc.add_argument(
        "--sh-encoding-degree", type=int, default=_TC_DEFAULTS.get("sh_encoding_degree", 4),
        help="Max polynomial degree for SH-inspired angular encoding (1..8, default: 4).",
    )
    sh_raw_group = group_enc.add_mutually_exclusive_group()
    sh_raw_group.add_argument(
        "--sh-append-raw", action="store_true", dest="sh_append_raw",
        help="Append raw xyz coordinates to SH encoding output (required; default: True).",
    )
    sh_raw_group.add_argument(
        "--no-sh-append-raw", action="store_false", dest="sh_append_raw",
        help="Do not append raw xyz to SH encoding (will raise if SH encoding is active).",
    )
    enc_rad_group = group_enc.add_mutually_exclusive_group()
    enc_rad_group.add_argument(
        "--use-radial-separation", action="store_true", dest="use_radial_separation",
        help="Use RadialSeparationEncoding [r_norm, ux, uy, uz]. "
             "Mutually exclusive with --use-sh-encoding.",
    )
    enc_rad_group.add_argument(
        "--no-radial-separation", action="store_false", dest="use_radial_separation",
        help="Disable radial separation encoding (default).",
    )
    rad_raw_group = group_enc.add_mutually_exclusive_group()
    rad_raw_group.add_argument(
        "--radial-append-raw", action="store_true", dest="radial_append_raw",
        help="Append raw xyz to radial separation encoding (7-dim output).",
    )
    rad_raw_group.add_argument(
        "--no-radial-append-raw", action="store_false", dest="radial_append_raw",
        help="Do not append raw xyz to radial encoding (4-dim output, default).",
    )

    # Radial decay-aware encoding (experimental).
    dec_group = group_enc.add_mutually_exclusive_group()
    dec_group.add_argument(
        "--use-radial-decay-encoding", action="store_true", dest="use_radial_decay_encoding",
        help="Use RadialDecayEncoding (R/r inverse-radial decay powers). Experimental; "
             "mutually exclusive with the other encodings.",
    )
    dec_group.add_argument(
        "--no-radial-decay-encoding", action="store_false", dest="use_radial_decay_encoding",
        help="Disable radial decay encoding (default).",
    )
    group_enc.add_argument(
        "--radial-decay-max-power", type=int, default=_TC_DEFAULTS.get("radial_decay_max_power", 4),
        help="Highest inverse-radial power for RadialDecayEncoding (default: 4).",
    )
    dec_raw_group = group_enc.add_mutually_exclusive_group()
    dec_raw_group.add_argument(
        "--radial-decay-append-raw", action="store_true", dest="radial_decay_append_raw",
        help="Append raw xyz to radial decay encoding (default).",
    )
    dec_raw_group.add_argument(
        "--no-radial-decay-append-raw", action="store_false", dest="radial_decay_append_raw",
        help="Do not append raw xyz to radial decay encoding.",
    )

    # Real spherical-harmonic angular basis (experimental).
    rsh_group = group_enc.add_mutually_exclusive_group()
    rsh_group.add_argument(
        "--use-real-sh-basis", action="store_true", dest="use_real_sh_basis",
        help="Use RealSHBasisEncoding (genuine real spherical harmonics). Experimental; "
             "mutually exclusive with the other encodings.",
    )
    rsh_group.add_argument(
        "--no-real-sh-basis", action="store_false", dest="use_real_sh_basis",
        help="Disable real SH basis encoding (default).",
    )
    group_enc.add_argument(
        "--real-sh-degree", type=int, default=_TC_DEFAULTS.get("real_sh_degree", 4),
        help="Max degree L for RealSHBasisEncoding ((L+1)^2 angular terms, default: 4).",
    )
    rsh_raw_group = group_enc.add_mutually_exclusive_group()
    rsh_raw_group.add_argument(
        "--real-sh-append-raw", action="store_true", dest="real_sh_append_raw",
        help="Append raw xyz to real SH basis encoding (default).",
    )
    rsh_raw_group.add_argument(
        "--no-real-sh-append-raw", action="store_false", dest="real_sh_append_raw",
        help="Do not append raw xyz to real SH basis encoding.",
    )
    rsh_rad_group = group_enc.add_mutually_exclusive_group()
    rsh_rad_group.add_argument(
        "--real-sh-include-radial", action="store_true", dest="real_sh_include_radial",
        help="Prepend the scaled radial magnitude to the real SH basis (default).",
    )
    rsh_rad_group.add_argument(
        "--no-real-sh-include-radial", action="store_false", dest="real_sh_include_radial",
        help="Angular-only real SH basis (no radial feature).",
    )

    ap.set_defaults(
        use_sh_encoding=False, sh_encoding_degree=_TC_DEFAULTS.get("sh_encoding_degree", 4),
        sh_append_raw=True,
        use_radial_separation=False, radial_append_raw=False,
        use_radial_decay_encoding=False,
        radial_decay_append_raw=_TC_DEFAULTS.get("radial_decay_append_raw", True),
        use_real_sh_basis=False,
        real_sh_append_raw=_TC_DEFAULTS.get("real_sh_append_raw", True),
        real_sh_include_radial=_TC_DEFAULTS.get("real_sh_include_radial", True),
    )

    # PINN architecture
    group_pinn = ap.add_argument_group("PINN Architecture (residual & multi-scale SIREN)")
    res_group = group_pinn.add_mutually_exclusive_group()
    res_group.add_argument("--use-residual-blocks", action="store_true", dest="use_residual_blocks",
                           help="Wrap SIREN hidden layers in SirenResBlock (pre-norm + zero-init skip). "
                                "Recommended for --depth >= 6.")
    res_group.add_argument("--no-residual-blocks", action="store_false", dest="use_residual_blocks",
                           help="Use plain Linear+Sine hidden layers instead of residual blocks.")
    group_pinn.add_argument("--n-bands", type=int, default=_TC_DEFAULTS["n_bands"],
                            help="Number of harmonic frequency bands for multi-scale SIREN. "
                                 ">1 uses a multi-scale SIREN with band w0s derived from "
                                 "degree_min/degree_max. (default: 3; requires degree_max "
                                 "metadata. Use 1 for a standard single-scale SirenMLP.)")
    group_pinn.add_argument("--multiscale-mode", choices=["concat_shared", "additive"],
                            default=_TC_DEFAULTS.get("multiscale_mode", "concat_shared"),
                            help="Multi-scale composition when n_bands>1: 'concat_shared' "
                                 "(parallel bands -> concat -> shared trunk, default) or "
                                 "'additive' (per-band trunks summed; experimental).")
    group_pinn.add_argument("--grad-accumulation-steps", type=int, default=1,
                            help="Accumulate gradients over N batches before optimizer step. "
                                 "Effective batch = batch_size × N. (default: 1 = no accumulation)")

    # Scaler robustness
    group_scaler = ap.add_argument_group("Target Scaler")
    group_scaler.add_argument("--u-scale-mode", choices=["max", "rms", "hybrid"],
                              default=_TC_DEFAULTS.get("u_scale_mode", "hybrid"),
                              help="Isometric scale rule for the residual potential target "
                                   "(default: hybrid = robust to outliers).")
    group_scaler.add_argument("--a-scale-mode", choices=["max", "rms", "hybrid"],
                              default=_TC_DEFAULTS.get("a_scale_mode", "hybrid"),
                              help="Isometric scale rule for the residual acceleration target "
                                   "(default: hybrid).")
    group_scaler.add_argument("--target-scale-multiplier", type=float,
                              default=_TC_DEFAULTS.get("target_scale_multiplier", 6.0),
                              help="RMS expansion factor for rms/hybrid target scaling (default: 6.0).")

    # Dataset convention / determinism
    group_safety = ap.add_argument_group("Dataset Safety & Determinism")
    group_safety.add_argument("--allow-legacy-derivative-convention", action="store_true",
                              default=False,
                              help="Permit training on datasets generated before the dP_dphi "
                                   "sign fix (sign-flipped latitude acceleration). Inspection only.")
    det_group = group_safety.add_mutually_exclusive_group()
    det_group.add_argument("--deterministic", action="store_true", dest="deterministic",
                           help="Set deterministic cuDNN (default: True).")
    det_group.add_argument("--no-deterministic", action="store_false", dest="deterministic",
                           help="Disable deterministic cuDNN.")
    group_safety.add_argument("--benchmark-cudnn", action="store_true", default=False,
                              help="Enable cudnn.benchmark autotuner (non-deterministic).")
    ap.set_defaults(deterministic=True)

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

    # ---------------------------------------------------------------------------
    # TrainConfig is the single source of truth for the recommended configuration.
    # There is no hidden "legacy mode": the dataclass defaults ARE the recommended
    # production/research architecture. Any older configuration is reproduced by
    # passing the corresponding CLI flags explicitly (or via run_ablation_matrix.py).
    #
    # The minimal recommended run is simply:
    #
    #   python st_lrps_train.py --data path/to/train.h5 --epochs 250
    #
    # Notes:
    #   - n_bands=3 (multi-scale SIREN) REQUIRES degree_max in the dataset metadata.
    #     Use --n-bands 1 for datasets without it.
    #   - If direction-loss-floor-abs=1e-7 causes noise in low-residual regions,
    #     increase to 3e-7 or 1e-6.
    #   - If VRAM is insufficient: --batch-size 4096 --grad-accumulation-steps 4
    #     (an advisory warning is printed at startup when batch_size looks large
    #     for the detected GPU).
    #   - Experimental input encodings (off by default): --use-radial-decay-encoding
    #     (physically motivated R/r decay) and --use-real-sh-basis (real spherical
    #     harmonic angular basis). See run_ablation_matrix.py for controlled studies.
    # ---------------------------------------------------------------------------

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
        preload_policy=("never" if getattr(a, "no_auto_preload", False) else str(a.preload_policy)),
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
        use_sh_encoding=bool(a.use_sh_encoding),
        sh_encoding_degree=max(1, min(8, int(a.sh_encoding_degree))),
        sh_append_raw=bool(a.sh_append_raw),
        use_radial_separation=bool(a.use_radial_separation),
        radial_append_raw=bool(a.radial_append_raw),
        use_radial_decay_encoding=bool(a.use_radial_decay_encoding),
        radial_decay_max_power=max(1, int(a.radial_decay_max_power)),
        radial_decay_append_raw=bool(a.radial_decay_append_raw),
        use_real_sh_basis=bool(a.use_real_sh_basis),
        real_sh_degree=max(0, min(8, int(a.real_sh_degree))),
        real_sh_append_raw=bool(a.real_sh_append_raw),
        real_sh_include_radial=bool(a.real_sh_include_radial),
        use_residual_blocks=bool(a.use_residual_blocks),
        n_bands=max(1, int(a.n_bands)),
        multiscale_mode=str(a.multiscale_mode),
        grad_accumulation_steps=max(1, int(a.grad_accumulation_steps)),
        best_metric=str(a.best_metric),
        hybrid_direction_alpha=float(a.hybrid_direction_alpha),
        save_epoch_snapshots=bool(a.save_epoch_snapshots),
        epoch_snapshot_every=max(1, int(a.epoch_snapshot_every)),
        u_scale_mode=str(a.u_scale_mode),
        a_scale_mode=str(a.a_scale_mode),
        target_scale_multiplier=float(a.target_scale_multiplier),
        allow_legacy_derivative_convention=bool(a.allow_legacy_derivative_convention),
        deterministic=bool(a.deterministic),
        benchmark_cudnn=bool(a.benchmark_cudnn),
        laplacian_mode=str(a.laplacian_mode),
        collocation_laplacian_every=max(1, int(a.collocation_laplacian_every)),
        collocation_alt_min_km=(float(a.collocation_alt_min_km) if a.collocation_alt_min_km is not None else None),
        collocation_alt_max_km=(float(a.collocation_alt_max_km) if a.collocation_alt_max_km is not None else None),
        collocation_laplacian_weight=float(a.collocation_laplacian_weight),
        collocation_laplacian_samples=max(1, int(a.collocation_laplacian_samples)),
        collocation_laplacian_hutchinson_samples=max(1, int(a.collocation_laplacian_hutchinson_samples)),
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
