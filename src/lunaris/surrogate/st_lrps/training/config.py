"""
Configuration and CLI parsing for the lunar potential surrogate trainer.

This module is the single source of truth for training CLI defaults. The PyQt
dashboard builds commands against these names, and ``st_lrps.training.cli``
delegates all argument parsing here. Defaults that describe generated cloud
geometry (altitude range in particular) are pulled from
``st_lrps.data.spatial_cloud_parameters`` so the generator and trainer do not
drift apart.

Configuration policy
--------------------
* ``TrainConfig`` defaults are the strong benchmark-candidate configuration:
  SH25 -> SH200 data, 100-1000 km training shell, physical radial features,
  residual SIREN blocks, and multi-band SIREN. Use explicit CLI flags or
  scenario presets for ablations away from that profile.
* Experimental input encodings (off by default): ``--use-radial-decay-encoding``
  (scaled inverse-radius decay features inspired by the R/r radial decay of
  spherical-harmonic terms; evaluate through ablation) and
  ``--use-real-sh-basis`` (genuine 4π-normalized real spherical harmonics).
  Treat both as ablation/experimental until benchmarked.
* The sparse Laplacian regulariser follows the strong benchmark-candidate
  profile by default and can be disabled with
  ``--no-laplacian-regularization``.
"""

from __future__ import annotations

import argparse
import dataclasses as _dataclasses
import datetime
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lunaris.common.paths import project_root_from_file

# Pull altitude defaults from the cloud-generation SSOT so both modules
# always agree on the training envelope without manual synchronisation.
try:
    from lunaris.surrogate.st_lrps.data.spatial_cloud_parameters import (
        DEFAULT_SPATIAL_CLOUD_CONFIG as _CLOUD_CFG,
    )
except ImportError:  # pragma: no cover - cloud-param defaults are optional
    _CLOUD_CFG = None

_DEFAULT_ALT_MIN_KM: float = float(getattr(_CLOUD_CFG, "alt_min_km", 100.0))
_DEFAULT_ALT_MAX_KM: float = float(getattr(_CLOUD_CFG, "alt_max_km", 1000.0))


def _load_dataset_helpers() -> tuple[Any, Any]:
    from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta, _find_latest_dataset

    return DatasetMeta, _find_latest_dataset


@dataclass
class TrainConfig:
    """Hyperparameter configuration for the Physics-Informed Neural Network."""

    # Runtime marker (NOT a dataclass field — left unannotated so it is excluded
    # from asdict()/serialization): set True by the CLI when --model-preset was
    # passed explicitly, read by apply_model_preset to distinguish an explicit
    # preset from the default.
    _model_preset_explicit = False
    _altitude_bounds_explicit = False

    data: str
    out: str
    dataset_name: str = "data"
    train_data: str | None = None
    val_data: str | None = None
    test_data: str | None = None
    ood_data: str | None = None
    suite_manifest: str | None = None  # path to suite manifest.json (provenance only)

    # Run-level preset (orthogonal to model_preset, which is architecture-only).
    # "custom" (default) applies nothing and preserves every other default, so it
    # is fully backward-compatible. "development" is the fast interpolation default,
    # "quick" is a pipeline smoke test, and "paper" is the scientifically defensible
    # posture (generalization split + deterministic + enforced preflight + no AMP).
    # See RUN_PRESETS / apply_run_preset.
    run_preset: str = "custom"

    seed: int = 42
    epochs: int = 400
    batch_size: int = 8192

    cache_rows: int = 65_536
    sampler_block_size: int = 65_536
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int | None = None  # only used when num_workers > 0

    val_ratio: float = 0.1
    split_seed: int | None = None
    split_policy: str = "seeded_random"
    test_fraction: float = 0.0
    # Spatial-block split knobs (Moon-fixed lon/lat grid holdout).
    spatial_lon_bins: int = 12
    spatial_lat_bins: int = 6
    spatial_val_block_fraction: float | None = None  # defaults to val_ratio
    spatial_test_block_fraction: float | None = None  # defaults to test_fraction
    spatial_altitude_bins: int = 4
    # OOD altitude split knobs. Thresholds override the fraction-based holdout.
    ood_low_altitude_max_km: float | None = None
    ood_high_altitude_min_km: float | None = None
    ood_holdout_fraction: float = 0.2

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
    t_max: int | None = 390
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.05
    patience: int = 30

    # Loss weighting
    w_u: float = 1.0
    w_a: float = 1.0
    # gradnorm_mode: "ntk_init" (default) | "fixed" | "dynamic"
    # "ntk_init" computes the gradient-norm ratio once at training start then
    # freezes w_a. This avoids the instability of repeated Hessian-level updates
    # (a_pred = ∇U makes ∂L_a/∂W a second-order quantity). "dynamic" is the
    # EMA-based GradNorm variant, kept for ablation only.
    gradnorm_mode: str = "ntk_init"
    gradnorm_w_a_min: float = 0.05
    gradnorm_w_a_max: float = 2.0
    potential_only_epochs: int = 10
    accel_ramp_epochs: int = 80
    # Minimum acceleration factor applied even during potential_only phase.
    # Prevents the derivative field from drifting completely unconstrained.
    # Set to 0.0 for a pure potential-only warm-up (no acceleration floor).
    accel_min_factor: float = 0.05
    a_sign: float | str = "auto"

    # SSOT / Physics Meta behavior
    use_si: bool = True
    fit_rows: int = 500_000
    fit_seed: int = 123
    fit_chunk_rows: int = 131_072

    # CUDA-only optimizer fusion is an opt-in performance detail with a safe
    # AdamW fallback when the installed torch build does not support it.
    use_fused_optimizer: bool = True

    amp: bool = False

    # Architecture preset layer. "baseline_raw" keeps all input encodings off
    # and is the control representation. "recommended_physical_radial_decay"
    # enables the physically scaled R_ref/r encoding. "custom" preserves manual
    # flag-level control for ablations and old configs.
    model_preset: str = "recommended_physical_radial_decay"
    runtime_model_kind: str = "potential_autograd"
    output_dim: int = 1

    # Fourier/RFF embedding → only for non-sine MLPs (activation="silu"/"tanh"/"softplus").
    # MUST NOT be combined with activation="sine" (SIREN): train() raises ValueError.
    use_fourier: bool = False
    fourier_append_raw: bool = True
    fourier_n_features: int = 256   # n → 2n-dim embedding (sin + cos)
    fourier_sigma: float = 1.0      # std of frequency matrix B
    fourier_seed: int = 42

    # Progress logging → log every N batches; 0 to disable
    log_every: int = 10
    # "fixed" honors log_every literally; "auto" derives ~10 progress updates
    # per epoch from the batch count (always logging the first and last batch).
    log_every_mode: str = "auto"

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
    max_train_batches: int | None = None  # cap training batches (None = full epoch)
    max_val_batches: int | None = None    # cap validation batches (None = full epoch)
    overfit_batches: int | None = None    # repeat the first N train batches every epoch

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

    # Best-checkpoint metric selection.
    # "hybrid" (default): val_base_loss + hybrid_direction_alpha * val_direction_loss.
    # "val_total_loss": validation reference loss only ("total_loss" is accepted as an old alias).
    # "val_base_loss": validation U + acceleration MSE only.
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
    cross_loss_weight: float = 0.05

    # Optional sparse Laplacian regularisation for the residual potential.
    # Uses an exact 3D coordinate-basis trace (Hutchinson fallback for other
    # dimensions; AMP-compatible).
    use_laplacian_regularization: bool = True
    laplacian_weight: float = 2e-9
    laplacian_every_n_batches: int = 100
    laplacian_subset_size: int = 512
    n_hutchinson_samples: int = 4   # Rademacher samples per Laplacian estimate
    collocation_laplacian_weight: float = 0.0
    laplacian_mode: str = "diagnostic"    # "off" | "diagnostic" | "train"
    collocation_laplacian_every: int = 25  # optimizer steps between collocation Laplacian evaluations
    # Collocation altitude bounds (defaults to altitude_min_km / altitude_max_km when None)
    collocation_alt_min_km: float | None = None
    collocation_alt_max_km: float | None = None
    # Separate control over collocation samples (alias for laplacian_subset_size in collocation call)
    collocation_laplacian_samples: int = 512
    collocation_laplacian_hutchinson_samples: int = 4

    # Input encodings. At most ONE of {use_fourier, use_sh_encoding,
    # use_radial_separation, use_radial_decay_encoding, use_real_sh_basis} may be
    # True. All default to False → raw Cartesian xyz input.
    #   use_sh_encoding         : SHInspiredAngularEncoding (Cartesian angular polynomial).
    #   use_radial_separation   : RadialSeparationEncoding [r, ux, uy, uz].
    #   use_radial_decay_encoding: RadialDecayEncoding (scaled inverse-radius; experimental).
    #   use_real_sh_basis       : RealSHBasisEncoding (real spherical harmonics; experimental).
    use_sh_encoding: bool = False
    sh_encoding_degree: int = 4          # max polynomial degree (1..8)
    sh_append_raw: bool = True           # always True (required by SHInspiredAngularEncoding)
    use_radial_separation: bool = False
    radial_append_raw: bool = False      # True → 7-dim output, False → 4-dim

    # Radial decay-aware encoding (experimental, off by default). Provides scaled
    # inverse-radius decay features inspired by the R/r radial decay of
    # spherical-harmonic terms. This is not exactly R_ref / r_phys; evaluate via
    # ablation before using it as a research claim. See RadialDecayEncoding.
    use_radial_decay_encoding: bool = False
    radial_decay_max_power: int = 4
    radial_decay_append_raw: bool = True

    # Physical radial-decay encoding: computes true rho = R_ref / r_phys using
    # scaler metadata. This is separate from the older scaled-coordinate
    # RadialDecayEncoding and is the physically informed recommended option.
    use_physical_radial_decay_encoding: bool = False
    physical_radial_decay_max_power: int = 4
    physical_radial_decay_append_raw: bool = True
    physical_radial_decay_include_unit: bool = True
    physical_radial_decay_include_r_scaled: bool = True
    x_scale_m: float | None = None
    resolved_r_ref_m: float | None = None

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
    # Default 3 is the strong benchmark-candidate multi-scale SIREN.
    n_bands: int = 3
    # Multi-scale composition: "concat_shared" (parallel bands -> concat -> shared
    # trunk, default) or "additive" (per-band trunks summed: dU = sum_k dU_k).
    multiscale_mode: str = "concat_shared"

    # Harmonic degree range of the dataset. Resolved from HDF5 metadata by the
    # engine BEFORE the model is built, then persisted to config.json and the
    # checkpoint so evaluation reconstructs the identical multi-scale spectrum.
    # Leaving these None at build time for n_bands>1 is a hard error (no silent
    # fallback to 0/50, which silently corrupted reloaded MultiScale SIRENs).
    degree_min: int | None = None
    degree_max: int | None = None
    # Resolved per-band SIREN frequencies (filled in by the engine for n_bands>1).
    w0_bands: list | None = None

    # Target scaler robustness. "max" lets a single outlier shrink every
    # normalized residual target; "hybrid" caps the scale at
    # target_scale_multiplier * RMS, which is far more robust. x scaling is
    # always origin-fixed max-radius and is NOT affected by these.
    u_scale_mode: str = "hybrid"   # "max" | "rms" | "hybrid"
    a_scale_mode: str = "hybrid"
    target_scale_multiplier: float = 6.0

    # Dataset convention safety. Old or incomplete dataset contracts are always
    # rejected; regenerate clouds with the current generator before training.
    allow_dataset_validation_fail: bool = False

    # Pre-training preflight gate (single go/no-go before training starts).
    skip_preflight: bool = False
    allow_preflight_fail: bool = False

    # Determinism / cuDNN. Defaults preserve prior behavior.
    deterministic: bool = True
    benchmark_cudnn: bool = False

    # Gradient accumulation — accumulate gradients over N batches before stepping.
    # Effective batch size = batch_size * grad_accumulation_steps.
    grad_accumulation_steps: int = 1

    # Resume / continuation. When resume_from is set, the engine restores model,
    # optimizer, GradNorm and RNG state from a previous run and continues from the
    # last completed epoch. --epochs is the TOTAL target epoch count (not extra
    # epochs). Defaults preserve existing behavior when resume is not used.
    resume_from: str | None = None
    resume_checkpoint: str = "last"          # "last" (continue training) | "best" (fine-tune)
    resume_strict: bool = True               # fail on architecture/dataset/scaler-critical mismatch
    resume_allow_longer_epochs: bool = True  # allow extending the epoch target on resume
    resume_append_history: bool = True       # preserve/append previous history by default

    # Periodic Evaluation During Training (monitoring only; OFF by default).
    # At selected epochs, AFTER the epoch's validation and ckpt_last save, the
    # evaluation CLI is run as a subprocess on the current checkpoint to produce
    # field-level diagnostics (parity, acceleration/potential/angular metrics).
    # This NEVER feeds back into the optimizer, scheduler, GradNorm, checkpoint
    # selection, gradients, RNG, or model weights. periodic_eval_count and
    # periodic_eval_every_epochs are mutually exclusive; both None = disabled.
    periodic_eval_count: int | None = None          # run N evals across the full horizon
    periodic_eval_every_epochs: int | None = None   # alternative: run every K epochs
    periodic_eval_dataset: str = "val"                 # "val" | "test" | "ood"
    periodic_eval_max_samples: int = 200_000           # keep monitoring eval lightweight
    periodic_eval_batch_size: int | None = None     # None = reuse training batch_size
    # CPU avoids competing with the live training process for a single CUDA
    # device. Users can explicitly request cuda/mps when a separate device is
    # available.
    periodic_eval_device: str = "cpu"                 # "auto" | "cpu" | "cuda" | "mps"
    periodic_eval_prefer_checkpoint: str = "last"      # "last" (default) | "best"
    periodic_eval_timeout_sec: int | None = None    # per-eval subprocess timeout
    periodic_eval_continue_on_fail: bool = True        # failure must not abort training

    # Optional exponential-moving-average weights. Raw weights remain the
    # checkpoint-selection source; evaluation must opt into EMA explicitly.
    ema_decay: float | None = None


MODEL_PRESETS = (
    "baseline_raw",
    "recommended_physical_radial_decay",
    "ablation_radial_separation",
    "ablation_radial_decay_scaled",
    "ablation_real_sh_low_degree",
    "custom",
)


_ENCODING_FLAGS = (
    "use_fourier",
    "use_sh_encoding",
    "use_radial_separation",
    "use_radial_decay_encoding",
    "use_physical_radial_decay_encoding",
    "use_real_sh_basis",
)


def apply_model_preset(cfg: TrainConfig) -> TrainConfig:
    """Apply named architecture presets in-place and return ``cfg``.

    Non-custom presets define the input encoding. If a caller also sets manual
    encoding flags that conflict with the preset, fail loudly so ablations are
    not mislabeled.
    """

    preset = str(getattr(cfg, "model_preset", "custom") or "custom").strip().lower()
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown model_preset={preset!r}; expected one of {MODEL_PRESETS}.")
    if preset == "custom":
        return cfg

    implied: dict[str, bool] = {name: False for name in _ENCODING_FLAGS}
    if preset == "recommended_physical_radial_decay":
        implied["use_physical_radial_decay_encoding"] = True
        cfg.physical_radial_decay_max_power = 4
        cfg.physical_radial_decay_append_raw = True
        cfg.physical_radial_decay_include_unit = True
        cfg.physical_radial_decay_include_r_scaled = True
    elif preset == "ablation_radial_separation":
        implied["use_radial_separation"] = True
    elif preset == "ablation_radial_decay_scaled":
        implied["use_radial_decay_encoding"] = True
    elif preset == "ablation_real_sh_low_degree":
        implied["use_real_sh_basis"] = True
        cfg.real_sh_degree = min(int(getattr(cfg, "real_sh_degree", 4)), 4)

    active = {name for name in _ENCODING_FLAGS if bool(getattr(cfg, name, False))}
    implied_active = {name for name, value in implied.items() if value}
    if active and active != implied_active:
        if not bool(getattr(cfg, "_model_preset_explicit", False)):
            cfg.model_preset = "custom"
            return cfg
        raise ValueError(
            f"model_preset={preset!r} conflicts with manual encoding flags {sorted(active)}. "
            "Use --model-preset custom for manual encoding ablations."
        )
    for name, value in implied.items():
        setattr(cfg, name, bool(value))
    return cfg


# Run-level presets. These are ORTHOGONAL to MODEL_PRESETS: model_preset selects
# the input-encoding architecture, run_preset selects the reproducibility /
# evaluation posture of the run. "custom" applies nothing.
RUN_PRESETS = (
    "custom",
    "development",
    "quick",
    "paper",
)

# Interpolation splits sample train/val/test from the same spatial+altitude
# distribution; generalization splits hold out a region (spatial block) or an
# altitude band (OOD). A "paper" run must use a generalization split so the
# reported error is not an in-distribution interpolation number.
_INTERPOLATION_SPLIT_POLICIES = frozenset(
    {"seeded_random", "random", "altitude_stratified"}
)
_GENERALIZATION_SPLIT_POLICIES = frozenset(
    {
        "spatial_block",
        "ood_low_altitude",
        "ood_high_altitude",
        "spatial_plus_altitude_stratified",
    }
)

# CLI flags that, when present, mark a run-preset-governed field as explicitly
# set by the user. Used to decide whether the "paper" preset should defer to the
# user (and only raise on a genuine conflict) instead of silently overriding.
RUN_PRESET_EXPLICIT_FLAGS: dict[str, tuple[str, ...]] = {
    "split_policy": ("--split-policy",),
    "deterministic": ("--deterministic", "--no-deterministic"),
    "benchmark_cudnn": ("--benchmark-cudnn",),
    "amp": ("--amp", "--no-amp"),
    "quick_check": ("--quick-check",),
    "skip_preflight": ("--skip-preflight",),
    "allow_preflight_fail": ("--allow-preflight-fail",),
    "allow_dataset_validation_fail": ("--allow-dataset-validation-fail",),
    "use_si": ("--use-si", "--no-si"),
}

# Boolean posture for the "paper" preset: (field -> required value). These are
# authoritative — an explicit conflicting CLI flag is a hard error so a "paper"
# run can never silently disable a reproducibility guard.
_PAPER_BOOL_POSTURE: dict[str, bool] = {
    "deterministic": True,
    "benchmark_cudnn": False,
    "amp": False,
    "quick_check": False,
    "skip_preflight": False,
    "allow_preflight_fail": False,
    "allow_dataset_validation_fail": False,
    "use_si": True,
}


def detect_explicit_run_preset_fields(argv_tokens: list[str]) -> set[str]:
    """Return the set of run-preset-governed field names explicitly set on argv."""

    explicit: set[str] = set()
    for field, flags in RUN_PRESET_EXPLICIT_FLAGS.items():
        for token in argv_tokens:
            if token.split("=", 1)[0] in flags:
                explicit.add(field)
                break
    return explicit


def _enforce_paper_bool(
    cfg: TrainConfig, explicit: set[str], field: str, target: bool
) -> None:
    current = bool(getattr(cfg, field, target))
    if field in explicit and current != target:
        raise ValueError(
            f"run_preset='paper' requires {field}={target} but it was explicitly "
            f"set to {current}. Remove the conflicting flag or use "
            f"--run-preset custom for a non-paper run."
        )
    setattr(cfg, field, target)


def apply_run_preset(
    cfg: TrainConfig, explicit_fields: set[str] | None = None
) -> TrainConfig:
    """Apply a named run-level preset in-place and return ``cfg``.

    ``explicit_fields`` lists the run-preset-governed fields the user set
    explicitly (from the CLI). For the authoritative "paper" preset an explicit
    conflicting value is a hard error; soft presets ("development"/"quick") defer
    to an explicit user value. When called programmatically with
    ``explicit_fields=None`` nothing is treated as explicit, so "paper" enforces
    its full posture (this makes engine re-application idempotent and safe).
    """

    preset = str(getattr(cfg, "run_preset", "custom") or "custom").strip().lower()
    if preset not in RUN_PRESETS:
        raise ValueError(
            f"Unknown run_preset={preset!r}; expected one of {RUN_PRESETS}."
        )
    cfg.run_preset = preset
    if preset == "custom":
        return cfg

    explicit = set(explicit_fields or ())

    if preset == "development":
        # Fast interpolation default: only soft-default the split policy.
        if "split_policy" not in explicit:
            cfg.split_policy = "seeded_random"
        if "deterministic" not in explicit:
            cfg.deterministic = False
        if "benchmark_cudnn" not in explicit:
            cfg.benchmark_cudnn = True
        return cfg

    if preset == "quick":
        # Pipeline smoke test: run the quick-check path and never let preflight
        # resource/headroom checks block a deliberate smoke run.
        if "quick_check" not in explicit:
            cfg.quick_check = True
        if "allow_preflight_fail" not in explicit:
            cfg.allow_preflight_fail = True
        return cfg

    # preset == "paper": authoritative, reproducibility-defensible posture.
    sp = str(getattr(cfg, "split_policy", "") or "").strip().lower()
    if "split_policy" in explicit:
        if sp in _INTERPOLATION_SPLIT_POLICIES:
            raise ValueError(
                f"run_preset='paper' requires a generalization split "
                f"(one of {sorted(_GENERALIZATION_SPLIT_POLICIES)}), but "
                f"--split-policy {sp} is an interpolation split. A paper result "
                f"must not be reported on an in-distribution interpolation split."
            )
        # An explicit generalization split is respected.
    elif sp not in _GENERALIZATION_SPLIT_POLICIES:
        cfg.split_policy = "spatial_block"
    for field, target in _PAPER_BOOL_POSTURE.items():
        _enforce_paper_bool(cfg, explicit, field, target)
    return cfg


_TC_DEFAULTS: dict = {
    f.name: f.default
    for f in _dataclasses.fields(TrainConfig)
    if f.default is not _dataclasses.MISSING
}


def _default_outdir(base: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / "outputs" / "training" / f"st_lrps_train_{ts}"


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
    group_data.add_argument(
        "--split-policy",
        choices=[
            "seeded_random",
            "random",
            "altitude_stratified",
            "spatial_block",
            "ood_low_altitude",
            "ood_high_altitude",
            "spatial_plus_altitude_stratified",
        ],
        default=_TC_DEFAULTS.get("split_policy", "seeded_random"),
        help=(
            "Dataset split policy. 'seeded_random'/'altitude_stratified' are "
            "interpolation splits; 'spatial_block' holds out Moon-fixed lon/lat "
            "blocks (spatial generalization); 'ood_low_altitude'/'ood_high_altitude' "
            "hold out an altitude band (extrapolation). Recorded in split_manifest.json."
        ),
    )
    group_data.add_argument("--test-fraction", type=float, default=_TC_DEFAULTS.get("test_fraction", 0.0),
                            help="Fraction reserved for an in-distribution test split.")
    group_data.add_argument("--spatial-lon-bins", type=int, default=_TC_DEFAULTS.get("spatial_lon_bins", 12),
                            help="spatial_block: number of longitude bins for the holdout grid.")
    group_data.add_argument("--spatial-lat-bins", type=int, default=_TC_DEFAULTS.get("spatial_lat_bins", 6),
                            help="spatial_block: number of latitude bins for the holdout grid.")
    group_data.add_argument("--spatial-val-block-fraction", type=float, default=None,
                            help="spatial_block: fraction of blocks held out for validation (default: val fraction).")
    group_data.add_argument("--spatial-test-block-fraction", type=float, default=None,
                            help="spatial_block: fraction of blocks held out for test (default: test fraction).")
    group_data.add_argument("--spatial-altitude-bins", type=int, default=_TC_DEFAULTS.get("spatial_altitude_bins", 4),
                            help="spatial_plus_altitude_stratified: altitude strata for balanced spatial holdout.")
    group_data.add_argument("--ood-low-altitude-max-km", type=float, default=None,
                            help="ood_low_altitude: hold out altitudes <= this value (km). Overrides --ood-holdout-fraction.")
    group_data.add_argument("--ood-high-altitude-min-km", type=float, default=None,
                            help="ood_high_altitude: hold out altitudes >= this value (km). Overrides --ood-holdout-fraction.")
    group_data.add_argument("--ood-holdout-fraction", type=float, default=_TC_DEFAULTS.get("ood_holdout_fraction", 0.2),
                            help="ood_*_altitude: fraction of the altitude range held out when no explicit threshold is given.")

    group_data.add_argument(
        "--run-preset",
        choices=RUN_PRESETS,
        default=_TC_DEFAULTS.get("run_preset", "custom"),
        help=(
            "Run-level reproducibility/evaluation posture (orthogonal to "
            "--model-preset). 'custom' (default) changes nothing. 'development' "
            "uses the fast seeded_random interpolation split. 'quick' runs the "
            "pipeline smoke check. 'paper' enforces a scientifically defensible "
            "run: a generalization split (spatial_block by default), deterministic "
            "execution, an enforced preflight gate, and no AMP/quick/legacy "
            "shortcuts (a conflicting explicit flag is a hard error)."
        ),
    )

    # Architecture
    group_arch = ap.add_argument_group("Model Architecture")
    group_arch.add_argument("--hidden", type=int, default=_TC_DEFAULTS["hidden"], help="Neurons per hidden layer.")
    group_arch.add_argument("--depth", type=int, default=_TC_DEFAULTS["depth"],
                            help="Number of hidden layers.")
    group_arch.add_argument("--activation", type=str, default=_TC_DEFAULTS["activation"],
                            choices=["sine", "silu", "tanh", "softplus"],
                            help="Activation function. 'sine' = SIREN.")
    group_arch.add_argument(
        "--model-preset",
        choices=MODEL_PRESETS,
        default=_TC_DEFAULTS.get("model_preset", "recommended_physical_radial_decay"),
        help=(
            "Named architecture preset. baseline_raw keeps raw xyz as the control; "
            "recommended_physical_radial_decay enables true R_ref/r radial features; "
            "custom respects manual encoding flags."
        ),
    )
    group_arch.add_argument(
        "--runtime-model-kind",
        choices=["potential_autograd"],
        default=_TC_DEFAULTS.get("runtime_model_kind", "potential_autograd"),
        help=(
            "Runtime model contract. Only potential_autograd is supported; the "
            "force_direct variant is archived in experimental/force-direct-archive."
        ),
    )
    group_arch.add_argument(
        "--output-dim",
        type=int,
        default=_TC_DEFAULTS.get("output_dim", 1),
        help="Model output dimension. potential_autograd uses 1.",
    )
    group_arch.add_argument("--w0-first", type=float, default=None,
                            help="SIREN w0 for first layer (default: auto from dataset degree_max; fallback 30.0).")
    group_arch.add_argument("--w0-hidden", type=float, default=None,
                            help="SIREN w0 for hidden layers (default: auto from dataset degree_max; fallback 30.0).")
    group_arch.add_argument("--dropout", type=float, default=_TC_DEFAULTS["dropout"])
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
    group_arch.add_argument("--fourier-n", type=int, default=_TC_DEFAULTS["fourier_n_features"],
                            help="Number of Fourier features (embedding dim = 2*n).")
    group_arch.add_argument("--fourier-sigma", type=float, default=_TC_DEFAULTS["fourier_sigma"],
                            help="Std of frequency matrix B; larger = finer spatial detail.")
    group_arch.add_argument("--fourier-seed", type=int, default=_TC_DEFAULTS["fourier_seed"],
                            help="Seed used to construct the fixed Fourier feature matrix.")

    # Optimization
    group_opt = ap.add_argument_group("Optimization")
    group_opt.add_argument("--epochs", type=int, default=_TC_DEFAULTS["epochs"])
    group_opt.add_argument("--batch-size", type=int, default=_TC_DEFAULTS["batch_size"])
    group_opt.add_argument("--lr", type=float, default=_TC_DEFAULTS["lr"])
    group_opt.add_argument("--weight-decay", type=float, default=_TC_DEFAULTS["weight_decay"])
    group_opt.add_argument("--output-head-lr-mult", type=float, default=_TC_DEFAULTS["output_head_lr_mult"],
                           help="Learning-rate multiplier applied only to the final scalar output head.")
    group_opt.add_argument("--grad-clip", "--max-grad-norm", dest="grad_clip", type=float, default=_TC_DEFAULTS["max_grad_norm"],
                           help="Global gradient clipping threshold.")
    group_opt.add_argument("--t-max", type=int, default=_TC_DEFAULTS["t_max"],
                           help="Cosine scheduler T_max.")
    group_opt.add_argument("--warmup-epochs", type=int, default=_TC_DEFAULTS["warmup_epochs"],
                           help="Linear learning-rate warm-up duration before cosine decay.")
    group_opt.add_argument("--min-lr-ratio", type=float, default=_TC_DEFAULTS["min_lr_ratio"],
                           help="Final cosine-decay learning-rate ratio relative to the base LR.")
    group_opt.add_argument("--patience", type=int, default=_TC_DEFAULTS["patience"],
                           help="Early-stopping patience measured on validation total loss.")
    amp_group = group_opt.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", action="store_true", dest="amp",
                           help="Enable CUDA AMP when the derivative path supports it safely.")
    amp_group.add_argument("--no-amp", action="store_false", dest="amp",
                           help="Disable Automatic Mixed Precision.")

    # Physics & Sobolev Weights
    group_phys = ap.add_argument_group("Physics & Loss Weights")
    group_phys.add_argument("--w-u", type=float, default=_TC_DEFAULTS["w_u"], help="Initial weight for Potential (ΔU) loss.")
    group_phys.add_argument("--w-a", type=float, default=_TC_DEFAULTS["w_a"], help="Initial weight for Acceleration (Δa) loss.")
    group_phys.add_argument("--gradnorm-mode", choices=["fixed", "ntk_init", "dynamic"], default=_TC_DEFAULTS["gradnorm_mode"],
                            help="Loss-weighting policy for the Sobolev objective: 'ntk_init' "
                                 "(default; compute w_a once then freeze), 'fixed' (use w_u/w_a "
                                 "as set), or 'dynamic' (EMA GradNorm; ablation only).")
    group_phys.add_argument("--gradnorm-w-a-min", type=float, default=_TC_DEFAULTS["gradnorm_w_a_min"],
                            help="Lower clamp for NTK/dynamic acceleration-loss weight.")
    group_phys.add_argument("--gradnorm-w-a-max", type=float, default=_TC_DEFAULTS["gradnorm_w_a_max"],
                            help="Upper clamp for NTK/dynamic acceleration-loss weight.")
    group_phys.add_argument("--potential-only-epochs", type=int, default=_TC_DEFAULTS["potential_only_epochs"],
                            help="Initial epochs that optimise only the residual potential ΔU.")
    group_phys.add_argument("--accel-ramp-epochs", type=int, default=_TC_DEFAULTS["accel_ramp_epochs"],
                            help="Epochs used to linearly ramp the acceleration loss from accel_min_factor to full weight.")
    group_phys.add_argument("--accel-min-factor", type=float, default=_TC_DEFAULTS["accel_min_factor"],
                            help="Minimum acceleration loss factor during curriculum warm-up (floor). "
                                 "0.0 = pure potential-only; positive values keep a derivative floor.")
    group_phys.add_argument("--a-sign", default=_TC_DEFAULTS["a_sign"], help="Sign of -grad(U). 'auto' or +1/-1.")
    group_phys.add_argument("--use-si", action="store_true", dest="use_si", help="Convert canonical units to SI.")
    group_phys.add_argument("--no-si", action="store_false", dest="use_si", help="Keep dataset units as-is.")
    ap.set_defaults(use_si=_TC_DEFAULTS["use_si"], pin_memory=_TC_DEFAULTS["pin_memory"])
    ap.set_defaults(use_fourier=_TC_DEFAULTS["use_fourier"], fourier_append_raw=_TC_DEFAULTS["fourier_append_raw"], amp=_TC_DEFAULTS["amp"])
    ap.set_defaults(use_residual_blocks=_TC_DEFAULTS["use_residual_blocks"])
    ap.set_defaults(
        use_altitude_balanced_loss=_TC_DEFAULTS["use_altitude_balanced_loss"],
        use_radial_cross_loss=_TC_DEFAULTS["use_radial_cross_loss"],
    )

    # Hardware & Performance
    group_perf = ap.add_argument_group("Performance & Scaler")
    group_perf.add_argument("--num-workers", type=int, default=_TC_DEFAULTS["num_workers"])
    group_perf.add_argument("--cache-rows", type=int, default=_TC_DEFAULTS["cache_rows"], help="H5BlockDataset cache size.")
    group_perf.add_argument("--sampler-block-size", type=int, default=_TC_DEFAULTS["sampler_block_size"],
                            help="Local block size used by the streaming shuffle sampler.")
    group_perf.add_argument("--fit-rows", type=int, default=_TC_DEFAULTS["fit_rows"], help="Rows for isometric scaler fitting.")
    group_perf.add_argument("--fit-seed", type=int, default=_TC_DEFAULTS["fit_seed"],
                            help="Seed used when sampling rows for scaler fitting.")
    group_perf.add_argument("--fit-chunk-rows", type=int, default=_TC_DEFAULTS["fit_chunk_rows"],
                            help="HDF5 row chunk size used during scaler fitting.")
    group_perf.add_argument("--seed", type=int, default=_TC_DEFAULTS["seed"])
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
                            help="Auto-preload when dataset size is at most this many MB.")
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
    group_dir.add_argument("--best-ckpt-start-epoch", type=int, default=_TC_DEFAULTS["best_ckpt_start_epoch"],
                           help="Epoch from which best-checkpoint tracking and patience counting begin. "
                                "-1 = auto (delays to direction_loss_start_epoch + "
                                "direction_loss_ramp_epochs + checkpoint_settle_epochs when direction loss is active).")
    group_dir.add_argument("--checkpoint-settle-epochs", type=int, default=_TC_DEFAULTS["checkpoint_settle_epochs"],
                           help="Additional settled epochs after the direction-loss ramp before auto best-checkpoint tracking starts.")
    group_dir.add_argument("--best-metric",
                           choices=["val_total_loss", "val_base_loss", "total_loss", "direction_loss", "hybrid"],
                           default=_TC_DEFAULTS['best_metric'],
                           help="Metric used for best-checkpoint selection. "
                                "'hybrid': val_base_loss + alpha * val_loss_dir (default). "
                                "'val_total_loss': validation reference loss only ('total_loss' alias accepted). "
                                "'val_base_loss': validation U + acceleration MSE only. "
                                "'direction_loss': validation direction loss only (experimental).")
    group_dir.add_argument("--hybrid-direction-alpha", type=float, default=_TC_DEFAULTS['hybrid_direction_alpha'],
                           help="Weight alpha for direction loss in hybrid best-metric: "
                                "score = val_base_loss + alpha * val_loss_dir.")
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
    group_alt.add_argument("--altitude-bin-width-km", type=float, default=_TC_DEFAULTS["altitude_bin_width_km"], help="Bin width in km.")
    group_alt.add_argument("--altitude-min-km", type=float, default=_TC_DEFAULTS["altitude_min_km"], help="Min altitude in km.")
    group_alt.add_argument("--altitude-max-km", type=float, default=_TC_DEFAULTS["altitude_max_km"], help="Max altitude in km.")

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
                           help="Weight for cross-radial loss (default: 0.05).")

    # Sparse Laplacian Regularization
    group_lap = ap.add_argument_group("Sparse Laplacian Regularization")
    lap_reg_group = group_lap.add_mutually_exclusive_group()
    lap_reg_group.add_argument("--use-laplacian-regularization", action="store_true", dest="use_laplacian_regularization",
                           help="Apply sparse Laplacian regularization (∇²U=0 physics constraint).")
    lap_reg_group.add_argument("--no-laplacian-regularization", action="store_false", dest="use_laplacian_regularization",
                               help="Disable sparse Laplacian regularization.")
    ap.set_defaults(use_laplacian_regularization=_TC_DEFAULTS["use_laplacian_regularization"])
    group_lap.add_argument("--laplacian-weight", type=float, default=_TC_DEFAULTS["laplacian_weight"], help="Weight for Laplacian loss.")
    group_lap.add_argument("--laplacian-every-n-batches", type=int, default=_TC_DEFAULTS["laplacian_every_n_batches"], help="Compute every N batches.")
    group_lap.add_argument("--laplacian-subset-size", type=int, default=_TC_DEFAULTS["laplacian_subset_size"],
                           help="Batch subset size for Hutchinson Laplacian estimator.")
    group_lap.add_argument("--n-hutchinson-samples", type=int, default=_TC_DEFAULTS["n_hutchinson_samples"],
                           help="Rademacher samples per Hutchinson trace estimate (K=4 → ~50%% relative error).")
    group_lap.add_argument("--laplacian-mode",
        choices=["off", "diagnostic", "train"], default=_TC_DEFAULTS.get('laplacian_mode', 'diagnostic'),
        help="Laplacian regularization mode. "
             "diagnostic = logs the physics (Laplace) violation only, no gradient is backpropagated; "
             "train = backpropagates the Laplacian penalty into model weights (create_graph=True); "
             "off = skip entirely. Default: diagnostic.")
    group_lap.add_argument("--collocation-laplacian-every", type=int,
        default=_TC_DEFAULTS["collocation_laplacian_every"],
        help="Optimizer steps between collocation Laplacian evaluations.")
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
        "--sh-encoding-degree", type=int, choices=range(0, 17),
        default=_TC_DEFAULTS.get("sh_encoding_degree", 4),
        help="Max polynomial degree for SH-inspired angular encoding (0..16).",
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
        help="Use RadialDecayEncoding: scaled inverse-radius decay features inspired "
             "by the R/r radial decay of spherical-harmonic terms. Experimental; "
             "off by default; evaluate through ablation; mutually exclusive with "
             "the other encodings.",
    )
    dec_group.add_argument(
        "--no-radial-decay-encoding", action="store_false", dest="use_radial_decay_encoding",
        help="Disable radial decay encoding (default).",
    )
    group_enc.add_argument(
        "--radial-decay-max-power", type=int, default=_TC_DEFAULTS.get("radial_decay_max_power", 4),
        help="Highest scaled inverse-radius power for RadialDecayEncoding (default: 4).",
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

    # Physical radial decay encoding (true R_ref / r_phys).
    pdec_group = group_enc.add_mutually_exclusive_group()
    pdec_group.add_argument(
        "--use-physical-radial-decay-encoding",
        action="store_true",
        dest="use_physical_radial_decay_encoding",
        help="Use PhysicalRadialDecayEncoding with true rho=R_ref/r_phys. "
             "Mutually exclusive with the other encodings.",
    )
    pdec_group.add_argument(
        "--no-physical-radial-decay-encoding",
        action="store_false",
        dest="use_physical_radial_decay_encoding",
        help="Disable physical radial decay encoding.",
    )
    group_enc.add_argument(
        "--physical-radial-decay-max-power",
        type=int,
        default=_TC_DEFAULTS.get("physical_radial_decay_max_power", 4),
        help="Highest power of true rho=R_ref/r_phys for PhysicalRadialDecayEncoding.",
    )
    pdec_raw_group = group_enc.add_mutually_exclusive_group()
    pdec_raw_group.add_argument(
        "--physical-radial-decay-append-raw",
        action="store_true",
        dest="physical_radial_decay_append_raw",
        help="Append raw scaled xyz to physical radial decay features (default).",
    )
    pdec_raw_group.add_argument(
        "--no-physical-radial-decay-append-raw",
        action="store_false",
        dest="physical_radial_decay_append_raw",
        help="Do not append raw scaled xyz to physical radial decay features.",
    )
    pdec_unit_group = group_enc.add_mutually_exclusive_group()
    pdec_unit_group.add_argument(
        "--physical-radial-decay-include-unit",
        action="store_true",
        dest="physical_radial_decay_include_unit",
        help="Include unit direction vector in physical radial decay features (default).",
    )
    pdec_unit_group.add_argument(
        "--no-physical-radial-decay-include-unit",
        action="store_false",
        dest="physical_radial_decay_include_unit",
        help="Omit unit direction vector from physical radial decay features.",
    )
    pdec_r_group = group_enc.add_mutually_exclusive_group()
    pdec_r_group.add_argument(
        "--physical-radial-decay-include-r-scaled",
        action="store_true",
        dest="physical_radial_decay_include_r_scaled",
        help="Include scaled radius ||x_scaled|| in physical radial decay features (default).",
    )
    pdec_r_group.add_argument(
        "--no-physical-radial-decay-include-r-scaled",
        action="store_false",
        dest="physical_radial_decay_include_r_scaled",
        help="Omit scaled radius from physical radial decay features.",
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
        "--real-sh-degree", type=int, choices=range(0, 9),
        default=_TC_DEFAULTS.get("real_sh_degree", 4),
        help="Max degree L for RealSHBasisEncoding ((L+1)^2 angular terms, 0..8).",
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
        use_physical_radial_decay_encoding=False,
        physical_radial_decay_append_raw=_TC_DEFAULTS.get("physical_radial_decay_append_raw", True),
        physical_radial_decay_include_unit=_TC_DEFAULTS.get("physical_radial_decay_include_unit", True),
        physical_radial_decay_include_r_scaled=_TC_DEFAULTS.get("physical_radial_decay_include_r_scaled", True),
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
                                 "degree_min/degree_max. Values >1 require degree_max "
                                 "metadata. Use 1 for a standard single-scale SirenMLP.)")
    group_pinn.add_argument("--multiscale-mode", choices=["concat_shared", "additive"],
                            default=_TC_DEFAULTS.get("multiscale_mode", "concat_shared"),
                            help="Multi-scale composition when n_bands>1: 'concat_shared' "
                                 "(parallel bands -> concat -> shared trunk, default) or "
                                 "'additive' (per-band trunks summed; experimental).")
    group_pinn.add_argument("--grad-accumulation-steps", type=int, default=_TC_DEFAULTS["grad_accumulation_steps"],
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
    group_safety.add_argument("--allow-dataset-validation-fail", action="store_true",
                              default=False,
                              help="Record but do not abort when lightweight dataset validation fails.")
    group_safety.add_argument("--skip-preflight", action="store_true", default=False,
                              help="Skip the single pre-training preflight go/no-go gate entirely.")
    group_safety.add_argument("--allow-preflight-fail", action="store_true", default=False,
                              help="Run the preflight gate and write its report, but do not abort on failure.")
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
    group_log.add_argument("--log-every", type=int, default=_TC_DEFAULTS["log_every"],
                           help="Print batch-level progress every N batches (0 to disable). "
                                "Used when --log-every-mode is 'fixed'.")
    group_log.add_argument("--log-every-mode", choices=["fixed", "auto"], default=_TC_DEFAULTS["log_every_mode"],
                           help="'fixed' uses --log-every literally; 'auto' logs roughly 10 "
                                "progress updates per epoch (always including the first and "
                                "last batch).")
    group_log.add_argument("--quick-check", action="store_true", default=False,
                           help="Run 1 epoch with 5 train + 2 val batches to verify the full pipeline.")
    group_log.add_argument("--max-train-batches", type=int, default=None,
                           help="Cap the number of training batches per epoch (None = full epoch).")
    group_log.add_argument("--max-val-batches", type=int, default=None,
                           help="Cap the number of validation batches per epoch (None = full epoch).")
    group_log.add_argument("--overfit-batches", type=int, default=_TC_DEFAULTS["overfit_batches"],
                           help="Repeat the first N training batches every epoch for pipeline sanity checks.")

    # Resume / Continuation
    group_resume = ap.add_argument_group("Resume / Continuation")
    group_resume.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume training from a previous ST-LRPS run directory, its checkpoints/ "
             "directory, or a specific .pt checkpoint. A run directory loads "
             "checkpoints/ckpt_last.pt by default. --data/--out are inferred from the "
             "previous run when omitted. --epochs is the TOTAL target epoch count.",
    )
    group_resume.add_argument(
        "--resume-checkpoint", choices=["last", "best"], default="last",
        help="Which checkpoint to prefer when --resume-from points to a run directory. "
             "Default: last (continues training/optimizer state). Use 'best' to "
             "fine-tune from the best-selected checkpoint.",
    )
    group_resume.add_argument(
        "--resume-nonstrict", action="store_true", default=False,
        help="Allow limited non-critical config differences when resuming. "
             "Architecture/dataset/scaler-critical mismatches still fail.",
    )
    resume_hist_group = group_resume.add_mutually_exclusive_group()
    resume_hist_group.add_argument(
        "--resume-append-history", action="store_true", dest="resume_append_history",
        help="Append to / preserve the existing history when resuming (default).",
    )
    resume_hist_group.add_argument(
        "--resume-overwrite-history", action="store_false", dest="resume_append_history",
        help="Overwrite the existing history when resuming.",
    )
    ap.set_defaults(resume_append_history=True)

    # Periodic Evaluation During Training (monitoring only; OFF by default)
    group_peval = ap.add_argument_group("Periodic Evaluation During Training (monitoring only)")
    peval_mode_group = group_peval.add_mutually_exclusive_group()
    peval_mode_group.add_argument(
        "--periodic-eval-count", type=int, default=None,
        help="Run periodic evaluation N times spread across the full --epochs horizon "
             "(e.g. --epochs 400 --periodic-eval-count 10 -> epochs 40,80,...,400). "
             "Mutually exclusive with --periodic-eval-every-epochs. Disabled by default.",
    )
    peval_mode_group.add_argument(
        "--periodic-eval-every-epochs", type=int, default=None,
        help="Run periodic evaluation every K epochs (e.g. 25 -> 25,50,75,...). "
             "Mutually exclusive with --periodic-eval-count. Disabled by default.",
    )
    group_peval.add_argument(
        "--periodic-eval-dataset", choices=["val", "test", "ood"],
        default=_TC_DEFAULTS["periodic_eval_dataset"],
        help="Dataset used for periodic evaluation (default: val). val falls back to "
             "--data for single-dataset runs.",
    )
    group_peval.add_argument(
        "--periodic-eval-max-samples", type=int, default=_TC_DEFAULTS["periodic_eval_max_samples"],
        help="Cap rows evaluated per periodic evaluation to keep it lightweight.",
    )
    group_peval.add_argument(
        "--periodic-eval-batch-size", type=int, default=_TC_DEFAULTS["periodic_eval_batch_size"],
        help="Batch size for periodic evaluation (default: reuse the training batch size).",
    )
    group_peval.add_argument(
        "--periodic-eval-device", choices=["auto", "cpu", "cuda", "mps"],
        default=_TC_DEFAULTS["periodic_eval_device"],
        help="Device for the periodic evaluation subprocess (default: cpu to avoid single-GPU contention).",
    )
    group_peval.add_argument(
        "--periodic-eval-prefer-checkpoint", choices=["last", "best"],
        default=_TC_DEFAULTS["periodic_eval_prefer_checkpoint"],
        help="Which checkpoint periodic evaluation should use (default: last — ckpt_best "
             "may not be active during early training).",
    )
    group_peval.add_argument(
        "--periodic-eval-timeout-sec", type=int, default=_TC_DEFAULTS["periodic_eval_timeout_sec"],
        help="Optional per-evaluation subprocess timeout in seconds (default: no timeout).",
    )
    peval_fail_group = group_peval.add_mutually_exclusive_group()
    peval_fail_group.add_argument(
        "--periodic-eval-continue-on-fail", action="store_true", dest="periodic_eval_continue_on_fail",
        help="A failed periodic evaluation does not abort training (default).",
    )
    peval_fail_group.add_argument(
        "--periodic-eval-fail-fast", action="store_false", dest="periodic_eval_continue_on_fail",
        help="Abort training if a periodic evaluation fails.",
    )
    ap.set_defaults(periodic_eval_continue_on_fail=_TC_DEFAULTS["periodic_eval_continue_on_fail"])

    group_ema = ap.add_argument_group("EMA (optional ablation)")
    group_ema.add_argument(
        "--ema-decay", type=float, default=_TC_DEFAULTS["ema_decay"],
        help="Enable exponential-moving-average weights with this decay (e.g. 0.999); raw weights remain the selection source.",
    )

    fused_group = group_opt.add_mutually_exclusive_group()
    fused_group.add_argument(
        "--fused-optimizer", action="store_true", dest="use_fused_optimizer",
        help="Try CUDA fused AdamW when supported; otherwise fall back to regular AdamW.",
    )
    fused_group.add_argument(
        "--no-fused-optimizer", action="store_false", dest="use_fused_optimizer",
        help="Disable the CUDA fused AdamW attempt.",
    )
    ap.set_defaults(use_fused_optimizer=_TC_DEFAULTS["use_fused_optimizer"])

    # ---------------------------------------------------------------------------
    # TrainConfig is the single source of truth for the current default
    # configuration. For this benchmark-reproduction phase, that default is the
    # strong SH25 -> SH200 multi-band SIREN profile.
    #
    # The minimal recommended run is simply:
    #
    #   python -m lunaris.surrogate.st_lrps.training.cli --data path/to/train.h5
    #
    # Notes:
    #   - n_bands>1 (multi-scale SIREN) REQUIRES degree_max in the dataset metadata.
    #   - If direction-loss-floor-abs=3e-6 masks too much of a low-residual region,
    #     lower it deliberately and record the ablation.
    #   - If VRAM is insufficient: --batch-size 4096 --grad-accumulation-steps 4
    #     (an advisory warning is printed at startup when batch_size looks large
    #     for the detected GPU).
    #   - Experimental input encodings (off by default): --use-radial-decay-encoding
    #     (scaled inverse-radius decay features inspired by the R/r radial decay of
    #     spherical-harmonic terms) and --use-real-sh-basis (real spherical harmonic
    #     angular basis). Evaluate both via ablation.
    # ---------------------------------------------------------------------------

    a = ap.parse_args()

    # Backward-compatible CLI behavior: older commands that directly selected
    # an encoding flag should continue to work without also adding
    # ``--model-preset custom``. If the preset was not explicitly supplied,
    # positive manual encoding flags switch the config to custom.
    argv_tokens = list(sys.argv[1:])
    preset_explicit = any(
        tok == "--model-preset" or tok.startswith("--model-preset=")
        for tok in argv_tokens
    )
    manual_encoding_requested = any(
        tok in {
            "--use-fourier",
            "--use-sh-encoding",
            "--use-radial-separation",
            "--use-radial-decay-encoding",
            "--use-physical-radial-decay-encoding",
            "--use-real-sh-basis",
        }
        for tok in argv_tokens
    )
    if manual_encoding_requested and not preset_explicit:
        a.model_preset = "custom"

    # 0. Resume pre-resolution.
    # When --resume-from is given, default --data/--out from the previous run so
    # the user only needs `--resume-from <run> [--epochs N]`. Full checkpoint
    # loading + architecture locking happens later in the training engine.
    resume_from = getattr(a, "resume_from", None)
    if resume_from:
        from lunaris.surrogate.st_lrps.artifacts.manager import resolve_run_dir as _resolve_run_dir
        resume_run_dir = _resolve_run_dir(Path(resume_from).expanduser())
        prev_cfg: dict = {}
        prev_cfg_path = resume_run_dir / "config.json"
        if prev_cfg_path.is_file():
            try:
                prev_cfg = json.loads(prev_cfg_path.read_text(encoding="utf-8"))
            except Exception as _e:  # pragma: no cover - defensive
                print(f"[RESUME] Warning: could not read previous config.json: {_e}")
        # Infer data/out from the previous run when not explicitly provided.
        if a.data is None and a.train_data is None:
            prev_data = prev_cfg.get("data")
            if prev_data:
                a.data = str(prev_data)
            else:
                a.train_data = a.train_data or prev_cfg.get("train_data_path") or prev_cfg.get("train_data")
                a.val_data = a.val_data or prev_cfg.get("val_data_path") or prev_cfg.get("val_data")
        if not a.out:
            a.out = str(resume_run_dir)
        print(f"[RESUME] Resuming run: {resume_run_dir}  (prefer={a.resume_checkpoint})")

    # 1. Resolve Data Path
    # Anchor dataset auto-discovery at the ST-LRPS package root, but place new
    # generated training runs under the repository-level outputs/ convention.
    script_dir = Path(__file__).resolve().parents[1]
    repo_root = project_root_from_file(__file__)
    data_path_raw = a.data or os.environ.get("SPATIAL_CLOUD_INPUT") or os.environ.get("DATASET_PATH")

    if data_path_raw is None and a.train_data is None:
        _, find_latest_dataset = _load_dataset_helpers()
        found = find_latest_dataset(script_dir)
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
    out_dir = Path(a.out) if a.out else _default_outdir(repo_root)
    if not a.out:
        print(f"[AUTO] Using default output directory: {out_dir}")

    # 3. Auto-sync: read dataset metadata and print auto-detected parameters
    w0_first_val = a.w0_first
    w0_hidden_val = a.w0_hidden

    if data_path.suffix.lower() in (".h5", ".hdf5"):
        try:
            DatasetMeta, _ = _load_dataset_helpers()
            meta_early = DatasetMeta.from_h5(data_path)
            degree_max_meta = meta_early.degree_max or meta_early.requested_degree
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
    a_sign_val: float | str = "auto"
    if str(a.a_sign).lower() != "auto":
        try:
            a_sign_val = float(a.a_sign)
        except ValueError:
            print(f"Error: --a-sign must be 'auto', '1.0', or '-1.0'. Got: {a.a_sign}")
            sys.exit(1)

    cfg = TrainConfig(
        data=str(data_path),
        train_data=a.train_data,
        val_data=a.val_data,
        test_data=a.test_data,
        ood_data=a.ood_data,
        suite_manifest=a.suite_manifest,
        out=str(out_dir),
        dataset_name=a.dataset_name,
        run_preset=str(a.run_preset),
        seed=a.seed,
        epochs=a.epochs,
        batch_size=a.batch_size,
        val_ratio=a.val_fraction,
        split_seed=(a.split_seed if a.split_seed is not None else a.seed),
        split_policy=str(a.split_policy),
        test_fraction=float(a.test_fraction),
        spatial_lon_bins=int(a.spatial_lon_bins),
        spatial_lat_bins=int(a.spatial_lat_bins),
        spatial_val_block_fraction=a.spatial_val_block_fraction,
        spatial_test_block_fraction=a.spatial_test_block_fraction,
        spatial_altitude_bins=int(a.spatial_altitude_bins),
        ood_low_altitude_max_km=a.ood_low_altitude_max_km,
        ood_high_altitude_min_km=a.ood_high_altitude_min_km,
        ood_holdout_fraction=float(a.ood_holdout_fraction),
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
        fit_seed=int(a.fit_seed),
        fit_chunk_rows=max(1, int(a.fit_chunk_rows)),
        sampler_block_size=max(1, int(a.sampler_block_size)),
        use_fused_optimizer=bool(a.use_fused_optimizer),
        amp=bool(a.amp),
        model_preset=str(a.model_preset),
        runtime_model_kind=str(a.runtime_model_kind),
        output_dim=int(a.output_dim),
        use_fourier=bool(a.use_fourier),
        fourier_append_raw=bool(a.fourier_append_raw),
        fourier_n_features=int(a.fourier_n),
        fourier_sigma=float(a.fourier_sigma),
        fourier_seed=int(a.fourier_seed),
        log_every=max(0, int(a.log_every)),
        log_every_mode=str(getattr(a, "log_every_mode", "fixed")),
        preload_data=bool(a.preload_data),
        auto_preload_mb=float(a.auto_preload_mb) if not getattr(a, "no_auto_preload", False) else 0.0,
        preload_policy=("never" if getattr(a, "no_auto_preload", False) else str(a.preload_policy)),
        quick_check=bool(a.quick_check),
        max_train_batches=(int(a.max_train_batches) if a.max_train_batches is not None else None),
        max_val_batches=(int(a.max_val_batches) if a.max_val_batches is not None else None),
        overfit_batches=(max(1, int(a.overfit_batches)) if a.overfit_batches is not None and int(a.overfit_batches) > 0 else None),
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
        sh_encoding_degree=int(a.sh_encoding_degree),
        sh_append_raw=bool(a.sh_append_raw),
        use_radial_separation=bool(a.use_radial_separation),
        radial_append_raw=bool(a.radial_append_raw),
        use_radial_decay_encoding=bool(a.use_radial_decay_encoding),
        radial_decay_max_power=max(1, int(a.radial_decay_max_power)),
        radial_decay_append_raw=bool(a.radial_decay_append_raw),
        use_physical_radial_decay_encoding=bool(a.use_physical_radial_decay_encoding),
        physical_radial_decay_max_power=max(1, int(a.physical_radial_decay_max_power)),
        physical_radial_decay_append_raw=bool(a.physical_radial_decay_append_raw),
        physical_radial_decay_include_unit=bool(a.physical_radial_decay_include_unit),
        physical_radial_decay_include_r_scaled=bool(a.physical_radial_decay_include_r_scaled),
        use_real_sh_basis=bool(a.use_real_sh_basis),
        real_sh_degree=int(a.real_sh_degree),
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
        allow_dataset_validation_fail=bool(a.allow_dataset_validation_fail),
        skip_preflight=bool(a.skip_preflight),
        allow_preflight_fail=bool(a.allow_preflight_fail),
        deterministic=bool(a.deterministic),
        benchmark_cudnn=bool(a.benchmark_cudnn),
        laplacian_mode=str(a.laplacian_mode),
        collocation_laplacian_every=max(1, int(a.collocation_laplacian_every)),
        collocation_alt_min_km=(float(a.collocation_alt_min_km) if a.collocation_alt_min_km is not None else None),
        collocation_alt_max_km=(float(a.collocation_alt_max_km) if a.collocation_alt_max_km is not None else None),
        collocation_laplacian_weight=float(a.collocation_laplacian_weight),
        collocation_laplacian_samples=max(1, int(a.collocation_laplacian_samples)),
        collocation_laplacian_hutchinson_samples=max(1, int(a.collocation_laplacian_hutchinson_samples)),
        resume_from=(str(a.resume_from) if getattr(a, "resume_from", None) else None),
        resume_checkpoint=str(getattr(a, "resume_checkpoint", "last")),
        resume_strict=(not bool(getattr(a, "resume_nonstrict", False))),
        resume_allow_longer_epochs=True,
        resume_append_history=bool(getattr(a, "resume_append_history", True)),
        periodic_eval_count=(int(a.periodic_eval_count) if a.periodic_eval_count is not None else None),
        periodic_eval_every_epochs=(
            int(a.periodic_eval_every_epochs) if a.periodic_eval_every_epochs is not None else None
        ),
        periodic_eval_dataset=str(a.periodic_eval_dataset),
        periodic_eval_max_samples=max(1, int(a.periodic_eval_max_samples)),
        periodic_eval_batch_size=(
            int(a.periodic_eval_batch_size) if a.periodic_eval_batch_size is not None else None
        ),
        periodic_eval_device=str(a.periodic_eval_device),
        periodic_eval_prefer_checkpoint=str(a.periodic_eval_prefer_checkpoint),
        periodic_eval_timeout_sec=(
            int(a.periodic_eval_timeout_sec) if a.periodic_eval_timeout_sec is not None else None
        ),
        periodic_eval_continue_on_fail=bool(a.periodic_eval_continue_on_fail),
        ema_decay=(float(a.ema_decay) if a.ema_decay is not None else None),
    )
    # Mutual-exclusivity is enforced by the argparse group, but guard explicitly
    # in case TrainConfig is constructed programmatically.
    if cfg.periodic_eval_count is not None and cfg.periodic_eval_every_epochs is not None:
        print(
            "Error: --periodic-eval-count and --periodic-eval-every-epochs are mutually "
            "exclusive. Set at most one."
        )
        sys.exit(1)
    if cfg.ema_decay is not None and not (0.0 < float(cfg.ema_decay) < 1.0):
        raise ValueError(f"--ema-decay must be in (0, 1), got {cfg.ema_decay!r}")
    cfg._model_preset_explicit = bool(preset_explicit)
    cfg._altitude_bounds_explicit = any(
        tok.split("=", 1)[0] in {"--altitude-min-km", "--altitude-max-km"}
        for tok in argv_tokens
    )
    # Run-level preset is applied BEFORE the architecture preset. It may flip the
    # split policy and reproducibility flags; an explicit conflicting flag under
    # --run-preset paper is a hard error (see apply_run_preset).
    apply_run_preset(cfg, detect_explicit_run_preset_fields(argv_tokens))
    return apply_model_preset(cfg)


# =============================================================================
# DEBUG ENTRY POINT
# =============================================================================
# st_lrps.training.config is a configuration module, NOT a training entry point.
# Launch training via:  python -m lunaris.surrogate.st_lrps.training.cli [--data ...] [--out ...]

if __name__ == "__main__":
    import json as _json
    from dataclasses import asdict as _asdict
    _cfg = parse_args()
    print(_json.dumps(_asdict(_cfg), indent=2, default=str))


__all__ = [
    'TrainConfig',
    'parse_args',
    'MODEL_PRESETS',
    'apply_model_preset',
    'RUN_PRESETS',
    'apply_run_preset',
    'detect_explicit_run_preset_fields',
]
