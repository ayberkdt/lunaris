"""Translate an explicit paper training config into ST-LRPS trainer CLI flags.

The canonical trainer is ``python -m lunaris.surrogate.st_lrps.training.cli``.
Every scientifically relevant option in the config is emitted as an explicit
flag so nothing relies on an implicit default. The legacy/debug safety flags are
deliberately **never emitted**: they are ``store_true`` and default to the safe
(false) value, and the config validator already requires the config to declare
them false.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRAINING_MODULE = "lunaris.surrogate.st_lrps.training.cli"


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    return value if isinstance(value, Mapping) else {}


def _flag(argv: list[str], name: str, value: Any) -> None:
    if value is not None:
        argv.extend([name, str(value)])


def _bool_flag(argv: list[str], value: Any, *, on: str, off: str) -> None:
    if value is None:
        return
    argv.append(on if bool(value) else off)


def build_training_argv(config: Mapping[str, Any]) -> list[str]:
    """Return the trainer flag list (without the ``python -m <module>`` prefix)."""
    dataset = _section(config, "dataset")
    output = _section(config, "output")
    split = _section(config, "split")
    scaler = _section(config, "scaler")
    model = _section(config, "model")
    loss = _section(config, "loss")
    optimizer = _section(config, "optimizer")
    early = _section(config, "early_stopping")
    periodic = _section(config, "periodic_eval")

    argv: list[str] = []

    # --- data ---
    train_data = dataset.get("train_data")
    val_data = dataset.get("val_data")
    if train_data and val_data:
        argv.extend(["--train-data", str(train_data), "--val-data", str(val_data)])
    elif train_data:
        argv.extend(["--data", str(train_data)])
    _flag(argv, "--test-data", dataset.get("test_data"))
    _flag(argv, "--ood-data", dataset.get("ood_data"))
    _flag(argv, "--dataset-name", dataset.get("dataset_name"))
    _flag(argv, "--out", output.get("out_dir"))

    # --- seeds + split ---
    _flag(argv, "--seed", config.get("seed"))
    _flag(argv, "--split-seed", split.get("split_seed", config.get("seed")))
    _flag(argv, "--split-policy", split.get("split_policy"))
    _flag(argv, "--val-fraction", split.get("val_fraction"))
    _flag(argv, "--test-fraction", split.get("test_fraction"))
    _flag(argv, "--spatial-lon-bins", split.get("spatial_lon_bins"))
    _flag(argv, "--spatial-lat-bins", split.get("spatial_lat_bins"))
    _flag(argv, "--spatial-val-block-fraction", split.get("spatial_val_block_fraction"))
    _flag(argv, "--spatial-test-block-fraction", split.get("spatial_test_block_fraction"))
    _flag(argv, "--spatial-altitude-bins", split.get("spatial_altitude_bins"))
    _flag(argv, "--ood-low-altitude-max-km", split.get("ood_low_altitude_max_km"))
    _flag(argv, "--ood-high-altitude-min-km", split.get("ood_high_altitude_min_km"))
    _flag(argv, "--ood-holdout-fraction", split.get("ood_holdout_fraction"))

    # --- scaler (train-only enforced by the engine; modes are explicit here) ---
    _flag(argv, "--u-scale-mode", scaler.get("u_scale_mode"))
    _flag(argv, "--a-scale-mode", scaler.get("a_scale_mode"))
    _flag(argv, "--target-scale-multiplier", scaler.get("target_scale_multiplier"))

    # --- model architecture ---
    _flag(argv, "--model-preset", model.get("model_preset"))
    _flag(argv, "--hidden", model.get("hidden"))
    _flag(argv, "--depth", model.get("depth"))
    _flag(argv, "--activation", model.get("activation"))
    _flag(argv, "--n-bands", model.get("n_bands"))
    _flag(argv, "--w0-first", model.get("w0_first"))
    _flag(argv, "--w0-hidden", model.get("w0_hidden"))
    _bool_flag(argv, model.get("use_residual_blocks"), on="--use-residual-blocks", off="--no-residual-blocks")

    # --- loss ---
    _flag(argv, "--w-u", loss.get("w_u"))
    _flag(argv, "--w-a", loss.get("w_a"))
    _flag(argv, "--direction-loss-weight", loss.get("direction_loss_weight"))
    _flag(argv, "--potential-only-epochs", loss.get("potential_only_epochs"))
    _flag(argv, "--accel-ramp-epochs", loss.get("accel_ramp_epochs"))
    _bool_flag(argv, loss.get("use_altitude_balanced_loss"), on="--use-altitude-balanced-loss", off="--no-altitude-balanced-loss")
    _bool_flag(argv, loss.get("use_radial_cross_loss"), on="--use-radial-cross-loss", off="--no-radial-cross-loss")

    # --- optimizer ---
    _flag(argv, "--lr", optimizer.get("lr"))
    _flag(argv, "--weight-decay", optimizer.get("weight_decay"))
    _flag(argv, "--max-grad-norm", optimizer.get("max_grad_norm"))
    _flag(argv, "--warmup-epochs", optimizer.get("warmup_epochs"))
    _flag(argv, "--min-lr-ratio", optimizer.get("min_lr_ratio"))

    # --- schedule ---
    _flag(argv, "--batch-size", config.get("batch_size"))
    _flag(argv, "--epochs", config.get("epochs"))
    if _truthy(early.get("enabled", True)):
        _flag(argv, "--patience", early.get("patience"))

    # --- periodic evaluation (monitor-only; never influences checkpoint choice) ---
    if _truthy(periodic.get("enabled", False)):
        _flag(argv, "--periodic-eval-count", periodic.get("count"))
        _flag(argv, "--periodic-eval-every-epochs", periodic.get("every_epochs"))
        _flag(argv, "--periodic-eval-dataset", periodic.get("dataset"))
        _flag(argv, "--periodic-eval-prefer-checkpoint", periodic.get("prefer_checkpoint"))

    return argv


def build_training_command(config: Mapping[str, Any], *, python: str) -> list[str]:
    """Full subprocess command: ``[python, -m, <module>, *flags]``."""
    return [str(python), "-m", TRAINING_MODULE, *build_training_argv(config)]


def find_unfilled_placeholders(config: Mapping[str, Any]) -> list[str]:
    """Return dataset paths that are still placeholders (``<FILL ...>``)."""
    dataset = _section(config, "dataset")
    unfilled: list[str] = []
    for key in ("train_data", "val_data", "test_data", "ood_data"):
        value = dataset.get(key)
        if isinstance(value, str) and _is_placeholder(value):
            unfilled.append(f"dataset.{key}={value}")
    return unfilled


def _is_placeholder(value: str) -> bool:
    text = value.strip()
    return text.startswith("<") or "FILL" in text.upper() or text.upper() in {"TODO", "TBD"}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def resolve_out_dir(config: Mapping[str, Any]) -> Path:
    out = _section(config, "output").get("out_dir")
    if not out:
        raise ValueError("config.output.out_dir is required")
    return Path(str(out))


__all__ = [
    "TRAINING_MODULE",
    "build_training_argv",
    "build_training_command",
    "find_unfilled_placeholders",
    "resolve_out_dir",
]
