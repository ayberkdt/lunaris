"""Safety validation for ST-LRPS paper training configs.

A paper config must be explicit and paper-safe: train-only scaler, strict
dataset-contract validation, and no legacy/debug escape hatches. This validator
rejects an unsafe config *before* any training is launched, with messages that
say exactly what to fix.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class PaperConfigError(ValueError):
    """Raised when an ST-LRPS paper training config is unsafe or incomplete."""


# Fields that must exist and be explicitly safe. Maps a human label to a
# predicate that returns an error string when the config is unsafe.
def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    return value if isinstance(value, Mapping) else {}


def load_paper_training_config(path: str | Path) -> dict[str, Any]:
    """Load a paper training config (JSON) and validate it. Raises on unsafe."""
    p = Path(path).expanduser()
    if not p.exists():
        raise PaperConfigError(f"paper training config does not exist: {p}")
    try:
        config = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PaperConfigError(f"paper training config is not valid JSON ({p}): {exc}") from exc
    if not isinstance(config, dict):
        raise PaperConfigError(f"paper training config must be a JSON object: {p}")
    validate_st_lrps_paper_training_config(config)
    return config


def validate_st_lrps_paper_training_config(config: Mapping[str, Any]) -> None:
    """Validate a paper training config; raise :class:`PaperConfigError` if unsafe.

    Checks (all must hold):
      * scaler fit scope is ``train_only``,
      * split policy is present,
      * strict dataset-contract validation is enabled,
      * no legacy/missing dataset contract, no validation-fail bypass,
      * no legacy derivative convention, no legacy target-mode inference,
      * output directory, seed, target mode, and base/target SH degrees present,
      * artifact-contract output enabled.

    Dataset *paths* may be placeholders here — they are filled in by the user and
    checked at run time, not at config-validation time.
    """
    errors: list[str] = []

    scaler = _section(config, "scaler")
    split = _section(config, "split")
    target = _section(config, "target")
    output = _section(config, "output")
    safety = _section(config, "contract_safety")

    # --- scaler must be train-only ---
    fit_scope = str(scaler.get("fit_scope", "")).strip().lower()
    if fit_scope != "train_only":
        errors.append(
            f"scaler.fit_scope must be 'train_only' for paper configs, got {scaler.get('fit_scope')!r}. "
            "Full-dataset scaler fitting leaks validation target statistics into training."
        )

    # --- split policy present ---
    if not str(split.get("split_policy", "")).strip():
        errors.append("split.split_policy is required (e.g. 'spatial_block' or 'ood_low_altitude').")

    # --- strict dataset-contract validation, no legacy/missing escapes ---
    if not _is_true(safety.get("strict_dataset_contract")):
        errors.append("contract_safety.strict_dataset_contract must be true.")
    unsafe_allows = {
        "allow_legacy_dataset_contract": "legacy dataset contracts",
        "allow_missing_dataset_contract": "missing dataset contracts",
        "allow_dataset_validation_fail": "dataset validation failures",
        "allow_legacy_derivative_convention": "legacy derivative convention",
        "allow_legacy_target_mode_inference": "legacy target-mode inference",
    }
    for key, what in unsafe_allows.items():
        if _is_true(safety.get(key)):
            errors.append(f"contract_safety.{key} must be false; paper configs may not allow {what}.")

    # --- required scientific fields ---
    if not str(output.get("out_dir", "")).strip():
        errors.append("output.out_dir is required.")
    if config.get("seed") is None:
        errors.append("seed is required.")
    if not str(target.get("target_mode", "")).strip():
        errors.append("target.target_mode is required ('residual' or 'full').")
    else:
        tmode = str(target.get("target_mode")).strip().lower()
        if tmode not in ("residual", "full"):
            errors.append(f"target.target_mode must be 'residual' or 'full', got {target.get('target_mode')!r}.")
    if target.get("base_sh_degree") is None:
        errors.append("target.base_sh_degree is required (the analytical SH baseline degree).")
    if target.get("target_sh_degree") is None:
        errors.append("target.target_sh_degree is required (the high-fidelity SH degree).")
    if (
        target.get("base_sh_degree") is not None
        and target.get("target_sh_degree") is not None
        and int(target["target_sh_degree"]) <= int(target["base_sh_degree"])
    ):
        errors.append("target.target_sh_degree must exceed target.base_sh_degree.")

    # --- artifact-contract output required ---
    if not _is_true(config.get("artifact_contract_output")):
        errors.append("artifact_contract_output must be true so the run is contract-checkable.")

    if errors:
        raise PaperConfigError(
            "Unsafe/incomplete ST-LRPS paper training config:\n  - " + "\n  - ".join(errors)
        )


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


__all__ = [
    "PaperConfigError",
    "load_paper_training_config",
    "validate_st_lrps_paper_training_config",
]
