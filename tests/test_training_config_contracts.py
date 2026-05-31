# tests/test_training_config_contracts.py
# -*- coding: utf-8 -*-
"""
Contract tests for ``lunaris.surrogate.st_lrps.training.config``.

These are pure-Python (no torch / h5py / dataset files): they exercise
``TrainConfig`` defaults and ``apply_model_preset`` only. The risk being guarded
is *input-encoding drift* — a preset silently enabling the wrong feature
encoding, or a manual flag silently overriding a named preset, would mislabel an
ablation and quietly change the learned physics.
"""

from __future__ import annotations

import pytest

from lunaris.surrogate.st_lrps.training.config import (
    MODEL_PRESETS,
    TrainConfig,
    _ENCODING_FLAGS,
    apply_model_preset,
)


def _cfg(**kw) -> TrainConfig:
    base = dict(data="dummy.h5", out="run_out")
    base.update(kw)
    return TrainConfig(**base)


def _active_encodings(cfg: TrainConfig) -> set[str]:
    return {name for name in _ENCODING_FLAGS if bool(getattr(cfg, name))}


# =============================================================================
# Presets
# =============================================================================

def test_baseline_raw_preset_leaves_all_encodings_off():
    cfg = apply_model_preset(_cfg(model_preset="baseline_raw"))
    assert _active_encodings(cfg) == set()


def test_recommended_preset_enables_only_physical_radial_decay():
    cfg = apply_model_preset(_cfg(model_preset="recommended_physical_radial_decay"))
    assert _active_encodings(cfg) == {"use_physical_radial_decay_encoding"}
    # Recommended encoding ships with its physically informed sub-options on.
    assert cfg.physical_radial_decay_include_unit is True
    assert cfg.physical_radial_decay_include_r_scaled is True


@pytest.mark.parametrize("preset,expected_flag", [
    ("ablation_radial_separation", "use_radial_separation"),
    ("ablation_radial_decay_scaled", "use_radial_decay_encoding"),
    ("ablation_real_sh_low_degree", "use_real_sh_basis"),
])
def test_ablation_presets_enable_exactly_one_encoding(preset, expected_flag):
    cfg = apply_model_preset(_cfg(model_preset=preset))
    assert _active_encodings(cfg) == {expected_flag}


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown model_preset"):
        apply_model_preset(_cfg(model_preset="totally_made_up"))


# =============================================================================
# Manual encoding vs preset conflicts
# =============================================================================

def test_explicit_preset_with_conflicting_manual_flag_raises():
    cfg = _cfg(model_preset="baseline_raw", use_fourier=True)
    cfg._model_preset_explicit = True  # user explicitly asked for the preset
    with pytest.raises(ValueError, match="conflicts with manual encoding"):
        apply_model_preset(cfg)


def test_non_explicit_preset_with_manual_flag_downgrades_to_custom():
    # When the preset was not explicitly requested, a manual encoding flag silently
    # downgrades to 'custom' (preserving the manual choice) rather than erroring.
    cfg = _cfg(model_preset="baseline_raw", use_fourier=True)
    apply_model_preset(cfg)
    assert cfg.model_preset == "custom"
    assert cfg.use_fourier is True


def test_custom_preset_preserves_manual_encoding_choices():
    cfg = apply_model_preset(_cfg(model_preset="custom", use_radial_separation=True))
    assert cfg.model_preset == "custom"
    assert cfg.use_radial_separation is True
    assert _active_encodings(cfg) == {"use_radial_separation"}


# =============================================================================
# Runtime-kind handling
# =============================================================================

def test_runtime_model_kind_default_and_reserved_value_preserved():
    assert _cfg().runtime_model_kind == "potential_autograd"
    assert TrainConfig.__dataclass_fields__["runtime_model_kind"].default == "potential_autograd"

    # 'force_direct' is a reserved future value: it must be preserved (not silently
    # rewritten) by preset application, but it remains unsupported at runtime
    # (enforced by the runtime layer, see test_st_lrps_runtime_contracts).
    cfg = apply_model_preset(_cfg(model_preset="custom", runtime_model_kind="force_direct"))
    assert cfg.runtime_model_kind == "force_direct"


# =============================================================================
# Default-config internal consistency
# =============================================================================

def test_default_config_values_are_internally_consistent():
    cfg = _cfg()

    # Positive structural / optimization quantities.
    assert cfg.epochs > 0
    assert cfg.batch_size > 0
    assert cfg.depth > 0
    assert cfg.hidden > 0
    assert cfg.lr > 0.0
    assert cfg.patience > 0
    assert cfg.n_bands >= 1

    # Loss weights are non-negative.
    for w in (cfg.w_u, cfg.w_a, cfg.radial_loss_weight, cfg.cross_loss_weight,
              cfg.direction_loss_weight, cfg.weight_decay):
        assert w >= 0.0

    # Valid ratios / ranges.
    assert 0.0 < cfg.val_ratio < 1.0
    assert 0.0 < cfg.min_lr_ratio <= 1.0
    assert cfg.altitude_min_km < cfg.altitude_max_km
    assert 1 <= cfg.sh_encoding_degree <= 8

    # The default preset implies no manual encodings are active yet.
    assert _active_encodings(cfg) == set()
    assert cfg.model_preset in MODEL_PRESETS
