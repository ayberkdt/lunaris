# tests/test_st_lrps_run_preset.py
"""
Contract tests for the run-level preset in
``lunaris.surrogate.st_lrps.training.config``.

``run_preset`` is orthogonal to ``model_preset``: ``model_preset`` selects the
input-encoding architecture, ``run_preset`` selects the reproducibility /
evaluation posture of the run. The risk being guarded is *silent scientific
overreach* — a "paper" run reported on an in-distribution interpolation split, or
with a reproducibility guard (determinism / preflight / AMP / legacy bypass)
quietly disabled. ``apply_run_preset`` must therefore fail loudly on a
conflicting explicit flag and never weaken the paper posture.

These are pure-Python tests (no torch / h5py / dataset files) plus a couple of
``parse_args`` integration tests that exercise the CLI wiring with a throwaway
``.h5`` path.
"""

from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import sys

import pytest

from lunaris.surrogate.st_lrps.training.config import (
    RUN_PRESETS,
    TrainConfig,
    apply_run_preset,
    detect_explicit_run_preset_fields,
    parse_args,
)


def _cfg(**kw) -> TrainConfig:
    base = dict(data="dummy.h5", out="run_out")
    base.update(kw)
    return TrainConfig(**base)


# =============================================================================
# custom / development / quick
# =============================================================================

def test_custom_preset_is_a_noop():
    cfg = apply_run_preset(_cfg(run_preset="custom"))
    assert cfg.run_preset == "custom"
    # Every default that the other presets touch is left untouched.
    assert cfg.split_policy == "seeded_random"
    assert cfg.deterministic is True
    assert cfg.amp is False
    assert cfg.quick_check is False


def test_default_train_config_uses_custom_run_preset():
    # Backward-compatibility guard: nothing changes unless a preset is requested.
    assert _cfg().run_preset == "custom"


def test_development_defaults_to_interpolation_split():
    cfg = apply_run_preset(_cfg(run_preset="development"))
    assert cfg.split_policy == "seeded_random"


def test_development_respects_explicit_split():
    cfg = apply_run_preset(
        _cfg(run_preset="development", split_policy="spatial_block"),
        explicit_fields={"split_policy"},
    )
    assert cfg.split_policy == "spatial_block"


def test_quick_enables_smoke_path():
    cfg = apply_run_preset(_cfg(run_preset="quick"))
    assert cfg.quick_check is True
    assert cfg.allow_preflight_fail is True


def test_quick_respects_explicit_flags():
    cfg = apply_run_preset(
        _cfg(run_preset="quick", quick_check=False),
        explicit_fields={"quick_check"},
    )
    assert cfg.quick_check is False


# =============================================================================
# paper — split policy
# =============================================================================

def test_paper_forces_spatial_block_from_interpolation_default():
    cfg = apply_run_preset(_cfg(run_preset="paper"))
    assert cfg.split_policy == "spatial_block"


def test_paper_keeps_explicit_generalization_split():
    cfg = apply_run_preset(
        _cfg(run_preset="paper", split_policy="ood_high_altitude"),
        explicit_fields={"split_policy"},
    )
    assert cfg.split_policy == "ood_high_altitude"


def test_paper_rejects_explicit_interpolation_split():
    with pytest.raises(ValueError, match="generalization split"):
        apply_run_preset(
            _cfg(run_preset="paper", split_policy="seeded_random"),
            explicit_fields={"split_policy"},
        )


# =============================================================================
# paper — reproducibility posture
# =============================================================================

def test_paper_enforces_full_reproducibility_posture():
    cfg = apply_run_preset(
        _cfg(
            run_preset="paper",
            # Start from a deliberately unsafe programmatic config.
            deterministic=False,
            benchmark_cudnn=True,
            amp=True,
            quick_check=True,
            skip_preflight=True,
            allow_preflight_fail=True,
            allow_dataset_validation_fail=True,
        )
    )
    # No explicit_fields → programmatic enforcement (never raises, always fixes).
    assert cfg.deterministic is True
    assert cfg.benchmark_cudnn is False
    assert cfg.amp is False
    assert cfg.quick_check is False
    assert cfg.skip_preflight is False
    assert cfg.allow_preflight_fail is False
    assert cfg.allow_dataset_validation_fail is False
    assert cfg.split_policy == "spatial_block"


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("deterministic", False),
        ("benchmark_cudnn", True),
        ("amp", True),
        ("quick_check", True),
        ("skip_preflight", True),
        ("allow_preflight_fail", True),
        ("allow_dataset_validation_fail", True),
    ],
)
def test_paper_raises_on_explicit_conflicting_flag(field, bad_value):
    with pytest.raises(ValueError, match="run_preset='paper'"):
        apply_run_preset(
            _cfg(run_preset="paper", **{field: bad_value}),
            explicit_fields={field},
        )


def test_paper_is_idempotent():
    cfg = apply_run_preset(_cfg(run_preset="paper", split_policy="ood_low_altitude"))
    # Re-applying (as the engine does) must not clobber the generalization split.
    first_split = cfg.split_policy
    apply_run_preset(cfg)
    assert cfg.split_policy == first_split
    assert cfg.deterministic is True
    assert cfg.amp is False


def test_unknown_run_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown run_preset"):
        apply_run_preset(_cfg(run_preset="totally_made_up"))


def test_run_presets_tuple_is_the_authoritative_list():
    assert RUN_PRESETS == ("custom", "development", "quick", "paper")


# =============================================================================
# explicit-flag detection
# =============================================================================

def test_detect_explicit_fields_handles_value_and_negation_flags():
    argv = [
        "--data", "x.h5",
        "--split-policy", "spatial_block",
        "--no-deterministic",
        "--amp",
        "--allow-preflight-fail",
    ]
    explicit = detect_explicit_run_preset_fields(argv)
    assert "split_policy" in explicit
    assert "deterministic" in explicit
    assert "amp" in explicit
    assert "allow_preflight_fail" in explicit
    assert "benchmark_cudnn" not in explicit


def test_detect_explicit_fields_handles_equals_form():
    explicit = detect_explicit_run_preset_fields(["--split-policy=spatial_block"])
    assert "split_policy" in explicit


# =============================================================================
# parse_args integration (CLI wiring)
# =============================================================================

def test_parse_args_paper_preset_sets_posture(monkeypatch, tmp_path):
    fake = tmp_path / "fake.h5"
    fake.write_bytes(b"not a real h5")  # DatasetMeta.from_h5 fails -> caught
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--data", str(fake), "--out", str(tmp_path / "out"), "--run-preset", "paper"],
    )
    cfg = parse_args()
    assert cfg.run_preset == "paper"
    assert cfg.split_policy == "spatial_block"
    assert cfg.deterministic is True
    assert cfg.amp is False


def test_parse_args_paper_with_conflicting_flag_raises(monkeypatch, tmp_path):
    fake = tmp_path / "fake.h5"
    fake.write_bytes(b"not a real h5")
    monkeypatch.setattr(
        sys, "argv",
        [
            "prog", "--data", str(fake), "--out", str(tmp_path / "out"),
            "--run-preset", "paper", "--no-deterministic",
        ],
    )
    with pytest.raises(ValueError, match="run_preset='paper'"):
        parse_args()
