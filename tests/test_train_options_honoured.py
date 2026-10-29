"""Training options must be honoured, or the run must say loudly that they are not.

Each test pins one option that used to be dropped without a word:

1. an explicit ``--split-policy`` / ``--deterministic`` under ``--run-preset development``
2. ``--physical-radial-decay-*`` flags overwritten by ``--model-preset``
3. ``--w0-first`` / ``--w0-hidden`` ignored when ``n_bands > 1``
4. ``--output-envelope-power`` overridden by ``--output-envelope-auto``
5. ``--w-a`` discarded under ``--gradnorm-mode ntk_init``
6. one explicit altitude bound dropping the dataset value of the other
7. a fresh run overwriting an existing run directory before it refuses
8. ``--patience 0`` / ``--warmup-epochs -3`` clipped to a valid value

The valid production commands (hpc/scenarios, the L1000 command) must keep resolving
to the same configuration; the last group of tests pins the stage-1 screening flags.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch.nn")

from lunaris.surrogate.st_lrps.training import engine  # noqa: E402
from lunaris.surrogate.st_lrps.training.config import (  # noqa: E402
    TrainConfig,
    parse_args,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The flags of hpc/scenarios/st_lrps_variant_screen_stage1.jsonl that touch these items.
_STAGE1_FLAGS = [
    "--w0-first", "30.0", "--w0-hidden", "30.0", "--w-a", "1.0",
    "--gradnorm-mode", "ntk_init", "--gradnorm-w-a-max", "2.0",
    "--output-envelope-auto", "--n-bands", "2",
    "--altitude-min-km", "100.0", "--altitude-max-km", "1000.0",
]


def _parse(monkeypatch: pytest.MonkeyPatch, flags: list[str], tmp_path: Path,
           *, train_data: Path | None = None) -> TrainConfig:
    data = ["--train-data", str(train_data or tmp_path / "train.h5"),
            "--val-data", str(tmp_path / "val.h5")]
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *flags, *data, "--out", str(tmp_path / "run")])
    return parse_args()


# --------------------------------------------------------------------------- 1
def test_run_preset_development_keeps_explicit_split_policy_and_determinism(
    monkeypatch, tmp_path
) -> None:
    cfg = _parse(
        monkeypatch,
        ["--run-preset", "development", "--split-policy", "spatial_block", "--deterministic"],
        tmp_path,
    )
    assert cfg.split_policy == "spatial_block" and cfg.deterministic is True

    class _Stop(Exception):
        pass

    seen: dict[str, object] = {}

    def _record(seed, *, deterministic, benchmark):
        seen["deterministic"] = deterministic
        raise _Stop

    # set_seed is the first consumer of the flags after the engine re-applies the presets.
    monkeypatch.setattr(engine, "set_seed", _record)
    with pytest.raises(_Stop):
        engine.build_training_session(cfg)
    assert cfg.split_policy == "spatial_block", "engine re-applied the preset over an explicit value"
    assert seen["deterministic"] is True


def test_run_preset_development_still_defaults_unset_fields(monkeypatch, tmp_path) -> None:
    cfg = _parse(monkeypatch, ["--run-preset", "development"], tmp_path)
    assert cfg.split_policy == "seeded_random"
    assert cfg.deterministic is False
    assert cfg.benchmark_cudnn is True


def test_paper_preset_engine_reapplication_is_unchanged(monkeypatch, tmp_path) -> None:
    import dataclasses

    from lunaris.surrogate.st_lrps.training.config import apply_run_preset

    cfg = _parse(
        monkeypatch,
        ["--run-preset", "paper", "--split-policy", "spatial_block", "--deterministic", "--no-amp"],
        tmp_path,
    )
    before = dataclasses.asdict(cfg)
    apply_run_preset(cfg, getattr(cfg, "_explicit_run_preset_fields", None))
    assert dataclasses.asdict(cfg) == before
    apply_run_preset(cfg)
    assert dataclasses.asdict(cfg) == before


# --------------------------------------------------------------------------- 2
@pytest.mark.parametrize("flags", [
    ["--physical-radial-decay-max-power", "3"],
    ["--physical-radial-decay-max-power=3"],
    ["--no-physical-radial-decay-append-raw"],
    ["--no-physical-radial-decay-include-unit"],
    ["--no-physical-radial-decay-include-r-scaled"],
])
def test_model_preset_overwritten_physical_decay_flags_are_refused(
    flags, monkeypatch, tmp_path
) -> None:
    with pytest.raises(ValueError, match="model-preset"):
        _parse(monkeypatch, ["--model-preset", "recommended_physical_radial_decay", *flags], tmp_path)


@pytest.mark.parametrize("flags", [
    ["--physical-radial-decay-max-power", "4"],
    ["--physical-radial-decay-append-raw", "--physical-radial-decay-include-unit"],
    ["--physical-radial-decay-include-r-scaled"],
])
def test_physical_decay_flags_equal_to_the_preset_values_are_accepted(
    flags, monkeypatch, tmp_path
) -> None:
    cfg = _parse(monkeypatch, ["--model-preset", "recommended_physical_radial_decay", *flags], tmp_path)
    assert cfg.physical_radial_decay_max_power == 4
    assert cfg.physical_radial_decay_append_raw is True


def test_physical_decay_flags_are_free_under_the_custom_preset(monkeypatch, tmp_path) -> None:
    cfg = _parse(
        monkeypatch,
        ["--model-preset", "custom", "--use-physical-radial-decay-encoding",
         "--physical-radial-decay-max-power", "3", "--no-physical-radial-decay-append-raw"],
        tmp_path,
    )
    assert cfg.physical_radial_decay_max_power == 3
    assert cfg.physical_radial_decay_append_raw is False


# ----------------------------------------------------------------------- 3 + 5
def _tiny_cfg(data_path: Path, out_dir: Path, **overrides) -> TrainConfig:
    kwargs = dict(
        data=str(data_path), out=str(out_dir), epochs=1, batch_size=8,
        num_workers=0, pin_memory=False, quick_check=True, max_train_batches=1,
        max_val_batches=1, activation="silu", hidden=8, depth=2, model_preset="custom",
        use_residual_blocks=False, n_bands=1, val_ratio=0.25, split_seed=11,
        a_sign=1.0, gradnorm_mode="fixed", direction_loss_weight=0.0,
        use_altitude_balanced_loss=False, use_radial_cross_loss=False,
        preload_policy="never", auto_preload_mb=0.0, fit_rows=32, fit_chunk_rows=32,
        cache_rows=16, sampler_block_size=16, warmup_epochs=0, best_ckpt_start_epoch=0,
        log_every=0,
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_parse_marks_w0_and_w_a_as_explicit(monkeypatch, tmp_path) -> None:
    explicit = _parse(monkeypatch, _STAGE1_FLAGS, tmp_path)
    assert explicit._w0_explicit is True
    assert explicit._w_a_explicit is True
    implicit = _parse(monkeypatch, [], tmp_path)
    assert implicit._w0_explicit is False
    assert implicit._w_a_explicit is False


def test_explicit_w0_with_multiband_and_explicit_w_a_under_ntk_are_warned_and_recorded(
    tmp_path, caplog
) -> None:
    from dataset_pipeline_test_utils import write_toy_contract_h5

    data_path = write_toy_contract_h5(tmp_path / "toy_train.h5", n=48)
    run_dir = tmp_path / "run"
    cfg = _tiny_cfg(
        data_path, run_dir, activation="sine", n_bands=2, use_residual_blocks=True,
        gradnorm_mode="ntk_init", w_a=1.0, w0_first=30.0, w0_hidden=30.0,
    )
    cfg._w0_explicit = True
    cfg._w_a_explicit = True
    caplog.set_level(logging.INFO, logger=engine.logger.name)
    caplog.set_level(logging.INFO, logger="lunaris.surrogate.st_lrps.training.losses")
    engine.train(cfg)

    text = caplog.text
    assert "--w0-first/--w0-hidden have no effect with n_bands=2" in text
    assert "--w-a=1 is only the starting value under --gradnorm-mode ntk_init" in text
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    recorded = " | ".join(manifest["warnings"])
    assert "--w0-first/--w0-hidden" in recorded and "--w-a" in recorded
    # the value the run really used is in train.log
    assert "NTK-init: w_a=" in (run_dir / "train.log").read_text(encoding="utf-8")


def test_no_ignored_option_warning_when_the_options_were_not_given() -> None:
    cfg = _tiny_cfg(Path("x.h5"), Path("o"), activation="sine", n_bands=2, gradnorm_mode="ntk_init")
    assert engine._ignored_option_warnings(cfg) == []


def test_explicit_w0_with_a_single_band_is_not_warned() -> None:
    cfg = _tiny_cfg(Path("x.h5"), Path("o"), activation="sine", n_bands=1, gradnorm_mode="fixed")
    cfg._w0_explicit = True
    cfg._w_a_explicit = True
    assert engine._ignored_option_warnings(cfg) == []


def test_w0_summary_does_not_call_an_unused_value_user_specified() -> None:
    from lunaris.surrogate.st_lrps.training.config import _w0_origin_note

    note = _w0_origin_note(30.0, 12.7, bands_ignore_w0=True)
    assert "unused" in note.lower() and "user-specified" not in note
    assert _w0_origin_note(30.0, 12.7, bands_ignore_w0=False) == "[user-specified, auto would be 12.7]"


# --------------------------------------------------------------------------- 4
def test_output_envelope_auto_with_explicit_power_is_refused(monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="output-envelope"):
        _parse(monkeypatch, ["--output-envelope-auto", "--output-envelope-power", "3"], tmp_path)
    with pytest.raises(ValueError, match="output-envelope"):
        _parse(monkeypatch, ["--output-envelope-auto", "--output-envelope-power=0"], tmp_path)


def test_output_envelope_auto_or_power_alone_is_accepted(monkeypatch, tmp_path) -> None:
    assert _parse(monkeypatch, ["--output-envelope-auto"], tmp_path).output_envelope_auto is True
    cfg = _parse(monkeypatch, ["--output-envelope-power", "3"], tmp_path)
    assert cfg.output_envelope_power == 3.0 and cfg.output_envelope_auto is False


# --------------------------------------------------------------------------- 6
@pytest.mark.parametrize(("flags", "expected"), [
    (["--altitude-min-km", "150"], (150.0, 500.0)),
    (["--altitude-max-km", "300"], (120.0, 300.0)),
    (["--altitude-min-km", "150", "--altitude-max-km", "300"], (150.0, 300.0)),
    ([], (120.0, 500.0)),
])
def test_one_sided_altitude_bound_keeps_the_dataset_value_of_the_other(
    flags, expected, monkeypatch, tmp_path
) -> None:
    from dataset_pipeline_test_utils import write_toy_contract_h5
    from lunaris.surrogate.st_lrps.artifacts.manager import ensure_run_layout

    # the toy cloud spans 120--500 km, the TrainConfig defaults are 100--1000 km
    data_path = write_toy_contract_h5(tmp_path / "toy_train.h5", n=48, alt_min_km=120.0, alt_max_km=500.0)
    cfg = _parse(monkeypatch, flags, tmp_path, train_data=data_path)
    cfg.val_data = None
    engine._load_dataset_context(cfg, layout=ensure_run_layout(tmp_path / "ctx_run"))
    assert (cfg.altitude_min_km, cfg.altitude_max_km) == expected


# --------------------------------------------------------------------------- 7
def _seed_finished_run(run_dir: Path) -> dict[str, bytes]:
    (run_dir / "checkpoints").mkdir(parents=True)
    files = {
        run_dir / "checkpoints" / "ckpt_last.pt": b"irreplaceable weights",
        run_dir / "run_manifest.json": json.dumps({"status": "completed", "run_id": "old"}).encode(),
        run_dir / "command.txt": b"the original command\n",
        run_dir / "train.log": b"the original log\n",
    }
    for path, payload in files.items():
        path.write_bytes(payload)
    return {str(p): b for p, b in files.items()}


def test_fresh_run_into_a_finished_run_directory_refuses_before_writing(tmp_path) -> None:
    from dataset_pipeline_test_utils import write_toy_contract_h5

    data_path = write_toy_contract_h5(tmp_path / "toy_train.h5", n=48)
    run_dir = tmp_path / "run"
    before = _seed_finished_run(run_dir)
    with pytest.raises(FileExistsError, match="already holds"):
        engine.build_training_session(_tiny_cfg(data_path, run_dir))
    assert {str(p): p.read_bytes() for p in map(Path, before)} == before
    assert not (run_dir / "provenance" / "split_manifest.json").exists()


def test_run_directory_guard_accepts_what_the_hpc_launcher_leaves(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LUNARIS_ALLOW_RUN_OVERWRITE", raising=False)
    missing = tmp_path / "not_there"
    engine._refuse_fresh_run_over_existing(missing)            # directory not created yet
    empty = tmp_path / "empty"
    empty.mkdir()
    engine._refuse_fresh_run_over_existing(empty)              # launcher mkdir
    launched = tmp_path / "launched"
    launched.mkdir()
    for name in ("scenario.json", "scenario_command.txt", "scenario_environment.json"):
        (launched / name).write_text("{}", encoding="utf-8")
    engine._refuse_fresh_run_over_existing(launched)           # launcher provenance files
    # --resume --restart-if-no-checkpoint: the first attempt died before its first save
    stopped = tmp_path / "stopped"
    (stopped / "checkpoints").mkdir(parents=True)
    (stopped / "run_manifest.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    engine._refuse_fresh_run_over_existing(stopped)


def test_run_directory_guard_refuses_checkpoints_and_completed_manifests(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LUNARIS_ALLOW_RUN_OVERWRITE", raising=False)
    with_ckpt = tmp_path / "with_ckpt"
    (with_ckpt / "checkpoints").mkdir(parents=True)
    (with_ckpt / "checkpoints" / "ckpt_best.pt").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="already holds"):
        engine._refuse_fresh_run_over_existing(with_ckpt)
    done = tmp_path / "done"
    done.mkdir()
    (done / "run_manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="already holds"):
        engine._refuse_fresh_run_over_existing(done)


def test_run_directory_guard_has_an_explicit_override(tmp_path, monkeypatch) -> None:
    with_ckpt = tmp_path / "with_ckpt"
    (with_ckpt / "checkpoints").mkdir(parents=True)
    (with_ckpt / "checkpoints" / "ckpt_best.pt").write_bytes(b"x")
    monkeypatch.setenv("LUNARIS_ALLOW_RUN_OVERWRITE", "1")
    engine._refuse_fresh_run_over_existing(with_ckpt)


# --------------------------------------------------------------------------- 8
@pytest.mark.parametrize("flags", [
    ["--patience", "0"], ["--patience", "-4"], ["--patience=0"],
    ["--warmup-epochs", "-3"], ["--warmup-epochs=-1"],
])
def test_invalid_patience_and_warmup_are_refused_not_clipped(flags, monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match=r"--patience|--warmup-epochs"):
        _parse(monkeypatch, flags, tmp_path)


def test_boundary_patience_and_warmup_are_accepted(monkeypatch, tmp_path) -> None:
    cfg = _parse(monkeypatch, ["--patience", "1", "--warmup-epochs", "0"], tmp_path)
    assert (cfg.patience, cfg.warmup_epochs) == (1, 0)


# ------------------------------------------------- valid production flags unchanged
def test_stage1_flags_still_resolve_to_the_same_configuration(monkeypatch, tmp_path) -> None:
    cfg = _parse(monkeypatch, _STAGE1_FLAGS, tmp_path)
    assert (cfg.w0_first, cfg.w0_hidden, cfg.w_a) == (30.0, 30.0, 1.0)
    assert cfg.gradnorm_mode == "ntk_init" and cfg.gradnorm_w_a_max == 2.0
    assert cfg.output_envelope_auto is True and cfg.output_envelope_power == 0.0
    assert cfg.n_bands == 2
    assert (cfg.altitude_min_km, cfg.altitude_max_km) == (100.0, 1000.0)
    assert cfg._altitude_bounds_explicit is True
