"""Every lunaris-train row in hpc/scenarios must pass the trainer's own parser.

A scenario row is submitted to the cluster as-is, so a flag the trainer rejects
or silently ignores costs a whole A100 run. This parses each row exactly as the
launcher passes it (row flags + data flags + --out) without starting training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch.nn")

from lunaris.surrogate.st_lrps.networks.models import build_model_from_config  # noqa: E402
from lunaris.surrogate.st_lrps.training.config import (  # noqa: E402
    TrainConfig,
    check_residual_block_support,
    parse_args,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tools" / "hpc"))

import run_training_scenario as launcher  # noqa: E402

_TRAIN_ROWS = [
    pytest.param(scenario, id=f"{path.name}:{scenario['name']}")
    for path in sorted((_REPO_ROOT / "hpc" / "scenarios").glob("*.jsonl"))
    for scenario in launcher.load_scenarios(path)
    if scenario.get("entrypoint") == "lunaris-train"
]


def _parse(monkeypatch: pytest.MonkeyPatch, flags: list[str], tmp_path: Path) -> TrainConfig:
    data = ["--train-data", str(tmp_path / "train.h5"), "--val-data", str(tmp_path / "val.h5")]
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *flags, *data, "--out", str(tmp_path / "run")])
    return parse_args()


@pytest.mark.parametrize("scenario", _TRAIN_ROWS)
def test_scenario_row_parses_with_the_trainer(scenario, monkeypatch, tmp_path) -> None:
    launcher.validate_scenario(scenario)
    cfg = _parse(monkeypatch, [str(t) for t in scenario["flags"]], tmp_path)
    assert not (tmp_path / "run").exists(), "parsing must not create the run directory"
    assert int(cfg.seed) >= 0


def test_there_are_training_rows_to_check() -> None:
    assert len(_TRAIN_ROWS) >= 3


def test_no_residual_blocks_with_multiband_concat_is_refused(monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="no-residual-blocks"):
        _parse(monkeypatch, ["--no-residual-blocks", "--n-bands", "2"], tmp_path)


@pytest.mark.parametrize("flags", [
    ["--no-residual-blocks", "--n-bands", "1"],
    ["--no-residual-blocks", "--n-bands", "2", "--multiscale-mode", "additive"],
    ["--use-residual-blocks", "--n-bands", "2"],
])
def test_supported_residual_combinations_parse(flags, monkeypatch, tmp_path) -> None:
    cfg = _parse(monkeypatch, flags, tmp_path)
    check_residual_block_support(cfg)


def test_legacy_multiband_artifact_without_residual_flag_still_builds_residual(caplog) -> None:
    """Artifacts trained before the guard record use_residual_blocks=False but hold
    residual-block weights; they must keep loading into the network they trained."""
    cfg = {
        "activation": "sine", "hidden": 32, "depth": 4, "n_bands": 2,
        "multiscale_mode": "concat_shared", "degree_min": 26, "degree_max": 200,
        "use_residual_blocks": False, "model_preset": "baseline_raw",
    }
    legacy = build_model_from_config(cfg)
    residual = build_model_from_config({**cfg, "use_residual_blocks": True})
    assert sorted(legacy.state_dict()) == sorted(residual.state_dict())
    assert "always uses residual blocks" in caplog.text
