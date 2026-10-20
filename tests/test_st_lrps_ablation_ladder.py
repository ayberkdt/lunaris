"""ST-LRPS ablation ladder: one component per step, seeds, comparable metric.

* A0..A6 resolve (through the real training parser) to configs that differ in
  exactly one component per step, and A6 is the shipped TrainConfig default.
* Every arm runs once per seed on one shared data split.
* The validation metric that compares arms (``val_accel_rmse_m_s2``) is an
  unweighted residual-acceleration RMS: it does not move with the
  altitude-balanced option that changes ``val_loss_a``.
* The HPC scenario file is generated from the registry.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lunaris.surrogate.st_lrps.evaluation import ablation as ram

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ladder():
    return [spec for spec in ram.ABLATION_REGISTRY if spec.include_in_default_matrix]


# --------------------------------------------------------------------------- #
# Ladder
# --------------------------------------------------------------------------- #


def _resolve(monkeypatch, overrides):
    from lunaris.surrogate.st_lrps.training.config import parse_args

    monkeypatch.setattr(
        sys, "argv", ["lunaris-train", "--train-data", "t.h5", "--val-data", "v.h5", "--out", "o", *overrides]
    )
    return dataclasses.asdict(parse_args())


def test_each_ladder_step_changes_exactly_one_component(monkeypatch):
    pytest.importorskip("torch")
    cfgs = [_resolve(monkeypatch, spec.cli_overrides) for spec in _ladder()]
    changed = [
        sorted(k for k in prev if prev[k] != cur[k]) for prev, cur in itertools.pairwise(cfgs)
    ]
    assert changed == [
        ["use_residual_blocks"],
        ["n_bands"],
        ["use_altitude_balanced_loss"],
        ["direction_loss_weight"],
        ["use_radial_cross_loss"],
        # One component: the preset and the encoding flag it implies.
        ["model_preset", "use_physical_radial_decay_encoding"],
    ]
    assert cfgs[0]["model_preset"] == "baseline_raw"


def test_a6_is_the_shipped_default(monkeypatch):
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.training.config import TrainConfig

    a6 = _resolve(monkeypatch, _ladder()[-1].cli_overrides)
    shipped = _resolve(monkeypatch, [])
    assert a6 == shipped
    assert a6["model_preset"] == TrainConfig.model_preset
    # A4/A5 use the shipped direction weight, so A6 changes only the encoding.
    a5 = _resolve(monkeypatch, _ladder()[-2].cli_overrides)
    assert a5["direction_loss_weight"] == TrainConfig.direction_loss_weight


def test_no_registered_arm_duplicates_the_shipped_default(monkeypatch):
    pytest.importorskip("torch")
    shipped = _resolve(monkeypatch, [])
    duplicates = [
        spec.name for spec in ram.ABLATION_REGISTRY
        if spec.name != "A6_full_recommended" and _resolve(monkeypatch, spec.cli_overrides) == shipped
    ]
    assert duplicates == []


# --------------------------------------------------------------------------- #
# Seeds and split
# --------------------------------------------------------------------------- #


def test_every_arm_runs_three_seeds_on_one_split(tmp_path):
    args = ram.parse_args(["--train-data", "t.h5", "--out-root", str(tmp_path), "--dry-run"])
    entries = ram.build_matrix(args)
    assert len(entries) == 7 * 3
    for entry in entries:
        cmd = entry["command"]
        assert cmd[cmd.index("--seed") + 1] == str(entry["seed"])
        assert cmd[cmd.index("--split-seed") + 1] == "42"
        assert Path(entry["out_dir"]) == tmp_path / entry["name"] / f"seed{entry['seed']}"
    assert sorted({e["seed"] for e in entries}) == [42, 43, 44]
    assert len({e["run_id"] for e in entries}) == len(entries)


def test_explicit_split_seed_and_duplicate_seeds(tmp_path):
    args = ram.parse_args(["--seeds", "5", "6", "--split-seed", "9", "--out-root", str(tmp_path)])
    cmds = [e["command"] for e in ram.build_matrix(args)]
    assert {c[c.index("--split-seed") + 1] for c in cmds} == {"9"}
    with pytest.raises(SystemExit):
        ram.parse_args(["--seeds", "5", "5"])


# --------------------------------------------------------------------------- #
# Summary: best-epoch metric and per-arm aggregation
# --------------------------------------------------------------------------- #


def _fake_run(run_dir: Path, *, best_epoch: int, val_rmse: list[float], test_rmse: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "best_epoch": best_epoch}), encoding="utf-8"
    )
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
        for i, value in enumerate(val_rmse):
            handle.write(json.dumps({"epoch": i, "epoch_display": i + 1, "val_accel_rmse_m_s2": value}) + "\n")
    (run_dir / "evals" / "test").mkdir(parents=True)
    (run_dir / "evals" / "test" / "summary_metrics.json").write_text(
        json.dumps([{"rmse_a_vec": test_rmse}]), encoding="utf-8"
    )


def test_summary_reads_best_epoch_metric_and_aggregates_seeds(tmp_path):
    out_root = tmp_path / "abl"
    args = ram.parse_args([
        "--train-data", "t.h5", "--out-root", str(out_root), "--only", "A3_plus_altitude_balanced", "--dry-run",
    ])
    entries = ram.build_matrix(args)
    # best epoch 2 (1-based): the last epoch's value must not be reported.
    for entry, (val, test) in zip(entries, [(1.0e-6, 2.0e-6), (3.0e-6, 4.0e-6), (5.0e-6, 6.0e-6)], strict=True):
        _fake_run(Path(entry["out_dir"]), best_epoch=2, val_rmse=[9.0, val, 9.0], test_rmse=test)

    ram.aggregate(entries, out_root)
    runs = json.loads((out_root / "ablation_summary.json").read_text(encoding="utf-8"))
    assert [r["val_accel_rmse_m_s2"] for r in runs] == [1.0e-6, 3.0e-6, 5.0e-6]

    (arm,) = ram._arm_rows(runs)
    assert arm["n_runs"] == 3 and arm["n_completed"] == 3
    assert arm["val_accel_rmse_m_s2_mean"] == pytest.approx(3.0e-6)
    assert arm["val_accel_rmse_m_s2_std"] == pytest.approx(2.0e-6)  # sample std of 1, 3, 5
    assert arm["test_rmse_a_mean"] == pytest.approx(4.0e-6)
    md = (out_root / "st_lrps_ablation_summary.md").read_text(encoding="utf-8")
    assert "3e-06 ± 2e-06 (n=3)" in md
    assert (out_root / "ablation_ranked_by_val_accel_rmse.csv").is_file()
    assert not (out_root / "ablation_ranked_by_val_base_loss.csv").exists()


def test_single_seed_arm_has_no_std():
    (arm,) = ram._arm_rows([{"name": "A0", "seed": 1, "status": "completed", "val_accel_rmse_m_s2": 2.0}])
    assert arm["val_accel_rmse_m_s2_mean"] == 2.0 and arm["val_accel_rmse_m_s2_std"] is None
    assert ram._md_mean_std(arm, "val_accel_rmse_m_s2") == "2 (n=1)"


# --------------------------------------------------------------------------- #
# Unweighted acceleration metric
# --------------------------------------------------------------------------- #


def _loss_fixture():
    torch = pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
    from lunaris.surrogate.st_lrps.training.losses import GradNormWeights, SobolevLoss

    sc = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0e4),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    loss = SobolevLoss(sc, a_sign=1.0)
    g = torch.Generator().manual_seed(0)
    r = 1.737e6 + torch.rand(64, 1, generator=g) * 600.0e3
    d = torch.randn(64, 3, generator=g)
    x = r * d / d.norm(dim=-1, keepdim=True)
    u = torch.randn(64, 1, generator=g)
    a = torch.randn(64, 3, generator=g) * 1.0e-3
    return torch, loss, GradNormWeights(mode="fixed"), x, u, a


def test_accel_sq_err_matches_point_mass_residual_for_a_zero_model():
    torch, loss, weights, x, u, a = _loss_fixture()
    model = torch.nn.Linear(3, 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    _, stats = loss(model, x, u, a, weights, is_train=False)

    # Default contract: full target over a point-mass baseline, a = grad(mu/r).
    xd, ad = x.double().numpy(), a.double().numpy()
    rd = np.linalg.norm(xd, axis=1, keepdims=True)
    a_base = -loss.target_contract.mu_si * xd / rd**3
    expected = float(np.mean(np.sum((ad - a_base) ** 2, axis=1)))
    # float32 forward pass: relative agreement to ~1e-6 expected, 1e-5 asserted.
    assert stats["accel_sq_err"] == pytest.approx(expected, rel=1.0e-5)


def test_accel_sq_err_does_not_move_with_altitude_balancing():
    torch, loss, weights, x, u, a = _loss_fixture()
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    _, plain = loss(model, x, u, a, weights, is_train=False, use_altitude_balanced_loss=False)
    _, balanced = loss(model, x, u, a, weights, is_train=False, use_altitude_balanced_loss=True)
    assert plain["mse_a"] != pytest.approx(balanced["mse_a"], rel=1.0e-3)  # the option matters here
    assert plain["accel_sq_err"] == balanced["accel_sq_err"]


def test_history_row_carries_the_epoch_rmse():
    from lunaris.surrogate.st_lrps.training.metrics import HISTORY_FIELDNAMES, flatten_epoch_metrics

    row = flatten_epoch_metrics(
        0, {"accel_rmse_m_s2": 2.0e-5}, {"accel_rmse_m_s2": 3.0e-5}, {"score": 1.0}, cfg=None
    )
    assert row["train_accel_rmse_m_s2"] == 2.0e-5
    assert row["val_accel_rmse_m_s2"] == 3.0e-5
    assert {"train_accel_rmse_m_s2", "val_accel_rmse_m_s2"} <= set(HISTORY_FIELDNAMES)
    assert math.isfinite(row["val_accel_rmse_m_s2"])


# --------------------------------------------------------------------------- #
# HPC scenario file is generated from the registry
# --------------------------------------------------------------------------- #


def test_hpc_scenario_file_matches_the_registry():
    path = _REPO_ROOT / "hpc" / "scenarios" / ram.HPC_SCENARIO_FILE
    assert path.read_text(encoding="utf-8") == ram.render_hpc_scenarios(), (
        "hpc/scenarios ablation file is stale; run python tools/hpc/generate_ablation_scenarios.py"
    )


def test_hpc_scenarios_mirror_the_matrix_commands():
    records = ram.hpc_scenario_records()
    args = ram.parse_args(["--train-data", "t.h5", "--dry-run"])
    entries = ram.build_matrix(args)
    assert len(records) == len(entries) == 21
    # Seed-major: the first seven lines are one complete ladder.
    assert [r["tags"][2] for r in records[:7]] == [s.name.split("_", 1)[0] for s in _ladder()]
    by_name = {f"PotentialAutograd_{e['name']}_Seed{e['seed']}": e for e in entries}
    assert set(by_name) == {r["name"] for r in records}
    for record in records:
        entry = by_name[record["name"]]
        # Same overrides, seed and split seed as the lunaris-ablation command.
        assert record["flags"] == [*entry["overrides"], "--seed", str(entry["seed"]), "--split-seed", "42"]


def test_generator_check_mode_passes_on_the_committed_file():
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "tools" / "hpc" / "generate_ablation_scenarios.py"), "--check"],
        capture_output=True, text=True, check=False, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
