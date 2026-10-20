"""Task 7 — ST-LRPS A0..A9 ablation suite."""

from __future__ import annotations

import pytest

from lunaris.surrogate.st_lrps.evaluation import ablation as ram


def _spec(name):
    return next(s for s in ram.ABLATION_REGISTRY if s.name == name)


def test_a0_to_a9_specs_exist():
    names = {s.name for s in ram.ABLATION_REGISTRY}
    for expected in (
        "A0_raw_siren_sobolev",
        "A1_plus_residual_blocks",
        "A2_plus_multiscale",
        "A3_plus_altitude_balanced",
        "A4_plus_direction",
        "A5_plus_radial_cross",
        "A6_full_recommended",
        "A7_physical_radial_decay",
        "A8_real_sh_basis",
        "A9_additive_multiband",
    ):
        assert expected in names


def test_default_matrix_is_a0_to_a6():
    default = [s.name for s in ram.ABLATION_REGISTRY if s.include_in_default_matrix]
    assert default == [
        "A0_raw_siren_sobolev",
        "A1_plus_residual_blocks",
        "A2_plus_multiscale",
        "A3_plus_altitude_balanced",
        "A4_plus_direction",
        "A5_plus_radial_cross",
        "A6_full_recommended",
    ]


def test_cumulative_progression_adds_one_feature_each_step():
    a0 = _spec("A0_raw_siren_sobolev").cli_overrides
    assert "--no-residual-blocks" in a0 and "1" in a0 and "--direction-loss-weight" in a0

    # A1 turns residual blocks ON (still single band).
    a1 = _spec("A1_plus_residual_blocks").cli_overrides
    assert "--use-residual-blocks" in a1 and "--no-residual-blocks" not in a1
    assert a1[a1.index("--n-bands") + 1] == "1"

    # A2 turns multi-scale ON.
    a2 = _spec("A2_plus_multiscale").cli_overrides
    assert a2[a2.index("--n-bands") + 1] == "3"
    assert "--no-altitude-balanced-loss" in a2

    # A3 turns altitude-balanced loss ON.
    a3 = _spec("A3_plus_altitude_balanced").cli_overrides
    assert "--use-altitude-balanced-loss" in a3 and "--no-radial-cross-loss" in a3

    # A4 turns direction loss ON.
    a4 = _spec("A4_plus_direction").cli_overrides
    assert a4[a4.index("--direction-loss-weight") + 1] == "0.2"
    assert "--no-radial-cross-loss" in a4

    # A5 turns radial/cross loss ON -> full feature set.
    a5 = _spec("A5_plus_radial_cross").cli_overrides
    assert "--use-radial-cross-loss" in a5 and "--no-radial-cross-loss" not in a5

    # A6 = recommended control, no overrides.
    assert _spec("A6_full_recommended").cli_overrides == []


def test_matrix_all_includes_optional_encodings():
    args = ram.parse_args(["--train-data", "t.h5", "--val-data", "v.h5", "--matrix", "all", "--dry-run"])
    names = [e["name"] for e in ram.build_matrix(args)]
    for optional in ("A7_physical_radial_decay", "A8_real_sh_basis", "A9_additive_multiband"):
        assert optional in names


def test_dry_run_writes_paper_summary_artifacts(tmp_path):
    out_root = tmp_path / "abl"
    rc = ram.main(
        ["--train-data", "t.h5", "--val-data", "v.h5", "--out-root", str(out_root), "--seed", "3", "--dry-run"]
    )
    assert rc == 0
    assert (out_root / "st_lrps_ablation_summary.csv").exists()
    assert (out_root / "st_lrps_ablation_summary.md").exists()
    md = (out_root / "st_lrps_ablation_summary.md").read_text(encoding="utf-8")
    assert "A6_full_recommended" in md
    assert "monitor-only" in md.lower()


def test_only_selection_and_eval_commands(tmp_path):
    args = ram.parse_args(
        [
            "--train-data", "t.h5", "--val-data", "v.h5",
            "--out-root", str(tmp_path / "abl"),
            "--only", "A2_plus_multiscale",
            "--ood-data", "ood.h5", "--run-eval-after-training", "--dry-run",
        ]
    )
    entries = ram.build_matrix(args)
    assert [e["name"] for e in entries] == ["A2_plus_multiscale"]
    assert entries[0]["eval_commands"], "eval commands should be generated when --run-eval-after-training is set"


def _ablation_entry_with_evals(tmp_path):
    args = ram.parse_args([
        "--run-eval-after-training", "--eval-streaming", "--dry-run",
        "--train-data", "train.h5", "--val-data", "val.h5",
        "--test-data", "test.h5", "--ood-data", "ood.h5",
        "--out-root", str(tmp_path / "abl"),
        "--only", "A0_raw_siren_sobolev",
    ])
    return ram.build_matrix(args)[0]


def test_ablation_eval_commands_are_accepted_by_the_real_eval_parser(monkeypatch, tmp_path):
    """Regression (ported from Sidera 4a8784c1): the launcher passed --out-dir,
    which the evaluator never accepted (argparse exit 2 under --run-eval-after-training)."""
    import sys

    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.evaluation.cli import parse_args as eval_parse_args

    entry = _ablation_entry_with_evals(tmp_path)
    assert len(entry["eval_commands"]) == 2
    for cmd, split, data in zip(
        entry["eval_commands"], ("test", "ood_high"), ("test.h5", "ood.h5"), strict=True
    ):
        assert cmd[1] == "-m" and cmd[2] == ram._EVALUATE_MODULE
        monkeypatch.setattr(sys, "argv", ["lunaris-eval", *cmd[3:]])
        ns = eval_parse_args()
        assert ns.data == data
        assert ns.streaming is True
        assert ns.model_dir == entry["out_dir"]
        assert ns.out.replace("\\", "/").endswith(f"evals/{split}")


def test_ablation_train_command_is_accepted_by_the_real_training_parser(monkeypatch, tmp_path):
    import sys

    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.training.config import parse_args as train_parse_args

    entry = _ablation_entry_with_evals(tmp_path)
    cmd = entry["command"]
    assert cmd[1] == "-m" and cmd[2] == ram._TRAIN_MODULE
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *cmd[3:]])
    cfg = train_parse_args()
    assert str(cfg.out) == entry["out_dir"]
    assert cfg.train_data == "train.h5" and cfg.val_data == "val.h5"


def test_ablation_summary_reads_the_keys_the_trainer_writes(tmp_path):
    """param_count / training_time_s come from run_manifest compute_accounting;
    the OOD column names the split it was read from."""
    import json

    entry = _ablation_entry_with_evals(tmp_path)
    run_dir = tmp_path / "abl" / entry["name"]
    (run_dir / "evals" / "ood_high").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "status": "completed",
        "compute_accounting": {"n_params": 1234, "hardware": {"wall_clock_seconds": 56.5}},
    }), encoding="utf-8")
    (run_dir / "evals" / "ood_high" / "summary_metrics.json").write_text(
        json.dumps([{"rmse_a_vec": 2.5e-5}]), encoding="utf-8"
    )
    row = ram._ablation_summary_row(entry)
    assert row["param_count"] == 1234
    assert row["training_time_s"] == 56.5
    assert row["ood_high_rmse_a"] == 2.5e-5
    assert row["ood_rmse_a"] == 2.5e-5 and row["ood_rmse_a_split"] == "ood_high"

    md_path = tmp_path / "summary.md"
    ram._write_summary_md(md_path, [row])
    md = md_path.read_text(encoding="utf-8")
    assert "final val base loss" in md
    assert "2.5e-05 (ood_high)" in md
    assert "| 1234 | 56.5 |" in md
