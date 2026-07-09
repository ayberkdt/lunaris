"""Task 7 — ST-LRPS A0..A9 ablation suite."""

from __future__ import annotations

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
