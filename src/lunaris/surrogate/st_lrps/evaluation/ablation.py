#!/usr/bin/env python3
"""Practical ST-LRPS ablation launcher and aggregator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lunaris.common.paths import project_root_from_file

_SCRIPT_DIR = Path(__file__).resolve().parent
# Project root holds editable outputs; subprocesses are launched
# from here with ``python -m`` so module resolution does not depend on CWD.
_REPO_ROOT = project_root_from_file(__file__)
_TRAIN_MODULE = "lunaris.surrogate.st_lrps.training.cli"
_EVALUATE_MODULE = "lunaris.surrogate.st_lrps.evaluation.cli"
# Every arm trains once per seed: one seed cannot separate a component's effect
# from seed-to-seed scatter. All runs share one data split (--split-seed).
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    cli_overrides: list[str]
    expected_purpose: str
    experimental: bool = False
    include_in_default_matrix: bool = True


# A0..A6 cumulative ablation ladder: raw xyz -> shipped model. Each step
# changes exactly ONE component relative to the previous one, and A6 is the
# shipped TrainConfig default (A5 + physical radial-decay encoding).
# tests/test_st_lrps_ablation_ladder.py resolves every step through the real
# training parser and checks this one-component-per-step property; it also
# fails when any other registered arm resolves to the shipped default.
#
# A0..A5 name the raw-xyz preset explicitly. Without it they inherit the default
# recommended_physical_radial_decay encoding, the ladder never measures the
# encoding, and A6 differs from A5 only in a loss weight. The direction weight
# of A4/A5 is the shipped default for the same reason.
#
# A8/A9 are optional encoding variants (not in the default matrix). There is no
# A7: the old A7 (A6 + physical radial decay) was the shipped default itself
# (so was the legacy alias multiscale_siren).
#
# Monitor-only contract: the training engine selects checkpoints purely from the
# validation metric (compute_checkpoint_score); periodic evaluation runs AFTER
# checkpoint selection and never influences it.
_RAW = ("--model-preset", "baseline_raw")
_SHIPPED_DIRECTION_WEIGHT = "0.1"  # TrainConfig.direction_loss_weight (test-pinned)

ABLATION_REGISTRY: list[AblationSpec] = [
    AblationSpec(
        name="A0_raw_siren_sobolev",
        description="Raw-xyz single-scale SIREN with Sobolev U/a loss only (no residual blocks, no auxiliary losses).",
        cli_overrides=[
            *_RAW, "--no-residual-blocks", "--n-bands", "1",
            "--no-altitude-balanced-loss", "--no-radial-cross-loss",
            "--direction-loss-weight", "0.0",
        ],
        expected_purpose="Establish the minimal scalar-potential SIREN + Sobolev baseline.",
    ),
    AblationSpec(
        name="A1_plus_residual_blocks",
        description="A0 + residual SIREN blocks.",
        cli_overrides=[
            *_RAW, "--use-residual-blocks", "--n-bands", "1",
            "--no-altitude-balanced-loss", "--no-radial-cross-loss",
            "--direction-loss-weight", "0.0",
        ],
        expected_purpose="Measure the contribution of residual SIREN blocks.",
    ),
    AblationSpec(
        name="A2_plus_multiscale",
        description="A1 + multi-scale SIREN (n_bands=3).",
        cli_overrides=[
            *_RAW, "--use-residual-blocks", "--n-bands", "3",
            "--no-altitude-balanced-loss", "--no-radial-cross-loss",
            "--direction-loss-weight", "0.0",
        ],
        expected_purpose="Measure the contribution of multi-scale bands.",
    ),
    AblationSpec(
        name="A3_plus_altitude_balanced",
        description="A2 + altitude-balanced loss.",
        cli_overrides=[
            *_RAW, "--use-residual-blocks", "--n-bands", "3",
            "--use-altitude-balanced-loss", "--no-radial-cross-loss",
            "--direction-loss-weight", "0.0",
        ],
        expected_purpose="Measure the contribution of altitude-balanced residual weighting.",
    ),
    AblationSpec(
        name="A4_plus_direction",
        description="A3 + direction (angular acceleration) loss at the shipped weight.",
        cli_overrides=[
            *_RAW, "--use-residual-blocks", "--n-bands", "3",
            "--use-altitude-balanced-loss", "--no-radial-cross-loss",
            "--direction-loss-weight", _SHIPPED_DIRECTION_WEIGHT,
        ],
        expected_purpose="Measure the contribution of the angular acceleration objective.",
    ),
    AblationSpec(
        name="A5_plus_radial_cross",
        description="A4 + radial/cross-radial loss decomposition (shipped losses on raw xyz).",
        cli_overrides=[
            *_RAW, "--use-residual-blocks", "--n-bands", "3",
            "--use-altitude-balanced-loss", "--use-radial-cross-loss",
            "--direction-loss-weight", _SHIPPED_DIRECTION_WEIGHT,
        ],
        expected_purpose="Measure the contribution of the radial/cross-radial loss.",
    ),
    AblationSpec(
        name="A6_full_recommended",
        description=(
            "A5 + physical radial-decay input encoding = the shipped ST-LRPS "
            "(TrainConfig defaults, no overrides)."
        ),
        cli_overrides=[],
        expected_purpose=(
            "Measure the physical radial-decay encoding; reference configuration "
            "the ablations are measured against."
        ),
    ),
    AblationSpec(
        name="A8_real_sh_basis",
        description="A6 with the input encoding replaced by a real spherical-harmonic basis (custom preset).",
        cli_overrides=[
            "--model-preset", "custom", "--use-real-sh-basis", "--real-sh-degree", "4",
            "--real-sh-append-raw", "--real-sh-include-radial", "--use-residual-blocks", "--n-bands", "3",
        ],
        expected_purpose="Optional: real SH angular basis input encoding.",
        experimental=True,
        include_in_default_matrix=False,
    ),
    AblationSpec(
        name="A9_additive_multiband",
        description="A6 with additive multi-band SIREN composition.",
        cli_overrides=["--multiscale-mode", "additive", "--use-residual-blocks", "--n-bands", "3"],
        expected_purpose="Optional: additive multi-band composition.",
        experimental=True,
        include_in_default_matrix=False,
    ),
    # --- Legacy named deviations (kept for backward compat; run via --matrix all
    #     or --only NAME). Not part of the A0..A6 default ladder. The two
    #     encoding variants below are the same configs as A8/A9 under their
    #     original names. ---
    AblationSpec(
        name="baseline_single_siren",
        description="Single-scale SIREN baseline without residual blocks or auxiliary balancing losses.",
        cli_overrides=["--no-residual-blocks", "--n-bands", "1", "--no-altitude-balanced-loss", "--no-radial-cross-loss"],
        expected_purpose="Legacy alias near A0; simplest scalar-potential SIREN baseline.",
        include_in_default_matrix=False,
    ),
    AblationSpec(
        name="multiscale_no_resblocks",
        description="Three-band multi-scale SIREN without residual blocks.",
        cli_overrides=["--no-residual-blocks", "--n-bands", "3"],
        expected_purpose="Legacy deviation: measure residual SIREN blocks.",
        include_in_default_matrix=False,
    ),
    AblationSpec(
        name="multiscale_no_direction",
        description="Production multi-scale SIREN with direction loss disabled.",
        cli_overrides=["--direction-loss-weight", "0.0"],
        expected_purpose="Legacy deviation: measure the angular acceleration objective.",
        include_in_default_matrix=False,
    ),
    AblationSpec(
        name="multiscale_no_altitude_balance",
        description="Production multi-scale SIREN without altitude-balanced loss.",
        cli_overrides=["--no-altitude-balanced-loss"],
        expected_purpose="Legacy deviation: measure altitude-balanced residual weighting.",
        include_in_default_matrix=False,
    ),
    AblationSpec(
        name="multiscale_no_radial_cross",
        description="Production multi-scale SIREN without radial/cross-radial penalties.",
        cli_overrides=["--no-radial-cross-loss"],
        expected_purpose="Legacy deviation: measure radial/cross-radial loss decomposition.",
        include_in_default_matrix=False,
    ),
    AblationSpec(
        name="real_sh_basis_encoding_optional",
        description="A8 (legacy name): torch-native real spherical-harmonic basis encoding.",
        cli_overrides=["--model-preset", "custom", "--use-real-sh-basis", "--real-sh-degree", "4", "--real-sh-append-raw", "--real-sh-include-radial", "--use-residual-blocks", "--n-bands", "3"],
        expected_purpose="Legacy alias of A8_real_sh_basis.",
        experimental=True,
        include_in_default_matrix=False,
    ),
    AblationSpec(
        name="additive_multiband",
        description="A9 (legacy name): additive multi-band SIREN with per-band trunks summed.",
        cli_overrides=["--multiscale-mode", "additive", "--use-residual-blocks", "--n-bands", "3"],
        expected_purpose="Legacy alias of A9_additive_multiband.",
        experimental=True,
        include_in_default_matrix=False,
    ),
    AblationSpec(
        name="radial_decay_encoding",
        description="Scaled inverse-radius decay features inspired by R/r radial decay.",
        cli_overrides=["--model-preset", "custom", "--use-radial-decay-encoding", "--radial-decay-max-power", "4", "--radial-decay-append-raw", "--use-residual-blocks", "--n-bands", "3"],
        expected_purpose="Test the experimental scaled inverse-radius input encoding.",
        experimental=True,
        include_in_default_matrix=False,
    ),
    # NOTE: A direct-acceleration baseline is intentionally NOT registered here.
    # The scalar-potential ablation matrix trains potential_autograd models only.
    # The direct residual-acceleration (force_direct) runtime is archived in the
    # experimental/force-direct-archive branch and is not part of this matrix.
]

# Backward-compatible list-of-dicts shape used by older tests/callers.
ABLATIONS: list[dict[str, Any]] = [
    {
        "name": spec.name,
        "description": spec.description,
        "flags": list(spec.cli_overrides),
        "cli_overrides": list(spec.cli_overrides),
        "expected_purpose": spec.expected_purpose,
        "experimental": bool(spec.experimental),
        "include_in_default_matrix": bool(spec.include_in_default_matrix),
    }
    for spec in ABLATION_REGISTRY
]


# The HPC array file is generated from ABLATION_REGISTRY (tools/hpc/
# generate_ablation_scenarios.py); a test checks the committed file matches.
HPC_SCENARIO_FILE = "st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl"


def hpc_scenario_records(
    seeds: Iterable[int] = DEFAULT_SEEDS, split_seed: int | None = None
) -> list[dict[str, Any]]:
    """Scenario lines for the default A0..A6 ladder, seed-major.

    Seed-major order puts one complete ladder in array indices 0..6, the next
    seed in 7..13, and so on.
    """
    seed_list = [int(seed) for seed in seeds]
    split = int(seed_list[0] if split_seed is None else split_seed)
    ladder = [spec for spec in ABLATION_REGISTRY if spec.include_in_default_matrix]
    return [
        {
            "name": f"PotentialAutograd_{spec.name}_Seed{seed}",
            "entrypoint": "lunaris-train",
            "description": spec.description,
            "runtime_model_kind": "potential_autograd",
            "tags": ["potential_autograd", "ablation", spec.name.split("_", 1)[0], f"seed{seed}"],
            "flags": [*spec.cli_overrides, "--seed", str(seed), "--split-seed", str(split)],
        }
        for seed in seed_list
        for spec in ladder
    ]


def render_hpc_scenarios(seeds: Iterable[int] = DEFAULT_SEEDS, split_seed: int | None = None) -> str:
    records = hpc_scenario_records(seeds, split_seed)
    header = [
        "# ST-LRPS scalar-potential paper ablation A0..A6 (raw xyz -> shipped model, one component per step).",
        "# GENERATED from lunaris.surrogate.st_lrps.evaluation.ablation.ABLATION_REGISTRY;",
        "# do not edit by hand. Regenerate: python tools/hpc/generate_ablation_scenarios.py",
        "# Seed-major: indices 0-6 are one complete ladder; all lines share one --split-seed.",
        f"# Submit: sbatch --array=0-{len(records) - 1} hpc/slurm_train_scenario_array.sbatch <this file> "
        "--train-data ... --val-data ...",
    ]
    return "\n".join(header + [json.dumps(record, ensure_ascii=False) for record in records]) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate, run, and aggregate an ST-LRPS ablation matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train-data", default=None, help="Path to the training HDF5 cloud.")
    ap.add_argument("--val-data", default=None, help="Path to the validation HDF5 cloud.")
    ap.add_argument("--test-data", default=None, help="Optional test HDF5 cloud stored in each run config.")
    ap.add_argument("--ood-data", default=None, help="Optional OOD HDF5 cloud stored in each run config.")
    ap.add_argument("--suite-manifest", default=None, help="Optional dataset suite manifest.json for provenance.")
    ap.add_argument("--out-root", default="ablation_runs", help="Root directory for ablation outputs.")
    ap.add_argument(
        "--seeds", "--seed", dest="seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Training seeds; every ablation arm runs once per seed.",
    )
    ap.add_argument(
        "--split-seed", type=int, default=None,
        help="Data-split seed shared by every run (default: the first --seeds value).",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Optional --epochs override forwarded to every run.")
    ap.add_argument("--matrix", choices=["default", "all"], default="default", help="Ablation matrix to prepare.")
    ap.add_argument("--only", nargs="+", default=None, help="Restrict to these ablation names.")
    ap.add_argument("--force", "--overwrite", dest="force", action="store_true", default=False, help="Re-run ablations even when a completed run manifest exists.")
    ap.add_argument("--run-eval-after-training", action="store_true", default=False, help="Automatically run evaluator on test/ood data after training completes.")
    ap.add_argument("--eval-streaming", action="store_true", default=False, help="Run the evaluator in memory-safe streaming mode.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", dest="execute", action="store_false", help="Only write commands + manifest; do not launch training.")
    grp.add_argument("--execute", dest="execute", action="store_true", help="Launch each ablation run sequentially.")
    ap.set_defaults(execute=False)
    args = ap.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        ap.error(f"--seeds has duplicates: {args.seeds}")
    if args.split_seed is None:
        args.split_seed = int(args.seeds[0])
    return args


def _data_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.train_data and args.val_data:
        flags += ["--train-data", str(args.train_data), "--val-data", str(args.val_data)]
    elif args.train_data:
        flags += ["--data", str(args.train_data)]
    if args.test_data:
        flags += ["--test-data", str(args.test_data)]
    if args.ood_data:
        flags += ["--ood-data", str(args.ood_data)]
    if args.suite_manifest:
        flags += ["--suite-manifest", str(args.suite_manifest)]
    return flags


def _selected_specs(args: argparse.Namespace) -> list[AblationSpec]:
    selected = set(args.only or [])
    if selected:
        # --only selects by name from the FULL registry, regardless of whether a
        # spec is in the default matrix, so any named ablation can be run directly.
        known = {spec.name for spec in ABLATION_REGISTRY}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Unknown ablation name(s): {', '.join(unknown)}")
        return [spec for spec in ABLATION_REGISTRY if spec.name in selected]
    return [
        spec for spec in ABLATION_REGISTRY
        if (args.matrix == "all" or spec.include_in_default_matrix)
    ]


def build_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    base_data = _data_flags(args)
    return [
        _matrix_entry(args, spec, int(seed), out_root / spec.name / f"seed{int(seed)}", base_data)
        for spec in _selected_specs(args)
        for seed in args.seeds
    ]


def _matrix_entry(
    args: argparse.Namespace, spec: AblationSpec, seed: int, run_dir: Path, base_data: list[str]
) -> dict[str, Any]:
    # The training seed alone would also move the split (split_seed defaults to
    # seed), so the arms would differ in their data, not only in their init.
    cmd: list[str] = [sys.executable, "-m", _TRAIN_MODULE]
    cmd += base_data
    cmd += ["--out", str(run_dir), "--seed", str(seed), "--split-seed", str(int(args.split_seed))]
    if args.epochs is not None:
        cmd += ["--epochs", str(int(args.epochs))]
    cmd += list(spec.cli_overrides)
    eval_cmds = []
    if args.run_eval_after_training:
        base_eval = [sys.executable, "-m", _EVALUATE_MODULE, "--model-dir", str(run_dir)]
        if args.eval_streaming:
            base_eval.append("--streaming")
        # The evaluator's output flag is --out (alias --output-dir); it has no --out-dir.
        if args.test_data:
            eval_cmds.append(base_eval + ["--data", str(args.test_data), "--out", str(run_dir / "evals" / "test")])
        if args.ood_data:
            eval_cmds.append(base_eval + ["--data", str(args.ood_data), "--out", str(run_dir / "evals" / "ood_high")])

    return {
        **asdict(spec),
        "run_id": f"{spec.name}_seed{seed}",
        "flags": list(spec.cli_overrides),
        "overrides": list(spec.cli_overrides),
        "out_dir": str(run_dir),
        "seed": seed,
        "split_seed": int(args.split_seed),
        "command": cmd,
        "eval_commands": eval_cmds,
    }


def _command_to_str(cmd: Iterable[str]) -> str:
    parts = []
    for tok in cmd:
        text = str(tok)
        parts.append(f'"{text}"' if (" " in text) else text)
    return " ".join(parts)


def _run_completed(run_dir: Path) -> bool:
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("status", "")).lower() == "completed"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _last_history_row(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "history.jsonl"
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def _history_row_at_epoch(run_dir: Path, epoch_display: Any) -> dict[str, Any]:
    """History row of the 1-based ``epoch_display`` (the best checkpoint's epoch)."""
    path = run_dir / "history.jsonl"
    try:
        target = int(epoch_display)
    except (TypeError, ValueError):
        return {}
    if not path.exists():
        return {}
    found: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("epoch_display") == target:
                found = row  # a resumed run may repeat an epoch; keep the last
    return found


def _ablation_summary_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(entry["out_dir"]))
    manifest = _read_json(run_dir / "run_manifest.json")
    config = _read_json(run_dir / "config.json")
    hist = _last_history_row(run_dir)
    eval_root = run_dir / "evals"
    # The trainer records capacity and cost only under run_manifest.json
    # "compute_accounting" (n_params, hardware.wall_clock_seconds); no producer
    # writes the flat param_count / training_time_s keys.
    accounting = manifest.get("compute_accounting") or {}

    best_epoch = manifest.get("best_epoch", config.get("best_epoch", hist.get("best_epoch")))
    best_hist = _history_row_at_epoch(run_dir, best_epoch)

    row = {
        "name": entry["name"],
        "run_id": entry.get("run_id", entry["name"]),
        "seed": entry.get("seed"),
        "description": entry["description"],
        "expected_purpose": entry.get("expected_purpose"),
        "experimental": entry.get("experimental"),
        "included_in_default_matrix": entry.get("include_in_default_matrix"),
        "out_dir": str(run_dir),
        "status": manifest.get("status", "missing"),
        "trained_run": str(run_dir) if run_dir.exists() else None,
        "overrides": " ".join(str(x) for x in entry.get("overrides", entry.get("flags", []))),
        # Capacity + cost.
        "param_count": accounting.get("n_params"),
        "training_time_s": (accounting.get("hardware") or {}).get("wall_clock_seconds"),
        "inference_runtime_path": str(run_dir / "runtime") if (run_dir / "runtime").exists() else None,
        # Checkpoint selection (validation-only; periodic eval is monitor-only).
        "best_checkpoint_score": manifest.get("best_score", config.get("best_score", hist.get("best_score"))),
        "best_epoch": best_epoch,
        "best_metric": manifest.get("best_metric", config.get("best_metric", hist.get("best_metric"))),
        # Unweighted residual-acceleration RMS [m/s^2] on the validation split at
        # the best epoch. Unlike val_loss_* it does not depend on the arm's loss
        # options (altitude balancing changes how val_loss_u/a are averaged), so
        # it is the validation number that compares arms.
        "val_accel_rmse_m_s2": best_hist.get("val_accel_rmse_m_s2"),
        "final_val_total_loss": hist.get("val_loss_total"),
        "final_val_base_loss": hist.get("val_loss_base"),
        "final_val_loss_dir": hist.get("val_loss_dir"),
        "final_checkpoint_score": hist.get("checkpoint_score"),
        "test_eval_path": str(eval_root / "test") if (eval_root / "test").exists() else None,
        "ood_eval_path": str(eval_root / "ood_high") if (eval_root / "ood_high").exists() else None,
        "test_rmse_a": None,
        # ood_high_rmse_a is the evals/ood_high split (what this suite's eval command
        # writes). ood_rmse_a is the evals/ood split when it has a summary, else the
        # ood_high value; ood_rmse_a_split names the split it came from and
        # ood_eval_path points at that split's directory.
        "ood_high_rmse_a": None,
        "ood_rmse_a": None,
        "ood_rmse_a_split": None,
        # Validation-suite (field + spatial/OOD) accel RMSE per split, when present.
        "validation_suite_path": None,
    }
    for split, key in (("test", "test_rmse_a"), ("ood_high", "ood_high_rmse_a"), ("ood", "ood_rmse_a")):
        summary = _read_json(eval_root / split / "summary_metrics.json")
        if isinstance(summary, list) and summary:
            row[key] = summary[0].get("rmse_a_vec")
    if row["ood_rmse_a"] is not None:
        row["ood_rmse_a_split"] = "ood"
        row["ood_eval_path"] = str(eval_root / "ood")
    elif row["ood_high_rmse_a"] is not None:
        row["ood_rmse_a"] = row["ood_high_rmse_a"]
        row["ood_rmse_a_split"] = "ood_high"
    suite = _read_json(eval_root / "validation_suite.json")
    if isinstance(suite, Mapping) and suite.get("field_validation"):
        row["validation_suite_path"] = str(eval_root / "validation_suite.json")
        for policy, metrics in suite["field_validation"].items():
            if isinstance(metrics, Mapping) and "residual_accel_rmse_m_s2" in metrics:
                row[f"suite_{policy}_accel_rmse"] = metrics.get("residual_accel_rmse_m_s2")
    return row


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


# Per-arm columns aggregated over seeds (mean, sample std, count). All are
# loss-configuration independent, so they compare arms.
_ARM_METRICS: tuple[str, ...] = ("val_accel_rmse_m_s2", "test_rmse_a", "ood_rmse_a", "training_time_s")


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _arm_rows(run_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per ablation arm: mean / sample std / n over its seeds."""
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for row in run_rows:
        by_arm.setdefault(str(row["name"]), []).append(row)
    out: list[dict[str, Any]] = []
    for name, rows in by_arm.items():
        first = rows[0]
        arm: dict[str, Any] = {
            "name": name,
            "description": first.get("description"),
            "overrides": first.get("overrides"),
            "seeds": " ".join(str(r.get("seed")) for r in rows),
            "n_runs": len(rows),
            "n_completed": sum(1 for r in rows if str(r.get("status")) == "completed"),
            "param_count": first.get("param_count"),
            "ood_rmse_a_split": next((r.get("ood_rmse_a_split") for r in rows if r.get("ood_rmse_a_split")), None),
        }
        for key in _ARM_METRICS:
            values = [v for v in (_finite(r.get(key)) for r in rows) if v is not None]
            arm[f"{key}_n"] = len(values)
            arm[f"{key}_mean"] = statistics.fmean(values) if values else None
            arm[f"{key}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        out.append(arm)
    return out


def _md_mean_std(row: Mapping[str, Any], key: str) -> str:
    mean, std, n = row.get(f"{key}_mean"), row.get(f"{key}_std"), row.get(f"{key}_n") or 0
    if mean is None:
        return "-"
    if std is None:
        return f"{_md_num(mean)} (n={n})"
    return f"{_md_num(mean)} ± {_md_num(std)} (n={n})"


def _write_summary_md(path: Path, arm_rows: list[Mapping[str, Any]]) -> None:
    lines = [
        "# ST-LRPS Ablation Summary",
        "",
        "A0..A6 add one component per step, from raw xyz to the shipped ST-LRPS "
        "(`A6_full_recommended`). Each cell is mean ± sample std over the arm's "
        "seeds; all runs share one data split. Checkpoints are selected on the "
        "validation metric only; periodic evaluation is monitor-only and never "
        "influences selection.",
        "",
        "Compare arms on these columns, not on val_loss_*: the altitude-balanced "
        "option changes how the validation losses are averaged.",
        "",
        "| ablation | params | val accel RMSE [m/s²] | test RMSE a [m/s²] | ood RMSE a [m/s²] (split) | "
        "train time [s] | completed |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        "| {name} | {params} | {val} | {test} | {ood} | {tt} | {done}/{n} |".format(
            name=row.get("name"),
            params=_md_num(row.get("param_count")),
            val=_md_mean_std(row, "val_accel_rmse_m_s2"),
            test=_md_mean_std(row, "test_rmse_a"),
            ood=(
                f"{_md_mean_std(row, 'ood_rmse_a')} ({row.get('ood_rmse_a_split')})"
                if row.get("ood_rmse_a_split") else _md_mean_std(row, "ood_rmse_a")
            ),
            tt=_md_mean_std(row, "training_time_s"),
            done=row.get("n_completed"),
            n=row.get("n_runs"),
        )
        for row in arm_rows
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_num(value: Any) -> str:
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{f:.4g}"


def aggregate(entries: list[dict[str, Any]], out_root: Path) -> None:
    rows = [_ablation_summary_row(entry) for entry in entries]
    arms = _arm_rows(rows)
    # Per-run rows (one per arm and seed).
    (out_root / "ablation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(out_root / "ablation_summary.csv", rows)
    # Canonical paper artifacts: one row per arm, aggregated over seeds.
    _write_csv(out_root / "st_lrps_ablation_summary.csv", arms)
    _write_summary_md(out_root / "st_lrps_ablation_summary.md", arms)

    def _rank(name: str, key: str) -> None:
        def sort_key(row: Mapping[str, Any]) -> float:
            value = _finite(row.get(key))
            return float("inf") if value is None else value
        _write_csv(out_root / name, sorted(arms, key=sort_key))

    _rank("ablation_ranked_by_val_accel_rmse.csv", "val_accel_rmse_m_s2_mean")
    _rank("ablation_ranked_by_test_rmse_a.csv", "test_rmse_a_mean")
    _rank("ablation_ranked_by_ood_rmse_a.csv", "ood_rmse_a_mean")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.train_data:
        print("[ablation] WARNING: no --train-data provided; commands may not be runnable.", file=sys.stderr)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        entries = build_matrix(args)
    except ValueError as exc:
        print(f"[ablation] {exc}", file=sys.stderr)
        return 1
    if not entries:
        print("[ablation] No ablations selected.", file=sys.stderr)
        return 1

    commands_path = out_root / "ablation_commands.txt"
    manifest_path = out_root / "ablation_manifest.json"
    with commands_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"# {entry['run_id']}: {entry['description']}\n")
            handle.write(_command_to_str(entry["command"]) + "\n\n")

    manifest = {
        "schema_version": "st_lrps_ablation_matrix_v3",
        "note": (
            "The default matrix is the A0..A6 ladder from raw xyz to the "
            "recommended production ST-LRPS scalar-potential configuration "
            "(A6 = shipped defaults), one component per step; every arm runs "
            "once per seed on one shared data split."
        ),
        "matrix": args.matrix,
        "seeds": [int(seed) for seed in args.seeds],
        "split_seed": int(args.split_seed),
        "out_root": str(out_root),
        "execute": bool(args.execute),
        "force": bool(args.force),
        "data": {
            "train_data": args.train_data,
            "val_data": args.val_data,
            "test_data": args.test_data,
            "ood_data": args.ood_data,
            "suite_manifest": args.suite_manifest,
        },
        "ablations": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_arms = len({entry["name"] for entry in entries})
    print(f"[ablation] {len(entries)} run(s) prepared: {n_arms} arm(s) x {len(args.seeds)} seed(s).")
    print(f"[ablation] commands -> {commands_path}")
    print(f"[ablation] manifest -> {manifest_path}")

    if not args.execute:
        print("[ablation] DRY RUN: nothing launched.")
        for entry in entries:
            print(f"  - {entry['run_id']}: {_command_to_str(entry['command'])}")
            for ecmd in entry.get("eval_commands", []):
                print(f"    (eval) -> {_command_to_str(ecmd)}")
        aggregate(entries, out_root)
        return 0

    failures = 0
    for entry in entries:
        run_dir = Path(str(entry["out_dir"]))
        if _run_completed(run_dir) and not args.force:
            print(f"[ablation] SKIP {entry['run_id']}: completed manifest exists (use --force to rerun).")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ablation_spec.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        print(f"[ablation] RUN  {entry['run_id']} -> {run_dir}")
        result = subprocess.run(entry["command"], cwd=str(_REPO_ROOT), check=False)
        if result.returncode != 0:
            failures += 1
            print(f"[ablation] FAILED {entry['run_id']} (exit {result.returncode}).", file=sys.stderr)
        else:
            for ecmd in entry.get("eval_commands", []):
                print(f"[ablation] EVAL {entry['run_id']} -> {ecmd[-1]}")
                eres = subprocess.run(ecmd, cwd=str(_REPO_ROOT), check=False)
                if eres.returncode != 0:
                    failures += 1
                    print(f"[ablation] EVAL FAILED {entry['run_id']} (exit {eres.returncode}).", file=sys.stderr)
        aggregate(entries, out_root)

    aggregate(entries, out_root)
    if failures:
        print(f"[ablation] Completed with {failures} failure(s).", file=sys.stderr)
        return 2
    print("[ablation] All requested ablations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
