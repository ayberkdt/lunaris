#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Practical ST-LRPS ablation launcher and aggregator."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
# Repo root holds the importable ``st_lrps`` package; subprocesses are launched
# from here with ``python -m`` so module resolution does not depend on CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_MODULE = "st_lrps.training.cli"
_EVALUATE_MODULE = "st_lrps.evaluation.cli"


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    cli_overrides: List[str]
    expected_purpose: str
    experimental: bool = False
    include_in_default_matrix: bool = True


ABLATION_REGISTRY: List[AblationSpec] = [
    AblationSpec(
        name="baseline_single_siren",
        description="Single-scale SIREN baseline without residual blocks or auxiliary balancing losses.",
        cli_overrides=["--no-residual-blocks", "--n-bands", "1", "--no-altitude-balanced-loss", "--no-radial-cross-loss"],
        expected_purpose="Establish the simplest scalar-potential SIREN baseline.",
    ),
    AblationSpec(
        name="multiscale_siren",
        description="Current multi-scale SIREN default with residual blocks.",
        cli_overrides=["--use-residual-blocks", "--n-bands", "3"],
        expected_purpose="Reference production architecture.",
    ),
    AblationSpec(
        name="multiscale_no_resblocks",
        description="Three-band multi-scale SIREN without residual blocks.",
        cli_overrides=["--no-residual-blocks", "--n-bands", "3"],
        expected_purpose="Measure the contribution of residual SIREN blocks.",
    ),
    AblationSpec(
        name="multiscale_no_direction",
        description="Production multi-scale SIREN with direction loss disabled.",
        cli_overrides=["--direction-loss-weight", "0.0"],
        expected_purpose="Measure the contribution of the angular acceleration objective.",
    ),
    AblationSpec(
        name="multiscale_no_altitude_balance",
        description="Production multi-scale SIREN without altitude-balanced loss.",
        cli_overrides=["--no-altitude-balanced-loss"],
        expected_purpose="Measure the contribution of altitude-balanced residual weighting.",
    ),
    AblationSpec(
        name="multiscale_no_radial_cross",
        description="Production multi-scale SIREN without radial/cross-radial penalties.",
        cli_overrides=["--no-radial-cross-loss"],
        expected_purpose="Measure the contribution of radial/cross-radial loss decomposition.",
    ),
    AblationSpec(
        name="radial_decay_encoding",
        description="Scaled inverse-radius decay features inspired by R/r radial decay.",
        cli_overrides=["--use-radial-decay-encoding", "--radial-decay-max-power", "4", "--radial-decay-append-raw", "--use-residual-blocks", "--n-bands", "3"],
        expected_purpose="Test the experimental scaled inverse-radius input encoding.",
        experimental=True,
    ),
    AblationSpec(
        name="real_sh_basis_encoding_optional",
        description="Torch-native real spherical-harmonic basis encoding.",
        cli_overrides=["--use-real-sh-basis", "--real-sh-degree", "4", "--real-sh-append-raw", "--real-sh-include-radial", "--use-residual-blocks", "--n-bands", "3"],
        expected_purpose="Test the experimental angular SH basis encoding.",
        experimental=True,
    ),
    AblationSpec(
        name="additive_multiband",
        description="Additive multi-band SIREN with per-band trunks summed.",
        cli_overrides=["--multiscale-mode", "additive", "--use-residual-blocks", "--n-bands", "3"],
        expected_purpose="Test the experimental additive multi-band composition.",
        experimental=True,
    ),
    AblationSpec(
        name="direct_accel_baseline_optional_only_if_easy",
        description="Placeholder for a direct-acceleration baseline; not part of default matrix.",
        cli_overrides=[],
        expected_purpose="Reserved for a future non-ST-LRPS baseline; omitted to preserve scalar potential design.",
        experimental=True,
        include_in_default_matrix=False,
    ),
]

# Backward-compatible list-of-dicts shape used by older tests/callers.
ABLATIONS: List[Dict[str, Any]] = [
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
    ap.add_argument("--seed", type=int, default=42, help="Seed shared by every ablation run.")
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
    return ap.parse_args(argv)


def _data_flags(args: argparse.Namespace) -> List[str]:
    flags: List[str] = []
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


def _selected_specs(args: argparse.Namespace) -> List[AblationSpec]:
    selected = set(args.only or [])
    specs = [
        spec for spec in ABLATION_REGISTRY
        if (args.matrix == "all" or spec.include_in_default_matrix)
    ]
    if selected:
        known = {spec.name for spec in ABLATION_REGISTRY}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Unknown ablation name(s): {', '.join(unknown)}")
        specs = [spec for spec in specs if spec.name in selected]
    return specs


def build_matrix(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_root = Path(args.out_root)
    base_data = _data_flags(args)
    entries: List[Dict[str, Any]] = []
    for spec in _selected_specs(args):
        run_dir = out_root / spec.name
        cmd: List[str] = [sys.executable, "-m", _TRAIN_MODULE]
        cmd += base_data
        cmd += ["--out", str(run_dir), "--seed", str(int(args.seed))]
        if args.epochs is not None:
            cmd += ["--epochs", str(int(args.epochs))]
        cmd += list(spec.cli_overrides)
        eval_cmds = []
        if args.run_eval_after_training:
            base_eval = [sys.executable, "-m", _EVALUATE_MODULE, "--model-dir", str(run_dir)]
            if args.eval_streaming:
                base_eval.append("--streaming")
            if args.test_data:
                eval_cmds.append(base_eval + ["--data", str(args.test_data), "--out-dir", str(run_dir / "evals" / "test")])
            if args.ood_data:
                eval_cmds.append(base_eval + ["--data", str(args.ood_data), "--out-dir", str(run_dir / "evals" / "ood_high")])

        entry = {
            **asdict(spec),
            "flags": list(spec.cli_overrides),
            "overrides": list(spec.cli_overrides),
            "out_dir": str(run_dir),
            "seed": int(args.seed),
            "command": cmd,
            "eval_commands": eval_cmds,
        }
        entries.append(entry)
    return entries


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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _last_history_row(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "history.jsonl"
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def _ablation_summary_row(entry: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = Path(str(entry["out_dir"]))
    manifest = _read_json(run_dir / "run_manifest.json")
    config = _read_json(run_dir / "config.json")
    hist = _last_history_row(run_dir)
    eval_root = run_dir / "evals"
    row = {
        "name": entry["name"],
        "description": entry["description"],
        "expected_purpose": entry.get("expected_purpose"),
        "experimental": entry.get("experimental"),
        "included_in_default_matrix": entry.get("include_in_default_matrix"),
        "out_dir": str(run_dir),
        "status": manifest.get("status", "missing"),
        "trained_run": str(run_dir) if run_dir.exists() else None,
        "overrides": " ".join(str(x) for x in entry.get("overrides", entry.get("flags", []))),
        "best_checkpoint_score": manifest.get("best_score", config.get("best_score", hist.get("best_score"))),
        "best_epoch": manifest.get("best_epoch", config.get("best_epoch", hist.get("best_epoch"))),
        "best_metric": manifest.get("best_metric", config.get("best_metric", hist.get("best_metric"))),
        "final_val_total_loss": hist.get("val_loss_total"),
        "final_val_base_loss": hist.get("val_loss_base"),
        "final_val_loss_dir": hist.get("val_loss_dir"),
        "final_checkpoint_score": hist.get("checkpoint_score"),
        "test_eval_path": str(eval_root / "test") if (eval_root / "test").exists() else None,
        "ood_eval_path": str(eval_root / "ood_high") if (eval_root / "ood_high").exists() else None,
        "test_rmse_a": None,
        "ood_rmse_a": None,
    }
    for split, key in (("test", "test_rmse_a"), ("ood_high", "ood_rmse_a"), ("ood", "ood_rmse_a")):
        summary = _read_json(eval_root / split / "summary_metrics.json")
        if isinstance(summary, list) and summary:
            row[key] = summary[0].get("rmse_a_vec")
    return row


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def aggregate(entries: List[Dict[str, Any]], out_root: Path) -> None:
    rows = [_ablation_summary_row(entry) for entry in entries]
    (out_root / "ablation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(out_root / "ablation_summary.csv", rows)

    def _rank(name: str, key: str) -> None:
        def sort_key(row: Mapping[str, Any]) -> float:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                value = float("inf")
            return value
        _write_csv(out_root / name, sorted(rows, key=sort_key))

    _rank("ablation_ranked_by_val_base_loss.csv", "final_val_base_loss")
    _rank("ablation_ranked_by_test_rmse_a.csv", "test_rmse_a")
    _rank("ablation_ranked_by_ood_rmse_a.csv", "ood_rmse_a")


def main(argv: Optional[List[str]] = None) -> int:
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
            handle.write(f"# {entry['name']}: {entry['description']}\n")
            handle.write(_command_to_str(entry["command"]) + "\n\n")

    manifest = {
        "schema_version": "st_lrps_ablation_matrix_v2",
        "note": (
            "The default matrix compares explicit, named deviations around the "
            "recommended production ST-LRPS scalar-potential configuration."
        ),
        "matrix": args.matrix,
        "seed": int(args.seed),
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

    print(f"[ablation] {len(entries)} ablation(s) prepared.")
    print(f"[ablation] commands -> {commands_path}")
    print(f"[ablation] manifest -> {manifest_path}")

    if not args.execute:
        print("[ablation] DRY RUN: nothing launched.")
        for entry in entries:
            print(f"  - {entry['name']}: {_command_to_str(entry['command'])}")
            for ecmd in entry.get("eval_commands", []):
                print(f"    (eval) -> {_command_to_str(ecmd)}")
        aggregate(entries, out_root)
        return 0

    failures = 0
    for entry in entries:
        run_dir = Path(str(entry["out_dir"]))
        if _run_completed(run_dir) and not args.force:
            print(f"[ablation] SKIP {entry['name']}: completed manifest exists (use --force to rerun).")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ablation_spec.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        print(f"[ablation] RUN  {entry['name']} -> {run_dir}")
        result = subprocess.run(entry["command"], cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            failures += 1
            print(f"[ablation] FAILED {entry['name']} (exit {result.returncode}).", file=sys.stderr)
        else:
            for ecmd in entry.get("eval_commands", []):
                print(f"[ablation] EVAL {entry['name']} -> {ecmd[-1]}")
                eres = subprocess.run(ecmd, cwd=str(_REPO_ROOT))
                if eres.returncode != 0:
                    failures += 1
                    print(f"[ablation] EVAL FAILED {entry['name']} (exit {eres.returncode}).", file=sys.stderr)
        aggregate(entries, out_root)

    aggregate(entries, out_root)
    if failures:
        print(f"[ablation] Completed with {failures} failure(s).", file=sys.stderr)
        return 2
    print("[ablation] All requested ablations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
