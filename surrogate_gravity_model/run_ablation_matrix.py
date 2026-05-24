#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_matrix.py - Generate (and optionally execute) the standard ST-LRPS
ablation matrix for the lunar residual gravity surrogate.

Each ablation is a single ``st_lrps_train.py`` run with a fixed set of
architecture / loss flags layered on top of the shared dataset + seed. The
script writes:

* ``<out_root>/ablation_commands.txt`` - one shell command per ablation
* ``<out_root>/ablation_manifest.json`` - compact machine-readable description
* ``<out_root>/<ablation_name>/`` - per-run output directory (created lazily)

By default it is a DRY RUN (commands are written but not executed). Pass
``--execute`` to actually launch the runs sequentially. Existing run
directories are skipped unless ``--overwrite`` is given.

Usage
-----
    python run_ablation_matrix.py \
        --train-data suite/train_hybrid.h5 \
        --val-data   suite/val_uniform.h5 \
        --out-root   ablation_runs \
        --dry-run

    python run_ablation_matrix.py --train-data train.h5 --val-data val.h5 \
        --out-root ablation_runs --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# The trainer entry point lives next to this script.
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _SCRIPT_DIR / "st_lrps_train.py"

# Small, safe collocation Laplacian weight for the laplacian_train ablation.
_LAPLACIAN_TRAIN_WEIGHT = "1e-12"

# Ordered list of (name, description, extra-flags). Flags are appended verbatim
# to the base command, so they override the (now production) defaults.
ABLATIONS: List[Dict[str, object]] = [
    {
        "name": "plain_siren",
        "description": "Single-scale SIREN, no residual blocks, no balanced/radial-cross losses.",
        "flags": ["--no-residual-blocks", "--n-bands", "1",
                  "--no-altitude-balanced-loss", "--no-radial-cross-loss"],
    },
    {
        "name": "residual_siren",
        "description": "Residual SIREN blocks, single scale (n_bands=1).",
        "flags": ["--use-residual-blocks", "--n-bands", "1"],
    },
    {
        "name": "multiscale_siren_3band",
        "description": "Residual blocks + 3-band multi-scale SIREN (recommended default).",
        "flags": ["--use-residual-blocks", "--n-bands", "3"],
    },
    {
        "name": "multiscale_siren_5band",
        "description": "Residual blocks + 5-band multi-scale SIREN.",
        "flags": ["--use-residual-blocks", "--n-bands", "5"],
    },
    {
        "name": "radial_encoding",
        "description": "Radial separation encoding [r, ux, uy, uz] + raw, residual blocks, 3 bands.",
        "flags": ["--use-radial-separation", "--radial-append-raw",
                  "--use-residual-blocks", "--n-bands", "3"],
    },
    {
        "name": "sh_encoding",
        "description": "SH-inspired angular encoding (degree 4), residual blocks, 3 bands.",
        "flags": ["--use-sh-encoding", "--sh-encoding-degree", "4",
                  "--use-residual-blocks", "--n-bands", "3"],
    },
    {
        "name": "no_direction_loss",
        "description": "Production defaults but with the direction (cosine) loss disabled.",
        "flags": ["--direction-loss-weight", "0.0"],
    },
    {
        "name": "no_altitude_balance",
        "description": "Production defaults but without altitude-balanced loss.",
        "flags": ["--no-altitude-balanced-loss"],
    },
    {
        "name": "no_radial_cross",
        "description": "Production defaults but without radial/cross-radial penalties.",
        "flags": ["--no-radial-cross-loss"],
    },
    {
        "name": "laplacian_train",
        "description": "Production defaults + trainable collocation Laplacian regulariser.",
        "flags": ["--laplacian-mode", "train",
                  "--collocation-laplacian-weight", _LAPLACIAN_TRAIN_WEIGHT],
    },
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate / run the standard ST-LRPS ablation matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train-data", default=None, help="Path to the training HDF5 cloud.")
    ap.add_argument("--val-data", default=None, help="Path to the validation HDF5 cloud (independent val).")
    ap.add_argument("--test-data", default=None, help="Optional test HDF5 cloud (stored in each run config).")
    ap.add_argument("--ood-data", default=None, help="Optional OOD HDF5 cloud (stored in each run config).")
    ap.add_argument("--suite-manifest", default=None, help="Optional dataset suite manifest.json for provenance.")
    ap.add_argument("--out-root", default="ablation_runs", help="Root directory for ablation outputs.")
    ap.add_argument("--seed", type=int, default=42, help="Seed shared by every ablation run.")
    ap.add_argument("--epochs", type=int, default=None, help="Optional --epochs override forwarded to every run.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Restrict to these ablation names (default: all).")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Re-run / overwrite ablations whose output directory already exists.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", dest="execute", action="store_false",
                     help="Only write commands + manifest; do not launch anything (default).")
    grp.add_argument("--execute", dest="execute", action="store_true",
                     help="Launch each ablation run sequentially via subprocess.")
    ap.set_defaults(execute=False)
    return ap.parse_args(argv)


def _data_flags(args: argparse.Namespace) -> List[str]:
    """Build the dataset-selection flags shared by every ablation command."""
    flags: List[str] = []
    if args.train_data and args.val_data:
        flags += ["--train-data", str(args.train_data), "--val-data", str(args.val_data)]
    elif args.train_data:
        # No independent validation set: feed the trainer's internal split path.
        flags += ["--data", str(args.train_data)]
    if args.test_data:
        flags += ["--test-data", str(args.test_data)]
    if args.ood_data:
        flags += ["--ood-data", str(args.ood_data)]
    if args.suite_manifest:
        flags += ["--suite-manifest", str(args.suite_manifest)]
    return flags


def build_matrix(args: argparse.Namespace) -> List[Dict[str, object]]:
    """Return a list of resolved ablation entries (name, out_dir, flags, command)."""
    out_root = Path(args.out_root)
    base_data = _data_flags(args)
    selected = set(args.only) if args.only else None

    entries: List[Dict[str, object]] = []
    for ab in ABLATIONS:
        name = str(ab["name"])
        if selected is not None and name not in selected:
            continue
        run_dir = out_root / name
        cmd: List[str] = [sys.executable, str(_TRAIN_SCRIPT)]
        cmd += base_data
        cmd += ["--out", str(run_dir), "--seed", str(int(args.seed))]
        if args.epochs is not None:
            cmd += ["--epochs", str(int(args.epochs))]
        cmd += [str(f) for f in ab["flags"]]
        entries.append({
            "name": name,
            "description": ab["description"],
            "out_dir": str(run_dir),
            "seed": int(args.seed),
            "flags": [str(f) for f in ab["flags"]],
            "command": cmd,
        })
    return entries


def _command_to_str(cmd: List[str]) -> str:
    """Render a command list as a copy-pasteable single line."""
    parts = []
    for tok in cmd:
        parts.append(f'"{tok}"' if (" " in tok) else tok)
    return " ".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.train_data:
        print("[ablation] WARNING: no --train-data provided. Commands will be written but "
              "are not runnable until you fill in a dataset.", file=sys.stderr)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    entries = build_matrix(args)
    if not entries:
        print("[ablation] No ablations selected (check --only).", file=sys.stderr)
        return 1

    # Write the commands file and the JSON manifest (always, even on dry-run).
    commands_path = out_root / "ablation_commands.txt"
    manifest_path = out_root / "ablation_manifest.json"

    with commands_path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(f"# {e['name']}: {e['description']}\n")
            fh.write(_command_to_str(e["command"]) + "\n\n")

    manifest = {
        "schema_version": "st_lrps_ablation_matrix_v1",
        "seed": int(args.seed),
        "out_root": str(out_root),
        "execute": bool(args.execute),
        "overwrite": bool(args.overwrite),
        "data": {
            "train_data": args.train_data,
            "val_data": args.val_data,
            "test_data": args.test_data,
            "ood_data": args.ood_data,
            "suite_manifest": args.suite_manifest,
        },
        "ablations": [
            {k: e[k] for k in ("name", "description", "out_dir", "seed", "flags", "command")}
            for e in entries
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[ablation] {len(entries)} ablation(s) prepared.")
    print(f"[ablation] commands -> {commands_path}")
    print(f"[ablation] manifest -> {manifest_path}")

    if not args.execute:
        print("[ablation] DRY RUN: nothing launched. Re-run with --execute to train.")
        for e in entries:
            print(f"  - {e['name']}: {_command_to_str(e['command'])}")
        return 0

    # Execute each ablation sequentially.
    failures = 0
    for e in entries:
        run_dir = Path(str(e["out_dir"]))
        if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
            print(f"[ablation] SKIP {e['name']}: {run_dir} already exists (use --overwrite to re-run).")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ablation] RUN  {e['name']} -> {run_dir}")
        result = subprocess.run(e["command"])
        if result.returncode != 0:
            failures += 1
            print(f"[ablation] FAILED {e['name']} (exit {result.returncode}).", file=sys.stderr)

    if failures:
        print(f"[ablation] Completed with {failures} failure(s).", file=sys.stderr)
        return 2
    print("[ablation] All requested ablations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
