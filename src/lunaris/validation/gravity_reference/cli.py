"""Command-line interface for external gravity-reference validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from lunaris.validation.gravity_reference.lunaris_field_runner import run_field_validation
from lunaris.validation.gravity_reference.lunaris_trajectory_runner import (
    run_trajectory_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunaris-validate",
        description="Run Lunaris external gravity-reference validation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    field = sub.add_parser("gravity-field", help="Validate fixed Moon-body field samples.")
    field.add_argument("--manifest", required=True, help="Field reference manifest JSON.")
    field.add_argument("--out", required=True, help="Output directory under outputs/.")

    traj = sub.add_parser("gravity-trajectory", help="Validate a gravity-only trajectory contract.")
    traj.add_argument("--manifest", required=True, help="Trajectory reference manifest JSON.")
    traj.add_argument("--out", required=True, help="Output directory under outputs/.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gravity-field":
        result: dict[str, Any] = run_field_validation(args.manifest, args.out)
    elif args.command == "gravity-trajectory":
        result = run_trajectory_validation(args.manifest, args.out)
    else:
        parser.error(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

