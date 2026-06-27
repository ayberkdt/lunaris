#!/usr/bin/env python3
"""Reproducible ST-LRPS scenario launcher for Slurm array jobs.

Each *scenario* is one self-describing experiment stored as a single JSON line
in a ``.jsonl`` file under ``hpc/scenarios/``. A Slurm array job selects the
line matching ``$SLURM_ARRAY_TASK_ID`` (or an explicit ``--index``), validates
it, builds an output directory under ``$LUNARIS_OUTPUT_DIR/training/<name>``,
records provenance (``scenario.json``, ``scenario_command.txt``,
``scenario_environment.json``), and launches the matching training/eval CLI.

Design goals
------------
* **Self-describing**: scenario ``name`` doubles as the run-directory name, so
  long explicit names (e.g. ``PotentialAutograd_A6FullRecommended_3Band_RawXYZ_
  DirectionW020_Seed42``) are encouraged.
* **Safe**: the launcher owns the ``--out`` flag, refuses to overwrite an
  existing run unless ``--force``/``--overwrite`` is passed, and never writes
  generated outputs under ``src/``.
* **Reproducible**: the resolved command and a curated environment snapshot are
  written next to the run before anything is launched.

Usage (driven by ``hpc/slurm_train_scenario_array.sbatch``)::

    python tools/hpc/run_training_scenario.py <scenario.jsonl> \
        [--index N] [--dry-run] [--force] [--output-root DIR] \
        -- <common flags forwarded to every scenario's entrypoint>

Any argument the launcher does not recognise (everything after the JSONL path
that is not a launcher option) is treated as a *common flag* and appended to the
entrypoint command, so the same data/epoch flags apply to every array task.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #

# Entry points the launcher is allowed to start, mapped to the module run via
# ``python -m`` (more robust than relying on console scripts being on PATH).
ENTRYPOINT_MODULES: dict[str, str] = {
    "lunaris-train": "lunaris.surrogate.st_lrps.training.cli",
    "lunaris-train-force-direct": "lunaris.surrogate.st_lrps.training.force_direct_cli",
    "lunaris-eval": "lunaris.surrogate.st_lrps.evaluation.cli",
    "lunaris-benchmark": "lunaris.surrogate.st_lrps.evaluation.compare_gravity_models",
}

REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "entrypoint",
    "description",
    "runtime_model_kind",
    "flags",
)

# The launcher injects the output directory itself; scenarios and common flags
# must not also set an output flag, or the run directory would be ambiguous.
OUTPUT_FLAGS: tuple[str, ...] = ("--out", "--out-dir", "--output-dir", "--output")

# Filesystem-safe scenario name: starts alphanumeric, then word/dot/dash chars.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_NAME_LEN = 200

# Expected (runtime_model_kind -> entrypoint) consistency for training kinds.
_KIND_TO_ENTRYPOINT: dict[str, str] = {
    "potential_autograd": "lunaris-train",
    "force_direct": "lunaris-train-force-direct",
}


class ScenarioError(ValueError):
    """Raised when a scenario file or line is malformed or unsafe."""


# --------------------------------------------------------------------------- #
# Loading & validation
# --------------------------------------------------------------------------- #


def is_safe_name(name: Any) -> bool:
    """Return True when ``name`` is a safe single path component."""
    if not isinstance(name, str) or not name or len(name) > _MAX_NAME_LEN:
        return False
    if name in {".", ".."}:
        return False
    return bool(_SAFE_NAME_RE.match(name))


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Parse a scenario JSONL file into a list of scenario dicts.

    Blank lines and ``#`` comment lines are skipped but counted so error
    messages can point at the original file line number.
    """
    if not path.exists():
        raise ScenarioError(f"Scenario file not found: {path}")
    scenarios: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ScenarioError(
                    f"{path}:{line_no}: invalid JSON ({exc.msg})"
                ) from exc
            if not isinstance(obj, dict):
                raise ScenarioError(
                    f"{path}:{line_no}: each scenario line must be a JSON object."
                )
            obj["_source_line"] = line_no
            scenarios.append(obj)
    if not scenarios:
        raise ScenarioError(f"No scenarios found in {path}.")
    return scenarios


def validate_scenario(scenario: Mapping[str, Any]) -> None:
    """Validate a single scenario dict; raise ScenarioError on any problem."""
    where = scenario.get("name", "<unnamed>")
    missing = [f for f in REQUIRED_FIELDS if f not in scenario]
    if missing:
        raise ScenarioError(
            f"Scenario {where!r} is missing required field(s): {', '.join(missing)}."
        )

    name = scenario["name"]
    if not is_safe_name(name):
        raise ScenarioError(
            f"Scenario name {name!r} is not filesystem-safe "
            "(use [A-Za-z0-9._-], no path separators)."
        )

    entrypoint = scenario["entrypoint"]
    if entrypoint not in ENTRYPOINT_MODULES:
        raise ScenarioError(
            f"Scenario {name!r}: entrypoint {entrypoint!r} is not allowed. "
            f"Choose one of: {', '.join(sorted(ENTRYPOINT_MODULES))}."
        )

    if not isinstance(scenario["description"], str) or not scenario["description"].strip():
        raise ScenarioError(f"Scenario {name!r}: 'description' must be a non-empty string.")

    if not isinstance(scenario["runtime_model_kind"], str) or not scenario["runtime_model_kind"].strip():
        raise ScenarioError(f"Scenario {name!r}: 'runtime_model_kind' must be a non-empty string.")

    flags = scenario["flags"]
    if not isinstance(flags, list) or not all(isinstance(tok, str) for tok in flags):
        raise ScenarioError(f"Scenario {name!r}: 'flags' must be a list of strings.")

    _reject_output_flags(name, flags, origin="scenario flags")
    _reject_src_paths(name, flags, origin="scenario flags")

    # Training kinds must use the matching training entrypoint so a force_direct
    # scenario can never silently train the scalar-potential model (or vice versa).
    kind = scenario["runtime_model_kind"]
    expected = _KIND_TO_ENTRYPOINT.get(kind)
    if expected is not None and entrypoint != expected:
        raise ScenarioError(
            f"Scenario {name!r}: runtime_model_kind={kind!r} requires "
            f"entrypoint={expected!r}, but got {entrypoint!r}."
        )

    tags = scenario.get("tags")
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)
    ):
        raise ScenarioError(f"Scenario {name!r}: 'tags' must be a list of strings when present.")


def _reject_output_flags(name: str, flags: Sequence[str], *, origin: str) -> None:
    for tok in flags:
        head = tok.split("=", 1)[0]
        if head in OUTPUT_FLAGS:
            raise ScenarioError(
                f"Scenario {name!r}: {origin} must not set an output flag "
                f"({head}); the launcher injects --out automatically."
            )


def _reject_src_paths(name: str, flags: Sequence[str], *, origin: str) -> None:
    for tok in flags:
        value = tok.split("=", 1)[-1]
        norm = value.replace("\\", "/")
        if norm == "src" or norm.startswith("src/") or "/src/" in norm:
            raise ScenarioError(
                f"Scenario {name!r}: {origin} must not write generated outputs "
                f"under src/ (offending token: {tok!r})."
            )


# --------------------------------------------------------------------------- #
# Command resolution
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    # tools/hpc/run_training_scenario.py -> repo root is two parents up.
    return Path(__file__).resolve().parents[2]


def resolve_output_root(explicit: str | None) -> Path:
    """Resolve the base directory that holds per-scenario run directories."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("LUNARIS_OUTPUT_DIR")
    if env:
        return (Path(env).expanduser() / "training").resolve()
    # Local fallback keeps generated artifacts out of src/ (see CLAUDE.md).
    return (_repo_root() / "outputs" / "training").resolve()


def _assert_not_under_src(run_dir: Path) -> None:
    src_dir = (_repo_root() / "src").resolve()
    try:
        run_dir.resolve().relative_to(src_dir)
    except ValueError:
        return
    raise ScenarioError(
        f"Refusing to write run outputs under src/: {run_dir}. "
        "Set LUNARIS_OUTPUT_DIR or --output-root to a runtime location."
    )


def build_command(
    scenario: Mapping[str, Any],
    run_dir: Path,
    common_flags: Sequence[str],
) -> list[str]:
    """Build the ``python -m <module> ...`` command for a scenario."""
    module = ENTRYPOINT_MODULES[scenario["entrypoint"]]
    cmd: list[str] = [sys.executable, "-m", module]
    cmd += [str(tok) for tok in scenario["flags"]]
    cmd += list(common_flags)
    # Inject the output directory last so the launcher's value is unambiguous.
    cmd += ["--out", str(run_dir)]
    return cmd


def command_to_str(cmd: Sequence[str]) -> str:
    parts = []
    for tok in cmd:
        text = str(tok)
        parts.append(f'"{text}"' if (" " in text) else text)
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def _environment_snapshot(scenario_file: Path, index: int) -> dict[str, Any]:
    """Capture a curated, secret-free snapshot of the launch environment."""
    captured: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(("SLURM_", "LUNARIS_", "STLRPS_", "CUDA_")):
            captured[key] = value
    for key in ("HOSTNAME", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if key in os.environ:
            captured[key] = os.environ[key]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_file": str(scenario_file),
        "scenario_index": index,
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "env": captured,
    }


def _write_provenance(
    run_dir: Path,
    scenario: Mapping[str, Any],
    cmd: Sequence[str],
    scenario_file: Path,
    index: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in scenario.items() if k != "_source_line"}
    payload["resolved_out"] = str(run_dir)
    (run_dir / "scenario.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (run_dir / "scenario_command.txt").write_text(
        command_to_str(cmd) + "\n", encoding="utf-8"
    )
    (run_dir / "scenario_environment.json").write_text(
        json.dumps(_environment_snapshot(scenario_file, index), indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        prog="run_training_scenario.py",
        description="Launch one ST-LRPS scenario from a JSONL sweep file.",
        # Disable prefix abbreviation so a forwarded entrypoint flag such as
        # ``--out`` is never silently matched to ``--output-root``; it must fall
        # through to the common-flags safety check instead.
        allow_abbrev=False,
    )
    ap.add_argument("scenario_file", help="Path to the scenario .jsonl file.")
    ap.add_argument(
        "--index",
        type=int,
        default=None,
        help="Scenario line index (0-based). Defaults to $SLURM_ARRAY_TASK_ID.",
    )
    ap.add_argument(
        "--output-root",
        default=None,
        help="Override the run-directory parent (default: $LUNARIS_OUTPUT_DIR/training).",
    )
    ap.add_argument(
        "--force",
        "--overwrite",
        dest="force",
        action="store_true",
        default=False,
        help="Reuse the output directory even if it already exists.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the resolved command and output directory without launching.",
    )
    # Everything else is forwarded verbatim to the entrypoint as common flags.
    return ap.parse_known_args(argv)


def _resolve_index(args: argparse.Namespace) -> int:
    if args.index is not None:
        return args.index
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None or env == "":
        raise ScenarioError(
            "No scenario index: pass --index N or run inside a Slurm array job "
            "(SLURM_ARRAY_TASK_ID)."
        )
    try:
        return int(env)
    except ValueError as exc:
        raise ScenarioError(f"SLURM_ARRAY_TASK_ID={env!r} is not an integer.") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args, common_flags = parse_args(argv)

    # Strip a lone "--" separator if the caller used one before common flags.
    common_flags = [tok for tok in common_flags if tok != "--"]

    scenario_file = Path(args.scenario_file).expanduser()
    try:
        scenarios = load_scenarios(scenario_file)
        index = _resolve_index(args)
        if not (0 <= index < len(scenarios)):
            raise ScenarioError(
                f"Scenario index {index} out of range: file has {len(scenarios)} "
                f"scenario(s) (valid 0..{len(scenarios) - 1})."
            )
        scenario = scenarios[index]
        validate_scenario(scenario)

        # Common flags share the same safety rules as scenario flags.
        name = scenario["name"]
        _reject_output_flags(name, common_flags, origin="common flags")
        _reject_src_paths(name, common_flags, origin="common flags")

        output_root = resolve_output_root(args.output_root)
        run_dir = output_root / name
        _assert_not_under_src(run_dir)
        cmd = build_command(scenario, run_dir, common_flags)
    except ScenarioError as exc:
        print(f"[scenario] ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"[scenario] file        : {scenario_file}")
    print(f"[scenario] index       : {index} / {len(scenarios) - 1}")
    print(f"[scenario] name        : {name}")
    print(f"[scenario] entrypoint  : {scenario['entrypoint']}")
    print(f"[scenario] kind        : {scenario['runtime_model_kind']}")
    print(f"[scenario] output dir  : {run_dir}")
    print(f"[scenario] command     : {command_to_str(cmd)}")

    if args.dry_run:
        print("[scenario] DRY RUN: nothing launched.")
        return 0

    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        print(
            f"[scenario] ERROR: output directory already exists and is not empty: "
            f"{run_dir}\n[scenario] Pass --force/--overwrite to reuse it.",
            file=sys.stderr,
        )
        return 3

    _write_provenance(run_dir, scenario, cmd, scenario_file, index)
    print(f"[scenario] provenance written -> {run_dir}")
    print(f"[scenario] launching {scenario['entrypoint']} ...")
    result = subprocess.run(cmd, cwd=str(_repo_root()))
    if result.returncode != 0:
        print(
            f"[scenario] FAILED {name} (exit {result.returncode}).",
            file=sys.stderr,
        )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
