#!/usr/bin/env python3
"""Validation tests for the ST-LRPS HPC scenario sweep files and launcher.

These tests never launch training; they only parse the scenario JSONL files and
exercise the launcher's dry-run path.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCENARIO_DIR = _REPO_ROOT / "hpc" / "scenarios"
_LAUNCHER = _REPO_ROOT / "tools" / "hpc" / "run_training_scenario.py"

# Expected line counts per the task contract.
_EXPECTED_COUNTS = {
    "st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl": 7,
    "st_lrps_potential_autograd_capacity_sweep_A6_full.jsonl": 5,
    "st_lrps_potential_autograd_encoding_and_loss_sweep.jsonl": 7,
}

_REQUIRED_FIELDS = ("name", "entrypoint", "description", "runtime_model_kind", "flags")
_KIND_TO_ENTRYPOINT = {
    "potential_autograd": "lunaris-train",
}
_ALLOWED_ENTRYPOINTS = {
    "lunaris-train",
    "lunaris-eval",
    "lunaris-benchmark",
}
# A representative potential_autograd scenario file for launcher-behaviour tests
# that are not specific to any one scenario.
_SAMPLE_SCENARIO_FILE = "st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl"
_OUTPUT_FLAGS = {"--out", "--out-dir", "--output-dir", "--output"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_launcher_module():
    spec = importlib.util.spec_from_file_location("run_training_scenario", _LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_scenario_files():
    files = sorted(_SCENARIO_DIR.glob("*.jsonl"))
    assert files, f"No scenario files found under {_SCENARIO_DIR}"
    return files


def _parse_lines(path: Path):
    scenarios = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path.name}:{line_no}: invalid JSON: {exc}")
        scenarios.append(obj)
    return scenarios


def _all_scenarios():
    out = []
    for path in _iter_scenario_files():
        for obj in _parse_lines(path):
            out.append((path.name, obj))
    return out


# --------------------------------------------------------------------------- #
# File-level contracts
# --------------------------------------------------------------------------- #


def test_launcher_script_exists():
    assert _LAUNCHER.is_file()


def test_every_scenario_file_parses():
    for path in _iter_scenario_files():
        scenarios = _parse_lines(path)
        assert scenarios, f"{path.name} has no scenario lines"


@pytest.mark.parametrize("filename, expected", sorted(_EXPECTED_COUNTS.items()))
def test_expected_line_counts(filename, expected):
    path = _SCENARIO_DIR / filename
    assert path.is_file(), f"missing scenario file {filename}"
    assert len(_parse_lines(path)) == expected


# --------------------------------------------------------------------------- #
# Scenario-level contracts
# --------------------------------------------------------------------------- #


def test_required_fields_present():
    for fname, obj in _all_scenarios():
        for field in _REQUIRED_FIELDS:
            assert field in obj, f"{fname}: scenario {obj.get('name')!r} missing {field!r}"
        assert isinstance(obj["flags"], list)
        assert all(isinstance(tok, str) for tok in obj["flags"])
        assert isinstance(obj["description"], str) and obj["description"].strip()


def test_entrypoints_allowed():
    for fname, obj in _all_scenarios():
        assert obj["entrypoint"] in _ALLOWED_ENTRYPOINTS, (
            f"{fname}: scenario {obj['name']!r} has disallowed entrypoint {obj['entrypoint']!r}"
        )


def test_names_unique_across_all_files():
    names = [obj["name"] for _, obj in _all_scenarios()]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"duplicate scenario names: {duplicates}"


def test_names_are_filesystem_safe():
    module = _load_launcher_module()
    for fname, obj in _all_scenarios():
        assert module.is_safe_name(obj["name"]), (
            f"{fname}: scenario name {obj['name']!r} is not filesystem-safe"
        )


def test_kind_entrypoint_consistency():
    for fname, obj in _all_scenarios():
        kind = obj["runtime_model_kind"]
        expected = _KIND_TO_ENTRYPOINT.get(kind)
        if expected is not None:
            assert obj["entrypoint"] == expected, (
                f"{fname}: scenario {obj['name']!r} kind={kind} must use entrypoint={expected}"
            )


def test_no_force_direct_scenarios_remain():
    # force_direct is archived in experimental/force-direct-archive; no shipped
    # scenario may declare or be tagged with it.
    for fname, obj in _all_scenarios():
        tags = {str(t).lower() for t in obj.get("tags", [])}
        assert obj["runtime_model_kind"] != "force_direct", (
            f"{fname}: scenario {obj['name']!r} declares archived force_direct kind"
        )
        assert "force_direct" not in tags, (
            f"{fname}: scenario {obj['name']!r} carries archived force_direct tag"
        )


def test_no_scenario_writes_under_src():
    for fname, obj in _all_scenarios():
        for tok in obj["flags"]:
            value = tok.split("=", 1)[-1].replace("\\", "/")
            assert not (value == "src" or value.startswith("src/") or "/src/" in value), (
                f"{fname}: scenario {obj['name']!r} flag {tok!r} writes under src/"
            )


def test_no_scenario_flags_set_output_flag():
    # The launcher injects --out; scenarios must not set an output flag.
    for fname, obj in _all_scenarios():
        for tok in obj["flags"]:
            head = tok.split("=", 1)[0]
            assert head not in _OUTPUT_FLAGS, (
                f"{fname}: scenario {obj['name']!r} must not set output flag {head}"
            )


def test_every_scenario_validates_via_launcher():
    module = _load_launcher_module()
    for _fname, obj in _all_scenarios():
        # Strip the loader's bookkeeping key if present.
        scenario = {k: v for k, v in obj.items() if k != "_source_line"}
        module.validate_scenario(scenario)  # must not raise


# --------------------------------------------------------------------------- #
# Launcher dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_prints_command_without_launching(tmp_path):
    scenario_file = _SCENARIO_DIR / "st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(_LAUNCHER),
            str(scenario_file),
            "--index",
            "0",
            "--dry-run",
            "--output-root",
            str(tmp_path / "training"),
            "--train-data",
            "dummy_train.h5",
            "--val-data",
            "dummy_val.h5",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "DRY RUN" in out
    assert "PotentialAutograd_A0RawSirenSobolev" in out
    # Resolved command references the training module and the injected --out.
    assert "lunaris.surrogate.st_lrps.training.cli" in out
    assert "--out" in out
    assert "--train-data" in out and "dummy_train.h5" in out
    # Dry run must not create the run directory.
    assert not (tmp_path / "training" / "PotentialAutograd_A0RawSirenSobolev_1Band_RawXYZ_NoAuxLosses_Seed42").exists()


def test_common_flag_output_override_is_rejected(tmp_path):
    # A forwarded --out must not silently change the launcher-owned output dir.
    scenario_file = _SCENARIO_DIR / _SAMPLE_SCENARIO_FILE
    result = subprocess.run(
        [
            sys.executable,
            str(_LAUNCHER),
            str(scenario_file),
            "--index",
            "0",
            "--dry-run",
            "--out",
            str(tmp_path / "sneaky"),
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "must not set an output flag" in result.stderr


def test_index_out_of_range_fails_cleanly(tmp_path):
    scenario_file = _SCENARIO_DIR / _SAMPLE_SCENARIO_FILE
    result = subprocess.run(
        [
            sys.executable,
            str(_LAUNCHER),
            str(scenario_file),
            "--index",
            "999",
            "--dry-run",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "out of range" in result.stderr
