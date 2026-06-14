"""Repository contracts for the ``SimConfig`` single source of truth."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.common.type_defs import GravityConfig, InitialState, PerturbationFlags
from lunaris.core.config import SimConfig, replace_sim_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "lunaris"


def _python_sources() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def test_only_core_config_constructs_sim_config_directly() -> None:
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "SimConfig":
                relative = path.relative_to(REPO_ROOT).as_posix()
                if relative != "src/lunaris/core/config.py":
                    offenders.append(f"{relative}:{node.lineno}")
    assert offenders == [], f"SimConfig must be constructed only by core.config: {offenders}"


def test_full_sim_config_replacements_use_the_validating_helper() -> None:
    offenders: list[str] = []
    for path in _python_sources():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in {
            "src/lunaris/core/config.py",
            "src/lunaris/cli/common_args.py",
        }:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        imports_sim_config = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "lunaris.core.config"
            and any(alias.name == "SimConfig" for alias in node.names)
            for node in ast.walk(tree)
        )
        if not imports_sim_config:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "replace":
                continue
            first = node.args[0]
            if isinstance(first, ast.Name) and first.id in {"cfg", "cfg_base", "sim_cfg"}:
                offenders.append(f"{relative}:{node.lineno}")
    assert offenders == [], (
        "Full SimConfig copies must use replace_sim_config so validation cannot be skipped: "
        f"{offenders}"
    )


def test_replace_sim_config_validates_the_result() -> None:
    cfg = SimConfig(
        gravity=GravityConfig(file_path="gravity.tab"),
        spice=SimpleNamespace(include_third_body=True),
        initial_state=InitialState(x=1.0, y=0.0, z=0.0, vx=0.0, vy=1.0, vz=0.0),
    )

    with pytest.raises(ValueError, match="enable_srp"):
        replace_sim_config(
            cfg,
            flags=PerturbationFlags(enable_srp=True),
        )
