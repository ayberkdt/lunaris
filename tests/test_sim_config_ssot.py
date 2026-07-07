"""Repository contracts for the ``SimConfig`` single source of truth."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.cli.common_args import apply_args_to_config
from lunaris.cli.options import parse_args
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


def test_cli_enable_force_flags_create_required_model_configs() -> None:
    cfg = SimConfig(
        gravity=GravityConfig(file_path="gravity.tab"),
        spice=SimpleNamespace(include_third_body=True, kernels=()),
        initial_state=InitialState(x=1.0, y=0.0, z=0.0, vx=0.0, vy=1.0, vz=0.0),
    )
    args = parse_args(["--enable-srp", "on", "--enable-earth-j2", "on"])

    updated = apply_args_to_config(cfg, args)

    assert updated.flags.enable_srp is True
    assert updated.srp is not None
    assert updated.flags.enable_earth_j2 is True
    assert updated.earth_j2 is not None


def test_albedo_root_selects_grid_mode_without_implicitly_enabling_force(tmp_path: Path) -> None:
    cfg = SimConfig(
        gravity=GravityConfig(file_path="gravity.tab"),
        spice=SimpleNamespace(include_third_body=True, kernels=()),
        initial_state=InitialState(x=1.0, y=0.0, z=0.0, vx=0.0, vy=1.0, vz=0.0),
    )
    args = parse_args(["--albedo-root", str(tmp_path)])

    updated = apply_args_to_config(cfg, args)

    assert updated.albedo is not None
    assert updated.albedo.albedo_mode == "scaled_dn_grid"
    assert updated.flags.enable_albedo is False
