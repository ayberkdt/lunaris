# -*- coding: utf-8 -*-
"""
Regression tests for the UI-layer cleanup (root launcher + lunaris.ui.widgets).

These guard the UI public contract after the ST-LRPS rebrand and policy
consolidation:

- no stale LunarSim / Lunar Mission Studio identity remains,
- ui_commons exposes canonical APP_NAME and project-root env var,
- the command builder targets the canonical CLI with canonical flags,
- surrogate preflight agrees with the canonical runtime validator,
- session persistence writes/upgrades through one migration boundary,
- MainWindow no longer creates legacy widget aliases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

# Qt pages must be importable; force offscreen so widget construction is headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

REPO_ROOT = Path(__file__).resolve().parents[1]
STALE_TOKENS = (
    "LUNAR_SIMULATION",
    "LunarSim",
    "Lunar Mission Studio",
    "LUNARSIM_",
    ".lunarsim_stop",
    "LunarMissionStudio",
    ".lunarmission",
)


def _ui_source_files() -> list[Path]:
    files = [REPO_ROOT / "src" / "lunaris" / "ui" / "app.py"]
    files.extend(sorted((REPO_ROOT / "src" / "lunaris" / "ui" / "widgets").glob("*.py")))
    return files


# ---------------------------------------------------------------------------
# 1) No stale UI names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source_file", _ui_source_files(), ids=lambda p: p.name)
def test_no_stale_ui_names(source_file: Path) -> None:
    text = source_file.read_text(encoding="utf-8")
    for token in STALE_TOKENS:
        assert token not in text, f"{source_file.name} still contains stale token {token!r}"


# ---------------------------------------------------------------------------
# 2) UI commons canonical constants / env var
# ---------------------------------------------------------------------------

def test_app_name_is_canonical() -> None:
    from lunaris.ui.core.ui_commons import APP_NAME

    assert APP_NAME == "ST-LRPS Studio"


def test_project_root_env_var_is_canonical(monkeypatch, tmp_path: Path) -> None:
    from lunaris.ui.core import ui_commons

    monkeypatch.setenv("STLRPS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("LUNARSIM_PROJECT_ROOT", raising=False)

    resolved = ui_commons.find_project_root()
    assert resolved == tmp_path.expanduser().resolve()

    # Source-level guarantee that the legacy variable is no longer consulted.
    src = (REPO_ROOT / "src" / "lunaris" / "ui" / "core" / "ui_commons.py").read_text(encoding="utf-8")
    assert "STLRPS_PROJECT_ROOT" in src
    assert "LUNARSIM_PROJECT_ROOT" not in src


# ---------------------------------------------------------------------------
# 3) Command builder produces canonical CLI commands
# ---------------------------------------------------------------------------

def _orbit() -> dict:
    return {
        "mode": "circular",
        "alt_km": 100.0,
        "inc_deg": 90.0,
        "raan_deg": 0.0,
        "argp_deg": 0.0,
        "ta_deg": 0.0,
    }


def _forces() -> dict:
    return {"gravity": {"enabled": True}, "sun": True, "earth": True}


def _propagation() -> dict:
    return {
        "timeline": {"duration": "1", "unit": "Days", "epoch": ""},
        "integrator": {"method": "DOP853 (Adaptive)", "dt_out": "60", "output_mode": "dt"},
    }


def _gravity_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        backend="classic_sh",
        st_lrps_model_dir="",
        degree=100,
        file_path="",
        adaptive_enabled=False,
        adaptive_table=None,
    )


def _solver_cfg() -> SimpleNamespace:
    return SimpleNamespace(rtol=1e-12, atol=1e-14, max_step=60.0)


def _spacecraft_cfg() -> SimpleNamespace:
    return SimpleNamespace(mass_kg=1000.0, area_m2=5.0, cd=2.2, cr=1.5)


def _output() -> SimpleNamespace:
    return SimpleNamespace(output_dir="results", generate_3d_plots=False, downsample_3d=1)


def _data_files() -> SimpleNamespace:
    return SimpleNamespace(ldem_root="", albedo_root="", kernel_dir="", ldem_ppd=4)


def test_build_command_targets_main_with_canonical_flags() -> None:
    from lunaris.ui.core.command_builder import build_command, build_command_preview

    cmd = build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=_orbit(),
        forces=_forces(),
        propagation=_propagation(),
        output=_output(),
        data_files=_data_files(),
        gravity_cfg=_gravity_cfg(),
        solver_cfg=_solver_cfg(),
        spacecraft_cfg=_spacecraft_cfg(),
    )

    assert cmd[0] == "python"
    assert cmd[1] == "main.py"
    assert "--gravity-backend" in cmd
    assert "--enable-sh" in cmd
    assert "--alt-km" in cmd

    preview = build_command_preview(cmd)
    for token in STALE_TOKENS:
        assert token not in preview


def test_build_mc_command_targets_runner_with_canonical_flags() -> None:
    from lunaris.ui.core.command_builder import build_mc_command, build_command_preview

    cmd = build_mc_command(
        python_executable="python",
        mc_runner_path=Path("mc_runner.py"),
        orbit=_orbit(),
        forces=_forces(),
        propagation=_propagation(),
        mc_data={"n_samples": 16, "gravity_mode_override": "follow_mission"},
        data_files=_data_files(),
        gravity_cfg=_gravity_cfg(),
        solver_cfg=_solver_cfg(),
        spacecraft_cfg=_spacecraft_cfg(),
    )

    assert cmd[1] == "mc_runner.py"
    assert "--gravity-backend" in cmd
    assert "--n-samples" in cmd
    assert "--mc-gravity-mode" in cmd

    preview = build_command_preview(cmd)
    for token in STALE_TOKENS:
        assert token not in preview


# ---------------------------------------------------------------------------
# 4) Surrogate preflight agrees with the canonical runtime validator
# ---------------------------------------------------------------------------

def _make_run(run_dir: Path, *, config: bool, ckpt: str | None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if config:
        (run_dir / "config.json").write_text("{}", encoding="utf-8")
    if ckpt:
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "checkpoints" / ckpt).write_bytes(b"")
    return run_dir


def _runtime_accepts(run_dir: Path) -> bool:
    from lunaris.common.montecarlo_defs import validate_st_lrps_model_dir

    try:
        validate_st_lrps_model_dir(run_dir)
        return True
    except ValueError:
        return False


@pytest.mark.parametrize(
    "config, ckpt, expect_ok",
    [
        (False, None, False),          # (a) missing/empty directory
        (True, None, False),           # (b) config.json but no checkpoint
        (True, "ckpt_best.pt", True),  # (c) config.json + ckpt_best
        (True, "ckpt_last.pt", True),  # (d) config.json + ckpt_last (runtime accepts)
        (False, "ckpt_best.pt", False),  # (e) checkpoint but no config.json (runtime requires it)
    ],
)
def test_surrogate_preflight_matches_runtime(tmp_path, config, ckpt, expect_ok) -> None:
    from lunaris.ui.core.surrogate_artifacts import validate_surrogate_run_preflight

    run_dir = tmp_path / "run"
    if config or ckpt:
        _make_run(run_dir, config=config, ckpt=ckpt)
    # else: leave run_dir non-existent for case (a)

    ok, _summary, _warnings = validate_surrogate_run_preflight(str(run_dir))
    assert ok is expect_ok

    if run_dir.exists():
        # UI verdict must agree with the canonical runtime validator.
        assert ok == _runtime_accepts(run_dir)


def test_surrogate_preflight_ckpt_last_emits_warning(tmp_path) -> None:
    from lunaris.ui.core.surrogate_artifacts import validate_surrogate_run_preflight

    run_dir = _make_run(tmp_path / "run", config=True, ckpt="ckpt_last.pt")
    ok, _summary, warnings = validate_surrogate_run_preflight(str(run_dir))
    assert ok is True
    assert any("ckpt_last" in w for w in warnings)


# ---------------------------------------------------------------------------
# 5) Session persistence: canonical schema + migration boundary
# ---------------------------------------------------------------------------

@dataclass
class _MiniCfg:
    value: int = 1


def test_collect_session_snapshot_writes_canonical_meta() -> None:
    from lunaris.ui.core.session_persistence import collect_session_snapshot

    snapshot = collect_session_snapshot(
        orbit_page=object(),
        propagation_page=object(),
        force_page=object(),
        output_page=object(),
        data_page=object(),
        gravity_cfg=SimpleNamespace(to_dict=lambda: {}),
        albedo_cfg=_MiniCfg(),
        solver_cfg=_MiniCfg(),
        spacecraft_cfg=_MiniCfg(),
        app_version="test-1.0",
    )

    assert snapshot["meta"]["schema_version"] == 2
    assert snapshot["meta"]["app"] == "ST-LRPS Studio"


def test_migrate_session_payload_upgrades_legacy() -> None:
    from lunaris.ui.core.session_persistence import (
        SESSION_APP_NAME,
        SESSION_SCHEMA_VERSION,
        migrate_session_payload,
    )

    warnings: list[str] = []
    legacy = {
        "meta": {"version": "12.0"},
        "gravity_config": {"degree": 50},
        "forces": {},
    }

    migrated = migrate_session_payload(legacy, log_warning=warnings.append)

    assert migrated["meta"]["schema_version"] == SESSION_SCHEMA_VERSION == 2
    assert migrated["meta"]["app"] == SESSION_APP_NAME == "ST-LRPS Studio"
    # Legacy top-level gravity_config folded into forces.gravity exactly once.
    assert migrated["forces"]["gravity"]["config"]["degree"] == 50
    # A migration warning was emitted for the legacy payload.
    assert any("Migrating legacy session" in w for w in warnings)


def test_migrate_session_payload_is_idempotent_for_canonical() -> None:
    from lunaris.ui.core.session_persistence import migrate_session_payload

    warnings: list[str] = []
    canonical = {"meta": {"schema_version": 2, "app": "ST-LRPS Studio"}}
    migrated = migrate_session_payload(canonical, log_warning=warnings.append)

    assert migrated["meta"]["schema_version"] == 2
    # No migration warning for an already-canonical payload.
    assert warnings == []


# ---------------------------------------------------------------------------
# 6) MainWindow no longer exposes legacy widget aliases
# ---------------------------------------------------------------------------

REMOVED_ALIASES = (
    "ent_out_dir",
    "toggle_anim3d",
    "spin_downsample_3d",
    "txt_preview",
    "ent_downsample_3d",
)


def test_mainwindow_has_no_legacy_aliases() -> None:
    from PySide6 import QtWidgets

    import lunaris.ui.app as ui

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        window = ui.MainWindow()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"MainWindow could not be constructed headlessly: {exc}")

    try:
        for alias in REMOVED_ALIASES:
            assert not hasattr(window, alias), f"MainWindow still exposes legacy alias {alias!r}"
    finally:
        window.deleteLater()
    _ = app
