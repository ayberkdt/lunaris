# -*- coding: utf-8 -*-
"""
Regression tests for the top-level entry/config layer cleanup.

Covers lunaris.core.config / launchers and the shared lunaris.cli.common_args
package. These guard the public CLI/config contract:

- importing the entrypoints must NOT eagerly load the default configuration,
- no stale LunarSim project identity remains,
- the duplicate COE Keplerian fallback is gone,
- ST-LRPS model-dir validation goes through the canonical helper,
- shared CLI helpers live in the neutral cli.common_args module.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_FILES = (
    "src/lunaris/core/config.py",
    "src/lunaris/cli/main.py",
    "src/lunaris/core/mc_runner.py",
)
STALE_NAMES = ("LUNAR_SIMULATION", "LUNAR SIMULATION", "LunarSim")


def _read_source(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _load_module_by_path(unique_name: str, file_name: str):
    """Execute a source file under a throwaway module name.

    Lets us re-run module-level import code (to prove it does not eagerly load
    config) without clobbering the canonical entry in ``sys.modules``.
    """
    spec = importlib.util.spec_from_file_location(unique_name, REPO_ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1) Import safety
# ---------------------------------------------------------------------------

def test_importing_config_does_not_warn_or_load(tmp_path: Path) -> None:
    """Importing config must be side-effect free (no asset discovery / warning)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import lunaris.core.config as config; print('OK')"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "[CONFIG WARNING]" not in proc.stderr
    assert "[CONFIG FATAL]" not in proc.stderr
    # No traceback leaked from an eager load attempt.
    assert "Traceback" not in proc.stderr


def test_config_has_no_module_level_default_instance() -> None:
    import lunaris.core.config as config

    assert not hasattr(config, "cfg"), "config must not expose an eager default instance"


def test_entrypoints_do_not_call_load_default_config_at_import(monkeypatch) -> None:
    """Sabotage load_default_config; importing the entrypoints must still work."""
    import lunaris.core.config as config

    def _boom(*args, **kwargs):
        raise RuntimeError("load_default_config must not run at import time")

    monkeypatch.setattr(config, "load_default_config", _boom)

    # Re-execute the module bodies under throwaway names so the global
    # sys.modules entries stay clean for the rest of the suite.
    main_mod = _load_module_by_path("_fresh_main_import", "src/lunaris/cli/main.py")
    mc_mod = _load_module_by_path("_fresh_mc_runner_import", "src/lunaris/core/mc_runner.py")

    assert main_mod is not None
    assert mc_mod is not None


# ---------------------------------------------------------------------------
# 2) Config public API
# ---------------------------------------------------------------------------

def test_config_all_names_exist_and_cfg_not_exported() -> None:
    import lunaris.core.config as config

    for name in config.__all__:
        assert hasattr(config, name), f"config.__all__ lists missing name: {name}"
    assert "cfg" not in config.__all__
    assert "get_default_config" in config.__all__
    assert callable(config.get_default_config)


# ---------------------------------------------------------------------------
# 3) No stale project identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("file_name", ENTRY_FILES)
def test_no_stale_lunar_names(file_name: str) -> None:
    source = _read_source(file_name)
    for stale in STALE_NAMES:
        assert stale not in source, f"{file_name} still contains stale name {stale!r}"


# ---------------------------------------------------------------------------
# 4) No duplicate / private COE fallback
# ---------------------------------------------------------------------------

def test_no_keplerian_fallback_in_main() -> None:
    import lunaris.cli.main as main

    assert not hasattr(main, "_initial_state_from_keplerian_fallback")


def test_mc_runner_does_not_depend_on_main_private_helpers() -> None:
    source = _read_source("src/lunaris/core/mc_runner.py")
    assert "from lunaris.cli.main import" not in source
    assert "_initial_state_from_keplerian_fallback" not in source


def test_no_duplicate_st_lrps_artifact_checks_in_main() -> None:
    """Artifact checks must be delegated, not reimplemented in main.py."""
    source = _read_source("src/lunaris/cli/main.py")
    # Canonical helper must be referenced.
    assert "validate_st_lrps_model_dir" in source
    # Old manual artifact-validation logic / messages must be gone.
    assert '"config.json").is_file()' not in source
    assert 'ckpt_best.pt").is_file()' not in source
    assert "is missing config.json" not in source


# ---------------------------------------------------------------------------
# 5) Canonical ST-LRPS model-dir validation
# ---------------------------------------------------------------------------

def test_validate_args_uses_canonical_st_lrps_helper(monkeypatch, tmp_path: Path) -> None:
    import lunaris.cli.main as main

    model_dir = tmp_path / "run"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    calls: list[str] = []

    def _spy(path):
        calls.append(str(path))
        return model_dir

    monkeypatch.setattr("lunaris.common.montecarlo_defs.validate_st_lrps_model_dir", _spy)
    monkeypatch.setattr(
        "lunaris.surrogate.st_lrps.data.dataset_parameters.looks_like_lunar_run_config",
        lambda cfg: True,
    )

    args = main.parse_args(["--surrogate-gravity-model-dir", str(model_dir)])

    assert calls == [str(model_dir)], "validate_args must delegate to the canonical helper"
    assert args.surrogate_gravity_model_dir == str(model_dir)


def test_validate_args_accepts_ckpt_last_via_canonical_helper(monkeypatch, tmp_path: Path) -> None:
    """A run with only ckpt_last.pt (no ckpt_best.pt) must be accepted."""
    import lunaris.cli.main as main

    model_dir = tmp_path / "run"
    (model_dir / "checkpoints").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"")

    monkeypatch.setattr(
        "lunaris.surrogate.st_lrps.data.dataset_parameters.looks_like_lunar_run_config",
        lambda cfg: True,
    )

    # Uses the real validate_st_lrps_model_dir; must not raise SystemExit.
    args = main.parse_args(["--surrogate-gravity-model-dir", str(model_dir)])
    assert args.surrogate_gravity_model_dir == str(model_dir)


def test_validate_args_rejects_invalid_st_lrps_dir(tmp_path: Path) -> None:
    import lunaris.cli.main as main

    missing = tmp_path / "does_not_exist"
    with pytest.raises(SystemExit):
        main.parse_args(["--surrogate-gravity-model-dir", str(missing)])


# ---------------------------------------------------------------------------
# 6) Shared CLI helpers live in cli.common_args
# ---------------------------------------------------------------------------

def test_entrypoints_import_shared_helpers_from_cli() -> None:
    import lunaris.cli.common_args as ca
    import lunaris.cli.main as main
    import lunaris.core.mc_runner as mc_runner

    # main re-exports the moved helpers from the neutral module (identity check).
    assert main.str2bool is ca.str2bool
    assert main.parse_adaptive_table is ca.parse_adaptive_table
    assert main.resolve_orbit_elements is ca.resolve_orbit_elements
    assert main.init_surface_provider is ca.init_surface_provider
    assert main.need_ephemeris is ca.need_ephemeris
    assert main.apply_args_to_config is ca.apply_args_to_config

    # mc_runner pulls the same shared helpers (not from main).
    assert mc_runner.apply_args_to_config is ca.apply_args_to_config
    assert mc_runner.resolve_orbit_elements is ca.resolve_orbit_elements
    assert mc_runner.init_surface_provider is ca.init_surface_provider
    assert mc_runner.str2bool is ca.str2bool
    assert mc_runner.parse_adaptive_table is ca.parse_adaptive_table


def test_cli_common_args_is_lightweight() -> None:
    """Importing cli.common_args must not pull in heavy runtime deps."""
    code = (
        "import sys, lunaris.cli.common_args; "
        "heavy = [m for m in ('numba', 'torch', 'spiceypy', 'scipy') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"cli.common_args imported heavy modules: {proc.stdout!r}"
