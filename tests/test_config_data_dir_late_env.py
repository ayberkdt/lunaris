"""A data-dir environment override set after import must still be honoured."""

from __future__ import annotations

from pathlib import Path

import lunaris.core.config as config_mod


def test_late_env_override_is_resolved(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config_mod, "_DATA_DIR_ENV_AT_IMPORT", ("", ""))
    monkeypatch.setenv("LUNARIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("STLRPS_DATA_DIR", raising=False)
    got = config_mod._asset_dir(Path("unused_default"), "gravity_models")
    assert got == tmp_path.resolve() / "gravity_models"


def test_unchanged_env_keeps_the_module_constant(monkeypatch, tmp_path: Path) -> None:
    # Tests and callers replace KERNEL_DIR/GRAV_DIR on purpose; with the env
    # override unchanged since import, that replacement must still win.
    monkeypatch.setattr(config_mod, "_DATA_DIR_ENV_AT_IMPORT", (str(tmp_path), ""))
    monkeypatch.setenv("LUNARIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("STLRPS_DATA_DIR", raising=False)
    patched = tmp_path / "patched_kernels"
    assert config_mod._asset_dir(patched, "ephemeris_models") == patched


def test_no_override_keeps_the_module_constant(monkeypatch) -> None:
    monkeypatch.setattr(config_mod, "_DATA_DIR_ENV_AT_IMPORT", ("", ""))
    monkeypatch.delenv("LUNARIS_DATA_DIR", raising=False)
    monkeypatch.delenv("STLRPS_DATA_DIR", raising=False)
    default = Path("some_default")
    assert config_mod._asset_dir(default, "gravity_models") == default
