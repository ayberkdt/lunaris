"""Regression tests for CLI asset-path overrides in the config factory.

Guards the bootstrap-ordering bug: on a machine without the default ``data/``
layout, supplying an explicit asset path (``--kernel-dir`` /
``--gravity-file-path`` / a custom data root) must let ``load_default_config``
build a config, instead of failing while resolving the (absent) defaults before
the override could ever apply. The no-override path must still fail fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lunaris.core import config as config_mod
from lunaris.core.config import load_default_config

# Required default asset filenames the factory resolves (first candidate of each).
_KERNEL_FILES = (
    "naif0012.tls",
    "pck00011.tpc",
    "moon_pa_de440_200625.bpc",
    "de440.bsp",
)
_GRAVITY_FILE = "jggrx_1800f_sha.tab"


def _make_data_dir(root: Path) -> Path:
    """Create a data dir with empty placeholder assets under the canonical layout.

    The factory only checks path existence (it does not read the kernels), so
    empty files are sufficient to exercise resolution end-to-end.
    """
    kernels = root / "ephemeris_models"
    gravity = root / "gravity_models"
    kernels.mkdir(parents=True)
    gravity.mkdir(parents=True)
    for name in _KERNEL_FILES:
        (kernels / name).write_bytes(b"")
    (gravity / _GRAVITY_FILE).write_bytes(b"")
    return root


def _break_default_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the module default dirs at a nonexistent location (clean machine)."""
    missing = tmp_path / "no_such_data"
    monkeypatch.setattr(config_mod, "KERNEL_DIR", missing / "ephemeris_models")
    monkeypatch.setattr(config_mod, "GRAV_DIR", missing / "gravity_models")
    monkeypatch.setattr(config_mod, "DATA_DIR", missing)


def test_data_dir_override_builds_when_default_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _break_default_dirs(monkeypatch, tmp_path)
    data_dir = _make_data_dir(tmp_path / "custom_data")

    cfg = load_default_config(data_dir=data_dir)

    # Assets resolved from the custom root, not the (missing) default.
    assert str(cfg.gravity.file_path).startswith(str(data_dir.resolve()))
    assert all(str(data_dir.resolve()) in k for k in cfg.spice.kernels)


def test_kernel_dir_and_gravity_file_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _break_default_dirs(monkeypatch, tmp_path)
    data_dir = _make_data_dir(tmp_path / "custom_data")
    kernel_dir = data_dir / "ephemeris_models"
    gravity_file = data_dir / "gravity_models" / _GRAVITY_FILE

    cfg = load_default_config(
        kernel_dir=kernel_dir, gravity_file_path=gravity_file
    )

    assert Path(cfg.gravity.file_path) == gravity_file.resolve()
    assert all(str(kernel_dir.resolve()) in k for k in cfg.spice.kernels)


def test_missing_assets_without_override_still_fail_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _break_default_dirs(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        load_default_config()


def test_gravity_file_override_missing_file_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _break_default_dirs(monkeypatch, tmp_path)
    # Kernels present, but the explicitly named gravity file does not exist.
    data_dir = _make_data_dir(tmp_path / "custom_data")
    kernel_dir = data_dir / "ephemeris_models"
    with pytest.raises(FileNotFoundError):
        load_default_config(
            kernel_dir=kernel_dir,
            gravity_file_path=tmp_path / "does_not_exist.tab",
        )
