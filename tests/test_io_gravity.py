"""Fail-closed gravity-file header and coefficient-loading contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.cli.batch_runner import _resolve_initial_state_mu
from lunaris.common.constants import MU_MOON
from lunaris.loaders.io_gravity import read_gravity_model_header


def _write_shadr(path: Path, *, radius_km: float = 1738.0, gm_km3s2: float = 4902.8) -> Path:
    path.write_text(
        f"{radius_km}, {gm_km3s2}, 0, 1, 1, 1, 0, 0\n"
        "1, 0, 0, 0\n"
        "1, 1, 0, 0\n",
        encoding="utf-8",
    )
    return path


def test_read_gravity_model_header_returns_si_values_without_loading_coefficients(
    tmp_path: Path,
) -> None:
    path = _write_shadr(tmp_path / "model.tab", gm_km3s2=4902.8003063302)

    degree, radius_m, gm_m3s2, normalization = read_gravity_model_header(str(path))

    assert degree == 1
    assert radius_m == pytest.approx(1_738_000.0)
    assert gm_m3s2 == pytest.approx(4.9028003063302e12)
    assert normalization == 1


def test_batch_orbit_initialization_uses_selected_classic_model_gm(tmp_path: Path) -> None:
    path = _write_shadr(tmp_path / "model.tab", gm_km3s2=4902.8003063302)
    cfg = SimpleNamespace(
        flags=SimpleNamespace(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False, file_path=path),
    )

    assert _resolve_initial_state_mu(cfg) == pytest.approx(4.9028003063302e12)


def test_batch_point_mass_initialization_uses_canonical_moon_gm() -> None:
    cfg = SimpleNamespace(
        flags=SimpleNamespace(enable_sh=False),
        gravity=SimpleNamespace(uses_st_lrps=False, file_path=None),
    )

    assert _resolve_initial_state_mu(cfg) == float(MU_MOON)
