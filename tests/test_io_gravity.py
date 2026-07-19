"""Fail-closed gravity-file header and coefficient-loading contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.cli.batch_runner import _resolve_initial_state_mu
from lunaris.common.constants import MU_MOON
from lunaris.loaders.io_gravity import (
    load_shadr_ascii,
    read_gravity_model_header,
    read_gravity_model_metadata,
)
from lunaris.physics.spherical_harmonics import GravityModel


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


@pytest.mark.parametrize(
    ("radius_km", "gm_km3s2"),
    [(0.0, 4902.8), (-1.0, 4902.8), (1738.0, 0.0), (1738.0, -1.0)],
)
def test_shadr_rejects_nonpositive_physical_header_values(
    tmp_path: Path, radius_km: float, gm_km3s2: float
) -> None:
    path = _write_shadr(tmp_path / "bad_header.tab", radius_km=radius_km, gm_km3s2=gm_km3s2)

    with pytest.raises(ValueError, match="finite and > 0"):
        load_shadr_ascii(str(path), strict=True)


@pytest.mark.parametrize("coefficient", ["nan", "inf", "-inf"])
def test_strict_shadr_rejects_nonfinite_coefficients(
    tmp_path: Path, coefficient: str
) -> None:
    path = tmp_path / "bad_coefficient.tab"
    path.write_text(
        "1738, 4902.8, 0, 1, 1, 1, 0, 0\n"
        f"1, 0, {coefficient}, 0\n"
        "1, 1, 0, 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Non-finite coefficient"):
        load_shadr_ascii(str(path), strict=True)


def test_gravity_metadata_reads_companion_pds_physical_contract(tmp_path: Path) -> None:
    path = _write_shadr(tmp_path / "model.tab.txt", gm_km3s2=4902.8003063302)
    (tmp_path / "model.lbl.txt").write_text(
        'ORIGINAL_PRODUCT_ID = "GL_TEST"\n'
        'DESCRIPTION = "Coefficients are fully normalized. The DE440 lunar body-fixed '
        'principal axes (PA) coordinate system is used. Degree coefficients do not \n'
        'include the permanent tide."\n',
        encoding="utf-8",
    )

    metadata = read_gravity_model_metadata(str(path))

    assert metadata["model_id"] == "GL_TEST"
    assert metadata["normalization"] == "fully_normalized_4pi"
    assert metadata["coefficient_frame"] == "MOON_PA"
    assert metadata["tide_system"] == "tide_free"
    assert metadata["source_gm_m3s2"] == pytest.approx(4.9028003063302e12)
    assert metadata["source_radius_m"] == pytest.approx(1_738_000.0)
    assert len(str(metadata["source_sha256"])) == 64

    model = GravityModel.from_file(str(path))
    assert model.metadata.model_id == "GL_TEST"
    assert model.metadata.coefficient_frame == "MOON_PA"
    assert model.metadata.tide_system == "tide_free"
