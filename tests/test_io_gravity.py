"""Fail-closed gravity-file header and coefficient-loading contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.cli.batch_runner import _resolve_initial_state_mu
from lunaris.common.constants import MU_MOON
from lunaris.loaders.io_gravity import (
    load_gravity_model,
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


def _write_lines(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("token", ["1.0-08", "-9.0E-05junk", "1.0E-0.5"])
def test_strict_shadr_rejects_malformed_numeric_tokens(tmp_path: Path, token: str) -> None:
    """The regex fallback used to re-read ``1.0-08`` as the two numbers 1.0 and
    -8.0 (shifting every later column) and ``-9.0E-05junk`` as -9.0e-5."""
    path = _write_lines(
        tmp_path / "bad_token.tab",
        "1738, 4902.8, 0, 1, 1, 1, 0, 0",
        f"1, 0, {token}, 0",
        "1, 1, 0, 0",
    )
    with pytest.raises(ValueError, match="Malformed numeric field"):
        load_shadr_ascii(str(path), strict=True)


def test_strict_shadr_reads_fortran_d_exponent_as_one_number(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "fortran.tab",
        "1738, 4902.8, 0, 1, 1, 1, 0, 0",
        "1, 0, -2.0321D-04, 0",
        "1, 1, 0, 1.5d-06",
    )
    _n, _r, _gm, c, s = load_shadr_ascii(str(path), strict=True)
    assert c[1, 0] == -2.0321e-4
    assert s[1, 1] == 1.5e-6


@pytest.mark.parametrize("state", ["nan", "1.5", "0.999"])
def test_strict_shadr_rejects_non_integral_normalization_state(tmp_path: Path, state: str) -> None:
    """``int(1.5)`` truncated to 1 and a nan state became None; both silently
    passed the strict normalization check."""
    path = _write_lines(
        tmp_path / "bad_norm.tab",
        f"1738, 4902.8, 0, 1, 1, {state}, 0, 0",
        "1, 0, 0, 0",
        "1, 1, 0, 0",
    )
    with pytest.raises(ValueError, match="normalization state"):
        load_shadr_ascii(str(path), strict=True)
    with pytest.raises(ValueError, match="normalization state"):
        read_gravity_model_header(str(path))


def test_strict_shadr_rejects_non_integral_index(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "bad_index.tab",
        "1738, 4902.8, 0, 1, 1, 1, 0, 0",
        "1, 0, 0, 0",
        "1.7, 1, 0, 0",
    )
    with pytest.raises(ValueError, match="integral"):
        load_shadr_ascii(str(path), strict=True)


def test_icgem_gfc_is_refused_with_a_clear_error(tmp_path: Path) -> None:
    """A .gfc file used to fall through to the binary SHBDR parser and fail
    with an unrelated struct/IO error (or parse garbage)."""
    path = _write_lines(tmp_path / "model.gfc", "begin_of_head", "end_of_head", "gfc 2 0 -4.8e-4 0")
    with pytest.raises(ValueError, match="ICGEM"):
        load_gravity_model(str(path))
    with pytest.raises(ValueError, match="ICGEM"):
        read_gravity_model_header(str(path))


def test_cli_gravity_file_help_does_not_offer_gfc() -> None:
    import argparse

    from lunaris.cli import options

    parser = argparse.ArgumentParser()
    options._add_gravity_args(parser)
    help_text = parser.format_help()
    assert "--gravity-file-path" in help_text
    assert ".gfc" not in help_text
