"""PDS3 LDEM label UNIT handling and LDEM product identity.

Regression tests for two loader defects:

1. ``parse_ldem_label`` read ``UNIT`` but every consumer assumed kilometres, so
   a label declaring ``UNIT = METER`` (the LOLA integer-LDEM convention:
   ``SCALING_FACTOR = 0.5``, ``OFFSET = 1737400.``) produced a surface radius
   1000x too large. The real ``LDEM_64_FLOAT`` label declares
   ``UNIT = KILOMETER``, ``SCALING_FACTOR = 1``, ``OFFSET = 1737.4``.
2. ``find_ldem_product`` accepted any label that parsed, so a LOLA LDAM albedo
   raster (whose label also carries the cylindrical map fields) or a product
   for another body could be loaded as topography, and an explicit ``ppd`` was
   silently replaced by the "closest" resolution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunaris.loaders.io_surface import (
    PDS3ParseError,
    TopographyGrid,
    find_ldem_product,
    parse_ldem_label,
)

_R_KM = 1737.4


def _write_pair(
    root: Path,
    stem: str,
    *,
    product_id: str = "LDEM_4",
    unit: str = "KILOMETER",
    scaling: float = 1.0,
    offset: float = _R_KM,
    ppd: int = 1,
    target: str | None = "MOON",
    dn: float = 2.0,
) -> Path:
    """Write a global 1-ppd (or coarser) float LDEM-like label + IMG."""
    lines, samples = 180 * ppd, 360 * ppd
    img = root / f"{stem}.img"
    np.full((lines, samples), dn, dtype="<f4").tofile(img)
    rows = [
        "PDS_VERSION_ID = PDS3",
        f'PRODUCT_ID = "{product_id}"',
    ]
    if target is not None:
        rows.append(f"TARGET_NAME = {target}")
    rows += [
        f'^IMAGE = "{img.name}"',
        "OBJECT = IMAGE",
        f"  LINES = {lines}",
        f"  LINE_SAMPLES = {samples}",
        "  SAMPLE_TYPE = PC_REAL",
        "  SAMPLE_BITS = 32",
        f"  UNIT = {unit}",
        f"  SCALING_FACTOR = {scaling}",
        f"  OFFSET = {offset}",
        "END_OBJECT = IMAGE",
        'MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"',
        f"MAP_RESOLUTION = {ppd} <pix/deg>",
        "MAXIMUM_LATITUDE = 90.0",
        "MINIMUM_LATITUDE = -90.0",
        "WESTERNMOST_LONGITUDE = 0.0",
        "EASTERNMOST_LONGITUDE = 360.0",
        'POSITIVE_LONGITUDE_DIRECTION = "EAST"',
        "CENTER_LATITUDE = 0.0 <deg>",
        "CENTER_LONGITUDE = 180.0 <deg>",
        "LINE_PROJECTION_OFFSET = 89.5 <pix>",
        "SAMPLE_PROJECTION_OFFSET = 179.5 <pix>",
        "END",
    ]
    lbl = root / f"{stem}.lbl"
    lbl.write_text("\n".join(rows) + "\n", encoding="utf-8")
    parse_ldem_label.cache_clear()
    return lbl


# --------------------------------------------------------------------------- 1


def test_kilometre_label_radius(tmp_path: Path) -> None:
    lbl = _write_pair(tmp_path, "ldem_km", unit="KILOMETER", scaling=1.0, offset=_R_KM, dn=2.0)
    grid = TopographyGrid(lbl, mmap=False)
    assert grid.radius_m(0.1, 0.2) == pytest.approx((_R_KM + 2.0) * 1000.0)
    assert grid.reference_radius_m == pytest.approx(_R_KM * 1000.0)


def test_metre_label_is_converted_not_read_as_km(tmp_path: Path) -> None:
    # LOLA integer-LDEM style: DN in 0.5 m steps above a 1737400 m sphere.
    lbl = _write_pair(tmp_path, "ldem_m", unit="METER", scaling=0.5, offset=1737400.0, dn=4000.0)
    grid = TopographyGrid(lbl, mmap=False)
    # height = 4000 * 0.5 m = 2000 m; radius = 1737400 + 2000 m.
    assert grid.height_m(0.1, 0.2) == pytest.approx(2000.0)
    assert grid.radius_m(0.1, 0.2) == pytest.approx(1739400.0)
    assert grid.reference_radius_m == pytest.approx(1737400.0)


def test_metre_and_kilometre_labels_agree(tmp_path: Path) -> None:
    km = _write_pair(tmp_path, "a", unit="KM", scaling=1.0, offset=_R_KM, dn=1.25)
    m = _write_pair(tmp_path, "b", unit="METERS", scaling=1000.0, offset=_R_KM * 1000.0, dn=1.25)
    r_km = TopographyGrid(km, mmap=False).radius_m(0.3, 1.0)
    r_m = TopographyGrid(m, mmap=False).radius_m(0.3, 1.0)
    assert r_m == pytest.approx(r_km, rel=0, abs=1e-6)


@pytest.mark.parametrize("unit", ["NONE", "UNITLESS", "DEGREE", "FURLONG"])
def test_non_length_unit_is_refused(tmp_path: Path, unit: str) -> None:
    lbl = _write_pair(tmp_path, f"bad_{unit.lower()}", unit=unit)
    with pytest.raises(PDS3ParseError, match="UNIT"):
        parse_ldem_label(lbl)


# --------------------------------------------------------------------------- 2


def test_explicit_ppd_does_not_fall_back_to_ldam_albedo(tmp_path: Path) -> None:
    # The LDAM label parses as a cylindrical map (UNIT = NONE) and sits at the
    # requested resolution; the LDEM is at another resolution.
    _write_pair(tmp_path, "ldem_1", product_id="LDEM_1", ppd=1)
    _write_pair(tmp_path, "ldam_2", product_id="LDAM_2_FLOAT", unit="NONE", offset=0.0, ppd=2)
    with pytest.raises(FileNotFoundError):
        find_ldem_product(tmp_path, ppd=2)


def test_highest_resolution_choice_skips_non_ldem_products(tmp_path: Path) -> None:
    ldem = _write_pair(tmp_path, "ldem_1", product_id="LDEM_1", ppd=1)
    # A higher-resolution raster with a length unit that is not an LDEM product.
    _write_pair(tmp_path, "dgdr_2", product_id="DGDR_RA_AVG_CYL_002_IMG", ppd=2)
    lbl, _img = find_ldem_product(tmp_path)
    assert lbl == ldem


def test_other_body_is_rejected(tmp_path: Path) -> None:
    _write_pair(tmp_path, "mars", product_id="LDEM_1", target="MARS", ppd=1)
    with pytest.raises(FileNotFoundError):
        find_ldem_product(tmp_path)


def test_direct_label_identity_is_checked(tmp_path: Path) -> None:
    ldam = _write_pair(tmp_path, "ldam_1", product_id="LDAM_1_FLOAT", unit="NONE", offset=0.0)
    with pytest.raises(FileNotFoundError, match="LDEM"):
        find_ldem_product(ldam)


def test_explicit_ppd_requires_exact_resolution(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ldem_1", product_id="LDEM_1", ppd=1)
    with pytest.raises(FileNotFoundError, match="ppd"):
        find_ldem_product(tmp_path, ppd=2)


def test_exact_ppd_and_missing_target_name_are_accepted(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ldem_1", product_id="LDEM_1", ppd=1)
    want = _write_pair(tmp_path, "ldem_2", product_id="LDEM_2", ppd=2, target=None)
    lbl, img = find_ldem_product(tmp_path, ppd=2)
    assert lbl == want
    assert img.name == "ldem_2.img"


def test_sldem_product_is_an_ldem(tmp_path: Path) -> None:
    want = _write_pair(tmp_path, "sldem", product_id="SLDEM2015_1_FLOAT", ppd=1)
    lbl, _ = find_ldem_product(tmp_path)
    assert lbl == want
