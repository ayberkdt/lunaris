"""PDS3 ``MISSING_CONSTANT`` written as an ODL based integer.

Regression test: ``parse_pds3_cyl_label`` matched ``16#FF7FFFFB#`` with a
decimal regex and read it as ``16.0``, so a DN of 16 was masked as missing and
the real null value was not. ODL based integers are ``radix#[sign]digits#``
(PDS3 Standards Reference, 12.3.1.2). For a real ``SAMPLE_TYPE`` the integer
is read as the IEEE bit pattern of the sample word (the LRO convention:
``16#FF7FFFFB#`` is the float32 -3.4028226550889045e+38).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from lunaris.loaders.io_surface import PDS3ParseError, parse_pds3_cyl_label


def _label(tmp_path: Path, name: str, *, sample_type: str, sample_bits: int, missing: str) -> Path:
    lbl = tmp_path / f"{name}.lbl"
    lbl.write_text(
        "\n".join(
            [
                f'^IMAGE = "{name}.img"',
                "LINES = 2",
                "LINE_SAMPLES = 2",
                f"SAMPLE_TYPE = {sample_type}",
                f"SAMPLE_BITS = {sample_bits}",
                "UNIT = UNITLESS",
                "SCALING_FACTOR = 1.0",
                "OFFSET = 0.0",
                f"MISSING_CONSTANT = {missing}",
                'MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"',
                "MAP_RESOLUTION = 1 <PIXEL/DEGREE>",
                "MAXIMUM_LATITUDE = 90.0",
                "MINIMUM_LATITUDE = -90.0",
                "WESTERNMOST_LONGITUDE = 0.0",
                "EASTERNMOST_LONGITUDE = 360.0",
                'POSITIVE_LONGITUDE_DIRECTION = "EAST"',
                "CENTER_LATITUDE = 0.0 <DEGREE>",
                "CENTER_LONGITUDE = 0.0 <DEGREE>",
            ]
        ),
        encoding="utf-8",
    )
    return lbl


def test_based_missing_constant_on_a_float_raster_is_the_ieee_pattern(tmp_path: Path) -> None:
    info = parse_pds3_cyl_label(
        _label(tmp_path, "f32", sample_type="PC_REAL", sample_bits=32, missing="16#FF7FFFFB#")
    )
    # Independent decode of the bit pattern through struct (big-endian word).
    expected = struct.unpack(">f", bytes.fromhex("FF7FFFFB"))[0]
    assert info.missing_constant == expected
    assert info.missing_constant == pytest.approx(-3.4028226550889045e38, rel=0.0)
    # A float32 null sample compares equal after promotion.
    assert float(np.float32(expected)) == info.missing_constant


def test_based_missing_constant_on_an_integer_raster(tmp_path: Path) -> None:
    info = parse_pds3_cyl_label(
        _label(tmp_path, "i16", sample_type="LSB_INTEGER", sample_bits=16, missing="16#-8000#")
    )
    assert info.missing_constant == -32768.0


def test_based_missing_constant_in_base_two(tmp_path: Path) -> None:
    info = parse_pds3_cyl_label(
        _label(tmp_path, "b2", sample_type="LSB_INTEGER", sample_bits=16, missing="2#1010#")
    )
    assert info.missing_constant == 10.0


def test_decimal_missing_constant_is_unchanged(tmp_path: Path) -> None:
    info = parse_pds3_cyl_label(
        _label(tmp_path, "dec", sample_type="LSB_INTEGER", sample_bits=16, missing="-32768")
    )
    assert info.missing_constant == -32768.0


@pytest.mark.parametrize(
    ("sample_type", "bits", "expected"),
    [
        ("PC_REAL", 32, "<f4"),
        ("PC_REAL", 64, "<f8"),
        ("IEEE_REAL", 32, ">f4"),  # PDS3 Appendix C: IEEE_REAL is big-endian
        ("REAL", 64, ">f8"),
        ("MAC_REAL", 32, ">f4"),
        ("SUN_REAL", 32, ">f4"),
    ],
)
def test_real_sample_type_byte_order(sample_type: str, bits: int, expected: str) -> None:
    """Regression: IEEE_REAL / REAL were read as little-endian like PC_REAL."""
    from lunaris.loaders.io_surface import dtype_from_pds3_sample

    dtype = dtype_from_pds3_sample(sample_type, bits)
    assert dtype == np.dtype(expected)
    # A big-endian word decodes to the value it encodes.
    raw = struct.pack(">f", 1737.4) if expected == ">f4" else None
    if raw is not None:
        assert np.frombuffer(raw, dtype=dtype)[0] == np.float32(1737.4)


@pytest.mark.parametrize(
    ("sample_type", "bits", "missing"),
    [
        ("LSB_INTEGER", 16, "16#10000#"),  # outside int16
        ("LSB_INTEGER", 16, "17#10#"),  # radix above 16
        ("LSB_INTEGER", 16, "8#9#"),  # not base-8 digits
        ("PC_REAL", 32, "16#1FFFFFFFF#"),  # wider than a 32-bit word
    ],
)
def test_malformed_based_missing_constant_is_refused(
    tmp_path: Path, sample_type: str, bits: int, missing: str
) -> None:
    name = f"bad_{abs(hash((sample_type, missing)))}"
    with pytest.raises(PDS3ParseError, match="MISSING_CONSTANT"):
        parse_pds3_cyl_label(_label(tmp_path, name, sample_type=sample_type, sample_bits=bits, missing=missing))
