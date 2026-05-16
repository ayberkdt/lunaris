# tests/test_time_utils.py
# -*- coding: utf-8 -*-
"""
Unit tests for common.time_utils
================================

These tests mirror the built-in self-test in common/time_utils.py, but in pytest
form so they can run in CI and give targeted failures.

Run:
    pytest -q
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

# -----------------------------------------------------------------------------
# Import helper (run from repo root without installing)
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from common import time_utils  # project layout: <root>/common/time_utils.py
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import 'common.time_utils'. "
        "Run pytest from the repository root (the folder that contains 'common/')."
    ) from e

# constants live in common.constants (single source of truth)
try:
    from common import constants as C
except Exception:  # pragma: no cover
    C = None


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
_TWOPI = 2.0 * math.pi


def test_constants_exist_and_match_module_contract():
    """time_utils depends on constants.DAY_S and constants.MJD_J2000_TT."""
    if C is None:
        pytest.skip("common.constants not importable in this environment")

    assert hasattr(C, "DAY_S")
    assert hasattr(C, "MJD_J2000_TT")
    # sanity only; values are project-owned
    assert C.DAY_S > 0.0


def test_j2000_epoch_consistency_tt():
    # J2000(TT) = 2000-01-01 12:00:00 -> JD 2451545.0
    jd = time_utils.date_to_jd(2000, 1, 1, 12, 0, 0.0)
    assert_allclose(jd, 2451545.0, atol=1e-12, rtol=0.0)

    mjd = time_utils.jd_to_mjd(jd)
    assert_allclose(mjd, float(time_utils.MJD_J2000_TT), atol=1e-12, rtol=0.0)

    j2 = time_utils.mjd_tt_to_j2000_seconds(mjd)
    assert_allclose(j2, 0.0, atol=1e-9, rtol=0.0)

    # convenience
    j2b = time_utils.date_tt_to_j2000_seconds(2000, 1, 1, 12, 0, 0.0)
    assert_allclose(j2b, 0.0, atol=1e-9, rtol=0.0)


def test_jd_mjd_are_pure_inverse_offsets():
    jd_ref = 2460000.123456
    mjd = time_utils.jd_to_mjd(jd_ref)
    jd_back = time_utils.mjd_to_jd(mjd)
    assert_allclose(jd_back, jd_ref, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "y,mo,d,h,mi,s",
    [
        (2024, 2, 29, 12, 0, 0.0),              # leap day
        (2024, 1, 31, 23, 59, 59.5),            # month end + fractional
        (2000, 1, 1, 12, 0, 0.0),               # J2000
        (1999, 12, 31, 23, 59, 59.9),           # near day boundary
        (2025, 6, 1, 0, 0, 0.123456),           # microseconds
    ],
)
def test_calendar_roundtrip_date_jd_date(y, mo, d, h, mi, s):
    jd = time_utils.date_to_jd(y, mo, d, h, mi, s)
    y2, mo2, d2, h2, mi2, s2 = time_utils.jd_to_date_tuple(jd)

    assert (y2, mo2, d2) == (y, mo, d)
    assert (h2, mi2) == (h, mi)

    # JD float can't guarantee microseconds near day boundary; allow same tolerance as self-test.
    assert_allclose(float(s2), float(s), atol=6e-5, rtol=0.0)


def test_microsecond_boundary_stability_in_j2000_seconds_domain():
    # This test is intentionally in J2000-seconds domain (as per module notes).
    y, mo, d, h, mi, s = (1999, 12, 31, 23, 59, 59.999999)
    j2 = time_utils.date_tt_to_j2000_seconds(y, mo, d, h, mi, s)

    iso6 = time_utils.format_j2000_seconds_as_iso(j2, precision=6)
    assert iso6 == "1999-12-31 23:59:59.999999"

    j2_back = time_utils.parse_iso_to_j2000_seconds_tt(iso6)
    assert_allclose(j2_back, j2, atol=2e-6, rtol=0.0)


def test_iso_parse_format_roundtrip_precision_stable():
    s_in = "2024-05-19 15:30:45.123"
    j2 = time_utils.parse_iso_to_j2000_seconds_tt(s_in)
    s_out = time_utils.format_j2000_seconds_as_iso(j2, precision=3)
    assert s_out == s_in


def test_iso_z_suffix_accepted_and_normalized():
    s_in_z = "2024-05-19T12:00:00Z"
    j2 = time_utils.parse_iso_to_j2000_seconds_tt(s_in_z)
    s_out = time_utils.format_j2000_seconds_as_iso(j2, precision=0)
    assert s_out == "2024-05-19 12:00:00"


def test_civil_iso_offset_normalizes_to_canonical_utc_string():
    normalized = time_utils.normalize_iso_datetime_to_utc_string(
        "2026-05-10T19:19:47+03:00",
        precision=0,
    )
    assert normalized == "2026-05-10T16:19:47Z"


def test_naive_civil_iso_is_treated_as_utc_for_normalization():
    dt_utc = time_utils.parse_iso_datetime_to_utc_datetime("2026-05-10 16:19:47")
    assert dt_utc.isoformat() == "2026-05-10T16:19:47+00:00"


def test_rollover_rounding_to_milliseconds_next_day():
    # 23:59:59.999999 rounded to milliseconds -> next day 00:00:00.000
    j2_roll = time_utils.date_tt_to_j2000_seconds(2024, 1, 1, 23, 59, 59.999999)
    s_roll = time_utils.format_j2000_seconds_as_iso(j2_roll, precision=3)
    assert s_roll == "2024-01-02 00:00:00.000"


def test_gregorian_helpers_sanity():
    assert bool(time_utils.is_leap_year_gregorian(2024)) is True
    assert bool(time_utils.is_leap_year_gregorian(2023)) is False
    assert int(time_utils.days_in_month_gregorian(2024, 2)) == 29
    assert int(time_utils.days_in_month_gregorian(2023, 2)) == 28


def test_mjd_utc_tt_inverse_for_same_delta_at():
    mjd_utc = 60000.25
    delta_at = 123.0  # arbitrary; only inverse behavior tested
    mjd_tt = time_utils.mjd_utc_to_mjd_tt(mjd_utc, delta_at)
    mjd_utc_back = time_utils.mjd_tt_to_mjd_utc(mjd_tt, delta_at)
    assert_allclose(mjd_utc_back, mjd_utc, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize("delta_at", [-1.0, 201.0, 1e9])
def test_mjd_utc_to_mjd_tt_invalid_delta_returns_nan(delta_at):
    out = time_utils.mjd_utc_to_mjd_tt(60000.0, float(delta_at))
    assert math.isnan(float(out))


if __name__ == "__main__":
    import sys

    print("This is a pytest test module. Run it with:")
    print("python -m pytest -vv -rA --durations=10 tests/test_time_utils.py")
    sys.exit(0)
