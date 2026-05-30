# lunaris/common/time_utils.py
"""
Time Utilities (Analytical, SPICE-Free)
======================================

This module provides fast, dependency-light time conversions for the simulation core,
without requiring SPICE. The primary internal time coordinate is:

    seconds past J2000 epoch (TT)

Design Goals
------------
- Deterministic: Pure arithmetic, no external kernels, consistent across platforms.
- Fast: Numba-accelerated arithmetic kernels.
  Numba is required. This module intentionally keeps time conversions dependency-light
  otherwise, but the hot-path scalar conversion kernels use `@njit`.
- Portable: Works in pure Python when `numba` is unavailable.

What’s Included
---------------
- JD <-> MJD: Pure epoch-offset conversions (no time-scale change).
- Gregorian <-> JD: Proleptic Gregorian calendar conversions (JDN-based, integer-safe).
- J2000 Seconds (TT): Conversion between MJD(TT) and seconds past J2000(TT).
- String Helpers: Parse/format ISO-like timestamps for configs, logging, and plots.

Time Scales (Important)
-----------------------
This module is analytical and does not perform UTC<->TT/TDB alignment via a leap-second table.

- If an input timestamp is intended to represent TT (or a continuous dynamical scale),
  the conversions are consistent and suitable for simulation time stepping.
- If an input timestamp is UTC, converting it to TT requires the leap-second offset
  ΔAT = (TAI-UTC), which is not provided here.

For convenience, helper functions are available:

    mjd_utc_to_mjd_tt(mjd_utc, tai_minus_utc_s)
    mjd_tt_to_mjd_utc(mjd_tt, tai_minus_utc_s)

You must supply `tai_minus_utc_s` from a leap-second table if true UTC alignment matters.

Recommended Use
---------------
Use this module for:
- Relative propagation time, integrator stepping, and checkpoints
- Visualization / labeling / debugging
- Unit tests that do not require historical UTC leap-second fidelity

For mission-critical UTC alignment or spacecraft operations timelines, use a SPICE-based layer.

Notes
-----
- Calendar conversions use the proleptic Gregorian calendar (no 1582 switch).
- Microsecond-level round-trips are guaranteed in the J2000-seconds domain.
  Round-tripping via absolute JD floats can lose microseconds near day boundaries due to
  floating-point resolution limits.
"""


# =============================================================================
# 0.                                 IMPORTS
# =============================================================================
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from numba import njit

from .constants import MJD_J2000_TT, DAY_S



# =============================================================================
# 1.                    JULIAN DATE, MJD, and J2000 (TT) SECONDS
# =============================================================================

JD_MJD_OFFSET: float = 2400000.5          # JD = MJD + 2400000.5
JDN_J2000_TT: int = 2451545               # 2000-01-01 12:00:00 TT
US_PER_DAY: int = 86_400_000_000
J2000_NOON_US: int = 43_200_000_000


@njit(cache=True)
def jd_to_mjd(jd: float) -> float:
    """
    JD -> MJD (pure epoch offset; does NOT change time scale).

    MJD = JD - 2400000.5
    """
    return jd - JD_MJD_OFFSET


@njit(cache=True)
def mjd_to_jd(mjd: float) -> float:
    """
    MJD -> JD (pure epoch offset; does NOT change time scale).

    JD = MJD + 2400000.5
    """
    return mjd + JD_MJD_OFFSET


@njit(cache=True)
def mjd_tt_to_j2000_seconds(mjd_tt: float) -> float:
    """
    MJD(TT) -> seconds past J2000(TT).

    J2000 epoch here is 2000-01-01 12:00:00 (TT) => MJD 51544.5.
    """
    return (mjd_tt - MJD_J2000_TT) * DAY_S


@njit(cache=True)
def j2000_seconds_to_mjd_tt(j2000_s: float) -> float:
    """
    seconds past J2000(TT) -> MJD(TT).
    """
    return MJD_J2000_TT + (j2000_s / DAY_S)


@njit(cache=True)
def mjd_utc_to_mjd_tt(mjd_utc: float, tai_minus_utc_s: float) -> float:
    """
    MJD(UTC) -> MJD(TT).

    TT = UTC + (TAI-UTC) + 32.184 s  = UTC + ΔAT + 32.184 s

    Parameters
    ----------
    tai_minus_utc_s : float
        ΔAT = (TAI - UTC) in seconds (accumulated leap seconds).
        Must be provided from a leap-second table for the given date.
    """

    if not (0.0 <= tai_minus_utc_s <= 200.0):
        return float('nan')  # Signal invalid input
    return mjd_utc + (tai_minus_utc_s + 32.184) / DAY_S


@njit(cache=True)
def mjd_tt_to_mjd_utc(mjd_tt: float, tai_minus_utc_s: float) -> float:
    """
    MJD(TT) -> MJD(UTC) (inverse of mjd_utc_to_mjd_tt).

    UTC = TT - (ΔAT + 32.184 s)
    """
    return mjd_tt - (tai_minus_utc_s + 32.184) / DAY_S



# =============================================================================
# 2.                            DATE HELPERS
# =============================================================================

# Month lengths for a common (non-leap) year; index 1..12 is used.
_DAYS_IN_MONTH_COMMON = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@njit(cache=True)
def is_leap_year_gregorian(year: int) -> bool:
    """
    Gregorian leap-year rule.

    True for:
    - divisible by 4
    - except centuries not divisible by 400
    """
    if year % 4 != 0:
        return False
    if year % 100 != 0:
        return True
    return year % 400 == 0


@njit(cache=True)
def days_in_month_gregorian(year: int, month: int) -> int:
    """
    Number of days in the given month for a Gregorian calendar year.

    Parameters
    ----------
    year : int
    month : int
        1..12 (no validation inside njit for speed; caller should ensure validity)

    Returns
    -------
    int
        Days in month.
    """
    d = _DAYS_IN_MONTH_COMMON[month]  # month must be 1..12
    if month == 2 and is_leap_year_gregorian(year):
        return d + 1
    return d



# =============================================================================
# 3.             CALENDAR ALGORITHMS (Gregorian <-> JD)
# =============================================================================
# Clean, integer-based conversions via JDN (proleptic Gregorian).
# JD is returned in the astronomical convention where days start at noon:
#   JD = JDN - 0.5 + (seconds_into_day / 86400)

@njit(cache=True)
def _floor_float(x: float) -> int:
    """Numba-friendly floor(x) -> int without importing math."""
    i = int(x)
    return i - 1 if x < i else i


@njit(cache=True)
def _ymd_to_jdn_gregorian(year: int, month: int, day: int) -> int:
    """Gregorian calendar date -> Julian Day Number (JDN) at 00:00."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


@njit(cache=True)
def _jdn_to_ymd_gregorian(jdn: int):
    """Julian Day Number (JDN) -> (year, month, day) in Gregorian calendar."""
    a = jdn + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153

    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + (m // 10)
    return year, month, day


@njit(cache=True)
def date_to_jd(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """
    Gregorian date/time -> Julian Date (JD), proleptic Gregorian.
    Uses integer microseconds-of-day for stable round-trips.
    """
    jdn = _ymd_to_jdn_gregorian(year, month, day)

    # Convert time to integer microseconds (rounded to nearest µs)
    us = int((second * 1_000_000.0) + 0.5)
    us_day = hour * 3_600_000_000 + minute * 60_000_000 + us

    # Carry if rounding pushed beyond a day
    if us_day >= US_PER_DAY:
        us_day -= US_PER_DAY
        jdn += 1

    return (float(jdn) - 0.5) + (float(us_day) / float(US_PER_DAY))


@njit(cache=True)
def jd_to_date_tuple(jd: float) -> tuple[int, int, int, int, int, float]:
    """
    JD -> (year, month, day, hour, minute, second) proleptic Gregorian.
    Extracts time via integer microseconds-of-day to avoid float rollover bugs.
    """
    jd_adj = jd + 0.5
    jdn = _floor_float(jd_adj)
    f = jd_adj - float(jdn)

    # Guard against float edge cases
    if f >= 1.0:
        f -= 1.0
        jdn += 1
    elif f < 0.0:
        f += 1.0
        jdn -= 1

    # Convert fractional day -> integer microseconds (rounded to nearest µs)

    US_PER_DAY = 86_400_000_000
    us_day = int(f * float(US_PER_DAY) + 0.5)

    # IMPORTANT: Do NOT carry to next day here. JD(float) cannot represent µs near day-boundary reliably.
    # Clamp instead to keep the calendar day stable and deterministic.
    if us_day < 0:
        us_day = 0
    elif us_day >= US_PER_DAY:
        us_day = US_PER_DAY - 1


    year, month, day = _jdn_to_ymd_gregorian(jdn)

    # Decompose microseconds-of-day
    h = us_day // 3_600_000_000
    us_day -= h * 3_600_000_000

    mi = us_day // 60_000_000
    us_day -= mi * 60_000_000

    s_int = us_day // 1_000_000
    us = us_day - s_int * 1_000_000

    s = float(s_int) + float(us) / 1_000_000.0
    return int(year), int(month), int(day), int(h), int(mi), s


@njit(cache=True)
def date_tt_to_j2000_seconds(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """
    Convenience: Gregorian date/time interpreted as TT -> seconds past J2000(TT).

    Pipeline
    --------
    date(TT) -> JD -> MJD -> seconds past J2000(TT)

    Returns
    -------
    float
        Seconds past J2000 (TT).
    """
    jdn = _ymd_to_jdn_gregorian(year, month, day)

    us = int((second * 1_000_000.0) + 0.5)
    us_day = hour * 3_600_000_000 + minute * 60_000_000 + us

    if us_day >= US_PER_DAY:
        us_day -= US_PER_DAY
        jdn += 1

    delta_us = (jdn - JDN_J2000_TT) * US_PER_DAY + us_day - J2000_NOON_US
    return float(delta_us) / 1_000_000.0



# =============================================================================
# 4.             PYTHONIC WRAPPERS (UI & Config Helpers)
# =============================================================================

def parse_iso_to_j2000_seconds_tt(iso_str: str) -> float:
    """
    Parse a date/time string and return seconds past J2000(TT).

    Notes
    -----
    - This is an analytic conversion (no leap-second table lookup).
    - If the input contains a timezone offset, it is normalized to UTC and then made naive.
      (Still analytic; no UTC<->TT leap-second handling is performed.)
    - If the input has no timezone info, it is treated as a naive datetime (user-defined scale).

    Parameters
    ----------
    iso_str : str
        Examples:
        - "2025-01-01"
        - "2025-01-01 12:00"
        - "2025-01-01 12:00:00.123"
        - "2025-01-01T12:00:00+03:00"
        - "2025-01-01T09:00:00Z"

    Returns
    -------
    float
        Seconds past J2000(TT).
    """
    dt = _parse_datetime_loose(iso_str)
    sec_float = dt.second + dt.microsecond / 1_000_000.0
    return date_tt_to_j2000_seconds(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        sec_float,
    )


def parse_iso_datetime_to_utc_datetime(
    value: str | datetime,
    *,
    assume_naive_utc: bool = True,
) -> datetime:
    """
    Parse a civil date/time input and return a timezone-aware UTC datetime.

    Why this helper exists
    ----------------------
    The UI and CLI both accept human-entered ISO-like timestamps.  Some callers
    provide explicit offsets (for example ``+03:00``), while older code paths
    store naive timestamps without any timezone suffix.  Centralizing the
    normalization policy keeps every entry point aligned before times reach the
    SPICE ephemeris layer.

    Policy
    ------
    - Offset-aware inputs are converted to UTC.
    - Inputs ending in ``Z`` are treated as UTC.
    - Naive inputs are interpreted as UTC when `assume_naive_utc=True`.

    Notes
    -----
    This helper normalizes civil timezone offsets only.  It does not perform
    leap-second handling or UTC<->TT/TDB conversion; those are separate concerns.
    """

    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError("Empty datetime string is not allowed.")

        normalized = raw.replace("T", " ")
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            dt = _parse_datetime_loose(raw)

    if dt.tzinfo is None:
        if not assume_naive_utc:
            raise ValueError("Naive datetime input requires assume_naive_utc=True.")
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def normalize_iso_datetime_to_utc_string(
    value: str | datetime,
    *,
    precision: int = 0,
    assume_naive_utc: bool = True,
) -> str:
    """
    Canonicalize a civil date/time input to an ISO-like UTC string ending in `Z`.

    Examples
    --------
    - ``2026-05-10T19:19:47+03:00`` -> ``2026-05-10T16:19:47Z``
    - ``2026-05-10 16:19:47``       -> ``2026-05-10T16:19:47Z``

    Parameters
    ----------
    precision:
        Fractional-second digits to keep in the canonical output (0..6).
    """

    dt_utc = parse_iso_datetime_to_utc_datetime(
        value,
        assume_naive_utc=assume_naive_utc,
    )

    p = int(precision)
    if p < 0:
        p = 0
    elif p > 6:
        p = 6

    if p < 6:
        dt_utc = _round_dt_to_precision(dt_utc, p)

    base_str = dt_utc.strftime("%Y-%m-%dT%H:%M:%S")
    if p == 0:
        return f"{base_str}Z"

    us_str = f"{dt_utc.microsecond:06d}"
    return f"{base_str}.{us_str[:p]}Z"


def _round_dt_to_precision(dt: datetime, precision: int) -> datetime:
    """
    Round a datetime's fractional seconds to `precision` digits (0..6).

    Rounds microseconds and handles rollover (e.g., 59.999999 -> next second).
    """
    p = int(precision)
    if p <= 0:
        # drop microseconds entirely
        return dt.replace(microsecond=0)
    if p >= 6:
        return dt

    unit = 10 ** (6 - p)  # e.g., p=3 -> 1000 us
    us = dt.microsecond

    rounded_us = int((us + unit / 2) // unit) * unit
    if rounded_us >= 1_000_000:
        dt = dt + timedelta(seconds=1)
        rounded_us = 0

    return dt.replace(microsecond=rounded_us)


def format_j2000_seconds_as_iso(j2000_s: float, precision: int = 3) -> str:
    """
    Format seconds past J2000(TT) as an ISO-like string.

    Parameters
    ----------
    j2000_s : float
        Seconds past J2000(TT).
    precision : int
        Decimal places for seconds (0..6).

    Returns
    -------
    str
        Example: "2024-05-19 12:00:00.123"
    """
    p = int(precision)
    if p < 0:
        p = 0
    elif p > 6:
        p = 6

    total_us = int(float(j2000_s) * 1_000_000.0 + (0.5 if j2000_s >= 0.0 else -0.5))
    day_offset, us_day = divmod(total_us + J2000_NOON_US, US_PER_DAY)
    jdn = JDN_J2000_TT + int(day_offset)
    y, mo, d = _jdn_to_ymd_gregorian(int(jdn))

    h = us_day // 3_600_000_000
    us_day -= h * 3_600_000_000
    mi = us_day // 60_000_000
    us_day -= mi * 60_000_000
    s_int = us_day // 1_000_000
    microseconds = int(us_day - s_int * 1_000_000)

    dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s_int), microseconds)
    if p < 6:
        dt = _round_dt_to_precision(dt, p)

    if p == 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    base_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    us_str = f"{dt.microsecond:06d}"
    return f"{base_str}.{us_str[:p]}"


def _parse_datetime_loose(s: str) -> datetime:
    """
    Best-effort parse of common date/time strings into a Python datetime.

    Supported (examples)
    --------------------
    - "YYYY-MM-DD"
    - "YYYY-MM-DD HH:MM"
    - "YYYY-MM-DD HH:MM:SS"
    - "YYYY-MM-DD HH:MM:SS.ffffff"
    - "YYYY-MM-DDTHH:MM:SS"
    - With timezone offsets: "+03:00", "Z" (normalized to UTC then tzinfo removed)

    Returns
    -------
    datetime
        Naive datetime (timezone removed). If input had tz info, it is first converted to UTC.

    Raises
    ------
    ValueError
        If the format is unrecognizable.
    """
    s = s.strip().replace("T", " ")

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # 1) fromisoformat (handles timezone offsets like +03:00)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        pass

    # 2) common fallbacks
    fmts = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported date/time format: '{s}'")



# =============================================================================
# 5.                           PUBLIC API
# =============================================================================

__all__ = [
    # --- Core epoch/scale converters ---
    "jd_to_mjd",                      # JD <-> MJD (pure epoch offset; no time-scale change)
    "mjd_to_jd",                      # JD <-> MJD (pure epoch offset; no time-scale change)

    "mjd_tt_to_j2000_seconds",        # MJD(TT) -> seconds past J2000(TT)
    "j2000_seconds_to_mjd_tt",        # seconds past J2000(TT) -> MJD(TT)

    "mjd_utc_to_mjd_tt",              # MJD(UTC) -> MJD(TT) using ΔAT (TAI-UTC) + 32.184s
    "mjd_tt_to_mjd_utc",              # MJD(TT) -> MJD(UTC) (inverse, requires same ΔAT)

    # --- Gregorian date helpers (proleptic Gregorian) ---
    "is_leap_year_gregorian",         # Gregorian leap-year rule
    "days_in_month_gregorian",        # Days in a month for a given year (handles Feb leap day)

    # --- Calendar algorithms ---
    "date_to_jd",                     # Gregorian date/time -> JD (proleptic Gregorian)
    "jd_to_date_tuple",               # JD -> (Y,M,D,h,m,s) (proleptic Gregorian)
    "date_tt_to_j2000_seconds",       # Gregorian date/time (interpreted as TT) -> seconds past J2000(TT)

    # --- UI / string helpers (analytic; no leap-second lookup) ---
    "parse_iso_to_j2000_seconds_tt",  # Parse ISO-like string -> seconds past J2000(TT) (analytic)
    "format_j2000_seconds_as_iso",    # seconds past J2000(TT) -> ISO-like string
    "parse_iso_datetime_to_utc_datetime",  # Civil ISO-like string/datetime -> timezone-aware UTC datetime
    "normalize_iso_datetime_to_utc_string",  # Civil ISO-like string/datetime -> canonical UTC string ending in Z
]
