"""Presentation-layer formatting for Mission Monitor widgets (Qt-free).

Stored telemetry stays SI; every unit conversion happens here, at render time,
so the science data path never carries display units. ``None`` (missing
channel value) always renders as :data:`UNAVAILABLE` — never "0".
"""

from __future__ import annotations

import math

#: Canonical "no data" glyph. Widgets pair it with an explanation, never with
#: a fake zero or an empty axis.
UNAVAILABLE = "—"

_DAY_S = 86_400.0


def format_length(meters: float | None, *, decimals: int = 3) -> str:
    """Length with automatic m/km selection (SI in, display out)."""
    if meters is None or not math.isfinite(meters):
        return UNAVAILABLE
    if abs(meters) >= 10_000.0:
        return f"{meters / 1000.0:,.{decimals}f} km"
    return f"{meters:,.1f} m"


def format_speed(m_s: float | None, *, decimals: int = 3) -> str:
    if m_s is None or not math.isfinite(m_s):
        return UNAVAILABLE
    if abs(m_s) >= 1000.0:
        return f"{m_s / 1000.0:,.{decimals}f} km/s"
    return f"{m_s:,.2f} m/s"


def format_angle_from_rad(rad: float | None, *, decimals: int = 3) -> str:
    if rad is None or not math.isfinite(rad):
        return UNAVAILABLE
    return f"{math.degrees(rad):.{decimals}f}°"


def format_dimensionless(value: float | None, *, decimals: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return UNAVAILABLE
    return f"{value:.{decimals}f}"


def format_duration(seconds: float | None) -> str:
    """Human-scale duration: s → min → h → d, one unit, no false precision."""
    if seconds is None or not math.isfinite(seconds):
        return UNAVAILABLE
    magnitude = abs(seconds)
    if magnitude < 120.0:
        return f"{seconds:.1f} s"
    if magnitude < 2.0 * 3600.0:
        return f"{seconds / 60.0:.1f} min"
    if magnitude < 2.0 * _DAY_S:
        return f"{seconds / 3600.0:.2f} h"
    return f"{seconds / _DAY_S:.2f} d"


def format_count(value: float | int | None) -> str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        if not math.isfinite(value):
            return UNAVAILABLE
        value = int(value)
    return f"{value:,}"


__all__ = [
    "UNAVAILABLE",
    "format_angle_from_rad",
    "format_count",
    "format_dimensionless",
    "format_duration",
    "format_length",
    "format_speed",
]
