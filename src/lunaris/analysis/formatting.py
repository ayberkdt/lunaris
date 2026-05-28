import numpy as np
from typing import Any

def safe_float(value: Any) -> float:
    """Best-effort float conversion used by formatting helpers."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "0 s"

    if not np.isfinite(s) or s <= 0.0:
        return "0 s"
    if s < 60.0:
        return f"{s:.2f} s"
    if s < 3600.0:
        return f"{s / 60.0:.2f} min"
    if s < 86400.0:
        return f"{s / 3600.0:.2f} h"
    return f"{s / 86400.0:.2f} d"

def format_count(value: Any) -> str:
    """Render integer-like counters with thousands separators."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"

def format_percent(value: Any, *, decimals: int = 2) -> str:
    """Render fractional values as percentages."""
    f = safe_float(value)
    if not np.isfinite(f):
        return "N/A"
    return f"{f * 100.0:.{decimals}f}%"

def format_days(seconds: Any, *, decimals: int = 3) -> str:
    """Render elapsed seconds as a day-based engineering quantity."""
    f = safe_float(seconds)
    if not np.isfinite(f):
        return "N/A"
    return f"{f / 86400.0:.{decimals}f} d"

def format_km(value: Any, *, decimals: int = 3) -> str:
    """Render kilometer-scale values with a consistent suffix."""
    f = safe_float(value)
    if not np.isfinite(f):
        return "N/A"
    return f"{f:.{decimals}f} km"

def format_sci_or_na(value: Any, *, decimals: int = 3) -> str:
    """Render scientific-notation diagnostics or return a neutral placeholder."""
    f = safe_float(value)
    if not np.isfinite(f):
        return "N/A"
    return f"{f:.{decimals}e}"
