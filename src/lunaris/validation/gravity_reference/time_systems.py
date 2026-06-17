"""Time-scale validation helpers."""

from __future__ import annotations

SUPPORTED_TIME_SCALES = {"UTC", "TAI", "TT", "TDB", "ET"}


def require_supported_time_scale(value: str) -> str:
    scale = str(value).strip().upper()
    if scale not in SUPPORTED_TIME_SCALES:
        raise ValueError(f"Unsupported time scale: {value!r}")
    return scale

