"""Shared output time-grid contract for single-run and ensemble propagation."""

from __future__ import annotations

import math

import numpy as np


def build_output_time_grid(
    duration_s: float,
    output_dt_s: float,
) -> tuple[np.ndarray, int, float]:
    """Build the canonical snapshot grid with an exact final epoch.

    Contract:
    - ``t[0] == 0.0``
    - ``t[-1] == duration_s``
    - ``np.all(np.diff(t) > 0.0)``
    - ``len(t) == n_snaps + 1``
    """
    duration = float(duration_s)
    out_dt = float(output_dt_s)
    if not (duration > 0.0):
        raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
    if not (out_dt > 0.0):
        raise ValueError(f"output_dt_s must be > 0, got {output_dt_s!r}")

    n_snaps = max(1, int(math.ceil(duration / out_dt)))
    t_out = np.linspace(0.0, duration, n_snaps + 1, dtype=np.float64)
    snap_interval_s = duration / n_snaps
    return t_out, n_snaps, snap_interval_s


__all__ = ["build_output_time_grid"]
