"""The output-point cap must bound the grid that is actually built.

``build_output_time_grid`` makes ``ceil(D / dt) + 1`` points. ``TimeConfig``
estimated ``floor(D / dt) + 1``, so a config one point over the cap passed the
"strict" config-time check; and ``_clamp_output_dt`` picked
``dt = D / (cap - 1)``, whose ``ceil(D / dt)`` can round up to ``cap`` and
yield ``cap + 1`` points.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.time_grid_contract import build_output_time_grid
from lunaris.common.type_defs import TimeConfig
from lunaris.core.propagation.time_grid import _clamp_output_dt


def test_time_config_counts_points_like_the_grid() -> None:
    t, _, _ = build_output_time_grid(1000.0, 3.0)
    assert t.size == 335
    with pytest.raises(ValueError, match="max_points_cap"):
        TimeConfig(duration_s=1000.0, output_dt_s=3.0, max_points_cap=334)
    TimeConfig(duration_s=1000.0, output_dt_s=3.0, max_points_cap=335)
    # Exact multiples are unchanged: 1000 / 2 -> 501 points.
    TimeConfig(duration_s=1000.0, output_dt_s=2.0, max_points_cap=501)


def test_clamped_output_dt_never_exceeds_the_cap() -> None:
    over = []
    rng = np.random.default_rng(0)
    for cap in range(3, 400):
        for dur in rng.uniform(10.0, 1.0e7, size=30):
            dt = _clamp_output_dt(0.0, float(dur), 1.0e-3, cap, False)
            n = build_output_time_grid(float(dur), dt)[0].size
            if n > cap:
                over.append((cap, float(dur), n))
    assert not over, over[:5]
