"""A1: the engine must not fabricate Moon-fixed impact positions without ephemeris.

Without the inertial->Moon-fixed rotation table there is no physically
meaningful body-fixed impact position. The previous behavior treated inertial
coordinates as if they were body-fixed (identity rotation), which silently
produced a wrong geographic (lat/lon) impact distribution. The guard now returns
NaN so downstream lat/lon reporting is skipped instead.
"""

from __future__ import annotations

import numpy as np

from lunaris.core.monte_carlo_engine import _impact_positions_fixed


def test_impact_positions_fixed_all_nan_without_ephemeris() -> None:
    t_impact = np.array([10.0, np.nan, 20.0], dtype=np.float64)
    positions_inertial = np.array(
        [[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )

    out = _impact_positions_fixed(None, t_impact, positions_inertial)

    assert out.shape == positions_inertial.shape
    # No identity fabrication: every entry is NaN even where the inertial
    # position was finite.
    assert np.all(np.isnan(out))


def test_impact_positions_fixed_empty_when_no_impacts() -> None:
    t_impact = np.array([np.nan, np.nan], dtype=np.float64)
    positions_inertial = np.full((2, 3), np.nan, dtype=np.float64)

    out = _impact_positions_fixed(None, t_impact, positions_inertial)

    assert out.shape == (2, 3)
    assert np.all(np.isnan(out))
