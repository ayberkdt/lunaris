"""A manoeuvre run keeps the requested output grid aligned to t0.

Regression test: ``_propagate_with_maneuvers`` called ``propagate`` per segment
with the segment's own ``t0_s``/``duration_s``, and the time grid re-spaced each
burn-shifted duration evenly: ``output_dt_s=60`` with a burn at 1000 s gave a
58.82 s spacing before the burn and 59.09 s after it, neither the requested
60 s nor aligned to t0. The output is now the no-burn grid plus the burn epoch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.propagation import ImpulsiveManeuver, ManeuverPlan, propagate


class _PointMassDynamics:
    grav = None
    ephem = None
    flags = None

    def build_rhs(self):
        def rhs(_t, y):
            state = np.asarray(y, dtype=np.float64)
            r = state[:3]
            rn = float(np.linalg.norm(r))
            out = np.empty_like(state)
            out[:3] = state[3:6]
            out[3:6] = -MU_MOON * r / rn**3
            return out

        return rhs


def _cfg(method: str) -> PropagatorConfig:
    return PropagatorConfig(
        method=method, rtol=1.0e-11, atol=1.0e-12, verbose=False, use_nyquist_max_step=False,
        user_max_step_s=10.0,
        events=EventConfig(detect_impact=False, enable_peri_apo_events=False),
    )


@pytest.mark.parametrize("method", ["DOP853", "RK4"])
def test_output_grid_is_the_no_burn_grid_plus_the_burn_epoch(method: str) -> None:
    r = float(R_MOON + 100_000.0)
    y0 = np.array([r, 0.0, 0.0, 0.0, math.sqrt(MU_MOON / r), 0.0])
    plan = ManeuverPlan((ImpulsiveManeuver(1000.0, (0.0, 5.0, 0.0), frame="ric"),))
    result = propagate(
        _PointMassDynamics(), y0, _cfg(method),
        time_cfg=TimeConfig(duration_s=3600.0, output_dt_s=60.0), maneuver_plan=plan,
    )
    expected = np.union1d(np.arange(0.0, 3600.0 + 1e-9, 60.0), [1000.0])
    np.testing.assert_allclose(result.t, expected, rtol=0.0, atol=1e-9)
