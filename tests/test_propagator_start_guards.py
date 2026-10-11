"""Pre-integration guards in ``propagate``.

- A start below the impact sphere with impact detection on never crosses the
  sphere downward, so the run integrated through the Moon and reported no
  impact; it is refused.
- A solver window outside the sampled ephemeris table silently froze the
  Sun/Earth vectors and the attitude at the end rows; it is refused.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("spiceypy")

from lunaris.common.constants import MU_MOON, R_MOON  # noqa: E402
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig  # noqa: E402
from lunaris.core.propagation.propagator import propagate  # noqa: E402
from lunaris.physics.ephemeris import EphemerisManager, EphemerisTables  # noqa: E402

MU = float(MU_MOON)
R = float(R_MOON)


class _PointMass:
    grav = None

    def __init__(self, ephem=None):
        self.ephem = ephem

    def build_rhs(self):
        def rhs(_t, y):
            y = np.asarray(y, dtype=np.float64)
            dy = np.empty_like(y)
            dy[:3] = y[3:6]
            dy[3:6] = -MU * y[:3] / float(np.linalg.norm(y[:3])) ** 3
            return dy

        return rhs


def _ephem(n: int, dt: float) -> EphemerisManager:
    q = np.zeros((n, 4), dtype=np.float64)
    q[:, 0] = 1.0
    return EphemerisManager.from_tables(EphemerisTables(
        dt_s=dt,
        t_tab_s=np.arange(n, dtype=np.float64) * dt,
        et0=0.0,
        q_i2f_tab=q,
        r_earth_tab_m=np.ones((n, 3)),
        r_sun_tab_m=np.ones((n, 3)),
        v_earth_tab_m_s=np.zeros((n, 3)),
        v_sun_tab_m_s=np.zeros((n, 3)),
        mu_earth_m3s2=3.986004418e14,
        mu_sun_m3s2=1.32712440018e20,
    ))


def _cfg(detect_impact: bool = True, impact_alt_km: float = 0.0) -> PropagatorConfig:
    return PropagatorConfig(
        method="DOP853",
        rtol=1e-10,
        atol=1e-6,
        verbose=False,
        compute_2body_baseline=False,
        use_nyquist_max_step=False,
        events=EventConfig(
            detect_impact=detect_impact,
            impact_alt_km=impact_alt_km,
            enable_peri_apo_events=False,
        ),
    )


def _state(r: float) -> np.ndarray:
    return np.array([r, 0.0, 0.0, 0.0, math.sqrt(MU / r), 0.0])


def test_start_below_impact_sphere_is_refused():
    tc = TimeConfig(duration_s=60.0, output_dt_s=20.0)
    with pytest.raises(ValueError, match="below the impact sphere"):
        propagate(_PointMass(), _state(R - 1_000.0), _cfg(), time_cfg=tc)
    # impact_alt_km raises the sphere: 3 km altitude is below a 5 km sphere.
    with pytest.raises(ValueError, match="below the impact sphere"):
        propagate(_PointMass(), _state(R + 3_000.0), _cfg(impact_alt_km=5.0), time_cfg=tc)


def test_start_on_or_above_sphere_or_without_detection_runs():
    tc = TimeConfig(duration_s=60.0, output_dt_s=20.0)
    propagate(_PointMass(), _state(R), _cfg(), time_cfg=tc)
    propagate(_PointMass(), _state(R + 100e3), _cfg(), time_cfg=tc)
    res = propagate(_PointMass(), _state(R - 1_000.0), _cfg(detect_impact=False), time_cfg=tc)
    assert not res.impacted


def test_window_outside_ephemeris_table_is_refused():
    # Table spans [0, 3600] s; the run asks for [3000, 4000] s.
    tc = TimeConfig(t0_s=3_000.0, duration_s=1_000.0, output_dt_s=100.0)
    with pytest.raises(ValueError, match="outside the ephemeris table"):
        propagate(_PointMass(_ephem(61, 60.0)), _state(R + 100e3), _cfg(), time_cfg=tc)


def test_window_inside_ephemeris_table_runs():
    tc = TimeConfig(t0_s=2_000.0, duration_s=1_600.0, output_dt_s=100.0)
    res = propagate(_PointMass(_ephem(61, 60.0)), _state(R + 100e3), _cfg(), time_cfg=tc)
    assert res.t[-1] == pytest.approx(3_600.0)
