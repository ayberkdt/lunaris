# tests/test_events.py
# -*- coding: utf-8 -*-
"""
Direct unit tests for the SciPy-compatible event factories in ``lunaris.core.events``.

Events are scalar root functions ``g(t, y) -> float`` carrying ``.terminal`` and
``.direction`` attributes. A wrong sign convention or a missing attribute here is
a silent integration bug (the solver simply never stops, or stops at the wrong
crossing). These tests pin the sign conventions, the SciPy attribute contract,
and the terrain-aware hybrid-impact fallback using a fake topography provider.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.core.events import (
    default_events,
    make_altitude_crossing_event,
    make_aposelene_event,
    make_escape_event,
    make_impact_event,
    make_hybrid_impact_event,
    make_periselene_event,
    make_radius_event,
)

R = 1_737_400.0  # m, lunar mean radius
MU = 4.9048695e12  # m^3/s^2


def _state(r_x, vx=0.0, vy=0.0, vz=0.0):
    return np.array([r_x, 0.0, 0.0, vx, vy, vz], dtype=np.float64)


# =============================================================================
# Impact / altitude sign convention
# =============================================================================

def test_impact_event_sign_convention_and_attributes():
    ev = make_impact_event(R, 0.0)
    # Above the surface -> positive, below -> negative, at threshold -> ~0.
    assert ev(0.0, _state(R + 10_000.0)) > 0.0
    assert ev(0.0, _state(R - 5_000.0)) < 0.0
    assert ev(0.0, _state(R)) == pytest.approx(0.0, abs=1e-6)
    # SciPy contract: terminal stop on a downward crossing.
    assert ev.terminal is True
    assert ev.direction == pytest.approx(-1.0)


def test_impact_event_respects_nonzero_threshold_altitude():
    ev = make_impact_event(R, 5_000.0)  # impact declared 5 km above the sphere
    assert ev(0.0, _state(R + 6_000.0)) > 0.0   # 1 km above threshold
    assert ev(0.0, _state(R + 5_000.0)) == pytest.approx(0.0, abs=1e-6)
    assert ev(0.0, _state(R + 4_000.0)) < 0.0   # 1 km below threshold


def test_altitude_crossing_event_value_is_altitude_minus_target():
    ev = make_altitude_crossing_event(R, 100_000.0, direction=+1.0, terminal=False)
    assert ev(0.0, _state(R + 150_000.0)) == pytest.approx(50_000.0)
    assert ev(0.0, _state(R + 50_000.0)) == pytest.approx(-50_000.0)
    assert ev.direction == pytest.approx(+1.0)
    assert ev.terminal is False


def test_radius_event_value_and_soi_like_direction():
    ev = make_radius_event(R + 200_000.0, direction=+1.0, terminal=True)
    assert ev(0.0, _state(R + 250_000.0)) == pytest.approx(50_000.0)
    assert ev.terminal is True
    assert ev.direction == pytest.approx(+1.0)


# =============================================================================
# Hybrid impact: far-field sphere vs near-field terrain
# =============================================================================

class _FakeTopo:
    """Topography provider exposing the degree-based radius sampler contract."""

    def __init__(self, terrain_radius_m: float):
        self._r = float(terrain_radius_m)
        self.calls = 0

    def radius_m_deg(self, lat_deg: float, lon_deg: float) -> float:
        self.calls += 1
        return self._r


def _identity_rotation(t, r_i):
    return np.asarray(r_i, dtype=np.float64)


def test_hybrid_impact_uses_sphere_far_field_and_terrain_near_field():
    terrain = _FakeTopo(R + 2_000.0)  # local terrain sits 2 km above the mean sphere
    ev = make_hybrid_impact_event(
        R, 0.0, topo=terrain, r_i_to_bf=_identity_rotation, switch_alt_m=11_000.0,
    )
    assert ev.terminal is True
    assert ev.direction == pytest.approx(-1.0)

    # Far field (alt_ref 50 km > 11 km switch): pure sphere altitude, topo untouched.
    far = ev(0.0, _state(R + 50_000.0))
    assert far == pytest.approx(50_000.0)
    assert terrain.calls == 0

    # Near field (alt_ref 1 km <= switch): clearance vs terrain -> (R+1km)-(R+2km) = -1 km.
    near = ev(0.0, _state(R + 1_000.0))
    assert near == pytest.approx(-1_000.0)
    assert terrain.calls >= 1


def test_hybrid_impact_without_topo_falls_back_to_sphere():
    ev = make_hybrid_impact_event(R, 0.0, topo=None, r_i_to_bf=None, switch_alt_m=11_000.0)
    # No topo -> always the cheap reference-sphere altitude, even near the surface.
    assert ev(0.0, _state(R + 1_000.0)) == pytest.approx(1_000.0)
    assert ev(0.0, _state(R - 1_000.0)) == pytest.approx(-1_000.0)


# =============================================================================
# Periapsis / apoapsis (r . v sign changes)
# =============================================================================

def test_periselene_and_aposelene_directions_and_rdot_value():
    peri = make_periselene_event(t_guard_s=0.0)
    apo = make_aposelene_event(t_guard_s=0.0)
    # periapsis = r.v crossing 0 with + slope; apoapsis = - slope.
    assert peri.direction == pytest.approx(+1.0)
    assert apo.direction == pytest.approx(-1.0)
    assert peri.terminal is False and apo.terminal is False

    outbound = _state(R + 100e3, vx=1.0)  # r.v = (R+100km) * 1 > 0
    inbound = _state(R + 100e3, vx=-1.0)  # r.v < 0
    assert peri(10.0, outbound) > 0.0
    assert peri(10.0, inbound) < 0.0
    # Same r.v root function underlies both; the slope (direction) distinguishes them.
    assert apo(10.0, outbound) == pytest.approx(peri(10.0, outbound))


def test_periselene_time_guard_suppresses_root_near_t0():
    peri = make_periselene_event(t_guard_s=5.0)
    # Before the guard the function returns a constant +1 so no root is localized.
    assert peri(1.0, _state(R + 100e3, vx=-1.0)) == pytest.approx(1.0)
    # After the guard it reports the true r.v sign.
    assert peri(10.0, _state(R + 100e3, vx=-1.0)) < 0.0


# =============================================================================
# Escape (specific-energy) diagnostic
# =============================================================================

def test_escape_event_value_and_validation():
    ev = make_escape_event(MU)
    r0 = R + 100e3
    v_circ = np.sqrt(MU / r0)              # bound circular -> eps < 0
    v_esc = np.sqrt(2.0 * MU / r0)         # escape speed -> eps == 0
    assert ev(0.0, _state(r0, vy=v_circ)) < 0.0
    assert ev(0.0, _state(r0, vy=v_esc)) == pytest.approx(0.0, abs=1e-3)
    assert ev.direction == pytest.approx(+1.0)

    with pytest.raises(ValueError):
        make_escape_event(-1.0)


# =============================================================================
# default_events bundle
# =============================================================================

def test_default_events_bundle_is_scipy_compatible():
    events = default_events(R, impact_alt_km=5.0, add_periapo=True)
    assert len(events) == 3  # impact + peri + apo
    for ev in events:
        assert hasattr(ev, "terminal")
        assert hasattr(ev, "direction")
        assert isinstance(float(ev(100.0, _state(R + 100e3, vy=1500.0))), float)
    # Only the impact event is terminal in the default bundle.
    assert events[0].terminal is True
    assert events[1].terminal is False
    assert events[2].terminal is False
