# tests/test_events.py
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
    make_hybrid_impact_event,
    make_impact_event,
    make_is_eclipsed_solar,
    make_longitude_crossing_event,
    make_maneuver_trigger_event,
    make_max_eclipse_duration_event,
    make_node_crossing_event,
    make_occultation_event,
    make_periselene_event,
    make_radius_event,
    make_soi_event,
    make_solar_eclipse_event,
    make_stability_violation_event,
    make_target_flyover_event,
    make_terminator_crossing_event,
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


def test_default_events_without_periapo_and_with_topo():
    # add_periapo False -> only the impact event
    only_impact = default_events(R, add_periapo=False)
    assert len(only_impact) == 1
    # topo + r_i_to_bf -> the hybrid impact branch is selected
    topo = _FakeTopo(R)
    bundle = default_events(R, topo=topo, r_i_to_bf=_identity_rotation)
    assert len(bundle) == 3
    assert bundle[0].terminal is True


# =============================================================================
# SOI (geometric radius wrapper)
# =============================================================================

def test_soi_event_is_outward_radius_crossing():
    ev = make_soi_event(R + 1_000_000.0)
    assert ev(0.0, _state(R + 1_100_000.0)) > 0.0
    assert ev(0.0, _state(R + 900_000.0)) < 0.0
    assert ev.terminal is True
    assert ev.direction == pytest.approx(+1.0)


# =============================================================================
# Hybrid impact: every supported topo sampler interface
# =============================================================================

class _BilinearTopo:
    def __init__(self, r):
        self._r = float(r)

    def sample_bilinear(self, lat_deg, lon_deg, kind="radius_m"):
        return self._r


class _NearestTopo:
    def __init__(self, r):
        self._r = float(r)

    def sample_nearest(self, lat_deg, lon_deg, kind="radius_m"):
        return self._r


class _RadiansTopo:
    def __init__(self, r):
        self._r = float(r)

    def radius_m(self, lat_rad, lon_rad):  # radians interface
        return self._r


@pytest.mark.parametrize(
    "topo, kind",
    [
        (_BilinearTopo(R + 2_000.0), "bilinear"),
        (_NearestTopo(R + 2_000.0), "nearest"),
        (_RadiansTopo(R + 2_000.0), "bilinear"),
    ],
)
def test_hybrid_impact_supports_all_topo_interfaces(topo, kind):
    ev = make_hybrid_impact_event(
        R, 0.0, topo=topo, r_i_to_bf=_identity_rotation, switch_alt_m=11_000.0, kind=kind
    )
    # Near field: clearance vs terrain (R+1km)-(R+2km) = -1 km
    assert ev(0.0, _state(R + 1_000.0)) == pytest.approx(-1_000.0)


def test_hybrid_impact_raises_when_topo_has_no_usable_sampler():
    class _BadTopo:
        sample_bilinear = None  # present (so use_topo) but not callable

    with pytest.raises(AttributeError):
        make_hybrid_impact_event(
            R, 0.0, topo=_BadTopo(), r_i_to_bf=_identity_rotation, switch_alt_m=11_000.0
        )


# =============================================================================
# Solar eclipse / occultation geometry
# =============================================================================

def _sun_far_plus_x(t):
    return np.array([1.0e9, 0.0, 0.0])


def test_solar_eclipse_validation():
    with pytest.raises(ValueError):
        make_solar_eclipse_event(get_sun_vec_m=_sun_far_plus_x, R_moon_m=-1.0)
    with pytest.raises(ValueError):
        make_solar_eclipse_event(get_sun_vec_m=_sun_far_plus_x, R_moon_m=R, R_earth_m=0.0)
    with pytest.raises(ValueError):
        # no active occultor
        make_solar_eclipse_event(
            get_sun_vec_m=_sun_far_plus_x, R_moon_m=R, include_moon=False, include_earth=False
        )


def test_solar_eclipse_moon_shadow_sign_and_guard():
    ev = make_solar_eclipse_event(get_sun_vec_m=_sun_far_plus_x, R_moon_m=R, t_guard_s=5.0)
    # Before guard -> no root
    assert ev(1.0, _state(-(R + 2_000.0))) == pytest.approx(1.0)
    # Night side (behind Moon from Sun): LOS passes through Moon -> eclipsed (<0)
    assert ev(10.0, _state(-(R + 2_000.0))) < 0.0
    # Day side (between Moon and Sun): clear LOS (>0)
    assert ev(10.0, _state(R + 2_000.0)) > 0.0


def test_solar_eclipse_with_earth_occultor_branch():
    def earth_vec(t):
        return np.array([-1.0e8, 0.0, 0.0])  # Earth on -X

    ev = make_solar_eclipse_event(
        get_sun_vec_m=_sun_far_plus_x,
        R_moon_m=R,
        get_earth_vec_m=earth_vec,
        include_moon=True,
        include_earth=True,
        t_guard_s=0.0,
    )
    val = ev(10.0, _state(R + 2_000.0))
    assert np.isfinite(val)


def test_is_eclipsed_solar_predicate():
    pred = make_is_eclipsed_solar(get_sun_vec_m=_sun_far_plus_x, R_moon_m=R)
    assert pred(10.0, _state(-(R + 2_000.0))) is True
    assert pred(10.0, _state(R + 2_000.0)) is False


def test_max_eclipse_duration_accumulates_and_resets():
    with pytest.raises(ValueError):
        make_max_eclipse_duration_event(is_eclipsed=lambda t, y: True, max_duration_s=0.0)

    eclipsed = lambda t, y: bool(y[0] < 0.0)  # noqa: E731
    ev = make_max_eclipse_duration_event(is_eclipsed=eclipsed, max_duration_s=15.0)
    night = _state(-(R + 1_000.0))
    day = _state(R + 1_000.0)
    assert ev(0.0, night) == pytest.approx(15.0)   # first sample, dur=0
    assert ev(10.0, night) == pytest.approx(5.0)    # dur=10 -> 15-10
    assert ev(20.0, night) < 0.0                    # dur=20 -> triggered
    # Out-of-order probe must not mutate state
    assert ev(5.0, night) < 0.0
    # Leaving eclipse resets the accumulator
    assert ev(30.0, day) == pytest.approx(15.0)
    assert ev.terminal is True
    assert ev.direction == pytest.approx(-1.0)


def test_occultation_event_sign():
    def body_far(t):
        return np.array([1.0e9, 0.0, 0.0])

    ev = make_occultation_event(body="earth", get_body_vec_m=body_far, R_ref_m=R, t_guard_s=0.0)
    assert ev(10.0, _state(-(R + 2_000.0))) < 0.0  # Moon blocks LOS
    assert ev(10.0, _state(R + 2_000.0)) > 0.0     # clear


# =============================================================================
# Surface-geometry crossings (identity inertial<->body-fixed)
# =============================================================================

def _ident_xform(t, vec):
    return np.asarray(vec, dtype=np.float64)


def _sun_hat_x(t):
    return np.array([1.0, 0.0, 0.0])


def test_terminator_crossing_subsolar_vs_terminator():
    ev = make_terminator_crossing_event(sun_hat_i=_sun_hat_x, r_i_to_bf=_ident_xform, t_guard_s=0.0)
    # Subsolar point: r aligned with Sun -> dot = +1
    assert ev(10.0, _state(R + 100e3)) == pytest.approx(1.0)
    # On the terminator: r perpendicular to Sun -> dot = 0
    perp = np.array([0.0, R + 100e3, 0.0, 0.0, 0.0, 0.0])
    assert ev(10.0, perp) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("which, expected_dir", [("asc", +1.0), ("desc", -1.0), ("both", 0.0)])
def test_node_crossing_returns_z_and_direction(which, expected_dir):
    ev = make_node_crossing_event(which=which, t_guard_s=0.0)
    up = np.array([R + 100e3, 0.0, 1234.0, 0.0, 0.0, 0.0])
    assert ev(10.0, up) == pytest.approx(1234.0)
    assert ev.direction == pytest.approx(expected_dir)


def test_node_crossing_rejects_bad_which():
    with pytest.raises(ValueError):
        make_node_crossing_event(which="sideways")


def test_longitude_crossing_zero_at_target_meridian():
    ev = make_longitude_crossing_event(lon0_deg=0.0, r_i_to_bf=_ident_xform, t_guard_s=0.0)
    # Position on the +X meridian -> lon = 0 -> root value 0
    assert ev(10.0, _state(R + 100e3)) == pytest.approx(0.0, abs=1e-9)


def test_target_flyover_inside_and_outside_cap():
    ev = make_target_flyover_event(
        target_lat_deg=0.0, target_lon_deg=0.0, max_central_angle_deg=5.0,
        r_i_to_bf=_ident_xform, t_guard_s=0.0,
    )
    # Subsatellite point exactly over the target -> gamma=0 < cap -> negative (inside)
    assert ev(10.0, _state(R + 100e3)) < 0.0
    # Over the opposite side (~180 deg away) -> outside the cap -> positive
    assert ev(10.0, _state(-(R + 100e3))) > 0.0
    assert ev.direction == pytest.approx(-1.0)


# =============================================================================
# Operational events
# =============================================================================

def test_maneuver_trigger_is_time_only():
    ev = make_maneuver_trigger_event(t_trigger_s=100.0)
    assert ev(50.0, _state(R + 100e3)) < 0.0
    assert ev(150.0, _state(R + 100e3)) > 0.0
    assert ev.direction == pytest.approx(+1.0)


def test_stability_violation_validation():
    with pytest.raises(ValueError):
        make_stability_violation_event(mu=-1.0, R_ref_m=R)
    with pytest.raises(ValueError):
        make_stability_violation_event(mu=MU, R_ref_m=0.0)


def test_stability_violation_within_bounds_and_violation():
    r0 = R + 500_000.0
    v_circ = np.sqrt(MU / r0)
    circular = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
    # Loose bounds exercising every guard branch -> inside bounds (negative).
    ev_ok = make_stability_violation_event(
        mu=MU, R_ref_m=R, e_max=0.1, e_min=-0.1, i_max_deg=10.0, i_min_deg=-10.0,
        rp_min_alt_km=100.0, ra_max_alt_km=1000.0, t_guard_s=0.0,
    )
    assert ev_ok(10.0, circular) < 0.0
    # An impossible eccentricity ceiling forces a violation (positive), past the guard.
    ev_bad = make_stability_violation_event(mu=MU, R_ref_m=R, e_max=-0.001, t_guard_s=5.0)
    assert ev_bad(10.0, circular) > 0.0
    # Degenerate state (r=0) is treated as a violation.
    assert ev_bad(10.0, np.zeros(6)) == pytest.approx(1.0)
    # Time guard suppresses the root near t=0 (guard_value = -1 -> inside bounds).
    assert ev_bad(0.0, circular) == pytest.approx(-1.0)
