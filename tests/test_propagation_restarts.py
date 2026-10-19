"""Shadow contacts stop the adaptive integrator instead of falling inside a step.

The SRP force is scaled by the conical shadow factor, which has a kink at each
contact of the apparent solar and occulter disks. An embedded Runge-Kutta pair
does not see a kink inside a step (its error estimate shares the corrupted
stages), so before the restart plan a shadowed SRP run drifted far from its own
converged answer. These tests rebuild the effect with a fixed Sun and Earth
(constant ephemeris tables) so they need no kernels; the oracle is the same
right-hand side integrated by SciPy with a 0.5 s step cap and no events at all,
which resolves the kinks by brute force and shares no restart logic.

Ported from Sidera (tests/core/test_propagation_restarts.py at 8ad733c5),
adapted to the lunar engine: Sidera's conical eclipse events do not exist in
Lunaris (its eclipse events are line-of-sight margins), so the user events that
share the restart roots are built from ``conical_shadow_boundaries`` itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from lunaris.common.constants import (
    AU,
    MU_EARTH,
    MU_MOON,
    MU_SUN,
    R_EARTH_MEAN,
    R_MOON_MEAN,
    R_SUN_MEAN,
)
from lunaris.common.type_defs import (
    EventConfig,
    PerturbationFlags,
    PropagatorConfig,
    SpacecraftProps,
    TimeConfig,
)
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.dynamics.contracts import RestartPlan, restart_event_family
from lunaris.core.events import (
    make_is_eclipsed_solar,
    make_max_eclipse_duration_event,
    make_solar_eclipse_event,
)
from lunaris.core.propagation import propagate
from lunaris.core.propagation.scipy_runner import _solve_with_restarts
from lunaris.physics.solar_effects import (
    PENUMBRA_STEPS,
    SRPConfig,
    conical_shadow_boundaries,
    moon_shadow_factor_conical,
    penumbra_max_step_s,
)

SUN = np.array([AU, 0.0, 0.0])
FAR_EARTH = np.array([0.0, 3.844e8, 0.0])  # off the Sun line: no Earth shadow near the Moon
RADIUS_M = R_MOON_MEAN + 100e3
SPEED = math.sqrt(MU_MOON / RADIUS_M)
PERIOD_S = 2.0 * math.pi * RADIUS_M / SPEED
CRAFT = SpacecraftProps(mass_kg=100.0, area_m2=100.0, cr=1.5)


class _FixedTables:
    """Constant Sun/Earth tables over [0, 1e6] s and an identity attitude."""

    def __init__(self, sun: np.ndarray, earth: np.ndarray) -> None:
        self.sun, self.earth = sun, earth

    def get_data_provider(self) -> dict[str, Any]:
        return {
            "dt_s": 1.0e6,
            "r_sun_tab_m": np.tile(self.sun, (2, 1)),
            "r_earth_tab_m": np.tile(self.earth, (2, 1)),
            "q_i2f_tab": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1)),
            "mu_sun_m3s2": float(MU_SUN),
            "mu_earth_m3s2": float(MU_EARTH),
        }


def _engine(*, sun: np.ndarray = SUN, earth: np.ndarray = FAR_EARTH, srp: SRPConfig | None = None,
            enable_srp: bool = True, restarts: bool = True) -> DynamicsEngine:
    engine = DynamicsEngine(
        sc_props=CRAFT,
        flags=PerturbationFlags(enable_sh=False, enable_srp=enable_srp),
        ephem_manager=_FixedTables(sun, earth),
        srp=srp if srp is not None else SRPConfig(),
        allow_identity_rotation=True,
    )
    if not restarts:
        engine.restart_plan = lambda: RestartPlan()  # type: ignore[method-assign]
    return engine


def _settings(rtol: float = 1e-11, atol: float = 1e-12, max_step: float | None = 60.0) -> PropagatorConfig:
    return PropagatorConfig(
        method="DOP853", rtol=rtol, atol=atol, verbose=False, use_nyquist_max_step=False,
        user_max_step_s=max_step, events=EventConfig(detect_impact=False, enable_peri_apo_events=False),
    )


def _timing(duration_s: float = PERIOD_S, output_dt_s: float = 60.0) -> TimeConfig:
    return TimeConfig(duration_s=duration_s, output_dt_s=output_dt_s, t0_s=0.0)


# Equatorial 100 km orbit starting on the sunlit side, moving towards the shadow behind -x.
Y0 = np.array([0.0, RADIUS_M, 0.0, -SPEED, 0.0, 0.0])


def _run(engine: DynamicsEngine, *, settings: PropagatorConfig | None = None, timing: TimeConfig | None = None,
         extra_events: list[Any] | None = None, y0: np.ndarray = Y0) -> Any:
    return propagate(engine, y0, settings or _settings(), time_cfg=timing or _timing(), extra_events=extra_events)


def _states(result: Any) -> np.ndarray:
    states = np.asarray(result.y, dtype=np.float64)
    return states if states.shape[1] == 6 else states.T


def _oracle(engine: DynamicsEngine, t: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Brute force: the engine's right-hand side, a 0.5 s step cap, no events."""
    rhs = engine.build_rhs()
    return solve_ivp(rhs, (t[0], t[-1]), y0, method="DOP853", rtol=1e-13, atol=1e-10, max_step=0.5, t_eval=t).y.T


def _miss(states: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(states[:, :3] - reference[:, :3], axis=1)))


@pytest.fixture(scope="module")
def moon_oracle() -> np.ndarray:
    t = np.asarray(_run(_engine()).t, dtype=np.float64)
    return _oracle(_engine(), t, Y0)


def test_restarts_bring_a_default_tolerance_run_to_the_converged_answer(moon_oracle: np.ndarray) -> None:
    # Measured 2026-10-20 (rtol 1e-11, 60 s steps, one revolution, SRP 6.8e-6 m/s^2):
    # 4.9e-6 m from this oracle and 1.1e-5 m from one with a 0.125 s cap (the
    # oracle's own error at 0.5 s is 5.4e-6 m); without restarts 1.3e-2 m.
    result = _run(_engine())
    plain = _run(_engine(restarts=False))
    error = _miss(_states(result), moon_oracle)
    plain_error = _miss(_states(plain), moon_oracle)
    # One shadow passage: penumbra and umbra contacts on entry and exit.
    assert int(getattr(result.ode, "restarts", 0)) == 4
    assert error < 3e-5
    # Without restarts the same settings miss the kinks.
    assert plain_error > 3e-3
    np.testing.assert_array_equal(result.t, plain.t)


def test_the_earth_shadow_contacts_restart_the_run_too() -> None:
    # The Earth between the Sun and the Moon, offset 6360 km sideways: the orbit
    # (1838 km radius) crosses the edge of the Earth's umbra (4578 km off the
    # axis at the Moon) and of its penumbra (8144 km). The Moon's own shadow is
    # off, so these are the Earth contacts alone, Earth-centred.
    earth = np.array([3.844e8, 6.36e6, 0.0])
    sun = earth + SUN
    srp = SRPConfig(enable_moon_eclipse=False, enable_earth_eclipse=True)
    result = _run(_engine(sun=sun, earth=earth, srp=srp))
    plain = _run(_engine(sun=sun, earth=earth, srp=srp, restarts=False))
    oracle = _oracle(_engine(sun=sun, earth=earth, srp=srp), np.asarray(result.t), Y0)
    # Measured 2026-10-20: 1.4e-7 m from the oracle (whose own error at 0.5 s is
    # 2.7e-7 m), without restarts 4.6e-5 m.
    assert int(getattr(result.ode, "restarts", 0)) == 4
    assert _miss(_states(result), oracle) < 2e-6
    assert _miss(_states(plain), oracle) > 2e-5


def test_a_run_without_radiation_pressure_or_shadows_has_no_restart_plan() -> None:
    for engine in (
        _engine(enable_srp=False),
        _engine(srp=SRPConfig(enable_moon_eclipse=False, enable_earth_eclipse=False)),
    ):
        plan = engine.restart_plan()
        assert plan.events == () and plan.max_step_s is None


def test_the_plan_covers_each_enabled_occulter() -> None:
    assert len(_engine(srp=SRPConfig(enable_moon_eclipse=True, enable_earth_eclipse=False)).restart_plan().events) == 2
    assert len(_engine(srp=SRPConfig(enable_moon_eclipse=True, enable_earth_eclipse=True)).restart_plan().events) == 4
    # No Earth table (a zero vector): the right-hand side has no Earth shadow, nor do the contacts.
    events = _engine(earth=np.zeros(3), srp=SRPConfig(enable_earth_eclipse=True)).restart_plan().events
    assert [events[k](0.0, Y0) for k in (2, 3)] == [1.0, 1.0]


def test_a_user_terminal_event_still_ends_a_restarted_run() -> None:
    def stop_at(t: float, y: np.ndarray) -> float:
        return t - 0.8 * PERIOD_S

    stop_at.terminal = True  # type: ignore[attr-defined]
    result = _run(_engine(), extra_events=[stop_at])
    assert result.stopped_early
    assert math.isclose(float(result.t[-1]), 0.8 * PERIOD_S, abs_tol=1e-6)
    assert int(getattr(result.ode, "restarts", 0)) >= 2


def _contact_event(index: int, *, terminal: bool = False, direction: float = 0.0) -> Any:
    """A user event on the Moon's shadow contacts, as an eclipse event would compute them.

    The same angles as ``conical_shadow_boundaries`` in other float64
    operations (Sidera's eclipse-event form), so its roots lie a rounding
    distance from the restart roots, on either side of them.
    """

    def g(t: float, y: np.ndarray) -> float:
        to_body = -np.asarray(y[:3], dtype=np.float64)
        to_sun = SUN - np.asarray(y[:3], dtype=np.float64)
        a = math.asin(R_SUN_MEAN / float(np.linalg.norm(to_sun)))
        b = math.asin(R_MOON_MEAN / float(np.linalg.norm(to_body)))
        c = math.atan2(float(np.linalg.norm(np.cross(to_body, to_sun))), float(np.dot(to_body, to_sun)))
        return (c - (a + b), c - abs(b - a))[index]

    g.terminal = terminal  # type: ignore[attr-defined]
    g.direction = direction  # type: ignore[attr-defined]
    return g


@pytest.mark.parametrize("index", [0, 1], ids=["penumbra", "umbra"])
def test_user_events_on_the_restart_contacts_are_all_reported(index: int) -> None:
    # A user event on the shadow contacts has its roots at the restart roots.
    # Each segment used to start it "on its surface" and drop the crossing
    # (Sidera: a GEO over three days lost 7 of 12).
    def contacts(restarts: bool) -> np.ndarray:
        result = _run(_engine(restarts=restarts), timing=_timing(2.0 * PERIOD_S), extra_events=[_contact_event(index)])
        assert (int(getattr(result.ode, "restarts", 0)) >= 4) == restarts
        return np.asarray(result.t_events[-1], dtype=np.float64)

    plain, restarted = contacts(False), contacts(True)
    # Two passages behind the Moon: entry and exit each.
    assert plain.size == 4
    assert restarted.size == plain.size
    # The trajectories differ by the kink error of the plain run; the contact times far less than a step.
    np.testing.assert_allclose(restarted, plain, rtol=0.0, atol=1e-3)


@pytest.mark.parametrize(
    ("beta_offset_rad", "start_in_shadow"),
    [(-0.02, False), (-0.005, False), (0.001, False), (0.003, False), (0.0, True), (0.0053, True)],
)
def test_contact_events_match_a_plain_run_on_grazing_and_shadowed_starts(
    beta_offset_rad: float, start_in_shadow: bool,
) -> None:
    # Sidera's randomized-stress geometries: orbit planes at the eclipse limit
    # (short passages, a penumbra without an umbra) and a start inside the umbra.
    if start_in_shadow:
        e = beta_offset_rad  # the eccentricity here: periapsis in the umbra
        r_p = RADIUS_M * (1.0 - e)
        position = np.array([-r_p, 0.0, 0.0])
        velocity = np.array([0.0, -math.sqrt(MU_MOON * (1.0 + e) / r_p), 0.0])
    else:
        beta = math.asin(R_MOON_MEAN / RADIUS_M) + beta_offset_rad
        normal = np.array([math.sin(beta), math.cos(beta), 0.0])
        along = np.cross(normal, [0.0, 0.0, 1.0])
        along /= np.linalg.norm(along)
        position, velocity = RADIUS_M * along, SPEED * np.cross(normal, along)
    y0 = np.concatenate([position, velocity])
    duration = 2.2 * PERIOD_S
    timing = _timing(duration, duration / 20.0)

    def contacts(restarts: bool) -> list[np.ndarray]:
        result = _run(_engine(restarts=restarts), timing=timing, y0=y0,
                      extra_events=[_contact_event(0), _contact_event(1)])
        return [np.asarray(times, dtype=np.float64) for times in result.t_events[-2:]]

    plain, restarted = contacts(False), contacts(True)
    for name, p, r in zip(("penumbra", "umbra"), plain, restarted, strict=True):
        assert p.size == r.size, (name, p, r)
        np.testing.assert_allclose(r, p, rtol=0.0, atol=1e-3, err_msg=name)


def _random_orbits(picks: tuple[int, ...]) -> list[np.ndarray]:
    """Circular orbits of a fixed random draw (inclination, node, phase, radius 1-3 lunar orbit radii)."""
    rng = np.random.default_rng(1)
    states = []
    for _ in range(max(picks) + 1):
        inc, node, phase = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        radius = RADIUS_M * rng.uniform(1.0, 3.0)
        speed = math.sqrt(MU_MOON / radius)
        normal = np.array([math.sin(inc) * math.sin(node), -math.sin(inc) * math.cos(node), math.cos(inc)])
        p = np.array([math.cos(node), math.sin(node), 0.0])
        q = np.cross(normal, p)
        states.append(np.concatenate([
            radius * (math.cos(phase) * p + math.sin(phase) * q),
            speed * (-math.sin(phase) * p + math.cos(phase) * q),
        ]))
    return [states[k] for k in picks]


@pytest.mark.parametrize("y0", _random_orbits((7, 9, 23)), ids=["draw7", "draw9", "draw23"])
def test_contact_events_on_restart_roots_are_reported_once(y0: np.ndarray) -> None:
    # Three draws of a 40-orbit randomized run (2026-10-20) on which a segment
    # started without the side past the root (the seeds of _side_after_root)
    # reported a contact twice: 3 of 294 contacts. With the seeds all 294
    # matched a run without restarts.
    radius = float(np.linalg.norm(y0[:3]))
    period = 2.0 * math.pi * radius / math.sqrt(MU_MOON / radius)
    timing = _timing(3.0 * period, 0.1 * period)

    def contacts(restarts: bool) -> list[np.ndarray]:
        result = _run(_engine(restarts=restarts), timing=timing, y0=y0,
                      extra_events=[_contact_event(0), _contact_event(1)])
        return [np.asarray(times, dtype=np.float64) for times in result.t_events[-2:]]

    plain, restarted = contacts(False), contacts(True)
    for name, p, r in zip(("penumbra", "umbra"), plain, restarted, strict=True):
        assert p.size == r.size == 6, (name, p, r)
        np.testing.assert_allclose(r, p, rtol=0.0, atol=1e-3, err_msg=name)


@pytest.mark.parametrize("index", [0, 1], ids=["penumbra", "umbra"])
def test_a_terminal_contact_event_on_a_restart_root_ends_the_run_there(index: int) -> None:
    def stop(restarts: bool) -> tuple[bool, float]:
        result = _run(_engine(restarts=restarts), extra_events=[_contact_event(index, terminal=True, direction=-1.0)])
        return bool(result.stopped_early), float(np.asarray(result.t, dtype=np.float64)[-1])

    plain_stopped, plain_end = stop(False)
    stopped, end = stop(True)
    assert plain_stopped and stopped
    assert end < PERIOD_S
    assert math.isclose(end, plain_end, abs_tol=1e-4)


def test_an_event_with_state_of_its_own_still_fires_with_restarts() -> None:
    # make_max_eclipse_duration_event remembers when the eclipse began and only
    # moves forward in time; a design that re-ran times it had seen left it frozen.
    def fired(restarts: bool) -> np.ndarray:
        eclipsed = make_is_eclipsed_solar(get_sun_vec_m=lambda t: SUN, R_moon_m=R_MOON_MEAN, include_earth=False)
        event = make_max_eclipse_duration_event(is_eclipsed=eclipsed, max_duration_s=600.0, terminal=False)
        result = _run(_engine(restarts=restarts), timing=_timing(2.0 * PERIOD_S), extra_events=[event])
        assert (int(getattr(result.ode, "restarts", 0)) >= 4) == restarts
        return np.asarray(result.t_events[-1], dtype=np.float64)

    plain, restarted = fired(False), fired(True)
    # Two passages, each in the shadow for longer than 600 s.
    assert plain.size == 2
    assert restarted.size == plain.size
    # Both note the entry at the first step end inside the shadow; their step ends differ: within one 60 s step.
    assert np.all(np.abs(restarted - plain) <= 60.0)


def test_the_eclipse_events_keep_their_times_with_restarts(moon_oracle: np.ndarray) -> None:
    # The line-of-sight eclipse events are untouched by the restarts: their
    # times move only by the plain run's kink error.
    def times(restarts: bool) -> np.ndarray:
        event = make_solar_eclipse_event(get_sun_vec_m=lambda t: SUN, R_moon_m=R_MOON_MEAN, include_earth=False)
        result = _run(_engine(restarts=restarts), extra_events=[event])
        return np.asarray(result.t_events[-1], dtype=np.float64)

    plain, restarted = times(False), times(True)
    assert plain.size == restarted.size == 2
    np.testing.assert_allclose(restarted, plain, rtol=0.0, atol=1e-3)


def _oscillator_solver(calls: dict[str, int], on_call: Any = None, method: str = "DOP853") -> Any:
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        if on_call is not None:
            on_call(t)
        return np.array([y[1], -y[0]])

    def solve_once(t0, t1, y, t_eval, events, cap=None, *, first_step=None):  # noqa: ANN001, ANN202
        return solve_ivp(rhs, (t0, t1), y, method=method, t_eval=t_eval, events=events or None, rtol=1e-9,
                         atol=1e-12, max_step=1.0 if cap is None else cap, first_step=first_step)

    return solve_once


def test_a_user_event_reading_outside_state_stops_a_restarted_run_as_a_single_solve() -> None:
    # The stop file's event is +1 or -1 by whether the file exists, so it is
    # seen only by the solve that is running when the file appears.
    flag = {"set": False}

    def stop(t: float, y: np.ndarray) -> float:
        return -1.0 if flag["set"] else 1.0

    stop.terminal = True  # type: ignore[attr-defined]
    stop.direction = 0.0  # type: ignore[attr-defined]

    def raise_flag(t: float) -> None:
        if t > 50.0:
            flag["set"] = True

    def never(t: float, y: np.ndarray) -> float:
        return 10.0 + float(y[0])

    grid = np.linspace(0.0, 200.0, 201)
    result = _solve_with_restarts(_oscillator_solver({"n": 0}, raise_flag), 0.0, 200.0, np.array([1.0, 0.0]), grid,
                                  [stop], [never])
    assert result.status == 1
    assert result.t_events[0].size == 1 and 48.0 < float(result.t_events[0][0]) < 52.0


def test_a_restart_plan_without_roots_costs_a_single_solve() -> None:
    def never(t: float, y: np.ndarray) -> float:
        return 10.0 + float(y[0])

    grid = np.linspace(0.0, 200.0, 3)
    with_plan, alone = {"n": 0}, {"n": 0}
    _solve_with_restarts(_oscillator_solver(with_plan), 0.0, 200.0, np.array([1.0, 0.0]), grid, [], [never])
    _oscillator_solver(alone)(0.0, 200.0, np.array([1.0, 0.0]), grid, [never])
    assert with_plan["n"] == alone["n"]


@pytest.mark.parametrize("method", ["DOP853", "BDF"])
def test_a_restart_root_on_or_a_few_ulp_past_a_step_end_neither_crashes_nor_loses_the_output(method: str) -> None:
    # brentq can return a step's own start as the root: the cut step then has no length.
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -y[0]])

    steps = solve_ivp(rhs, (0.0, 20.0), np.array([1.0, 0.0]), method=method, rtol=1e-9, atol=1e-12, max_step=1.0).t
    grid = np.linspace(0.0, 20.0, 5)
    for k in (3, 7):
        for ulps in (0, 1, 3, 40):
            root = float(steps[k] + ulps * np.spacing(steps[k]))
            result = _solve_with_restarts(_oscillator_solver({"n": 0}, method=method), 0.0, 20.0,
                                          np.array([1.0, 0.0]), grid, [], [lambda t, y, r=root: t - r])
            assert result.status == 0 and result.restarts == 1
            np.testing.assert_array_equal(result.t, grid)
            np.testing.assert_allclose(result.y[0], np.cos(grid), rtol=0.0, atol=1e-6)


def _behind(offset_m: float, distance_m: float = RADIUS_M) -> tuple[float, float, float]:
    """A point on the night side of the Moon, ``offset_m`` off the Sun-Moon line."""
    return -math.sqrt(distance_m**2 - offset_m**2), offset_m, 0.0


def test_the_contact_functions_vanish_where_the_shadow_factor_leaves_one_and_reaches_zero() -> None:
    def outer(offset: float) -> float:
        return conical_shadow_boundaries(*_behind(offset), *SUN, R_MOON_MEAN)[0]

    def inner(offset: float) -> float:
        return conical_shadow_boundaries(*_behind(offset), *SUN, R_MOON_MEAN)[1]

    penumbra_edge = brentq(outer, 1.6e6, RADIUS_M - 1.0, xtol=1e-6)
    umbra_edge = brentq(inner, 1.6e6, RADIUS_M - 1.0, xtol=1e-6)
    assert umbra_edge < penumbra_edge

    def nu(offset: float) -> float:
        return moon_shadow_factor_conical(*_behind(offset), *SUN, R_MOON_MEAN)

    step = 1.0  # m across a penumbra about 17 km wide
    assert nu(penumbra_edge + step) == 1.0
    assert 0.0 < nu(penumbra_edge - step) < 1.0
    assert nu(umbra_edge - step) == 0.0
    assert 0.0 < nu(umbra_edge + step) < 1.0


def test_the_penumbra_step_cap_applies_only_inside_the_penumbra() -> None:
    inner_edge = brentq(lambda o: conical_shadow_boundaries(*_behind(o), *SUN, R_MOON_MEAN)[1], 1.6e6, RADIUS_M - 1.0)
    outer_edge = brentq(lambda o: conical_shadow_boundaries(*_behind(o), *SUN, R_MOON_MEAN)[0], 1.6e6, RADIUS_M - 1.0)

    def at(offset: float) -> tuple[np.ndarray, np.ndarray]:
        r = np.array(_behind(offset))
        v = SPEED * np.array([-r[1], r[0], 0.0]) / RADIUS_M  # prograde, towards +y
        return r, v

    middle = 0.5 * (inner_edge + outer_edge)
    cap = penumbra_max_step_s(*at(middle), SUN, R_MOON_MEAN)
    duration = 2.0 * math.asin(R_SUN_MEAN / float(np.linalg.norm(SUN - at(middle)[0]))) / (SPEED / RADIUS_M)
    assert cap == pytest.approx(duration / PENUMBRA_STEPS, rel=1e-9)
    assert penumbra_max_step_s(*at(0.0), SUN, R_MOON_MEAN) is None  # umbra
    assert penumbra_max_step_s(np.array([RADIUS_M, 0.0, 0.0]), np.array([0.0, SPEED, 0.0]), SUN, R_MOON_MEAN) is None


def test_the_earth_contacts_are_earth_centred() -> None:
    # The Earth contacts are the Moon-centred kernel's own, shifted to the Earth.
    earth = np.array([3.844e8, 6.36e6, 0.0])
    sun = earth + SUN
    events = _engine(sun=sun, earth=earth, srp=SRPConfig(enable_earth_eclipse=True)).restart_plan().events
    y = np.array([1.0e6, -1.2e6, 3.0e5, 0.0, 0.0, 0.0])
    expected = conical_shadow_boundaries(*(y[:3] - earth), *(sun - earth), R_EARTH_MEAN)
    assert (events[2](5.0, y), events[3](5.0, y)) == expected


def test_a_restart_event_family_evaluates_once_per_state() -> None:
    calls = {"n": 0}

    def values(t: float, y: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return np.array([t, float(y[0]), -1.0])

    first, second, third = restart_event_family(values, 3)
    y = np.array([4.0, 5.0])
    assert (first(2.0, y), second(2.0, y), third(2.0, y)) == (2.0, 4.0, -1.0)
    assert calls["n"] == 1
    assert second(3.0, np.array([7.0])) == 7.0
    assert calls["n"] == 2
