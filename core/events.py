# ST_LRPS/core/events.py
# -*- coding: utf-8 -*-
"""
core.events
===========

SciPy `solve_ivp`-compatible event factories.

Each factory returns a callable root function:

    g(t, y) -> float

where the packed Cartesian state is assumed to be:

    y = [rx, ry, rz, vx, vy, vz]   (SI units)

The returned callable may expose SciPy event attributes:

    g.terminal  : bool   # stop integration when the event triggers
    g.direction : float  # 0: any crossing, +1: upward only, -1: downward only


What this module provides
-------------------------
This file focuses on “mission-analysis” style events that are commonly required
around an orbital propagator:

Geometry / safety
- Impact / altitude threshold crossing (sphere or terrain-aware hybrid)
- Generic radius / altitude crossings (utility building blocks)
- Simple SOI boundary (geometric, not dynamical)

Orbit characterization
- Periapsis / apoapsis detection via r·v = 0
- Two-body escape diagnostic via specific orbital energy (ε > 0)

Illumination / visibility
- Solar eclipse (hard-shadow, point Sun) via line-of-sight sphere intersection
- Max continuous eclipse duration guard (battery / thermal survival constraint)
- Occultation (Moon blocks line-of-sight to Earth/Sun or other target)

Surface / ground-track
- Terminator crossing (day/night boundary in Moon-fixed frame)
- Ascending / descending node crossings (z=0 reference plane)
- Fixed-frame longitude crossings (ground-track synchronization)
- Target flyover (within an angular radius of a surface target)

Operational hooks
- Maneuver trigger (time-based placeholder; easy to replace with other triggers)
- “Stability” violation guard (osculating e/i/rp/ra bounds from two-body elements)


Design notes
------------
SciPy events are scalar root functions. Many operational concepts are naturally
boolean (eclipsed/not-eclipsed, occulted/not-occulted). In those cases we expose
a signed *margin* function:

    margin < 0  => condition is active (e.g., eclipsed / occulted)
    margin > 0  => clear / inactive

so that entry/exit corresponds to a sign change and `solve_ivp` can locate the
boundary with root finding.

Several event types require geometry beyond (t, y), such as Sun/Earth vectors or
an inertial→Moon-fixed rotation. To keep the solver interface unchanged, those
dependencies are injected as callables captured by closures (e.g., `get_sun_vec_m(t)`).


Example
-------
    from scipy.integrate import solve_ivp

    # Provide time-dependent geometry via callables
    sun_hat_i = lambda t: get_sun_dir_hat_i(t)         # unit 3-vector in inertial frame
    r_i_to_bf = lambda t, r_i: rotate_i_to_fixed(t, r_i)  # inertial -> Moon-fixed

    ev_term = make_terminator_crossing_event(sun_hat_i=sun_hat_i, r_i_to_bf=r_i_to_bf)
    ev_lon0 = make_longitude_crossing_event(lon0_deg=0.0, r_i_to_bf=r_i_to_bf)

    sol = solve_ivp(rhs, (0.0, tf), y0, events=[ev_term, ev_lon0])

"""


# =============================================================================
# 0.                               IMPORTS
# =============================================================================

from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Optional

from common.type_defs import F64
from core.state import as_vec3



# =============================================================================
# 1.                              HELPERS
# =============================================================================


def _unit(v: ArrayLike, eps: float = 1e-12) -> NDArray[np.float64]:
    """
    Return a unit vector. If norm <= eps, returns a default axis (1,0,0).
    (Consider raising instead if you want strict behavior.)
    """
    vv = as_vec3(v, name="v")
    n = float(np.linalg.norm(vv))
    if not math.isfinite(n) or n <= eps:
        return np.array([1.0, 0.0, 0.0], dtype=F64)
    return vv / n


def _wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _latlon_from_r_bf(r_bf: ArrayLike) -> tuple[float, float]:
    """Return (lat_rad, lon_rad) from Moon-fixed position vector."""
    r = as_vec3(r_bf, name="r_bf")
    rn = float(np.linalg.norm(r))
    if not math.isfinite(rn) or rn <= 0.0:
        return 0.0, 0.0
    x, y, z = (r / rn).tolist()
    z = max(-1.0, min(1.0, z))
    lat = math.asin(z)
    lon = math.atan2(y, x)
    return float(lat), float(lon)


def _central_angle(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Spherical central angle between two (lat,lon) points (radians)."""
    dlon = float(lon2) - float(lon1)
    c = (math.sin(lat1) * math.sin(lat2) +
         math.cos(lat1) * math.cos(lat2) * math.cos(dlon))
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def _los_margin_sphere(
    p_sc: ArrayLike,
    p_body: ArrayLike,
    center: ArrayLike,
    R_m: float,
) -> float:
    """
    Signed margin for LOS occultation against a sphere:
      margin = d_min - R_m
    where d_min is the minimum distance from `center` to the segment [p_sc, p_body].

    margin < 0 => occulted
    """
    R = float(R_m)
    p1 = as_vec3(p_sc, name="p_sc")
    p2 = as_vec3(p_body, name="p_body")
    c = as_vec3(center, name="center")

    d = p2 - p1
    dd = float(np.dot(d, d))
    if dd <= 0.0:
        return float(np.linalg.norm(p1 - c) - R)

    u = float(np.dot(c - p1, d)) / dd
    u = max(0.0, min(1.0, u))
    closest = p1 + u * d
    return float(np.linalg.norm(closest - c) - R)


def _los_blocked_by_sphere(p_sc: ArrayLike, p_body: ArrayLike, R_m: float) -> float:
    """LOS margin vs sphere centered at origin."""
    return _los_margin_sphere(p_sc, p_body, center=(0.0, 0.0, 0.0), R_m=R_m)


def _los_blocked_by_sphere_center(
    p_sc: ArrayLike, p_body: ArrayLike, center: ArrayLike, R_m: float
) -> float:
    """LOS margin vs sphere centered at `center`."""
    return _los_margin_sphere(p_sc, p_body, center=center, R_m=R_m)



# =============================================================================
# 2.                      IMPACT & GEOMETRY EVENTS
# =============================================================================

EventFn = Callable[[float, NDArray[np.float64]], float]

def _set_event_props(fn: EventFn, *, terminal: bool, direction: float) -> EventFn:
    """Attach SciPy-style event attributes and return the same callable."""
    fn.terminal = bool(terminal)   # type: ignore[attr-defined]
    fn.direction = float(direction)  # type: ignore[attr-defined]
    return fn


def make_altitude_crossing_event(
    R_ref_m: float,
    target_alt_m: float,
    *,
    direction: float = 0.0,
    terminal: bool = False,
) -> EventFn:
    """
    Generic altitude crossing event:

        g(t, y) = (||r|| - R_ref_m) - target_alt_m

    direction:
      0  -> any crossing
      +1 -> upward crossing only
      -1 -> downward crossing only
    """
    R_ref = float(R_ref_m)
    target_alt = float(target_alt_m)

    def alt_event(t: float, y: NDArray[np.float64]) -> float:
        r_i = y[:3]
        r_m = float(np.linalg.norm(r_i))
        return (r_m - R_ref) - target_alt

    return _set_event_props(alt_event, terminal=terminal, direction=direction)


def make_impact_event(
    R_ref_m: float,
    impact_alt_m: float,
    *,
    terminal: bool = True,
) -> EventFn:
    """
    Stop when altitude drops to impact_alt_m above reference radius.

    Equivalent to a downward altitude crossing:
      direction = -1, terminal = terminal
    """
    return make_altitude_crossing_event(
        R_ref_m,
        impact_alt_m,
        direction=-1.0,
        terminal=terminal,
    )


def make_hybrid_impact_event(
    R_ref_m: float,
    impact_alt_m: float,
    *,
    topo: Optional[object] = None,
    r_i_to_bf: Optional[Callable[[float, NDArray[np.float64]], NDArray[np.float64]]] = None,
    switch_alt_m: float = 11_000.0,
    kind: str = "bilinear",
    terminal: bool = True,
) -> EventFn:
    """
    Hybrid impact event:

      - Far-field (alt_ref > switch_alt): uses reference-sphere altitude (fast).
      - Near-field (alt_ref <= switch_alt): uses terrain clearance (accurate) if `topo` is provided.

    Supported topo interfaces (any one is sufficient)
    -------------------------------------------------
    * Raster/grid style (degrees):
        topo.sample_bilinear(lat_deg, lon_deg, kind="radius_m")  # preferred continuous sampler
        topo.sample_nearest(lat_deg, lon_deg, kind="radius_m")   # optional
    * Provider style (degrees):
        topo.radius_m_deg(lat_deg, lon_deg) -> radius_m
    * Provider style (radians):
        topo.radius_m(lat_rad, lon_rad) -> radius_m

    Notes
    -----
    - When topo is used, the inertial position is first mapped to the body-fixed frame via `r_i_to_bf(t, r_i)`.
    - The sub-point lat/lon is computed from the body-fixed position (no dependency on topo implementing
      a lat/lon helper).
    - Root function triggers on a downward crossing of:
        alt_terrain(t) - impact_alt_m
      where alt_terrain = ||r|| - terrain_radius(lat, lon).
    """
    R_ref = float(R_ref_m)
    impact_alt = float(impact_alt_m)
    switch_alt = float(switch_alt_m)
    kind_l = str(kind).lower().strip()

    # Is topo usable at all?
    use_topo = (
        topo is not None
        and r_i_to_bf is not None
        and (
            hasattr(topo, "sample_bilinear")
            or hasattr(topo, "sample_nearest")
            or hasattr(topo, "radius_m_deg")
            or hasattr(topo, "radius_m")
        )
    )

    # Resolve a radius sampler once (avoid repeated hasattr() checks inside the event)
    radius_sampler: Optional[Callable[[float, float], float]] = None
    if use_topo:
        # Choose "nearest" only if explicitly requested; otherwise prefer continuous sampling.
        prefer_nearest = ("near" in kind_l)

        if hasattr(topo, "sample_bilinear") or hasattr(topo, "sample_nearest"):
            bilinear = getattr(topo, "sample_bilinear", None)
            nearest = getattr(topo, "sample_nearest", None)

            if prefer_nearest and callable(nearest):
                def _r_m(lat_rad: float, lon_rad: float) -> float:
                    lat_deg = math.degrees(lat_rad)
                    lon_deg = math.degrees(lon_rad) % 360.0
                    return float(nearest(lat_deg, lon_deg, kind="radius_m"))  # type: ignore[misc]
                radius_sampler = _r_m
            elif callable(bilinear):
                def _r_m(lat_rad: float, lon_rad: float) -> float:
                    lat_deg = math.degrees(lat_rad)
                    lon_deg = math.degrees(lon_rad) % 360.0
                    return float(bilinear(lat_deg, lon_deg, kind="radius_m"))  # type: ignore[misc]
                radius_sampler = _r_m
            elif callable(nearest):
                def _r_m(lat_rad: float, lon_rad: float) -> float:
                    lat_deg = math.degrees(lat_rad)
                    lon_deg = math.degrees(lon_rad) % 360.0
                    return float(nearest(lat_deg, lon_deg, kind="radius_m"))  # type: ignore[misc]
                radius_sampler = _r_m

        if radius_sampler is None and hasattr(topo, "radius_m_deg"):
            fn = getattr(topo, "radius_m_deg")
            if callable(fn):
                def _r_m(lat_rad: float, lon_rad: float) -> float:
                    lat_deg = math.degrees(lat_rad)
                    lon_deg = math.degrees(lon_rad) % 360.0
                    return float(fn(lat_deg, lon_deg))  # type: ignore[misc]
                radius_sampler = _r_m

        if radius_sampler is None and hasattr(topo, "radius_m"):
            fn = getattr(topo, "radius_m")
            if callable(fn):
                def _r_m(lat_rad: float, lon_rad: float) -> float:
                    # Expect radians
                    return float(fn(float(lat_rad), float(lon_rad)))  # type: ignore[misc]
                radius_sampler = _r_m

        if radius_sampler is None:
            raise AttributeError(
                "Hybrid impact event: topo provided but no usable radius sampler found. "
                "Expected one of: sample_bilinear/sample_nearest (deg), radius_m_deg (deg), radius_m (rad)."
            )

    def impact_event(t: float, y: NDArray[np.float64]) -> float:
        r_i = y[:3]
        r_m = float(np.linalg.norm(r_i))
        alt_ref = r_m - R_ref

        # Cheap branch (or topo unavailable)
        if (not use_topo) or (alt_ref > switch_alt):
            return alt_ref - impact_alt

        # Near the surface: evaluate terrain radius at current sub-point
        r_bf = r_i_to_bf(t, r_i)  # type: ignore[misc]
        lat_rad, lon_rad = _latlon_from_r_bf(r_bf)
        terrain_r_m = float(radius_sampler(lat_rad, lon_rad))  # type: ignore[misc]
        alt_terrain = r_m - terrain_r_m
        return alt_terrain - impact_alt

    return _set_event_props(impact_event, terminal=terminal, direction=-1.0)


def make_radius_event(
    target_r_m: float,
    *,
    direction: float = 0.0,
    terminal: bool = False,
) -> EventFn:
    """
    Generic radius crossing event:

        g(t, y) = ||r|| - target_r_m

    direction:
      0  -> any crossing
      +1 -> outward crossing only
      -1 -> inward crossing only
    """
    target_r = float(target_r_m)

    def radius_event(t: float, y: NDArray[np.float64]) -> float:
        r_m = float(np.linalg.norm(y[:3]))
        return r_m - target_r

    return _set_event_props(radius_event, terminal=terminal, direction=direction)


def make_soi_event(soi_r_m: float, *, terminal: bool = True) -> EventFn:
    """
    Sphere-of-influence boundary (geometric): triggers when ||r|| exceeds soi_r_m (outward).

    Note: This is a simple geometric stop; it is NOT a rigorous dynamical SOI condition.
    """
    return make_radius_event(float(soi_r_m), direction=+1.0, terminal=terminal)



# =============================================================================
# 3.                         ORBITAL EVENTS
# =============================================================================

def _make_rdot_event(*, t_guard_s: float, direction: float, terminal: bool) -> EventFn:
    """
    Event for r·v = 0 (radial velocity zero crossing).
    direction=+1 => periapsis (from - to +)
    direction=-1 => apoapsis (from + to -)
    """
    t_guard = float(t_guard_s)

    def rdot_event(t: float, y: NDArray[np.float64]) -> float:
        if t < t_guard:
            return 1.0  # keep away from root near t=0
        r = y[:3]
        v = y[3:]
        return float(np.dot(r, v))

    return _set_event_props(rdot_event, terminal=terminal, direction=direction)


def make_periselene_event(t_guard_s: float = 1.0, terminal: bool = False) -> EventFn:
    """Periapsis when r·v crosses 0 with + slope (from - to +)."""
    return _make_rdot_event(t_guard_s=t_guard_s, direction=+1.0, terminal=terminal)


def make_aposelene_event(t_guard_s: float = 1.0, terminal: bool = False) -> EventFn:
    """Apoapsis when r·v crosses 0 with - slope (from + to -)."""
    return _make_rdot_event(t_guard_s=t_guard_s, direction=-1.0, terminal=terminal)


def make_escape_event(mu: float, terminal: bool = True, *, eps_r_guard_m: float = 1e-9) -> EventFn:
    """
    Escape condition (two-body diagnostic) based on specific orbital energy:

        eps = v^2/2 - mu/r

    Triggers when eps crosses 0 upward.

    Note: With strong perturbations, this is a diagnostic rather than a strict guarantee.
    """
    mu_ = float(mu)
    if not math.isfinite(mu_) or mu_ <= 0.0:
        raise ValueError("mu must be finite and > 0.")
    r_guard = float(eps_r_guard_m)

    def escape_event(t: float, y: NDArray[np.float64]) -> float:
        r = float(np.linalg.norm(y[:3]))
        v2 = float(np.dot(y[3:], y[3:]))
        r_eff = max(r, r_guard)
        return 0.5 * v2 - mu_ / r_eff

    return _set_event_props(escape_event, terminal=terminal, direction=+1.0)



# =============================================================================
# 4.                           ECLIPSE EVENTS
# =============================================================================

VecFn = Callable[[float], ArrayLike]

def make_solar_eclipse_event(
    *,
    get_sun_vec_m: VecFn,
    R_moon_m: float,
    get_earth_vec_m: Optional[VecFn] = None,
    R_earth_m: float = 6_378_137.0,
    include_moon: bool = True,
    include_earth: bool = True,
    terminal: bool = False,
    direction: float = 0.0,
    t_guard_s: float = 1.0,
) -> EventFn:
    """
    Solar eclipse (hard-shadow) event: LOS(SC->Sun) occulted by Moon and/or Earth.

    Root function:
      g(t,y) = min(moon_margin, earth_margin)
      margin < 0 => eclipsed

    Notes:
    - Sun is treated as a point target at position get_sun_vec_m(t) (Moon-centered inertial).
    - Earth occultation requires get_earth_vec_m(t). If None, Earth branch is disabled.
    """
    R_moon = float(R_moon_m)
    R_earth = float(R_earth_m)
    t_guard = float(t_guard_s)

    if not (math.isfinite(R_moon) and R_moon > 0.0):
        raise ValueError("R_moon_m must be finite and > 0.")
    if not (math.isfinite(R_earth) and R_earth > 0.0):
        raise ValueError("R_earth_m must be finite and > 0.")

    use_moon = bool(include_moon)
    use_earth = bool(include_earth) and (get_earth_vec_m is not None)

    if not (use_moon or use_earth):
        raise ValueError(
            "Solar eclipse event has no active occultor. "
            "Enable include_moon/include_earth and provide get_earth_vec_m if Earth is enabled."
        )

    def ecl_event(t: float, y: NDArray[np.float64]) -> float:
        if t < t_guard:
            return 1.0

        r_sc = as_vec3(y[:3], name="r_sc")
        r_sun = as_vec3(get_sun_vec_m(t), name="r_sun")

        best = math.inf

        if use_moon:
            best = min(best, float(_los_blocked_by_sphere(r_sc, r_sun, R_moon)))

        if use_earth:
            r_earth = as_vec3(get_earth_vec_m(t), name="r_earth")  # type: ignore[misc]
            best = min(best, float(_los_blocked_by_sphere_center(r_sc, r_sun, r_earth, R_earth)))

        return float(best)

    return _set_event_props(ecl_event, terminal=terminal, direction=direction)


def make_is_eclipsed_solar(
    *,
    get_sun_vec_m: VecFn,
    R_moon_m: float,
    get_earth_vec_m: Optional[VecFn] = None,
    R_earth_m: float = 6_378_137.0,
    include_moon: bool = True,
    include_earth: bool = True,
    t_guard_s: float = 0.0,
) -> Callable[[float, NDArray[np.float64]], bool]:
    """
    Convenience boolean predicate:
        is_eclipsed(t, y) -> bool
    """
    ev = make_solar_eclipse_event(
        get_sun_vec_m=get_sun_vec_m,
        R_moon_m=R_moon_m,
        get_earth_vec_m=get_earth_vec_m,
        R_earth_m=R_earth_m,
        include_moon=include_moon,
        include_earth=include_earth,
        terminal=False,
        direction=0.0,
        t_guard_s=float(t_guard_s),
    )

    def is_eclipsed(t: float, y: NDArray[np.float64]) -> bool:
        return ev(t, y) < 0.0

    return is_eclipsed


def make_max_eclipse_duration_event(
    *,
    is_eclipsed: Callable[[float, NDArray[np.float64]], bool],
    max_duration_s: float,
    terminal: bool = True,
) -> EventFn:
    """
    Guard on *continuous* eclipse time. Triggers when contiguous eclipse duration exceeds max_duration_s.

    Implementation notes:
    - Uses closure state (last_t, in_ecl, dur).
    - If solver probes non-monotone times during root localization, we do NOT update state.
      (This avoids inconsistent toggles.)
    """
    max_dur = float(max_duration_s)
    if not (math.isfinite(max_dur) and max_dur > 0.0):
        raise ValueError("max_duration_s must be finite and > 0.")

    last_t = -math.inf
    in_ecl = False
    dur = 0.0

    def ecl_dur_event(t: float, y: NDArray[np.float64]) -> float:
        nonlocal last_t, in_ecl, dur

        tt = float(t)

        # If solver calls out-of-order times, avoid state mutation (keep monotone behavior).
        if tt < last_t:
            return max_dur - dur

        dt = 0.0 if not math.isfinite(last_t) else (tt - last_t)
        last_t = tt

        ecl = bool(is_eclipsed(tt, y))

        if ecl:
            dur = (dur + dt) if in_ecl else 0.0
            in_ecl = True
        else:
            in_ecl = False
            dur = 0.0

        return max_dur - dur

    return _set_event_props(ecl_dur_event, terminal=terminal, direction=-1.0)



# =============================================================================
# 5.                       SURFACE GEOMETRY EVENTS
# =============================================================================

XformFn_t = Callable[[float, ArrayLike], ArrayLike]

def _with_t_guard(fn: EventFn, *, t_guard_s: float, guard_value: float = 1.0) -> EventFn:
    """Wrap an event to avoid spurious root detection near t=0."""
    t_guard = float(t_guard_s)

    def wrapped(t: float, y: NDArray[np.float64]) -> float:
        if t < t_guard:
            return float(guard_value)
        return fn(t, y)

    return wrapped


def make_terminator_crossing_event(
    *,
    sun_hat_i: VecFn_t,
    r_i_to_bf: XformFn_t,
    terminal: bool = False,
    direction: float = 0.0,
    t_guard_s: float = 1.0,
) -> EventFn:
    """
    Terminator crossing (day/night boundary) event.

    Condition (Moon-fixed): dot(r_hat_fixed, sun_hat_fixed) = 0
    """
    def core(t: float, y: NDArray[np.float64]) -> float:
        r_i = as_vec3(y[:3], name="r_i")
        r_bf = as_vec3(r_i_to_bf(t, r_i), name="r_bf")

        sun_i = as_vec3(sun_hat_i(t), name="sun_hat_i")
        # Treat r_i_to_bf as a pure rotation (direction-vector transform).
        sun_bf = as_vec3(r_i_to_bf(t, sun_i), name="sun_hat_bf")

        return float(np.dot(_unit(r_bf), _unit(sun_bf)))

    ev = _with_t_guard(core, t_guard_s=t_guard_s)
    return _set_event_props(ev, terminal=terminal, direction=direction)


def make_node_crossing_event(
    *,
    r_i_to_ref: Optional[XformFn_t] = None,
    which: str = "both",
    terminal: bool = False,
    t_guard_s: float = 1.0,
) -> EventFn:
    """
    Ascending/descending node crossing w.r.t. reference plane z=0.

    which:
      "asc"  -> z crosses 0 from - to + (direction=+1)
      "desc" -> z crosses 0 from + to - (direction=-1)
      "both" -> any crossing (direction=0)
    """
    w = str(which).lower().strip()
    dir_map = {"asc": +1.0, "desc": -1.0, "both": 0.0}
    if w not in dir_map:
        raise ValueError("which must be 'asc', 'desc', or 'both'")

    def core(t: float, y: NDArray[np.float64]) -> float:
        r_i = as_vec3(y[:3], name="r_i")
        r_ref = as_vec3(r_i_to_ref(t, r_i), name="r_ref") if r_i_to_ref else r_i
        return float(r_ref[2])

    ev = _with_t_guard(core, t_guard_s=t_guard_s)
    return _set_event_props(ev, terminal=terminal, direction=dir_map[w])


def make_longitude_crossing_event(
    *,
    lon0_deg: float,
    r_i_to_bf: XformFn_t,
    terminal: bool = False,
    direction: float = 0.0,
    t_guard_s: float = 1.0,
) -> EventFn:
    """
    Moon-fixed longitude crossing event.

    Root function: wrap_pi(lon(t) - lon0)  [radians]
    """
    lon0 = math.radians(float(lon0_deg))

    def core(t: float, y: NDArray[np.float64]) -> float:
        r_i = as_vec3(y[:3], name="r_i")
        r_bf = as_vec3(r_i_to_bf(t, r_i), name="r_bf")
        _, lon = _latlon_from_r_bf(r_bf)
        return float(_wrap_pi(lon - lon0))

    ev = _with_t_guard(core, t_guard_s=t_guard_s)
    return _set_event_props(ev, terminal=terminal, direction=direction)


def make_target_flyover_event(
    *,
    target_lat_deg: float,
    target_lon_deg: float,
    max_central_angle_deg: float,
    r_i_to_bf: XformFn_t,
    terminal: bool = False,
    t_guard_s: float = 1.0,
) -> EventFn:
    """
    Target flyover: triggers when subsatellite point enters a spherical cap.

    Root: gamma(t) - gamma_max  (downward crossing => direction = -1)
    """
    lat0 = math.radians(float(target_lat_deg))
    lon0 = math.radians(float(target_lon_deg))
    gam_max = math.radians(float(max_central_angle_deg))

    def core(t: float, y: NDArray[np.float64]) -> float:
        r_i = as_vec3(y[:3], name="r_i")
        r_bf = as_vec3(r_i_to_bf(t, r_i), name="r_bf")
        lat, lon = _latlon_from_r_bf(r_bf)
        gamma = _central_angle(lat, lon, lat0, lon0)
        return float(gamma - gam_max)

    ev = _with_t_guard(core, t_guard_s=t_guard_s)
    return _set_event_props(ev, terminal=terminal, direction=-1.0)


# =============================================================================
# 6.                         COMMUNICATION EVENTS
# =============================================================================
VecFn_t = Callable[[float], ArrayLike]

def make_occultation_event(
    *,
    body: str,
    get_body_vec_m: VecFn_t,
    R_ref_m: float,
    terminal: bool = False,
    direction: float = 0.0,
    t_guard_s: float = 1.0,
) -> EventFn:
    """
    Occultation: Moon blocks LOS between spacecraft and a body (Earth/Sun).

    margin = d_min(segment(sc->body), origin) - R_ref_m
      margin < 0 => occulted
      margin > 0 => clear LOS
    """
    label = str(body)
    R_ref = float(R_ref_m)

    def core(t: float, y: NDArray[np.float64]) -> float:
        r_sc = as_vec3(y[:3], name="r_sc")
        r_body = as_vec3(get_body_vec_m(t), name=f"r_{label}")
        return float(_los_blocked_by_sphere(r_sc, r_body, R_ref))

    ev = _with_t_guard(core, t_guard_s=t_guard_s)
    ev = _set_event_props(ev, terminal=terminal, direction=direction)

    # Debug-friendly name (optional)
    try:
        ev.__name__ = f"occultation_{label}"
    except Exception:
        pass

    return ev



# =============================================================================
# 7.                          OPERATIONAL EVENTS
# =============================================================================

def make_maneuver_trigger_event(
    *,
    t_trigger_s: float,
    terminal: bool = False,
    direction: float = +1.0,
) -> EventFn:
    """
    Simple time-based maneuver trigger.

    Root: g(t,y) = t - t_trigger_s
    """
    t0 = float(t_trigger_s)

    def core(t: float, y: NDArray[np.float64]) -> float:
        _ = y  # unused
        return float(t - t0)

    return _set_event_props(core, terminal=terminal, direction=direction)


def make_stability_violation_event(
    *,
    mu: float,
    R_ref_m: float,
    e_max: float | None = None,
    e_min: float | None = None,
    i_max_deg: float | None = None,
    i_min_deg: float | None = None,
    rp_min_alt_km: float | None = None,
    ra_max_alt_km: float | None = None,
    terminal: bool = True,
    t_guard_s: float = 1.0,
) -> EventFn:
    """
    Frozen-orbit / stability guard: triggers when e/i/periapsis/apoapsis exceed bounds.

    Root returns the largest violation margin:
      - negative => inside bounds
      - positive => violated

    Event triggers when crossing upward (direction=+1).
    """
    mu_ = float(mu)
    R_ref = float(R_ref_m)

    if not (math.isfinite(mu_) and mu_ > 0.0):
        raise ValueError("mu must be finite and > 0.")
    if not (math.isfinite(R_ref) and R_ref > 0.0):
        raise ValueError("R_ref_m must be finite and > 0.")

    # Prefer project constants if present; otherwise fallback.
    KM2M = float(globals().get("KM_TO_M", 1000.0))
    M2KM = float(globals().get("M_TO_KM", 1.0 / 1000.0))

    def _coe_from_rv(r_i: NDArray[np.float64], v_i: NDArray[np.float64]) -> tuple[float, float, float, float]:
        """
        Minimal osculating elements needed for guards:
          e, inc_rad, a, p

        Returns (e, i, a, p). For non-elliptic cases, a may be +/-inf.
        """
        r = as_vec3(r_i, name="r")
        v = as_vec3(v_i, name="v")

        R = float(np.linalg.norm(r))
        if not math.isfinite(R) or R <= 0.0:
            return math.nan, math.nan, math.nan, math.nan

        h = np.cross(r, v)
        hnorm = float(np.linalg.norm(h))
        if not math.isfinite(hnorm) or hnorm <= 0.0:
            inc = 0.0
        else:
            c = h[2] / hnorm
            c = max(-1.0, min(1.0, float(c)))
            inc = math.acos(c)

        e_vec = (np.cross(v, h) / mu_) - (r / R)
        e = float(np.linalg.norm(e_vec))

        V2 = float(np.dot(v, v))
        eps = 0.5 * V2 - mu_ / R  # specific orbital energy

        # a is well-defined for eps != 0; sign follows conic type
        if not math.isfinite(eps) or abs(eps) <= 0.0:
            a = math.inf
        else:
            a = -mu_ / (2.0 * eps)

        p = (hnorm * hnorm) / mu_ if (math.isfinite(hnorm) and hnorm > 0.0) else math.nan
        return float(e), float(inc), float(a), float(p)

    def core(t: float, y: NDArray[np.float64]) -> float:
        r = y[:3]
        v = y[3:]
        e, inc_rad, a, p = _coe_from_rv(r, v)

        # If we can't compute meaningful values, treat as violation.
        if not (math.isfinite(e) and math.isfinite(inc_rad)):
            return 1.0

        i_deg = math.degrees(inc_rad)
        viol: list[float] = []

        # e bounds
        if e_max is not None:
            viol.append(e - float(e_max))
        if e_min is not None:
            viol.append(float(e_min) - e)

        # i bounds
        if i_max_deg is not None:
            viol.append(i_deg - float(i_max_deg))
        if i_min_deg is not None:
            viol.append(float(i_min_deg) - i_deg)

        # peri/apo altitude bounds (elliptic only)
        if math.isfinite(a) and a > 0.0 and e < 1.0:
            rp = a * (1.0 - e)
            ra = a * (1.0 + e)

            if rp_min_alt_km is not None:
                rp_alt_km = (rp - R_ref) * M2KM
                viol.append(float(rp_min_alt_km) - rp_alt_km)  # positive if below min

            if ra_max_alt_km is not None:
                ra_alt_km = (ra - R_ref) * M2KM
                viol.append(ra_alt_km - float(ra_max_alt_km))  # positive if above max

        return float(max(viol)) if viol else -1.0

    ev = _with_t_guard(core, t_guard_s=t_guard_s, guard_value=-1.0)
    return _set_event_props(ev, terminal=terminal, direction=+1.0)



# =============================================================================
# 8.                           CONVENIENCE EVENTS
# =============================================================================

def default_events(
    R_ref_m: float,
    impact_alt_km: float = 5.0,
    add_periapo: bool = True,
    t_guard_s: float = 1.0,
    *,
    topo=None,
    r_i_to_bf=None,
    switch_alt_km: float = 11.0,
    topo_kind: str = "bilinear",
) -> list[EventFn]:
    """
    Convenience bundle for the propagator:
      - impact threshold (terminal)
      - periapsis/apoapsis events (optional, non-terminal)
    """
    KM2M = float(globals().get("KM_TO_M", 1000.0))

    impact_alt_m = float(impact_alt_km) * KM2M
    switch_alt_m = float(switch_alt_km) * KM2M

    events: list[EventFn] = []

    # Prefer topo-aware impact event when both are provided
    if (topo is not None) and (r_i_to_bf is not None):
        events.append(
            make_hybrid_impact_event(
                R_ref_m=float(R_ref_m),
                impact_alt_m=impact_alt_m,
                topo=topo,
                r_i_to_bf=r_i_to_bf,
                switch_alt_m=switch_alt_m,
                kind=str(topo_kind),
                terminal=True,
            )
        )
    else:
        events.append(make_impact_event(float(R_ref_m), impact_alt_m, terminal=True))

    if add_periapo:
        events.append(make_periselene_event(t_guard_s=float(t_guard_s), terminal=False))
        events.append(make_aposelene_event(t_guard_s=float(t_guard_s), terminal=False))

    return events



# =============================================================================
# 9.                            TESTING __MAIN__
# =============================================================================

if __name__ == "__main__":
    """
    Minimal, deterministic self-test for the event factories.

    Notes
    -----
    - Avoid importing this module inside this block (can create a second module instance).
      We call the factories directly.
    - Keep all events non-terminal here so one event doesn't stop the run before others fire.
    """
    import sys
    import numpy as np

    try:
        from scipy.integrate import solve_ivp
    except Exception as e:  # pragma: no cover
        print("\n[ERROR] SciPy is required for this self-test.", file=sys.stderr)
        print("Install with:  pip install scipy\n", file=sys.stderr)
        raise SystemExit(2) from e

    # -----------------------------
    # Constants (approx, deterministic)
    # -----------------------------
    MU = 4.9048695e12     # [m^3/s^2] Moon GM
    R_M = 1_737_400.0     # [m] Moon mean radius

    # -----------------------------
    # Simple 2-body RHS (Moon-centered inertial)
    # -----------------------------
    def rhs_2body(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        rn = float(np.linalg.norm(r))
        # guard against rn=0 in pathological cases
        inv_r3 = 1.0 / (rn * rn * rn + 1e-30)
        a = -MU * r * inv_r3
        return np.concatenate((v, a))

    # -----------------------------
    # Geometry providers (test stubs)
    # -----------------------------
    # For testing: assume inertial == Moon-fixed, so transform is identity.
    def r_i_to_bf(t: float, vec: np.ndarray) -> np.ndarray:
        return np.asarray(vec, dtype=float)

    # Sun direction: constant +X
    def sun_hat_i(t: float) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    # "Earth" position: far away +X (LOS mostly clear unless SC is between origin and Earth)
    def earth_vec_m(t: float) -> np.ndarray:
        return np.array([50.0 * R_M, 0.0, 0.0], dtype=float)

    # Simple eclipse predicate for duration test: night-side if x<0
    def is_eclipsed(t: float, y: np.ndarray) -> bool:
        return bool(y[0] < 0.0)

    # -----------------------------
    # Initial orbit (elliptic, inclined)
    # -----------------------------
    rp = R_M + 100_000.0
    ra = R_M + 1_500_000.0
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)

    r0 = np.array([rp, 0.0, 0.0], dtype=float)
    v0_mag = float(np.sqrt(MU * (1.0 + e) / rp))

    inc = np.deg2rad(30.0)
    v0 = np.array([0.0, v0_mag * np.cos(inc), v0_mag * np.sin(inc)], dtype=float)

    y0 = np.concatenate((r0, v0))

    T = float(2.0 * np.pi * np.sqrt(a**3 / MU))
    tf = 1.5 * T  # enough time for multiple crossings

    # -----------------------------
    # Build events (non-terminal for test robustness)
    # -----------------------------
    events_named = [
        ("terminator", make_terminator_crossing_event(
            sun_hat_i=sun_hat_i,
            r_i_to_bf=r_i_to_bf,
            terminal=False,
            direction=0.0,
            t_guard_s=10.0,
        )),
        ("node_asc", make_node_crossing_event(
            which="asc",
            terminal=False,
            t_guard_s=10.0,
        )),
        ("lon0", make_longitude_crossing_event(
            lon0_deg=0.0,
            r_i_to_bf=r_i_to_bf,
            terminal=False,
            direction=0.0,
            t_guard_s=10.0,
        )),
        ("flyover", make_target_flyover_event(
            target_lat_deg=0.0,
            target_lon_deg=0.0,
            max_central_angle_deg=5.0,
            r_i_to_bf=r_i_to_bf,
            terminal=False,
            t_guard_s=10.0,
        )),
        ("occult_earth", make_occultation_event(
            body="earth",
            get_body_vec_m=earth_vec_m,
            R_ref_m=R_M,
            terminal=False,
            direction=0.0,
            t_guard_s=10.0,
        )),
        ("maneuver", make_maneuver_trigger_event(
            t_trigger_s=0.25 * T,
            terminal=False,
            direction=+1.0,
        )),
        ("max_eclipse_dur", make_max_eclipse_duration_event(
            is_eclipsed=is_eclipsed,
            max_duration_s=1e9,   # huge => should not trigger
            terminal=False,
        )),
    ]

    events = [ev for _, ev in events_named]

    # Give events debug-friendly names (optional)
    for name, ev in events_named:
        try:
            ev.__name__ = name
        except Exception:
            pass

    # -----------------------------
    # Solve
    # -----------------------------
    sol = solve_ivp(
        rhs_2body,
        (0.0, tf),
        y0,
        rtol=1e-9,
        atol=1e-12,
        max_step=T / 2000.0,
        events=events,
        dense_output=False,
    )

    # -----------------------------
    # Report
    # -----------------------------
    print("\n=== events self-test ===")
    print(f"status = {sol.status} (0=OK, 1=terminal-event), message={sol.message}")

    # SciPy: sol.t_events is list aligned with events
    for i, (name, _) in enumerate(events_named):
        t_hits = sol.t_events[i]
        n = len(t_hits)
        print(f"- {name:14s}: {n} hits", end="")
        if n:
            print(f" | first={t_hits[0]:.3f} s, last={t_hits[-1]:.3f} s")
        else:
            print()

    # -----------------------------
    # Soft asserts (geometry-dependent)
    # -----------------------------
    def _assert(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    _assert(len(sol.t_events[0]) > 0, "terminator crossing did not trigger (check sun_hat_i / r_i_to_bf)")
    _assert(len(sol.t_events[1]) > 0, "ascending node crossing did not trigger (check inclination)")
    _assert(len(sol.t_events[2]) > 0, "longitude crossing did not trigger (check lon definition / wrap)")
    _assert(len(sol.t_events[5]) == 1, "maneuver trigger should hit exactly once")
    _assert(len(sol.t_events[6]) == 0, "max eclipse duration should NOT trigger in this test")

    print("✅ OK: core events triggered as expected.\n")


# =============================================================================
# 10.                             Public API
# =============================================================================

__all__ = (
    # -------------------------------------------------------------------------
    # Impact & geometry events
    # -------------------------------------------------------------------------
    "make_impact_event",              # Stop when (||r|| - R_ref) drops to impact_alt_m (downward crossing)
    "make_hybrid_impact_event",       # Two-stage impact: sphere altitude far-field, topo clearance near-field
    "make_altitude_crossing_event",   # Generic altitude threshold: (||r|| - R_ref) - target_alt_m
    "make_radius_event",              # Generic radius threshold: ||r|| - target_r_m
    "make_soi_event",                 # Geometric SOI boundary: outward radius crossing at soi_r_m

    # -------------------------------------------------------------------------
    # Orbital events
    # -------------------------------------------------------------------------
    "make_periselene_event",          # Periapsis: r·v = 0 crossing with + slope (direction=+1)
    "make_aposelene_event",           # Apoapsis: r·v = 0 crossing with - slope (direction=-1)
    "make_escape_event",              # Two-body escape diagnostic: eps = v^2/2 - mu/r crosses 0 upward

    # -------------------------------------------------------------------------
    # Eclipse events (hard-shadow / LOS occultation)
    # -------------------------------------------------------------------------
    "make_solar_eclipse_event",       # LOS(SC->Sun) occulted by Moon and/or Earth (sphere intersection test)
    "make_is_eclipsed_solar",         # Convenience boolean wrapper around make_solar_eclipse_event
    "make_max_eclipse_duration_event",# Guard on continuous eclipse time: triggers when duration > max_duration_s

    # -------------------------------------------------------------------------
    # Surface geometry / ground-track events (Moon-fixed frame)
    # -------------------------------------------------------------------------
    "make_terminator_crossing_event", # Terminator: dot(r_hat_fixed, sun_hat_fixed) = 0
    "make_node_crossing_event",       # Node crossing in reference plane: z=0 (asc/desc/both)
    "make_longitude_crossing_event",  # Fixed-frame longitude crossing: wrap_pi(lon - lon0) = 0
    "make_target_flyover_event",      # Flyover: central angle to target <= gamma_max (downward crossing)

    # -------------------------------------------------------------------------
    # Communication / line-of-sight events
    # -------------------------------------------------------------------------
    "make_occultation_event",         # LOS(SC->body) occulted by Moon sphere (margin crosses 0)

    # -------------------------------------------------------------------------
    # Operational events
    # -------------------------------------------------------------------------
    "make_maneuver_trigger_event",    # Simple time trigger: t - t_trigger_s
    "make_stability_violation_event", # Osculating guard: e/i/rp/ra bounds (max violation crosses 0 upward)

    # -------------------------------------------------------------------------
    # Convenience bundles
    # -------------------------------------------------------------------------
    "default_events",                 # Standard event list for a propagator (impact + optional peri/apo)
)

