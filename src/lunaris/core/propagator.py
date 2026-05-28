# LUNAR_SIMULATION/core/propagator.py
# -*- coding: utf-8 -*-
"""
core.propagator
===============

High-Level Orbit Propagator & Simulation Orchestrator.

This module serves as the **single authoritative entry point** for orbit propagation.
It bridges the high-level Application/UI layers with the low-level physics engine
(:class:`core.dynamics.DynamicsEngine`) and numerical integrators (SciPy).

Design Philosophy & Key Decisions
---------------------------------
1. **Event Compatibility:**
   The event logic in `core.events` remains strictly geometric. This propagator
   wraps the state vector passed to events to ensure only the kinematic state
   ``[x, y, z, vx, vy, vz]`` is exposed, ignoring auxiliary states (e.g., mass)
   to maintain compatibility with existing event factories.

2. **Dynamics Source:**
   The Equations of Motion (RHS) are sourced exclusively from
   ``core.dynamics.DynamicsEngine.build_rhs()``. Legacy factories are bypassed
   to guarantee JIT-compiled performance and correctness.

3. **Output Standardization:**
   - Numerical solvers (SciPy) return column-major arrays ``(n_states, n_steps)``.
   - This module transposes outputs to **row-major** ``(n_steps, n_states)``
     standardized containers (:class:`common.type_defs.PropagationResult`)
     for easier slicing and plotting.

4. **Step-Size Control (Anti-Aliasing):**
   Automatic step-size limiting is applied using the Nyquist-Shannon theorem via
   ``common.math_utils.nyquist_max_step_s``, based on the active spherical
   harmonic degree. This prevents gravity-field aliasing in high-fidelity runs.

5. **Long-Duration Operations:**
   - Supports **chunked integration** to manage RAM usage during long missions.
   - Implements cooperative stopping via ``stop_file`` detection.
   - Checkpointing support for crash recovery.

Note
----
This module defines its own result extraction logic to avoid dependencies on
potentially deprecated legacy integrator modules.
"""


# =============================================================================
# 0.                                 IMPORTS
# =============================================================================

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


from scipy.integrate import solve_ivp  # type: ignore


from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.math_utils import nyquist_max_step_s, quat_slerp_np, quat_rotate_np
from lunaris.common.type_defs import PropagationResult, PropagatorConfig, TimeConfig


from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.state import OrbitState
from lunaris.core.events import (
    make_impact_event,
    make_hybrid_impact_event,
    make_periselene_event,
    make_aposelene_event,
)



# =============================================================================
# 1.                             UTILITIES
# =============================================================================

STATE_MIN_SIZE = 6  # [x,y,z,vx,vy,vz]

def _make_telem_dict(
    t_s: float,
    y: np.ndarray,
    R_ref_m: float,
    mu_m3s2: float,
    *,
    t_frame_s: Optional[float] = None,
    r_i_to_bf: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    surface_radius_m: Optional[Callable[[float, float], float]] = None,
) -> Optional[Dict[str, float]]:
    """
    Create a compact telemetry dict for the desktop UI (JSON-lines on stdout).

    Base keys
    ---------
    - `t_s`
    - `alt_km`
    - `v_km_s`
    - `ecc`

    Optional terrain-aware keys
    ---------------------------
    When both a body-fixed mapper and a surface radius sampler are available, the
    telemetry also includes:
    - `surface_r_km`
    - `surface_alt_km`
    - `terrain_clearance_km`

    This lets the UI distinguish "below mean radius" from "below actual local
    terrain", which is essential when topography-backed collision analysis is
    active.
    """
    try:
        y = np.asarray(y, dtype=np.float64)
        if y.size < 6:
            return None
        r = y[0:3]
        v = y[3:6]
        r_norm = float(np.linalg.norm(r))
        if not np.isfinite(r_norm) or r_norm <= 0.0:
            return None

        # Altitude & speed
        alt_km = (r_norm - float(R_ref_m)) / 1000.0
        v_km_s = float(np.linalg.norm(v)) / 1000.0

        # Eccentricity (2-body osculating, Moon-centered)
        mu = float(mu_m3s2)
        if not np.isfinite(mu) or mu <= 0.0:
            ecc = float("nan")
        else:
            h = np.cross(r, v)
            e_vec = (np.cross(v, h) / mu) - (r / r_norm)
            ecc = float(np.linalg.norm(e_vec))

        telem = {
            "t_s": float(t_s),
            "alt_km": float(alt_km),
            "v_km_s": float(v_km_s),
            "ecc": float(ecc),
        }

        if r_i_to_bf is not None and surface_radius_m is not None:
            try:
                rotation_time_s = float(t_s if t_frame_s is None else t_frame_s)
                r_bf = np.asarray(r_i_to_bf(rotation_time_s, r), dtype=np.float64).reshape(3)
                lat_rad, lon_rad = _latlon_from_r_bf(r_bf)
                terrain_r_m = float(surface_radius_m(lat_rad, lon_rad))
                if math.isfinite(terrain_r_m) and terrain_r_m > 0.0:
                    telem["surface_r_km"] = float(terrain_r_m / 1000.0)
                    telem["surface_alt_km"] = float((terrain_r_m - float(R_ref_m)) / 1000.0)
                    telem["terrain_clearance_km"] = float((r_norm - terrain_r_m) / 1000.0)
            except Exception:
                # Telemetry streaming must stay best-effort; skipping the optional
                # terrain fields is preferable to breaking the entire run.
                pass

        return telem
    except Exception:
        return None


def _latlon_from_r_bf(r_bf: np.ndarray) -> Tuple[float, float]:
    """
    Compute body-fixed geocentric latitude/longitude from a Cartesian vector.

    The helper is intentionally small and dependency-light so telemetry and
    hybrid-impact helpers can reuse it without reaching back into heavier model
    modules.
    """

    x = float(r_bf[0])
    y = float(r_bf[1])
    z = float(r_bf[2])
    radius = math.sqrt(x * x + y * y + z * z)
    if radius <= 0.0:
        return 0.0, 0.0
    lat_rad = math.asin(max(-1.0, min(1.0, z / radius)))
    lon_rad = math.atan2(y, x)
    return lat_rad, lon_rad


def _build_surface_radius_sampler(topo: Any) -> Callable[[float, float], float]:
    """
    Build a `(lat_rad, lon_rad) -> radius_m` sampler from a topo provider.

    Supported contracts intentionally mirror the hybrid impact-event code:
    - `sample_bilinear(lat_deg, lon_deg, kind="radius_m")`
    - `sample_nearest(lat_deg, lon_deg, kind="radius_m")`
    - `radius_m_deg(lat_deg, lon_deg)`
    - `radius_m(lat_rad, lon_rad)`
    """

    if hasattr(topo, "sample_bilinear") and callable(getattr(topo, "sample_bilinear")):
        fn = getattr(topo, "sample_bilinear")

        def _radius_from_bilinear(lat_rad: float, lon_rad: float) -> float:
            return float(fn(math.degrees(lat_rad), math.degrees(lon_rad) % 360.0, kind="radius_m"))

        return _radius_from_bilinear

    if hasattr(topo, "sample_nearest") and callable(getattr(topo, "sample_nearest")):
        fn = getattr(topo, "sample_nearest")

        def _radius_from_nearest(lat_rad: float, lon_rad: float) -> float:
            return float(fn(math.degrees(lat_rad), math.degrees(lon_rad) % 360.0, kind="radius_m"))

        return _radius_from_nearest

    if hasattr(topo, "radius_m_deg") and callable(getattr(topo, "radius_m_deg")):
        fn = getattr(topo, "radius_m_deg")

        def _radius_from_deg(lat_rad: float, lon_rad: float) -> float:
            return float(fn(math.degrees(lat_rad), math.degrees(lon_rad) % 360.0))

        return _radius_from_deg

    if hasattr(topo, "radius_m") and callable(getattr(topo, "radius_m")):
        fn = getattr(topo, "radius_m")

        def _radius_from_rad(lat_rad: float, lon_rad: float) -> float:
            return float(fn(float(lat_rad), float(lon_rad)))

        return _radius_from_rad

    raise AttributeError(
        "Topography object does not expose a usable radius sampler "
        "(expected sample_bilinear/sample_nearest/radius_m_deg/radius_m)."
    )


def _as_state_array(y0: Any) -> np.ndarray:
    """
    Normalize initial state to a contiguous float64 1D array.
    Accepts OrbitState or array-like. Requires at least 6 elements.
    """
    if isinstance(y0, OrbitState):
        y = np.asarray(y0.y, dtype=np.float64).reshape(-1)
    else:
        y = np.asarray(y0, dtype=np.float64).reshape(-1)

    if y.size < STATE_MIN_SIZE:
        raise ValueError("Initial state must have at least 6 elements: [x,y,z,vx,vy,vz].")

    return np.array(y, dtype=np.float64, copy=True)


def _norm_method(method: Any) -> str:
    """Normalize integrator method names to a stable canonical form."""
    m = str(method).strip().upper()
    m = m.replace("-", "_").replace(" ", "_")
    return m


def _is_symplectic_method(method: str) -> bool:
    m = _norm_method(method)
    return m in ("VV", "VERLET", "STORMER_VERLET", "STÖRMER_VERLET", "YOSHIDA4", "Y4")


def _sympl_name(method: str) -> str:
    m = _norm_method(method)
    return "Y4" if m in ("YOSHIDA4", "Y4") else "VV"


def _stop_requested(stop_file: Optional[str]) -> bool:
    """Stop if a sentinel file exists (safe on all platforms)."""
    if not stop_file:
        return False
    try:
        return Path(os.fspath(stop_file)).exists()
    except Exception:
        return False


def _atomic_save_npz(path: str, **arrays) -> None:
    """
    Atomically write an NPZ file:
      - writes to <path>.tmp.npz
      - then os.replace() onto final path
    """
    if not path:
        return

    dst = Path(os.fspath(path))
    if dst.suffix.lower() != ".npz":
        dst = dst.with_suffix(".npz")

    tmp = dst.with_name(dst.name + ".tmp")  # e.g., out.npz.tmp
    tmp_npz = tmp.with_suffix(".npz")       # ensure numpy does not auto-append

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        np.savez(str(tmp_npz), **arrays)
        os.replace(str(tmp_npz), str(dst))
    finally:
        for p in (tmp, tmp_npz):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def make_time_grid(t0: float, tf: float, dt: float) -> np.ndarray:
    t0 = float(t0); tf = float(tf); dt = float(dt)
    if (not np.isfinite(t0)) or (not np.isfinite(tf)) or (tf <= t0) or (dt <= 0.0) or (not np.isfinite(dt)):
        return np.array([t0, tf], dtype=np.float64)

    n = int(math.floor((tf - t0) / dt))
    if n < 1:
        return np.array([t0, tf], dtype=np.float64)

    t = t0 + np.arange(n + 1, dtype=np.float64) * dt
    if t[-1] < tf:
        t = np.append(t, tf)
    else:
        t[-1] = tf
    return t


def _clamp_output_dt(t0: float, tf: float, dt_out: float, cap: int, verbose: bool) -> float:
    dt = float(dt_out)
    if dt <= 0.0 or (not np.isfinite(dt)):
        raise ValueError("output_dt_s must be positive and finite.")
    if tf <= t0:
        return dt

    n = int(math.ceil((tf - t0) / dt)) + 1
    if n > int(cap):
        dt = (tf - t0) / max(2, int(cap) - 1)
        if verbose:
            print(f"[OUT] max_points_cap exceeded -> increasing output_dt_s to {dt:g} s", flush=True)
    return dt


def _get_ref_radius_and_mu(dynamics: DynamicsEngine) -> Tuple[float, float]:
    """
    STRICT gravity SSOT:
      - grav.R_ref_m
      - grav.GM_m3s2
    Falls back to constants only if dynamics.grav is None (point-mass baseline).
    """
    grav = getattr(dynamics, "grav", None)
    if grav is None:
        return float(R_MOON), float(MU_MOON)

    missing = [name for name in ("R_ref_m", "GM_m3s2") if not hasattr(grav, name)]
    if missing:
        raise AttributeError(
            "Gravity model attached to DynamicsEngine is missing required strict attributes: "
            + ", ".join(missing)
            + ". Expected SSOT fields: R_ref_m, GM_m3s2."
        )

    return float(getattr(grav, "R_ref_m")), float(getattr(grav, "GM_m3s2"))


def _get_sh_degree(dynamics: DynamicsEngine) -> int:
    """
    STRICT gravity SSOT:
      - grav.degree_max
    Returns 1 only if no gravity model is attached.
    """
    grav = getattr(dynamics, "grav", None)
    if grav is None:
        return 1

    if not hasattr(grav, "degree_max"):
        raise AttributeError("Gravity model missing required strict attribute: degree_max.")

    d = int(getattr(grav, "degree_max"))
    return max(1, d)



# =============================================================================
# 2.            Rotation helper for hybrid topo impact
# =============================================================================

def _build_r_i_to_bf_from_rot_table(
    dynamics: DynamicsEngine,
) -> Optional[Callable[[float, np.ndarray], np.ndarray]]:
    """Build an inertial→body-fixed position mapper from an ephemeris quaternion table.

    Expected rotation table format:
      - q_i2f(t): quaternion (w, x, y, z) mapping inertial → fixed/body frame

    Supported provider:
      - eph.tables -> dt_s, q_i2f_tab

    Returns None if no valid rotation table is available.

    Why single-row tables are supported
    -----------------------------------
    Short runs can legitimately produce an ephemeris attitude table with only
    one quaternion sample. In that case the rotation is still perfectly usable:
    it simply behaves as a constant inertial→fixed transform over the requested
    interval. Rejecting `N=1` tables would unnecessarily disable:

    - terrain-aware telemetry, and
    - hybrid impact events

    for otherwise valid short-duration propagations.
    """
    eph = getattr(dynamics, "ephem", None)
    if eph is None:
        return None

    def _try_get_dt_and_qtab() -> Optional[Tuple[float, np.ndarray]]:
        # Tables-style (strict)
        if hasattr(eph, "tables"):

            try:
                tab = eph.tables
                dt = float(getattr(tab, "dt_s", 0.0))
                q = getattr(tab, "q_i2f_tab", None)
            except Exception:
                dt = 0.0
                q = None
            if dt > 0.0 and q is not None:
                try:
                    qtab = np.asarray(q, dtype=np.float64)
                    return dt, qtab
                except Exception:
                    pass

        return None

    got = _try_get_dt_and_qtab()
    if got is None:
        return None

    dt, q_tab = got
    if dt <= 0.0 or q_tab.ndim != 2 or q_tab.shape[1] != 4 or q_tab.shape[0] < 1:
        return None

    q_tab = np.ascontiguousarray(q_tab, dtype=np.float64)
    n = int(q_tab.shape[0])
    dt_f = float(dt)

    if n == 1:
        q_const = np.asarray(q_tab[0], dtype=np.float64).reshape(4)

        def r_i_to_bf_constant(_t: float, r_i: np.ndarray) -> np.ndarray:
            """
            Apply a constant inertial→fixed quaternion.

            The timestamp is intentionally ignored because a single-table sample
            means the caller asked for a degenerate/constant frame history.
            """

            r = np.asarray(r_i, dtype=np.float64).reshape(3)
            return quat_rotate_np(q_const, r)

        return r_i_to_bf_constant

    def r_i_to_bf(t: float, r_i: np.ndarray) -> np.ndarray:
        # t assumed seconds aligned with ephemeris start
        u = float(t) / dt_f
        if u <= 0.0:
            q = q_tab[0]
        elif u >= float(n - 1):
            q = q_tab[n - 1]
        else:
            i = int(u)  # floor for u>=0
            a = float(u - i)
            q = quat_slerp_np(q_tab[i], q_tab[i + 1], a)

        r = np.asarray(r_i, dtype=np.float64).reshape(3)
        return quat_rotate_np(q, r)

    return r_i_to_bf



# =============================================================================
# 3.                  Event construction & wrappers
# =============================================================================

def _wrap_event_first6(ev: Callable[[float, np.ndarray], float]) -> Callable[[float, np.ndarray], float]:
    """Wrap an event function so it ignores any augmented state beyond the first 6 elements."""
    def g(t: float, y: np.ndarray) -> float:
        return float(ev(t, y[:6]))

    # SciPy event attributes
    g.terminal = bool(getattr(ev, "terminal", False))     # type: ignore[attr-defined]
    g.direction = float(getattr(ev, "direction", 0.0))    # type: ignore[attr-defined]

    # Help debugging / introspection
    g.__name__ = getattr(ev, "__name__", "event_first6")
    g.__doc__ = getattr(ev, "__doc__", None)
    return g


def _get_event_cfg(cfg: Any) -> Any:
    return getattr(cfg, "events", None)


def _get_cfg_bool(cfg: Any, name: str, default: bool) -> bool:
    evc = _get_event_cfg(cfg)
    if evc is not None and hasattr(evc, name):
        try:
            return bool(getattr(evc, name))
        except Exception:
            pass
    if hasattr(cfg, name):
        try:
            return bool(getattr(cfg, name))
        except Exception:
            pass
    return bool(default)


def _get_cfg_float(cfg: Any, name: str, default: float) -> float:
    evc = _get_event_cfg(cfg)
    if evc is not None and hasattr(evc, name):
        try:
            return float(getattr(evc, name))
        except Exception:
            pass
    if hasattr(cfg, name):
        try:
            return float(getattr(cfg, name))
        except Exception:
            pass
    return float(default)


def _get_detect_impact(cfg: Any) -> bool:
    return _get_cfg_bool(cfg, "detect_impact", True)


def _get_impact_alt_km(cfg: Any) -> float:
    return _get_cfg_float(cfg, "impact_alt_km", 0.0)


def _get_enable_peri_apo_events(cfg: Any) -> bool:
    return _get_cfg_bool(cfg, "enable_peri_apo_events", True)


def _find_event_index(events: Optional[List[Callable[[float, np.ndarray], float]]], role: str) -> Optional[int]:
    if not events:
        return None
    for i, ev in enumerate(events):
        if getattr(ev, "_event_role", None) == role:
            return i
    return None


def build_events(
    dynamics: DynamicsEngine,
    cfg: PropagatorConfig,
    *,
    topo_grid: Any = None,
    add_stop_event: bool = True,
) -> List[Callable[[float, np.ndarray], float]]:
    """Build SciPy-compatible event callables based on PropagatorConfig (+ optional topo grid)."""
    R_ref, _mu = _get_ref_radius_and_mu(dynamics)

    detect_impact = _get_detect_impact(cfg)
    impact_alt_km = _get_impact_alt_km(cfg)
    impact_alt_m = float(impact_alt_km) * 1000.0

    events: List[Callable[[float, np.ndarray], float]] = []

    # Impact event (terminal)
    if detect_impact:
        if topo_grid is not None:
            r_i_to_bf = _build_r_i_to_bf_from_rot_table(dynamics)
            if r_i_to_bf is not None:
                ev_imp = make_hybrid_impact_event(
                    R_ref_m=float(R_ref),
                    impact_alt_m=float(impact_alt_m),
                    topo=topo_grid,
                    r_i_to_bf=r_i_to_bf,
                    switch_alt_m=float(getattr(cfg, "hybrid_switch_alt_m", 0.0)),
                    kind=str(getattr(cfg, "hybrid_kind", "radial")),
                    terminal=True,
                )
            else:
                ev_imp = make_impact_event(R_ref_m=float(R_ref), impact_alt_m=float(impact_alt_m), terminal=True)
        else:
            ev_imp = make_impact_event(R_ref_m=float(R_ref), impact_alt_m=float(impact_alt_m), terminal=True)

        ev_imp6 = _wrap_event_first6(ev_imp)
        setattr(ev_imp6, "_event_role", "impact")
        events.append(ev_imp6)

    # Peri/Apo events (non-terminal)
    if _get_enable_peri_apo_events(cfg):
        ev_peri = _wrap_event_first6(make_periselene_event(terminal=False))
        ev_apo = _wrap_event_first6(make_aposelene_event(terminal=False))
        setattr(ev_peri, "_event_role", "peri")
        setattr(ev_apo, "_event_role", "apo")
        events.append(ev_peri)
        events.append(ev_apo)

    # Stop file event (optional) – use +/-1 for sign change
    stop_file = getattr(cfg, "stop_file", None)
    stop_in_scipy = bool(getattr(cfg, "stop_event_in_scipy", False))
    if add_stop_event and stop_file and stop_in_scipy:
        def _stop_ev(t: float, y: np.ndarray) -> float:
            # Use +/-1 so a change in stop-file state produces a sign change
            return -1.0 if _stop_requested(str(stop_file)) else 1.0
        _stop_ev.terminal = True           # type: ignore[attr-defined]
        _stop_ev.direction = 0.0           # type: ignore[attr-defined]
        setattr(_stop_ev, "_event_role", "stop")
        events.append(_stop_ev)

    return events



# =============================================================================
# 4.                 Fixed-step integrators (VV / Yoshida4)
# =============================================================================

_Y4_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_Y4_W0 = - (2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))


def _vv_step(accel: Callable[[float, np.ndarray], np.ndarray], t: float, y6: np.ndarray, h: float) -> np.ndarray:
    r = y6[:3]
    v = y6[3:6]
    a0 = accel(t, y6)

    v_half = v + 0.5 * h * a0
    r1 = r + h * v_half

    y_half = np.empty(6, dtype=np.float64)
    y_half[0:3] = r1
    y_half[3:6] = v_half
    a1 = accel(t + h, y_half)

    v1 = v_half + 0.5 * h * a1

    y1 = np.empty(6, dtype=np.float64)
    y1[0:3] = r1
    y1[3:6] = v1
    return y1


def _y4_step(accel: Callable[[float, np.ndarray], np.ndarray], t: float, y6: np.ndarray, h: float) -> np.ndarray:
    y1 = _vv_step(accel, t, y6, _Y4_W1 * h)
    t1 = t + _Y4_W1 * h
    y2 = _vv_step(accel, t1, y1, _Y4_W0 * h)
    t2 = t1 + _Y4_W0 * h
    y3 = _vv_step(accel, t2, y2, _Y4_W1 * h)
    return y3


def _event_crossed(g0: float, g1: float, direction: float = 0.0) -> bool:
    """Return True if an event root is bracketed in [g0,g1] given direction."""
    if not (np.isfinite(g0) and np.isfinite(g1)):
        return False
    # Treat exact zeros robustly
    if g0 == 0.0:
        # If we start exactly at root, do not trigger unless moving away then back.
        # Here we simply ignore the start-point root to avoid duplicate detections.
        return False
    if direction > 0.0:
        return (g0 < 0.0) and (g1 >= 0.0)
    if direction < 0.0:
        return (g0 > 0.0) and (g1 <= 0.0)
    return (g0 > 0.0) != (g1 > 0.0) or (g1 == 0.0)


def _refine_event_time_bisect(
    *,
    stepper: Callable[[Callable[[float, np.ndarray], np.ndarray], float, np.ndarray, float], np.ndarray],
    accel: Callable[[float, np.ndarray], np.ndarray],
    ev: Callable[[float, np.ndarray], float],
    t0: float,
    y0: np.ndarray,
    h: float,
    g0: float,
    g1: float,
    max_iter: int = 30,
    tol_s: float = 1e-6,
) -> Tuple[float, np.ndarray]:
    """Refine a single event root inside a step using bisection + final linear-in-g correction.

    We re-integrate short substeps from (t0,y0) to candidate times. This does NOT modify the
    main integrator's state advancement (y_next is still computed once in the main loop), so
    symplectic stepping remains unchanged; we only improve the reported event time/state.
    """
    # Ensure a valid bracket
    if not (np.isfinite(g0) and np.isfinite(g1)):
        t_lin = t0
        return t_lin, y0

    a = 0.0
    b = 1.0
    ga = float(g0)
    gb = float(g1)

    # Cache endpoints' states when needed
    ya = y0
    yb = None

    # Early exit if already extremely close
    if abs(h) <= tol_s:
        yb = stepper(accel, t0, y0, h)
        return t0 + h, np.asarray(yb, dtype=np.float64)

    # Bisection iterations
    for _ in range(max_iter):
        if (b - a) * abs(h) <= tol_s:
            break
        m = 0.5 * (a + b)
        hm = m * h
        ym = stepper(accel, t0, y0, hm)
        gm = float(ev(t0 + hm, ym))

        # Narrow the bracket by sign
        if (ga > 0.0) == (gm > 0.0):
            a = m
            ga = gm
            ya = ym
        else:
            b = m
            gb = gm
            yb = ym

    # If yb not computed (possible if we always moved left), compute it
    if yb is None:
        yb = stepper(accel, t0, y0, b * h)
        gb = float(ev(t0 + b * h, yb))

    # Final linear-in-g correction inside last bracket (a,b)
    denom = (gb - ga)
    if denom != 0.0 and np.isfinite(denom):
        tau = a + (-ga) * (b - a) / denom
        tau = float(min(1.0, max(0.0, tau)))
    else:
        tau = b

    ht = tau * h
    yt = stepper(accel, t0, y0, ht)
    return float(t0 + ht), np.asarray(yt, dtype=np.float64)


def _integrate_fixed_step(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    *,
    max_step: float,
    method: str,
    events: Optional[List[Callable[[float, np.ndarray], float]]],
    R_ref_m: float,
    mu_m3s2: float,
    verbose: bool,
    heartbeat_hours: float,
    stop_file: Optional[str],
    checkpoint_path: Optional[str],
) -> Tuple[Any, bool, Optional[float], Optional[np.ndarray], bool, Optional[str], Optional[float]]:
    """Integrate a 6D state with a fixed-step symplectic method (VV / Yoshida4).

    Notes
    -----
    - This fixed-step path supports ONLY the 6D state [r,v]. Any augmented state
      (e.g. mass) should be handled via the SciPy path.
    - Events are supported (including non-impact events) and can be refined inside
      each step using a bisection scheme.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if t_eval.size < 2 or np.any(np.diff(t_eval) <= 0.0):
        raise ValueError("t_eval must be strictly increasing and contain at least 2 points.")

    y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
    if y0.size != 6:
        raise ValueError("Fixed-step VV/Y4 currently supports only 6D state vectors.")

    max_step = float(max_step)
    if (not np.isfinite(max_step)) or max_step <= 0.0:
        raise ValueError("max_step must be positive and finite for fixed-step integration.")

    meth = _sympl_name(method)
    stepper = _vv_step if meth == "VV" else _y4_step

    # Acceleration adapter: avoid extra allocations when rhs already returns ndarray
    def accel(t: float, y6: np.ndarray) -> np.ndarray:
        dy = rhs(t, y6)
        if isinstance(dy, np.ndarray):
            return dy[3:6]
        # Fallback (should be rare)
        a = np.asarray(dy, dtype=np.float64).reshape(-1)
        return a[3:6]

    # ------------------------------------------------------------------
    # Events: support all events, with optional refinement
    # ------------------------------------------------------------------
    ev_list: List[Callable[[float, np.ndarray], float]] = list(events) if events else []

    n_ev = len(ev_list)
    t_events_acc: List[List[float]] = [[] for _ in range(n_ev)]
    y_events_acc: List[List[np.ndarray]] = [[] for _ in range(n_ev)]

    # Initialize previous event values at the start time
    t_start = float(t_eval[0])
    g_prev: List[float] = []
    for ev in ev_list:
        try:
            g_prev.append(float(ev(t_start, y0)))
        except Exception:
            g_prev.append(float("nan"))

    t_list: List[float] = [t_start]
    y_list: List[np.ndarray] = [y0.copy()]

    impacted = False
    t_imp: Optional[float] = None
    y_imp: Optional[np.ndarray] = None

    stopped_early = False
    stop_reason: Optional[str] = None
    t_stop: Optional[float] = None

    last_hb_hr = 0.0
    alt_min_km = float("inf")
    alt_max_km = float("-inf")




    # Refinement controls (can be overridden by attaching attrs to rhs)
    refine_tol_s = float(getattr(rhs, "_fixed_step_event_tol_s", 1e-6))
    refine_max_iter = int(getattr(rhs, "_fixed_step_event_max_iter", 30))

    for k in range(t_eval.size - 1):
        if _stop_requested(stop_file):
            stopped_early = True
            stop_reason = "stop file"
            break

        t_seg0 = float(t_list[-1])
        t_target = float(t_eval[k + 1])
        dt_seg = t_target - t_seg0
        if dt_seg <= 0.0:
            continue

        n_sub = int(math.ceil(dt_seg / max_step)) if dt_seg > max_step else 1
        n_sub = max(1, n_sub)
        h = dt_seg / float(n_sub)

        y_curr = y_list[-1].copy()

        for j in range(n_sub):
            tj = t_seg0 + j * h
            y_next = stepper(accel, tj, y_curr, h)
            t_next = tj + h

            earliest_terminal: Optional[Tuple[float, int, np.ndarray]] = None  # (t_event, idx, y_event)

            for i, ev in enumerate(ev_list):
                try:
                    g0 = float(g_prev[i])
                    g1 = float(ev(t_next, y_next))
                except Exception:
                    g_prev[i] = float("nan")
                    continue

                direction = float(getattr(ev, "direction", 0.0))
                terminal = bool(getattr(ev, "terminal", False))

                if _event_crossed(g0, g1, direction):
                    # Refine root within this substep
                    try:
                        t_ev, y_ev = _refine_event_time_bisect(
                            stepper=stepper,
                            accel=accel,
                            ev=ev,
                            t0=tj,
                            y0=y_curr,
                            h=h,
                            g0=g0,
                            g1=g1,
                            max_iter=refine_max_iter,
                            tol_s=refine_tol_s,
                        )
                    except Exception:
                        # Fallback: linear interpolation in event function value
                        denom = (g0 - g1)
                        if denom != 0.0 and np.isfinite(denom):
                            tau = float(min(1.0, max(0.0, g0 / denom)))
                        else:
                            tau = 0.5
                        t_ev = tj + tau * h
                        y_ev = np.asarray(y_curr + tau * (y_next - y_curr), dtype=np.float64)

                    t_events_acc[i].append(float(t_ev))
                    y_events_acc[i].append(np.asarray(y_ev, dtype=np.float64))

                    if terminal:
                        if (earliest_terminal is None) or (t_ev < earliest_terminal[0]):
                            earliest_terminal = (float(t_ev), i, np.asarray(y_ev, dtype=np.float64))

                # Update previous value for next substep
                g_prev[i] = g1

            # Terminal event: stop at earliest terminal root in this substep
            if earliest_terminal is not None:
                t_ev, i_ev, y_ev = earliest_terminal
                ev_role = getattr(ev_list[i_ev], "_event_role", None)

                t_list.append(float(t_ev))
                y_list.append(np.asarray(y_ev, dtype=np.float64))

                stopped_early = True
                if ev_role == "impact":
                    impacted = True
                    t_imp = float(t_ev)
                    y_imp = np.asarray(y_ev, dtype=np.float64)
                    stop_reason = "impact"
                elif ev_role in ("stop", "stop_file", "stopfile"):
                    stop_reason = "stop file"
                else:
                    stop_reason = "event"
                break

            # No terminal event: accept full step
            y_curr = np.asarray(y_next, dtype=np.float64)

            # Periodic stop-file polling
            if (j % 50) == 0 and _stop_requested(stop_file):
                stopped_early = True
                stop_reason = "stop file"
                break

        if stopped_early:
            break

        t_list.append(float(t_target))
        y_list.append(np.asarray(y_curr, dtype=np.float64))

        # Heartbeat
        if heartbeat_hours and heartbeat_hours > 0.0:
            t_hr = (t_target - float(t_eval[0])) / 3600.0
            alt_now_km = (float(np.linalg.norm(y_curr[0:3])) - float(R_ref_m)) / 1000.0
            alt_min_km = min(alt_min_km, alt_now_km)
            alt_max_km = max(alt_max_km, alt_now_km)
            if (t_hr - last_hb_hr) >= float(heartbeat_hours):
                if verbose:
                    print(
                        f"[HB] t={t_hr:7.2f} h | alt={alt_now_km:9.3f} km | min={alt_min_km:9.3f} | max={alt_max_km:9.3f}",
                        flush=True,
                    )
                last_hb_hr = t_hr

    if stopped_early and t_list:
        t_stop = float(t_list[-1])

    t_arr = np.asarray(t_list, dtype=np.float64)
    y_arr = np.asarray(y_list, dtype=np.float64)  # (N,6)

    # Convert accumulated event hits
    t_events = [np.asarray(te, dtype=np.float64) for te in t_events_acc]
    y_events = [
        (np.vstack(ye).astype(np.float64, copy=False) if len(ye) else np.zeros((0, y0.size), dtype=np.float64))
        for ye in y_events_acc
    ]

    ode_like = SimpleNamespace(
        t=t_arr,
        y=y_arr.T,  # mimic SciPy (6,N)
        success=True,
        status=(1 if (impacted or stopped_early) else 0),
        message=("fixed-step ok" if not stopped_early else "stopped early"),
        nfev=np.nan,
        t_events=t_events,
        y_events=y_events,
    )

    if checkpoint_path:
        try:
            _atomic_save_npz(checkpoint_path, t=t_arr, y_row=y_arr)
        except Exception as exc:
            import warnings
            warnings.warn(f"Checkpoint write failed: {exc}", RuntimeWarning)

    return (
        ode_like,
        impacted,
        (float(t_imp) if t_imp is not None else None),
        (np.asarray(y_imp, dtype=np.float64) if y_imp is not None else None),
        stopped_early,
        stop_reason,
        t_stop,
    )



# =============================================================================
# 5.                       MAIN API / propagate()
# =============================================================================

def propagate(
    dynamics: DynamicsEngine,
    y0: Any,
    cfg: PropagatorConfig,
    *,
    time_cfg: Optional["TimeConfig"] = None,
    topo_grid: Any = None,
    extra_events: Optional[Sequence[Callable[[float, np.ndarray], float]]] = None,
) -> PropagationResult:
    """
    Propagate the trajectory for a configured duration and output sampling grid.

    SSOT
    ----
    Duration and output sampling are owned by TimeConfig:
        time_cfg.duration_s
        time_cfg.output_dt_s

    Notes
    -----
    - t0 (start epoch in seconds) is owned by TimeConfig when provided (time_cfg.t0_s).
    - max_points_cap / verbosity / integration tolerances remain in PropagatorConfig unless also provided in TimeConfig.
    """
    y0_arr = _as_state_array(y0)

    t_wall0 = time.perf_counter()

    verbose = bool(getattr(cfg, "verbose", False))

    # Normalize optional filesystem paths to str (or None) for string ops in helpers.
    stop_file: Optional[str] = None
    try:
        sf = getattr(cfg, "stop_file", None)
        stop_file = (str(sf) if sf else None)
    except Exception:
        stop_file = None

    checkpoint_path: Optional[str] = None
    try:
        cp = getattr(cfg, "checkpoint_path", None)
        checkpoint_path = (str(cp) if cp else None)
    except Exception:
        checkpoint_path = None

    # -------------------------------------------------------------------------
    # 1) Resolve time grid (STRICT: TimeConfig required)
    # -------------------------------------------------------------------------
    if time_cfg is None:
        raise ValueError("time_cfg is required (STRICT). Provide TimeConfig(duration_s=..., output_dt_s=...).")

    if getattr(time_cfg, "duration_s", None) is None:
        raise ValueError("time_cfg.duration_s is required and must be finite/positive.")
    dt_out_raw = getattr(time_cfg, "output_dt_s", None)
    dur_s = float(time_cfg.duration_s)
    if dur_s <= 0.0 or (not np.isfinite(dur_s)):
        raise ValueError("Duration must be positive and finite.")

    # Start/end times (t0 belongs to TimeConfig; default 0 if omitted)
    t0 = float(getattr(time_cfg, "t0_s", 0.0) or 0.0)
    if not np.isfinite(t0):
        raise ValueError("time_cfg.t0_s must be finite.")
    tf = t0 + dur_s

    # Resolve output sampling step
    if dt_out_raw is None:
        # Allow "output_dt_s=None" by deriving a reasonable sampling step from the
        # osculating Keplerian period estimated from the initial state (Kepler two-body).
        # (This matches the intent of TimeConfig.samples_per_period.)
        _, mu = _get_ref_radius_and_mu(dynamics)
        mu = float(mu)

        r0 = float(np.linalg.norm(y0_arr[:3]))
        v0 = float(np.linalg.norm(y0_arr[3:6]))
        if not (math.isfinite(r0) and math.isfinite(v0) and r0 > 0.0 and mu > 0.0):
            raise ValueError("Cannot derive output_dt_s: invalid initial state or mu.")

        denom = (2.0 / r0) - (v0 * v0 / mu)
        if denom <= 0.0 or (not math.isfinite(denom)):
            raise ValueError(
                "time_cfg.output_dt_s is None, but the orbit appears unbound/degenerate. "
                "Set output_dt_s explicitly."
            )

        a = 1.0 / denom
        T = 2.0 * math.pi * math.sqrt((a * a * a) / mu)

        spp = int(getattr(time_cfg, "samples_per_period", 360) or 360)
        spp = max(1, spp)
        dt_out_user = float(T) / float(spp)
    else:
        dt_out_user = float(dt_out_raw)

    if dt_out_user <= 0.0 or (not np.isfinite(dt_out_user)):
        raise ValueError("time_cfg.output_dt_s must be positive and finite.")
    # Cap output points (owned by PropagatorConfig; TimeConfig may optionally override)
    max_points_cap = int(getattr(time_cfg, "max_points_cap", getattr(cfg, "max_points_cap", 200_000)))

    dt_out = _clamp_output_dt(t0, tf, float(dt_out_user), max_points_cap, verbose)
    t_eval = make_time_grid(t0, tf, dt_out)

    rhs = dynamics.build_rhs()
    R_ref_m, mu_m3s2 = _get_ref_radius_and_mu(dynamics)

    # Terrain-aware telemetry is optional. The actual hybrid impact event uses a
    # similar capability deeper in the propagator, but surfacing the sampled
    # local radius here lets the desktop UI explain *why* a run stopped near the
    # surface instead of only showing mean-radius altitude.
    telem_r_i_to_bf: Optional[Callable[[float, np.ndarray], np.ndarray]] = None
    telem_surface_radius_m: Optional[Callable[[float, float], float]] = None
    if topo_grid is not None:
        try:
            telem_r_i_to_bf = _build_r_i_to_bf_from_rot_table(dynamics)
            if telem_r_i_to_bf is not None:
                telem_surface_radius_m = _build_surface_radius_sampler(topo_grid)
        except Exception:
            telem_r_i_to_bf = None
            telem_surface_radius_m = None

    # Optional: stream compact JSON telemetry for UI live plots/progress.
    # Controlled explicitly via config (no env-var "magic").
    # Telemetry is always enabled (project-wide default).
    # Cadence can be tuned via cfg.telem_cadence_s; if <=0, a sensible default is derived.
    enable_telem_json = True
    telem_cadence_s: float = float(getattr(cfg, "telem_cadence_s", getattr(cfg, "telemetry_cadence_s", 0.0)) or 0.0)
    if enable_telem_json and telem_cadence_s <= 0.0:
        hb_h = float(getattr(cfg, "heartbeat_hours", 0.0) or 0.0)
        if hb_h > 0.0:
            telem_cadence_s = max(5.0, hb_h * 3600.0)
        else:
            # Fallback: ~60 output samples, but at least 60s
            telem_cadence_s = max(60.0, float(dt_out) * 60.0)

    if enable_telem_json and telem_cadence_s > 0.0:
        last_telem_t = float(t0) - float(telem_cadence_s)
        rhs_base = rhs

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            nonlocal last_telem_t
            dy = rhs_base(t, y)
            if (float(t) - float(last_telem_t)) >= float(telem_cadence_s):
                telem = _make_telem_dict(
                    t_s=float(t - t0),
                    y=y,
                    R_ref_m=float(R_ref_m),
                    mu_m3s2=float(mu_m3s2),
                    t_frame_s=float(t),
                    r_i_to_bf=telem_r_i_to_bf,
                    surface_radius_m=telem_surface_radius_m,
                )
                if telem is not None:
                    print(json.dumps(telem, separators=(",", ":")), flush=True)
                last_telem_t = float(t)
            return dy

    # -------------------------------------------------------------------------
    # 3) Max-step logic (Nyquist cap vs user cap)
    # -------------------------------------------------------------------------
    degree = _get_sh_degree(dynamics)
    topo_present = topo_grid is not None

    nyq_max: Optional[float] = None
    if bool(getattr(cfg, "use_nyquist_max_step", False)):
        try:
            nyq_max = float(nyquist_max_step_s(
                R_ref_m=float(R_ref_m),
                mu_m3s2=float(mu_m3s2),
                degree=int(max(1, degree)),
                r_min_alt_km=(0.0 if topo_present else float(_get_impact_alt_km(cfg) if _get_detect_impact(cfg) else 0.0)),
                safety_div=float(getattr(cfg, "nyquist_safety_div", 8.0)),
                v_margin=float(getattr(cfg, "nyquist_v_margin", 1.2)),
            ))
        except Exception:
            nyq_max = None

    if nyq_max is None or (not np.isfinite(nyq_max)) or nyq_max <= 0.0:
        nyq_max = float(dt_out)

    if getattr(cfg, "user_max_step_s", None) is None:
        max_step = float(nyq_max)
        if verbose:
            print(f"[STEP] Nyquist max_step_s={max_step:.6f} (deg={degree})", flush=True)
    else:
        max_step = min(float(cfg.user_max_step_s), float(nyq_max))
        if verbose:
            print(f"[STEP] user_max_step={float(cfg.user_max_step_s):g}s, nyquist={nyq_max:.6f}s -> using {max_step:.6f}s", flush=True)

    # -------------------------------------------------------------------------
    # 4) Events
    # -------------------------------------------------------------------------
    events = build_events(dynamics, cfg, topo_grid=topo_grid, add_stop_event=bool(stop_file))
    if extra_events:
        for ev in list(extra_events):
            events.append(_wrap_event_first6(ev))


    # -------------------------------------------------------------------------
    # 5) Integrate
    # -------------------------------------------------------------------------
    if _is_symplectic_method(getattr(cfg, "method", "DOP853")):
        meth_name = str(getattr(cfg, "method", "VV"))
        if verbose:
            print(f"[PROP] Fixed-step {meth_name}: dt_out={dt_out:g}s, max_step={max_step:.6f}s", flush=True)

        if y0_arr.size != 6:
            raise ValueError(
                "Fixed-step symplectic integrators (VV/Y4) support only the 6D state [x,y,z,vx,vy,vz]. "
                f"Got initial state size={int(y0_arr.size)}. Use a SciPy integrator (e.g., DOP853/RK45) for augmented states."
            )

        ode_like, impacted, t_imp, y_imp, stopped_early, stop_reason, t_stop = _integrate_fixed_step(
            rhs=rhs,
            t_eval=t_eval,
            y0=y0_arr[:6],
            max_step=max_step,
            method=meth_name,
            events=events,
            R_ref_m=float(R_ref_m),
            mu_m3s2=float(mu_m3s2),
            verbose=verbose,
            heartbeat_hours=float(getattr(cfg, "heartbeat_hours", 0.0)),
            stop_file=stop_file,
            checkpoint_path=checkpoint_path,
        )

        t_out = np.asarray(ode_like.t, dtype=np.float64)
        y_row = np.asarray(ode_like.y, dtype=np.float64).T  # (N,6)

        res = PropagationResult(
            t=t_out,
            y=y_row,
            ode=ode_like,
            impacted=bool(impacted),
            t_impact_s=(float(t_imp) if t_imp is not None else None),
            y_impact=(np.asarray(y_imp, dtype=np.float64) if y_imp is not None else None),
            stopped_early=bool(stopped_early),
            stop_reason=stop_reason if stop_reason else ("impact" if impacted else None),
            t_stop_s=t_stop,
            diagnostics={},
        )

    else:
        if solve_ivp is None:
            raise ImportError("SciPy is required for adaptive integration (solve_ivp not available).")

        method = str(getattr(cfg, "method", "DOP853")).strip().upper()
        if method not in ("DOP853", "RK45", "RK23", "RADAU", "BDF", "LSODA"):
            method = "DOP853"

        if verbose:
            print(f"[PROP] solve_ivp method={method} | dt_out={dt_out:g}s | max_step={max_step:.6f}s", flush=True)

        def _solve_span(t_start: float, t_end: float, y_start: np.ndarray, t_eval_span: np.ndarray):
            return solve_ivp(
                fun=rhs,
                t_span=(float(t_start), float(t_end)),
                y0=np.asarray(y_start, dtype=np.float64),
                method=method,
                t_eval=np.asarray(t_eval_span, dtype=np.float64),
                rtol=float(getattr(cfg, "rtol", 1e-9)),
                atol=float(getattr(cfg, "atol", 1e-12)),
                max_step=float(max_step),
                events=(events if events else None),
                dense_output=False,
                vectorized=False,
            )

        total_span = tf - t0
        chunk_s = getattr(cfg, "chunk_s", None)
        if chunk_s is not None:
            chunk_s = float(chunk_s)
            if chunk_s <= 0.0 or chunk_s >= total_span:
                chunk_s = None

        stopped_early = False
        stop_reason: Optional[str] = None
        chunk_idx = 0

        if chunk_s is None:
            sol = _solve_span(t0, tf, y0_arr, t_eval)
            t_cat = np.asarray(sol.t, dtype=np.float64)
            y_cat = np.asarray(sol.y, dtype=np.float64)
            t_events = [np.asarray(te, dtype=np.float64) for te in (sol.t_events or [])]
            y_events = [np.asarray(ye, dtype=np.float64) for ye in (sol.y_events or [])]
        else:
            t_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []

            n_ev = len(events) if events else 0
            t_events_acc: List[List[np.ndarray]] = [[] for _ in range(n_ev)]
            y_events_acc: List[List[np.ndarray]] = [[] for _ in range(n_ev)]

            y_curr = y0_arr.copy()
            t_curr = float(t0)

            while t_curr < tf - 1e-12:
                if _stop_requested(stop_file) and (not bool(getattr(cfg, "stop_event_in_scipy", False))):
                    stopped_early = True
                    stop_reason = "stop file"
                    break

                t_next = min(tf, t_curr + float(chunk_s))
                mask = (t_eval >= t_curr - 1e-12) & (t_eval <= t_next + 1e-12)
                t_eval_span = t_eval[mask]
                if t_eval_span.size < 2:
                    t_eval_span = np.array([t_curr, t_next], dtype=np.float64)

                sol_k = _solve_span(t_curr, t_next, y_curr, t_eval_span)

                if not t_parts:
                    t_parts.append(np.asarray(sol_k.t, dtype=np.float64))
                    y_parts.append(np.asarray(sol_k.y, dtype=np.float64))
                else:
                    t_parts.append(np.asarray(sol_k.t[1:], dtype=np.float64))
                    y_parts.append(np.asarray(sol_k.y[:, 1:], dtype=np.float64))

                if getattr(sol_k, "t_events", None) is not None:
                    for i in range(n_ev):
                        te = sol_k.t_events[i] if i < len(sol_k.t_events) else np.array([], dtype=np.float64)
                        ye = sol_k.y_events[i] if i < len(sol_k.y_events) else np.zeros((0, y0_arr.size), dtype=np.float64)
                        t_events_acc[i].append(np.asarray(te, dtype=np.float64))
                        y_events_acc[i].append(np.asarray(ye, dtype=np.float64))

                if checkpoint_path and bool(getattr(cfg, "checkpoint_every_chunk", False)):
                    try:
                        ck_mode = str(getattr(cfg, "checkpoint_mode", "full")).strip().lower()
                        if ck_mode in ("latest", "state", "last"):
                            _atomic_save_npz(
                                checkpoint_path,
                                t=np.asarray([t_curr], dtype=np.float64),
                                y_row=y_curr.reshape(1, -1),
                            )
                        elif ck_mode in ("chunks", "chunk"):
                            base = str(checkpoint_path)
                            chunk_path = f"{base}.chunk{chunk_idx:06d}.npz"
                            _atomic_save_npz(chunk_path, t=np.asarray(sol_k.t, dtype=np.float64), y_row=np.asarray(sol_k.y, dtype=np.float64).T)
                            _atomic_save_npz(
                                checkpoint_path,
                                t=np.asarray([t_curr], dtype=np.float64),
                                y_row=y_curr.reshape(1, -1),
                            )
                        else:
                            t_tmp = np.concatenate(t_parts) if t_parts else np.array([], dtype=np.float64)
                            y_tmp = np.concatenate(y_parts, axis=1) if y_parts else np.zeros((y0_arr.size, 0), dtype=np.float64)
                            _atomic_save_npz(checkpoint_path, t=t_tmp, y_row=y_tmp.T)
                    except Exception as exc:
                        import warnings
                        warnings.warn(f"Checkpoint write failed: {exc}", RuntimeWarning)

                if int(getattr(sol_k, "status", 0)) == 1:
                    stopped_early = True
                    stop_reason = "event"
                    break

                if not bool(getattr(sol_k, "success", True)):
                    stopped_early = True
                    stop_reason = "integration failed"
                    break

                y_curr = np.asarray(sol_k.y[:, -1], dtype=np.float64).copy()
                t_curr = float(sol_k.t[-1])
                chunk_idx += 1

            t_cat = np.concatenate(t_parts) if t_parts else np.array([t0], dtype=np.float64)
            y_cat = np.concatenate(y_parts, axis=1) if y_parts else y0_arr.reshape(-1, 1)

            t_events = [np.concatenate(ch) if ch else np.array([], dtype=np.float64) for ch in t_events_acc]
            y_events = [np.concatenate(ch, axis=0) if ch else np.zeros((0, y0_arr.size), dtype=np.float64) for ch in y_events_acc]

            sol = SimpleNamespace(
                t=t_cat,
                y=y_cat,
                t_events=t_events,
                y_events=y_events,
                success=True,
                status=(1 if stopped_early else 0),
                message=("chunked ok" if not stopped_early else "stopped early"),
                nfev=np.nan,
            )

        y_row = np.asarray(y_cat, dtype=np.float64).T

        impacted = False
        t_imp = None
        y_imp = None
        idx_impact = _find_event_index(events, "impact")
        if idx_impact is not None:
            try:
                if idx_impact < len(t_events) and np.asarray(t_events[idx_impact]).size > 0:
                    impacted = True
                    t_imp = float(np.asarray(t_events[idx_impact])[0])
                    y_imp = np.asarray(np.asarray(y_events[idx_impact])[0], dtype=np.float64)
            except Exception:
                pass

        if stop_reason is None:
            try:
                if impacted:
                    stop_reason = "impact"
                else:
                    idx_stop = _find_event_index(events, "stop")
                    if (
                        stop_file and bool(getattr(cfg, "stop_event_in_scipy", False))
                        and idx_stop is not None and idx_stop < len(t_events)
                        and np.asarray(t_events[idx_stop]).size > 0
                    ):
                        stop_reason = "stop file"
                if stop_reason is None and any((te is not None and np.asarray(te).size > 0) for te in t_events):
                    stop_reason = "event"
            except Exception:
                pass

        t_stop = None
        if stop_file and bool(getattr(cfg, "stop_event_in_scipy", False)) and (not impacted):
            idx_stop = _find_event_index(events, "stop")
            if idx_stop is not None:
                try:
                    if idx_stop < len(t_events) and np.asarray(t_events[idx_stop]).size > 0:
                        t_stop = float(np.asarray(t_events[idx_stop])[0])
                except Exception:
                    pass

        if checkpoint_path:
            try:
                _atomic_save_npz(checkpoint_path, t=np.asarray(t_cat, dtype=np.float64), y_row=y_row)
            except Exception as exc:
                import warnings
                warnings.warn(f"Checkpoint write failed: {exc}", RuntimeWarning)

        res = PropagationResult(
            t=np.asarray(t_cat, dtype=np.float64),
            y=y_row,
            ode=sol,
            t_events=list(t_events),
            y_events=list(y_events),
            impacted=bool(impacted),
            t_impact_s=(float(t_imp) if t_imp is not None else None),
            y_impact=(np.asarray(y_imp, dtype=np.float64) if y_imp is not None else None),
            stopped_early=bool(stopped_early) or bool(impacted),
            stop_reason=stop_reason,
            t_stop_s=t_stop,
            diagnostics={},
        )

    # -------------------------------------------------------------------------
    # 6) Diagnostics + Optional 2-body baseline
    # -------------------------------------------------------------------------
    wall = time.perf_counter() - t_wall0
    nfev = float(getattr(res.ode, "nfev", np.nan)) if res.ode is not None else np.nan
    res.diagnostics = {
        "wall_time_s": float(wall),
        "output_dt_s": float(dt_out),
        "max_step_s": float(max_step),
        "degree": float(degree),
        "n_points": float(res.t.size),
        "nfev": float(nfev) if np.isfinite(nfev) else float("nan"),
        "method_symplectic": float(1.0 if _is_symplectic_method(getattr(cfg, "method", "DOP853")) else 0.0),
    }

    if bool(getattr(cfg, "compute_2body_baseline", False)):
        res.baseline = _compute_2body_baseline(
            t_eval=res.t,
            y0=y0_arr[:6],
            mu_m3s2=float(mu_m3s2),
            cfg=cfg,
            max_step=float(max_step),
        )

    return res



# =============================================================================
# 6.                       2-body baseline helper
# =============================================================================

def _compute_2body_baseline(
    *,
    t_eval: np.ndarray,
    y0: np.ndarray,
    mu_m3s2: float,
    cfg: PropagatorConfig,
    max_step: float,
) -> Optional[PropagationResult]:
    """Compute a simple 2-body (central-gravity) reference trajectory.

    This is a diagnostic baseline to compare against the full dynamics model.
    Returns None if the time grid is invalid or SciPy is unavailable (adaptive path).
    """
    t_eval = np.asarray(t_eval, dtype=np.float64).reshape(-1)
    if t_eval.size < 2 or np.any(np.diff(t_eval) <= 0.0):
        return None

    y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
    if y0.size < 6:
        return None
    y0 = y0[:6].copy()

    mu = float(mu_m3s2)
    if (not np.isfinite(mu)) or mu <= 0.0:
        return None

    def rhs2(t: float, y: np.ndarray) -> np.ndarray:
        rx, ry, rz = float(y[0]), float(y[1]), float(y[2])
        vx, vy, vz = float(y[3]), float(y[4]), float(y[5])

        r2 = rx * rx + ry * ry + rz * rz
        r2 = max(r2, 1e-30)
        inv_r = 1.0 / math.sqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r

        ax = -mu * rx * inv_r3
        ay = -mu * ry * inv_r3
        az = -mu * rz * inv_r3

        dy = np.empty(6, dtype=np.float64)
        dy[0] = vx
        dy[1] = vy
        dy[2] = vz
        dy[3] = ax
        dy[4] = ay
        dy[5] = az
        return dy

    # Symplectic baseline (fixed-step)
    if _is_symplectic_method(cfg.method):
        ode_like, _, _, _, _, _, _ = _integrate_fixed_step(
            rhs=rhs2,
            t_eval=t_eval,
            y0=y0,
            max_step=float(max_step),
            method=str(cfg.method),
            events=None,
            R_ref_m=float(R_MOON),
            mu_m3s2=float(mu),
            verbose=False,
            heartbeat_hours=0.0,
            stop_file=None,
            checkpoint_path=None,
        )
        t_out = np.asarray(ode_like.t, dtype=np.float64)
        y_row = np.asarray(ode_like.y, dtype=np.float64).T
        return PropagationResult(
            t=t_out,
            y=y_row,
            ode=ode_like,
            diagnostics={"baseline": 1.0, "solver": "fixed-step", "success": 1.0},
        )

    # Adaptive baseline (solve_ivp)
    if solve_ivp is None:
        return None

    method = str(cfg.method).strip().upper() or "DOP853"
    rtol = float(getattr(cfg, "baseline_rtol", getattr(cfg, "rtol", 1e-9)))
    atol = float(getattr(cfg, "baseline_atol", getattr(cfg, "atol", 1e-12)))

    try:
        sol = solve_ivp(
            fun=rhs2,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=float(max_step),
            events=None,
            dense_output=False,
            vectorized=False,
        )
    except Exception:
        sol = solve_ivp(
            fun=rhs2,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=y0,
            method="DOP853",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=float(max_step),
            events=None,
            dense_output=False,
            vectorized=False,
        )

    return PropagationResult(
        t=np.asarray(sol.t, dtype=np.float64),
        y=np.asarray(sol.y, dtype=np.float64).T,
        ode=sol,
        diagnostics={"baseline": 1.0, "solver": method, "success": float(bool(getattr(sol, "success", True)))},
    )



# =============================================================================
# 8.                             SMOKE TEST
# =============================================================================

__all__ = [
    # Main entry point
    "propagate",

    # Return type (handy for callers)
    "PropagationResult",

    # Advanced / optional helpers
    "make_time_grid",
    "build_events",
]
