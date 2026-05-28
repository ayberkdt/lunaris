# ST_LRPS/analysis/postprocess.py
# -*- coding: utf-8 -*-
"""
ST_LRPS post-processing utilities.

This module converts raw propagation output (time grid + state history) into a
plot/report-friendly "history" dictionary with derived series and optional
diagnostics.

Core outputs
------------
- Osculating orbital elements:
  a [km], e [-], i/RAAN/argp/nu [deg]
- Altitude above the reference radius [km]
- Specific orbital energy ε [J/kg] and angular momentum norm h [m²/s]
- Relative drifts:
  Δε / max(|ε0|, eps), Δh / max(|h0|, eps)
- Event indices:
  periapsis, apoapsis, impact (optional threshold altitude)

Optional products (ctx-dependent)
---------------------------------
- Acceleration component magnitude breakdown (if ctx exposes a breakdown API)
- Ground track (lat/lon) using inertial->body-fixed attitude tables (quaternion or DCM)
- Eclipse mask (hard-shadow approximation) using ephemeris Sun/Earth vectors

Design notes
------------
- The API is intentionally "fail-soft" for optional products: if a required table
  is missing, the core history is still returned.
- The core (t, y, mu, R_body) contract is validated strictly.
"""

# =============================================================================
# 0.                               IMPORTS
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# Prefer package-relative imports; fall back to local imports when running as a script.
try:
    from ..common.math_utils import batch_y_to_elements, quat_rotate_vec
except Exception:  # pragma: no cover
    from lunaris.common.math_utils import batch_y_to_elements, quat_rotate_vec



# =============================================================================
# 1.                           INTERNAL HELPERS
# =============================================================================

def _as_np(x, dtype=None) -> np.ndarray:
    """
    Safe NumPy conversion.

    - None -> empty 1D array
    - dtype is applied only if provided (prevents accidental float-casting)
    """
    if x is None:
        return np.empty((0,), dtype=float if dtype is None else dtype)

    if isinstance(x, np.ndarray):
        if dtype is None:
            return x
        try:
            return x.astype(dtype, copy=False)
        except Exception:
            return np.asarray(x, dtype=dtype)

    try:
        return np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)
    except Exception:
        return np.asarray(x)


def _ensure_3xN(a: np.ndarray) -> np.ndarray:
    """
    Ensure vector history is shaped (3, N).
    Accepts (3, N) or (N, 3). Returns a view when possible.
    """
    a = _as_np(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array for vector history, got shape={a.shape}")
    if a.shape[0] == 3:
        return a
    if a.shape[1] == 3:
        return a.T
    raise ValueError(f"Expected shape (3,N) or (N,3), got shape={a.shape}")


def _norm(v: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.linalg.norm(v, axis=axis)


def _central_specific_energy(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    """Specific orbital energy for central gravity."""
    r = _ensure_3xN(r)
    v = _ensure_3xN(v)

    rr = _norm(r, axis=0)
    rr = np.maximum(rr, 1e-30)
    vv2 = np.einsum("ij,ij->j", v, v)  # stable + fast
    return 0.5 * vv2 - float(mu) / rr


def _specific_ang_mom_norm(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angular momentum magnitude |h| where h = r x v."""
    r = _ensure_3xN(r)
    v = _ensure_3xN(v)
    if r.size == 0 or v.size == 0:
        return np.empty((0,), dtype=float)

    h = np.cross(r.T, v.T).T  # (3,N)
    return _norm(h, axis=0)


def _detect_peri_apo_indices(r: np.ndarray, v: np.ndarray, *, vr_eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect periapsis/apoapsis indices via sign changes of radial velocity:
        vr = dot(r, v)/|r|
    Robust against vr ~ 0 plateaus by applying a small epsilon and forward-filling zeros.
    """
    r = _ensure_3xN(r)
    v = _ensure_3xN(v)

    rr = _norm(r, axis=0)
    rr = np.maximum(rr, 1e-30)
    vr = np.einsum("ij,ij->j", r, v) / rr

    # Treat tiny |vr| as zero, then forward-fill sign to avoid spurious events
    s = np.sign(vr)
    if s.size:
        zero = np.abs(vr) <= float(vr_eps)
        if np.any(zero):
            # forward-fill zeros with previous non-zero sign
            for i in range(1, s.size):
                if zero[i]:
                    s[i] = s[i-1]
            # if the first is zero, back-fill from first non-zero (if exists)
            if zero[0]:
                nz = np.flatnonzero(~zero)
                if nz.size:
                    s[0] = s[nz[0]]

    ds = np.diff(s)

    peri = np.where(ds > 0)[0] + 1  # - to +
    apo  = np.where(ds < 0)[0] + 1  # + to -

    return peri.astype(int), apo.astype(int)


def _detect_impact_index(alt_km: np.ndarray, impact_alt_km: Optional[float]) -> Optional[int]:
    """Return first index where altitude crosses below impact_alt_km (if provided)."""
    if impact_alt_km is None:
        return None
    alt = _as_np(alt_km, dtype=float)
    if alt.size == 0:
        return None
    hit = np.where(alt <= float(impact_alt_km))[0]
    return int(hit[0]) if hit.size else None



# =============================================================================
# 2.                 HISTORY NORMALIZATION / EXTRACTORS
# =============================================================================

def _first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    """Return the first key that exists in `d` with a non-None value."""
    for k in keys:
        if k in d:
            v = d.get(k)
            if v is not None:
                return v
    return None


def _maybe_km(x: np.ndarray) -> np.ndarray:
    """
    Best-effort length unit normalization to kilometers.

    Heuristic:
    - If median(|x|) is large (typical of meters), assume meters and divide by 1000.
    - Otherwise, return as-is (assume already in km).
    """
    x = _as_np(x, dtype=float)
    if x.size == 0:
        return x
    with np.errstate(invalid="ignore"):
        m = float(np.nanmedian(np.abs(x)))
    if np.isfinite(m) and m > 1e5:  # meters-scale typical orbital radii
        return x / 1000.0
    return x


def _maybe_deg(x: np.ndarray) -> np.ndarray:
    """
    Best-effort angle unit normalization to degrees.

    Heuristic:
    - If 95th percentile is within a few radians, assume radians and convert to degrees.
    - Otherwise, return as-is (assume already in degrees).
    """
    x = _as_np(x, dtype=float)
    if x.size == 0:
        return x
    with np.errstate(invalid="ignore"):
        p95 = float(np.nanpercentile(np.abs(x), 95))
    if np.isfinite(p95) and p95 <= (2.0 * np.pi + 0.5):
        return np.degrees(x)
    return x


def _meta_get(meta: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Return the first non-None value from `meta` matching any of the provided keys.

    Supports dotted-path lookup for nested mappings:
        _meta_get(meta, "integrator.method", "propagator.method")
    """
    def _get_path(d: Any, path: str) -> Any:
        cur = d
        for part in path.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                return None
            cur = cur[part]
        return cur

    for k in keys:
        if isinstance(meta, Mapping) and k in meta:
            v = meta.get(k)
            if v is not None:
                return v
        if "." in k:
            v = _get_path(meta, k)
            if v is not None:
                return v
        if "/" in k:
            v = _get_path(meta, k.replace("/", "."))
            if v is not None:
                return v
    return default


def _cfg_first(cfg: Any, *paths: str, default: Any = None) -> Any:
    """
    Return the first non-None attribute/value for dotted paths.

    Works with dataclass-like objects (attribute access) and nested dict snapshots.
    Example:
        _cfg_first(cfg, "propagator.events.impact_alt_km", default=None)
    """
    def _get_one(obj: Any, path: str) -> Any:
        cur = obj
        for part in path.split("."):
            if cur is None:
                return None
            if isinstance(cur, dict):
                if part not in cur:
                    return None
                cur = cur.get(part)
                continue
            if hasattr(cur, part):
                cur = getattr(cur, part)
                continue
            return None
        return cur

    for p in paths:
        try:
            v = _get_one(cfg, p)
        except Exception:
            v = None
        if v is not None:
            return v
    return default


# ------------------------------------------------------------
# Time extractors
# ------------------------------------------------------------

def extract_time_seconds(history: Mapping[str, Any]) -> np.ndarray:
    """
    Return simulation time in seconds.

    Expected canonical keys:
      - "t_s" (preferred)
      - "t_days" (converted to seconds)
    """
    t_s = _first_present(history, ["t_s"])
    if t_s is not None:
        return _as_np(t_s, dtype=float)

    t_days = _first_present(history, ["t_days"])
    if t_days is not None:
        return _as_np(t_days, dtype=float) * 86400.0

    return np.empty((0,), dtype=float)


def extract_time_days(history: Mapping[str, Any]) -> np.ndarray:
    """
    Return simulation time in days.

    Expected canonical keys:
      - "t_days" (preferred)
      - "t_s" (converted to days)
    """
    t_days = _first_present(history, ["t_days"])
    if t_days is not None:
        return _as_np(t_days, dtype=float)

    t_s = _first_present(history, ["t_s"])
    t_s = _as_np(t_s, dtype=float)
    if t_s.size == 0:
        return t_s
    return t_s / 86400.0


# ------------------------------------------------------------
# State extraction (position/velocity)
# ------------------------------------------------------------

def _maybe_r_m(r: np.ndarray) -> np.ndarray:
    """
    Normalize position vectors to meters.

    Heuristic:
    - If median radius looks like km-scale, multiply by 1000.
    """
    r = _as_np(r, dtype=float)
    if r.size == 0:
        return r
    with np.errstate(invalid="ignore"):
        m = float(np.nanmedian(np.linalg.norm(r, axis=1))) if r.ndim == 2 and r.shape[1] == 3 else float(np.nanmedian(np.abs(r)))
    if np.isfinite(m) and m < 1e5:  # likely km-scale (e.g., ~1700 km)
        return r * 1000.0
    return r


def _maybe_v_mps(v: np.ndarray) -> np.ndarray:
    """
    Normalize velocity vectors to m/s.

    Heuristic:
    - If median speed looks like km/s-scale, multiply by 1000.
    """
    v = _as_np(v, dtype=float)
    if v.size == 0:
        return v
    with np.errstate(invalid="ignore"):
        m = float(np.nanmedian(np.linalg.norm(v, axis=1))) if v.ndim == 2 and v.shape[1] == 3 else float(np.nanmedian(np.abs(v)))
    if np.isfinite(m) and m < 50.0:  # likely km/s (e.g., ~1.6 km/s)
        return v * 1000.0
    return v


def _extract_rv_vectors(history: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return position/velocity histories as (N, 3) arrays in SI units:
      - r: meters
      - v: m/s

    Expected canonical representation:
      - history["y"] shaped (6, N) or (N, 6)
    Optional canonical fallbacks:
      - history["r_m"] shaped (N, 3) or (3, N)
      - history["v_mps"] shaped (N, 3) or (3, N)
    """
    y = _first_present(history, ["y"])
    y = _as_np(y, dtype=float)

    if y.ndim == 2:
        if y.shape[0] >= 6 and y.shape[1] >= 1:
            # solve_ivp-style (6, N)
            r = y[0:3, :].T
            v = y[3:6, :].T
            return _maybe_r_m(r), _maybe_v_mps(v)

        if y.shape[1] >= 6 and y.shape[0] >= 1:
            # row-major (N, 6+)
            r = y[:, 0:3]
            v = y[:, 3:6]
            return _maybe_r_m(r), _maybe_v_mps(v)

    # Explicit r/v fallbacks (canonical names only)
    r = _as_np(_first_present(history, ["r_m"]), dtype=float)
    v = _as_np(_first_present(history, ["v_mps"]), dtype=float)

    def _to_N3(a: np.ndarray) -> np.ndarray:
        if a.ndim != 2:
            return np.empty((0, 3), dtype=float)
        if a.shape[1] == 3:
            return a
        if a.shape[0] == 3:
            return a.T
        return np.empty((0, 3), dtype=float)

    rN3 = _to_N3(r)
    vN3 = _to_N3(v)

    if rN3.size and vN3.size:
        n = min(rN3.shape[0], vN3.shape[0])
        return _maybe_r_m(rN3[:n]), _maybe_v_mps(vN3[:n])

    return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)


# ------------------------------------------------------------
# Orbital elements
# ------------------------------------------------------------

def extract_elements(history: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract (or derive) osculating orbital elements for plotting.

    Returns:
      - a_km
      - e
      - i_deg
      - raan_deg
      - argp_deg

    Canonical keys (preferred if present):
      - a_km, e, i_deg, raan_deg, argp_deg

    If any are missing, elements are derived from history["y"] using batch_y_to_elements
    with mu taken from history["meta"] (mu_m3s2) or history["mu_m3s2"].
    """
    a = _as_np(_first_present(history, ["a_km"]), dtype=float)
    e = _as_np(_first_present(history, ["e"]), dtype=float)
    i = _as_np(_first_present(history, ["i_deg"]), dtype=float)
    raan = _as_np(_first_present(history, ["raan_deg"]), dtype=float)
    argp = _as_np(_first_present(history, ["argp_deg"]), dtype=float)

    need = (a.size == 0) or (e.size == 0) or (i.size == 0) or (raan.size == 0) or (argp.size == 0)
    if not need:
        return {"a_km": a, "e": e, "i_deg": i, "raan_deg": raan, "argp_deg": argp}

    # Derive from state if possible
    y = _first_present(history, ["y"])
    y = _as_np(y, dtype=float)

    y6N = None
    if y.ndim == 2:
        if y.shape[0] >= 6:
            y6N = y[:6, :]
        elif y.shape[1] >= 6:
            y6N = y[:, :6].T

    mu = None
    meta = history.get("meta") if isinstance(history, dict) else None
    if isinstance(meta, Mapping):
        mu = _meta_get(meta, "mu_m3s2", "mu", "GM", default=None)
    if mu is None:
        mu = _first_present(history, ["mu_m3s2", "mu", "GM"])

    if y6N is not None and y6N.size and mu is not None:
        try:
            mu_f = float(mu)
            if mu_f > 0.0:
                a_m, e2, inc, raan2, argp2, *_ = batch_y_to_elements(y6N.astype(float), mu_f, mode="coe")
                a = _as_np(a_m, dtype=float) / 1000.0
                e = _as_np(e2, dtype=float)
                i = np.degrees(_as_np(inc, dtype=float))
                raan = np.degrees(_as_np(raan2, dtype=float))
                argp = np.degrees(_as_np(argp2, dtype=float))
        except Exception:
            pass

    return {"a_km": a, "e": e, "i_deg": i, "raan_deg": raan, "argp_deg": argp}


# ------------------------------------------------------------
# Invariants and diagnostics
# ------------------------------------------------------------

def extract_invariants(history: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract common invariants/diagnostics.

    Canonical preferred keys:
      - r_norm_km
      - v_norm_kmps
      - energy_Jkg
      - h_norm_m2s
      - rel_energy_drift
      - rel_h_drift

    If norms are missing, they are derived from r/v vectors.
    """
    r_norm = _maybe_km(_as_np(_first_present(history, ["r_norm_km"]), dtype=float))
    v_norm = _as_np(_first_present(history, ["v_norm_kmps"]), dtype=float)

    # If velocity norm looks like m/s, convert to km/s
    if v_norm.size:
        with np.errstate(invalid="ignore"):
            if float(np.nanmedian(np.abs(v_norm))) > 50.0:
                v_norm = v_norm / 1000.0

    if r_norm.size == 0 or v_norm.size == 0:
        r_vec, v_vec = _extract_rv_vectors(history)
        if r_vec.size and r_norm.size == 0:
            r_norm = _maybe_km(np.linalg.norm(r_vec, axis=1))
        if v_vec.size and v_norm.size == 0:
            vn = np.linalg.norm(v_vec, axis=1)  # m/s
            with np.errstate(invalid="ignore"):
                if float(np.nanmedian(np.abs(vn))) > 50.0:
                    vn = vn / 1000.0
            v_norm = vn

    energy = _as_np(_first_present(history, ["energy_Jkg"]), dtype=float)
    h = _as_np(_first_present(history, ["h_norm_m2s"]), dtype=float)

    relE = _as_np(_first_present(history, ["rel_energy_drift"]), dtype=float)
    relh = _as_np(_first_present(history, ["rel_h_drift"]), dtype=float)

    return {
        "r_norm_km": r_norm,
        "v_norm_kmps": v_norm,
        "energy_Jkg": energy,
        "h_norm_m2s": h,
        "rel_energy_drift": relE,
        "rel_h_drift": relh,
    }


def extract_altitude_km(history: Mapping[str, Any], meta: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """
    Altitude [km].

    Preferred canonical key:
      - "alt_km"

    If missing, derive from |r| - R_ref using:
      - r extracted from history["y"] (meters)
      - R_ref from meta/body_radius_m or history["body_radius_m"]
    """
    alt = _first_present(history, ["alt_km"])
    if alt is not None:
        return _as_np(alt, dtype=float)

    r_vec, _ = _extract_rv_vectors(history)
    if r_vec.size == 0:
        return np.empty((0,), dtype=float)

    r_norm_m = np.linalg.norm(r_vec, axis=1)

    Rm = None
    if meta is not None:
        Rm = _meta_get(meta, "body_radius_m", "R_ref_m", default=None)
    if Rm is None:
        Rm = _first_present(history, ["body_radius_m", "R_ref_m"])

    if Rm is None:
        return np.empty((0,), dtype=float)

    try:
        return ((r_norm_m - float(Rm)) / 1000.0).astype(float)
    except Exception:
        return np.empty((0,), dtype=float)


def extract_events(history: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return event indices dict:
      - peri_idx: int array
      - apo_idx: int array
      - impact_idx: Optional[int]

    Preferred canonical structure:
      history["events"] = {"peri_idx": [...], "apo_idx": [...], "impact_idx": int|None}
    """
    ev = _first_present(history, ["events"])
    if isinstance(ev, Mapping):
        peri = _as_np(ev.get("peri_idx", []), dtype=int)
        apo = _as_np(ev.get("apo_idx", []), dtype=int)
        impact = ev.get("impact_idx", None)
        try:
            impact_i = int(impact) if impact is not None else None
        except Exception:
            impact_i = None
        return {"peri_idx": peri, "apo_idx": apo, "impact_idx": impact_i}

    return {"peri_idx": np.empty((0,), dtype=int), "apo_idx": np.empty((0,), dtype=int), "impact_idx": None}



# =============================================================================
# 3.                        OPTIONAL DERIVED PRODUCTS
# =============================================================================

def _downsample_indices(n: int, max_samples: int) -> Optional[np.ndarray]:
    """Return evenly spaced indices if downsampling is needed, otherwise None."""
    if (max_samples is None) or (max_samples <= 0) or (n <= max_samples):
        return None
    return np.linspace(0, n - 1, int(max_samples), dtype=np.int64)


def _require_ephem(ctx: Any) -> Dict[str, Any]:
    """Fetch and validate ctx.ephem_data (strict)."""
    if ctx is None or not hasattr(ctx, "ephem_data"):
        raise ValueError("Strict ephemeris required: ctx.ephem_data is missing.")

    ephem = getattr(ctx, "ephem_data")
    if not isinstance(ephem, dict):
        raise TypeError("Strict ephemeris required: ctx.ephem_data must be a dict.")

    t_s = _as_np(ephem.get("t_s", None), dtype=float)
    if t_s.ndim != 1 or t_s.size < 2:
        raise ValueError("ephem_data['t_s'] must be a 1D array with length >= 2.")

    t0_s = ephem.get("t0_s", 0.0)
    try:
        t0_s = float(t0_s)
    except Exception as e:
        raise ValueError("ephem_data['t0_s'] must be float-like.") from e

    q_i2f = ephem.get("q_i2f", None)
    R_i2f = ephem.get("R_i2f", None)

    if (q_i2f is None) == (R_i2f is None):
        raise ValueError("ephem_data must provide exactly one of {'q_i2f', 'R_i2f'}.")

    if q_i2f is not None:
        q = _as_np(q_i2f, dtype=float)
        if q.ndim != 2 or q.shape[1] != 4 or q.shape[0] != t_s.size:
            raise ValueError("ephem_data['q_i2f'] must have shape (Nt,4) matching t_s.")
        ephem["q_i2f"] = q

    if R_i2f is not None:
        R = _as_np(R_i2f, dtype=float)
        if R.ndim != 3 or R.shape[1:] != (3, 3) or R.shape[0] != t_s.size:
            raise ValueError("ephem_data['R_i2f'] must have shape (Nt,3,3) matching t_s.")
        ephem["R_i2f"] = R

    rS = _as_np(ephem.get("r_sun_i_m", None), dtype=float)
    if rS.ndim != 2 or rS.shape[1] != 3 or rS.shape[0] != t_s.size:
        raise ValueError("ephem_data['r_sun_i_m'] must have shape (Nt,3) matching t_s.")
    ephem["r_sun_i_m"] = rS

    rE = ephem.get("r_earth_i_m", None)
    if rE is not None:
        rE = _as_np(rE, dtype=float)
        if rE.ndim != 2 or rE.shape[1] != 3 or rE.shape[0] != t_s.size:
            raise ValueError("ephem_data['r_earth_i_m'] must have shape (Nt,3) matching t_s.")
        ephem["r_earth_i_m"] = rE

    ephem["t_s"] = t_s
    ephem["t0_s"] = t0_s
    return ephem


def _map_sim_time_to_ephem_index(t_sim_s: np.ndarray, t_ephem_s: np.ndarray, t0_s: float) -> np.ndarray:
    """Map simulation times to ephemeris indices with nearest-left lookup."""
    t_sim_s = _as_np(t_sim_s, dtype=float)
    t_rel = t_sim_s - float(t0_s)
    j = np.searchsorted(t_ephem_s, t_rel, side="left")
    return np.clip(j, 0, int(t_ephem_s.size) - 1).astype(np.int64)


def _accel_breakdown_if_available(
    ctx: Any,
    t_s: np.ndarray,
    y: np.ndarray,
    max_samples: int = 20000,
) -> Dict[str, np.ndarray]:
    """
    Compute acceleration component magnitudes using strict API:
        ctx.accel_breakdown(t_s: float, y6: np.ndarray) -> dict[name -> vec3]

    Returns sampled magnitudes as:
        dict[name -> np.ndarray]
    """
    if ctx is None:
        return {}

    fn = getattr(ctx, "accel_breakdown", None)
    if not callable(fn):
        return {}

    t_s = _as_np(t_s, dtype=float)
    y = _as_np(y, dtype=float)
    if t_s.size == 0 or y.size == 0:
        return {}
    if y.ndim != 2 or y.shape[0] < 6 or y.shape[1] != t_s.size:
        raise ValueError("Strict state required: y must be (6,N) and match t_s length.")

    N = int(t_s.size)
    idx = _downsample_indices(N, max_samples)
    if idx is not None:
        t_work = t_s[idx]
        y_work = y[:6, idx]
    else:
        t_work = t_s
        y_work = y[:6, :]

    mags: Dict[str, list[float]] = {}
    for k in range(int(t_work.size)):
        d = fn(float(t_work[k]), y_work[:, k])
        if not isinstance(d, dict):
            raise TypeError("ctx.accel_breakdown must return dict[str, vec3].")
        for comp_name, avec in d.items():
            a = _as_np(avec, dtype=float).reshape(-1)
            if a.size < 3:
                raise ValueError(f"accel_breakdown component '{comp_name}' is not a 3-vector.")
            mags.setdefault(str(comp_name), []).append(float(np.linalg.norm(a[:3])))

    return {k: _as_np(v, dtype=float) for k, v in mags.items()}


def _groundtrack_if_available(
    ctx: Any,
    t_s: np.ndarray,
    y: np.ndarray,
    max_samples: int = 50000,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Convert inertial position history to body-fixed latitude/longitude (strict).

    Requires:
      - ctx.ephem_data validated by _require_ephem()
      - y is (6,N) or at least (3,N), with N == len(t_s)

    Returns downsampled lat/lon if max_samples applies:
      {"lat_deg": np.ndarray, "lon_deg": np.ndarray}
    """
    if ctx is None:
        return None

    t_s = _as_np(t_s, dtype=float)
    y = _as_np(y, dtype=float)
    if t_s.size == 0 or y.size == 0:
        return None
    if y.ndim != 2 or y.shape[0] < 3 or y.shape[1] != t_s.size:
        raise ValueError("Strict state required: y must be (>=3,N) and match t_s length.")

    N = int(t_s.size)
    idx = _downsample_indices(N, max_samples)
    if idx is not None:
        t_work = t_s[idx]
        r_i = y[:3, idx]
    else:
        t_work = t_s
        r_i = y[:3, :]

    ephem = _require_ephem(ctx)
    t_ephem = ephem["t_s"]
    j = _map_sim_time_to_ephem_index(t_work, t_ephem, ephem["t0_s"])

    n_points = int(t_work.size)
    r_bf = np.empty((3, n_points), dtype=np.float64)

    if "q_i2f" in ephem:
        q_tab = ephem["q_i2f"]  # (Nt,4)
        for k in range(n_points):
            q = q_tab[j[k]]
            rx, ry, rz = quat_rotate_vec(
                q[0], q[1], q[2], q[3],
                r_i[0, k], r_i[1, k], r_i[2, k],
            )
            r_bf[:, k] = (rx, ry, rz)
    else:
        Rtab = ephem["R_i2f"]  # (Nt,3,3)
        Rt = Rtab[j, :, :]     # (n,3,3)
        r_bf = np.einsum("nij,jn->in", Rt, r_i)

    x, yb, z = r_bf[0], r_bf[1], r_bf[2]
    rr = np.maximum(np.sqrt(x * x + yb * yb + z * z), 1e-30)

    lat = np.degrees(np.arcsin(np.clip(z / rr, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(yb, x))
    lon = (lon + 180.0) % 360.0 - 180.0

    return {"lat_deg": lat.astype(float), "lon_deg": lon.astype(float)}


def _eclipse_if_available(
    ctx: Any,
    t_s: np.ndarray,
    y: np.ndarray,
    R_body: float,
) -> Optional[Dict[str, Any]]:
    """
    Eclipse mask + summary (strict hard-shadow segment test).

    Requires:
      - ctx.ephem_data validated by _require_ephem()
      - ephem_data['r_sun_i_m'] (Nt,3)
    Optional:
      - ephem_data['r_earth_i_m'] (Nt,3)

    Returns:
      {"mask": bool[N], "total_s": float, "fraction": float}
    """
    if ctx is None:
        return None

    t_s = _as_np(t_s, dtype=float)
    y = _as_np(y, dtype=float)
    if t_s.size == 0 or y.size == 0:
        return None
    if y.ndim != 2 or y.shape[0] < 3 or y.shape[1] != t_s.size:
        raise ValueError("Strict state required: y must be (>=3,N) and match t_s length.")

    ephem = _require_ephem(ctx)
    t_ephem = ephem["t_s"]
    j = _map_sim_time_to_ephem_index(t_s, t_ephem, ephem["t0_s"])

    r_sc = y[:3, :].T              # (N,3) Moon->SC
    r_sun = ephem["r_sun_i_m"][j]  # (N,3) Moon->Sun

    r_earth = ephem.get("r_earth_i_m", None)
    if r_earth is not None:
        r_earth = r_earth[j]       # (N,3)

    # Earth radius (strict default unless ctx provides a valid override)
    R_earth_m = 6_378_137.0
    v = getattr(ctx, "R_earth_m", None)
    if v is not None:
        try:
            vv = float(v)
            if np.isfinite(vv) and vv > 0.0:
                R_earth_m = vv
        except Exception:
            pass

    # Segment-sphere intersection test for Moon and (optionally) Earth
    p1 = r_sc
    p2 = r_sun
    d = p2 - p1
    dd = np.einsum("ij,ij->i", d, d)
    dd = np.where(dd > 0.0, dd, 1.0)

    # Moon occultation (center at origin)
    u0 = -np.einsum("ij,ij->i", p1, d) / dd
    u = np.clip(u0, 0.0, 1.0)
    closest = p1 + (u[:, None] * d)
    moon_mask = np.einsum("ij,ij->i", closest, closest) <= float(R_body) ** 2

    earth_mask = np.zeros_like(moon_mask, dtype=bool)
    if r_earth is not None:
        c = r_earth
        u0e = np.einsum("ij,ij->i", (c - p1), d) / dd
        ue = np.clip(u0e, 0.0, 1.0)
        closest_e = p1 + (ue[:, None] * d)
        diff = closest_e - c
        earth_mask = np.einsum("ij,ij->i", diff, diff) <= float(R_earth_m) ** 2

    mask = moon_mask | earth_mask

    if t_s.size >= 2:
        dt = np.diff(t_s)
        total_s = float(np.sum(dt * mask[:-1].astype(np.float64)))
        span = float(t_s[-1] - t_s[0])
        frac = float(total_s / span) if span > 0 else 0.0
    else:
        total_s = 0.0
        frac = 0.0

    return {"mask": mask, "total_s": total_s, "fraction": frac}



# =============================================================================
# 4.                     CORE POSTPROCESS API (CLEAN)
# =============================================================================

def compute_history(
    t_s,
    y,
    mu: float,
    R_body: float,
    ctx: Any = None,
    impact_alt_km: Optional[float] = None,
    max_samples: int = 50000,
    *,
    detect_peri_apo: bool = True,
    detect_impact: bool = True,
    compute_eclipse: bool = True,
    compute_groundtrack: bool = True,
    compute_accel_breakdown: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Compute derived time series and event indices from a propagated state history.

    Inputs
    ------
    t_s : array-like
        Time grid in seconds (N,).
    y : array-like
        State matrix in meters and m/s, shape (6, N). The first 3 rows are r [m],
        the next 3 rows are v [m/s].
    mu : float
        Central body gravitational parameter [m^3/s^2].
    R_body : float
        Central body reference radius [m].
    ctx : Any
        Optional dynamics context. Used only for optional derived products:
        acceleration breakdown, ground track, eclipse mask.
    impact_alt_km : float | None
        If provided and detect_impact=True, impact event is detected when alt_km <= impact_alt_km.
    max_samples : int
        Downsampling cap applied inside compute_history (affects returned history arrays).

    Returns
    -------
    dict
        A history dictionary with canonical keys (t_s, y, elements, altitude, invariants, events, optional products).
    """
    t_s = _as_np(t_s, float)
    y = _as_np(y, float)

    if t_s.ndim != 1 or t_s.size < 2:
        raise ValueError("compute_history: t_s must be a 1D array with length >= 2.")
    if y.ndim != 2 or y.shape[0] < 6:
        raise ValueError("compute_history: y must be a 2D array with shape (>=6, N).")

    # Allow (N,6) input as a last-resort convenience, but enforce canonical internal form (6,N).
    if y.shape[1] != t_s.size and y.shape[0] == t_s.size and y.shape[1] >= 6:
        y = y.T

    if y.shape[1] != t_s.size:
        raise ValueError("compute_history: y length must match t_s length (expected y.shape[1] == len(t_s)).")

    mu = float(mu)
    R_body = float(R_body)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("compute_history: mu must be a positive finite float.")
    if not np.isfinite(R_body) or R_body <= 0.0:
        raise ValueError("compute_history: R_body must be a positive finite float.")

    # ------------------------------------------------------------------
    # 1) Downsample (for plotting/reporting performance)
    # ------------------------------------------------------------------
    N = int(t_s.size)
    if (max_samples is not None) and (int(max_samples) > 0) and (N > int(max_samples)):
        idx = np.linspace(0, N - 1, int(max_samples), dtype=np.int64)
        t = t_s[idx]
        ys = y[:, idx]
    else:
        idx = None
        t = t_s
        ys = y

    r = ys[:3, :]
    v = ys[3:6, :]

    # ------------------------------------------------------------------
    # 2) Classical orbital elements (batch)
    # Returns: a, e, inc, raan, argp, nu, eps, rnorm, vnorm, hnorm
    # Units: a[m], angles[rad], eps[J/kg], rnorm[m], vnorm[m/s], hnorm[m^2/s]
    # ------------------------------------------------------------------
    a, e, inc, raan, argp, nu, eps, rnorm, vnorm, hnorm = batch_y_to_elements(
        ys, mu, mode="coe10"
    )

    # ------------------------------------------------------------------
    # 3) Altitude + relative drift series
    # ------------------------------------------------------------------
    alt_km = (rnorm - R_body) / 1000.0

    E0 = float(eps[0]) if eps.size else 0.0
    h0 = float(hnorm[0]) if hnorm.size else 0.0
    relE = (eps - E0) / max(abs(E0), 1e-30) if eps.size else _as_np([], float)
    relh = (hnorm - h0) / max(abs(h0), 1e-30) if hnorm.size else _as_np([], float)

    # ------------------------------------------------------------------
    # 4) Event detection
    # ------------------------------------------------------------------
    if detect_peri_apo:
        peri_idx, apo_idx = _detect_peri_apo_indices(r, v)
    else:
        peri_idx = np.array([], dtype=np.int64)
        apo_idx = np.array([], dtype=np.int64)

    if detect_impact and (impact_alt_km is not None):
        impact_idx = _detect_impact_index(alt_km, float(impact_alt_km))
    else:
        impact_idx = None

    # ------------------------------------------------------------------
    # 5) History (canonical keys)
    # ------------------------------------------------------------------
    hist: Dict[str, Any] = {
        "t_s": t,
        "y": ys,
        "r_m": r,
        "v_mps": v,
        "a_km": _as_np(a, float) / 1000.0,
        "e": _as_np(e, float),
        "i_deg": np.degrees(_as_np(inc, float)),
        "raan_deg": np.degrees(_as_np(raan, float)),
        "argp_deg": np.degrees(_as_np(argp, float)),
        "nu_deg": np.degrees(_as_np(nu, float)),
        "alt_km": _as_np(alt_km, float),
        "energy_Jkg": _as_np(eps, float),
        "h_norm_m2s": _as_np(hnorm, float),
        "rel_energy_drift": _as_np(relE, float),
        "rel_h_drift": _as_np(relh, float),
        "events": {
            "peri_idx": _as_np(peri_idx, float).astype(np.int64, copy=False),
            "apo_idx": _as_np(apo_idx, float).astype(np.int64, copy=False),
            "impact_idx": int(impact_idx) if impact_idx is not None else None,
            "impact_alt_km": float(impact_alt_km) if impact_alt_km is not None else None,
            "detect_peri_apo": bool(detect_peri_apo),
            "detect_impact": bool(detect_impact),
        },
    }

    # ------------------------------------------------------------------
    # 6) Optional derived products
    # NOTE: these are allowed to fail-soft (return None/{}), but compute_history
    # itself remains strict about (t, y, mu, R_body) validity.
    # ------------------------------------------------------------------
    if compute_accel_breakdown:
        try:
            acc_mag = _accel_breakdown_if_available(ctx, t, ys, max_samples=min(20000, int(t.size)))
            if acc_mag:
                hist["accel_mag"] = acc_mag
                hist["t_s_accel"] = t
        except Exception:
            if strict:
                raise
            # keep core history valid even if optional product fails
            pass

    if compute_groundtrack:
        try:
            gt = _groundtrack_if_available(ctx, t, ys, max_samples=int(max_samples))
            if gt is not None:
                hist["groundtrack"] = gt
        except Exception:
            if strict:
                raise
            pass

    if compute_eclipse:
        try:
            ecl = _eclipse_if_available(ctx, t, ys, R_body=R_body)
            if ecl is not None:
                hist["eclipse"] = ecl["mask"]
                hist["eclipse_total_s"] = float(ecl.get("total_s", 0.0))
                hist["eclipse_fraction"] = float(ecl.get("fraction", 0.0))
        except Exception:
            if strict:
                raise
            pass

    # Preserve downsampling index if downstream wants to relate to original arrays
    if idx is not None:
        hist["downsample_idx"] = idx

    return hist



# =============================================================================
# 5.                   REPORTING & SUMMARY HELPERS (CLEAN)
# =============================================================================

def summarize_history(hist: Dict[str, Any]) -> Dict[str, float]:
    """Return basic summary statistics for altitude [km]."""
    alt = _as_np(hist.get("alt_km", []), float)
    if alt.size == 0:
        return {}
    return {
        "alt_min_km": float(np.nanmin(alt)),
        "alt_max_km": float(np.nanmax(alt)),
        "alt_mean_km": float(np.nanmean(alt)),
    }



# =============================================================================
# 6.                       HIGH-LEVEL ENTRYPOINT (STRICT)
# =============================================================================

def process_simulation_results(result: Any, ctx: Any = None, cfg: Any = None, *, strict: bool = False) -> Dict[str, Any]:
    """
    Preferred postprocess entry point.

    Contract
    --------
    - We must obtain (t_s, y) where:
        t_s is (N,) seconds,
        y is (6,N) state in meters and m/s.

    - cfg is treated as strict schema:
        cfg.time.max_points_cap
        cfg.propagator.events.detect_impact
        cfg.propagator.events.enable_peri_apo_events
        cfg.propagator.events.detect_eclipse
        cfg.propagator.events.impact_alt_km

      No legacy alternative paths are consulted here.

    - Optional 2-body baseline:
        result.sol_2body with .t and .y on the same grid (or at least same length).
    """
    # ---------------------------------------------------------------------
    # 0) Extract solution (t, y)
    # ---------------------------------------------------------------------
    sol = getattr(result, "sol", None)

    if sol is not None and hasattr(sol, "t") and hasattr(sol, "y"):
        t = np.asarray(sol.t, dtype=float)
        y = np.asarray(sol.y, dtype=float)
    else:
        # Fallback: raw result.t / result.y only (strict; no broad alias normalization)
        t = np.asarray(getattr(result, "t", None), dtype=float) if getattr(result, "t", None) is not None else None
        y = np.asarray(getattr(result, "y", None), dtype=float) if getattr(result, "y", None) is not None else None
        if t is None or y is None:
            raise ValueError("process_simulation_results: expected result.sol or (result.t, result.y).")

    if t.ndim != 1 or t.size < 2:
        raise ValueError("process_simulation_results: expected 1D time vector with length >= 2.")
    if y.ndim != 2:
        raise ValueError("process_simulation_results: expected 2D state array.")

    # Canonicalize y to (6,N)
    if y.shape[1] != t.size and y.shape[0] == t.size and y.shape[1] >= 6:
        y = y.T
    if y.shape[1] != t.size:
        raise ValueError("process_simulation_results: y does not match t length.")
    if y.shape[0] < 6:
        raise ValueError("process_simulation_results: y must have at least 6 rows (r,v).")

    # ---------------------------------------------------------------------
    # 1) Read strict cfg knobs (no legacy keys)
    # ---------------------------------------------------------------------
    max_samples = 50000
    if cfg is not None:
        try:
            cap = getattr(getattr(cfg, "time"), "max_points_cap", None)
            if cap is not None:
                cap_i = int(cap)
                if cap_i > 0:
                    max_samples = min(max_samples, cap_i)
        except Exception:
            pass

    detect_impact = True
    detect_peri_apo = True
    detect_eclipse = False
    impact_alt_km = 0.0

    if cfg is not None:
        try:
            ev = getattr(getattr(cfg, "propagator"), "events")
            detect_impact = bool(getattr(ev, "detect_impact", True))
            detect_peri_apo = bool(getattr(ev, "enable_peri_apo_events", True))
            detect_eclipse = bool(getattr(ev, "detect_eclipse", False))
            impact_alt_km = float(getattr(ev, "impact_alt_km", 0.0))
        except Exception:
            pass

    compute_groundtrack = True
    compute_accel_breakdown = True

    # If eclipse is requested or radiation effects are enabled, compute eclipse products.
    compute_eclipse = bool(detect_eclipse)
    if (not compute_eclipse) and (cfg is not None):
        try:
            flags = getattr(cfg, "flags", None)
            if flags is not None:
                srp_on = bool(getattr(flags, "enable_srp", False))
                alb_on = bool(getattr(flags, "enable_albedo", False))
                th_on = bool(getattr(flags, "enable_thermal", False) or getattr(flags, "enable_thermal_ir", False))
                compute_eclipse = bool(srp_on or alb_on or th_on)
        except Exception:
            pass

    impact_alt_km_for_post = (impact_alt_km if detect_impact else None)

    # ---------------------------------------------------------------------
    # 2) mu and body radius (prefer ctx)
    # ---------------------------------------------------------------------
    mu_fallback = 4.902801e12
    R_fallback = 1_737_400.0

    mu = mu_fallback
    R = R_fallback
    if ctx is not None:
        try:
            mu = float(getattr(ctx, "mu_m3s2", None) or getattr(ctx, "mu", None) or mu_fallback)
        except Exception:
            mu = mu_fallback
        try:
            R = float(getattr(ctx, "R_body_m", None) or getattr(ctx, "R_m", None) or getattr(ctx, "radius_m", None) or R_fallback)
        except Exception:
            R = R_fallback

    # ---------------------------------------------------------------------
    # 3) Build history
    # ---------------------------------------------------------------------
    hist = compute_history(
        t, y,
        mu=float(mu), R_body=float(R),
        ctx=ctx,
        impact_alt_km=impact_alt_km_for_post,
        max_samples=int(max_samples),
        detect_peri_apo=bool(detect_peri_apo),
        detect_impact=bool(detect_impact),
        compute_eclipse=bool(compute_eclipse),
        compute_groundtrack=bool(compute_groundtrack),
        compute_accel_breakdown=bool(compute_accel_breakdown),
        strict=bool(strict),
    )

    # ---------------------------------------------------------------------
    # 4) Baseline diffs vs 2-body (strict)
    # ---------------------------------------------------------------------
    sol0 = getattr(result, "sol_2body", None)
    if sol0 is not None and hasattr(sol0, "t") and hasattr(sol0, "y"):
        try:
            t0 = np.asarray(sol0.t, dtype=float)
            y0 = np.asarray(sol0.y, dtype=float)
            if y0.ndim == 2 and y0.shape[1] != t0.size and y0.shape[0] == t0.size and y0.shape[1] >= 6:
                y0 = y0.T

            # Compare only if the grids match length (we assume same sampling)
            if t0.ndim == 1 and t0.size == t.size and y0.ndim == 2 and y0.shape[1] == t0.size and y0.shape[0] >= 6:
                # Apply the same downsampling index used by compute_history if available
                idx = hist.get("downsample_idx", None)
                if idx is None:
                    idx = np.arange(int(t.size), dtype=np.int64)

                r = y[:3, idx]
                v = y[3:6, idx]
                r0 = y0[:3, idx]
                v0 = y0[3:6, idx]

                hist["dr_2body_km"] = (np.linalg.norm(r - r0, axis=0) / 1000.0).astype(float)
                hist["dv_2body_mps"] = np.linalg.norm(v - v0, axis=0).astype(float)

                alt = np.linalg.norm(r, axis=0) - float(R)
                alt0 = np.linalg.norm(r0, axis=0) - float(R)
                hist["dalt_2body_km"] = ((alt - alt0) / 1000.0).astype(float)
        except Exception:
            # keep history usable even if baseline comparison fails
            pass

    return hist
