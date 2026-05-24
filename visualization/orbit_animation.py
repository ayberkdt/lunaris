# ST_LRPS/visualization/orbit_animation.py
# -*- coding: utf-8 -*-
"""
ST_LRPS — Scientific-grade 3D orbit animation (Moon-centered)

Produces a publication / agency-documentation quality 3D animation of a
Moon-centered trajectory with physically meaningful lighting, a dense
starfield, and a live telemetry HUD.

Primary API (called from main.py):
    render_orbit_animation(result, config, output_file)

History API (for notebooks/tests):
    render_orbit_animation_from_history(history, output_file)

Key features
- Space-black background with high-density starfield (log-normal star sizes).
- Lambertian Moon shading driven by Sun direction; day/night terminator ring.
- Moon equatorial ring + prime meridian reference lines.
- Smooth animation via uniform-time resampling with CubicHermiteSpline fallback.
- Animated satellite trail (inferno colormap, time-gradient).
- Ground-track projection onto the Moon surface.
- Periapsis and apoapsis markers with altitude annotations.
- Live HUD: altitude, inclination, eccentricity, speed, semi-major axis,
  orbital period, mission time, Sun direction arrow.
- MP4 via ffmpeg (libx264, crf=18, preset='slow'); GIF fallback if needed.
"""

from __future__ import annotations

import os
import math
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

try:
    from scipy.interpolate import CubicSpline, CubicHermiteSpline  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    CubicSpline = None  # type: ignore
    _HAVE_SCIPY = False

try:
    from .styling import COLORS as _COLORS  # type: ignore
except Exception:
    _COLORS = {}

# ==========================
# Theme
# ==========================

@dataclass(frozen=True)
class Theme:
    bg: str = "#000000"
    text: str = "#E8EEF8"
    hud_bg: str = "#0A0F1E"
    hud_edge: str = "#2B3555"
    orbit: str = "#5ED0FF"
    orbit_faint: str = "#2B6C80"
    moon_base: Tuple[float, float, float] = (0.75, 0.76, 0.80)
    sun: str = "#FFD166"
    star: str = "#D6E2FF"
    trail_cmap: str = "inferno"
    peri: str = "#FF6B6B"      # periapsis marker
    apo: str = "#98FB98"       # apoapsis marker
    ground_track: str = "#40E0D0"  # ground track projection


def _hex_to_rgb01(h: str):
    h = str(h).strip().lstrip("#")
    if len(h) == 3:
        h = "".join([c * 2 for c in h])
    if len(h) != 6:
        return None
    try:
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    except Exception:
        return None


def _rel_luminance(rgb):
    def _lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _pick_dark_safe_color(candidate, fallback: str, *, want_light_text: bool = False) -> str:
    if not candidate:
        return fallback
    rgb = _hex_to_rgb01(candidate)
    if rgb is None:
        return fallback
    lum = _rel_luminance(rgb)
    if want_light_text and lum < 0.35:
        return fallback
    return candidate


THEME = Theme(
    text=_pick_dark_safe_color(_COLORS.get("text", None), "#E8EEF8", want_light_text=True),
    hud_bg=_pick_dark_safe_color(_COLORS.get("hud_bg", None), "#0A0F1E"),
    hud_edge=_pick_dark_safe_color(_COLORS.get("hud_edge", None), "#2B3555"),
    orbit=_pick_dark_safe_color(_COLORS.get("orbit", None), "#5ED0FF"),
    orbit_faint=_pick_dark_safe_color(_COLORS.get("orbit_faint", None), "#2B6C80"),
    sun=_pick_dark_safe_color(_COLORS.get("sun", None), "#FFD166"),
    star=_pick_dark_safe_color(_COLORS.get("star", None), "#D6E2FF"),
)

R_MOON_KM_DEFAULT = 1737.4
MU_MOON_DEFAULT = 4.902800066e12  # m^3/s^2


# ==========================
# Math helpers
# ==========================

def _as_np(x, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _norm(v: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.linalg.norm(v, axis=axis)


def _unit(v: np.ndarray, axis: int = -1, eps: float = 1e-30) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _meta_get(meta: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    if not isinstance(meta, dict):
        return default
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    return default


def _pick_first(hist: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in hist and hist[k] is not None:
            return hist[k]
    return default


def _ensure_monotonic_time(
    t_s: np.ndarray,
    arrs: Tuple[np.ndarray, ...],
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    t_s = _as_np(t_s, float).ravel()
    ok = np.isfinite(t_s)
    if ok.sum() < 2:
        return t_s[ok], tuple(_as_np(a, float)[..., ok] for a in arrs)

    for a in arrs:
        aa = _as_np(a, float)
        if aa.ndim == 2:
            ok &= np.all(np.isfinite(aa), axis=0)
        else:
            ok &= np.isfinite(aa)

    t = t_s[ok]
    idx = np.argsort(t)
    t = t[idx]

    out = []
    for a in arrs:
        aa = _as_np(a, float)
        if aa.ndim == 2:
            out.append(aa[:, ok][:, idx])
        else:
            out.append(aa[ok][idx])

    _, uniq = np.unique(t, return_index=True)
    t = t[uniq]
    out = [a[:, uniq] if a.ndim == 2 else a[uniq] for a in out]
    return t, tuple(out)


# ==========================
# Resampling
# ==========================

def _resample_vec3(t: np.ndarray, vec3: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    vec3 = _as_np(vec3, float)
    out = np.empty((3, t_new.size), dtype=float)
    if _HAVE_SCIPY and CubicSpline is not None and t.size >= 3:
        for i in range(3):
            cs = CubicSpline(t, vec3[i, :], bc_type="natural")
            out[i, :] = cs(t_new)
    else:
        for i in range(3):
            out[i, :] = np.interp(t_new, t, vec3[i, :])
    return out


def _resample_pos_with_vel(
    t: np.ndarray,
    r_m: np.ndarray,
    v_mps: Optional[np.ndarray],
    t_new: np.ndarray,
) -> np.ndarray:
    r_m = _as_np(r_m, float)
    if v_mps is not None:
        v_mps = _as_np(v_mps, float)
    if _HAVE_SCIPY and v_mps is not None and CubicHermiteSpline is not None and t.size >= 2:
        out = np.empty((3, t_new.size), dtype=float)
        for i in range(3):
            hs = CubicHermiteSpline(t, r_m[i, :], v_mps[i, :])
            out[i, :] = hs(t_new)
        return out
    return _resample_vec3(t, r_m, t_new)


def _resample_1d(t: np.ndarray, x: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    x = _as_np(x, float).ravel()
    if _HAVE_SCIPY and CubicSpline is not None and t.size >= 3:
        cs = CubicSpline(t, x, bc_type="natural")
        return cs(t_new)
    return np.interp(t_new, t, x)


# ==========================
# Orbital elements
# ==========================

def _compute_orbital_elements_series(
    r_m: np.ndarray,
    v_mps: Optional[np.ndarray],
    R_body_m: float,
    mu_m3s2: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Compute per-sample orbital elements. Returns dict with keys:
    alt_km, inc_deg, e, speed_kmps, sma_km, period_h
    """
    rr = _norm(r_m, axis=0)                           # (N,)
    alt_km = (rr - float(R_body_m)) / 1000.0

    N = rr.shape[0]
    inc_deg = np.full(N, np.nan)
    ecc = np.full(N, np.nan)
    speed_kmps = np.full(N, np.nan)
    sma_km = np.full(N, np.nan)
    period_h = np.full(N, np.nan)

    if v_mps is None or v_mps.size == 0:
        return dict(alt_km=alt_km, inc_deg=inc_deg, e=ecc,
                    speed_kmps=speed_kmps, sma_km=sma_km, period_h=period_h)

    v_mps = _as_np(v_mps, float)
    if v_mps.shape[0] < 3:
        return dict(alt_km=alt_km, inc_deg=inc_deg, e=ecc,
                    speed_kmps=speed_kmps, sma_km=sma_km, period_h=period_h)

    # Specific angular momentum
    h = np.cross(r_m.T, v_mps.T).T        # (3, N)
    hn = _norm(h, axis=0)
    inc = np.arccos(np.clip(h[2, :] / np.maximum(hn, 1e-30), -1.0, 1.0))
    inc_deg = np.degrees(inc)

    vn = _norm(v_mps, axis=0)
    speed_kmps = vn / 1000.0

    if mu_m3s2 is not None and float(mu_m3s2) > 0.0:
        mu = float(mu_m3s2)

        # Eccentricity
        vxh = np.cross(v_mps.T, h.T).T    # (3, N)
        evec = vxh / mu - r_m / np.maximum(rr, 1e-30)
        ecc = _norm(evec, axis=0)

        # Semi-major axis from vis-viva: a = 1 / (2/r - v²/mu)
        vv = vn ** 2
        denom = 2.0 / np.maximum(rr, 1.0) - vv / mu
        valid = np.abs(denom) > 1e-30
        a = np.where(valid, 1.0 / denom, np.nan)
        sma_km = a / 1000.0

        # Orbital period T = 2pi * sqrt(a³/mu)   [hours]
        pos_a = a > 0.0
        period_s = np.where(pos_a, 2.0 * np.pi * np.sqrt(np.where(pos_a, a**3, 1.0) / mu), np.nan)
        period_h = period_s / 3600.0

    return dict(alt_km=alt_km, inc_deg=inc_deg, e=ecc,
                speed_kmps=speed_kmps, sma_km=sma_km, period_h=period_h)


def _find_extrema(alt_km: np.ndarray) -> Tuple[int, int]:
    """Return (idx_periapsis, idx_apoapsis) — indices of min/max altitude."""
    valid = np.isfinite(alt_km)
    if not valid.any():
        return 0, 0
    masked = np.where(valid, alt_km, np.inf)
    idx_peri = int(np.argmin(masked))
    masked2 = np.where(valid, alt_km, -np.inf)
    idx_apo = int(np.argmax(masked2))
    return idx_peri, idx_apo


# ==========================
# API adapter: PropagationResult → hist dict
# ==========================

def _result_to_hist(result: Any, config: Any) -> Dict[str, Any]:
    """Convert a PropagationResult + SimConfig to the internal hist dict format."""
    t_s = np.asarray(getattr(result, "t", []), dtype=float).ravel()
    y = getattr(result, "y", None)
    if y is None:
        return {"t_s": t_s}

    y = np.asarray(y, dtype=float)
    # y is (N, n_state) row-major; convert to (n_state, N) column-major
    if y.ndim == 2 and y.shape[1] >= 6 and y.shape[0] != 6:
        y = y.T  # now (n_state, N)

    r_m = y[0:3, :]   # (3, N) meters
    v_mps = y[3:6, :] if y.shape[0] >= 6 else None

    # Body radius and mu from config or defaults
    R_body_m = float(R_MOON_KM_DEFAULT * 1000.0)
    mu_m3s2 = float(MU_MOON_DEFAULT)

    if config is not None:
        try:
            R_body_m = float(config.body.radius_m)
        except Exception:
            pass
        try:
            mu_m3s2 = float(config.gravity.mu)
        except Exception:
            try:
                mu_m3s2 = float(config.mu_m3s2)
            except Exception:
                pass

    hist: Dict[str, Any] = {
        "t_s": t_s,
        "r_m": r_m,
        "v_mps": v_mps,
        "meta": {
            "body_radius_m": R_body_m,
            "mu_m3s2": mu_m3s2,
        },
    }
    return hist


# ==========================
# Visual elements
# ==========================

def _create_moon_mesh(R_km: float, nu: int = 120, nv: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.0, np.pi, nv)
    x = R_km * np.outer(np.cos(u), np.sin(v))
    y = R_km * np.outer(np.sin(u), np.sin(v))
    z = R_km * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _moon_facecolors(
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    sun_dir: np.ndarray,
    *,
    ambient: float = 0.06,
    diffuse: float = 0.94,
    gamma: float = 1.05,
    base_rgb: Tuple[float, float, float] = (0.75, 0.76, 0.80),
) -> np.ndarray:
    sun = _unit(np.asarray(sun_dir, float).reshape(3,), axis=0)
    xc = 0.25 * (mx[:-1, :-1] + mx[1:, :-1] + mx[:-1, 1:] + mx[1:, 1:])
    yc = 0.25 * (my[:-1, :-1] + my[1:, :-1] + my[:-1, 1:] + my[1:, 1:])
    zc = 0.25 * (mz[:-1, :-1] + mz[1:, :-1] + mz[:-1, 1:] + mz[1:, 1:])
    n = _unit(np.stack([xc, yc, zc], axis=0), axis=0)
    lambert = np.clip(n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2], 0.0, 1.0)
    intensity = np.clip(ambient + diffuse * lambert, 0.0, 1.0) ** gamma
    base = np.array(base_rgb, dtype=float)[None, None, :]
    rgb = base * intensity[..., None]
    alpha = np.ones_like(intensity)[..., None]
    return np.concatenate([rgb, alpha], axis=2)


def _setup_3d_axis(fig: plt.Figure) -> Any:
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(THEME.bg)
    ax.set_facecolor(THEME.bg)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((0, 0, 0, 0))
        except Exception:
            pass
    try:
        ax.set_axis_off()
    except Exception:
        pass
    return ax


def _add_stars(ax: Any, max_range_km: float, n_stars: int = 1500, seed: int = 7) -> None:
    """High-density starfield with realistic log-normal brightness distribution."""
    rng = np.random.default_rng(seed)

    u = rng.random(n_stars)
    v = rng.random(n_stars)
    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)
    r = max_range_km * 1.35

    xs = r * np.sin(phi) * np.cos(theta)
    ys = r * np.sin(phi) * np.sin(theta)
    zs = r * np.cos(phi)

    # Log-normal sizes mimic a stellar luminosity function:
    # many faint stars, few bright ones
    log_sizes = rng.normal(loc=0.0, scale=0.6, size=n_stars)
    sizes = np.clip(np.exp(log_sizes) * 0.6, 0.05, 4.5)

    # Brightness variety: brighter stars are slightly whiter/larger
    alphas = np.clip(0.2 + 0.6 * (sizes / sizes.max()), 0.15, 0.90)

    ax.scatter(
        xs, ys, zs,
        s=sizes, c=THEME.star, alpha=alphas,
        linewidths=0, depthshade=False, zorder=0,
    )


def _add_moon_reference_lines(ax: Any, R_body_km: float) -> None:
    """Draw a subtle equatorial ring and prime meridian arc on the Moon surface."""
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    r = R_body_km * 1.002  # just outside surface to avoid z-fighting

    # Equatorial ring (z = 0 plane)
    eq_x = r * np.cos(t)
    eq_y = r * np.sin(t)
    eq_z = np.zeros_like(t)
    ax.plot(eq_x, eq_y, eq_z, color="#556688", lw=0.7, alpha=0.35, zorder=2, linestyle="--")

    # Prime meridian arc (y = 0 plane, x ≥ 0)
    pm_x = r * np.cos(t)
    pm_y = np.zeros_like(t)
    pm_z = r * np.sin(t)
    ax.plot(pm_x, pm_y, pm_z, color="#556688", lw=0.5, alpha=0.25, zorder=2, linestyle=":")


def _add_terminator_ring(ax: Any, R_body_km: float, sun_dir: np.ndarray) -> Any:
    """Draw the day/night terminator circle on the Moon surface.

    Returns the plot object so it can be updated in the animation loop.
    """
    s = _unit(np.asarray(sun_dir, float).reshape(3,), axis=0)
    r = R_body_km * 1.003  # slightly above surface

    # The terminator is the great circle perpendicular to sun_dir.
    # Build two orthogonal vectors in the terminator plane.
    ax_ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(s, ax_ref))) > 0.9:
        ax_ref = np.array([1.0, 0.0, 0.0])
    u1 = _unit(np.cross(s, ax_ref).reshape(3,), axis=0)
    u2 = _unit(np.cross(s, u1).reshape(3,), axis=0)

    t = np.linspace(0.0, 2.0 * np.pi, 200)
    pts = r * (np.outer(np.cos(t), u1) + np.outer(np.sin(t), u2))  # (200, 3)
    line, = ax.plot(
        pts[:, 0], pts[:, 1], pts[:, 2],
        color="#AACCFF", lw=1.0, alpha=0.40, zorder=3, linestyle="-",
    )
    return line


def _update_terminator_ring(line: Any, R_body_km: float, sun_dir: np.ndarray) -> None:
    s = _unit(np.asarray(sun_dir, float).reshape(3,), axis=0)
    r = R_body_km * 1.003

    ax_ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(s, ax_ref))) > 0.9:
        ax_ref = np.array([1.0, 0.0, 0.0])
    u1 = _unit(np.cross(s, ax_ref).reshape(3,), axis=0)
    u2 = _unit(np.cross(s, u1).reshape(3,), axis=0)

    t = np.linspace(0.0, 2.0 * np.pi, 200)
    pts = r * (np.outer(np.cos(t), u1) + np.outer(np.sin(t), u2))
    line.set_data(pts[:, 0], pts[:, 1])
    line.set_3d_properties(pts[:, 2])


# ==========================
# Sun direction
# ==========================

def _try_spice_sun_dir(meta: Dict[str, Any], t_s: np.ndarray) -> Optional[np.ndarray]:
    spice_meta = meta.get("spice") if isinstance(meta, dict) else None
    if not isinstance(spice_meta, dict):
        return None
    start_utc = spice_meta.get("start_utc")
    if not isinstance(start_utc, str) or not start_utc:
        return None
    try:
        import spiceypy as spice  # type: ignore
    except Exception:
        return None
    try:
        et0 = float(spice.utc2et(start_utc))
    except Exception:
        return None

    sun_dir = np.empty((3, t_s.size), dtype=float)
    for i in range(t_s.size):
        et = et0 + float(t_s[i])
        try:
            st, _ = spice.spkezr("SUN", et, "J2000", "NONE", "MOON")
            r_sun_km = np.asarray(st[:3], dtype=float)
        except Exception:
            return None
        sun_dir[:, i] = _unit(r_sun_km, axis=0).reshape(3,)
    return sun_dir


def _get_sun_dir_series(hist: Dict[str, Any], t_ref: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    meta = hist.get("meta", {}) if isinstance(hist.get("meta", {}), dict) else {}

    sun_spice = _try_spice_sun_dir(meta, t_new)
    if sun_spice is not None:
        return sun_spice

    for k in ("sun_dir_eci", "sun_vec_eci", "sun_dir", "sun_vec"):
        tab = hist.get(k)
        if tab is None:
            continue
        tab = _as_np(tab, float)
        if tab.ndim == 2 and tab.shape[0] == 3 and tab.shape[1] >= 2 and t_ref.size == tab.shape[1]:
            t0, (tab2,) = _ensure_monotonic_time(t_ref, (tab,))
            sun = _resample_vec3(t0, tab2, t_new)
            return _unit(sun, axis=0)

    return np.tile(np.array([[1.0], [0.0], [0.0]]), (1, t_new.size))


# ==========================
# HUD helpers
# ==========================

def _format_mission_time(t_s: float) -> Tuple[float, float]:
    days = t_s / 86400.0
    hours = t_s / 3600.0
    return float(days), float(hours)


def _sun_arrow_text(s: np.ndarray) -> str:
    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    ang = math.atan2(sy, sx)
    dirs = ["->", "/^", "^", "^\\", "<-", "\\/", "v", "v/"]
    k = int(np.floor((ang + math.pi) / (2 * math.pi) * 8.0)) % 8
    return f"{dirs[k]}  [{sx:+.3f}, {sy:+.3f}, {sz:+.3f}]"


# ==========================
# Public API
# ==========================

def render_orbit_animation(
    result,
    config,
    output_file: str = "orbit_3d.mp4",
    **kwargs
) -> Optional[str]:
    """
    Create a high-quality Moon-centered orbit animation from propagation results.
    """
    hist = _result_to_hist(result, config)
    return render_orbit_animation_from_history(hist, output_file, **kwargs)

def render_orbit_animation_from_history(
    history: dict,
    output_file: str,
    # --- keyword-only visual / encoding params ---
    frames: int = 900,
    fps: int = 30,
    trail_length: int = 260,
    rotate_camera: bool = True,
    elev_deg: float = 22.0,
    azim0_deg: float = 35.0,
    cam_rate_deg_per_day: float = 30.0,
    moon_mesh_u: int = 120,
    moon_mesh_v: int = 60,
    show_stars: bool = True,
    n_stars: int = 1500,
    show_hud: bool = True,
    light_update_every: int = 8,
    dpi: int = 170,
    crf: int = 18,
    preset: str = "slow",
) -> Optional[str]:
    """
    Create a high-quality Moon-centered orbit animation from a history dict.
    """
    hist = history
    save_path = str(output_file)
    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------
    # Validate + load inputs
    # -------------------------------------------------------
    t_s = _as_np(hist.get("t_s", []), float).ravel()
    r_m = _pick_first(hist, "r_m", default=None)
    if r_m is None or t_s.size < 2:
        print("[3D] No trajectory data (need 't_s' and 'r_m').")
        return None

    r_m = _as_np(r_m, float)
    if r_m.ndim != 2 or r_m.shape[0] < 3:
        print("[3D] Invalid r_m shape; expected (3,N).")
        return None

    v_mps = _pick_first(hist, "v_mps", default=None)
    if v_mps is None and "y" in hist and hist["y"] is not None:
        y = _as_np(hist["y"], float)
        if y.ndim == 2 and y.shape[0] >= 6:
            v_mps = y[3:6, :]
    if v_mps is not None:
        v_mps = _as_np(v_mps, float)

    meta = hist.get("meta", {}) if isinstance(hist.get("meta", {}), dict) else {}
    R_body_m = _meta_get(meta, "body_radius_m", "R_body_m", default=None)
    if R_body_m is None:
        R_body_m = float(R_MOON_KM_DEFAULT) * 1000.0
    mu_m3s2 = _meta_get(meta, "mu_m3s2", "mu", default=None)
    try:
        mu_m3s2 = float(mu_m3s2) if mu_m3s2 is not None else MU_MOON_DEFAULT
    except Exception:
        mu_m3s2 = MU_MOON_DEFAULT

    if v_mps is not None and v_mps.shape[1] == r_m.shape[1]:
        t_s, (r_m, v_mps) = _ensure_monotonic_time(t_s, (r_m, v_mps))
    else:
        t_s, (r_m,) = _ensure_monotonic_time(t_s, (r_m,))
        v_mps = None

    if t_s.size < 2:
        print("[3D] Not enough valid samples after sanitization.")
        return None

    # -------------------------------------------------------
    # Uniform time base and resampling
    # -------------------------------------------------------
    duration_days = (float(t_s[-1]) - float(t_s[0])) / 86400.0
    if frames is None or int(frames) <= 0:
        frames = int(np.clip(duration_days * 30.0, 240, 2400))
    else:
        frames = int(np.clip(int(frames), 120, 2400))

    t_anim = np.linspace(float(t_s[0]), float(t_s[-1]), frames)

    r_anim_m = _resample_pos_with_vel(t_s, r_m, v_mps, t_anim)
    v_anim_mps = _resample_vec3(t_s, v_mps, t_anim) if v_mps is not None else None

    # Compute extended orbital elements
    oe = _compute_orbital_elements_series(r_anim_m, v_anim_mps, float(R_body_m), mu_m3s2)

    # Override with hist series if available and aligned
    for key, hist_key in [("alt_km", "alt_km"), ("inc_deg", "i_deg"), ("e", "e")]:
        val = hist.get(hist_key)
        if val is not None and _as_np(val).size == t_s.size:
            oe[key] = _resample_1d(t_s, _as_np(val, float), t_anim)

    alt_anim_km = oe["alt_km"]
    inc_anim = oe["inc_deg"]
    e_anim = oe["e"]
    speed_anim = oe["speed_kmps"]
    sma_anim = oe["sma_km"]
    period_anim = oe["period_h"]

    # Sun direction series
    sun_dir = _get_sun_dir_series(hist, t_s, t_anim)

    # Convert to km
    r_anim_km = r_anim_m / 1000.0
    R_body_km = float(R_body_m) / 1000.0

    max_range = float(np.nanmax(np.abs(r_anim_km)))
    if not np.isfinite(max_range) or max_range <= 0.0:
        max_range = max(R_body_km * 2.2, 2500.0)
    max_range *= 1.10

    # Periapsis and apoapsis
    idx_peri, idx_apo = _find_extrema(alt_anim_km)
    peri_pos = r_anim_km[:, idx_peri]
    apo_pos = r_anim_km[:, idx_apo]
    peri_alt = float(alt_anim_km[idx_peri]) if np.isfinite(alt_anim_km[idx_peri]) else 0.0
    apo_alt = float(alt_anim_km[idx_apo]) if np.isfinite(alt_anim_km[idx_apo]) else 0.0

    # -------------------------------------------------------
    # Figure / axis
    # -------------------------------------------------------
    fig = plt.figure(figsize=(12, 8), dpi=int(dpi))
    ax = _setup_3d_axis(fig)

    if show_stars:
        _add_stars(ax, max_range_km=max_range, n_stars=int(n_stars))

    # Moon
    mx, my, mz = _create_moon_mesh(R_body_km, nu=int(moon_mesh_u), nv=int(moon_mesh_v))
    moon_fc = _moon_facecolors(mx, my, mz, sun_dir[:, 0], ambient=0.06, diffuse=0.94,
                                base_rgb=THEME.moon_base)
    moon_surf = ax.plot_surface(
        mx, my, mz,
        facecolors=moon_fc,
        shade=False,
        linewidth=0.0,
        antialiased=True,
        rcount=int(moon_mesh_u),
        ccount=int(moon_mesh_v),
        zorder=1,
    )
    try:
        moon_surf.set_alpha(1.0)
    except Exception:
        pass

    # Moon reference lines (equatorial ring + prime meridian)
    _add_moon_reference_lines(ax, R_body_km)

    # Terminator ring (animated)
    term_line = _add_terminator_ring(ax, R_body_km, sun_dir[:, 0])

    # Static full-orbit ghost
    ax.plot(r_anim_km[0, :], r_anim_km[1, :], r_anim_km[2, :],
            color=THEME.orbit, lw=3.5, alpha=0.05, zorder=2)
    ax.plot(r_anim_km[0, :], r_anim_km[1, :], r_anim_km[2, :],
            color=THEME.orbit_faint, lw=1.2, alpha=0.20, zorder=3)

    # Periapsis marker
    peri_dot, = ax.plot(
        [peri_pos[0]], [peri_pos[1]], [peri_pos[2]],
        marker="v", markersize=7, color=THEME.peri,
        markeredgecolor="#220000", markeredgewidth=0.7,
        linestyle="none", zorder=7, alpha=0.90,
    )
    # Apoapsis marker
    apo_dot, = ax.plot(
        [apo_pos[0]], [apo_pos[1]], [apo_pos[2]],
        marker="^", markersize=7, color=THEME.apo,
        markeredgecolor="#002200", markeredgewidth=0.7,
        linestyle="none", zorder=7, alpha=0.90,
    )

    # Ground track: project onto Moon surface radius
    gt_factor = R_body_km / np.maximum(np.linalg.norm(r_anim_km, axis=0), 1e-12)
    gt_km = r_anim_km * gt_factor[np.newaxis, :]

    ground_track_line, = ax.plot(
        gt_km[0, :], gt_km[1, :], gt_km[2, :],
        color=THEME.ground_track, lw=0.6, alpha=0.20, zorder=2, linestyle="-",
    )

    # Animated trail
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.colors import Normalize

    trail_norm = Normalize(vmin=0.0, vmax=1.0)
    trail_coll = Line3DCollection(
        [], cmap=THEME.trail_cmap, norm=trail_norm, linewidths=2.2, alpha=0.98
    )
    ax.add_collection3d(trail_coll)

    # Animated ground track trail (short, faint)
    gt_coll = Line3DCollection(
        [], cmap="cool", norm=trail_norm, linewidths=0.9, alpha=0.40
    )
    ax.add_collection3d(gt_coll)

    # Spacecraft marker
    sat, = ax.plot(
        [], [], [],
        marker="o", markersize=6.5,
        color=THEME.text, markeredgecolor=THEME.orbit, markeredgewidth=1.2,
        zorder=6,
    )

    # Sun direction vector
    sun_line_glow, = ax.plot([], [], [], color=THEME.sun, lw=5.5, alpha=0.15, zorder=2)
    sun_line,      = ax.plot([], [], [], color=THEME.sun, lw=2.4, alpha=0.95, zorder=5)
    sun_head1,     = ax.plot([], [], [], color=THEME.sun, lw=2.2, alpha=0.95, zorder=6)
    sun_head2,     = ax.plot([], [], [], color=THEME.sun, lw=2.2, alpha=0.95, zorder=6)
    sun_pt,        = ax.plot([], [], [], marker="o", markersize=4.5, color=THEME.sun,
                             alpha=0.95, zorder=7)

    # Limits and aspect
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    # Summary title (static, built once)
    mean_alt = float(np.nanmean(alt_anim_km)) if np.any(np.isfinite(alt_anim_km)) else 0.0
    mean_sma = float(np.nanmean(sma_anim)) if np.any(np.isfinite(sma_anim)) else 0.0
    mean_per = float(np.nanmean(period_anim)) if np.any(np.isfinite(period_anim)) else 0.0
    title_info = (
        f"ST_LRPS — Orbit Evolution  |  "
        f"{duration_days:.2f} d  |  "
        f"mean alt {mean_alt:.0f} km  |  "
        f"SMA {mean_sma:.0f} km  |  "
        f"T {mean_per:.2f} h"
    )
    fig.text(
        0.5, 0.967,
        title_info,
        color=THEME.text,
        fontsize=11,
        ha="center", va="top", weight="bold",
        family="monospace",
    )

    # Legend labels for periapsis / apoapsis
    fig.text(
        0.018, 0.960,
        f"v  peri {peri_alt:.1f} km",
        color=THEME.peri, fontsize=8.5, ha="left", va="top",
        family="monospace",
    )
    fig.text(
        0.018, 0.946,
        f"^  apo  {apo_alt:.1f} km",
        color=THEME.apo, fontsize=8.5, ha="left", va="top",
        family="monospace",
    )

    # -------------------------------------------------------
    # HUD (optional)
    # -------------------------------------------------------
    hud_items = None
    grad_ax = None
    minimap_ax = None
    sun_arrow2d_glow = None
    sun_arrow2d = None
    sun_arrow2d_tip = None

    if show_hud:
        x0, y0f, w0, h0 = 0.018, 0.018, 0.57, 0.350

        hud_shadow = FancyBboxPatch(
            (x0 + 0.004, y0f - 0.004), w0, h0,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=fig.transFigure,
            facecolor="#000000", edgecolor="none", alpha=0.35, zorder=20,
        )
        fig.patches.append(hud_shadow)

        hud_panel = FancyBboxPatch(
            (x0, y0f), w0, h0,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=fig.transFigure,
            facecolor="#070A12", edgecolor="#2C4A7A",
            linewidth=1.25, alpha=0.80, zorder=21,
        )
        fig.patches.append(hud_panel)

        # Trail age gradient bar
        try:
            dt_med = float(np.nanmedian(np.diff(t_anim))) if t_anim.size >= 2 else 0.0
            trail_dur_d = (dt_med * max(0, int(trail_length) - 1)) / 86400.0

            g_left, g_bottom = x0 + 0.19, y0f + 0.012
            g_width, g_height = w0 - 0.21, 0.022

            grad_ax = fig.add_axes([g_left, g_bottom, g_width, g_height], zorder=23)
            grad_ax.set_facecolor((0, 0, 0, 0.35))
            for sp in grad_ax.spines.values():
                sp.set_edgecolor("#2C4A7A"); sp.set_linewidth(1.0); sp.set_alpha(0.9)

            _grad = np.linspace(0.0, 1.0, 256, dtype=float)[None, :]
            grad_ax.imshow(_grad, aspect="auto", cmap=THEME.trail_cmap,
                           origin="lower", extent=[0, 1, 0, 1])
            grad_ax.set_yticks([])
            grad_ax.set_xticks([0.0, 0.5, 1.0])
            lbl_0 = f"-{trail_dur_d:.2f} d" if trail_dur_d > 0 else "old"
            grad_ax.set_xticklabels([lbl_0, "", "0"], fontsize=9,
                                     family="monospace", color=THEME.text)
            grad_ax.tick_params(axis="x", colors=THEME.text, length=0, pad=2)
            grad_ax.set_title("TRAIL AGE", color="#9FB2D8", fontsize=8, pad=2,
                               family="monospace")
        except Exception:
            grad_ax = None

        glow = [
            pe.withStroke(linewidth=6, foreground="#0B3D91", alpha=0.18),
            pe.withStroke(linewidth=3, foreground="#0B3D91", alpha=0.30),
            pe.Normal(),
        ]
        mono = dict(family="monospace", fontsize=9.5, va="top", ha="left",
                    zorder=22, path_effects=glow)

        fig.text(x0 + 0.014, y0f + h0 - 0.018, "TELEMETRY",
                 color="#E8EEF8", weight="bold", **mono)
        fig.text(x0 + 0.014, y0f + h0 - 0.038, "-" * 46,
                 color="#7FB7FF", **mono)

        lbl = "#9FB2D8"
        valY = "#FFE08A"
        valG = "#B6FFB0"
        valC = "#84D6FF"
        valW = "#F8FAFF"
        valO = "#FFBB77"
        valP = "#FF9999"

        y = y0f + h0 - 0.057
        dy = 0.026

        fig.text(x0 + 0.014, y, "MISSION TIME:", color=lbl, **mono)
        t_val = fig.text(x0 + 0.185, y, "", color=valY, **mono)

        y -= dy
        fig.text(x0 + 0.014, y, "ALTITUDE:", color=lbl, **mono)
        o_val = fig.text(x0 + 0.185, y, "", color=valG, **mono)

        y -= dy
        fig.text(x0 + 0.014, y, "SPEED:", color=lbl, **mono)
        v_val = fig.text(x0 + 0.185, y, "", color=valO, **mono)

        y -= dy
        fig.text(x0 + 0.014, y, "INCLINATION:", color=lbl, **mono)
        i_val = fig.text(x0 + 0.185, y, "", color=valC, **mono)

        y -= dy
        fig.text(x0 + 0.014, y, "ECCENTRICITY:", color=lbl, **mono)
        e_val = fig.text(x0 + 0.185, y, "", color=valW, **mono)

        y -= dy
        fig.text(x0 + 0.014, y, "SMA:", color=lbl, **mono)
        sma_val = fig.text(x0 + 0.185, y, "", color=valP, **mono)

        y -= dy
        fig.text(x0 + 0.014, y, "PERIOD:", color=lbl, **mono)
        per_val = fig.text(x0 + 0.185, y, "", color=valP, **mono)

        y -= dy * 1.1
        fig.text(x0 + 0.014, y, "SUN:", color=lbl, **mono)
        s_val = fig.text(x0 + 0.185, y, "", color=valW, **mono)

        hud_items = dict(
            t_val=t_val, o_val=o_val, v_val=v_val,
            i_val=i_val, e_val=e_val, sma_val=sma_val,
            per_val=per_val, s_val=s_val,
        )

        # Mini sun compass
        minimap_ax = fig.add_axes([0.025, 0.115, 0.130, 0.130], zorder=24)
        minimap_ax.set_facecolor((0, 0, 0, 0))
        minimap_ax.set_aspect("equal", adjustable="box")
        minimap_ax.set_xlim(-1.05, 1.05)
        minimap_ax.set_ylim(-1.05, 1.05)
        minimap_ax.set_xticks([])
        minimap_ax.set_yticks([])
        for sp in minimap_ax.spines.values():
            sp.set_visible(False)

        circ = plt.Circle((0, 0), 1.0, facecolor=(0, 0, 0, 0.35),
                           edgecolor=THEME.hud_edge, linewidth=1.0)
        minimap_ax.add_patch(circ)
        minimap_ax.plot([0, 0], [-1, 1], color=THEME.hud_edge, lw=0.7, alpha=0.50)
        minimap_ax.plot([-1, 1], [0, 0], color=THEME.hud_edge, lw=0.7, alpha=0.50)
        minimap_ax.text(0.0, 1.13, "SUN", color=THEME.text, fontsize=7.5,
                        ha="center", va="bottom", family="monospace")

        sun_arrow2d_glow, = minimap_ax.plot(
            [0, 0], [0, 1], color=THEME.sun, lw=5.5, alpha=0.12, solid_capstyle="round"
        )
        sun_arrow2d, = minimap_ax.plot(
            [0, 0], [0, 1], color=THEME.sun, lw=2.0, alpha=0.95, solid_capstyle="round"
        )
        sun_arrow2d_tip, = minimap_ax.plot(
            [0], [1], marker="^", markersize=6.5, color=THEME.sun,
            markeredgecolor=THEME.text, markeredgewidth=0.5, alpha=0.98,
        )

    # -------------------------------------------------------
    # Animation update
    # -------------------------------------------------------
    def _update(i: int):
        i = int(i)

        # Trail
        j0 = max(0, i - int(trail_length))
        tr = r_anim_km[:, j0:i + 1]
        if tr.shape[1] >= 2:
            pts = np.column_stack([tr[0], tr[1], tr[2]]).reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            tt = np.linspace(0.0, 1.0, segs.shape[0])
            trail_coll.set_segments(segs)
            trail_coll.set_array(tt)
        else:
            trail_coll.set_segments([])

        # Ground-track trail (shorter, more subtle)
        gt_j0 = max(0, i - int(trail_length) // 3)
        gt_tr = gt_km[:, gt_j0:i + 1]
        if gt_tr.shape[1] >= 2:
            gpts = np.column_stack([gt_tr[0], gt_tr[1], gt_tr[2]]).reshape(-1, 1, 3)
            gsegs = np.concatenate([gpts[:-1], gpts[1:]], axis=1)
            gtt = np.linspace(0.0, 1.0, gsegs.shape[0])
            gt_coll.set_segments(gsegs)
            gt_coll.set_array(gtt)
        else:
            gt_coll.set_segments([])

        # Spacecraft marker
        p = r_anim_km[:, i]
        sat.set_data([p[0]], [p[1]])
        sat.set_3d_properties([p[2]])

        # Sun direction vector
        s = sun_dir[:, i]
        if sun_arrow2d is not None:
            sx, sy = float(s[0]), float(s[1])
            nxy = math.sqrt(sx * sx + sy * sy)
            if nxy < 1e-12:
                sx, sy, nxy = 0.0, 1.0, 1.0
            sx /= nxy; sy /= nxy
            if sun_arrow2d_glow is not None:
                sun_arrow2d_glow.set_data([0.0, sx], [0.0, sy])
            sun_arrow2d.set_data([0.0, sx], [0.0, sy])
            if sun_arrow2d_tip is not None:
                sun_arrow2d_tip.set_data([sx], [sy])

        L = max(R_body_km * 1.75, 0.55 * max_range)
        tip = (s * L).reshape(3,)

        sun_line_glow.set_data([0.0, tip[0]], [0.0, tip[1]])
        sun_line_glow.set_3d_properties([0.0, tip[2]])
        sun_line.set_data([0.0, tip[0]], [0.0, tip[1]])
        sun_line.set_3d_properties([0.0, tip[2]])

        ax_ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(s, ax_ref))) > 0.92:
            ax_ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(s, ax_ref)
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            u = np.array([1.0, 0.0, 0.0]); nu = 1.0
        u = u / nu

        head_len = 0.12 * L
        head_w = 0.06 * L
        base = tip - head_len * s
        p1 = base + head_w * u
        p2 = base - head_w * u

        sun_head1.set_data([tip[0], p1[0]], [tip[1], p1[1]])
        sun_head1.set_3d_properties([tip[2], p1[2]])
        sun_head2.set_data([tip[0], p2[0]], [tip[1], p2[1]])
        sun_head2.set_3d_properties([tip[2], p2[2]])

        sun_pt.set_data([tip[0]], [tip[1]])
        sun_pt.set_3d_properties([tip[2]])

        # HUD telemetry
        if hud_items is not None:
            days, hours = _format_mission_time(float(t_anim[i]))
            alt = float(alt_anim_km[i]) if np.isfinite(alt_anim_km[i]) else float("nan")
            inc = float(inc_anim[i])    if np.isfinite(inc_anim[i])    else float("nan")
            ee  = float(e_anim[i])      if np.isfinite(e_anim[i])      else float("nan")
            spd = float(speed_anim[i])  if np.isfinite(speed_anim[i])  else float("nan")
            sma = float(sma_anim[i])    if np.isfinite(sma_anim[i])    else float("nan")
            per = float(period_anim[i]) if np.isfinite(period_anim[i]) else float("nan")

            hud_items["t_val"].set_text(f"{days:7.3f} d  ({hours:7.2f} h)")
            hud_items["o_val"].set_text(f"{alt:9.3f} km")
            hud_items["v_val"].set_text(f"{spd:9.4f} km/s")
            hud_items["i_val"].set_text(f"{inc:9.3f} deg")
            hud_items["e_val"].set_text(f"{ee:9.5f}")
            hud_items["sma_val"].set_text(f"{sma:9.1f} km")
            hud_items["per_val"].set_text(f"{per:9.4f} h")
            hud_items["s_val"].set_text(_sun_arrow_text(s))

        # Camera rotation
        if rotate_camera:
            dur_days = max(1e-9, (float(t_anim[-1]) - float(t_anim[0])) / 86400.0)
            sweep = float(np.clip(float(cam_rate_deg_per_day) * dur_days, 0.0, 160.0))
            az = float(azim0_deg) + sweep * ((float(t_anim[i]) - float(t_anim[0])) / 86400.0) / dur_days
            ax.view_init(elev=float(elev_deg), azim=az)
        else:
            ax.view_init(elev=float(elev_deg), azim=float(azim0_deg))

        # Moon lighting + terminator update (amortised)
        if light_update_every and int(light_update_every) > 0 and (i % int(light_update_every) == 0):
            try:
                fc = _moon_facecolors(mx, my, mz, s, ambient=0.06, diffuse=0.94,
                                      base_rgb=THEME.moon_base)
                moon_surf.set_facecolors(fc.reshape(-1, 4))
            except Exception:
                pass
            _update_terminator_ring(term_line, R_body_km, s)

        return trail_coll, gt_coll, sat, sun_line_glow, sun_line, sun_head1, sun_head2, sun_pt

    anim_obj = animation.FuncAnimation(
        fig, _update,
        frames=frames,
        interval=1000.0 / max(1, int(fps)),
        blit=False,
        repeat=False,
    )

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------
    ext = os.path.splitext(save_path)[1].lower()

    try:
        if ext == ".gif":
            anim_obj.save(save_path, writer="pillow", fps=int(fps), dpi=int(dpi))
            print(f"[3D] Saved GIF: {save_path}")
        else:
            ffmpeg_ok = shutil.which("ffmpeg") is not None
            if not ffmpeg_ok:
                gif_path = os.path.splitext(save_path)[0] + ".gif"
                print(f"[3D] ffmpeg not found. Falling back to GIF: {gif_path}")
                anim_obj.save(gif_path, writer="pillow", fps=int(fps), dpi=int(dpi))
                save_path = gif_path
            else:
                writer = animation.FFMpegWriter(
                    fps=int(fps),
                    codec="libx264",
                    bitrate=0,
                    extra_args=[
                        "-pix_fmt", "yuv420p",
                        "-preset", str(preset),
                        "-crf", str(int(crf)),
                    ],
                )
                anim_obj.save(save_path, writer=writer, dpi=int(dpi))
                print(f"[3D] Saved MP4: {save_path}")
    except Exception as exc:
        print(f"[3D] Failed to save animation: {exc}")
        save_path = None

    plt.close(fig)
    return save_path


if __name__ == "__main__":
    # Minimal self-test (synthetic trajectory)
    t = np.linspace(0.0, 6.0 * 86400.0, 12000)
    mu = MU_MOON_DEFAULT
    Rm = R_MOON_KM_DEFAULT * 1000.0
    a = Rm + 200e3
    omega = math.sqrt(mu / (a ** 3))

    r = np.zeros((3, t.size))
    v = np.zeros((3, t.size))
    r[0] = a * np.cos(omega * t)
    r[1] = a * np.sin(omega * t)
    v[0] = -a * omega * np.sin(omega * t)
    v[1] = a * omega * np.cos(omega * t)
    r[2] = 50e3 * np.sin(0.05 * omega * t)

    sun = np.zeros((3, t.size))
    sun[0] = 1.0
    sun[1] = 0.15 * np.sin(2.0 * np.pi * t / (7.0 * 86400.0))
    sun = _unit(sun, axis=0)

    mock = {
        "t_s": t, "r_m": r, "v_mps": v, "sun_dir_eci": sun,
        "meta": {"body_radius_m": Rm, "mu_m3s2": mu,
                 "spice": {"start_utc": "2025-01-01T00:00:00"}},
    }
    out = os.path.join(os.path.dirname(__file__), "_test_anim")
    os.makedirs(out, exist_ok=True)
    render_orbit_animation(mock, out, filename="test_orbit.mp4",
                  frames=900, fps=30, trail_length=260, preset="slow", crf=18)
