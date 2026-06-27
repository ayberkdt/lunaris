"""Telemetry and terrain-radius helpers for propagation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np


def _make_telem_dict(
    t_s: float,
    y: np.ndarray,
    R_ref_m: float,
    mu_m3s2: float,
    *,
    t_frame_s: float | None = None,
    r_i_to_bf: Callable[[float, np.ndarray], np.ndarray] | None = None,
    surface_radius_m: Callable[[float, float], float] | None = None,
) -> dict[str, float] | None:
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

def _latlon_from_r_bf(r_bf: np.ndarray) -> tuple[float, float]:
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

    if hasattr(topo, "sample_bilinear") and callable(topo.sample_bilinear):
        fn = topo.sample_bilinear

        def _radius_from_bilinear(lat_rad: float, lon_rad: float) -> float:
            return float(fn(math.degrees(lat_rad), math.degrees(lon_rad) % 360.0, kind="radius_m"))

        return _radius_from_bilinear

    if hasattr(topo, "sample_nearest") and callable(topo.sample_nearest):
        fn = topo.sample_nearest

        def _radius_from_nearest(lat_rad: float, lon_rad: float) -> float:
            return float(fn(math.degrees(lat_rad), math.degrees(lon_rad) % 360.0, kind="radius_m"))

        return _radius_from_nearest

    if hasattr(topo, "radius_m_deg") and callable(topo.radius_m_deg):
        fn = topo.radius_m_deg

        def _radius_from_deg(lat_rad: float, lon_rad: float) -> float:
            return float(fn(math.degrees(lat_rad), math.degrees(lon_rad) % 360.0))

        return _radius_from_deg

    if hasattr(topo, "radius_m") and callable(topo.radius_m):
        fn = topo.radius_m

        def _radius_from_rad(lat_rad: float, lon_rad: float) -> float:
            return float(fn(float(lat_rad), float(lon_rad)))

        return _radius_from_rad

    raise AttributeError(
        "Topography object does not expose a usable radius sampler "
        "(expected sample_bilinear/sample_nearest/radius_m_deg/radius_m)."
    )


__all__ = ["_make_telem_dict", "_latlon_from_r_bf", "_build_surface_radius_sampler"]
