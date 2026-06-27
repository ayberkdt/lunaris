"""Adaptive spherical-harmonic degree and albedo sampling kernels."""

from __future__ import annotations

import numpy as np
from numba import njit

from lunaris.common.math_utils import _sample_2d_scaled_bilinear_kernel, clamp, wrap_lon_deg


@njit(cache=True)
def _sample_albedo_dn_scaled(
    lat_deg: float,
    lon_deg: float,
    dn_grid: np.ndarray,
    n_rows: int,
    n_cols: int,
    deg_per_px: float,
    lon_ref_deg: float,
    lat_ref_deg: float,
    flip_lat: int,
    scale: float,
    bias: float,
    nodata_dn: float,
    lat_min_deg: float,
    lat_max_deg: float,
) -> float:
    """
    Bilinearly sample a scaled-DN albedo grid at (lat, lon).

    Longitude is wrapped to [0, 360) and latitude is clamped to [lat_min_deg, lat_max_deg].
    Returns scale * DN + bias via a scaled bilinear sampler (nodata handled by the sampler).
    """

    # Guard against degenerate grids / invalid resolution
    if deg_per_px <= 0.0 or n_rows <= 0 or n_cols <= 0:
        return bias

    # These helpers must be Numba-compatible
    lon_wrapped = wrap_lon_deg(lon_deg)
    lat_clamped = clamp(lat_deg, lat_min_deg, lat_max_deg)

    # Latitude -> fractional row index
    # flip_lat == 0: i_f = (lat_ref - lat)/deg_per_px
    # else        : i_f = (lat - lat_ref)/deg_per_px
    lat_sign = 1.0 if flip_lat == 0 else -1.0
    i_f = (lat_sign * (lat_ref_deg - lat_clamped)) / deg_per_px

    # Longitude -> fractional col index (periodic)
    j_f = (lon_wrapped - lon_ref_deg) / deg_per_px
    j_f = j_f % n_cols

    return _sample_2d_scaled_bilinear_kernel(
        dn_grid, i_f, j_f, n_rows, n_cols, scale, bias, nodata_dn
    )


@njit(cache=True)
def _select_adaptive_sh_degree(
    r_norm_m: float,
    r_ref_m: float,
    degree_max: int,
    adaptive_mode: int,
    adaptive_power: float,
    adaptive_min_degree: int,
    quantization_step: int,
    table_alt_km: np.ndarray,
    table_degree: np.ndarray,
    table_len: int,
) -> int:
    """
    Pick the spherical-harmonic degree to evaluate for the current radius.

    Modes
    -----
    0:
        Adaptive logic disabled; always return `degree_max`.
    1:
        Piecewise altitude table. The first threshold where `alt_km <= threshold`
        wins; above the last threshold we fall back to `adaptive_min_degree`.
    2:
        Smooth power-law heuristic based on `(R_ref / r) ** power`.

    Notes
    -----
    The return value is always clamped to `[adaptive_min_degree, degree_max]` and
    quantized downward when `quantization_step > 1`. Quantizing downward avoids
    unexpectedly increasing runtime cost above the policy's requested level.
    """

    if degree_max <= 0:
        return 0

    n_eval = int(degree_max)
    min_degree = int(adaptive_min_degree)
    if min_degree < 0:
        min_degree = 0
    if min_degree > degree_max:
        min_degree = int(degree_max)

    if adaptive_mode == 1 and table_len > 0:
        alt_km = (float(r_norm_m) - float(r_ref_m)) / 1000.0
        if alt_km < 0.0:
            alt_km = 0.0

        n_eval = min_degree
        for i in range(int(table_len)):
            if alt_km <= float(table_alt_km[i]):
                n_eval = int(table_degree[i])
                break
    elif adaptive_mode == 2:
        safe_r = float(r_norm_m)
        if safe_r < float(r_ref_m):
            safe_r = float(r_ref_m)
        ratio = float(r_ref_m) / safe_r
        n_eval = int(float(degree_max) * (ratio ** float(adaptive_power)))

    if n_eval < min_degree:
        n_eval = min_degree
    if n_eval > degree_max:
        n_eval = int(degree_max)

    qstep = int(quantization_step)
    if qstep > 1:
        n_eval = (n_eval // qstep) * qstep
        if n_eval < min_degree:
            n_eval = min_degree

    return int(n_eval)

__all__ = ["_sample_albedo_dn_scaled", "_select_adaptive_sh_degree"]
