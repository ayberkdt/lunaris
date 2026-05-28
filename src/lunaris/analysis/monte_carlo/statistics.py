# ST_LRPS/analysis/monte_carlo/statistics.py
# -*- coding: utf-8 -*-
"""
Monte Carlo Statistical Analysis
==================================

Post-processes a ``MCRunResult`` ensemble into actionable statistics:

1. **Ensemble statistics** (mean, covariance, σ-bounds as a function of time):
   - Mean trajectory:  μ(t) = E[r(t), v(t)]
   - Covariance tube:  P(t) = Cov[Y(t)]  →  6×6 matrix at each epoch
   - σ-bounds:         ±nσ position / velocity spread

2. **Error ellipsoids**:
   - 3-σ position ellipsoid semi-axes and orientation at each epoch
   - Suitable for 3-D visualisation as wireframes or tubes

3. **Impact statistics**:
   - Impact probability estimate with Binomial confidence interval
   - Impact time / location distribution
   - Geographic impact spread (lat/lon histogram)

4. **Orbital element dispersion**:
   - Semi-major axis, eccentricity, inclination spread vs time

All computations are pure NumPy (no Numba / CUDA dependencies), so this
module can run without a GPU environment.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from lunaris.common.montecarlo_defs import MCRunResult
from lunaris.common.type_defs import F64Array
from lunaris.core.state import cartesian_to_keplerian
from lunaris.common.constants import R_MOON


# =============================================================================
# 1.              HELPER FUNCTIONS
# =============================================================================

def _cov6(Y_t: F64Array) -> F64Array:
    """
    Compute 6×6 sample covariance matrix of Y_t of shape (N, 6).

    Uses ddof=1 (unbiased estimator).  Returns zeros if N < 2.
    """
    N = int(Y_t.shape[0])
    if N < 2:
        return np.zeros((6, 6), dtype=np.float64)
    return np.cov(Y_t.T, ddof=1).astype(np.float64)


def _position_ellipsoid_axes(P_pos: F64Array) -> Tuple[F64Array, F64Array]:
    """
    Compute 3-σ semi-axes and orientation of the position error ellipsoid.

    Parameters
    ----------
    P_pos : (3, 3) position covariance sub-matrix.

    Returns
    -------
    semi_axes : (3,) 3-σ semi-axis lengths [m]
    eigvecs   : (3, 3) columns are principal-axis unit vectors
    """
    P_pos = np.asarray(P_pos, dtype=np.float64)
    # Symmetrize to guard against floating-point asymmetry
    P_sym = 0.5 * (P_pos + P_pos.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    eigvals = np.maximum(eigvals, 0.0)    # numerical positivity
    semi_axes = 3.0 * np.sqrt(eigvals)   # 3-σ ellipsoid
    return semi_axes, eigvecs




def _binomial_ci_wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score 95% confidence interval for a Binomial proportion k/n.

    Returns (lower, upper) as probabilities in [0, 1].
    """
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    z2    = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    half   = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# =============================================================================
# 2.              RESULT CONTAINERS
# =============================================================================

@dataclass
class EnsembleStatistics:
    """
    Time-varying ensemble statistics for the position-velocity state.

    Shapes
    ------
    t          : (T,)      time [s]
    mean       : (T, 6)    ensemble mean [m, m/s]
    cov        : (T, 6, 6) full covariance matrices
    std        : (T, 6)    element-wise standard deviations
    alt_mean   : (T,)      mean altitude [km]
    alt_std    : (T,)      altitude 1-sigma [km]
    """
    t:          F64Array
    mean:       F64Array
    cov:        F64Array
    std:        F64Array
    alt_mean:   F64Array
    alt_std:    F64Array

    def sigma_bounds(self, n: float = 3.0) -> Tuple[F64Array, F64Array]:
        """Return (mean - n*std, mean + n*std) bounds for all 6 state components."""
        return self.mean - n * self.std, self.mean + n * self.std

    def pos_cov(self) -> F64Array:
        """Return (T, 3, 3) position covariance sub-matrices."""
        return self.cov[:, :3, :3]

    def vel_cov(self) -> F64Array:
        """Return (T, 3, 3) velocity covariance sub-matrices."""
        return self.cov[:, 3:, 3:]


@dataclass
class ErrorEllipsoids:
    """
    3-σ position error ellipsoids at each output epoch.

    Attributes
    ----------
    t          : (T,)      epoch times [s]
    semi_axes  : (T, 3)    3-σ semi-axis lengths [m], sorted ascending
    eigvecs    : (T, 3, 3) columns = principal-axis unit vectors
    centres    : (T, 3)    ellipsoid centres (= ensemble mean position) [m]
    """
    t:          F64Array
    semi_axes:  F64Array
    eigvecs:    F64Array
    centres:    F64Array

    def tube_radii(self) -> F64Array:
        """RMS of the three semi-axes – a scalar 'tube radius' at each epoch."""
        return np.sqrt(np.mean(self.semi_axes ** 2, axis=1))


@dataclass
class ImpactStatistics:
    """
    Impact probability and geographic distribution.

    Attributes
    ----------
    n_total       : total number of MC samples
    n_impacts     : number that hit the surface
    p_impact      : MLE estimate of impact probability
    p_impact_ci95 : (lower, upper) Wilson 95% confidence interval
    t_impact_mean : mean impact time [s]  (NaN if n_impacts == 0)
    t_impact_std  : std of impact times [s]
    lat_deg       : (K,) geodetic latitudes of impact sites [deg]
    lon_deg       : (K,) longitudes of impact sites [deg]
    """
    n_total:        int
    n_impacts:      int
    p_impact:       float
    p_impact_ci95:  Tuple[float, float]
    t_impact_mean:  float
    t_impact_std:   float
    lat_deg:        F64Array
    lon_deg:        F64Array


@dataclass
class OEDispersion:
    """
    Osculating Keplerian element dispersion vs time.

    Shapes: all arrays (T,).
    """
    t:          F64Array
    a_mean_km:  F64Array
    a_std_km:   F64Array
    e_mean:     F64Array
    e_std:      F64Array
    inc_mean_deg: F64Array
    inc_std_deg:  F64Array


@dataclass
class MCStatistics:
    """
    Complete Monte Carlo analysis output.

    Produced by :func:`compute_mc_statistics`.
    """
    ensemble:   EnsembleStatistics
    ellipsoids: ErrorEllipsoids
    impacts:    ImpactStatistics
    oe_disp:    Optional[OEDispersion] = None

    # Raw result reference (not serialised by default)
    _raw: Optional[MCRunResult] = field(default=None, repr=False)


# =============================================================================
# 3.              ANALYSIS FUNCTIONS
# =============================================================================

def compute_ensemble_statistics(
    result: MCRunResult,
    *,
    use_survived_only: bool = False,
    r_ref_m: float = R_MOON,
) -> EnsembleStatistics:
    """
    Compute time-varying mean, covariance, and altitude statistics.

    Parameters
    ----------
    result : MCRunResult
    use_survived_only : bool
        If True, exclude impacted samples from statistics.
    r_ref_m : float
        Reference radius for altitude computation [m].

    Returns
    -------
    EnsembleStatistics
    """
    Y = result.Y          # (T, N, 6)
    t = result.t          # (T,)

    if use_survived_only:
        mask = result.impact_mask < 0.5
        if mask.sum() < 2:
            warnings.warn(
                "[mc_analysis] Fewer than 2 survived samples; using full ensemble.",
                RuntimeWarning,
            )
            mask = np.ones(result.n_samples, dtype=bool)
        Y = Y[:, mask, :]

    T = int(Y.shape[0])
    N = int(Y.shape[1])

    mean  = np.mean(Y, axis=1)           # (T, 6)
    std   = np.std(Y, axis=1, ddof=1)    # (T, 6)
    cov   = np.zeros((T, 6, 6), dtype=np.float64)

    for k in range(T):
        cov[k] = _cov6(Y[k])

    # Altitude statistics
    pos_norms = np.linalg.norm(Y[:, :, :3], axis=2)   # (T, N)
    alt_km    = (pos_norms - r_ref_m) / 1_000.0        # (T, N)
    alt_mean  = np.mean(alt_km, axis=1)                # (T,)
    alt_std   = np.std(alt_km, axis=1, ddof=1 if N > 1 else 0)  # (T,)

    return EnsembleStatistics(
        t=np.ascontiguousarray(t, dtype=np.float64),
        mean=np.ascontiguousarray(mean, dtype=np.float64),
        cov=np.ascontiguousarray(cov, dtype=np.float64),
        std=np.ascontiguousarray(std, dtype=np.float64),
        alt_mean=np.ascontiguousarray(alt_mean, dtype=np.float64),
        alt_std=np.ascontiguousarray(alt_std, dtype=np.float64),
    )


def compute_error_ellipsoids(
    ens: EnsembleStatistics,
) -> ErrorEllipsoids:
    """
    Compute 3-σ position error ellipsoids from the ensemble covariance history.

    Parameters
    ----------
    ens : EnsembleStatistics from :func:`compute_ensemble_statistics`

    Returns
    -------
    ErrorEllipsoids
    """
    T = int(ens.t.shape[0])
    semi_axes_all = np.zeros((T, 3), dtype=np.float64)
    eigvecs_all   = np.zeros((T, 3, 3), dtype=np.float64)

    for k in range(T):
        P_pos = ens.cov[k, :3, :3]
        axes, vecs = _position_ellipsoid_axes(P_pos)
        semi_axes_all[k] = axes
        eigvecs_all[k]   = vecs

    return ErrorEllipsoids(
        t=ens.t.copy(),
        semi_axes=semi_axes_all,
        eigvecs=eigvecs_all,
        centres=ens.mean[:, :3].copy(),
    )


def compute_impact_statistics(
    result: MCRunResult,
    *,
    r_ref_m: float = R_MOON,
) -> ImpactStatistics:
    """
    Compute impact probability and geographic distribution of impact sites.

    Parameters
    ----------
    result : MCRunResult
    r_ref_m : float
        Reference radius for lat/lon computation [m].

    Returns
    -------
    ImpactStatistics
    """
    N     = result.n_samples
    mask  = result.impact_mask > 0.5
    n_hit = int(mask.sum())
    p_mle = float(n_hit) / max(1, N)
    ci95  = _binomial_ci_wilson(n_hit, N)

    t_hit = result.t_impact[mask]
    t_hit = t_hit[np.isfinite(t_hit)]
    t_mean = float(np.mean(t_hit)) if len(t_hit) > 0 else math.nan
    t_std  = float(np.std(t_hit, ddof=1)) if len(t_hit) > 1 else 0.0

    # Impact lat/lon: use last known position for impacting samples
    lat_arr = np.zeros(n_hit, dtype=np.float64)
    lon_arr = np.zeros(n_hit, dtype=np.float64)

    hit_indices = np.where(mask)[0]
    for j, i in enumerate(hit_indices):
        # Last snapshot position (proxy for impact site)
        r = result.Y[-1, i, :3]
        r_n = float(np.linalg.norm(r))
        if r_n > 0.0:
            lat_arr[j] = math.degrees(math.asin(float(np.clip(r[2] / r_n, -1.0, 1.0))))
            lon_arr[j] = math.degrees(math.atan2(float(r[1]), float(r[0])))

    return ImpactStatistics(
        n_total=N,
        n_impacts=n_hit,
        p_impact=p_mle,
        p_impact_ci95=ci95,
        t_impact_mean=t_mean,
        t_impact_std=t_std,
        lat_deg=lat_arr,
        lon_deg=lon_arr,
    )


def compute_oe_dispersion(
    result: MCRunResult,
    mu: float = 4.9048695e12,     # μ_Moon [m³/s²]
    *,
    max_samples: int = 500,       # cap for Keplerian conversion (expensive)
) -> OEDispersion:
    """
    Compute osculating Keplerian element dispersion at each output epoch.

    Parameters
    ----------
    result : MCRunResult
    mu : float
        Gravitational parameter of the central body [m³/s²].
    max_samples : int
        Cap the number of samples used (random draw) to keep runtime bounded.

    Returns
    -------
    OEDispersion
    """
    Y = result.Y     # (T, N, 6)
    t = result.t     # (T,)
    T = int(t.shape[0])
    N = int(Y.shape[1])

    sub = min(N, max_samples)
    idx = np.random.default_rng(0).choice(N, size=sub, replace=False)

    a_mean  = np.zeros(T); a_std  = np.zeros(T)
    e_mean  = np.zeros(T); e_std  = np.zeros(T)
    inc_mean= np.zeros(T); inc_std= np.zeros(T)

    for k in range(T):
        a_arr   = np.zeros(sub); e_arr   = np.zeros(sub); inc_arr = np.zeros(sub)
        for j, i in enumerate(idx):
            try:
                a_m, e_val, inc_rad, _, _, _ = cartesian_to_keplerian(Y[k, i, :3], Y[k, i, 3:], mu=mu)
                a_arr[j]   = (a_m / 1000.0) if math.isfinite(a_m) else math.nan
                e_arr[j]   = e_val          if math.isfinite(e_val) else math.nan
                inc_arr[j] = math.degrees(inc_rad) if math.isfinite(inc_rad) else math.nan
            except Exception:
                a_arr[j]   = math.nan
                e_arr[j]   = math.nan
                inc_arr[j] = math.nan

        a_mean[k]   = float(np.nanmean(a_arr))
        a_std[k]    = float(np.nanstd(a_arr,  ddof=1))
        e_mean[k]   = float(np.nanmean(e_arr))
        e_std[k]    = float(np.nanstd(e_arr,  ddof=1))
        inc_mean[k] = float(np.nanmean(inc_arr))
        inc_std[k]  = float(np.nanstd(inc_arr, ddof=1))

    return OEDispersion(
        t=np.ascontiguousarray(t, dtype=np.float64),
        a_mean_km=a_mean, a_std_km=a_std,
        e_mean=e_mean, e_std=e_std,
        inc_mean_deg=inc_mean, inc_std_deg=inc_std,
    )


def compute_mc_statistics(
    result: MCRunResult,
    *,
    mu: float = 4.9048695e12,
    r_ref_m: float = R_MOON,
    compute_oe: bool = True,
    use_survived_only: bool = False,
) -> MCStatistics:
    """
    Master function: runs all analyses and returns ``MCStatistics``.

    Parameters
    ----------
    result : MCRunResult
    mu : float
        Lunar gravitational parameter [m³/s²].
    r_ref_m : float
        Reference radius for altitude computation [m].
    compute_oe : bool
        Whether to compute orbital element dispersion (can be slow for large N).
    use_survived_only : bool
        Exclude impacting samples from ensemble statistics.

    Returns
    -------
    MCStatistics
    """
    ens   = compute_ensemble_statistics(result, use_survived_only=use_survived_only, r_ref_m=r_ref_m)
    ell   = compute_error_ellipsoids(ens)
    imps  = compute_impact_statistics(result, r_ref_m=r_ref_m)
    oe    = compute_oe_dispersion(result, mu=mu) if compute_oe else None

    return MCStatistics(ensemble=ens, ellipsoids=ell, impacts=imps, oe_disp=oe, _raw=result)


# =============================================================================
# 4.              COVARIANCE PROPAGATION HELPERS (linear theory)
# =============================================================================

def propagate_covariance_linear(
    P0: F64Array,       # (6, 6) initial covariance
    Phi: F64Array,      # (T, 6, 6) state transition matrices
) -> F64Array:
    """
    Propagate covariance via linear (STM) theory:
        P(t) = Φ(t) P0 Φ(t)ᵀ

    Parameters
    ----------
    P0 : (6, 6) initial covariance
    Phi : (T, 6, 6) state transition matrices from t0 to each epoch

    Returns
    -------
    P_hist : (T, 6, 6)
    """
    T = int(Phi.shape[0])
    P_hist = np.zeros((T, 6, 6), dtype=np.float64)
    for k in range(T):
        F = Phi[k]
        P_hist[k] = F @ P0 @ F.T
    return P_hist


def mahalanobis_distance(
    y: F64Array,        # (6,) query state
    mean: F64Array,     # (6,) ensemble mean
    P: F64Array,        # (6, 6) covariance
) -> float:
    """
    Compute Mahalanobis distance: d² = (y - μ)ᵀ P⁻¹ (y - μ).

    Returns NaN if P is singular.
    """
    delta = np.asarray(y, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    try:
        P_inv = np.linalg.inv(P)
        return float(delta @ P_inv @ delta)
    except np.linalg.LinAlgError:
        return math.nan


# =============================================================================
# 5.                        PUBLIC API
# =============================================================================

__all__ = [
    # Data containers
    "EnsembleStatistics",
    "ErrorEllipsoids",
    "ImpactStatistics",
    "OEDispersion",
    "MCStatistics",
    # Analysis functions
    "compute_ensemble_statistics",
    "compute_error_ellipsoids",
    "compute_impact_statistics",
    "compute_oe_dispersion",
    "compute_mc_statistics",
    # Helpers
    "propagate_covariance_linear",
    "mahalanobis_distance",
]

