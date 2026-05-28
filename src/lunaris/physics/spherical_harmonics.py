"""
Spherical Harmonic Gravity Kernels (High-Performance Lunar Engine)

This module serves as the primary numerical engine for evaluating gravitational 
acceleration using Spherical Harmonic models. It is strictly 'compute-only', 
meaning it contains no I/O, file parsing, or dataset management.

Core Philosophy:
----------------
1. Separation of Concerns: Numerical kernels live here; I/O lives in loaders/.
2. Frame Integrity: All calculations are performed in the BODY-FIXED frame.
3. Speed: All kernels are JIT-compiled via Numba with high-degree optimizations.
4. Stability: Implements Kahan summation and pole-safe guards for high-degree models.

Component Overview:
-------------------
- Foundations & Infrastructure:
    * SHWorkspace                  : Pre-allocated scratch buffers (zero-allocation per call).
    * get_legendre_recurrence_tables: Precomputes recurrence constants for stable ALFs.
    * slice_gravity_model          : Normalizes and truncates coefficient matrices.

- Numerical Acceleration Kernels (Body-Fixed):
    * sh_accel_fixed_numba          : Deterministic, Kahan-compensated serial kernel.
    * sh_accel_adaptive_blend_numba : Altitude-aware, dual-fidelity blended kernel.
    * compute_point_mass_acceleration: High-performance Newtonian monopole baseline.

- Dispatch & API:
    * GravityModel (Class)         : The high-level container managing data and state.
    * sh_accel_fixed               : Python-level dispatch (Serial/Parallel) logic.

Data Flow & Design:
-------------------
- Input: Position (x, y, z) in the BODY-FIXED frame.
- Output: Acceleration vector (ax, ay, az) in the BODY-FIXED frame.
- Rotations: Frame transformations (Inertial <-> Body-Fixed) must be handled 
  upstream by specialized frame/attitude modules.

Numerical Specifications:
-------------------------
- Normalization: Fully-normalized (4π) Associated Legendre Functions (ALFs).
- Precision: Compensated (Kahan) summation protects against rounding error in large sums.
- Singularity Protection: Pole-safe Cartesian mapping prevents divergence at latitudes ±90°.

Unit System (SI):
-----------------
- Reference Radius (r_ref)       : Meters [m]
- Gravitational Parameter (mu)   : [m^3 / s^2]
- Position & Distance            : Meters [m]
- Acceleration Output            : [m / s^2]

Dependencies:
-------------
- NumPy: Array operations and data storage.
- Numba: Just-In-Time compilation (nopython=True, parallel=True).
"""


# =============================================================================
# 0.                               IMPORTS
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, TypeAlias

import math
import numpy as np

from numba import njit, prange

from lunaris.common.constants import EPS_1E12, EPS_1E24, NEARLY_UNIT
from lunaris.common.type_defs import Vec3, Arr1, Arr2, F64
from lunaris.common.math_utils import clamp

logger = logging.getLogger(__name__)



# =============================================================================
# 1.                       SETUP & PRECOMPUTATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class SHWorkspace:
    """Reusable work arrays for SH evaluation (fully-normalized, real form)."""
    P: Arr2      # (N+1, N+1)
    dP: Arr2     # (N+1, N+1)
    cos_m: Arr1  # (N+1,)
    sin_m: Arr1  # (N+1,)


def build_legendre_coeffs(n_max: int) -> Tuple[Arr1, Arr1, Arr2, Arr2, Arr1]:
    """
    Precompute recurrence constants for fully-normalized associated Legendre functions (P̄_n^m).

    Convention
    ----------
    - Normalization: fully-normalized (4π, geodesy convention)
    - Condon–Shortley phase: (-1)^m is applied via `scale_m`
    - Real-harmonic scaling: m>0 multiplied by sqrt(2)

    Parameters
    ----------
    n_max:
        Maximum degree N (>= 0).

    Returns
    -------
    diag:
        (N+1,) float64
        Diagonal recurrence coefficients for P̄_n^n.
    subdiag:
        (N+1,) float64
        Sub-diagonal recurrence coefficients for P̄_n^{n-1}.
    A:
        (N+1, N+1) float64
        Vertical recursion coefficient A[n, m] valid for n>=2, m<=n-2.
    B:
        (N+1, N+1) float64
        Vertical recursion coefficient B[n, m] valid for n>=2, m<=n-2.
    scale_m:
        (N+1,) float64
        Per-order scaling = sqrt(2) for m>0 and (-1)^m sign (Condon–Shortley).
    """
    N = int(n_max)
    if N < 0:
        raise ValueError(f"n_max must be >= 0. Got {n_max}.")

    diag: Arr1 = np.zeros(N + 1, dtype=F64)
    subdiag: Arr1 = np.zeros(N + 1, dtype=F64)
    A: Arr2 = np.zeros((N + 1, N + 1), dtype=F64)
    B: Arr2 = np.zeros((N + 1, N + 1), dtype=F64)

    # 1) Diagonal & sub-diagonal coefficients
    if N >= 1:
        n = np.arange(1, N + 1, dtype=F64)
        diag[1:] = np.sqrt((2.0 * n + 1.0) / (2.0 * n))
        subdiag[1:] = np.sqrt(2.0 * n + 1.0)

    # 2) Vertical recursion coefficients A, B for n>=2 and m=0..n-2
    for n_int in range(2, N + 1):
        n = float(n_int)
        m = np.arange(0.0, n - 1.0, dtype=F64)  # 0..n-2

        # A[n,m] = sqrt(((2n-1)(2n+1))/((n-m)(n+m)))
        anm = np.sqrt(((2.0 * n - 1.0) * (2.0 * n + 1.0)) / ((n - m) * (n + m)))

        # B[n,m] = sqrt(((2n+1)(n-m-1)(n+m-1))/((2n-3)(n+m)(n-m)))
        bnm = np.sqrt(
            ((2.0 * n + 1.0) * (n - m - 1.0) * (n + m - 1.0))
            / ((2.0 * n - 3.0) * (n + m) * (n - m))
        )

        A[n_int, : n_int - 1] = anm
        B[n_int, : n_int - 1] = bnm

    # 3) Order scaling: sqrt(2) for m>0 and (-1)^m sign
    scale_m: Arr1 = np.ones(N + 1, dtype=F64)
    if N >= 1:
        scale_m[1:] *= math.sqrt(2.0)

    m_idx = np.arange(N + 1, dtype=np.int64)
    sign = 1.0 - 2.0 * (m_idx & 1)  # even->+1, odd->-1
    scale_m *= sign.astype(F64)

    return diag, subdiag, A, B, scale_m


def slice_gravity_model(
    Cnm_full: np.ndarray,
    Snm_full: np.ndarray,
    degree: int,
) -> Tuple[Arr2, Arr2]:
    """
    Extract coefficient matrices up to degree N and return strict square blocks.

    The Numba kernels in this module assume coefficient arrays are always shaped (N+1, N+1).
    Therefore this helper *always* returns square, C-contiguous float64 arrays.

    - Square inputs:      (N_full+1, N_full+1) -> sliced
    - Rectangular inputs: (n_max+1, m_max+1)   -> padded with zeros in the order dimension

    Parameters
    ----------
    Cnm_full, Snm_full:
        Full coefficient arrays.
    degree:
        Target degree N for truncation (>= 0).

    Returns
    -------
    Cnm, Snm:
        (N+1, N+1) contiguous float64 coefficient blocks.
    """
    N = int(degree)
    if N < 0:
        raise ValueError(f"degree must be >= 0. Got {degree}.")

    C = np.asarray(Cnm_full)
    S = np.asarray(Snm_full)

    if C.ndim != 2 or S.ndim != 2:
        raise ValueError("Cnm_full and Snm_full must be 2D arrays.")
    if C.shape != S.shape:
        raise ValueError(f"Cnm_full shape {C.shape} and Snm_full shape {S.shape} must match.")

    need = N + 1
    n0, m0 = C.shape
    if n0 < need:
        raise ValueError(f"Not enough degree rows: Cnm_full has {n0} rows, need {need} (degree={N}).")

    # Always return (need, need)
    outC: Arr2 = np.zeros((need, need), dtype=F64)
    outS: Arr2 = np.zeros((need, need), dtype=F64)

    m_copy = min(need, m0)
    outC[:, :m_copy] = C[:need, :m_copy]
    outS[:, :m_copy] = S[:need, :m_copy]

    return np.ascontiguousarray(outC), np.ascontiguousarray(outS)


def make_sh_workspace(degree: int) -> SHWorkspace:
    """
    Allocate reusable workspace arrays for SH evaluation (typed dataclass form).

    Safer than dict; no string keys.
    """
    N = int(degree)
    if N < 0:
        raise ValueError(f"degree must be >= 0. Got {degree}.")

    P: Arr2 = np.zeros((N + 1, N + 1), dtype=F64)
    dP: Arr2 = np.zeros((N + 1, N + 1), dtype=F64)
    cos_m: Arr1 = np.zeros(N + 1, dtype=F64)
    sin_m: Arr1 = np.zeros(N + 1, dtype=F64)

    return SHWorkspace(P=P, dP=dP, cos_m=cos_m, sin_m=sin_m)


# Cache: degree -> (diag, subdiag, A, B, scale_m)
_LEGENDRE_CACHE: dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_legendre_tables(
    degree: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cached access to Legendre recurrence constants for a given degree.

    Returned arrays should be treated as read-only.
    """
    N = int(degree)
    if N < 0:
        raise ValueError(f"degree must be >= 0. Got {degree}.")

    try:
        return _LEGENDRE_CACHE[N]
    except KeyError:
        tables = build_legendre_coeffs(N)
        # Reduce accidental mutation bugs (best-effort)
        for a in tables:
            try:
                a.flags.writeable = False
            except Exception:
                pass
        _LEGENDRE_CACHE[N] = tables
        return tables



# =============================================================================
# 2.                           THE CORE KERNELS
# =============================================================================

LOG_UNDERFLOW_LIMIT = -690.7755278982137  # ln(1e-300)


@njit(cache=True)
def _imin(a: int, b: int) -> int:
    return a if a < b else b


@njit(cache=True)
def _imax(a: int, b: int) -> int:
    return a if a > b else b


@njit(cache=True)
def _compute_stable_m_limit(cos_phi: float, max_degree: int) -> int:
    """
    Computes the maximum order 'm' where cos(phi)^m stays above float64 floor.
    Used to skip underflowing sectoral terms near the poles for performance.
    """
    if max_degree <= 0:
        return 0

    # abs_cos_phi represents the magnitude of the cosine of geocentric latitude
    abs_cos_phi = abs(cos_phi)
    
    # Handle singularities: exact pole (cos=0) or equator (cos=1)
    if abs_cos_phi == 0.0:
        return 0
    if abs_cos_phi >= NEARLY_UNIT:
        return max_degree

    # Mathematical justification:
    # We require: cos(phi)^m > exp(LOG_UNDERFLOW_LIMIT)
    # m * ln|cos_phi| > LOG_UNDERFLOW_LIMIT
    # m < LOG_UNDERFLOW_LIMIT / ln|cos_phi|
    
    ln_cos_phi = math.log(abs_cos_phi)
    m_suggested = int(LOG_UNDERFLOW_LIMIT / ln_cos_phi)

    # Return the stable order limit within [0, max_degree]
    return int(clamp(m_suggested, 0, max_degree))


@njit(cache=True)
def _fill_longitude_trig_tables(
    cos_lon: float, 
    sin_lon: float, 
    max_order: int,
    cos_m_lon: np.ndarray, 
    sin_m_lon: np.ndarray
) -> None:
    """
    Computes cos(m*lambda) and sin(m*lambda) tables using trigonometric addition formulas.
    This avoids repeated expensive calls to transcendental functions in the hot loop.
    """
    # Base case: m = 0
    cos_m_lon[0] = 1.0
    sin_m_lon[0] = 0.0
    
    if max_order <= 0:
        return

    # Base case: m = 1
    cos_m_lon[1] = cos_lon
    sin_m_lon[1] = sin_lon

    # Recursion using:
    # cos(m*L) = cos((m-1)*L)*cos(L) - sin((m-1)*L)*sin(L)
    # sin(m*L) = sin((m-1)*L)*cos(L) + cos((m-1)*L)*sin(L)
    for m in range(2, max_order + 1):
        cos_prev = cos_m_lon[m - 1]
        sin_prev = sin_m_lon[m - 1]
        
        cos_m_lon[m] = cos_prev * cos_lon - sin_prev * sin_lon
        sin_m_lon[m] = sin_prev * cos_lon + cos_prev * sin_lon


@njit(cache=True)
def _apply_legendre_normalization(
    p_matrix: np.ndarray, 
    dp_matrix: np.ndarray,
    max_degree: int, 
    max_order: int, 
    stable_m_limit: int,
    scale_m_table: np.ndarray
) -> None:
    """
    Applies per-order scaling (sqrt(2) and Condon–Shortley phase) in-place.
    
    This final step adjusts the ALFs and their derivatives to match the 
    fully-normalized geodesy convention.
    """
    for n in range(max_degree + 1):
        # Accessing rows directly for better cache locality in Numba
        p_row = p_matrix[n]
        dp_row = dp_matrix[n]

        # 1. Scale Legendre Polynomials (P)
        # Limit by the smaller of current degree 'n' or model order
        p_limit = _imin(max_order, n)
        for m in range(p_limit + 1):
            p_row[m] *= scale_m_table[m]

        # 2. Scale Derivatives (dP)
        # Limit by the stable 'm' cutoff to avoid unnecessary scaling of underflown terms
        dp_limit = _imin(stable_m_limit, n)
        for m in range(dp_limit + 1):
            dp_row[m] *= scale_m_table[m]


@njit(cache=True)
def _compute_legendre_polynomials_inplace(
    sin_phi: float, 
    cos_phi: float,
    max_degree: int, 
    max_order: int, 
    stable_m_limit: int,
    diag_coeffs: np.ndarray, 
    subdiag_coeffs: np.ndarray,
    A_coeffs: np.ndarray, 
    B_coeffs: np.ndarray, 
    scale_m_table: np.ndarray,
    p_matrix: np.ndarray, 
    dp_matrix: np.ndarray
) -> None:
    """
    Computes fully-normalized associated Legendre polynomials (ALFs) and their 
    derivatives dP/dphi using stable recursive algorithms.
    
    This is the core numerical hot-path. It performs zero heap allocations.
    """

    # 1. Coordinate and Range Setup
    workspace_n_max = p_matrix.shape[0] - 1
    max_degree = int(clamp(max_degree, 0, workspace_n_max))

    # Derivative dP[n,m] requires P[n,m+1]. 
    # We must ensure max_order is sufficient for derivative calculation.
    safe_order_for_deriv = _imin(stable_m_limit + 1, max_degree)
    max_order = _imax(max_order, safe_order_for_deriv)

    # 2. Base Case: n=0, m=0
    p_matrix[0, 0] = 1.0
    dp_matrix[0, 0] = 0.0

    # 3. Main Recurrence Loop (Degree-by-Degree)
    for n in range(1, max_degree + 1):
        p_curr = p_matrix[n]
        p_prev = p_matrix[n - 1]
        
        # Sectoral terms (Diagonal): P[n,n]
        if n <= max_order:
            p_curr[n] = diag_coeffs[n] * cos_phi * p_prev[n - 1]

        # Tesseral terms (Sub-diagonal): P[n,n-1]
        if (n - 1) <= max_order:
            p_curr[n - 1] = subdiag_coeffs[n] * sin_phi * p_prev[n - 1]

        # Vertical recursion: P[n,m] for m = 0..n-2
        if n >= 2:
            p_prev2 = p_matrix[n - 2]
            vertical_limit = _imin(max_order, n - 2)
            for m in range(vertical_limit + 1):
                p_curr[m] = A_coeffs[n, m] * sin_phi * p_prev[m] - B_coeffs[n, m] * p_prev2[m]

        # 4. Derivative Computation (dP/dphi)
        dp_curr = dp_matrix[n]

        # Case m = 0: dP[n,0] = sqrt(n*(n+1)) * P[n,1]
        if max_order >= 1:
            dp_curr[0] = math.sqrt(n * (n + 1.0)) * p_curr[1]
        else:
            dp_curr[0] = 0.0

        # Case m > 0: Standard derivative recurrence
        derivative_limit = _imin(stable_m_limit, n)
        for m in range(1, derivative_limit + 1):
            # Term involving P[n, m-1]
            coeff_minus = math.sqrt((n + m) * (n - m + 1.0))
            term_minus = coeff_minus * p_curr[m - 1]

            # Term involving P[n, m+1]
            m_plus_1 = m + 1
            if m_plus_1 <= n and m_plus_1 <= max_order:
                coeff_plus = math.sqrt((n - m) * (n + m + 1.0))
                term_plus = coeff_plus * p_curr[m_plus_1]
            else:
                term_plus = 0.0

            dp_curr[m] = 0.5 * (term_plus - term_minus)

    # 5. Final Normalization Scaling
    # Apply sqrt(2) and phase factor to bring ALFs to the geodesy convention.
    _apply_legendre_normalization(
        p_matrix, dp_matrix, 
        max_degree, max_order, stable_m_limit, 
        scale_m_table
    )



# =============================================================================
# 3.               ACCELERATION FUNCTIONS (SERIAL / PARALLEL)
# =============================================================================

@njit(cache=True)
def _determine_effective_degree(c_coeffs: np.ndarray, s_coeffs: np.ndarray) -> int:
    """
    Determines the maximum safe degree (N) based on the coefficient array dimensions.
    Supports rectangular inputs by selecting the most conservative bound.
    """
    # n_rows corresponds to Degree (N), m_cols corresponds to Order (M)
    n_rows_c, m_cols_c = c_coeffs.shape
    n_rows_s, m_cols_s = s_coeffs.shape

    # Max usable index is (size - 1) for 0-based indexing.
    # We find the global minimum across all four dimensions to stay within bounds.
    min_dim = _imin(_imin(n_rows_c, m_cols_c), 
                    _imin(n_rows_s, m_cols_s))
    
    max_safe_degree = min_dim - 1

    return _imax(0, max_safe_degree)


@njit(cache=True)
def _transform_cartesian_to_spherical_basis(x: float, y: float, z: float):
    """
    Transforms Cartesian (x, y, z) to Spherical components and calculates 
    the radial (u_r) and meridional (u_phi) basis vectors.
    """
    # Radial distance calculation
    rho_sq = x*x + y*y     # Planar distance squared (x-y plane)
    r_sq = rho_sq + z*z    # Total distance squared
    r = math.sqrt(r_sq)

    # Collision/Center Guard: Avoid division by zero near the origin
    if r <= 1.0:
        return (False, 
                0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 1.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Precompute inverses for performance
    inv_r = 1.0 / r
    inv_r_sq = inv_r * inv_r

    rho = math.sqrt(rho_sq)
    
    # Latitude components (sin_phi is z-component, cos_phi is planar component)
    sin_phi = z * inv_r
    cos_phi = rho * inv_r

    # Longitude components with Pole-Safety
    if rho > EPS_1E12:
        inv_rho = 1.0 / rho
        cos_lon = x * inv_rho
        sin_lon = y * inv_rho
    else:
        # At the exact pole, longitude is undefined; default to 0 degrees
        cos_lon = 1.0
        sin_lon = 0.0

    # u_r: Radial Unit Vector (points away from center)
    u_r_x = x * inv_r
    u_r_y = y * inv_r
    u_r_z = z * inv_r

    # u_phi: Meridional Unit Vector (points North along the meridian)
    # Formula: [-sin(phi)cos(lambda), -sin(phi)sin(lambda), cos(phi)]
    u_phi_x = -sin_phi * cos_lon
    u_phi_y = -sin_phi * sin_lon
    u_phi_z = cos_phi

    return (True,
            r, inv_r, inv_r_sq, rho_sq,
            sin_phi, cos_phi,
            cos_lon, sin_lon,
            u_r_x, u_r_y, u_r_z,
            u_phi_x, u_phi_y, u_phi_z)


@njit(cache=True)
def _compute_pole_safe_inv_rho_sq(rho_sq: float, r: float) -> float:
    """
    Computes 1/rho^2 safely near the planetary poles.
    
    Uses a relative softening factor (epsilon) based on radial distance.
    If the position is too close to the pole, it nullifies the term to prevent 
    numerical spikes in the longitudinal acceleration.
    """
    
    # If we are effectively at the pole, return 0 to bypass longitudinal forces
    if rho_sq < EPS_1E24:
        return 0.0
    
    # Standard inverse with a small softening constant to ensure stability
    return 1.0 / (rho_sq + EPS_1E24)


@njit(cache=True)
def _convert_spherical_gradients_to_cartesian(
    x: float, y: float,
    r: float, inv_r: float, rho_sq: float,
    dv_dr: float, dv_dphi: float, dv_dlambda: float,
    u_r_x: float, u_r_y: float, u_r_z: float,
    u_phi_x: float, u_phi_y: float, u_phi_z: float
) -> Tuple[float, float, float]:
    """
    Converts spherical potential gradients (dV/dr, dV/dphi, dV/dlambda) 
    into Cartesian acceleration components (ax, ay, az).
    """
    # 1. Radial and Meridional scaling
    # The meridional (phi) gradient is scaled by 1/r
    phi_factor = dv_dphi * inv_r
    
    # 2. Longitudinal scaling (The most sensitive part near poles)
    # The term (1 / rho^2) is used to project the dV/dlambda component
    inv_rho_sq = _compute_pole_safe_inv_rho_sq(rho_sq, r)

    # 3. Vector Recomposition
    # Radial component + Meridional component + Longitudinal component
    ax = dv_dr * u_r_x + phi_factor * u_phi_x - dv_dlambda * y * inv_rho_sq
    ay = dv_dr * u_r_y + phi_factor * u_phi_y + dv_dlambda * x * inv_rho_sq
    az = dv_dr * u_r_z + phi_factor * u_phi_z
    
    return ax, ay, az


@njit(cache=True)
def _kahan_sum_step(running_sum: float, error_compensation: float, value: float):
    """
    Performs one step of Kahan Summation to minimize floating-point errors.
    
    This is critical for high-degree models where thousands of small 
    harmonic terms are accumulated.
    """
    # 1. Adjust the current value by subtracting the previously accumulated error
    # (Note: error_compensation is subtracted because it's effectively a 'loss')
    corrected_value = value - error_compensation
    
    # 2. Add the corrected value to the running total. 
    # High-order bits are stored in 'new_sum'.
    new_sum = running_sum + corrected_value
    
    # 3. Calculate the new error compensation.
    # (new_sum - running_sum) gives the actual increase seen by the large number.
    # Subtracting corrected_value from this recovery gives the lost low-order bits.
    new_error = (new_sum - running_sum) - corrected_value
    
    return new_sum, new_error


@njit(cache=True)
def _prepare_evaluation_tables(
    sin_phi: float, cos_phi: float,
    cos_lon: float, sin_lon: float,
    max_degree: int,
    diag_coeffs: np.ndarray, subdiag_coeffs: np.ndarray,
    A_coeffs: np.ndarray, B_coeffs: np.ndarray, 
    scale_m_table: np.ndarray,
    p_matrix: np.ndarray, dp_matrix: np.ndarray,
    cos_m_table: np.ndarray, sin_m_table: np.ndarray
) -> int:
    """
    Coordinates the computation of all reusable tables (Legendre and Trigonometric).
    
    1. Computes the stability limit (m_cutoff) for the current latitude.
    2. Fills Legendre polynomials (P) and their derivatives (dP) in-place.
    3. Fills the longitude recurrence tables (cos/sin m*lambda) in-place.
    """
    
    # 1. Determine the stable order limit for this latitude
    stable_m_limit = _compute_stable_m_limit(cos_phi, max_degree)

    # 2. Derivative dP[n,m] needs P[n,m+1]. 
    # We ensure P is computed up to (m_limit + 1) to avoid index errors or zeroed derivatives.
    max_required_order = _imin(stable_m_limit + 1, max_degree)
    
    # 3. Compute Legendre Polynomials (In-place)
    _compute_legendre_polynomials_inplace(
        sin_phi, cos_phi, max_degree, max_required_order, stable_m_limit,
        diag_coeffs, subdiag_coeffs, A_coeffs, B_coeffs, scale_m_table, 
        p_matrix, dp_matrix
    )

    # 4. Compute Trigonometric Longitude Tables (In-place)
    _fill_longitude_trig_tables(
        cos_lon, sin_lon, stable_m_limit, 
        cos_m_table, sin_m_table
    )

    return stable_m_limit


@njit(cache=True)
def _prepare_simulation_preamble(
    x: float, y: float, z: float,
    requested_degree: int,
    c_coeffs: np.ndarray, s_coeffs: np.ndarray
):
    """
    Common setup for gravity evaluation. 
    Validates position, transforms coordinates, and determines safe evaluation degree.

    Returns a flat tuple of all geometric and administrative constants.
    """
    # 1. Coordinate transformation and basis vector calculation
    # Refactored call to our previously optimized basis function
    (valid_position, r, inv_r, inv_r_sq, rho_sq, 
     sin_phi, cos_phi, cos_lon, sin_lon,
     u_r_x, u_r_y, u_r_z, 
     u_phi_x, u_phi_y, u_phi_z) = _transform_cartesian_to_spherical_basis(x, y, z)

    # 2. Safety Guard: If position is invalid (e.g., too close to center)
    if not valid_position:
        return (False,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0)

    # 3. Determine the maximum degree supported by the data vs requested by the user
    # Refactored call to our optimized degree checker
    available_n_max = _determine_effective_degree(c_coeffs, s_coeffs)
    effective_eval_degree = int(clamp(requested_degree, 0, available_n_max))

    return (True,
            r, inv_r, inv_r_sq, rho_sq,
            sin_phi, cos_phi, cos_lon, sin_lon,
            u_r_x, u_r_y, u_r_z, 
            u_phi_x, u_phi_y, u_phi_z,
            effective_eval_degree)

# ------------------------ SERIAL (accurate, Kahan) ------------------------

@njit(cache=True)
def _compute_sh_acceleration_serial(
    x: float, y: float, z: float,
    requested_degree: int,
    r_ref: float, mu: float,
    c_coeffs: np.ndarray, s_coeffs: np.ndarray,
    diag_coeffs: np.ndarray, subdiag_coeffs: np.ndarray,
    a_coeffs: np.ndarray, b_coeffs: np.ndarray, 
    scale_m_table: np.ndarray,
    p_matrix: np.ndarray, dp_matrix: np.ndarray,
    cos_m_table: np.ndarray, sin_m_table: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes the gravitational acceleration vector using a high-degree 
    Spherical Harmonics model. Optimized for serial execution with 
    Kahan summation for double-precision stability.
    """

    # 1. Preamble: Coordinate transformation and degree validation
    (is_valid, r, inv_r, inv_r_sq, rho_sq, 
     sin_phi, cos_phi, cos_lon, sin_lon,
     u_r_x, u_r_y, u_r_z, 
     u_phi_x, u_phi_y, u_phi_z, 
     eval_degree) = _prepare_simulation_preamble(x, y, z, requested_degree, c_coeffs, s_coeffs)

    if not is_valid:
        return 0.0, 0.0, 0.0

    # 2. Initialize Potentials and Gradients
    # Central gravity term: dV/dr = -GM/r^2
    dv_dr = -mu * inv_r_sq
    dv_dphi = 0.0
    dv_dlambda = 0.0

    # Global error compensators for Kahan summation
    err_dr, err_dphi, err_dlambda = 0.0, 0.0, 0.0

    # 3. Non-Spherical Terms (Perturbations)
    if eval_degree >= 2:
        # Precompute Legendre and Trigonometric lookup tables
        m_cutoff = _prepare_evaluation_tables(
            sin_phi, cos_phi, cos_lon, sin_lon, eval_degree,
            diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
            p_matrix, dp_matrix, cos_m_table, sin_m_table
        )

        r_ratio_base = r_ref * inv_r
        # r_ratio starts at (R_ref/r)^2 for degree n=2
        r_ratio_n = r_ratio_base * r_ratio_base 
        mu_inv_r = mu * inv_r
        mu_inv_r_sq = mu * inv_r_sq

        # Outer Loop: Summation by Degree (n)
        for n in range(2, eval_degree + 1):
            m_limit = _imin(m_cutoff, n)

            # Row access for current degree
            c_row, s_row = c_coeffs[n], s_coeffs[n]
            p_row, dp_row = p_matrix[n], dp_matrix[n]

            # Per-degree partial sums and their Kahan compensators
            s_r, k_r = 0.0, 0.0
            s_p, k_p = 0.0, 0.0
            s_l, k_l = 0.0, 0.0

            # Inner Loop: Summation by Order (m)
            for m in range(m_limit + 1):
                cos_m_lon = cos_m_table[m]
                sin_m_lon = sin_m_table[m]

                # Longitudinal terms: C*cos(mL) + S*sin(mL)
                term_lon = c_row[m] * cos_m_lon + s_row[m] * sin_m_lon
                deriv_lon = -c_row[m] * sin_m_lon + s_row[m] * cos_m_lon

                p_nm = p_row[m]
                dp_nm = dp_row[m]

                # Accumulate partial sums for each gradient component
                s_r, k_r = _kahan_sum_step(s_r, k_r, p_nm * term_lon)
                s_p, k_p = _kahan_sum_step(s_p, k_p, dp_nm * term_lon)
                s_l, k_l = _kahan_sum_step(s_l, k_l, (m * p_nm) * deriv_lon)

            # Apply common factors for the current degree 'n'
            delta_dr = -mu_inv_r_sq * (n + 1.0) * r_ratio_n * s_r
            delta_dp =  mu_inv_r * r_ratio_n * s_p
            delta_dl =  mu_inv_r * r_ratio_n * s_l

            # Update global gradients using global Kahan compensators
            dv_dr, err_dr = _kahan_sum_step(dv_dr, err_dr, delta_dr)
            dv_dphi, err_dphi = _kahan_sum_step(dv_dphi, err_dphi, delta_dp)
            dv_dlambda, err_dlambda = _kahan_sum_step(dv_dlambda, err_dlambda, delta_dl)

            # Advance (R_ref/r)^n for the next degree
            r_ratio_n *= r_ratio_base

    # 4. Final Conversion to Cartesian Acceleration
    return _convert_spherical_gradients_to_cartesian(
        x, y, r, inv_r, rho_sq,
        dv_dr, dv_dphi, dv_dlambda,
        u_r_x, u_r_y, u_r_z,
        u_phi_x, u_phi_y, u_phi_z
    )


# ------------------------ PARALLEL (race-free reduction) ------------------------

@njit(cache=True)
def _get_optimal_block_size(max_degree: int) -> int:
    """Heuristic to determine how many degrees each thread should process."""
    return 16 if max_degree > 256 else 8


@njit(parallel=True, fastmath=True, cache=True)
def _compute_sh_acceleration_parallel(
    x: float, y: float, z: float,
    requested_degree: int,
    r_ref: float, mu: float,
    c_coeffs: np.ndarray, s_coeffs: np.ndarray,
    diag_coeffs: np.ndarray, subdiag_coeffs: np.ndarray,
    a_coeffs: np.ndarray, b_coeffs: np.ndarray, 
    scale_m_table: np.ndarray,
    p_matrix: np.ndarray, dp_matrix: np.ndarray,
    cos_m_table: np.ndarray, sin_m_table: np.ndarray
) -> Tuple[float, float, float]:
    """
    Parallel implementation of SH acceleration. 
    Distributes degree-blocks across multiple CPU cores using a race-free 
    partial-sum reduction strategy.
    """

    # 1. Preamble and Setup
    (is_valid, r, inv_r, inv_r_sq, rho_sq, 
     sin_phi, cos_phi, cos_lon, sin_lon,
     u_r_x, u_r_y, u_r_z, 
     u_phi_x, u_phi_y, u_phi_z, 
     eval_degree) = _prepare_simulation_preamble(x, y, z, requested_degree, c_coeffs, s_coeffs)

    if not is_valid:
        return 0.0, 0.0, 0.0

    # Initialize global gradients (central body term)
    dv_dr = -mu * inv_r_sq
    dv_dphi = 0.0
    dv_dlambda = 0.0

    if eval_degree >= 2:
        # Precompute shared tables (Legendre/Trig) - Note: This part is serial!
        m_cutoff = _prepare_evaluation_tables(
            sin_phi, cos_phi, cos_lon, sin_lon, eval_degree,
            diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
            p_matrix, dp_matrix, cos_m_table, sin_m_table
        )

        r_ratio_base = r_ref * inv_r
        mu_inv_r = mu * inv_r
        mu_inv_r_sq = mu * inv_r_sq

        # 2. Parallel Partitioning
        block_size = _get_optimal_block_size(eval_degree)
        n_start, n_stop = 2, eval_degree + 1
        num_blocks = (n_stop - n_start + block_size - 1) // block_size

        # Race-free partial sum buffers
        partial_dr = np.zeros(num_blocks, dtype=F64)
        partial_dp = np.zeros(num_blocks, dtype=F64)
        partial_dl = np.zeros(num_blocks, dtype=F64)

        # 3. Parallel Work Loop
        for block_idx in prange(num_blocks):
            deg_0 = n_start + block_idx * block_size
            deg_1 = _imin(deg_0 + block_size, n_stop)

            # Calculate (R/r)^n for the start of this specific block
            r_ratio_n = math.pow(r_ratio_base, deg_0)

            # Local accumulators for this thread's block
            block_sum_dr = 0.0
            block_sum_dp = 0.0
            block_sum_dl = 0.0

            for n in range(deg_0, deg_1):
                m_limit = _imin(m_cutoff, n)
                
                c_row, s_row = c_coeffs[n], s_coeffs[n]
                p_row, dp_row = p_matrix[n], dp_matrix[n]

                s_r, s_p, s_l = 0.0, 0.0, 0.0

                # Inner Order Summation
                for m in range(m_limit + 1):
                    cos_ml = cos_m_table[m]
                    sin_ml = sin_m_table[m]

                    term_lon = c_row[m] * cos_ml + s_row[m] * sin_ml
                    deriv_lon = -c_row[m] * sin_ml + s_row[m] * cos_ml

                    p_nm = p_row[m]
                    s_r += p_nm * term_lon
                    s_p += dp_row[m] * term_lon
                    s_l += (m * p_nm) * deriv_lon

                # Accumulate local gradients
                block_sum_dr += -mu_inv_r_sq * (n + 1.0) * r_ratio_n * s_r
                block_sum_dp +=  mu_inv_r * r_ratio_n * s_p
                block_sum_dl +=  mu_inv_r * r_ratio_n * s_l

                r_ratio_n *= r_ratio_base

            # Store block results in shared buffers
            partial_dr[block_idx] = block_sum_dr
            partial_dp[block_idx] = block_sum_dp
            partial_dl[block_idx] = block_sum_dl

        # 4. Final Reduction (Deterministic Serial Combine)
        for i in range(num_blocks):
            dv_dr += partial_dr[i]
            dv_dphi += partial_dp[i]
            dv_dlambda += partial_dl[i]

    # 5. Transform to Cartesian
    return _convert_spherical_gradients_to_cartesian(
        x, y, r, inv_r, rho_sq,
        dv_dr, dv_dphi, dv_dlambda,
        u_r_x, u_r_y, u_r_z,
        u_phi_x, u_phi_y, u_phi_z
    )



# =============================================================================
# 4.           HIGH-LEVEL ACCELERATION WRAPPERS (FIXED & ADAPTIVE)
# =============================================================================

SERIAL_PARALLEL_THRESHOLD = 80  # degrees; heuristic

# ------------------------ Interpolation Helpers ------------------------

@njit(cache=True)
def _apply_smoothstep(t: float) -> float:
    """
    Applies Cubic Hermite interpolation. 
    Expects t in [0, 1]. Clamps internally to ensure stability.
    """
    # Mevcut clamp fonksiyonunu kullanarak t'yi [0, 1] arasına hapsediyoruz.
    t_clamped = clamp(t, 0.0, 1.0)
    
    # S-Curve (Smoothstep) formülü
    return t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)


# ------------------------ dispatch wrapper ------------------------

def sh_accel_fixed(
    x: float, y: float, z: float,
    degree: int,
    r_ref: float, mu: float,
    c_coeffs: np.ndarray, s_coeffs: np.ndarray,
    diag_coeffs: np.ndarray, subdiag_coeffs: np.ndarray,
    a_coeffs: np.ndarray, b_coeffs: np.ndarray, 
    scale_m_table: np.ndarray,
    p_matrix: np.ndarray, dp_matrix: np.ndarray,
    cos_m_table: np.ndarray, sin_m_table: np.ndarray,
    *,
    use_parallel: bool = False,
    parallel_threshold: int = SERIAL_PARALLEL_THRESHOLD,
) -> Tuple[float, float, float]:
    """
    Fixed-degree acceleration with explicit caller-controlled dispatch.
    
    Chooses between the high-precision Serial kernel (Kahan) and the 
    high-speed Parallel kernel (Fastmath) based on the degree threshold.
    """
    # 1. Determine safe evaluation degree
    max_safe_n = _determine_effective_degree(c_coeffs, s_coeffs)
    eval_degree = int(clamp(degree, 0, max_safe_n))

    # 2. Decision Logic: Parallel vs Serial
    # Parallel is only triggered if requested AND degree is high enough.
    if use_parallel and (eval_degree > int(parallel_threshold)):
        return _compute_sh_acceleration_parallel(
            x, y, z, eval_degree,
            r_ref, mu, c_coeffs, s_coeffs,
            diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
            p_matrix, dp_matrix, cos_m_table, sin_m_table
        )

    # Default to high-precision serial kernel
    return _compute_sh_acceleration_serial(
        x, y, z, eval_degree,
        r_ref, mu, c_coeffs, s_coeffs,
        diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
        p_matrix, dp_matrix, cos_m_table, sin_m_table
    )


@njit(cache=True)
def sh_accel_fixed_numba(
    x: float, y: float, z: float,
    degree: int,
    r_ref: float, mu: float,
    c_coeffs: np.ndarray, s_coeffs: np.ndarray,
    diag_coeffs: np.ndarray, subdiag_coeffs: np.ndarray,
    a_coeffs: np.ndarray, b_coeffs: np.ndarray, 
    scale_m_table: np.ndarray,
    p_matrix: np.ndarray, dp_matrix: np.ndarray,
    cos_m_table: np.ndarray, sin_m_table: np.ndarray
) -> Tuple[float, float, float]:
    """
    Jit-compatible fixed-degree wrapper. Always uses the serial kernel 
    to ensure deterministic results when called from within other Numba kernels.
    """
    # Quick limit check
    max_safe_n = _determine_effective_degree(c_coeffs, s_coeffs)
    eval_degree = int(clamp(degree, 0, max_safe_n))

    return _compute_sh_acceleration_serial(
        x, y, z, eval_degree,
        r_ref, mu, c_coeffs, s_coeffs,
        diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
        p_matrix, dp_matrix, cos_m_table, sin_m_table
    )


# ------------------------ dual-degree in one pass ------------------------

@njit(cache=True)
def _compute_sh_acceleration_dual_numba(
    x: float, y: float, z: float,
    degree_low: int, degree_high: int,
    r_ref: float, mu: float,
    c_coeffs: np.ndarray, s_coeffs: np.ndarray,
    diag_coeffs: np.ndarray, subdiag_coeffs: np.ndarray,
    a_coeffs: np.ndarray, b_coeffs: np.ndarray, 
    scale_m_table: np.ndarray,
    p_matrix: np.ndarray, dp_matrix: np.ndarray,
    cos_m_table: np.ndarray, sin_m_table: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """
    Computes two different gravity solutions (Low-Degree and High-Degree) 
    simultaneously in a single pass. 
    
    Optimized for blending/interpolation between different model resolutions 
    without redundant Legendre or Trigonometric calculations.
    """

    # 1. Preamble (Calculates coordinates and validates the highest degree)
    (is_valid, r, inv_r, inv_r_sq, rho_sq, 
     sin_phi, cos_phi, cos_lon, sin_lon,
     u_r_x, u_r_y, u_r_z, 
     u_phi_x, u_phi_y, u_phi_z, 
     max_safe_degree) = _prepare_simulation_preamble(x, y, z, degree_high, c_coeffs, s_coeffs)

    if not is_valid:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Ensure degrees are within safe limits and lo <= hi
    n_lo = int(clamp(degree_low, 0, max_safe_degree))
    n_hi = int(clamp(degree_high, n_lo, max_safe_degree))

    # 2. Initialize Two Sets of Potential Gradients
    # Low-degree accumulators
    dv_dr_lo, dv_dp_lo, dv_dl_lo = -mu * inv_r_sq, 0.0, 0.0
    err_dr_lo, err_dp_lo, err_dl_lo = 0.0, 0.0, 0.0

    # High-degree accumulators
    dv_dr_hi, dv_dp_hi, dv_dl_hi = -mu * inv_r_sq, 0.0, 0.0
    err_dr_hi, err_dp_hi, err_dl_hi = 0.0, 0.0, 0.0

    if n_hi >= 2:
        # Precompute tables for the HIGHER degree (covers both)
        m_cutoff = _prepare_evaluation_tables(
            sin_phi, cos_phi, cos_lon, sin_lon, n_hi,
            diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
            p_matrix, dp_matrix, cos_m_table, sin_m_table
        )

        r_ratio_base = r_ref * inv_r
        r_ratio_n = r_ratio_base * r_ratio_base
        mu_inv_r = mu * inv_r
        mu_inv_r_sq = mu * inv_r_sq

        # 3. Main Loop: Unified Summation
        for n in range(2, n_hi + 1):
            m_limit = _imin(m_cutoff, n)
            
            c_row, s_row = c_coeffs[n], s_coeffs[n]
            p_row, dp_row = p_matrix[n], dp_matrix[n]

            s_r, k_r = 0.0, 0.0
            s_p, k_p = 0.0, 0.0
            s_l, k_l = 0.0, 0.0

            # Compute harmonic terms for degree 'n'
            for m in range(m_limit + 1):
                cos_ml = cos_m_table[m]
                sin_ml = sin_m_table[m]

                term_lon = c_row[m] * cos_ml + s_row[m] * sin_ml
                deriv_lon = -c_row[m] * sin_ml + s_row[m] * cos_ml

                p_nm = p_row[m]
                s_r, k_r = _kahan_sum_step(s_r, k_r, p_nm * term_lon)
                s_p, k_p = _kahan_sum_step(s_p, k_p, dp_row[m] * term_lon)
                s_l, k_l = _kahan_sum_step(s_l, k_l, (m * p_nm) * deriv_lon)

            # Calculate physical delta for this degree
            delta_dr = -mu_inv_r_sq * (n + 1.0) * r_ratio_n * s_r
            delta_dp =  mu_inv_r * r_ratio_n * s_p
            delta_dl =  mu_inv_r * r_ratio_n * s_l

            # UPDATE HIGH: Always added
            dv_dr_hi, err_dr_hi = _kahan_sum_step(dv_dr_hi, err_dr_hi, delta_dr)
            dv_dp_hi, err_dp_hi = _kahan_sum_step(dv_dp_hi, err_dp_hi, delta_dp)
            dv_dl_hi, err_dl_hi = _kahan_sum_step(dv_dl_hi, err_dl_hi, delta_dl)

            # UPDATE LOW: Only added if degree n is within n_lo range
            if n <= n_lo:
                dv_dr_lo, err_dr_lo = _kahan_sum_step(dv_dr_lo, err_dr_lo, delta_dr)
                dv_dp_lo, err_dp_lo = _kahan_sum_step(dv_dp_lo, err_dp_lo, delta_dp)
                dv_dl_lo, err_dl_lo = _kahan_sum_step(dv_dl_lo, err_dl_lo, delta_dl)

            r_ratio_n *= r_ratio_base

    # 4. Final Projection to Cartesian (Two independent results)
    ax_lo, ay_lo, az_lo = _convert_spherical_gradients_to_cartesian(
        x, y, r, inv_r, rho_sq, dv_dr_lo, dv_dp_lo, dv_dl_lo,
        u_r_x, u_r_y, u_r_z, u_phi_x, u_phi_y, u_phi_z
    )
    ax_hi, ay_hi, az_hi = _convert_spherical_gradients_to_cartesian(
        x, y, r, inv_r, rho_sq, dv_dr_hi, dv_dp_hi, dv_dl_hi,
        u_r_x, u_r_y, u_r_z, u_phi_x, u_phi_y, u_phi_z
    )

    return ax_lo, ay_lo, az_lo, ax_hi, ay_hi, az_hi

# ------------------------ adaptive blend ------------------------

@njit(cache=True)
def sh_accel_adaptive_blend_numba(
    x: float, y: float, z: float,
    degree_far: int, degree_near: int,
    alt_limit_far: float, alt_limit_near: float,
    degree_step: int,
    r_ref: float, mu: float,
    c_coeffs: np.ndarray, s_coeffs: np.ndarray,
    diag_coeffs: np.ndarray, subdiag_coeffs: np.ndarray,
    a_coeffs: np.ndarray, b_coeffs: np.ndarray, 
    scale_m_table: np.ndarray,
    p_matrix: np.ndarray, dp_matrix: np.ndarray,
    cos_m_table: np.ndarray, sin_m_table: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes gravity acceleration with a dynamic degree that scales with altitude.
    Uses 'Dual-Fidelity' evaluation to blend between two discrete degrees for 
    numerical continuity (C1-smoothness).
    """

    # 1. Degree Limits & Step Validation
    max_safe_n = _determine_effective_degree(c_coeffs, s_coeffs)
    n_min = int(clamp(degree_far, 0, max_safe_n))
    n_max = int(clamp(degree_near, n_min, max_safe_n))
    
    step = _imax(1, int(degree_step))

    # 2. Geometry: Distance and Altitude
    r = math.sqrt(x*x + y*y + z*z)
    altitude = r - r_ref

    # 3. Transition Logic (Determine Blending Factor 's')
    # If altitude limits are invalid, fallback to discrete selection
    if alt_limit_far <= alt_limit_near:
        target_degree = n_max if altitude <= alt_limit_near else n_min
        return sh_accel_fixed_numba(
            x, y, z, target_degree, r_ref, mu, c_coeffs, s_coeffs,
            diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
            p_matrix, dp_matrix, cos_m_table, sin_m_table
        )

    # Handle cases outside the transition band
    if altitude >= alt_limit_far:
        return sh_accel_fixed_numba(
            x, y, z, n_min, r_ref, mu, c_coeffs, s_coeffs,
            diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
            p_matrix, dp_matrix, cos_m_table, sin_m_table
        )
    if altitude <= alt_limit_near:
        return sh_accel_fixed_numba(
            x, y, z, n_max, r_ref, mu, c_coeffs, s_coeffs,
            diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
            p_matrix, dp_matrix, cos_m_table, sin_m_table
        )

    # 4. Smoothstep Interpolation
    # t moves from 0 (at far limit) to 1 (at near limit)
    t = (alt_limit_far - altitude) / (alt_limit_far - alt_limit_near)
    s = _apply_smoothstep(t)

    # Desired floating-point degree (e.g., 42.7)
    desired_degree = n_min + s * (n_max - n_min)

    # 5. Discrete Ladder Selection (The 'Dual' steps)
    # Example: If desired is 42.7 and step is 10, we blend between degree 40 and 50.
    k = int(math.floor((desired_degree - n_min) / step))
    deg_lo = int(clamp(n_min + k * step, n_min, n_max))
    deg_hi = int(clamp(deg_lo + step, n_min, n_max))

    # If steps collapsed to the same value, no blending needed
    if deg_hi == deg_lo:
        return sh_accel_fixed_numba(
            x, y, z, deg_lo, r_ref, mu, c_coeffs, s_coeffs,
            diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
            p_matrix, dp_matrix, cos_m_table, sin_m_table
        )

    # 6. Final Dual Evaluation and Linear Blend
    # 'w' is the weight between the two discrete degree steps
    weight = (desired_degree - deg_lo) / (deg_hi - deg_lo)
    weight = clamp(weight, 0.0, 1.0)

    # Execute both degrees in a single optimized pass
    (ax_lo, ay_lo, az_lo, 
     ax_hi, ay_hi, az_hi) = _compute_sh_acceleration_dual_numba(
        x, y, z, deg_lo, deg_hi, r_ref, mu, c_coeffs, s_coeffs,
        diag_coeffs, subdiag_coeffs, a_coeffs, b_coeffs, scale_m_table,
        p_matrix, dp_matrix, cos_m_table, sin_m_table
    )

    # Blend the results
    ax = (1.0 - weight) * ax_lo + weight * ax_hi
    ay = (1.0 - weight) * ay_lo + weight * ay_hi
    az = (1.0 - weight) * az_lo + weight * az_hi
    
    return ax, ay, az



# =============================================================================
# 5.                 BASIC POINT MASS GRAVITY (BASELINE)
# =============================================================================

@njit(cache=True)
def compute_point_mass_acceleration(
    x: float, y: float, z: float, 
    mu: float
) -> Tuple[float, float, float]:
    """
    Computes the standard Newtonian point-mass acceleration (monopole).
    
    This serves as the baseline gravity model, representing the planet 
    as a single point mass at the origin.
    """
    # Calculate squared distance to avoid early square root
    dist_sq = x*x + y*y + z*z
    
    # Numerical Safety Guard: Avoid singularity at the center (r -> 0)
    # Using a threshold (e.g., 1e-24 m^2) to prevent division by zero.
    if dist_sq <= EPS_1E24:
        return 0.0, 0.0, 0.0

    # Optimization: Calculate 1/r first, then cube it to minimize divisions.
    inv_dist = 1.0 / math.sqrt(dist_sq)
    inv_dist_cubed = inv_dist * inv_dist * inv_dist
    
    # Gravitational magnitude factor: -mu / r^3
    accel_factor = -mu * inv_dist_cubed
    
    # Return Cartesian components
    return accel_factor * x, accel_factor * y, accel_factor * z



# =============================================================================
# 6.                 HIGH-LEVEL WRAPPER: GRAVITY MODEL
# =============================================================================

Workspace: TypeAlias = "SHWorkspace"

@dataclass(frozen=True, slots=True)
class GravityModel:
    """
    High-level container for a spherical-harmonic gravity model.
    
    Coordinates the numerical kernels, manages precomputed recurrence tables, 
    and handles thread-local workspaces for orbital simulations.
    """

    max_degree: int
    r_ref: float  # Reference radius (m)
    mu: float     # Gravitational parameter (m^3/s^2)

    # Precomputed Kernel Data (Read-only arrays)
    c_coeffs: np.ndarray
    s_coeffs: np.ndarray
    diag_coeffs: np.ndarray
    subdiag_coeffs: np.ndarray
    a_coeffs: np.ndarray
    b_coeffs: np.ndarray
    scale_m_table: np.ndarray

    # Default workspace for single-threaded usage
    workspace: Workspace

    @property
    def degree_max(self) -> int:
        return self.max_degree

    @property
    def R_ref_m(self) -> float:
        return self.r_ref
        
    @property
    def GM_m3s2(self) -> float:
        return self.mu
        
    @property
    def Cnm(self) -> np.ndarray:
        return self.c_coeffs
        
    @property
    def Snm(self) -> np.ndarray:
        return self.s_coeffs
        
    @property
    def diag(self) -> np.ndarray:
        return self.diag_coeffs
        
    @property
    def subdiag(self) -> np.ndarray:
        return self.subdiag_coeffs
        
    @property
    def A(self) -> np.ndarray:
        return self.a_coeffs
        
    @property
    def B(self) -> np.ndarray:
        return self.b_coeffs
        
    @property
    def scale_m(self) -> np.ndarray:
        return self.scale_m_table

    # --------------------------- Internal Helpers ---------------------------

    @staticmethod
    def _ensure_finite(name: str, value: float) -> float:
        val = float(value)
        if not math.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite. Got {value}.")
        return val

    # --------------------------- Factories ---------------------------

    @classmethod
    def from_file(
        cls,
        path: str,
        requested_degree: Optional[int] = None,
    ) -> "GravityModel":
        """
        Factory: Loads coefficients from a file and prepares kernel arrays.
        Separation of concerns: parsing is handled by loaders.io_gravity.
        """
        from lunaris.loaders.io_gravity import load_gravity_model as _load_file

        # Load raw data via the external IO utility
        n_file, r_val, mu_val, c_raw, s_raw = _load_file(path, degree_max=requested_degree)

        return cls.from_arrays(
            degree_max=requested_degree if requested_degree is not None else n_file,
            r_ref=r_val,
            mu=mu_val,
            c_coeffs_full=c_raw,
            s_coeffs_full=s_raw
        )

    @classmethod
    def from_arrays(
        cls,
        degree_max: int,
        r_ref: float,
        mu: float,
        c_coeffs_full: np.ndarray,
        s_coeffs_full: np.ndarray,
    ) -> "GravityModel":
        """
        Factory: Build a model from in-memory arrays and precompute tables.
        """
        c_full = np.asarray(c_coeffs_full, dtype=np.float64)
        s_full = np.asarray(s_coeffs_full, dtype=np.float64)

        # 1. Determine safe bounds
        n_supported = _determine_effective_degree(c_full, s_full)
        final_degree = int(clamp(degree_max, 0, n_supported))

        # 2. Slice and precompute tables
        c_sliced, s_sliced = slice_gravity_model(c_full, s_full, final_degree)
        diag, subdiag, a, b, scale_m = _get_legendre_tables(final_degree)
        
        # 3. Allocation
        ws = make_sh_workspace(final_degree)

        return cls(
            max_degree=final_degree,
            r_ref=cls._ensure_finite("r_ref", r_ref),
            mu=cls._ensure_finite("mu", mu),
            c_coeffs=c_sliced,
            s_coeffs=s_sliced,
            diag_coeffs=diag,
            subdiag_coeffs=subdiag,
            a_coeffs=a,
            b_coeffs=b,
            scale_m_table=scale_m,
            workspace=ws
        )

    # --------------------------- Workspace Management ---------------------------

    def make_workspace(self) -> Workspace:
        """Allocates an independent workspace. Essential for multi-threaded solvers."""
        return make_sh_workspace(self.max_degree)

    def _resolve_ws(self, ws: Optional[Workspace]) -> Workspace:
        """Selects between the default workspace or a user-provided one."""
        return self.workspace if ws is None else ws

    # --------------------------- Acceleration APIs ---------------------------

    def accel_fixed(
        self,
        r_body_fixed: Vec3,
        degree: Optional[int] = None,
        workspace: Optional[Workspace] = None,
    ) -> np.ndarray:
        """
        Fixed-degree gravity acceleration in the Body-Fixed frame.
        """
        # Convert input to raw floats for Numba speed
        x = float(r_body_fixed[0])
        y = float(r_body_fixed[1])
        z = float(r_body_fixed[2])
        
        n_eval = int(clamp(degree if degree is not None else self.max_degree, 0, self.max_degree))
        ws = self._resolve_ws(workspace)

        ax, ay, az = sh_accel_fixed_numba(
            x, y, z, n_eval,
            self.r_ref, self.mu,
            self.c_coeffs, self.s_coeffs,
            self.diag_coeffs, self.subdiag_coeffs,
            self.a_coeffs, self.b_coeffs, self.scale_m_table,
            ws.P, ws.dP, ws.cos_m, ws.sin_m
        )
        return np.array([ax, ay, az], dtype=np.float64)

    def accel_adaptive(
        self,
        r_body_fixed: Vec3,
        degree_far: int,
        degree_near: int,
        alt_far: float,
        alt_near: float,
        degree_step: int = 10,
        workspace: Optional[Workspace] = None,
    ) -> np.ndarray:
        """
        Altitude-based adaptive blending (smooth transitions between resolutions).
        """
        x = float(r_body_fixed[0])
        y = float(r_body_fixed[1])
        z = float(r_body_fixed[2])
        
        ws = self._resolve_ws(workspace)

        ax, ay, az = sh_accel_adaptive_blend_numba(
            x, y, z,
            int(degree_far), int(degree_near),
            float(alt_far), float(alt_near),
            int(degree_step),
            self.r_ref, self.mu,
            self.c_coeffs, self.s_coeffs,
            self.diag_coeffs, self.subdiag_coeffs,
            self.a_coeffs, self.b_coeffs, self.scale_m_table,
            ws.P, ws.dP, ws.cos_m, ws.sin_m
        )
        return np.array([ax, ay, az], dtype=np.float64)



# =============================================================================
# 7.                                 PUBLIC API
# =============================================================================

__all__ = (
    # --- Core Containers & Factories ---
    "GravityModel",                    # Main container: coefficients + tables + API
    "SHWorkspace",                    # Thread-local scratch buffers (pre-allocated)
    "make_sh_workspace",              # Factory for SHWorkspace buffers

    # --- High-Level Acceleration APIs (Python Dispatch) ---
    "sh_accel_fixed",                 # Smart dispatch (Serial/Parallel) for fixed degree

    # --- Numerical JIT-Kernels (Numba-Safe) ---
    "sh_accel_fixed_numba",           # Serial Kahan-compensated fixed-degree kernel
    "sh_accel_adaptive_blend_numba",  # Altitude-based adaptive resolution kernel
    "compute_point_mass_acceleration",# Newtonian monopole baseline (point-mass)

    # --- Utility & Setup Functions ---
    "slice_gravity_model",            # Truncates/pads coefficient arrays to degree N
)