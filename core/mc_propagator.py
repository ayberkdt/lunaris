# ST_LRPS/core/mc_propagator.py
# -*- coding: utf-8 -*-
"""
Batch Monte Carlo Propagators (GPU + CPU)
=========================================

This module provides two parallel batch propagators that share the same
public interface but target different hardware:

GPU path  — ``GPUBatchPropagator``
    Uses CUDA (via ``numba.cuda``) to propagate N samples in parallel with a
    fixed-step RK4 integrator.  Each CUDA thread handles exactly one sample.

    Physics available on GPU:
      * Spherical harmonics gravity up to degree 24 (compile-time fixed workspace).
      * Point-mass fallback (when ``gpu_sh_degree=0``).
      * Third-body Sun / Earth point-mass perturbations.
      * Solar Radiation Pressure (flat-plate, eclipse-checked).
      * 1PN Schwarzschild relativistic correction.

    Physics CPU-only (not on GPU):
      * Albedo / thermal surface forces.
      * High-degree SH (> 24).
      * Tide models.

CPU path  — ``CPUBatchPropagator``
    Runs the full-fidelity ``core.propagator.propagate()`` path for each
    sample while reusing the same validated runtime assets as the main mission
    analysis pipeline.  All physics enabled in the parent ``SimConfig`` are
    therefore available on the CPU Monte Carlo backend.

Architecture notes
------------------
- Both propagators consume a (N, 6) initial-state matrix and a pre-computed
  ephemeris pack for Sun/Earth positions and attitude quaternions.
- Output is a (T, N, 6) snapshot tensor at ``output_dt_s`` intervals, written
  to a caller-supplied callback or accumulated in RAM.
- GPU workspace arrays (P/dP/cos/sin for ALF recurrence) are thread-local
  ``cuda.local.array`` objects so there are no race conditions.

GPU SH degree constraint
------------------------
The GPU SH kernel allocates workspace via ``cuda.local.array`` which requires
compile-time constant shapes.  The module-level constant ``_GPU_WS = 26``
gives workspace arrays of shape (26, 26), supporting SH degree ≤ 24.
"""


from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common.constants import AU, MU_EARTH, MU_MOON, MU_SUN, P_SUN_1AU, R_EARTH_MEAN, R_MOON
from common.type_defs import F64Array


# =============================================================================
# 0.                     CUDA AVAILABILITY GUARD
# =============================================================================

_CUDA_AVAILABLE: bool = False
try:
    from numba import cuda, float64 as nb_f64  # type: ignore[attr-defined]
    from numba import njit  # noqa: F401 – used in CPU helpers
    import numba
    _CUDA_AVAILABLE = bool(cuda.is_available())
except ImportError:
    cuda = None  # type: ignore[assignment]
    nb_f64 = None
    numba = None  # type: ignore[assignment]


# Compile-time workspace size: supports SH degree up to _GPU_WS-2 = 24
_GPU_WS: int = 26


def gpu_unsupported_features(flags: Any) -> Tuple[str, ...]:
    """
    Return the active physics options that the current CUDA backend cannot model.

    The GPU path now covers Moon SH gravity, Sun/Earth third-body terms, Earth
    J2, SRP, and 1PN relativity.  Surface-lighting forces and solid tides still
    require the CPU full-fidelity propagator.
    """

    unsupported: List[str] = []
    if bool(getattr(flags, "enable_albedo", False)):
        unsupported.append("albedo")
    if bool(getattr(flags, "enable_thermal", False)):
        unsupported.append("thermal IR")
    if bool(getattr(flags, "enable_tides_k2", False)) or bool(getattr(flags, "enable_tides_k3", False)):
        unsupported.append("solid tides")
    return tuple(unsupported)


def _sanitize_gpu_threads_per_block(
    requested: int,
    *,
    warp_size: int = 32,
    max_threads_per_block: int = 1024,
) -> int:
    """
    Align a user-provided CUDA launch width to device-safe values.

    The UI exposes threads-per-block as an expert tuning control, but GPU
    devices still require sane launch shapes.  We therefore clamp the request
    to the current device limit and align it down to a whole warp.
    """

    warp = max(1, int(warp_size))
    hard_max = max(warp, int(max_threads_per_block))
    value = int(requested) if requested is not None else 128
    if value <= 0:
        value = 128
    value = max(warp, min(value, hard_max))
    value = (value // warp) * warp
    if value < warp:
        value = warp
    return min(value, hard_max)


# =============================================================================
# 1.              CUDA DEVICE FUNCTIONS (physics primitives)
# =============================================================================

if _CUDA_AVAILABLE:

    @cuda.jit(device=True, inline=True)
    def _pm_accel_cuda(rx, ry, rz, mu, out):
        """Point-mass gravitational acceleration (body-fixed or inertial)."""
        r2 = rx * rx + ry * ry + rz * rz
        if r2 < 1.0:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
            return
        inv_r3 = 1.0 / (r2 * math.sqrt(r2))
        out[0] = -mu * rx * inv_r3
        out[1] = -mu * ry * inv_r3
        out[2] = -mu * rz * inv_r3

    @cuda.jit(device=True, inline=True)
    def _third_body_cuda(rx, ry, rz, bx, by, bz, mu_b, out):
        """Third-body acceleration (differential formulation)."""
        dx = bx - rx; dy = by - ry; dz = bz - rz
        d2 = dx * dx + dy * dy + dz * dz
        b2 = bx * bx + by * by + bz * bz
        if d2 < 1.0 or b2 < 1.0:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
            return
        inv_d3 = 1.0 / (d2 * math.sqrt(d2))
        inv_b3 = 1.0 / (b2 * math.sqrt(b2))
        out[0] = mu_b * (dx * inv_d3 - bx * inv_b3)
        out[1] = mu_b * (dy * inv_d3 - by * inv_b3)
        out[2] = mu_b * (dz * inv_d3 - bz * inv_b3)

    @cuda.jit(device=True, inline=True)
    def _interp3_cuda(t, dt_s, tab, n_tab, result):
        """Linear interpolation of a (N, 3) pre-tabulated vector at time t."""
        u = t / dt_s
        i = int(u)
        if i < 0:
            i = 0
        if i >= n_tab - 1:
            i = n_tab - 2
        a = u - float(i)
        result[0] = tab[i, 0] * (1.0 - a) + tab[i + 1, 0] * a
        result[1] = tab[i, 1] * (1.0 - a) + tab[i + 1, 1] * a
        result[2] = tab[i, 2] * (1.0 - a) + tab[i + 1, 2] * a

    @cuda.jit(device=True, inline=True)
    def _interp4_cuda(t, dt_s, tab, n_tab, result):
        """Linear interpolation of a (N, 4) pre-tabulated quaternion at time t."""
        u = t / dt_s
        i = int(u)
        if i < 0:
            i = 0
        if i >= n_tab - 1:
            i = n_tab - 2
        a = u - float(i)
        b = 1.0 - a
        result[0] = tab[i, 0] * b + tab[i + 1, 0] * a
        result[1] = tab[i, 1] * b + tab[i + 1, 1] * a
        result[2] = tab[i, 2] * b + tab[i + 1, 2] * a
        result[3] = tab[i, 3] * b + tab[i + 1, 3] * a
        # Renormalise (avoid drift)
        nrm = math.sqrt(
            result[0] * result[0] + result[1] * result[1]
            + result[2] * result[2] + result[3] * result[3]
        )
        if nrm > 1e-15:
            inv_n = 1.0 / nrm
            result[0] *= inv_n; result[1] *= inv_n
            result[2] *= inv_n; result[3] *= inv_n

    @cuda.jit(device=True, inline=True)
    def _quat_rot_cuda(q0, q1, q2, q3, vx, vy, vz, out):
        """Rotate vector (vx,vy,vz) by quaternion (q0,q1,q2,q3) [scalar-first]."""
        # t = 2 * cross(q_vec, v)
        tx = 2.0 * (q2 * vz - q3 * vy)
        ty = 2.0 * (q3 * vx - q1 * vz)
        tz = 2.0 * (q1 * vy - q2 * vx)
        # v' = v + q0*t + cross(q_vec, t)
        out[0] = vx + q0 * tx + q2 * tz - q3 * ty
        out[1] = vy + q0 * ty + q3 * tx - q1 * tz
        out[2] = vz + q0 * tz + q1 * ty - q2 * tx

    @cuda.jit(device=True)
    def _sh_accel_cuda(
        rx, ry, rz,
        n_eval,
        r_ref, gm,
        Cnm, Snm,
        diag, subdiag, A_coef, B_coef, scale_m,
        out,
    ):
        """
        Spherical-harmonic gravity acceleration in the body-fixed frame.

        Implements the same algorithm as ``_compute_sh_acceleration_serial``
        (models/spherical_harmonics.py) but with:
          * thread-local workspace arrays (``cuda.local.array``).
          * No Kahan summation (lower accuracy, adequate for MC spread).
          * Compile-time fixed workspace size (_GPU_WS × _GPU_WS).

        Constraint: n_eval must be ≤ _GPU_WS - 2 = 24.
        """
        # ----------------------------------------------------------------
        # Thread-local workspace: fixed at compile time
        # ----------------------------------------------------------------
        P     = cuda.local.array((_GPU_WS, _GPU_WS), numba.float64)
        dP    = cuda.local.array((_GPU_WS, _GPU_WS), numba.float64)
        cos_m = cuda.local.array(_GPU_WS, numba.float64)
        sin_m = cuda.local.array(_GPU_WS, numba.float64)

        # ----------------------------------------------------------------
        # Coordinate transform → spherical
        # ----------------------------------------------------------------
        rho2 = rx * rx + ry * ry
        r2   = rho2 + rz * rz
        if r2 < 1.0:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
            return

        r      = math.sqrt(r2)
        inv_r  = 1.0 / r
        rho    = math.sqrt(rho2)
        sin_ph = rz * inv_r
        cos_ph = rho * inv_r

        if rho > 1e-6:
            inv_rho = 1.0 / rho
            cos_lon = rx * inv_rho
            sin_lon = ry * inv_rho
        else:
            cos_lon = 1.0
            sin_lon = 0.0

        # Meridional unit vector
        u_phi_x = -sin_ph * cos_lon
        u_phi_y = -sin_ph * sin_lon
        u_phi_z = cos_ph

        # ----------------------------------------------------------------
        # ALF recurrence: P[n,m] and dP[n,m]
        # ----------------------------------------------------------------
        P[0][0]  = 1.0
        dP[0][0] = 0.0

        n_eff = n_eval if n_eval < _GPU_WS - 1 else _GPU_WS - 2

        for n in range(1, n_eff + 1):
            # Sectoral: P[n,n]
            P[n][n]  = diag[n] * cos_ph * P[n - 1][n - 1]
            dP[n][n] = 0.0   # filled below via derivative recurrence

            # Sub-sectoral: P[n,n-1]
            P[n][n - 1]  = subdiag[n] * sin_ph * P[n - 1][n - 1]
            dP[n][n - 1] = 0.0

            # Zonal/tesseral: P[n,m] for m = 0..n-2
            if n >= 2:
                for m in range(n - 1):
                    P[n][m] = (
                        A_coef[n][m] * sin_ph * P[n - 1][m]
                        - B_coef[n][m] * P[n - 2][m]
                    )

            # Derivative dP[n,0] = sqrt(n(n+1)) * P[n,1]
            if n_eff >= 1 and n >= 1:
                dP[n][0] = math.sqrt(float(n) * float(n + 1)) * P[n][1]

            # Derivative dP[n,m] for m >= 1
            for m in range(1, n + 1):
                c_m = math.sqrt(float(n + m) * float(n - m + 1))
                term_minus = c_m * P[n][m - 1]
                if m + 1 <= n:
                    c_p = math.sqrt(float(n - m) * float(n + m + 1))
                    term_plus = c_p * P[n][m + 1]
                else:
                    term_plus = 0.0
                dP[n][m] = 0.5 * (term_plus - term_minus)

        # Apply scale_m (sqrt(2) for m>0 + Condon-Shortley phase)
        for n in range(n_eff + 1):
            for m in range(n + 1):
                P[n][m]  *= scale_m[m]
                dP[n][m] *= scale_m[m]

        # ----------------------------------------------------------------
        # Longitude trig tables
        # ----------------------------------------------------------------
        cos_m[0] = 1.0
        sin_m[0] = 0.0
        if n_eff >= 1:
            cos_m[1] = cos_lon
            sin_m[1] = sin_lon
        for m in range(2, n_eff + 1):
            cos_m[m] = cos_m[m - 1] * cos_lon - sin_m[m - 1] * sin_lon
            sin_m[m] = sin_m[m - 1] * cos_lon + cos_m[m - 1] * sin_lon

        # ----------------------------------------------------------------
        # Gradient summation (spherical components)
        # ----------------------------------------------------------------
        dv_dr     = -gm / r2                  # central term
        dv_dphi   = 0.0
        dv_dlambda= 0.0

        r_ratio = r_ref * inv_r
        r_ratio_n = r_ratio * r_ratio         # (r_ref/r)^2 for n=2

        mu_inv_r   = gm * inv_r
        mu_inv_r2  = gm / r2

        for n in range(2, n_eff + 1):
            s_r = 0.0; s_p = 0.0; s_l = 0.0
            for m in range(n + 1):
                c_lon = cos_m[m]
                s_lon = sin_m[m]
                H    =  Cnm[n][m] * c_lon + Snm[n][m] * s_lon
                dH_dl= -Cnm[n][m] * s_lon + Snm[n][m] * c_lon
                pnm  = P[n][m]
                dpnm = dP[n][m]
                s_r  += pnm  * H
                s_p  += dpnm * H
                s_l  += float(m) * pnm * dH_dl

            dv_dr      += -mu_inv_r2 * float(n + 1) * r_ratio_n * s_r
            dv_dphi    +=  mu_inv_r  * r_ratio_n * s_p
            dv_dlambda +=  mu_inv_r  * r_ratio_n * s_l

            r_ratio_n *= r_ratio

        # ----------------------------------------------------------------
        # Gradient → Cartesian acceleration
        # ----------------------------------------------------------------
        phi_fac   = dv_dphi * inv_r
        inv_rho2  = 1.0 / (rho2 + 1e-24)

        out[0] = (dv_dr * rx * inv_r
                  + phi_fac * u_phi_x
                  - dv_dlambda * ry * inv_rho2)
        out[1] = (dv_dr * ry * inv_r
                  + phi_fac * u_phi_y
                  + dv_dlambda * rx * inv_rho2)
        out[2] = (dv_dr * rz * inv_r
                  + phi_fac * u_phi_z)

    @cuda.jit(device=True, inline=True)
    def _srp_accel_cuda(
        rx, ry, rz,
        sun_x, sun_y, sun_z,
        r_moon, r_earth_aux, au, p_1au,
        cr, area, mass,
        out,
    ):
        """
        Solar Radiation Pressure acceleration with lunar shadow check.

        Conical shadow (umbra only) using spherical Moon.
        No Earth shadow on the GPU path (negligible contribution near Moon).
        """
        # Sun-spacecraft vector
        s2sc_x = rx - sun_x
        s2sc_y = ry - sun_y
        s2sc_z = rz - sun_z
        r2_sc_sun = s2sc_x * s2sc_x + s2sc_y * s2sc_y + s2sc_z * s2sc_z
        if r2_sc_sun < 1.0:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
            return

        r_sc_sun = math.sqrt(r2_sc_sun)
        inv_rss  = 1.0 / r_sc_sun

        # Moon shadow: project Sun→SC line onto Sun→Moon direction
        sun_mag2 = sun_x * sun_x + sun_y * sun_y + sun_z * sun_z
        if sun_mag2 < 1.0:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
            return

        inv_sm = 1.0 / math.sqrt(sun_mag2)
        # unit vector Sun → Moon (Moon is at origin, so Moon - Sun = -sun)
        d_x = -sun_x * inv_sm
        d_y = -sun_y * inv_sm
        d_z = -sun_z * inv_sm

        # Projection of (SC - Sun) onto shadow axis
        proj = s2sc_x * d_x + s2sc_y * d_y + s2sc_z * d_z

        # Shadow cylinder check (simplified umbra)
        in_shadow = 0
        if proj > 0.0:
            perp2 = (r2_sc_sun - proj * proj)
            if perp2 < r_moon * r_moon:
                in_shadow = 1

        if in_shadow:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
            return

        # SRP magnitude: F/m = (P_1AU * CR * A/m) * (AU/r)²
        au2 = au * au
        factor = (p_1au * cr * area / mass) * (au2 / r2_sc_sun)

        # Direction: unit vector from Sun to SC
        out[0] = factor * s2sc_x * inv_rss
        out[1] = factor * s2sc_y * inv_rss
        out[2] = factor * s2sc_z * inv_rss

    @cuda.jit(device=True, inline=True)
    def _relativity_1pn_cuda(rx, ry, rz, vx, vy, vz, mu, out):
        """1PN Schwarzschild acceleration correction."""
        c_light = 299_792_458.0
        c2 = c_light * c_light
        r2 = rx * rx + ry * ry + rz * rz
        if r2 < 1.0:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
            return
        r     = math.sqrt(r2)
        inv_r = 1.0 / r
        v2    = vx * vx + vy * vy + vz * vz
        rdotv = rx * vx + ry * vy + rz * vz
        inv_r3 = inv_r / r2
        fac = mu * inv_r3 / c2
        # Schwarzschild term
        A = (4.0 * mu * inv_r - v2)
        B = 4.0 * rdotv
        out[0] = fac * (A * rx + B * vx)
        out[1] = fac * (A * ry + B * vy)
        out[2] = fac * (A * rz + B * vz)

    @cuda.jit(device=True, inline=True)
    def _j2_oblate_unit_cuda(x, y, z, mu, r_ref, j2, kx, ky, kz, out):
        """J2 acceleration for an oblate body in its inertial frame."""
        r2 = x * x + y * y + z * z
        if r2 <= 1.0:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
            return

        rk = x * kx + y * ky + z * kz
        rk2 = rk * rk
        inv_r = 1.0 / math.sqrt(r2)
        inv_r2 = inv_r * inv_r
        inv_r5 = inv_r2 * inv_r2 * inv_r

        pref = 1.5 * j2 * mu * (r_ref * r_ref) * inv_r5
        term_a = 5.0 * (rk2 * inv_r2) - 1.0
        term_b = 2.0 * rk

        out[0] = pref * (term_a * x - term_b * kx)
        out[1] = pref * (term_a * y - term_b * ky)
        out[2] = pref * (term_a * z - term_b * kz)

    @cuda.jit(device=True, inline=True)
    def _earth_j2_diff_cuda(rx, ry, rz, bx, by, bz, mu_body, r_ref, j2, kx, ky, kz, out):
        """Differential Earth-J2 term in the Moon-centered inertial frame."""
        sc = cuda.local.array(3, numba.float64)
        moon = cuda.local.array(3, numba.float64)

        _j2_oblate_unit_cuda(rx - bx, ry - by, rz - bz, mu_body, r_ref, j2, kx, ky, kz, sc)
        _j2_oblate_unit_cuda(-bx, -by, -bz, mu_body, r_ref, j2, kx, ky, kz, moon)

        out[0] = sc[0] - moon[0]
        out[1] = sc[1] - moon[1]
        out[2] = sc[2] - moon[2]

    # ------------------------------------------------------------------
    # Main RHS device function
    # ------------------------------------------------------------------

    @cuda.jit(device=True)
    def _rhs_cuda(
        t, y,
        # Ephemeris tables
        ephem_dt, sun_tab, earth_tab, q_tab, n_ephem,
        # Gravity
        n_sh, r_ref, gm,
        Cnm, Snm, diag_sh, subdiag_sh, A_sh, B_sh, scale_sh,
        # Physics flags (0/1 int)
        use_sh, use_sun, use_earth, use_ej2, use_srp, use_rel,
        # Spacecraft
        mass, area, cr,
        # Constants
        mu_sun, mu_earth, earth_j2_r_ref, earth_j2_j2, earth_j2_kx, earth_j2_ky, earth_j2_kz, r_moon, r_earth, au, p1au,
        # Output
        dydt,
    ):
        """
        Full equations of motion for one sample on GPU.

        ``y``    : (6,) state [rx, ry, rz, vx, vy, vz]  (inertial frame, m, m/s)
        ``dydt`` : (6,) output derivative
        """
        rx, ry, rz = y[0], y[1], y[2]
        vx, vy, vz = y[3], y[4], y[5]

        dydt[0] = vx; dydt[1] = vy; dydt[2] = vz

        # Thread-local temporaries
        accel = cuda.local.array(3, numba.float64)
        sun   = cuda.local.array(3, numba.float64)
        earth = cuda.local.array(3, numba.float64)
        quat  = cuda.local.array(4, numba.float64)
        rf    = cuda.local.array(3, numba.float64)   # body-fixed position

        ax = 0.0; ay = 0.0; az = 0.0

        # Ephemeris interpolation
        _interp3_cuda(t, ephem_dt, sun_tab,   n_ephem, sun)
        _interp3_cuda(t, ephem_dt, earth_tab, n_ephem, earth)
        _interp4_cuda(t, ephem_dt, q_tab,     n_ephem, quat)

        # A) Gravity (SH or PM)
        if use_sh == 1 and n_sh >= 2:
            # Rotate inertial → body-fixed
            _quat_rot_cuda(quat[0], quat[1], quat[2], quat[3], rx, ry, rz, rf)
            _sh_accel_cuda(
                rf[0], rf[1], rf[2],
                n_sh, r_ref, gm,
                Cnm, Snm, diag_sh, subdiag_sh, A_sh, B_sh, scale_sh,
                accel,
            )
            # Rotate body-fixed acceleration → inertial (conjugate quaternion)
            _quat_rot_cuda(quat[0], -quat[1], -quat[2], -quat[3], accel[0], accel[1], accel[2], accel)
        else:
            _pm_accel_cuda(rx, ry, rz, gm, accel)

        ax += accel[0]; ay += accel[1]; az += accel[2]

        # B) Third-body Sun
        if use_sun == 1:
            _third_body_cuda(rx, ry, rz, sun[0], sun[1], sun[2], mu_sun, accel)
            ax += accel[0]; ay += accel[1]; az += accel[2]

        # C) Third-body Earth
        if use_earth == 1:
            _third_body_cuda(rx, ry, rz, earth[0], earth[1], earth[2], mu_earth, accel)
            ax += accel[0]; ay += accel[1]; az += accel[2]

        # D) Earth J2
        if use_ej2 == 1:
            _earth_j2_diff_cuda(
                rx, ry, rz,
                earth[0], earth[1], earth[2],
                mu_earth,
                earth_j2_r_ref,
                earth_j2_j2,
                earth_j2_kx,
                earth_j2_ky,
                earth_j2_kz,
                accel,
            )
            ax += accel[0]; ay += accel[1]; az += accel[2]

        # E) SRP
        if use_srp == 1:
            _srp_accel_cuda(
                rx, ry, rz,
                sun[0], sun[1], sun[2],
                r_moon, r_earth, au, p1au,
                cr, area, mass,
                accel,
            )
            ax += accel[0]; ay += accel[1]; az += accel[2]

        # F) 1PN Relativity
        if use_rel == 1:
            _relativity_1pn_cuda(rx, ry, rz, vx, vy, vz, gm, accel)
            ax += accel[0]; ay += accel[1]; az += accel[2]

        dydt[3] = ax; dydt[4] = ay; dydt[5] = az

    # ------------------------------------------------------------------
    # CUDA RK4 batch kernel
    # ------------------------------------------------------------------

    @cuda.jit
    def _rk4_batch_kernel(
        Y,              # (N, 6) current state [in-place updated]
        t_val,          # scalar: current simulation time
        dt,             # scalar: RK4 step size
        # Ephemeris
        ephem_dt, sun_tab, earth_tab, q_tab, n_ephem,
        # Gravity
        n_sh, r_ref, gm,
        Cnm, Snm, diag_sh, subdiag_sh, A_sh, B_sh, scale_sh,
        # Flags
        use_sh, use_sun, use_earth, use_ej2, use_srp, use_rel,
        # Spacecraft (per-sample)
        masses, areas, crs,
        # Constants
        mu_sun, mu_earth, earth_j2_r_ref, earth_j2_j2, earth_j2_kx, earth_j2_ky, earth_j2_kz, r_moon, r_earth, au, p1au,
        # Impact detection
        impact_flags,   # (N,) int32 – set to 1 on impact
        impact_times,   # (N,) float64 – first step-crossing time, NaN if untouched
        r_impact,       # impact radius [m]
    ):
        """
        One RK4 step for all N samples in parallel.

        Thread layout: 1D grid, 1 thread per sample.
        Y is updated in-place; samples already flagged as impacted are skipped.
        """
        i = cuda.grid(1)
        if i >= Y.shape[0]:
            return
        if impact_flags[i] != 0:
            return   # this sample already impacted

        # Load state
        y  = cuda.local.array(6, numba.float64)
        k1 = cuda.local.array(6, numba.float64)
        k2 = cuda.local.array(6, numba.float64)
        k3 = cuda.local.array(6, numba.float64)
        k4 = cuda.local.array(6, numba.float64)
        ytmp = cuda.local.array(6, numba.float64)

        for j in range(6):
            y[j] = Y[i, j]

        sc_mass = masses[i]
        sc_area = areas[i]
        sc_cr   = crs[i]

        # k1 = f(t, y)
        _rhs_cuda(
            t_val, y,
            ephem_dt, sun_tab, earth_tab, q_tab, n_ephem,
            n_sh, r_ref, gm, Cnm, Snm, diag_sh, subdiag_sh, A_sh, B_sh, scale_sh,
            use_sh, use_sun, use_earth, use_ej2, use_srp, use_rel,
            sc_mass, sc_area, sc_cr,
            mu_sun, mu_earth, earth_j2_r_ref, earth_j2_j2, earth_j2_kx, earth_j2_ky, earth_j2_kz, r_moon, r_earth, au, p1au,
            k1,
        )

        # k2 = f(t + dt/2, y + dt/2 * k1)
        h2 = 0.5 * dt
        for j in range(6):
            ytmp[j] = y[j] + h2 * k1[j]
        _rhs_cuda(
            t_val + h2, ytmp,
            ephem_dt, sun_tab, earth_tab, q_tab, n_ephem,
            n_sh, r_ref, gm, Cnm, Snm, diag_sh, subdiag_sh, A_sh, B_sh, scale_sh,
            use_sh, use_sun, use_earth, use_ej2, use_srp, use_rel,
            sc_mass, sc_area, sc_cr,
            mu_sun, mu_earth, earth_j2_r_ref, earth_j2_j2, earth_j2_kx, earth_j2_ky, earth_j2_kz, r_moon, r_earth, au, p1au,
            k2,
        )

        # k3 = f(t + dt/2, y + dt/2 * k2)
        for j in range(6):
            ytmp[j] = y[j] + h2 * k2[j]
        _rhs_cuda(
            t_val + h2, ytmp,
            ephem_dt, sun_tab, earth_tab, q_tab, n_ephem,
            n_sh, r_ref, gm, Cnm, Snm, diag_sh, subdiag_sh, A_sh, B_sh, scale_sh,
            use_sh, use_sun, use_earth, use_ej2, use_srp, use_rel,
            sc_mass, sc_area, sc_cr,
            mu_sun, mu_earth, earth_j2_r_ref, earth_j2_j2, earth_j2_kx, earth_j2_ky, earth_j2_kz, r_moon, r_earth, au, p1au,
            k3,
        )

        # k4 = f(t + dt, y + dt * k3)
        for j in range(6):
            ytmp[j] = y[j] + dt * k3[j]
        _rhs_cuda(
            t_val + dt, ytmp,
            ephem_dt, sun_tab, earth_tab, q_tab, n_ephem,
            n_sh, r_ref, gm, Cnm, Snm, diag_sh, subdiag_sh, A_sh, B_sh, scale_sh,
            use_sh, use_sun, use_earth, use_ej2, use_srp, use_rel,
            sc_mass, sc_area, sc_cr,
            mu_sun, mu_earth, earth_j2_r_ref, earth_j2_j2, earth_j2_kx, earth_j2_ky, earth_j2_kz, r_moon, r_earth, au, p1au,
            k4,
        )

        # RK4 update: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        w = dt / 6.0
        for j in range(6):
            y[j] += w * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])

        # Impact detection
        r_sc = math.sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2])
        if r_sc <= r_impact:
            impact_flags[i] = 1
            if math.isnan(impact_times[i]):
                impact_times[i] = t_val + dt

        # Write back
        for j in range(6):
            Y[i, j] = y[j]


# =============================================================================
# 2.              GPU BATCH PROPAGATOR
# =============================================================================

@dataclass
class _EphemGPUPack:
    """Pre-validated ephemeris data transferred to GPU device memory."""
    dt_s: float
    d_sun:   Any   # device array (N, 3)
    d_earth: Any   # device array (N, 3)
    d_quat:  Any   # device array (N, 4)
    n_rows:  int


@dataclass
class _GravGPUPack:
    """Gravity model arrays transferred to GPU device memory."""
    n_sh:       int
    r_ref:      float
    gm:         float
    d_Cnm:      Any
    d_Snm:      Any
    d_diag:     Any
    d_subdiag:  Any
    d_A:        Any
    d_B:        Any
    d_scale_m:  Any


class GPUBatchPropagator:
    """
    Fixed-step RK4 Monte Carlo propagator running on CUDA GPU.

    Parameters
    ----------
    dynamics_engine : DynamicsEngine
        Must have been prepared (``build_rhs()`` called or assets loaded).
    mc_cfg : MonteCarloConfig
        Determines step size, SH degree, output cadence, VRAM budget.
    flags : PerturbationFlags
        Which physics terms to evaluate on GPU.
    """

    def __init__(
        self,
        dynamics_engine: Any,          # core.dynamics.DynamicsEngine
        mc_cfg: Any,                   # MonteCarloConfig
        flags: Any,                    # PerturbationFlags
    ) -> None:
        """
        Pre-transfer all heavy arrays (gravity coefficients, ephemeris tables)
        to GPU device memory so that repeated kernel launches during propagation
        do not pay the PCIe transfer cost at every time step.
        """
        if not _CUDA_AVAILABLE:
            raise RuntimeError(
                "numba.cuda is not available.  Install CUDA toolkit and numba, "
                "or use CPUBatchPropagator."
            )

        self._mc = mc_cfg
        self._flags = flags
        self._device_id = int(mc_cfg.gpu_device_id)

        with cuda.gpus[self._device_id]:
            dev = cuda.get_current_device()
            dev_name = getattr(dev, "name", f"cuda:{self._device_id}")
            if isinstance(dev_name, bytes):
                dev_name = dev_name.decode(errors="ignore")
            self._device_name = str(dev_name).strip() or f"cuda:{self._device_id}"
            self._warp_size = int(getattr(dev, "WARP_SIZE", 32))
            self._max_threads_per_block = int(getattr(dev, "MAX_THREADS_PER_BLOCK", 1024))
            self._launch_tpb = _sanitize_gpu_threads_per_block(
                getattr(mc_cfg, "gpu_threads_per_block", 128),
                warp_size=self._warp_size,
                max_threads_per_block=self._max_threads_per_block,
            )
            try:
                self._free_mem_bytes, self._total_mem_bytes = cuda.current_context().get_memory_info()
            except Exception:
                self._free_mem_bytes, self._total_mem_bytes = (0, 0)

        self._ephem_pack = self._build_ephem_pack(dynamics_engine)
        self._grav_pack  = self._build_grav_pack(dynamics_engine)
        self._earth_j2_pack = self._build_earth_j2_pack(dynamics_engine)
        self._recommended_max_batch = self._estimate_recommended_max_batch()

        # Physics flags → integers (branchless in CUDA)
        f = flags
        self._use_sh    = int(bool(getattr(f, "enable_sh", True)))
        self._use_sun   = int(bool(getattr(f, "enable_3rd_body_sun", False)))
        self._use_earth = int(bool(getattr(f, "enable_3rd_body_earth", False)))
        self._use_ej2   = int(bool(getattr(f, "enable_earth_j2", False)) and bool(self._earth_j2_pack["enabled"]))
        self._use_srp   = int(bool(getattr(f, "enable_srp", False)))
        self._use_rel   = int(bool(getattr(f, "enable_relativity_1pn", False)))

        # Physical constants
        self._mu_sun   = float(MU_SUN)
        self._mu_earth = float(MU_EARTH)
        self._r_moon   = float(R_MOON)
        self._r_earth  = float(R_EARTH_MEAN)
        self._au       = float(AU)
        self._p1au     = float(P_SUN_1AU)

    # ----------------------------------------------------------------
    # Setup helpers
    # ----------------------------------------------------------------

    def _build_ephem_pack(self, dyn: Any) -> _EphemGPUPack:
        """
        Extract ephemeris tables and move them to GPU device memory.

        The strict ephemeris contract now allows Sun/Earth vector tables to
        collapse to a single constant row when only the Moon-fixed attitude
        quaternion timeline is needed.  The CUDA kernel still expects a single
        shared row count for all three tables, so we expand those constant rows
        to the quaternion timeline length here.
        """
        from core.dynamics import extract_ephem_tables_strict
        ep = getattr(dyn, "ephem", None)
        if ep is None:
            # Minimal stub: 2-row tables at t=0
            rows = 2
            sun_h   = np.zeros((rows, 3), dtype=np.float64)
            earth_h = np.zeros((rows, 3), dtype=np.float64)
            q_h     = np.tile([1.0, 0.0, 0.0, 0.0], (rows, 1)).astype(np.float64)
            dt_s    = 1.0
        else:
            dt_s, sun_h, earth_h, q_h = extract_ephem_tables_strict(ep)
            q_rows = int(q_h.shape[0])
            if int(sun_h.shape[0]) == 1 and q_rows > 1:
                sun_h = np.repeat(sun_h, q_rows, axis=0)
            if int(earth_h.shape[0]) == 1 and q_rows > 1:
                earth_h = np.repeat(earth_h, q_rows, axis=0)
            if q_rows == 1:
                max_rows = max(int(sun_h.shape[0]), int(earth_h.shape[0]), 1)
                if max_rows > 1:
                    q_h = np.repeat(q_h, max_rows, axis=0)

        with cuda.gpus[self._device_id]:
            d_sun   = cuda.to_device(np.ascontiguousarray(sun_h,   dtype=np.float64))
            d_earth = cuda.to_device(np.ascontiguousarray(earth_h, dtype=np.float64))
            d_q     = cuda.to_device(np.ascontiguousarray(q_h,     dtype=np.float64))

        return _EphemGPUPack(
            dt_s=float(dt_s),
            d_sun=d_sun,
            d_earth=d_earth,
            d_quat=d_q,
            n_rows=max(int(sun_h.shape[0]), int(earth_h.shape[0]), int(q_h.shape[0])),
        )

    def _build_grav_pack(self, dyn: Any) -> _GravGPUPack:
        """Extract gravity model and move coefficient arrays to GPU."""
        from core.dynamics import extract_gravity_strict
        from models.spherical_harmonics import build_legendre_coeffs

        grav = getattr(dyn, "grav", None)
        n_sh_req = int(self._mc.gpu_sh_degree)

        if grav is None or n_sh_req == 0:
            # Point-mass only
            dummy1 = np.zeros((2, 2), dtype=np.float64)
            dummy_1d = np.zeros(2, dtype=np.float64)
            with cuda.gpus[self._device_id]:
                return _GravGPUPack(
                    n_sh=0, r_ref=float(R_MOON), gm=float(MU_MOON),
                    d_Cnm=cuda.to_device(dummy1),
                    d_Snm=cuda.to_device(dummy1),
                    d_diag=cuda.to_device(dummy_1d),
                    d_subdiag=cuda.to_device(dummy_1d),
                    d_A=cuda.to_device(dummy1),
                    d_B=cuda.to_device(dummy1),
                    d_scale_m=cuda.to_device(dummy_1d),
                )

        nmax, r_ref, gm, Cnm, Snm, *_ = extract_gravity_strict(grav)
        n_use = min(int(nmax), n_sh_req, _GPU_WS - 2)

        # Slice coefficients to n_use
        Cnm_s = np.ascontiguousarray(Cnm[:n_use + 1, :n_use + 1], dtype=np.float64)
        Snm_s = np.ascontiguousarray(Snm[:n_use + 1, :n_use + 1], dtype=np.float64)

        diag, subdiag, A, B, scale_m = build_legendre_coeffs(n_use)

        with cuda.gpus[self._device_id]:
            return _GravGPUPack(
                n_sh=n_use,
                r_ref=float(r_ref),
                gm=float(gm),
                d_Cnm=cuda.to_device(Cnm_s),
                d_Snm=cuda.to_device(Snm_s),
                d_diag=cuda.to_device(np.ascontiguousarray(diag,    dtype=np.float64)),
                d_subdiag=cuda.to_device(np.ascontiguousarray(subdiag, dtype=np.float64)),
                d_A=cuda.to_device(np.ascontiguousarray(A, dtype=np.float64)),
                d_B=cuda.to_device(np.ascontiguousarray(B, dtype=np.float64)),
                d_scale_m=cuda.to_device(np.ascontiguousarray(scale_m, dtype=np.float64)),
            )

    def _build_earth_j2_pack(self, dyn: Any) -> Dict[str, float]:
        """
        Normalize optional Earth-J2 parameters into GPU-friendly scalars.

        The main dynamics engine already stores Earth J2 as simple physical
        parameters, so the GPU backend only needs a validated, unit-length spin
        axis and a boolean telling the kernel whether the term is active.
        """

        ej2 = getattr(dyn, "earth_j2", None)
        if ej2 is None:
            return {
                "enabled": 0.0,
                "j2": 0.0,
                "r_ref_m": 1.0,
                "kx": 0.0,
                "ky": 0.0,
                "kz": 1.0,
            }

        j2 = float(getattr(ej2, "j2_coeff", 0.0))
        r_ref = float(getattr(ej2, "r_eq_m", 1.0))
        kx, ky, kz = getattr(ej2, "spin_axis_i", (0.0, 0.0, 1.0))
        k_norm = math.sqrt(float(kx) * float(kx) + float(ky) * float(ky) + float(kz) * float(kz))
        if r_ref <= 0.0 or j2 == 0.0 or k_norm <= 1.0e-15:
            return {
                "enabled": 0.0,
                "j2": 0.0,
                "r_ref_m": 1.0,
                "kx": 0.0,
                "ky": 0.0,
                "kz": 1.0,
            }

        inv_k = 1.0 / k_norm
        return {
            "enabled": 1.0,
            "j2": j2,
            "r_ref_m": r_ref,
            "kx": float(kx) * inv_k,
            "ky": float(ky) * inv_k,
            "kz": float(kz) * inv_k,
        }

    def _estimate_recommended_max_batch(self) -> int:
        """
        Estimate a safe GPU sub-batch size using live device memory when possible.

        This stays conservative on purpose. A slightly smaller batch is far less
        harmful than a launch that overruns device memory and aborts the MC run.
        """

        bytes_per_sample = (
            6 * 8    # state vector
            + 3 * 8  # mass / area / cr
            + 4      # impact flag
            + 8      # impact time
            + 32     # launch / scratch safety margin
        )
        cfg_budget = float(getattr(self._mc, "max_vram_gb", 4.0)) * (1024.0 ** 3) * 0.80
        live_budget = float(self._free_mem_bytes) * 0.70 if self._free_mem_bytes else cfg_budget
        budget = max(1.0, min(cfg_budget, live_budget))
        return max(1, int(budget / max(1, bytes_per_sample)))

    def recommended_max_batch(self, requested_max_batch: Optional[int] = None) -> int:
        """Return the backend-specific batch cap after device-memory tuning."""

        requested = int(requested_max_batch) if requested_max_batch is not None else int(self._recommended_max_batch)
        return max(1, min(requested, int(self._recommended_max_batch)))

    def diagnostics_snapshot(self) -> Dict[str, Any]:
        """Expose lightweight runtime diagnostics for logs, reports, and tests."""

        return {
            "device_name": self._device_name,
            "device_id": int(self._device_id),
            "threads_per_block": int(self._launch_tpb),
            "warp_size": int(self._warp_size),
            "gpu_free_mem_bytes": int(self._free_mem_bytes),
            "gpu_total_mem_bytes": int(self._total_mem_bytes),
            "recommended_max_batch": int(self._recommended_max_batch),
            "gpu_sh_degree": int(getattr(self._grav_pack, "n_sh", 0)),
            "supports_earth_j2": bool(self._earth_j2_pack["enabled"]),
        }

    # ----------------------------------------------------------------
    # Public: propagate batch
    # ----------------------------------------------------------------

    def propagate(
        self,
        Y0: F64Array,               # (N, 6) initial states
        masses: F64Array,           # (N,)
        areas: F64Array,            # (N,)
        cds: F64Array,              # (N,) — ignored on GPU path (no drag term in kernel)
        crs: F64Array,              # (N,)
        duration_s: float,
        output_dt_s: float,
        callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[F64Array, F64Array, F64Array, F64Array]:
        """
        Propagate N samples from t=0 to t=duration_s with fixed RK4 step dt_s.

        Parameters
        ----------
        Y0 : (N, 6) initial state ensemble
        masses, areas, cds, crs : (N,) per-sample spacecraft properties.
            ``cds`` is accepted for API parity with CPUBatchPropagator but is
            not forwarded to the CUDA kernel (the GPU physics model has no drag
            term — only gravity, third-body, SRP, and 1PN relativity).
        duration_s : total propagation span [s]
        output_dt_s : snapshot interval [s]; must be a multiple of dt_s
        callback : optional ``f(progress_fraction)``
            Receives a normalized snapshot-progress fraction in ``[0, 1]``.
            Using a lightweight scalar callback keeps the GPU and CPU backends
            on the same progress-reporting contract for the desktop UI.

        Returns
        -------
        t_out : (T,) snapshot times [s]
        Y_out : (T, N, 6) state snapshots
        impact_flags : (N,) float64 – 1.0 if impacted, else 0.0
        t_impact : (N,) float64 – NaN on GPU path (exact event time unavailable)
        """
        N   = int(Y0.shape[0])
        dt  = float(self._mc.dt_s)
        T   = float(duration_s)
        out_dt = float(output_dt_s)

        steps_per_snap = max(1, int(round(out_dt / dt)))
        n_snaps = max(1, int(round(T / out_dt)))

        ep = self._ephem_pack
        gp = self._grav_pack
        tpb = int(self._launch_tpb)
        bpg = (N + tpb - 1) // tpb

        r_impact = float(R_MOON) + float(self._mc.impact_alt_km) * 1_000.0

        t_out = np.empty(n_snaps, dtype=np.float64)
        Y_out = np.empty((n_snaps, N, 6), dtype=np.float64)
        t_curr = 0.0
        with cuda.gpus[self._device_id]:
            stream = cuda.stream()
            d_Y = cuda.to_device(np.ascontiguousarray(Y0, dtype=np.float64), stream=stream)
            d_masses = cuda.to_device(np.ascontiguousarray(masses, dtype=np.float64), stream=stream)
            d_areas  = cuda.to_device(np.ascontiguousarray(areas,  dtype=np.float64), stream=stream)
            d_crs    = cuda.to_device(np.ascontiguousarray(crs,    dtype=np.float64), stream=stream)
            d_impact = cuda.to_device(np.zeros(N, dtype=np.int32), stream=stream)
            d_t_impact = cuda.to_device(np.full(N, np.nan, dtype=np.float64), stream=stream)
            stream.synchronize()

            for snap_idx in range(n_snaps):
                for _ in range(steps_per_snap):
                    _rk4_batch_kernel[bpg, tpb, stream](
                        d_Y,
                        np.float64(t_curr),
                        np.float64(dt),
                        np.float64(ep.dt_s),
                        ep.d_sun, ep.d_earth, ep.d_quat,
                        np.int32(ep.n_rows),
                        np.int32(gp.n_sh),
                        np.float64(gp.r_ref),
                        np.float64(gp.gm),
                        gp.d_Cnm, gp.d_Snm,
                        gp.d_diag, gp.d_subdiag,
                        gp.d_A, gp.d_B, gp.d_scale_m,
                        np.int32(self._use_sh),
                        np.int32(self._use_sun),
                        np.int32(self._use_earth),
                        np.int32(self._use_ej2),
                        np.int32(self._use_srp),
                        np.int32(self._use_rel),
                        d_masses, d_areas, d_crs,
                        np.float64(self._mu_sun),
                        np.float64(self._mu_earth),
                        np.float64(self._earth_j2_pack["r_ref_m"]),
                        np.float64(self._earth_j2_pack["j2"]),
                        np.float64(self._earth_j2_pack["kx"]),
                        np.float64(self._earth_j2_pack["ky"]),
                        np.float64(self._earth_j2_pack["kz"]),
                        np.float64(self._r_moon),
                        np.float64(self._r_earth),
                        np.float64(self._au),
                        np.float64(self._p1au),
                        d_impact,
                        d_t_impact,
                        np.float64(r_impact),
                    )
                    t_curr += dt

                d_Y.copy_to_host(ary=Y_out[snap_idx], stream=stream)
                stream.synchronize()
                t_out[snap_idx] = t_curr

                if callback is not None:
                    callback(float(snap_idx + 1) / float(max(n_snaps, 1)))

            impact_host = d_impact.copy_to_host(stream=stream).astype(np.float64)
            t_impact_host = d_t_impact.copy_to_host(stream=stream).astype(np.float64)
            stream.synchronize()

        return t_out, Y_out, impact_host, t_impact_host


# =============================================================================
# 3.              CPU BATCH PROPAGATOR (full-fidelity multiprocessing)
# =============================================================================

def _build_cpu_time_and_solver_config(sim_cfg: Any, mc_cfg: Any, duration_s: float, output_dt_s: float) -> Tuple[Any, Any]:
    """
    Clone the nominal run configs with MC-specific time and impact settings applied.

    The single-run UI stores the impact threshold inside ``PropagatorConfig.events``,
    while the MC page owns its own ``impact_alt_km`` control.  This helper makes
    that mapping explicit so the CPU Monte Carlo path uses the same event logic
    the rest of the project uses.
    """

    time_cfg = replace(
        sim_cfg.time,
        duration_s=float(duration_s),
        output_dt_s=float(output_dt_s),
    )
    events_cfg = replace(
        sim_cfg.propagator.events,
        detect_impact=bool(getattr(mc_cfg, "compute_impact_probability", True)),
        impact_alt_km=float(getattr(mc_cfg, "impact_alt_km", 0.0)),
    )
    prop_cfg = replace(sim_cfg.propagator, events=events_cfg)
    return time_cfg, prop_cfg


class CPUBatchPropagator:
    """
    Full-fidelity Monte Carlo propagator for the CPU backend.

    The previous implementation attempted to rebuild each sample inside worker
    processes using legacy helper APIs that no longer match the rest of the
    codebase.  This version instead reuses the already-validated runtime assets
    (gravity, ephemeris, surface providers) and constructs a fresh lightweight
    ``DynamicsEngine`` per sample in-process.  That keeps the CPU MC path fully
    aligned with the main propagation pipeline and, most importantly, reliable.

    Notes
    -----
    - The approach is intentionally conservative: correctness and API
      compatibility are prioritized over multiprocessing throughput.
    - Heavy read-only assets are reused from the template dynamics instance, so
      per-sample overhead remains manageable for the UI-driven MC workloads.
    """

    def __init__(
        self,
        sim_cfg: Any,
        mc_cfg: Any,
        dynamics_template: Optional[Any] = None,
        surface_provider: Any = None,
        topo_grid: Any = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        sim_cfg : SimConfig
            Full simulation configuration used to rebuild a ``DynamicsEngine``
            inside each worker process (Numba JIT objects are not picklable).
        mc_cfg : MonteCarloConfig
            Monte Carlo parameters (not used directly here; workers read
            ``sim_cfg`` to reconstruct their own state).
        dynamics_template : DynamicsEngine, optional
            Pre-built dynamics instance whose validated gravity / ephemeris
            assets can be reused for each sampled spacecraft.
        surface_provider / topo_grid :
            Optional terrain assets forwarded to the single-sample propagator.
        max_workers : int, optional
            Retained for API compatibility with older callers.  The current
            implementation runs samples in-process, so this acts only as
            metadata for future extensions.
        """
        self._sim_cfg = sim_cfg
        self._mc = mc_cfg
        self._dyn_template = dynamics_template
        self._surface_provider = surface_provider
        self._topo_grid = topo_grid
        self._max_workers = max_workers or os.cpu_count() or 1

    def _make_sample_dynamics(self, *, mass_kg: float, area_m2: float, cd: float, cr: float) -> Any:
        """
        Create a per-sample ``DynamicsEngine`` that reuses heavy shared assets.

        Each sample perturbs spacecraft properties, so the dynamics object must
        reflect those values.  Gravity coefficients, ephemeris tables, and
        surface providers are read-only and can therefore be reused safely in
        this sequential CPU execution model.
        """

        from common.type_defs import SpacecraftProps
        from core.dynamics import DynamicsEngine

        sc = SpacecraftProps(
            mass_kg=float(mass_kg),
            area_m2=float(area_m2),
            cd=float(cd),
            cr=float(cr),
        )

        template = self._dyn_template
        grav_model = getattr(template, "grav", None) if template is not None else None
        gravity_adaptive = getattr(template, "gravity_adaptive", None) if template is not None else None
        ephem_manager = getattr(template, "ephem", None) if template is not None else None
        earth_j2 = getattr(template, "earth_j2", None) if template is not None else None

        return DynamicsEngine(
            sc_props=sc,
            flags=self._sim_cfg.flags,
            gravity_model=grav_model,
            gravity_adaptive=gravity_adaptive,
            ephem_manager=ephem_manager,
            surface_provider=self._surface_provider,
            earth_j2=earth_j2,
            allow_identity_rotation=(ephem_manager is None),
        )

    def validate_gravity_assets(self) -> None:
        """
        Probe gravity attributes before entering the Monte Carlo sample loop.

        Calls the same helper functions that ``core.propagator.propagate()``
        uses internally so any missing attribute is caught once, with a single
        clear error message, instead of being repeated for every sample.

        Raises
        ------
        RuntimeError
            When the gravity model attached to the template dynamics engine is
            missing a required attribute such as ``degree_max``.
        """

        from core.propagator import _get_ref_radius_and_mu, _get_sh_degree

        dyn = self._make_sample_dynamics(
            mass_kg=float(self._sim_cfg.spacecraft.mass_kg),
            area_m2=float(self._sim_cfg.spacecraft.area_m2),
            cd=float(self._sim_cfg.spacecraft.cd),
            cr=float(self._sim_cfg.spacecraft.cr),
        )
        try:
            _get_ref_radius_and_mu(dyn)
            _get_sh_degree(dyn)
        except AttributeError as exc:
            grav = getattr(dyn, "grav", None)
            kind = getattr(grav, "model_kind", "unknown")
            raise RuntimeError(
                f"[MC] Pre-flight gravity validation failed (backend='{kind}'): {exc}. "
                "Cannot start Monte Carlo run."
            ) from exc

    def propagate(
        self,
        Y0: F64Array,           # (N, 6)
        masses: F64Array,       # (N,)
        areas: F64Array,        # (N,)
        cds: F64Array,          # (N,) — sampled drag coefficients
        crs: F64Array,          # (N,)
        duration_s: float,
        output_dt_s: float,
        callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[F64Array, F64Array, F64Array, F64Array]:
        """
        Propagate N samples on CPU using the full-fidelity single-run solver.

        Returns the same ``(t_out, Y_out, impact_flags, t_impact)`` contract as
        the GPU path. The first successful sample defines the reference time
        grid and later samples are linearly resampled onto that grid when
        needed.
        """
        from core.propagator import propagate as propagate_single

        N = int(Y0.shape[0])
        time_cfg, prop_cfg = _build_cpu_time_and_solver_config(
            self._sim_cfg,
            self._mc,
            duration_s=float(duration_s),
            output_dt_s=float(output_dt_s),
        )

        results_by_idx: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, float]] = {}
        for i in range(N):
            try:
                dyn = self._make_sample_dynamics(
                    mass_kg=float(masses[i]),
                    area_m2=float(areas[i]),
                    cd=float(cds[i]),
                    cr=float(crs[i]),
                )
                result = propagate_single(
                    dynamics=dyn,
                    y0=np.ascontiguousarray(Y0[i], dtype=np.float64),
                    cfg=prop_cfg,
                    time_cfg=time_cfg,
                    topo_grid=self._topo_grid,
                )
                results_by_idx[i] = (
                    np.asarray(result.t, dtype=np.float64),
                    np.asarray(result.y, dtype=np.float64),
                    bool(result.impacted),
                    float(result.t_impact_s) if result.t_impact_s is not None else float("nan"),
                )
            except Exception as exc:
                if not getattr(self._mc, "allow_sample_failures", False):
                    raise RuntimeError(f"Monte Carlo CPU sample {i} failed: {exc}") from exc
                warnings.warn(f"[MC][CPU] Sample {i} failed: {exc}", RuntimeWarning)
                results_by_idx[i] = (None, None, False, float("nan"))

            if callback is not None:
                callback(float(i + 1) / float(max(N, 1)))

        # Use first successful sample's time grid as reference
        ref_t = None
        for i in range(N):
            t_i, y_i, _, _ = results_by_idx.get(i, (None, None, None, None))
            if t_i is not None and len(t_i) >= 2:
                ref_t = t_i
                break

        if ref_t is None:
            raise RuntimeError("All MC samples failed in CPUBatchPropagator.")

        T = len(ref_t)
        Y_out = np.zeros((T, N, 6), dtype=np.float64)
        impact_flags = np.zeros(N, dtype=np.float64)
        t_impact = np.full(N, np.nan, dtype=np.float64)

        for i in range(N):
            t_i, y_i, imp, t_imp = results_by_idx.get(i, (None, None, False, np.nan))
            if t_i is None or y_i is None:
                continue
            # Resample to ref_t grid
            for state_col in range(6):
                Y_out[:, i, state_col] = np.interp(ref_t, t_i, y_i[:, state_col])
            if imp:
                impact_flags[i] = 1.0
                if np.isfinite(float(t_imp)):
                    t_impact[i] = float(t_imp)

        return ref_t, Y_out, impact_flags, t_impact


# =============================================================================
# 4.                        PUBLIC API
# =============================================================================

__all__ = [
    "GPUBatchPropagator",
    "CPUBatchPropagator",
    "gpu_unsupported_features",
]
