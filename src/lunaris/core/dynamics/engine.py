# lunaris.core.dynamics
"""
Core Dynamics Engine (EOM / RHS Builder)
=======================================

This module builds the equations of motion (RHS) used by numerical integrators
(e.g. SciPy ``solve_ivp`` or fixed-step propagators).

Architecture
------------
High-level Python objects (gravity, ephemeris, surface data, etc.) are kept
outside the inner loop. ``build_rhs()`` extracts all required inputs into
Numba-friendly primitives (floats, arrays, booleans) and constructs a JIT-
compiled RHS closure. The compiled RHS must not allocate heap memory or access
Python objects.

Reference frames
----------------
- Integration frame: Moon-Centered Inertial (MCI, J2000-like).
- Body-fixed frame: Moon-Centered Fixed (MCF).
- The ephemeris provides the inertial→fixed attitude quaternion ``q_i2f``,
  stored scalar-first as ``(w, x, y, z)``.

State convention
----------------
The state vector is

    ``y = [rx, ry, rz, vx, vy, vz]``

with an optional 7th element interpreted as spacecraft mass (``dm/dt = 0``).

Implementation notes
--------------------
- Ephemeris sampling uses an allocation-free kernel (unpacked float return).
- Perturbation models used in the RHS are written as allocation-free kernels.
- A single internal contract is enforced for provider inputs (gravity/ephem/
  surface) to keep the propagation core consistent across modules.
"""


# =============================================================================
# 0.                                IMPORTS
# =============================================================================

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit

from lunaris.common.constants import (
    AU,
    MU_EARTH,
    MU_SUN,
    P_SUN_1AU,
    R_EARTH_MEAN,
    R_MOON,
)
from lunaris.common.frame_policy import FrameModeInput, resolve_frame_policy
from lunaris.common.math_utils import (
    # Numba-callable kernels for use inside @njit code (the public
    # sample_*/sample_grid_* wrappers validate in Python and cannot be called
    # from nopython mode).
    _sample_grid_bilinear_kernel,
    latlon_from_xyz_m,
    quat_rotate_vec,
    sample_grid_bilinear,
)
from lunaris.common.type_defs import PerturbationFlags, SolidTideConfig, SpacecraftProps
from lunaris.core.dynamics.adaptive_degree import (
    _sample_albedo_dn_scaled,
    _select_adaptive_sh_degree,
)
from lunaris.core.dynamics.ephemeris_pack import _EphemPack
from lunaris.core.dynamics.gravity_pack import _GravPack
from lunaris.core.dynamics.perturbation_packs import (
    _AlbedoPack,
    _EarthJ2Pack,
    _ThermalPack,
    _TidePack,
)
from lunaris.core.dynamics.preparation import (
    DynamicsRequirements,
    compute_requirements,
    prepare_albedo,
    prepare_earth_j2,
    prepare_ephem,
    prepare_gravity,
    prepare_solid_tides,
    prepare_thermal,
    resolve_effective_requirements,
    validate_dependencies,
)
from lunaris.physics.ephemeris import get_ephem_state, interp_vec3_derivative_safe
from lunaris.physics.lunar_albedo import (
    accel_albedo_facets_numba,
)
from lunaris.physics.relativity_effects import _external_1pn_components, _schwarzschild_components
from lunaris.physics.solar_effects import SRPConfig, accel_srp
from lunaris.physics.solid_tides import accel_solid_tides_numba
from lunaris.physics.spherical_harmonics import (
    compute_point_mass_acceleration,
    sh_accel_fixed_numba,
)
from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig, accel_albedo_simple
from lunaris.physics.thermal_ir import (
    accel_thermal_ir_facets_numba,
)
from lunaris.physics.third_body_effects import accel_j2_oblate_diff_numba, accel_third_body_numba

logger = logging.getLogger(__name__)


def _validate_rhs_state_vector(y: Any) -> np.ndarray:
    """Return a 1D state vector accepted by the dynamics RHS."""

    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] not in (6, 7):
        raise ValueError(
            "DynamicsEngine RHS supports state vectors with exactly 6 elements "
            "or 7 elements when y[6] is spacecraft mass."
        )
    return arr


# =============================================================================
# 1.                             DYNAMICS ENGINE
# =============================================================================

class DynamicsEngine:
    """
    Dynamics RHS builder for high-performance orbit propagation.

    State (SI):
        y = [rx, ry, rz, vx, vy, vz]  (+ optional mass_kg in y[6])
    """

    def __init__(
        self,
        sc_props: SpacecraftProps,
        flags: PerturbationFlags,
        *,
        gravity_model: Any = None,
        gravity_adaptive: Any = None,
        ephem_manager: Any = None,
        surface_provider: Any = None,
        earth_j2: Any = None,
        srp: SRPConfig | None = None,
        thermal: Any = None,
        albedo: Any = None,
        solid_tides: SolidTideConfig | None = None,
        allow_identity_rotation: bool = False,
        frame_mode: FrameModeInput = None,
    ) -> None:
        self.sc_props = sc_props
        self.flags = flags

        self.grav = gravity_model
        self.gravity_adaptive = gravity_adaptive
        self.ephem = ephem_manager
        self.surf = surface_provider
        self.earth_j2 = earth_j2
        self.srp = srp if srp is not None else SRPConfig()
        self.thermal = thermal if thermal is not None else ThermalConfig()
        self.albedo = albedo if albedo is not None else AlbedoConfig()
        self.solid_tides = solid_tides if solid_tides is not None else SolidTideConfig()

        # If True, q_i2f is treated as identity when ephemeris is absent.
        # This only substitutes for frame rotation (q), NOT for Sun/Earth vectors.
        self.frame_policy = resolve_frame_policy(
            ephem_manager=self.ephem,
            allow_identity_rotation=bool(allow_identity_rotation),
            frame_mode=frame_mode,
        )
        self.frame_mode = self.frame_policy.mode
        self.allow_identity_rotation = self.frame_policy.allow_identity_rotation

        self._rhs_cache: Callable[[float, np.ndarray], np.ndarray] | None = None
        self._prep: dict[str, Any] = {}  # debug/reporting packs + requirements

        self._validate_dependencies()

    # -------------------------------------------------------------------------
    # Requirements / validation
    # -------------------------------------------------------------------------
    def _requirements(self) -> DynamicsRequirements:
        """Raw config-derived requirement set (see ``dynamics.preparation``)."""
        return compute_requirements(
            flags=self.flags,
            gravity_model=self.grav,
            earth_j2=self.earth_j2,
            albedo=self.albedo,
            thermal=self.thermal,
            solid_tides=self.solid_tides,
            allow_identity_rotation=self.allow_identity_rotation,
            have_ephem=self.ephem is not None,
        )

    def _validate_dependencies(self) -> None:
        validate_dependencies(
            req=self._requirements(),
            gravity_model=self.grav,
            surface_provider=self.surf,
            albedo=self.albedo,
            sc_props=self.sc_props,
            ephem_manager=self.ephem,
            flags=self.flags,
            solid_tides=self.solid_tides,
            earth_j2=self.earth_j2,
        )

    # -------------------------------------------------------------------------
    # Public: build RHS
    # -------------------------------------------------------------------------
    def build_rhs(self, *, force_rebuild: bool = False) -> Callable[[float, np.ndarray], np.ndarray]:
        if self._rhs_cache is not None and not force_rebuild:
            return self._rhs_cache

        t0 = time.perf_counter()

        req = self._requirements()
        gp = prepare_gravity(self.grav, gravity_adaptive=self.gravity_adaptive)
        ep = prepare_ephem(self.ephem, req)
        # Raw config-derived requirements -> effective runtime requirements.
        # Kept as an explicit, side-effect-free step: prepare_ephem never
        # mutates `req`, so the closure captures below cannot silently depend
        # on call order.
        req = resolve_effective_requirements(req, ep)
        ap = prepare_albedo(req, albedo=self.albedo, surface_provider=self.surf)
        ej = prepare_earth_j2(req, self.earth_j2)
        tp = prepare_solid_tides(req, self.solid_tides)
        th = prepare_thermal(req, thermal=self.thermal, surface_provider=self.surf)

        # Cache for debug/reporting
        self._prep = {"req": req, "grav": gp, "eph": ep, "alb": ap, "earth_j2": ej, "tides": tp, "thermal": th}

        # Flags captured into closure
        USE_SH = bool(req.use_sh)
        USE_SURROGATE = bool(req.use_surrogate_gravity)
        USE_3RD_SUN = bool(req.use_3rd_sun)
        USE_3RD_EARTH = bool(req.use_3rd_earth)
        USE_SRP = bool(req.use_srp)
        USE_ALBEDO = bool(req.use_albedo)
        USE_REL = bool(req.use_rel)
        USE_REL_EXTERNAL = bool(req.use_rel_external)
        USE_EJ2 = bool(req.use_earth_j2)
        USE_TIDES = bool(req.use_tides)
        USE_TIDE_EARTH = bool(req.use_tide_earth)
        USE_TIDE_SUN = bool(req.use_tide_sun)
        USE_THERMAL = bool(req.use_thermal)
        SH_ONLY_FAST_PATH = bool(
            USE_SH
            and not USE_SURROGATE
            and not USE_3RD_SUN
            and not USE_3RD_EARTH
            and not USE_SRP
            and not USE_ALBEDO
            and not USE_REL
            and not USE_EJ2
            and not USE_TIDES
            and not USE_THERMAL
        )
        NEED_FIXED_FRAME = bool(USE_SH or USE_TIDES or USE_ALBEDO or USE_THERMAL)
        NEED_SUN_FIXED = bool(USE_TIDE_SUN or USE_ALBEDO or USE_THERMAL)
        NEED_EARTH_FIXED = bool(USE_TIDE_EARTH or USE_ALBEDO or USE_THERMAL)

        # Ephemeris fetch needed inside kernel only if we actually have ephem_manager.
        HAVE_EPH = bool(self.ephem is not None)
        NEED_EPH = bool(HAVE_EPH and (req.need_sun or req.need_earth or req.need_q))

        # Spacecraft constants
        SC_MASS = float(self.sc_props.mass_kg)
        SC_AREA = float(self.sc_props.area_m2)
        SC_CR = float(self.sc_props.cr)

        # Core constants
        MU_M = float(gp.gm_m3s2)  # prefer model GM even if SH is disabled
        MU_S = float(MU_SUN)
        MU_E = float(MU_EARTH)

        srp_cfg = self.srp
        RMOON = float(getattr(srp_cfg, "R_moon_m", R_MOON))
        R_EARTH = float(getattr(srp_cfg, "R_earth_m", R_EARTH_MEAN))
        AU_ = float(getattr(srp_cfg, "AU_m", AU))
        P1AU = float(getattr(srp_cfg, "P0", P_SUN_1AU))
        SRP_MOON_ECLIPSE = bool(getattr(srp_cfg, "enable_moon_eclipse", True))
        SRP_EARTH_ECLIPSE = bool(getattr(srp_cfg, "enable_earth_eclipse", False))

        ENABLE_ECLIPSE = True

        # Ephemeris arrays/scalars
        EPH_DT_S = float(ep.dt_s)
        EPH_SUN = ep.r_sun_tab_m
        EPH_EARTH = ep.r_earth_tab_m
        EPH_QTAB = ep.q_i2f_tab

        # Gravity scalars/arrays
        G_NMAX = int(gp.nmax)
        G_RREF = float(gp.r_ref_m)
        G_GM = float(gp.gm_m3s2)
        G_CNM = gp.Cnm
        G_SNM = gp.Snm
        G_DIAG = gp.diag
        G_SUB = gp.subdiag
        G_A = gp.A
        G_B = gp.B
        G_SCL = gp.scale_m
        G_ADAPTIVE_ENABLED = bool(gp.adaptive_enabled)
        G_ADAPTIVE_MODE = int(gp.adaptive_mode)
        G_ADAPTIVE_POWER = float(gp.adaptive_power)
        G_ADAPTIVE_MIN_DEG = int(gp.adaptive_min_degree)
        G_ADAPTIVE_QSTEP = int(gp.adaptive_quantization_step)
        G_ADAPTIVE_ALT_KM = gp.adaptive_table_alt_km
        G_ADAPTIVE_DEG = gp.adaptive_table_degree
        G_ADAPTIVE_TABLE_LEN = int(gp.adaptive_table_len)

        WS_P = gp.ws_P
        WS_DP = gp.ws_dP
        WS_COS = gp.ws_cos_m
        WS_SIN = gp.ws_sin_m

        # Albedo scalars/arrays (provide stable arrays to closure)
        ALB_MODE = int(ap.mode)
        ALB_CONST = float(ap.alb_const)
        ALB_SCALE = float(ap.alb_scale)
        ALB_KLAMB = float(ap.k_lambert)

        ALB_GRID = ap.grid_alb if ap.grid_alb is not None else np.zeros((1, 1), dtype=np.float64)
        ALB_DN = ap.dn if ap.dn is not None else np.zeros((1, 1), dtype=np.float64)

        ALB_NLINES = int(ap.n_lines)
        ALB_NSAMPLES = int(ap.n_samples)
        ALB_RES = float(ap.res_deg)
        ALB_LON0 = float(ap.lon0_deg)
        ALB_LAT0 = float(ap.lat0_deg)
        ALB_SF = float(ap.sf)
        ALB_OFF = float(ap.off)
        ALB_MISSING = float(ap.missing)
        ALB_FLIP = int(ap.flip)
        ALB_LATMIN = float(ap.latmin)
        ALB_LATMAX = float(ap.latmax)

        # Albedo backend selector + lambert_facets arrays/coefficients.
        ALB_BACKEND = int(ap.backend)  # 0 = simple cannonball, 1 = lambert_facets
        ALB_FACET_POS = ap.facet_pos_m
        ALB_FACET_NORM = ap.facet_normals
        ALB_FACET_AREA = ap.facet_areas_m2
        ALB_FACET_ALB = ap.facet_albedo
        ALB_PCOEF = float(ap.pressure_coefficient)
        ALB_SOLAR_FLUX = float(ap.solar_flux_1au_W_m2)
        ALB_AU = float(ap.au_m)
        ALB_C = float(ap.c_light_m_s)
        ALB_R_EARTH = float(ap.r_earth_m)
        ALB_SCALE_SUN_DIST = bool(ap.include_sun_distance_scaling)
        ALB_ECLIPSE = bool(ap.enable_eclipse)

        # Earth J2 axis normalize once outside kernel
        EJ2_J2 = float(ej.j2)
        EJ2_RREF = float(ej.r_ref_m)
        kx, ky, kz = float(ej.ax), float(ej.ay), float(ej.az)
        k2 = kx * kx + ky * ky + kz * kz
        if k2 > 0.0:
            invk = 1.0 / math.sqrt(k2)
            EJ2_KX, EJ2_KY, EJ2_KZ = kx * invk, ky * invk, kz * invk
        else:
            EJ2_KX, EJ2_KY, EJ2_KZ = 0.0, 0.0, 1.0

        # Solid tides
        TIDE_USE_K2 = bool(tp.use_k2)
        TIDE_USE_K3 = bool(tp.use_k3)
        TIDE_K2 = float(tp.k2)
        TIDE_K3 = float(tp.k3)
        TIDE_RREF = float(tp.r_ref_m)

        # Thermal IR
        TH_MODE = int(th.mode)
        TH_EPS = float(th.surface_emissivity)
        TH_ALB = float(th.surface_albedo)
        TH_TEMP = float(th.temperature_K)
        TH_NIGHT_TEMP = float(th.night_temperature_K)
        TH_FLOOR = float(th.thermal_floor_flux_W_m2)
        TH_COEFF = float(th.ir_pressure_coefficient)
        TH_SOLAR_FLUX = float(th.solar_flux_1au_W_m2)
        TH_AU = float(th.au_m)
        TH_C = float(th.c_light_m_s)
        TH_SIGMA = float(th.sigma_sb)
        TH_SCALE_SUN_DIST = bool(th.include_sun_distance_scaling)
        TH_ECLIPSE = bool(th.enable_eclipse)
        TH_R_EARTH = float(th.r_earth_m)
        TH_POS = th.facet_pos_m
        TH_NORMALS = th.facet_normals
        TH_AREAS = th.facet_areas_m2
        TH_TEMPS = th.facet_temperatures_K

        if USE_SURROGATE:
            surrogate = self.grav

            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                y = _validate_rhs_state_vector(y)
                rx, ry, rz = float(y[0]), float(y[1]), float(y[2])
                vx, vy, vz = float(y[3]), float(y[4]), float(y[5])

                n = int(y.shape[0])
                mass = float(SC_MASS if n <= 6 else y[6])

                ax = 0.0
                ay = 0.0
                az = 0.0

                sunx = 0.0
                suny = 0.0
                sunz = 0.0
                earthx = 0.0
                earthy = 0.0
                earthz = 0.0
                q0 = 1.0
                q1 = 0.0
                q2 = 0.0
                q3 = 0.0

                if NEED_EPH:
                    sunx, suny, sunz, earthx, earthy, earthz, q0, q1, q2, q3 = get_ephem_state(
                        float(t), EPH_DT_S, EPH_SUN, EPH_EARTH, EPH_QTAB
                    )

                rfx = rx
                rfy = ry
                rfz = rz
                sfx = sunx
                sfy = suny
                sfz = sunz
                efx = earthx
                efy = earthy
                efz = earthz
                if NEED_FIXED_FRAME:
                    rfx, rfy, rfz = quat_rotate_vec(q0, q1, q2, q3, rx, ry, rz)
                if NEED_SUN_FIXED:
                    sfx, sfy, sfz = quat_rotate_vec(q0, q1, q2, q3, sunx, suny, sunz)
                if NEED_EARTH_FIXED:
                    efx, efy, efz = quat_rotate_vec(q0, q1, q2, q3, earthx, earthy, earthz)

                if USE_SH:
                    s_ax, s_ay, s_az = surrogate.acceleration_fixed((rfx, rfy, rfz))
                    agx, agy, agz = quat_rotate_vec(
                        q0, -q1, -q2, -q3, float(s_ax), float(s_ay), float(s_az)
                    )
                    ax += agx
                    ay += agy
                    az += agz
                else:
                    gax, gay, gaz = compute_point_mass_acceleration(rx, ry, rz, MU_M)
                    ax += gax
                    ay += gay
                    az += gaz

                if USE_3RD_SUN:
                    a3x, a3y, a3z = accel_third_body_numba(rx, ry, rz, sunx, suny, sunz, MU_S)
                    ax += a3x
                    ay += a3y
                    az += a3z

                if USE_3RD_EARTH:
                    a3x, a3y, a3z = accel_third_body_numba(rx, ry, rz, earthx, earthy, earthz, MU_E)
                    ax += a3x
                    ay += a3y
                    az += a3z

                if USE_EJ2:
                    j2x, j2y, j2z = accel_j2_oblate_diff_numba(
                        rx, ry, rz, earthx, earthy, earthz, MU_E, EJ2_RREF, EJ2_J2, EJ2_KX, EJ2_KY, EJ2_KZ
                    )
                    ax += j2x
                    ay += j2y
                    az += j2z

                if USE_TIDES:
                    if USE_TIDE_EARTH:
                        atx_f, aty_f, atz_f = accel_solid_tides_numba(
                            rfx, rfy, rfz, efx, efy, efz, MU_E, TIDE_RREF, TIDE_K2, TIDE_K3, TIDE_USE_K2, TIDE_USE_K3
                        )
                        atx, aty, atz = quat_rotate_vec(q0, -q1, -q2, -q3, atx_f, aty_f, atz_f)
                        ax += atx
                        ay += aty
                        az += atz

                    if USE_TIDE_SUN:
                        atx_f, aty_f, atz_f = accel_solid_tides_numba(
                            rfx, rfy, rfz, sfx, sfy, sfz, MU_S, TIDE_RREF, TIDE_K2, TIDE_K3, TIDE_USE_K2, TIDE_USE_K3
                        )
                        atx, aty, atz = quat_rotate_vec(q0, -q1, -q2, -q3, atx_f, aty_f, atz_f)
                        ax += atx
                        ay += aty
                        az += atz

                if USE_SRP:
                    earth_r2 = earthx * earthx + earthy * earthy + earthz * earthz
                    enable_earth = SRP_EARTH_ECLIPSE and (earth_r2 > 1.0e12)
                    asx, asy, asz = accel_srp(
                        rx, ry, rz, sunx, suny, sunz, earthx, earthy, earthz,
                        RMOON, R_EARTH, AU_, P1AU, SC_CR, SC_AREA, mass,
                        SRP_MOON_ECLIPSE, enable_earth,
                    )
                    ax += asx
                    ay += asy
                    az += asz

                if USE_ALBEDO:
                    if ALB_BACKEND == 1:
                        # Lambertian facet model (reflected solar; Moon-fixed sum).
                        aax_f, aay_f, aaz_f = accel_albedo_facets_numba(
                            rfx, rfy, rfz, sfx, sfy, sfz, efx, efy, efz,
                            ALB_FACET_POS, ALB_FACET_NORM, ALB_FACET_AREA, ALB_FACET_ALB,
                            ALB_PCOEF, SC_AREA, mass,
                            ALB_SOLAR_FLUX, ALB_AU, ALB_C, ALB_R_EARTH,
                            ALB_SCALE_SUN_DIST, ALB_ECLIPSE,
                        )
                    else:
                        # Legacy cannonball (sub-satellite albedo sample).
                        lat_deg, lon_deg, _ = latlon_from_xyz_m(rfx, rfy, rfz)
                        alb_val = ALB_CONST
                        if ALB_MODE == 0:
                            alb_val = sample_grid_bilinear(
                                lat_deg, lon_deg, ALB_GRID, ALB_NLINES, ALB_NSAMPLES, ALB_RES, ALB_LON0, ALB_LAT0
                            )
                        elif ALB_MODE == 1:
                            alb_val = _sample_albedo_dn_scaled(
                                lat_deg, lon_deg, ALB_DN, ALB_NLINES, ALB_NSAMPLES, ALB_RES,
                                ALB_LON0, ALB_LAT0, ALB_FLIP, ALB_SF, ALB_OFF, ALB_MISSING, ALB_LATMIN, ALB_LATMAX,
                            )
                        if math.isnan(alb_val):
                            alb_val = ALB_CONST
                        if alb_val < 0.0:
                            alb_val = 0.0
                        elif alb_val > 1.0:
                            alb_val = 1.0
                        alb_val *= ALB_SCALE
                        aax_f, aay_f, aaz_f = accel_albedo_simple(
                            rfx, rfy, rfz, sfx, sfy, sfz, RMOON, AU_, P1AU,
                            alb_val, ALB_KLAMB, SC_CR, SC_AREA, mass, 1 if ENABLE_ECLIPSE else 0,
                        )

                    aax, aay, aaz = quat_rotate_vec(q0, -q1, -q2, -q3, aax_f, aay_f, aaz_f)
                    ax += aax
                    ay += aay
                    az += aaz

                if USE_THERMAL:
                    athx_f, athy_f, athz_f = accel_thermal_ir_facets_numba(
                        rfx,
                        rfy,
                        rfz,
                        sfx,
                        sfy,
                        sfz,
                        efx,
                        efy,
                        efz,
                        TH_POS,
                        TH_NORMALS,
                        TH_AREAS,
                        TH_TEMPS,
                        TH_MODE,
                        TH_EPS,
                        TH_ALB,
                        TH_TEMP,
                        TH_NIGHT_TEMP,
                        TH_FLOOR,
                        TH_COEFF,
                        SC_AREA,
                        mass,
                        TH_SOLAR_FLUX,
                        TH_AU,
                        TH_C,
                        TH_SIGMA,
                        TH_SCALE_SUN_DIST,
                        TH_R_EARTH,
                        TH_ECLIPSE,
                    )
                    athx, athy, athz = quat_rotate_vec(q0, -q1, -q2, -q3, athx_f, athy_f, athz_f)
                    ax += athx
                    ay += athy
                    az += athz

                if USE_REL:
                    arx, ary, arz = _schwarzschild_components(rx, ry, rz, vx, vy, vz, MU_M)
                    ax += arx
                    ay += ary
                    az += arz
                    if USE_REL_EXTERNAL:
                        svx, svy, svz = interp_vec3_derivative_safe(float(t), EPH_DT_S, EPH_SUN)
                        erx, ery, erz = _external_1pn_components(
                            rx, ry, rz, vx, vy, vz,
                            sunx, suny, sunz,
                            svx, svy, svz,
                            MU_S,
                        )
                        ax += erx
                        ay += ery
                        az += erz

                        evx, evy, evz = interp_vec3_derivative_safe(float(t), EPH_DT_S, EPH_EARTH)
                        erx, ery, erz = _external_1pn_components(
                            rx, ry, rz, vx, vy, vz,
                            earthx, earthy, earthz,
                            evx, evy, evz,
                            MU_E,
                        )
                        ax += erx
                        ay += ery
                        az += erz

                dydt = np.empty_like(y)
                dydt[0] = vx
                dydt[1] = vy
                dydt[2] = vz
                dydt[3] = ax
                dydt[4] = ay
                dydt[5] = az
                if n > 6:
                    dydt[6] = 0.0
                return dydt

            self._rhs_cache = rhs
            self._prep["rhs_path"] = "surrogate_python"
            dt_build = time.perf_counter() - t0
            # Unlike the SH path below, this RHS is a plain Python closure (it
            # calls the PyTorch surrogate, which Numba cannot compile). It pays
            # Python-call + autograd overhead on every evaluation, so a
            # single-trajectory CPU run is NOT a like-for-like speed comparison
            # against the @njit SH kernel; the surrogate only amortizes that
            # overhead in the GPU batch path. See the dynamics path asymmetry note.
            logger.info(
                f"[Dynamics] RHS ready. (build={dt_build:.3f}s | surrogate gravity, "
                "interpreted Python+autograd path -- not @njit; single-trajectory CPU "
                "timings are not comparable to the SH kernel)"
            )
            return rhs

        if SH_ONLY_FAST_PATH:
            # R13: minimal classical-SH RHS for paper-safe SH-only baselines.
            # It keeps only state unpacking, optional Moon-fixed frame rotation,
            # SH acceleration, inverse rotation, and dy/dt assembly.
            @njit(cache=False, nogil=True)
            def _rhs_sh_only_numba(
                t: float,
                y: np.ndarray,
                WS_P: np.ndarray,
                WS_DP: np.ndarray,
                WS_COS: np.ndarray,
                WS_SIN: np.ndarray,
            ) -> np.ndarray:
                rx, ry, rz = y[0], y[1], y[2]
                vx, vy, vz = y[3], y[4], y[5]

                q0 = 1.0
                q1 = 0.0
                q2 = 0.0
                q3 = 0.0
                if NEED_EPH:
                    _sunx, _suny, _sunz, _earthx, _earthy, _earthz, q0, q1, q2, q3 = get_ephem_state(
                        t, EPH_DT_S, EPH_SUN, EPH_EARTH, EPH_QTAB
                    )

                rfx, rfy, rfz = quat_rotate_vec(q0, q1, q2, q3, rx, ry, rz)
                n_eval = G_NMAX
                if G_ADAPTIVE_ENABLED:
                    r_norm = math.sqrt(rx * rx + ry * ry + rz * rz)
                    n_eval = _select_adaptive_sh_degree(
                        r_norm,
                        G_RREF,
                        G_NMAX,
                        G_ADAPTIVE_MODE,
                        G_ADAPTIVE_POWER,
                        G_ADAPTIVE_MIN_DEG,
                        G_ADAPTIVE_QSTEP,
                        G_ADAPTIVE_ALT_KM,
                        G_ADAPTIVE_DEG,
                        G_ADAPTIVE_TABLE_LEN,
                    )
                afx, afy, afz = sh_accel_fixed_numba(
                    rfx,
                    rfy,
                    rfz,
                    n_eval,
                    G_RREF,
                    G_GM,
                    G_CNM,
                    G_SNM,
                    G_DIAG,
                    G_SUB,
                    G_A,
                    G_B,
                    G_SCL,
                    WS_P,
                    WS_DP,
                    WS_COS,
                    WS_SIN,
                )
                ax, ay, az = quat_rotate_vec(q0, -q1, -q2, -q3, afx, afy, afz)

                dydt = np.empty_like(y)
                dydt[0] = vx
                dydt[1] = vy
                dydt[2] = vz
                dydt[3] = ax
                dydt[4] = ay
                dydt[5] = az
                if y.shape[0] > 6:
                    dydt[6] = 0.0
                return dydt

            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                y = _validate_rhs_state_vector(y)
                return _rhs_sh_only_numba(t, y, WS_P, WS_DP, WS_COS, WS_SIN)

            self._rhs_cache = rhs
            self._prep["rhs_path"] = "sh_only_numba"

            dt_build = time.perf_counter() - t0
            logger.info(f"[Dynamics] RHS ready. (build={dt_build:.3f}s | sh-only numba fast path)")

            return rhs

        # This closure captures runtime-sized arrays/config values, so Numba
        # cannot persist it to disk cache reliably. Disabling cache avoids a
        # noisy warning on every run without changing numerical behavior.
        @njit(cache=False, nogil=True)
        def _rhs_kernel_numba(
            t: float,
            y: np.ndarray,
            WS_P: np.ndarray,
            WS_DP: np.ndarray,
            WS_COS: np.ndarray,
            WS_SIN: np.ndarray,
        ) -> np.ndarray:
            rx, ry, rz = y[0], y[1], y[2]
            vx, vy, vz = y[3], y[4], y[5]

            n = y.shape[0]
            mass = SC_MASS if n <= 6 else y[6]

            ax = 0.0
            ay = 0.0
            az = 0.0

            # Ephemeris state (defaults)
            sunx = 0.0
            suny = 0.0
            sunz = 0.0
            earthx = 0.0
            earthy = 0.0
            earthz = 0.0
            q0 = 1.0
            q1 = 0.0
            q2 = 0.0
            q3 = 0.0

            if NEED_EPH:
                sunx, suny, sunz, earthx, earthy, earthz, q0, q1, q2, q3 = get_ephem_state(
                    t, EPH_DT_S, EPH_SUN, EPH_EARTH, EPH_QTAB
                )

            rfx = rx
            rfy = ry
            rfz = rz
            sfx = sunx
            sfy = suny
            sfz = sunz
            efx = earthx
            efy = earthy
            efz = earthz
            if NEED_FIXED_FRAME:
                rfx, rfy, rfz = quat_rotate_vec(q0, q1, q2, q3, rx, ry, rz)
            if NEED_SUN_FIXED:
                sfx, sfy, sfz = quat_rotate_vec(q0, q1, q2, q3, sunx, suny, sunz)
            if NEED_EARTH_FIXED:
                efx, efy, efz = quat_rotate_vec(q0, q1, q2, q3, earthx, earthy, earthz)

            # A) Central gravity
            if USE_SH:
                n_eval = G_NMAX
                if G_ADAPTIVE_ENABLED:
                    r_norm = math.sqrt(rx * rx + ry * ry + rz * rz)
                    n_eval = _select_adaptive_sh_degree(
                        r_norm,
                        G_RREF,
                        G_NMAX,
                        G_ADAPTIVE_MODE,
                        G_ADAPTIVE_POWER,
                        G_ADAPTIVE_MIN_DEG,
                        G_ADAPTIVE_QSTEP,
                        G_ADAPTIVE_ALT_KM,
                        G_ADAPTIVE_DEG,
                        G_ADAPTIVE_TABLE_LEN,
                    )
                afx, afy, afz = sh_accel_fixed_numba(
                    rfx,
                    rfy,
                    rfz,
                    n_eval,
                    G_RREF,
                    G_GM,
                    G_CNM,
                    G_SNM,
                    G_DIAG,
                    G_SUB,
                    G_A,
                    G_B,
                    G_SCL,
                    WS_P,
                    WS_DP,
                    WS_COS,
                    WS_SIN,
                )
                agx, agy, agz = quat_rotate_vec(q0, -q1, -q2, -q3, afx, afy, afz)
                ax += agx
                ay += agy
                az += agz
            else:
                gax, gay, gaz = compute_point_mass_acceleration(rx, ry, rz, MU_M)
                ax += gax
                ay += gay
                az += gaz

            # B) Third-body
            if USE_3RD_SUN:
                a3x, a3y, a3z = accel_third_body_numba(rx, ry, rz, sunx, suny, sunz, MU_S)
                ax += a3x
                ay += a3y
                az += a3z

            if USE_3RD_EARTH:
                a3x, a3y, a3z = accel_third_body_numba(rx, ry, rz, earthx, earthy, earthz, MU_E)
                ax += a3x
                ay += a3y
                az += a3z

            if USE_EJ2:
                j2x, j2y, j2z = accel_j2_oblate_diff_numba(
                    rx,
                    ry,
                    rz,
                    earthx,
                    earthy,
                    earthz,
                    MU_E,
                    EJ2_RREF,
                    EJ2_J2,
                    EJ2_KX,
                    EJ2_KY,
                    EJ2_KZ,
                )
                ax += j2x
                ay += j2y
                az += j2z

            # C) Solid-body tides (Moon-fixed potential gradient)
            if USE_TIDES:
                if USE_TIDE_EARTH:
                    atx_f, aty_f, atz_f = accel_solid_tides_numba(
                        rfx,
                        rfy,
                        rfz,
                        efx,
                        efy,
                        efz,
                        MU_E,
                        TIDE_RREF,
                        TIDE_K2,
                        TIDE_K3,
                        TIDE_USE_K2,
                        TIDE_USE_K3,
                    )
                    atx, aty, atz = quat_rotate_vec(q0, -q1, -q2, -q3, atx_f, aty_f, atz_f)
                    ax += atx
                    ay += aty
                    az += atz

                if USE_TIDE_SUN:
                    atx_f, aty_f, atz_f = accel_solid_tides_numba(
                        rfx,
                        rfy,
                        rfz,
                        sfx,
                        sfy,
                        sfz,
                        MU_S,
                        TIDE_RREF,
                        TIDE_K2,
                        TIDE_K3,
                        TIDE_USE_K2,
                        TIDE_USE_K3,
                    )
                    atx, aty, atz = quat_rotate_vec(q0, -q1, -q2, -q3, atx_f, aty_f, atz_f)
                    ax += atx
                    ay += aty
                    az += atz

            # D) SRP
            if USE_SRP:
                earth_r2 = earthx * earthx + earthy * earthy + earthz * earthz
                enable_earth = SRP_EARTH_ECLIPSE and (earth_r2 > 1.0e12)

                asx, asy, asz = accel_srp(
                    rx,
                    ry,
                    rz,
                    sunx,
                    suny,
                    sunz,
                    earthx,
                    earthy,
                    earthz,
                    RMOON,
                    R_EARTH,
                    AU_,
                    P1AU,
                    SC_CR,
                    SC_AREA,
                    mass,
                    SRP_MOON_ECLIPSE,
                    enable_earth,
                )
                ax += asx
                ay += asy
                az += asz

            # E) Albedo (reflected solar radiation pressure)
            if USE_ALBEDO:
                if ALB_BACKEND == 1:
                    # Lambertian facet model: sum reflected-solar contributions
                    # from sunlit, visible facets in the Moon-fixed frame.
                    aax_f, aay_f, aaz_f = accel_albedo_facets_numba(
                        rfx,
                        rfy,
                        rfz,
                        sfx,
                        sfy,
                        sfz,
                        efx,
                        efy,
                        efz,
                        ALB_FACET_POS,
                        ALB_FACET_NORM,
                        ALB_FACET_AREA,
                        ALB_FACET_ALB,
                        ALB_PCOEF,
                        SC_AREA,
                        mass,
                        ALB_SOLAR_FLUX,
                        ALB_AU,
                        ALB_C,
                        ALB_R_EARTH,
                        ALB_SCALE_SUN_DIST,
                        ALB_ECLIPSE,
                    )
                else:
                    # Legacy cannonball: single sub-satellite albedo sample.
                    lat_deg, lon_deg, _ = latlon_from_xyz_m(rfx, rfy, rfz)

                    alb_val = ALB_CONST
                    if ALB_MODE == 0:
                        alb_val = _sample_grid_bilinear_kernel(
                            lat_deg,
                            lon_deg,
                            ALB_GRID,
                            ALB_NLINES,
                            ALB_NSAMPLES,
                            ALB_RES,
                            ALB_LON0,
                            ALB_LAT0,
                        )
                    elif ALB_MODE == 1:
                        alb_val = _sample_albedo_dn_scaled(
                            lat_deg,
                            lon_deg,
                            ALB_DN,
                            ALB_NLINES,
                            ALB_NSAMPLES,
                            ALB_RES,
                            ALB_LON0,
                            ALB_LAT0,
                            ALB_FLIP,
                            ALB_SF,
                            ALB_OFF,
                            ALB_MISSING,
                            ALB_LATMIN,
                            ALB_LATMAX,
                        )

                    if math.isnan(alb_val):
                        alb_val = ALB_CONST
                    if alb_val < 0.0:
                        alb_val = 0.0
                    elif alb_val > 1.0:
                        alb_val = 1.0
                    alb_val *= ALB_SCALE

                    aax_f, aay_f, aaz_f = accel_albedo_simple(
                        rfx,
                        rfy,
                        rfz,
                        sfx,
                        sfy,
                        sfz,
                        RMOON,
                        AU_,
                        P1AU,
                        alb_val,
                        ALB_KLAMB,
                        SC_CR,
                        SC_AREA,
                        mass,
                        1 if ENABLE_ECLIPSE else 0,
                    )

                aax, aay, aaz = quat_rotate_vec(q0, -q1, -q2, -q3, aax_f, aay_f, aaz_f)
                ax += aax
                ay += aay
                az += aaz

            # F) Lunar thermal IR radiation pressure
            if USE_THERMAL:
                athx_f, athy_f, athz_f = accel_thermal_ir_facets_numba(
                    rfx,
                    rfy,
                    rfz,
                    sfx,
                    sfy,
                    sfz,
                    efx,
                    efy,
                    efz,
                    TH_POS,
                    TH_NORMALS,
                    TH_AREAS,
                    TH_TEMPS,
                    TH_MODE,
                    TH_EPS,
                    TH_ALB,
                    TH_TEMP,
                    TH_NIGHT_TEMP,
                    TH_FLOOR,
                    TH_COEFF,
                    SC_AREA,
                    mass,
                    TH_SOLAR_FLUX,
                    TH_AU,
                    TH_C,
                    TH_SIGMA,
                    TH_SCALE_SUN_DIST,
                    TH_R_EARTH,
                    TH_ECLIPSE,
                )
                athx, athy, athz = quat_rotate_vec(q0, -q1, -q2, -q3, athx_f, athy_f, athz_f)
                ax += athx
                ay += athy
                az += athz

            # G) Relativity
            if USE_REL:
                arx, ary, arz = _schwarzschild_components(rx, ry, rz, vx, vy, vz, MU_M)
                ax += arx
                ay += ary
                az += arz
                if USE_REL_EXTERNAL:
                    svx, svy, svz = interp_vec3_derivative_safe(t, EPH_DT_S, EPH_SUN)
                    erx, ery, erz = _external_1pn_components(
                        rx, ry, rz, vx, vy, vz,
                        sunx, suny, sunz,
                        svx, svy, svz,
                        MU_S,
                    )
                    ax += erx
                    ay += ery
                    az += erz

                    evx, evy, evz = interp_vec3_derivative_safe(t, EPH_DT_S, EPH_EARTH)
                    erx, ery, erz = _external_1pn_components(
                        rx, ry, rz, vx, vy, vz,
                        earthx, earthy, earthz,
                        evx, evy, evz,
                        MU_E,
                    )
                    ax += erx
                    ay += ery
                    az += erz

            dydt = np.empty_like(y)
            dydt[0] = vx
            dydt[1] = vy
            dydt[2] = vz
            dydt[3] = ax
            dydt[4] = ay
            dydt[5] = az
            if n > 6:
                dydt[6] = 0.0
            return dydt

        def rhs(t: float, y: np.ndarray) -> np.ndarray:  # type: ignore[no-redef]
            y = _validate_rhs_state_vector(y)
            return _rhs_kernel_numba(t, y, WS_P, WS_DP, WS_COS, WS_SIN)

        self._rhs_cache = rhs
        self._prep["rhs_path"] = "general_numba"

        dt_build = time.perf_counter() - t0
        logger.info(f"[Dynamics] RHS ready. (build={dt_build:.3f}s)")

        return rhs

    # -------------------------------------------------------------------------
    # Debug / reporting
    # -------------------------------------------------------------------------
    def get_acceleration_breakdown(self, t: float, y: np.ndarray) -> dict[str, float]:
        """Return acceleration component norms at epoch t (debug/reporting)."""
        if not self._prep:
            self.build_rhs(force_rebuild=False)

        req: DynamicsRequirements = self._prep["req"]
        gp: _GravPack = self._prep["grav"]
        ep: _EphemPack = self._prep["eph"]
        ap: _AlbedoPack = self._prep["alb"]
        ej: _EarthJ2Pack = self._prep["earth_j2"]
        tp: _TidePack = self._prep["tides"]
        th: _ThermalPack = self._prep["thermal"]

        r = np.asarray(y[0:3], dtype=float)
        v = np.asarray(y[3:6], dtype=float)
        mass = float(y[6]) if (y.size > 6) else float(self.sc_props.mass_kg)

        mu_m = float(gp.gm_m3s2)  # consistent with RHS

        out: dict[str, float] = {}

        # Ephemeris (Python-side fetch)
        sun = np.zeros(3, dtype=float)
        earth = np.zeros(3, dtype=float)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        have_eph = bool(self.ephem is not None)
        need_eph = bool(have_eph and (req.need_sun or req.need_earth or req.need_q))
        if need_eph:
            sx, sy, sz, ex, ey, ez, q0, q1, q2, q3 = get_ephem_state(
                float(t),
                float(ep.dt_s),
                np.ascontiguousarray(ep.r_sun_tab_m, dtype=np.float64),
                np.ascontiguousarray(ep.r_earth_tab_m, dtype=np.float64),
                np.ascontiguousarray(ep.q_i2f_tab, dtype=np.float64),
            )
            sun[:] = (sx, sy, sz)
            earth[:] = (ex, ey, ez)
            q[:] = (q0, q1, q2, q3)

        def _norm3(ax: float, ay: float, az: float) -> float:
            return float(math.sqrt(ax * ax + ay * ay + az * az))

        # Gravity
        if req.use_sh:
            if bool(req.use_surrogate_gravity):
                rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])
                ax_f, ay_f, az_f = self.grav.acceleration_fixed((rfx, rfy, rfz))
                ax_i, ay_i, az_i = quat_rotate_vec(
                    q[0], -q[1], -q[2], -q[3], float(ax_f), float(ay_f), float(az_f)
                )
                out["Gravity (ST-LRPS)"] = _norm3(ax_i, ay_i, az_i)
            else:
                rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])

                # Copy workspace to avoid contaminating runtime scratch
                WP = np.ascontiguousarray(gp.ws_P, dtype=np.float64).copy()
                WDP = np.ascontiguousarray(gp.ws_dP, dtype=np.float64).copy()
                WC = np.ascontiguousarray(gp.ws_cos_m, dtype=np.float64).copy()
                WS = np.ascontiguousarray(gp.ws_sin_m, dtype=np.float64).copy()
                n_eval = int(gp.nmax)
                if bool(gp.adaptive_enabled):
                    n_eval = _select_adaptive_sh_degree(
                        float(np.linalg.norm(r)),
                        float(gp.r_ref_m),
                        int(gp.nmax),
                        int(gp.adaptive_mode),
                        float(gp.adaptive_power),
                        int(gp.adaptive_min_degree),
                        int(gp.adaptive_quantization_step),
                        np.ascontiguousarray(gp.adaptive_table_alt_km, dtype=np.float64),
                        np.ascontiguousarray(gp.adaptive_table_degree, dtype=np.int64),
                        int(gp.adaptive_table_len),
                    )

                ax_f, ay_f, az_f = sh_accel_fixed_numba(
                    rfx,
                    rfy,
                    rfz,
                    n_eval,
                    float(gp.r_ref_m),
                    float(gp.gm_m3s2),
                    gp.Cnm,
                    gp.Snm,
                    gp.diag,
                    gp.subdiag,
                    gp.A,
                    gp.B,
                    gp.scale_m,
                    WP,
                    WDP,
                    WC,
                    WS,
                )
                ax_i, ay_i, az_i = quat_rotate_vec(q[0], -q[1], -q[2], -q[3], ax_f, ay_f, az_f)
                out["Gravity (SH)"] = _norm3(ax_i, ay_i, az_i)
        else:
            ax0, ay0, az0 = compute_point_mass_acceleration(r[0], r[1], r[2], float(mu_m))
            out["Gravity (PM)"] = _norm3(ax0, ay0, az0)

        # Third body
        if req.use_3rd_sun:
            ax3, ay3, az3 = accel_third_body_numba(r[0], r[1], r[2], sun[0], sun[1], sun[2], float(MU_SUN))
            out["3rd Body (Sun)"] = _norm3(ax3, ay3, az3)

        if req.use_3rd_earth:
            ax3, ay3, az3 = accel_third_body_numba(r[0], r[1], r[2], earth[0], earth[1], earth[2], float(MU_EARTH))
            out["3rd Body (Earth)"] = _norm3(ax3, ay3, az3)

        if req.use_earth_j2:
            j2x, j2y, j2z = accel_j2_oblate_diff_numba(
                float(r[0]),
                float(r[1]),
                float(r[2]),
                float(earth[0]),
                float(earth[1]),
                float(earth[2]),
                float(MU_EARTH),
                float(ej.r_ref_m),
                float(ej.j2),
                float(ej.ax),
                float(ej.ay),
                float(ej.az),
            )
            out["3rd Body (Earth J2)"] = _norm3(j2x, j2y, j2z)

        # Solid-body tides
        if req.use_tides:
            rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])
            tide_x = 0.0
            tide_y = 0.0
            tide_z = 0.0

            if req.use_tide_earth:
                efx, efy, efz = quat_rotate_vec(q[0], q[1], q[2], q[3], earth[0], earth[1], earth[2])
                atx_f, aty_f, atz_f = accel_solid_tides_numba(
                    rfx,
                    rfy,
                    rfz,
                    efx,
                    efy,
                    efz,
                    float(MU_EARTH),
                    float(tp.r_ref_m),
                    float(tp.k2),
                    float(tp.k3),
                    bool(tp.use_k2),
                    bool(tp.use_k3),
                )
                atx_i, aty_i, atz_i = quat_rotate_vec(q[0], -q[1], -q[2], -q[3], atx_f, aty_f, atz_f)
                earth_norm = _norm3(atx_i, aty_i, atz_i)
                out["Solid Tides (Earth)"] = earth_norm
                tide_x += atx_i
                tide_y += aty_i
                tide_z += atz_i

            if req.use_tide_sun:
                sfx, sfy, sfz = quat_rotate_vec(q[0], q[1], q[2], q[3], sun[0], sun[1], sun[2])
                atx_f, aty_f, atz_f = accel_solid_tides_numba(
                    rfx,
                    rfy,
                    rfz,
                    sfx,
                    sfy,
                    sfz,
                    float(MU_SUN),
                    float(tp.r_ref_m),
                    float(tp.k2),
                    float(tp.k3),
                    bool(tp.use_k2),
                    bool(tp.use_k3),
                )
                atx_i, aty_i, atz_i = quat_rotate_vec(q[0], -q[1], -q[2], -q[3], atx_f, aty_f, atz_f)
                sun_norm = _norm3(atx_i, aty_i, atz_i)
                out["Solid Tides (Sun)"] = sun_norm
                tide_x += atx_i
                tide_y += aty_i
                tide_z += atz_i

            out["Solid Tides"] = _norm3(tide_x, tide_y, tide_z)

        # SRP
        if req.use_srp:
            srp_cfg = self.srp
            earth_r2 = float(earth[0] * earth[0] + earth[1] * earth[1] + earth[2] * earth[2])
            enable_earth = bool(getattr(srp_cfg, "enable_earth_eclipse", False)) and bool(earth_r2 > 1.0e12)

            asx, asy, asz = accel_srp(
                r[0],
                r[1],
                r[2],
                sun[0],
                sun[1],
                sun[2],
                earth[0],
                earth[1],
                earth[2],
                float(getattr(srp_cfg, "R_moon_m", R_MOON)),
                float(getattr(srp_cfg, "R_earth_m", R_EARTH_MEAN)),
                float(getattr(srp_cfg, "AU_m", AU)),
                float(getattr(srp_cfg, "P0", P_SUN_1AU)),
                float(self.sc_props.cr),
                float(self.sc_props.area_m2),
                float(mass),
                bool(getattr(srp_cfg, "enable_moon_eclipse", True)),
                enable_earth,
            )
            out["SRP"] = _norm3(asx, asy, asz)

        # Albedo (reflected solar radiation pressure)
        if req.use_albedo:
            rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])
            sfx, sfy, sfz = quat_rotate_vec(q[0], q[1], q[2], q[3], sun[0], sun[1], sun[2])

            if ap.backend == 1:
                efx, efy, efz = quat_rotate_vec(q[0], q[1], q[2], q[3], earth[0], earth[1], earth[2])
                aax_f, aay_f, aaz_f = accel_albedo_facets_numba(
                    rfx,
                    rfy,
                    rfz,
                    sfx,
                    sfy,
                    sfz,
                    efx,
                    efy,
                    efz,
                    np.ascontiguousarray(ap.facet_pos_m, dtype=np.float64),
                    np.ascontiguousarray(ap.facet_normals, dtype=np.float64),
                    np.ascontiguousarray(ap.facet_areas_m2, dtype=np.float64),
                    np.ascontiguousarray(ap.facet_albedo, dtype=np.float64),
                    float(ap.pressure_coefficient),
                    float(self.sc_props.area_m2),
                    float(mass),
                    float(ap.solar_flux_1au_W_m2),
                    float(ap.au_m),
                    float(ap.c_light_m_s),
                    float(ap.r_earth_m),
                    bool(ap.include_sun_distance_scaling),
                    bool(ap.enable_eclipse),
                )
            else:
                lat_deg, lon_deg, _ = latlon_from_xyz_m(rfx, rfy, rfz)

                aval = float(ap.alb_const)
                if ap.mode == 0 and ap.grid_alb is not None:
                    aval = float(
                        sample_grid_bilinear(
                            lat_deg,
                            lon_deg,
                            np.ascontiguousarray(ap.grid_alb, dtype=np.float64),
                            int(ap.n_lines),
                            int(ap.n_samples),
                            float(ap.res_deg),
                            float(ap.lon0_deg),
                            float(ap.lat0_deg),
                        )
                    )
                elif ap.mode == 1 and ap.dn is not None:
                    aval = float(
                        _sample_albedo_dn_scaled(
                            lat_deg,
                            lon_deg,
                            np.ascontiguousarray(ap.dn, dtype=np.float64),
                            int(ap.n_lines),
                            int(ap.n_samples),
                            float(ap.res_deg),
                            float(ap.lon0_deg),
                            float(ap.lat0_deg),
                            int(ap.flip),
                            float(ap.sf),
                            float(ap.off),
                            float(ap.missing),
                            float(ap.latmin),
                            float(ap.latmax),
                        )
                    )

                if math.isnan(aval):
                    aval = float(ap.alb_const)
                aval = max(0.0, min(1.0, aval)) * float(ap.alb_scale)

                # Use the same SRPConfig-derived constants as the runtime RHS
                # (which reads R_moon_m / AU_m / P0 from self.srp), so the
                # debug breakdown cannot diverge from the integrated force.
                _srp_cfg = self.srp
                aax_f, aay_f, aaz_f = accel_albedo_simple(
                    rfx,
                    rfy,
                    rfz,
                    sfx,
                    sfy,
                    sfz,
                    float(getattr(_srp_cfg, "R_moon_m", R_MOON)),
                    float(getattr(_srp_cfg, "AU_m", AU)),
                    float(getattr(_srp_cfg, "P0", P_SUN_1AU)),
                    float(aval),
                    float(ap.k_lambert),
                    float(self.sc_props.cr),
                    float(self.sc_props.area_m2),
                    float(mass),
                    1,
                )

            aax_i, aay_i, aaz_i = quat_rotate_vec(q[0], -q[1], -q[2], -q[3], aax_f, aay_f, aaz_f)
            out["Albedo"] = _norm3(aax_i, aay_i, aaz_i)

        # Lunar thermal IR radiation pressure
        if req.use_thermal:
            rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])
            sfx, sfy, sfz = quat_rotate_vec(q[0], q[1], q[2], q[3], sun[0], sun[1], sun[2])
            efx, efy, efz = quat_rotate_vec(q[0], q[1], q[2], q[3], earth[0], earth[1], earth[2])
            athx_f, athy_f, athz_f = accel_thermal_ir_facets_numba(
                rfx,
                rfy,
                rfz,
                sfx,
                sfy,
                sfz,
                efx,
                efy,
                efz,
                np.ascontiguousarray(th.facet_pos_m, dtype=np.float64),
                np.ascontiguousarray(th.facet_normals, dtype=np.float64),
                np.ascontiguousarray(th.facet_areas_m2, dtype=np.float64),
                np.ascontiguousarray(th.facet_temperatures_K, dtype=np.float64),
                int(th.mode),
                float(th.surface_emissivity),
                float(th.surface_albedo),
                float(th.temperature_K),
                float(th.night_temperature_K),
                float(th.thermal_floor_flux_W_m2),
                float(th.ir_pressure_coefficient),
                float(self.sc_props.area_m2),
                float(mass),
                float(th.solar_flux_1au_W_m2),
                float(th.au_m),
                float(th.c_light_m_s),
                float(th.sigma_sb),
                bool(th.include_sun_distance_scaling),
                float(th.r_earth_m),
                bool(th.enable_eclipse),
            )
            athx_i, athy_i, athz_i = quat_rotate_vec(q[0], -q[1], -q[2], -q[3], athx_f, athy_f, athz_f)
            out["Thermal IR"] = _norm3(athx_i, athy_i, athz_i)

        # Relativity
        if req.use_rel:
            arx, ary, arz = _schwarzschild_components(r[0], r[1], r[2], v[0], v[1], v[2], float(mu_m))
            rel_x = arx
            rel_y = ary
            rel_z = arz
            out["Relativity (Moon Schwarzschild)"] = _norm3(arx, ary, arz)

            if req.use_rel_external:
                svx, svy, svz = interp_vec3_derivative_safe(
                    float(t),
                    float(ep.dt_s),
                    np.ascontiguousarray(ep.r_sun_tab_m, dtype=np.float64),
                )
                exx, exy, exz = _external_1pn_components(
                    float(r[0]), float(r[1]), float(r[2]),
                    float(v[0]), float(v[1]), float(v[2]),
                    float(sun[0]), float(sun[1]), float(sun[2]),
                    float(svx), float(svy), float(svz),
                    float(MU_SUN),
                )
                evx, evy, evz = interp_vec3_derivative_safe(
                    float(t),
                    float(ep.dt_s),
                    np.ascontiguousarray(ep.r_earth_tab_m, dtype=np.float64),
                )
                eex, eey, eez = _external_1pn_components(
                    float(r[0]), float(r[1]), float(r[2]),
                    float(v[0]), float(v[1]), float(v[2]),
                    float(earth[0]), float(earth[1]), float(earth[2]),
                    float(evx), float(evy), float(evz),
                    float(MU_EARTH),
                )
                ext_x = exx + eex
                ext_y = exy + eey
                ext_z = exz + eez
                out["Relativity (External 1PN)"] = _norm3(ext_x, ext_y, ext_z)
                rel_x += ext_x
                rel_y += ext_y
                rel_z += ext_z

            out["Relativity (1PN)"] = _norm3(rel_x, rel_y, rel_z)

        return out



# =============================================================================
# 4.                                PUBLIC API
# =============================================================================

__all__ = (
    # Main engine
    "DynamicsEngine",
)
