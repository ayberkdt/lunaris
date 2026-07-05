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
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit

from lunaris.common.constants import (
    AU,
    C_LIGHT,
    MU_EARTH,
    MU_MOON,
    MU_SUN,
    P_SUN_1AU,
    R_EARTH_MEAN,
    R_MOON,
    SIGMA_SB,
    SOLAR_FLUX_1AU,
)
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
from lunaris.core.dynamics.requirements import (
    extract_ephem_tables_strict,
    extract_gravity_strict,
    extract_surface_provider_strict,
)
from lunaris.core.dynamics.surrogate_bridge import _is_surrogate_gravity_provider
from lunaris.physics.ephemeris import get_ephem_state, interp_vec3_derivative_safe
from lunaris.physics.lunar_albedo import (
    ALBEDO_SOURCE_CONSTANT,
    ALBEDO_SOURCE_GRID,
    accel_albedo_facets_numba,
    normalize_albedo_mode,
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
    THERMAL_MODE_CONSTANT,
    THERMAL_MODE_EQUILIBRIUM,
    THERMAL_MODE_TEMPERATURE_GRID,
    accel_thermal_ir_facets_numba,
    build_latlon_facets,
    normalize_thermal_mode,
)
from lunaris.physics.third_body_effects import accel_j2_oblate_diff_numba, accel_third_body_numba

logger = logging.getLogger(__name__)

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
        self.allow_identity_rotation = bool(allow_identity_rotation)

        self._rhs_cache: Callable[[float, np.ndarray], np.ndarray] | None = None
        self._prep: dict[str, Any] = {}  # debug/reporting packs + requirements

        self._validate_dependencies()

    # -------------------------------------------------------------------------
    # Requirements / validation
    # -------------------------------------------------------------------------
    def _requirements(self) -> dict[str, Any]:
        f = self.flags

        use_sh = bool(getattr(f, "enable_sh", False))
        use_surrogate_gravity = bool(use_sh and _is_surrogate_gravity_provider(self.grav))
        use_albedo = bool(getattr(f, "enable_albedo", False))

        alb_cfg = self.albedo if self.albedo is not None else AlbedoConfig()
        albedo_model = str(getattr(alb_cfg, "albedo_model", "lambert_facets")).strip().lower()
        albedo_source = normalize_albedo_mode(getattr(alb_cfg, "albedo_mode", "constant_albedo"))
        albedo_needs_provider = bool(
            use_albedo
            and (
                albedo_source != ALBEDO_SOURCE_CONSTANT
                or bool(getattr(alb_cfg, "require_surface_provider", False))
            )
        )

        use_thermal = bool(getattr(f, "enable_thermal", getattr(f, "enable_thermal_ir", False)))
        use_srp = bool(getattr(f, "enable_srp", False))
        use_3rd_sun = bool(getattr(f, "enable_3rd_body_sun", False))
        use_3rd_earth = bool(getattr(f, "enable_3rd_body_earth", False))
        use_tides_k2 = bool(getattr(f, "enable_tides_k2", False))
        use_tides_k3 = bool(getattr(f, "enable_tides_k3", False))
        use_tides = bool(use_tides_k2 or use_tides_k3)

        # Backwards compatibility: enable_relativity_1pn or enable_relativity
        use_rel = bool(getattr(f, "enable_relativity_1pn", getattr(f, "enable_relativity", False)))
        # Design (#4): there is intentionally no separate "external relativity"
        # flag. The external-body Schwarzschild differential and the de Sitter
        # geodetic term are the physically dominant relativistic effects for a
        # lunar orbiter, but they can only be evaluated when Sun/Earth ephemeris
        # states are available. So they auto-enable whenever 1PN relativity is on
        # AND an ephemeris is present, and silently degrade to the central-body
        # Schwarzschild term alone otherwise. The body velocities they require are
        # a central finite difference of the ephemeris position table
        # (interp_vec3_derivative_safe, O(dt^2), the same scheme used on the GPU
        # via _interp3_derivative_cuda) rather than the Catmull-Rom position
        # interpolant's analytic derivative — adequate for the ~1e-11 m/s^2 term.
        use_rel_external = bool(use_rel and self.ephem is not None)

        use_earth_j2_flag = bool(getattr(f, "enable_earth_j2", False))
        use_earth_j2 = bool(use_earth_j2_flag and (self.earth_j2 is not None))
        tide_bodies = getattr(self.solid_tides, "tide_bodies", ("earth", "sun"))
        use_tide_earth = bool(use_tides and ("earth" in tide_bodies))
        use_tide_sun = bool(use_tides and ("sun" in tide_bodies))
        thermal_mode = normalize_thermal_mode(getattr(self.thermal, "thermal_mode", "constant_temperature"))
        use_thermal_equilibrium = bool(use_thermal and thermal_mode == THERMAL_MODE_EQUILIBRIUM)
        use_thermal_grid = bool(use_thermal and thermal_mode == THERMAL_MODE_TEMPERATURE_GRID)
        use_thermal_eclipse = bool(
            use_thermal_equilibrium
            and bool(getattr(self.thermal, "enable_eclipse", True))
        )

        # What data do we need from ephemeris?
        need_sun = bool(
            use_srp
            or use_3rd_sun
            or use_albedo
            or use_tide_sun
            or use_thermal_equilibrium
            or use_rel_external
        )
        need_earth = bool(
            use_3rd_earth
            or use_earth_j2
            or use_tide_earth
            or use_thermal_eclipse
            or use_rel_external
        )
        need_q = bool(use_sh or use_surrogate_gravity or use_albedo or use_tides or use_thermal)

        # Ephemeris manager is required if Sun/Earth vectors are needed.
        need_vectors = bool(need_sun or need_earth)

        # Quaternion can be substituted with identity if explicitly allowed.
        need_quat_from_ephem = bool(need_q and (not self.allow_identity_rotation))

        need_ephem = bool(need_vectors or need_quat_from_ephem)

        return {
            "use_sh": use_sh,
            "use_surrogate_gravity": use_surrogate_gravity,
            "use_albedo": use_albedo,
            "albedo_model": albedo_model,
            "albedo_needs_provider": albedo_needs_provider,
            "use_thermal": use_thermal,
            "use_thermal_equilibrium": use_thermal_equilibrium,
            "use_thermal_eclipse": use_thermal_eclipse,
            "use_thermal_grid": use_thermal_grid,
            "use_srp": use_srp,
            "use_3rd_sun": use_3rd_sun,
            "use_3rd_earth": use_3rd_earth,
            "use_tides": use_tides,
            "use_tides_k2": use_tides_k2,
            "use_tides_k3": use_tides_k3,
            "use_tide_earth": use_tide_earth,
            "use_tide_sun": use_tide_sun,
            "use_rel": use_rel,
            "use_rel_external": use_rel_external,
            "use_earth_j2": use_earth_j2,
            "need_sun": need_sun,
            "need_earth": need_earth,
            "need_q": need_q,
            "need_vectors": need_vectors,
            "need_quat_from_ephem": need_quat_from_ephem,
            "need_ephem": need_ephem,
        }

    def _validate_dependencies(self) -> None:
        req = self._requirements()

        if req["use_sh"] and self.grav is None:
            raise ValueError("enable_sh=True but gravity_model is None.")

        if req["albedo_needs_provider"] and self.surf is None:
            cfg = self.albedo if self.albedo is not None else AlbedoConfig()
            raise ValueError(
                f"enable_albedo with albedo_mode={cfg.albedo_mode!r} requires a "
                "surface_provider supplying an albedo grid, but none was provided. "
                "Use albedo_mode='constant_albedo' for a provider-free model, or "
                "supply a surface_provider (e.g. --albedo-root)."
            )

        if req["use_thermal_grid"] and self.surf is None:
            raise ValueError(
                "ThermalConfig.thermal_mode='temperature_grid' requires surface_provider "
                "with thermal_temperature_cells_K, temperature_cells_K, or temperature_grid."
            )

        # SRP / Albedo / Thermal IR require valid spacecraft optical area and mass
        if req["use_srp"] or req["use_albedo"] or req["use_thermal"]:
            if self.sc_props.mass_kg <= 0.0:
                raise ValueError(f"mass_kg must be > 0, got {self.sc_props.mass_kg}")
            if self.sc_props.area_m2 <= 0.0:
                raise ValueError(f"area_m2 must be > 0 for SRP/Albedo/Thermal IR, got {self.sc_props.area_m2}")
        # The SRP and simple/legacy albedo backends reuse the spacecraft SRP
        # coefficient cr; the lambert_facets albedo backend does not (it uses
        # albedo_pressure_coefficient), so cr is only enforced when actually used.
        if req["use_srp"] or (req["use_albedo"] and req["albedo_model"] == "simple"):
            if not (0.0 < self.sc_props.cr <= 2.5):
                raise ValueError(f"cr looks invalid, got {self.sc_props.cr}")

        if req["need_ephem"] and (self.ephem is None):
            reasons: list[str] = []
            if req["need_vectors"]:
                if req["need_sun"]:
                    reasons.append("Sun vector (SRP / 3rd-body Sun / Albedo / Thermal IR equilibrium / solid tides)")
                if req["need_earth"]:
                    reasons.append("Earth vector (3rd-body Earth / Earth J2 / Thermal IR eclipse / solid tides)")
            if req["need_quat_from_ephem"]:
                reasons.append("q_i2f (SH / Albedo / Thermal IR / solid tides)")

            why = "; ".join(reasons) if reasons else "Sun/Earth vectors and/or q_i2f"
            raise ValueError(
                f"Ephemeris is required for selected perturbations: {why}. "
                "Provide ephem_manager, or disable those perturbations. "
                "Note: allow_identity_rotation only replaces q_i2f, not Sun/Earth vectors."
            )

        f = self.flags
        if req["use_tides_k3"] and getattr(self.solid_tides, "k3", None) is None:
            raise ValueError(
                "enable_tides_k3=True requires solid_tides.k3 to be set explicitly; "
                "no degree-3 lunar Love number default is assumed."
            )

        if bool(getattr(f, "enable_earth_j2", False)) and (self.earth_j2 is None):
            raise ValueError(
                "enable_earth_j2=True but earth_j2 params are None."
            )

    # -------------------------------------------------------------------------
    # Providers -> prepared packs (strict)
    # -------------------------------------------------------------------------
    def _prepare_adaptive_gravity_policy(self, nmax: int) -> dict[str, Any]:
        """
        Normalize optional adaptive-degree settings into kernel-friendly arrays.

        The backend config already validates table ordering, but the dynamics
        layer still clamps each requested degree to the loaded model's actual
        maximum. This keeps runtime evaluation robust even when older session
        files or UI presets request degrees higher than the active gravity file.
        """

        adaptive = self.gravity_adaptive
        if adaptive is None and self.grav is not None:
            adaptive = getattr(self.grav, "adaptive", None)

        disabled = {
            "adaptive_enabled": False,
            "adaptive_mode": 0,
            "adaptive_power": 2.5,
            "adaptive_min_degree": max(0, min(int(nmax), 4)),
            "adaptive_quantization_step": 1,
            "adaptive_table_alt_km": np.zeros(1, dtype=np.float64),
            "adaptive_table_degree": np.zeros(1, dtype=np.int64),
            "adaptive_table_len": 0,
        }
        if adaptive is None or not bool(getattr(adaptive, "enabled", False)) or int(nmax) <= 0:
            return disabled

        min_degree = max(0, min(int(nmax), int(getattr(adaptive, "min_degree", 4) or 4)))
        quant_step = max(1, int(getattr(adaptive, "quantization_step", 10) or 10))
        power = float(getattr(adaptive, "power", 2.5) or 2.5)

        raw_table = getattr(adaptive, "altitude_table", None)
        if raw_table:
            parsed_rows: list[tuple[float, int]] = []
            skipped_rows: list[Any] = []
            for row in raw_table:
                try:
                    alt_km = max(0.0, float(row[0]))
                    degree = max(min_degree, min(int(nmax), int(row[1])))
                except (TypeError, ValueError, IndexError):
                    # Malformed user row (wrong arity / non-numeric). Dropping it
                    # changes the effective gravity degree schedule, so it must
                    # never be silent (R29b).
                    skipped_rows.append(row)
                    continue
                parsed_rows.append((alt_km, degree))
            if skipped_rows:
                warnings.warn(
                    f"Adaptive SH degree table: skipped {len(skipped_rows)} malformed "
                    f"row(s) {skipped_rows!r}; the remaining rows define the degree "
                    "schedule. Fix the altitude_table entries.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            parsed_rows.sort(key=lambda item: item[0])
            cleaned_rows: list[tuple[float, int]] = []
            prev_alt = -1.0
            for alt_km, degree in parsed_rows:
                if alt_km <= prev_alt:
                    continue
                cleaned_rows.append((alt_km, degree))
                prev_alt = alt_km

            if cleaned_rows:
                return {
                    "adaptive_enabled": True,
                    "adaptive_mode": 1,
                    "adaptive_power": power,
                    "adaptive_min_degree": min_degree,
                    "adaptive_quantization_step": quant_step,
                    "adaptive_table_alt_km": np.ascontiguousarray(
                        np.asarray([row[0] for row in cleaned_rows], dtype=np.float64)
                    ),
                    "adaptive_table_degree": np.ascontiguousarray(
                        np.asarray([row[1] for row in cleaned_rows], dtype=np.int64)
                    ),
                    "adaptive_table_len": len(cleaned_rows),
                }

        return {
            "adaptive_enabled": True,
            "adaptive_mode": 2,
            "adaptive_power": power,
            "adaptive_min_degree": min_degree,
            "adaptive_quantization_step": quant_step,
            "adaptive_table_alt_km": np.zeros(1, dtype=np.float64),
            "adaptive_table_degree": np.zeros(1, dtype=np.int64),
            "adaptive_table_len": 0,
        }

    def _prepare_gravity(self) -> _GravPack:
        if self.grav is None:
            z11 = np.zeros((1, 1), dtype=np.float64)
            z1 = np.zeros(1, dtype=np.float64)
            return _GravPack(
                nmax=0,
                r_ref_m=float(R_MOON),
                gm_m3s2=float(MU_MOON),
                Cnm=z11,
                Snm=z11,
                diag=z1,
                subdiag=z1,
                A=z1,
                B=z1,
                scale_m=z1,
                ws_P=z11,
                ws_dP=z11,
                ws_cos_m=z1,
                ws_sin_m=z1,
                adaptive_enabled=False,
                adaptive_mode=0,
                adaptive_power=2.5,
                adaptive_min_degree=0,
                adaptive_quantization_step=1,
                adaptive_table_alt_km=np.zeros(1, dtype=np.float64),
                adaptive_table_degree=np.zeros(1, dtype=np.int64),
                adaptive_table_len=0,
            )

        if _is_surrogate_gravity_provider(self.grav):
            z11 = np.zeros((1, 1), dtype=np.float64)
            z1 = np.zeros(1, dtype=np.float64)
            return _GravPack(
                nmax=0,
                r_ref_m=float(getattr(self.grav, "R_ref_m", getattr(self.grav, "r_ref_m", R_MOON))),
                gm_m3s2=float(getattr(self.grav, "GM_m3s2", getattr(self.grav, "gm_m3s2", MU_MOON))),
                Cnm=z11,
                Snm=z11,
                diag=z1,
                subdiag=z1,
                A=z1,
                B=z1,
                scale_m=np.ones(1, dtype=np.float64),
                ws_P=z11,
                ws_dP=z11,
                ws_cos_m=np.ones(1, dtype=np.float64),
                ws_sin_m=np.zeros(1, dtype=np.float64),
                adaptive_enabled=False,
                adaptive_mode=0,
                adaptive_power=2.5,
                adaptive_min_degree=0,
                adaptive_quantization_step=1,
                adaptive_table_alt_km=np.zeros(1, dtype=np.float64),
                adaptive_table_degree=np.zeros(1, dtype=np.int64),
                adaptive_table_len=0,
            )

        nmax, r_ref, gm, Cnm, Snm, diag, subdiag, A, B, scale_m, ws_P, ws_dP, ws_cos, ws_sin = extract_gravity_strict(
            self.grav
        )
        adaptive_policy = self._prepare_adaptive_gravity_policy(int(nmax))

        return _GravPack(
            nmax=int(nmax),
            r_ref_m=float(r_ref),
            gm_m3s2=float(gm),
            Cnm=Cnm,
            Snm=Snm,
            diag=diag,
            subdiag=subdiag,
            A=A,
            B=B,
            scale_m=scale_m,
            ws_P=ws_P,
            ws_dP=ws_dP,
            ws_cos_m=ws_cos,
            ws_sin_m=ws_sin,
            adaptive_enabled=bool(adaptive_policy["adaptive_enabled"]),
            adaptive_mode=int(adaptive_policy["adaptive_mode"]),
            adaptive_power=float(adaptive_policy["adaptive_power"]),
            adaptive_min_degree=int(adaptive_policy["adaptive_min_degree"]),
            adaptive_quantization_step=int(adaptive_policy["adaptive_quantization_step"]),
            adaptive_table_alt_km=adaptive_policy["adaptive_table_alt_km"],
            adaptive_table_degree=adaptive_policy["adaptive_table_degree"],
            adaptive_table_len=int(adaptive_policy["adaptive_table_len"]),
        )

    def _prepare_ephem(self, req: dict[str, bool]) -> _EphemPack:
        if self.ephem is None:
            # Only valid if we do NOT need Sun/Earth vectors AND we allow identity rotation for q_i2f.
            q_ident = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
            z23 = np.zeros((2, 3), dtype=np.float64)
            return _EphemPack(dt_s=1.0, r_sun_tab_m=z23, r_earth_tab_m=z23, q_i2f_tab=q_ident)

        dt_s, sun_tab, earth_tab, qtab = extract_ephem_tables_strict(self.ephem)

        # Fail closed on degenerate body tables. An ephemeris built with
        # include_third_body=False stores all-zero Sun/Earth rows; feeding those
        # into an enabled Sun/Earth-dependent force does not fail loudly -- the
        # third-body/J2 kernels guard-return zero (the force silently vanishes)
        # and SRP evaluates with the Sun at the Moon's center (catastrophically
        # wrong magnitude). SimConfig.validate() catches this at the config
        # layer; this guard covers direct DynamicsEngine construction.
        #
        # Only *explicitly enabled* forces raise. The external-1PN relativity
        # terms are auto-enabled from `ephem is not None` and are documented to
        # silently degrade to the central-body Schwarzschild term when Sun/Earth
        # vectors are unavailable, so a quaternion-only ephemeris downgrades
        # them (with a warning) instead of failing the run.
        explicit_sun = bool(
            req["use_srp"]
            or req["use_3rd_sun"]
            or req["use_albedo"]
            or req["use_tide_sun"]
            or req["use_thermal_equilibrium"]
        )
        explicit_earth = bool(
            req["use_3rd_earth"]
            or req["use_earth_j2"]
            or req["use_tide_earth"]
            or req["use_thermal_eclipse"]
        )
        sun_degenerate = not np.any(sun_tab)
        earth_degenerate = not np.any(earth_tab)

        if explicit_sun and sun_degenerate:
            raise ValueError(
                "Enabled perturbations require the Sun position (SRP / 3rd-body Sun / "
                "Albedo / Thermal IR equilibrium / solid tides), but the ephemeris Sun "
                "table is all zeros. Rebuild the ephemeris with include_third_body=True, "
                "or disable those perturbations."
            )
        if explicit_earth and earth_degenerate:
            raise ValueError(
                "Enabled perturbations require the Earth position (3rd-body Earth / "
                "Earth J2 / Thermal IR eclipse / solid tides), but the ephemeris Earth "
                "table is all zeros. Rebuild the ephemeris with include_third_body=True, "
                "or disable those perturbations."
            )
        if req["use_rel_external"] and (sun_degenerate or earth_degenerate):
            req["use_rel_external"] = False
            warnings.warn(
                "1PN external-body relativity terms disabled: the ephemeris does not "
                "provide Sun/Earth position tables (all zeros). Only the central-body "
                "Schwarzschild term remains active.",
                RuntimeWarning,
                stacklevel=3,
            )

        return _EphemPack(dt_s=float(dt_s), r_sun_tab_m=sun_tab, r_earth_tab_m=earth_tab, q_i2f_tab=qtab)

    def _prepare_albedo(self, req: dict[str, bool]) -> _AlbedoPack:
        if not req["use_albedo"]:
            return _AlbedoPack(backend=1, mode=2, alb_const=0.12, alb_scale=1.0, k_lambert=1.0)

        cfg = self.albedo if self.albedo is not None else AlbedoConfig()
        model = str(getattr(cfg, "albedo_model", "lambert_facets")).strip().lower()

        if model == "simple":
            return self._prepare_albedo_simple(cfg)

        # --- lambert_facets (default): build facets + per-facet albedo once. ---
        lat_count = int(getattr(cfg, "facet_lat_count", 18))
        lon_count = int(getattr(cfg, "facet_lon_count", 36))
        pos, normals, areas, lat_c, lon_c = build_latlon_facets(
            lat_count, lon_count, radius_m=float(R_MOON)
        )

        source = normalize_albedo_mode(getattr(cfg, "albedo_mode", "constant_albedo"))
        fallback = float(getattr(cfg, "albedo_const", 0.12))

        if source == ALBEDO_SOURCE_CONSTANT:
            facet_albedo = np.full(areas.shape[0], fallback, dtype=np.float64)
        else:
            if self.surf is None:
                raise ValueError(
                    f"albedo_mode={cfg.albedo_mode!r} requires a surface_provider "
                    "supplying an albedo grid, but none was provided."
                )
            surf = extract_surface_provider_strict(self.surf)
            facet_albedo = self._sample_facet_albedo_from_provider(
                surf, source, lat_c, lon_c, fallback
            )

        np.clip(facet_albedo, 0.0, 1.0, out=facet_albedo)

        return _AlbedoPack(
            backend=1,
            mode=2,
            facet_pos_m=pos,
            facet_normals=normals,
            facet_areas_m2=areas,
            facet_albedo=np.ascontiguousarray(facet_albedo, dtype=np.float64),
            pressure_coefficient=float(getattr(cfg, "albedo_pressure_coefficient", 1.0)),
            solar_flux_1au_W_m2=float(getattr(cfg, "solar_flux_1au_W_m2", SOLAR_FLUX_1AU)),
            au_m=float(getattr(cfg, "AU_m", AU)),
            c_light_m_s=float(getattr(cfg, "c_light_m_s", C_LIGHT)),
            r_earth_m=float(R_EARTH_MEAN),
            include_sun_distance_scaling=bool(getattr(cfg, "include_sun_distance_scaling", True)),
            enable_eclipse=bool(getattr(cfg, "enable_eclipse", True)),
        )

    def _prepare_albedo_simple(self, cfg: AlbedoConfig) -> _AlbedoPack:
        """Legacy cannonball backend: parameters sourced from the surface provider.

        Reproduces the historical albedo behavior (single sub-satellite albedo
        sample, Sun->spacecraft push, SRP coefficient ``cr``). Works without a
        provider using a constant albedo from the config.
        """
        const = float(getattr(cfg, "albedo_const", 0.12))
        klamb = float(getattr(cfg, "k_lambert", 1.0))
        if self.surf is None:
            return _AlbedoPack(backend=0, mode=2, alb_const=const, alb_scale=1.0, k_lambert=klamb)

        surf = extract_surface_provider_strict(self.surf)
        alb_const = float(surf.get("albedo_const", const))
        alb_scale = float(surf.get("scale", 1.0))
        k_lambert = float(surf.get("k_lambert", klamb))

        if "albedo_grid" in surf:
            return _AlbedoPack(
                backend=0, mode=0, alb_const=alb_const, alb_scale=alb_scale, k_lambert=k_lambert,
                grid_alb=surf["albedo_grid"],
                n_lines=int(surf["n_lines"]), n_samples=int(surf["n_samples"]),
                res_deg=float(surf["res_deg"]), lon0_deg=float(surf["lon0_deg"]), lat0_deg=float(surf["lat0_deg"]),
            )
        if "dn" in surf:
            return _AlbedoPack(
                backend=0, mode=1, alb_const=alb_const, alb_scale=alb_scale, k_lambert=k_lambert,
                dn=surf["dn"],
                n_lines=int(surf["n_lines"]), n_samples=int(surf["n_samples"]),
                res_deg=float(surf["res_deg"]), lon0_deg=float(surf["lon0_deg"]), lat0_deg=float(surf["lat0_deg"]),
                sf=float(surf.get("scale_factor", 1.0)), off=float(surf.get("offset", 0.0)),
                missing=float(surf.get("missing_dn", -1.0)), flip=int(surf.get("flip_lat", 0)),
                latmin=float(surf.get("lat_min_deg", -90.0)), latmax=float(surf.get("lat_max_deg", 90.0)),
            )
        return _AlbedoPack(backend=0, mode=2, alb_const=alb_const, alb_scale=alb_scale, k_lambert=k_lambert)

    def _sample_facet_albedo_from_provider(
        self,
        surf: dict[str, Any],
        source: int,
        lat_c_rad: np.ndarray,
        lon_c_rad: np.ndarray,
        fallback: float,
    ) -> np.ndarray:
        """Precompute per-facet albedo by sampling a provider grid at facet centers.

        ``lat_c_rad`` / ``lon_c_rad`` are facet-center latitudes/longitudes [rad].
        NaN / nodata samples fall back to ``fallback`` (documented policy). Grid
        shape/metadata mismatches raise via the underlying samplers.
        """
        n = int(lat_c_rad.shape[0])
        out = np.empty(n, dtype=np.float64)

        if source == ALBEDO_SOURCE_GRID:
            if "albedo_grid" not in surf:
                raise ValueError(
                    "albedo_mode='albedo_grid' requires surface provider key 'albedo_grid'."
                )
            grid = np.ascontiguousarray(np.asarray(surf["albedo_grid"], dtype=np.float64))
            nl = int(surf["n_lines"])
            ns = int(surf["n_samples"])
            res = float(surf["res_deg"])
            lon0 = float(surf["lon0_deg"])
            lat0 = float(surf["lat0_deg"])
            for i in range(n):
                lat_deg = math.degrees(float(lat_c_rad[i]))
                lon_deg = math.degrees(float(lon_c_rad[i]))
                a = float(sample_grid_bilinear(lat_deg, lon_deg, grid, nl, ns, res, lon0, lat0))
                out[i] = fallback if not math.isfinite(a) else a
        else:  # ALBEDO_SOURCE_SCALED_DN
            if "dn" not in surf:
                raise ValueError(
                    "albedo_mode='scaled_dn_grid' requires surface provider DN payload ('dn')."
                )
            dn = np.ascontiguousarray(np.asarray(surf["dn"], dtype=np.float64))
            nl = int(surf["n_lines"])
            ns = int(surf["n_samples"])
            res = float(surf["res_deg"])
            lon0 = float(surf["lon0_deg"])
            lat0 = float(surf["lat0_deg"])
            sf = float(surf.get("scale_factor", 1.0))
            off = float(surf.get("offset", 0.0))
            missing = float(surf.get("missing_dn", -1.0))
            flip = int(surf.get("flip_lat", 0))
            latmin = float(surf.get("lat_min_deg", -90.0))
            latmax = float(surf.get("lat_max_deg", 90.0))
            for i in range(n):
                lat_deg = math.degrees(float(lat_c_rad[i]))
                lon_deg = math.degrees(float(lon_c_rad[i]))
                a = float(
                    _sample_albedo_dn_scaled(
                        lat_deg, lon_deg, dn, nl, ns, res, lon0, lat0,
                        flip, sf, off, missing, latmin, latmax,
                    )
                )
                out[i] = fallback if not math.isfinite(a) else a

        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _prepare_earth_j2(self, req: dict[str, bool]) -> _EarthJ2Pack:
        if not req["use_earth_j2"] or (self.earth_j2 is None):
            return _EarthJ2Pack(j2=0.0, r_ref_m=1.0, ax=0.0, ay=0.0, az=1.0)

        ej2 = self.earth_j2
        j2 = float(ej2.j2_coeff)
        r_ref = float(ej2.r_eq_m)
        kx, ky, kz = ej2.spin_axis_i
        return _EarthJ2Pack(j2=j2, r_ref_m=r_ref, ax=float(kx), ay=float(ky), az=float(kz))

    def _prepare_solid_tides(self, req: dict[str, bool]) -> _TidePack:
        if not req["use_tides"]:
            return _TidePack(
                use_k2=False,
                use_k3=False,
                use_earth=False,
                use_sun=False,
                k2=0.0,
                k3=0.0,
                r_ref_m=float(R_MOON),
            )

        cfg = self.solid_tides
        k3_raw = getattr(cfg, "k3", None)
        if req["use_tides_k3"] and k3_raw is None:
            raise ValueError(
                "enable_tides_k3=True requires solid_tides.k3 to be set explicitly; "
                "no degree-3 lunar Love number default is assumed."
            )

        return _TidePack(
            use_k2=bool(req["use_tides_k2"]),
            use_k3=bool(req["use_tides_k3"]),
            use_earth=bool(req["use_tide_earth"]),
            use_sun=bool(req["use_tide_sun"]),
            k2=float(getattr(cfg, "k2", 0.0)),
            k3=0.0 if k3_raw is None else float(k3_raw),
            r_ref_m=float(getattr(cfg, "r_ref_m", R_MOON)),
        )

    def _prepare_thermal(self, req: dict[str, bool]) -> _ThermalPack:
        if not req["use_thermal"]:
            return _ThermalPack(
                mode=THERMAL_MODE_CONSTANT,
                surface_emissivity=1.0,
                surface_albedo=0.0,
                temperature_K=0.0,
                night_temperature_K=0.0,
                thermal_floor_flux_W_m2=0.0,
                ir_pressure_coefficient=0.0,
                solar_flux_1au_W_m2=0.0,
                au_m=float(AU),
                c_light_m_s=1.0,
                sigma_sb=1.0,
                include_sun_distance_scaling=True,
                enable_eclipse=False,
                r_earth_m=float(R_EARTH_MEAN),
                facet_pos_m=np.zeros((1, 3), dtype=np.float64),
                facet_normals=np.zeros((1, 3), dtype=np.float64),
                facet_areas_m2=np.zeros(1, dtype=np.float64),
                facet_temperatures_K=np.zeros(1, dtype=np.float64),
            )

        cfg = self.thermal if self.thermal is not None else ThermalConfig()
        mode = normalize_thermal_mode(getattr(cfg, "thermal_mode", "constant_temperature"))
        lat_count = int(getattr(cfg, "facet_lat_count", 18))
        lon_count = int(getattr(cfg, "facet_lon_count", 36))
        pos, normals, areas, _lat, _lon = build_latlon_facets(lat_count, lon_count, radius_m=float(R_MOON))

        temps: np.ndarray = np.zeros(1, dtype=np.float64)
        if mode == THERMAL_MODE_TEMPERATURE_GRID:
            surf = extract_surface_provider_strict(self.surf)
            raw_temps = None
            for key in ("thermal_temperature_cells_K", "temperature_cells_K", "facet_temperatures_K"):
                if key in surf:
                    raw_temps = surf[key]
                    break
            if raw_temps is None and "temperature_grid" in surf:
                raw_temps = surf["temperature_grid"]

            if raw_temps is None:
                raise ValueError(
                    "temperature_grid thermal mode requires surface_provider data: "
                    "thermal_temperature_cells_K, temperature_cells_K, facet_temperatures_K, or temperature_grid."
                )

            arr = np.asarray(raw_temps, dtype=np.float64)
            if arr.ndim == 2:
                if arr.shape != (lat_count, lon_count):
                    raise ValueError(
                        "temperature_grid shape must match thermal facet grid "
                        f"({lat_count}, {lon_count}), got {arr.shape}."
                    )
                arr = arr.reshape(-1)
            else:
                arr = arr.reshape(-1)

            if arr.shape[0] != areas.shape[0]:
                raise ValueError(
                    "temperature_grid mode requires one temperature per thermal facet "
                    f"(got temps={arr.shape[0]}, facets={areas.shape[0]})."
                )
            temps = np.ascontiguousarray(arr, dtype=np.float64)

        return _ThermalPack(
            mode=int(mode),
            surface_emissivity=float(getattr(cfg, "surface_emissivity", 0.95)),
            surface_albedo=float(getattr(cfg, "surface_albedo", 0.12)),
            temperature_K=float(getattr(cfg, "temperature_K", 250.0)),
            night_temperature_K=float(getattr(cfg, "night_temperature_K", 100.0)),
            thermal_floor_flux_W_m2=float(getattr(cfg, "thermal_floor_flux_W_m2", 0.0)),
            ir_pressure_coefficient=float(getattr(cfg, "ir_pressure_coefficient", getattr(cfg, "k_thermal", 1.0))),
            solar_flux_1au_W_m2=float(getattr(cfg, "solar_flux_1au_W_m2", SOLAR_FLUX_1AU)),
            au_m=float(getattr(cfg, "AU_m", AU)),
            c_light_m_s=float(getattr(cfg, "c_light_m_s", C_LIGHT)),
            sigma_sb=float(getattr(cfg, "sigma_sb", SIGMA_SB)),
            include_sun_distance_scaling=bool(getattr(cfg, "include_sun_distance_scaling", True)),
            enable_eclipse=bool(getattr(cfg, "enable_eclipse", True)),
            r_earth_m=float(R_EARTH_MEAN),
            facet_pos_m=pos,
            facet_normals=normals,
            facet_areas_m2=areas,
            facet_temperatures_K=temps,
        )

    # -------------------------------------------------------------------------
    # Public: build RHS
    # -------------------------------------------------------------------------
    def build_rhs(self, *, force_rebuild: bool = False) -> Callable[[float, np.ndarray], np.ndarray]:
        if self._rhs_cache is not None and not force_rebuild:
            return self._rhs_cache

        t0 = time.perf_counter()

        req = self._requirements()
        gp = self._prepare_gravity()
        ep = self._prepare_ephem(req)
        ap = self._prepare_albedo(req)
        ej = self._prepare_earth_j2(req)
        tp = self._prepare_solid_tides(req)
        th = self._prepare_thermal(req)

        # Cache for debug/reporting
        self._prep = {"req": req, "grav": gp, "eph": ep, "alb": ap, "earth_j2": ej, "tides": tp, "thermal": th}

        # Flags captured into closure
        USE_SH = bool(req["use_sh"])
        USE_SURROGATE = bool(req.get("use_surrogate_gravity", False))
        USE_3RD_SUN = bool(req["use_3rd_sun"])
        USE_3RD_EARTH = bool(req["use_3rd_earth"])
        USE_SRP = bool(req["use_srp"])
        USE_ALBEDO = bool(req["use_albedo"])
        USE_REL = bool(req["use_rel"])
        USE_REL_EXTERNAL = bool(req["use_rel_external"])
        USE_EJ2 = bool(req["use_earth_j2"])
        USE_TIDES = bool(req["use_tides"])
        USE_TIDE_EARTH = bool(req["use_tide_earth"])
        USE_TIDE_SUN = bool(req["use_tide_sun"])
        USE_THERMAL = bool(req["use_thermal"])
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
        NEED_EPH = bool(HAVE_EPH and (req["need_sun"] or req["need_earth"] or req["need_q"]))

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

            def rhs(t: float, y: np.ndarray) -> np.ndarray:  # type: ignore[no-redef]
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

        req: dict[str, bool] = self._prep["req"]
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
        need_eph = bool(have_eph and (req["need_sun"] or req["need_earth"] or req["need_q"]))
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
        if req["use_sh"]:
            if bool(req.get("use_surrogate_gravity", False)):
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
        if req["use_3rd_sun"]:
            ax3, ay3, az3 = accel_third_body_numba(r[0], r[1], r[2], sun[0], sun[1], sun[2], float(MU_SUN))
            out["3rd Body (Sun)"] = _norm3(ax3, ay3, az3)

        if req["use_3rd_earth"]:
            ax3, ay3, az3 = accel_third_body_numba(r[0], r[1], r[2], earth[0], earth[1], earth[2], float(MU_EARTH))
            out["3rd Body (Earth)"] = _norm3(ax3, ay3, az3)

        if req["use_earth_j2"]:
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
        if req["use_tides"]:
            rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])
            tide_x = 0.0
            tide_y = 0.0
            tide_z = 0.0

            if req["use_tide_earth"]:
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

            if req["use_tide_sun"]:
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
        if req["use_srp"]:
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
        if req["use_albedo"]:
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
        if req["use_thermal"]:
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
        if req["use_rel"]:
            arx, ary, arz = _schwarzschild_components(r[0], r[1], r[2], v[0], v[1], v[2], float(mu_m))
            rel_x = arx
            rel_y = ary
            rel_z = arz
            out["Relativity (Moon Schwarzschild)"] = _norm3(arx, ary, arz)

            if req.get("use_rel_external", False):
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
