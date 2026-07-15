"""Config -> validated-pack preparation for the dynamics engine.

This module owns everything that happens *before* the jitted RHS closure is
assembled: computing the requirement set for the active perturbation flags,
validating provider dependencies, and converting high-level provider objects
(gravity model, ephemeris manager, surface provider, force configs) into the
frozen Numba-friendly packs consumed by ``engine.build_rhs``.

Every function is a pure module-level function with explicit inputs — no
engine state, no side effects on its arguments (``resolve_effective_requirements``
returns a new :class:`DynamicsRequirements` rather than mutating the raw one).
The jitted RHS closures intentionally remain inside ``engine.py`` so this split
does not alter hot-loop object boundaries (see ``rhs_numba``).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from lunaris.common.constants import (
    AU,
    C_LIGHT,
    MU_EARTH,
    MU_MOON,
    MU_SUN,
    R_EARTH_MEAN,
    R_MOON,
    SIGMA_SB,
    SOLAR_FLUX_1AU,
)
from lunaris.common.force_requirements import force_requirements
from lunaris.common.math_utils import sample_grid_bilinear
from lunaris.common.type_defs import SpacecraftProps
from lunaris.core.dynamics.adaptive_degree import _sample_albedo_dn_scaled
from lunaris.core.dynamics.contracts import DynamicsRequirements
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
from lunaris.physics.lunar_albedo import (
    ALBEDO_SOURCE_CONSTANT,
    ALBEDO_SOURCE_GRID,
    normalize_albedo_mode,
)
from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig
from lunaris.physics.thermal_ir import (
    THERMAL_MODE_CONSTANT,
    THERMAL_MODE_TEMPERATURE_GRID,
    build_latlon_facets,
    normalize_thermal_mode,
)


def _provider_get(provider: Any, key: str, default: Any) -> Any:
    if isinstance(provider, dict):
        return provider.get(key, default)
    return getattr(provider, key, default)


def _provider_has(provider: Any, key: str) -> bool:
    if isinstance(provider, dict):
        return key in provider
    return hasattr(provider, key)


# =============================================================================
# 1.                     Requirements / dependency validation
# =============================================================================


def compute_requirements(
    *,
    flags: Any,
    gravity_model: Any,
    earth_j2: Any,
    albedo: Any,
    thermal: Any,
    solid_tides: Any,
    allow_identity_rotation: bool,
    have_ephem: bool,
) -> DynamicsRequirements:
    """Raw config-derived requirement set for the active perturbation flags."""
    alb_cfg = albedo if albedo is not None else AlbedoConfig()
    albedo_model = str(getattr(alb_cfg, "albedo_model", "lambert_facets")).strip().lower()

    # Design (#4): there is intentionally no separate "external relativity"
    # flag. The external-body Schwarzschild differential and the de Sitter
    # geodetic term are the physically dominant relativistic effects for a
    # lunar orbiter, but they can only be evaluated when Sun/Earth ephemeris
    # states are available. So they auto-enable whenever 1PN relativity is on
    # AND an ephemeris is present, and silently degrade to the central-body
    # Schwarzschild term alone otherwise. The body velocities they require are
    # the analytic derivatives of the same Catmull-Rom ephemeris interpolant
    # used for body positions (interp_vec3_derivative_safe; mirrored by the
    # GPU _interp3_derivative_cuda), so each external 1PN state is kinematically
    # consistent within an ephemeris cell.
    req = force_requirements(
        flags,
        gravity_uses_st_lrps=_is_surrogate_gravity_provider(gravity_model),
        earth_j2_available=earth_j2 is not None,
        albedo=alb_cfg,
        thermal=thermal,
        solid_tides=solid_tides,
        allow_identity_rotation=allow_identity_rotation,
        request_external_relativity=have_ephem,
    )

    return DynamicsRequirements(force=req, albedo_model=albedo_model)


def validate_dependencies(
    *,
    req: DynamicsRequirements,
    gravity_model: Any,
    surface_provider: Any,
    albedo: Any,
    sc_props: SpacecraftProps,
    ephem_manager: Any,
    flags: Any,
    solid_tides: Any,
    earth_j2: Any,
) -> None:
    """Fail fast when an enabled perturbation is missing a provider/config."""
    if req.use_sh and gravity_model is None:
        raise ValueError("enable_sh=True but gravity_model is None.")

    if req.albedo_needs_provider and surface_provider is None:
        cfg = albedo if albedo is not None else AlbedoConfig()
        raise ValueError(
            f"enable_albedo with albedo_mode={cfg.albedo_mode!r} requires a "
            "surface_provider supplying an albedo grid, but none was provided. "
            "Use albedo_mode='constant_albedo' for a provider-free model, or "
            "supply a surface_provider (e.g. --albedo-root)."
        )

    if req.use_thermal_grid and surface_provider is None:
        raise ValueError(
            "ThermalConfig.thermal_mode='temperature_grid' requires surface_provider "
            "with thermal_temperature_cells_K, temperature_cells_K, or temperature_grid."
        )

    # SRP / Albedo / Thermal IR require valid spacecraft optical area and mass
    if req.use_srp or req.use_albedo or req.use_thermal:
        if sc_props.mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be > 0, got {sc_props.mass_kg}")
        if sc_props.area_m2 <= 0.0:
            raise ValueError(
                f"area_m2 must be > 0 for SRP/Albedo/Thermal IR, got {sc_props.area_m2}"
            )
    # The SRP and simple/legacy albedo backends reuse the spacecraft SRP
    # coefficient cr; the lambert_facets albedo backend does not (it uses
    # albedo_pressure_coefficient), so cr is only enforced when actually used.
    if req.use_srp or (req.use_albedo and req.albedo_model == "simple"):
        if not (0.0 < sc_props.cr <= 2.5):
            raise ValueError(f"cr looks invalid, got {sc_props.cr}")

    if req.need_ephem and (ephem_manager is None):
        reasons: list[str] = []
        if req.need_vectors:
            if req.need_sun:
                reasons.append(
                    "Sun vector (SRP / 3rd-body Sun / Albedo / Thermal IR equilibrium / solid tides)"
                )
            if req.need_earth:
                reasons.append(
                    "Earth vector (3rd-body Earth / Earth J2 / Thermal IR eclipse / solid tides)"
                )
        if req.need_quat_from_ephem:
            reasons.append("q_i2f (SH / Albedo / Thermal IR / solid tides)")

        why = "; ".join(reasons) if reasons else "Sun/Earth vectors and/or q_i2f"
        raise ValueError(
            f"Ephemeris is required for selected perturbations: {why}. "
            "Provide ephem_manager, or disable those perturbations. "
            "Note: allow_identity_rotation only replaces q_i2f, not Sun/Earth vectors."
        )

    if req.use_tides_k3 and getattr(solid_tides, "k3", None) is None:
        raise ValueError(
            "enable_tides_k3=True requires solid_tides.k3 to be set explicitly; "
            "no degree-3 lunar Love number default is assumed."
        )

    if bool(getattr(flags, "enable_earth_j2", False)) and (earth_j2 is None):
        raise ValueError("enable_earth_j2=True but earth_j2 params are None.")


# =============================================================================
# 2.                     Providers -> prepared packs (strict)
# =============================================================================


def prepare_adaptive_gravity_policy(
    nmax: int,
    *,
    gravity_model: Any,
    gravity_adaptive: Any,
    strict_fixed_degree: bool = False,
) -> dict[str, Any]:
    """
    Normalize optional adaptive-degree settings into kernel-friendly arrays.

    The backend config already validates table ordering, but the dynamics
    layer still clamps each requested degree to the loaded model's actual
    maximum. This keeps runtime evaluation robust even when older session
    files or UI presets request degrees higher than the active gravity file.

    ``strict_fixed_degree`` (paper-safe / benchmark / strict runs) forbids the
    adaptive blend outright: a reference result must use a single fixed degree so
    its error is attributable to one model, not to an altitude-dependent blend of
    two. Enabling adaptive degree under a strict posture is a hard error, never a
    silent downgrade.
    """

    adaptive = gravity_adaptive
    if adaptive is None and gravity_model is not None:
        adaptive = getattr(gravity_model, "adaptive", None)

    would_enable = (
        adaptive is not None and bool(getattr(adaptive, "enabled", False)) and int(nmax) > 0
    )
    if strict_fixed_degree and would_enable:
        raise ValueError(
            "Adaptive SH degree blending is forbidden in paper-safe / benchmark / "
            "strict runs: a reference result must use a single fixed degree so its "
            "error is attributable to one model, not an altitude-dependent blend. "
            "Set gravity.adaptive.enabled=False (fixed degree) for reference runs, "
            "or drop the strict/paper-safe flag for exploratory use."
        )

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


def prepare_gravity(
    gravity_model: Any,
    *,
    gravity_adaptive: Any = None,
    strict_fixed_degree: bool = False,
) -> _GravPack:
    """Validated gravity pack: point-mass, surrogate, or strict classical SH.

    ``strict_fixed_degree`` forbids adaptive-degree blending (paper-safe /
    benchmark / strict reference runs must use a single fixed degree).
    """
    if gravity_model is None:
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

    if _is_surrogate_gravity_provider(gravity_model):
        z11 = np.zeros((1, 1), dtype=np.float64)
        z1 = np.zeros(1, dtype=np.float64)
        return _GravPack(
            nmax=0,
            r_ref_m=float(
                getattr(gravity_model, "R_ref_m", getattr(gravity_model, "r_ref_m", R_MOON))
            ),
            gm_m3s2=float(
                getattr(gravity_model, "GM_m3s2", getattr(gravity_model, "gm_m3s2", MU_MOON))
            ),
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

    nmax, r_ref, gm, Cnm, Snm, diag, subdiag, A, B, scale_m, ws_P, ws_dP, ws_cos, ws_sin = (
        extract_gravity_strict(gravity_model)
    )
    adaptive_policy = prepare_adaptive_gravity_policy(
        int(nmax),
        gravity_model=gravity_model,
        gravity_adaptive=gravity_adaptive,
        strict_fixed_degree=strict_fixed_degree,
    )

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


def prepare_ephem(ephem_manager: Any, req: DynamicsRequirements) -> _EphemPack:
    """Validated ephemeris pack; fails closed on degenerate body tables."""
    if ephem_manager is None:
        # Only valid if we do NOT need Sun/Earth vectors AND we allow identity rotation for q_i2f.
        q_ident = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        z23 = np.zeros((2, 3), dtype=np.float64)
        return _EphemPack(
            dt_s=1.0,
            r_sun_tab_m=z23,
            r_earth_tab_m=z23,
            q_i2f_tab=q_ident,
            mu_earth_m3s2=float(MU_EARTH),
            mu_sun_m3s2=float(MU_SUN),
        )

    dt_s, sun_tab, earth_tab, qtab = extract_ephem_tables_strict(ephem_manager)
    provider = ephem_manager.get_data_provider()
    mu_earth = float(_provider_get(provider, "mu_earth_m3s2", MU_EARTH))
    mu_sun = float(_provider_get(provider, "mu_sun_m3s2", MU_SUN))

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
        req.use_srp
        or req.use_3rd_sun
        or req.use_albedo
        or req.use_tide_sun
        or req.use_thermal_equilibrium
    )
    explicit_earth = bool(
        req.use_3rd_earth or req.use_earth_j2 or req.use_tide_earth or req.use_thermal_eclipse
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

    return _EphemPack(
        dt_s=float(dt_s),
        r_sun_tab_m=sun_tab,
        r_earth_tab_m=earth_tab,
        q_i2f_tab=qtab,
        mu_earth_m3s2=mu_earth,
        mu_sun_m3s2=mu_sun,
    )


def resolve_effective_requirements(
    req: DynamicsRequirements, ep: _EphemPack
) -> DynamicsRequirements:
    """Derive the effective runtime requirements from the raw config-derived ones.

    The external-1PN relativity terms are documented to silently degrade to
    the central-body Schwarzschild term when the ephemeris lacks Sun/Earth
    position tables (see ``prepare_ephem``). That downgrade is resolved
    here, in one explicit step that returns a *new* object: the raw
    requirements are never mutated, so no consumer can observe a stale flag
    depending on when it reads ``req``.
    """
    sun_degenerate = not np.any(ep.r_sun_tab_m)
    earth_degenerate = not np.any(ep.r_earth_tab_m)
    if req.use_rel_external and (sun_degenerate or earth_degenerate):
        warnings.warn(
            "1PN external-body relativity terms disabled: the ephemeris does not "
            "provide Sun/Earth position tables (all zeros). Only the central-body "
            "Schwarzschild term remains active.",
            RuntimeWarning,
            stacklevel=3,
        )
        return req.without_external_relativity()
    return req


def prepare_albedo(
    req: DynamicsRequirements,
    *,
    albedo: Any,
    surface_provider: Any,
) -> _AlbedoPack:
    """Validated albedo pack (lambert_facets or legacy simple backend)."""
    if not req.use_albedo:
        return _AlbedoPack(backend=1, mode=2, alb_const=0.12, alb_scale=1.0, k_lambert=1.0)

    cfg = albedo if albedo is not None else AlbedoConfig()
    model = str(getattr(cfg, "albedo_model", "lambert_facets")).strip().lower()

    if model == "simple":
        return _prepare_albedo_simple(cfg, surface_provider)

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
        if surface_provider is None:
            raise ValueError(
                f"albedo_mode={cfg.albedo_mode!r} requires a surface_provider "
                "supplying an albedo grid, but none was provided."
            )
        surf = extract_surface_provider_strict(surface_provider)
        facet_albedo = _sample_facet_albedo_from_provider(surf, source, lat_c, lon_c, fallback)

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


def _prepare_albedo_simple(cfg: AlbedoConfig, surface_provider: Any) -> _AlbedoPack:
    """Legacy cannonball backend: parameters sourced from the surface provider.

    Reproduces the historical albedo behavior (single sub-satellite albedo
    sample, Sun->spacecraft push, SRP coefficient ``cr``). Works without a
    provider using a constant albedo from the config.
    """
    const = float(getattr(cfg, "albedo_const", 0.12))
    klamb = float(getattr(cfg, "k_lambert", 1.0))
    if surface_provider is None:
        return _AlbedoPack(backend=0, mode=2, alb_const=const, alb_scale=1.0, k_lambert=klamb)

    surf = extract_surface_provider_strict(surface_provider)
    alb_const = float(surf.get("albedo_const", const))
    alb_scale = float(surf.get("scale", 1.0))
    k_lambert = float(surf.get("k_lambert", klamb))

    if "albedo_grid" in surf:
        return _AlbedoPack(
            backend=0,
            mode=0,
            alb_const=alb_const,
            alb_scale=alb_scale,
            k_lambert=k_lambert,
            grid_alb=surf["albedo_grid"],
            n_lines=int(surf["n_lines"]),
            n_samples=int(surf["n_samples"]),
            res_deg=float(surf["res_deg"]),
            lon0_deg=float(surf["lon0_deg"]),
            lat0_deg=float(surf["lat0_deg"]),
        )
    if "dn" in surf:
        return _AlbedoPack(
            backend=0,
            mode=1,
            alb_const=alb_const,
            alb_scale=alb_scale,
            k_lambert=k_lambert,
            dn=surf["dn"],
            n_lines=int(surf["n_lines"]),
            n_samples=int(surf["n_samples"]),
            res_deg=float(surf["res_deg"]),
            lon0_deg=float(surf["lon0_deg"]),
            lat0_deg=float(surf["lat0_deg"]),
            sf=float(surf.get("scale_factor", 1.0)),
            off=float(surf.get("offset", 0.0)),
            missing=float(surf.get("missing_dn", -1.0)),
            flip=int(surf.get("flip_lat", 0)),
            latmin=float(surf.get("lat_min_deg", -90.0)),
            latmax=float(surf.get("lat_max_deg", 90.0)),
        )
    return _AlbedoPack(
        backend=0, mode=2, alb_const=alb_const, alb_scale=alb_scale, k_lambert=k_lambert
    )


def _sample_facet_albedo_from_provider(
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
                    lat_deg,
                    lon_deg,
                    dn,
                    nl,
                    ns,
                    res,
                    lon0,
                    lat0,
                    flip,
                    sf,
                    off,
                    missing,
                    latmin,
                    latmax,
                )
            )
            out[i] = fallback if not math.isfinite(a) else a

    np.clip(out, 0.0, 1.0, out=out)
    return out


def prepare_earth_j2(req: DynamicsRequirements, earth_j2: Any) -> _EarthJ2Pack:
    """Earth-J2 pack (inert when the force is disabled or params missing)."""
    if not req.use_earth_j2 or (earth_j2 is None):
        return _EarthJ2Pack(j2=0.0, r_ref_m=1.0, ax=0.0, ay=0.0, az=1.0)

    j2 = float(earth_j2.j2_coeff)
    r_ref = float(earth_j2.r_eq_m)
    kx, ky, kz = earth_j2.spin_axis_i
    return _EarthJ2Pack(j2=j2, r_ref_m=r_ref, ax=float(kx), ay=float(ky), az=float(kz))


def prepare_solid_tides(req: DynamicsRequirements, solid_tides: Any) -> _TidePack:
    """Solid-tide pack (k2/k3 Love numbers, per-body switches)."""
    if not req.use_tides:
        return _TidePack(
            use_k2=False,
            use_k3=False,
            use_earth=False,
            use_sun=False,
            k2=0.0,
            k3=0.0,
            r_ref_m=float(R_MOON),
        )

    cfg = solid_tides
    k3_raw = getattr(cfg, "k3", None)
    if req.use_tides_k3 and k3_raw is None:
        raise ValueError(
            "enable_tides_k3=True requires solid_tides.k3 to be set explicitly; "
            "no degree-3 lunar Love number default is assumed."
        )

    return _TidePack(
        use_k2=bool(req.use_tides_k2),
        use_k3=bool(req.use_tides_k3),
        use_earth=bool(req.use_tide_earth),
        use_sun=bool(req.use_tide_sun),
        k2=float(getattr(cfg, "k2", 0.0)),
        k3=0.0 if k3_raw is None else float(k3_raw),
        r_ref_m=float(getattr(cfg, "r_ref_m", R_MOON)),
    )


def prepare_thermal(
    req: DynamicsRequirements,
    *,
    thermal: Any,
    surface_provider: Any,
) -> _ThermalPack:
    """Thermal-IR pack (constant / equilibrium / temperature-grid modes)."""
    if not req.use_thermal:
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

    cfg = thermal if thermal is not None else ThermalConfig()
    mode = normalize_thermal_mode(getattr(cfg, "thermal_mode", "constant_temperature"))
    lat_count = int(getattr(cfg, "facet_lat_count", 18))
    lon_count = int(getattr(cfg, "facet_lon_count", 36))
    pos, normals, areas, _lat, _lon = build_latlon_facets(
        lat_count, lon_count, radius_m=float(R_MOON)
    )

    temps: np.ndarray = np.zeros(1, dtype=np.float64)
    if mode == THERMAL_MODE_TEMPERATURE_GRID:
        surf = extract_surface_provider_strict(surface_provider)
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
        ir_pressure_coefficient=float(
            getattr(cfg, "ir_pressure_coefficient", getattr(cfg, "k_thermal", 1.0))
        ),
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


__all__ = [
    "DynamicsRequirements",
    "compute_requirements",
    "validate_dependencies",
    "prepare_adaptive_gravity_policy",
    "prepare_gravity",
    "prepare_ephem",
    "resolve_effective_requirements",
    "prepare_albedo",
    "prepare_earth_j2",
    "prepare_solid_tides",
    "prepare_thermal",
]
