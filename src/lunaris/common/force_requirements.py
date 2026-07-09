"""Shared force-model runtime requirement helpers.

This module is dependency-light by design: it only inspects config-shaped
objects and perturbation flags, so core, CLI, and batch orchestration can share
one decision tree without importing physics kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

_THERMAL_EQUILIBRIUM_MODES = {
    "equilibrium",
    "equilibrium_temperature",
    "instantaneous_equilibrium",
}
_THERMAL_GRID_MODES = {"temperature_grid", "grid"}
_ALBEDO_CONSTANT_MODES = {"constant", "constant_albedo"}


@dataclass(frozen=True, slots=True)
class ForceRequirements:
    """Data tables required by the active perturbation set."""

    use_sh: bool
    use_surrogate_gravity: bool
    use_albedo: bool
    albedo_needs_provider: bool
    use_thermal: bool
    use_thermal_equilibrium: bool
    use_thermal_eclipse: bool
    use_thermal_grid: bool
    use_srp: bool
    use_3rd_sun: bool
    use_3rd_earth: bool
    use_tides: bool
    use_tides_k2: bool
    use_tides_k3: bool
    use_tide_earth: bool
    use_tide_sun: bool
    use_rel: bool
    use_rel_external: bool
    use_earth_j2: bool
    need_sun: bool
    need_earth: bool
    need_q_i2f: bool
    need_surface_provider: bool
    need_body_vectors: bool
    need_quat_from_ephem: bool
    need_ephem: bool

    def without_external_relativity(self) -> ForceRequirements:
        """Return a copy with external 1PN disabled and derived flags refreshed."""
        need_sun = bool(
            self.use_srp
            or self.use_3rd_sun
            or self.use_albedo
            or self.use_tide_sun
            or self.use_thermal_equilibrium
        )
        need_earth = bool(
            self.use_3rd_earth
            or self.use_earth_j2
            or self.use_tide_earth
            or self.use_thermal_eclipse
        )
        need_body_vectors = bool(need_sun or need_earth)
        need_ephem = bool(need_body_vectors or self.need_quat_from_ephem)
        return replace(
            self,
            use_rel_external=False,
            need_sun=need_sun,
            need_earth=need_earth,
            need_body_vectors=need_body_vectors,
            need_ephem=need_ephem,
        )


def _mode(value: Any, default: str) -> str:
    return str(value if value is not None else default).strip().lower()


def force_requirements(
    flags: Any,
    *,
    gravity_uses_st_lrps: bool = False,
    earth_j2_available: bool = True,
    albedo: Any = None,
    thermal: Any = None,
    solid_tides: Any = None,
    allow_identity_rotation: bool = False,
    request_external_relativity: bool = False,
) -> ForceRequirements:
    """Return the ephemeris/surface requirements for a config-like object."""

    use_sh = bool(getattr(flags, "enable_sh", False))
    if gravity_uses_st_lrps and not use_sh:
        raise ValueError(
            "ST-LRPS backend requires enable_sh=True because enable_sh currently "
            "gates the central lunar gravity field."
        )
    use_surrogate_gravity = bool(gravity_uses_st_lrps)
    use_albedo = bool(getattr(flags, "enable_albedo", False))
    use_thermal = bool(getattr(flags, "enable_thermal", getattr(flags, "enable_thermal_ir", False)))
    use_srp = bool(getattr(flags, "enable_srp", False))
    use_3rd_sun = bool(getattr(flags, "enable_3rd_body_sun", False))
    use_3rd_earth = bool(getattr(flags, "enable_3rd_body_earth", False))
    use_tides_k2 = bool(getattr(flags, "enable_tides_k2", False))
    use_tides_k3 = bool(getattr(flags, "enable_tides_k3", False))
    use_tides = bool(use_tides_k2 or use_tides_k3)
    use_rel = bool(getattr(flags, "enable_relativity_1pn", getattr(flags, "enable_relativity", False)))
    use_rel_external = bool(use_rel and request_external_relativity)
    use_earth_j2 = bool(getattr(flags, "enable_earth_j2", False) and earth_j2_available)

    albedo_mode = _mode(getattr(albedo, "albedo_mode", "constant_albedo"), "constant_albedo")
    albedo_needs_provider = bool(
        use_albedo
        and (
            albedo_mode not in _ALBEDO_CONSTANT_MODES
            or bool(getattr(albedo, "require_surface_provider", False))
        )
    )

    thermal_mode = _mode(getattr(thermal, "thermal_mode", "constant_temperature"), "constant_temperature")
    use_thermal_equilibrium = bool(use_thermal and thermal_mode in _THERMAL_EQUILIBRIUM_MODES)
    use_thermal_grid = bool(use_thermal and thermal_mode in _THERMAL_GRID_MODES)
    use_thermal_eclipse = bool(
        use_thermal_equilibrium
        and bool(getattr(thermal, "enable_eclipse", True))
    )

    tide_bodies = tuple(
        str(body).strip().lower()
        for body in getattr(solid_tides, "tide_bodies", ("earth", "sun"))
    )
    use_tide_earth = bool(use_tides and "earth" in tide_bodies)
    use_tide_sun = bool(use_tides and "sun" in tide_bodies)

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
    need_q_i2f = bool(use_sh or use_surrogate_gravity or use_albedo or use_tides or use_thermal)
    need_body_vectors = bool(need_sun or need_earth)
    need_quat_from_ephem = bool(need_q_i2f and not allow_identity_rotation)
    need_ephem = bool(need_body_vectors or need_quat_from_ephem)
    need_surface_provider = bool(albedo_needs_provider or use_thermal_grid)

    return ForceRequirements(
        use_sh=use_sh,
        use_surrogate_gravity=use_surrogate_gravity,
        use_albedo=use_albedo,
        albedo_needs_provider=albedo_needs_provider,
        use_thermal=use_thermal,
        use_thermal_equilibrium=use_thermal_equilibrium,
        use_thermal_eclipse=use_thermal_eclipse,
        use_thermal_grid=use_thermal_grid,
        use_srp=use_srp,
        use_3rd_sun=use_3rd_sun,
        use_3rd_earth=use_3rd_earth,
        use_tides=use_tides,
        use_tides_k2=use_tides_k2,
        use_tides_k3=use_tides_k3,
        use_tide_earth=use_tide_earth,
        use_tide_sun=use_tide_sun,
        use_rel=use_rel,
        use_rel_external=use_rel_external,
        use_earth_j2=use_earth_j2,
        need_sun=need_sun,
        need_earth=need_earth,
        need_q_i2f=need_q_i2f,
        need_surface_provider=need_surface_provider,
        need_body_vectors=need_body_vectors,
        need_quat_from_ephem=need_quat_from_ephem,
        need_ephem=need_ephem,
    )


def force_requirements_for_config(
    cfg: Any,
    *,
    gravity_uses_st_lrps: bool | None = None,
    allow_identity_rotation: bool = False,
    request_external_relativity: bool = False,
) -> ForceRequirements:
    """Return force requirements from a SimConfig-like object."""

    if gravity_uses_st_lrps is None:
        gravity = getattr(cfg, "gravity", None)
        uses = getattr(gravity, "uses_st_lrps", False)
        gravity_uses_st_lrps = bool(uses() if callable(uses) else uses)

    return force_requirements(
        cfg.flags,
        gravity_uses_st_lrps=gravity_uses_st_lrps,
        earth_j2_available=True,
        albedo=getattr(cfg, "albedo", None),
        thermal=getattr(cfg, "thermal", None),
        solid_tides=getattr(cfg, "solid_tides", None),
        allow_identity_rotation=allow_identity_rotation,
        request_external_relativity=request_external_relativity,
    )


__all__ = [
    "ForceRequirements",
    "force_requirements",
    "force_requirements_for_config",
]
