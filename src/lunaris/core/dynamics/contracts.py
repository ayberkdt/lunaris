"""Typed dynamics-layer contracts shared by preparation and engine code."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from lunaris.common.force_requirements import ForceRequirements


@dataclass(frozen=True, slots=True)
class DynamicsRequirements:
    """Typed requirement set consumed by dependency validation and pack prep.

    Wraps the shared :class:`ForceRequirements` decision tree and adds the
    dynamics-layer albedo-model selector. All flags are exposed as typed
    read-only properties so a typo fails at type-check time instead of as a
    runtime ``KeyError``. ``to_dict()`` exists only for serialization /
    provenance boundaries and preserves the historical key names
    (``need_q``, ``need_vectors``).
    """

    force: ForceRequirements
    albedo_model: str

    @property
    def use_sh(self) -> bool:
        return self.force.use_sh

    @property
    def use_surrogate_gravity(self) -> bool:
        return self.force.use_surrogate_gravity

    @property
    def use_albedo(self) -> bool:
        return self.force.use_albedo

    @property
    def albedo_needs_provider(self) -> bool:
        return self.force.albedo_needs_provider

    @property
    def use_thermal(self) -> bool:
        return self.force.use_thermal

    @property
    def use_thermal_equilibrium(self) -> bool:
        return self.force.use_thermal_equilibrium

    @property
    def use_thermal_eclipse(self) -> bool:
        return self.force.use_thermal_eclipse

    @property
    def use_thermal_grid(self) -> bool:
        return self.force.use_thermal_grid

    @property
    def use_srp(self) -> bool:
        return self.force.use_srp

    @property
    def use_3rd_sun(self) -> bool:
        return self.force.use_3rd_sun

    @property
    def use_3rd_earth(self) -> bool:
        return self.force.use_3rd_earth

    @property
    def use_tides(self) -> bool:
        return self.force.use_tides

    @property
    def use_tides_k2(self) -> bool:
        return self.force.use_tides_k2

    @property
    def use_tides_k3(self) -> bool:
        return self.force.use_tides_k3

    @property
    def use_tide_earth(self) -> bool:
        return self.force.use_tide_earth

    @property
    def use_tide_sun(self) -> bool:
        return self.force.use_tide_sun

    @property
    def use_rel(self) -> bool:
        return self.force.use_rel

    @property
    def use_rel_external(self) -> bool:
        return self.force.use_rel_external

    @property
    def use_earth_j2(self) -> bool:
        return self.force.use_earth_j2

    @property
    def need_sun(self) -> bool:
        return self.force.need_sun

    @property
    def need_earth(self) -> bool:
        return self.force.need_earth

    @property
    def need_q(self) -> bool:
        """Historical alias for ``force.need_q_i2f``."""
        return self.force.need_q_i2f

    @property
    def need_vectors(self) -> bool:
        """Historical alias for ``force.need_body_vectors``."""
        return self.force.need_body_vectors

    @property
    def need_quat_from_ephem(self) -> bool:
        return self.force.need_quat_from_ephem

    @property
    def need_ephem(self) -> bool:
        return self.force.need_ephem

    def without_external_relativity(self) -> DynamicsRequirements:
        """Return a new requirement set with external 1PN disabled."""
        return replace(self, force=self.force.without_external_relativity())

    def to_dict(self) -> dict[str, Any]:
        """Serialization/provenance boundary only; not for flag lookups."""
        return {
            "use_sh": self.use_sh,
            "use_surrogate_gravity": self.use_surrogate_gravity,
            "use_albedo": self.use_albedo,
            "albedo_model": self.albedo_model,
            "albedo_needs_provider": self.albedo_needs_provider,
            "use_thermal": self.use_thermal,
            "use_thermal_equilibrium": self.use_thermal_equilibrium,
            "use_thermal_eclipse": self.use_thermal_eclipse,
            "use_thermal_grid": self.use_thermal_grid,
            "use_srp": self.use_srp,
            "use_3rd_sun": self.use_3rd_sun,
            "use_3rd_earth": self.use_3rd_earth,
            "use_tides": self.use_tides,
            "use_tides_k2": self.use_tides_k2,
            "use_tides_k3": self.use_tides_k3,
            "use_tide_earth": self.use_tide_earth,
            "use_tide_sun": self.use_tide_sun,
            "use_rel": self.use_rel,
            "use_rel_external": self.use_rel_external,
            "use_earth_j2": self.use_earth_j2,
            "need_sun": self.need_sun,
            "need_earth": self.need_earth,
            "need_q": self.need_q,
            "need_vectors": self.need_vectors,
            "need_quat_from_ephem": self.need_quat_from_ephem,
            "need_ephem": self.need_ephem,
        }


__all__ = ["DynamicsRequirements"]
