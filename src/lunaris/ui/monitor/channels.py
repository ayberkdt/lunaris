"""Channel catalogue for the Mission Monitor.

A *channel* is one observable stream a widget can subscribe to. The catalogue
is the single source of truth for channel identity, base unit (always SI —
presentation converts), frame semantics, and whether the value is measured
from the propagated state or derived from it. Widgets declare
``required_channels`` against these IDs and the store keys its ring buffers by
them, so availability ("this run/backend does not provide X") is decidable
without touching widget code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChannelKind = Literal["scalar", "state", "events", "diagnostics", "provenance"]

#: Orbital-element channels are namespaced so new elements never collide with
#: top-level sample fields.
ELEMENT_CHANNEL_PREFIX = "elements."


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """Identity and semantics of one monitor channel."""

    channel_id: str
    label: str
    unit: str  # SI base unit of the stored values ("" when dimensionless)
    kind: ChannelKind = "scalar"
    description: str = ""
    #: True when the value is derived (e.g. osculating elements) rather than a
    #: direct function of the integrated state. Widgets badge derived data.
    derived: bool = False
    #: Frame the values are expressed in; None = frame-independent magnitude.
    frame: str | None = None


_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec("radius_m", "Radius", "m",
                description="Moon-centered radial distance |r|."),
    ChannelSpec("altitude_m", "Altitude", "m",
                description="Altitude above the mean reference radius (r - R_ref)."),
    ChannelSpec("speed_m_s", "Speed", "m/s",
                description="Inertial speed |v|."),
    ChannelSpec("surface_radius_m", "Local surface radius", "m",
                description="Topography-sampled local surface radius under the spacecraft."),
    ChannelSpec("terrain_clearance_m", "Terrain clearance", "m",
                description="Radial clearance above local terrain (r - r_surface)."),
    ChannelSpec("elements.sma_m", "Semi-major axis", "m", derived=True,
                description="Osculating 2-body semi-major axis."),
    ChannelSpec("elements.ecc", "Eccentricity", "", derived=True,
                description="Osculating 2-body eccentricity."),
    ChannelSpec("elements.inc_rad", "Inclination", "rad", derived=True,
                description="Osculating inclination."),
    ChannelSpec("elements.raan_rad", "RAAN", "rad", derived=True,
                description="Right ascension of the ascending node (undefined for equatorial orbits)."),
    ChannelSpec("elements.argp_rad", "Arg. of periapsis", "rad", derived=True,
                description="Argument of periapsis (undefined for circular orbits)."),
    ChannelSpec("elements.nu_rad", "True anomaly", "rad", derived=True,
                description="True anomaly (or its singular-orbit substitute)."),
    ChannelSpec("state_inertial", "State (inertial)", "m, m/s", kind="state",
                description="Cartesian position/velocity in the inertial frame."),
    ChannelSpec("state_fixed", "State (body-fixed)", "m, m/s", kind="state",
                description="Cartesian position/velocity in the Moon-fixed frame."),
    ChannelSpec("events", "Mission events", "", kind="events",
                description="Discrete events (periapsis, impact, fallback, stop)."),
    ChannelSpec("diagnostics", "Integrator diagnostics", "", kind="diagnostics",
                description="Backend-provided integrator/runtime diagnostics."),
    ChannelSpec("provenance", "Run provenance", "", kind="provenance",
                description="Requested/effective backend, models, hashes."),
)

CHANNELS: dict[str, ChannelSpec] = {spec.channel_id: spec for spec in _SPECS}

#: Scalar channels that map 1:1 onto TelemetrySample attributes.
SCALAR_SAMPLE_FIELDS: dict[str, str] = {
    "radius_m": "radius_m",
    "altitude_m": "altitude_m",
    "speed_m_s": "speed_m_s",
    "surface_radius_m": "surface_radius_m",
    "terrain_clearance_m": "terrain_clearance_m",
}


def channel_spec(channel_id: str) -> ChannelSpec | None:
    """Look up a channel; unknown element channels get a synthetic spec."""
    spec = CHANNELS.get(channel_id)
    if spec is not None:
        return spec
    if channel_id.startswith(ELEMENT_CHANNEL_PREFIX):
        name = channel_id[len(ELEMENT_CHANNEL_PREFIX):]
        return ChannelSpec(channel_id, name, "", derived=True,
                           description="Osculating orbital element (uncatalogued).")
    return None


__all__ = [
    "CHANNELS",
    "ELEMENT_CHANNEL_PREFIX",
    "SCALAR_SAMPLE_FIELDS",
    "ChannelKind",
    "ChannelSpec",
    "channel_spec",
]
