"""Shared Moon-fixed frame policy helpers.

The numerical backends may implement frame transforms differently (Numba,
Torch, or Python), but the public policy is intentionally small and explicit:
use a real ephemeris-backed Moon-fixed frame, or opt into an identity diagnostic
frame for smoke/regression work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FRAME_MODE_MOON_FIXED_EPHEMERIS: Literal["moon_fixed_ephemeris"] = "moon_fixed_ephemeris"
FRAME_MODE_IDENTITY_DIAGNOSTIC: Literal["identity_diagnostic"] = "identity_diagnostic"

FrameMode = Literal["moon_fixed_ephemeris", "identity_diagnostic"]
FrameModeInput = FrameMode | str | None

_MOON_FIXED_ALIASES = frozenset(
    {
        FRAME_MODE_MOON_FIXED_EPHEMERIS,
        "moon_fixed_slerp",
        "moon-fixed-slerp",
        "ephemeris",
        "ephemeris_wired",
        "match_dynamics_engine",
        "precomputed_slerp",
    }
)
_IDENTITY_ALIASES = frozenset(
    {
        FRAME_MODE_IDENTITY_DIAGNOSTIC,
        "identity",
        "identity_rotation",
        "identity-diagnostic",
        "inertial",
        "inertial_fixed_legacy",
    }
)


@dataclass(frozen=True, slots=True)
class FramePolicy:
    """Resolved frame policy consumed by propagation backends."""

    mode: FrameMode
    allow_identity_rotation: bool
    uses_frame_rotation: bool
    requires_ephemeris: bool
    provenance: str


def normalize_frame_mode(value: FrameModeInput) -> FrameMode | None:
    """Normalize frame-mode aliases to the canonical frame policy names."""

    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in _MOON_FIXED_ALIASES:
        return FRAME_MODE_MOON_FIXED_EPHEMERIS
    if text in _IDENTITY_ALIASES:
        return FRAME_MODE_IDENTITY_DIAGNOSTIC
    raise ValueError(
        f"Unknown frame_mode {value!r}; expected "
        f"{FRAME_MODE_MOON_FIXED_EPHEMERIS!r} or {FRAME_MODE_IDENTITY_DIAGNOSTIC!r}."
    )


def canonical_frame_mode(
    value: FrameModeInput,
    *,
    default: FrameMode = FRAME_MODE_MOON_FIXED_EPHEMERIS,
) -> FrameMode:
    """Return a concrete canonical frame mode, applying *default* for blanks."""

    mode = normalize_frame_mode(value)
    return default if mode is None else mode


def is_identity_frame_mode(value: FrameModeInput) -> bool:
    """Return True when *value* selects the diagnostic identity frame."""

    return normalize_frame_mode(value) == FRAME_MODE_IDENTITY_DIAGNOSTIC


def frame_mode_uses_rotation(value: FrameModeInput) -> bool:
    """Return True when *value* selects the ephemeris-backed Moon-fixed frame."""

    return normalize_frame_mode(value) == FRAME_MODE_MOON_FIXED_EPHEMERIS


def frame_provenance_label(mode: FrameMode, *, ephem_available: bool) -> str:
    """Return the canonical human-readable frame provenance string."""

    if mode == FRAME_MODE_IDENTITY_DIAGNOSTIC:
        return "identity (gravity field fixed in the integration frame)"
    if ephem_available:
        return "moon_fixed_ephemeris (ephemeris-wired q_i2f)"
    return "unresolved (ephemeris required)"


def resolve_frame_policy(
    *,
    ephem_manager: Any = None,
    allow_identity_rotation: bool | None = None,
    frame_mode: FrameModeInput = None,
    strict: bool = False,
    role: str = "backend",
) -> FramePolicy:
    """Resolve legacy boolean frame settings into an explicit frame policy.

    ``allow_identity_rotation`` remains accepted for backwards compatibility.
    When no explicit ``frame_mode`` is supplied, an available ephemeris selects
    the rotating Moon-fixed frame; otherwise ``allow_identity_rotation=True`` is
    required to select the identity diagnostic mode.
    """

    ephem_available = ephem_manager is not None
    explicit_mode = normalize_frame_mode(frame_mode)
    if explicit_mode is not None:
        mode = explicit_mode
        if mode == FRAME_MODE_IDENTITY_DIAGNOSTIC and ephem_available:
            raise ValueError(
                "frame_mode='identity_diagnostic' must not be combined with an "
                "ephemeris manager; omit ephem_manager for identity diagnostic runs."
            )
    elif ephem_available:
        mode = FRAME_MODE_MOON_FIXED_EPHEMERIS
    elif allow_identity_rotation is True:
        mode = FRAME_MODE_IDENTITY_DIAGNOSTIC
    else:
        mode = FRAME_MODE_MOON_FIXED_EPHEMERIS

    allow_identity = mode == FRAME_MODE_IDENTITY_DIAGNOSTIC
    uses_rotation = bool(mode == FRAME_MODE_MOON_FIXED_EPHEMERIS and ephem_available)
    requires_ephem = mode == FRAME_MODE_MOON_FIXED_EPHEMERIS
    provenance = frame_provenance_label(mode, ephem_available=ephem_available)

    if strict and (allow_identity or not uses_rotation):
        raise ValueError(
            f"{role} requires an ephemeris-backed Moon-fixed frame, got {provenance!r}."
        )

    return FramePolicy(
        mode=mode,
        allow_identity_rotation=allow_identity,
        uses_frame_rotation=uses_rotation,
        requires_ephemeris=requires_ephem,
        provenance=provenance,
    )


__all__ = [
    "FRAME_MODE_IDENTITY_DIAGNOSTIC",
    "FRAME_MODE_MOON_FIXED_EPHEMERIS",
    "FrameMode",
    "FrameModeInput",
    "FramePolicy",
    "canonical_frame_mode",
    "frame_provenance_label",
    "frame_mode_uses_rotation",
    "is_identity_frame_mode",
    "normalize_frame_mode",
    "resolve_frame_policy",
]
