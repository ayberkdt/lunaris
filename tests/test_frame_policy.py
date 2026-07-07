from __future__ import annotations

import pytest

from lunaris.common.frame_policy import (
    FRAME_MODE_IDENTITY_DIAGNOSTIC,
    FRAME_MODE_MOON_FIXED_EPHEMERIS,
    normalize_frame_mode,
    resolve_frame_policy,
)
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine


def _sc() -> SpacecraftProps:
    return SpacecraftProps(mass_kg=10.0, area_m2=1.0, cd=2.2, cr=1.5)


def test_frame_policy_normalizes_legacy_aliases() -> None:
    assert normalize_frame_mode("match_dynamics_engine") == FRAME_MODE_MOON_FIXED_EPHEMERIS
    assert normalize_frame_mode("precomputed_slerp") == FRAME_MODE_MOON_FIXED_EPHEMERIS
    assert normalize_frame_mode("inertial_fixed_legacy") == FRAME_MODE_IDENTITY_DIAGNOSTIC


def test_frame_policy_legacy_bool_resolves_to_identity_without_ephemeris() -> None:
    policy = resolve_frame_policy(allow_identity_rotation=True)

    assert policy.mode == FRAME_MODE_IDENTITY_DIAGNOSTIC
    assert policy.allow_identity_rotation is True
    assert policy.uses_frame_rotation is False


def test_frame_policy_strict_rejects_identity_or_unresolved_frame() -> None:
    with pytest.raises(ValueError, match="requires an ephemeris-backed"):
        resolve_frame_policy(
            allow_identity_rotation=True,
            strict=True,
            role="paper_safe screening",
        )
    with pytest.raises(ValueError, match="requires an ephemeris-backed"):
        resolve_frame_policy(
            frame_mode=FRAME_MODE_MOON_FIXED_EPHEMERIS,
            strict=True,
            role="paper_safe screening",
        )


def test_frame_policy_rejects_ambiguous_identity_with_ephemeris() -> None:
    with pytest.raises(ValueError, match="must not be combined"):
        resolve_frame_policy(
            ephem_manager=object(),
            frame_mode=FRAME_MODE_IDENTITY_DIAGNOSTIC,
        )


def test_dynamics_engine_accepts_explicit_identity_frame_mode() -> None:
    eng = DynamicsEngine(
        _sc(),
        PerturbationFlags(enable_sh=True),
        gravity_model=object(),
        ephem_manager=None,
        frame_mode=FRAME_MODE_IDENTITY_DIAGNOSTIC,
    )

    assert eng.frame_mode == FRAME_MODE_IDENTITY_DIAGNOSTIC
    assert eng.allow_identity_rotation is True


def test_dynamics_engine_moon_fixed_frame_mode_requires_q_when_needed() -> None:
    with pytest.raises(ValueError, match="q_i2f"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=True),
            gravity_model=object(),
            ephem_manager=None,
            frame_mode=FRAME_MODE_MOON_FIXED_EPHEMERIS,
        )
