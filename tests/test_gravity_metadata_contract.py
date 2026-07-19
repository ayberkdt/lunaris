"""Gravity coefficient frame/tide metadata enforcement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lunaris.common.type_defs import PerturbationFlags, SolidTideConfig
from lunaris.core.dynamics.preparation import (
    _validate_gravity_metadata_contract,
    compute_requirements,
)
from lunaris.physics.spherical_harmonics import GravityModelMetadata


def _metadata(
    *,
    frame: str = "MOON_PA",
    tide: str = "tide_free",
    normalization: str = "fully_normalized_4pi",
    source_gm_m3s2: float = 4.9028003063302e12,
) -> GravityModelMetadata:
    return GravityModelMetadata(
        model_id="test_model",
        source_sha256="a" * 64,
        normalization=normalization,
        coefficient_frame=frame,
        tide_system=tide,
        source_gm_m3s2=source_gm_m3s2,
        source_radius_m=1_738_000.0,
    )


def _requirements(*, tides: bool = False):
    flags = PerturbationFlags(enable_sh=True, enable_tides_k2=tides)
    return compute_requirements(
        flags=flags,
        gravity_model=object(),
        earth_j2=None,
        albedo=None,
        thermal=None,
        solid_tides=SolidTideConfig(),
        allow_identity_rotation=False,
        have_ephem=True,
    )


def test_gravity_frame_must_match_ephemeris_fixed_frame() -> None:
    gravity = SimpleNamespace(metadata=_metadata(frame="MOON_ME"))
    ephemeris = SimpleNamespace(tables=SimpleNamespace(fixed_frame="MOON_PA"))

    with pytest.raises(ValueError, match="coefficient frame"):
        _validate_gravity_metadata_contract(
            req=_requirements(), gravity_model=gravity, ephem_manager=ephemeris, strict=False
        )


def test_solid_tides_require_tide_free_static_gravity() -> None:
    gravity = SimpleNamespace(metadata=_metadata(tide="mean_tide"))
    ephemeris = SimpleNamespace(tables=SimpleNamespace(fixed_frame="MOON_PA"))

    with pytest.raises(ValueError, match="tide-free"):
        _validate_gravity_metadata_contract(
            req=_requirements(tides=True),
            gravity_model=gravity,
            ephem_manager=ephemeris,
            strict=False,
        )


def test_strict_run_requires_complete_gravity_metadata() -> None:
    gravity = SimpleNamespace(metadata=_metadata(frame="unspecified", tide="unspecified"))
    ephemeris = SimpleNamespace(tables=SimpleNamespace(fixed_frame="MOON_PA"))

    with pytest.raises(ValueError, match="complete model metadata"):
        _validate_gravity_metadata_contract(
            req=_requirements(), gravity_model=gravity, ephem_manager=ephemeris, strict=True
        )


def test_de_specific_pa_alias_matches_generic_moon_pa() -> None:
    gravity = SimpleNamespace(metadata=_metadata(frame="MOON_PA"))
    ephemeris = SimpleNamespace(tables=SimpleNamespace(fixed_frame="MOON_PA_DE440"))

    _validate_gravity_metadata_contract(
        req=_requirements(), gravity_model=gravity, ephem_manager=ephemeris, strict=True
    )


def test_strict_run_requires_explicit_4pi_normalization() -> None:
    gravity = SimpleNamespace(metadata=_metadata(normalization="normalized"))
    ephemeris = SimpleNamespace(tables=SimpleNamespace(fixed_frame="MOON_PA"))

    with pytest.raises(ValueError, match="4-pi"):
        _validate_gravity_metadata_contract(
            req=_requirements(), gravity_model=gravity, ephem_manager=ephemeris, strict=True
        )


def test_strict_run_rejects_nonlunar_gravity_constants() -> None:
    gravity = SimpleNamespace(metadata=_metadata(source_gm_m3s2=3.986004418e14))
    ephemeris = SimpleNamespace(tables=SimpleNamespace(fixed_frame="MOON_PA"))

    with pytest.raises(ValueError, match="lunar-compatible"):
        _validate_gravity_metadata_contract(
            req=_requirements(), gravity_model=gravity, ephem_manager=ephemeris, strict=True
        )
