"""Limiting cases, model on/off behaviour, and unsupported-combination errors.

The ``astrodynamics-validation`` gate requires that toggling a perturbation flag
changes the result, and that unsupported configurations fail loudly rather than
silently dropping a force. This module exercises the ``DynamicsEngine`` contract
at that level, plus the closed-form relativity kernel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import C_LIGHT, MU_MOON
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine
from lunaris.physics.relativity_effects import _schwarzschild_components

_SC = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)


def _state(r_km: float = 1837.4, v_ms: float = 1600.0) -> np.ndarray:
    y = np.zeros(6, dtype=np.float64)
    y[0] = r_km * 1000.0
    y[4] = v_ms
    return y


def _point_mass_engine(**flag_overrides) -> DynamicsEngine:
    defaults = dict(
        enable_sh=False,
        enable_3rd_body_sun=False,
        enable_3rd_body_earth=False,
        enable_srp=False,
        enable_relativity_1pn=False,
        enable_earth_j2=False,
    )
    defaults.update(flag_overrides)
    flags = PerturbationFlags(**defaults)
    return DynamicsEngine(
        sc_props=_SC,
        flags=flags,
        gravity_model=None,
        ephem_manager=None,
        surface_provider=None,
        earth_j2=None,
        allow_identity_rotation=True,
    )


# -----------------------------------------------------------------------------
# Model on/off must change the result
# -----------------------------------------------------------------------------
def test_relativity_toggle_changes_acceleration() -> None:
    y = _state()
    a_off = _point_mass_engine(enable_relativity_1pn=False).build_rhs(force_rebuild=True)(0.0, y)
    a_on = _point_mass_engine(enable_relativity_1pn=True).build_rhs(force_rebuild=True)(0.0, y)
    delta = np.linalg.norm(a_on[3:6] - a_off[3:6])
    assert delta > 0.0
    # The 1PN correction is tiny relative to Newtonian gravity but nonzero.
    assert delta < 1e-3 * np.linalg.norm(a_off[3:6])


def test_point_mass_engine_is_newtonian() -> None:
    y = _state()
    a = _point_mass_engine().build_rhs(force_rebuild=True)(0.0, y)
    r = y[:3]
    expected = -MU_MOON * r / np.linalg.norm(r) ** 3
    np.testing.assert_allclose(a[3:6], expected, rtol=1e-12, atol=0.0)


# -----------------------------------------------------------------------------
# Unsupported combinations must raise (never silently drop a force)
# -----------------------------------------------------------------------------
def test_sh_without_gravity_model_raises() -> None:
    with pytest.raises(ValueError, match="gravity_model"):
        DynamicsEngine(
            sc_props=_SC,
            flags=PerturbationFlags(enable_sh=True),
            gravity_model=None,
            ephem_manager=None,
            allow_identity_rotation=True,
        )


def test_third_body_without_ephemeris_raises() -> None:
    with pytest.raises(ValueError, match="[Ee]phemeris"):
        DynamicsEngine(
            sc_props=_SC,
            flags=PerturbationFlags(enable_sh=False, enable_3rd_body_sun=True),
            gravity_model=None,
            ephem_manager=None,
            allow_identity_rotation=True,
        )


def test_earth_j2_without_params_raises() -> None:
    with pytest.raises(ValueError, match="earth_j2"):
        DynamicsEngine(
            sc_props=_SC,
            flags=PerturbationFlags(enable_sh=False, enable_earth_j2=True),
            gravity_model=None,
            ephem_manager=None,
            earth_j2=None,
            allow_identity_rotation=True,
        )


def test_tides_k3_requires_k2_at_flag_construction() -> None:
    with pytest.raises(ValueError, match="k3"):
        PerturbationFlags(enable_sh=False, enable_tides_k3=True, enable_tides_k2=False)


def test_srp_with_zero_area_raises() -> None:
    bad_sc = SpacecraftProps(mass_kg=12.0, area_m2=0.0, cr=1.3)
    with pytest.raises(ValueError, match="area_m2"):
        DynamicsEngine(
            sc_props=bad_sc,
            flags=PerturbationFlags(enable_sh=False, enable_srp=True),
            gravity_model=None,
            ephem_manager=None,
            allow_identity_rotation=True,
        )


# -----------------------------------------------------------------------------
# Relativity closed form
# -----------------------------------------------------------------------------
def test_schwarzschild_matches_closed_form() -> None:
    r = np.array([1.95e6, 3.0e5, -1.2e5])
    v = np.array([50.0, 1500.0, -200.0])
    a = np.array(_schwarzschild_components(r[0], r[1], r[2], v[0], v[1], v[2], MU_MOON))

    rn = np.linalg.norm(r)
    v2 = float(v @ v)
    rv = float(r @ v)
    expected = (MU_MOON / (C_LIGHT**2 * rn**3)) * ((4.0 * MU_MOON / rn - v2) * r + 4.0 * rv * v)
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=0.0)


def test_schwarzschild_guards_singularity() -> None:
    a = _schwarzschild_components(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, MU_MOON)
    assert a == (0.0, 0.0, 0.0)
