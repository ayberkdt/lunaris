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
from lunaris.common.force_requirements import force_requirements
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine
from lunaris.physics.relativity_effects import _schwarzschild_components
from lunaris.physics.spherical_harmonics import compute_point_mass_acceleration
from lunaris.physics.third_body_effects import (
    EarthJ2Params,
    accel_j2_oblate_diff_numba,
    accel_third_body_numba,
)

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


def test_surrogate_gravity_cannot_disable_central_gravity_gate() -> None:
    with pytest.raises(ValueError, match="ST-LRPS backend requires enable_sh=True"):
        force_requirements(
            PerturbationFlags(enable_sh=False),
            gravity_uses_st_lrps=True,
        )


def test_dynamics_uses_ephemeris_provider_gm_for_third_body_and_earth_j2() -> None:
    class _Ephem:
        sun = np.array([1.5e11, 2.0e9, 0.0], dtype=np.float64)
        earth = np.array([3.84e8, -1.0e7, 2.0e6], dtype=np.float64)
        mu_sun = 1.2345e20
        mu_earth = 4.321e14

        def get_data_provider(self):
            return {
                "dt_s": 60.0,
                "r_sun_tab_m": np.vstack([self.sun, self.sun]),
                "r_earth_tab_m": np.vstack([self.earth, self.earth]),
                "q_i2f_tab": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (2, 1)),
                "mu_sun_m3s2": self.mu_sun,
                "mu_earth_m3s2": self.mu_earth,
            }

    y = np.array([1.84e6, 2.0e5, -5.0e4, 0.0, 1600.0, 0.0], dtype=np.float64)
    flags = PerturbationFlags(
        enable_sh=False,
        enable_3rd_body_sun=True,
        enable_3rd_body_earth=True,
        enable_earth_j2=True,
    )
    earth_j2 = EarthJ2Params(spin_axis_i=(0.0, 0.0, 1.0))
    engine = DynamicsEngine(
        sc_props=_SC,
        flags=flags,
        gravity_model=None,
        ephem_manager=_Ephem(),
        earth_j2=earth_j2,
        allow_identity_rotation=True,
    )

    dy = engine.build_rhs(force_rebuild=True)(0.0, y)
    central = np.array(
        compute_point_mass_acceleration(y[0], y[1], y[2], MU_MOON),
        dtype=np.float64,
    )
    sun = np.array(accel_third_body_numba(y[0], y[1], y[2], *_Ephem.sun, _Ephem.mu_sun))
    earth = np.array(accel_third_body_numba(y[0], y[1], y[2], *_Ephem.earth, _Ephem.mu_earth))
    j2 = np.array(
        accel_j2_oblate_diff_numba(
            y[0],
            y[1],
            y[2],
            *_Ephem.earth,
            _Ephem.mu_earth,
            earth_j2.r_eq_m,
            earth_j2.j2_coeff,
            0.0,
            0.0,
            1.0,
        )
    )

    np.testing.assert_allclose(dy[3:6], central + sun + earth + j2, rtol=1e-12, atol=1e-18)
    breakdown = engine.get_acceleration_breakdown(0.0, y)
    assert breakdown["3rd Body (Sun)"] == pytest.approx(float(np.linalg.norm(sun)))
    assert breakdown["3rd Body (Earth)"] == pytest.approx(float(np.linalg.norm(earth)))
    assert breakdown["3rd Body (Earth J2)"] == pytest.approx(float(np.linalg.norm(j2)))


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
