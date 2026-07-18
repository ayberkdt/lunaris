# tests/test_dynamics_contracts.py
"""
Dependency / provider *contract* tests for ``lunaris.core.dynamics``.

These tests pin down ``DynamicsEngine._validate_dependencies()`` and the strict
provider-extraction helpers WITHOUT requiring real SPICE kernels, gravity files,
or Numba RHS compilation. ``DynamicsEngine.__init__`` runs validation eagerly, so
constructing an engine is enough to exercise the contract; we never call
``build_rhs()`` for the heavy classical path here.

Why this matters: a misconfigured force model that *silently* runs with the wrong
provider (or no provider) is exactly the class of bug that produces plausible-but-
wrong physics. Each test asserts a clear, early failure.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import (
    DynamicsEngine,
    extract_gravity_strict,
)
from lunaris.core.dynamics.surrogate_bridge import _is_surrogate_gravity_provider
from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------
def _sc(**kw) -> SpacecraftProps:
    base = dict(mass_kg=10.0, area_m2=1.0, cd=2.2, cr=1.5)
    base.update(kw)
    return SpacecraftProps(**base)


class _Workspace:
    """Minimal Numba-style scratch workspace expected by the strict gravity contract."""

    def __init__(self, nmax: int):
        n = nmax + 1
        self.P = np.zeros((n, n), dtype=np.float64)
        self.dP = np.zeros((n, n), dtype=np.float64)
        self.cos_m = np.zeros(n, dtype=np.float64)
        self.sin_m = np.zeros(n, dtype=np.float64)


def _good_gravity_model(nmax: int = 2):
    """A minimal object satisfying the strict classical-SH gravity contract."""
    n = nmax + 1

    class _G:
        degree_max = nmax
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12
        Cnm = np.zeros((n, n), dtype=np.float64)
        Snm = np.zeros((n, n), dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        subdiag = np.zeros(n, dtype=np.float64)
        A = np.zeros(n, dtype=np.float64)
        B = np.zeros(n, dtype=np.float64)
        scale_m = np.ones(n, dtype=np.float64)
        ws = _Workspace(nmax)

    return _G()


class _StubSurrogate:
    model_kind = "st_lrps"
    R_ref_m = 1_737_400.0
    GM_m3s2 = 4.9048695e12

    def acceleration_fixed(self, r_fixed):
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)


# =============================================================================
# _validate_dependencies — required providers
# =============================================================================

def test_enable_sh_without_gravity_model_raises():
    with pytest.raises(ValueError, match="enable_sh=True but gravity_model is None"):
        DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                       gravity_model=None, allow_identity_rotation=True)


def test_enable_albedo_constant_requires_ephemeris_not_provider():
    # The facet model in constant-albedo mode needs no surface_provider, but it
    # does need the Sun vector from an ephemeris (it is reflected sunlight).
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_albedo=True),
            surface_provider=None,
            allow_identity_rotation=True,
        )


def test_enable_albedo_grid_mode_without_provider_raises():
    # Grid-sourced albedo requires a surface provider; the error fires before the
    # ephemeris check during dependency validation.
    with pytest.raises(ValueError, match="requires a surface_provider"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_albedo=True),
            albedo=AlbedoConfig(albedo_mode="albedo_grid"),
            surface_provider=None,
            allow_identity_rotation=True,
        )


@pytest.mark.parametrize("bad_sc", [
    _sc(area_m2=0.0),   # SRP/albedo need a positive optical area
    _sc(cr=0.0),        # cr must be in (0, 2.5]
    _sc(cr=3.0),
])
def test_srp_rejects_invalid_spacecraft_properties(bad_sc):
    # The spacecraft-property check fires before the ephemeris check, so no
    # ephemeris stub is required to reach it.
    with pytest.raises(ValueError):
        DynamicsEngine(bad_sc, PerturbationFlags(enable_sh=False, enable_srp=True),
                       allow_identity_rotation=True)


def test_third_body_sun_requires_ephemeris():
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_3rd_body_sun=True),
            ephem_manager=None,
            allow_identity_rotation=True,
        )


def test_earth_j2_without_params_raises():
    with pytest.raises(ValueError, match="earth_j2 params are None"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_earth_j2=True),
            earth_j2=None,
            allow_identity_rotation=True,
        )


# =============================================================================
# allow_identity_rotation only substitutes the frame quaternion
# =============================================================================

def test_allow_identity_rotation_substitutes_quaternion_only():
    """With SH enabled and no ephemeris, the engine needs q_i2f. That single
    dependency may be replaced by identity, but only when explicitly allowed."""
    grav = object()  # non-None, non-surrogate -> use_sh path needs q

    # allow_identity_rotation=False -> q must come from ephemeris -> error.
    with pytest.raises(ValueError, match="q_i2f"):
        DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                       gravity_model=grav, ephem_manager=None,
                       allow_identity_rotation=False)

    # allow_identity_rotation=True -> identity q is accepted; construction succeeds.
    eng = DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                         gravity_model=grav, ephem_manager=None,
                         allow_identity_rotation=True)
    assert eng is not None


def test_allow_identity_rotation_does_not_substitute_sun_or_earth_vectors():
    """Sun/Earth *vectors* are physical inputs, not a frame convention: identity
    rotation must NOT paper over a missing ephemeris when a vector is required."""
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_3rd_body_sun=True),
            ephem_manager=None,
            allow_identity_rotation=True,  # cannot rescue a missing Sun vector
        )


# =============================================================================
# Unsupported or missing dependencies must fail loudly (not silently no-op)
# =============================================================================

def test_thermal_constant_mode_is_supported_without_surface_provider():
    eng = DynamicsEngine(
        _sc(),
        PerturbationFlags(enable_sh=False, enable_thermal=True),
        thermal=ThermalConfig(thermal_mode="constant_temperature", facet_lat_count=2, facet_lon_count=4),
        allow_identity_rotation=True,
    )
    assert eng._requirements().use_thermal is True


def test_thermal_equilibrium_eclipse_requests_earth_vector():
    eng = DynamicsEngine(
        _sc(),
        PerturbationFlags(enable_sh=False, enable_thermal=True),
        thermal=ThermalConfig(
            thermal_mode="equilibrium_temperature",
            enable_eclipse=True,
            facet_lat_count=2,
            facet_lon_count=4,
        ),
        ephem_manager=object(),
    )
    req = eng._requirements()
    assert req.need_sun is True
    assert req.need_earth is True


def test_thermal_equilibrium_without_eclipse_only_requests_sun_vector():
    eng = DynamicsEngine(
        _sc(),
        PerturbationFlags(enable_sh=False, enable_thermal=True),
        thermal=ThermalConfig(
            thermal_mode="equilibrium_temperature",
            enable_eclipse=False,
            facet_lat_count=2,
            facet_lon_count=4,
        ),
        ephem_manager=object(),
    )
    req = eng._requirements()
    assert req.need_sun is True
    assert req.need_earth is False


def test_thermal_equilibrium_requires_sun_ephemeris():
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_thermal=True),
            thermal=ThermalConfig(thermal_mode="equilibrium_temperature", facet_lat_count=2, facet_lon_count=4),
            ephem_manager=None,
            allow_identity_rotation=True,
        )


def test_solid_tides_require_ephemeris_vectors():
    # enable_sh disabled so the gravity-model check does not fire first.
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(_sc(), PerturbationFlags(enable_sh=False, enable_tides_k2=True),
                       allow_identity_rotation=True)


# =============================================================================
# Surrogate-gravity provider detection
# =============================================================================

def test_surrogate_provider_detection():
    assert _is_surrogate_gravity_provider(_StubSurrogate()) is True
    assert _is_surrogate_gravity_provider(object()) is False
    assert _is_surrogate_gravity_provider(None) is False

    # An object claiming the kind but lacking acceleration_fixed is NOT a surrogate.
    class _Partial:
        model_kind = "st_lrps"
    assert _is_surrogate_gravity_provider(_Partial()) is False


def test_engine_routes_surrogate_provider_through_python_path():
    eng = DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                         gravity_model=_StubSurrogate(), ephem_manager=None,
                         allow_identity_rotation=True)
    req = eng._requirements()
    assert req.use_sh is True
    assert req.use_surrogate_gravity is True


# =============================================================================
# Strict classical-SH gravity contract (extract_gravity_strict)
# =============================================================================

def test_extract_gravity_strict_accepts_valid_model():
    nmax, r_ref, gm, *_ = extract_gravity_strict(_good_gravity_model(nmax=3))
    assert nmax == 3
    assert r_ref > 0.0 and gm > 0.0


# =============================================================================
# RHS state-vector length contract
# =============================================================================

def test_rhs_state_vector_guard_accepts_6_and_7_elements():
    from lunaris.core.dynamics.engine import _validate_rhs_state_vector

    y6 = _validate_rhs_state_vector(np.arange(6, dtype=np.float64))
    y7 = _validate_rhs_state_vector(np.arange(7, dtype=np.float64))
    assert y6.shape == (6,) and y7.shape == (7,)


@pytest.mark.parametrize("bad_length", [0, 5, 8, 12])
def test_rhs_state_vector_guard_hard_fails_on_other_lengths(bad_length):
    """Scientific-code contract: an RHS must never return a derivative with
    uninitialized entries. Lengths outside (6, 7) fail loudly instead of
    silently producing np.empty_like garbage for the extra components."""
    from lunaris.core.dynamics.engine import _validate_rhs_state_vector

    with pytest.raises(ValueError, match="6 elements"):
        _validate_rhs_state_vector(np.zeros(bad_length, dtype=np.float64))


def test_rhs_state_vector_guard_rejects_2d_input():
    from lunaris.core.dynamics.engine import _validate_rhs_state_vector

    with pytest.raises(ValueError, match="6 elements"):
        _validate_rhs_state_vector(np.zeros((2, 3), dtype=np.float64))


def _surrogate_rhs():
    eng = DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                         gravity_model=_StubSurrogate(), ephem_manager=None,
                         allow_identity_rotation=True)
    return eng.build_rhs()


def test_public_rhs_validates_every_call_and_results_are_identical():
    rhs = _surrogate_rhs()
    y = np.array([2.0e6, 0.0, 0.0, 0.0, 1.5e3, 0.0])
    first = rhs(0.0, y)
    second = rhs(0.0, y)
    third = rhs(0.0, list(y))  # non-ndarray input still coerced after first call
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)
    with pytest.raises(ValueError, match="6 elements"):
        rhs(0.0, np.zeros(5))


def test_trusted_solver_rhs_validates_first_call_only():
    eng = DynamicsEngine(
        _sc(),
        PerturbationFlags(enable_sh=True),
        gravity_model=_StubSurrogate(),
        ephem_manager=None,
        allow_identity_rotation=True,
    )
    rhs = eng._build_solver_rhs()
    y = np.array([2.0e6, 0.0, 0.0, 0.0, 1.5e3, 0.0])
    first = rhs(0.0, y)
    second = rhs(0.0, list(y))
    np.testing.assert_array_equal(first, second)


def test_rhs_rejects_bad_state_on_first_call_and_stays_armed_after_failure():
    rhs = _surrogate_rhs()
    with pytest.raises(ValueError, match="6 elements"):
        rhs(0.0, np.zeros(5))
    # A failed first call must not disarm the guard.
    with pytest.raises(ValueError, match="6 elements"):
        rhs(0.0, np.zeros((2, 3)))
    # A valid state then evaluates normally.
    out = rhs(0.0, np.array([2.0e6, 0.0, 0.0, 0.0, 1.5e3, 0.0]))
    assert out.shape == (6,) and np.all(np.isfinite(out))


def test_extract_gravity_strict_rejects_none():
    with pytest.raises(ValueError, match="gravity_model is None"):
        extract_gravity_strict(None)


def test_extract_gravity_strict_requires_degree_max():
    g = _good_gravity_model()
    delattr(type(g), "degree_max")
    with pytest.raises(AttributeError, match="degree_max"):
        extract_gravity_strict(g)


def test_extract_gravity_strict_requires_workspace():
    class _NoWs:
        degree_max = 2
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12
        Cnm = np.zeros((3, 3))
        Snm = np.zeros((3, 3))
        diag = np.zeros(3)
        subdiag = np.zeros(3)
        A = np.zeros(3)
        B = np.zeros(3)
        scale_m = np.ones(3)

    with pytest.raises(AttributeError, match="ws.*make_workspace|workspace"):
        extract_gravity_strict(_NoWs())


def test_extract_gravity_strict_rejects_too_small_coeff_arrays():
    class _Small:
        degree_max = 5            # claims degree 5 but coeff arrays are 1x1
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12
        Cnm = np.zeros((1, 1))
        Snm = np.zeros((1, 1))
        diag = np.zeros(1)
        subdiag = np.zeros(1)
        A = np.zeros(1)
        B = np.zeros(1)
        scale_m = np.ones(6)
        ws = _Workspace(5)

    with pytest.raises(ValueError, match="Cnm shape too small"):
        extract_gravity_strict(_Small())


@pytest.mark.parametrize("attr,value", [("GM_m3s2", -1.0), ("R_ref_m", 0.0)])
def test_extract_gravity_strict_rejects_nonpositive_scalars(attr, value):
    g = _good_gravity_model()
    setattr(type(g), attr, value)
    with pytest.raises(ValueError, match="must be positive"):
        extract_gravity_strict(g)


# =============================================================================
# Typed DynamicsRequirements contract (P1: no more dict drift)
# =============================================================================

def _compute_req(flags: PerturbationFlags, *, have_ephem: bool):
    from lunaris.core.dynamics.preparation import compute_requirements

    return compute_requirements(
        flags=flags,
        gravity_model=None,
        earth_j2=None,
        albedo=None,
        thermal=None,
        solid_tides=None,
        allow_identity_rotation=True,
        have_ephem=have_ephem,
    )


def test_compute_requirements_returns_frozen_typed_object():
    from lunaris.core.dynamics.contracts import DynamicsRequirements
    from lunaris.core.dynamics.preparation import DynamicsRequirements as CompatRequirements

    req = _compute_req(
        PerturbationFlags(enable_sh=False, enable_srp=True), have_ephem=True
    )
    assert isinstance(req, DynamicsRequirements)
    assert CompatRequirements is DynamicsRequirements
    # mypy-friendly usage: attribute access, no dict lookups.
    assert req.use_srp is True
    assert req.need_sun is True
    # Historical aliases stay wired to the shared ForceRequirements fields.
    assert req.need_q == req.force.need_q_i2f
    assert req.need_vectors == req.force.need_body_vectors
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        req.albedo_model = "simple"


def test_external_1pn_downgrade_returns_new_object_and_never_mutates_raw():
    from lunaris.core.dynamics.ephemeris_pack import _EphemPack
    from lunaris.core.dynamics.preparation import resolve_effective_requirements

    raw = _compute_req(
        PerturbationFlags(enable_sh=False, enable_relativity_1pn=True),
        have_ephem=True,
    )
    assert raw.use_rel_external is True

    q_ident = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    zeros = np.zeros((2, 3))
    degenerate = _EphemPack(
        dt_s=1.0, r_sun_tab_m=zeros, r_earth_tab_m=zeros,
        v_sun_tab_m_s=zeros, v_earth_tab_m_s=zeros, q_i2f_tab=q_ident
    )
    with pytest.warns(RuntimeWarning, match="external-body relativity terms disabled"):
        eff = resolve_effective_requirements(raw, degenerate)

    assert eff is not raw
    assert eff.use_rel_external is False
    assert raw.use_rel_external is True  # raw requirements never mutated
    assert eff.use_rel is True  # central-body Schwarzschild term survives


def test_external_relativity_downgrade_recomputes_derived_requirements():
    raw = _compute_req(
        PerturbationFlags(enable_sh=False, enable_srp=True, enable_relativity_1pn=True),
        have_ephem=True,
    )
    assert raw.use_rel_external is True
    assert raw.need_sun is True
    assert raw.need_earth is True
    assert raw.need_vectors is True
    assert raw.need_ephem is True

    downgraded = raw.without_external_relativity()

    assert downgraded.use_rel_external is False
    assert downgraded.need_sun is True
    assert downgraded.need_earth is False
    assert downgraded.need_vectors is True
    assert downgraded.need_ephem is True
    assert raw.need_sun is True
    assert raw.need_earth is True


def test_engine_prep_stores_contract_requirements_object():
    from lunaris.core.dynamics.contracts import DynamicsRequirements

    eng = DynamicsEngine(
        _sc(),
        PerturbationFlags(enable_sh=False),
        allow_identity_rotation=True,
    )
    eng.build_rhs()

    assert isinstance(eng._prep["req"], DynamicsRequirements)


def test_effective_requirements_pass_through_when_tables_present():
    from lunaris.core.dynamics.ephemeris_pack import _EphemPack
    from lunaris.core.dynamics.preparation import resolve_effective_requirements

    raw = _compute_req(
        PerturbationFlags(enable_sh=False, enable_relativity_1pn=True),
        have_ephem=True,
    )
    q_ident = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    ones = np.ones((2, 3))
    ep = _EphemPack(
        dt_s=1.0, r_sun_tab_m=ones, r_earth_tab_m=ones,
        v_sun_tab_m_s=np.zeros_like(ones), v_earth_tab_m_s=np.zeros_like(ones),
        q_i2f_tab=q_ident,
    )
    eff = resolve_effective_requirements(raw, ep)
    assert eff.use_rel_external is True


def test_requirements_to_dict_is_a_provenance_boundary_with_legacy_keys():
    req = _compute_req(
        PerturbationFlags(enable_sh=False, enable_srp=True), have_ephem=True
    )
    d = req.to_dict()
    # Historical key names preserved for serialization/provenance consumers.
    for key in ("need_q", "need_vectors", "albedo_model", "use_srp", "need_ephem"):
        assert key in d
    assert d["need_q"] == req.force.need_q_i2f
    assert d["use_srp"] is True
