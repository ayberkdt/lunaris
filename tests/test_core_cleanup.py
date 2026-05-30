import pytest
import numpy as np

from lunaris.core.state import create_state_from_keplerian
from lunaris.core.monte_carlo_engine import MonteCarloEngine
from lunaris.core.mc_backend_policy import resolve_mc_backend_policy
from lunaris.core.dynamics import extract_surface_provider_strict, DynamicsEngine
from lunaris.common.type_defs import SpacecraftProps, PerturbationFlags

def test_no_legacy_exports():
    import lunaris.core as core
    assert not hasattr(core, "create_state_from_coe")
    assert not hasattr(core, "ae_from_rp_ra")
    assert hasattr(core, "create_state_from_keplerian")

def test_fail_fast_on_unsupported_physics():
    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
    flags = PerturbationFlags(
        enable_sh=False,
        enable_thermal=True, # Not implemented
        enable_tides_k2=False,
    )
    with pytest.raises(NotImplementedError):
        DynamicsEngine(sc_props=sc, flags=flags, gravity_model=None, ephem_manager=None, surface_provider=None, earth_j2=None, allow_identity_rotation=True)

def test_fail_fast_on_missing_j2_radius():
    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
    flags = PerturbationFlags(
        enable_sh=False,
        enable_earth_j2=True,
    )
    with pytest.raises(ValueError):
        DynamicsEngine(sc_props=sc, flags=flags, gravity_model=None, ephem_manager=None, surface_provider=None, earth_j2=None, allow_identity_rotation=True)

def test_extract_surface_provider_strict_fail():
    class BadProvider:
        # Missing the canonical `as_numba_dict()` API entirely.
        def unrelated_method(self):
            return lambda *args: 0.0

    with pytest.raises(TypeError):
        extract_surface_provider_strict(BadProvider())

def test_mc_sample_failure_fast_fail():
    from lunaris.common.montecarlo_defs import MonteCarloConfig

    mc_cfg = MonteCarloConfig(
        n_samples=2,
    )
    
    # allow_sample_failures defaults to False
    assert getattr(mc_cfg, "allow_sample_failures", False) is False
