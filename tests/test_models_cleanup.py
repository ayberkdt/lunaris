# -*- coding: utf-8 -*-
import pytest
import lunaris.physics as models
from lunaris.physics import surface_effects
from lunaris.physics.surrogate_gravity import _build_model_from_config

def test_models_init_exports():
    """Verify models/__init__.py canonical exports are correct."""
    assert hasattr(models, "compute_point_mass_acceleration"), "Expected compute_point_mass_acceleration to be exported."
    assert not hasattr(models, "accel_point_mass"), "accel_point_mass legacy symbol should be removed."
    assert not hasattr(models, "sh_accel_fixed_numba_dual"), "sh_accel_fixed_numba_dual legacy symbol should be removed."

def test_surface_effects_cleanup():
    """Verify dead/legacy surface-radiation kernels were removed from surface_effects."""
    assert not hasattr(surface_effects, "accel_thermal_lommel_seeliger"), "accel_thermal_lommel_seeliger should be removed."
    # accel_thermal_simple: dead engineering recoil kernel, superseded by the
    # Lambertian facet model in lunaris.physics.thermal_ir.
    assert not hasattr(surface_effects, "accel_thermal_simple"), "accel_thermal_simple (dead) should be removed."
    # accel_albedo_lommel_seeliger: unvalidated engineering proxy; albedo now has
    # exactly two backends (lambert_facets default + simple cannonball).
    assert not hasattr(surface_effects, "accel_albedo_lommel_seeliger"), "accel_albedo_lommel_seeliger (legacy) should be removed."
    assert not hasattr(surface_effects, "_ALBEDO_KERNELS"), "_ALBEDO_KERNELS dispatch dict should be removed."

def test_surrogate_gravity_fail_fast():
    """Verify surrogate_gravity._build_model_from_config rejects advanced architectures."""
    cfg_multiscale = {"architecture": "MultiScale", "activation": "sine", "hidden": 256, "depth": 4}
    with pytest.raises(ValueError, match="does not support MultiScale or advanced Residual models"):
        _build_model_from_config(cfg_multiscale)

    cfg_nbands = {"n_bands": 2, "activation": "sine", "hidden": 256, "depth": 4}
    with pytest.raises(ValueError, match="does not support MultiScale or advanced Residual models"):
        _build_model_from_config(cfg_nbands)

    # Should not raise for legacy supported formats
    cfg_legacy = {"activation": "sine", "hidden": 64, "depth": 2}
    net = _build_model_from_config(cfg_legacy)
    assert net is not None
