# ST_LRPS/models/__init__.py
# -*- coding: utf-8 -*-
"""
Lunar Simulation - Models package
=================================

Stable, high-level physics/environment model layer.

Design
------
- Strict imports only (package-relative).
- No backward-compat aliases.
- Lazy re-exports (PEP 562): importing `models` does NOT eagerly import heavy
  submodules. Symbols/modules are imported on first access and cached.
- File I/O lives under `loaders.*` and is intentionally not re-exported here.

Usage
-----
Preferred:
    from lunaris.physics import GravityModel, ThirdBodyModel

If you need the full submodule (advanced / bulk access):
    from lunaris.physics import spherical_harmonics   # lazy module
    from lunaris.physics import surface_effects       # lazy module
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Final, TYPE_CHECKING

# -----------------------------------------------------------------------------
# Lazy symbol exports: name -> (relative_module, attribute, short purpose)
# Keep this list small & stable. Add only truly user-facing entry points.
# -----------------------------------------------------------------------------
_EXPORTS: Final[dict[str, tuple[str, str, str]]] = {
    # Gravity / Spherical Harmonics (compute-only)
    "GravityModel": (".spherical_harmonics", "GravityModel", "High-level SH gravity wrapper."), 
    "SHWorkspace": (".spherical_harmonics", "SHWorkspace", "Reusable scratch buffers for SH kernels."),
    "compute_point_mass_acceleration": (".spherical_harmonics", "compute_point_mass_acceleration", "Baseline point-mass gravity."),
    "build_legendre_coeffs": (".spherical_harmonics", "build_legendre_coeffs", "Legendre recurrence constants."),
    "slice_gravity_model": (".spherical_harmonics", "slice_gravity_model", "Truncate/pad SH coefficient matrices."),
    "make_sh_workspace": (".spherical_harmonics", "make_sh_workspace", "Allocate SHWorkspace for a max degree."),
    "sh_accel_fixed": (".spherical_harmonics", "sh_accel_fixed", "Fixed-degree SH acceleration (dispatcher)."),
    "sh_accel_fixed_numba": (".spherical_harmonics", "sh_accel_fixed_numba", "Numba kernel: fixed-degree SH accel."),
    "sh_accel_adaptive_blend_numba": (".spherical_harmonics", "sh_accel_adaptive_blend_numba", "Numba kernel: adaptive/blended degree SH accel."),

    # Ephemeris & SPICE
    "SpiceBuildConfig": (".ephemeris", "SpiceBuildConfig", "SPICE build configuration."), 
    "EphemerisTables": (".ephemeris", "EphemerisTables", "Prebuilt ephemeris tables container."),
    "EphemerisManager": (".ephemeris", "EphemerisManager", "Ephemeris accessor/manager."),
    "build_spice_tables": (".ephemeris", "build_spice_tables", "Build ephemeris tables from SPICE."),
    "get_ephem_state": (".ephemeris", "get_ephem_state", "Interpolate/query ephemeris state."),

    # Relativity
    "RelativityModel": (".relativity_effects", "RelativityModel", "Relativistic corrections model."),
    "calc_schwarzschild_accel": (".relativity_effects", "calc_schwarzschild_accel", "Schwarzschild acceleration term."),

    # Third-body effects
    "ThirdBodyModel": (".third_body_effects", "ThirdBodyModel", "Earth/Sun third-body effects model."),
    "LoveParams": (".third_body_effects", "LoveParams", "Solid-tide Love parameter set."),
    "EarthJ2Params": (".third_body_effects", "EarthJ2Params", "Earth J2 differential parameter set."),
    "calc_3rd_body_accel": (".third_body_effects", "calc_3rd_body_accel", "Analytic third-body differential accel."),
    "calc_j2_oblate_diff_accel": (".third_body_effects", "calc_j2_oblate_diff_accel", "Analytic J2 differential accel."),
    "accel_third_body_numba": (".third_body_effects", "accel_third_body_numba", "Numba kernel: third-body accel."),
    "accel_solid_tide": (".third_body_effects", "accel_solid_tide", "Solid-tide acceleration term."),

    # Solid-body tides
    "calc_solid_tide_accel": (".solid_tides", "calc_solid_tide_accel", "Elastic lunar solid-tide acceleration."),
    "accel_solid_tides_numba": (".solid_tides", "accel_solid_tides_numba", "Numba kernel: elastic solid-tide accel."),
    "solid_tide_potential_degree": (".solid_tides", "solid_tide_potential_degree", "Solid-tide disturbing potential."),

    # Solar effects (SRP + shadow)
    "SRPConfig": (".solar_effects", "SRPConfig", "Solar Radiation Pressure configuration."), 
    "compute_srp_accel": (".solar_effects", "compute_srp_accel", "Compute SRP acceleration."), 
    "accel_srp": (".solar_effects", "accel_srp", "SRP acceleration helper (low-level)."), 
    "moon_shadow_factor_conical": (".solar_effects", "moon_shadow_factor_conical", "Conical umbra shadow factor."), 
    "in_moon_umbra_conical": (".solar_effects", "in_moon_umbra_conical", "Umbra test (conical)."), 
}

# -----------------------------------------------------------------------------
# Lazy modules (attribute name -> short purpose)
# Exposed so users can do `from lunaris.physics import surface_effects` without eager import.
# -----------------------------------------------------------------------------
_LAZY_MODULES: Final[dict[str, str]] = {
    "spherical_harmonics": "Spherical harmonics gravity (compute-only).", 
    "ephemeris": "Ephemeris & SPICE utilities (can be heavy).", 
    "relativity_effects": "Relativistic corrections.", 
    "third_body_effects": "Earth/Sun third-body effects.", 
    "solid_tides": "Elastic lunar solid-body tides.",
    "solar_effects": "SRP + eclipse/shadow geometry.", 
    "surface_effects": "Surface environment (topography/albedo/thermal; often heavy).", 
}

# Public API list (symbols + module shortcuts)
__all__: tuple[str, ...] = tuple(_EXPORTS.keys()) + tuple(_LAZY_MODULES.keys())


def __getattr__(name: str) -> Any:
    """
    Lazy-load public symbols and selected submodules.

    Behavior
    --------
    - If `name` is a public symbol in _EXPORTS: import its module, fetch attribute,
      cache it in this package namespace, and return it.
    - If `name` is a lazy module in _LAZY_MODULES: import the submodule, cache, return it.
    - Otherwise: AttributeError with helpful hints.
    """
    # 1) Public symbols
    if name in _EXPORTS:
        rel_mod, attr, _purpose = _EXPORTS[name]
        mod = importlib.import_module(rel_mod, __name__)
        try:
            obj = getattr(mod, attr)
        except AttributeError as e:
            # surface mismatch etc. should be loud and precise
            raise AttributeError(
                f"Failed to import public symbol {name!r}: {rel_mod}.{attr} is missing"
            ) from e
        globals()[name] = obj  # cache
        return obj

    # 2) Lazy modules (exposed as attributes)
    if name in _LAZY_MODULES:
        full_name = f"{__name__}.{name}"
        try:
            mod = importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as e:
            # If it's not "module file missing", let dependencies bubble
            if getattr(e, "name", None) != full_name:
                raise
            hint = (
                f"Could not import lazy module {full_name!r}.\n"
                f"Purpose: {_LAZY_MODULES[name]}\n"
                "Likely causes:\n"
                f" - Missing file: models/{name}.py\n"
                " - Repo root not on PYTHONPATH\n"
                "Fix:\n"
                " - Run from project root, or install editable: pip install -e .\n"
                f"Python: {sys.executable}"
            )
            raise AttributeError(hint) from e

        globals()[name] = mod  # cache
        return mod

    valid_syms = ", ".join(sorted(_EXPORTS))
    valid_mods = ", ".join(sorted(_LAZY_MODULES))
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}.\n"
        f"Public symbols: {valid_syms}\n"
        f"Lazy modules: {valid_mods}"
    )


def __dir__() -> list[str]:
    """Expose public exports + lazy modules for IDE completion."""
    return sorted(set(globals()) | set(__all__))


# -----------------------------------------------------------------------------
# Static type checking support (no runtime cost)
# -----------------------------------------------------------------------------
if TYPE_CHECKING:
    from . import (  # noqa: F401
        spherical_harmonics,
        ephemeris,
        relativity_effects,
        third_body_effects,
        solid_tides,
        solar_effects,
        surface_effects,
    )

    from .spherical_harmonics import (  # noqa: F401
        GravityModel,
        SHWorkspace,
        compute_point_mass_acceleration,
        build_legendre_coeffs,
        slice_gravity_model,
        make_sh_workspace,
        sh_accel_fixed,
        sh_accel_fixed_numba,
        sh_accel_adaptive_blend_numba,
    )
    from .ephemeris import (  # noqa: F401
        SpiceBuildConfig,
        EphemerisTables,
        EphemerisManager,
        build_spice_tables,
        get_ephem_state,
    )
    from .relativity_effects import (  # noqa: F401
        RelativityModel,
        calc_schwarzschild_accel,
    )
    from .third_body_effects import (  # noqa: F401
        ThirdBodyModel,
        LoveParams,
        EarthJ2Params,
        calc_3rd_body_accel,
        calc_j2_oblate_diff_accel,
        accel_third_body_numba,
        accel_solid_tide,
    )
    from .solid_tides import (  # noqa: F401
        calc_solid_tide_accel,
        accel_solid_tides_numba,
        solid_tide_potential_degree,
    )
    from .solar_effects import (  # noqa: F401
        SRPConfig,
        compute_srp_accel,
        accel_srp,
        moon_shadow_factor_conical,
        in_moon_umbra_conical,
    )
