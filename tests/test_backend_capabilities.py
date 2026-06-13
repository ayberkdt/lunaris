# -*- coding: utf-8 -*-
"""
Tests for the central backend capability registry.

These tests lock :mod:`lunaris.core.backend_capabilities` to the *existing*
Monte Carlo behavior so the registry is a faithful SSOT, not a new opinion:

* completeness — the brief's five required backends are registered;
* drift guard — the recorded classic-SH GPU degree limit equals the real CUDA
  workspace constant;
* consistency — for every individual perturbation flag, the registry's
  ``unsupported_force_models`` agrees (active/inactive) with the legacy
  ``gpu_unsupported_features`` (classic SH) and
  ``_st_lrps_gpu_unsupported_features`` (ST-LRPS torch path);
* the MC policy degree-limit helper now reads from the registry.

All tests are CPU-only and require no CUDA device.
"""

from __future__ import annotations

import pytest

from lunaris.common.type_defs import PerturbationFlags
from lunaris.core.backend_capabilities import (
    BACKEND_REGISTRY,
    REQUIRED_BACKEND_NAMES,
    get_capabilities,
    gpu_sh_max_degree,
    gpu_sh_supported_tiers,
    list_backend_names,
    unsupported_force_models,
)

# The optional perturbation flags (gravity/enable_sh excluded — it is the
# central force, not an optional perturbation).
PERTURBATION_FLAGS = (
    "enable_3rd_body_sun",
    "enable_3rd_body_earth",
    "enable_earth_j2",
    "enable_srp",
    "enable_albedo",
    "enable_thermal",
    "enable_tides_k2",
    "enable_tides_k3",
    "enable_relativity_1pn",
)


def _flags_with(attr: str) -> PerturbationFlags:
    kwargs = {"enable_sh": True, attr: True}
    # Config invariant: k3 solid tides require k2 to be enabled as well.
    if attr == "enable_tides_k3":
        kwargs["enable_tides_k2"] = True
    return PerturbationFlags(**kwargs)


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

def test_required_backends_are_registered() -> None:
    for name in REQUIRED_BACKEND_NAMES:
        assert name in BACKEND_REGISTRY, f"required backend {name!r} missing from registry"
        assert get_capabilities(name).name == name


def test_list_backend_names_is_sorted_and_complete() -> None:
    names = list_backend_names()
    assert names == tuple(sorted(BACKEND_REGISTRY))
    assert set(REQUIRED_BACKEND_NAMES).issubset(set(names))


def test_get_capabilities_unknown_raises_with_known_list() -> None:
    with pytest.raises(KeyError) as exc:
        get_capabilities("totally_made_up_backend")
    # The error must enumerate the known backends so misconfiguration is caught
    # before a run starts (never a silent fallback).
    assert "cpu_sh" in str(exc.value)


# ---------------------------------------------------------------------------
# Degree-limit drift guard
# ---------------------------------------------------------------------------

def test_recorded_gpu_sh_degree_matches_kernel_constant() -> None:
    from lunaris.core.mc_propagator import GPU_SH_MAX_DEGREE, GPU_SH_SUPPORTED_TIERS

    assert BACKEND_REGISTRY["gpu_sh"].max_sh_degree == int(GPU_SH_MAX_DEGREE)
    assert gpu_sh_max_degree() == int(GPU_SH_MAX_DEGREE)
    assert gpu_sh_supported_tiers() == tuple(int(v) for v in GPU_SH_SUPPORTED_TIERS)


def test_mc_policy_degree_limit_reads_registry() -> None:
    from lunaris.core.mc_backend_policy import _gpu_sh_limits

    assert _gpu_sh_limits() == (gpu_sh_max_degree(), gpu_sh_supported_tiers())


# ---------------------------------------------------------------------------
# Consistency with the existing MC capability functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr", PERTURBATION_FLAGS)
def test_gpu_sh_matrix_matches_legacy_classic_function(attr: str) -> None:
    """Classic-SH GPU: registry agrees with mc_propagator.gpu_unsupported_features."""
    from lunaris.core.mc_propagator import gpu_unsupported_features

    flags = _flags_with(attr)
    registry_blocked = bool(unsupported_force_models("gpu_sh", flags))
    legacy_blocked = bool(gpu_unsupported_features(flags))
    assert registry_blocked == legacy_blocked, (
        f"gpu_sh disagreement on {attr}: registry={registry_blocked} legacy={legacy_blocked}"
    )


@pytest.mark.parametrize("attr", PERTURBATION_FLAGS)
def test_gpu_st_lrps_matrix_matches_legacy_function(attr: str) -> None:
    """ST-LRPS GPU: registry agrees with policy._st_lrps_gpu_unsupported_features."""
    from lunaris.core.mc_backend_policy import _st_lrps_gpu_unsupported_features

    flags = _flags_with(attr)
    registry_blocked = bool(unsupported_force_models("gpu_st_lrps_potential", flags))
    legacy_blocked = bool(_st_lrps_gpu_unsupported_features(flags))
    assert registry_blocked == legacy_blocked, (
        f"gpu_st_lrps disagreement on {attr}: registry={registry_blocked} legacy={legacy_blocked}"
    )


def test_gpu_st_lrps_direct_matches_potential_matrix() -> None:
    """The direct and potential ST-LRPS paths share one capability matrix."""
    flags = PerturbationFlags(
        enable_sh=True,
        enable_3rd_body_sun=True,
        enable_srp=True,
        enable_albedo=True,
    )
    assert unsupported_force_models("gpu_st_lrps_direct", flags) == unsupported_force_models(
        "gpu_st_lrps_potential", flags
    )


# ---------------------------------------------------------------------------
# CPU backend supports everything
# ---------------------------------------------------------------------------

def test_cpu_backends_support_all_force_models() -> None:
    flags = PerturbationFlags(
        enable_sh=True,
        enable_3rd_body_sun=True,
        enable_3rd_body_earth=True,
        enable_earth_j2=True,
        enable_srp=True,
        enable_albedo=True,
        enable_thermal=True,
        enable_tides_k2=True,
        enable_tides_k3=True,
        enable_relativity_1pn=True,
    )
    assert unsupported_force_models("cpu_sh", flags) == ()
    assert unsupported_force_models("cpu_st_lrps", flags) == ()


def test_gpu_sh_blocks_surface_and_tides_but_allows_classic_perturbations() -> None:
    caps = get_capabilities("gpu_sh")
    assert caps.supports_third_body and caps.supports_earth_j2 and caps.supports_srp
    assert caps.supports_relativity_1pn
    assert not caps.supports_albedo and not caps.supports_thermal and not caps.supports_solid_tides
    assert caps.device == "cuda" and caps.default_dtype == "float64"
