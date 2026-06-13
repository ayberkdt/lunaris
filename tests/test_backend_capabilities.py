# -*- coding: utf-8 -*-
"""
Tests for the central backend capability registry after the GPU SH split.

The two GPU spherical-harmonics implementations are kept distinct:
``numba_cuda_sh`` (degree <= 24 kernel-workspace limit) and ``torch_cuda_sh``
(arbitrary degree, no hard cap). ``gpu_sh`` survives only as a legacy alias to
``numba_cuda_sh``.

These lock the registry to the *existing* Monte Carlo behavior (the Numba
force-model matrix and degree limit) so the split changes labels/structure, not
physics. CPU-only; no CUDA device required.
"""

from __future__ import annotations

import pytest

from lunaris.common.type_defs import PerturbationFlags
from lunaris.core.backend_capabilities import (
    BACKEND_ALIASES,
    BACKEND_REGISTRY,
    REQUIRED_BACKEND_NAMES,
    get_capabilities,
    gpu_sh_max_degree,
    gpu_sh_supported_tiers,
    list_backend_names,
    resolve_backend_alias,
    unsupported_force_models,
)

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
    if attr == "enable_tides_k3":
        kwargs["enable_tides_k2"] = True
    return PerturbationFlags(**kwargs)


# ---------------------------------------------------------------------------
# Completeness & the two-implementation split
# ---------------------------------------------------------------------------

def test_required_backends_registered() -> None:
    for name in REQUIRED_BACKEND_NAMES:
        assert name in BACKEND_REGISTRY, f"required backend {name!r} missing"
        assert get_capabilities(name).name == name


def test_numba_and_torch_sh_are_distinct_backends() -> None:
    numba = get_capabilities("numba_cuda_sh")
    torch = get_capabilities("torch_cuda_sh")
    assert numba.name != torch.name
    assert numba.implementation == "numba_cuda"
    assert torch.implementation == "torch"
    assert numba.family == torch.family == "classic_sh"


def test_numba_cuda_sh_max_degree_is_24() -> None:
    caps = get_capabilities("numba_cuda_sh")
    assert caps.max_runtime_sh_degree == 24
    # Drift guard against the real kernel workspace constant.
    from lunaris.core.mc_propagator import GPU_SH_MAX_DEGREE

    assert caps.max_runtime_sh_degree == int(GPU_SH_MAX_DEGREE)
    assert gpu_sh_max_degree() == int(GPU_SH_MAX_DEGREE)
    assert gpu_sh_supported_tiers() == (int(GPU_SH_MAX_DEGREE),)


def test_torch_sh_backends_have_no_hard_degree_cap() -> None:
    assert get_capabilities("torch_cuda_sh").max_runtime_sh_degree is None
    assert get_capabilities("torch_cpu_sh").max_runtime_sh_degree is None


def test_gpu_sh_is_legacy_alias_to_numba() -> None:
    assert resolve_backend_alias("gpu_sh") == "numba_cuda_sh"
    assert "gpu_sh" in BACKEND_ALIASES
    # get_capabilities resolves the alias transparently.
    assert get_capabilities("gpu_sh").name == "numba_cuda_sh"


def test_unknown_backend_raises_with_known_and_aliases() -> None:
    with pytest.raises(KeyError) as exc:
        get_capabilities("totally_made_up")
    text = str(exc.value)
    assert "numba_cuda_sh" in text and "gpu_sh" in text


def test_list_backend_names_can_include_aliases() -> None:
    assert "gpu_sh" not in list_backend_names()
    assert "gpu_sh" in list_backend_names(include_aliases=True)


# ---------------------------------------------------------------------------
# Numba matrix locked to legacy MC behavior (physics unchanged by the split)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr", PERTURBATION_FLAGS)
def test_numba_cuda_sh_matrix_matches_legacy_classic_function(attr: str) -> None:
    from lunaris.core.mc_propagator import gpu_unsupported_features

    flags = _flags_with(attr)
    registry_blocked = bool(unsupported_force_models("numba_cuda_sh", flags))
    legacy_blocked = bool(gpu_unsupported_features(flags))
    assert registry_blocked == legacy_blocked, (
        f"numba_cuda_sh disagreement on {attr}: registry={registry_blocked} legacy={legacy_blocked}"
    )


@pytest.mark.parametrize("attr", PERTURBATION_FLAGS)
def test_gpu_st_lrps_matrix_matches_legacy_function(attr: str) -> None:
    from lunaris.core.mc_backend_policy import _st_lrps_gpu_unsupported_features

    flags = _flags_with(attr)
    registry_blocked = bool(unsupported_force_models("gpu_st_lrps_potential", flags))
    legacy_blocked = bool(_st_lrps_gpu_unsupported_features(flags))
    assert registry_blocked == legacy_blocked


def test_torch_sh_first_form_is_gravity_only() -> None:
    # The first torch-SH runtime form is gravity-only: any active perturbation
    # is reported as unsupported (forces an explicit fallback, never silent).
    flags = PerturbationFlags(enable_sh=True, enable_3rd_body_sun=True, enable_srp=True)
    assert "third_body_sun" in unsupported_force_models("torch_cuda_sh", flags)
    assert "srp" in unsupported_force_models("torch_cuda_sh", flags)


def test_cpu_backends_support_all_force_models() -> None:
    flags = PerturbationFlags(
        enable_sh=True, enable_3rd_body_sun=True, enable_3rd_body_earth=True,
        enable_earth_j2=True, enable_srp=True, enable_albedo=True, enable_thermal=True,
        enable_tides_k2=True, enable_tides_k3=True, enable_relativity_1pn=True,
    )
    assert unsupported_force_models("cpu_sh", flags) == ()
    assert unsupported_force_models("cpu_st_lrps", flags) == ()


# ---------------------------------------------------------------------------
# Back-compat property aliases
# ---------------------------------------------------------------------------

def test_backcompat_property_aliases() -> None:
    caps = get_capabilities("numba_cuda_sh")
    assert caps.max_sh_degree == caps.max_runtime_sh_degree == 24
    assert caps.gravity_kind == caps.family == "classic_sh"
    assert caps.supports_float64 is True and caps.supports_float32 is False
    torch = get_capabilities("torch_cuda_sh")
    assert torch.supports_float32 is True and torch.supports_float64 is True
