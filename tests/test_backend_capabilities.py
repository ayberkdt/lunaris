"""
Tests for the central backend capability registry after the GPU SH split.

The two GPU spherical-harmonics implementations are kept distinct:
``numba_cuda_sh`` (degree <= 24 kernel-workspace limit) and ``torch_cuda_sh``
(no hard-coded degree ceiling). Backend requests use canonical names only.

These lock the registry to the *existing* batch propagation behavior (the Numba
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
    list_backend_names,
    numba_cuda_sh_max_degree,
    numba_cuda_sh_supported_tiers,
    resolve_effective_dtype,
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
    from lunaris.core.batch_propagator import NUMBA_CUDA_SH_MAX_DEGREE

    assert caps.max_runtime_sh_degree == int(NUMBA_CUDA_SH_MAX_DEGREE)
    assert numba_cuda_sh_max_degree() == int(NUMBA_CUDA_SH_MAX_DEGREE)
    assert numba_cuda_sh_supported_tiers() == (int(NUMBA_CUDA_SH_MAX_DEGREE),)


def test_torch_sh_backends_have_no_hard_degree_cap() -> None:
    assert get_capabilities("torch_cuda_sh").max_runtime_sh_degree is None
    assert get_capabilities("torch_cpu_sh").max_runtime_sh_degree is None


def test_no_backend_aliases_are_registered() -> None:
    assert BACKEND_ALIASES == {}
    with pytest.raises(KeyError):
        get_capabilities("totally_made_up")


def test_unknown_backend_raises_with_known_and_aliases() -> None:
    with pytest.raises(KeyError) as exc:
        get_capabilities("totally_made_up")
    text = str(exc.value)
    assert "numba_cuda_sh" in text
    assert "no backend aliases are registered" in text


def test_list_backend_names_can_include_aliases() -> None:
    assert list_backend_names(include_aliases=True) == list_backend_names()


# ---------------------------------------------------------------------------
# Numba matrix locked to the existing batch behavior (physics unchanged by the split)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr", PERTURBATION_FLAGS)
def test_numba_cuda_sh_matrix_matches_legacy_classic_function(attr: str) -> None:
    from lunaris.core.batch_propagator import gpu_unsupported_features

    flags = _flags_with(attr)
    registry_blocked = bool(unsupported_force_models("numba_cuda_sh", flags))
    legacy_blocked = bool(gpu_unsupported_features(flags))
    assert registry_blocked == legacy_blocked, (
        f"numba_cuda_sh disagreement on {attr}: registry={registry_blocked} legacy={legacy_blocked}"
    )


@pytest.mark.parametrize("attr", PERTURBATION_FLAGS)
def test_gpu_st_lrps_matrix_matches_legacy_function(attr: str) -> None:
    from lunaris.batch.backend_policy import _st_lrps_gpu_unsupported_features

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


# ---------------------------------------------------------------------------
# R09: the capability registry is the single source of support decisions
# ---------------------------------------------------------------------------

def test_st_lrps_gpu_unsupported_delegates_to_registry() -> None:
    # The ST-LRPS GPU support decision must match the registry exactly, so a new
    # force flag is honored without editing backend_policy.
    from lunaris.batch.backend_policy import _st_lrps_gpu_unsupported_features

    flags = PerturbationFlags(
        enable_sh=True, enable_3rd_body_sun=True, enable_srp=True,
        enable_albedo=True, enable_relativity_1pn=True,
    )
    assert (
        set(_st_lrps_gpu_unsupported_features(flags))
        == set(unsupported_force_models("gpu_st_lrps_potential", flags))
    )
    assert _st_lrps_gpu_unsupported_features(None) == ()


def test_resolve_effective_dtype_honors_supported_request() -> None:
    # cpu_sh supports float64; torch_cuda_sh supports both — a supported request
    # passes through unchanged, no downgrade.
    res = resolve_effective_dtype("float64", "torch_cuda_sh")
    assert (res.requested, res.effective, res.downgraded) == ("float64", "float64", False)
    res32 = resolve_effective_dtype("float32", "torch_cuda_sh")
    assert (res32.effective, res32.downgraded) == ("float32", False)


def test_resolve_effective_dtype_records_downgrade_when_unsupported() -> None:
    # cpu_sh runs float64 only; a float32 request is downgraded, not silently run.
    res = resolve_effective_dtype("float32", "cpu_sh")
    assert res.requested == "float32"
    assert res.effective == "float64"
    assert res.downgraded is True
    assert "float32" in res.reason and "cpu_sh" in res.reason


def test_resolve_effective_dtype_defaults_to_backend_default() -> None:
    # An empty/None request falls back to the backend's registered default.
    res = resolve_effective_dtype(None, "gpu_st_lrps_potential")
    assert res.requested == get_capabilities("gpu_st_lrps_potential").default_dtype
    assert res.downgraded is False


def test_backend_policy_holds_no_hardcoded_perturbation_flag_literals() -> None:
    # Contract: support/fallback decisions live only in the capability registry.
    # backend_policy must not name a perturbation flag attribute directly (that
    # would be a second, drift-prone source of truth). enable_sh is central
    # gravity, not an optional perturbation, so it is allowed.
    import pathlib

    from lunaris.batch import backend_policy

    source = pathlib.Path(backend_policy.__file__).read_text(encoding="utf-8")
    offenders = [flag for flag in PERTURBATION_FLAGS if flag in source]
    assert not offenders, (
        "backend_policy.py hardcodes perturbation flag literals "
        f"{offenders}; route support decisions through "
        "core.backend_capabilities.unsupported_force_models instead."
    )
