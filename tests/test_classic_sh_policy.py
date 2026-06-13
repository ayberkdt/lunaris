# -*- coding: utf-8 -*-
"""
Degree-aware classic-SH backend selection policy (task §8).

These test the pure decision function ``select_classic_sh_backend`` — the layer
that chooses between numba_cuda_sh / torch_cuda_sh / cpu_sh — without touching
the live MC runtime. The key guarantees: degree > 24 is routed to torch before
CPU, the requested degree is never silently clipped, and explicit requests obey
the fallback policy.
"""

from __future__ import annotations

from lunaris.core.mc_backend_policy import select_classic_sh_backend


def _sel(**kw):
    base = dict(
        requested_backend="auto",
        requested_degree=20,
        numba_cuda_available=False,
        torch_cuda_available=False,
    )
    base.update(kw)
    return select_classic_sh_backend(**base)


# --- §17 auto-policy table -------------------------------------------------

def test_degree20_numba_available_picks_numba() -> None:
    d = _sel(requested_degree=20, numba_cuda_available=True, torch_cuda_available=True)
    assert d.backend == "numba_cuda_sh"
    assert d.fallback_applied is False


def test_degree20_numba_unavailable_torch_available_picks_torch() -> None:
    d = _sel(requested_degree=20, numba_cuda_available=False, torch_cuda_available=True)
    assert d.backend == "torch_cuda_sh"


def test_degree100_torch_available_picks_torch() -> None:
    d = _sel(requested_degree=100, numba_cuda_available=True, torch_cuda_available=True)
    assert d.backend == "torch_cuda_sh"


def test_degree100_torch_unavailable_falls_back_to_cpu() -> None:
    d = _sel(requested_degree=100, numba_cuda_available=True, torch_cuda_available=False)
    assert d.backend == "cpu_sh"
    assert d.fallback_applied is True
    assert d.fallback_reason  # must not be empty


def test_degree100_is_never_silently_clipped() -> None:
    for torch_avail in (True, False):
        d = _sel(requested_degree=100, numba_cuda_available=True, torch_cuda_available=torch_avail)
        # The requested degree is preserved verbatim regardless of routing.
        assert d.requested_degree == 100


def test_unsupported_physics_forces_explicit_fallback() -> None:
    # torch path can't honor the requested physics -> explicit CPU fallback.
    d = _sel(
        requested_backend="torch_cuda_sh",
        requested_degree=100,
        torch_cuda_available=True,
        torch_physics_ok=False,
    )
    assert d.backend == "cpu_sh"
    assert d.fallback_applied is True
    assert "physics" in d.fallback_reason.lower()


# --- explicit numba + high degree obeys fallback_policy --------------------

def test_explicit_numba_high_degree_error_policy_requires_error() -> None:
    d = _sel(
        requested_backend="numba_cuda_sh",
        requested_degree=100,
        numba_cuda_available=True,
        torch_cuda_available=True,
        fallback_policy="error",
    )
    assert d.requires_error is True
    assert d.fallback_reason


def test_explicit_numba_high_degree_compatible_gpu_uses_torch() -> None:
    d = _sel(
        requested_backend="numba_cuda_sh",
        requested_degree=100,
        numba_cuda_available=True,
        torch_cuda_available=True,
        fallback_policy="compatible_gpu",
    )
    assert d.backend == "torch_cuda_sh"
    assert d.fallback_applied is True


def test_explicit_numba_high_degree_cpu_policy_uses_cpu() -> None:
    d = _sel(
        requested_backend="numba_cuda_sh",
        requested_degree=100,
        numba_cuda_available=True,
        torch_cuda_available=True,
        fallback_policy="cpu",
    )
    assert d.backend == "cpu_sh"


def test_gpu_sh_alias_resolves_in_policy() -> None:
    # Legacy alias request behaves exactly like numba_cuda_sh.
    d = _sel(requested_backend="gpu_sh", requested_degree=20, numba_cuda_available=True)
    assert d.backend == "numba_cuda_sh"
