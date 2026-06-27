"""Compatibility adapter over the existing core Monte Carlo backend policy."""

from __future__ import annotations

from lunaris.core.mc_backend_policy import (
    MC_BACKEND_REQUESTS,
    ClassicSHDecision,
    MCBackend,
    MCBackendPlan,
    _numba_cuda_available,
    _numba_cuda_device_name,
    _st_lrps_gpu_unsupported_features,
    _torch_cuda_available,
    _torch_cuda_device_name,
    resolve_mc_backend_policy,
    select_classic_sh_backend,
)

__all__ = [
    "MCBackend",
    "MCBackendPlan",
    "resolve_mc_backend_policy",
    "select_classic_sh_backend",
    "ClassicSHDecision",
    "MC_BACKEND_REQUESTS",
    "_torch_cuda_available",
    "_numba_cuda_available",
    "_numba_cuda_device_name",
    "_torch_cuda_device_name",
    "_st_lrps_gpu_unsupported_features",
]
