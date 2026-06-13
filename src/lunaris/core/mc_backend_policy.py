# ST_LRPS/core/mc_backend_policy.py
# -*- coding: utf-8 -*-
"""
Monte Carlo Backend Capability Matrix and Policy Resolver
=========================================================

Single source of truth for deciding which propagator backend is used for a
Monte Carlo run.  All GPU/CPU routing that was previously scattered across
``MonteCarloEngine._build_propagator()`` is consolidated here so the decision
is testable in isolation.

Capability matrix
-----------------
+--------------------+----------+---------+---------------------+
| Gravity backend    | Numba    | PyTorch | Selected backend    |
|                    | CUDA     | CUDA    |                     |
+====================+==========+=========+=====================+
| Classic SH         | yes      | —       | GPU_CLASSIC_SH      |
| Classic SH         | no       | —       | CPU                 |
| ST-LRPS            | —        | yes     | GPU_ST_LRPS         |
| ST-LRPS            | —        | no      | CPU                 |
+--------------------+----------+---------+---------------------+

Notes
-----
- GPU_ST_LRPS uses PyTorch fixed-step RK4 with the surrogate model on CUDA.
  It currently supports gravity only (no third-body/SRP/relativity on this path).
  Those perturbations force a CPU fallback.
- GPU_CLASSIC_SH uses the existing Numba CUDA RK4 kernel.  The current true
  GPU classic-SH evaluator supports degree <= 24; higher requested degrees are
  routed through an explicit CPU fallback instead of being clipped.
- CPU always uses the full-fidelity scipy DOP853 per-sample path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple


# =============================================================================
# 1.                      CUDA AVAILABILITY PROBES
# =============================================================================


def _torch_cuda_available() -> bool:
    """Return True when PyTorch can use at least one CUDA device."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _numba_cuda_available() -> bool:
    """Return True when Numba CUDA is installed and sees a device."""
    try:
        from numba import cuda  # type: ignore

        return bool(cuda.is_available())
    except Exception:
        return False


def _numba_cuda_device_name() -> str | None:
    """Best-effort CUDA device name for provenance metadata."""

    try:
        from numba import cuda  # type: ignore

        if not cuda.is_available():
            return None
        dev = cuda.get_current_device()
        name = getattr(dev, "name", None)
        if isinstance(name, bytes):
            name = name.decode(errors="ignore")
        return str(name).strip() if name else None
    except Exception:
        return None


def _torch_cuda_device_name() -> str | None:
    """Best-effort PyTorch CUDA device name for provenance metadata."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None


def _gpu_sh_limits() -> tuple[int, Tuple[int, ...]]:
    """Return current true-GPU classic-SH max degree and supported tiers.

    Sourced from the central backend capability registry
    (:mod:`lunaris.core.backend_capabilities`) so the MC engine, the benchmark
    runner, the UI, and provenance writers all agree on the same limit. The
    registry in turn reads the real Numba CUDA workspace constant.
    """

    try:
        from lunaris.core.backend_capabilities import (
            gpu_sh_max_degree,
            gpu_sh_supported_tiers,
        )

        return gpu_sh_max_degree(), gpu_sh_supported_tiers()
    except Exception:
        return 24, (24,)


def _clean_requested_backend(value: Any) -> str:
    backend = str(value or "auto").strip().lower()
    if backend not in MC_BACKEND_REQUESTS:
        raise ValueError(
            "mc_backend must be one of: "
            + ", ".join(sorted(MC_BACKEND_REQUESTS))
            + f"; got {backend!r}."
        )
    return backend


def _read_st_lrps_runtime_kind(mc_cfg: Any, sim_cfg: Any) -> str | None:
    """Read runtime_model_kind from an ST-LRPS config.json when cheaply available."""

    candidates: list[Any] = [
        getattr(mc_cfg, "st_lrps_model_dir", None),
        getattr(getattr(sim_cfg, "gravity", None), "st_lrps_model_dir", None),
    ]
    for raw in candidates:
        if not raw:
            continue
        try:
            cfg_path = Path(raw).expanduser().resolve() / "config.json"
            if cfg_path.exists():
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                kind = str(payload.get("runtime_model_kind", "") or "").strip()
                if kind:
                    return kind
        except Exception:
            continue
    return None


# =============================================================================
# 2.                      BACKEND ENUM + PLAN
# =============================================================================


class MCBackend(str, Enum):
    """
    Available Monte Carlo propagator backends.

    ``GPU_ST_LRPS``
        PyTorch CUDA fixed-step RK4.  All N trajectories are kept as a single
        ``[N, 6]`` CUDA float32 tensor.  The ST-LRPS neural surrogate is
        evaluated as a batched PyTorch forward pass + ``autograd.grad`` for the
        acceleration.  Gravity only (no third-body / SRP / relativity).

    ``GPU_CLASSIC_SH``
        Numba CUDA fixed-step RK4 with per-thread SH workspace (degree ≤ 24).
        Supports third-body Sun/Earth, SRP, and 1PN relativity on GPU.

    ``CPU``
        Sequential full-fidelity per-sample scipy DOP853.  All physics flags
        supported.
    """

    CPU = "cpu"
    GPU_CLASSIC_SH = "gpu_classic_sh"
    GPU_ST_LRPS = "gpu_st_lrps"


MC_BACKEND_REQUESTS = frozenset(
    {
        "auto",
        "cpu_sh",
        "gpu_sh",
        "gpu_st_lrps_potential",
        "gpu_st_lrps_direct",
    }
)


@dataclass
class MCBackendPlan:
    """
    Fully resolved backend decision including availability diagnostics.

    Consumers should treat this as read-only after construction.
    """

    final_backend: MCBackend
    use_gpu: bool
    gravity_backend: str
    torch_cuda_available: bool
    numba_cuda_available: bool
    requested_backend: str = "auto"
    actual_backend: str = "cpu_sh"
    requested_sh_degree: int = 0
    actual_sh_degree: int | None = None
    gpu_sh_max_degree: int = 24
    gpu_sh_supported_tiers: Tuple[int, ...] = (24,)
    runtime_model_kind: str | None = None
    cuda_device_name: str | None = None
    dtype: str = "float64"
    warnings: List[str] = field(default_factory=list)
    reason: str = ""
    fallback_reason: str = ""
    integrator: str = "adaptive (DOP853)"
    batch_note: str = ""

    def log_summary(self) -> None:
        """Print a one-line backend decision summary suitable for the MC log."""
        print(
            f"[MC] Backend plan: {self.final_backend.value}  "
            f"gravity={self.gravity_backend}  "
            f"requested_backend={self.requested_backend}  "
            f"actual_backend={self.actual_backend}  "
            f"torch_cuda={self.torch_cuda_available}  "
            f"numba_cuda={self.numba_cuda_available}  "
            f"integrator={self.integrator}",
            flush=True,
        )
        if self.batch_note:
            print(f"[MC] {self.batch_note}", flush=True)


# =============================================================================
# 3.                   CPU-ONLY PHYSICS CHECKS
# =============================================================================


def _st_lrps_gpu_unsupported_features(flags: Any) -> Tuple[str, ...]:
    """
    Return physics flags that are active but unsupported on the GPU ST-LRPS path.

    The torch RK4 propagator currently handles gravity only (point-mass +
    neural residual).  Any additional perturbation forces a CPU fallback.
    """

    if flags is None:
        return ()

    unsupported: List[str] = []
    if bool(getattr(flags, "enable_3rd_body_sun", False)):
        unsupported.append("third-body Sun")
    if bool(getattr(flags, "enable_3rd_body_earth", False)):
        unsupported.append("third-body Earth")
    if bool(getattr(flags, "enable_earth_j2", False)):
        unsupported.append("Earth J2")
    if bool(getattr(flags, "enable_srp", False)):
        unsupported.append("SRP")
    if bool(getattr(flags, "enable_albedo", False)):
        unsupported.append("albedo")
    if bool(getattr(flags, "enable_thermal", False)):
        unsupported.append("thermal IR")
    if bool(getattr(flags, "enable_tides_k2", False)):
        unsupported.append("solid tides k2")
    if bool(getattr(flags, "enable_tides_k3", False)):
        unsupported.append("solid tides k3")
    if bool(getattr(flags, "enable_relativity_1pn", False)):
        unsupported.append("1PN relativity")
    return tuple(unsupported)


# =============================================================================
# 4.                   MAIN POLICY RESOLVER
# =============================================================================


def resolve_mc_backend_policy(
    mc_cfg: Any,
    sim_cfg: Any,
) -> MCBackendPlan:
    """
    Resolve the best available Monte Carlo backend given config and hardware.

    Parameters
    ----------
    mc_cfg : MonteCarloConfig
        Requested Monte Carlo settings (``use_gpu``, ``gravity_mode_override``, …).
    sim_cfg : SimConfig
        Full simulation configuration used to read ``gravity.uses_st_lrps`` and
        active perturbation flags.

    Returns
    -------
    MCBackendPlan
        Fully resolved plan.  ``plan.warnings`` contains human-readable
        fallback reasons; callers should emit these as ``RuntimeWarning``.
    """

    warns: List[str] = []

    # --- Hardware probes ------------------------------------------------------
    torch_cuda = _torch_cuda_available()
    numba_cuda = _numba_cuda_available()
    gpu_sh_max_degree, gpu_sh_tiers = _gpu_sh_limits()

    # --- Determine gravity mode -----------------------------------------------
    gravity_cfg = getattr(sim_cfg, "gravity", None)
    mission_st_lrps = bool(getattr(gravity_cfg, "uses_st_lrps", False))
    mc_override = str(getattr(mc_cfg, "gravity_mode_override", "follow_mission") or "follow_mission")
    requested_backend = _clean_requested_backend(getattr(mc_cfg, "mc_backend", "auto"))
    requested_sh_degree = int(getattr(mc_cfg, "gpu_sh_degree", 0) or 0)

    if requested_backend in {"cpu_sh", "gpu_sh"}:
        is_st_lrps = False
    elif requested_backend in {"gpu_st_lrps_potential", "gpu_st_lrps_direct"}:
        is_st_lrps = True
    else:
        is_st_lrps = mission_st_lrps or (mc_override == "st_lrps")
    gravity_label = "st_lrps" if is_st_lrps else "classic_sh"

    requested_gpu = bool(getattr(mc_cfg, "use_gpu", False))
    if requested_backend == "cpu_sh":
        requested_gpu = False
    elif requested_backend in {"gpu_sh", "gpu_st_lrps_potential", "gpu_st_lrps_direct"}:
        requested_gpu = True

    runtime_model_kind = _read_st_lrps_runtime_kind(mc_cfg, sim_cfg) if is_st_lrps else None
    if requested_backend == "gpu_st_lrps_potential":
        runtime_model_kind = runtime_model_kind or "potential_autograd"
    elif requested_backend == "gpu_st_lrps_direct":
        runtime_model_kind = runtime_model_kind or "force_direct"

    # Log availability (always useful for diagnostics)
    _avail_str = (
        f"PyTorch CUDA available: {'yes' if torch_cuda else 'no'}  "
        f"Numba CUDA available: {'yes' if numba_cuda else 'no'}"
    )

    # --- CPU-only request -------------------------------------------------
    if not requested_gpu:
        return MCBackendPlan(
            final_backend=MCBackend.CPU,
            use_gpu=False,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            requested_backend=requested_backend,
            actual_backend="cpu_sh" if not is_st_lrps else "cpu_st_lrps",
            requested_sh_degree=requested_sh_degree,
            actual_sh_degree=None,
            gpu_sh_max_degree=gpu_sh_max_degree,
            gpu_sh_supported_tiers=gpu_sh_tiers,
            runtime_model_kind=runtime_model_kind,
            dtype="float64",
            reason="CPU backend explicitly requested",
            integrator="adaptive (DOP853)",
        )

    flags = getattr(sim_cfg, "flags", None)

    # =========================================================================
    # ST-LRPS path
    # =========================================================================
    if is_st_lrps:
        if not torch_cuda:
            msg = (
                f"[MC] use_gpu=True with ST-LRPS gravity, but PyTorch CUDA is unavailable. "
                f"{_avail_str}. "
                "Falling back to the CPU full-fidelity backend. "
                "Selected MC backend: CPU."
            )
            warns.append(msg)
            return MCBackendPlan(
                final_backend=MCBackend.CPU,
                use_gpu=False,
                gravity_backend=gravity_label,
                torch_cuda_available=torch_cuda,
                numba_cuda_available=numba_cuda,
                requested_backend=requested_backend,
                actual_backend="cpu_st_lrps",
                requested_sh_degree=requested_sh_degree,
                actual_sh_degree=None,
                gpu_sh_max_degree=gpu_sh_max_degree,
                gpu_sh_supported_tiers=gpu_sh_tiers,
                runtime_model_kind=runtime_model_kind,
                dtype="float64",
                warnings=warns,
                reason="ST-LRPS GPU requested but PyTorch CUDA is unavailable",
                fallback_reason="ST-LRPS GPU requested but PyTorch CUDA is unavailable",
                integrator="adaptive (DOP853)",
            )

        # PyTorch CUDA is available — check for incompatible perturbations
        gpu_st_lrps_unsupported = _st_lrps_gpu_unsupported_features(flags)
        if gpu_st_lrps_unsupported:
            pretty = ", ".join(gpu_st_lrps_unsupported)
            msg = (
                f"[MC] GPU ST-LRPS batch propagator does not currently model: {pretty}. "
                "Falling back to the CPU full-fidelity backend. "
                "Selected MC backend: CPU."
            )
            warns.append(msg)
            return MCBackendPlan(
                final_backend=MCBackend.CPU,
                use_gpu=False,
                gravity_backend=gravity_label,
                torch_cuda_available=torch_cuda,
                numba_cuda_available=numba_cuda,
                requested_backend=requested_backend,
                actual_backend="cpu_st_lrps",
                requested_sh_degree=requested_sh_degree,
                actual_sh_degree=None,
                gpu_sh_max_degree=gpu_sh_max_degree,
                gpu_sh_supported_tiers=gpu_sh_tiers,
                runtime_model_kind=runtime_model_kind,
                dtype="float64",
                warnings=warns,
                reason=f"ST-LRPS GPU: unsupported physics on this path: {pretty}",
                fallback_reason=f"ST-LRPS GPU unsupported physics: {pretty}",
                integrator="adaptive (DOP853)",
            )

        actual_stlrps_backend = (
            "gpu_st_lrps_direct"
            if str(runtime_model_kind or "").strip() == "force_direct"
            else "gpu_st_lrps_potential"
        )
        stlrps_note = (
            "ST-LRPS direct residual acceleration via batched no-grad CUDA forward pass."
            if actual_stlrps_backend == "gpu_st_lrps_direct"
            else "ST-LRPS acceleration via batched autograd on CUDA device."
        )
        return MCBackendPlan(
            final_backend=MCBackend.GPU_ST_LRPS,
            use_gpu=True,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            requested_backend=requested_backend,
            actual_backend=actual_stlrps_backend,
            requested_sh_degree=requested_sh_degree,
            actual_sh_degree=None,
            gpu_sh_max_degree=gpu_sh_max_degree,
            gpu_sh_supported_tiers=gpu_sh_tiers,
            runtime_model_kind=runtime_model_kind or (
                "force_direct" if actual_stlrps_backend == "gpu_st_lrps_direct" else "potential_autograd"
            ),
            cuda_device_name=_torch_cuda_device_name(),
            dtype="float32",
            reason=f"ST-LRPS + PyTorch CUDA available. {_avail_str}. Selected MC backend: GPU-ST-LRPS.",
            integrator="fixed-step RK4",
            batch_note=(
                "Batch propagation: N trajectories simultaneously on CUDA tensor [N, 6]. "
                f"{stlrps_note}"
            ),
        )

    # =========================================================================
    # Classic SH path
    # =========================================================================
    if not numba_cuda:
        msg = (
            f"[MC] use_gpu=True but Numba CUDA is unavailable. "
            f"{_avail_str}. "
            "Falling back to CPU. Selected MC backend: CPU."
        )
        warns.append(msg)
        return MCBackendPlan(
            final_backend=MCBackend.CPU,
            use_gpu=False,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            requested_backend=requested_backend,
            actual_backend="cpu_sh",
            requested_sh_degree=requested_sh_degree,
            actual_sh_degree=None,
            gpu_sh_max_degree=gpu_sh_max_degree,
            gpu_sh_supported_tiers=gpu_sh_tiers,
            dtype="float64",
            warnings=warns,
            reason="Classic SH GPU requested but Numba CUDA is unavailable",
            fallback_reason="Classic SH GPU requested but Numba CUDA is unavailable",
            integrator="adaptive (DOP853)",
        )

    # Numba CUDA available — check for CPU-only perturbations
    sh_enabled = bool(getattr(flags, "enable_sh", True)) if flags is not None else True
    if sh_enabled and requested_sh_degree > gpu_sh_max_degree:
        msg = (
            f"[MC] Requested gpu_sh_degree={requested_sh_degree}, but the current "
            f"Numba CUDA classic-SH kernel supports true GPU SH only through "
            f"degree {gpu_sh_max_degree}. Falling back to CPU without clipping "
            "the requested degree. Selected MC backend: CPU."
        )
        warns.append(msg)
        return MCBackendPlan(
            final_backend=MCBackend.CPU,
            use_gpu=False,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            requested_backend=requested_backend,
            actual_backend="cpu_sh",
            requested_sh_degree=requested_sh_degree,
            actual_sh_degree=None,
            gpu_sh_max_degree=gpu_sh_max_degree,
            gpu_sh_supported_tiers=gpu_sh_tiers,
            dtype="float64",
            warnings=warns,
            reason=f"Classic SH GPU degree {requested_sh_degree} exceeds true GPU max {gpu_sh_max_degree}",
            fallback_reason=f"gpu_sh_degree>{gpu_sh_max_degree}",
            integrator="adaptive (DOP853)",
        )

    from lunaris.core.mc_propagator import gpu_unsupported_features  # noqa: PLC0415

    classic_unsupported = gpu_unsupported_features(flags) if flags is not None else ()
    if classic_unsupported:
        pretty = ", ".join(classic_unsupported)
        msg = (
            f"[MC] GPU classic-SH backend does not model: {pretty}. "
            "Falling back to CPU. Selected MC backend: CPU."
        )
        warns.append(msg)
        return MCBackendPlan(
            final_backend=MCBackend.CPU,
            use_gpu=False,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            requested_backend=requested_backend,
            actual_backend="cpu_sh",
            requested_sh_degree=requested_sh_degree,
            actual_sh_degree=None,
            gpu_sh_max_degree=gpu_sh_max_degree,
            gpu_sh_supported_tiers=gpu_sh_tiers,
            dtype="float64",
            warnings=warns,
            reason=f"Classic SH GPU: unsupported physics: {pretty}",
            fallback_reason=f"Classic SH GPU unsupported physics: {pretty}",
            integrator="adaptive (DOP853)",
        )

    return MCBackendPlan(
        final_backend=MCBackend.GPU_CLASSIC_SH,
        use_gpu=True,
        gravity_backend=gravity_label,
        torch_cuda_available=torch_cuda,
        numba_cuda_available=numba_cuda,
        requested_backend=requested_backend,
        actual_backend="gpu_sh",
        requested_sh_degree=requested_sh_degree,
        actual_sh_degree=requested_sh_degree,
        gpu_sh_max_degree=gpu_sh_max_degree,
        gpu_sh_supported_tiers=gpu_sh_tiers,
        cuda_device_name=_numba_cuda_device_name(),
        dtype="float64",
        reason=f"Classic SH + Numba CUDA available. {_avail_str}. Selected MC backend: GPU-classic-SH.",
        integrator="fixed-step RK4",
    )


__all__ = [
    "MCBackend",
    "MCBackendPlan",
    "resolve_mc_backend_policy",
    "MC_BACKEND_REQUESTS",
    "_torch_cuda_available",
    "_numba_cuda_available",
    "_numba_cuda_device_name",
    "_torch_cuda_device_name",
    "_st_lrps_gpu_unsupported_features",
]
