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
- GPU_CLASSIC_SH uses the existing Numba CUDA RK4 kernel (degree ≤ 24).
- CPU always uses the full-fidelity scipy DOP853 per-sample path.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from enum import Enum
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
    warnings: List[str] = field(default_factory=list)
    reason: str = ""
    integrator: str = "adaptive (DOP853)"
    batch_note: str = ""

    def log_summary(self) -> None:
        """Print a one-line backend decision summary suitable for the MC log."""
        print(
            f"[MC] Backend plan: {self.final_backend.value}  "
            f"gravity={self.gravity_backend}  "
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

    # --- Determine gravity mode -----------------------------------------------
    gravity_cfg = getattr(sim_cfg, "gravity", None)
    mission_st_lrps = bool(getattr(gravity_cfg, "uses_st_lrps", False))
    mc_override = str(getattr(mc_cfg, "gravity_mode_override", "follow_mission") or "follow_mission")
    is_st_lrps = mission_st_lrps or (mc_override == "st_lrps")
    gravity_label = "st_lrps" if is_st_lrps else "classic_sh"

    requested_gpu = bool(getattr(mc_cfg, "use_gpu", False))

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
                warnings=warns,
                reason="ST-LRPS GPU requested but PyTorch CUDA is unavailable",
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
                warnings=warns,
                reason=f"ST-LRPS GPU: unsupported physics on this path: {pretty}",
                integrator="adaptive (DOP853)",
            )

        return MCBackendPlan(
            final_backend=MCBackend.GPU_ST_LRPS,
            use_gpu=True,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            reason=f"ST-LRPS + PyTorch CUDA available. {_avail_str}. Selected MC backend: GPU-ST-LRPS.",
            integrator="fixed-step RK4",
            batch_note=(
                "Batch propagation: N trajectories simultaneously on CUDA tensor [N, 6]. "
                "ST-LRPS acceleration via batched autograd on CUDA device."
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
            warnings=warns,
            reason="Classic SH GPU requested but Numba CUDA is unavailable",
            integrator="adaptive (DOP853)",
        )

    # Numba CUDA available — check for CPU-only perturbations
    from core.mc_propagator import gpu_unsupported_features  # noqa: PLC0415

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
            warnings=warns,
            reason=f"Classic SH GPU: unsupported physics: {pretty}",
            integrator="adaptive (DOP853)",
        )

    return MCBackendPlan(
        final_backend=MCBackend.GPU_CLASSIC_SH,
        use_gpu=True,
        gravity_backend=gravity_label,
        torch_cuda_available=torch_cuda,
        numba_cuda_available=numba_cuda,
        reason=f"Classic SH + Numba CUDA available. {_avail_str}. Selected MC backend: GPU-classic-SH.",
        integrator="fixed-step RK4",
    )


__all__ = [
    "MCBackend",
    "MCBackendPlan",
    "resolve_mc_backend_policy",
    "_torch_cuda_available",
    "_numba_cuda_available",
    "_st_lrps_gpu_unsupported_features",
]
