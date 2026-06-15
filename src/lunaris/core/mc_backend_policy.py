# ST_LRPS/core/mc_backend_policy.py
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
- GPU_CLASSIC_SH uses the existing Numba CUDA RK4 kernel (``numba_cuda_sh``).
  Its degree-24 ceiling is a thread-local kernel-workspace limit, not a physical
  one. Requested degrees above 24 are never clipped.
- GPU_TORCH_SH uses the PyTorch fixed-step RK4 classic-SH path
  (``torch_cuda_sh``). It evaluates arbitrary degree (bounded by the loaded
  coefficient file, VRAM, batch size, dtype, and step) and is the live
  high-degree GPU route: ``degree > 24`` with PyTorch CUDA available selects
  ``torch_cuda_sh`` when the requested physics is supported. This first runtime
  form is gravity-only; any added perturbation forces an explicit, recorded
  fallback (to ``numba_cuda_sh`` when it can honor the physics, else CPU).
- CPU always uses the full-fidelity scipy DOP853 per-sample path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

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
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _numba_cuda_device_name() -> str | None:
    """Best-effort CUDA device name for provenance metadata."""

    try:
        from numba import cuda

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


def _gpu_sh_limits() -> tuple[int, tuple[int, ...]]:
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


@dataclass
class ClassicSHDecision:
    """Result of the degree-aware classic-SH backend selection (task §8).

    ``backend`` is the resolved backend name. ``requires_error`` is True when an
    explicit request cannot be honored and ``fallback_policy='error'`` was set —
    the caller must raise rather than silently substitute. The requested degree
    is preserved verbatim (never clipped).
    """

    backend: str
    requested_backend: str
    requested_degree: int
    fallback_applied: bool = False
    fallback_reason: str = ""
    requires_error: bool = False


def select_classic_sh_backend(
    *,
    requested_backend: str,
    requested_degree: int,
    numba_cuda_available: bool,
    torch_cuda_available: bool,
    numba_physics_ok: bool = True,
    torch_physics_ok: bool = True,
    fallback_policy: str = "compatible_gpu",
    numba_max_degree: int | None = None,
) -> ClassicSHDecision:
    """Pure, testable classic-SH backend selection policy (task §8).

    Implements the documented routing without ever silently clipping the degree
    or silently disabling a backend:

    * degree <= numba limit: numba_cuda_sh -> torch_cuda_sh -> cpu_sh;
    * degree  > numba limit: torch_cuda_sh -> cpu_sh (CPU is never the *only*
      option — torch is tried first);
    * explicit ``numba_cuda_sh`` with degree > limit obeys ``fallback_policy``
      (``error`` | ``compatible_gpu`` | ``cpu``).

    This is the single source of truth for the classic-SH backend choice.
    :func:`resolve_mc_backend_policy` consumes this decision directly (it does
    not re-derive the choice), and the live MC engine dispatches the resulting
    ``torch_cuda_sh`` selection to the PyTorch SH batch propagator
    (:mod:`lunaris.core.torch_sh_propagator`).
    """
    req = resolve_backend_alias_local(requested_backend)
    limit = int(numba_max_degree if numba_max_degree is not None else _gpu_sh_limits()[0])
    degree = int(requested_degree or 0)

    def _decide(backend: str, *, applied: bool = False, reason: str = "", err: bool = False) -> ClassicSHDecision:
        return ClassicSHDecision(
            backend=backend,
            requested_backend=requested_backend,
            requested_degree=degree,
            fallback_applied=applied,
            fallback_reason=reason,
            requires_error=err,
        )

    def _high_degree_gpu_or_cpu(reason: str) -> ClassicSHDecision:
        if torch_cuda_available and torch_physics_ok:
            return _decide("torch_cuda_sh", applied=True, reason=reason)
        return _decide("cpu_sh", applied=True, reason=reason)

    # --- Explicit CPU --------------------------------------------------------
    if req == "cpu_sh":
        return _decide("cpu_sh")

    # --- Explicit Torch CPU -----------------------------------------------
    if req == "torch_cpu_sh":
        try:
            import torch  # noqa: F401, F811  # availability probe only
            return _decide("torch_cpu_sh")
        except ImportError:
            return _decide("torch_cpu_sh", applied=False, reason="PyTorch not importable for torch_cpu_sh", err=True)

    # --- Explicit torch ------------------------------------------------------
    if req == "torch_cuda_sh":
        if not torch_cuda_available:
            policy = str(fallback_policy or "compatible_gpu").strip().lower()
            if policy == "error":
                return _decide("torch_cuda_sh", applied=False,
                               reason="PyTorch CUDA unavailable for torch_cuda_sh", err=True)
            if policy == "cpu":
                return _decide("torch_cpu_sh", applied=True,
                               reason="PyTorch CUDA unavailable for torch_cuda_sh; falling back to torch_cpu_sh")
            # compatible_gpu: no other compatible GPU exists, raise error
            return _decide("torch_cuda_sh", applied=False,
                           reason="PyTorch CUDA unavailable for torch_cuda_sh and no compatible GPU backend available",
                           err=True)
        if not torch_physics_ok:
            return _decide("cpu_sh", applied=True, reason="requested physics not supported by torch_cuda_sh")
        return _decide("torch_cuda_sh")

    # --- Explicit Numba ------------------------------------------------------
    if req == "numba_cuda_sh":
        if degree > limit:
            reason = f"numba_cuda_sh supports degree <= {limit}"
            policy = str(fallback_policy or "compatible_gpu").strip().lower()
            if policy == "error":
                return _decide("numba_cuda_sh", applied=False, reason=reason, err=True)
            if policy == "cpu":
                return _decide("cpu_sh", applied=True, reason=reason)
            return _high_degree_gpu_or_cpu(reason)  # compatible_gpu (default)
        if not numba_cuda_available:
            return _high_degree_gpu_or_cpu("Numba CUDA unavailable for numba_cuda_sh") \
                if (str(fallback_policy).lower() != "cpu") else _decide(
                    "cpu_sh", applied=True, reason="Numba CUDA unavailable for numba_cuda_sh")
        if not numba_physics_ok:
            return _decide("cpu_sh", applied=True, reason="requested physics not supported by numba_cuda_sh")
        return _decide("numba_cuda_sh")

    # --- auto ----------------------------------------------------------------
    if degree <= limit:
        if numba_cuda_available and numba_physics_ok:
            return _decide("numba_cuda_sh")
        if torch_cuda_available and torch_physics_ok:
            return _decide("torch_cuda_sh", applied=True, reason="numba_cuda_sh unavailable; using torch_cuda_sh")
        return _decide(
            "cpu_sh",
            applied=True,
            reason="no compatible GPU classic-SH backend available (Numba CUDA and PyTorch CUDA unavailable)",
        )
    # degree > numba limit: prefer torch over CPU
    if torch_cuda_available and torch_physics_ok:
        return _decide("torch_cuda_sh")
    return _decide(
        "cpu_sh",
        applied=True,
        reason=f"degree {degree} exceeds numba_cuda_sh limit {limit} and torch_cuda_sh unavailable",
    )


def resolve_backend_alias_local(name: str) -> str:
    """Resolve a backend alias (``gpu_sh`` -> ``numba_cuda_sh``) via the registry."""
    try:
        from lunaris.core.backend_capabilities import resolve_backend_alias

        return resolve_backend_alias(name)
    except Exception:
        return str(name).strip()


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
        Maps to the ``numba_cuda_sh`` backend.  Supports third-body Sun/Earth,
        Earth J2, SRP, and 1PN relativity on GPU.

    ``GPU_TORCH_SH``
        PyTorch (CUDA or CPU) fixed-step RK4 evaluating classic spherical
        harmonics at arbitrary degree via the canonical
        :class:`~lunaris.physics.torch_spherical_harmonics.TorchSHGravityEvaluator`.
        Maps to the ``torch_cuda_sh`` backend.  Distinct runtime implementation
        from ``GPU_CLASSIC_SH`` — the two must never share an enum value.  First
        runtime form is gravity-only; any added perturbation forces an explicit,
        recorded fallback.

    ``CPU``
        Sequential full-fidelity per-sample scipy DOP853.  All physics flags
        supported.
    """

    CPU = "cpu"
    GPU_CLASSIC_SH = "gpu_classic_sh"
    GPU_TORCH_SH = "gpu_torch_sh"
    TORCH_CPU_SH = "torch_cpu_sh"
    GPU_ST_LRPS = "gpu_st_lrps"


MC_BACKEND_REQUESTS = frozenset(
    {
        "auto",
        "cpu_sh",
        "numba_cuda_sh",
        "torch_cuda_sh",
        "torch_cpu_sh",
        "gpu_sh",  # legacy alias -> numba_cuda_sh
        "gpu_st_lrps_potential",
        "gpu_st_lrps_direct",
    }
)

# Classic-SH request names that select the Numba CUDA backend (degree <= 24).
_NUMBA_SH_REQUESTS = frozenset({"gpu_sh", "numba_cuda_sh"})
# Classic-SH request names that select the PyTorch SH backend (arbitrary degree).
_TORCH_SH_REQUESTS = frozenset({"torch_cuda_sh", "torch_cpu_sh"})


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
    backend_family: str = ""
    backend_implementation: str = ""
    requested_sh_degree: int = 0
    actual_sh_degree: int | None = None
    gpu_sh_max_degree: int = 24
    gpu_sh_supported_tiers: tuple[int, ...] = (24,)
    runtime_model_kind: str | None = None
    cuda_device_name: str | None = None
    requested_device: str = ""
    actual_device: str = ""
    dtype: str = "float64"
    warnings: list[str] = field(default_factory=list)
    reason: str = ""
    fallback_applied: bool = False
    fallback_reason: str = ""
    integrator: str = "adaptive (DOP853)"
    batch_note: str = ""

    def __post_init__(self) -> None:
        # Auto-populate provenance family/implementation from the central
        # capability registry based on the resolved backend, unless explicitly
        # set. Keeps every return site labeled without per-site duplication.
        if not self.backend_family or not self.backend_implementation:
            try:
                from lunaris.core.backend_capabilities import get_capabilities

                caps = get_capabilities(self.actual_backend)
                if not self.backend_family:
                    self.backend_family = caps.family
                if not self.backend_implementation:
                    self.backend_implementation = caps.implementation
            except Exception:
                pass

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


def _st_lrps_gpu_unsupported_features(flags: Any) -> tuple[str, ...]:
    """
    Return physics flags that are active but unsupported on the GPU ST-LRPS path.

    The torch RK4 propagator currently handles gravity only (point-mass +
    neural residual).  Any additional perturbation forces a CPU fallback.
    """

    if flags is None:
        return ()

    unsupported: list[str] = []
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

    warns: list[str] = []

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

    if requested_backend in {"cpu_sh"} | _NUMBA_SH_REQUESTS | _TORCH_SH_REQUESTS:
        is_st_lrps = False
    elif requested_backend in {"gpu_st_lrps_potential", "gpu_st_lrps_direct"}:
        is_st_lrps = True
    else:
        is_st_lrps = mission_st_lrps or (mc_override == "st_lrps")
    gravity_label = "st_lrps" if is_st_lrps else "classic_sh"

    requested_gpu = bool(getattr(mc_cfg, "use_gpu", False))
    if requested_backend == "cpu_sh":
        requested_gpu = False
    elif requested_backend in (
        _NUMBA_SH_REQUESTS | _TORCH_SH_REQUESTS | {"gpu_st_lrps_potential", "gpu_st_lrps_direct"}
    ):
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
                requested_device="cuda",
                actual_device="cpu",
                dtype="float64",
                warnings=warns,
                fallback_applied=True,
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
                requested_device="cuda",
                actual_device="cpu",
                dtype="float64",
                warnings=warns,
                fallback_applied=True,
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
    # Classic SH path — single source of truth is select_classic_sh_backend()
    # =========================================================================
    # Every classic-SH request (auto / cpu_sh / gpu_sh / numba_cuda_sh /
    # torch_cuda_sh) is routed through ONE decision function; the backend choice
    # is never re-derived in a second if/elif chain here. This is the piece that
    # finally wires torch_cuda_sh into the live MC runtime: degree > 24 with
    # PyTorch CUDA available now selects torch_cuda_sh instead of dropping to CPU.
    from lunaris.core.backend_capabilities import (  # noqa: PLC0415
        get_capabilities,
        resolve_backend_alias,
        unsupported_force_models,
    )

    sh_enabled = bool(getattr(flags, "enable_sh", True)) if flags is not None else True
    # When SH is disabled the run is point-mass; degree is irrelevant and must
    # not push an otherwise-fine run off the degree-24 Numba screening kernel.
    effective_degree = int(requested_sh_degree) if sh_enabled else 0

    # Gravity-only capability checks from the central registry. The registry is
    # locked to the legacy Numba force-model matrix by
    # tests/test_backend_capabilities.py, so this stays consistent with the
    # Numba kernel without importing the CUDA stack here.
    numba_unsupported = unsupported_force_models("numba_cuda_sh", flags) if flags is not None else ()
    torch_unsupported = unsupported_force_models("torch_cuda_sh", flags) if flags is not None else ()
    fallback_policy = str(getattr(mc_cfg, "gpu_sh_fallback_policy", "compatible_gpu") or "compatible_gpu")
    torch_dtype = str(getattr(mc_cfg, "torch_dtype", "float64") or "float64").lower()
    if torch_dtype not in ("float32", "float64"):
        torch_dtype = "float64"

    decision = select_classic_sh_backend(
        requested_backend=requested_backend,
        requested_degree=effective_degree,
        numba_cuda_available=numba_cuda,
        torch_cuda_available=torch_cuda,
        numba_physics_ok=(not numba_unsupported),
        torch_physics_ok=(not torch_unsupported),
        fallback_policy=fallback_policy,
        numba_max_degree=gpu_sh_max_degree,
    )

    # Explicit request the policy cannot honor under fallback_policy='error'.
    if decision.requires_error:
        raise RuntimeError(
            f"[MC] mc_backend={requested_backend!r} cannot be honored "
            f"({decision.fallback_reason}); "
            "the selected fallback policy forbids substituting another backend. "
            "Check CUDA availability, lower the degree, or set gpu_sh_fallback_policy "
            "to 'compatible_gpu' or 'cpu'."
        )

    requested_device = str(get_capabilities(resolve_backend_alias(requested_backend)).device)

    # ----- numba_cuda_sh (degree <= 24 Numba CUDA screening kernel) -----------
    if decision.backend == "numba_cuda_sh":
        return MCBackendPlan(
            final_backend=MCBackend.GPU_CLASSIC_SH,
            use_gpu=True,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            requested_backend=requested_backend,
            actual_backend="numba_cuda_sh",
            requested_sh_degree=requested_sh_degree,
            actual_sh_degree=effective_degree,
            gpu_sh_max_degree=gpu_sh_max_degree,
            gpu_sh_supported_tiers=gpu_sh_tiers,
            cuda_device_name=_numba_cuda_device_name(),
            requested_device=requested_device,
            actual_device=_numba_cuda_device_name() or "cuda",
            dtype="float64",
            reason=f"Classic SH + Numba CUDA available. {_avail_str}. Selected MC backend: numba_cuda_sh.",
            integrator="fixed-step RK4",
        )

    # ----- torch_cuda_sh (arbitrary-degree PyTorch classic SH, gravity-only) ---
    if decision.backend == "torch_cuda_sh":
        if decision.fallback_applied and decision.fallback_reason:
            warns.append(
                f"[MC] {decision.fallback_reason}. Selected MC backend: torch_cuda_sh "
                "(PyTorch high-degree classic SH, gravity-only)."
            )
        return MCBackendPlan(
            final_backend=MCBackend.GPU_TORCH_SH,
            use_gpu=True,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            requested_backend=requested_backend,
            actual_backend="torch_cuda_sh",
            requested_sh_degree=requested_sh_degree,
            actual_sh_degree=effective_degree,
            gpu_sh_max_degree=gpu_sh_max_degree,
            gpu_sh_supported_tiers=gpu_sh_tiers,
            cuda_device_name=_torch_cuda_device_name(),
            requested_device=requested_device,
            actual_device=_torch_cuda_device_name() or "cuda",
            dtype=torch_dtype,
            warnings=warns,
            fallback_applied=bool(decision.fallback_applied),
            fallback_reason=decision.fallback_reason,
            reason=(
                f"Classic SH degree {effective_degree} on PyTorch CUDA. {_avail_str}. "
                "Selected MC backend: torch_cuda_sh."
            ),
            integrator="fixed-step RK4",
            batch_note=(
                "Batch propagation: N trajectories as a single torch tensor [N, 6]; "
                "per-RK-stage Moon inertial<->fixed frame transform; fixed-step RK4."
            ),
        )

    # ----- torch_cpu_sh (PyTorch SH on CPU, fixed-step RK4) -----------------
    if decision.backend == "torch_cpu_sh":
        if decision.requires_error:
            raise RuntimeError(
                f"[MC] mc_backend='torch_cpu_sh' cannot be used: {decision.fallback_reason}. "
                "Install PyTorch to use the torch_cpu_sh backend."
            )
        return MCBackendPlan(
            final_backend=MCBackend.TORCH_CPU_SH,
            use_gpu=False,
            gravity_backend=gravity_label,
            torch_cuda_available=torch_cuda,
            numba_cuda_available=numba_cuda,
            requested_backend=requested_backend,
            actual_backend="torch_cpu_sh",
            requested_sh_degree=requested_sh_degree,
            actual_sh_degree=effective_degree,
            gpu_sh_max_degree=gpu_sh_max_degree,
            gpu_sh_supported_tiers=gpu_sh_tiers,
            requested_device=requested_device,
            actual_device="cpu",
            dtype=torch_dtype,
            warnings=warns,
            fallback_applied=bool(decision.fallback_applied),
            fallback_reason=decision.fallback_reason,
            reason=(
                f"Classic SH degree {effective_degree} on PyTorch CPU. "
                "Selected MC backend: torch_cpu_sh."
            ),
            integrator="fixed-step RK4",
            batch_note=(
                "Batch propagation: N trajectories as a single torch tensor [N, 6] on CPU; "
                "per-RK-stage Moon inertial<->fixed frame transform; fixed-step RK4."
            ),
        )

    # ----- cpu_sh (explicit, recorded fallback off the classic-SH GPU path) ----
    # Build an honest, specific warning naming the actual cause.
    blocking: tuple[str, ...] = ()
    if numba_cuda and numba_unsupported:
        blocking = numba_unsupported
    elif torch_cuda and torch_unsupported:
        blocking = torch_unsupported

    high_degree_numba = bool(
        resolve_backend_alias(requested_backend) == "numba_cuda_sh"
        and sh_enabled
        and requested_sh_degree > gpu_sh_max_degree
    )

    if blocking:
        pretty = ", ".join(blocking)
        fallback_reason = f"requested physics not supported by GPU classic-SH path: {pretty}"
        msg = (
            f"[MC] GPU classic-SH backend does not model: {pretty}. "
            "Falling back to CPU. Selected MC backend: CPU."
        )
    elif high_degree_numba:
        fallback_reason = decision.fallback_reason
        msg = (
            f"[MC] Requested classic-SH degree {requested_sh_degree} exceeds the "
            f"numba_cuda_sh kernel limit (degree <= {gpu_sh_max_degree}; a "
            "thread-local workspace limit, not a physical one) and torch_cuda_sh "
            "is unavailable. Falling back to CPU without clipping the requested "
            "degree. Selected MC backend: CPU."
        )
    else:
        fallback_reason = decision.fallback_reason or "no compatible GPU classic-SH backend available"
        msg = f"[MC] {fallback_reason}. Falling back to CPU. Selected MC backend: CPU."
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
        requested_device=requested_device,
        actual_device="cpu",
        dtype="float64",
        warnings=warns,
        fallback_applied=True,
        fallback_reason=fallback_reason,
        reason=msg,
        integrator="adaptive (DOP853)",
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
