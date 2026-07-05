# lunaris.core.torch_sh_propagator
"""
Torch Classic-SH Batch Propagator (``torch_cuda_sh`` / ``torch_cpu_sh`` runtime)
=========================================================================

This is the live runtime behind the ``torch_cuda_sh`` and ``torch_cpu_sh`` backends.  It propagates
``N`` ensemble samples simultaneously as a single ``[N, 6]`` PyTorch tensor
using a fixed-step RK4 integrator whose gravity is the canonical batched
spherical-harmonic evaluator
:class:`lunaris.physics.torch_spherical_harmonics.TorchSHGravityEvaluator`.

Why this module exists
----------------------
The Numba CUDA classic-SH kernel (``numba_cuda_sh``) is capped at degree 24 by
its compile-time thread-local Legendre workspace.  That ceiling is a kernel
limitation, **not** a physical one.  This module provides the high-degree GPU
path: arbitrary SH degree bounded only by the loaded coefficient file, GPU
memory, batch size, dtype, and step size.

Contract (matches :class:`lunaris.core.batch_propagator.GPUBatchPropagator`)
------------------------------------------------------------------------
``propagate(Y0, masses, areas, cds, crs, duration_s, output_dt_s, callback)``
returns ``(t_out, Y_out, impact_flags, t_impact)``.  Spacecraft properties are
accepted for API parity but ignored: this first runtime form is **gravity-only**
(lunar SH + Moon inertial<->fixed frame transform).  Any active perturbation is
a hard contract violation here — :func:`resolve_batch_backend_policy` is responsible
for routing physics-incompatible runs elsewhere; if one reaches this module it
raises :class:`TorchSHPreflightError` rather than silently dropping physics.

Frame handling (task §7)
------------------------
At every RK4 stage the inertial position is rotated into the Moon-fixed frame at
that stage's epoch (SLERP-interpolated ``q_i2f`` quaternion), the SH acceleration
is evaluated in the fixed frame, then rotated back to inertial with the conjugate
quaternion.  The quaternion convention (scalar-first ``q_i2f``) matches the Numba
CUDA kernel and the ST-LRPS benchmark, so the three paths agree.

Device / dtype
--------------
Runs on CUDA or CPU; float32 or float64.  CPU + float64 is the validation path
(deterministic, no GPU required); CUDA + float64 is the default production path.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from lunaris.common.batch_defs import build_batch_output_grid
from lunaris.common.constants import R_MOON
from lunaris.core.backend_capabilities import unsupported_force_models
from lunaris.core.batched_fixed_step import (
    rhs_batch,
    rk4_step,
    run_batched_fixed_step,
)
from lunaris.core.torch_frame import (
    TorchFrameError,
    TorchMoonFrame,
    quat_conjugate_torch,
    quat_rotate_torch,
    topo_payload_to_torch,
)


class TorchSHPreflightError(RuntimeError):
    """Hard contract violation for the torch_cuda_sh runtime.

    Raised for problems that must NOT be silently worked around by dropping to
    another backend: a requested SH degree above the loaded coefficient file, an
    unsupported active perturbation, or a missing/!invalid gravity model.  The batch
    engine re-raises this instead of falling back to CPU so the failure is loud.
    """


# Conservative per-sample VRAM safety multiplier for the chunk estimate.  The
# Legendre workspace (P and dP) dominates; the evaluator also materialises a
# handful of (N, degree)-shaped intermediate tensors per call, so we inflate the
# raw P+dP footprint by this factor rather than modelling every temporary.
_VRAM_SAFETY_FACTOR = 8.0
# Fraction of *free* VRAM a single chunk is allowed to occupy.
_VRAM_SAFE_FRACTION = 0.80
# Default chunk when no explicit chunk_size is given and memory info is absent.
_DEFAULT_CHUNK = 1024


_quat_rotate = quat_rotate_torch
_TorchMoonFrame = TorchMoonFrame


class _SHAccelerationProvider:
    """R07 acceleration provider: per-stage frame rotation + batched SH gravity.

    The quaternion is resolved once per stage and reused for the forward and
    inverse rotation (identical numerics to the pre-R07 in-class ``_rhs``).
    """

    def __init__(self, evaluator: Any, frame: Any) -> None:
        self._evaluator = evaluator
        self._frame = frame

    def acceleration(self, t_s: float, s: Any) -> Any:
        r_i = s[:, :3]
        if self._frame.uses_rotation:
            q = self._frame.quat_i2f(t_s)
            r_f = quat_rotate_torch(q, r_i)
            a_f = self._evaluator.acceleration(r_f)
            return quat_rotate_torch(quat_conjugate_torch(q), a_f)
        return self._evaluator.acceleration(r_i)


class TorchSHBatchPropagator:
    """Fixed-step RK4 batch propagator for ``torch_cuda_sh`` (gravity-only).

    Parameters
    ----------
    dynamics_engine :
        Prepared ``DynamicsEngine``.  ``.grav`` must satisfy the normalized
        gravity contract read by :class:`TorchSHGravityEvaluator` (``R_ref_m``,
        ``GM_m3s2``, ``Cnm``, ``Snm``, recurrence tables, ``scale_m``,
        ``degree_max``).  ``.ephem`` (optional) supplies the Moon-fixed
        attitude timeline.
    batch_cfg :
        ``BatchPropagationConfig`` (reads ``sh_degree``, ``dt_s``, ``impact_alt_km``,
        ``torch_dtype``, ``torch_sh_chunk_size``, ``gpu_device_id``).
    flags :
        ``PerturbationFlags``.  Must be gravity-only on this path.
    device / dtype / chunk_size :
        Optional overrides (used by tests and the CPU validation path).
    """

    def __init__(
        self,
        dynamics_engine: Any,
        batch_cfg: Any,
        flags: Any,
        *,
        device: Any = None,
        dtype: Any = None,
        chunk_size: int | None = None,
        topo_payload: dict | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a hard dep of this path
            raise TorchSHPreflightError("PyTorch is required for torch_cuda_sh.") from exc

        self._torch = torch
        self._cfg = batch_cfg

        # --- Resolve device --------------------------------------------------
        if device is not None:
            self._device = torch.device(device)
        else:
            dev_id = int(getattr(batch_cfg, "gpu_device_id", 0) or 0)
            self._device = (
                torch.device(f"cuda:{dev_id}") if torch.cuda.is_available() else torch.device("cpu")
            )

        # Honest device contract: an explicit CUDA device must never silently
        # degrade to CPU. If CUDA is unavailable, raise so the batch engine performs
        # a *recorded* fallback (downgrade_plan_to_cpu) instead of a hidden CPU
        # run mislabeled as torch_cuda_sh. This is a plain RuntimeError (not a
        # TorchSHPreflightError) precisely so the engine catches it and falls back.
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"torch_cuda_sh requested CUDA device {self._device}, but PyTorch "
                "CUDA is unavailable; refusing to silently run on CPU."
            )

        # --- Resolve dtype ---------------------------------------------------
        if dtype is not None:
            self._dtype = dtype
        else:
            dtype_name = str(getattr(batch_cfg, "torch_dtype", "float64") or "float64").lower()
            self._dtype = torch.float64 if dtype_name == "float64" else torch.float32

        self._dt = float(getattr(batch_cfg, "dt_s", 60.0))
        self._impact_alt_m = float(getattr(batch_cfg, "impact_alt_km", 0.0)) * 1_000.0
        self._impact_r = float(R_MOON) + self._impact_alt_m
        self._detect_impact = bool(getattr(batch_cfg, "impact_detection_enabled", True))
        self._terrain_requested = self._detect_impact and (
            str(getattr(batch_cfg, "impact_surface_mode", "sphere")) == "terrain"
        )
        self._topo_payload = topo_payload

        # --- Physics preflight: gravity-only --------------------------------
        unsupported = unsupported_force_models("torch_cuda_sh", flags) if flags is not None else ()
        if unsupported:
            raise TorchSHPreflightError(
                "torch_cuda_sh is gravity-only; it cannot model: "
                + ", ".join(unsupported)
                + ". Route this run through batch_backend='auto' (CPU fallback) instead."
            )

        # --- Gravity model + coefficient/degree preflight (task §8) ----------
        grav = getattr(dynamics_engine, "grav", None)
        if grav is None:
            raise TorchSHPreflightError(
                "torch_cuda_sh requires a classic spherical-harmonic gravity model "
                "on the dynamics engine, but none is attached."
            )
        loaded_max = int(getattr(grav, "degree_max", getattr(grav, "max_degree", 0)))
        sh_enabled = bool(getattr(flags, "enable_sh", True)) if flags is not None else True
        requested = int(getattr(batch_cfg, "sh_degree", 0) or 0) if sh_enabled else 0
        if requested > loaded_max:
            raise TorchSHPreflightError(
                f"Requested SH degree {requested}, but loaded gravity model supports "
                f"degree {loaded_max}. The degree is never silently reduced; load a "
                f"higher-degree coefficient file or lower sh_degree."
            )
        # Verify the coefficient arrays are actually dimensioned for the request.
        c_arr = np.asarray(grav.Cnm)
        if c_arr.ndim < 2 or c_arr.shape[0] < (requested + 1) or c_arr.shape[1] < (requested + 1):
            raise TorchSHPreflightError(
                f"Cnm array shape {c_arr.shape} is too small for requested degree {requested}."
            )

        self._requested_degree = int(requested)
        self._actual_degree = int(requested)  # never clipped
        self._loaded_degree_max = int(loaded_max)

        # --- Build the canonical evaluator + frame provider ------------------
        from lunaris.physics.torch_spherical_harmonics import TorchSHGravityEvaluator

        self._evaluator = TorchSHGravityEvaluator(
            grav, degree=self._actual_degree, device=self._device, dtype=self._dtype
        )
        try:
            self._frame = TorchMoonFrame(
                getattr(dynamics_engine, "ephem", None),
                device=self._device,
                dtype=self._dtype,
                allow_identity=bool(getattr(dynamics_engine, "allow_identity_rotation", True)),
            )
        except TorchFrameError as exc:
            raise TorchSHPreflightError(str(exc)) from exc

        # Terrain-aware impact freeze: device-resident topography payload, only
        # when requested AND a usable grid is present (else constant-sphere path).
        self._topo = (
            topo_payload_to_torch(self._topo_payload, device=self._device, dtype=self._dtype)
            if self._terrain_requested
            else None
        )
        self._terrain_enabled = self._topo is not None
        if self._terrain_requested and not self._terrain_enabled:
            warnings.warn(
                "impact_surface_mode='terrain' requested but no usable topography "
                "payload was provided; falling back to constant-sphere impact freeze.",
                RuntimeWarning,
                stacklevel=2,
            )

        # --- GPU memory preflight + chunk sizing (task §11/§12) --------------
        requested_chunk = (
            int(chunk_size) if chunk_size is not None
            else int(getattr(batch_cfg, "torch_sh_chunk_size", 0) or 0)
        )
        self._bytes_per_sample = self._estimate_bytes_per_sample()
        self._free_mem_bytes, self._total_mem_bytes = self._query_device_memory()
        self._chunk_size = self._resolve_chunk_size(requested_chunk)

        self._device_name = self._resolve_device_name()
        self._throughput_metrics: dict = {}

    # ------------------------------------------------------------------
    # Preflight helpers
    # ------------------------------------------------------------------

    def _dtype_bytes(self) -> int:
        return 8 if self._dtype == self._torch.float64 else 4

    def _estimate_bytes_per_sample(self) -> int:
        """Per-sample device-memory estimate for the chunk preflight.

        Assumptions (documented so the number is auditable, not magic):
          * the dominant cost is the Legendre workspace P and dP, each shaped
            ``(chunk, degree+1, degree+2)`` -> ``2 * (d+1)(d+2)`` elements/sample;
          * ``cos_m`` / ``sin_m`` and the per-degree intermediate term tensors add
            ``~6 (d+1)`` elements/sample;
          * RK4 keeps state + 4 stage derivatives + temporaries -> ``~60`` elems;
          * the whole footprint is inflated by ``_VRAM_SAFETY_FACTOR`` to cover the
            transient tensors PyTorch allocates inside the evaluator.
        """
        d = int(self._actual_degree)
        workspace_elems = 2 * (d + 1) * (d + 2) + 6 * (d + 1) + 60
        return int(workspace_elems * self._dtype_bytes() * _VRAM_SAFETY_FACTOR)

    def _query_device_memory(self) -> tuple[int, int]:
        torch = self._torch
        if self._device.type != "cuda":
            return (0, 0)
        try:
            free, total = torch.cuda.mem_get_info(self._device)
            return (int(free), int(total))
        except Exception:
            return (0, 0)

    def _resolve_chunk_size(self, requested_chunk: int) -> int:
        """Pick a chunk size that fits the per-sample VRAM budget.

        Chunking changes only memory use, never the numbers: each chunk is an
        independent slice of the (per-sample-independent) sample axis, so any
        chunk size yields identical trajectories.
        """
        base = int(requested_chunk) if requested_chunk and requested_chunk > 0 else _DEFAULT_CHUNK
        if self._device.type != "cuda" or self._free_mem_bytes <= 0:
            return max(1, base)

        budget = float(self._free_mem_bytes) * _VRAM_SAFE_FRACTION
        cap = int(budget / max(1, self._bytes_per_sample))
        if cap < 1:
            # A single sample does not fit the safe VRAM fraction: fail loudly
            # rather than launching into a silent OOM.
            raise TorchSHPreflightError(
                f"Estimated {self._bytes_per_sample / 1e6:.1f} MB/sample for torch_cuda_sh "
                f"at degree {self._actual_degree} ({str(self._dtype).replace('torch.', '')}) "
                f"exceeds the safe VRAM budget ({budget / 1e6:.1f} MB free*{_VRAM_SAFE_FRACTION:g}). "
                "Lower the degree, use float32, or free GPU memory."
            )
        return max(1, min(base, cap))

    def _resolve_device_name(self) -> str:
        torch = self._torch
        if self._device.type == "cuda":
            try:
                return str(torch.cuda.get_device_name(self._device.index or 0))
            except Exception:
                return f"cuda:{self._device.index or 0}"
        return "cpu"

    # ------------------------------------------------------------------
    # Public interface (matches GPUBatchPropagator / TorchBatchPropagator)
    # ------------------------------------------------------------------

    def recommended_max_batch(self, requested_max_batch: int | None = None) -> int:
        """Return the engine sub-batch cap.

        The torch SH path does its own VRAM-aware chunking internally, so the
        engine may hand us the whole ensemble; we keep the caller's requested cap.
        """
        if requested_max_batch is None:
            return max(1, int(self._chunk_size))
        return max(1, int(requested_max_batch))

    def diagnostics_snapshot(self) -> dict:
        """Lightweight runtime diagnostics for logs, reports, and provenance."""
        backend_name = "torch_cuda_sh" if self._device.type == "cuda" else "torch_cpu_sh"
        diag = {
            "backend": backend_name,
            "backend_implementation": "torch",
            "device_name": self._device_name,
            "device": str(self._device),
            "dtype": str(self._dtype).replace("torch.", ""),
            "integrator": "fixed-step RK4",
            "requested_sh_degree": int(self._requested_degree),
            "sh_degree": int(self._actual_degree),
            "actual_sh_degree": int(self._actual_degree),
            "loaded_degree_max": int(self._loaded_degree_max),
            "chunk_size": int(self._chunk_size),
            "bytes_per_sample": int(self._bytes_per_sample),
            "gpu_free_mem_bytes": int(self._free_mem_bytes),
            "gpu_total_mem_bytes": int(self._total_mem_bytes),
            "frame_mode": "moon_fixed_slerp" if self._frame.uses_rotation else "identity",
            "frame_interpolation": "slerp_shortest_path",
            "uses_frame_rotation": bool(self._frame.uses_rotation),
            "impact_position_method": (
                "terrain_bisection_hybrid"
                if getattr(self, "_terrain_enabled", False)
                else "line_sphere_quadratic"
            ),
            "impact_surface_mode": "terrain" if getattr(self, "_terrain_enabled", False) else "sphere",
        }
        diag.update(self._throughput_metrics)
        return diag

    def provenance(self) -> dict:
        """Backend provenance for the run manifest (task §13)."""
        backend_name = "torch_cuda_sh" if self._device.type == "cuda" else "torch_cpu_sh"
        return {
            "actual_backend": backend_name,
            "backend_family": "classic_sh",
            "backend_implementation": "torch",
            "actual_device": self._device_name,
            "requested_sh_degree": int(self._requested_degree),
            "actual_sh_degree": int(self._actual_degree),
            "integrator": "fixed-step RK4",
            "dtype": str(self._dtype).replace("torch.", ""),
            "chunk_size": int(self._chunk_size),
            "impact_position_method": (
                "terrain_bisection_hybrid"
                if getattr(self, "_terrain_enabled", False)
                else "line_sphere_quadratic"
            ),
            "impact_surface_mode": "terrain" if getattr(self, "_terrain_enabled", False) else "sphere",
        }

    # ------------------------------------------------------------------
    # RHS + RK4 — thin wrappers over the shared loop (R07)
    # ------------------------------------------------------------------

    @property
    def _provider(self) -> _SHAccelerationProvider:
        return _SHAccelerationProvider(self._evaluator, self._frame)

    def _rhs(self, t_s: float, s: Any) -> Any:
        """Evaluate ``[v; a]`` for state ``[N, 6]`` at epoch ``t_s``."""
        return rhs_batch(self._torch, self._provider, t_s, s)

    def _rk4_step(self, s: Any, t_s: float, h: float) -> Any:
        """One classic RK4 step with per-stage frame transforms (task §7)."""
        return rk4_step(self._torch, self._provider, s, t_s, h)

    # ------------------------------------------------------------------
    # Public: propagate batch
    # ------------------------------------------------------------------

    def propagate(
        self,
        Y0: np.ndarray,            # (N, 6) initial states
        masses: np.ndarray,        # (N,)  — accepted but unused (gravity only)
        areas: np.ndarray,         # (N,)  — accepted but unused
        cds: np.ndarray,           # (N,)  — accepted but unused
        crs: np.ndarray,           # (N,)  — accepted but unused
        duration_s: float,
        output_dt_s: float,
        callback: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propagate ``N`` samples with fixed-step RK4 + per-stage SH gravity.

        Returns ``(t_out, Y_out, impact_flags, t_impact)`` with ``Y_out`` shaped
        ``(n_snaps + 1, N, 6)`` (the initial state is snapshot 0).
        """
        N = int(Y0.shape[0])
        dt = float(self._dt)

        # Grid preview for the run header only; the shared loop rebuilds the
        # same grid (single contract: build_batch_output_grid).
        _t_out, n_snaps, snap_interval = build_batch_output_grid(duration_s, output_dt_s)
        steps_per_snap = max(1, int(round(snap_interval / dt)))
        dt_eff = snap_interval / steps_per_snap

        chunk = max(1, int(self._chunk_size))
        n_chunks = int(math.ceil(N / chunk))
        log_backend = "torch_cuda_sh" if self._device.type == "cuda" else "torch_cpu_sh"

        print(
            f"[BATCH][{log_backend}] N={N}  device={self._device} ({self._device_name})  "
            f"degree={self._actual_degree}  dtype={str(self._dtype).replace('torch.', '')}  "
            f"chunk={chunk}  chunks={n_chunks}  dt={dt_eff:.1f}s  snaps={n_snaps}  "
            f"frame={'moon-fixed' if self._frame.uses_rotation else 'identity'}",
            flush=True,
        )

        result = run_batched_fixed_step(
            torch_mod=self._torch,
            device=self._device,
            dtype=self._dtype,
            provider=self._provider,
            frame=self._frame,
            Y0=Y0,
            duration_s=duration_s,
            output_dt_s=output_dt_s,
            dt_s=dt,
            impact_r_m=self._impact_r,
            detect_impact=self._detect_impact,
            topo=getattr(self, "_topo", None),
            impact_alt_m=float(getattr(self, "_impact_alt_m", 0.0)),
            chunk_size=chunk,
            callback=callback,
            callback_granularity="chunk",
        )
        t_out = result.t_out
        Y_out = result.Y_out
        impact_flags = result.impact_flags
        t_impact = result.t_impact
        impact_positions = result.impact_positions_inertial
        elapsed = float(result.metrics["propagation_elapsed_s"])

        n_impacts = int(result.metrics["impacted_sample_count"])
        self._throughput_metrics = {
            "raw_batch_state_steps_per_second": result.metrics["raw_batch_state_steps_per_second"],
            "active_state_steps_per_second": result.metrics["active_state_steps_per_second"],
            "active_sample_count": int(N - n_impacts),
            "impacted_sample_count": n_impacts,
            "impact_fraction": float(n_impacts) / max(N, 1),
            "total_raw_state_steps": int(result.metrics["total_raw_state_steps"]),
            "total_active_state_steps": int(result.metrics["total_active_state_steps"]),
            "propagation_elapsed_s": float(elapsed),
            "impact_position_method": (
                "terrain_bisection_hybrid"
                if getattr(self, "_terrain_enabled", False)
                else "line_sphere_quadratic"
            ),
            "impact_time_resolution_s": float(dt_eff),
        }
        print(
            f"[BATCH][{log_backend}] done: {elapsed:.2f}s  "
            f"{self._throughput_metrics['raw_batch_state_steps_per_second']:,.0f} raw-steps/s  "
            f"{self._throughput_metrics['active_state_steps_per_second']:,.0f} active-steps/s",
            flush=True,
        )
        self._last_impact_positions_inertial = impact_positions
        return t_out, Y_out, impact_flags, t_impact

    def last_impact_positions_inertial(self) -> np.ndarray:
        """Return fixed-step endpoint impact positions for the latest batch."""
        return np.asarray(
            getattr(self, "_last_impact_positions_inertial", np.empty((0, 3))),
            dtype=np.float64,
        )


__all__ = ["TorchSHBatchPropagator", "TorchSHPreflightError"]
