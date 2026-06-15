# ST_LRPS/core/torch_sh_propagator.py
"""
Torch Classic-SH Batch Monte Carlo Propagator (``torch_cuda_sh`` / ``torch_cpu_sh`` runtime)
=========================================================================

This is the live runtime behind the ``torch_cuda_sh`` and ``torch_cpu_sh`` backends.  It propagates
``N`` Monte Carlo samples simultaneously as a single ``[N, 6]`` PyTorch tensor
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

Contract (matches :class:`lunaris.core.mc_propagator.GPUBatchPropagator`)
------------------------------------------------------------------------
``propagate(Y0, masses, areas, cds, crs, duration_s, output_dt_s, callback)``
returns ``(t_out, Y_out, impact_flags, t_impact)``.  Spacecraft properties are
accepted for API parity but ignored: this first runtime form is **gravity-only**
(lunar SH + Moon inertial<->fixed frame transform).  Any active perturbation is
a hard contract violation here — :func:`resolve_mc_backend_policy` is responsible
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
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from lunaris.common.constants import R_MOON
from lunaris.common.montecarlo_defs import build_mc_output_grid
from lunaris.core.backend_capabilities import unsupported_force_models


class TorchSHPreflightError(RuntimeError):
    """Hard contract violation for the torch_cuda_sh runtime.

    Raised for problems that must NOT be silently worked around by dropping to
    another backend: a requested SH degree above the loaded coefficient file, an
    unsupported active perturbation, or a missing/!invalid gravity model.  The MC
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


def _quat_rotate(q: Any, v: Any) -> Any:
    """Rotate a batch of vectors ``v`` (``[N, 3]``) by scalar-first quaternion ``q`` (``[4]``).

    Active rotation ``v' = v + 2 q0 (q_vec x v) + 2 q_vec x (q_vec x v)``; matches
    ``_quat_rot_cuda`` in :mod:`lunaris.core.mc_propagator` and ``_quat_rotate_torch``
    in the ST-LRPS benchmark.
    """
    import torch

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    tx = 2.0 * (q2 * vz - q3 * vy)
    ty = 2.0 * (q3 * vx - q1 * vz)
    tz = 2.0 * (q1 * vy - q2 * vx)
    rx = vx + q0 * tx + (q2 * tz - q3 * ty)
    ry = vy + q0 * ty + (q3 * tx - q1 * tz)
    rz = vz + q0 * tz + (q1 * ty - q2 * tx)
    return torch.stack((rx, ry, rz), dim=1)


class _TorchMoonFrame:
    """Moon inertial<->fixed frame provider for the torch SH RK4 path.

    Reads the ``q_i2f`` (inertial->fixed, scalar-first) quaternion timeline from
    an :class:`~lunaris.physics.ephemeris.EphemerisManager` data provider and
    SLERP-interpolates it at arbitrary RK-stage epochs.  When no ephemeris is
    supplied the frame is the identity (body-fixed == inertial), matching
    ``DynamicsEngine(allow_identity_rotation=True)``.
    """

    def __init__(self, ephem: Any, *, device: Any, dtype: Any) -> None:
        import torch

        self.device = device
        self.dtype = dtype
        if ephem is None:
            self.uses_rotation = False
            self.dt_s = 1.0
            self.q_tab = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
            return

        provider = ephem.get_data_provider()
        q_np = np.asarray(provider["q_i2f_tab"], dtype=np.float64)
        if q_np.ndim != 2 or q_np.shape[1] != 4:
            raise TorchSHPreflightError(
                f"ephemeris q_i2f_tab must have shape (N, 4); got {q_np.shape}."
            )
        self.dt_s = float(provider["dt_s"])
        self.q_tab = torch.as_tensor(q_np, device=device, dtype=dtype)
        self.uses_rotation = bool(self.q_tab.shape[0] >= 1)

    def quat_i2f(self, t_s: float) -> Any:
        """Return the SLERP-interpolated ``q_i2f`` quaternion ``[4]`` at time ``t_s``."""
        import torch

        if self.q_tab.shape[0] <= 1:
            return self.q_tab[0]
        u = max(0.0, float(t_s) / max(self.dt_s, 1e-12))
        i0 = int(math.floor(u))
        if i0 >= self.q_tab.shape[0] - 1:
            return self.q_tab[-1]
        frac = torch.tensor(u - i0, device=self.device, dtype=self.dtype)
        qa = self.q_tab[i0]
        qb = self.q_tab[i0 + 1]
        dot = torch.dot(qa, qb)
        sign = torch.where(dot < 0.0, -torch.ones_like(dot), torch.ones_like(dot))
        qb = qb * sign
        dot = torch.clamp(dot * sign, -1.0, 1.0)
        q_linear = (1.0 - frac) * qa + frac * qb
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0).clamp_min(1e-30)
        theta = theta_0 * frac
        s0 = torch.sin(theta_0 - theta) / sin_theta_0
        s1 = torch.sin(theta) / sin_theta_0
        q_slerp = s0 * qa + s1 * qb
        # Near-parallel quaternions: linear blend avoids 0/0 in the SLERP weights.
        q = torch.where(dot > 0.9995, q_linear, q_slerp)
        return q / torch.linalg.norm(q).clamp_min(1e-30)


class TorchSHBatchPropagator:
    """Fixed-step RK4 Monte Carlo propagator for ``torch_cuda_sh`` (gravity-only).

    Parameters
    ----------
    dynamics_engine :
        Prepared ``DynamicsEngine``.  ``.grav`` must satisfy the normalized
        gravity contract read by :class:`TorchSHGravityEvaluator` (``R_ref_m``,
        ``GM_m3s2``, ``Cnm``, ``Snm``, recurrence tables, ``scale_m``,
        ``degree_max``).  ``.ephem`` (optional) supplies the Moon-fixed
        attitude timeline.
    mc_cfg :
        ``MonteCarloConfig`` (reads ``gpu_sh_degree``, ``dt_s``, ``impact_alt_km``,
        ``torch_dtype``, ``torch_sh_chunk_size``, ``gpu_device_id``).
    flags :
        ``PerturbationFlags``.  Must be gravity-only on this path.
    device / dtype / chunk_size :
        Optional overrides (used by tests and the CPU validation path).
    """

    def __init__(
        self,
        dynamics_engine: Any,
        mc_cfg: Any,
        flags: Any,
        *,
        device: Any = None,
        dtype: Any = None,
        chunk_size: int | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a hard dep of this path
            raise TorchSHPreflightError("PyTorch is required for torch_cuda_sh.") from exc

        self._torch = torch
        self._mc = mc_cfg

        # --- Resolve device --------------------------------------------------
        if device is not None:
            self._device = torch.device(device)
        else:
            dev_id = int(getattr(mc_cfg, "gpu_device_id", 0) or 0)
            self._device = (
                torch.device(f"cuda:{dev_id}") if torch.cuda.is_available() else torch.device("cpu")
            )

        # Honest device contract: an explicit CUDA device must never silently
        # degrade to CPU. If CUDA is unavailable, raise so the MC engine performs
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
            dtype_name = str(getattr(mc_cfg, "torch_dtype", "float64") or "float64").lower()
            self._dtype = torch.float64 if dtype_name == "float64" else torch.float32

        self._dt = float(getattr(mc_cfg, "dt_s", 60.0))
        self._impact_r = float(R_MOON) + float(getattr(mc_cfg, "impact_alt_km", 0.0)) * 1_000.0

        # --- Physics preflight: gravity-only --------------------------------
        unsupported = unsupported_force_models("torch_cuda_sh", flags) if flags is not None else ()
        if unsupported:
            raise TorchSHPreflightError(
                "torch_cuda_sh is gravity-only; it cannot model: "
                + ", ".join(unsupported)
                + ". Route this run through mc_backend='auto' (CPU fallback) instead."
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
        requested = int(getattr(mc_cfg, "gpu_sh_degree", 0) or 0) if sh_enabled else 0
        if requested > loaded_max:
            raise TorchSHPreflightError(
                f"Requested SH degree {requested}, but loaded gravity model supports "
                f"degree {loaded_max}. The degree is never silently reduced; load a "
                f"higher-degree coefficient file or lower gpu_sh_degree."
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
        self._frame = _TorchMoonFrame(
            getattr(dynamics_engine, "ephem", None), device=self._device, dtype=self._dtype
        )

        # --- GPU memory preflight + chunk sizing (task §11/§12) --------------
        requested_chunk = (
            int(chunk_size) if chunk_size is not None
            else int(getattr(mc_cfg, "torch_sh_chunk_size", 0) or 0)
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
            "requested_gpu_sh_degree": int(self._requested_degree),
            "gpu_sh_degree": int(self._actual_degree),
            "actual_gpu_sh_degree": int(self._actual_degree),
            "loaded_degree_max": int(self._loaded_degree_max),
            "chunk_size": int(self._chunk_size),
            "bytes_per_sample": int(self._bytes_per_sample),
            "gpu_free_mem_bytes": int(self._free_mem_bytes),
            "gpu_total_mem_bytes": int(self._total_mem_bytes),
            "frame_mode": "moon_fixed_slerp" if self._frame.uses_rotation else "identity",
            "uses_frame_rotation": bool(self._frame.uses_rotation),
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
        }

    # ------------------------------------------------------------------
    # RHS + RK4
    # ------------------------------------------------------------------

    def _rhs(self, t_s: float, s: Any) -> Any:
        """Evaluate ``[v; a]`` for state ``[N, 6]`` at epoch ``t_s``."""
        torch = self._torch
        r_i = s[:, :3]
        v_i = s[:, 3:]
        if self._frame.uses_rotation:
            q = self._frame.quat_i2f(t_s)
            r_f = _quat_rotate(q, r_i)
            a_f = self._evaluator.acceleration(r_f)
            q_conj = q.clone()
            q_conj[1:] = -q_conj[1:]
            a_i = _quat_rotate(q_conj, a_f)
        else:
            a_i = self._evaluator.acceleration(r_i)
        return torch.cat((v_i, a_i), dim=1)

    def _rk4_step(self, s: Any, t_s: float, h: float) -> Any:
        """One classic RK4 step with per-stage frame transforms (task §7)."""
        k1 = self._rhs(t_s, s)
        k2 = self._rhs(t_s + 0.5 * h, s + (0.5 * h) * k1)
        k3 = self._rhs(t_s + 0.5 * h, s + (0.5 * h) * k2)
        k4 = self._rhs(t_s + h, s + h * k3)
        return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

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

        # Shared output grid contract: t[0]=0, t[-1]=duration_s, uniform.
        t_out, n_snaps, snap_interval = build_mc_output_grid(duration_s, output_dt_s)
        steps_per_snap = max(1, int(round(snap_interval / dt)))
        dt_eff = snap_interval / steps_per_snap
        Y_out = np.empty((n_snaps + 1, N, 6), dtype=np.float64)
        impact_flags = np.zeros(N, dtype=np.float64)
        t_impact = np.full(N, np.nan, dtype=np.float64)

        chunk = max(1, int(self._chunk_size))
        n_chunks = int(math.ceil(N / chunk))

        total_raw_state_steps = 0
        total_active_state_steps = 0
        total_steps_per_sample = n_snaps * steps_per_snap

        t_start = time.perf_counter()
        print(
            f"[MC][torch_cuda_sh] N={N}  device={self._device} ({self._device_name})  "
            f"degree={self._actual_degree}  dtype={str(self._dtype).replace('torch.', '')}  "
            f"chunk={chunk}  chunks={n_chunks}  dt={dt_eff:.1f}s  snaps={n_snaps}  "
            f"frame={'moon-fixed' if self._frame.uses_rotation else 'identity'}",
            flush=True,
        )

        for ci in range(n_chunks):
            a = ci * chunk
            b = min(N, a + chunk)
            chunk_n = b - a
            active_steps = self._propagate_chunk(
                Y0[a:b], a, b, steps_per_snap, dt_eff, n_snaps, Y_out, impact_flags, t_impact
            )
            total_raw_state_steps += chunk_n * total_steps_per_sample
            total_active_state_steps += active_steps
            if callback is not None:
                callback(float(b) / float(max(N, 1)))

        if self._device.type == "cuda":
            try:
                self._torch.cuda.synchronize(self._device)
            except Exception:
                pass
        elapsed = time.perf_counter() - t_start

        n_impacts = int(np.sum(impact_flags > 0.5))
        self._throughput_metrics = {
            "raw_batch_state_steps_per_second": float(total_raw_state_steps) / max(elapsed, 1e-9),
            "active_state_steps_per_second": float(total_active_state_steps) / max(elapsed, 1e-9),
            "active_sample_count": int(N - n_impacts),
            "impacted_sample_count": n_impacts,
            "impact_fraction": float(n_impacts) / max(N, 1),
            "total_raw_state_steps": total_raw_state_steps,
            "total_active_state_steps": total_active_state_steps,
            "propagation_elapsed_s": float(elapsed),
        }
        print(
            f"[MC][torch_cuda_sh] done: {elapsed:.2f}s  "
            f"{self._throughput_metrics['raw_batch_state_steps_per_second']:,.0f} raw-steps/s  "
            f"{self._throughput_metrics['active_state_steps_per_second']:,.0f} active-steps/s",
            flush=True,
        )
        return t_out, Y_out, impact_flags, t_impact

    def _propagate_chunk(
        self,
        Y0_chunk: np.ndarray,
        a: int,
        b: int,
        steps_per_snap: int,
        dt_eff: float,
        n_snaps: int,
        Y_out: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
    ) -> int:
        """Propagate one chunk of samples through the RK4 loop.

        Returns the total number of active state-step evaluations executed
        in this chunk (for throughput accounting).
        """
        torch = self._torch
        device = self._device

        state = torch.as_tensor(
            np.ascontiguousarray(Y0_chunk, dtype=np.float64), device=device, dtype=self._dtype
        )
        n = int(state.shape[0])
        alive = torch.ones(n, dtype=torch.bool, device=device)

        Y_out[0, a:b, :] = state.detach().cpu().numpy().astype(np.float64)

        # Device-side accumulators: avoid per-step host synchronization. Both the
        # active-step count and the first-impact step index are tracked on device
        # and converted to host scalars exactly once, at the end of the chunk
        # (task §3: no per-step .item()/CPU<->GPU sync in the RK4 hot loop — a
        # device-to-host sync every step would distort the very throughput we are
        # trying to measure on CUDA).
        active_steps_acc = torch.zeros((), dtype=torch.int64, device=device)
        impact_step = torch.full((n,), -1, dtype=torch.int64, device=device)

        t_curr = 0.0
        global_step = 0
        # TODO(perf): Benchmark masking or compacting impacted samples so they
        # no longer participate in SH evaluation. Deferred to preserve fixed
        # batch shapes and avoid gather/scatter overhead until profiling shows
        # a net benefit.
        with torch.no_grad():
            for snap_idx in range(n_snaps):
                for _ in range(steps_per_snap):
                    active_steps_acc += alive.sum()
                    state = self._rk4_step(state, t_curr, dt_eff)
                    t_curr += dt_eff
                    global_step += 1
                    r_mag = torch.linalg.norm(state[:, :3], dim=1)
                    newly = alive & (r_mag <= self._impact_r)
                    # Record the 1-based step index of first impact on device;
                    # already-impacted samples keep their earlier index. No host
                    # sync here — the bookkeeping is resolved once below.
                    impact_step = torch.where(
                        newly, torch.full_like(impact_step, global_step), impact_step
                    )
                    alive = alive & ~newly
                Y_out[snap_idx + 1, a:b, :] = state.detach().cpu().numpy().astype(np.float64)

        # Single host sync per chunk: resolve impact bookkeeping. ``step * dt_eff``
        # equals the ``t_curr`` value immediately after the impacting step, so the
        # recorded impact epoch is identical to the per-step accounting it replaces.
        impact_step_host = impact_step.detach().cpu().numpy()
        for li in np.nonzero(impact_step_host >= 0)[0].tolist():
            gi = a + int(li)
            if impact_flags[gi] == 0.0:
                impact_flags[gi] = 1.0
                t_impact[gi] = float(impact_step_host[li]) * dt_eff

        return int(active_steps_acc.item())


__all__ = ["TorchSHBatchPropagator", "TorchSHPreflightError"]
