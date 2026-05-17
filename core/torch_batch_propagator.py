# LUNAR_SIMULATION/core/torch_batch_propagator.py
# -*- coding: utf-8 -*-
"""
GPU-Accelerated Batched Monte Carlo Propagator — ST-LRPS Path
==============================================================

Propagates N trajectories simultaneously as a single ``[N, 6]`` CUDA float32
tensor using PyTorch fixed-step RK4 and the ST-LRPS neural surrogate for
gravity.

Architecture
------------
``rhs_batch(state)``
    Splits the ``[N, 6]`` state into positions ``[N, 3]`` and velocities
    ``[N, 3]``, evaluates the total acceleration via the surrogate's
    ``predict_total_accel_torch`` (which internally runs the neural forward
    pass + ``torch.autograd.grad`` on the CUDA device), then concatenates
    ``[v, a]`` to return the derivative tensor.

``rk4_step(state, dt)``
    Standard four-stage RK4; all intermediate tensors stay on CUDA:
    ::
        k1 = rhs(s)
        k2 = rhs(s + 0.5*dt*k1)
        k3 = rhs(s + 0.5*dt*k2)
        k4 = rhs(s + dt*k3)
        s_next = s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

Limitations (current version)
------------------------------
- Gravity only: point-mass + ST-LRPS neural residual.
- No third-body, SRP, albedo, tides, or relativity.
  Enabling any of those perturbations in ``SimConfig`` forces a CPU fallback
  (detected by ``core.mc_backend_policy.resolve_mc_backend_policy``).
- Fixed step size; no adaptive step control.
- State is float32 (sufficient for ensemble statistics with ≥ 500 m σ_r).

Performance notes
-----------------
Each RK4 step launches 4 batched neural forward passes + 4 autograd calls on
the CUDA device.  No per-step CPU round-trips occur once the run is started.
Snapshots are copied to host only at the ``output_dt_s`` cadence.

Timing metrics are printed to stdout at run start and end.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

import numpy as np

from common.constants import R_MOON


class TorchBatchPropagator:
    """
    Fixed-step RK4 Monte Carlo propagator backed by PyTorch CUDA.

    Parameters
    ----------
    surrogate_model : SurrogateGravityModel
        A loaded ST-LRPS model.  ``to_device`` is called during ``__init__``
        so the model and its scaling tensors are transferred to *device_id*.
    mc_cfg : MonteCarloConfig
        Monte Carlo configuration (``dt_s``, ``impact_alt_km``, …).
    device_id : int
        CUDA device index (default 0).
    """

    def __init__(
        self,
        surrogate_model: Any,
        mc_cfg: Any,
        device_id: int = 0,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for TorchBatchPropagator."
            ) from exc

        if not torch.cuda.is_available():  # pragma: no cover
            raise RuntimeError(
                "TorchBatchPropagator requires a CUDA device but "
                "torch.cuda.is_available() returned False."
            )

        self._torch = torch
        self._device = torch.device(f"cuda:{int(device_id)}")
        self._dt = float(getattr(mc_cfg, "dt_s", 60.0))
        self._impact_r = float(R_MOON) + float(getattr(mc_cfg, "impact_alt_km", 0.0)) * 1_000.0
        dtype_name = str(getattr(mc_cfg, "torch_dtype", "float32") or "float32").lower()
        self._dtype = torch.float64 if dtype_name == "float64" else torch.float32

        # Move surrogate model (weights + scaling tensors) to CUDA
        self._model = surrogate_model
        self._model.to_device(self._device)

    # ------------------------------------------------------------------
    # Public interface (matches GPUBatchPropagator / CPUBatchPropagator)
    # ------------------------------------------------------------------

    def diagnostics_snapshot(self) -> dict:
        """Return a diagnostics dict for the progress log."""
        torch = self._torch
        dev = self._device
        return {
            "backend": "GPU-ST-LRPS",
            "device_name": torch.cuda.get_device_name(dev.index or 0),
            "torch_cuda_version": str(torch.version.cuda or "unknown"),
            "threads_per_block": "managed by PyTorch",
        }

    def recommended_max_batch(self, budget: int) -> int:
        """
        Conservative VRAM-aware batch cap.

        Each sample needs ≈ 24 bytes (float32 [6]) × 4 RK4 stages + model
        activations.  We let the caller's budget dominate and just cap at
        10 000 as a safety ceiling for common GPU sizes.
        """
        return min(int(budget), 10_000)

    def propagate(
        self,
        Y0: np.ndarray,            # (N, 6) float64
        masses: np.ndarray,        # (N,)  — accepted but not used (gravity only)
        areas: np.ndarray,         # (N,)  — accepted but not used
        cds: np.ndarray,           # (N,)  — accepted but not used
        crs: np.ndarray,           # (N,)  — accepted but not used
        duration_s: float,
        output_dt_s: float,
        callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagate N samples simultaneously on CUDA.

        Returns
        -------
        t_out : (T,) float64   — snapshot times [s]
        Y_out : (T, N, 6) float64 — state ensemble [m, m/s]
        impact_flags : (N,) float64 — 1.0 if impacted, else 0.0
        t_impact : (N,) float64 — impact time (NaN if none)
        """
        torch = self._torch
        device = self._device
        model = self._model

        N = int(Y0.shape[0])
        dt = self._dt
        snap_interval = float(output_dt_s)
        total_time = float(duration_s)

        # Derive integration plan
        steps_per_snap = max(1, round(snap_interval / dt))
        dt_eff = snap_interval / steps_per_snap  # may differ slightly from dt
        n_snaps = max(1, round(total_time / snap_interval))

        t_out = np.linspace(0.0, n_snaps * snap_interval, n_snaps + 1, dtype=np.float64)
        Y_out = np.empty((n_snaps + 1, N, 6), dtype=np.float64)
        impact_flags = np.zeros(N, dtype=np.float64)
        t_impact_arr = np.full(N, np.nan, dtype=np.float64)

        # Transfer initial state to CUDA (float32 for performance)
        state = torch.as_tensor(Y0, dtype=self._dtype, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        r_impact_t = torch.tensor(self._impact_r, dtype=self._dtype, device=device)

        # ------------------------------------------------------------------
        # Inner helpers (closures capture `model` and `device`)
        # ------------------------------------------------------------------

        def _rhs(s: "torch.Tensor") -> "torch.Tensor":
            """Evaluate [v; a] for state [N, 6]."""
            r = s[:, :3]                               # positions [N, 3]
            v = s[:, 3:]                               # velocities [N, 3]
            a = model.predict_total_accel_torch(r)     # [N, 3]
            return torch.cat([v, a], dim=1)

        def _rk4(s: "torch.Tensor", h: float) -> "torch.Tensor":
            k1 = _rhs(s)
            k2 = _rhs(s + (h * 0.5) * k1)
            k3 = _rhs(s + (h * 0.5) * k2)
            k4 = _rhs(s + h * k3)
            return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # ------------------------------------------------------------------
        # Print run header
        # ------------------------------------------------------------------
        deg_min = getattr(model, "degree_min", "?")
        deg_max = getattr(model, "degree_max", "?")
        dev_name = torch.cuda.get_device_name(device.index or 0)
        print(
            f"[MC][GPU-STLRPS] N={N}  device={device} ({dev_name})",
            flush=True,
        )
        print(
            f"[MC][GPU-STLRPS] degree_min={deg_min}  degree_max={deg_max}  "
            f"dt={dt_eff:.1f}s  snaps={n_snaps}  steps/snap={steps_per_snap}  "
            f"dtype={str(self._dtype).replace('torch.', '')}",
            flush=True,
        )

        # Time one batched acceleration call for the log
        _ = model.predict_total_accel_torch(state[:, :3])
        torch.cuda.synchronize(device)
        _t0 = time.perf_counter()
        _ = model.predict_total_accel_torch(state[:, :3])
        torch.cuda.synchronize(device)
        accel_ms = (time.perf_counter() - _t0) * 1_000.0
        print(
            f"[MC][GPU-STLRPS] one batched accel call: {accel_ms:.2f} ms  "
            f"state=[{N}, 6]",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Initial snapshot
        # ------------------------------------------------------------------
        Y_out[0] = state.detach().cpu().numpy().astype(np.float64)

        t_curr = 0.0
        t_prop_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Main integration loop
        # ------------------------------------------------------------------
        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                state = _rk4(state, dt_eff)
                t_curr += dt_eff

                # Impact detection on GPU — only alive samples
                if alive.any():
                    r_mag = torch.linalg.norm(state[:, :3], dim=1)  # [N]
                    newly_hit = alive & (r_mag <= r_impact_t)
                    if newly_hit.any():
                        hit_indices = newly_hit.nonzero(as_tuple=False).view(-1)
                        for idx in hit_indices.cpu().tolist():
                            if impact_flags[idx] == 0.0:
                                impact_flags[idx] = 1.0
                                t_impact_arr[idx] = t_curr
                        alive = alive & ~newly_hit

            Y_out[snap_idx + 1] = state.detach().cpu().numpy().astype(np.float64)

            if callback is not None:
                callback(float(snap_idx + 1) / float(n_snaps))

        # ------------------------------------------------------------------
        # Print timing summary
        # ------------------------------------------------------------------
        t_prop = time.perf_counter() - t_prop_start
        total_steps = n_snaps * steps_per_snap
        traj_steps_per_s = (N * total_steps) / max(t_prop, 1e-9)
        print(
            f"[MC][GPU-STLRPS] propagation complete: "
            f"{t_prop:.2f}s  {traj_steps_per_s:,.0f} trajectory-steps/s",
            flush=True,
        )

        return t_out, Y_out, impact_flags, t_impact_arr


__all__ = ["TorchBatchPropagator"]
