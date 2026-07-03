# lunaris.core.torch_batch_propagator
"""
GPU-Accelerated Batch Propagator — ST-LRPS Path
==============================================================

Propagates N trajectories simultaneously as a single ``[N, 6]`` CUDA tensor
using PyTorch fixed-step RK4 and the ST-LRPS neural surrogate for
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
  (detected by ``batch.backend_policy.resolve_batch_backend_policy``).
- Fixed step size; no adaptive step control.
- State dtype follows ``BatchPropagationConfig.torch_dtype`` (float32 by default for
  throughput, float64 when explicitly requested).

Performance notes
-----------------
Each RK4 step launches 4 batched neural forward passes + 4 autograd calls on
the CUDA device.  No per-step CPU round-trips occur once the run is started.
Snapshots are copied to host only at the ``output_dt_s`` cadence.

Timing metrics are printed to stdout at run start and end.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from lunaris.common.batch_defs import build_batch_output_grid
from lunaris.common.constants import R_MOON
from lunaris.core.torch_frame import (
    TorchFrameError,
    TorchMoonFrame,
    line_sphere_intersection,
    terrain_segment_intersection,
    topo_payload_to_torch,
)

if TYPE_CHECKING:
    # Annotation-only alias. The runtime ``torch`` handle is the per-instance
    # ``self._torch`` (optional dependency), which shadows a module import; a
    # dedicated name keeps the type annotations resolvable.
    from torch import Tensor


class TorchSTLRPSPreflightError(RuntimeError):
    """Hard ST-LRPS runtime contract violation that must not fall back silently."""


class TorchBatchPropagator:
    """
    Fixed-step RK4 batch propagator backed by PyTorch CUDA.

    Parameters
    ----------
    surrogate_model : SurrogateGravityModel
        A loaded ST-LRPS model.  ``to_device`` is called during ``__init__``
        so the model and its scaling tensors are transferred to *device_id*.
    batch_cfg : BatchPropagationConfig
        Batch propagation configuration (``dt_s``, ``impact_alt_km``, …).
    device_id : int
        CUDA device index (default 0).
    """

    def __init__(
        self,
        surrogate_model: Any,
        batch_cfg: Any,
        device_id: int = 0,
        ephem: Any = None,
        allow_identity_rotation: bool = False,
        topo_payload: dict | None = None,
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
        self._dt = float(getattr(batch_cfg, "dt_s", 60.0))
        self._impact_alt_m = float(getattr(batch_cfg, "impact_alt_km", 0.0)) * 1_000.0
        self._impact_r = float(R_MOON) + self._impact_alt_m
        self._detect_impact = bool(getattr(batch_cfg, "impact_detection_enabled", True))
        self._terrain_requested = self._detect_impact and (
            str(getattr(batch_cfg, "impact_surface_mode", "sphere")) == "terrain"
        )
        dtype_name = str(getattr(batch_cfg, "torch_dtype", "float32") or "float32").lower()
        self._dtype = torch.float64 if dtype_name == "float64" else torch.float32

        # Move surrogate model (weights + scaling tensors) to CUDA
        self._model = surrogate_model
        self._model.to_device(self._device)
        try:
            self._frame = TorchMoonFrame(
                ephem,
                device=self._device,
                dtype=self._dtype,
                allow_identity=bool(allow_identity_rotation),
            )
        except TorchFrameError as exc:
            raise TorchSTLRPSPreflightError(
                f"GPU ST-LRPS frame preflight failed: {exc}"
            ) from exc

        # Terrain-aware impact freeze: move the topography payload onto the device
        # only when requested AND a usable grid is present; otherwise stay on the
        # constant-sphere path (zero behaviour change).
        self._topo = (
            topo_payload_to_torch(topo_payload, device=self._device, dtype=self._dtype)
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
        self._throughput_metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface (matches GPUBatchPropagator / CPUBatchPropagator)
    # ------------------------------------------------------------------

    def diagnostics_snapshot(self) -> dict:
        """Return a diagnostics dict for the progress log."""
        torch = self._torch
        dev = self._device
        runtime = getattr(self._model, "_force_runtime", None)
        model_dtype = "float32"
        try:
            model_obj = getattr(runtime, "model", None)
            if model_obj is not None:
                model_dtype = str(next(model_obj.parameters()).dtype).replace("torch.", "")
        except Exception:
            pass
        diagnostics = {
            "backend": "GPU-ST-LRPS",
            "device_name": torch.cuda.get_device_name(dev.index or 0),
            "torch_cuda_version": str(torch.version.cuda or "unknown"),
            "threads_per_block": "managed by PyTorch",
            "runtime_model_kind": str(
                getattr(getattr(self._model, "_force_runtime", None), "runtime_model_kind", "")
                or getattr(self._model, "config", {}).get("runtime_model_kind", "potential_autograd")
            ),
            "dtype": str(self._dtype).replace("torch.", ""),
            "state_dtype": str(self._dtype).replace("torch.", ""),
            "model_dtype": model_dtype,
            "acceleration_output_dtype": str(self._dtype).replace("torch.", ""),
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
        diagnostics.update(self._throughput_metrics)
        return diagnostics

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
        callback: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        frame = getattr(self, "_frame", None)
        if frame is None:
            frame = TorchMoonFrame(
                None,
                device=device,
                dtype=self._dtype,
                allow_identity=True,
            )
        detect_impact = bool(getattr(self, "_detect_impact", True))
        # Defensive (mirrors `_frame`): smoke tests construct via __new__ and may
        # not set the terrain attributes; absent topo => constant-sphere freeze.
        topo = getattr(self, "_topo", None)
        impact_alt_m = float(getattr(self, "_impact_alt_m", 0.0))

        N = int(Y0.shape[0])
        dt = self._dt

        # Shared output grid contract: t[0]=0, t[-1]=duration_s, uniform.
        t_out, n_snaps, snap_interval = build_batch_output_grid(duration_s, output_dt_s)
        steps_per_snap = max(1, round(snap_interval / dt))
        dt_eff = snap_interval / steps_per_snap  # may differ slightly from dt
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

        def _rhs(t_s: float, s: Tensor) -> Tensor:
            """Evaluate [v; a] for state [N, 6]."""
            r_i = s[:, :3]                             # positions [N, 3]
            v = s[:, 3:]                               # velocities [N, 3]
            r_f = frame.inertial_to_fixed(t_s, r_i)
            a_f = model.predict_total_accel_torch(r_f)
            a_i = frame.fixed_to_inertial(t_s, a_f)
            return torch.cat([v, a_i], dim=1)

        def _rk4(t_s: float, s: Tensor, h: float) -> Tensor:
            k1 = _rhs(t_s, s)
            k2 = _rhs(t_s + 0.5 * h, s + (h * 0.5) * k1)
            k3 = _rhs(t_s + 0.5 * h, s + (h * 0.5) * k2)
            k4 = _rhs(t_s + h, s + h * k3)
            return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # ------------------------------------------------------------------
        # Print run header
        # ------------------------------------------------------------------
        deg_min = getattr(model, "degree_min", "?")
        deg_max = getattr(model, "degree_max", "?")
        runtime_kind = str(
            getattr(getattr(model, "_force_runtime", None), "runtime_model_kind", "")
            or getattr(model, "config", {}).get("runtime_model_kind", "potential_autograd")
        )
        dev_name = torch.cuda.get_device_name(device.index or 0)
        print(
            f"[BATCH][GPU-STLRPS] N={N}  device={device} ({dev_name})",
            flush=True,
        )
        print(
            f"[BATCH][GPU-STLRPS] degree_min={deg_min}  degree_max={deg_max}  "
            f"runtime_model_kind={runtime_kind}  "
            f"dt={dt_eff:.1f}s  snaps={n_snaps}  steps/snap={steps_per_snap}  "
            f"dtype={str(self._dtype).replace('torch.', '')}",
            flush=True,
        )

        # Time one batched acceleration call for the log
        _ = _rhs(0.0, state)
        torch.cuda.synchronize(device)
        _t0 = time.perf_counter()
        _ = _rhs(0.0, state)
        torch.cuda.synchronize(device)
        accel_ms = (time.perf_counter() - _t0) * 1_000.0
        print(
            f"[BATCH][GPU-STLRPS] one batched accel call: {accel_ms:.2f} ms  "
            f"state=[{N}, 6]",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Initial snapshot
        # ------------------------------------------------------------------
        Y_out[0] = state.detach().cpu().numpy().astype(np.float64)

        # t=0 surface check: samples already at/under the impact radius impact at
        # t=0 and are frozen there instead of being propagated through the body.
        r0 = torch.linalg.norm(state[:, :3], dim=1)
        hit0 = r0 <= r_impact_t
        impact_step = torch.full((N,), -1, dtype=torch.int64, device=device)
        # Interpolated crossing time / inertial position (NaN until a sample hits).
        impact_time_t = torch.full((N,), float("nan"), dtype=self._dtype, device=device)
        impact_pos_t = torch.full((N, 3), float("nan"), dtype=self._dtype, device=device)
        if detect_impact:
            impact_step = torch.where(hit0, torch.zeros_like(impact_step), impact_step)
            impact_time_t = torch.where(hit0, torch.zeros_like(impact_time_t), impact_time_t)
            impact_pos_t = torch.where(hit0.unsqueeze(1), state[:, :3], impact_pos_t)
            alive = alive & ~hit0

        t_curr = 0.0
        t_prop_start = time.perf_counter()
        active_steps_acc = torch.zeros((), dtype=torch.int64, device=device)
        global_step = 0

        # ------------------------------------------------------------------
        # Main integration loop
        # ------------------------------------------------------------------
        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                # Freeze impacted samples: only alive trajectories advance; an
                # impacted sample holds its last state (no propagation through
                # the Moon). Fixed batch shape is preserved (no compaction).
                active_steps_acc += alive.sum()
                prev_state = state
                candidate = _rk4(t_curr, state, dt_eff)
                state = torch.where(alive.unsqueeze(1), candidate, state)
                t_curr += dt_eff
                global_step += 1

                # Impact detection on GPU — only alive samples. The crossing is
                # the true line-sphere intersection over the step segment, and the
                # main propagated state is replaced by that crossing state (on the
                # impact sphere) so impacted trajectories freeze on the surface
                # instead of at the sub-surface step endpoint.
                if detect_impact:
                    if topo is not None:
                        segment_hit, alpha = terrain_segment_intersection(
                            prev_state[:, :3],
                            state[:, :3],
                            t_prev_s=t_curr - dt_eff,
                            dt_s=dt_eff,
                            frame=frame,
                            topo=topo,
                            impact_alt_m=impact_alt_m,
                        )
                    else:
                        segment_hit, alpha = line_sphere_intersection(
                            prev_state[:, :3],
                            state[:, :3],
                            r_impact_t,
                        )
                    newly_hit = alive & segment_hit
                    cross_state = prev_state + alpha.unsqueeze(1) * (state - prev_state)
                    t_cross = (float(global_step - 1) + alpha) * dt_eff
                    impact_step = torch.where(
                        newly_hit,
                        torch.full_like(impact_step, global_step),
                        impact_step,
                    )
                    impact_time_t = torch.where(newly_hit, t_cross, impact_time_t)
                    impact_pos_t = torch.where(
                        newly_hit.unsqueeze(1), cross_state[:, :3], impact_pos_t
                    )
                    # Freeze the main state at the surface crossing (position+velocity).
                    state = torch.where(newly_hit.unsqueeze(1), cross_state, state)
                    alive = alive & ~newly_hit

            Y_out[snap_idx + 1] = state.detach().cpu().numpy().astype(np.float64)

            if callback is not None:
                callback(float(snap_idx + 1) / float(n_snaps))

        # ------------------------------------------------------------------
        # Print timing summary
        # ------------------------------------------------------------------
        t_prop = time.perf_counter() - t_prop_start
        total_steps = n_snaps * steps_per_snap
        impact_step_host = impact_step.detach().cpu().numpy()
        hit_indices = np.nonzero(impact_step_host >= 0)[0]
        impact_flags[hit_indices] = 1.0
        t_impact_arr = impact_time_t.detach().cpu().numpy().astype(np.float64)
        total_raw_steps = N * total_steps
        total_active_steps = int(active_steps_acc.item())
        traj_steps_per_s = total_raw_steps / max(t_prop, 1e-9)
        self._throughput_metrics = {
            "total_raw_state_steps": int(total_raw_steps),
            "total_active_state_steps": int(total_active_steps),
            "raw_batch_state_steps_per_second": float(traj_steps_per_s),
            "active_state_steps_per_second": float(total_active_steps) / max(t_prop, 1e-9),
            "propagation_elapsed_s": float(t_prop),
            "impacted_sample_count": int(hit_indices.size),
            "impact_position_method": (
                "terrain_bisection_hybrid"
                if getattr(self, "_terrain_enabled", False)
                else "line_sphere_quadratic"
            ),
            "impact_time_resolution_s": float(dt_eff),
        }
        impact_positions = impact_pos_t.detach().cpu().numpy().astype(np.float64)
        self._last_impact_positions_inertial = impact_positions
        print(
            f"[BATCH][GPU-STLRPS] propagation complete: "
            f"{t_prop:.2f}s  {traj_steps_per_s:,.0f} trajectory-steps/s",
            flush=True,
        )

        return t_out, Y_out, impact_flags, t_impact_arr

    def last_impact_positions_inertial(self) -> np.ndarray:
        """Return fixed-step endpoint impact positions for the latest batch."""
        return np.asarray(
            getattr(self, "_last_impact_positions_inertial", np.empty((0, 3))),
            dtype=np.float64,
        )


__all__ = ["TorchBatchPropagator", "TorchSTLRPSPreflightError"]
