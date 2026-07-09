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
- Base path: ST-LRPS lunar gravity only.
- Hybrid path: ST-LRPS lunar gravity plus analytic Sun/Earth third-body
  acceleration.
- SRP, albedo, tides, Earth J2, and relativity still force a CPU fallback
  (detected by ``batch.backend_policy.resolve_batch_backend_policy``).
- Fixed step size; no adaptive step control.
- State dtype follows ``BatchPropagationConfig.torch_dtype`` (float32 by default for
  throughput, float64 when explicitly requested).

Performance notes
-----------------
Each RK4 step launches 4 batched neural forward passes + 4 autograd calls on
the CUDA device.  No per-step CPU round-trips occur once the run is started.
Snapshots are copied to host only at the ``output_dt_s`` cadence.

Timing metrics are emitted through this module's logger at run start and end.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from lunaris.common.batch_defs import build_batch_output_grid
from lunaris.common.constants import MU_EARTH, MU_SUN, R_MOON
from lunaris.common.frame_policy import (
    FRAME_MODE_IDENTITY_DIAGNOSTIC,
    FRAME_MODE_MOON_FIXED_EPHEMERIS,
)

logger = logging.getLogger(__name__)
from lunaris.core.batched_fixed_step import (
    query_device_memory,
    resolve_vram_aware_chunk_size,
    rhs_batch,
    run_batched_fixed_step,
)
from lunaris.core.torch_frame import (
    TorchFrameError,
    TorchMoonFrame,
    topo_payload_to_torch,
)

if TYPE_CHECKING:
    # Annotation-only alias. The runtime ``torch`` handle is the per-instance
    # ``self._torch`` (optional dependency), which shadows a module import; a
    # dedicated name keeps the type annotations resolvable.
    from torch import Tensor


class TorchSTLRPSPreflightError(RuntimeError):
    """Hard ST-LRPS runtime contract violation that must not fall back silently."""


def _is_torch_cuda_oom(torch_mod: Any, exc: BaseException) -> bool:
    """Return True for typed or message-based torch CUDA OOM exceptions."""

    oom_type = getattr(getattr(torch_mod, "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class _STLRPSAccelerationProvider:
    """R07 acceleration provider: frame rotation + ST-LRPS total acceleration.

    Same composition as the pre-R07 in-loop ``_rhs``: rotate the inertial
    position into the Moon-fixed frame, evaluate the surrogate total
    acceleration there, rotate the result back to inertial.
    """

    def __init__(self, model: Any, frame: Any) -> None:
        self._model = model
        self._frame = frame

    def acceleration(self, t_s: float, s: Tensor) -> Tensor:
        r_i = s[:, :3]
        r_f = self._frame.inertial_to_fixed(t_s, r_i)
        a_f = self._model.predict_total_accel_torch(r_f)
        return self._frame.fixed_to_inertial(t_s, a_f)


class _STLRPSThirdBodyAccelerationProvider(_STLRPSAccelerationProvider):
    """R03 hybrid provider: ST-LRPS lunar gravity + analytic Earth/Sun third-body.

    ``a_total = a_STLRPS(lunar, via frame rotation) + a_3rd(Sun) + a_3rd(Earth)``
    with the third-body terms evaluated directly in the Moon-centred inertial
    frame from Catmull-Rom-interpolated ephemeris positions — the same
    formulation (cancellation-free Battin F(q)) and tables as the CPU path.
    """

    def __init__(
        self,
        model: Any,
        frame: Any,
        *,
        ephem_tables: Any,
        use_sun: bool,
        use_earth: bool,
        mu_sun: float,
        mu_earth: float,
    ) -> None:
        super().__init__(model, frame)
        self._tables = ephem_tables
        self._use_sun = bool(use_sun)
        self._use_earth = bool(use_earth)
        self._mu_sun = float(mu_sun)
        self._mu_earth = float(mu_earth)

    def acceleration(self, t_s: float, s: Tensor) -> Tensor:
        from lunaris.core.torch_third_body import third_body_accel_batch

        a_i = super().acceleration(t_s, s)
        r_i = s[:, :3]
        if self._use_sun:
            a_i = a_i + third_body_accel_batch(
                r_i, self._tables.sun_position(t_s).to(dtype=r_i.dtype), self._mu_sun
            )
        if self._use_earth:
            a_i = a_i + third_body_accel_batch(
                r_i, self._tables.earth_position(t_s).to(dtype=r_i.dtype), self._mu_earth
            )
        return a_i


def _resolve_third_body_tables(
    third_body: tuple[str, ...],
    ephem: Any,
    *,
    device: Any,
    dtype: Any,
) -> Any:
    """Build device-resident ephemeris tables for the ST-LRPS third-body hybrid."""

    bodies = tuple(str(body) for body in (third_body or ()) if str(body))
    if not bodies:
        return None

    supported = {"third_body_sun", "third_body_earth"}
    unknown = tuple(sorted(set(bodies) - supported))
    if unknown:
        raise TorchSTLRPSPreflightError(
            "gpu_st_lrps_third_body received unsupported third-body selector(s): "
            + ", ".join(unknown)
        )

    if ephem is None:
        raise TorchSTLRPSPreflightError(
            "gpu_st_lrps_third_body requires an ephemeris (Sun/Earth position "
            "tables), but none is attached. Load an ephemeris or disable the "
            "third-body perturbations."
        )

    try:
        from lunaris.core.dynamics import extract_ephem_tables_strict
        from lunaris.core.torch_third_body import TorchEphemerisTables

        from lunaris.core.dynamics.preparation import _provider_get, _provider_has

        dt_s, sun_tab, earth_tab, _q_tab = extract_ephem_tables_strict(ephem)
        provider = ephem.get_data_provider()
        has_mu_sun = _provider_has(provider, "mu_sun_m3s2")
        has_mu_earth = _provider_has(provider, "mu_earth_m3s2")
        if has_mu_sun and has_mu_earth:
            mu_source = "ephemeris_provider"
        elif has_mu_sun or has_mu_earth:
            mu_source = "mixed_ephemeris_provider_and_module_constants"
        else:
            mu_source = "module_constants_fallback"
        return TorchEphemerisTables(
            dt_s=dt_s,
            r_sun_tab_m=sun_tab,
            r_earth_tab_m=earth_tab,
            device=device,
            dtype=dtype,
            need_sun="third_body_sun" in bodies,
            need_earth="third_body_earth" in bodies,
            mu_sun_m3s2=float(_provider_get(provider, "mu_sun_m3s2", MU_SUN)),
            mu_earth_m3s2=float(_provider_get(provider, "mu_earth_m3s2", MU_EARTH)),
            mu_source=mu_source,
        )
    except TorchSTLRPSPreflightError:
        raise
    except Exception as exc:
        raise TorchSTLRPSPreflightError(
            f"gpu_st_lrps_third_body ephemeris preflight failed: {exc}"
        ) from exc


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
        third_body: tuple[str, ...] = (),
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

        # Move surrogate model (weights + scaling tensors) to CUDA with the
        # requested torch dtype; paper-safe diagnostics must reflect the dtype
        # actually used by the neural runtime, not only the state tensor dtype.
        self._model = surrogate_model
        self._model.to_device(self._device, dtype=self._dtype)
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

        # R03: analytic vectorized third-body gravity (hybrid backend).
        self._third_body = tuple(str(b) for b in (third_body or ()))
        self._ephem_tables = _resolve_third_body_tables(
            self._third_body, ephem, device=self._device, dtype=self._dtype
        )

        # R06: VRAM-aware chunking. The shared loop merges chunk results into
        # one output and halves the chunk on OOM (recorded in the metrics).
        requested_chunk = int(getattr(batch_cfg, "torch_sh_chunk_size", 0) or 0)
        self._bytes_per_sample = self._estimate_bytes_per_sample()
        self._free_mem_bytes, self._total_mem_bytes = query_device_memory(torch, self._device)
        self._chunk_size, self._chunk_provenance = resolve_vram_aware_chunk_size(
            bytes_per_sample=self._bytes_per_sample,
            free_bytes=self._free_mem_bytes,
            total_bytes=self._total_mem_bytes,
            requested=requested_chunk,
        )
        self._throughput_metrics: dict[str, Any] = {}

    def _estimate_bytes_per_sample(self) -> int:
        """Per-sample device-memory estimate for the ST-LRPS chunk preflight.

        Assumptions (auditable, not magic):
          * dominant cost is the MLP activations kept for the autograd pass:
            ``~hidden * (depth + 1)`` elements/sample per forward, doubled for
            the backward workspace;
          * RK4 keeps state + 4 stage derivatives + temporaries -> ``~60`` elems;
          * inflated 8x for transient tensors inside the evaluator (same safety
            posture as the classic-SH estimate).
        """
        cfg = getattr(self._model, "config", {}) or {}
        hidden = int(cfg.get("hidden", 256) or 256)
        depth = int(cfg.get("depth", 4) or 4)
        dtype_bytes = 8 if self._dtype == self._torch.float64 else 4
        activation_elems = 2 * hidden * (depth + 1) + 60
        return int(activation_elems * dtype_bytes * 8.0)

    # ------------------------------------------------------------------
    # Public interface (matches GPUBatchPropagator / CPUBatchPropagator)
    # ------------------------------------------------------------------

    def diagnostics_snapshot(self) -> dict:
        """Return a diagnostics dict for the progress log."""
        torch = self._torch
        dev = self._device
        runtime = getattr(self._model, "_force_runtime", None)
        dtype_diag = {}
        if hasattr(self._model, "dtype_diagnostics"):
            dtype_diag = dict(self._model.dtype_diagnostics(requested_dtype=self._dtype))
        model_dtype = "float32"
        try:
            model_obj = getattr(runtime, "model", None)
            if model_obj is not None:
                model_dtype = str(next(model_obj.parameters()).dtype).replace("torch.", "")
        except Exception:
            # R29b-justified: dtype probe for the progress-log diagnostics only;
            # authoritative dtype provenance comes from the backend plan
            # (requested_dtype/effective_dtype), not this display string.
            pass
        third_body = tuple(getattr(self, "_third_body", ()) or ())
        diagnostics = {
            "backend": "GPU-ST-LRPS",
            "device_name": torch.cuda.get_device_name(dev.index or 0),
            "torch_cuda_version": str(torch.version.cuda or "unknown"),
            "threads_per_block": "managed by PyTorch",
            # R03 provenance: gravity source + on-device third-body modeling.
            "lunar_gravity_backend": "st_lrps",
            "third_body_backend": "analytic_vectorized" if third_body else "",
            "third_body_bodies": list(third_body),
            "third_body_mu_source": str(
                getattr(getattr(self, "_ephem_tables", None), "mu_source", "")
            ),
            "mu_sun_m3s2": getattr(getattr(self, "_ephem_tables", None), "mu_sun_m3s2", None),
            "mu_earth_m3s2": getattr(getattr(self, "_ephem_tables", None), "mu_earth_m3s2", None),
            "runtime_model_kind": str(
                getattr(getattr(self._model, "_force_runtime", None), "runtime_model_kind", "")
                or getattr(self._model, "config", {}).get("runtime_model_kind", "potential_autograd")
            ),
            "dtype": str(self._dtype).replace("torch.", ""),
            "requested_dtype": dtype_diag.get("requested_dtype") or str(self._dtype).replace("torch.", ""),
            "effective_dtype": dtype_diag.get("effective_dtype") or model_dtype,
            "dtype_downgraded": bool(dtype_diag.get("dtype_downgraded", False)),
            "state_dtype": str(self._dtype).replace("torch.", ""),
            "model_dtype": dtype_diag.get("model_dtype") or model_dtype,
            "scaler_dtype": dtype_diag.get("scaler_dtype"),
            "force_runtime_scaler_dtype": dtype_diag.get("force_runtime_scaler_dtype"),
            "acceleration_output_dtype": str(self._dtype).replace("torch.", ""),
            "frame_mode": (
                FRAME_MODE_MOON_FIXED_EPHEMERIS
                if self._frame.uses_rotation
                else FRAME_MODE_IDENTITY_DIAGNOSTIC
            ),
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

        # Grid preview for the run header only; the shared loop (R07) rebuilds
        # the same grid (single contract: build_batch_output_grid).
        _t_preview, n_snaps, snap_interval = build_batch_output_grid(duration_s, output_dt_s)
        steps_per_snap = max(1, round(snap_interval / dt))
        dt_eff = snap_interval / steps_per_snap  # may differ slightly from dt

        third_body = tuple(getattr(self, "_third_body", ()) or ())
        if third_body:
            provider: _STLRPSAccelerationProvider = _STLRPSThirdBodyAccelerationProvider(
                model,
                frame,
                ephem_tables=self._ephem_tables,
                use_sun="third_body_sun" in third_body,
                use_earth="third_body_earth" in third_body,
                mu_sun=float(getattr(self._ephem_tables, "mu_sun_m3s2", MU_SUN)),
                mu_earth=float(getattr(self._ephem_tables, "mu_earth_m3s2", MU_EARTH)),
            )
        else:
            provider = _STLRPSAccelerationProvider(model, frame)

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
        logger.info(f"[BATCH][GPU-STLRPS] N={N}  device={device} ({dev_name})")
        logger.info(
            f"[BATCH][GPU-STLRPS] degree_min={deg_min}  degree_max={deg_max}  "
            f"runtime_model_kind={runtime_kind}  "
            f"dt={dt_eff:.1f}s  snaps={n_snaps}  steps/snap={steps_per_snap}  "
            f"dtype={str(self._dtype).replace('torch.', '')}"
        )

        # Time one batched acceleration call for the log. Keep this diagnostic
        # sample no larger than the effective chunk so a large ensemble does not
        # OOM before the VRAM-aware chunked propagation loop can recover.
        warmup_chunk = int(getattr(self, "_chunk_size", 0) or N)
        warmup_n = max(1, min(N, max(1, warmup_chunk)))
        warmup_metrics: dict[str, Any] = {
            "accel_warmup_sample_count": int(warmup_n),
            "accel_warmup_full_batch": bool(warmup_n == N),
        }
        try:
            state = torch.as_tensor(Y0[:warmup_n], dtype=self._dtype, device=device)
            _ = rhs_batch(torch, provider, 0.0, state)
            torch.cuda.synchronize(device)
            _t0 = time.perf_counter()
            _ = rhs_batch(torch, provider, 0.0, state)
            torch.cuda.synchronize(device)
            accel_ms = (time.perf_counter() - _t0) * 1_000.0
            warmup_metrics["accel_warmup_ms"] = float(accel_ms)
            logger.info(
                f"[BATCH][GPU-STLRPS] one batched accel call: {accel_ms:.2f} ms  "
                f"warmup_state=[{warmup_n}, 6] full_N={N}"
            )
        except Exception as exc:
            if not _is_torch_cuda_oom(torch, exc):
                raise
            warmup_metrics["accel_warmup_skipped_reason"] = "cuda_oom"
            try:
                torch.cuda.empty_cache()
            except Exception:
                # Best-effort after a diagnostic-only OOM; propagation below
                # still gets the shared loop's chunk-halving recovery.
                pass
            logger.warning(
                "[BATCH][GPU-STLRPS] skipped warmup/timing acceleration due to "
                "CUDA OOM at warmup_n=%d; continuing with chunked propagation.",
                warmup_n,
            )

        # ------------------------------------------------------------------
        # Shared batched fixed-step RK4 + impact loop (R07)
        # ------------------------------------------------------------------
        result = run_batched_fixed_step(
            torch_mod=torch,
            device=device,
            dtype=self._dtype,
            provider=provider,
            frame=frame,
            Y0=Y0,
            duration_s=duration_s,
            output_dt_s=output_dt_s,
            dt_s=dt,
            impact_r_m=float(self._impact_r),
            detect_impact=detect_impact,
            topo=topo,
            impact_alt_m=impact_alt_m,
            # R06: VRAM-aware chunk (falls back to a single chunk for the
            # __new__-constructed smoke-test path that skips __init__).
            chunk_size=getattr(self, "_chunk_size", None),
            callback=callback,
            callback_granularity="snapshot",
        )

        t_prop = float(result.metrics["propagation_elapsed_s"])
        traj_steps_per_s = float(result.metrics["raw_batch_state_steps_per_second"])
        self._throughput_metrics = {
            "total_raw_state_steps": int(result.metrics["total_raw_state_steps"]),
            "total_active_state_steps": int(result.metrics["total_active_state_steps"]),
            "raw_batch_state_steps_per_second": traj_steps_per_s,
            "active_state_steps_per_second": float(result.metrics["active_state_steps_per_second"]),
            "propagation_elapsed_s": t_prop,
            "impacted_sample_count": int(result.metrics["impacted_sample_count"]),
            "impact_position_method": (
                "terrain_bisection_hybrid"
                if getattr(self, "_terrain_enabled", False)
                else "line_sphere_quadratic"
            ),
            "impact_time_resolution_s": float(dt_eff),
            "requested_dt_s": float(result.metrics["requested_dt_s"]),
            "effective_dt_s": float(result.metrics["effective_dt_s"]),
            "steps_per_snapshot": int(result.metrics["steps_per_snapshot"]),
            "requested_output_dt_s": float(result.metrics["requested_output_dt_s"]),
            "effective_output_dt_s": float(result.metrics["effective_output_dt_s"]),
            "n_output_snapshots": int(result.metrics["n_output_snapshots"]),
            # R06 chunk provenance from the shared loop (OOM recoveries included).
            "chunk_size_requested": int(result.metrics["chunk_size_requested"]),
            "chunk_size_effective": int(result.metrics["chunk_size_effective"]),
            "oom_recoveries": result.metrics["oom_recoveries"],
            **warmup_metrics,
        }
        self._last_impact_positions_inertial = result.impact_positions_inertial
        logger.info(
            f"[BATCH][GPU-STLRPS] propagation complete: "
            f"{t_prop:.2f}s  {traj_steps_per_s:,.0f} trajectory-steps/s"
        )

        return result.t_out, result.Y_out, result.impact_flags, result.t_impact

    def last_impact_positions_inertial(self) -> np.ndarray:
        """Return fixed-step endpoint impact positions for the latest batch."""
        return np.asarray(
            getattr(self, "_last_impact_positions_inertial", np.empty((0, 3))),
            dtype=np.float64,
        )


__all__ = ["TorchBatchPropagator", "TorchSTLRPSPreflightError"]
