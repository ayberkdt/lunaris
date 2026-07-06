# lunaris.batch.engine
"""
Batch / Ensemble Dispatch Engine
================================

Canonical batch ensemble orchestration. Compatibility imports remain in
``lunaris.core.monte_carlo_engine`` for the historical public path.
"""

from __future__ import annotations

import inspect
import math
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lunaris.batch.provenance import _active_physics_capabilities, _sha256_file
from lunaris.batch.requirements import (
    _build_ephemeris_manager,
    _impact_positions_fixed,
    _need_ephemeris,
    _state_to_array,
    _surface_topography_requested,
)
from lunaris.batch.sampling import (
    _sobol_size_note,
    generate_standard_normal_design,
    sample_initial_states,
    sample_spacecraft_props,
)
from lunaris.batch.storage import (
    HDF5TrajectoryView,
    _allocate_result_buffer,
    _make_writer,
    _resolve_result_storage,
    load_mc_result,
)
from lunaris.common.batch_defs import (
    MCRunResult,
    MonteCarloConfig,
    build_mc_output_grid,
)
from lunaris.common.constants import DAY_S, MU_MOON, R_MOON

if TYPE_CHECKING:
    from lunaris.core.mc_backend_policy import MCBackendPlan

_BACKEND_DISPLAY_NAMES = {
    "cpu_sh": "CPU",
    "cpu_st_lrps": "CPU-ST-LRPS",
    "numba_cuda_sh": "GPU-CLASSIC-SH",
    "torch_cuda_sh": "GPU-TORCH-SH",
    "torch_cpu_sh": "CPU-TORCH-SH",
    "gpu_st_lrps_potential": "GPU-ST-LRPS",
    "gpu_st_lrps_direct": "GPU-ST-LRPS",
}


def _st_lrps_kind_mismatch(expected_kind: Any, actual_kind: Any) -> str | None:
    """Return the preflight error message when the artifact kind is unacceptable.

    ``expected_kind`` is what the backend policy resolved (from the request /
    config.json); ``actual_kind`` is what the loaded artifact actually declares.
    Rules:

    - expected empty: nothing to enforce (``None``).
    - expected ``force_direct`` but the artifact declares nothing: **fail
      closed**. An explicit direct-force request must be provable from the
      artifact — legacy kind-less artifacts are potential-only by construction,
      so assuming ``force_direct`` would run the wrong physics.
    - expected ``potential_autograd`` with a kind-less artifact: allowed (the
      legacy loader only ever builds scalar-potential models).
    - both declared and different: fail.
    """
    expected = str(expected_kind or "").strip()
    actual = str(actual_kind or "").strip()
    if not expected:
        return None
    if not actual:
        if expected == "force_direct":
            return (
                "GPU ST-LRPS artifact kind mismatch: backend policy expects "
                "'force_direct', but the loaded artifact does not declare "
                "runtime_model_kind. An explicit direct-force request must be "
                "provable from the artifact (legacy kind-less artifacts are "
                "potential-only); refusing to assume."
            )
        return None
    if actual != expected:
        return (
            "GPU ST-LRPS artifact kind mismatch: backend policy expects "
            f"{expected!r}, loaded runtime is {actual!r}."
        )
    return None


class MonteCarloEngine:
    """
    Orchestrates a full batch/ensemble injection-dispersion propagation run.

    Workflow
    ----------
    1. ``__init__``: validate configs, select backend (GPU / CPU).
    2. ``run()``:
       a. Draw N initial state samples + spacecraft property samples.
       b. Open output writer.
       c. For each sub-batch (VRAM-bounded):
          - Transfer arrays to device (GPU) or dispatch workers (CPU).
          - Iterate over time steps; write snapshots to disk.
       d. Aggregate impact statistics.
       e. Return ``BatchPropagationResult`` / legacy ``MCRunResult``.

    Parameters
    ----------
    sim_cfg : SimConfig
        Full simulation configuration (physics flags, gravity, ephemeris, ...).
    mc_cfg : MonteCarloConfig
        Batch/ensemble parameters (N, uncertainties, GPU flags, output format).
    dynamics_engine : optional pre-built DynamicsEngine
        If None, the engine builds one from ``sim_cfg``.
    progress_callback : optional ``f(payload: dict)``
        Receives structured progress payloads containing stage, percent,
        done/total scenario counts, and ETA hints suitable for UI progress bars.
    """

    def __init__(
        self,
        sim_cfg: Any,                       # config.SimConfig
        mc_cfg: MonteCarloConfig,
        dynamics_engine: Any = None,        # core.dynamics.DynamicsEngine
        surface_provider: Any = None,
        topo_grid: Any = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._sim_cfg = sim_cfg
        self._mc      = mc_cfg
        self._cb      = progress_callback
        self._surface_provider = surface_provider
        self._topo_grid = topo_grid
        self._backend_note = ""
        self._backend_plan: MCBackendPlan | None = None
        if self._topo_grid is None and self._surface_provider is not None and hasattr(self._surface_provider, "grids"):
            try:
                self._topo_grid = self._surface_provider.grids().topo
            except Exception:
                self._topo_grid = None
        self._dyn     = dynamics_engine or self._build_dynamics()

    def _publish_progress(
        self,
        *,
        stage: str,
        stage_fraction: float,
        total_samples: int,
        done_samples: float,
        elapsed_s: float,
        backend: str,
        batch_index: int | None = None,
        batch_count: int | None = None,
        detail: str = "",
    ) -> None:
        """
        Emit a structured progress payload for the UI layer.

        Progress is modeled in phases rather than as a single opaque counter:
        sampling, propagating, and writing each occupy a weighted slice of the
        full run.  This keeps the progress bar visually honest and avoids the
        "stuck at 99%" anti-pattern.
        """

        if self._cb is None:
            return

        stage_offsets = {
            "sampling": (0.00, 0.05),
            "propagating": (0.05, 0.90),
            "writing": (0.95, 0.05),
            "finalizing": (0.995, 0.005),
        }
        offset, weight = stage_offsets.get(stage, (0.0, 1.0))
        stage_fraction = max(0.0, min(1.0, float(stage_fraction)))
        overall_fraction = max(0.0, min(1.0, offset + weight * stage_fraction))
        eta_s: float | None = None
        if overall_fraction > 1.0e-6:
            eta_s = max(0.0, float(elapsed_s) * (1.0 - overall_fraction) / overall_fraction)

        payload = {
            "stage": str(stage),
            "percent": round(overall_fraction * 100.0, 3),
            "fraction": overall_fraction,
            "done_samples": float(done_samples),
            "total_samples": int(total_samples),
            "elapsed_s": round(float(elapsed_s), 3),
            "eta_s": (round(float(eta_s), 3) if eta_s is not None else None),
            "backend": str(backend),
            "detail": str(detail),
        }
        if batch_index is not None:
            payload["batch_index"] = int(batch_index)
        if batch_count is not None:
            payload["batch_count"] = int(batch_count)

        self._cb(payload)

    # ----------------------------------------------------------------
    # Internal: build dynamics engine from SimConfig
    # ----------------------------------------------------------------

    def _build_dynamics(self) -> Any:
        """
        Lazily build a DynamicsEngine from the stored SimConfig.

        The MC path intentionally reuses the same gravity / ephemeris bootstrap
        policy as the single-run path so users do not hit "works in Run, breaks
        in batch/ensemble" divergences.
        """
        from lunaris.core.dynamics import DynamicsEngine

        cfg = self._sim_cfg
        mc_backend = str(getattr(self._mc, "mc_backend", "auto") or "auto")
        mc_forces_classic_sh = mc_backend in {"cpu_sh", "gpu_sh", "numba_cuda_sh", "torch_cuda_sh", "torch_cpu_sh"}
        mc_forces_st_lrps = mc_backend in {"gpu_st_lrps_potential", "gpu_st_lrps_direct"}
        grav_model = None
        ephem_manager = None
        use_st_lrps_gravity = False
        surface_provider = self._surface_provider
        topo_requested = _surface_topography_requested(surface_provider, self._topo_grid)

        if bool(cfg.flags.enable_sh):
            try:
                use_st_lrps_gravity = (
                    mc_forces_st_lrps
                    or (
                        not mc_forces_classic_sh
                        and bool(getattr(cfg.gravity, "uses_st_lrps", False))
                    )
                )
                if use_st_lrps_gravity:
                    from lunaris.surrogate.runtime import SurrogateGravityModel

                    # Prioritize the MC-specific ST-LRPS run directory if provided.
                    st_lrps_dir = self._mc.st_lrps_model_dir or cfg.gravity.st_lrps_model_dir

                    from lunaris.common.batch_defs import validate_st_lrps_model_dir
                    valid_dir = validate_st_lrps_model_dir(st_lrps_dir)

                    grav_model = SurrogateGravityModel.from_model_dir(
                        str(valid_dir),
                        mu_override=float(MU_MOON),
                        r_ref_override=float(R_MOON),
                        device_preference="cpu",
                    )
                else:
                    from lunaris.physics.spherical_harmonics import GravityModel

                    requested_degree = int(cfg.gravity.degree) if cfg.gravity.degree is not None else None

                    # The classic-SH GPU batch paths (numba_cuda_sh /
                    # torch_cuda_sh / torch_cpu_sh, or use_gpu auto) evaluate SH
                    # up to mc.gpu_sh_degree. Load coefficients to at least that
                    # degree (clamped to the file's own max by the loader) so a
                    # high gpu_sh_degree is not rejected by the propagator
                    # preflight merely because the mission's nominal degree is
                    # lower. Pure-CPU runs keep the mission degree unchanged so
                    # their physics is not silently altered.
                    mc_gpu_degree = int(getattr(self._mc, "gpu_sh_degree", 0) or 0)
                    gpu_sh_path_requested = (
                        mc_backend in {"gpu_sh", "numba_cuda_sh", "torch_cuda_sh", "torch_cpu_sh"}
                        or (mc_backend == "auto" and bool(getattr(self._mc, "use_gpu", False)))
                    )
                    if gpu_sh_path_requested and mc_gpu_degree > 0:
                        requested_degree = (
                            mc_gpu_degree
                            if requested_degree is None
                            else max(requested_degree, mc_gpu_degree)
                        )

                    # GravityModel already exposes the full dynamics gravity
                    # contract (degree_max, R_ref_m, GM_m3s2, Cnm ... ws).
                    grav_model = GravityModel.from_file(
                        path=str(cfg.gravity.file_path),
                        requested_degree=requested_degree,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Batch ensemble bootstrap failed: Could not load gravity model.\n"
                    f"ST-LRPS mode: {getattr(cfg.gravity, 'uses_st_lrps', False)}\n"
                    f"Error: {exc}"
                ) from exc

        if _need_ephemeris(cfg, topo_requested=topo_requested):
            try:
                ephem_manager = _build_ephemeris_manager(cfg)
            except Exception as exc:
                raise RuntimeError(
                    f"Batch ensemble bootstrap failed: Could not load ephemeris.\n"
                    f"Error: {exc}"
                ) from exc

        earth_j2 = getattr(cfg, "earth_j2", None)
        srp = getattr(cfg, "srp", None)
        albedo = getattr(cfg, "albedo", None)
        thermal = getattr(cfg, "thermal", None)
        solid_tides = getattr(cfg, "solid_tides", None)

        return DynamicsEngine(
            sc_props=cfg.spacecraft,
            flags=cfg.flags,
            gravity_model=grav_model,
            gravity_adaptive=(
                None if use_st_lrps_gravity
                else getattr(cfg.gravity, "adaptive", None)
            ),
            ephem_manager=ephem_manager,
            surface_provider=surface_provider,
            earth_j2=earth_j2,
            srp=srp,
            albedo=albedo,
            thermal=thermal,
            solid_tides=solid_tides,
            allow_identity_rotation=(ephem_manager is None),
        )

    # ----------------------------------------------------------------
    # Internal: select and initialise backend
    # ----------------------------------------------------------------

    def _resolve_topo_payload(self) -> dict[str, Any] | None:
        """Topography payload for terrain-aware impact freeze, or ``None``.

        Built only when the config requests ``impact_surface_mode='terrain'`` AND
        a topography grid/provider is available; otherwise the batch backends keep
        the constant-sphere impact freeze (zero behaviour change). The payload is
        the same POD contract the CPU ground-truth event consumes, so all backends
        share one terrain definition.
        """
        if not bool(getattr(self._mc, "impact_surface_terrain_enabled", False)):
            return None

        prov = self._surface_provider
        if prov is not None and hasattr(prov, "topo_payload"):
            try:
                payload = prov.topo_payload()
            except Exception:
                payload = None
            if payload is not None and payload.get("dn", None) is not None:
                return payload

        if self._topo_grid is not None:
            from lunaris.loaders.io_surface import _grid_topo_payload
            return _grid_topo_payload(self._topo_grid)
        return None

    def _build_propagator(self) -> Any:
        """
        Instantiate the appropriate batch propagator using the backend policy.

        Backend selection is fully delegated to
        ``core.mc_backend_policy.resolve_mc_backend_policy`` so the routing
        logic is testable in isolation without constructing a full engine.
        """
        from lunaris.core.mc_backend_policy import MCBackend, resolve_mc_backend_policy
        from lunaris.core.mc_propagator import CPUBatchPropagator

        plan = resolve_mc_backend_policy(self._mc, self._sim_cfg)
        self._backend_plan = plan

        # Terrain-aware impact freeze payload (None unless requested + available).
        # Shared across every batch backend so they agree on the surface.
        topo_payload = self._resolve_topo_payload()

        # Emit all warnings produced by the policy resolver
        for w in plan.warnings:
            warnings.warn(w, RuntimeWarning, stacklevel=2)
            self._backend_note = w  # keep the most recent one for the run log

        # Log the resolved plan
        plan.log_summary()

        # ----------------------------------------------------------------
        # GPU ST-LRPS path - PyTorch fixed-step RK4
        # ----------------------------------------------------------------
        if plan.final_backend == MCBackend.GPU_ST_LRPS:
            try:
                from lunaris.core.torch_batch_propagator import (
                    TorchBatchPropagator,
                    TorchSTLRPSPreflightError,
                )

                grav_model = getattr(self._dyn, "grav", None)
                if grav_model is None or getattr(grav_model, "model_kind", None) != "st_lrps":
                    raise RuntimeError(
                        "GPU ST-LRPS backend selected but no SurrogateGravityModel "
                        "is attached to the dynamics engine."
                    )
                deg_min = getattr(grav_model, "degree_min", "?")
                deg_max = getattr(grav_model, "degree_max", "?")
                print(
                    f"[MC][GPU-STLRPS] Loading surrogate: degree_min={deg_min}  "
                    f"degree_max={deg_max}  model_dir={grav_model.model_dir}",
                    flush=True,
                )
                actual_runtime_kind = str(
                    getattr(getattr(grav_model, "_force_runtime", None), "runtime_model_kind", "")
                    or getattr(grav_model, "config", {}).get("runtime_model_kind", "")
                ).strip()
                expected_runtime_kind = str(getattr(plan, "runtime_model_kind", "") or "").strip()
                _kind_error = _st_lrps_kind_mismatch(expected_runtime_kind, actual_runtime_kind)
                if _kind_error:
                    raise TorchSTLRPSPreflightError(_kind_error)
                prop_kwargs: dict[str, Any] = {
                    "surrogate_model": grav_model,
                    "mc_cfg": self._mc,
                    "device_id": int(getattr(self._mc, "gpu_device_id", 0)),
                }
                constructor_params = inspect.signature(TorchBatchPropagator).parameters
                if "ephem" in constructor_params:
                    prop_kwargs["ephem"] = getattr(self._dyn, "ephem", None)
                if "allow_identity_rotation" in constructor_params:
                    prop_kwargs["allow_identity_rotation"] = bool(
                        getattr(self._dyn, "allow_identity_rotation", False)
                    )
                if "topo_payload" in constructor_params and topo_payload is not None:
                    prop_kwargs["topo_payload"] = topo_payload
                return TorchBatchPropagator(**prop_kwargs)
            except TorchSTLRPSPreflightError:
                raise
            except Exception as exc:
                note = (
                    f"[MC] GPU ST-LRPS backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._handle_backend_init_failure(plan, note, exc)

        # ----------------------------------------------------------------
        # GPU classic-SH path - Numba CUDA fixed-step RK4 (torch_cuda_sh's sibling)
        # ----------------------------------------------------------------
        elif plan.final_backend == MCBackend.GPU_CLASSIC_SH:
            try:
                from lunaris.core.mc_propagator import GPUBatchPropagator

                gpu_kwargs: dict[str, Any] = {}
                if "topo_payload" in inspect.signature(GPUBatchPropagator).parameters and topo_payload is not None:
                    gpu_kwargs["topo_payload"] = topo_payload
                return GPUBatchPropagator(
                    self._dyn,
                    self._mc,
                    self._sim_cfg.flags,
                    **gpu_kwargs,
                )
            except Exception as exc:
                note = (
                    f"[MC] GPU classic-SH backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._handle_backend_init_failure(plan, note, exc)

        # ----------------------------------------------------------------
        # GPU torch classic-SH path - PyTorch fixed-step RK4 (high-degree)
        # ----------------------------------------------------------------
        elif plan.final_backend == MCBackend.GPU_TORCH_SH:
            from lunaris.core.torch_sh_propagator import (
                TorchSHBatchPropagator,
                TorchSHPreflightError,
            )

            try:
                return TorchSHBatchPropagator(
                    self._dyn,
                    self._mc,
                    self._sim_cfg.flags,
                    device=f"cuda:{int(getattr(self._mc, 'gpu_device_id', 0) or 0)}",
                    topo_payload=topo_payload,
                )
            except TorchSHPreflightError:
                # Hard contract violation (degree above the coefficient file,
                # unsupported physics, missing model). Never silently fall back -
                # surface it so the requested degree is not quietly reduced.
                raise
            except Exception as exc:
                note = (
                    f"[MC] torch_cuda_sh backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._handle_backend_init_failure(plan, note, exc)

        # ----------------------------------------------------------------
        # Torch CPU classic-SH path - PyTorch fixed-step RK4 on CPU
        # ----------------------------------------------------------------
        elif plan.final_backend == MCBackend.TORCH_CPU_SH:
            from lunaris.core.torch_sh_propagator import (
                TorchSHBatchPropagator,
                TorchSHPreflightError,
            )

            try:
                return TorchSHBatchPropagator(
                    self._dyn,
                    self._mc,
                    self._sim_cfg.flags,
                    device="cpu",
                    topo_payload=topo_payload,
                )
            except TorchSHPreflightError:
                raise
            except Exception as exc:
                note = (
                    f"[MC] torch_cpu_sh backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._handle_backend_init_failure(plan, note, exc)

        # ----------------------------------------------------------------
        # CPU path (default / fallback)
        # ----------------------------------------------------------------
        return CPUBatchPropagator(
            self._sim_cfg,
            self._mc,
            dynamics_template=self._dyn,
            surface_provider=self._surface_provider,
            topo_grid=self._topo_grid,
        )

    def _fallback_forbidden(self) -> bool:
        """True when a silent GPU->CPU downgrade must NOT happen.

        A benchmark / paper-evidence run that asks for a GPU backend and then
        quietly executes on CPU produces a misleading speed/throughput table. The
        existing ``gpu_sh_fallback_policy='error'`` already hard-fails at backend
        *planning*; this extends the same intent to *initialization* failures
        (GPU selected by the policy, but the propagator could not be built), plus
        any explicit paper-safe / strict-backend flag on the MC config.
        """
        policy = str(getattr(self._mc, "gpu_sh_fallback_policy", "compatible_gpu") or "").strip().lower()
        if policy == "error":
            return True
        return any(
            bool(getattr(self._mc, attr, False))
            for attr in ("paper_safe", "strict_backend", "benchmark_mode")
        )

    def _handle_backend_init_failure(self, plan: Any, note: str, exc: Exception) -> None:
        """Either hard-fail (strict/paper-safe) or downgrade to CPU with provenance."""
        if self._fallback_forbidden():
            raise RuntimeError(
                f"{note} However, fallback is forbidden (gpu_sh_fallback_policy='error' or "
                "paper-safe mode): refusing to silently run on CPU for a benchmark/paper run. "
                "Fix the GPU backend or relax the fallback policy explicitly."
            ) from exc
        self._backend_note = note
        warnings.warn(note, RuntimeWarning, stacklevel=3)
        self._downgrade_plan_to_cpu(plan, note)

    @staticmethod
    def _downgrade_plan_to_cpu(plan: Any, reason: str) -> None:
        """Rewrite a backend plan to CPU after a GPU propagator failed to build.

        Keeps provenance honest: a run that actually executes on CPU must not be
        labeled with a GPU backend, device, or integrator (task section 13).
        """
        from lunaris.core.mc_backend_policy import MCBackend

        plan.final_backend = MCBackend.CPU
        plan.use_gpu = False
        plan.actual_backend = "cpu_st_lrps" if plan.gravity_backend == "st_lrps" else "cpu_sh"
        plan.actual_sh_degree = None
        plan.actual_device = "cpu"
        plan.cuda_device_name = None
        plan.dtype = "float64"
        plan.integrator = "adaptive (DOP853)"
        plan.fallback_applied = True
        plan.fallback_reason = reason
        # Refresh family/implementation labels for the new actual backend.
        try:
            from lunaris.core.backend_capabilities import get_capabilities

            caps = get_capabilities(plan.actual_backend)
            plan.backend_family = caps.family
            plan.backend_implementation = caps.implementation
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Public: run
    # ----------------------------------------------------------------

    def run(self) -> MCRunResult:
        """
        Execute the full batch/ensemble propagation.

        Returns
        ----------
        MCRunResult
            Ensemble trajectories, spacecraft samples, impact bookkeeping.
        """
        mc  = self._mc
        cfg = self._sim_cfg
        N   = int(mc.n_samples)

        t_wall0 = time.perf_counter()
        rng = np.random.default_rng(int(mc.seed))
        self._publish_progress(
            stage="sampling",
            stage_fraction=0.0,
            total_samples=N,
            done_samples=0.0,
            elapsed_s=0.0,
            backend="pending",
            detail="Preparing ensemble sample set",
        )

        # ----------------------------------------------------------------
        # 1) Generate samples
        # ----------------------------------------------------------------
        nominal = _state_to_array(cfg.initial_state)   # (6,)
        sampling_method = str(getattr(mc, "sampling_method", "random") or "random")
        qmc_design_note = _sobol_size_note(sampling_method, N)
        if sampling_method == "random":
            joint_standard_normals = None
        else:
            joint_standard_normals = generate_standard_normal_design(
                N,
                10,
                sampling_method,
                int(mc.seed),
            )

        Y0 = sample_initial_states(
            nominal,
            mc.state,
            N,
            rng,
            sampling_method=sampling_method,
            seed=int(mc.seed),
            standard_normal_samples=(
                None if joint_standard_normals is None else joint_standard_normals[:, :6]
            ),
        )
        sc_samples = sample_spacecraft_props(
            nominal_mass=float(cfg.spacecraft.mass_kg),
            nominal_area=float(cfg.spacecraft.area_m2),
            nominal_cd=float(cfg.spacecraft.cd),
            nominal_cr=float(cfg.spacecraft.cr),
            uncertainty=mc.spacecraft,
            n_samples=N,
            rng=rng,
            sampling_method=sampling_method,
            seed=int(mc.seed) + 1,
            standard_normal_samples=(
                None if joint_standard_normals is None else joint_standard_normals[:, 6:10]
            ),
        )

        masses = sc_samples[:, 0]
        areas  = sc_samples[:, 1]
        cds    = sc_samples[:, 2]
        crs    = sc_samples[:, 3]
        self._publish_progress(
            stage="sampling",
            stage_fraction=1.0,
            total_samples=N,
            done_samples=0.0,
            elapsed_s=time.perf_counter() - t_wall0,
            backend="pending",
            detail=f"Samples generated ({sampling_method})",
        )

        # ----------------------------------------------------------------
        # 2) Propagator + output storage contract
        # ----------------------------------------------------------------
        prop   = self._build_propagator()

        # Fail-fast: validate gravity model contract before entering the sample
        # loop.  Without this check the CPU propagator catches the same missing-
        # attribute error N times and prints N identical "Sample i failed" lines.
        if hasattr(prop, "validate_gravity_assets"):
            prop.validate_gravity_assets()

        # Human-readable backend label derived from the *resolved plan's* actual
        # backend - never from the propagator class name. The class name cannot
        # distinguish torch_cuda_sh from torch_cpu_sh (same TorchSHBatchPropagator
        # class) and would mislabel a CPU run as GPU. After a GPU-build failure the
        # plan has already been downgraded to CPU (see _downgrade_plan_to_cpu), so
        # this label stays consistent with what actually executes.
        _plan_actual = str(getattr(self._backend_plan, "actual_backend", "") or "")
        backend_name = _BACKEND_DISPLAY_NAMES.get(_plan_actual, "CPU")
        backend_diag = prop.diagnostics_snapshot() if hasattr(prop, "diagnostics_snapshot") else {}
        backend_plan = getattr(self, "_backend_plan", None)
        requested_max_batch = mc.effective_max_batch()
        if hasattr(prop, "recommended_max_batch"):
            max_batch = int(prop.recommended_max_batch(requested_max_batch))
        else:
            max_batch = int(requested_max_batch)

        duration_s  = float(cfg.time.duration_s)
        output_dt_s = float(cfg.time.output_dt_s or mc.dt_s * 10)
        t_out_contract, _, _ = build_mc_output_grid(duration_s, output_dt_s)
        storage_mode, result_bytes, memory_limit_bytes = _resolve_result_storage(
            mc,
            len(t_out_contract),
        )
        writer = _make_writer(mc, N, t_out_contract)

        print(
            f"[MC] N={N}  backend={backend_name}  "
            f"T={duration_s / DAY_S:.2f} d  "
            f"step={mc.dt_s:.1f} s  snap={output_dt_s:.1f} s",
            flush=True,
        )
        if self._backend_note:
            print(self._backend_note, flush=True)
        if backend_diag:
            device_name = str(backend_diag.get("device_name", "")).strip()
            tpb = backend_diag.get("threads_per_block")
            if device_name:
                print(
                    f"[MC] runtime device={device_name}  tpb={tpb}  "
                    f"batch_cap~{max_batch}",
                    flush=True,
                )

        # ----------------------------------------------------------------
        # 3) Sub-batch loop (VRAM + host-RAM budget)
        # ----------------------------------------------------------------
        # The per-batch host buffer is (T, b_n, 6) float64. A batch that fits in
        # VRAM can still exhaust host RAM for long / high-cadence runs because the
        # VRAM cap only accounts for a single state vector per sample, not the full
        # snapshot history kept on the host. Bound the sample batch by the host
        # memory budget as well so max_batch never blows out resident memory. The
        # budget already folds in the available-RAM safety factor (see
        # _resolve_result_storage), so a busy host tightens the batch cap too.
        host_bytes_per_sample = len(t_out_contract) * 6 * np.dtype(np.float64).itemsize
        host_batch_cap = max(
            1, int(memory_limit_bytes / max(1, host_bytes_per_sample))
        )
        if host_batch_cap < max_batch:
            print(
                f"[MC] Host-RAM cap reduced batch {max_batch} -> {host_batch_cap} "
                f"(per-batch host buffer ~{host_bytes_per_sample / 1e6:.1f} MB/sample "
                f"x T={len(t_out_contract)}).",
                flush=True,
            )
            max_batch = host_batch_cap

        n_batches = math.ceil(N / max_batch)
        self._publish_progress(
            stage="propagating",
            stage_fraction=0.0,
            total_samples=N,
            done_samples=0.0,
            elapsed_s=time.perf_counter() - t_wall0,
            backend=backend_name,
            batch_count=n_batches if n_batches > 0 else None,
            detail="Propagation starting",
        )

        # Result arrays stay eager only in memory mode. Disk mode writes each
        # sample batch directly into the final HDF5 trajectory dataset.
        t_out_ref = t_out_contract
        writer_buffer = getattr(writer, "memory_buffer", None)
        Y_all = _allocate_result_buffer(
            storage_mode,
            writer_buffer,
            (len(t_out_ref), N, 6),
        )
        impact_all   = np.zeros(N, dtype=np.float64)
        t_impact_all = np.full(N, np.nan, dtype=np.float64)
        valid_all = np.zeros(N, dtype=np.float64)
        impact_position_inertial = np.full((N, 3), np.nan, dtype=np.float64)
        impact_position_fixed = np.full((N, 3), np.nan, dtype=np.float64)

        # Throughput accumulators across engine sub-batches. Aggregated as
        # total_state_steps / total_propagation_time (NOT an average of per-batch
        # rates) so the recorded diagnostics match the work actually done.
        _agg_raw_steps = 0
        _agg_active_steps = 0
        _agg_elapsed_s = 0.0

        for b_idx in range(n_batches):
            b_start = b_idx * max_batch
            b_end   = min(N, b_start + max_batch)
            b_n     = b_end - b_start

            print(
                f"[MC] Batch {b_idx + 1}/{n_batches}  "
                f"samples {b_start}-{b_end - 1}",
                flush=True,
            )

            # Loop variables are bound as defaults: the callback is invoked
            # synchronously within this iteration's propagate() call, but binding
            # makes that explicit and silences B023 (late-binding closure).
            def _batch_progress(
                local_fraction: float,
                _b_start: int = b_start,
                _b_n: int = b_n,
                _b_idx: int = b_idx,
            ) -> None:
                effective_done = float(_b_start) + float(_b_n) * max(0.0, min(1.0, float(local_fraction)))
                self._publish_progress(
                    stage="propagating",
                    stage_fraction=(effective_done / max(N, 1)),
                    total_samples=N,
                    done_samples=effective_done,
                    elapsed_s=time.perf_counter() - t_wall0,
                    backend=backend_name,
                    batch_index=_b_idx + 1,
                    batch_count=n_batches,
                    detail=f"Batch {_b_idx + 1}/{n_batches}",
                )

            try:
                t_b, Y_b, imp_b, t_imp_b = prop.propagate(
                    Y0[b_start:b_end],
                    masses[b_start:b_end],
                    areas[b_start:b_end],
                    cds[b_start:b_end],
                    crs[b_start:b_end],
                    duration_s=duration_s,
                    output_dt_s=output_dt_s,
                    callback=_batch_progress,
                )
            except Exception:
                writer.abort()
                raise

            # Accumulate this batch's throughput counters (only backends that
            # expose them populate these keys; others contribute nothing).
            if hasattr(prop, "diagnostics_snapshot"):
                _bd = prop.diagnostics_snapshot()
                _agg_raw_steps += int(_bd.get("total_raw_state_steps", 0) or 0)
                _agg_active_steps += int(_bd.get("total_active_state_steps", 0) or 0)
                _agg_elapsed_s += float(_bd.get("propagation_elapsed_s", 0.0) or 0.0)

            # Resample to reference grid if needed
            if len(t_b) == len(t_out_ref) and np.allclose(t_b, t_out_ref, rtol=1e-6):
                Y_ref = np.ascontiguousarray(Y_b, dtype=np.float64)
            else:
                # Linear interpolation to reference grid
                Y_ref = np.empty((len(t_out_ref), b_n, 6), dtype=np.float64)
                for j in range(b_n):
                    for c in range(6):
                        Y_ref[:, j, c] = np.interp(
                            t_out_ref, t_b, Y_b[:, j, c]
                        )

            impact_all[b_start:b_end] = imp_b
            valid_b = np.isfinite(Y_ref).all(axis=(0, 2))
            valid_all[b_start:b_end] = valid_b.astype(np.float64)

            batch_impact_positions = np.full((b_n, 3), np.nan, dtype=np.float64)
            if hasattr(prop, "last_impact_positions_inertial"):
                candidate_positions = np.asarray(
                    prop.last_impact_positions_inertial(), dtype=np.float64
                )
                if candidate_positions.shape == (b_n, 3):
                    batch_impact_positions[:] = candidate_positions
            for j in range(b_n):
                if (
                    imp_b[j] > 0.5
                    and not np.isfinite(batch_impact_positions[j]).all()
                ):
                    if np.isfinite(t_imp_b[j]):
                        hit_idx = int(np.argmin(np.abs(t_b - float(t_imp_b[j]))))
                    else:
                        radii = np.linalg.norm(Y_b[:, j, :3], axis=1)
                        hits = np.where(
                            radii
                            <= float(R_MOON) + float(mc.impact_alt_km) * 1_000.0
                        )[0]
                        hit_idx = int(hits[0]) if hits.size else len(t_b) - 1
                        t_imp_b[j] = float(t_b[hit_idx])
                    batch_impact_positions[j] = Y_b[hit_idx, j, :3]
            t_impact_all[b_start:b_end] = t_imp_b
            impact_position_inertial[b_start:b_end] = batch_impact_positions
            impact_position_fixed[b_start:b_end] = _impact_positions_fixed(
                getattr(self._dyn, "ephem", None),
                np.asarray(t_imp_b, dtype=np.float64),
                batch_impact_positions,
            )

            try:
                writer.write_sample_batch(b_start, b_end, Y_ref)
            except Exception:
                writer.abort()
                raise
            if Y_all is not None and Y_all is not writer_buffer:
                Y_all[:, b_start:b_end, :] = Y_ref

            self._publish_progress(
                stage="propagating",
                stage_fraction=(float(b_end) / max(N, 1)),
                total_samples=N,
                done_samples=float(b_end),
                elapsed_s=time.perf_counter() - t_wall0,
                backend=backend_name,
                batch_index=b_idx + 1,
                batch_count=n_batches,
                detail=f"Batch {b_idx + 1}/{n_batches} complete",
            )

        # ----------------------------------------------------------------
        # 3b) Refresh diagnostics AFTER propagation (throughput is only known
        #     post-run) and fold in the cross-batch aggregate. The pre-run
        #     snapshot above carried static device info but no throughput.
        # ----------------------------------------------------------------
        if hasattr(prop, "diagnostics_snapshot"):
            backend_diag = dict(prop.diagnostics_snapshot())
            if _agg_elapsed_s > 0.0:
                backend_diag["total_raw_state_steps"] = _agg_raw_steps
                backend_diag["total_active_state_steps"] = _agg_active_steps
                backend_diag["propagation_elapsed_s"] = _agg_elapsed_s
                backend_diag["raw_batch_state_steps_per_second"] = _agg_raw_steps / _agg_elapsed_s
                backend_diag["active_state_steps_per_second"] = _agg_active_steps / _agg_elapsed_s

        # ----------------------------------------------------------------
        # 4) Finalize archive metadata
        # ----------------------------------------------------------------
        # Collect ST-LRPS provenance metadata when the surrogate backend is active.
        _grav_model = getattr(self._dyn, "grav", None)
        _st_lrps_meta: dict[str, Any] = {}
        if getattr(_grav_model, "model_kind", None) == "st_lrps":
            _st_lrps_meta = {
                "gravity_backend": "st_lrps",
                "st_lrps_model_dir": str(getattr(_grav_model, "model_dir", "") or ""),
                "st_lrps_degree_min": getattr(_grav_model, "degree_min", None),
                "st_lrps_degree_max": getattr(_grav_model, "degree_max", None),
                "effective_degree_max": getattr(_grav_model, "effective_degree_max", None),
                "runtime_model_kind": str(
                    getattr(getattr(_grav_model, "_force_runtime", None), "runtime_model_kind", "")
                    or getattr(_grav_model, "config", {}).get("runtime_model_kind", "potential_autograd")
                ),
            }

        # Collect backend-plan provenance for the archive
        try:
            _plan = backend_plan
            if _plan is None:
                from lunaris.core.mc_backend_policy import resolve_mc_backend_policy as _resolve
                _plan = _resolve(mc, self._sim_cfg)
            actual_sh_degree = backend_diag.get("actual_gpu_sh_degree", backend_diag.get("gpu_sh_degree"))
            if actual_sh_degree is None and _grav_model is not None:
                actual_sh_degree = getattr(_grav_model, "effective_degree_max", getattr(_grav_model, "degree", None))
            _plan_meta: dict[str, Any] = {
                "requested_mc_backend": getattr(_plan, "requested_backend", "auto"),
                "actual_mc_backend": getattr(_plan, "actual_backend", _plan.final_backend.value),
                "mc_backend": _plan.final_backend.value,
                "backend_family": getattr(_plan, "backend_family", ""),
                "backend_implementation": backend_diag.get("backend_implementation")
                    or getattr(_plan, "backend_implementation", ""),
                "requested_use_gpu": bool(mc.use_gpu),
                "final_use_gpu": _plan.use_gpu,
                "plan_gravity_backend": _plan.gravity_backend,   # renamed: avoids collision with _st_lrps_meta["gravity_backend"]
                "requested_device": getattr(_plan, "requested_device", ""),
                "actual_device": backend_diag.get("device_name") or getattr(_plan, "actual_device", ""),
                "requested_sh_degree": getattr(_plan, "requested_sh_degree", int(mc.gpu_sh_degree)),
                "actual_sh_degree": actual_sh_degree,
                "gpu_sh_max_degree": getattr(_plan, "gpu_sh_max_degree", None),
                "gpu_sh_supported_tiers": list(getattr(_plan, "gpu_sh_supported_tiers", ())),
                "runtime_model_kind": _st_lrps_meta.get(
                    "runtime_model_kind",
                    getattr(_plan, "runtime_model_kind", None),
                ),
                "torch_cuda_available": _plan.torch_cuda_available,
                "numba_cuda_available": _plan.numba_cuda_available,
                "cuda_device_name": backend_diag.get("device_name") or getattr(_plan, "cuda_device_name", None),
                "dtype": backend_diag.get("dtype") or getattr(_plan, "dtype", "float64"),
                "state_dtype": backend_diag.get("state_dtype")
                    or backend_diag.get("dtype")
                    or getattr(_plan, "dtype", "float64"),
                "model_dtype": backend_diag.get("model_dtype"),
                "acceleration_output_dtype": backend_diag.get("acceleration_output_dtype"),
                "frame_mode": backend_diag.get("frame_mode", "unknown"),
                "integrator": backend_diag.get("integrator") or _plan.integrator,
                "batch_size": max_batch,
                "chunk_size": backend_diag.get("chunk_size", max_batch),
                "fallback_applied": bool(getattr(_plan, "fallback_applied", False)),
                "fallback_reason": (
                    getattr(_plan, "fallback_reason", "")
                    if bool(getattr(_plan, "fallback_applied", False))
                    else ""
                ),
                "selection_reason": getattr(_plan, "reason", ""),
                "physics_capabilities": _active_physics_capabilities(self._sim_cfg),
            }
        except Exception:
            # Even on the degenerate provenance path the required v2 manifest
            # fields must be present (and non-null) so the archive still loads
            # under load_mc_result(strict=True).
            _fallback_backend = str(getattr(mc, "mc_backend", "auto") or "auto")
            _plan_meta = {
                "requested_mc_backend": _fallback_backend,
                "actual_mc_backend": _fallback_backend,
                "mc_backend": _fallback_backend,
                "requested_sh_degree": int(mc.gpu_sh_degree),
            }

        # Artifact + coefficient + kernel hash provenance: a path string alone is
        # not reproducible evidence. Stamp content hashes so a reader can verify
        # exactly which weights, gravity coefficients, and GPU kernel produced
        # this archive. _sha256_file never raises (missing file -> None, dropped).
        _provenance_hashes: dict[str, Any] = {}
        if getattr(_grav_model, "model_kind", None) == "st_lrps":
            _force_runtime = getattr(_grav_model, "_force_runtime", None)
            _ckpt_path = getattr(_force_runtime, "checkpoint_path", None)
            if _ckpt_path:
                _provenance_hashes["st_lrps_checkpoint_sha256"] = _sha256_file(_ckpt_path)
            _model_dir = getattr(_grav_model, "model_dir", None)
            if _model_dir:
                _provenance_hashes["st_lrps_config_sha256"] = _sha256_file(
                    Path(_model_dir) / "config.json"
                )
            _run_manifest = getattr(_force_runtime, "run_manifest", {}) or {}
            for _key in ("checkpoint_hash", "scaler_hash", "training_config_hash"):
                _val = _run_manifest.get(_key)
                if _val:
                    _provenance_hashes[f"st_lrps_{_key}"] = _val
        _grav_file = getattr(getattr(cfg, "gravity", None), "file_path", None)
        if _grav_file:
            _provenance_hashes["sh_coefficient_sha256"] = _sha256_file(_grav_file)
        try:
            _provenance_hashes["kernel_module"] = str(getattr(type(prop), "__module__", "") or "")
            _provenance_hashes["kernel_source_sha256"] = _sha256_file(
                inspect.getsourcefile(type(prop))
            )
        except Exception:
            pass

        try:
            writer.write_metadata(
                archive_schema_version=2,
                n_samples=N,
                seed=int(mc.seed),
                sampling_method=sampling_method,
                sampling_note=qmc_design_note,
                duration_s=duration_s,
                output_dt_s=output_dt_s,
                requested_backend="GPU" if bool(mc.use_gpu) else "CPU",
                gpu_sh_degree=int(mc.gpu_sh_degree),
                backend=backend_name,
                backend_note=self._backend_note,
                backend_diagnostics=backend_diag,
                result_storage_mode=storage_mode,
                estimated_result_bytes=result_bytes,
                detect_impact=bool(mc.impact_detection_enabled),
                compute_impact_statistics=bool(mc.impact_statistics_enabled),
                impact_frame_available=bool(getattr(self._dyn, "ephem", None) is not None),
                **_provenance_hashes,
                **_st_lrps_meta,
                **_plan_meta,
            )
            writer.write_final(
                sc_samples,
                impact_all,
                t_impact_all,
                valid_all,
                impact_position_inertial,
                impact_position_fixed,
            )
            self._publish_progress(
                stage="writing",
                stage_fraction=1.0,
                total_samples=N,
                done_samples=float(N),
                elapsed_s=time.perf_counter() - t_wall0,
                backend=backend_name,
                batch_index=n_batches,
                batch_count=n_batches,
                detail="Finalizing archive",
            )
            writer.finalize()
        except Exception:
            writer.abort()
            raise

        t_wall = time.perf_counter() - t_wall0
        valid_bool = valid_all > 0.5
        n_valid = int(np.sum(valid_bool))
        n_hit = int(np.sum(valid_bool & (impact_all > 0.5)))
        impact_fraction = float(n_hit) / n_valid if n_valid else math.nan
        print(
            f"[MC] Done. Wall={t_wall:.1f}s  "
            f"impacts={n_hit}/{n_valid} "
            f"({100.0 * impact_fraction:.1f}%)",
            flush=True,
        )
        self._publish_progress(
            stage="finalizing",
            stage_fraction=1.0,
            total_samples=N,
            done_samples=float(N),
            elapsed_s=t_wall,
            backend=backend_name,
            batch_index=n_batches,
            batch_count=n_batches,
            detail="Run completed",
        )

        # ----------------------------------------------------------------
        # 5) Build result
        # ----------------------------------------------------------------
        if storage_mode == "disk":
            Y_result: Any = HDF5TrajectoryView(mc.output_path_resolved)
        else:
            if Y_all is None:
                raise RuntimeError("Eager MC result buffer was not initialized.")
            Y_result = Y_all
        n_failed = int(np.sum(valid_all < 0.5))

        return MCRunResult(
            t=t_out_ref,
            Y=Y_result,
            sc_samples=sc_samples,
            impact_mask=impact_all,
            t_impact=t_impact_all,
            valid_mask=valid_all,
            impact_position_inertial_m=impact_position_inertial,
            impact_position_fixed_m=impact_position_fixed,
            archive_path=str(mc.output_path_resolved),
            diagnostics={
                "wall_time_s": float(t_wall),
                "n_samples": N,
                "sampling_method": sampling_method,
                "sampling_note": qmc_design_note,
                "n_valid_samples": n_valid,
                "n_failed_samples": n_failed,
                "n_impacts": n_hit,
                "impact_fraction": impact_fraction,
                "impact_detection_enabled": bool(mc.impact_detection_enabled),
                "impact_statistics_enabled": bool(mc.impact_statistics_enabled),
                "impact_frame_available": bool(getattr(self._dyn, "ephem", None) is not None),
                "backend": backend_name,
                "backend_note": self._backend_note,
                "output_path": str(mc.output_path_resolved),
                "backend_diagnostics": backend_diag,
                # Throughput metrics from the batched propagator (if available)
                "raw_batch_state_steps_per_second": backend_diag.get("raw_batch_state_steps_per_second"),
                "active_state_steps_per_second": backend_diag.get("active_state_steps_per_second"),
                "requested_mc_backend": _plan_meta.get("requested_mc_backend"),
                "actual_mc_backend": _plan_meta.get("actual_mc_backend"),
                "requested_sh_degree": _plan_meta.get("requested_sh_degree"),
                "actual_sh_degree": _plan_meta.get("actual_sh_degree"),
                "runtime_model_kind": _plan_meta.get("runtime_model_kind"),
                "fallback_reason": _plan_meta.get("fallback_reason"),
                "selection_reason": _plan_meta.get("selection_reason"),
                "result_storage_mode": storage_mode,
            },
        )


def mc_entry() -> int:
    """Historical console-script alias for batch/ensemble propagation."""
    from lunaris.cli.batch_runner import main as _mc_main

    return int(_mc_main())


def batch_entry() -> int:
    """Console-script alias for the batch propagation terminology."""
    return mc_entry()


BatchPropagationEngine = MonteCarloEngine


__all__ = [
    "BatchPropagationEngine",
    "MonteCarloEngine",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
    "HDF5TrajectoryView",
    "load_mc_result",
    "mc_entry",
    "batch_entry",
]
