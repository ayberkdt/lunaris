# lunaris.batch.engine
"""
Batch / Ensemble Dispatch Engine
================================

Canonical batch ensemble orchestration.
"""

from __future__ import annotations

import inspect
import logging
import math
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lunaris.batch.provenance import (
    _active_physics_capabilities,
    _sha256_file,
    build_degraded_batch_backend_metadata,
)
from lunaris.batch.requirements import (
    _build_ephemeris_manager,
    _impact_positions_fixed,
    _need_ephemeris,
    _state_to_array,
    _topography_requested,
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
    _SummaryOnlyWriter,
    load_batch_result,
)
from lunaris.common.batch_defs import (
    BatchPropagationConfig,
    BatchPropagationResult,
    build_batch_output_grid,
    build_fixed_step_grid_metadata,
)
from lunaris.common.constants import DAY_S, MU_MOON, R_MOON
from lunaris.common.contracts.batch_archive import BATCH_ARCHIVE_SCHEMA_VERSION

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lunaris.batch.backend_policy import BatchBackendPlan

_surface_topography_requested = _topography_requested

_BACKEND_DISPLAY_NAMES = {
    "cpu_sh": "CPU",
    "cpu_st_lrps": "CPU-ST-LRPS",
    "numba_cuda_sh": "GPU-CLASSIC-SH",
    "torch_cuda_sh": "GPU-TORCH-SH",
    "torch_cpu_sh": "CPU-TORCH-SH",
    "gpu_st_lrps_potential": "GPU-ST-LRPS",
    "gpu_st_lrps_third_body": "GPU-ST-LRPS+3B",
}

_FIXED_STEP_BATCH_BACKENDS = frozenset(
    {
        "numba_cuda_sh",
        "torch_cuda_sh",
        "torch_cpu_sh",
        "gpu_st_lrps_potential",
        "gpu_st_lrps_third_body",
    }
)


def _batch_timestep_provenance(
    batch_cfg: Any,
    *,
    duration_s: float,
    output_dt_s: float,
    actual_backend: str,
    backend_diag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return manifest fields for requested vs realized batch time stepping."""

    _t_out, n_snaps, snap_interval = build_batch_output_grid(duration_s, output_dt_s)
    meta: dict[str, Any] = {
        "requested_dt_s": float(getattr(batch_cfg, "dt_s", 0.0)),
        "requested_output_dt_s": float(output_dt_s),
        "effective_output_dt_s": float(snap_interval),
        "n_output_snapshots": int(n_snaps + 1),
        "fixed_step_grid_aligned": str(actual_backend) in _FIXED_STEP_BATCH_BACKENDS,
    }
    if str(actual_backend) in _FIXED_STEP_BATCH_BACKENDS:
        meta.update(
            build_fixed_step_grid_metadata(
                duration_s,
                output_dt_s,
                float(getattr(batch_cfg, "dt_s", 0.0)),
            )
        )
    diagnostics = backend_diag or {}
    for key in (
        "requested_dt_s",
        "effective_dt_s",
        "steps_per_snapshot",
        "requested_output_dt_s",
        "effective_output_dt_s",
        "n_output_snapshots",
    ):
        if diagnostics.get(key) is not None:
            meta[key] = diagnostics[key]
    return meta


def _force_model_fidelity_provenance(
    sim_cfg: Any,
    *,
    backend_diag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return paper-facing fidelity labels for enabled non-gravity models."""

    flags = getattr(sim_cfg, "flags", None)
    diagnostics = backend_diag or {}
    meta: dict[str, Any] = {}

    if bool(getattr(flags, "enable_srp", False)):
        srp_cfg = getattr(sim_cfg, "srp", None)
        moon_eclipse = bool(getattr(srp_cfg, "enable_moon_eclipse", True))
        earth_eclipse = bool(getattr(srp_cfg, "enable_earth_eclipse", False))
        meta.update(
            {
                "srp_force_model": "cannonball_cr_area_over_mass",
                "srp_attitude_model": "none",
                "srp_moon_eclipse_enabled": moon_eclipse,
                "srp_earth_eclipse_enabled": earth_eclipse,
            }
        )
        if diagnostics.get("srp_shadow_model"):
            for key in (
                "srp_shadow_model",
                "srp_shadow_model_fidelity",
                "srp_earth_eclipse_supported",
                "srp_shadow_model_note",
            ):
                value = diagnostics.get(key)
                if value is not None and value != "":
                    meta[key] = value
        else:
            if moon_eclipse and earth_eclipse:
                shadow_model = "conical_moon_and_earth_shadow_factor"
            elif moon_eclipse:
                shadow_model = "conical_moon_shadow_factor"
            elif earth_eclipse:
                shadow_model = "conical_earth_shadow_factor"
            else:
                shadow_model = "disabled"
            meta.update(
                {
                    "srp_shadow_model": shadow_model,
                    "srp_shadow_model_fidelity": "engineering_conical_shadow",
                    "srp_earth_eclipse_supported": True,
                }
            )

    if bool(getattr(flags, "enable_relativity_1pn", False)):
        meta.update(
            {
                "relativity_model": "selected_1pn_corrections",
                "relativity_terms": [
                    "central_body_schwarzschild",
                    "external_body_schwarzschild_differential_when_ephemeris_available",
                    "de_sitter_geodetic_when_ephemeris_available",
                ],
                "relativity_excluded_terms": [
                    "full_eih_n_body",
                    "lense_thirring_frame_dragging",
                    "j2_relativistic_coupling",
                    "clock_time_dilation_model",
                ],
            }
        )

    if bool(getattr(flags, "enable_albedo", False)):
        albedo_cfg = getattr(sim_cfg, "albedo", None)
        albedo_model = str(getattr(albedo_cfg, "albedo_model", "lambert_facets") or "lambert_facets")
        eclipse_enabled = bool(getattr(albedo_cfg, "enable_eclipse", True))
        meta.update(
            {
                "albedo_radiation_model": albedo_model,
                "albedo_model_fidelity": "engineering_approximation",
                "albedo_eclipse_model": (
                    "moon_center_global_earth_shadow_factor"
                    if eclipse_enabled
                    else "disabled"
                ),
                "albedo_eclipse_fidelity": (
                    "global_moon_center_proxy_not_per_facet"
                    if eclipse_enabled
                    else "disabled"
                ),
            }
        )

    if bool(getattr(flags, "enable_thermal", False)):
        thermal_cfg = getattr(sim_cfg, "thermal", None)
        thermal_mode = str(
            getattr(thermal_cfg, "thermal_mode", "equilibrium_temperature")
            or "equilibrium_temperature"
        )
        eclipse_enabled = bool(getattr(thermal_cfg, "enable_eclipse", True))
        eclipse_model = "disabled"
        eclipse_fidelity = "disabled"
        if eclipse_enabled and thermal_mode == "equilibrium_temperature":
            eclipse_model = "moon_center_global_earth_shadow_factor_on_solar_input"
            eclipse_fidelity = "global_moon_center_proxy_not_per_facet"
        elif eclipse_enabled:
            eclipse_model = "configured_but_only_equilibrium_temperature_uses_eclipse"
            eclipse_fidelity = "not_applied_to_prescribed_temperature_modes"
        meta.update(
            {
                "thermal_ir_radiation_model": "lambert_facets",
                "thermal_ir_temperature_mode": thermal_mode,
                "thermal_ir_model_fidelity": "engineering_approximation",
                "thermal_ir_eclipse_model": eclipse_model,
                "thermal_ir_eclipse_fidelity": eclipse_fidelity,
            }
        )

    return meta


def _st_lrps_kind_mismatch(expected_kind: Any, actual_kind: Any) -> str | None:
    """Return the preflight error message when the artifact kind is unacceptable.

    ``expected_kind`` is what the backend policy resolved (from the request /
    config.json); ``actual_kind`` is what the loaded artifact actually declares.
    Rules:

    - expected empty: nothing to enforce (``None``).
    - expected ``potential_autograd`` with a kind-less artifact: allowed (the
      legacy loader only ever builds scalar-potential models).
    - both declared and different: fail.
    """
    expected = str(expected_kind or "").strip()
    actual = str(actual_kind or "").strip()
    if not expected:
        return None
    if not actual:
        return None
    if actual != expected:
        return (
            "GPU ST-LRPS artifact kind mismatch: backend policy expects "
            f"{expected!r}, loaded runtime is {actual!r}."
        )
    return None


class BatchPropagationEngine:
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
       e. Return ``BatchPropagationResult``.

    Parameters
    ----------
    sim_cfg : SimConfig
        Full simulation configuration (physics flags, gravity, ephemeris, ...).
    batch_cfg : BatchPropagationConfig
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
        batch_cfg: BatchPropagationConfig,
        dynamics_engine: Any = None,        # core.dynamics.DynamicsEngine
        surface_provider: Any = None,
        topo_grid: Any = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._sim_cfg = sim_cfg
        self._cfg      = batch_cfg
        self._cb      = progress_callback
        self._surface_provider = surface_provider
        self._topo_grid = topo_grid
        self._backend_note = ""
        self._backend_plan: BatchBackendPlan | None = None
        # R12: "sphere" once a terrain impact request was downgraded to a sphere
        # in research mode; None when terrain was honored or never requested.
        self._terrain_fallback: str | None = None
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

        The batch path intentionally reuses the same gravity / ephemeris bootstrap
        policy as the single-run path so users do not hit "works in Run, breaks
        in batch/ensemble" divergences.
        """
        from lunaris.core.dynamics import DynamicsEngine

        cfg = self._sim_cfg
        batch_backend = str(getattr(self._cfg, "batch_backend", "auto") or "auto")
        backend_forces_classic_sh = batch_backend in {"cpu_sh", "numba_cuda_sh", "torch_cuda_sh", "torch_cpu_sh"}
        backend_forces_st_lrps = batch_backend in {
            "gpu_st_lrps_potential",
            "gpu_st_lrps_third_body",
        }
        grav_model = None
        ephem_manager = None
        use_st_lrps_gravity = False
        surface_provider = self._surface_provider
        topo_requested = _topography_requested(surface_provider, self._topo_grid)

        if bool(cfg.flags.enable_sh):
            try:
                use_st_lrps_gravity = (
                    backend_forces_st_lrps
                    or (
                        not backend_forces_classic_sh
                        and bool(getattr(cfg.gravity, "uses_st_lrps", False))
                    )
                )
                if use_st_lrps_gravity:
                    from lunaris.surrogate.runtime import SurrogateGravityModel

                    # Prioritize the batch-specific ST-LRPS run directory if provided.
                    st_lrps_dir = self._cfg.st_lrps_model_dir or cfg.gravity.st_lrps_model_dir

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

                    # The classic-SH batch paths (numba_cuda_sh /
                    # torch_cuda_sh / torch_cpu_sh, or use_gpu auto) evaluate SH
                    # up to batch_cfg.sh_degree. Load coefficients to at
                    # least that degree (clamped to the file's own max by the
                    # loader) so a high sh_degree is not rejected by the
                    # propagator preflight merely because the mission's nominal
                    # degree is lower. Pure-CPU runs keep the mission degree
                    # unchanged so their physics is not silently altered.
                    batch_sh_degree = int(getattr(self._cfg, "sh_degree", 0) or 0)
                    numba_cuda_sh_path_requested = (
                        batch_backend in {"numba_cuda_sh", "torch_cuda_sh", "torch_cpu_sh"}
                        or (batch_backend == "auto" and bool(getattr(self._cfg, "use_gpu", False)))
                    )
                    if numba_cuda_sh_path_requested and batch_sh_degree > 0:
                        requested_degree = (
                            batch_sh_degree
                            if requested_degree is None
                            else max(requested_degree, batch_sh_degree)
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
        if not bool(getattr(self._cfg, "impact_surface_terrain_enabled", False)):
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
        ``batch.backend_policy.resolve_batch_backend_policy`` so the routing
        logic is testable in isolation without constructing a full engine.
        """
        from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy
        from lunaris.core.batch_propagator import CPUBatchPropagator

        plan = resolve_batch_backend_policy(self._cfg, self._sim_cfg)
        self._backend_plan = plan

        # Terrain-aware impact freeze payload (None unless requested + available).
        # Shared across every batch backend so they agree on the surface.
        topo_payload = self._resolve_topo_payload()

        # R12: terrain impact requested but no usable topography payload would
        # silently degrade the impact freeze to a constant sphere. Paper-safe /
        # strict runs hard-fail; research mode warns and records the fallback so
        # it is never silent.
        if (
            bool(getattr(self._cfg, "impact_surface_terrain_enabled", False))
            and topo_payload is None
        ):
            terrain_msg = (
                "impact_surface_mode='terrain' was requested but no usable "
                "topography payload is available (no surface provider or topo grid "
                "with elevation data); the impact freeze would fall back to a "
                "constant sphere."
            )
            if self._fallback_forbidden():
                raise RuntimeError(
                    terrain_msg
                    + " Paper-safe/strict mode forbids this silent simplification: "
                    "provide a topography grid/provider or set "
                    "impact_surface_mode='sphere' explicitly."
                )
            warnings.warn(
                terrain_msg + " Falling back to a constant sphere (research mode).",
                RuntimeWarning,
                stacklevel=2,
            )
            self._terrain_fallback = "sphere"

        # Emit all warnings produced by the policy resolver
        for w in plan.warnings:
            warnings.warn(w, RuntimeWarning, stacklevel=2)
            self._backend_note = w  # keep the most recent one for the run log

        # Log the resolved plan
        plan.log_summary()

        # ----------------------------------------------------------------
        # GPU ST-LRPS path - PyTorch fixed-step RK4
        # ----------------------------------------------------------------
        if plan.final_backend == BatchBackend.GPU_ST_LRPS:
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
                logger.info(
                    f"[BATCH][GPU-STLRPS] Loading surrogate: degree_min={deg_min}  "
                    f"degree_max={deg_max}  model_dir={grav_model.model_dir}"
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
                    "batch_cfg": self._cfg,
                    "device_id": int(getattr(self._cfg, "gpu_device_id", 0)),
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
                # R03: the hybrid backend models Earth/Sun third-body on-device.
                if (
                    "third_body" in constructor_params
                    and str(getattr(plan, "actual_backend", "")) == "gpu_st_lrps_third_body"
                ):
                    from lunaris.core.backend_capabilities import FORCE_MODEL_FLAG_ATTR

                    _flags = getattr(self._sim_cfg, "flags", None)
                    _bodies = tuple(
                        name
                        for name in ("third_body_sun", "third_body_earth")
                        if bool(getattr(_flags, FORCE_MODEL_FLAG_ATTR[name], False))
                    )
                    prop_kwargs["third_body"] = _bodies
                return TorchBatchPropagator(**prop_kwargs)
            except TorchSTLRPSPreflightError:
                raise
            except Exception as exc:
                note = (
                    f"[BATCH] GPU ST-LRPS backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._handle_backend_init_failure(plan, note, exc)

        # ----------------------------------------------------------------
        # GPU classic-SH path - Numba CUDA fixed-step RK4 (torch_cuda_sh's sibling)
        # ----------------------------------------------------------------
        elif plan.final_backend == BatchBackend.GPU_CLASSIC_SH:
            try:
                from lunaris.core.batch_propagator import GPUBatchPropagator

                gpu_kwargs: dict[str, Any] = {}
                if "topo_payload" in inspect.signature(GPUBatchPropagator).parameters and topo_payload is not None:
                    gpu_kwargs["topo_payload"] = topo_payload
                return GPUBatchPropagator(
                    self._dyn,
                    self._cfg,
                    self._sim_cfg.flags,
                    **gpu_kwargs,
                )
            except Exception as exc:
                note = (
                    f"[BATCH] GPU classic-SH backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._handle_backend_init_failure(plan, note, exc)

        # ----------------------------------------------------------------
        # GPU torch classic-SH path - PyTorch fixed-step RK4 (high-degree)
        # ----------------------------------------------------------------
        elif plan.final_backend == BatchBackend.GPU_TORCH_SH:
            from lunaris.core.torch_sh_propagator import (
                TorchSHBatchPropagator,
                TorchSHPreflightError,
            )

            try:
                return TorchSHBatchPropagator(
                    self._dyn,
                    self._cfg,
                    self._sim_cfg.flags,
                    device=f"cuda:{int(getattr(self._cfg, 'gpu_device_id', 0) or 0)}",
                    topo_payload=topo_payload,
                )
            except TorchSHPreflightError:
                # Hard contract violation (degree above the coefficient file,
                # unsupported physics, missing model). Never silently fall back -
                # surface it so the requested degree is not quietly reduced.
                raise
            except Exception as exc:
                note = (
                    f"[BATCH] torch_cuda_sh backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._handle_backend_init_failure(plan, note, exc)

        # ----------------------------------------------------------------
        # Torch CPU classic-SH path - PyTorch fixed-step RK4 on CPU
        # ----------------------------------------------------------------
        elif plan.final_backend == BatchBackend.TORCH_CPU_SH:
            from lunaris.core.torch_sh_propagator import (
                TorchSHBatchPropagator,
                TorchSHPreflightError,
            )

            try:
                return TorchSHBatchPropagator(
                    self._dyn,
                    self._cfg,
                    self._sim_cfg.flags,
                    device="cpu",
                    topo_payload=topo_payload,
                )
            except TorchSHPreflightError:
                raise
            except Exception as exc:
                note = (
                    f"[BATCH] torch_cpu_sh backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._handle_backend_init_failure(plan, note, exc)

        # ----------------------------------------------------------------
        # CPU path (default / fallback)
        # ----------------------------------------------------------------
        return CPUBatchPropagator(
            self._sim_cfg,
            self._cfg,
            dynamics_template=self._dyn,
            surface_provider=self._surface_provider,
            topo_grid=self._topo_grid,
        )

    def _fallback_forbidden(self) -> bool:
        """True when a silent GPU->CPU downgrade must NOT happen.

        Delegates to ``backend_policy.fallback_forbidden`` (single source,
        R29b) — the same posture that hard-fails at backend *planning* also
        applies to *initialization* failures
        (GPU selected by the policy, but the propagator could not be built), plus
        any explicit paper-safe / strict-backend flag on the batch config.
        """
        from lunaris.batch.backend_policy import fallback_forbidden

        return fallback_forbidden(self._cfg)

    def _handle_backend_init_failure(self, plan: Any, note: str, exc: Exception) -> None:
        """Either hard-fail (strict/paper-safe) or downgrade to CPU with provenance."""
        if self._fallback_forbidden():
            raise RuntimeError(
                f"{note} However, fallback is forbidden (sh_fallback_policy='error' or "
                "paper-safe mode): refusing to silently run on CPU for a benchmark/paper run. "
                "Fix the GPU backend or relax the fallback policy explicitly."
            ) from exc
        self._backend_note = note
        warnings.warn(note, RuntimeWarning, stacklevel=3)
        # BatchBackendPlan is immutable: the downgrade produces a fresh plan and
        # replaces the stored one, so no consumer can observe a half-rewritten
        # plan (provenance stays honest, task section 13).
        self._backend_plan = plan.as_cpu_fallback(note)

    # ----------------------------------------------------------------
    # Public: run
    # ----------------------------------------------------------------

    def run(self) -> BatchPropagationResult:
        """
        Execute the full batch/ensemble propagation.

        Returns
        ----------
        BatchPropagationResult
            Ensemble trajectories, spacecraft samples, impact bookkeeping.
        """
        batch_cfg = self._cfg
        cfg = self._sim_cfg
        N   = int(batch_cfg.n_samples)

        t_wall0 = time.perf_counter()
        rng = np.random.default_rng(int(batch_cfg.seed))
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
        sampling_method = str(getattr(batch_cfg, "sampling_method", "random") or "random")
        qmc_design_note = _sobol_size_note(sampling_method, N)
        if sampling_method == "random":
            joint_standard_normals = None
        else:
            joint_standard_normals = generate_standard_normal_design(
                N,
                10,
                sampling_method,
                int(batch_cfg.seed),
            )

        Y0 = sample_initial_states(
            nominal,
            batch_cfg.state,
            N,
            rng,
            sampling_method=sampling_method,
            seed=int(batch_cfg.seed),
            standard_normal_samples=(
                None if joint_standard_normals is None else joint_standard_normals[:, :6]
            ),
        )
        sc_samples = sample_spacecraft_props(
            nominal_mass=float(cfg.spacecraft.mass_kg),
            nominal_area=float(cfg.spacecraft.area_m2),
            nominal_cd=float(cfg.spacecraft.cd),
            nominal_cr=float(cfg.spacecraft.cr),
            uncertainty=batch_cfg.spacecraft,
            n_samples=N,
            rng=rng,
            sampling_method=sampling_method,
            seed=int(batch_cfg.seed) + 1,
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
        # class) and would mislabel a CPU run as GPU. After a GPU-build failure
        # the stored plan has already been replaced by its CPU fallback (see
        # BatchBackendPlan.as_cpu_fallback), so this label stays consistent with
        # what actually executes.
        _plan_actual = str(getattr(self._backend_plan, "actual_backend", "") or "")
        backend_name = _BACKEND_DISPLAY_NAMES.get(_plan_actual, "CPU")
        backend_diag = prop.diagnostics_snapshot() if hasattr(prop, "diagnostics_snapshot") else {}
        backend_plan = getattr(self, "_backend_plan", None)
        requested_max_batch = batch_cfg.effective_max_batch()
        if hasattr(prop, "recommended_max_batch"):
            max_batch = int(prop.recommended_max_batch(requested_max_batch))
        else:
            max_batch = int(requested_max_batch)

        duration_s  = float(cfg.time.duration_s)
        output_dt_s = float(cfg.time.output_dt_s or batch_cfg.dt_s * 10)
        t_out_contract, _, _ = build_batch_output_grid(duration_s, output_dt_s)

        # R23: summary-only screening mode never materializes or archives the
        # full (T, N, 6) ensemble tensor. Each sub-batch is reduced to the
        # versioned screening summary; only the top-K full histories survive.
        summary_only = str(getattr(batch_cfg, "output_mode", "full")) == "summary_only"

        if summary_only:
            storage_mode = "summary_only"
            result_bytes = 0
            memory_limit_bytes = 0
            from lunaris.batch.summary import TopKTrajectoryBuffer, summarize_ensemble

            writer = _SummaryOnlyWriter()
            _summary_parts: list[dict[str, Any]] = []
            _topk_buffer = TopKTrajectoryBuffer(int(getattr(batch_cfg, "summary_top_k", 16)))
            _summary_mu = float(
                getattr(getattr(self._dyn, "grav", None), "GM_m3s2", 0.0) or MU_MOON
            )
            _summary_r_ref = float(
                getattr(getattr(self._dyn, "grav", None), "R_ref_m", 0.0) or R_MOON
            )
        else:
            storage_mode, result_bytes, memory_limit_bytes = _resolve_result_storage(
                batch_cfg, len(t_out_contract)
            )
            writer = _make_writer(batch_cfg, N, t_out_contract)

        logger.info(
            f"[BATCH] N={N}  backend={backend_name}  "
            f"T={duration_s / DAY_S:.2f} d  "
            f"step={batch_cfg.dt_s:.1f} s  snap={output_dt_s:.1f} s"
        )
        if self._backend_note:
            logger.info("%s", self._backend_note)
        if backend_diag:
            device_name = str(backend_diag.get("device_name", "")).strip()
            tpb = backend_diag.get("threads_per_block")
            if device_name:
                logger.info(
                    f"[BATCH] runtime device={device_name}  tpb={tpb}  "
                    f"batch_cap~{max_batch}"
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
            logger.info(
                f"[BATCH] Host-RAM cap reduced batch {max_batch} -> {host_batch_cap} "
                f"(per-batch host buffer ~{host_bytes_per_sample / 1e6:.1f} MB/sample "
                f"x T={len(t_out_contract)})."
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
        Y_all = (
            None
            if summary_only
            else _allocate_result_buffer(
                storage_mode,
                writer_buffer,
                (len(t_out_ref), N, 6),
            )
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

            logger.info(
                f"[BATCH] Batch {b_idx + 1}/{n_batches}  "
                f"samples {b_start}-{b_end - 1}"
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
                            <= float(R_MOON) + float(batch_cfg.impact_alt_km) * 1_000.0
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

            if summary_only:
                # R23: reduce the batch to the screening summary + top-K full
                # histories, then let the (T, b_n, 6) block go out of scope.
                _part = summarize_ensemble(
                    t_out_ref,
                    Y_ref,
                    imp_b,
                    t_imp_b,
                    mu_m3s2=_summary_mu,
                    r_ref_m=_summary_r_ref,
                    valid_mask=(
                        np.asarray(valid_b, dtype=np.bool_)
                        if valid_b is not None
                        else None
                    ),
                )
                _summary_parts.append(_part)
                _topk_buffer.offer_batch(
                    global_start=b_start,
                    scores=np.asarray(_part["fields"]["score"], dtype=np.float64),
                    Y_batch=Y_ref,
                    impact_flags=np.asarray(imp_b, dtype=np.float64),
                    t_impact=np.asarray(t_imp_b, dtype=np.float64),
                )
            else:
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
                from lunaris.batch.backend_policy import resolve_batch_backend_policy as _resolve
                _plan = _resolve(batch_cfg, self._sim_cfg)
            actual_sh_degree = backend_diag.get("actual_sh_degree", backend_diag.get("sh_degree"))
            if actual_sh_degree is None and _grav_model is not None:
                actual_sh_degree = getattr(_grav_model, "effective_degree_max", getattr(_grav_model, "degree", None))
            actual_backend_name = str(
                getattr(_plan, "actual_backend", _plan.final_backend.value)
            )
            _time_step_meta = _batch_timestep_provenance(
                batch_cfg,
                duration_s=duration_s,
                output_dt_s=output_dt_s,
                actual_backend=actual_backend_name,
                backend_diag=backend_diag,
            )
            _plan_meta: dict[str, Any] = {
                "requested_batch_backend": getattr(_plan, "requested_backend", "auto"),
                "actual_batch_backend": actual_backend_name,
                "batch_backend": _plan.final_backend.value,
                "backend_family": getattr(_plan, "backend_family", ""),
                "backend_implementation": backend_diag.get("backend_implementation")
                    or getattr(_plan, "backend_implementation", ""),
                "requested_use_gpu": bool(batch_cfg.use_gpu),
                "final_use_gpu": _plan.use_gpu,
                "plan_gravity_backend": _plan.gravity_backend,   # renamed: avoids collision with _st_lrps_meta["gravity_backend"]
                "requested_device": getattr(_plan, "requested_device", ""),
                "actual_device": backend_diag.get("device_name") or getattr(_plan, "actual_device", ""),
                "requested_sh_degree": getattr(_plan, "requested_sh_degree", int(batch_cfg.sh_degree)),
                "actual_sh_degree": actual_sh_degree,
                "numba_cuda_sh_max_degree": getattr(_plan, "numba_cuda_sh_max_degree", None),
                "numba_cuda_sh_supported_tiers": list(getattr(_plan, "numba_cuda_sh_supported_tiers", ())),
                "runtime_model_kind": _st_lrps_meta.get(
                    "runtime_model_kind",
                    getattr(_plan, "runtime_model_kind", None),
                ),
                "torch_cuda_available": _plan.torch_cuda_available,
                "numba_cuda_available": _plan.numba_cuda_available,
                "cuda_device_name": backend_diag.get("device_name") or getattr(_plan, "cuda_device_name", None),
                "dtype": backend_diag.get("dtype") or getattr(_plan, "dtype", "float64"),
                # Dtype provenance (R10): what was requested vs what actually ran,
                # plus whether an unsupported request was downgraded.
                "requested_dtype": getattr(_plan, "requested_dtype", "")
                    or getattr(_plan, "dtype", "float64"),
                "effective_dtype": backend_diag.get("dtype")
                    or getattr(_plan, "effective_dtype", "")
                    or getattr(_plan, "dtype", "float64"),
                "dtype_downgraded": bool(getattr(_plan, "dtype_downgraded", False)),
                # R03 provenance: on-device third-body modeling + the backend's
                # statically unsupported force models (capability registry).
                "third_body_backend": backend_diag.get("third_body_backend")
                    or getattr(_plan, "third_body_backend", ""),
                "third_body_mu_source": backend_diag.get("third_body_mu_source"),
                "mu_sun_m3s2": backend_diag.get("mu_sun_m3s2"),
                "mu_earth_m3s2": backend_diag.get("mu_earth_m3s2"),
                "unsupported_forces": list(getattr(_plan, "unsupported_forces", ())),
                "state_dtype": backend_diag.get("state_dtype")
                    or backend_diag.get("dtype")
                    or getattr(_plan, "dtype", "float64"),
                "model_dtype": backend_diag.get("model_dtype"),
                "acceleration_output_dtype": backend_diag.get("acceleration_output_dtype"),
                "frame_mode": backend_diag.get("frame_mode", "unknown"),
                # R12: records a research-mode terrain->sphere downgrade so the
                # impact surface used is never ambiguous. None when terrain was
                # honored or never requested.
                "terrain_fallback": getattr(self, "_terrain_fallback", None),
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
            _plan_meta.update(_time_step_meta)
            _force_fidelity_meta = _force_model_fidelity_provenance(
                self._sim_cfg,
                backend_diag=backend_diag,
            )
            if _force_fidelity_meta:
                _plan_meta.update(_force_fidelity_meta)
                _plan_meta["force_model_fidelity"] = dict(_force_fidelity_meta)
        except Exception as exc:
            # Do not fabricate a successful provenance record after a completed
            # propagation. The fallback preserves required schema-v2 fields,
            # records the failure, and uses only observed diagnostics or the
            # resolved policy plan for the actual backend.
            logger.warning(
                "Batch backend provenance is degraded; archive records the "
                "available runtime facts instead of guessing.",
                exc_info=exc,
            )
            _plan_meta = build_degraded_batch_backend_metadata(
                requested_backend=getattr(batch_cfg, "batch_backend", "auto"),
                backend_plan=_plan,
                backend_diagnostics=backend_diag,
                requested_sh_degree=int(batch_cfg.sh_degree),
                error=exc,
            )
            _plan_meta.update(
                _batch_timestep_provenance(
                    batch_cfg,
                    duration_s=duration_s,
                    output_dt_s=output_dt_s,
                    actual_backend=_plan_meta["actual_batch_backend"],
                    backend_diag=backend_diag,
                )
            )

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
            # R29b-justified: supplementary kernel-source fingerprint (frozen /
            # zipapp installs have no source file). Mandatory identity hashes
            # (gravity file, ST-LRPS artifact) are recorded unconditionally above.
            pass

        try:
            writer.write_metadata(
                archive_schema_version=BATCH_ARCHIVE_SCHEMA_VERSION,
                n_samples=N,
                seed=int(batch_cfg.seed),
                sampling_method=sampling_method,
                sampling_note=qmc_design_note,
                duration_s=duration_s,
                output_dt_s=output_dt_s,
                requested_backend="GPU" if bool(batch_cfg.use_gpu) else "CPU",
                sh_degree=int(batch_cfg.sh_degree),
                backend=backend_name,
                backend_note=self._backend_note,
                backend_diagnostics=backend_diag,
                result_storage_mode=storage_mode,
                estimated_result_bytes=result_bytes,
                detect_impact=bool(batch_cfg.impact_detection_enabled),
                compute_impact_statistics=bool(batch_cfg.impact_statistics_enabled),
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
        logger.info(
            f"[BATCH] Done. Wall={t_wall:.1f}s  "
            f"impacts={n_hit}/{n_valid} "
            f"({100.0 * impact_fraction:.1f}%)"
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
        n_failed = int(np.sum(valid_all < 0.5))

        if summary_only:
            from lunaris.batch.summary import merge_summaries

            merged_summary = merge_summaries(_summary_parts)
            sel = np.asarray(_topk_buffer.selected_indices, dtype=np.int64)
            return BatchPropagationResult(
                t=t_out_ref,
                Y=_topk_buffer.stacked_trajectories(len(t_out_ref)),
                sc_samples=(
                    sc_samples[sel] if sel.size else np.empty((0, 4), dtype=np.float64)
                ),
                impact_mask=impact_all[sel],
                t_impact=t_impact_all[sel],
                valid_mask=valid_all[sel],
                impact_position_inertial_m=impact_position_inertial[sel],
                impact_position_fixed_m=impact_position_fixed[sel],
                archive_path=None,
                diagnostics={
                    "wall_time_s": float(t_wall),
                    "n_samples": N,
                    "sampling_method": sampling_method,
                    "sampling_note": qmc_design_note,
                    "n_valid_samples": n_valid,
                    "n_failed_samples": n_failed,
                    "n_impacts": n_hit,
                    "impact_fraction": impact_fraction,
                    "backend": backend_name,
                    "backend_note": self._backend_note,
                    "backend_diagnostics": backend_diag,
                    "requested_batch_backend": _plan_meta.get("requested_batch_backend"),
                    "actual_batch_backend": _plan_meta.get("actual_batch_backend"),
                    "requested_sh_degree": _plan_meta.get("requested_sh_degree"),
                    "actual_sh_degree": _plan_meta.get("actual_sh_degree"),
                    "runtime_model_kind": _plan_meta.get("runtime_model_kind"),
                    "third_body_mu_source": _plan_meta.get("third_body_mu_source"),
                    "mu_sun_m3s2": _plan_meta.get("mu_sun_m3s2"),
                    "mu_earth_m3s2": _plan_meta.get("mu_earth_m3s2"),
                    "requested_dt_s": _plan_meta.get("requested_dt_s"),
                    "effective_dt_s": _plan_meta.get("effective_dt_s"),
                    "steps_per_snapshot": _plan_meta.get("steps_per_snapshot"),
                    "requested_output_dt_s": _plan_meta.get("requested_output_dt_s"),
                    "effective_output_dt_s": _plan_meta.get("effective_output_dt_s"),
                    "fixed_step_grid_aligned": _plan_meta.get("fixed_step_grid_aligned"),
                    "srp_shadow_model": _plan_meta.get("srp_shadow_model"),
                    "srp_shadow_model_fidelity": _plan_meta.get("srp_shadow_model_fidelity"),
                    "srp_earth_eclipse_supported": _plan_meta.get("srp_earth_eclipse_supported"),
                    "force_model_fidelity": _plan_meta.get("force_model_fidelity"),
                    "fallback_reason": _plan_meta.get("fallback_reason"),
                    "selection_reason": _plan_meta.get("selection_reason"),
                    # R23 summary-mode payload: full-N screening summary +
                    # top-K bookkeeping. Y above holds ONLY the top-K histories.
                    "output_mode": "summary_only",
                    "batch_summary": merged_summary,
                    "summary_top_k": int(getattr(batch_cfg, "summary_top_k", 16)),
                    "summary_selected_indices": sel.tolist(),
                    "summary_selected_scores": list(_topk_buffer.scores),
                    "result_storage_mode": "summary_only",
                },
            )

        if storage_mode == "disk":
            Y_result: Any = HDF5TrajectoryView(batch_cfg.output_path_resolved)
        else:
            if Y_all is None:
                raise RuntimeError("Eager batch result buffer was not initialized.")
            Y_result = Y_all

        return BatchPropagationResult(
            t=t_out_ref,
            Y=Y_result,
            sc_samples=sc_samples,
            impact_mask=impact_all,
            t_impact=t_impact_all,
            valid_mask=valid_all,
            impact_position_inertial_m=impact_position_inertial,
            impact_position_fixed_m=impact_position_fixed,
            archive_path=str(batch_cfg.output_path_resolved),
            diagnostics={
                "wall_time_s": float(t_wall),
                "n_samples": N,
                "sampling_method": sampling_method,
                "sampling_note": qmc_design_note,
                "n_valid_samples": n_valid,
                "n_failed_samples": n_failed,
                "n_impacts": n_hit,
                "impact_fraction": impact_fraction,
                "impact_detection_enabled": bool(batch_cfg.impact_detection_enabled),
                "impact_statistics_enabled": bool(batch_cfg.impact_statistics_enabled),
                "impact_frame_available": bool(getattr(self._dyn, "ephem", None) is not None),
                "backend": backend_name,
                "backend_note": self._backend_note,
                "output_path": str(batch_cfg.output_path_resolved),
                "backend_diagnostics": backend_diag,
                # Throughput metrics from the batched propagator (if available)
                "raw_batch_state_steps_per_second": backend_diag.get("raw_batch_state_steps_per_second"),
                "active_state_steps_per_second": backend_diag.get("active_state_steps_per_second"),
                "requested_batch_backend": _plan_meta.get("requested_batch_backend"),
                "actual_batch_backend": _plan_meta.get("actual_batch_backend"),
                "requested_sh_degree": _plan_meta.get("requested_sh_degree"),
                "actual_sh_degree": _plan_meta.get("actual_sh_degree"),
                "runtime_model_kind": _plan_meta.get("runtime_model_kind"),
                "third_body_mu_source": _plan_meta.get("third_body_mu_source"),
                "mu_sun_m3s2": _plan_meta.get("mu_sun_m3s2"),
                "mu_earth_m3s2": _plan_meta.get("mu_earth_m3s2"),
                "requested_dt_s": _plan_meta.get("requested_dt_s"),
                "effective_dt_s": _plan_meta.get("effective_dt_s"),
                "steps_per_snapshot": _plan_meta.get("steps_per_snapshot"),
                "requested_output_dt_s": _plan_meta.get("requested_output_dt_s"),
                "effective_output_dt_s": _plan_meta.get("effective_output_dt_s"),
                "fixed_step_grid_aligned": _plan_meta.get("fixed_step_grid_aligned"),
                "srp_shadow_model": _plan_meta.get("srp_shadow_model"),
                "srp_shadow_model_fidelity": _plan_meta.get("srp_shadow_model_fidelity"),
                "srp_earth_eclipse_supported": _plan_meta.get("srp_earth_eclipse_supported"),
                "force_model_fidelity": _plan_meta.get("force_model_fidelity"),
                "fallback_reason": _plan_meta.get("fallback_reason"),
                "selection_reason": _plan_meta.get("selection_reason"),
                "result_storage_mode": storage_mode,
            },
        )


def batch_entry() -> int:
    """Console-script entry point for batch/ensemble propagation."""
    from lunaris.cli.batch_runner import main as _batch_main

    return int(_batch_main())


__all__ = [
    "BatchPropagationEngine",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
    "HDF5TrajectoryView",
    "load_batch_result",
    "batch_entry",
]
