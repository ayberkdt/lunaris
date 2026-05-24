# LUNAR_SIMULATION/core/monte_carlo_engine.py
# -*- coding: utf-8 -*-
"""
Monte Carlo Dispatch Engine
============================

This module is the single entry point for running Monte Carlo simulations.
It handles:

1. Sample generation  — multivariate Gaussian draws for the 6D state and
   optional spacecraft property perturbations.
2. Backend dispatch   — routes to GPUBatchPropagator (CUDA) or
   CPUBatchPropagator (multiprocessing) based on ``MonteCarloConfig.use_gpu``
   and hardware availability.
3. VRAM / RAM budget  — automatically tiles large ensembles into sub-batches
   that fit within ``mc_cfg.max_vram_gb``.
4. Streaming output   — snapshot data is written to HDF5 or NPZ at
   ``output_dt_s`` intervals to avoid memory exhaustion.
5. Progress reporting — optional structured ``progress_callback(payload)`` hook.

Usage example
-------------
::

    from config import load_default_config
    from common.montecarlo_defs import MonteCarloConfig, StateUncertainty
    from core.monte_carlo_engine import MonteCarloEngine

    sim_cfg = load_default_config()
    mc_cfg  = MonteCarloConfig(
        n_samples=500,
        state=StateUncertainty(sigma_r_m=500.0, sigma_v_m_s=0.5),
        use_gpu=True,
        gpu_sh_degree=10,
    )

    engine = MonteCarloEngine(sim_cfg, mc_cfg)
    result = engine.run()          # MCRunResult

Architecture note
-----------------
This module is **layer 3** (core); it may import from ``common`` and
``models`` but must not import from ``analysis`` or ``app``.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from common.constants import DAY_S, MU_MOON, R_MOON
from common.montecarlo_defs import MCRunResult, MonteCarloConfig, StateUncertainty
from common.type_defs import F64Array


# =============================================================================
# 0.                    LOCAL BOOTSTRAP / COMPAT HELPERS
# =============================================================================

class _GravityModelAdapter:
    """
    Normalize the modern ``GravityModel`` API to the strict core.dynamics contract.

    The Monte Carlo engine intentionally mirrors the runtime bootstrap used by
    the main single-run pipeline.  The dynamics layer expects attributes such
    as ``degree_max`` / ``R_ref_m`` / ``GM_m3s2`` while the modern
    ``models.spherical_harmonics.GravityModel`` exposes ``max_degree`` /
    ``r_ref`` / ``mu``.  This lightweight adapter keeps the MC path aligned
    with the rest of the project without mutating the source model object.
    """

    __slots__ = ("_model",)

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    @property
    def degree_max(self) -> int:
        return int(getattr(self._model, "degree_max", getattr(self._model, "max_degree")))

    @property
    def R_ref_m(self) -> float:
        return float(getattr(self._model, "R_ref_m", getattr(self._model, "r_ref")))

    @property
    def GM_m3s2(self) -> float:
        return float(getattr(self._model, "GM_m3s2", getattr(self._model, "mu")))

    @property
    def ws(self) -> Any:
        return getattr(self._model, "ws", getattr(self._model, "workspace"))

    def make_workspace(self) -> Any:
        if hasattr(self._model, "make_workspace"):
            return self._model.make_workspace()
        return self.ws

    @property
    def Cnm(self) -> Any:
        return getattr(self._model, "Cnm", getattr(self._model, "c_coeffs"))

    @property
    def Snm(self) -> Any:
        return getattr(self._model, "Snm", getattr(self._model, "s_coeffs"))

    @property
    def diag(self) -> Any:
        return getattr(self._model, "diag", getattr(self._model, "diag_coeffs"))

    @property
    def subdiag(self) -> Any:
        return getattr(self._model, "subdiag", getattr(self._model, "subdiag_coeffs"))

    @property
    def A(self) -> Any:
        return getattr(self._model, "A", getattr(self._model, "a_coeffs"))

    @property
    def B(self) -> Any:
        return getattr(self._model, "B", getattr(self._model, "b_coeffs"))

    @property
    def scale_m(self) -> Any:
        return getattr(self._model, "scale_m", getattr(self._model, "scale_m_table"))


def _adapt_gravity_model(model: Any) -> Any:
    """
    Return the model unchanged when it already satisfies the strict contract.

    Keeping this as a helper instead of inlining the attribute checks makes the
    MC bootstrap easier to follow and keeps the compatibility story explicit.
    """

    required = ("degree_max", "R_ref_m", "GM_m3s2", "Cnm", "Snm", "diag", "subdiag", "A", "B")
    if all(hasattr(model, name) for name in required) and hasattr(model, "ws"):
        return model
    return _GravityModelAdapter(model)


def _state_to_array(state_like: Any) -> np.ndarray:
    """
    Convert the configured nominal state to a plain row-major float64 vector.

    MC sampling works in Cartesian state space, so we accept the same state
    container styles as the single-run pipeline: ``InitialState``,
    ``OrbitState``-like objects exposing ``.y``, or raw array-likes.
    """

    if state_like is None:
        raise ValueError("Nominal state is None.")

    if hasattr(state_like, "to_array"):
        arr = np.asarray(getattr(state_like, "to_array")(), dtype=np.float64).reshape(-1)
    elif hasattr(state_like, "y"):
        arr = np.asarray(getattr(state_like, "y"), dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(state_like, dtype=np.float64).reshape(-1)

    if arr.size < 6:
        raise ValueError(f"Nominal state must contain at least 6 elements, got {arr.size}.")
    return np.ascontiguousarray(arr[:6], dtype=np.float64)


def _surface_topography_requested(surface_provider: Any, topo_grid: Any) -> bool:
    """
    Return True when the MC run needs Moon-fixed ephemeris because terrain is active.

    Topography can influence both surface-force sampling and impact detection, so
    the MC bootstrap mirrors the main runner by treating terrain availability as
    an ephemeris requirement even when third-body vectors are disabled.
    """

    return bool(surface_provider is not None or topo_grid is not None)


def _need_ephemeris(cfg: Any, *, topo_requested: bool) -> bool:
    """
    Match the main runner's ephemeris policy for consistent physics coverage.

    The Monte Carlo path should not secretly use a different decision tree than
    the single-run path.  Repeating the logic locally keeps this core module
    self-contained while preserving the same "SH/topography implies q_i2f"
    behavior.
    """

    flags = cfg.flags
    physics_need = (
        flags.enable_sh
        or flags.enable_3rd_body_sun
        or flags.enable_3rd_body_earth
        or flags.enable_earth_j2
        or flags.enable_srp
        or flags.enable_albedo
        or flags.enable_thermal
        or flags.enable_surface_forces
        or flags.enable_tides_k2
        or flags.enable_tides_k3
        or flags.enable_relativity_1pn
    )
    return bool(physics_need or topo_requested)


def _need_body_vectors(cfg: Any) -> bool:
    """
    Return True only when Sun/Earth position tables are physically required.

    SH-only or topo-only Monte Carlo runs still need the Moon-fixed attitude
    quaternion table, but they do not need Sun/Earth vectors.  Using this split
    keeps ephemeris initialization lighter and avoids misleading SPICE warnings.
    """

    flags = cfg.flags
    return bool(
        flags.enable_3rd_body_sun
        or flags.enable_3rd_body_earth
        or flags.enable_earth_j2
        or flags.enable_srp
        or flags.enable_albedo
        or flags.enable_thermal
        or flags.enable_tides_k2
        or flags.enable_tides_k3
    )


def _build_ephemeris_manager(cfg: Any) -> Any:
    """
    Build an ``EphemerisManager`` using the same buffered timeline as main.py.

    A small duration buffer protects interpolation near the last requested
    sample, which is especially helpful in Monte Carlo runs where many samples
    stop at slightly different times due to impact events.
    """

    from models.ephemeris import EphemerisManager

    start_utc = str(cfg.time.start_date).strip()
    if not start_utc:
        raise ValueError("cfg.time.start_date is empty.")

    time_cfg = replace(cfg.time, duration_s=float(cfg.time.duration_s) + 0.1 * DAY_S)
    spice_cfg = replace(cfg.spice, include_third_body=_need_body_vectors(cfg))
    return EphemerisManager.from_time_and_spice(
        time_cfg,
        spice_cfg,
        auto_fix_kernel_paths=True,
        need_moon_fixed_rotation=True,
    )


# =============================================================================
# 1.                      SAMPLE GENERATION
# =============================================================================

def sample_initial_states(
    nominal_state: F64Array,         # (6,) [x,y,z,vx,vy,vz]
    uncertainty: "StateUncertainty",
    n_samples: int,
    rng: np.random.Generator,
) -> F64Array:
    """
    Draw N Gaussian samples around the nominal state.

    Returns
    -------
    Y0 : (N, 6) float64 perturbed initial states
    """
    L = uncertainty.cholesky_factor()           # (6, 6) lower-triangular
    Z = rng.standard_normal((n_samples, 6))     # (N, 6) i.i.d.
    delta = Z @ L.T                             # (N, 6)  – broadcasted perturbation
    return np.ascontiguousarray(
        nominal_state[None, :] + delta, dtype=np.float64
    )


def sample_spacecraft_props(
    nominal_mass: float,
    nominal_area: float,
    nominal_cd: float,
    nominal_cr: float,
    uncertainty: Any,               # SpacecraftUncertainty
    n_samples: int,
    rng: np.random.Generator,
) -> F64Array:
    """
    Sample spacecraft physical properties (truncated normal at zero).

    Returns
    -------
    sc_samples : (N, 4) float64 — columns [mass_kg, area_m2, cd, cr]
    """
    sc = np.zeros((n_samples, 4), dtype=np.float64)

    def _trunc_normal(mu: float, sigma: float) -> np.ndarray:
        """Sample with sigma; clip at 0.01 * mu to keep values positive."""
        if sigma <= 0.0:
            return np.full(n_samples, mu, dtype=np.float64)
        raw = rng.normal(mu, sigma, n_samples)
        return np.clip(raw, 0.01 * max(mu, 1e-30), None)

    sc[:, 0] = _trunc_normal(nominal_mass, float(getattr(uncertainty, "sigma_mass_kg", 0.0)))
    sc[:, 1] = _trunc_normal(nominal_area, float(getattr(uncertainty, "sigma_area_m2", 0.0)))
    sc[:, 2] = _trunc_normal(nominal_cd,   float(getattr(uncertainty, "sigma_cd",     0.0)))
    sc[:, 3] = _trunc_normal(nominal_cr,   float(getattr(uncertainty, "sigma_cr",     0.0)))

    return sc


# =============================================================================
# 2.               OUTPUT WRITERS (HDF5 / NPZ)
# =============================================================================

def _metadata_value_to_jsonable(value: Any) -> Any:
    """
    Convert runtime metadata values into JSON-safe primitives.

    Monte Carlo archives are often reopened long after the run completed, so
    even lightweight metadata such as seed, cadence, and backend selection is
    worth preserving in a transport-safe form across both HDF5 and NPZ outputs.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_metadata_value_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value_to_jsonable(val) for key, val in value.items()}
    return str(value)


def _decode_archive_metadata(raw: Any) -> dict[str, Any]:
    """
    Decode a metadata payload stored inside an archive.

    The helper is intentionally permissive: missing or malformed metadata
    should not block result loading.
    """

    if raw is None:
        return {}

    try:
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        text = str(raw).strip()
        if not text:
            return {}
        decoded = json.loads(text)
        return decoded if isinstance(decoded, dict) else {}
    except Exception:
        return {}

class _HDF5Writer:
    """Streaming HDF5 writer.  Opens the file once, appends snapshot blocks."""

    def __init__(self, path: Path, n_samples: int, n_state: int = 6) -> None:
        try:
            import h5py
            self._h5py = h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 output. "
                "Install via:  pip install h5py"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(str(path), "w")
        self._n = n_samples
        self._s = n_state

        # Extendable datasets (chunked along time axis)
        self._ds_t = self._f.create_dataset(
            "t", shape=(0,), maxshape=(None,), dtype=np.float64,
            chunks=(256,), compression="lzf",
        )
        self._ds_Y = self._f.create_dataset(
            "Y", shape=(0, n_samples, n_state),
            maxshape=(None, n_samples, n_state),
            dtype=np.float64,
            chunks=(16, min(n_samples, 128), n_state),
            compression="lzf",
        )
        self._row = 0

    def write_snapshot(self, t: float, Y: np.ndarray) -> None:
        """Append one time snapshot (Y shape: (N, 6))."""
        self._ds_t.resize(self._row + 1, axis=0)
        self._ds_Y.resize(self._row + 1, axis=0)
        self._ds_t[self._row] = t
        self._ds_Y[self._row] = Y
        self._row += 1

    def write_metadata(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            try:
                payload = _metadata_value_to_jsonable(v)
                if isinstance(payload, (dict, list)):
                    payload = json.dumps(payload, sort_keys=True)
                self._f.attrs[k] = payload
            except Exception:
                pass

    def write_final(
        self,
        sc_samples: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
    ) -> None:
        self._f.create_dataset("sc_samples",  data=sc_samples)
        self._f.create_dataset("impact_flags", data=impact_flags)
        self._f.create_dataset("t_impact",    data=t_impact)

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


class _NPZWriter:
    """Accumulates snapshots in RAM and writes a single NPZ at the end."""

    def __init__(self, path: Path, n_samples: int, n_state: int = 6) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._t_list: List[float] = []
        self._Y_list: List[np.ndarray] = []
        self._metadata: dict[str, Any] = {}

    def write_snapshot(self, t: float, Y: np.ndarray) -> None:
        self._t_list.append(t)
        self._Y_list.append(Y.copy())

    def write_metadata(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self._metadata[str(key)] = _metadata_value_to_jsonable(value)

    def write_final(
        self,
        sc_samples: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
    ) -> None:
        t_arr = np.asarray(self._t_list, dtype=np.float64)
        Y_arr = np.stack(self._Y_list, axis=0)  # (T, N, 6)
        np.savez_compressed(
            str(self._path),
            t=t_arr,
            Y=Y_arr,
            sc_samples=sc_samples,
            impact_flags=impact_flags,
            t_impact=t_impact,
            metadata_json=np.asarray(json.dumps(self._metadata, sort_keys=True), dtype=np.str_),
        )

    def close(self) -> None:
        pass


def _make_writer(mc_cfg: MonteCarloConfig, n_samples: int) -> Any:
    """Factory: return the appropriate writer based on output_format."""
    p = mc_cfg.output_path_resolved
    if mc_cfg.output_format == "hdf5":
        return _HDF5Writer(p, n_samples)
    return _NPZWriter(p, n_samples)


# =============================================================================
# 3.               MONTE CARLO ENGINE
# =============================================================================

class MonteCarloEngine:
    """
    Orchestrates a full Monte Carlo orbital uncertainty propagation run.

    Workflow
    --------
    1. ``__init__``: validate configs, select backend (GPU / CPU).
    2. ``run()``:
       a. Draw N initial state samples + spacecraft property samples.
       b. Open output writer.
       c. For each sub-batch (VRAM-bounded):
          - Transfer arrays to device (GPU) or dispatch workers (CPU).
          - Iterate over time steps; write snapshots to disk.
       d. Aggregate impact statistics.
       e. Return ``MCRunResult``.

    Parameters
    ----------
    sim_cfg : SimConfig
        Full simulation configuration (physics flags, gravity, ephemeris, …).
    mc_cfg : MonteCarloConfig
        Monte Carlo parameters (N, uncertainties, GPU flags, output format).
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
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self._sim_cfg = sim_cfg
        self._mc      = mc_cfg
        self._cb      = progress_callback
        self._surface_provider = surface_provider
        self._topo_grid = topo_grid
        self._backend_note = ""
        if self._topo_grid is None and self._surface_provider is not None and hasattr(self._surface_provider, "grids"):
            try:
                self._topo_grid = self._surface_provider.grids().topo  # type: ignore[attr-defined]
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
        batch_index: Optional[int] = None,
        batch_count: Optional[int] = None,
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
        eta_s: Optional[float] = None
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
        in Monte Carlo" divergences.
        """
        from core.dynamics import DynamicsEngine

        cfg = self._sim_cfg
        grav_model = None
        ephem_manager = None
        surface_provider = self._surface_provider
        topo_requested = _surface_topography_requested(surface_provider, self._topo_grid)

        if bool(cfg.flags.enable_sh):
            try:
                if bool(getattr(cfg.gravity, "uses_st_lrps", False)):
                    from models.surrogate_gravity import SurrogateGravityModel

                    # Prioritize the MC-specific ST-LRPS run directory if provided.
                    st_lrps_dir = self._mc.st_lrps_model_dir or cfg.gravity.st_lrps_model_dir
                    
                    from common.montecarlo_defs import validate_st_lrps_model_dir
                    valid_dir = validate_st_lrps_model_dir(st_lrps_dir)

                    grav_model = SurrogateGravityModel.from_model_dir(
                        str(valid_dir),
                        mu_override=float(MU_MOON),
                        r_ref_override=float(R_MOON),
                        device_preference="cpu",
                    )
                else:
                    from models.spherical_harmonics import GravityModel

                    requested_degree = int(cfg.gravity.degree) if cfg.gravity.degree is not None else None
                    grav_model = _adapt_gravity_model(
                        GravityModel.from_file(
                            path=str(cfg.gravity.file_path),
                            requested_degree=requested_degree,
                        )
                    )
            except Exception as exc:
                warnings.warn(f"[MC] Could not load gravity model: {exc}", RuntimeWarning)

        if _need_ephemeris(cfg, topo_requested=topo_requested):
            try:
                ephem_manager = _build_ephemeris_manager(cfg)
            except Exception as exc:
                warnings.warn(f"[MC] Could not load ephemeris: {exc}", RuntimeWarning)

        earth_j2 = getattr(cfg, "earth_j2", None)

        return DynamicsEngine(
            sc_props=cfg.spacecraft,
            flags=cfg.flags,
            gravity_model=grav_model,
            gravity_adaptive=(
                None if bool(getattr(cfg.gravity, "uses_st_lrps", False))
                else getattr(cfg.gravity, "adaptive", None)
            ),
            ephem_manager=ephem_manager,
            surface_provider=surface_provider,
            earth_j2=earth_j2,
            allow_identity_rotation=(ephem_manager is None),
        )

    # ----------------------------------------------------------------
    # Internal: select and initialise backend
    # ----------------------------------------------------------------

    def _build_propagator(self) -> Any:
        """
        Instantiate the appropriate batch propagator using the backend policy.

        Backend selection is fully delegated to
        ``core.mc_backend_policy.resolve_mc_backend_policy`` so the routing
        logic is testable in isolation without constructing a full engine.
        """
        from core.mc_backend_policy import MCBackend, resolve_mc_backend_policy
        from core.mc_propagator import CPUBatchPropagator

        plan = resolve_mc_backend_policy(self._mc, self._sim_cfg)

        # Emit all warnings produced by the policy resolver
        for w in plan.warnings:
            warnings.warn(w, RuntimeWarning)
            self._backend_note = w  # keep the most recent one for the run log

        # Log the resolved plan
        plan.log_summary()

        # ----------------------------------------------------------------
        # GPU ST-LRPS path — PyTorch fixed-step RK4
        # ----------------------------------------------------------------
        if plan.final_backend == MCBackend.GPU_ST_LRPS:
            try:
                from core.torch_batch_propagator import TorchBatchPropagator

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
                return TorchBatchPropagator(
                    surrogate_model=grav_model,
                    mc_cfg=self._mc,
                    device_id=int(getattr(self._mc, "gpu_device_id", 0)),
                )
            except Exception as exc:
                note = (
                    f"[MC] GPU ST-LRPS backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._backend_note = note
                warnings.warn(note, RuntimeWarning)

        # ----------------------------------------------------------------
        # GPU classic-SH path — Numba CUDA fixed-step RK4
        # ----------------------------------------------------------------
        elif plan.final_backend == MCBackend.GPU_CLASSIC_SH:
            try:
                from core.mc_propagator import GPUBatchPropagator

                return GPUBatchPropagator(
                    self._dyn,
                    self._mc,
                    self._sim_cfg.flags,
                )
            except Exception as exc:
                note = (
                    f"[MC] GPU classic-SH backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._backend_note = note
                warnings.warn(note, RuntimeWarning)

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

    # ----------------------------------------------------------------
    # Public: run
    # ----------------------------------------------------------------

    def run(self) -> MCRunResult:
        """
        Execute the full Monte Carlo simulation.

        Returns
        -------
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
            detail="Preparing Monte Carlo sample set",
        )

        # -----------------------------------------------------------------
        # 1) Generate samples
        # -----------------------------------------------------------------
        nominal = _state_to_array(cfg.initial_state)   # (6,)

        Y0 = sample_initial_states(nominal, mc.state, N, rng)
        sc_samples = sample_spacecraft_props(
            nominal_mass=float(cfg.spacecraft.mass_kg),
            nominal_area=float(cfg.spacecraft.area_m2),
            nominal_cd=float(cfg.spacecraft.cd),
            nominal_cr=float(cfg.spacecraft.cr),
            uncertainty=mc.spacecraft,
            n_samples=N,
            rng=rng,
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
            detail="Samples generated",
        )

        # -----------------------------------------------------------------
        # 2) Output writer + propagator
        # -----------------------------------------------------------------
        writer = _make_writer(mc, N)
        prop   = self._build_propagator()

        # Fail-fast: validate gravity model contract before entering the sample
        # loop.  Without this check the CPU propagator catches the same missing-
        # attribute error N times and prints N identical "Sample i failed" lines.
        if hasattr(prop, "validate_gravity_assets"):
            try:
                prop.validate_gravity_assets()
            except RuntimeError:
                writer.close()
                raise

        _cls = prop.__class__.__name__
        if _cls == "TorchBatchPropagator":
            backend_name = "GPU-ST-LRPS"
        elif _cls.startswith("GPU"):
            backend_name = "GPU"
        else:
            backend_name = "CPU"
        backend_diag = prop.diagnostics_snapshot() if hasattr(prop, "diagnostics_snapshot") else {}
        requested_max_batch = mc.effective_max_batch()
        if hasattr(prop, "recommended_max_batch"):
            max_batch = int(prop.recommended_max_batch(requested_max_batch))
        else:
            max_batch = int(requested_max_batch)

        duration_s  = float(cfg.time.duration_s)
        output_dt_s = float(cfg.time.output_dt_s or mc.dt_s * 10)

        print(
            f"[MC] N={N}  backend={backend_name}  "
            f"T={duration_s / 86400:.2f} d  "
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
                    f"[MC] GPU device={device_name}  tpb={tpb}  "
                    f"batch_cap~{max_batch}",
                    flush=True,
                )

        # -----------------------------------------------------------------
        # 3) Sub-batch loop (VRAM budget)
        # -----------------------------------------------------------------
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

        # Accumulators for the full ensemble
        t_out_ref: Optional[np.ndarray] = None
        Y_all     = None   # will be (T, N, 6) after first batch
        impact_all   = np.zeros(N, dtype=np.float64)
        t_impact_all = np.full(N, np.nan, dtype=np.float64)

        for b_idx in range(n_batches):
            b_start = b_idx * max_batch
            b_end   = min(N, b_start + max_batch)
            b_n     = b_end - b_start

            print(
                f"[MC] Batch {b_idx + 1}/{n_batches}  "
                f"samples {b_start}–{b_end - 1}",
                flush=True,
            )

            def _batch_progress(local_fraction: float) -> None:
                effective_done = float(b_start) + float(b_n) * max(0.0, min(1.0, float(local_fraction)))
                self._publish_progress(
                    stage="propagating",
                    stage_fraction=(effective_done / max(N, 1)),
                    total_samples=N,
                    done_samples=effective_done,
                    elapsed_s=time.perf_counter() - t_wall0,
                    backend=backend_name,
                    batch_index=b_idx + 1,
                    batch_count=n_batches,
                    detail=f"Batch {b_idx + 1}/{n_batches}",
                )

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

            # First batch defines the reference time grid
            if t_out_ref is None:
                t_out_ref = t_b
                Y_all = np.zeros((len(t_b), N, 6), dtype=np.float64)

            # Resample to reference grid if needed
            if len(t_b) == len(t_out_ref) and np.allclose(t_b, t_out_ref, rtol=1e-6):
                Y_all[:, b_start:b_end, :] = Y_b
            else:
                # Linear interpolation to reference grid
                T_ref = len(t_out_ref)
                for j in range(b_n):
                    for c in range(6):
                        Y_all[:, b_start + j, c] = np.interp(
                            t_out_ref, t_b, Y_b[:, j, c]
                        )

            impact_all[b_start:b_end] = imp_b
            t_impact_all[b_start:b_end] = t_imp_b

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

        # -----------------------------------------------------------------
        # 4) Compute impact times (t_impact_s)
        # -----------------------------------------------------------------
        r_impact = float(R_MOON) + float(mc.impact_alt_km) * 1_000.0
        for i in range(N):
            if impact_all[i] > 0.5 and not np.isfinite(float(t_impact_all[i])):
                # Find first time the spacecraft crossed the impact sphere
                r_history = np.linalg.norm(Y_all[:, i, :3], axis=1)
                hit_idx = np.argmax(r_history <= r_impact)
                if hit_idx > 0 and t_out_ref is not None:
                    t_impact_all[i] = float(t_out_ref[hit_idx])

        # -----------------------------------------------------------------
        # 5) Write to disk
        # -----------------------------------------------------------------
        # Collect ST-LRPS provenance metadata when the surrogate backend is active.
        _grav_cfg = getattr(self._sim_cfg, "gravity", None)
        _grav_model = getattr(self._dyn, "grav", None)
        _st_lrps_meta: dict[str, Any] = {}
        if getattr(_grav_model, "model_kind", None) == "st_lrps":
            _st_lrps_meta = {
                "gravity_backend": "st_lrps",
                "st_lrps_model_dir": str(getattr(_grav_model, "model_dir", "") or ""),
                "st_lrps_degree_min": getattr(_grav_model, "degree_min", None),
                "st_lrps_degree_max": getattr(_grav_model, "degree_max", None),
                "effective_degree_max": getattr(_grav_model, "effective_degree_max", None),
            }

        # Collect backend-plan provenance for the archive
        try:
            from core.mc_backend_policy import resolve_mc_backend_policy as _resolve
            _plan = _resolve(mc, self._sim_cfg)
            _plan_meta: dict[str, Any] = {
                "mc_backend": _plan.final_backend.value,
                "requested_use_gpu": bool(mc.use_gpu),
                "final_use_gpu": _plan.use_gpu,
                "plan_gravity_backend": _plan.gravity_backend,   # renamed: avoids collision with _st_lrps_meta["gravity_backend"]
                "torch_cuda_available": _plan.torch_cuda_available,
                "numba_cuda_available": _plan.numba_cuda_available,
                "integrator": _plan.integrator,
                "batch_size": max_batch,
                "fallback_reason": _plan.reason if not _plan.use_gpu else "",
            }
        except Exception:
            _plan_meta = {}

        writer.write_metadata(
            n_samples=N,
            seed=int(mc.seed),
            duration_s=duration_s,
            output_dt_s=output_dt_s,
            requested_backend="GPU" if bool(mc.use_gpu) else "CPU",
            gpu_sh_degree=int(mc.gpu_sh_degree),
            backend=backend_name,
            backend_note=self._backend_note,
            backend_diagnostics=backend_diag,
            **_st_lrps_meta,
            **_plan_meta,
        )
        if t_out_ref is not None and Y_all is not None:
            total_rows = max(int(len(t_out_ref)), 1)
            for k, t_k in enumerate(t_out_ref):
                writer.write_snapshot(float(t_k), Y_all[k])
                if k == 0 or (k + 1) == total_rows or ((k + 1) % max(1, total_rows // 20) == 0):
                    self._publish_progress(
                        stage="writing",
                        stage_fraction=(float(k + 1) / float(total_rows)),
                        total_samples=N,
                        done_samples=float(N),
                        elapsed_s=time.perf_counter() - t_wall0,
                        backend=backend_name,
                        batch_index=n_batches,
                        batch_count=n_batches,
                        detail=f"Writing snapshots {k + 1}/{total_rows}",
                    )

        try:
            writer.write_final(sc_samples, impact_all, t_impact_all)
        finally:
            writer.close()

        t_wall = time.perf_counter() - t_wall0
        n_hit  = int(np.sum(impact_all > 0.5))
        print(
            f"[MC] Done. Wall={t_wall:.1f}s  "
            f"impacts={n_hit}/{N} ({100.0 * n_hit / N:.1f}%)",
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

        # -----------------------------------------------------------------
        # 6) Build result
        # -----------------------------------------------------------------
        if t_out_ref is None or Y_all is None:
            t_out_ref = np.array([0.0], dtype=np.float64)
            Y_all = np.zeros((1, N, 6), dtype=np.float64)

        return MCRunResult(
            t=t_out_ref,
            Y=Y_all,
            sc_samples=sc_samples,
            impact_mask=impact_all,
            t_impact=t_impact_all,
            diagnostics={
                "wall_time_s": float(t_wall),
                "n_samples": N,
                "n_impacts": n_hit,
                "impact_fraction": float(n_hit) / N,
                "backend": backend_name,
                "backend_note": self._backend_note,
                "output_path": str(mc.output_path_resolved),
                "backend_diagnostics": backend_diag,
            },
        )


# =============================================================================
# 4.             LOADER: read back a previously saved run
# =============================================================================

def load_mc_result(path: str) -> MCRunResult:
    """
    Reload a saved ``MCRunResult`` from HDF5 or NPZ file.

    Parameters
    ----------
    path : str
        Path produced by ``MonteCarloEngine.run()``.

    Returns
    -------
    MCRunResult
    """
    p = Path(path).expanduser().resolve()
    suffix = p.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required to read HDF5 MC output.")
        with h5py.File(str(p), "r") as f:
            t_arr  = np.asarray(f["t"],           dtype=np.float64)
            Y_arr  = np.asarray(f["Y"],           dtype=np.float64)
            sc     = np.asarray(f["sc_samples"],  dtype=np.float64)
            imask  = np.asarray(f["impact_flags"], dtype=np.float64)
            t_imp  = np.asarray(f["t_impact"],    dtype=np.float64)
            diagnostics = {
                str(key): _metadata_value_to_jsonable(value)
                for key, value in dict(f.attrs).items()
            }
        return MCRunResult(
            t=t_arr, Y=Y_arr, sc_samples=sc,
            impact_mask=imask, t_impact=t_imp,
            diagnostics=diagnostics,
        )

    if suffix == ".npz":
        with np.load(str(p), allow_pickle=False) as data:
            diagnostics = {}
            if "metadata_json" in data.files:
                diagnostics = _decode_archive_metadata(data["metadata_json"])
            return MCRunResult(
                t=data["t"],
                Y=data["Y"],
                sc_samples=data["sc_samples"],
                impact_mask=data["impact_flags"],
                t_impact=data["t_impact"],
                diagnostics=diagnostics,
            )

    raise ValueError(f"Unrecognised MC output format: {suffix!r} (expected .h5 or .npz)")


# =============================================================================
# 5.                        PUBLIC API
# =============================================================================

__all__ = [
    "MonteCarloEngine",
    "sample_initial_states",
    "sample_spacecraft_props",
    "load_mc_result",
]
