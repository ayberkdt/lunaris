"""Internal module of the lunar gravity-model benchmark harness.

Part of :mod:`lunaris.surrogate.st_lrps.evaluation.compare_gravity_models`;
this is an implementation detail, not a public API. See that module's
docstring for CLI usage.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lunaris.common.constants import DAY_S
from lunaris.core.config import SimConfig, load_default_config, replace_sim_config
from lunaris.physics.spherical_harmonics import GravityModel
from lunaris.surrogate.runtime import (
    SurrogateGravityModel,
    find_checkpoint_for_st_lrps_run,
)

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class Scenario:
    scenario_id: int
    hp_km: float
    ha_km: float
    a_km: float
    e: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    ta_deg: float
    initial_state: np.ndarray = field(repr=False)
    raw_unit_sample: list[float] | None = None
    sampling_method: str = "random"


@dataclass
class BatchModelResult:
    """Container for one fixed-step batch propagation result."""

    model_name: str
    display_name: str
    backend: str
    device: str
    dtype: str
    t: np.ndarray
    y: np.ndarray
    runtime_s: float
    n_steps: int
    n_scenarios: int
    rk4_dt_s: float
    output_dt_s: float
    # "ok" (every scenario completed), "partial" (some scenarios failed and
    # are NaN-filled / excluded from metrics), or "failed" (nothing usable).
    status: str
    failure_reason: str = ""
    # "fixed_step_rk" for the GPU batch kernels; "cpu_adaptive" for the
    # per-scenario SciPy adaptive series (rk4_dt_s is NaN there).
    integrator: str = "fixed_step_rk"
    # Total RHS (acceleration) evaluations across all scenarios, when the
    # propagator reports them (adaptive series); None for fixed-step results
    # whose eval count is derived from n_steps * evals_per_step.
    accel_evals_total: int | None = None
    # Field-evaluation provenance when it differs from the integrator/state
    # provenance carried by ``device``/``dtype``. Only the CPU adaptive
    # surrogate series sets these: its integrator loop and state arrays live
    # on the CPU in float64 while the surrogate field evaluates on the device
    # and dtype the loaded model uses. None means "same as device/dtype".
    model_device: str | None = None
    model_dtype: str | None = None


@dataclass
class TruthTrajectorySet:
    """SH200 DOP853 truth trajectories keyed by scenario id."""

    model_name: str
    t_by_scenario: dict[int, np.ndarray]
    y_by_scenario: dict[int, np.ndarray]
    runtime_by_scenario: dict[int, float]

    @property
    def total_runtime_s(self) -> float:
        return float(sum(self.runtime_by_scenario.values()))

    @property
    def mean_runtime_s(self) -> float:
        if not self.runtime_by_scenario:
            return float("nan")
        return float(np.mean(list(self.runtime_by_scenario.values())))


@dataclass
class CachedTrajectory:
    t: np.ndarray
    y: np.ndarray
    runtime_s: float = float("nan")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GpuBatchTask:
    model_name: str
    cache_name: str
    display_name: str
    rk4_dt_s: float


_METRICS_FIELDNAMES = [
    "scenario_id", "model",
    "runtime_s", "runtime_rel_to_truth",
    "rms_pos_err_km", "final_pos_err_km", "max_pos_err_km", "p95_pos_err_km",
    "rms_vel_err_ms", "final_vel_err_ms", "max_vel_err_ms", "p95_vel_err_ms",
    "radial_rms_km", "along_rms_km", "cross_rms_km",
    "radial_max_km", "along_max_km", "cross_max_km",
    "phase_lag_final_s", "phase_lag_slope_s_per_day",
    "phase_corrected_rms_km", "phase_explained_fraction",
    "final_alt_err_km", "rms_alt_err_km", "max_abs_alt_err_km",
    "min_alt_model_km", "min_alt_truth_km",
    "status",
]

_BATCH_METRICS_FIELDNAMES = [
    "scenario_id", "model", "reference",
    "rms_pos_err_km", "final_pos_err_km", "max_pos_err_km", "p95_pos_err_km",
    "rms_vel_err_ms", "final_vel_err_ms",
    "radial_rms_km", "along_rms_km", "cross_rms_km",
    "phase_lag_final_s", "phase_lag_slope_s_per_day",
    "phase_corrected_rms_km", "phase_explained_fraction",
    "rms_alt_err_km", "hp_km", "inc_deg", "status",
]

_GPU_BATCH_METRICS_FIELDNAMES = [
    "scenario_id", "model", "reference", "backend", "device", "rk4_dt_s",
    "duration_days", "hp_km", "ha_km", "a_km", "e", "inc_deg", "raan_deg",
    "argp_deg", "ta_deg",
    "rms_pos_err_km", "final_pos_err_km", "max_pos_err_km", "p95_pos_err_km",
    "rms_vel_err_ms", "final_vel_err_ms", "max_vel_err_ms", "p95_vel_err_ms",
    "radial_rms_km", "along_rms_km", "cross_rms_km",
    "radial_max_km", "along_max_km", "cross_max_km",
    "phase_lag_final_s", "phase_lag_slope_s_per_day",
    "phase_corrected_rms_km", "phase_explained_fraction",
    "rms_alt_err_km", "final_alt_err_km", "max_abs_alt_err_km",
    "min_alt_model_km", "min_alt_truth_km", "status", "failure_reason",
]

SAMPLING_METHODS = ("random", "lhs", "sobol", "sobol_scrambled")
INCLINATION_SAMPLING_METHODS = ("uniform_deg", "uniform_cos")
SCENARIO_UNIT_DIM = 6
SCENARIO_MANIFEST_CSV = "scenario_manifest.csv"
SCENARIO_MANIFEST_JSON = "scenario_manifest.json"
BENCHMARK_CACHE_SCHEMA_VERSION = 1
# =============================================================================
# Config helpers
# =============================================================================

def build_base_config(args: argparse.Namespace) -> SimConfig:
    cfg = load_default_config()
    new_time = replace(cfg.time,
        duration_s=args.duration_days * DAY_S,
        output_dt_s=args.dt_out,
    )
    new_prop = replace(cfg.propagator,
        method=args.integrator,
        rtol=args.rtol,
        atol=args.atol,
        user_max_step_s=args.max_step,
    )
    new_flags = replace(cfg.flags,
        enable_sh=True,
        enable_3rd_body_sun=False,
        enable_3rd_body_earth=False,
        enable_srp=False,
        enable_albedo=False,
        enable_thermal=False,
        enable_earth_j2=False,
    )
    return replace_sim_config(cfg, time=new_time, propagator=new_prop, flags=new_flags)


def _cfg_with_tolerances(cfg: SimConfig, rtol: float, atol: float) -> SimConfig:
    """Return a copy of ``cfg`` whose propagator uses the given tolerances.

    The CPU adaptive surrogate series needs looser tolerances than the truth
    run: a float32 surrogate RHS carries a relative noise floor around 1e-7,
    and truth-grade tolerances (rtol 1e-10) push the step controller into a
    rejection crawl against that noise instead of buying accuracy.
    """
    return replace_sim_config(
        cfg,
        propagator=replace(cfg.propagator, rtol=float(rtol), atol=float(atol)),
    )


def _cfg_with_integrator(cfg: SimConfig, integrator: str) -> SimConfig:
    """Return a copy of ``cfg`` whose propagator uses ``integrator`` (e.g. RK45/DOP853).

    Used to let the ground-truth reference run a different adaptive integrator
    from the compared models without mutating the shared base config.
    """
    return replace_sim_config(
        cfg,
        propagator=replace(cfg.propagator, method=str(integrator)),
    )


# =============================================================================
# Time interpolation
# =============================================================================

def interpolate_state_to_times(
    src_t: np.ndarray,
    src_y: np.ndarray,
    tgt_t: np.ndarray,
    tol: float = 1e-3,
) -> np.ndarray:
    """Interpolate state (N_src, 6) to target times (N_tgt,). Returns (N_tgt, 6)."""
    if len(src_t) == len(tgt_t) and np.max(np.abs(src_t - tgt_t)) < tol:
        return src_y
    result = np.empty((len(tgt_t), 6), dtype=np.float64)
    for k in range(6):
        result[:, k] = np.interp(tgt_t, src_t, src_y[:, k])
    return result


# =============================================================================
# RIC decomposition
# =============================================================================

def decompose_vector_ric(
    vec: np.ndarray,
    r_ref: np.ndarray,
    v_ref: np.ndarray,
) -> np.ndarray:
    scalar = vec.ndim == 1
    if scalar:
        vec   = vec[None, :]
        r_ref = r_ref[None, :]
        v_ref = v_ref[None, :]

    N = r_ref.shape[0]
    out = np.zeros((N, 3), dtype=np.float64)

    r_norms = np.linalg.norm(r_ref, axis=1, keepdims=True)
    r_hat   = r_ref / np.maximum(r_norms, 1e-12)

    h       = np.cross(r_ref, v_ref)
    h_norms = np.linalg.norm(h, axis=1, keepdims=True)
    c_hat   = h / np.maximum(h_norms, 1e-12)

    i_hat = np.cross(c_hat, r_hat)

    out[:, 0] = np.einsum("ij,ij->i", vec, r_hat)
    out[:, 1] = np.einsum("ij,ij->i", vec, i_hat)
    out[:, 2] = np.einsum("ij,ij->i", vec, c_hat)

    return out[0] if scalar else out


def compute_ric_errors(
    r_ref: np.ndarray,
    v_ref: np.ndarray,
    r_test: np.ndarray,
) -> np.ndarray:
    return decompose_vector_ric(r_test - r_ref, r_ref, v_ref)


# =============================================================================
# Gravity model cache
# =============================================================================

class GravityModelCache:
    """Loads each gravity model once and reuses it across scenarios."""

    def __init__(self, cfg: SimConfig, args: argparse.Namespace) -> None:
        self._cfg  = cfg
        self._args = args
        self._cache: dict[str, Any] = {}

    def get(self, model_name: str) -> Any:
        if model_name not in self._cache:
            self._cache[model_name] = self._load(model_name)
        return self._cache[model_name]

    def _load(self, model_name: str) -> Any:
        if model_name.startswith("sh"):
            degree = int(model_name.replace("sh", ""))
            print(f"  [cache] Loading SH{degree} gravity model ...", flush=True)
            # GravityModel already satisfies the dynamics gravity contract.
            return GravityModel.from_file(self._cfg.gravity.file_path, requested_degree=degree)

        if model_name == "st_lrps":
            if not self._args.st_lrps_model_dir:
                raise ValueError("--st-lrps-model-dir required for st_lrps model")

            # Use GPU if batch-rk4 or gpu mode requested
            want_gpu = (
                getattr(self._args, "batch_rk4", False) or
                getattr(self._args, "gpu_batch_compare", False) or
                getattr(self._args, "st_lrps_mode", "cpu_dop853") != "cpu_dop853"
            )
            device_pref = "cpu"
            if want_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_pref = "cuda"
                    else:
                        fallback = getattr(self._args, "gpu_fallback", "cpu")
                        if fallback == "error":
                            raise RuntimeError(
                                "CUDA requested for ST-LRPS but torch.cuda.is_available()=False. "
                                "Use --gpu-fallback cpu to fall back to CPU."
                            )
                        print("  [cache] CUDA unavailable; loading ST-LRPS on CPU.", flush=True)
                except ImportError:
                    fallback = getattr(self._args, "gpu_fallback", "error")
                    if fallback == "error":
                        raise RuntimeError(
                            "CUDA/ST-LRPS GPU mode requested but PyTorch is not installed. "
                            "Install CUDA-enabled PyTorch or use --gpu-fallback cpu."
                        ) from None
                    print("  [cache] PyTorch not installed; loading ST-LRPS on CPU.", flush=True)

            print(f"  [cache] Loading ST-LRPS model from {self._args.st_lrps_model_dir} "
                  f"(device={device_pref}) ...", flush=True)
            weight = _find_st_lrps_weight_file(self._args.st_lrps_model_dir)
            if weight:
                print(f"  [cache] ST-LRPS checkpoint: {weight}", flush=True)
            return SurrogateGravityModel.from_model_dir(
                self._args.st_lrps_model_dir,
                device_preference=device_pref,
            )

        raise ValueError(f"Unknown model name: {model_name!r}")


# ---- moved here to keep the import graph acyclic ----
def _find_st_lrps_weight_file(model_dir: str | None) -> str | None:
    """Return the checkpoint path used by the ST-LRPS runtime, if available."""

    if not model_dir:
        return None
    try:
        return str(find_checkpoint_for_st_lrps_run(model_dir))
    except Exception:
        return None
