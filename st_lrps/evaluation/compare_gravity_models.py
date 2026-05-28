# st_lrps/evaluation/compare_gravity_models.py
# -*- coding: utf-8 -*-
"""
Lunar Gravity Model Validation Harness
=======================================

Compares SH20/SH80/SH120/SH160 (and optionally ST-LRPS) against SH200
as ground truth, either for a single orbit or across N random scenarios.

Smoke tests:

  # CPU smoke (no GPU needed)
  python -m st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 3 --duration-days 0.01 \\
      --models sh20,sh80 --truth sh200 \\
      --output-dir results/smoke_cpu

  # ST-LRPS force batch evaluation
  python -m st_lrps.evaluation.compare_gravity_models \\
      --force-sample-trajectory sh200 \\
      --models st_lrps,sh80 \\
      --st-lrps-mode gpu_rk4 --force-batch-size 8192 \\
      --output-dir results/smoke_force_gpu

  # 100-orbit ST-LRPS GPU batch RK4
  python -m st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 100 --scenario-seed 42 \\
      --scenario-mode near_circular_altitude \\
      --altitude-min-km 200 --altitude-max-km 400 \\
      --duration-days 1.0 --dt-out 60 \\
      --models st_lrps --truth sh200 \\
      --st-lrps-mode gpu_rk4 \\
      --batch-rk4 --st-lrps-rk4-dt 10 \\
      --output-dir results/stlrps_batch_rk4_100

  # Full comparison (DOP853 + batch RK4)
  python -m st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 100 --scenario-seed 42 \\
      --scenario-mode near_circular_altitude \\
      --altitude-min-km 200 --altitude-max-km 400 \\
      --duration-days 1.0 \\
      --models sh20,sh80,sh120,sh160,st_lrps --truth sh200 \\
      --st-lrps-mode gpu_rk4 \\
      --batch-rk4 --st-lrps-rk4-dt 10 \\
      --output-dir results/full_validation_100

  # Full GPU batch comparison: SH200 DOP853 truth vs GPU RK4 models
  python -m st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 100 --scenario-seed 42 \\
      --scenario-mode near_circular_altitude \\
      --altitude-min-km 200 --altitude-max-km 400 \\
      --duration-days 1.0 --dt-out 60 \\
      --truth sh200 \\
      --gpu-models sh200,sh160,sh120,sh60,sh20,st_lrps \\
      --gpu-batch-compare --rk4-dt-s 10 \\
      --torch-dtype float64 --plot-theme report_light \\
      --output-dir results/gpu_sh_vs_stlrps_100

  # Faster GPU batch smoke
  python -m st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 5 --duration-days 0.05 \\
      --truth sh200 \\
      --gpu-models sh200,sh60,sh20,st_lrps \\
      --gpu-batch-compare --rk4-dt-s 10 \\
      --output-dir results/smoke_gpu_batch_compare
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import load_default_config, SimConfig
    from core.state import create_state_from_keplerian, calculate_ae_from_altitudes
    from core.dynamics import DynamicsEngine
    from core.propagator import propagate
    from models.ephemeris import EphemerisManager
    from models.gravity_adapter import adapt_gravity_model
    from models.spherical_harmonics import GravityModel
    from models.surrogate_gravity import (
        SurrogateGravityModel,
        find_checkpoint_for_st_lrps_run,
        find_latest_st_lrps_model_dir,
    )
    from common.constants import MU_MOON, R_MOON
except ImportError as exc:
    print(f"CRITICAL: Must run from ST_LRPS root. Missing: {exc}", file=sys.stderr)
    sys.exit(1)

from st_lrps.evaluation import progress
from dataclasses import replace


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
    raw_unit_sample: Optional[List[float]] = None
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
    status: str
    failure_reason: str = ""


@dataclass
class TruthTrajectorySet:
    """SH200 DOP853 truth trajectories keyed by scenario id."""

    model_name: str
    t_by_scenario: Dict[int, np.ndarray]
    y_by_scenario: Dict[int, np.ndarray]
    runtime_by_scenario: Dict[int, float]

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
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    "final_alt_err_km", "rms_alt_err_km", "max_abs_alt_err_km",
    "min_alt_model_km", "min_alt_truth_km",
    "status",
]

_BATCH_METRICS_FIELDNAMES = [
    "scenario_id", "model", "reference",
    "rms_pos_err_km", "final_pos_err_km", "max_pos_err_km", "p95_pos_err_km",
    "rms_vel_err_ms", "final_vel_err_ms",
    "radial_rms_km", "along_rms_km", "cross_rms_km",
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
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lunar gravity model validation harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Random / sampled scenario mode ---
    p.add_argument("--random-scenarios", type=int, default=100,
                   help="Number of validation scenarios, used by all sampling methods")
    p.add_argument("--scenario-seed", type=int, default=42)
    p.add_argument("--scenario-mode",
                   choices=["bounded_keplerian", "near_circular_altitude"],
                   default="near_circular_altitude")
    p.add_argument("--sampling-method", choices=SAMPLING_METHODS, default="random",
                   help="Scenario sampler. 'random' preserves the legacy generator.")
    p.add_argument("--inclination-sampling", choices=INCLINATION_SAMPLING_METHODS,
                   default="uniform_deg",
                   help="Sample inclination uniformly in degrees or uniformly in cos(i).")
    p.add_argument("--altitude-min-km", type=float, default=100.0)
    p.add_argument("--altitude-max-km", type=float, default=1000.0)
    p.add_argument("--ecc-min", type=float, default=0.0)
    p.add_argument("--ecc-max", type=float, default=0.0)
    p.add_argument("--inc-min-deg", type=float, default=0.0)
    p.add_argument("--inc-max-deg", type=float, default=180.0)
    p.add_argument("--raan-min-deg", type=float, default=0.0)
    p.add_argument("--raan-max-deg", type=float, default=360.0)
    p.add_argument("--argp-min-deg", type=float, default=0.0)
    p.add_argument("--argp-max-deg", type=float, default=360.0)
    p.add_argument("--ta-min-deg", type=float, default=0.0)
    p.add_argument("--ta-max-deg", type=float, default=360.0)
    p.add_argument("--resume", action="store_true",
                   help="Skip scenarios already in per_scenario_metrics.csv and aggregate old rows")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--plot-scenario-id", type=int, default=None,
                   help="Scenario id to plot (default: median-difficulty scenario)")
    p.add_argument("--scenario-limit", type=int, default=None)

    # --- Propagation ---
    p.add_argument("--duration-days", type=float, default=1.0)
    p.add_argument("--dt-out", type=float, default=60.0)
    p.add_argument("--integrator", type=str, default="DOP853",
                   help="Adaptive integrator for the compared models in per-model "
                        "CPU mode (e.g. DOP853, RK45).")
    p.add_argument("--truth-integrator", choices=["RK45", "DOP853"], default="DOP853",
                   help="Adaptive integrator used to build the ground-truth "
                        "reference trajectories (default: DOP853).")
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--max-step", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=4,
                   help="CPU worker processes for adaptive DOP853/RK45 work. In CPU "
                        "mode this parallelizes truth + compared-model scenario sweeps; "
                        "in GPU batch mode this parallelizes CPU truth generation. "
                        "1 = sequential. Each worker rebuilds its own ephemeris + "
                        "gravity caches.")

    # --- Models ---
    p.add_argument("--models", type=str, default="sh20,sh80,sh120,sh160,st_lrps")
    p.add_argument("--truth", type=str, default="sh200")
    p.add_argument("--include-st-lrps", action="store_true")
    p.add_argument("--st-lrps-model-dir", type=str, default=None)
    p.add_argument("--st-lrps-mode", choices=["cpu_dop853", "gpu_rk4"], default="cpu_dop853")
    p.add_argument("--st-lrps-rk4-dt", type=float, default=30.0)
    p.add_argument("--output-dir", type=str, default="results/gravity_validation_circular")

    # --- Full GPU batch comparison ---
    p.add_argument("--gpu-batch-compare", action="store_true",
                   help="Compare GPU RK4 SH/ST-LRPS models against SH200 DOP853 truth")
    p.add_argument("--gpu-models", type=str, default="sh200,sh160,sh120,sh60,sh20,st_lrps",
                   help="Comma-separated GPU fixed-step model list")
    p.add_argument("--gpu-integrator", choices=list(GPU_INTEGRATORS), default="medium",
                   help="GPU fixed-step integrator tier: light (RK2 midpoint), "
                        "medium (classic RK4, default), or robust (RK4 + Richardson "
                        "extrapolation).")
    p.add_argument("--batch-frame-mode",
                   choices=["match_dynamics_engine", "inertial_fixed_legacy"],
                   default="match_dynamics_engine",
                   help="Frame convention for GPU batch RK4")
    p.add_argument("--cache-truth", action="store_true",
                   help="Save SH200 DOP853 truth trajectories under output_dir/truth")
    p.add_argument("--reuse-truth-cache", action="store_true",
                   help="Reuse valid truth cache if metadata matches the current run")
    p.add_argument("--cache-trajectories", action="store_true",
                   help="Persist per-scenario truth and comparison-model trajectories "
                        "under benchmark_cache")
    p.add_argument("--reuse-cache", action="store_true",
                   help="Reuse compatible per-scenario benchmark_cache trajectories")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Optional benchmark cache directory. Default: output_dir/benchmark_cache")
    p.add_argument("--append-scenarios", type=int, default=0,
                   help="Append N new scenarios to an existing manifest")
    p.add_argument("--rebuild-metrics", action="store_true",
                   help="Rebuild metrics/reports from cached trajectories without propagation")
    p.add_argument("--strict-complete", action="store_true",
                   help="Fail metric rebuild if any selected model is missing scenarios")
    p.add_argument("--allow-lhs-append", action="store_true",
                   help="Allow blockwise LHS append; not equivalent to one global LHS design")
    p.add_argument("--require-st-lrps", action="store_true",
                   help="Fail if ST-LRPS is requested but no valid model directory is found")
    p.add_argument("--plot-theme", choices=["report_light", "technical_dark"], default="report_light")
    p.add_argument("--plot-error-logscale", action="store_true")
    p.add_argument("--plot-3d", action="store_true")
    p.add_argument("--plot-best-scenario-id", type=int, default=None)
    p.add_argument("--plot-worst-scenario-id", type=int, default=None)
    p.add_argument("--plot-representative-scenario-id", type=int, default=None)

    # --- Batch RK4 ---
    p.add_argument("--batch-rk4", action="store_true",
                   help="Run ST-LRPS as batched GPU/CPU fixed-step RK4 for all scenarios")
    p.add_argument("--batch-rk4-reference",
                   choices=["none", "sh200_rk4", "sh200_dop853_interpolated"],
                   default="sh200_dop853_interpolated",
                   help="Reference for batch RK4 error comparison")
    p.add_argument("--rk4-dt-s", type=float, default=None,
                   help="RK4 fixed step size (s). Default: --st-lrps-rk4-dt value.")
    p.add_argument("--gpu-rk4-dt-s-list", type=str, default=None,
                   help="Optional comma-separated RK4 step sizes to compare for each "
                        "GPU model, e.g. '10,30'. When more than one value is "
                        "provided, each model/step pair is treated as a separate "
                        "comparison series.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="GPU batch size (scenarios per pass). Default: all scenarios.")
    p.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32",
                   help="Torch dtype for GPU batch propagation. float32 is the default "
                        "for laptop/workstation throughput; choose float64 explicitly "
                        "when precision/runtime tradeoffs justify it.")
    p.add_argument("--gpu-fallback", choices=["error", "cpu"], default="error",
                   help="What to do when CUDA unavailable for batch RK4")
    p.add_argument("--save-batch-trajectories", action="store_true",
                   help="Save full batch trajectory NPZ (can be large)")

    # --- Force evaluation ---
    p.add_argument("--force-sample-trajectory", type=str, default=None)
    p.add_argument("--force-batch-size", type=int, default=8192)

    # --- Single orbit mode (backwards compat) ---
    p.add_argument("--altitude-km", type=float, default=200.0)
    p.add_argument("--ecc", type=float, default=0.0)
    p.add_argument("--inc-deg", type=float, default=90.0)
    p.add_argument("--raan-deg", type=float, default=0.0)
    p.add_argument("--argp-deg", type=float, default=0.0)
    p.add_argument("--ta-deg", type=float, default=0.0)

    return p.parse_args()


# =============================================================================
# Config helpers
# =============================================================================

def build_base_config(args: argparse.Namespace) -> SimConfig:
    cfg = load_default_config()
    new_time = replace(cfg.time,
        duration_s=args.duration_days * 86400.0,
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
    return replace(cfg, time=new_time, propagator=new_prop, flags=new_flags)


def _cfg_with_integrator(cfg: SimConfig, integrator: str) -> SimConfig:
    """Return a copy of ``cfg`` whose propagator uses ``integrator`` (e.g. RK45/DOP853).

    Used to let the ground-truth reference run a different adaptive integrator
    from the compared models without mutating the shared base config.
    """
    return replace(cfg, propagator=replace(cfg.propagator, method=str(integrator)))


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
        self._cache: Dict[str, Any] = {}

    def get(self, model_name: str) -> Any:
        if model_name not in self._cache:
            self._cache[model_name] = self._load(model_name)
        return self._cache[model_name]

    def _load(self, model_name: str) -> Any:
        if model_name.startswith("sh"):
            degree = int(model_name.replace("sh", ""))
            print(f"  [cache] Loading SH{degree} gravity model ...", flush=True)
            raw = GravityModel.from_file(self._cfg.gravity.file_path, requested_degree=degree)
            return adapt_gravity_model(raw)

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
                        )
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


# =============================================================================
# GPU batch comparison helpers
# =============================================================================

def _model_display_name(model_name: str) -> str:
    name = str(model_name).lower()
    base_name, dt_label = _split_gpu_variant_name(name)
    name = base_name
    if name == "st_lrps":
        base = "GPU_ST_LRPS_RK4"
        return f"{base}_DT{dt_label}" if dt_label else base
    if name.startswith("sh"):
        base = f"GPU_{name.upper()}_RK4"
        return f"{base}_DT{dt_label}" if dt_label else base
    return name.upper()


def _parse_model_list_csv(value: str) -> List[str]:
    return [m.strip().lower() for m in str(value).split(",") if m.strip()]


def _parse_float_list_csv(value: Optional[str]) -> List[float]:
    if value is None or str(value).strip() == "":
        return []
    out: List[float] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        val = float(raw)
        if val <= 0.0:
            raise ValueError("--gpu-rk4-dt-s-list values must be positive.")
        out.append(val)
    return out


def _format_rk4_dt_label(dt_s: float) -> str:
    return f"{float(dt_s):g}"


def _format_rk4_dt_token(dt_s: float) -> str:
    return _format_rk4_dt_label(dt_s).replace("-", "m").replace(".", "p")


def _split_gpu_variant_name(model_name: str) -> Tuple[str, Optional[str]]:
    name = str(model_name).lower()
    marker = "_rk4_dt"
    if marker not in name:
        return name, None
    base, token = name.split(marker, 1)
    token = token.strip("_")
    label = token.replace("m", "-").replace("p", ".")
    return base, label or None


def _gpu_variant_cache_name(model_name: str, rk4_dt_s: float, include_dt: bool) -> str:
    base = str(model_name).strip().lower()
    if not include_dt:
        return base
    return f"{base}_rk4_dt{_format_rk4_dt_token(rk4_dt_s)}"


def _gpu_rk4_dt_values(args: argparse.Namespace) -> List[float]:
    values = _parse_float_list_csv(getattr(args, "gpu_rk4_dt_s_list", None))
    if values:
        return values
    return [float(args.rk4_dt_s if args.rk4_dt_s is not None else args.st_lrps_rk4_dt)]


def _build_gpu_batch_tasks(gpu_models: List[str], args: argparse.Namespace) -> List[GpuBatchTask]:
    dt_values = _gpu_rk4_dt_values(args)
    include_dt = bool(str(getattr(args, "gpu_rk4_dt_s_list", "") or "").strip()) or len(dt_values) > 1
    tasks: List[GpuBatchTask] = []
    for model in gpu_models:
        for dt_s in dt_values:
            cache_name = _gpu_variant_cache_name(model, dt_s, include_dt)
            tasks.append(GpuBatchTask(
                model_name=str(model).lower(),
                cache_name=cache_name,
                display_name=_model_display_name(cache_name),
                rk4_dt_s=float(dt_s),
            ))
    return tasks


def _torch_dtype_from_name(dtype_name: str) -> Any:
    import torch
    return torch.float64 if str(dtype_name).lower() == "float64" else torch.float32


def _quat_rotate_torch(q: Any, v: Any) -> Any:
    """Rotate a batch of vectors by scalar-first quaternion q."""

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    tx = 2.0 * (q2 * vz - q3 * vy)
    ty = 2.0 * (q3 * vx - q1 * vz)
    tz = 2.0 * (q1 * vy - q2 * vx)
    cx = q2 * tz - q3 * ty
    cy = q3 * tx - q1 * tz
    cz = q1 * ty - q2 * tx
    return v + torch_stack_like(v, (q0 * tx + cx, q0 * ty + cy, q0 * tz + cz))


def torch_stack_like(reference: Any, cols: Tuple[Any, Any, Any]) -> Any:
    """Stack columns using the torch module that owns *reference*."""

    import torch
    return torch.stack(cols, dim=1).to(device=reference.device, dtype=reference.dtype)


class TorchFrameProvider:
    """
    Torch-side inertial/body-fixed frame provider for the batch RK4 path.

    ``match_dynamics_engine`` samples the same q_i2f table used by
    ``DynamicsEngine``.  Interpolation is normalized linear interpolation, which
    is close to SLERP for the small ephemeris cadence used here and keeps the
    GPU path free of host round-trips.
    """

    def __init__(self, ephem: Any, *, device: Any, dtype: Any, mode: str) -> None:
        import torch
        self.mode = str(mode)
        self.device = device
        self.dtype = dtype
        if self.mode == "inertial_fixed_legacy":
            self.dt_s = 1.0
            self.q_tab = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0, 0.0]],
                                      device=device, dtype=dtype)
            self.uses_rotation = False
            return

        if ephem is None:
            raise ValueError(
                "batch-frame-mode=match_dynamics_engine requires an EphemerisManager."
            )
        provider = ephem.get_data_provider()
        q_np = np.asarray(provider["q_i2f_tab"], dtype=np.float64)
        if q_np.ndim != 2 or q_np.shape[1] != 4:
            raise ValueError(f"q_i2f_tab must be shape (N,4), got {q_np.shape}")
        self.dt_s = float(provider["dt_s"])
        self.q_tab = torch.as_tensor(q_np, device=device, dtype=dtype)
        self.uses_rotation = True

    def quat_i2f(self, t_s: float) -> Any:
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
        q = torch.where(dot > 0.9995, q_linear, q_slerp)
        return q / torch.linalg.norm(q).clamp_min(1e-30)

    def inertial_to_fixed(self, t_s: float, r_i: Any) -> Any:
        if not self.uses_rotation:
            return r_i
        return _quat_rotate_torch(self.quat_i2f(t_s), r_i)

    def fixed_to_inertial(self, t_s: float, a_f: Any) -> Any:
        if not self.uses_rotation:
            return a_f
        q = self.quat_i2f(t_s).clone()
        q[1:] = -q[1:]
        return _quat_rotate_torch(q, a_f)


class TorchSHGravityEvaluator:
    """
    Torch vectorized spherical-harmonic gravity evaluator.

    The implementation mirrors the repository SH kernel recurrence but evaluates
    all scenarios in a position batch at once on the selected torch device.
    """

    def __init__(self, gravity_model: Any, *, degree: int, device: Any, dtype: Any) -> None:
        import torch
        self.degree = int(degree)
        self.device = device
        self.dtype = dtype
        self.backend = "torch_sh"
        self.r_ref = torch.tensor(float(getattr(gravity_model, "R_ref_m")), device=device, dtype=dtype)
        self.mu = torch.tensor(float(getattr(gravity_model, "GM_m3s2")), device=device, dtype=dtype)
        self.C = torch.as_tensor(np.array(getattr(gravity_model, "Cnm"), dtype=np.float64, copy=True),
                                 device=device, dtype=dtype)
        self.S = torch.as_tensor(np.array(getattr(gravity_model, "Snm"), dtype=np.float64, copy=True),
                                 device=device, dtype=dtype)
        self.diag = torch.as_tensor(np.array(getattr(gravity_model, "diag"), dtype=np.float64, copy=True),
                                    device=device, dtype=dtype)
        self.subdiag = torch.as_tensor(np.array(getattr(gravity_model, "subdiag"), dtype=np.float64, copy=True),
                                       device=device, dtype=dtype)
        self.A = torch.as_tensor(np.array(getattr(gravity_model, "A"), dtype=np.float64, copy=True),
                                 device=device, dtype=dtype)
        self.B = torch.as_tensor(np.array(getattr(gravity_model, "B"), dtype=np.float64, copy=True),
                                 device=device, dtype=dtype)
        scale_np = np.asarray(getattr(gravity_model, "scale_m"), dtype=np.float64)
        scale_pad = np.ones(self.degree + 2, dtype=np.float64)
        scale_pad[:min(scale_np.size, scale_pad.size)] = scale_np[:min(scale_np.size, scale_pad.size)]
        self.scale = torch.as_tensor(scale_pad, device=device, dtype=dtype)
        self.m_all = torch.arange(self.degree + 1, device=device, dtype=dtype)

    def acceleration(self, positions_fixed_m: Any) -> Any:
        import torch

        x = positions_fixed_m[:, 0]
        y = positions_fixed_m[:, 1]
        z = positions_fixed_m[:, 2]
        rho_sq = x * x + y * y
        r_sq = rho_sq + z * z
        r = torch.sqrt(r_sq).clamp_min(1.0)
        inv_r = 1.0 / r
        inv_r_sq = inv_r * inv_r
        rho = torch.sqrt(rho_sq)

        sin_phi = z * inv_r
        cos_phi = rho * inv_r
        pole = rho > 1e-12
        cos_lon = torch.where(pole, x / rho.clamp_min(1e-30), torch.ones_like(x))
        sin_lon = torch.where(pole, y / rho.clamp_min(1e-30), torch.zeros_like(y))

        u_r = positions_fixed_m * inv_r[:, None]
        u_phi = torch.stack(
            (-sin_phi * cos_lon, -sin_phi * sin_lon, cos_phi),
            dim=1,
        )

        batch_n = positions_fixed_m.shape[0]
        nmax = self.degree
        P = torch.zeros((batch_n, nmax + 1, nmax + 2), device=self.device, dtype=self.dtype)
        dP = torch.zeros_like(P)
        P[:, 0, 0] = 1.0

        for n in range(1, nmax + 1):
            P[:, n, n] = self.diag[n] * cos_phi * P[:, n - 1, n - 1]
            P[:, n, n - 1] = self.subdiag[n] * sin_phi * P[:, n - 1, n - 1]
            if n >= 2:
                m_slice = slice(0, n - 1)
                P[:, n, m_slice] = (
                    self.A[n, m_slice][None, :] * sin_phi[:, None] * P[:, n - 1, m_slice]
                    - self.B[n, m_slice][None, :] * P[:, n - 2, m_slice]
                )

            dP[:, n, 0] = math.sqrt(n * (n + 1.0)) * P[:, n, 1]
            if n >= 1:
                m = torch.arange(1, n + 1, device=self.device, dtype=self.dtype)
                coeff_minus = torch.sqrt((n + m) * (n - m + 1.0))
                term_minus = coeff_minus[None, :] * P[:, n, 0:n]
                term_plus = torch.zeros((batch_n, n), device=self.device, dtype=self.dtype)
                if n >= 2:
                    m2 = torch.arange(1, n, device=self.device, dtype=self.dtype)
                    coeff_plus = torch.sqrt((n - m2) * (n + m2 + 1.0))
                    term_plus[:, 0:n - 1] = coeff_plus[None, :] * P[:, n, 2:n + 1]
                dP[:, n, 1:n + 1] = 0.5 * (term_plus - term_minus)

        scale = self.scale[:nmax + 2]
        P = P * scale[None, None, :]
        dP = dP * scale[None, None, :]

        cos_m = torch.empty((batch_n, nmax + 1), device=self.device, dtype=self.dtype)
        sin_m = torch.empty_like(cos_m)
        cos_m[:, 0] = 1.0
        sin_m[:, 0] = 0.0
        if nmax >= 1:
            cos_m[:, 1] = cos_lon
            sin_m[:, 1] = sin_lon
        for m_i in range(2, nmax + 1):
            cos_m[:, m_i] = cos_m[:, m_i - 1] * cos_lon - sin_m[:, m_i - 1] * sin_lon
            sin_m[:, m_i] = sin_m[:, m_i - 1] * cos_lon + cos_m[:, m_i - 1] * sin_lon

        dv_dr = -self.mu * inv_r_sq
        dv_dphi = torch.zeros_like(dv_dr)
        dv_dlambda = torch.zeros_like(dv_dr)

        if nmax >= 2:
            r_ratio_base = self.r_ref * inv_r
            r_ratio_n = r_ratio_base * r_ratio_base
            mu_inv_r = self.mu * inv_r
            mu_inv_r_sq = self.mu * inv_r_sq
            for n in range(2, nmax + 1):
                sl = slice(0, n + 1)
                term_lon = self.C[n, sl][None, :] * cos_m[:, sl] + self.S[n, sl][None, :] * sin_m[:, sl]
                deriv_lon = -self.C[n, sl][None, :] * sin_m[:, sl] + self.S[n, sl][None, :] * cos_m[:, sl]
                m = self.m_all[sl]
                s_r = torch.sum(P[:, n, sl] * term_lon, dim=1)
                s_p = torch.sum(dP[:, n, sl] * term_lon, dim=1)
                s_l = torch.sum(m[None, :] * P[:, n, sl] * deriv_lon, dim=1)
                dv_dr = dv_dr - mu_inv_r_sq * (n + 1.0) * r_ratio_n * s_r
                dv_dphi = dv_dphi + mu_inv_r * r_ratio_n * s_p
                dv_dlambda = dv_dlambda + mu_inv_r * r_ratio_n * s_l
                r_ratio_n = r_ratio_n * r_ratio_base

        phi_factor = dv_dphi * inv_r
        inv_rho_sq = torch.where(rho_sq < 1e-24, torch.zeros_like(rho_sq), 1.0 / (rho_sq + 1e-24))
        ax = dv_dr * u_r[:, 0] + phi_factor * u_phi[:, 0] - dv_dlambda * y * inv_rho_sq
        ay = dv_dr * u_r[:, 1] + phi_factor * u_phi[:, 1] + dv_dlambda * x * inv_rho_sq
        az = dv_dr * u_r[:, 2] + phi_factor * u_phi[:, 2]
        return torch.stack((ax, ay, az), dim=1)


def _make_gpu_accelerator(model_name: str, gravity_model: Any, *, device: Any, dtype: Any) -> Tuple[Any, str]:
    """Create a batched torch acceleration provider for one GPU model."""

    name = str(model_name).lower()
    if name == "st_lrps":
        if str(getattr(gravity_model, "device", "")) != str(device):
            gravity_model.to_device(device)

        def _accel_st(pos_fixed: Any) -> Any:
            return gravity_model.predict_total_accel_torch(pos_fixed).to(device=device, dtype=dtype)

        return _accel_st, "torch_st_lrps"

    if name.startswith("sh"):
        degree = int(name.replace("sh", ""))
        evaluator = TorchSHGravityEvaluator(gravity_model, degree=degree, device=device, dtype=dtype)
        return evaluator.acceleration, evaluator.backend

    raise ValueError(f"Unsupported GPU batch model: {model_name!r}")


# GPU fixed-step integrators. The GPU batch path must use fixed-step schemes
# (no adaptive error control on-device), so three fidelity tiers are offered:
#   light  -> RK2 midpoint           (order 2, 2 RHS evals/step, cheapest)
#   medium -> classic RK4            (order 4, 4 RHS evals/step, default)
#   robust -> RK4 + Richardson       (local order ~6, 12 RHS evals/step, accurate)
# The helpers only use +, *, / and the rhs callable, so they are backend-agnostic
# (work on torch tensors or numpy arrays) and unit-testable without CUDA.
GPU_INTEGRATORS = ("light", "medium", "robust")


def _rk4_step(rhs, t, state, dt):
    """One classic fourth-order Runge-Kutta step."""
    k1 = rhs(t, state)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = rhs(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def gpu_fixed_step_advance(rhs, t, state, dt, method: str = "medium"):
    """Advance ``state`` by one ``dt`` using the selected fixed-step method.

    ``method`` is one of :data:`GPU_INTEGRATORS`. Unknown values fall back to the
    medium (classic RK4) scheme. The integrator advances by exactly ``dt`` per
    call; ``robust`` performs internal half-steps and Richardson extrapolation
    but still represents a single output step.
    """
    m = str(method).lower()
    if m == "light":  # midpoint RK2
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        return state + dt * k2
    if m == "robust":  # RK4 with one level of Richardson extrapolation
        full = _rk4_step(rhs, t, state, dt)
        half = _rk4_step(rhs, t, state, 0.5 * dt)
        half2 = _rk4_step(rhs, t + 0.5 * dt, half, 0.5 * dt)
        return (16.0 * half2 - full) / 15.0
    # medium / default: classic RK4
    return _rk4_step(rhs, t, state, dt)


def propagate_gpu_batch_model(
    model_name: str,
    gravity_model: Any,
    y0_batch: np.ndarray,
    duration_s: float,
    rk4_dt_s: float,
    output_dt_s: float,
    ephem: Any,
    *,
    device: Any,
    dtype: Any,
    dtype_name: str,
    frame_mode: str,
    gpu_integrator: str = "medium",
    progress_cb: Optional[Any] = None,
) -> BatchModelResult:
    """Propagate one model for all scenarios using a fixed-step torch integrator.

    ``gpu_integrator`` selects the fidelity tier (light/medium/robust); see
    :func:`gpu_fixed_step_advance`.

    ``progress_cb`` (optional) is invoked as ``cb(current_step, total_steps,
    elapsed_s)`` at step 0, on a throttled cadence during integration, and once
    more on successful completion. It is logging-only and never affects the
    numerical result.
    """

    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    if rk4_dt_s <= 0.0 or output_dt_s <= 0.0:
        raise ValueError("rk4_dt_s and output_dt_s must be positive.")
    if rk4_dt_s > output_dt_s:
        print(f"[gpu-batch] WARNING: rk4_dt_s={rk4_dt_s} > output_dt_s={output_dt_s}; "
              "using output_dt_s as the effective RK4 step.", flush=True)
        rk4_dt_s = output_dt_s
    steps_per_snap = max(1, round(output_dt_s / rk4_dt_s))
    dt_eff = output_dt_s / steps_per_snap
    frac = output_dt_s / rk4_dt_s
    if abs(frac - round(frac)) > 1e-6:
        print(f"[gpu-batch] WARNING: output_dt_s={output_dt_s} is not divisible by "
              f"rk4_dt_s={rk4_dt_s}; effective dt={dt_eff:.6f}s.", flush=True)

    n_scenarios = int(y0_batch.shape[0])
    n_snaps = max(1, round(duration_s / output_dt_s))
    t_out = np.linspace(0.0, n_snaps * output_dt_s, n_snaps + 1, dtype=np.float64)

    frame = TorchFrameProvider(ephem, device=device, dtype=dtype, mode=frame_mode)
    accel_fixed, backend = _make_gpu_accelerator(model_name, gravity_model, device=device, dtype=dtype)

    state = torch.as_tensor(y0_batch, device=device, dtype=dtype)
    y_gpu = torch.empty((n_snaps + 1, n_scenarios, 6), device=device, dtype=dtype)
    y_gpu[0].copy_(state)

    def _rhs(t_s: float, s: Any) -> Any:
        r_i = s[:, :3]
        v_i = s[:, 3:]
        r_f = frame.inertial_to_fixed(t_s, r_i)
        a_f = accel_fixed(r_f)
        a_i = frame.fixed_to_inertial(t_s, a_f)
        return torch.cat((v_i, a_i), dim=1)

    total_steps = int(n_snaps * steps_per_snap)
    throttle = progress.StepThrottle(total_steps)

    def _emit_progress(step: int, elapsed: float) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(int(step), total_steps, float(elapsed))
        except Exception:
            pass

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_curr = 0.0
    status = "ok"
    failure_reason = ""
    step_count = 0
    bad_state = torch.zeros((), device=device, dtype=torch.bool)
    _emit_progress(0, 0.0)

    try:
        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                state = gpu_fixed_step_advance(_rhs, t_curr, state, dt_eff, gpu_integrator)
                t_curr += dt_eff
                step_count += 1
                bad_state.logical_or_(~torch.isfinite(state).all())
                now = time.perf_counter()
                if throttle.update(step_count, now):
                    _emit_progress(step_count, now - t0)
            y_gpu[snap_idx + 1].copy_(state)
    except Exception as exc:
        status = "failed"
        failure_reason = str(exc)
        print(f"[gpu-batch] {model_name.upper()} failed: {exc}", flush=True)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    if status == "ok" and bool(bad_state.detach().cpu().item()):
        status = "failed"
        failure_reason = f"non-finite state in {model_name}"
        print(f"[gpu-batch] {model_name.upper()} failed: {failure_reason}", flush=True)
    y_out = y_gpu.detach().cpu().numpy().astype(np.float64, copy=False)
    runtime_s = time.perf_counter() - t0
    n_steps = n_snaps * steps_per_snap
    if status == "ok":
        _emit_progress(total_steps, runtime_s)
    return BatchModelResult(
        model_name=str(model_name).lower(),
        display_name=_model_display_name(model_name),
        backend=backend,
        device=str(device),
        dtype=str(dtype_name),
        t=t_out,
        y=y_out,
        runtime_s=float(runtime_s),
        n_steps=int(n_steps),
        n_scenarios=n_scenarios,
        rk4_dt_s=float(dt_eff),
        output_dt_s=float(output_dt_s),
        status=status,
        failure_reason=failure_reason,
    )


# =============================================================================
# Scenario generation
# =============================================================================

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _sobol_note(method: str, n: int) -> str:
    if str(method).startswith("sobol") and n > 0 and not _is_power_of_two(int(n)):
        return ("Sobol sequences have their strongest balance properties when "
                "the scenario count is a power of two; this run generated the "
                "next power-of-two sequence and truncated to the requested count.")
    return ""


def _require_qmc():
    try:
        from scipy.stats import qmc
    except Exception as exc:
        raise ImportError("scipy.stats.qmc is required for LHS/Sobol sampling.") from exc
    return qmc


def generate_unit_samples(
    n: int,
    dim: int,
    method: str,
    seed: int,
) -> np.ndarray:
    """Generate unit-hypercube samples for scenario construction."""

    n = int(n)
    dim = int(dim)
    method = str(method)
    if n < 0:
        raise ValueError("n must be non-negative")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if n == 0:
        return np.empty((0, dim), dtype=np.float64)
    if method not in SAMPLING_METHODS:
        raise ValueError(f"Unknown sampling method: {method}")

    if method == "random":
        return np.asarray(np.random.default_rng(int(seed)).random((n, dim)), dtype=np.float64)

    qmc = _require_qmc()
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=dim, seed=int(seed))
        return np.asarray(sampler.random(n), dtype=np.float64)

    scramble = method == "sobol_scrambled"
    sampler = qmc.Sobol(d=dim, scramble=scramble, seed=int(seed) if scramble else None)
    if _is_power_of_two(n):
        m = int(math.log2(n))
        samples = sampler.random_base2(m=m)
    else:
        # Generate a balanced Sobol block and truncate so arbitrary N is allowed
        # without changing the requested scenario count.
        m = int(math.log2(_next_power_of_two(n)))
        samples = sampler.random_base2(m=m)[:n]
    return np.asarray(samples, dtype=np.float64)


def _map_unit_linear(u: float, lo: float, hi: float) -> float:
    return float(lo + float(u) * (hi - lo))


def _map_inclination_deg(u: float, args: argparse.Namespace) -> float:
    inc_min = float(args.inc_min_deg)
    inc_max = float(args.inc_max_deg)
    if str(getattr(args, "inclination_sampling", "uniform_deg")) == "uniform_cos":
        cos_i_min = math.cos(math.radians(inc_max))
        cos_i_max = math.cos(math.radians(inc_min))
        cos_i = _map_unit_linear(float(u), cos_i_min, cos_i_max)
        cos_i = max(-1.0, min(1.0, cos_i))
        return float(math.degrees(math.acos(cos_i)))
    return _map_unit_linear(float(u), inc_min, inc_max)


def _validate_sampling_bounds(args: argparse.Namespace) -> None:
    if float(args.altitude_min_km) > float(args.altitude_max_km):
        raise ValueError("--altitude-min-km must be <= --altitude-max-km")
    if float(args.inc_min_deg) > float(args.inc_max_deg):
        raise ValueError("--inc-min-deg must be <= --inc-max-deg")
    if str(args.scenario_mode) == "near_circular_altitude":
        if float(args.ecc_min) < 0.0 or float(args.ecc_max) >= 1.0:
            raise ValueError("near_circular_altitude requires 0 <= --ecc-min <= --ecc-max < 1")
        if float(args.ecc_min) > float(args.ecc_max):
            raise ValueError("--ecc-min must be <= --ecc-max")


def _state_from_elements(
    a_m: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    ta_deg: float,
) -> np.ndarray:
    return create_state_from_keplerian(
        semi_major_axis=float(a_m),
        eccentricity=float(e),
        inclination=math.radians(float(inc_deg)),
        raan=math.radians(float(raan_deg)),
        argp=math.radians(float(argp_deg)),
        true_anomaly=math.radians(float(ta_deg)),
        mu=MU_MOON,
    ).y


def generate_scenarios_from_samples(
    samples: np.ndarray,
    args: argparse.Namespace,
) -> List[Scenario]:
    """Map unit-hypercube samples into validation scenarios without propagation."""

    _validate_sampling_bounds(args)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array")
    if samples.shape[1] < SCENARIO_UNIT_DIM:
        raise ValueError(f"samples must have at least {SCENARIO_UNIT_DIM} columns")

    scenarios: List[Scenario] = []
    moon_r_km = float(R_MOON) / 1_000.0
    for sid, u in enumerate(samples):
        raw = [float(x) for x in u[:SCENARIO_UNIT_DIM]]
        if str(args.scenario_mode) == "bounded_keplerian":
            raw_alt_1 = _map_unit_linear(raw[0], float(args.altitude_min_km), float(args.altitude_max_km))
            raw_alt_2 = _map_unit_linear(raw[1], float(args.altitude_min_km), float(args.altitude_max_km))
            hp_km = min(raw_alt_1, raw_alt_2)
            ha_km = max(raw_alt_1, raw_alt_2)
            rp_km = moon_r_km + hp_km
            ra_km = moon_r_km + ha_km
            a_km = 0.5 * (rp_km + ra_km)
            e = (ra_km - rp_km) / (ra_km + rp_km)
            inc_u, raan_u, argp_u, ta_u = raw[2], raw[3], raw[4], raw[5]
        else:
            alt_km = _map_unit_linear(raw[0], float(args.altitude_min_km), float(args.altitude_max_km))
            e = _map_unit_linear(raw[1], float(args.ecc_min), float(args.ecc_max))
            a_km = moon_r_km + alt_km
            if abs(e) <= 1e-15:
                hp_km = alt_km
                ha_km = alt_km
            else:
                hp_km = a_km * (1.0 - e) - moon_r_km
                ha_km = a_km * (1.0 + e) - moon_r_km
            inc_u, raan_u, argp_u, ta_u = raw[2], raw[3], raw[4], raw[5]

        if hp_km > ha_km:
            hp_km, ha_km = ha_km, hp_km
        if e < 0.0 or e >= 1.0:
            raise ValueError(f"Generated invalid eccentricity for scenario {sid}: {e}")
        if a_km <= moon_r_km:
            raise ValueError(f"Generated invalid semi-major axis for scenario {sid}: {a_km} km")

        inc_deg = _map_inclination_deg(inc_u, args)
        raan_deg = _map_unit_linear(raan_u, float(args.raan_min_deg), float(args.raan_max_deg))
        argp_deg = _map_unit_linear(argp_u, float(args.argp_min_deg), float(args.argp_max_deg))
        ta_deg = _map_unit_linear(ta_u, float(args.ta_min_deg), float(args.ta_max_deg))
        state = _state_from_elements(a_km * 1_000.0, e, inc_deg, raan_deg, argp_deg, ta_deg)
        if not np.isfinite(state).all():
            raise ValueError(f"Generated non-finite initial state for scenario {sid}")

        scenarios.append(Scenario(
            scenario_id=sid,
            hp_km=float(hp_km),
            ha_km=float(ha_km),
            a_km=float(a_km),
            e=float(e),
            inc_deg=float(inc_deg),
            raan_deg=float(raan_deg),
            argp_deg=float(argp_deg),
            ta_deg=float(ta_deg),
            initial_state=state,
            raw_unit_sample=raw,
            sampling_method=str(getattr(args, "sampling_method", "random")),
        ))
    return scenarios


def generate_random_scenarios(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[Scenario]:
    n = args.random_scenarios
    alt_min = args.altitude_min_km
    alt_max = args.altitude_max_km
    inc_min = math.radians(args.inc_min_deg)
    inc_max = math.radians(args.inc_max_deg)

    scenarios: List[Scenario] = []
    attempts = 0
    max_attempts = n * 20

    while len(scenarios) < n and attempts < max_attempts:
        attempts += 1
        sid = len(scenarios)

        try:
            if args.scenario_mode == "bounded_keplerian":
                hp_km = float(rng.uniform(alt_min, max(alt_min + 1.0, alt_max * 0.7)))
                ha_km = float(rng.uniform(hp_km, alt_max))
                a_m, e = calculate_ae_from_altitudes(R_MOON, hp_km, ha_km)
            else:  # near_circular_altitude
                alt_km = float(rng.uniform(alt_min, alt_max))
                e      = float(rng.uniform(args.ecc_min, args.ecc_max))
                a_m    = R_MOON + alt_km * 1_000.0
                if abs(e) <= 1e-15:
                    hp_km = alt_km
                    ha_km = alt_km
                else:
                    hp_km = (a_m * (1.0 - e) - R_MOON) / 1_000.0
                    ha_km = (a_m * (1.0 + e) - R_MOON) / 1_000.0

            if e < 0.0 or e >= 1.0:
                continue
            if a_m <= R_MOON:
                continue

            if str(getattr(args, "inclination_sampling", "uniform_deg")) == "uniform_cos":
                inc_deg = _map_inclination_deg(float(rng.random()), args)
            else:
                inc_deg  = float(math.degrees(rng.uniform(inc_min, inc_max)))
            raan_deg = float(rng.uniform(args.raan_min_deg, args.raan_max_deg))
            argp_deg = float(rng.uniform(args.argp_min_deg, args.argp_max_deg))
            ta_deg   = float(rng.uniform(args.ta_min_deg, args.ta_max_deg))

            state = _state_from_elements(a_m, e, inc_deg, raan_deg, argp_deg, ta_deg)

            if not np.isfinite(state).all():
                continue

            scenarios.append(Scenario(
                scenario_id=sid,
                hp_km=hp_km,
                ha_km=ha_km,
                a_km=a_m / 1_000.0,
                e=e,
                inc_deg=inc_deg,
                raan_deg=raan_deg,
                argp_deg=argp_deg,
                ta_deg=ta_deg,
                initial_state=state,
                sampling_method="random",
            ))
        except Exception:
            continue

    if len(scenarios) < n:
        print(f"WARNING: only generated {len(scenarios)}/{n} valid scenarios "
              f"after {attempts} attempts")

    return scenarios


def generate_validation_scenarios(args: argparse.Namespace) -> List[Scenario]:
    method = str(getattr(args, "sampling_method", "random"))
    if method == "random":
        rng = np.random.default_rng(args.scenario_seed)
        return generate_random_scenarios(args, rng)
    samples = generate_unit_samples(
        int(args.random_scenarios),
        SCENARIO_UNIT_DIM,
        method,
        int(args.scenario_seed),
    )
    return generate_scenarios_from_samples(samples, args)


# =============================================================================
# DOP853 propagation
# =============================================================================

def propagate_for_scenario(
    model_name: str,
    y0: np.ndarray,
    args: argparse.Namespace,
    cfg_base: SimConfig,
    ephem: Any,
    model_cache: GravityModelCache,
) -> Tuple[Optional[Any], float]:
    """Propagate with the named model. Returns (PropagationResult|None, runtime_s)."""
    grav = model_cache.get(model_name)
    cfg  = cfg_base

    if model_name == "st_lrps":
        # ST-LRPS has degree_max=200 (training target), which causes the Nyquist
        # criterion to demand ~5s steps.  We disable Nyquist by:
        # 1. Setting use_nyquist_max_step=False in PropagatorConfig
        # 2. Temporarily overriding grav.degree_max = grav.degree_min so that
        #    _get_sh_degree() returns the base degree (e.g. 10) as a belt-and-suspenders.
        new_prop = replace(cfg_base.propagator, use_nyquist_max_step=False)
        cfg = replace(cfg_base, propagator=new_prop)

        # Belt-and-suspenders: temporarily lower degree_max on the surrogate
        _orig_dmax = getattr(grav, "degree_max", 200)
        _base_deg  = max(1, int(getattr(grav, "degree_min", 20)))
        try:
            grav.degree_max = _base_deg
        except Exception:
            pass

    dyn = DynamicsEngine(
        sc_props=cfg.spacecraft,
        flags=cfg.flags,
        gravity_model=grav,
        ephem_manager=ephem,
        allow_identity_rotation=True,
    )
    t0 = time.perf_counter()
    try:
        res = propagate(dyn, y0, cfg.propagator, time_cfg=cfg.time)
    except Exception as exc:
        print(f"    ERROR propagating {model_name}: {exc}", flush=True)
        res = None
    finally:
        # Restore degree_max on surrogate
        if model_name == "st_lrps":
            try:
                grav.degree_max = _orig_dmax
            except Exception:
                pass
    rt = time.perf_counter() - t0

    if res is None or (res.ode is not None and not res.ode.success):
        return None, rt
    return res, rt


# =============================================================================
# Batched force evaluation (ST-LRPS)
# =============================================================================

def evaluate_st_lrps_forces_batched(
    model: Any,
    positions_m: np.ndarray,   # (N, 3) body-fixed
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Batch evaluate ST-LRPS acceleration at N body-fixed positions.
    Returns (N, 3) in m/s^2.
    """
    N = positions_m.shape[0]
    result = np.empty((N, 3), dtype=np.float64)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        chunk = positions_m[start:end]
        result[start:end] = model.acceleration_fixed_batch(chunk)

    return result


def _synchronize_model_device_if_cuda(model: Any) -> None:
    """Synchronize CUDA timing when *model* is resident on GPU."""

    dev = str(getattr(model, "device", "") or "").lower()
    if "cuda" not in dev:
        return
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        pass


# =============================================================================
# Batch GPU/CPU RK4 for ST-LRPS
# =============================================================================

class _BatchMCCfg:
    """Minimal mc_cfg duck-type for TorchBatchPropagator."""
    def __init__(self, dt_s: float, impact_alt_km: float = 0.0, torch_dtype: str = "float32") -> None:
        self.dt_s = float(dt_s)
        self.impact_alt_km = float(impact_alt_km)
        self.torch_dtype = str(torch_dtype)


def run_st_lrps_batch_rk4(
    surrogate_model: Any,
    y0_batch: np.ndarray,          # (N, 6) SI
    duration_s: float,
    dt_s: float,
    output_dt_s: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run ST-LRPS fixed-step RK4 for N scenarios.

    GPU path (CUDA available): uses core.torch_batch_propagator.TorchBatchPropagator.
    CPU fallback: sequential numpy batch RK4 using acceleration_fixed_batch.

    NOTE: Both paths evaluate gravity in the inertial frame without applying the
    Moon's rotation matrix (body-fixed != inertial approximation). This matches the
    TorchBatchPropagator's existing contract and is acceptable for short durations.
    """
    requested_chunk = getattr(args, "batch_size", None)
    if requested_chunk is not None and int(requested_chunk) > 0 and y0_batch.shape[0] > int(requested_chunk):
        chunk_size = int(requested_chunk)
        print(f"[batch-rk4] Splitting {y0_batch.shape[0]} scenarios into chunks of {chunk_size}.",
              flush=True)
        child_args = argparse.Namespace(**vars(args))
        child_args.batch_size = None
        chunk_results: List[Dict[str, Any]] = []
        for start in range(0, y0_batch.shape[0], chunk_size):
            end = min(start + chunk_size, y0_batch.shape[0])
            print(f"[batch-rk4] chunk {start}:{end}", flush=True)
            chunk_results.append(run_st_lrps_batch_rk4(
                surrogate_model,
                y0_batch[start:end],
                duration_s=duration_s,
                dt_s=dt_s,
                output_dt_s=output_dt_s,
                args=child_args,
            ))

        t_ref = chunk_results[0]["t"]
        if any(len(r["t"]) != len(t_ref) or np.max(np.abs(r["t"] - t_ref)) > 1e-6 for r in chunk_results):
            raise RuntimeError("Chunked RK4 runs produced inconsistent output time grids.")
        Y = np.concatenate([r["Y"] for r in chunk_results], axis=1)
        impact_flags = np.concatenate([r.get("impact_flags", np.zeros(r["Y"].shape[1])) for r in chunk_results])
        t_impact = np.concatenate([r.get("t_impact", np.full(r["Y"].shape[1], np.nan)) for r in chunk_results])
        runtime_s = float(sum(float(r.get("runtime_s", 0.0)) for r in chunk_results))
        n_steps = int(chunk_results[0].get("n_steps", 0))
        n_scenarios = int(y0_batch.shape[0])
        return {
            "t": t_ref,
            "Y": Y,
            "impact_flags": impact_flags,
            "t_impact": t_impact,
            "runtime_s": runtime_s,
            "device": chunk_results[0].get("device", "?"),
            "dt_s": float(chunk_results[0].get("dt_s", dt_s)),
            "n_scenarios": n_scenarios,
            "n_steps": n_steps,
            "samples_per_second": n_scenarios * n_steps / max(runtime_s, 1e-9),
            "mode": str(chunk_results[0].get("mode", "chunked_rk4")),
            "chunk_size": chunk_size,
            "torch_dtype": getattr(args, "torch_dtype", "float64"),
            "y_layout": "time_scenario_state",
        }

    # Warn if dt > output_dt
    if dt_s > output_dt_s:
        print(f"[batch-rk4] WARNING: rk4-dt ({dt_s}s) > dt-out ({output_dt_s}s); "
              f"clamping to dt_s = output_dt_s.", flush=True)
        dt_s = output_dt_s

    steps_per_snap = max(1, round(output_dt_s / dt_s))
    dt_eff = output_dt_s / steps_per_snap
    frac = output_dt_s / dt_s
    if abs(frac - round(frac)) > 0.01:
        print(f"[batch-rk4] WARNING: output_dt ({output_dt_s}s) not divisible by "
              f"rk4-dt ({dt_s}s). Effective dt = {dt_eff:.3f}s.", flush=True)

    # Try GPU path
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    if cuda_ok:
        return _run_batch_rk4_gpu(surrogate_model, y0_batch, duration_s, dt_eff,
                                   output_dt_s, args)
    else:
        fallback = getattr(args, "gpu_fallback", "cpu")
        if fallback == "error":
            raise RuntimeError(
                "GPU batch RK4 requested but CUDA is unavailable. "
                "Use --gpu-fallback cpu."
            )
        print("[batch-rk4] CUDA unavailable; using CPU batch RK4.", flush=True)
        return _run_batch_rk4_cpu(surrogate_model, y0_batch, duration_s, dt_eff, output_dt_s)


def _run_batch_rk4_gpu(
    surrogate_model: Any,
    y0_batch: np.ndarray,
    duration_s: float,
    dt_s: float,
    output_dt_s: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import torch
    from core.torch_batch_propagator import TorchBatchPropagator

    device = torch.device("cuda:0")
    dev_name = torch.cuda.get_device_name(0)

    # Move model to GPU if needed
    if str(surrogate_model.device) != str(device):
        surrogate_model.to_device(device)

    mc_cfg = _BatchMCCfg(dt_s=dt_s, torch_dtype=getattr(args, "torch_dtype", "float32"))
    prop   = TorchBatchPropagator(surrogate_model, mc_cfg, device_id=0)
    N      = y0_batch.shape[0]
    ones_N = np.ones(N, dtype=np.float64)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_out, Y_out, impact_flags, t_impact = prop.propagate(
        y0_batch, ones_N, ones_N, ones_N, ones_N,
        duration_s=duration_s,
        output_dt_s=output_dt_s,
    )
    torch.cuda.synchronize()
    runtime_s = time.perf_counter() - t0

    n_snaps = Y_out.shape[0] - 1
    n_steps = n_snaps * max(1, round(output_dt_s / dt_s))
    print(f"[batch-rk4] GPU done: {runtime_s:.2f}s  device={device} ({dev_name})  "
          f"scenarios={N}  steps={n_steps}  "
          f"throughput={N * n_steps / max(runtime_s, 1e-9):,.0f} traj-steps/s",
          flush=True)

    return {
        "t": t_out, "Y": Y_out,
        "impact_flags": impact_flags, "t_impact": t_impact,
        "runtime_s": runtime_s, "device": f"cuda:0 ({dev_name})",
        "dt_s": dt_s, "n_scenarios": N,
        "n_steps": n_steps,
        "samples_per_second": N * n_steps / max(runtime_s, 1e-9),
        "mode": "gpu_rk4",
        "torch_dtype": getattr(args, "torch_dtype", "float64"),
        "y_layout": "time_scenario_state",
    }


def _run_batch_rk4_cpu(
    surrogate_model: Any,
    y0_batch: np.ndarray,
    duration_s: float,
    dt_s: float,
    output_dt_s: float,
) -> Dict[str, Any]:
    """CPU sequential batch RK4 using acceleration_fixed_batch."""
    N = y0_batch.shape[0]
    steps_per_snap = max(1, round(output_dt_s / dt_s))
    n_snaps = max(1, round(duration_s / output_dt_s))
    t_out = np.linspace(0.0, n_snaps * output_dt_s, n_snaps + 1)
    Y_out = np.empty((n_snaps + 1, N, 6), dtype=np.float64)

    def _batch_accel(Y: np.ndarray) -> np.ndarray:
        return surrogate_model.acceleration_fixed_batch(Y[:, :3])

    def _rhs(Y: np.ndarray) -> np.ndarray:
        a = _batch_accel(Y)
        return np.concatenate([Y[:, 3:], a], axis=1)

    t0 = time.perf_counter()
    Y_curr = y0_batch.copy()
    Y_out[0] = Y_curr

    for snap_idx in range(n_snaps):
        for _ in range(steps_per_snap):
            k1 = _rhs(Y_curr)
            k2 = _rhs(Y_curr + 0.5 * dt_s * k1)
            k3 = _rhs(Y_curr + 0.5 * dt_s * k2)
            k4 = _rhs(Y_curr + dt_s * k3)
            Y_curr = Y_curr + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.isfinite(Y_curr).all():
                print("[batch-rk4] WARNING: non-finite state detected; "
                      "replacing with last good state.", flush=True)
                Y_curr = np.where(np.isfinite(Y_curr), Y_curr, Y_out[snap_idx])
                break
        Y_out[snap_idx + 1] = Y_curr
        if (snap_idx + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  [CPU-RK4] snap {snap_idx+1}/{n_snaps}  {el:.1f}s elapsed", flush=True)

    runtime_s = time.perf_counter() - t0
    n_steps = n_snaps * steps_per_snap
    print(f"[batch-rk4] CPU done: {runtime_s:.2f}s  scenarios={N}  steps={n_steps}  "
          f"throughput={N * n_steps / max(runtime_s, 1e-9):,.0f} traj-steps/s", flush=True)

    return {
        "t": t_out, "Y": Y_out,
        "impact_flags": np.zeros(N), "t_impact": np.full(N, np.nan),
        "runtime_s": runtime_s, "device": "cpu",
        "dt_s": dt_s, "n_scenarios": N, "n_steps": n_steps,
        "samples_per_second": N * n_steps / max(runtime_s, 1e-9),
        "mode": "cpu_rk4",
        "torch_dtype": "numpy_float64",
        "y_layout": "time_scenario_state",
    }


# =============================================================================
# SH200 CPU RK4 reference (for error decomposition)
# =============================================================================

def run_sh200_cpu_rk4_reference(
    grav: Any,                 # GravityModel (SH200)
    y0_batch: np.ndarray,     # (N, 6)
    duration_s: float,
    dt_s: float,
    output_dt_s: float,
) -> Dict[str, Any]:
    """
    Run SH200 fixed-step CPU RK4 for N scenarios sequentially.
    Gravity evaluated WITHOUT lunar rotation (same approximation as GPU batch RK4)
    so that the two can be directly subtracted for error decomposition.
    """
    from models.spherical_harmonics import sh_accel_fixed_numba

    N = y0_batch.shape[0]
    steps_per_snap = max(1, round(output_dt_s / dt_s))
    n_snaps = max(1, round(duration_s / output_dt_s))
    t_out = np.linspace(0.0, n_snaps * output_dt_s, n_snaps + 1)
    Y_out = np.empty((n_snaps + 1, N, 6), dtype=np.float64)

    # Pre-allocate workspace once
    ws = grav.make_workspace()

    def sh_accel(r: np.ndarray) -> np.ndarray:
        ax, ay, az = sh_accel_fixed_numba(
            float(r[0]), float(r[1]), float(r[2]),
            grav.max_degree, grav.r_ref, grav.mu,
            grav.c_coeffs, grav.s_coeffs,
            grav.diag_coeffs, grav.subdiag_coeffs,
            grav.a_coeffs, grav.b_coeffs, grav.scale_m_table,
            ws.P, ws.dP, ws.cos_m, ws.sin_m,
        )
        return np.array([ax, ay, az], dtype=np.float64)

    def rhs(y: np.ndarray) -> np.ndarray:
        r, v = y[:3], y[3:]
        return np.concatenate([v, sh_accel(r)])

    t0 = time.perf_counter()
    for i in range(N):
        state = y0_batch[i].copy()
        Y_out[0, i] = state
        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                k1 = rhs(state)
                k2 = rhs(state + 0.5 * dt_s * k1)
                k3 = rhs(state + 0.5 * dt_s * k2)
                k4 = rhs(state + dt_s * k3)
                state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Y_out[snap_idx + 1, i] = state
        if (i + 1) % 10 == 0:
            el = time.perf_counter() - t0
            rate = (i + 1) / el
            eta  = (N - i - 1) / max(rate, 1e-9)
            print(f"  [SH200-RK4] {i+1}/{N}  {el:.1f}s  ETA {eta:.0f}s", flush=True)

    runtime_s = time.perf_counter() - t0
    n_steps = n_snaps * steps_per_snap
    print(f"[SH200-RK4] done: {runtime_s:.2f}s  scenarios={N}", flush=True)
    return {"t": t_out, "Y": Y_out, "runtime_s": runtime_s, "device": "cpu",
            "dt_s": dt_s, "n_scenarios": N, "n_steps": n_steps,
            "samples_per_second": N * n_steps / max(runtime_s, 1e-9)}


# =============================================================================
# DOP853 trajectory metrics
# =============================================================================

def compute_trajectory_metrics(
    model_name: str,
    scenario: Scenario,
    truth_res: Any,
    model_res: Any,
    model_runtime_s: float,
    truth_runtime_s: float,
) -> Dict[str, Any]:
    t_ref  = truth_res.t
    r_ref  = truth_res.y[:, :3]
    v_ref  = truth_res.y[:, 3:6]

    # Use time-grid interpolation instead of raw index truncation
    y_model = interpolate_state_to_times(model_res.t, model_res.y, t_ref)
    r_test  = y_model[:, :3]
    v_test  = y_model[:, 3:6]

    # Validation
    if not np.isfinite(r_test).all():
        return {f: None for f in _METRICS_FIELDNAMES} | {
            "scenario_id": scenario.scenario_id,
            "model": model_name, "status": "failed_nonfinite",
        }

    dr    = r_test - r_ref
    dv    = v_test - v_ref
    dr_km = np.linalg.norm(dr, axis=1) / 1_000.0
    dv_ms = np.linalg.norm(dv, axis=1)

    ric_km = compute_ric_errors(r_ref, v_ref, r_test) / 1_000.0

    alt_truth_km = (np.linalg.norm(r_ref,  axis=1) - R_MOON) / 1_000.0
    alt_model_km = (np.linalg.norm(r_test, axis=1) - R_MOON) / 1_000.0
    alt_err_km   = alt_model_km - alt_truth_km

    if np.any(alt_model_km < 0):
        print(f"    WARNING: model {model_name} altitude went negative "
              f"(min {np.min(alt_model_km):.2f} km)", flush=True)

    return {
        "scenario_id": scenario.scenario_id,
        "model":       model_name,
        "runtime_s":   round(model_runtime_s, 4),
        "runtime_rel_to_truth": round(model_runtime_s / max(truth_runtime_s, 1e-9), 4),
        "rms_pos_err_km":   float(np.sqrt(np.mean(dr_km ** 2))),
        "final_pos_err_km": float(dr_km[-1]),
        "max_pos_err_km":   float(np.max(dr_km)),
        "p95_pos_err_km":   float(np.percentile(dr_km, 95)),
        "rms_vel_err_ms":   float(np.sqrt(np.mean(dv_ms ** 2))),
        "final_vel_err_ms": float(dv_ms[-1]),
        "max_vel_err_ms":   float(np.max(dv_ms)),
        "p95_vel_err_ms":   float(np.percentile(dv_ms, 95)),
        "radial_rms_km":    float(np.sqrt(np.mean(ric_km[:, 0] ** 2))),
        "along_rms_km":     float(np.sqrt(np.mean(ric_km[:, 1] ** 2))),
        "cross_rms_km":     float(np.sqrt(np.mean(ric_km[:, 2] ** 2))),
        "radial_max_km":    float(np.max(np.abs(ric_km[:, 0]))),
        "along_max_km":     float(np.max(np.abs(ric_km[:, 1]))),
        "cross_max_km":     float(np.max(np.abs(ric_km[:, 2]))),
        "final_alt_err_km":    float(alt_err_km[-1]),
        "rms_alt_err_km":      float(np.sqrt(np.mean(alt_err_km ** 2))),
        "max_abs_alt_err_km":  float(np.max(np.abs(alt_err_km))),
        "min_alt_model_km":    float(np.min(alt_model_km)),
        "min_alt_truth_km":    float(np.min(alt_truth_km)),
        "status": "ok",
    }


# =============================================================================
# Batch RK4 metrics
# =============================================================================

def compute_batch_rk4_metrics(
    batch_result: Dict[str, Any],
    truth_results: List[Optional[Any]],   # DOP853 truth per scenario
    scenarios: List[Scenario],
    sh200_rk4_result: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Returns:
        total_rows   : st_lrps_rk4 vs sh200_dop853
        model_rows   : st_lrps_rk4 vs sh200_rk4 (empty if no sh200_rk4_result)
        integr_rows  : sh200_rk4 vs sh200_dop853 (empty if no sh200_rk4_result)
    """
    t_batch = batch_result["t"]    # (T,)
    Y_batch = batch_result["Y"]    # (T, N, 6)

    total_rows: List[Dict] = []
    model_rows: List[Dict] = []
    integr_rows: List[Dict] = []

    for i, (scenario, truth_res) in enumerate(zip(scenarios, truth_results)):
        if truth_res is None:
            continue

        y_stlrps = Y_batch[:, i, :]    # (T, 6)

        # Validate
        if not np.isfinite(y_stlrps).all():
            total_rows.append({
                "scenario_id": scenario.scenario_id, "model": "st_lrps_batch_rk4",
                "reference": "sh200_dop853", "status": "failed_nonfinite",
                **{k: np.nan for k in _BATCH_METRICS_FIELDNAMES
                   if k not in ("scenario_id","model","reference","status")},
            })
            continue

        # Interpolate SH200 DOP853 to batch time grid
        y_truth = interpolate_state_to_times(truth_res.t, truth_res.y, t_batch)

        dr     = y_stlrps[:, :3] - y_truth[:, :3]
        dv     = y_stlrps[:, 3:] - y_truth[:, 3:]
        dr_km  = np.linalg.norm(dr, axis=1) / 1_000.0
        dv_ms  = np.linalg.norm(dv, axis=1)
        ric_km = compute_ric_errors(y_truth[:, :3], y_truth[:, 3:], y_stlrps[:, :3]) / 1_000.0

        alt_tr = (np.linalg.norm(y_truth[:, :3], axis=1) - R_MOON) / 1_000.0
        alt_st = (np.linalg.norm(y_stlrps[:, :3], axis=1) - R_MOON) / 1_000.0

        total_rows.append({
            "scenario_id": scenario.scenario_id,
            "model": "st_lrps_batch_rk4", "reference": "sh200_dop853",
            "rms_pos_err_km":   float(np.sqrt(np.mean(dr_km ** 2))),
            "final_pos_err_km": float(dr_km[-1]),
            "max_pos_err_km":   float(np.max(dr_km)),
            "p95_pos_err_km":   float(np.percentile(dr_km, 95)),
            "rms_vel_err_ms":   float(np.sqrt(np.mean(dv_ms ** 2))),
            "final_vel_err_ms": float(dv_ms[-1]),
            "radial_rms_km":    float(np.sqrt(np.mean(ric_km[:, 0] ** 2))),
            "along_rms_km":     float(np.sqrt(np.mean(ric_km[:, 1] ** 2))),
            "cross_rms_km":     float(np.sqrt(np.mean(ric_km[:, 2] ** 2))),
            "rms_alt_err_km":   float(np.sqrt(np.mean((alt_st - alt_tr) ** 2))),
            "hp_km": scenario.hp_km, "inc_deg": scenario.inc_deg,
            "status": "ok",
        })

        if sh200_rk4_result is not None:
            t_rk4   = sh200_rk4_result["t"]
            Y_rk4   = sh200_rk4_result["Y"][:, i, :]  # (T, 6)

            # model error: st_lrps_rk4 vs sh200_rk4
            y_rk4_at_batch = interpolate_state_to_times(t_rk4, Y_rk4, t_batch)
            dr_m   = y_stlrps[:, :3] - y_rk4_at_batch[:, :3]
            dv_m   = y_stlrps[:, 3:] - y_rk4_at_batch[:, 3:]
            dr_m_km = np.linalg.norm(dr_m, axis=1) / 1_000.0
            dv_m_ms = np.linalg.norm(dv_m, axis=1)

            model_rows.append({
                "scenario_id": scenario.scenario_id,
                "model": "st_lrps_batch_rk4", "reference": "sh200_rk4",
                "rms_pos_err_km":   float(np.sqrt(np.mean(dr_m_km ** 2))),
                "final_pos_err_km": float(dr_m_km[-1]),
                "max_pos_err_km":   float(np.max(dr_m_km)),
                "p95_pos_err_km":   float(np.percentile(dr_m_km, 95)),
                "rms_vel_err_ms":   float(np.sqrt(np.mean(dv_m_ms ** 2))),
                "final_vel_err_ms": float(dv_m_ms[-1]),
                "radial_rms_km":    np.nan, "along_rms_km": np.nan, "cross_rms_km": np.nan,
                "rms_alt_err_km":   np.nan,
                "hp_km": scenario.hp_km, "inc_deg": scenario.inc_deg,
                "status": "ok",
            })

            # integrator error: sh200_rk4 vs sh200_dop853
            y_rk4_at_truth = interpolate_state_to_times(t_rk4, Y_rk4, truth_res.t)
            dr_i   = y_rk4_at_truth[:, :3] - truth_res.y[:, :3]
            dv_i   = y_rk4_at_truth[:, 3:] - truth_res.y[:, 3:]
            dr_i_km = np.linalg.norm(dr_i, axis=1) / 1_000.0
            dv_i_ms = np.linalg.norm(dv_i, axis=1)

            integr_rows.append({
                "scenario_id": scenario.scenario_id,
                "model": "sh200_rk4", "reference": "sh200_dop853",
                "rms_pos_err_km":   float(np.sqrt(np.mean(dr_i_km ** 2))),
                "final_pos_err_km": float(dr_i_km[-1]),
                "max_pos_err_km":   float(np.max(dr_i_km)),
                "p95_pos_err_km":   float(np.percentile(dr_i_km, 95)),
                "rms_vel_err_ms":   float(np.sqrt(np.mean(dv_i_ms ** 2))),
                "final_vel_err_ms": float(dv_i_ms[-1]),
                "radial_rms_km":    np.nan, "along_rms_km": np.nan, "cross_rms_km": np.nan,
                "rms_alt_err_km":   np.nan,
                "hp_km": scenario.hp_km, "inc_deg": scenario.inc_deg,
                "status": "ok",
            })

    return total_rows, model_rows, integr_rows


def _batch_agg_stats(rows: List[Dict], key: str) -> Dict[str, float]:
    vals = np.array([r[key] for r in rows if r.get("status") == "ok"
                     and np.isfinite(r.get(key, np.nan))], dtype=np.float64)
    if len(vals) == 0:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def compute_gpu_batch_metrics_for_model(
    result: BatchModelResult,
    truth: TruthTrajectorySet,
    scenarios: List[Scenario],
    duration_days: float,
) -> List[Dict[str, Any]]:
    """Compute per-scenario metrics for one GPU RK4 model against truth."""

    rows: List[Dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
        base = {
            "scenario_id": scenario.scenario_id,
            "model": result.display_name,
            "reference": "sh200_dop853",
            "backend": result.backend,
            "device": result.device,
            "rk4_dt_s": result.rk4_dt_s,
            "duration_days": float(duration_days),
            "hp_km": scenario.hp_km,
            "ha_km": scenario.ha_km,
            "a_km": scenario.a_km,
            "e": scenario.e,
            "inc_deg": scenario.inc_deg,
            "raan_deg": scenario.raan_deg,
            "argp_deg": scenario.argp_deg,
            "ta_deg": scenario.ta_deg,
        }
        if result.status != "ok":
            rows.append({
                **base,
                **{k: np.nan for k in _GPU_BATCH_METRICS_FIELDNAMES if k not in base},
                "status": "failed",
                "failure_reason": result.failure_reason,
            })
            continue
        if scenario.scenario_id not in truth.t_by_scenario:
            rows.append({
                **base,
                **{k: np.nan for k in _GPU_BATCH_METRICS_FIELDNAMES if k not in base},
                "status": "failed",
                "failure_reason": "missing_truth",
            })
            continue
        y_model = np.asarray(result.y[:, i, :], dtype=np.float64)
        if not np.isfinite(y_model).all():
            rows.append({
                **base,
                **{k: np.nan for k in _GPU_BATCH_METRICS_FIELDNAMES if k not in base},
                "status": "failed",
                "failure_reason": "non_finite_model_state",
            })
            continue

        t_truth = truth.t_by_scenario[scenario.scenario_id]
        y_truth = truth.y_by_scenario[scenario.scenario_id]
        y_model_at_truth = interpolate_state_to_times(result.t, y_model, t_truth)
        r_ref = y_truth[:, :3]
        v_ref = y_truth[:, 3:]
        r_test = y_model_at_truth[:, :3]
        v_test = y_model_at_truth[:, 3:]

        dr = r_test - r_ref
        dv = v_test - v_ref
        dr_km = np.linalg.norm(dr, axis=1) / 1_000.0
        dv_ms = np.linalg.norm(dv, axis=1)
        ric_km = compute_ric_errors(r_ref, v_ref, r_test) / 1_000.0
        alt_truth_km = (np.linalg.norm(r_ref, axis=1) - R_MOON) / 1_000.0
        alt_model_km = (np.linalg.norm(r_test, axis=1) - R_MOON) / 1_000.0
        alt_err_km = alt_model_km - alt_truth_km

        status = "ok"
        failure = ""
        if np.any(alt_model_km < 0.0):
            status = "warning_negative_altitude"
            failure = "model_altitude_became_negative"

        rows.append({
            **base,
            "rms_pos_err_km": float(np.sqrt(np.mean(dr_km ** 2))),
            "final_pos_err_km": float(dr_km[-1]),
            "max_pos_err_km": float(np.max(dr_km)),
            "p95_pos_err_km": float(np.percentile(dr_km, 95)),
            "rms_vel_err_ms": float(np.sqrt(np.mean(dv_ms ** 2))),
            "final_vel_err_ms": float(dv_ms[-1]),
            "max_vel_err_ms": float(np.max(dv_ms)),
            "p95_vel_err_ms": float(np.percentile(dv_ms, 95)),
            "radial_rms_km": float(np.sqrt(np.mean(ric_km[:, 0] ** 2))),
            "along_rms_km": float(np.sqrt(np.mean(ric_km[:, 1] ** 2))),
            "cross_rms_km": float(np.sqrt(np.mean(ric_km[:, 2] ** 2))),
            "radial_max_km": float(np.max(np.abs(ric_km[:, 0]))),
            "along_max_km": float(np.max(np.abs(ric_km[:, 1]))),
            "cross_max_km": float(np.max(np.abs(ric_km[:, 2]))),
            "rms_alt_err_km": float(np.sqrt(np.mean(alt_err_km ** 2))),
            "final_alt_err_km": float(alt_err_km[-1]),
            "max_abs_alt_err_km": float(np.max(np.abs(alt_err_km))),
            "min_alt_model_km": float(np.min(alt_model_km)),
            "min_alt_truth_km": float(np.min(alt_truth_km)),
            "status": status,
            "failure_reason": failure,
        })
    return rows


def aggregate_gpu_batch_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate GPU batch metrics per model."""

    from collections import defaultdict
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    failed: Dict[str, int] = defaultdict(int)
    for row in rows:
        model = str(row.get("model", ""))
        if row.get("status") in {"ok", "warning_negative_altitude"}:
            grouped[model].append(row)
        else:
            failed[model] += 1

    def _vals(model_rows: List[Dict[str, Any]], key: str) -> np.ndarray:
        return np.array([
            float(r[key]) for r in model_rows
            if key in r and np.isfinite(float(r.get(key, np.nan)))
        ], dtype=np.float64)

    def _percentile(vals: np.ndarray, pct: float) -> float:
        return float(np.percentile(vals, pct)) if vals.size else np.nan

    out: List[Dict[str, Any]] = []
    for model, model_rows in grouped.items():
        rms = _vals(model_rows, "rms_pos_err_km")
        final = _vals(model_rows, "final_pos_err_km")
        mx = _vals(model_rows, "max_pos_err_km")
        vel = _vals(model_rows, "rms_vel_err_ms")
        radial = _vals(model_rows, "radial_rms_km")
        along = _vals(model_rows, "along_rms_km")
        cross = _vals(model_rows, "cross_rms_km")
        alt = _vals(model_rows, "rms_alt_err_km")
        out.append({
            "model": model,
            "n_scenarios_ok": len(model_rows),
            "n_scenarios_failed": int(failed.get(model, 0)),
            "mean_rms_pos_err_km": float(np.mean(rms)) if rms.size else np.nan,
            "median_rms_pos_err_km": float(np.median(rms)) if rms.size else np.nan,
            "p90_rms_pos_err_km": _percentile(rms, 90),
            "p95_rms_pos_err_km": _percentile(rms, 95),
            "p99_rms_pos_err_km": _percentile(rms, 99),
            "max_rms_pos_err_km": float(np.max(rms)) if rms.size else np.nan,
            "mean_final_pos_err_km": float(np.mean(final)) if final.size else np.nan,
            "median_final_pos_err_km": float(np.median(final)) if final.size else np.nan,
            "p95_final_pos_err_km": _percentile(final, 95),
            "max_final_pos_err_km": float(np.max(final)) if final.size else np.nan,
            "mean_max_pos_err_km": float(np.mean(mx)) if mx.size else np.nan,
            "p95_max_pos_err_km": _percentile(mx, 95),
            "max_max_pos_err_km": float(np.max(mx)) if mx.size else np.nan,
            "median_rms_vel_err_ms": float(np.median(vel)) if vel.size else np.nan,
            "p95_rms_vel_err_ms": _percentile(vel, 95),
            "median_radial_rms_km": float(np.median(radial)) if radial.size else np.nan,
            "median_along_rms_km": float(np.median(along)) if along.size else np.nan,
            "median_cross_rms_km": float(np.median(cross)) if cross.size else np.nan,
            "median_rms_alt_err_km": float(np.median(alt)) if alt.size else np.nan,
        })
    return sorted(out, key=lambda r: r.get("median_rms_pos_err_km", np.inf))


def load_cached_truth_set(
    args: argparse.Namespace,
    scenarios: List[Scenario],
    cache_dir: Path,
    *,
    strict: bool = False,
) -> TruthTrajectorySet:
    t_by: Dict[int, np.ndarray] = {}
    y_by: Dict[int, np.ndarray] = {}
    rt_by: Dict[int, float] = {}
    missing: List[int] = []
    for scenario in scenarios:
        cached = _load_cached_trajectory(_cached_truth_path(cache_dir, args, scenario.scenario_id))
        if cached is None:
            missing.append(int(scenario.scenario_id))
            continue
        t_by[scenario.scenario_id] = cached.t
        y_by[scenario.scenario_id] = cached.y
        rt_by[scenario.scenario_id] = float(cached.runtime_s)
    if missing and strict:
        raise RuntimeError(f"Truth cache missing {len(missing)} scenarios: {missing[:8]}")
    return TruthTrajectorySet(_truth_cache_name(args), t_by, y_by, rt_by)


def _cached_gpu_runtime_rows(
    args: argparse.Namespace,
    models: List[str],
    scenarios: List[Scenario],
    cache_dir: Path,
    truth: TruthTrajectorySet,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in models:
        runtime = 0.0
        n = 0
        steps = 0
        device = ""
        backend = ""
        for scenario in scenarios:
            cached = _load_cached_trajectory(_cached_model_path(cache_dir, model, scenario.scenario_id))
            if cached is None:
                continue
            if np.isfinite(cached.runtime_s):
                runtime += float(cached.runtime_s)
            n += 1
            steps += max(0, int(cached.t.shape[0] - 1))
            device = str(cached.metadata.get("device", device))
            backend = str(cached.metadata.get("backend", backend))
        if n == 0:
            continue
        rows.append({
            "model": _model_display_name(model),
            "backend": backend,
            "device": device,
            "dtype": str(getattr(args, "torch_dtype", "")),
            "n_scenarios": n,
            "n_steps": steps,
            "n_saved_outputs": "",
            "total_runtime_s": runtime,
            "runtime_per_scenario_s": runtime / max(n, 1),
            "trajectory_steps_per_second": n * steps / max(runtime, 1e-9),
            "truth_total_runtime_s": truth.total_runtime_s,
            "truth_mean_runtime_per_scenario_s": truth.mean_runtime_s,
            "speedup_vs_truth_total": truth.total_runtime_s / max(runtime, 1e-9),
            "speedup_vs_truth_per_scenario": truth.mean_runtime_s / max(runtime / max(n, 1), 1e-9),
            "status": "cached",
        })
    return sorted(rows, key=lambda r: r.get("total_runtime_s", np.inf))


def _load_cached_gpu_batch_results(
    args: argparse.Namespace,
    scenarios: List[Scenario],
    cache_dir: Path,
    models: List[str],
) -> List[BatchModelResult]:
    results: List[BatchModelResult] = []
    for model in models:
        cached_by_scenario: List[CachedTrajectory] = []
        complete = True
        for scenario in scenarios:
            cached = _load_cached_trajectory(_cached_model_path(cache_dir, model, scenario.scenario_id))
            if cached is None:
                complete = False
                break
            cached_by_scenario.append(cached)
        if not complete or not cached_by_scenario:
            continue
        t_ref = cached_by_scenario[0].t
        if any(c.t.shape != t_ref.shape or np.max(np.abs(c.t - t_ref)) > 1e-9 for c in cached_by_scenario):
            print(f"[cache] WARNING: cached model {model} has inconsistent time grids; "
                  "skipping time-series plots for this model.", flush=True)
            continue
        y = np.stack([c.y for c in cached_by_scenario], axis=1)
        meta = cached_by_scenario[0].metadata
        rk4_dt = float(meta.get(
            "rk4_dt_s",
            args.rk4_dt_s if args.rk4_dt_s is not None else args.st_lrps_rk4_dt,
        ))
        runtime = float(sum(c.runtime_s for c in cached_by_scenario if np.isfinite(c.runtime_s)))
        results.append(BatchModelResult(
            model_name=str(model),
            display_name=_model_display_name(model),
            backend=str(meta.get("backend", "cached")),
            device=str(meta.get("device", "")),
            dtype=str(meta.get("dtype", "")),
            t=t_ref,
            y=y,
            runtime_s=runtime,
            n_steps=max(0, int(t_ref.shape[0] - 1)),
            n_scenarios=len(cached_by_scenario),
            rk4_dt_s=rk4_dt,
            output_dt_s=float(args.dt_out),
            status="ok",
        ))
    return results


def rebuild_gpu_batch_metrics_from_cache(
    args: argparse.Namespace,
    scenarios: List[Scenario],
    cache_dir: Path,
    gpu_models: List[str],
    metrics_dir: Path,
    plots_dir: Path,
    reports_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    print("[cache] Rebuilding metrics from cached trajectories.", flush=True)
    truth = load_cached_truth_set(args, scenarios, cache_dir, strict=True)
    all_rows: List[Dict[str, Any]] = []
    for model in gpu_models:
        complete, missing = _model_cache_completion(cache_dir, model, scenarios)
        print(f"[cache] Model {model}: {complete}/{len(scenarios)} complete.", flush=True)
        if missing and getattr(args, "strict_complete", False):
            raise RuntimeError(
                f"Model {model} is missing {len(missing)} cached scenario trajectories."
            )
        for scenario in scenarios:
            cached = _load_cached_trajectory(_cached_model_path(cache_dir, model, scenario.scenario_id))
            if cached is None:
                continue
            result = BatchModelResult(
                model_name=model,
                display_name=_model_display_name(model),
                backend=str(cached.metadata.get("backend", "cached")),
                device=str(cached.metadata.get("device", "")),
                dtype=str(cached.metadata.get("dtype", "")),
                t=cached.t,
                y=cached.y[:, None, :],
                runtime_s=float(cached.runtime_s),
                n_steps=max(0, int(cached.t.shape[0] - 1)),
                n_scenarios=1,
                rk4_dt_s=float(
                    cached.metadata.get(
                        "rk4_dt_s",
                        args.rk4_dt_s if args.rk4_dt_s is not None else args.st_lrps_rk4_dt,
                    )
                ),
                output_dt_s=float(args.dt_out),
                status="ok",
            )
            all_rows.extend(compute_gpu_batch_metrics_for_model(
                result, truth, [scenario], args.duration_days
            ))

    aggregate_rows = aggregate_gpu_batch_metrics(all_rows)
    runtime_rows = _cached_gpu_runtime_rows(args, gpu_models, scenarios, cache_dir, truth)
    plot_results = _load_cached_gpu_batch_results(args, scenarios, cache_dir, gpu_models)
    ranking_rows = build_gpu_model_ranking(aggregate_rows)
    equivalent = estimate_stlrps_equivalent_sh_degree(aggregate_rows)
    selected = select_stlrps_scenarios(all_rows, {s.scenario_id: s for s in scenarios}, args)

    _write_csv(all_rows, metrics_dir / "gpu_batch_per_scenario_metrics.csv")
    _write_csv(aggregate_rows, metrics_dir / "gpu_batch_aggregate_metrics.csv")
    _write_csv(runtime_rows, metrics_dir / "gpu_batch_runtime_metrics.csv")
    _write_csv(ranking_rows, metrics_dir / "gpu_batch_model_ranking.csv")
    (metrics_dir / "stlrps_selected_scenarios.json").write_text(
        json.dumps(selected, indent=4, default=str), encoding="utf-8"
    )
    summary = {
        "truth": _truth_cache_name(args),
        "gpu_models": gpu_models,
        "gpu_model_variants": [_model_display_name(m) for m in gpu_models],
        "n_scenarios_total": len(scenarios),
        "n_scenarios_new_this_run": 0,
        "accumulated": bool(args.resume),
        "rebuilt_from_cache": True,
        "sampling": _sampling_metadata(args, len(scenarios)),
        "truth_workers": int(getattr(args, "workers", 1)),
        "gpu_integrator": str(getattr(args, "gpu_integrator", "medium")),
        "frame_mode": args.batch_frame_mode,
        "truth_total_runtime_s": truth.total_runtime_s,
        "truth_mean_runtime_per_scenario_s": truth.mean_runtime_s,
        "equivalent_sh_degree": equivalent,
        "selected_stlrps_scenarios": selected,
        "aggregate": aggregate_rows,
        "runtime": runtime_rows,
    }
    (metrics_dir / "gpu_batch_summary.json").write_text(
        json.dumps(summary, indent=4, default=str), encoding="utf-8"
    )
    cache_metrics_dir = cache_dir / "metrics"
    if cache_metrics_dir != metrics_dir:
        _write_csv(all_rows, cache_metrics_dir / "per_model_scenario_metrics.csv")
        _write_csv(aggregate_rows, cache_metrics_dir / "aggregate_metrics.csv")
        _write_csv(runtime_rows, cache_metrics_dir / "runtime_metrics.csv")
        _write_csv(ranking_rows, cache_metrics_dir / "model_ranking.csv")
        cache_metrics_dir.mkdir(parents=True, exist_ok=True)
        (cache_metrics_dir / "summary.json").write_text(
            json.dumps(summary, indent=4, default=str), encoding="utf-8"
        )
    if aggregate_rows:
        plot_gpu_batch_report_figures(
            aggregate_rows, runtime_rows, all_rows, plot_results, truth, scenarios,
            selected, equivalent, plots_dir, args
        )
        write_gpu_batch_report_pdf(args, aggregate_rows, runtime_rows, equivalent, selected, plots_dir, reports_dir)
    return aggregate_rows, runtime_rows, equivalent, selected


def build_gpu_runtime_metrics(
    results: List[BatchModelResult],
    truth: TruthTrajectorySet,
) -> List[Dict[str, Any]]:
    """Build per-model runtime and speedup rows."""

    base_rows: List[Dict[str, Any]] = []
    by_model: Dict[str, Dict[str, Any]] = {}
    truth_total = truth.total_runtime_s
    truth_mean = truth.mean_runtime_s
    for result in results:
        n_steps = max(int(result.n_steps), 1)
        n_scenarios = max(int(result.n_scenarios), 1)
        runtime = float(result.runtime_s)
        row = {
            "model": result.display_name,
            "backend": result.backend,
            "device": result.device,
            "dtype": result.dtype,
            "n_scenarios": n_scenarios,
            "n_steps": n_steps,
            "n_saved_outputs": int(len(result.t)),
            "total_runtime_s": runtime,
            "runtime_per_scenario_s": runtime / n_scenarios,
            "trajectory_steps_per_second": n_scenarios * n_steps / max(runtime, 1e-9),
            "acceleration_evaluations_per_second": n_scenarios * n_steps * 4 / max(runtime, 1e-9),
            "truth_total_runtime_s": truth_total,
            "truth_mean_runtime_per_scenario_s": truth_mean,
            "speedup_vs_truth_total": truth_total / max(runtime, 1e-9),
            "speedup_vs_truth_per_scenario": truth_mean / max(runtime / n_scenarios, 1e-9),
        }
        by_model[result.display_name] = row
        base_rows.append(row)

    for row in base_rows:
        for other in base_rows:
            key = "speedup_vs_" + other["model"].lower()
            row[key] = float(other["total_runtime_s"]) / max(float(row["total_runtime_s"]), 1e-9)
    return sorted(base_rows, key=lambda r: r["total_runtime_s"])


def build_gpu_model_ranking(aggregate_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for i, row in enumerate(sorted(aggregate_rows, key=lambda r: r.get("median_rms_pos_err_km", np.inf)), 1):
        rows.append({
            "rank_accuracy": i,
            "model": row["model"],
            "median_rms_pos_err_km": row.get("median_rms_pos_err_km", np.nan),
            "p95_rms_pos_err_km": row.get("p95_rms_pos_err_km", np.nan),
            "max_rms_pos_err_km": row.get("max_rms_pos_err_km", np.nan),
            "median_along_rms_km": row.get("median_along_rms_km", np.nan),
            "n_scenarios_ok": row.get("n_scenarios_ok", 0),
        })
    return rows


# =============================================================================
# DOP853 aggregate statistics
# =============================================================================

_AGG_KEYS = [
    ("rms_pos_err_km", ["mean", "median", "std", "p50", "p90", "p95", "p99", "max"]),
    ("final_pos_err_km", ["mean", "median", "p90", "p95", "max"]),
    ("max_pos_err_km", ["mean", "p90", "p95", "max"]),
    ("rms_vel_err_ms", ["mean", "median", "p90", "p95", "max"]),
    ("runtime_s", ["mean", "total"]),
]


def _stat(arr: np.ndarray, stat: str) -> float:
    if stat == "mean":   return float(np.mean(arr))
    if stat == "median": return float(np.median(arr))
    if stat == "std":    return float(np.std(arr))
    if stat == "total":  return float(np.sum(arr))
    if stat == "max":    return float(np.max(arr))
    pct = int(stat[1:])
    return float(np.percentile(arr, pct))


def aggregate_metrics(
    all_metrics: List[Dict],
    truth_runtime_mean: float,
) -> Dict[str, Dict]:
    from collections import defaultdict
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for m in all_metrics:
        if m.get("status") == "ok":
            grouped[m["model"]].append(m)

    result: Dict[str, Dict] = {}
    for model, rows in grouped.items():
        entry: Dict[str, Any] = {"n_scenarios": len(rows)}
        for key, stats in _AGG_KEYS:
            vals = np.array([r[key] for r in rows if r.get(key) is not None],
                            dtype=np.float64)
            if len(vals) == 0:
                continue
            for s in stats:
                entry[f"{key}__{s}"] = _stat(vals, s)
        rt = np.array([r["runtime_s"] for r in rows if r.get("runtime_s") is not None],
                      dtype=np.float64)
        if len(rt) > 0:
            entry["runtime_s__mean"]  = float(np.mean(rt))
            entry["runtime_s__total"] = float(np.sum(rt))
            entry["runtime_speed_rel_to_truth"] = float(
                np.mean(rt) / max(truth_runtime_mean, 1e-9)
            )
        result[model] = entry
    return result


def build_rankings(agg: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for model, stats in agg.items():
        rows.append({
            "model": model,
            "median_rms_pos_err_km": stats.get("rms_pos_err_km__median", np.nan),
            "p95_rms_pos_err_km":    stats.get("rms_pos_err_km__p95", np.nan),
            "max_pos_err_km__mean":  stats.get("max_pos_err_km__mean", np.nan),
            "runtime_s__mean":       stats.get("runtime_s__mean", np.nan),
            "n_scenarios":           stats.get("n_scenarios", 0),
        })

    for i, r in enumerate(sorted(rows, key=lambda r: r["median_rms_pos_err_km"])):
        r["rank_median_rms"] = i + 1
    for i, r in enumerate(sorted(rows, key=lambda r: r["p95_rms_pos_err_km"])):
        r["rank_p95_rms"] = i + 1
    for i, r in enumerate(sorted(rows, key=lambda r: r["max_pos_err_km__mean"])):
        r["rank_worst"] = i + 1
    for i, r in enumerate(sorted(rows, key=lambda r: r["runtime_s__mean"])):
        r["rank_runtime"] = i + 1

    combined = {r["model"]: r for r in rows}
    return sorted(combined.values(), key=lambda r: r.get("rank_median_rms", 999))


def find_worst_cases(
    all_metrics: List[Dict],
    scenarios_by_id: Dict[int, Scenario],
) -> List[Dict]:
    from collections import defaultdict
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for m in all_metrics:
        if m.get("status") == "ok":
            grouped[m["model"]].append(m)

    worst_rows = []
    metrics_to_check = [
        ("max_rms_pos_err",   "rms_pos_err_km"),
        ("max_final_pos_err", "final_pos_err_km"),
        ("max_max_pos_err",   "max_pos_err_km"),
        ("max_alt_err",       "max_abs_alt_err_km"),
    ]
    for model, rows in grouped.items():
        for label, key in metrics_to_check:
            valid = [r for r in rows if r.get(key) is not None]
            if not valid:
                continue
            worst = max(valid, key=lambda r: r[key])
            sc = scenarios_by_id.get(worst["scenario_id"])
            row = {"model": model, "metric_name": label,
                   "scenario_id": worst["scenario_id"], "metric_value": worst[key]}
            if sc is not None:
                row.update({
                    "hp_km": sc.hp_km, "ha_km": sc.ha_km, "a_km": sc.a_km,
                    "e": sc.e, "inc_deg": sc.inc_deg, "raan_deg": sc.raan_deg,
                    "argp_deg": sc.argp_deg, "ta_deg": sc.ta_deg,
                })
            worst_rows.append(row)
    return worst_rows


def select_median_difficulty_scenario(
    all_metrics: List[Dict],
    scenarios: List[Scenario],
) -> Optional[Scenario]:
    """Choose the scenario whose max-RMS across all models is nearest the median."""
    if not all_metrics or not scenarios:
        return scenarios[len(scenarios) // 2] if scenarios else None

    from collections import defaultdict
    rms_by_sc: Dict[int, List[float]] = defaultdict(list)
    for m in all_metrics:
        if m.get("status") == "ok" and m.get("rms_pos_err_km") is not None:
            rms_by_sc[m["scenario_id"]].append(float(m["rms_pos_err_km"]))

    if not rms_by_sc:
        return scenarios[len(scenarios) // 2]

    # Difficulty = mean RMS across models for each scenario
    sc_difficulty = {sid: float(np.mean(vals)) for sid, vals in rms_by_sc.items()}
    median_diff = float(np.median(list(sc_difficulty.values())))

    scenarios_dict = {s.scenario_id: s for s in scenarios}
    best_sid = min(sc_difficulty.keys(), key=lambda s: abs(sc_difficulty[s] - median_diff))
    return scenarios_dict.get(best_sid, scenarios[len(scenarios) // 2])


# =============================================================================
# Plotting helpers
# =============================================================================

# =============================================================================
# Publication-grade plotting style (visualization only; no numeric impact)
# =============================================================================
# Consistent, professional styling shared by every Orbit-Level Benchmark figure:
#   * ST-LRPS uses a single distinctive accent colour + star marker and a heavier
#     line so it always stands out.
#   * Spherical-harmonic baselines share a degree-ordered colour family that runs
#     warm (low degree) -> cool/dark (high degree), so SH20 and SH200 are easy to
#     tell apart and the ordering reads naturally.
#   * Helpers pick a sensible display unit (km/m/cm) and never leave a blank plot.

_ST_LRPS_COLOR = "#8E2DC4"   # deep violet accent — ST-LRPS stands out
_TRUTH_COLOR = "#15202B"     # near-black reference
_FALLBACK_COLOR = "#7A8699"

# Degree -> colour anchors (interpolated in RGB for arbitrary degrees).
_SH_DEGREE_ANCHORS = [
    (20,  "#D1495B"),  # muted red
    (30,  "#E8833A"),  # warm amber
    (60,  "#C9A227"),  # gold
    (80,  "#6C8EBF"),  # blue-gray
    (100, "#3D5A80"),  # slate blue
    (120, "#33518A"),  # deeper slate
    (160, "#23386B"),  # dark blue
    (200, "#1B2A41"),  # charcoal navy
]
_KNOWN_DEGREE_ORDER = [20, 30, 60, 80, 100, 120, 160, 200]
_SH_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
_SH_DEGREE_RE = re.compile(r"SH(\d+)")

# Legacy override table (kept for the older CPU-mode plots / batch-rk4 panels).
MODEL_COLORS = {
    "st_lrps_batch_rk4": _ST_LRPS_COLOR, "sh200_rk4": "#1B2A41",
}


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb)


def _model_degree(model: str) -> Optional[int]:
    m = str(model).upper()
    if "ST_LRPS" in m or "ST-LRPS" in m:
        return None
    found = _SH_DEGREE_RE.search(m)
    return int(found.group(1)) if found else None


def _is_stlrps(model: str) -> bool:
    m = str(model).upper()
    return "ST_LRPS" in m or "ST-LRPS" in m


def _sh_degree_color(deg: int) -> str:
    anchors = _SH_DEGREE_ANCHORS
    if deg <= anchors[0][0]:
        return anchors[0][1]
    if deg >= anchors[-1][0]:
        return anchors[-1][1]
    for (d0, c0), (d1, c1) in zip(anchors, anchors[1:]):
        if d0 <= deg <= d1:
            f = (deg - d0) / (d1 - d0) if d1 > d0 else 0.0
            r0, r1 = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return _rgb_to_hex(tuple(a + (b - a) * f for a, b in zip(r0, r1)))
    return anchors[-1][1]


def model_color(model: str) -> str:
    """Consistent colour for a model across every figure."""
    if _is_stlrps(model):
        return _ST_LRPS_COLOR
    deg = _model_degree(model)
    if deg is not None:
        return _sh_degree_color(deg)
    return MODEL_COLORS.get(str(model), MODEL_COLORS.get(str(model).lower(), _FALLBACK_COLOR))


def _color(m: str) -> str:  # backwards-compatible alias
    return model_color(m)


def model_marker(model: str) -> str:
    if _is_stlrps(model):
        return "*"
    deg = _model_degree(model)
    if deg is None:
        return "o"
    if deg in _KNOWN_DEGREE_ORDER:
        return _SH_MARKERS[_KNOWN_DEGREE_ORDER.index(deg) % len(_SH_MARKERS)]
    return _SH_MARKERS[deg % len(_SH_MARKERS)]


def model_linewidth(model: str) -> float:
    return 2.8 if _is_stlrps(model) else 1.6


def model_marker_size(model: str) -> float:
    return 210.0 if _is_stlrps(model) else 90.0


def model_zorder(model: str) -> int:
    return 7 if _is_stlrps(model) else 3


def display_label(model: str) -> str:
    """Human label, e.g. GPU_SH20_RK4 -> SH20, GPU_ST_LRPS_RK4 -> ST-LRPS."""
    m = str(model)
    dt_label: Optional[str] = None
    if "_DT" in m.upper():
        head, tail = re.split(r"_DT", m, maxsplit=1, flags=re.IGNORECASE)
        m = head
        dt_label = tail
    if _is_stlrps(m):
        label = "ST-LRPS"
    else:
        label = m.replace("GPU_", "").replace("_RK4", "").upper()
    return f"{label} dt{dt_label}" if dt_label else label


def select_length_unit(max_km: float) -> Tuple[str, float]:
    """Pick a readable display unit for a length given the largest value in km.

    Returns ``(unit_label, multiplier)`` where ``display = value_km * multiplier``.
    CSV/metric units are never changed — this only affects plotting.
    """
    try:
        v = float(max_km)
    except (TypeError, ValueError):
        return ("km", 1.0)
    if not math.isfinite(v) or v <= 0.0:
        return ("km", 1.0)
    if v < 1.0e-3:
        return ("cm", 1.0e5)   # 1 km = 1e5 cm
    if v < 1.0e-2:
        return ("m", 1.0e3)    # 1 km = 1e3 m
    return ("km", 1.0)


def _finite_positive(values: Sequence[float]) -> List[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0.0:
            out.append(f)
    return out


def _should_log(values: Sequence[float], ratio: float = 50.0) -> bool:
    """True when positive values span more than ``ratio`` orders-of-magnitude."""
    pos = _finite_positive(values)
    return len(pos) >= 2 and (max(pos) / min(pos)) > ratio


# Theme palettes (used by figure styling helpers + rcParams).
_PLOT_THEMES = {
    "report_light": dict(bg="#FFFFFF", ax_bg="#FFFFFF", text="#1A1F29",
                         grid="#D7DEE8", edge="#3A4452", muted="#5A6675", accent="#2A9D8F"),
    "technical_dark": dict(bg="#0E1116", ax_bg="#11151C", text="#E8ECF8",
                           grid="#2A3340", edge="#8A98AD", muted="#9AA7C7", accent="#35D0FF"),
}
_ACTIVE_PLOT_THEME = _PLOT_THEMES["report_light"]


def apply_plot_theme(theme: str) -> None:
    """Central publication-grade plotting style for validation figures."""
    global _ACTIVE_PLOT_THEME
    th = _PLOT_THEMES.get(str(theme), _PLOT_THEMES["report_light"])
    _ACTIVE_PLOT_THEME = th
    plt.style.use("dark_background" if theme == "technical_dark" else "default")
    plt.rcParams.update({
        "figure.facecolor": th["bg"],
        "axes.facecolor": th["ax_bg"],
        "savefig.facecolor": th["bg"],
        "text.color": th["text"],
        "axes.labelcolor": th["text"],
        "axes.edgecolor": th["edge"],
        "axes.linewidth": 0.9,
        "xtick.color": th["text"],
        "ytick.color": th["text"],
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.color": th["grid"],
        "grid.alpha": 0.55 if theme == "technical_dark" else 0.9,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
    })


def _style_ax(ax: Any, *, title: Optional[str] = None, xlabel: Optional[str] = None,
              ylabel: Optional[str] = None, subtitle: Optional[str] = None) -> None:
    th = _ACTIVE_PLOT_THEME
    if title:
        ax.set_title(title, color=th["text"], pad=30 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=th["muted"], va="bottom", ha="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.5, linewidth=0.7)
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)


def _legend(ax: Any, *, outside: bool = False, loc: str = "best", ncol: int = 1) -> Any:
    th = _ACTIVE_PLOT_THEME
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if outside:
        return ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                         frameon=False, ncol=ncol)
    leg = ax.legend(handles, labels, loc=loc, frameon=True, framealpha=0.9, ncol=ncol)
    if leg is not None:
        leg.get_frame().set_edgecolor(th["grid"])
        leg.get_frame().set_facecolor(th["ax_bg"])
        leg.get_frame().set_linewidth(0.6)
    return leg


def _legend_outside(ax: Any) -> None:  # backwards-compatible alias
    _legend(ax, outside=True)


def _empty_note(ax: Any, text: str) -> None:
    """Stamp an explanatory note on an otherwise-blank plot."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color=_ACTIVE_PLOT_THEME["muted"], style="italic", wrap=True)


def _highlight_ticklabels(ax: Any, labels: Sequence[str], axis: str = "y") -> None:
    """Bold + accent the ST-LRPS tick label on a category axis."""
    th = _ACTIVE_PLOT_THEME
    ticklabels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lbl, text in zip(ticklabels, labels):
        if _is_stlrps(text):
            lbl.set_color(_ST_LRPS_COLOR)
            lbl.set_fontweight("bold")
        else:
            lbl.set_color(th["text"])


def _model_sort_key(model: str) -> Tuple[int, int]:
    m = str(model).upper()
    if "SH200" in m:
        return (0, 200)
    if "SH160" in m:
        return (1, 160)
    if "SH120" in m:
        return (2, 120)
    if "SH60" in m:
        return (3, 60)
    if "SH20" in m:
        return (4, 20)
    if "ST_LRPS" in m:
        return (5, 0)
    return (9, 0)


def plot_selected_scenario(
    scenario: Scenario,
    truth_model: str,
    model_trajectories: Dict[str, Any],
    out_dir: Path,
    prefix: str = "selected",
) -> List[Path]:
    saved = []
    plt.style.use("dark_background")

    truth_res = model_trajectories.get(truth_model)
    if truth_res is None:
        return saved

    t_ref  = truth_res.t / 86400.0
    r_ref  = truth_res.y[:, :3]
    v_ref  = truth_res.y[:, 3:6]
    other_models = [m for m in model_trajectories if m != truth_model]

    # 3D orbit
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    r_km = r_ref / 1_000.0
    ax.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2],
            color=_color(truth_model), lw=2, label=truth_model.upper(), zorder=5)
    for m in other_models:
        res = model_trajectories[m]
        rk  = interpolate_state_to_times(res.t, res.y, truth_res.t)[:, :3] / 1_000.0
        ax.plot(rk[:, 0], rk[:, 1], rk[:, 2], color=_color(m), lw=1,
                alpha=0.8, label=m.upper())
    ax.set_title(f"3D Orbit — scenario {scenario.scenario_id}\n"
                 f"hp={scenario.hp_km:.0f} km  ha={scenario.ha_km:.0f} km  "
                 f"i={scenario.inc_deg:.1f} deg")
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    ax.legend(fontsize=8)
    p = out_dir / f"{prefix}_orbit_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Altitude
    fig, ax = plt.subplots(figsize=(10, 5))
    alt_ref = (np.linalg.norm(r_ref, axis=1) - R_MOON) / 1_000.0
    ax.plot(t_ref, alt_ref, color=_color(truth_model), lw=2, label=truth_model.upper(), zorder=5)
    for m in other_models:
        res = model_trajectories[m]
        y_m = interpolate_state_to_times(res.t, res.y, truth_res.t)
        alt = (np.linalg.norm(y_m[:, :3], axis=1) - R_MOON) / 1_000.0
        ax.plot(t_ref, alt, color=_color(m), lw=1, alpha=0.85, label=m.upper())
    ax.set_title(f"Altitude — scenario {scenario.scenario_id}")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("Altitude [km]")
    ax.grid(True, alpha=0.25); ax.legend()
    p = out_dir / f"{prefix}_altitude.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Position error
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in other_models:
        res = model_trajectories[m]
        y_m = interpolate_state_to_times(res.t, res.y, truth_res.t)
        dr  = np.linalg.norm(y_m[:, :3] - r_ref, axis=1) / 1_000.0
        ax.semilogy(t_ref, np.maximum(dr, 1e-9), color=_color(m), lw=1.2, label=m.upper())
    ax.set_title(f"Position Error vs {truth_model.upper()} — scenario {scenario.scenario_id}")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("Position Error [km]")
    ax.grid(True, alpha=0.25, which="both"); ax.legend()
    p = out_dir / f"{prefix}_position_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Velocity error
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in other_models:
        res = model_trajectories[m]
        y_m = interpolate_state_to_times(res.t, res.y, truth_res.t)
        dv  = np.linalg.norm(y_m[:, 3:] - v_ref, axis=1)
        ax.semilogy(t_ref, np.maximum(dv, 1e-9), color=_color(m), lw=1.2, label=m.upper())
    ax.set_title(f"Velocity Error vs {truth_model.upper()} — scenario {scenario.scenario_id}")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("Velocity Error [m/s]")
    ax.grid(True, alpha=0.25, which="both"); ax.legend()
    p = out_dir / f"{prefix}_velocity_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # RIC error
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels_ric = ["Radial", "In-track (Along)", "Cross-track"]
    for m in other_models:
        res = model_trajectories[m]
        y_m = interpolate_state_to_times(res.t, res.y, truth_res.t)
        ric = compute_ric_errors(r_ref, v_ref, y_m[:, :3]) / 1_000.0
        for k in range(3):
            axes[k].plot(t_ref, ric[:, k], color=_color(m), lw=1, label=m.upper())
    for k, lbl in enumerate(labels_ric):
        axes[k].set_ylabel(f"{lbl} [km]")
        axes[k].grid(True, alpha=0.25)
        axes[k].legend(fontsize=7)
    axes[0].set_title(f"RIC Position Error — scenario {scenario.scenario_id}")
    axes[2].set_xlabel("Time [days]")
    fig.tight_layout()
    p = out_dir / f"{prefix}_ric_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    return saved


def plot_aggregate_stats(
    all_metrics: List[Dict],
    agg: Dict[str, Dict],
    rankings: List[Dict],
    out_dir: Path,
) -> List[Path]:
    saved = []
    plt.style.use("dark_background")

    from collections import defaultdict
    grouped: Dict[str, List[float]] = defaultdict(list)
    for m in all_metrics:
        if m.get("status") == "ok" and m.get("rms_pos_err_km") is not None:
            grouped[m["model"]].append(m["rms_pos_err_km"])

    if not grouped:
        return saved

    models_sorted = [r["model"] for r in rankings if r["model"] in grouped]

    # Boxplot
    fig, ax = plt.subplots(figsize=(max(6, len(models_sorted) * 1.5), 6))
    data   = [grouped[m] for m in models_sorted]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", lw=2))
    for patch, m in zip(bp["boxes"], models_sorted):
        patch.set_facecolor(_color(m))
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(models_sorted) + 1))
    ax.set_xticklabels([m.upper() for m in models_sorted])
    ax.set_ylabel("RMS Position Error [km]")
    ax.set_title("RMS Position Error Distribution vs Truth")
    ax.grid(True, alpha=0.2, axis="y")
    p = out_dir / "aggregate_boxplot_rms_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # P95 bar
    p95_vals = [agg[m].get("rms_pos_err_km__p95", 0) for m in models_sorted]
    fig, ax  = plt.subplots(figsize=(max(6, len(models_sorted) * 1.5), 5))
    bars = ax.bar(range(len(models_sorted)), p95_vals,
                  color=[_color(m) for m in models_sorted], alpha=0.8)
    ax.set_xticks(range(len(models_sorted)))
    ax.set_xticklabels([m.upper() for m in models_sorted])
    ax.set_ylabel("P95 RMS Position Error [km]")
    ax.set_title("P95 RMS Position Error vs Truth")
    ax.grid(True, alpha=0.2, axis="y")
    for bar, val in zip(bars, p95_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    p = out_dir / "aggregate_p95_error_bar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Runtime vs accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models_sorted:
        rt  = agg[m].get("runtime_s__mean", np.nan)
        err = agg[m].get("rms_pos_err_km__median", np.nan)
        ax.scatter(rt, err, color=_color(m), s=120, zorder=5, label=m.upper())
        ax.annotate(m.upper(), (rt, err), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean Runtime per Scenario [s]")
    ax.set_ylabel("Median RMS Position Error [km]")
    ax.set_title("Runtime vs Accuracy (DOP853)")
    ax.grid(True, alpha=0.2); ax.legend()
    p = out_dir / "runtime_vs_accuracy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    return saved


def plot_batch_rk4_results(
    total_rows: List[Dict],
    model_rows: List[Dict],
    integr_rows: List[Dict],
    batch_meta: Dict[str, Any],
    out_dir: Path,
) -> List[Path]:
    saved = []
    plt.style.use("dark_background")

    ok_total = [r for r in total_rows if r.get("status") == "ok"]
    if not ok_total:
        return saved

    rms_total = np.array([r["rms_pos_err_km"] for r in ok_total])

    # Runtime vs accuracy single-point panel.  It looks simple, but it makes the
    # GPU batch result visually comparable to the CPU DOP853 runtime plots.
    fig, ax = plt.subplots(figsize=(7, 5))
    runtime_s = float(batch_meta.get("runtime_s", np.nan))
    ax.scatter(runtime_s, float(np.median(rms_total)), color=_color("st_lrps"), s=140)
    ax.annotate("ST-LRPS RK4", (runtime_s, float(np.median(rms_total))),
                textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Total Batch Runtime [s]")
    ax.set_ylabel("Median RMS Position Error [km]")
    ax.set_title("Batch Runtime vs Accuracy")
    ax.grid(True, alpha=0.2)
    p = out_dir / "batch_runtime_vs_accuracy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # RMS distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rms_total, bins=20, color=_color("st_lrps"), alpha=0.8, edgecolor="white", lw=0.5)
    ax.axvline(np.median(rms_total), color="yellow", lw=1.5, label=f"Median {np.median(rms_total):.3f} km")
    ax.axvline(np.percentile(rms_total, 95), color="orange", lw=1.5,
               label=f"P95 {np.percentile(rms_total, 95):.3f} km")
    ax.set_xlabel("RMS Position Error [km]")
    ax.set_ylabel("Count")
    ax.set_title("ST-LRPS Batch RK4 vs SH200 DOP853 — RMS Error Distribution")
    ax.legend(); ax.grid(True, alpha=0.2)
    p = out_dir / "batch_rms_error_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Error decomposition bar chart (if decomposition available)
    if model_rows and integr_rows:
        ok_model  = [r for r in model_rows if r.get("status") == "ok"]
        ok_integr = [r for r in integr_rows if r.get("status") == "ok"]

        rms_model  = np.median([r["rms_pos_err_km"] for r in ok_model]) if ok_model else 0
        rms_integr = np.median([r["rms_pos_err_km"] for r in ok_integr]) if ok_integr else 0
        rms_total_med = float(np.median(rms_total))

        labels = ["ST-LRPS RK4\nvs SH200 DOP853\n(total)", "ST-LRPS RK4\nvs SH200 RK4\n(model error)", "SH200 RK4\nvs SH200 DOP853\n(integrator error)"]
        vals   = [rms_total_med, rms_model, rms_integr]
        colors = [_color("st_lrps"), _color("st_lrps_batch_rk4"), _color("sh200_rk4")]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(range(3), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Median RMS Position Error [km]")
        ax.set_title("Error Decomposition (Batch RK4)")
        ax.grid(True, alpha=0.2, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        p = out_dir / "batch_error_decomposition_bar.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # RMS error vs inclination
    inc_vals = np.array([r["inc_deg"] for r in ok_total])
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(inc_vals, rms_total, c=rms_total, cmap="plasma", s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="RMS Error [km]")
    ax.set_xlabel("Inclination [deg]")
    ax.set_ylabel("RMS Position Error [km]")
    ax.set_title("ST-LRPS Batch RK4 Error vs Inclination")
    ax.grid(True, alpha=0.2)
    p = out_dir / "batch_error_vs_inclination.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    return saved


def plot_batch_selected_scenario(
    total_rows: List[Dict],
    batch_result: Dict[str, Any],
    truth_results: List[Optional[Any]],
    scenarios: List[Scenario],
    out_dir: Path,
) -> List[Path]:
    """Plot the median-error batch RK4 scenario against SH200 DOP853."""

    ok_rows = [
        r for r in total_rows
        if r.get("status") == "ok" and np.isfinite(r.get("rms_pos_err_km", np.nan))
    ]
    if not ok_rows:
        return []

    median_rms = float(np.median([r["rms_pos_err_km"] for r in ok_rows]))
    selected = min(ok_rows, key=lambda r: abs(float(r["rms_pos_err_km"]) - median_rms))
    sid = int(selected["scenario_id"])
    idx_by_sid = {sc.scenario_id: i for i, sc in enumerate(scenarios)}
    i = idx_by_sid.get(sid)
    if i is None or i >= len(truth_results) or truth_results[i] is None:
        return []

    truth = truth_results[i]
    assert truth is not None
    t_batch = np.asarray(batch_result["t"], dtype=np.float64)
    y_st = np.asarray(batch_result["Y"][:, i, :], dtype=np.float64)
    y_truth = interpolate_state_to_times(truth.t, truth.y, t_batch)
    t_days = t_batch / 86400.0

    saved: List[Path] = []
    plt.style.use("dark_background")

    pos_err_km = np.linalg.norm(y_st[:, :3] - y_truth[:, :3], axis=1) / 1_000.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(t_days, np.maximum(pos_err_km, 1e-9), color=_color("st_lrps"), lw=1.4)
    ax.set_title(f"Batch Selected Position Error - scenario {sid}")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Position Error [km]")
    ax.grid(True, alpha=0.25, which="both")
    p = out_dir / "batch_selected_position_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    ric_km = compute_ric_errors(y_truth[:, :3], y_truth[:, 3:], y_st[:, :3]) / 1_000.0
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels_ric = ["Radial", "In-track", "Cross-track"]
    for k, label in enumerate(labels_ric):
        axes[k].plot(t_days, ric_km[:, k], color=_color("st_lrps"), lw=1.1)
        axes[k].set_ylabel(f"{label} [km]")
        axes[k].grid(True, alpha=0.25)
    axes[0].set_title(f"Batch Selected RIC Error - scenario {sid}")
    axes[-1].set_xlabel("Time [days]")
    fig.tight_layout()
    p = out_dir / "batch_selected_ric_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    return saved


def estimate_stlrps_equivalent_sh_degree(aggregate_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate which classical SH degree ST-LRPS resembles by error level."""

    by_model = {str(r["model"]).upper(): r for r in aggregate_rows}
    st = by_model.get("GPU_ST_LRPS_RK4")
    if not st:
        return {"status": "missing_st_lrps"}

    def _metric(metric_key: str) -> Dict[str, Any]:
        sh_points = []
        for model, row in by_model.items():
            deg = _model_degree(model)
            if model.startswith("GPU_SH") and deg is not None:
                try:
                    sh_points.append((deg, float(row[metric_key]), model))
                except Exception:
                    pass
        sh_points.sort()
        st_err = float(st[metric_key])
        if not sh_points or not np.isfinite(st_err):
            return {"status": "insufficient_data", "st_lrps_error": st_err}
        errs = np.array([p[1] for p in sh_points], dtype=np.float64)
        degrees = np.array([p[0] for p in sh_points], dtype=np.float64)
        monotonic = bool(np.all(np.diff(errs) <= 1e-12))
        closest = min(sh_points, key=lambda p: abs(p[1] - st_err))
        out = {
            "status": "ok" if monotonic else "non_monotonic_unreliable",
            "st_lrps_error": st_err,
            "closest_model": closest[2],
            "closest_degree": closest[0],
            "closest_error": closest[1],
            "monotonic": monotonic,
        }
        if not monotonic:
            return out
        if st_err > errs[0]:
            out["equivalent_degree_status"] = "worse_than_sh20"
            return out
        if st_err < errs[-1]:
            out["equivalent_degree_status"] = f"better_than_sh{int(degrees[-1])}"
            return out
        # errors decrease with degree; find enclosing interval
        for i in range(len(degrees) - 1):
            e_lo, e_hi = errs[i], errs[i + 1]
            if e_lo >= st_err >= e_hi:
                x0, x1 = degrees[i], degrees[i + 1]
                y0, y1 = math.log(max(e_lo, 1e-30)), math.log(max(e_hi, 1e-30))
                ys = math.log(max(st_err, 1e-30))
                frac = 0.0 if abs(y1 - y0) < 1e-30 else (ys - y0) / (y1 - y0)
                out["equivalent_degree_status"] = "interpolated"
                out["equivalent_degree"] = float(x0 + frac * (x1 - x0))
                return out
        return out

    return {
        "median_rms": _metric("median_rms_pos_err_km"),
        "p95_rms": _metric("p95_rms_pos_err_km"),
    }


def select_stlrps_scenarios(rows: List[Dict[str, Any]], scenarios_by_id: Dict[int, Scenario],
                            args: argparse.Namespace) -> Dict[str, Any]:
    st_rows = [
        r for r in rows
        if str(r.get("model", "")).upper().startswith("GPU_ST_LRPS_RK4")
        and r.get("status") in {"ok", "warning_negative_altitude"}
        and np.isfinite(float(r.get("rms_pos_err_km", np.nan)))
    ]
    source_label = "ST-LRPS"
    if not st_rows:
        from collections import defaultdict
        vals_by_sid: Dict[int, List[float]] = defaultdict(list)
        base_by_sid: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            if r.get("status") not in {"ok", "warning_negative_altitude"}:
                continue
            try:
                sid = int(r["scenario_id"])
                val = float(r.get("rms_pos_err_km", np.nan))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            vals_by_sid[sid].append(val)
            base_by_sid.setdefault(sid, dict(r))
        if not vals_by_sid:
            return {}
        st_rows = []
        for sid, vals in vals_by_sid.items():
            row = dict(base_by_sid[sid])
            row["model"] = "ALL_GPU_MODELS"
            row["rms_pos_err_km"] = float(np.mean(vals))
            st_rows.append(row)
        source_label = "comparison set"

    by_id = {int(r["scenario_id"]): r for r in st_rows}

    def _pick(label: str, override: Optional[int], key_fn: Any) -> Dict[str, Any]:
        if override is not None and override in by_id:
            row = by_id[override]
        else:
            row = key_fn(st_rows)
        sid = int(row["scenario_id"])
        sc = scenarios_by_id.get(sid)
        payload = dict(row)
        if sc is not None:
            payload.update({
                "hp_km": sc.hp_km, "ha_km": sc.ha_km, "a_km": sc.a_km,
                "e": sc.e, "inc_deg": sc.inc_deg, "raan_deg": sc.raan_deg,
                "argp_deg": sc.argp_deg, "ta_deg": sc.ta_deg,
            })
        payload["selection"] = label
        payload["selection_source"] = source_label
        return payload

    vals = np.array([float(r["rms_pos_err_km"]) for r in st_rows], dtype=np.float64)
    median = float(np.median(vals))
    mean = float(np.mean(vals))
    selected = {
        "best": _pick("best", args.plot_best_scenario_id,
                      lambda rr: min(rr, key=lambda r: float(r["rms_pos_err_km"]))),
        "worst": _pick("worst", args.plot_worst_scenario_id,
                       lambda rr: max(rr, key=lambda r: float(r["rms_pos_err_km"]))),
        "representative": _pick("representative", args.plot_representative_scenario_id,
                                lambda rr: min(rr, key=lambda r: abs(float(r["rms_pos_err_km"]) - median))),
        "mean_error": _pick("mean_error", None,
                            lambda rr: min(rr, key=lambda r: abs(float(r["rms_pos_err_km"]) - mean))),
    }
    selected["_selection_source"] = source_label
    return selected


def plot_gpu_batch_report_figures(
    aggregate_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    metrics_rows: List[Dict[str, Any]],
    results: List[BatchModelResult],
    truth: TruthTrajectorySet,
    scenarios: List[Scenario],
    selected: Dict[str, Any],
    equivalent: Dict[str, Any],
    plots_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    """Create publication-grade report figures for the GPU batch comparison.

    Visualization only — no metric value is recomputed here. CSV units stay in
    km; figures pick a readable display unit (km/m/cm) per axis.
    """

    plots_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_theme(getattr(args, "plot_theme", "report_light"))
    th = _ACTIVE_PLOT_THEME
    saved: List[Path] = []
    agg_by_model = {r["model"]: r for r in aggregate_rows}
    runtime_by_model = {r["model"]: r for r in runtime_rows}

    truth_integrator = str(getattr(args, "truth_integrator", "DOP853"))
    truth_label = f"{str(args.truth).upper()} {truth_integrator}"
    n_scn = len(scenarios)
    duration = float(getattr(args, "duration_days", 0.0) or 0.0)
    ctx = f"N = {n_scn} scenarios  ·  {duration:g} d  ·  errors vs {truth_label}"

    def _safe(x: Any, default: float = float("inf")) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    def _fmt(v: float) -> str:
        return f"{v:.3g}"

    def _model_vals(m: str, key: str = "rms_pos_err_km") -> List[float]:
        out = []
        for r in metrics_rows:
            if r.get("model") != m or r.get("status") not in {"ok", "warning_negative_altitude"}:
                continue
            v = _safe(r.get(key), default=float("nan"))
            if math.isfinite(v):
                out.append(v)
        return out

    # Best (lowest median RMS) first.
    models = sorted(agg_by_model.keys(),
                    key=lambda m: _safe(agg_by_model[m].get("median_rms_pos_err_km")))
    counts = [len(_model_vals(m)) for m in models]
    n_dist = max(counts) if counts else 0
    small_n = 0 < n_dist < 8

    # ----- 1. Accuracy ranking (horizontal lollipop) ---------------------
    if models:
        med_km = [_safe(agg_by_model[m].get("median_rms_pos_err_km"), 0.0) for m in models]
        p95_km = [_safe(agg_by_model[m].get("p95_rms_pos_err_km"), 0.0) for m in models]
        unit, mult = select_length_unit(max(_finite_positive(med_km + p95_km) or [0.0]))
        med = [v * mult for v in med_km]
        p95 = [v * mult for v in p95_km]
        labels = [display_label(m) for m in models]
        logx = _should_log(med_km + p95_km)
        y = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.62 * len(models) + 1.6)))
        x0 = min(_finite_positive(med + p95) or [0.0]) * 0.5 if logx else 0.0
        for yi, m, mv, pv in zip(y, models, med, p95):
            c = model_color(m)
            ax.hlines(yi, x0, mv, color=c, lw=2.6, alpha=0.45, zorder=2)
            ax.scatter(mv, yi, color=c, marker=model_marker(m),
                       s=150 if _is_stlrps(m) else 80,
                       edgecolor=th["edge"], linewidth=0.6, zorder=model_zorder(m))
            ax.scatter(pv, yi, facecolors="none", edgecolors=c, marker="D",
                       s=46, linewidth=1.3, zorder=4)
            anchor = max(mv, pv)
            xt = anchor * 1.10 if logx else anchor + 0.02 * max(med + p95 + [1e-9])
            ax.text(xt, yi, _fmt(mv), va="center", ha="left", fontsize=9, color=th["text"])
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # best at top
        if logx:
            ax.set_xscale("log")
        ax.margins(x=0.16)  # headroom for end-of-bar value labels
        from matplotlib.lines import Line2D
        proxies = [
            Line2D([0], [0], marker="o", color=th["muted"], ls="none", label="Median RMS"),
            Line2D([0], [0], marker="D", markerfacecolor="none", markeredgecolor=th["muted"],
                   color=th["muted"], ls="none", label="P95 RMS"),
        ]
        ax.legend(handles=proxies, loc="upper right", frameon=False)
        _style_ax(ax, title="GPU RK4 Accuracy Ranking",
                  xlabel=f"RMS Position Error [{unit}]",
                  subtitle=f"Lower is better.  {ctx}")
        ax.grid(True, axis="y", alpha=0.0)
        _highlight_ticklabels(ax, labels, axis="y")
        fig.tight_layout()
        p = plots_dir / "gpu_accuracy_ranking_bar.png"
        fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 2. Runtime vs accuracy ----------------------------------------
    if models:
        pts = []
        for m in models:
            x = _safe(runtime_by_model.get(m, {}).get("total_runtime_s"), default=float("nan"))
            yk = _safe(agg_by_model[m].get("median_rms_pos_err_km"), default=float("nan"))
            if math.isfinite(x) and math.isfinite(yk):
                pts.append((x, yk, m))
        unit, mult = select_length_unit(max(_finite_positive([p[1] for p in pts]) or [0.0]))
        logy = _should_log([p[1] for p in pts])
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        if pts:
            # Pareto frontier (lower-left): cheapest run achieving each new best error.
            front = []
            best = float("inf")
            for x, yk, m in sorted(pts, key=lambda t: t[0]):
                if yk < best - 1e-30:
                    best = yk
                    front.append((x, yk * mult))
            if len(front) >= 2:
                ax.step([f[0] for f in front], [f[1] for f in front], where="post",
                        ls="--", lw=1.2, color=th["muted"], alpha=0.7, zorder=1,
                        label="Pareto front")
            for x, yk, m in pts:
                ax.scatter(x, yk * mult, color=model_color(m), marker=model_marker(m),
                           s=model_marker_size(m), edgecolor=th["edge"],
                           linewidth=1.0 if _is_stlrps(m) else 0.6, zorder=model_zorder(m))
                ax.annotate(display_label(m), (x, yk * mult), xytext=(8, 5),
                            textcoords="offset points", fontsize=9.5 if _is_stlrps(m) else 9,
                            fontweight="bold" if _is_stlrps(m) else "normal",
                            color=model_color(m) if _is_stlrps(m) else th["text"])
            if logy:
                ax.set_yscale("log")
            _legend(ax, loc="upper right")
        else:
            _empty_note(ax, "No runtime/accuracy data available.")
        _style_ax(ax, title="Runtime vs Accuracy",
                  xlabel="Total GPU Runtime [s]",
                  ylabel=f"Median RMS Position Error [{unit}]",
                  subtitle=f"Lower-left is better (faster + more accurate).  {ctx}")
        fig.tight_layout()
        p = plots_dir / "gpu_runtime_vs_accuracy.png"
        fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 3. RMS error distribution -------------------------------------
    data = [_model_vals(m) for m in models]
    all_vals = [v for d in data for v in d]
    unit, mult = select_length_unit(max(_finite_positive(all_vals) or [0.0]))
    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.62 * len(models) + 1.8)))
    if any(data):
        y = np.arange(len(models))
        logx = _should_log(all_vals)
        if small_n:
            rng = np.random.default_rng(0)
            for yi, m, vals in zip(y, models, data):
                if not vals:
                    continue
                vv = np.asarray(vals) * mult
                jitter = (rng.random(len(vv)) - 0.5) * 0.28
                ax.scatter(vv, np.full_like(vv, yi) + jitter, color=model_color(m),
                           marker=model_marker(m), s=70 if _is_stlrps(m) else 42,
                           alpha=0.85, edgecolor=th["edge"], linewidth=0.4,
                           zorder=model_zorder(m))
                ax.scatter(np.median(vv), yi, color=th["text"], marker="|", s=420,
                           linewidth=2.2, zorder=6)
            subtitle = f"N={n_dist} is small — strip plot is diagnostic, not statistical.  {ctx}"
        else:
            box = ax.boxplot(
                [np.asarray(d) * mult for d in data], vert=False, patch_artist=True,
                showfliers=False, widths=0.6, positions=y,
                medianprops=dict(color=th["text"], lw=1.8),
            )
            for patch, m in zip(box["boxes"], models):
                patch.set_facecolor(model_color(m))
                patch.set_alpha(0.45 if not _is_stlrps(m) else 0.65)
                patch.set_edgecolor(model_color(m))
            subtitle = ctx
        if logx:
            ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels([display_label(m) for m in models])
        ax.invert_yaxis()
        _style_ax(ax, title="RMS Position Error Distribution",
                  xlabel=f"RMS Position Error [{unit}]", subtitle=subtitle)
        ax.grid(True, axis="y", alpha=0.0)
        _highlight_ticklabels(ax, [display_label(m) for m in models], axis="y")
    else:
        _empty_note(ax, "Errors are below plotting threshold for this short run.")
        _style_ax(ax, title="RMS Position Error Distribution", subtitle=ctx)
    fig.tight_layout()
    p = plots_dir / "gpu_rms_error_distribution_boxplot.png"
    fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 4. Histograms --------------------------------------------------
    if models:
        fig, axes = plt.subplots(len(models), 1,
                                 figsize=(9, max(4.2, 1.4 * len(models))), sharex=True)
        if len(models) == 1:
            axes = [axes]
        for ax, m, vals in zip(axes, models, data):
            if vals:
                ax.hist(np.asarray(vals) * mult, bins=min(24, max(6, n_dist)),
                        color=model_color(m), alpha=0.85, edgecolor=th["bg"], linewidth=0.4)
            else:
                _empty_note(ax, "no data")
            ax.set_ylabel(display_label(m), rotation=0, ha="right", va="center",
                          fontsize=9, color=(_ST_LRPS_COLOR if _is_stlrps(m) else th["text"]),
                          fontweight="bold" if _is_stlrps(m) else "normal")
            ax.grid(True, alpha=0.4)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[-1].set_xlabel(f"RMS Position Error [{unit}]")
        axes[0].set_title("RMS Error Histograms per Model")
        fig.tight_layout()
        p = plots_dir / "gpu_rms_error_histograms.png"
        fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 5. ST-LRPS equivalent SH degree -------------------------------
    sh_points = []
    for m in models:
        deg = _model_degree(m)
        if deg is not None and m.upper().startswith("GPU_SH"):
            sh_points.append((deg, _safe(agg_by_model[m].get("median_rms_pos_err_km")),
                              _safe(agg_by_model[m].get("p95_rms_pos_err_km"))))
    sh_points.sort()
    st = agg_by_model.get("GPU_ST_LRPS_RK4")
    st_med_km = _safe(st.get("median_rms_pos_err_km")) if st else float("nan")
    st_p95_km = _safe(st.get("p95_rms_pos_err_km")) if st else float("nan")
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    span_km = [p[1] for p in sh_points] + [p[2] for p in sh_points]
    if math.isfinite(st_med_km):
        span_km.append(st_med_km)
    unit, mult = select_length_unit(max(_finite_positive(span_km) or [0.0]))
    if sh_points:
        degs = [p[0] for p in sh_points]
        ax.plot(degs, [p[1] * mult for p in sh_points], marker="o", color="#3D5A80",
                lw=2.0, label="SH median RMS", zorder=3)
        ax.plot(degs, [p[2] * mult for p in sh_points], marker="s", ls="--", color="#6C8EBF",
                lw=1.6, label="SH P95 RMS", zorder=3)
        if math.isfinite(st_med_km):
            ax.axhline(st_med_km * mult, color=_ST_LRPS_COLOR, lw=2.6, ls="-",
                       label="ST-LRPS median RMS", zorder=4)
        if math.isfinite(st_p95_km):
            ax.axhline(st_p95_km * mult, color=_ST_LRPS_COLOR, lw=1.6, ls=":",
                       alpha=0.8, label="ST-LRPS P95 RMS", zorder=4)
        # Annotate the equivalent-degree estimate, if available.
        med_eq = equivalent.get("median_rms", {}) if isinstance(equivalent, dict) else {}
        eq_txt = None
        if med_eq.get("equivalent_degree") is not None:
            eq_deg = float(med_eq["equivalent_degree"])
            eq_txt = f"≈ SH{eq_deg:.0f}"
            ax.axvline(eq_deg, color=_ST_LRPS_COLOR, lw=1.0, ls=":", alpha=0.6)
        else:
            status_map = {
                "worse_than_sh20": "below SH20",
                "better_than_sh200": "above SH200",
            }
            raw = str(med_eq.get("equivalent_degree_status", ""))
            eq_txt = status_map.get(raw)
            if eq_txt is None and raw.startswith("better_than_sh"):
                eq_txt = f"above {raw.replace('better_than_', '').upper()}"
        if eq_txt and math.isfinite(st_med_km):
            ax.annotate(f"ST-LRPS {eq_txt}", xy=(degs[len(degs) // 2], st_med_km * mult),
                        xytext=(0, 8), textcoords="offset points", color=_ST_LRPS_COLOR,
                        fontweight="bold", fontsize=10, ha="center")
        if _should_log(span_km):
            ax.set_yscale("log")
        _legend(ax, loc="best")
    else:
        _empty_note(ax, "No spherical-harmonic baselines available for comparison.")
    _style_ax(ax, title="ST-LRPS Equivalent Spherical-Harmonic Degree",
              xlabel="Spherical Harmonic Degree",
              ylabel=f"RMS Position Error [{unit}]",
              subtitle=f"Where ST-LRPS sits on the SH error ladder.  {ctx}")
    fig.tight_layout()
    p = plots_dir / "stlrps_equivalent_sh_degree.png"
    fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 6-7. Error vs inclination / altitude --------------------------
    for xkey, xlabel, fname in [
        ("inc_deg", "Inclination [deg]", "gpu_error_vs_inclination_all_models.png"),
        ("hp_km", "Periselene Altitude [km]", "gpu_error_vs_altitude_all_models.png"),
    ]:
        all_y = [v for m in models for v in _model_vals(m)]
        unit, mult = select_length_unit(max(_finite_positive(all_y) or [0.0]))
        fig, ax = plt.subplots(figsize=(9, 5.4))
        plotted = False
        for m in models:
            rows = [r for r in metrics_rows
                    if r.get("model") == m and r.get("status") in {"ok", "warning_negative_altitude"}]
            xs = [_safe(r.get(xkey), float("nan")) for r in rows]
            ys = [_safe(r.get("rms_pos_err_km"), float("nan")) * mult for r in rows]
            if rows:
                ax.scatter(xs, ys, color=model_color(m), marker=model_marker(m),
                           s=70 if _is_stlrps(m) else 26, alpha=0.85,
                           edgecolor=th["edge"], linewidth=0.3,
                           zorder=model_zorder(m), label=display_label(m))
                plotted = True
        if plotted:
            if _should_log(all_y):
                ax.set_yscale("log")
            _legend(ax, outside=True)
        else:
            _empty_note(ax, "Errors are below plotting threshold for this short run.")
        _style_ax(ax, title=f"Error vs {xlabel.split()[0]}", xlabel=xlabel,
                  ylabel=f"RMS Position Error [{unit}]", subtitle=ctx)
        fig.tight_layout()
        p = plots_dir / fname
        fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- Ensemble time-series ------------------------------------------
    if scenarios and truth.t_by_scenario:
        common_t = next(iter(truth.t_by_scenario.values()))
        t_days = np.asarray(common_t) / 86400.0
        med_by_model: Dict[str, np.ndarray] = {}
        band_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        ric_by_model: Dict[str, np.ndarray] = {}
        for result in results:
            if result.status != "ok":
                continue
            pos_err, ric_err = [], []
            for i, sc in enumerate(scenarios):
                if sc.scenario_id not in truth.t_by_scenario:
                    continue
                y_model = interpolate_state_to_times(result.t, result.y[:, i, :], common_t)
                y_truth = truth.y_by_scenario[sc.scenario_id]
                pos_err.append(np.linalg.norm(y_model[:, :3] - y_truth[:, :3], axis=1) / 1000.0)
                ric_err.append(compute_ric_errors(y_truth[:, :3], y_truth[:, 3:], y_model[:, :3]) / 1000.0)
            if not pos_err:
                continue
            pos_arr = np.asarray(pos_err)
            med_by_model[result.display_name] = np.median(pos_arr, axis=0)
            band_by_model[result.display_name] = (np.percentile(pos_arr, 25, axis=0),
                                                  np.percentile(pos_arr, 75, axis=0))
            ric_by_model[result.display_name] = np.asarray(ric_err)

        pos_max_km = max(_finite_positive([float(np.max(v)) for v in med_by_model.values()]) or [0.0])
        unit, mult = select_length_unit(pos_max_km)

        fig, ax = plt.subplots(figsize=(10, 5.6))
        if med_by_model and pos_max_km > 1e-12:
            for name, curve in med_by_model.items():
                lo, hi = band_by_model[name]
                ax.plot(t_days, curve * mult, color=model_color(name),
                        lw=model_linewidth(name), label=display_label(name),
                        zorder=model_zorder(name))
                ax.fill_between(t_days, lo * mult, hi * mult, color=model_color(name), alpha=0.10)
            _legend(ax, outside=True)
        else:
            _empty_note(ax, "Errors are below plotting threshold for this short run.")
        _style_ax(ax, title="Ensemble Position Error vs Time",
                  xlabel="Time [days]", ylabel=f"Median Position Error [{unit}]",
                  subtitle=f"Median across scenarios; shaded band = 25–75%.  {ctx}")
        fig.tight_layout()
        p = plots_dir / "ensemble_mean_position_error_vs_time.png"
        fig.savefig(p); plt.close(fig); saved.append(p)

        ric_curves = {name: np.sqrt(np.mean(arr ** 2, axis=0)) for name, arr in ric_by_model.items()}
        ric_max_km = max(_finite_positive(
            [float(np.max(c)) for c in ric_curves.values()]) or [0.0])
        runit, rmult = select_length_unit(ric_max_km)
        fig_ric, axes_ric = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True)
        for k, lbl in enumerate(["Radial", "Along-track", "Cross-track"]):
            if ric_curves and ric_max_km > 1e-12:
                for name, c in ric_curves.items():
                    axes_ric[k].plot(t_days, c[:, k] * rmult, color=model_color(name),
                                     lw=model_linewidth(name), label=display_label(name),
                                     zorder=model_zorder(name))
            else:
                _empty_note(axes_ric[k], "below plotting threshold")
            axes_ric[k].set_ylabel(f"{lbl} RMS [{runit}]")
            axes_ric[k].grid(True, alpha=0.45)
            for spine in ("top", "right"):
                axes_ric[k].spines[spine].set_visible(False)
        axes_ric[-1].set_xlabel("Time [days]")
        axes_ric[0].set_title("Ensemble RIC RMS Error vs Time")
        if ric_curves and ric_max_km > 1e-12:
            _legend(axes_ric[0], outside=True)
        fig_ric.tight_layout()
        p = plots_dir / "ensemble_ric_rms_vs_time.png"
        fig_ric.savefig(p); plt.close(fig_ric); saved.append(p)

    # ----- Selected ST-LRPS scenarios ------------------------------------
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    selection_source = str(selected.get("_selection_source", "ST-LRPS"))
    for label in ("best", "representative", "worst"):
        item = selected.get(label)
        if not item:
            continue
        sid = int(item["scenario_id"])
        sc = scenario_by_id.get(sid)
        if sc is None or sid not in truth.t_by_scenario:
            continue
        idx = scenarios.index(sc)
        t_truth = truth.t_by_scenario[sid]
        y_truth = truth.y_by_scenario[sid]
        t_days = np.asarray(t_truth) / 86400.0

        pos_by_model: Dict[str, np.ndarray] = {}
        alt_by_model: Dict[str, np.ndarray] = {}
        ric_by_model = {}
        for result in results:
            if result.status != "ok":
                continue
            y_model = interpolate_state_to_times(result.t, result.y[:, idx, :], t_truth)
            pos_by_model[result.display_name] = (
                np.linalg.norm(y_model[:, :3] - y_truth[:, :3], axis=1) / 1000.0)
            alt_by_model[result.display_name] = (
                np.linalg.norm(y_model[:, :3], axis=1) - np.linalg.norm(y_truth[:, :3], axis=1)) / 1000.0
            ric_by_model[result.display_name] = (
                compute_ric_errors(y_truth[:, :3], y_truth[:, 3:], y_model[:, :3]) / 1000.0)

        pos_max_km = max(_finite_positive([float(np.max(v)) for v in pos_by_model.values()]) or [0.0])
        unit, mult = select_length_unit(pos_max_km)
        sub = f"Scenario {sid}: hp={sc.hp_km:.0f} km, i={sc.inc_deg:.1f}°.  vs {truth_label}"

        # Position error
        fig_pos, ax_pos = plt.subplots(figsize=(10, 5.4))
        if pos_by_model and pos_max_km > 1e-12:
            use_log = bool(getattr(args, "plot_error_logscale", False)) or _should_log(
                [float(np.max(v)) for v in pos_by_model.values()])
            for name, curve in pos_by_model.items():
                ax_pos.plot(t_days, np.maximum(curve * mult, 1e-12 if use_log else 0.0),
                            color=model_color(name), lw=model_linewidth(name),
                            label=display_label(name), zorder=model_zorder(name))
            if use_log:
                ax_pos.set_yscale("log")
            _legend(ax_pos, outside=True)
        else:
            _empty_note(ax_pos, "Errors are below plotting threshold for this short run.")
        _style_ax(ax_pos, title=f"{label.title()} {selection_source} Scenario: Position Error",
                  xlabel="Time [days]", ylabel=f"Position Error [{unit}]", subtitle=sub)
        fig_pos.tight_layout()
        p = plots_dir / f"selected_{label}_position_error_all_models.png"
        fig_pos.savefig(p); plt.close(fig_pos); saved.append(p)

        # Altitude error
        alt_max_km = max(_finite_positive(
            [float(np.max(np.abs(v))) for v in alt_by_model.values()]) or [0.0])
        aunit, amult = select_length_unit(alt_max_km)
        fig_alt, ax_alt = plt.subplots(figsize=(10, 5.0))
        if alt_by_model and alt_max_km > 1e-12:
            for name, curve in alt_by_model.items():
                ax_alt.plot(t_days, curve * amult, color=model_color(name),
                            lw=model_linewidth(name), label=display_label(name),
                            zorder=model_zorder(name))
            _legend(ax_alt, outside=True)
        else:
            _empty_note(ax_alt, "Errors are below plotting threshold for this short run.")
        _style_ax(ax_alt, title=f"{label.title()} {selection_source} Scenario: Altitude Error",
                  xlabel="Time [days]", ylabel=f"Altitude Error [{aunit}]", subtitle=sub)
        fig_alt.tight_layout()
        p = plots_dir / f"selected_{label}_altitude_error_all_models.png"
        fig_alt.savefig(p); plt.close(fig_alt); saved.append(p)

        # RIC error
        ric_max_km = max(_finite_positive(
            [float(np.max(np.abs(v))) for v in ric_by_model.values()]) or [0.0])
        runit, rmult = select_length_unit(ric_max_km)
        fig_ric_sel, axes_sel = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True)
        for k, lbl in enumerate(["Radial", "Along-track", "Cross-track"]):
            if ric_by_model and ric_max_km > 1e-12:
                for name, curve in ric_by_model.items():
                    axes_sel[k].plot(t_days, curve[:, k] * rmult, color=model_color(name),
                                     lw=model_linewidth(name), label=display_label(name),
                                     zorder=model_zorder(name))
            else:
                _empty_note(axes_sel[k], "below plotting threshold")
            axes_sel[k].set_ylabel(f"{lbl} [{runit}]")
            axes_sel[k].grid(True, alpha=0.45)
            for spine in ("top", "right"):
                axes_sel[k].spines[spine].set_visible(False)
        axes_sel[0].set_title(f"{label.title()} {selection_source} Scenario {sid}: RIC Error")
        axes_sel[-1].set_xlabel("Time [days]")
        if ric_by_model and ric_max_km > 1e-12:
            _legend(axes_sel[0], outside=True)
        fig_ric_sel.tight_layout()
        p = plots_dir / f"selected_{label}_ric_error_all_models.png"
        fig_ric_sel.savefig(p); plt.close(fig_ric_sel); saved.append(p)

        # 3D trajectory (optional)
        if getattr(args, "plot_3d", False):
            fig_3d = plt.figure(figsize=(8, 7))
            ax_3d = fig_3d.add_subplot(111, projection="3d")
            rk = y_truth[:, :3] / 1000.0
            ax_3d.plot(rk[:, 0], rk[:, 1], rk[:, 2], color=_TRUTH_COLOR, lw=2.5, label=truth_label)
            for result in results:
                if result.status != "ok":
                    continue
                y_model = interpolate_state_to_times(result.t, result.y[:, idx, :], t_truth)
                rk = y_model[:, :3] / 1000.0
                ax_3d.plot(rk[:, 0], rk[:, 1], rk[:, 2], color=model_color(result.display_name),
                           lw=model_linewidth(result.display_name), label=display_label(result.display_name))
            ax_3d.set_title(f"{label.title()} ST-LRPS Scenario {sid}: 3D Trajectory")
            ax_3d.set_xlabel("X [km]"); ax_3d.set_ylabel("Y [km]"); ax_3d.set_zlabel("Z [km]")
            ax_3d.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
            p = plots_dir / f"selected_{label}_trajectory_3d_all_models.png"
            fig_3d.savefig(p, bbox_inches="tight"); plt.close(fig_3d); saved.append(p)

    return saved


# =============================================================================
# CSV / JSON helpers
# =============================================================================

def _ensure_dir(path: Path) -> None:
    """Create parent directories for a file path if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_scenarios_csv(scenarios: List[Scenario], out_dir: Path) -> None:
    fieldnames = ["scenario_id", "hp_km", "ha_km", "a_km", "e",
                  "inc_deg", "raan_deg", "argp_deg", "ta_deg"]
    p = out_dir / "scenarios.csv"
    _ensure_dir(p)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in scenarios:
            w.writerow({k: getattr(s, k) for k in fieldnames})


def _scenario_count_for_args(args: argparse.Namespace) -> int:
    count = max(0, int(args.random_scenarios))
    limit = getattr(args, "scenario_limit", None)
    if limit is not None:
        count = min(count, max(0, int(limit)))
    return count


def _sampling_metadata(args: argparse.Namespace, scenario_count: Optional[int] = None) -> Dict[str, Any]:
    count = _scenario_count_for_args(args) if scenario_count is None else int(scenario_count)
    method = str(getattr(args, "sampling_method", "random"))
    note = _sobol_note(method, int(args.random_scenarios))
    return {
        "scenario_count": count,
        "requested_random_scenarios": int(args.random_scenarios),
        "scenario_limit": (
            None if getattr(args, "scenario_limit", None) is None
            else int(args.scenario_limit)
        ),
        "sampling_method": method,
        "scenario_seed": int(args.scenario_seed),
        "scenario_mode": str(args.scenario_mode),
        "inclination_sampling": str(getattr(args, "inclination_sampling", "uniform_deg")),
        "altitude_min_km": float(args.altitude_min_km),
        "altitude_max_km": float(args.altitude_max_km),
        "ecc_min": float(args.ecc_min),
        "ecc_max": float(args.ecc_max),
        "inc_min_deg": float(args.inc_min_deg),
        "inc_max_deg": float(args.inc_max_deg),
        "raan_min_deg": float(args.raan_min_deg),
        "raan_max_deg": float(args.raan_max_deg),
        "argp_min_deg": float(args.argp_min_deg),
        "argp_max_deg": float(args.argp_max_deg),
        "ta_min_deg": float(args.ta_min_deg),
        "ta_max_deg": float(args.ta_max_deg),
        "altitude_bounds_km": {
            "min": float(args.altitude_min_km),
            "max": float(args.altitude_max_km),
        },
        "eccentricity_bounds": {
            "min": float(args.ecc_min),
            "max": float(args.ecc_max),
        },
        "inclination_bounds_deg": {
            "min": float(args.inc_min_deg),
            "max": float(args.inc_max_deg),
        },
        "angular_bounds_deg": {
            "raan": {"min": float(args.raan_min_deg), "max": float(args.raan_max_deg)},
            "argp": {"min": float(args.argp_min_deg), "max": float(args.argp_max_deg)},
            "ta": {"min": float(args.ta_min_deg), "max": float(args.ta_max_deg)},
        },
        "module_name": __name__,
        "code_path": str(Path(__file__).resolve()),
        "sampling_note": note,
        "lhs_append_mode": (
            "blockwise" if method == "lhs" and bool(getattr(args, "allow_lhs_append", False))
            else None
        ),
        "warning": (
            "Blockwise LHS append is not equivalent to a single global LHS design."
            if method == "lhs" and bool(getattr(args, "allow_lhs_append", False))
            else ""
        ),
    }


def _scenario_generation_args(args: argparse.Namespace, n: int) -> argparse.Namespace:
    child = argparse.Namespace(**vars(args))
    child.random_scenarios = int(n)
    child.scenario_limit = None
    return child


def _scenario_numeric_tuple(s: Scenario) -> Tuple[float, ...]:
    return (
        float(s.hp_km), float(s.ha_km), float(s.a_km), float(s.e),
        float(s.inc_deg), float(s.raan_deg), float(s.argp_deg), float(s.ta_deg),
    )


def _scenarios_match(a: Scenario, b: Scenario, atol: float = 1e-9) -> bool:
    if int(a.scenario_id) != int(b.scenario_id):
        return False
    av = _scenario_numeric_tuple(a)
    bv = _scenario_numeric_tuple(b)
    return all(math.isclose(x, y, rel_tol=0.0, abs_tol=atol) for x, y in zip(av, bv))


def _renumber_scenarios(scenarios: List[Scenario], start_id: int) -> List[Scenario]:
    out: List[Scenario] = []
    for offset, scenario in enumerate(scenarios):
        out.append(replace(scenario, scenario_id=int(start_id + offset)))
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _scenario_manifest_row(
    scenario: Scenario,
    args: argparse.Namespace,
    csv_mode: bool = False,
) -> Dict[str, Any]:
    raw = scenario.raw_unit_sample
    raw_list = None if raw is None else [float(x) for x in raw]
    row: Dict[str, Any] = {
        "scenario_id": int(scenario.scenario_id),
        "sampling_method": str(getattr(scenario, "sampling_method", getattr(args, "sampling_method", "random"))),
        "scenario_seed": int(args.scenario_seed),
        "scenario_mode": str(args.scenario_mode),
        "inclination_sampling": str(getattr(args, "inclination_sampling", "uniform_deg")),
        "raw_unit_sample": raw_list,
        "hp_km": float(scenario.hp_km),
        "ha_km": float(scenario.ha_km),
        "a_km": float(scenario.a_km),
        "e": float(scenario.e),
        "inc_deg": float(scenario.inc_deg),
        "raan_deg": float(scenario.raan_deg),
        "argp_deg": float(scenario.argp_deg),
        "ta_deg": float(scenario.ta_deg),
    }
    for i in range(SCENARIO_UNIT_DIM):
        row[f"unit_u{i}"] = "" if raw_list is None or i >= len(raw_list) else float(raw_list[i])
    if csv_mode:
        row["raw_unit_sample"] = "" if raw_list is None else json.dumps(raw_list, separators=(",", ":"))
    return row


def _write_scenario_manifest(
    scenarios: List[Scenario],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    metadata = _sampling_metadata(args, len(scenarios))
    rows = [_scenario_manifest_row(s, args, csv_mode=False) for s in scenarios]
    payload = {
        "metadata": metadata,
        "scenarios": rows,
    }

    json_path = out_dir / SCENARIO_MANIFEST_JSON
    _ensure_dir(json_path)
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=4),
        encoding="utf-8",
    )

    csv_path = out_dir / SCENARIO_MANIFEST_CSV
    csv_rows = [_scenario_manifest_row(s, args, csv_mode=True) for s in scenarios]
    fieldnames = [
        "scenario_id", "sampling_method", "scenario_seed", "scenario_mode",
        "inclination_sampling", "raw_unit_sample",
        "unit_u0", "unit_u1", "unit_u2", "unit_u3", "unit_u4", "unit_u5",
        "hp_km", "ha_km", "a_km", "e",
        "inc_deg", "raan_deg", "argp_deg", "ta_deg",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)


def _manifest_value_text(value: Any) -> str:
    if value is None:
        return "<missing>"
    return str(value)


def _manifest_values_equal(old: Any, new: Any) -> bool:
    if old is None:
        return new is None
    if isinstance(new, float):
        try:
            return math.isclose(float(old), float(new), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return old == new


def _verify_scenario_manifest_matches(
    manifest: Dict[str, Any],
    args: argparse.Namespace,
    *,
    require_count: bool = True,
) -> None:
    metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
    expected = _sampling_metadata(args)
    fields = [
        "sampling_method",
        "scenario_seed",
        "scenario_mode",
        "inclination_sampling",
        "altitude_min_km",
        "altitude_max_km",
        "ecc_min",
        "ecc_max",
        "inc_min_deg",
        "inc_max_deg",
        "raan_min_deg",
        "raan_max_deg",
        "argp_min_deg",
        "argp_max_deg",
        "ta_min_deg",
        "ta_max_deg",
    ]
    if require_count:
        fields[2:2] = ["scenario_count", "requested_random_scenarios", "scenario_limit"]
    for field in fields:
        old = metadata.get(field)
        new = expected.get(field)
        if not _manifest_values_equal(old, new):
            raise ValueError(
                "Existing scenario_manifest uses "
                f"{field}={_manifest_value_text(old)} but current request uses "
                f"{_manifest_value_text(new)}."
            )


def _scenario_from_manifest_row(row: Dict[str, Any]) -> Scenario:
    raw = row.get("raw_unit_sample")
    raw_list = None
    if isinstance(raw, list):
        raw_list = [float(x) for x in raw]
    a_km = float(row["a_km"])
    e = float(row["e"])
    inc_deg = float(row["inc_deg"])
    raan_deg = float(row["raan_deg"])
    argp_deg = float(row["argp_deg"])
    ta_deg = float(row["ta_deg"])
    state = _state_from_elements(a_km * 1_000.0, e, inc_deg, raan_deg, argp_deg, ta_deg)
    return Scenario(
        scenario_id=int(row["scenario_id"]),
        hp_km=float(row["hp_km"]),
        ha_km=float(row["ha_km"]),
        a_km=a_km,
        e=e,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        ta_deg=ta_deg,
        initial_state=state,
        raw_unit_sample=raw_list,
        sampling_method=str(row.get("sampling_method", "random")),
    )


def _load_scenarios_from_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    *,
    require_count: bool = True,
) -> List[Scenario]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_scenario_manifest_matches(manifest, args, require_count=require_count)
    rows = manifest.get("scenarios", [])
    if not isinstance(rows, list):
        raise ValueError("Existing scenario_manifest has invalid scenarios payload.")
    scenarios = [_scenario_from_manifest_row(row) for row in rows]
    if require_count and len(scenarios) != _scenario_count_for_args(args):
        raise ValueError(
            "Existing scenario_manifest uses scenario_count="
            f"{len(scenarios)} but current request uses {_scenario_count_for_args(args)}."
        )
    return scenarios


def prepare_scenarios(args: argparse.Namespace, out_dir: Path) -> List[Scenario]:
    manifest_path = out_dir / SCENARIO_MANIFEST_JSON
    use_existing = (
        bool(getattr(args, "resume", False))
        or bool(getattr(args, "reuse_cache", False))
        or bool(getattr(args, "rebuild_metrics", False))
        or int(getattr(args, "append_scenarios", 0) or 0) > 0
    )
    if use_existing and manifest_path.exists():
        existing = _load_scenarios_from_manifest(manifest_path, args, require_count=False)
        existing_count = len(existing)
        append_count = max(0, int(getattr(args, "append_scenarios", 0) or 0))
        if append_count > 0:
            target_count = existing_count + append_count
        elif bool(getattr(args, "rebuild_metrics", False)):
            target_count = existing_count
        else:
            target_count = int(args.random_scenarios)

        if target_count < existing_count:
            raise ValueError(
                "Existing scenario_manifest uses scenario_count="
                f"{existing_count} but current request targets {target_count}. "
                "Use a new output directory or request at least the existing count."
            )
        if target_count == existing_count:
            scenarios = existing
            _write_scenarios_csv(scenarios, out_dir)
            print(f"[cache] Scenario manifest found: {len(scenarios)} scenarios.", flush=True)
            return scenarios

        method = str(getattr(args, "sampling_method", "random"))
        if method == "lhs" and not bool(getattr(args, "allow_lhs_append", False)):
            raise ValueError(
                f"Existing LHS manifest has {existing_count} scenarios. "
                "LHS is not naturally nested. Use Sobol/Sobol-scrambled for "
                "extendable benchmark sets, or rerun a fresh benchmark. "
                "Use --allow-lhs-append for explicit blockwise LHS append."
            )

        if method == "lhs":
            block_args = _scenario_generation_args(args, append_count or (target_count - existing_count))
            block_args.scenario_seed = int(args.scenario_seed) + existing_count
            new_block = generate_validation_scenarios(block_args)
            scenarios = existing + _renumber_scenarios(new_block, existing_count)
            print("[cache] WARNING: blockwise LHS append is not equivalent to a single "
                  "global LHS design.", flush=True)
        else:
            generated = generate_validation_scenarios(_scenario_generation_args(args, target_count))
            for old, new in zip(existing, generated[:existing_count]):
                if not _scenarios_match(old, new):
                    raise ValueError(
                        "Existing scenario_manifest is incompatible with regenerated "
                        f"{method} sequence at scenario_id={old.scenario_id}."
                    )
            scenarios = generated

        print(f"[cache] Extending scenario manifest: {existing_count} -> {len(scenarios)}.",
              flush=True)
        _write_scenarios_csv(scenarios, out_dir)
        _write_scenario_manifest(scenarios, args, out_dir)
        return scenarios

    if bool(getattr(args, "resume", False)) and manifest_path.exists():
        scenarios = _load_scenarios_from_manifest(manifest_path, args)
        _write_scenarios_csv(scenarios, out_dir)
        print(f"[scenarios] resume: loaded {len(scenarios)} scenarios from {manifest_path}",
              flush=True)
        return scenarios

    scenarios = generate_validation_scenarios(args)
    if getattr(args, "scenario_limit", None) is not None:
        scenarios = scenarios[:int(args.scenario_limit)]
    note = _sobol_note(str(getattr(args, "sampling_method", "random")), int(args.random_scenarios))
    if note:
        print(f"[scenarios] NOTE: {note}", flush=True)
    _write_scenarios_csv(scenarios, out_dir)
    _write_scenario_manifest(scenarios, args, out_dir)
    return scenarios


def _append_metrics_csv(metrics: Dict, path: Path, write_header: bool) -> None:
    _ensure_dir(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METRICS_FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(metrics)


def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _cache_requested(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "cache_trajectories", False)
        or getattr(args, "reuse_cache", False)
        or getattr(args, "rebuild_metrics", False)
        or getattr(args, "resume", False)
        or int(getattr(args, "append_scenarios", 0) or 0) > 0
    )


def _benchmark_cache_dir(args: argparse.Namespace, out_dir: Path) -> Path:
    raw = getattr(args, "cache_dir", None)
    return Path(raw) if raw else out_dir / "benchmark_cache"


def _safe_cache_name(name: str) -> str:
    clean = str(name).strip().lower().replace("gpu_", "").replace("_rk4", "")
    clean = clean.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return clean or "unknown"


def _truth_cache_name(args: argparse.Namespace) -> str:
    return f"{str(args.truth).lower()}_{str(getattr(args, 'truth_integrator', 'DOP853')).lower()}"


def _trajectory_cache_path(
    cache_dir: Path,
    model_type: str,
    model_name: str,
    scenario_id: int,
    args: Optional[argparse.Namespace] = None,
) -> Path:
    if model_type == "truth":
        group = cache_dir / "truth" / _safe_cache_name(model_name)
    else:
        group = cache_dir / "models" / _safe_cache_name(model_name)
    return group / f"scenario_{int(scenario_id):06d}.npz"


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    weight_file = _find_st_lrps_weight_file(getattr(args, "st_lrps_model_dir", None))
    return {
        "cache_schema_version": BENCHMARK_CACHE_SCHEMA_VERSION,
        "truth": str(args.truth).lower(),
        "truth_integrator": str(getattr(args, "truth_integrator", "DOP853")),
        "duration_days": float(args.duration_days),
        "dt_out": float(args.dt_out),
        "rk4_dt_s": (
            None if getattr(args, "rk4_dt_s", None) is None
            else float(args.rk4_dt_s)
        ),
        "gpu_rk4_dt_s_list": _parse_float_list_csv(getattr(args, "gpu_rk4_dt_s_list", None)),
        "st_lrps_rk4_dt": float(getattr(args, "st_lrps_rk4_dt", 30.0)),
        "gpu_integrator": str(getattr(args, "gpu_integrator", "medium")),
        "torch_dtype": str(getattr(args, "torch_dtype", "float64")),
        "batch_frame_mode": str(getattr(args, "batch_frame_mode", "match_dynamics_engine")),
        "st_lrps_model_dir": getattr(args, "st_lrps_model_dir", None),
        "st_lrps_weight_file": weight_file,
        "st_lrps_weight_sha256": _file_sha256(weight_file),
    }


def _write_cache_manifest(
    args: argparse.Namespace,
    cache_dir: Path,
    scenarios: List[Scenario],
    selected_models: Optional[List[str]] = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_schema_version": BENCHMARK_CACHE_SCHEMA_VERSION,
        "metadata": _cache_metadata(args),
        "scenario_count": len(scenarios),
        "scenario_ids": [int(s.scenario_id) for s in scenarios],
        "selected_models": selected_models or [],
        "updated_utc_s": time.time(),
    }
    path = cache_dir / "cache_manifest.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=4), encoding="utf-8")
    os.replace(tmp, path)


def _validate_cache_compatibility(args: argparse.Namespace, cache_dir: Path) -> None:
    path = cache_dir / "cache_manifest.json"
    if not path.exists():
        return
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Existing benchmark cache manifest is unreadable: {exc}") from exc
    old_meta = old.get("metadata", {})
    new_meta = _cache_metadata(args)
    fields = [
        "cache_schema_version", "truth", "truth_integrator", "duration_days", "dt_out",
        "rk4_dt_s", "st_lrps_rk4_dt", "gpu_integrator", "torch_dtype",
        "batch_frame_mode",
    ]
    for field in fields:
        old_value = old_meta.get(field)
        new_value = new_meta.get(field)
        if not _manifest_values_equal(old_value, new_value):
            raise ValueError(
                "Existing benchmark cache uses "
                f"{field}={_manifest_value_text(old_value)} but current request uses "
                f"{_manifest_value_text(new_value)}. Refusing to reuse cached trajectories."
            )
    if "st_lrps" in str(getattr(args, "models", "")) or "st_lrps" in str(getattr(args, "gpu_models", "")):
        for field in ("st_lrps_model_dir", "st_lrps_weight_file", "st_lrps_weight_sha256"):
            old_value = old_meta.get(field)
            new_value = new_meta.get(field)
            if old_value and new_value and old_value != new_value:
                raise ValueError(
                    "Existing benchmark cache uses "
                    f"{field}={_manifest_value_text(old_value)} but current request uses "
                    f"{_manifest_value_text(new_value)}. Refusing to reuse ST-LRPS trajectories."
                )


def _save_cached_trajectory(
    cache_dir: Path,
    scenario: Scenario,
    model_name: str,
    model_type: str,
    t: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    *,
    runtime_s: float = float("nan"),
    integrator: str = "",
    rk4_dt_s: Optional[float] = None,
    dtype: str = "",
    device: str = "",
    backend: str = "",
    truth_model: str = "",
) -> Path:
    path = _trajectory_cache_path(cache_dir, model_type, model_name, scenario.scenario_id, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {
        "cache_schema_version": np.asarray(BENCHMARK_CACHE_SCHEMA_VERSION, dtype=np.int64),
        "scenario_id": np.asarray(int(scenario.scenario_id), dtype=np.int64),
        "model_name": np.asarray(str(model_name)),
        "model_type": np.asarray(str(model_type)),
        "integrator": np.asarray(str(integrator)),
        "t": np.asarray(t, dtype=np.float64),
        "state": np.asarray(y, dtype=np.float64),
        "position": np.asarray(y, dtype=np.float64)[:, :3],
        "velocity": np.asarray(y, dtype=np.float64)[:, 3:],
        "duration_days": np.asarray(float(args.duration_days), dtype=np.float64),
        "dt_out": np.asarray(float(args.dt_out), dtype=np.float64),
        "rk4_dt_s": np.asarray(float("nan") if rk4_dt_s is None else float(rk4_dt_s), dtype=np.float64),
        "dtype": np.asarray(str(dtype)),
        "device": np.asarray(str(device)),
        "backend": np.asarray(str(backend)),
        "truth_model": np.asarray(str(truth_model)),
        "runtime_s": np.asarray(float(runtime_s), dtype=np.float64),
        "hp_km": np.asarray(float(scenario.hp_km), dtype=np.float64),
        "ha_km": np.asarray(float(scenario.ha_km), dtype=np.float64),
        "a_km": np.asarray(float(scenario.a_km), dtype=np.float64),
        "e": np.asarray(float(scenario.e), dtype=np.float64),
        "inc_deg": np.asarray(float(scenario.inc_deg), dtype=np.float64),
        "raan_deg": np.asarray(float(scenario.raan_deg), dtype=np.float64),
        "argp_deg": np.asarray(float(scenario.argp_deg), dtype=np.float64),
        "ta_deg": np.asarray(float(scenario.ta_deg), dtype=np.float64),
        "frame_mode": np.asarray(str(getattr(args, "batch_frame_mode", ""))),
        "gpu_integrator": np.asarray(str(getattr(args, "gpu_integrator", ""))),
    }
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    print(f"[cache] Saved model={model_name} scenario={scenario.scenario_id:06d} file={path}",
          flush=True)
    return path


def _load_cached_trajectory(path: Path) -> Optional[CachedTrajectory]:
    if path.suffix != ".npz" or not path.exists() or path.name.endswith(".tmp"):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["cache_schema_version"]).item())
            if version != BENCHMARK_CACHE_SCHEMA_VERSION:
                return None
            t = np.asarray(data["t"], dtype=np.float64)
            y = np.asarray(data["state"], dtype=np.float64)
            if t.ndim != 1 or y.ndim != 2 or y.shape[1] != 6 or y.shape[0] != t.shape[0]:
                return None
            if not np.isfinite(t).all() or not np.isfinite(y).all():
                return None
            metadata = {k: data[k].tolist() for k in data.files if k not in {"t", "state", "position", "velocity"}}
            runtime = float(np.asarray(data["runtime_s"]).item()) if "runtime_s" in data.files else float("nan")
            return CachedTrajectory(t=t, y=y, runtime_s=runtime, metadata=metadata)
    except Exception:
        return None


def _cached_truth_path(cache_dir: Path, args: argparse.Namespace, scenario_id: int) -> Path:
    return _trajectory_cache_path(cache_dir, "truth", _truth_cache_name(args), scenario_id, args)


def _cached_model_path(cache_dir: Path, model_name: str, scenario_id: int) -> Path:
    return _trajectory_cache_path(cache_dir, "comparison_model", model_name, scenario_id)


def _truth_cache_completion(
    cache_dir: Path,
    args: argparse.Namespace,
    scenarios: List[Scenario],
) -> Tuple[int, List[Scenario]]:
    complete = 0
    missing: List[Scenario] = []
    for scenario in scenarios:
        path = _cached_truth_path(cache_dir, args, scenario.scenario_id)
        if _load_cached_trajectory(path) is not None:
            complete += 1
        else:
            missing.append(scenario)
    return complete, missing


def _model_cache_completion(
    cache_dir: Path,
    model_name: str,
    scenarios: List[Scenario],
) -> Tuple[int, List[Scenario]]:
    complete = 0
    missing: List[Scenario] = []
    for scenario in scenarios:
        path = _cached_model_path(cache_dir, model_name, scenario.scenario_id)
        if _load_cached_trajectory(path) is not None:
            complete += 1
        else:
            missing.append(scenario)
    return complete, missing


# String-valued metric columns that must not be coerced to float on reload.
_METRIC_STRING_KEYS = {"model", "reference", "status", "backend", "device", "failure_reason"}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a metrics CSV into a list of string-valued dict rows (empty if absent)."""
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _coerce_numeric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce CSV string values to float where possible for aggregation.

    Non-numeric / blank values become NaN (so finite-guards skip them); known
    string columns are preserved verbatim.
    """
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if key in _METRIC_STRING_KEYS:
            out[key] = value
            continue
        if value in (None, "", "None", "nan", "NaN"):
            out[key] = float("nan")
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def _find_st_lrps_weight_file(model_dir: Optional[str]) -> Optional[str]:
    """Return the checkpoint path used by the ST-LRPS runtime, if available."""

    if not model_dir:
        return None
    try:
        return str(find_checkpoint_for_st_lrps_run(model_dir))
    except Exception:
        return None


def _write_run_metadata(
    args: argparse.Namespace,
    out_dir: Path,
    scenarios: Optional[List[Scenario]] = None,
) -> None:
    """Persist reproducibility metadata for the validation run."""

    weight_file = _find_st_lrps_weight_file(getattr(args, "st_lrps_model_dir", None))
    scenario_count = len(scenarios) if scenarios is not None else _scenario_count_for_args(args)
    meta = {
        "models": [m.strip().lower() for m in str(args.models).split(",") if m.strip()],
        "truth": str(args.truth).lower(),
        "random_scenarios": int(args.random_scenarios),
        "scenario_count": int(scenario_count),
        "scenario_seed": int(args.scenario_seed),
        "scenario_mode": str(args.scenario_mode),
        "sampling_method": str(getattr(args, "sampling_method", "random")),
        "inclination_sampling": str(getattr(args, "inclination_sampling", "uniform_deg")),
        "altitude_min_km": float(args.altitude_min_km),
        "altitude_max_km": float(args.altitude_max_km),
        "ecc_min": float(args.ecc_min),
        "ecc_max": float(args.ecc_max),
        "inc_min_deg": float(args.inc_min_deg),
        "inc_max_deg": float(args.inc_max_deg),
        "raan_min_deg": float(args.raan_min_deg),
        "raan_max_deg": float(args.raan_max_deg),
        "argp_min_deg": float(args.argp_min_deg),
        "argp_max_deg": float(args.argp_max_deg),
        "ta_min_deg": float(args.ta_min_deg),
        "ta_max_deg": float(args.ta_max_deg),
        "duration_days": float(args.duration_days),
        "dt_out_s": float(args.dt_out),
        "integrator": str(args.integrator),
        "workers": int(getattr(args, "workers", 1)),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_step_s": float(args.max_step),
        "st_lrps_model_dir": getattr(args, "st_lrps_model_dir", None),
        "st_lrps_weight_file": weight_file,
        "st_lrps_mode": str(args.st_lrps_mode),
        "batch_rk4": bool(args.batch_rk4),
        "batch_rk4_reference": str(args.batch_rk4_reference),
        "rk4_dt_s": float(args.rk4_dt_s if args.rk4_dt_s is not None else args.st_lrps_rk4_dt),
        "gpu_rk4_dt_s_list": _parse_float_list_csv(getattr(args, "gpu_rk4_dt_s_list", None)),
        "gpu_fallback": str(args.gpu_fallback),
        "torch_dtype": str(args.torch_dtype),
        "force_batch_size": int(args.force_batch_size),
        "cache_trajectories": bool(getattr(args, "cache_trajectories", False)),
        "reuse_cache": bool(getattr(args, "reuse_cache", False)),
        "cache_dir": getattr(args, "cache_dir", None),
        "append_scenarios": int(getattr(args, "append_scenarios", 0) or 0),
        "rebuild_metrics": bool(getattr(args, "rebuild_metrics", False)),
        "strict_complete": bool(getattr(args, "strict_complete", False)),
        "allow_lhs_append": bool(getattr(args, "allow_lhs_append", False)),
    }
    meta["sampling"] = _sampling_metadata(args, scenario_count)
    p = out_dir / "run_metadata.json"
    _ensure_dir(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, default=str)


def _truth_cache_metadata(args: argparse.Namespace, scenarios: List[Scenario]) -> Dict[str, Any]:
    return {
        "truth": str(args.truth).lower(),
        "random_scenarios": int(len(scenarios)),
        "scenario_seed": int(args.scenario_seed),
        "scenario_mode": str(args.scenario_mode),
        "sampling_method": str(getattr(args, "sampling_method", "random")),
        "inclination_sampling": str(getattr(args, "inclination_sampling", "uniform_deg")),
        "altitude_min_km": float(args.altitude_min_km),
        "altitude_max_km": float(args.altitude_max_km),
        "ecc_min": float(args.ecc_min),
        "ecc_max": float(args.ecc_max),
        "inc_min_deg": float(args.inc_min_deg),
        "inc_max_deg": float(args.inc_max_deg),
        "raan_min_deg": float(args.raan_min_deg),
        "raan_max_deg": float(args.raan_max_deg),
        "argp_min_deg": float(args.argp_min_deg),
        "argp_max_deg": float(args.argp_max_deg),
        "ta_min_deg": float(args.ta_min_deg),
        "ta_max_deg": float(args.ta_max_deg),
        "duration_days": float(args.duration_days),
        "dt_out_s": float(args.dt_out),
        "integrator": str(args.integrator),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_step_s": float(args.max_step),
        "scenario_ids": [int(s.scenario_id) for s in scenarios],
    }


def _truth_cache_available(cache_dir: Path, args: argparse.Namespace, scenarios: List[Scenario]) -> bool:
    """Cheap predicate: would ``_load_truth_cache`` produce a valid hit?

    Checks file presence + metadata equality without loading the (large) NPZ.
    Used only to decide the overall-progress weighting (truth weight collapses
    when truth is served from cache).
    """
    meta_path = cache_dir / "truth_metadata.json"
    npz_path = cache_dir / "sh200_dop853_trajectories.npz"
    if not meta_path.exists() or not npz_path.exists():
        return False
    try:
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return old_meta == _truth_cache_metadata(args, scenarios)
    except Exception:
        return False


def _load_truth_cache(cache_dir: Path, args: argparse.Namespace, scenarios: List[Scenario]) -> Optional[TruthTrajectorySet]:
    meta_path = cache_dir / "truth_metadata.json"
    npz_path = cache_dir / "sh200_dop853_trajectories.npz"
    if not meta_path.exists() or not npz_path.exists():
        return None
    try:
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta != _truth_cache_metadata(args, scenarios):
            return None
        data = np.load(npz_path)
        t_common = data["t"]
        y_all = data["y"]
        runtime = data["runtime"]
        t_by = {s.scenario_id: np.asarray(t_common, dtype=np.float64) for s in scenarios}
        y_by = {s.scenario_id: np.asarray(y_all[i], dtype=np.float64) for i, s in enumerate(scenarios)}
        rt_by = {s.scenario_id: float(runtime[i]) for i, s in enumerate(scenarios)}
        print(f"[truth] Reused cache: {npz_path}", flush=True)
        return TruthTrajectorySet("sh200_dop853", t_by, y_by, rt_by)
    except Exception as exc:
        print(f"[truth] Cache ignored: {exc}", flush=True)
        return None


def _save_truth_cache(cache_dir: Path, args: argparse.Namespace, scenarios: List[Scenario], truth: TruthTrajectorySet) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = truth.t_by_scenario[scenarios[0].scenario_id]
    y_all = np.stack([truth.y_by_scenario[s.scenario_id] for s in scenarios], axis=0)
    runtime = np.asarray([truth.runtime_by_scenario[s.scenario_id] for s in scenarios], dtype=np.float64)
    np.savez_compressed(cache_dir / "sh200_dop853_trajectories.npz", t=t0, y=y_all, runtime=runtime)
    (cache_dir / "truth_metadata.json").write_text(
        json.dumps(_truth_cache_metadata(args, scenarios), indent=4),
        encoding="utf-8",
    )


def build_truth_trajectory_set(
    args: argparse.Namespace,
    scenarios: List[Scenario],
    cfg_base: SimConfig,
    ephem: Any,
    model_cache: GravityModelCache,
    truth_dir: Path,
    on_progress: Optional[Any] = None,
) -> TruthTrajectorySet:
    """Generate or load SH200 DOP853 truth trajectories for all scenarios.

    ``on_progress`` (optional) is invoked as ``cb(completed, total, elapsed_s,
    eta_s)`` after each scenario finishes; it is logging-only.
    """

    def _report(completed: int, total: int, elapsed_s: float) -> None:
        if on_progress is None:
            return
        rate = completed / max(elapsed_s, 1e-9)
        eta = (total - completed) / max(rate, 1e-9)
        try:
            on_progress(int(completed), int(total), float(elapsed_s), float(eta))
        except Exception:
            pass

    if args.reuse_truth_cache:
        cached = _load_truth_cache(truth_dir, args, scenarios)
        if cached is not None:
            return cached

    cache_enabled = _cache_requested(args) or bool(getattr(args, "cache_truth", False))
    cache_dir = _benchmark_cache_dir(args, Path(args.output_dir))
    t_by: Dict[int, np.ndarray] = {}
    y_by: Dict[int, np.ndarray] = {}
    rt_by: Dict[int, float] = {}
    truth_model = str(args.truth).lower()
    truth_integrator = str(getattr(args, "truth_integrator", "DOP853"))
    truth_cfg = _cfg_with_integrator(cfg_base, truth_integrator)
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    pending_scenarios = list(scenarios)
    if cache_enabled:
        complete, pending_scenarios = _truth_cache_completion(cache_dir, args, scenarios)
        print(f"[cache] Truth cache {_truth_cache_name(args)}: {complete}/{len(scenarios)} complete.",
              flush=True)
        for scenario in scenarios:
            cached = _load_cached_trajectory(_cached_truth_path(cache_dir, args, scenario.scenario_id))
            if cached is None:
                continue
            t_by[scenario.scenario_id] = cached.t
            y_by[scenario.scenario_id] = cached.y
            rt_by[scenario.scenario_id] = cached.runtime_s
    print(f"[truth] Building {truth_model.upper()} {truth_integrator} reference "
          f"for {len(pending_scenarios)} missing of {len(scenarios)} scenarios.", flush=True)
    t_truth_start = time.perf_counter()
    if workers > 1 and len(pending_scenarios) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        worker_count = min(workers, len(pending_scenarios))
        print(f"[truth] CPU parallel truth generation: {worker_count} workers "
              f"(integrator={truth_integrator}).", flush=True)
        payloads = [(scenario, truth_model) for scenario in pending_scenarios]
        completed = 0
        scenario_by_id = {s.scenario_id: s for s in pending_scenarios}
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_parallel_worker_init,
            initargs=(args, cfg_base),
        ) as executor:
            futures = {executor.submit(_parallel_worker_truth, p): p[0] for p in payloads}
            for future in as_completed(futures):
                scenario = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    if args.fail_fast:
                        raise RuntimeError(
                            f"truth propagation failed for scenario {scenario.scenario_id}: {exc}"
                        ) from exc
                    print(f"[truth] WARNING: scenario {scenario.scenario_id} failed: {exc}",
                          flush=True)
                    continue
                if result.get("truth_failed"):
                    if args.fail_fast:
                        raise RuntimeError(f"truth propagation failed for scenario {scenario.scenario_id}")
                    print(f"[truth] WARNING: scenario {scenario.scenario_id} failed; "
                          "omitted from metrics.", flush=True)
                    continue
                sid = int(result["scenario_id"])
                t_by[sid] = np.asarray(result["t"], dtype=np.float64)
                y_by[sid] = np.asarray(result["y"], dtype=np.float64)
                rt_by[sid] = float(result["truth_rt"])
                if cache_enabled and not result.get("saved_to_cache"):
                    _save_cached_trajectory(
                        cache_dir, scenario_by_id[sid], _truth_cache_name(args), "truth",
                        t_by[sid], y_by[sid], args,
                        runtime_s=rt_by[sid], integrator=truth_integrator,
                        dtype="float64", device="cpu", truth_model=truth_model,
                    )
                elapsed = time.perf_counter() - t_truth_start
                rate = completed / max(elapsed, 1e-9)
                remaining = (len(pending_scenarios) - completed) / max(rate, 1e-9)
                mm, ss = divmod(int(remaining), 60)
                hh, mm = divmod(mm, 60)
                print(f"[truth] Scenario {completed:03d}/{len(pending_scenarios)} done "
                      f"| id={sid} | runtime={rt_by[sid]:.2f}s "
                      f"| ETA {hh:02d}:{mm:02d}:{ss:02d} "
                      f"| elapsed {elapsed/60.0:.1f} min", flush=True)
                _report(completed, len(pending_scenarios), elapsed)
    else:
        for idx, scenario in enumerate(pending_scenarios, 1):
            print(f"\n[truth] Scenario {idx:03d}/{len(pending_scenarios)} | id={scenario.scenario_id} "
                  f"| hp={scenario.hp_km:.0f} km  ha={scenario.ha_km:.0f} km  "
                  f"i={scenario.inc_deg:.1f} deg", flush=True)
            res, runtime = propagate_for_scenario(
                truth_model, scenario.initial_state, args, truth_cfg, ephem, model_cache
            )
            if res is None:
                if args.fail_fast:
                    raise RuntimeError(f"truth propagation failed for scenario {scenario.scenario_id}")
                print(f"[truth] WARNING: scenario {scenario.scenario_id} failed; omitted from metrics.",
                      flush=True)
                continue
            t_by[scenario.scenario_id] = np.asarray(res.t, dtype=np.float64)
            y_by[scenario.scenario_id] = np.asarray(res.y, dtype=np.float64)
            rt_by[scenario.scenario_id] = float(runtime)
            if cache_enabled:
                _save_cached_trajectory(
                    cache_dir, scenario, _truth_cache_name(args), "truth",
                    t_by[scenario.scenario_id], y_by[scenario.scenario_id], args,
                    runtime_s=float(runtime), integrator=truth_integrator,
                    dtype="float64", device="cpu", truth_model=truth_model,
                )
            elapsed = time.perf_counter() - t_truth_start
            rate = idx / max(elapsed, 1e-9)
            remaining = (len(pending_scenarios) - idx) / max(rate, 1e-9)
            mm, ss = divmod(int(remaining), 60)
            hh, mm = divmod(mm, 60)
            print(f"[truth] Scenario {idx:03d}/{len(pending_scenarios)} done in {runtime:.2f}s "
                  f"| ETA {hh:02d}:{mm:02d}:{ss:02d} "
                  f"| elapsed {elapsed/60.0:.1f} min", flush=True)
            _report(idx, len(pending_scenarios), elapsed)

    truth = TruthTrajectorySet(f"{truth_model}_{truth_integrator.lower()}", t_by, y_by, rt_by)
    if args.cache_truth and len(t_by) == len(scenarios):
        _save_truth_cache(truth_dir, args, scenarios, truth)
    return truth


# =============================================================================
# PDF report — professional template
# =============================================================================
# A small, dependency-free (matplotlib-only) report toolkit that gives the
# generated PDFs a consistent, publication-grade look: a title cover page, a
# navy header band + accent rule, a footer with page numbers and timestamp,
# cleanly styled tables, and captioned figure pages. The numeric content is
# unchanged — only the presentation is upgraded.

_REPORT_THEME = {
    "navy": "#16314F",
    "navy_soft": "#1F4068",
    "accent": "#2A9D8F",
    "ink": "#1A1F29",
    "muted": "#5A6675",
    "rule": "#C9D2DE",
    "row_alt": "#EEF2F7",
    "highlight": "#E3F2EE",
    "page": "#FFFFFF",
}
_REPORT_PAGE_SIZE = (8.5, 11.0)  # US Letter, portrait


class _ReportPager:
    """Builds a multi-page PDF with a consistent header/footer and styled pages."""

    def __init__(self, pdf: PdfPages, title: str, subtitle: str) -> None:
        self.pdf = pdf
        self.title = title
        self.subtitle = subtitle
        self.page_no = 0
        self.generated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    # -- low-level page scaffolding ------------------------------------
    def _blank(self):
        fig = plt.figure(figsize=_REPORT_PAGE_SIZE)
        fig.patch.set_facecolor(_REPORT_THEME["page"])
        return fig

    def _chrome(self, fig, heading: Optional[str]) -> Any:
        """Draw header band + footer; return a content axes (0..1)."""
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        self.page_no += 1
        t = _REPORT_THEME
        # Header band
        fig.add_artist(Rectangle((0, 0.945), 1, 0.055, transform=fig.transFigure,
                                 facecolor=t["navy"], edgecolor="none", zorder=0))
        fig.add_artist(Rectangle((0, 0.941), 1, 0.004, transform=fig.transFigure,
                                 facecolor=t["accent"], edgecolor="none", zorder=0))
        fig.text(0.06, 0.973, self.title, color="white", fontsize=13,
                 fontweight="bold", va="center")
        fig.text(0.06, 0.954, self.subtitle, color="#A9C0D6", fontsize=8.5, va="center")
        # Footer
        fig.add_artist(Line2D([0.06, 0.94], [0.052, 0.052], color=t["rule"],
                              lw=0.8, transform=fig.transFigure))
        fig.text(0.06, 0.034, "ST-LRPS · Lunar Gravity Model Validation",
                 color=t["muted"], fontsize=8, va="center")
        fig.text(0.50, 0.034, self.generated, color=t["muted"], fontsize=8,
                 ha="center", va="center")
        fig.text(0.94, 0.034, f"Page {self.page_no}", color=t["muted"], fontsize=8,
                 ha="right", va="center")
        ax = fig.add_axes([0.06, 0.075, 0.88, 0.85])
        ax.axis("off")
        if heading:
            ax.text(0.0, 1.0, heading, transform=ax.transAxes, fontsize=15,
                    fontweight="bold", color=t["navy"], va="top")
        return ax

    def _save(self, fig) -> None:
        self.pdf.savefig(fig, facecolor=_REPORT_THEME["page"])
        plt.close(fig)

    # -- public page builders ------------------------------------------
    def cover(self, meta: List[Tuple[str, str]], note: str) -> None:
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        t = _REPORT_THEME
        self.page_no += 1
        fig = self._blank()
        # Full-bleed navy banner
        fig.add_artist(Rectangle((0, 0.62), 1, 0.38, transform=fig.transFigure,
                                 facecolor=t["navy"], edgecolor="none", zorder=0))
        fig.add_artist(Rectangle((0, 0.612), 1, 0.008, transform=fig.transFigure,
                                 facecolor=t["accent"], edgecolor="none", zorder=0))
        fig.text(0.08, 0.86, self.title, color="white", fontsize=26, fontweight="bold", va="center")
        fig.text(0.08, 0.80, self.subtitle, color="#A9C0D6", fontsize=13, va="center")
        fig.text(0.08, 0.665, self.generated, color="#7FA8C9", fontsize=10, va="center")
        # Metadata table area
        ax = fig.add_axes([0.08, 0.16, 0.84, 0.40])
        ax.axis("off")
        rows = [[k, v] for k, v in meta]
        if rows:
            tbl = ax.table(cellText=rows, colLabels=["Parameter", "Value"],
                           cellLoc="left", loc="upper left", bbox=[0, 0, 1, 1])
            _style_table(tbl, n_body=len(rows), first_col_left=True)
        # Disclaimer note
        fig.add_artist(Line2D([0.08, 0.92], [0.12, 0.12], color=t["rule"], lw=0.8,
                              transform=fig.transFigure))
        fig.text(0.08, 0.085, note, color=t["muted"], fontsize=9, va="center", wrap=True)
        fig.text(0.08, 0.045, "Generated by st_lrps.evaluation.compare_gravity_models",
                 color=t["muted"], fontsize=8, va="center")
        self._save(fig)

    def table_page(self, heading: str, col_labels: List[str], rows: List[List[str]],
                   *, highlight_row: Optional[int] = None, intro: Optional[str] = None) -> None:
        fig = self._blank()
        ax = self._chrome(fig, heading)  # content axes, axis off, coords 0..1
        top = 0.90
        if intro:
            ax.text(0.0, top, intro, transform=ax.transAxes, fontsize=9.5,
                    color=_REPORT_THEME["muted"], va="top", wrap=True)
            top -= 0.06
        # Bound the table to a sensible region under the heading (axes-relative).
        n = max(1, len(rows))
        tbl_h = min(top - 0.02, 0.05 * (n + 1))
        sub = ax.inset_axes([0.0, max(0.0, top - tbl_h), 1.0, tbl_h])
        sub.axis("off")
        tbl = sub.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                        loc="upper center", bbox=[0, 0, 1, 1])
        _style_table(tbl, n_body=n, first_col_left=True, highlight_row=highlight_row)
        self._save(fig)

    def figure_page(self, heading: str, image_path: Path, caption: str = "") -> bool:
        if not Path(image_path).exists():
            return False
        fig = self._blank()
        self._chrome(fig, heading)
        img = plt.imread(str(image_path))
        h = int(img.shape[0]) or 1
        w = int(img.shape[1]) or 1
        img_aspect = w / h
        page_w, page_h = _REPORT_PAGE_SIZE
        # Size the image box so the figure fills the page width (preserving its
        # aspect) and sits just under the heading — no tiny plot floating in a
        # large blank page.
        bw, x0, y_top = 0.88, 0.06, 0.90
        bh = bw * page_w / (img_aspect * page_h)
        max_bh = 0.74
        if bh > max_bh:
            bh = max_bh
            bw = min(0.88, bh * img_aspect * page_h / page_w)
            x0 = (1.0 - bw) / 2.0
        y0 = y_top - bh
        img_ax = fig.add_axes([x0, y0, bw, bh])
        img_ax.imshow(img)
        img_ax.axis("off")
        if caption:
            fig.text(0.06, max(0.07, y0 - 0.022), caption, color=_REPORT_THEME["muted"],
                     fontsize=9, va="top", wrap=True)
        self._save(fig)
        return True

    def text_page(self, heading: str, paragraphs: List[str]) -> None:
        fig = self._blank()
        ax = self._chrome(fig, heading)
        y = 0.92
        for para in paragraphs:
            ax.text(0.0, y, para, transform=ax.transAxes, fontsize=10,
                    color=_REPORT_THEME["ink"], va="top", wrap=True)
            y -= 0.035 + 0.02 * para.count("\n")
        self._save(fig)


def _style_table(tbl, *, n_body: int, first_col_left: bool = False,
                 highlight_row: Optional[int] = None) -> None:
    """Apply the professional table style to a matplotlib table in place."""
    t = _REPORT_THEME
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(t["rule"])
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(t["navy"])
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.5)
        else:
            if highlight_row is not None and (r - 1) == highlight_row:
                cell.set_facecolor(t["highlight"])
                cell.set_text_props(color=t["ink"], fontweight="bold")
            else:
                cell.set_facecolor("#FFFFFF" if (r % 2 == 1) else t["row_alt"])
                cell.set_text_props(color=t["ink"])
            cell.set_height(cell.get_height() * 1.25)
        if first_col_left and c == 0:
            cell.get_text().set_horizontalalignment("left")
            cell.PAD = 0.04


def write_report_pdf(
    args: argparse.Namespace,
    scenarios: List[Scenario],
    agg: Dict[str, Dict],
    rankings: List[Dict],
    worst_cases: List[Dict],
    plots_dir: Path,
    out_dir: Path,
) -> None:
    pdf_path = out_dir / "gravity_random_validation_report.pdf"
    _ensure_dir(pdf_path)
    truth_integ = str(getattr(args, "truth_integrator", args.integrator))
    with PdfPages(pdf_path) as pdf:
        pager = _ReportPager(
            pdf,
            title="Lunar Gravity Model Validation",
            subtitle="Orbit-level comparison vs a high-degree reference",
        )
        pager.cover(
            meta=[
                ("Scenarios", str(args.random_scenarios)),
                ("Sampling method", str(getattr(args, "sampling_method", "random"))),
                ("Scenario seed", str(args.scenario_seed)),
                ("Scenario mode", str(args.scenario_mode)),
                ("Inclination sampling", str(getattr(args, "inclination_sampling", "uniform_deg"))),
                ("Altitude range", f"{args.altitude_min_km:g} - {args.altitude_max_km:g} km"),
                ("Duration", f"{args.duration_days:g} days"),
                ("Truth model", f"{args.truth.upper()} ({truth_integ})"),
                ("Compared models", str(args.models)),
                ("Integrator", str(args.integrator)),
                ("CPU workers", str(getattr(args, "workers", 1))),
            ],
            note=("Reference note: the truth model is a high-accuracy numerical "
                  "reference, not physical ground truth. Reported errors are "
                  "relative to that reference."),
        )

        if rankings:
            best_idx = 0
            for i, r in enumerate(rankings):
                if r.get("rank_median_rms") in (1, "1"):
                    best_idx = i
                    break
            rows = [[
                r.get("model", "").upper(),
                str(r.get("rank_median_rms", "")),
                f"{r.get('median_rms_pos_err_km', 0):.4f}",
                f"{r.get('p95_rms_pos_err_km', 0):.4f}",
                f"{r.get('max_pos_err_km__mean', 0):.4f}",
                f"{r.get('runtime_s__mean', 0):.2f}",
            ] for r in rankings]
            pager.table_page(
                "Model Accuracy Ranking",
                ["Model", "Rank", "Median RMS [km]", "P95 RMS [km]",
                 "Mean Max [km]", "Mean Runtime [s]"],
                rows,
                highlight_row=best_idx,
                intro="Ranked by median RMS position error across all scenarios "
                      "(lower is better).",
            )

        figure_specs = [
            ("aggregate_boxplot_rms_error.png", "RMS position-error distribution across scenarios."),
            ("aggregate_p95_error_bar.png", "Per-model 95th-percentile RMS position error."),
            ("runtime_vs_accuracy.png", "Runtime vs accuracy tradeoff."),
            ("selected_position_error.png", "Selected scenario: position error vs time."),
            ("selected_ric_error.png", "Selected scenario: radial/in-track/cross-track error."),
            ("selected_orbit_3d.png", "Selected scenario: 3-D trajectory overlay."),
        ]
        for png_name, caption in figure_specs:
            for search_dir in (plots_dir, out_dir):
                if pager.figure_page("Figure", search_dir / png_name, caption):
                    break

    print(f"  [report] PDF saved: {pdf_path}", flush=True)


def write_gpu_batch_report_pdf(
    args: argparse.Namespace,
    aggregate_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    equivalent: Dict[str, Any],
    selected: Dict[str, Any],
    plots_dir: Path,
    reports_dir: Path,
) -> None:
    """Write the GPU batch validation report PDF (professional template)."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / "gpu_batch_validation_report.pdf"
    rk4_dt_values = _gpu_rk4_dt_values(args)
    rk4_dt_text = ", ".join(f"{v:g}" for v in rk4_dt_values)
    truth_integ = str(getattr(args, "truth_integrator", "DOP853"))
    gpu_integ = str(getattr(args, "gpu_integrator", "medium"))
    selection_source = str(selected.get("_selection_source", "ST-LRPS"))

    with PdfPages(pdf_path) as pdf:
        pager = _ReportPager(
            pdf,
            title="GPU Batch Lunar Gravity Validation",
            subtitle="Fixed-step GPU propagation vs an adaptive reference",
        )
        pager.cover(
            meta=[
                ("Scenarios", str(args.random_scenarios)),
                ("Sampling method", str(getattr(args, "sampling_method", "random"))),
                ("Scenario seed", str(args.scenario_seed)),
                ("Scenario mode", str(getattr(args, "scenario_mode", "near_circular_altitude"))),
                ("Inclination sampling", str(getattr(args, "inclination_sampling", "uniform_deg"))),
                ("Altitude range", f"{args.altitude_min_km:.0f} - {args.altitude_max_km:.0f} km"),
                ("Duration", f"{args.duration_days:g} days"),
                ("GPU integrator", f"{gpu_integ} (fixed step {rk4_dt_text} s)"),
                ("Truth workers", str(getattr(args, "workers", 1))),
                ("Output cadence", f"{args.dt_out:g} s"),
                ("Precision", str(args.torch_dtype)),
                ("Truth", f"{args.truth.upper()} {truth_integ}"),
                ("GPU models", str(args.gpu_models)),
                ("Frame mode", str(args.batch_frame_mode)),
            ],
            note=("Reference note: the truth trajectories are a high-accuracy "
                  "adaptive numerical reference, not physical ground truth. GPU "
                  "fixed-step models carry both model and integration error."),
        )

        # Executive summary as a styled key/value table.
        med_eq = equivalent.get("median_rms", {}) if isinstance(equivalent, dict) else {}
        p95_eq = equivalent.get("p95_rms", {}) if isinstance(equivalent, dict) else {}
        fastest = min(runtime_rows, key=lambda r: r.get("total_runtime_s", np.inf)) if runtime_rows else {}
        most_acc = min(aggregate_rows, key=lambda r: r.get("median_rms_pos_err_km", np.inf)) if aggregate_rows else {}
        pager.table_page(
            "Executive Summary",
            ["Metric", "Value"],
            [
                ["Most accurate GPU model", str(most_acc.get("model", "n/a"))],
                ["Fastest GPU model", str(fastest.get("model", "n/a"))],
                ["ST-LRPS closest by median RMS", str(med_eq.get("closest_model", "n/a"))],
                ["ST-LRPS median-equivalent status",
                 str(med_eq.get("equivalent_degree_status", med_eq.get("status", "n/a")))],
                ["ST-LRPS closest by P95 RMS", str(p95_eq.get("closest_model", "n/a"))],
                [f"Best {selection_source} scenario", str(selected.get("best", {}).get("scenario_id", "n/a"))],
                [f"Representative {selection_source} scenario",
                 str(selected.get("representative", {}).get("scenario_id", "n/a"))],
                [f"Worst {selection_source} scenario", str(selected.get("worst", {}).get("scenario_id", "n/a"))],
            ],
        )

        if aggregate_rows:
            acc_rows = [[
                str(r.get("model", "")),
                f"{r.get('median_rms_pos_err_km', float('nan')):.4f}",
                f"{r.get('p95_rms_pos_err_km', float('nan')):.4f}",
                f"{r.get('max_rms_pos_err_km', float('nan')):.4f}",
                f"{r.get('median_along_rms_km', float('nan')):.4f}",
            ] for r in aggregate_rows]
            pager.table_page(
                "Accuracy Ranking",
                ["Model", "Median RMS [km]", "P95 RMS [km]", "Max RMS [km]", "Median Along [km]"],
                acc_rows,
                highlight_row=0,
                intro="Sorted best-to-worst by median RMS position error.",
            )

        if runtime_rows:
            run_rows = [[
                str(r.get("model", "")),
                f"{r.get('total_runtime_s', float('nan')):.3f}",
                f"{r.get('runtime_per_scenario_s', float('nan')):.5f}",
                f"{r.get('trajectory_steps_per_second', float('nan')):.1f}",
                f"{r.get('speedup_vs_truth_total', float('nan')):.2f}",
            ] for r in runtime_rows]
            pager.table_page(
                "Runtime",
                ["Model", "Total [s]", "Per scenario [s]", "Steps/s", "Speedup vs truth"],
                run_rows,
                intro="Wall-clock runtime and throughput for the GPU fixed-step propagation.",
            )

        # Shared context appended to each caption (N, duration, truth, unit note).
        ctx = (f"N = {args.random_scenarios} scenarios over {args.duration_days:g} day(s); "
               f"errors are relative to the {args.truth.upper()} {truth_integ} reference. "
               f"Axes auto-select display units (km/m/cm); CSV metrics remain in km.")
        small_n_note = (" For small N, distribution panels are diagnostic rather than "
                        "statistical." if int(args.random_scenarios) < 8 else "")
        figure_specs = [
            ("gpu_runtime_vs_accuracy.png",
             "Runtime–accuracy tradeoff across GPU models. Lower-left is better "
             "(faster and more accurate); the dashed line marks the Pareto front. " + ctx),
            ("gpu_accuracy_ranking_bar.png",
             "Per-model accuracy ranking (lollipops = median RMS, open diamonds = P95 RMS), "
             "sorted best-to-worst; ST-LRPS is highlighted. Lower is better. " + ctx),
            ("stlrps_equivalent_sh_degree.png",
             "ST-LRPS equivalent SH-degree estimate by interpolating median RMS error "
             "across the spherical-harmonic baselines. " + ctx),
            ("gpu_rms_error_distribution_boxplot.png",
             "Distribution of per-scenario RMS position error per model." + small_n_note + " " + ctx),
            ("ensemble_mean_position_error_vs_time.png",
             "Ensemble median position error vs time (shaded band = 25–75% across scenarios). " + ctx),
            ("ensemble_ric_rms_vs_time.png",
             "Ensemble radial/along-track/cross-track RMS error vs time. " + ctx),
            ("selected_representative_position_error_all_models.png",
             "Representative scenario: position error vs time. " + ctx),
            ("selected_representative_ric_error_all_models.png",
             "Representative scenario: radial/along-track/cross-track error vs time. " + ctx),
            ("selected_worst_position_error_all_models.png",
             "Worst-case scenario: position error vs time. " + ctx),
            ("selected_worst_ric_error_all_models.png",
             "Worst-case scenario: radial/along-track/cross-track error vs time. " + ctx),
        ]
        for png_name, caption in figure_specs:
            pager.figure_page("Figure", plots_dir / png_name, caption)

        notes = [
            "- The truth trajectories are a high-accuracy adaptive numerical reference, "
            "not physical ground truth.",
            "- GPU fixed-step SH models include both spherical-harmonic truncation error "
            "and integration error.",
            "- ST-LRPS includes surrogate-model error plus integration error.",
            f"- Frame mode: {args.batch_frame_mode}.",
        ]
        if args.batch_frame_mode == "inertial_fixed_legacy":
            notes.append("- Legacy frame mode is approximate and should not be used for final claims.")
        pager.text_page("Notes & Caveats", notes)

    print(f"  [report] GPU batch PDF saved: {pdf_path}", flush=True)


# =============================================================================
# Force evaluation mode
# =============================================================================

def evaluate_forces(
    models_to_test: List[str],
    truth_model: str,
    args: argparse.Namespace,
    cfg: SimConfig,
    ephem: Any,
    out_dir: Path,
) -> None:
    print(f"\n--- Force Sample Evaluation vs {truth_model.upper()} ---", flush=True)

    model_cache = GravityModelCache(cfg, args)

    a_m = args.altitude_km * 1_000.0 + R_MOON
    y0  = create_state_from_keplerian(
        semi_major_axis=a_m, eccentricity=args.ecc,
        inclination=math.radians(args.inc_deg), raan=math.radians(args.raan_deg),
        argp=math.radians(args.argp_deg), true_anomaly=math.radians(args.ta_deg),
        mu=MU_MOON,
    ).y

    grav_truth = model_cache.get(truth_model)
    dyn_truth  = DynamicsEngine(cfg.spacecraft, cfg.flags,
                                gravity_model=grav_truth, ephem_manager=ephem,
                                allow_identity_rotation=True)
    try:
        res_truth = propagate(dyn_truth, y0, cfg.propagator, time_cfg=cfg.time)
        if res_truth is None or (res_truth.ode is not None and not res_truth.ode.success):
            raise RuntimeError("truth propagation failed")
    except Exception as exc:
        print(f"CRITICAL: truth propagation failed: {exc}", file=sys.stderr)
        return

    t_ref   = res_truth.t
    y_ref   = res_truth.y
    N       = len(t_ref)
    rhs_truth = dyn_truth.build_rhs()
    a_truth = np.array([rhs_truth(t_ref[i], y_ref[i])[3:6] for i in range(N)])

    summary = []
    for m in models_to_test:
        if m == truth_model:
            continue
        grav = model_cache.get(m)
        _synchronize_model_device_if_cuda(grav)
        t0   = time.perf_counter()

        if m == "st_lrps":
            # Use batched path for surrogate
            r_body_fixed = y_ref[:, :3]   # inertial ≈ body-fixed approximation
            a_test = evaluate_st_lrps_forces_batched(
                grav, r_body_fixed, batch_size=args.force_batch_size
            )
        else:
            dyn  = DynamicsEngine(cfg.spacecraft, cfg.flags,
                                  gravity_model=grav, ephem_manager=ephem,
                                  allow_identity_rotation=True)
            rhs  = dyn.build_rhs()
            a_test = np.array([rhs(t_ref[i], y_ref[i])[3:6] for i in range(N)])

        _synchronize_model_device_if_cuda(grav)
        eval_s = time.perf_counter() - t0

        da = a_test - a_truth
        da_norm_mGal = np.linalg.norm(da, axis=1) * 1e5
        ric_da = decompose_vector_ric(da, y_ref[:, :3], y_ref[:, 3:6])
        ric_mGal = ric_da * 1e5

        a_truth_norm = np.linalg.norm(a_truth, axis=1)
        cos_ang = np.clip(
            np.einsum("ij,ij->i", a_truth, a_test)
            / (a_truth_norm * np.linalg.norm(a_test, axis=1) + 1e-30),
            -1.0, 1.0,
        )
        ang_deg = np.degrees(np.arccos(cos_ang))

        samples_per_s = N / max(eval_s, 1e-9)
        print(f"  {m.upper()}: eval={eval_s:.2f}s  {samples_per_s:,.0f} pts/s  "
              f"RMS={float(np.sqrt(np.mean(da_norm_mGal**2))):.3f} mGal", flush=True)

        summary.append({
            "model": m,
            "eval_time_s": round(eval_s, 3),
            "samples_per_second": round(samples_per_s, 0),
            "batch_size": args.force_batch_size if m == "st_lrps" else 1,
            "device": str(grav.device) if hasattr(grav, "device") else "cpu",
            "accel_err_rms_mGal":   float(np.sqrt(np.mean(da_norm_mGal ** 2))),
            "accel_err_max_mGal":   float(np.max(da_norm_mGal)),
            "accel_rel_mean":       float(np.mean(da_norm_mGal / (a_truth_norm * 1e5 + 1e-30))),
            "accel_rel_p95":        float(np.percentile(da_norm_mGal / (a_truth_norm * 1e5 + 1e-30), 95)),
            "radial_accel_rms_mGal":  float(np.sqrt(np.mean(ric_mGal[:, 0] ** 2))),
            "along_accel_rms_mGal":   float(np.sqrt(np.mean(ric_mGal[:, 1] ** 2))),
            "cross_accel_rms_mGal":   float(np.sqrt(np.mean(ric_mGal[:, 2] ** 2))),
            "angular_error_deg_mean": float(np.mean(ang_deg)),
            "angular_error_deg_p95":  float(np.percentile(ang_deg, 95)),
        })

    summary.sort(key=lambda x: x["accel_err_rms_mGal"])
    _ensure_dir(out_dir / "force_sample_summary.json")
    with open(out_dir / "force_sample_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    _write_csv(summary, out_dir / "force_sample_summary.csv")
    _write_csv(
        [
            {
                "model": row["model"],
                "eval_time_s": row["eval_time_s"],
                "samples_per_second": row["samples_per_second"],
                "device": row["device"],
                "batch_size": row["batch_size"],
            }
            for row in summary
        ],
        out_dir / "force_runtime_summary.csv",
    )

    print("\n--- Force Evaluation Ranking ---")
    for i, s in enumerate(summary, 1):
        print(f"  {i}. {s['model'].upper():<12} "
              f"| RMS: {s['accel_err_rms_mGal']:.3f} mGal "
              f"| Max: {s['accel_err_max_mGal']:.3f} mGal "
              f"| {s['samples_per_second']:,.0f} pts/s "
              f"| device={s['device']}")
    print(f"\nForce evaluation complete -> {out_dir}", flush=True)


# =============================================================================
# Single orbit mode (original)
# =============================================================================

def run_single_orbit_mode(args: argparse.Namespace, cfg: SimConfig, ephem: Any) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    truth  = args.truth.strip().lower()
    if truth not in models:
        models.append(truth)

    a_m = args.altitude_km * 1_000.0 + R_MOON
    y0  = create_state_from_keplerian(
        semi_major_axis=a_m, eccentricity=args.ecc,
        inclination=math.radians(args.inc_deg), raan=math.radians(args.raan_deg),
        argp=math.radians(args.argp_deg), true_anomaly=math.radians(args.ta_deg),
        mu=MU_MOON,
    ).y

    model_cache = GravityModelCache(cfg, args)
    results: Dict[str, Any] = {}
    runtimes: Dict[str, float] = {}

    for m in models:
        print(f"\n--- Running {m.upper()} ---", flush=True)
        res, rt = propagate_for_scenario(m, y0, args, cfg, ephem, model_cache)
        if res is not None:
            results[m] = res
            runtimes[m] = rt
            print(f"  done: {rt:.2f}s", flush=True)
        else:
            print("  FAILED", flush=True)

    if truth not in results:
        print(f"CRITICAL: truth model {truth} failed.", file=sys.stderr)
        sys.exit(1)

    truth_res = results[truth]
    summary = []
    for m, res in results.items():
        if m == truth:
            continue
        sc = Scenario(0, 0, 0, a_m / 1_000.0, args.ecc, args.inc_deg,
                      args.raan_deg, args.argp_deg, args.ta_deg, initial_state=y0)
        met = compute_trajectory_metrics(m, sc, truth_res, res, runtimes[m], runtimes[truth])
        summary.append(met)

    summary.sort(key=lambda x: x.get("rms_pos_err_km") or 0)

    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    _write_csv(summary, out_dir / "comparison_summary.csv")

    print(f"\n--- Ranking (RMS pos error vs {truth.upper()}) ---")
    for i, s in enumerate(summary, 1):
        if s.get("rms_pos_err_km") is not None:
            print(f"  {i}. {s['model'].upper():<10} "
                  f"| RMS: {s['rms_pos_err_km']:.6f} km "
                  f"| Runtime: {s['runtime_s']:.2f}s")

    plt.style.use("dark_background")
    t_ref = truth_res.t / 86400.0
    r_ref = truth_res.y[:, :3]

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in [s["model"] for s in summary if s.get("rms_pos_err_km") is not None]:
        y_m = interpolate_state_to_times(results[m].t, results[m].y, truth_res.t)
        err = np.linalg.norm(y_m[:, :3] - r_ref, axis=1) / 1_000.0
        ax.semilogy(t_ref, np.maximum(err, 1e-9), color=_color(m), label=m.upper())
    ax.set_title(f"Position Error vs {truth.upper()}")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("Position Error [km]")
    ax.grid(True, alpha=0.25, which="both"); ax.legend()
    fig.savefig(out_dir / "position_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDone -> {out_dir}", flush=True)


# =============================================================================
# Ranking table printer
# =============================================================================

def _print_ranking_table(
    rankings: List[Dict],
    agg: Dict[str, Dict],
    args: argparse.Namespace,
) -> None:
    n_sc = max((r.get("n_scenarios", 0) for r in rankings), default=0)
    sep  = "=" * 95

    print(f"\n{sep}")
    print(f"  GRAVITY MODEL ACCURACY RANKING  |  truth={args.truth.upper()}  "
          f"|  {n_sc} scenarios  |  {args.duration_days:.2f} day(s)  "
          f"|  alt {args.altitude_min_km:.0f}-{args.altitude_max_km:.0f} km")
    print(sep)
    hdr = (f"  {'#':<3} {'Model':<12} "
           f"{'Median RMS':>12} {'P95 RMS':>10} {'Max err':>10} "
           f"{'Mean RMS':>10} {'Std RMS':>10} "
           f"{'Median Vel':>11} {'Runtime':>9}")
    print(hdr)
    print(f"  {'':3} {'':12} "
          f"{'[km]':>12} {'[km]':>10} {'[km]':>10} "
          f"{'[km]':>10} {'[km]':>10} "
          f"{'[m/s]':>11} {'[s/sc]':>9}")
    print("-" * 95)

    for r in rankings:
        m    = r["model"]
        s    = agg.get(m, {})
        line = (
            f"  {r.get('rank_median_rms', '-'):<3} {m.upper():<12} "
            f"{r.get('median_rms_pos_err_km', 0):>12.4f} "
            f"{r.get('p95_rms_pos_err_km',   0):>10.4f} "
            f"{s.get('max_pos_err_km__max',  0):>10.4f} "
            f"{s.get('rms_pos_err_km__mean', 0):>10.4f} "
            f"{s.get('rms_pos_err_km__std',  0):>10.4f} "
            f"{s.get('rms_vel_err_ms__median', 0):>11.4f} "
            f"{s.get('runtime_s__mean',       0):>9.2f}"
        )
        print(line)

    print(sep)
    print("\n  Worst-case per model:")
    for r in rankings:
        m   = r["model"]
        s   = agg.get(m, {})
        wc  = s.get("max_pos_err_km__max", 0) or 0
        p99 = s.get("rms_pos_err_km__p99", 0) or 0
        print(f"    {m.upper():<12}  worst max err: {wc:.4f} km  |  p99 RMS: {p99:.4f} km")

    print(f"\n  All errors are vs {args.truth.upper()} (not physical truth).")
    print(sep)


def _print_batch_summary(
    batch_result: Dict[str, Any],
    total_rows: List[Dict],
    model_rows: List[Dict],
    integr_rows: List[Dict],
    args: argparse.Namespace,
) -> None:
    sep = "=" * 80
    ok_total  = [r for r in total_rows  if r.get("status") == "ok"]
    ok_model  = [r for r in model_rows  if r.get("status") == "ok"]
    ok_integr = [r for r in integr_rows if r.get("status") == "ok"]

    def _rms_stats(rows: List[Dict]) -> str:
        if not rows:
            return "N/A"
        vals = [r["rms_pos_err_km"] for r in rows if np.isfinite(r.get("rms_pos_err_km", np.nan))]
        if not vals:
            return "N/A"
        return (f"median={np.median(vals):.4f} km  "
                f"p95={np.percentile(vals, 95):.4f} km  "
                f"max={np.max(vals):.4f} km")

    print(f"\n{sep}")
    print("  BATCH RK4 SUMMARY")
    print(sep)
    print(f"  Device:          {batch_result.get('device', '?')}")
    print(f"  Mode:            {batch_result.get('mode', '?')}")
    print(f"  Scenarios:       {batch_result.get('n_scenarios', '?')}")
    print(f"  RK4 dt:          {batch_result.get('dt_s', '?')} s")
    print(f"  Total runtime:   {batch_result.get('runtime_s', 0):.2f} s")
    n_sc = batch_result.get("n_scenarios", 1)
    n_steps = batch_result.get("n_steps", 1)
    rt = batch_result.get("runtime_s", 1)
    print(f"  Throughput:      {n_sc * n_steps / max(rt, 1e-9):,.0f} traj-steps/s")
    print(f"  Per-scenario:    {rt / max(n_sc, 1):.2f} s")
    print(sep)
    print(f"  ST-LRPS RK4 vs SH200 DOP853 (total error):")
    print(f"    {_rms_stats(ok_total)}")
    if ok_model:
        print(f"  ST-LRPS RK4 vs SH200 RK4 (model error):")
        print(f"    {_rms_stats(ok_model)}")
    if ok_integr:
        print(f"  SH200 RK4 vs SH200 DOP853 (integrator error):")
        print(f"    {_rms_stats(ok_integr)}")
    print(sep)


def _print_final_validation_summary(
    args: argparse.Namespace,
    agg: Dict[str, Dict],
    batch_result: Optional[Dict[str, Any]],
    total_rows: List[Dict],
    model_rows: List[Dict],
    integr_rows: List[Dict],
) -> None:
    """Print a compact end-of-run summary focused on runtime vs accuracy."""

    sep = "=" * 92
    print(f"\n{sep}")
    print("GRAVITY VALIDATION SUMMARY")
    print(sep)
    print(f"Truth reference: {args.truth.upper()} DOP853")
    print(f"Scenarios:       {args.random_scenarios}")
    print(f"Sampling:        {getattr(args, 'sampling_method', 'random')}")
    print(f"Inclination:     {getattr(args, 'inclination_sampling', 'uniform_deg')}")
    print(f"CPU workers:     {getattr(args, 'workers', 1)}")
    print(f"Altitude range:  {args.altitude_min_km:.0f}-{args.altitude_max_km:.0f} km")
    print(f"Duration:        {args.duration_days:g} day(s)")

    if agg:
        print("\nCPU DOP853 MODE:")
        print(f"{'Model':<14} {'median RMS km':>14} {'p95 RMS km':>12} "
              f"{'max km':>12} {'runtime/sc s':>13}")
        for model, stats in sorted(agg.items()):
            print(f"{model.upper():<14} "
                  f"{stats.get('rms_pos_err_km__median', np.nan):>14.4f} "
                  f"{stats.get('rms_pos_err_km__p95', np.nan):>12.4f} "
                  f"{stats.get('max_pos_err_km__max', np.nan):>12.4f} "
                  f"{stats.get('runtime_s__mean', np.nan):>13.3f}")

    if batch_result is not None:
        ok_total = [r for r in total_rows if r.get("status") == "ok"]
        ok_model = [r for r in model_rows if r.get("status") == "ok"]
        ok_integr = [r for r in integr_rows if r.get("status") == "ok"]

        def _med_p95(rows: List[Dict]) -> Tuple[float, float]:
            vals = np.array([r["rms_pos_err_km"] for r in rows], dtype=np.float64)
            if vals.size == 0:
                return np.nan, np.nan
            return float(np.median(vals)), float(np.percentile(vals, 95))

        med_total, p95_total = _med_p95(ok_total)
        print("\nBATCH GPU RK4 MODE:")
        print(f"{'Model':<14} {'median RMS km':>14} {'p95 RMS km':>12} "
              f"{'total runtime s':>16} {'scenario/s':>12}")
        runtime = float(batch_result.get("runtime_s", np.nan))
        n_sc = float(batch_result.get("n_scenarios", 0) or 0)
        print(f"{'ST-LRPS':<14} {med_total:>14.4f} {p95_total:>12.4f} "
              f"{runtime:>16.3f} {n_sc / max(runtime, 1e-9):>12.3f}")
        print(f"Device: {batch_result.get('device')} | batch size: "
              f"{args.batch_size or batch_result.get('n_scenarios')} | "
              f"RK4 dt: {batch_result.get('dt_s')} s | output dt: {args.dt_out} s")

        print("\nERROR DECOMPOSITION:")
        if ok_integr:
            med, p95 = _med_p95(ok_integr)
            print(f"SH200 RK4 vs SH200 DOP853: median RMS={med:.4f} km, p95={p95:.4f} km")
        else:
            print("SH200 RK4 vs SH200 DOP853: not run")
        if ok_model:
            med, p95 = _med_p95(ok_model)
            print(f"ST-LRPS RK4 vs SH200 RK4: median RMS={med:.4f} km, p95={p95:.4f} km")
        else:
            print("ST-LRPS RK4 vs SH200 RK4: not run")
        print(f"ST-LRPS RK4 vs SH200 DOP853: median RMS={med_total:.4f} km, p95={p95_total:.4f} km")

    print(sep)


def _print_gpu_batch_summary(
    args: argparse.Namespace,
    aggregate_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    equivalent: Dict[str, Any],
    selected: Dict[str, Any],
) -> None:
    sep = "=" * 96
    print(f"\n{sep}")
    print("GPU BATCH VALIDATION SUMMARY")
    print(sep)
    print(f"Truth:    {args.truth.upper()} DOP853")
    print(f"Scenarios:{args.random_scenarios}")
    print(f"Sampling: {getattr(args, 'sampling_method', 'random')}")
    print(f"Inc mode: {getattr(args, 'inclination_sampling', 'uniform_deg')}")
    print(f"Truth workers: {getattr(args, 'workers', 1)}")
    print(f"Duration: {args.duration_days:g} days")
    rk4_values = _gpu_rk4_dt_values(args)
    print(f"RK4 dt:   {', '.join(f'{v:g}' for v in rk4_values)} s")
    print(f"Dtype:    {args.torch_dtype}")
    print(f"Frame:    {args.batch_frame_mode}")

    print("\nAccuracy ranking:")
    print(f"{'Model':<22} {'Median RMS km':>14} {'P95 RMS km':>12} "
          f"{'Max RMS km':>12} {'Median Along km':>17}")
    for r in aggregate_rows:
        print(f"{r['model']:<22} {r.get('median_rms_pos_err_km', np.nan):>14.4f} "
              f"{r.get('p95_rms_pos_err_km', np.nan):>12.4f} "
              f"{r.get('max_rms_pos_err_km', np.nan):>12.4f} "
              f"{r.get('median_along_rms_km', np.nan):>17.4f}")

    print("\nRuntime ranking:")
    print(f"{'Model':<22} {'Runtime s':>10} {'Runtime/sc s':>14} "
          f"{'Traj-steps/s':>14} {'Speedup truth':>14}")
    for r in runtime_rows:
        print(f"{r['model']:<22} {r.get('total_runtime_s', np.nan):>10.3f} "
              f"{r.get('runtime_per_scenario_s', np.nan):>14.5f} "
              f"{r.get('trajectory_steps_per_second', np.nan):>14.1f} "
              f"{r.get('speedup_vs_truth_total', np.nan):>14.2f}")

    med_eq = equivalent.get("median_rms", {}) if isinstance(equivalent, dict) else {}
    st_row = next((r for r in aggregate_rows if r.get("model") == "GPU_ST_LRPS_RK4"), None)
    st_runtime = next((r for r in runtime_rows if r.get("model") == "GPU_ST_LRPS_RK4"), None)
    closest_model = med_eq.get("closest_model", "n/a")
    closest_runtime = next((r for r in runtime_rows if r.get("model") == closest_model), None)
    speedup_vs_closest = np.nan
    if st_runtime and closest_runtime:
        speedup_vs_closest = closest_runtime["total_runtime_s"] / max(st_runtime["total_runtime_s"], 1e-9)
    print("\nST-LRPS interpretation:")
    if st_row:
        print(f"- ST-LRPS median RMS = {st_row.get('median_rms_pos_err_km', np.nan):.4f} km")
    print(f"- Closest classical SH model by median RMS = {closest_model}")
    print(f"- Equivalent-degree status = {med_eq.get('equivalent_degree_status', med_eq.get('status', 'n/a'))}")
    print(f"- ST-LRPS speedup vs closest model = {speedup_vs_closest:.2f}x")
    print(f"- Best scenario id = {selected.get('best', {}).get('scenario_id', 'n/a')}")
    print(f"- Representative scenario id = {selected.get('representative', {}).get('scenario_id', 'n/a')}")
    print(f"- Worst scenario id = {selected.get('worst', {}).get('scenario_id', 'n/a')}")
    print(sep)


def run_gpu_batch_compare_mode(args: argparse.Namespace, cfg_base: SimConfig, ephem: Any) -> None:
    """New validation workflow: SH200 DOP853 truth vs GPU RK4 SH/ST-LRPS."""

    out_dir = Path(args.output_dir)
    truth_dir = out_dir / "truth"
    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"
    for d in (truth_dir, metrics_dir, plots_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    scenarios = prepare_scenarios(args, out_dir)
    _write_run_metadata(args, out_dir, scenarios)
    progress.emit_progress(
        "scenario", current=len(scenarios), total=len(scenarios),
        percent=100.0, message="Scenarios ready",
    )

    gpu_models = _parse_model_list_csv(args.gpu_models)
    if args.truth.lower() not in {"sh200"}:
        print(f"[gpu-batch] WARNING: requested truth={args.truth}; expected sh200 for this workflow.",
              flush=True)

    if "st_lrps" in gpu_models and not args.st_lrps_model_dir:
        auto_dir = _auto_find_st_lrps_dir()
        if auto_dir:
            args.st_lrps_model_dir = auto_dir
            print(f"[auto] ST-LRPS model dir: {auto_dir}", flush=True)
            weight = _find_st_lrps_weight_file(auto_dir)
            if weight:
                print(f"[auto] ST-LRPS weight file: {weight}", flush=True)
        elif args.require_st_lrps:
            raise FileNotFoundError("ST-LRPS requested but no valid model dir was found.")
        else:
            print("[gpu-batch] WARNING: ST-LRPS model missing; removing st_lrps from --gpu-models.",
                  flush=True)
            gpu_models = [m for m in gpu_models if m != "st_lrps"]
    gpu_tasks = _build_gpu_batch_tasks(gpu_models, args)
    gpu_cache_names = [task.cache_name for task in gpu_tasks]

    cache_enabled = _cache_requested(args) or bool(getattr(args, "cache_trajectories", False))
    cache_dir = _benchmark_cache_dir(args, out_dir)
    if cache_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[cache] Enabled trajectory cache: {cache_dir}", flush=True)
        _validate_cache_compatibility(args, cache_dir)
        _write_cache_manifest(args, cache_dir, scenarios, gpu_cache_names)

    if getattr(args, "rebuild_metrics", False):
        aggregate_rows, runtime_rows, equivalent, selected = rebuild_gpu_batch_metrics_from_cache(
            args, scenarios, cache_dir, gpu_cache_names, metrics_dir, plots_dir, reports_dir
        )
        _print_gpu_batch_summary(args, aggregate_rows, runtime_rows, equivalent, selected)
        return

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for --gpu-batch-compare.") from exc

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif args.gpu_fallback == "cpu":
        print("[gpu-batch] CUDA unavailable; explicit --gpu-fallback cpu selected.", flush=True)
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA unavailable for --gpu-batch-compare. Use --gpu-fallback cpu to continue.")
    dtype = _torch_dtype_from_name(args.torch_dtype)

    print(f"\n[gpu-batch] models={[task.display_name for task in gpu_tasks]} "
          f"device={device} dtype={args.torch_dtype} "
          f"frame={args.batch_frame_mode} gpu_integrator={getattr(args, 'gpu_integrator', 'medium')}",
          flush=True)

    # Accumulation / resume: skip scenarios already computed for every requested
    # model so a later run (same seed, larger --random-scenarios) only computes
    # the new orbits and the aggregate covers the cumulative set.
    per_scenario_csv = metrics_dir / "gpu_batch_per_scenario_metrics.csv"
    existing_rows: List[Dict[str, Any]] = []
    completed_ids: set = set()
    if args.resume and per_scenario_csv.exists():
        existing_rows = [_coerce_numeric_row(r) for r in _read_csv_rows(per_scenario_csv)]
        needed_models = {task.display_name for task in gpu_tasks}
        by_id: Dict[int, set] = {}
        for r in existing_rows:
            try:
                sid = int(float(r.get("scenario_id")))
            except (TypeError, ValueError):
                continue
            by_id.setdefault(sid, set()).add(str(r.get("model", "")))
        completed_ids = {sid for sid, mods in by_id.items() if needed_models.issubset(mods)}
        print(f"[gpu-batch] resume: {len(completed_ids)} scenarios already complete for all "
              f"requested models; {len(existing_rows)} stored metric rows loaded.", flush=True)

    if cache_enabled:
        completed_ids = set()
        existing_rows = []

    run_scenarios = [s for s in scenarios if s.scenario_id not in completed_ids]
    if args.resume and not run_scenarios:
        print("[gpu-batch] resume: no new scenarios to compute; re-aggregating stored results.",
              flush=True)

    model_cache = GravityModelCache(cfg_base, args)
    _truth_name = f"{str(args.truth).lower()}_{str(getattr(args, 'truth_integrator', 'DOP853')).lower()}"

    # Weighted overall progress. When truth is served from cache its weight
    # collapses so the GPU comparison phase dominates the bar.
    truth_cached = bool(
        run_scenarios
        and getattr(args, "reuse_truth_cache", False)
        and _truth_cache_available(truth_dir, args, run_scenarios)
    )
    overall_weights = (
        {"gpu": 0.50, "report": 0.10}
        if truth_cached
        else {"truth": 0.40, "gpu": 0.50, "report": 0.10}
    )
    overall = progress.OverallProgress(overall_weights)
    n_gpu_models = max(1, len(gpu_tasks))

    def _on_truth_progress(completed: int, total: int, elapsed_s: float, eta_s: float) -> None:
        pct = 100.0 * completed / max(1, total)
        progress.emit_progress(
            "truth", current=completed, total=total, percent=pct,
            elapsed_s=elapsed_s, eta_s=eta_s, message="SH200 DOP853 truth",
        )
        ov = overall.update("truth", completed / max(1, total))
        progress.emit_progress_total(
            ov, "truth", elapsed_s=overall.elapsed_s(), eta_s=overall.eta_s()
        )

    if run_scenarios:
        truth = build_truth_trajectory_set(
            args, run_scenarios, cfg_base, ephem, model_cache, truth_dir,
            on_progress=_on_truth_progress,
        )
        if not truth.t_by_scenario:
            raise RuntimeError("No truth trajectories were generated.")
    else:
        truth = TruthTrajectorySet(_truth_name, {}, {}, {})

    duration_s = float(args.duration_days) * 86400.0
    results: List[BatchModelResult] = []
    t_gpu_start = time.perf_counter()
    completed_gpu = 0
    for model_idx, task in enumerate(gpu_tasks if run_scenarios else [], 1):
        model_name = task.model_name
        model_scenarios = list(run_scenarios)
        if cache_enabled:
            complete, missing = _model_cache_completion(cache_dir, task.cache_name, scenarios)
            print(f"[cache] Model {task.display_name}: {complete}/{len(scenarios)} complete."
                  + (f" Recomputing {len(missing)} missing." if missing else ""),
                  flush=True)
            model_scenarios = missing
            if not model_scenarios:
                continue

        print(f"\n[gpu-batch] Model {model_idx:02d}/{len(gpu_tasks)} | "
              f"{task.display_name} starting for {len(model_scenarios)} scenario(s) "
              f"(rk4_dt={task.rk4_dt_s:g}s) ...",
              flush=True)
        y0_batch = np.asarray([s.initial_state for s in model_scenarios], dtype=np.float64)

        def _gpu_progress_cb(
            current_step: int, total_steps: int, elapsed_s: float,
            _name: str = task.display_name, _idx: int = model_idx,
        ) -> None:
            stats = progress.compute_step_stats(current_step, total_steps, elapsed_s)
            print(
                f"[gpu-batch][{_name}] step {stats['current_step']}/{stats['total_steps']} "
                f"| {stats['percent']:.1f}% | elapsed {progress.format_duration(elapsed_s)} "
                f"| ETA {progress.format_eta(stats['eta_s'])} "
                f"| {stats['steps_per_s']:.1f} steps/s",
                flush=True,
            )
            progress.emit_progress(
                "gpu_model", model=_name,
                current_step=stats["current_step"], total_steps=stats["total_steps"],
                percent=stats["percent"], elapsed_s=elapsed_s,
                eta_s=stats["eta_s"], steps_per_s=stats["steps_per_s"],
                device=str(device), dtype=str(args.torch_dtype),
                n_scenarios=len(model_scenarios),
            )
            model_frac = stats["current_step"] / max(1, stats["total_steps"])
            gpu_frac = ((_idx - 1) + model_frac) / n_gpu_models
            ov = overall.update("gpu", gpu_frac)
            progress.emit_progress_total(
                ov, "gpu_model", model=_name,
                elapsed_s=overall.elapsed_s(), eta_s=overall.eta_s(),
            )

        try:
            gravity = model_cache.get(model_name)
            result = propagate_gpu_batch_model(
                model_name,
                gravity,
                y0_batch,
                duration_s,
                task.rk4_dt_s,
                float(args.dt_out),
                ephem,
                device=device,
                dtype=dtype,
                dtype_name=args.torch_dtype,
                frame_mode=args.batch_frame_mode,
                gpu_integrator=str(getattr(args, "gpu_integrator", "medium")),
                progress_cb=_gpu_progress_cb,
            )
            result.display_name = task.display_name
            result.model_name = task.cache_name
            completed_gpu += 1
            elapsed_gpu = time.perf_counter() - t_gpu_start
            rate_gpu = completed_gpu / max(elapsed_gpu, 1e-9)
            remaining_gpu = (len(gpu_tasks) - completed_gpu) / max(rate_gpu, 1e-9)
            mm, ss = divmod(int(remaining_gpu), 60)
            hh, mm = divmod(mm, 60)
            print(f"[gpu-batch] Model {model_idx:02d}/{len(gpu_tasks)} done | "
                  f"{result.display_name}: {result.runtime_s:.2f}s "
                  f"backend={result.backend} status={result.status} "
                  f"| ETA {hh:02d}:{mm:02d}:{ss:02d}", flush=True)
            if cache_enabled and result.status == "ok":
                per_scenario_runtime = result.runtime_s / max(1, len(model_scenarios))
                for scenario_idx, scenario in enumerate(model_scenarios):
                    _save_cached_trajectory(
                        cache_dir, scenario, task.cache_name, "comparison_model",
                        result.t, result.y[:, scenario_idx, :], args,
                        runtime_s=per_scenario_runtime,
                        integrator="gpu_rk4",
                        rk4_dt_s=result.rk4_dt_s,
                        dtype=result.dtype,
                        device=result.device,
                        backend=result.backend,
                        truth_model=_truth_name,
                    )
            results.append(result)
        except Exception as exc:
            print(f"[gpu-batch] ERROR {task.display_name}: {exc}", flush=True)
            if args.fail_fast:
                raise
            completed_gpu += 1
            results.append(BatchModelResult(
                model_name=task.cache_name,
                display_name=task.display_name,
                backend="failed",
                device=str(device),
                dtype=args.torch_dtype,
                t=np.array([], dtype=np.float64),
                y=np.empty((0, len(model_scenarios), 6), dtype=np.float64),
                runtime_s=float("nan"),
                n_steps=0,
                n_scenarios=len(model_scenarios),
                rk4_dt_s=task.rk4_dt_s,
                output_dt_s=float(args.dt_out),
                status="failed",
                failure_reason=str(exc),
            ))

    if cache_enabled:
        _write_cache_manifest(args, cache_dir, scenarios, gpu_cache_names)
        aggregate_rows, runtime_rows, equivalent, selected = rebuild_gpu_batch_metrics_from_cache(
            args, scenarios, cache_dir, gpu_cache_names, metrics_dir, plots_dir, reports_dir
        )
        _print_gpu_batch_summary(args, aggregate_rows, runtime_rows, equivalent, selected)
        print(f"\n[gpu-batch] Complete -> {out_dir}", flush=True)
        print("  benchmark_cache/")
        print("  metrics/gpu_batch_per_scenario_metrics.csv")
        print("  metrics/gpu_batch_aggregate_metrics.csv")
        print("  metrics/gpu_batch_runtime_metrics.csv")
        print("  metrics/gpu_batch_model_ranking.csv")
        print("  metrics/gpu_batch_summary.json")
        print("  plots/")
        print("  reports/gpu_batch_validation_report.pdf")
        return

    progress.emit_progress(
        "aggregate", current=1, total=1, percent=100.0, message="Writing metrics"
    )
    _ov_agg = overall.update("report", 0.5)
    progress.emit_progress_total(
        _ov_agg, "aggregate", elapsed_s=overall.elapsed_s(), eta_s=overall.eta_s()
    )

    new_rows: List[Dict[str, Any]] = []
    for result in results:
        new_rows.extend(compute_gpu_batch_metrics_for_model(result, truth, run_scenarios, args.duration_days))
    # Cumulative union: previously-stored rows + newly-computed rows.
    all_rows: List[Dict[str, Any]] = list(existing_rows) + new_rows

    aggregate_rows = aggregate_gpu_batch_metrics(all_rows)
    runtime_rows = build_gpu_runtime_metrics(results, truth)
    ranking_rows = build_gpu_model_ranking(aggregate_rows)
    equivalent = estimate_stlrps_equivalent_sh_degree(aggregate_rows)
    selected = select_stlrps_scenarios(all_rows, {s.scenario_id: s for s in scenarios}, args)

    _write_csv(all_rows, per_scenario_csv)
    _write_csv(aggregate_rows, metrics_dir / "gpu_batch_aggregate_metrics.csv")
    _write_csv(runtime_rows, metrics_dir / "gpu_batch_runtime_metrics.csv")
    _write_csv(ranking_rows, metrics_dir / "gpu_batch_model_ranking.csv")
    (metrics_dir / "stlrps_selected_scenarios.json").write_text(
        json.dumps(selected, indent=4, default=str), encoding="utf-8"
    )
    summary = {
        "truth": _truth_name,
        "gpu_models": gpu_models,
        "gpu_model_variants": [task.display_name for task in gpu_tasks],
        "n_scenarios_total": len(scenarios),
        "n_scenarios_new_this_run": len(run_scenarios),
        "accumulated": bool(args.resume),
        "sampling": _sampling_metadata(args, len(scenarios)),
        "truth_workers": int(getattr(args, "workers", 1)),
        "gpu_integrator": str(getattr(args, "gpu_integrator", "medium")),
        "frame_mode": args.batch_frame_mode,
        "uses_lunar_rotation": args.batch_frame_mode == "match_dynamics_engine",
        "matches_dynamics_engine_frame": args.batch_frame_mode == "match_dynamics_engine",
        "truth_total_runtime_s": truth.total_runtime_s,
        "truth_mean_runtime_per_scenario_s": truth.mean_runtime_s,
        "equivalent_sh_degree": equivalent,
        "selected_stlrps_scenarios": selected,
        "aggregate": aggregate_rows,
        "runtime": runtime_rows,
    }
    (metrics_dir / "gpu_batch_summary.json").write_text(
        json.dumps(summary, indent=4, default=str), encoding="utf-8"
    )

    sh200_row = next((r for r in aggregate_rows if r.get("model") == "GPU_SH200_RK4"), None)
    if sh200_row and sh200_row.get("median_rms_pos_err_km", 0.0) > 10.0:
        print("[gpu-batch] WARNING: GPU SH200 RK4 vs SH200 DOP853 error is high. "
              "Check RK4 dt, frame mode, and rotation consistency.", flush=True)

    progress.emit_progress(
        "report", current=1, total=1, percent=100.0,
        message="Generating plots/report",
    )
    _ov_report = overall.update("report", 1.0)
    progress.emit_progress_total(
        _ov_report, "report", elapsed_s=overall.elapsed_s(), eta_s=overall.eta_s()
    )

    plot_gpu_batch_report_figures(
        aggregate_rows, runtime_rows, all_rows, results, truth, run_scenarios,
        selected, equivalent, plots_dir, args
    )
    write_gpu_batch_report_pdf(args, aggregate_rows, runtime_rows, equivalent, selected, plots_dir, reports_dir)
    _print_gpu_batch_summary(args, aggregate_rows, runtime_rows, equivalent, selected)

    print(f"\n[gpu-batch] Complete -> {out_dir}", flush=True)
    print("  metrics/gpu_batch_per_scenario_metrics.csv")
    print("  metrics/gpu_batch_aggregate_metrics.csv")
    print("  metrics/gpu_batch_runtime_metrics.csv")
    print("  metrics/gpu_batch_model_ranking.csv")
    print("  metrics/stlrps_selected_scenarios.json")
    print("  metrics/gpu_batch_summary.json")
    print("  plots/")
    print("  reports/gpu_batch_validation_report.pdf")


# =============================================================================
# CPU parallel scenario workers
# =============================================================================
# Per-process state, populated once by the pool initializer so the heavy
# ephemeris + gravity caches are built a single time per worker rather than
# pickled per task.
_PARALLEL_STATE: Dict[str, Any] = {}


def _parallel_worker_init(args: argparse.Namespace, cfg_base: SimConfig) -> None:
    """ProcessPool initializer: build per-worker ephemeris + gravity caches once."""
    ephem = EphemerisManager.from_time_and_spice(cfg_base.time, cfg_base.spice)
    _PARALLEL_STATE["args"] = args
    _PARALLEL_STATE["cfg_base"] = cfg_base
    _PARALLEL_STATE["truth_cfg"] = _cfg_with_integrator(
        cfg_base, str(getattr(args, "truth_integrator", "DOP853"))
    )
    _PARALLEL_STATE["ephem"] = ephem
    _PARALLEL_STATE["cache"] = GravityModelCache(cfg_base, args)


def _parallel_worker_truth(payload: Tuple[Scenario, str]) -> Dict[str, Any]:
    """Propagate only the adaptive truth trajectory for one scenario."""

    scenario, truth_model = payload
    st = _PARALLEL_STATE
    args = st["args"]
    truth_cfg = st["truth_cfg"]
    ephem = st["ephem"]
    cache = st["cache"]
    truth_res, truth_rt = propagate_for_scenario(
        truth_model, scenario.initial_state, args, truth_cfg, ephem, cache
    )
    if truth_res is None:
        return {
            "scenario_id": scenario.scenario_id,
            "truth_failed": True,
            "truth_rt": None,
        }
    saved = False
    if _cache_requested(args) or bool(getattr(args, "cache_truth", False)):
        try:
            cache_dir = _benchmark_cache_dir(args, Path(args.output_dir))
            _save_cached_trajectory(
                cache_dir, scenario, _truth_cache_name(args), "truth",
                np.asarray(truth_res.t, dtype=np.float64),
                np.asarray(truth_res.y, dtype=np.float64),
                args,
                runtime_s=float(truth_rt),
                integrator=str(getattr(args, "truth_integrator", "DOP853")),
                dtype="float64",
                device="cpu",
                truth_model=truth_model,
            )
            saved = True
        except Exception:
            saved = False
    return {
        "scenario_id": scenario.scenario_id,
        "truth_failed": False,
        "truth_rt": float(truth_rt),
        "t": np.asarray(truth_res.t, dtype=np.float64),
        "y": np.asarray(truth_res.y, dtype=np.float64),
        "saved_to_cache": saved,
    }


def _parallel_worker_scenario(payload: Tuple[Scenario, str, List[str]]) -> Dict[str, Any]:
    """Propagate truth + compared models for one scenario inside a worker.

    Only lightweight metric rows (not full trajectories) are returned so the
    inter-process payload stays small.
    """
    scenario, truth_model, compare_models = payload
    st = _PARALLEL_STATE
    args = st["args"]
    cfg_base = st["cfg_base"]
    truth_cfg = st["truth_cfg"]
    ephem = st["ephem"]
    cache = st["cache"]

    y0 = scenario.initial_state
    truth_res, truth_rt = propagate_for_scenario(truth_model, y0, args, truth_cfg, ephem, cache)
    if truth_res is None:
        return {"scenario_id": scenario.scenario_id, "truth_failed": True, "truth_rt": None, "rows": []}

    rows: List[Dict[str, Any]] = []
    for model in compare_models:
        try:
            res, rt = propagate_for_scenario(model, y0, args, cfg_base, ephem, cache)
        except Exception as exc:  # pragma: no cover - defensive (worker side)
            failed = {f: None for f in _METRICS_FIELDNAMES}
            failed.update({"scenario_id": scenario.scenario_id, "model": model,
                           "status": "exception", "failure_reason": str(exc)})
            rows.append(failed)
            continue
        if res is None:
            failed = {f: None for f in _METRICS_FIELDNAMES}
            failed.update({"scenario_id": scenario.scenario_id, "model": model, "status": "failed"})
            rows.append(failed)
            continue
        rows.append(compute_trajectory_metrics(model, scenario, truth_res, res, rt, truth_rt))

    return {
        "scenario_id": scenario.scenario_id,
        "truth_failed": False,
        "truth_rt": float(truth_rt),
        "rows": rows,
    }


# =============================================================================
# Random scenario validation mode
# =============================================================================

def run_random_scenario_mode(
    args: argparse.Namespace,
    cfg_base: SimConfig,
    ephem: Any,
) -> None:
    out_dir   = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    models_str  = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    truth_model = args.truth.strip().lower()
    if args.include_st_lrps and "st_lrps" not in models_str:
        models_str.append("st_lrps")
    compare_models = [m for m in models_str if m != truth_model]
    dop853_compare_models = [
        m for m in compare_models
        if not (m == "st_lrps" and args.batch_rk4 and args.st_lrps_mode == "gpu_rk4")
    ]
    if "st_lrps" in compare_models and "st_lrps" not in dop853_compare_models:
        print("[harness] ST-LRPS DOP853 scalar propagation skipped because "
              "--batch-rk4 with --st-lrps-mode gpu_rk4 was requested.",
              flush=True)

    scenarios = prepare_scenarios(args, out_dir)

    print(f"\n[harness] {len(scenarios)} scenarios  truth={truth_model.upper()}  "
          f"models={[m.upper() for m in compare_models]}", flush=True)

    _write_run_metadata(args, out_dir, scenarios)
    scenarios_by_id = {s.scenario_id: s for s in scenarios}
    progress.emit_progress(
        "scenario", current=len(scenarios), total=len(scenarios),
        percent=100.0, message="Scenarios ready",
    )
    overall_cpu = progress.OverallProgress({"sweep": 0.90, "report": 0.10})

    def _report_sweep(done: int, total: int, elapsed_s: float, eta_s: float) -> None:
        pct = 100.0 * done / max(1, total)
        progress.emit_progress(
            "sweep", current=done, total=total, percent=pct,
            elapsed_s=elapsed_s, eta_s=eta_s, message="CPU adaptive sweep",
        )
        ov = overall_cpu.update("sweep", done / max(1, total))
        progress.emit_progress_total(
            ov, "sweep", elapsed_s=overall_cpu.elapsed_s(), eta_s=overall_cpu.eta_s()
        )

    cache_enabled = _cache_requested(args) or bool(getattr(args, "cache_trajectories", False))
    cache_dir = _benchmark_cache_dir(args, out_dir)
    model_missing_by_name: Dict[str, List[Scenario]] = {}
    if cache_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[cache] Enabled trajectory cache: {cache_dir}", flush=True)
        _validate_cache_compatibility(args, cache_dir)
        _write_cache_manifest(args, cache_dir, scenarios, compare_models)
        truth_complete, truth_missing = _truth_cache_completion(cache_dir, args, scenarios)
        print(f"[cache] Truth {_truth_cache_name(args)}: {truth_complete}/{len(scenarios)} complete.",
              flush=True)
        for model in dop853_compare_models:
            complete, missing = _model_cache_completion(cache_dir, model, scenarios)
            model_missing_by_name[model] = missing
            print(f"[cache] Model {model}: {complete}/{len(scenarios)} complete.",
                  flush=True)
        if getattr(args, "rebuild_metrics", False):
            print("[cache] Rebuilding metrics from cached trajectories.", flush=True)
            if getattr(args, "strict_complete", False):
                if truth_missing:
                    raise RuntimeError(
                        f"--strict-complete requested but truth cache is missing "
                        f"{len(truth_missing)} scenario(s)."
                    )
                missing_models = {
                    m: len(missing)
                    for m, missing in model_missing_by_name.items()
                    if missing
                }
                if missing_models:
                    raise RuntimeError(
                        "--strict-complete requested but model cache is incomplete: "
                        + ", ".join(f"{m}={n}" for m, n in missing_models.items())
                    )

    # Resume support: load old rows into all_metrics
    metrics_path   = out_dir / "per_scenario_metrics.csv"
    completed_ids: set = set()
    all_metrics: List[Dict] = []

    if args.resume and metrics_path.exists() and not cache_enabled:
        try:
            with open(metrics_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        sid = int(row["scenario_id"])
                        completed_ids.add(sid)
                        # Convert numeric columns
                        converted = dict(row)
                        for k, v in converted.items():
                            if k not in ("model", "status") and v not in (None, "", "None"):
                                try:
                                    converted[k] = float(v)
                                except (ValueError, TypeError):
                                    pass
                        if converted.get("status") == "ok":
                            all_metrics.append(converted)
                    except (KeyError, ValueError):
                        pass
            print(f"[resume] {len(completed_ids)} scenarios complete, "
                  f"{len(all_metrics)} ok metric rows loaded", flush=True)
        except Exception as exc:
            print(f"[resume] WARNING: could not load old metrics: {exc}", flush=True)

    if metrics_path.exists() and (cache_enabled or not args.resume):
        metrics_path.unlink()

    truth_runtimes: List[float] = []
    truth_results_all: Dict[int, Any] = {}  # for batch RK4

    header_written = len(all_metrics) > 0 or (args.resume and metrics_path.exists())
    n_total = len(scenarios)
    t_start = time.perf_counter()
    n_done  = sum(1 for s in scenarios if s.scenario_id in completed_ids)
    model_cache = GravityModelCache(cfg_base, args)

    # Ground-truth integrator may differ from the compared-model integrator.
    truth_integrator = str(getattr(args, "truth_integrator", "DOP853"))
    truth_cfg = _cfg_with_integrator(cfg_base, truth_integrator)

    pending = scenarios if cache_enabled else [s for s in scenarios if s.scenario_id not in completed_ids]
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    # CPU parallelism applies to the per-model adaptive sweep only. batch-RK4
    # needs full truth trajectories in-process, so it stays sequential.
    parallel = (
        workers > 1
        and not bool(args.batch_rk4)
        and bool(dop853_compare_models)
        and not cache_enabled
    )

    if parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"\n[harness] CPU parallel sweep: {len(pending)} scenarios across "
              f"{workers} workers (truth integrator={truth_integrator}).", flush=True)
        payloads = [(s, truth_model, dop853_compare_models) for s in pending]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_parallel_worker_init,
            initargs=(args, cfg_base),
        ) as executor:
            futures = {executor.submit(_parallel_worker_scenario, p): p[0] for p in payloads}
            for future in as_completed(futures):
                scenario = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[harness] worker failed for scenario {scenario.scenario_id}: {exc}",
                          flush=True)
                    if args.fail_fast:
                        raise
                    n_done += 1
                    continue
                if result.get("truth_failed"):
                    print(f"  FAILED: truth model (scenario {result['scenario_id']})", flush=True)
                    if args.fail_fast:
                        sys.exit(1)
                    n_done += 1
                    continue
                if result.get("truth_rt") is not None:
                    truth_runtimes.append(float(result["truth_rt"]))
                for row in result.get("rows", []):
                    if row.get("status") == "ok":
                        all_metrics.append(row)
                    _append_metrics_csv(row, metrics_path, not header_written)
                    header_written = True
                n_done += 1
                elapsed = time.perf_counter() - t_start
                rate = n_done / max(elapsed, 1e-9)
                remaining = (n_total - n_done) / max(rate, 1e-9)
                mm, ss = divmod(int(remaining), 60)
                hh, mm = divmod(mm, 60)
                print(f"  [{n_done:03d}/{n_total}] scenario {result['scenario_id']} done "
                      f"| ETA {hh:02d}:{mm:02d}:{ss:02d}", flush=True)
                _report_sweep(n_done, n_total, elapsed, remaining)
    else:
        if workers > 1:
            reason = "trajectory cache requires per-file checkpointing" if cache_enabled else \
                "batch-RK4 or no adaptive compare models"
            print(f"[harness] --workers>1 ignored ({reason}); running sequentially.",
                  flush=True)
        for sc_i, scenario in enumerate(scenarios):
            if scenario.scenario_id in completed_ids and not cache_enabled:
                continue

            print(f"\nScenario {sc_i+1:03d}/{n_total} | id={scenario.scenario_id} "
                  f"| hp={scenario.hp_km:.0f} km  ha={scenario.ha_km:.0f} km  "
                  f"i={scenario.inc_deg:.1f} deg", flush=True)

            y0 = scenario.initial_state

            # Truth propagation (uses the selected ground-truth integrator).
            truth_res = None
            truth_rt = float("nan")
            if cache_enabled:
                cached_truth = _load_cached_trajectory(
                    _cached_truth_path(cache_dir, args, scenario.scenario_id)
                )
                if cached_truth is not None:
                    truth_res = cached_truth
                    truth_rt = cached_truth.runtime_s
                    print(f"  {truth_model.upper()} {truth_integrator} | cache hit", flush=True)
                elif getattr(args, "rebuild_metrics", False):
                    msg = (
                        f"missing cached truth for scenario {scenario.scenario_id:06d}; "
                        "skipping scenario."
                    )
                    if getattr(args, "strict_complete", False):
                        raise RuntimeError(msg)
                    print(f"  [cache] {msg}", flush=True)
                    n_done += 1
                    continue

            if truth_res is None:
                print(f"  {truth_model.upper()} {truth_integrator} | running ...", flush=True)
                truth_res, truth_rt = propagate_for_scenario(
                    truth_model, y0, args, truth_cfg, ephem, model_cache
                )
            if truth_res is None:
                print("  FAILED: truth model", flush=True)
                if args.fail_fast:
                    sys.exit(1)
                n_done += 1
                continue
            if cache_enabled and not getattr(args, "rebuild_metrics", False):
                _save_cached_trajectory(
                    cache_dir, scenario, _truth_cache_name(args), "truth",
                    truth_res.t, truth_res.y, args,
                    runtime_s=truth_rt,
                    integrator=truth_integrator,
                    dtype="float64",
                    device="cpu",
                    backend="cpu_truth",
                    truth_model=truth_model,
                )

            if np.isfinite(truth_rt):
                print(f"  {truth_model.upper()} | done {truth_rt:.2f}s", flush=True)
            truth_runtimes.append(truth_rt)
            truth_results_all[scenario.scenario_id] = truth_res

            # Compare models
            for model in dop853_compare_models:
                res = None
                rt = float("nan")
                if cache_enabled:
                    cached_model = _load_cached_trajectory(
                        _cached_model_path(cache_dir, model, scenario.scenario_id)
                    )
                    if cached_model is not None:
                        res = cached_model
                        rt = cached_model.runtime_s
                        print(f"  {model.upper()} | cache hit", flush=True)
                    elif getattr(args, "rebuild_metrics", False):
                        msg = (
                            f"missing cached model={model} scenario="
                            f"{scenario.scenario_id:06d}; skipping row."
                        )
                        if getattr(args, "strict_complete", False):
                            raise RuntimeError(msg)
                        print(f"  [cache] {msg}", flush=True)
                        continue

                if res is None:
                    print(f"  {model.upper()} | running ...", end=" ", flush=True)
                    try:
                        res, rt = propagate_for_scenario(
                            model, y0, args, cfg_base, ephem, model_cache
                        )
                    except Exception as exc:
                        print(f"EXCEPTION: {exc}", flush=True)
                        traceback.print_exc()
                        if args.fail_fast:
                            sys.exit(1)
                        failed_row = {f: None for f in _METRICS_FIELDNAMES}
                        failed_row.update({"scenario_id": scenario.scenario_id,
                                           "model": model, "status": "exception"})
                        _append_metrics_csv(failed_row, metrics_path, not header_written)
                        header_written = True
                        continue

                if res is None:
                    print("FAILED", flush=True)
                    if args.fail_fast:
                        sys.exit(1)
                    failed_row = {f: None for f in _METRICS_FIELDNAMES}
                    failed_row.update({"scenario_id": scenario.scenario_id,
                                       "model": model, "status": "failed"})
                    _append_metrics_csv(failed_row, metrics_path, not header_written)
                    header_written = True
                    continue
                if cache_enabled and not getattr(args, "rebuild_metrics", False):
                    _save_cached_trajectory(
                        cache_dir, scenario, model, "comparison_model",
                        res.t, res.y, args,
                        runtime_s=rt,
                        integrator=str(getattr(args, "integrator", "DOP853")),
                        dtype="float64",
                        device="cpu",
                        backend="cpu_adaptive",
                        truth_model=truth_model,
                    )

                metrics = compute_trajectory_metrics(
                    model, scenario, truth_res, res, rt, truth_rt
                )
                all_metrics.append(metrics)
                _append_metrics_csv(metrics, metrics_path, not header_written)
                header_written = True
                print(f"done {rt:.2f}s | RMS pos err: {metrics.get('rms_pos_err_km', 0):.4f} km",
                      flush=True)

            n_done += 1
            elapsed = time.perf_counter() - t_start
            rate    = n_done / max(elapsed, 1e-9)
            remaining = (n_total - n_done) / max(rate, 1e-9)
            mm, ss  = divmod(int(remaining), 60)
            hh, mm  = divmod(mm, 60)
            print(f"  ETA: {hh:02d}:{mm:02d}:{ss:02d} remaining  ({n_done}/{n_total} done)",
                  flush=True)
            _report_sweep(n_done, n_total, elapsed, remaining)

    # Aggregate statistics
    print("\n[harness] Computing aggregate statistics ...", flush=True)
    progress.emit_progress(
        "report", current=1, total=1, percent=100.0,
        message="Aggregating metrics and generating plots",
    )
    _ov_cpu_report = overall_cpu.update("report", 1.0)
    progress.emit_progress_total(
        _ov_cpu_report, "report",
        elapsed_s=overall_cpu.elapsed_s(), eta_s=overall_cpu.eta_s(),
    )
    truth_runtime_mean = float(np.mean(truth_runtimes)) if truth_runtimes else 1.0
    agg      = aggregate_metrics(all_metrics, truth_runtime_mean)
    rankings = build_rankings(agg)

    with open(out_dir / "aggregate_summary.json", "w") as f:
        json.dump(agg, f, indent=4, default=str)
    _write_csv(rankings, out_dir / "ranking_summary.csv")
    agg_rows = [{"model": m, **stats} for m, stats in agg.items()]
    _write_csv(agg_rows, out_dir / "aggregate_summary.csv")
    if cache_enabled:
        cache_metrics_dir = cache_dir / "metrics"
        _write_csv(all_metrics, cache_metrics_dir / "per_model_scenario_metrics.csv")
        _write_csv(agg_rows, cache_metrics_dir / "aggregate_metrics.csv")
        _write_csv(rankings, cache_metrics_dir / "ranking_summary.csv")

    worst_cases = find_worst_cases(all_metrics, scenarios_by_id)
    _write_csv(worst_cases, out_dir / "worst_cases_by_model.csv")

    _print_ranking_table(rankings, agg, args)

    # Aggregate plots
    print("\n[harness] Generating aggregate plots ...", flush=True)
    plot_aggregate_stats(all_metrics, agg, rankings, plots_dir)

    # Selected scenario overlay (median difficulty)
    selected_sc = None
    if args.plot_scenario_id is not None:
        selected_sc = scenarios_by_id.get(args.plot_scenario_id)
    if selected_sc is None:
        selected_sc = select_median_difficulty_scenario(all_metrics, scenarios)

    if selected_sc is not None and dop853_compare_models:
        print(f"\n[harness] Plotting selected scenario {selected_sc.scenario_id} "
              f"(median-difficulty) ...", flush=True)
        traj: Dict[str, Any] = {}
        y0 = selected_sc.initial_state
        for m in [truth_model] + dop853_compare_models:
            if cache_enabled:
                cache_path = (
                    _cached_truth_path(cache_dir, args, selected_sc.scenario_id)
                    if m == truth_model
                    else _cached_model_path(cache_dir, m, selected_sc.scenario_id)
                )
                cached = _load_cached_trajectory(cache_path)
                if cached is not None:
                    traj[m] = cached
                    continue
                if getattr(args, "rebuild_metrics", False):
                    continue
            _m_cfg = truth_cfg if m == truth_model else cfg_base
            res, _ = propagate_for_scenario(m, y0, args, _m_cfg, ephem, model_cache)
            if res is not None:
                traj[m] = res
        npz_path = out_dir / "trajectories_selected_scenario.npz"
        _ensure_dir(npz_path)
        np.savez_compressed(
            npz_path,
            **{f"{m}_t": r.t for m, r in traj.items()},
            **{f"{m}_y": r.y for m, r in traj.items()},
        )
        plot_selected_scenario(selected_sc, truth_model, traj, plots_dir, prefix="selected")
    elif selected_sc is not None:
        print("\n[harness] Skipping DOP853 selected-scenario overlay because no "
              "non-truth DOP853 models were requested.", flush=True)

    # Worst-case global plot
    if all_metrics:
        ok_rows = [m for m in all_metrics if m.get("status") == "ok"
                   and m.get("max_pos_err_km") is not None]
        if ok_rows:
            worst_global = max(ok_rows, key=lambda r: r["max_pos_err_km"])
            worst_sc = scenarios_by_id.get(worst_global["scenario_id"])
            if worst_sc is not None:
                print(f"\n[harness] Plotting worst-case scenario "
                      f"{worst_sc.scenario_id} ({worst_global['model'].upper()}) ...", flush=True)
                traj_w: Dict[str, Any] = {}
                y0w = worst_sc.initial_state
                for m in [truth_model] + dop853_compare_models:
                    if cache_enabled:
                        cache_path = (
                            _cached_truth_path(cache_dir, args, worst_sc.scenario_id)
                            if m == truth_model
                            else _cached_model_path(cache_dir, m, worst_sc.scenario_id)
                        )
                        cached = _load_cached_trajectory(cache_path)
                        if cached is not None:
                            traj_w[m] = cached
                            continue
                        if getattr(args, "rebuild_metrics", False):
                            continue
                    _m_cfg = truth_cfg if m == truth_model else cfg_base
                    res, _ = propagate_for_scenario(m, y0w, args, _m_cfg, ephem, model_cache)
                    if res is not None:
                        traj_w[m] = res
                npz_path_w = out_dir / "trajectories_worst_case.npz"
                _ensure_dir(npz_path_w)
                np.savez_compressed(
                    npz_path_w,
                    **{f"{m}_t": r.t for m, r in traj_w.items()},
                    **{f"{m}_y": r.y for m, r in traj_w.items()},
                )
                plot_selected_scenario(worst_sc, truth_model, traj_w, plots_dir,
                                       prefix="worst_case")

    # =========================================================================
    # Batch RK4 mode
    # =========================================================================
    batch_result     = None
    total_rows:  List[Dict] = []
    model_rows:  List[Dict] = []
    integr_rows: List[Dict] = []

    if args.batch_rk4 and "st_lrps" in compare_models and not getattr(args, "rebuild_metrics", False):
        print("\n[batch-rk4] Starting batched RK4 propagation ...", flush=True)

        rk4_dt = args.rk4_dt_s if args.rk4_dt_s is not None else args.st_lrps_rk4_dt
        duration_s   = args.duration_days * 86400.0
        output_dt_s  = args.dt_out

        missing_truth = [sc for sc in scenarios if sc.scenario_id not in truth_results_all]
        if missing_truth:
            print(f"[batch-rk4] Rebuilding {len(missing_truth)} SH200 DOP853 truth "
                  "trajectories skipped by resume so batch metrics stay complete.",
                  flush=True)
            for idx, sc in enumerate(missing_truth, 1):
                truth_res, truth_rt = propagate_for_scenario(
                    truth_model, sc.initial_state, args, truth_cfg, ephem, model_cache
                )
                if truth_res is not None:
                    truth_results_all[sc.scenario_id] = truth_res
                    truth_runtimes.append(truth_rt)
                elif args.fail_fast:
                    raise RuntimeError(
                        f"Could not rebuild truth trajectory for scenario {sc.scenario_id}."
                    )
                if idx % 10 == 0 or idx == len(missing_truth):
                    print(f"  [batch-rk4] truth rebuild {idx}/{len(missing_truth)}", flush=True)

        # Collect all initial states (including already-completed scenarios)
        all_y0 = np.array([sc.initial_state for sc in scenarios])

        surr_model = model_cache.get("st_lrps")

        print(f"  Scenarios: {len(scenarios)}  rk4_dt={rk4_dt}s  "
              f"output_dt={output_dt_s}s  duration={args.duration_days}d", flush=True)

        try:
            batch_result = run_st_lrps_batch_rk4(
                surr_model, all_y0,
                duration_s=duration_s,
                dt_s=rk4_dt,
                output_dt_s=output_dt_s,
                args=args,
            )
        except Exception as exc:
            print(f"[batch-rk4] ERROR: {exc}", flush=True)
            traceback.print_exc()
            batch_result = None

        if batch_result is not None:
            # Save batch trajectories optionally
            if args.save_batch_trajectories:
                npz_batch = out_dir / "trajectories_batch_rk4.npz"
                _ensure_dir(npz_batch)
                np.savez_compressed(npz_batch,
                                    t=batch_result["t"], Y=batch_result["Y"])
                print(f"  [batch-rk4] Trajectories saved: {npz_batch}", flush=True)

            # Collect truth results list aligned to scenarios
            truth_list = [truth_results_all.get(sc.scenario_id) for sc in scenarios]

            # SH200 RK4 reference for error decomposition
            sh200_rk4_result = None
            if args.batch_rk4_reference == "sh200_rk4":
                print("[batch-rk4] Running SH200 CPU RK4 reference "
                      "(may take several minutes) ...", flush=True)
                grav_sh200 = model_cache.get(truth_model)
                try:
                    sh200_rk4_result = run_sh200_cpu_rk4_reference(
                        grav_sh200, all_y0,
                        duration_s=duration_s,
                        dt_s=rk4_dt,
                        output_dt_s=output_dt_s,
                    )
                except Exception as exc:
                    print(f"[batch-rk4] SH200 RK4 reference failed: {exc}", flush=True)

            # Compute metrics
            total_rows, model_rows, integr_rows = compute_batch_rk4_metrics(
                batch_result, truth_list, scenarios, sh200_rk4_result
            )

            # Save batch metrics CSVs
            _write_csv(total_rows, out_dir / "batch_rk4_per_scenario_metrics.csv")
            if model_rows:
                _write_csv(model_rows, out_dir / "batch_rk4_model_error_metrics.csv")
            if integr_rows:
                _write_csv(integr_rows, out_dir / "batch_rk4_integrator_error_metrics.csv")

            # Aggregate
            agg_total = _batch_agg_stats([r for r in total_rows if r.get("status") == "ok"],
                                         "rms_pos_err_km")
            agg_model = _batch_agg_stats([r for r in model_rows if r.get("status") == "ok"],
                                         "rms_pos_err_km") if model_rows else {}
            agg_integr= _batch_agg_stats([r for r in integr_rows if r.get("status") == "ok"],
                                         "rms_pos_err_km") if integr_rows else {}

            batch_summary = {
                "device": batch_result.get("device"),
                "mode": batch_result.get("mode"),
                "n_scenarios": batch_result.get("n_scenarios"),
                "rk4_dt_s": batch_result.get("dt_s"),
                "runtime_s": batch_result.get("runtime_s"),
                "n_steps": batch_result.get("n_steps"),
                "throughput_traj_steps_per_s": (
                    batch_result.get("n_scenarios", 1) *
                    batch_result.get("n_steps", 1) /
                    max(batch_result.get("runtime_s", 1), 1e-9)
                ),
                "total_error_vs_sh200_dop853": agg_total,
                "model_error_vs_sh200_rk4":    agg_model,
                "integrator_error_sh200_rk4_vs_dop853": agg_integr,
            }
            with open(out_dir / "batch_rk4_summary.json", "w") as f:
                json.dump(batch_summary, f, indent=4, default=str)
            _write_csv(
                [
                    {"comparison": "stlrps_rk4_vs_sh200_dop853", **agg_total},
                    {"comparison": "stlrps_rk4_vs_sh200_rk4", **agg_model},
                    {"comparison": "sh200_rk4_vs_sh200_dop853", **agg_integr},
                ],
                out_dir / "batch_rk4_aggregate_summary.csv",
            )
            runtime_rows = [
                {
                    "model": "st_lrps_batch_rk4",
                    "reference": "sh200_dop853",
                    "runtime_s": batch_result.get("runtime_s"),
                    "n_scenarios": batch_result.get("n_scenarios"),
                    "n_steps": batch_result.get("n_steps"),
                    "scenario_per_second": (
                        batch_result.get("n_scenarios", 0)
                        / max(float(batch_result.get("runtime_s", 1.0)), 1e-9)
                    ),
                    "traj_steps_per_second": batch_summary["throughput_traj_steps_per_s"],
                    "device": batch_result.get("device"),
                    "batch_size": args.batch_size or batch_result.get("n_scenarios"),
                    "torch_dtype": batch_result.get("torch_dtype", args.torch_dtype),
                }
            ]
            if sh200_rk4_result is not None:
                sh_runtime = float(sh200_rk4_result.get("runtime_s", 0.0))
                sh_n = float(sh200_rk4_result.get("n_scenarios", 0) or 0)
                sh_steps = float(sh200_rk4_result.get("n_steps", 0) or 0)
                runtime_rows.append({
                    "model": "sh200_rk4",
                    "reference": "sh200_dop853",
                    "runtime_s": sh_runtime,
                    "n_scenarios": sh_n,
                    "n_steps": sh_steps,
                    "scenario_per_second": sh_n / max(sh_runtime, 1e-9),
                    "traj_steps_per_second": sh_n * sh_steps / max(sh_runtime, 1e-9),
                    "device": "cpu",
                    "batch_size": 1,
                    "torch_dtype": "numpy_float64",
                })
            _write_csv(runtime_rows, out_dir / "batch_rk4_runtime_summary.csv")

            # Plots
            print("\n[batch-rk4] Generating batch plots ...", flush=True)
            plot_batch_rk4_results(total_rows, model_rows, integr_rows, batch_result, plots_dir)
            plot_batch_selected_scenario(total_rows, batch_result, truth_list, scenarios, plots_dir)

            # Print summary
            _print_batch_summary(batch_result, total_rows, model_rows, integr_rows, args)

    elif args.batch_rk4 and getattr(args, "rebuild_metrics", False):
        print("[batch-rk4] Skipping batch RK4 propagation during --rebuild-metrics.",
              flush=True)

    elif args.batch_rk4 and "st_lrps" not in compare_models:
        print("[batch-rk4] WARNING: --batch-rk4 requires st_lrps in --models. Skipped.",
              flush=True)

    # PDF report
    print("\n[harness] Writing PDF report ...", flush=True)
    try:
        write_report_pdf(args, scenarios, agg, rankings, worst_cases, plots_dir, out_dir)
    except Exception as exc:
        print(f"  WARNING: PDF generation failed: {exc}", flush=True)

    _print_final_validation_summary(args, agg, batch_result, total_rows, model_rows, integr_rows)

    print(f"\n[harness] Complete -> {out_dir}", flush=True)
    print(f"  scenarios.csv              per_scenario_metrics.csv")
    print(f"  aggregate_summary.csv      aggregate_summary.json")
    print(f"  ranking_summary.csv        worst_cases_by_model.csv")
    if batch_result is not None:
        print(f"  batch_rk4_per_scenario_metrics.csv  batch_rk4_summary.json")
    print(f"  plots/                     gravity_random_validation_report.pdf")


# =============================================================================
# ST-LRPS auto-detection
# =============================================================================

def _auto_find_st_lrps_dir() -> Optional[str]:
    """
    Return the newest valid surrogate run directory.
    Uses models.surrogate_gravity.find_latest_st_lrps_model_dir which requires
    config.json AND checkpoints/ckpt_best.pt (or ckpt_last.pt).
    """
    result = find_latest_st_lrps_model_dir()
    if result is not None:
        return str(result)
    return None


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Lunar Gravity Validation ...", flush=True)

    # ST-LRPS auto-detection (requires real model file, not just config.json)
    models_raw = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    gpu_models_raw = _parse_model_list_csv(getattr(args, "gpu_models", ""))
    needs_stlrps = (
        ("st_lrps" in gpu_models_raw)
        if bool(args.gpu_batch_compare)
        else ("st_lrps" in models_raw)
    )
    if needs_stlrps and not args.st_lrps_model_dir:
        auto_dir = _auto_find_st_lrps_dir()
        if auto_dir:
            args.st_lrps_model_dir = auto_dir
            weight = _find_st_lrps_weight_file(auto_dir)
            print(f"[auto] ST-LRPS model dir: {auto_dir}", flush=True)
            if weight:
                print(f"[auto] ST-LRPS weight file: {weight}", flush=True)
        else:
            if args.require_st_lrps:
                raise FileNotFoundError("ST-LRPS requested but no valid model dir found.")
            print("WARNING: 'st_lrps' requested but no valid model dir found in "
                  "st_lrps/runs/. Removing st_lrps from comparison.",
                  flush=True)
            models_raw = [m for m in models_raw if m != "st_lrps"]
            gpu_models_raw = [m for m in gpu_models_raw if m != "st_lrps"]
            args.models = ",".join(models_raw)
            args.gpu_models = ",".join(gpu_models_raw)

    cfg   = build_base_config(args)
    _write_run_metadata(args, out_dir)
    ephem = EphemerisManager.from_time_and_spice(cfg.time, cfg.spice)

    if args.gpu_batch_compare:
        run_gpu_batch_compare_mode(args, cfg, ephem)
        return

    if args.force_sample_trajectory:
        models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        truth  = args.force_sample_trajectory.strip().lower()
        if truth not in models:
            models.append(truth)
        evaluate_forces(models, truth, args, cfg, ephem, out_dir)
        return

    if args.random_scenarios > 0:
        run_random_scenario_mode(args, cfg, ephem)
    else:
        run_single_orbit_mode(args, cfg, ephem)


if __name__ == "__main__":
    main()
