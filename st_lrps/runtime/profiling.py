"""Runtime profiling tools for ST-LRPS surrogate force-model inference.

Detected runtime path
---------------------
Propagation-facing ST-LRPS inference is loaded through
``st_lrps.runtime.force_model.load_surrogate_force_model``.  That loader
resolves a run directory, selects ``ckpt_best.pt`` with ``ckpt_last.pt``
fallback through ``st_lrps.artifacts.manager``, reconstructs the trained
network and scaler, then returns ``SurrogateForceModel``.  Acceleration is
computed as the autograd gradient of the learned residual potential, with
large inputs processed through the runtime chunk size.

This module measures that path without changing physics, model architecture,
checkpoint schema, loss functions, or propagation algorithms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import torch

from st_lrps.artifacts.manager import (
    load_checkpoint,
    load_scaler_for_run,
    make_run_layout,
    read_run_manifest,
    resolve_run_dir,
)
from st_lrps.networks.models import reconstruct_model_from_artifacts
from st_lrps.runtime.force_model import SurrogateForceModel


# ---------------------------------------------------------------------------
# Dataclass schema
# ---------------------------------------------------------------------------


@dataclass
class ProfileTimerResult:
    """Best-effort load-phase timing for one ST-LRPS runtime load."""

    total_load_s: Optional[float] = None
    checkpoint_select_s: Optional[float] = None
    config_load_s: Optional[float] = None
    checkpoint_load_s: Optional[float] = None
    scaler_load_s: Optional[float] = None
    model_reconstruct_s: Optional[float] = None
    model_to_device_s: Optional[float] = None


@dataclass
class InferenceProfileResult:
    """Steady-state inference timing and diagnostics for one batch/chunk case."""

    batch_size: int
    n_warmup: int
    n_repeat: int
    chunk_size: Optional[int]
    input_source: str
    device: str
    total_wall_s: float
    mean_call_s: float
    median_call_s: float
    p95_call_s: float
    min_call_s: float
    max_call_s: float
    samples_per_s: float
    microseconds_per_sample: float
    cuda_memory_allocated_mb_before: Optional[float] = None
    cuda_memory_allocated_mb_after: Optional[float] = None
    cuda_memory_reserved_mb_after: Optional[float] = None
    peak_cuda_memory_mb: Optional[float] = None
    output_shape: str = ""
    output_dtype: str = ""
    finite_output_fraction: float = 0.0
    accel_norm_mean: Optional[float] = None
    accel_norm_max: Optional[float] = None
    potential_mean_call_s: Optional[float] = None
    accel_minus_potential_mean_call_s: Optional[float] = None
    classic_sh_degree: Optional[int] = None
    classic_sh_mean_call_s: Optional[float] = None
    classic_sh_median_call_s: Optional[float] = None
    classic_sh_p95_call_s: Optional[float] = None
    speedup_vs_classic_sh: Optional[float] = None
    accuracy_diff_vs_classic_sh: Optional[float] = None


@dataclass
class RuntimeProfileConfig:
    """Configuration used to produce a runtime profile report."""

    model_dir: str
    device: str = "auto"
    batch_sizes: list[int] = field(default_factory=lambda: [1, 16, 128, 1024, 8192])
    n_warmup: int = 10
    n_repeat: int = 50
    chunk_sizes: list[Optional[int]] = field(default_factory=lambda: [None])
    input_source: str = "synthetic"
    data_path: Optional[str] = None
    dataset_name: str = "data"
    alt_min_km: float = 100.0
    alt_max_km: float = 2000.0
    seed: int = 42
    output_dir: Optional[str] = None
    compare_classic_sh: bool = False
    classic_sh_degree: int = 60


@dataclass
class RuntimeProfileReport:
    """Serializable ST-LRPS runtime profiling report."""

    config: RuntimeProfileConfig
    model_dir: str
    checkpoint_kind: Optional[str]
    checkpoint_path: Optional[str]
    device: str
    dtype: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: Optional[str]
    created_at_utc: str
    load: ProfileTimerResult
    inference_results: list[InferenceProfileResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def now_perf() -> float:
    """Return a monotonic high-resolution timestamp."""

    return time.perf_counter()


def _as_torch_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def sync_device(device: Any) -> None:
    """Synchronize CUDA work before/after timing when the target is CUDA."""

    dev = _as_torch_device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def timed_call(fn: Callable[[], Any], *, device: Any) -> tuple[float, Any]:
    """Run ``fn`` once and return ``(elapsed_seconds, result)``."""

    sync_device(device)
    start = now_perf()
    result = fn()
    sync_device(device)
    return now_perf() - start, result


def summarize_times(times: Sequence[float]) -> dict[str, float]:
    """Return mean, median, p95, min, and max for measured call times."""

    values = [float(t) for t in times]
    if not values:
        raise ValueError("summarize_times requires at least one timing value")
    sorted_values = sorted(values)
    return {
        "mean": float(statistics.fmean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "p95": float(np.percentile(sorted_values, 95)),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
    }


# ---------------------------------------------------------------------------
# Query generation / sampling
# ---------------------------------------------------------------------------


def generate_lunar_shell_queries(
    n: int,
    *,
    r_ref_m: float,
    alt_min_km: float,
    alt_max_km: float,
    seed: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> np.ndarray | torch.Tensor:
    """Generate random Moon-centered positions in SI meters with shape ``(n, 3)``.

    Directions are uniform on the sphere and altitude is uniform in the provided
    kilometer range.  When ``device`` or ``dtype`` is supplied, a torch tensor is
    returned; otherwise the result is a NumPy ``float64`` array.
    """

    count = int(n)
    if count <= 0:
        raise ValueError("n must be positive")
    if alt_max_km < alt_min_km:
        raise ValueError("alt_max_km must be >= alt_min_km")

    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-30)
    alt_m = rng.uniform(float(alt_min_km), float(alt_max_km), size=(count, 1)) * 1000.0
    positions = (float(r_ref_m) + alt_m) * directions

    if device is not None or dtype is not None:
        return torch.as_tensor(
            positions,
            device=device or torch.device("cpu"),
            dtype=dtype or torch.float32,
        )
    return positions.astype(np.float64, copy=False)


def sample_hdf5_position_queries(
    data_path: Path | str,
    *,
    n: int,
    dataset_name: str = "data",
    seed: int = 42,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> np.ndarray | torch.Tensor:
    """Sample ``x,y,z`` rows from an HDF5 dataset without loading it all."""

    try:
        import h5py
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Dataset-backed profiling requires h5py.") from exc

    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    if int(n) <= 0:
        raise ValueError("n must be positive")

    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            available = ", ".join(str(k) for k in handle.keys())
            raise KeyError(
                f"HDF5 dataset {dataset_name!r} not found in {path}. "
                f"Available top-level keys: {available or '<none>'}"
            )
        ds = handle[dataset_name]
        if len(ds.shape) != 2 or int(ds.shape[1]) < 3:
            raise ValueError(
                f"Dataset {dataset_name!r} must have shape (N, >=3); got {ds.shape}."
            )
        total = int(ds.shape[0])
        if total <= 0:
            raise ValueError(f"Dataset {dataset_name!r} is empty.")

        rng = np.random.default_rng(int(seed))
        sampled = rng.integers(0, total, size=int(n))
        unique_idx, inverse = np.unique(sampled, return_inverse=True)
        unique_rows = np.asarray(ds[unique_idx, :3], dtype=np.float64)
        positions = unique_rows[inverse]

    if device is not None or dtype is not None:
        return torch.as_tensor(
            positions,
            device=device or torch.device("cpu"),
            dtype=dtype or torch.float32,
        )
    return positions.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Runtime load profiling
# ---------------------------------------------------------------------------


def _resolve_profile_device(device: str) -> torch.device:
    dev = str(device or "auto").strip().lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)


def _select_checkpoint_path(layout: Any) -> Path:
    for candidate in (layout.ckpt_best, layout.ckpt_last):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found in {layout.checkpoints_dir}. "
        f"Expected {layout.ckpt_best.name} or {layout.ckpt_last.name}."
    )


def _load_profiled_surrogate_force_model(
    model_dir: Path | str,
    *,
    device: str,
    chunk_size: int,
    allow_config_mismatch: bool = False,
) -> tuple[SurrogateForceModel, ProfileTimerResult, dict[str, Any]]:
    """Load the canonical runtime object while collecting phase timings."""

    load_t0 = now_perf()
    dev = _resolve_profile_device(device)
    run_dir = resolve_run_dir(model_dir)
    layout = make_run_layout(run_dir)
    timings = ProfileTimerResult()

    elapsed, ckpt_path = timed_call(lambda: _select_checkpoint_path(layout), device=dev)
    timings.checkpoint_select_s = elapsed

    cfg_json: dict[str, Any]
    if layout.config_json.exists():
        elapsed, cfg_json = timed_call(
            lambda: json.loads(layout.config_json.read_text(encoding="utf-8")),
            device=dev,
        )
        timings.config_load_s = elapsed
        config_source = "config_json"
    else:
        cfg_json = {}
        config_source = "checkpoint"

    elapsed, ckpt = timed_call(lambda: load_checkpoint(ckpt_path, dev), device=dev)
    timings.checkpoint_load_s = elapsed

    if not cfg_json:
        ckpt_cfg = ckpt.get("config")
        if not isinstance(ckpt_cfg, dict):
            raise FileNotFoundError(
                f"Missing config artifact for run {layout.run_dir}. "
                f"Expected {layout.config_json} or checkpoint['config']."
            )
        cfg_json = dict(ckpt_cfg)

    elapsed, reconstructed = timed_call(
        lambda: reconstruct_model_from_artifacts(
            cfg_json,
            ckpt,
            dev,
            allow_config_mismatch=allow_config_mismatch,
        ),
        device=dev,
    )
    timings.model_reconstruct_s = elapsed
    model, merged_cfg, report = reconstructed

    elapsed, scaler_result = timed_call(
        lambda: load_scaler_for_run(layout, ckpt, device=dev, dtype=torch.float32),
        device=dev,
    )
    timings.scaler_load_s = elapsed
    scaler, scaler_report = scaler_result

    manifest = read_run_manifest(layout)
    report.update(
        {
            "checkpoint_schema_version": ckpt.get("schema_version"),
            "checkpoint_kind": ckpt.get("kind"),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_epoch_display": ckpt.get("epoch_display"),
            "checkpoint_metric": (
                (ckpt.get("scoring") or {}).get("score")
                if isinstance(ckpt.get("scoring"), dict)
                else None
            ),
            "checkpoint_hash": ckpt.get("checkpoint_hash"),
            "checkpoint_config_source": config_source,
            "run_manifest_path": str(layout.run_manifest_json)
            if layout.run_manifest_json.exists()
            else None,
            "run_manifest": manifest or None,
            "scaler_source": scaler_report["scaler_source"],
            "scaler_hash": scaler_report["scaler_hash"],
        }
    )

    runtime = SurrogateForceModel(
        model=model,
        scaler=scaler,
        cfg=merged_cfg,
        device=dev,
        chunk_size=int(chunk_size),
        checkpoint_path=report.get("checkpoint_path"),
        checkpoint_epoch=report.get("checkpoint_epoch"),
        architecture_signature=report.get("architecture_signature"),
        run_manifest=manifest,
        strict_domain=False,
    )
    sync_device(dev)
    timings.total_load_s = now_perf() - load_t0
    return runtime, timings, report


# ---------------------------------------------------------------------------
# Inference profiling
# ---------------------------------------------------------------------------


def _cuda_memory_mb(device: torch.device) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None, None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(idx) / (1024.0 * 1024.0)
    reserved = torch.cuda.memory_reserved(idx) / (1024.0 * 1024.0)
    peak = torch.cuda.max_memory_allocated(idx) / (1024.0 * 1024.0)
    return float(allocated), float(reserved), float(peak)


def _call_runtime_acceleration(runtime: Any, queries: np.ndarray) -> Any:
    if hasattr(runtime, "predict_residual_accel"):
        return runtime.predict_residual_accel(queries)
    if hasattr(runtime, "acceleration_fixed_batch"):
        return runtime.acceleration_fixed_batch(queries)
    if hasattr(runtime, "predict_acceleration"):
        return runtime.predict_acceleration(queries)
    raise TypeError(
        "Runtime object must expose predict_residual_accel, acceleration_fixed_batch, "
        "or predict_acceleration."
    )


def _call_runtime_potential(runtime: Any, queries: np.ndarray) -> Any:
    if not hasattr(runtime, "predict_residual_potential"):
        raise TypeError("Runtime object does not expose predict_residual_potential.")
    return runtime.predict_residual_potential(queries)


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, tuple) and len(value) >= 2:
        value = value[1]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _diagnose_output(output: Any) -> dict[str, Any]:
    arr = _as_array(output)
    finite = np.isfinite(arr)
    finite_fraction = float(finite.mean()) if arr.size else 0.0
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        norms = np.linalg.norm(arr.reshape(-1, 3), axis=1)
        norm_mean = float(np.nanmean(norms)) if norms.size else None
        norm_max = float(np.nanmax(norms)) if norms.size else None
    else:
        norm_mean = None
        norm_max = None
    return {
        "output_shape": str(tuple(arr.shape)),
        "output_dtype": str(arr.dtype),
        "finite_output_fraction": finite_fraction,
        "accel_norm_mean": norm_mean,
        "accel_norm_max": norm_max,
    }


def _profile_potential_only(
    runtime: Any,
    queries: np.ndarray,
    *,
    n_warmup: int,
    n_repeat: int,
    device: torch.device,
) -> Optional[dict[str, float]]:
    if not hasattr(runtime, "predict_residual_potential"):
        return None
    try:
        for _ in range(int(n_warmup)):
            _call_runtime_potential(runtime, queries)
        times = [
            timed_call(lambda: _call_runtime_potential(runtime, queries), device=device)[0]
            for _ in range(int(n_repeat))
        ]
    except Exception:
        return None
    return summarize_times(times)


def _profile_classic_sh_batch(
    gravity_model: Any,
    queries: np.ndarray,
    *,
    degree: int,
    n_warmup: int,
    n_repeat: int,
) -> tuple[dict[str, float], np.ndarray]:
    def call() -> np.ndarray:
        out = np.empty((queries.shape[0], 3), dtype=np.float64)
        for idx, row in enumerate(queries):
            out[idx, :] = gravity_model.accel_fixed(row, degree=int(degree))
        return out

    for _ in range(int(n_warmup)):
        call()
    times: list[float] = []
    last = None
    for _ in range(int(n_repeat)):
        elapsed, last = timed_call(call, device=torch.device("cpu"))
        times.append(elapsed)
    return summarize_times(times), np.asarray(last, dtype=np.float64)


def _profile_one_case(
    runtime: Any,
    queries: np.ndarray,
    *,
    batch_size: int,
    chunk_size: Optional[int],
    input_source: str,
    n_warmup: int,
    n_repeat: int,
    device: torch.device,
    classic_sh_model: Optional[Any],
    classic_sh_degree: int,
) -> InferenceProfileResult:
    old_chunk = getattr(runtime, "chunk_size", None)
    if chunk_size is not None and hasattr(runtime, "chunk_size"):
        runtime.chunk_size = int(chunk_size)

    try:
        for _ in range(int(n_warmup)):
            _call_runtime_acceleration(runtime, queries)

        sync_device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        mem_before, _, _ = _cuda_memory_mb(device)

        times: list[float] = []
        last_output = None
        loop_start = now_perf()
        for _ in range(int(n_repeat)):
            elapsed, last_output = timed_call(
                lambda: _call_runtime_acceleration(runtime, queries),
                device=device,
            )
            times.append(elapsed)
        total_wall_s = now_perf() - loop_start
        mem_after, mem_reserved_after, peak_mem = _cuda_memory_mb(device)

        stats = summarize_times(times)
        diagnostics = _diagnose_output(last_output)
        samples_per_s = float(batch_size) / stats["mean"] if stats["mean"] > 0.0 else math.inf
        us_per_sample = stats["mean"] * 1_000_000.0 / float(batch_size)

        potential_stats = _profile_potential_only(
            runtime,
            queries,
            n_warmup=max(1, min(int(n_warmup), 3)),
            n_repeat=max(1, min(int(n_repeat), 10)),
            device=device,
        )

        classic_stats = None
        classic_out = None
        if classic_sh_model is not None:
            classic_stats, classic_out = _profile_classic_sh_batch(
                classic_sh_model,
                queries,
                degree=int(classic_sh_degree),
                n_warmup=max(1, min(int(n_warmup), 3)),
                n_repeat=max(1, min(int(n_repeat), 10)),
            )

        speedup = None
        diff = None
        if classic_stats is not None:
            speedup = (
                classic_stats["mean"] / stats["mean"]
                if stats["mean"] > 0.0
                else math.inf
            )
            st_arr = _as_array(last_output).reshape(-1, 3)
            if classic_out is not None and classic_out.shape == st_arr.shape:
                diff = float(np.mean(np.linalg.norm(st_arr - classic_out, axis=1)))

        pot_mean = potential_stats["mean"] if potential_stats else None
        overhead = (stats["mean"] - pot_mean) if pot_mean is not None else None

        return InferenceProfileResult(
            batch_size=int(batch_size),
            n_warmup=int(n_warmup),
            n_repeat=int(n_repeat),
            chunk_size=int(chunk_size) if chunk_size is not None else None,
            input_source=str(input_source),
            device=str(device),
            total_wall_s=float(total_wall_s),
            mean_call_s=stats["mean"],
            median_call_s=stats["median"],
            p95_call_s=stats["p95"],
            min_call_s=stats["min"],
            max_call_s=stats["max"],
            samples_per_s=float(samples_per_s),
            microseconds_per_sample=float(us_per_sample),
            cuda_memory_allocated_mb_before=mem_before,
            cuda_memory_allocated_mb_after=mem_after,
            cuda_memory_reserved_mb_after=mem_reserved_after,
            peak_cuda_memory_mb=peak_mem,
            output_shape=diagnostics["output_shape"],
            output_dtype=diagnostics["output_dtype"],
            finite_output_fraction=diagnostics["finite_output_fraction"],
            accel_norm_mean=diagnostics["accel_norm_mean"],
            accel_norm_max=diagnostics["accel_norm_max"],
            potential_mean_call_s=pot_mean,
            accel_minus_potential_mean_call_s=overhead,
            classic_sh_degree=int(classic_sh_degree) if classic_stats is not None else None,
            classic_sh_mean_call_s=classic_stats["mean"] if classic_stats else None,
            classic_sh_median_call_s=classic_stats["median"] if classic_stats else None,
            classic_sh_p95_call_s=classic_stats["p95"] if classic_stats else None,
            speedup_vs_classic_sh=speedup,
            accuracy_diff_vs_classic_sh=diff,
        )
    finally:
        if chunk_size is not None and hasattr(runtime, "chunk_size") and old_chunk is not None:
            runtime.chunk_size = old_chunk


def _make_queries(
    *,
    input_source: str,
    batch_size: int,
    r_ref_m: float,
    alt_min_km: float,
    alt_max_km: float,
    seed: int,
    data_path: Optional[Path | str],
    dataset_name: str,
) -> np.ndarray:
    if input_source == "dataset":
        if data_path is None:
            raise ValueError("--input-source dataset requires --data")
        return np.asarray(
            sample_hdf5_position_queries(
                data_path,
                n=int(batch_size),
                dataset_name=dataset_name,
                seed=seed,
            ),
            dtype=np.float64,
        )
    if input_source != "synthetic":
        raise ValueError("input_source must be 'synthetic' or 'dataset'")
    return np.asarray(
        generate_lunar_shell_queries(
            int(batch_size),
            r_ref_m=float(r_ref_m),
            alt_min_km=float(alt_min_km),
            alt_max_km=float(alt_max_km),
            seed=int(seed),
        ),
        dtype=np.float64,
    )


def _try_load_classic_sh(degree: int) -> tuple[Optional[Any], Optional[str]]:
    try:
        from models.spherical_harmonics import GravityModel
        from st_lrps.data.dataset_parameters import DEFAULT_DATASET_CONFIG, resolve_lunar_gravity_path

        path = resolve_lunar_gravity_path(getattr(DEFAULT_DATASET_CONFIG, "gravity_gfc_path"))
        if not Path(path).exists():
            return None, f"Classic SH comparison skipped: gravity file not found at {path}"
        return GravityModel.from_file(str(path), requested_degree=int(degree)), None
    except Exception as exc:
        return None, f"Classic SH comparison skipped: {exc}"


def profile_surrogate_runtime(
    model_dir: Path | str,
    *,
    device: str = "auto",
    batch_sizes: Sequence[int] = (1, 16, 128, 1024, 8192),
    n_warmup: int = 10,
    n_repeat: int = 50,
    chunk_sizes: Sequence[int | None] = (None,),
    input_source: str = "synthetic",
    data_path: Optional[Path | str] = None,
    dataset_name: str = "data",
    alt_min_km: float = 100.0,
    alt_max_km: float = 2000.0,
    seed: int = 42,
    output_dir: Optional[Path | str] = None,
    compare_classic_sh: bool = False,
    classic_sh_degree: int = 60,
) -> RuntimeProfileReport:
    """Profile ST-LRPS runtime loading and steady-state acceleration inference."""

    if int(n_repeat) <= 0:
        raise ValueError("n_repeat must be positive")
    if int(n_warmup) < 0:
        raise ValueError("n_warmup must be >= 0")

    batch_list = [int(v) for v in batch_sizes]
    chunk_list = list(chunk_sizes) or [None]
    if any(v <= 0 for v in batch_list):
        raise ValueError("batch_sizes must all be positive")

    default_chunk = next((c for c in chunk_list if c is not None), 8192)
    runtime, load_timings, load_report = _load_profiled_surrogate_force_model(
        model_dir,
        device=device,
        chunk_size=int(default_chunk or 8192),
    )
    runtime_device = _as_torch_device(runtime.device)

    warnings_out: list[str] = []
    classic_model = None
    if compare_classic_sh:
        classic_model, warning_text = _try_load_classic_sh(int(classic_sh_degree))
        if warning_text:
            warnings.warn(warning_text)
            warnings_out.append(warning_text)

    cfg = RuntimeProfileConfig(
        model_dir=str(Path(model_dir).expanduser()),
        device=str(device),
        batch_sizes=batch_list,
        n_warmup=int(n_warmup),
        n_repeat=int(n_repeat),
        chunk_sizes=[int(c) if c is not None else None for c in chunk_list],
        input_source=str(input_source),
        data_path=str(data_path) if data_path is not None else None,
        dataset_name=str(dataset_name),
        alt_min_km=float(alt_min_km),
        alt_max_km=float(alt_max_km),
        seed=int(seed),
        output_dir=str(output_dir) if output_dir is not None else None,
        compare_classic_sh=bool(compare_classic_sh),
        classic_sh_degree=int(classic_sh_degree),
    )
    report = RuntimeProfileReport(
        config=cfg,
        model_dir=str(resolve_run_dir(model_dir)),
        checkpoint_kind=load_report.get("checkpoint_kind"),
        checkpoint_path=load_report.get("checkpoint_path"),
        device=str(runtime_device),
        dtype=str(torch.float32),
        torch_version=str(torch.__version__),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_name=(
            torch.cuda.get_device_name(runtime_device)
            if runtime_device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        load=load_timings,
        warnings=warnings_out,
    )

    for batch_size in batch_list:
        for chunk_size in chunk_list:
            queries = _make_queries(
                input_source=str(input_source),
                batch_size=int(batch_size),
                r_ref_m=float(getattr(runtime, "r_ref_m", 1737_400.0)),
                alt_min_km=float(alt_min_km),
                alt_max_km=float(alt_max_km),
                seed=int(seed) + int(batch_size) + (int(chunk_size) if chunk_size else 0),
                data_path=data_path,
                dataset_name=dataset_name,
            )
            result = _profile_one_case(
                runtime,
                queries,
                batch_size=int(batch_size),
                chunk_size=chunk_size,
                input_source=str(input_source),
                n_warmup=int(n_warmup),
                n_repeat=int(n_repeat),
                device=runtime_device,
                classic_sh_model=classic_model,
                classic_sh_degree=int(classic_sh_degree),
            )
            report.inference_results.append(result)

    if output_dir is not None:
        write_runtime_profile_outputs(report, output_dir)
    return report


# ---------------------------------------------------------------------------
# Serialization and output helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
    return value


def write_runtime_profile_json(report: RuntimeProfileReport, output_dir: Path | str) -> Path:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime_profile.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_runtime_profile_csv(report: RuntimeProfileReport, output_dir: Path | str) -> Path:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime_profile.csv"
    columns = [
        "batch_size",
        "chunk_size",
        "device",
        "n_repeat",
        "mean_call_s",
        "median_call_s",
        "p95_call_s",
        "samples_per_s",
        "microseconds_per_sample",
        "cuda_memory_allocated_mb_after",
        "cuda_memory_reserved_mb_after",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report.inference_results:
            payload = asdict(row)
            payload["chunk_size"] = "none" if row.chunk_size is None else row.chunk_size
            writer.writerow({key: payload.get(key) for key in columns})
    return path


def _best_by(
    rows: Sequence[InferenceProfileResult],
    key: Callable[[InferenceProfileResult], float],
    *,
    reverse: bool = False,
) -> Optional[InferenceProfileResult]:
    if not rows:
        return None
    return sorted(rows, key=key, reverse=reverse)[0]


def write_runtime_profile_summary(report: RuntimeProfileReport, output_dir: Path | str) -> Path:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime_profile_summary.md"
    best_throughput = _best_by(report.inference_results, lambda r: r.samples_per_s, reverse=True)
    lowest_latency = _best_by(report.inference_results, lambda r: r.median_call_s)

    lines = [
        "# ST-LRPS Runtime Profile Summary",
        "",
        f"- model_dir: `{report.model_dir}`",
        f"- checkpoint: `{report.checkpoint_path}`",
        f"- checkpoint_kind: `{report.checkpoint_kind}`",
        f"- device: `{report.device}`",
        f"- cuda_used: `{str(report.device).startswith('cuda')}`",
        f"- created_at_utc: `{report.created_at_utc}`",
        "",
    ]
    if best_throughput is not None:
        lines.append(
            f"- best throughput: batch `{best_throughput.batch_size}`, "
            f"chunk `{best_throughput.chunk_size}`, "
            f"{best_throughput.samples_per_s:.3g} samples/s"
        )
    if lowest_latency is not None:
        lines.append(
            f"- lowest median latency: batch `{lowest_latency.batch_size}`, "
            f"chunk `{lowest_latency.chunk_size}`, "
            f"{lowest_latency.median_call_s:.6g} s/call"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Batch throughput improves when batch size increases if Python overhead dominates.",
            "- If p95 is much higher than median, runtime jitter or memory pressure may be present.",
            "- If batch size 1 is slow but batch 1024 is fast per sample, Monte Carlo should use batch force evaluation.",
            "- Potential-only timing is a low-risk proxy for forward-path cost; full acceleration timing includes autograd-gradient work.",
        ]
    )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {text}" for text in report.warnings)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_runtime_profile_plots(report: RuntimeProfileReport, output_dir: Path | str) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        warnings.warn(f"Skipping runtime profile plots: matplotlib unavailable ({exc}).")
        return []

    rows = list(report.inference_results)
    if not rows:
        return []
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    labels = ["none" if r.chunk_size is None else str(r.chunk_size) for r in rows]
    batch = np.asarray([r.batch_size for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(batch, [r.microseconds_per_sample for r in rows])
    for x_val, y_val, label in zip(batch, [r.microseconds_per_sample for r in rows], labels):
        ax.annotate(label, (x_val, y_val), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Microseconds per sample")
    ax.set_title("ST-LRPS Runtime Latency")
    ax.grid(True, which="both", alpha=0.3)
    path = out_dir / "runtime_profile_latency.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(batch, [r.samples_per_s for r in rows])
    for x_val, y_val, label in zip(batch, [r.samples_per_s for r in rows], labels):
        ax.annotate(label, (x_val, y_val), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Samples per second")
    ax.set_title("ST-LRPS Runtime Throughput")
    ax.grid(True, which="both", alpha=0.3)
    path = out_dir / "runtime_profile_throughput.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_runtime_profile_outputs(
    report: RuntimeProfileReport,
    output_dir: Path | str,
    *,
    json_only: bool = False,
    make_plots: bool = True,
) -> dict[str, Path | list[Path]]:
    outputs: dict[str, Path | list[Path]] = {
        "json": write_runtime_profile_json(report, output_dir),
    }
    if not json_only:
        outputs["csv"] = write_runtime_profile_csv(report, output_dir)
        outputs["summary"] = write_runtime_profile_summary(report, output_dir)
        if make_plots:
            outputs["plots"] = write_runtime_profile_plots(report, output_dir)
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    try:
        parsed = [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {text}") from exc
    if any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError("All values must be positive.")
    return parsed


def _parse_chunk_list(text: str) -> list[Optional[int]]:
    values = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    if not values:
        return [None]
    parsed: list[Optional[int]] = []
    for item in values:
        if item in {"none", "default", "null"}:
            parsed.append(None)
        else:
            try:
                value = int(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid chunk size: {item}") from exc
            if value <= 0:
                raise argparse.ArgumentTypeError("Chunk sizes must be positive or 'none'.")
            parsed.append(value)
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile ST-LRPS runtime loading and batched acceleration inference."
    )
    parser.add_argument("--model-dir", required=True, help="Trained ST-LRPS run directory or checkpoint path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=_parse_int_list("1,16,128,1024,8192"))
    parser.add_argument("--chunk-sizes", type=_parse_chunk_list, default=_parse_chunk_list("none"))
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-repeat", type=int, default=50)
    parser.add_argument("--input-source", choices=("synthetic", "dataset"), default="synthetic")
    parser.add_argument("--data", default=None, help="HDF5 dataset path for dataset-backed query sampling.")
    parser.add_argument("--dataset-name", default="data")
    parser.add_argument("--alt-min-km", type=float, default=100.0)
    parser.add_argument("--alt-max-km", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: outputs/runtime_performance/<model>_<timestamp>.",
    )
    parser.add_argument("--compare-classic-sh", action="store_true")
    parser.add_argument("--classic-sh-degree", type=int, default=60)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.input_source == "dataset" and not args.data:
        parser.error("--input-source dataset requires --data PATH")
    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = Path(str(args.model_dir)).stem or "st_lrps_runtime"
        model_slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_slug).strip("._-") or "st_lrps_runtime"
        args.out_dir = str(Path("outputs") / "runtime_performance" / f"{model_slug}_{stamp}")

    report = profile_surrogate_runtime(
        args.model_dir,
        device=args.device,
        batch_sizes=args.batch_sizes,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
        chunk_sizes=args.chunk_sizes,
        input_source=args.input_source,
        data_path=args.data,
        dataset_name=args.dataset_name,
        alt_min_km=args.alt_min_km,
        alt_max_km=args.alt_max_km,
        seed=args.seed,
        output_dir=None,
        compare_classic_sh=args.compare_classic_sh,
        classic_sh_degree=args.classic_sh_degree,
    )
    outputs = write_runtime_profile_outputs(
        report,
        args.out_dir,
        json_only=bool(args.json_only),
    )
    if args.verbose:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Wrote ST-LRPS runtime profile to {Path(args.out_dir).expanduser().resolve()}")
        for key, value in outputs.items():
            print(f"{key}: {value}")
    return 0


__all__ = [
    "ProfileTimerResult",
    "InferenceProfileResult",
    "RuntimeProfileConfig",
    "RuntimeProfileReport",
    "generate_lunar_shell_queries",
    "sample_hdf5_position_queries",
    "now_perf",
    "sync_device",
    "timed_call",
    "summarize_times",
    "profile_surrogate_runtime",
    "write_runtime_profile_outputs",
    "write_runtime_profile_json",
    "write_runtime_profile_csv",
    "write_runtime_profile_summary",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
