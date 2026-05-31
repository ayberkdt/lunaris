# lunaris/surrogate/st_lrps/evaluation/compare_gravity_models.py
# -*- coding: utf-8 -*-
"""
Lunar Gravity Model Validation Harness
=======================================

Compares SH20/SH80/SH120/SH160 (and optionally ST-LRPS) against SH200
as ground truth, either for a single orbit or across N random scenarios.

Smoke tests:

  # CPU smoke (no GPU needed)
  python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 3 --duration-days 0.01 \\
      --models sh20,sh80 --truth sh200 \\
      --output-dir results/smoke_cpu

  # ST-LRPS force batch evaluation
  python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \\
      --force-sample-trajectory sh200 \\
      --models st_lrps,sh80 \\
      --st-lrps-mode gpu_rk4 --force-batch-size 8192 \\
      --output-dir results/smoke_force_gpu

  # 100-orbit ST-LRPS GPU batch RK4
  python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 100 --scenario-seed 42 \\
      --scenario-mode near_circular_altitude \\
      --altitude-min-km 200 --altitude-max-km 400 \\
      --duration-days 1.0 --dt-out 60 \\
      --models st_lrps --truth sh200 \\
      --st-lrps-mode gpu_rk4 \\
      --batch-rk4 --st-lrps-rk4-dt 10 \\
      --output-dir results/stlrps_batch_rk4_100

  # Full comparison (DOP853 + batch RK4)
  python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 100 --scenario-seed 42 \\
      --scenario-mode near_circular_altitude \\
      --altitude-min-km 200 --altitude-max-km 400 \\
      --duration-days 1.0 \\
      --models sh20,sh80,sh120,sh160,st_lrps --truth sh200 \\
      --st-lrps-mode gpu_rk4 \\
      --batch-rk4 --st-lrps-rk4-dt 10 \\
      --output-dir results/full_validation_100

  # Full GPU batch comparison: SH200 DOP853 truth vs GPU RK4 models
  python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \\
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
  python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \\
      --random-scenarios 5 --duration-days 0.05 \\
      --truth sh200 \\
      --gpu-models sh200,sh60,sh20,st_lrps \\
      --gpu-batch-compare --rk4-dt-s 10 \\
      --output-dir results/smoke_gpu_batch_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from lunaris.physics.ephemeris import EphemerisManager
except ImportError as exc:
    import sys

    print(f"CRITICAL: Lunaris package import failed. Missing: {exc}", file=sys.stderr)
    sys.exit(1)


# --- intra-package wiring (auto-generated split) ---
from ._gravity_benchmark.types import (
    INCLINATION_SAMPLING_METHODS,
    SAMPLING_METHODS,
    _find_st_lrps_weight_file,
    build_base_config,
)
from ._gravity_benchmark.compute import (
    GPU_INTEGRATORS,
    _model_display_name,
    _parse_model_list_csv,
)
from ._gravity_benchmark.results_io import (
    _benchmark_cache_dir,
    _build_gpu_batch_summary,
    _cache_provenance,
    _coerce_numeric_row,
    _read_csv_rows,
    _write_run_metadata,
)
from ._gravity_benchmark.plotting import (
    estimate_stlrps_equivalent_sh_degree,
)
from ._gravity_benchmark.modes import (
    _auto_find_st_lrps_dir,
    evaluate_forces,
    run_gpu_batch_compare_mode,
    run_random_scenario_mode,
    run_single_orbit_mode,
)

# ---------------------------------------------------------------------------
# Backward-compatible facade. The implementation now lives in the internal
# ``_gravity_benchmark`` subpackage, but ``compare_gravity_models`` remains the
# stable public import surface. Re-export every top-level symbol from the
# implementation modules so ``compare_gravity_models.X`` and
# ``from ...compare_gravity_models import X`` keep working unchanged.
# ---------------------------------------------------------------------------
from ._gravity_benchmark import (
    types as _gb_types,
    compute as _gb_compute,
    metrics as _gb_metrics,
    plotting as _gb_plotting,
    results_io as _gb_results_io,
    modes as _gb_modes,
)

for _gb_mod in (_gb_types, _gb_compute, _gb_results_io, _gb_plotting, _gb_metrics, _gb_modes):
    for _gb_name in vars(_gb_mod):
        if not _gb_name.startswith("__"):
            globals().setdefault(_gb_name, getattr(_gb_mod, _gb_name))
del _gb_mod, _gb_name


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lunar gravity model validation harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Config-driven reproducible benchmark mode ---
    p.add_argument("--config", type=str, default=None,
                   help="Benchmark config file. Preferred for reproducible benchmark runs.")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Config-mode override for surrogate.model_dir (alias for --st-lrps-model-dir in legacy mode).")
    p.add_argument("--out", type=str, default=None,
                   help="Config-mode output directory override.")
    p.add_argument("--scenario-count", type=int, default=None,
                   help="Config-mode override for scenario.count.")
    p.add_argument("--seed", type=int, default=None,
                   help="Config-mode override for scenario.seed.")
    p.add_argument("--dtype", choices=["float32", "float64"], default=None,
                   help="Config-mode override for propagation.dtype.")
    p.add_argument("--quick", action="store_true",
                   help="Run a lightweight synthetic benchmark that still writes manifest/config/validation outputs.")
    p.add_argument("--allow-validation-fail", action="store_true",
                   help="Return success even if standardized benchmark validation fails.")
    p.add_argument("--allow-contract-mismatch", action="store_true",
                   help="Config-mode: downgrade artifact/benchmark contract errors to warnings.")
    p.add_argument("--allow-domain-extrapolation", action="store_true",
                   help="Config-mode: allow benchmark altitude ranges outside artifact training envelope.")
    p.add_argument("--allow-legacy-artifact", action="store_true",
                   help="Config-mode: allow ST-LRPS artifacts without a full artifact_contract.")

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
    p.add_argument("--output-dir", type=str, default="outputs/gravity_benchmark")

    # --- Full GPU batch comparison ---
    p.add_argument("--gpu-batch-compare", action="store_true",
                   help="Compare GPU RK4 SH/ST-LRPS models against SH200 DOP853 truth")
    p.add_argument("--gpu-models", type=str, default="sh200,sh160,sh120,sh60,sh20,st_lrps",
                   help="Comma-separated GPU fixed-step model list")
    p.add_argument("--gpu-integrator", choices=list(GPU_INTEGRATORS), default="medium",
                   help="GPU fixed-step integrator tier: light (RK2 midpoint), "
                        "medium (classic RK4, default), or robust (RK4 + Richardson "
                        "extrapolation).")
    p.add_argument("--gpu-finite-check-mode",
                   choices=["step", "snapshot", "end", "off"],
                   default="snapshot",
                   help="When the GPU batch path scans for NaN/Inf state during "
                        "fixed-step propagation. step: after every RK step "
                        "(safest, highest overhead). snapshot: once per output "
                        "snapshot (recommended for benchmark speed; still catches "
                        "non-finite output before results are returned). end: "
                        "once over the full trajectory (minimal checking "
                        "overhead). off: skip the check in the hot loop (fastest "
                        "but does not detect invalid GPU states). snapshot/end "
                        "reduce CPU-side overhead in GPU batch propagation; the "
                        "actual speedup depends on model, batch size, device, OS, "
                        "and integrator settings. Numerical results are identical "
                        "across modes.")
    p.add_argument("--batch-frame-mode",
                   choices=["match_dynamics_engine", "inertial_fixed_legacy", "precomputed_slerp"],
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
    p.add_argument("--refresh-metadata", action="store_true",
                   help="Rebuild only the bookkeeping JSON (run_metadata.json + "
                        "gpu_batch_summary.json) from existing metrics CSVs and the cache "
                        "manifest. Does not import torch, propagate, or reload trajectories.")
    p.add_argument("--allow-stale-cache", action="store_true",
                   help="Downgrade cache-compatibility errors (mismatched config fields or "
                        "fingerprint) to warnings and reuse the cache anyway. Off by default: "
                        "a mismatched cache is refused so results stay scientifically valid.")
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

    return p.parse_args(argv)



# =============================================================================
# Entry point
# =============================================================================

def refresh_benchmark_metadata(args: argparse.Namespace) -> None:
    """Rebuild bookkeeping JSON from existing metrics + manifest (Fix 6).

    Re-emits ``run_metadata.json`` and ``gpu_batch_summary.json`` from the
    metrics CSVs and the cache manifest already on disk. It never imports torch,
    propagates, or reloads trajectory NPZ files, so numerical results are left
    byte-identical — only the lightweight metadata is regenerated.
    """
    out_dir = Path(args.output_dir)
    metrics_dir = out_dir / "metrics"
    cache_dir = _benchmark_cache_dir(args, out_dir)
    print(f"[refresh-metadata] Rebuilding bookkeeping for {out_dir}", flush=True)

    # Read the manifest first so we can restore the fingerprint-relevant model
    # dir (otherwise the recomputed fingerprint would drift from the run that
    # produced the cache). Never load trajectory NPZ files here.
    manifest: Dict[str, Any] = {}
    manifest_path = cache_dir / "cache_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[refresh-metadata] WARNING: unreadable cache manifest ({exc}); "
                  "continuing without it.", flush=True)
    md = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
    if md.get("st_lrps_model_dir") and not getattr(args, "st_lrps_model_dir", None):
        args.st_lrps_model_dir = md["st_lrps_model_dir"]

    # 1) run_metadata.json — skip the torch device probe (no imports on refresh).
    setattr(args, "_no_torch_probe", True)
    _write_run_metadata(args, out_dir)
    print(f"[refresh-metadata] Wrote {out_dir / 'run_metadata.json'}", flush=True)

    # 2) Existing GPU-batch metrics (string rows -> numeric where possible).
    aggregate_rows = [
        _coerce_numeric_row(r)
        for r in _read_csv_rows(metrics_dir / "gpu_batch_aggregate_metrics.csv")
    ]
    runtime_rows = [
        _coerce_numeric_row(r)
        for r in _read_csv_rows(metrics_dir / "gpu_batch_runtime_metrics.csv")
    ]
    if not aggregate_rows and not runtime_rows:
        print("[refresh-metadata] No GPU-batch metrics CSVs found; "
              "run_metadata.json refreshed only.", flush=True)
        return

    models_in_metrics = {str(r.get("model")) for r in aggregate_rows if r.get("model")}
    selected_models = list(manifest.get("selected_models", []) or [])
    if selected_models:
        requested_display = [_model_display_name(m) for m in selected_models]
    else:
        requested_display = sorted(models_in_metrics)
    status_by_model = {
        d: ("completed" if d in models_in_metrics else "skipped")
        for d in requested_display
    }

    # Scenario count: manifest first, else distinct ids in the per-scenario CSV.
    n_total = int(manifest.get("scenario_count", 0) or 0)
    if n_total <= 0:
        per_rows = _read_csv_rows(metrics_dir / "gpu_batch_per_scenario_metrics.csv")
        n_total = len({r.get("scenario_id") for r in per_rows if r.get("scenario_id")})

    # Truth runtime carried verbatim from the runtime CSV (already computed).
    truth_total = runtime_rows[0].get("truth_total_runtime_s") if runtime_rows else None
    truth_mean = (
        runtime_rows[0].get("truth_mean_runtime_per_scenario_s") if runtime_rows else None
    )

    equivalent = estimate_stlrps_equivalent_sh_degree(aggregate_rows)
    sel_path = metrics_dir / "stlrps_selected_scenarios.json"
    try:
        selected = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
    except Exception:
        selected = {}

    model_entries = {
        _model_display_name(m): {
            "cache_name": str(m),
            "status": status_by_model.get(_model_display_name(m), "skipped"),
            "in_metrics": _model_display_name(m) in models_in_metrics,
        }
        for m in selected_models
    }
    cache_provenance = _cache_provenance(
        args, cache_dir, enabled=bool(manifest),
        truth_counts={
            "requested": n_total,
            "loaded": None,
            "missing": None,
            "note": "counts not recomputed during --refresh-metadata",
        },
        model_entries=model_entries,
    )

    summary = _build_gpu_batch_summary(
        args,
        aggregate_rows=aggregate_rows,
        runtime_rows=runtime_rows,
        gpu_models=selected_models or sorted(models_in_metrics),
        requested_display=requested_display,
        status_by_model=status_by_model,
        n_scenarios_total=n_total,
        n_scenarios_new_this_run=0,
        truth_total_runtime_s=truth_total,
        truth_mean_runtime_per_scenario_s=truth_mean,
        equivalent=equivalent,
        selected=selected,
        cache_provenance=cache_provenance,
        rebuilt_from_cache=True,
        source="refresh",
        extra_warnings=[
            "Bookkeeping refreshed from existing metrics; trajectories were not recomputed."
        ],
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "gpu_batch_summary.json").write_text(
        json.dumps(summary, indent=4, default=str), encoding="utf-8"
    )
    print(f"[refresh-metadata] Wrote {metrics_dir / 'gpu_batch_summary.json'}", flush=True)

    cache_metrics_dir = cache_dir / "metrics"
    if manifest and cache_metrics_dir != metrics_dir:
        cache_metrics_dir.mkdir(parents=True, exist_ok=True)
        (cache_metrics_dir / "summary.json").write_text(
            json.dumps(summary, indent=4, default=str), encoding="utf-8"
        )
        print(f"[refresh-metadata] Wrote {cache_metrics_dir / 'summary.json'}", flush=True)


def _apply_common_aliases(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "model_dir", None) and not getattr(args, "st_lrps_model_dir", None):
        args.st_lrps_model_dir = args.model_dir
    if getattr(args, "out", None):
        args.output_dir = args.out
    if getattr(args, "scenario_count", None) is not None:
        args.random_scenarios = int(args.scenario_count)
    if getattr(args, "seed", None) is not None:
        args.scenario_seed = int(args.seed)
    if getattr(args, "dtype", None):
        args.torch_dtype = str(args.dtype)
    if getattr(args, "quick", False):
        args.random_scenarios = min(int(args.random_scenarios), 3)
        args.duration_days = min(float(args.duration_days), 0.01)
    return args


def run_from_args(args: argparse.Namespace) -> int:
    args = _apply_common_aliases(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "refresh_metadata", False):
        refresh_benchmark_metadata(args)
        return 0

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
        if getattr(args, "rebuild_metrics", False):
            print("[auto] Skipping ST-LRPS model dir resolution since --rebuild-metrics is active.", flush=True)
        else:
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
                if bool(args.gpu_batch_compare):
                    args.gpu_models = ",".join([m for m in gpu_models_raw if m != "st_lrps"])
                else:
                    args.models = ",".join([m for m in models_raw if m != "st_lrps"])

    cfg   = build_base_config(args)
    _write_run_metadata(args, out_dir)
    ephem = EphemerisManager.from_time_and_spice(cfg.time, cfg.spice)

    if args.gpu_batch_compare:
        run_gpu_batch_compare_mode(args, cfg, ephem)
        return 0

    if args.force_sample_trajectory:
        models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        truth  = args.force_sample_trajectory.strip().lower()
        if truth not in models:
            models.append(truth)
        evaluate_forces(models, truth, args, cfg, ephem, out_dir)
        return 0

    if args.random_scenarios > 0:
        run_random_scenario_mode(args, cfg, ephem)
    else:
        run_single_orbit_mode(args, cfg, ephem)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    if args.config:
        explicit_out = args.out
        if explicit_out is None and "--output-dir" in raw_argv:
            explicit_out = args.output_dir
        from .benchmark_pipeline import run_configured_benchmark

        return run_configured_benchmark(
            args.config,
            out_dir=explicit_out,
            model_dir=args.model_dir or args.st_lrps_model_dir,
            scenario_count=args.scenario_count,
            seed=args.seed,
            dtype=args.dtype,
            quick=bool(args.quick),
            allow_validation_fail=bool(args.allow_validation_fail),
            allow_contract_mismatch=bool(args.allow_contract_mismatch),
            allow_domain_extrapolation=bool(args.allow_domain_extrapolation),
            allow_legacy_artifact=bool(args.allow_legacy_artifact),
        )
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
