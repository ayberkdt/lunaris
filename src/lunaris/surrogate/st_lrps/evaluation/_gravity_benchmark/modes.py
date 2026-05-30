# -*- coding: utf-8 -*-
"""Internal module of the lunar gravity-model benchmark harness.

Part of :mod:`lunaris.surrogate.st_lrps.evaluation.compare_gravity_models`;
this is an implementation detail, not a public API. See that module's
docstring for CLI usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lunaris.core.config import SimConfig
from lunaris.core.state import create_state_from_keplerian
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.propagator import propagate
from lunaris.physics.surrogate_gravity import (
    find_latest_st_lrps_model_dir,
)
from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.surrogate.st_lrps.evaluation import progress

# --- intra-package wiring (auto-generated split) ---
from .types import (
    BatchModelResult,
    GravityModelCache,
    Scenario,
    TruthTrajectorySet,
    _METRICS_FIELDNAMES,
    _cfg_with_integrator,
    _find_st_lrps_weight_file,
    decompose_vector_ric,
    interpolate_state_to_times,
)
from .compute import (
    _build_gpu_batch_tasks,
    _gpu_integrator_evals_per_step,
    _gpu_rk4_dt_values,
    _parse_model_list_csv,
    _synchronize_model_device_if_cuda,
    _torch_dtype_from_name,
    evaluate_st_lrps_forces_batched,
    propagate_for_scenario,
    propagate_gpu_batch_model,
    run_sh200_cpu_rk4_reference,
    run_st_lrps_batch_rk4,
)
from .results_io import (
    _PARALLEL_STATE,
    _append_metrics_csv,
    _benchmark_cache_dir,
    _build_gpu_batch_summary,
    _cache_provenance,
    _cache_requested,
    _cached_model_path,
    _cached_truth_path,
    _coerce_numeric_row,
    _ensure_dir,
    _load_cached_trajectory,
    _model_cache_completion,
    _parallel_worker_init,
    _read_csv_rows,
    _save_cached_trajectory,
    _truth_cache_available,
    _truth_cache_completion,
    _truth_cache_name,
    _validate_cache_compatibility,
    _write_cache_manifest,
    _write_csv,
    _write_run_metadata,
    build_truth_trajectory_set,
    prepare_scenarios,
)
from .plotting import (
    _color,
    _fmt_km,
    estimate_stlrps_equivalent_sh_degree,
    plot_aggregate_stats,
    plot_batch_rk4_results,
    plot_batch_selected_scenario,
    plot_gpu_batch_report_figures,
    plot_selected_scenario,
    select_stlrps_scenarios,
    write_gpu_batch_report_pdf,
    write_report_pdf,
)
from .metrics import (
    _batch_agg_stats,
    aggregate_gpu_batch_metrics,
    aggregate_metrics,
    build_gpu_model_ranking,
    build_gpu_runtime_metrics,
    build_rankings,
    compute_batch_rk4_metrics,
    compute_gpu_batch_metrics_for_model,
    compute_trajectory_metrics,
    find_worst_cases,
    rebuild_gpu_batch_metrics_from_cache,
    select_median_difficulty_scenario,
)

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


def _load_written_summary(metrics_dir: Path) -> Optional[Dict[str, Any]]:
    """Best-effort read of the just-written gpu_batch_summary.json (for stdout)."""
    p = metrics_dir / "gpu_batch_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _print_gpu_batch_summary(
    args: argparse.Namespace,
    aggregate_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    equivalent: Dict[str, Any],
    selected: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
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

    # Honest scope (Fix 8): never imply "all models completed" unless true.
    if summary:
        scope = str(summary.get("summary_scope", ""))
        note = str(summary.get("summary_note", ""))
        if scope == "completed_models_only":
            print(f"Scope:    completed models only -- {note}")
        elif note:
            print(f"Scope:    {note}")
        for label in ("failed_models", "partial_models", "skipped_models"):
            names = summary.get(label) or []
            if names:
                print(f"  {label.replace('_', ' ')}: {', '.join(map(str, names))}")
        for warn in (summary.get("metadata_warnings") or []):
            print(f"  [warn] {warn}")

    print("\nAccuracy ranking:")
    print(f"{'Model':<22} {'Median RMS km':>14} {'P95 RMS km':>12} "
          f"{'Max RMS km':>12} {'Median Along km':>17}")
    for r in aggregate_rows:
        print(f"{r['model']:<22} {_fmt_km(r.get('median_rms_pos_err_km', np.nan)):>14} "
              f"{_fmt_km(r.get('p95_rms_pos_err_km', np.nan)):>12} "
              f"{_fmt_km(r.get('max_rms_pos_err_km', np.nan)):>12} "
              f"{_fmt_km(r.get('median_along_rms_km', np.nan)):>17}")

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
        print(f"- ST-LRPS median RMS = {_fmt_km(st_row.get('median_rms_pos_err_km', np.nan))} km")
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
    if not getattr(args, "rebuild_metrics", False):
        _write_run_metadata(args, out_dir, scenarios)
    progress.emit_progress(
        "scenario", current=len(scenarios), total=len(scenarios),
        percent=100.0, message="Scenarios ready",
    )

    gpu_models = _parse_model_list_csv(args.gpu_models)
    if args.truth.lower() not in {"sh200"}:
        print(f"[gpu-batch] WARNING: requested truth={args.truth}; expected sh200 for this workflow.",
              flush=True)

    if "st_lrps" in gpu_models and not getattr(args, "st_lrps_model_dir", None):
        if getattr(args, "rebuild_metrics", False):
            print("[gpu-batch] --rebuild-metrics is active; skipping ST-LRPS model dir lookup.", flush=True)
        else:
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
    cache_warnings: List[str] = []
    if cache_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[cache] Enabled trajectory cache: {cache_dir}", flush=True)
        cache_warnings = _validate_cache_compatibility(args, cache_dir)
        for _w in cache_warnings:
            print(f"[cache] WARNING: {_w}", flush=True)
        if not getattr(args, "rebuild_metrics", False):
            _write_cache_manifest(args, cache_dir, scenarios, gpu_cache_names)

    if getattr(args, "rebuild_metrics", False):
        aggregate_rows, runtime_rows, equivalent, selected = rebuild_gpu_batch_metrics_from_cache(
            args, scenarios, cache_dir, gpu_cache_names, metrics_dir, plots_dir, reports_dir
        )
        _print_gpu_batch_summary(args, aggregate_rows, runtime_rows, equivalent, selected,
                                 _load_written_summary(metrics_dir))
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

        # Per-model windowed-rate state (fresh dict captured per model iteration).
        cb_state = {"step": 0, "t": 0.0}

        def _gpu_progress_cb(
            current_step: int, total_steps: int, elapsed_s: float,
            _name: str = task.display_name, _idx: int = model_idx,
            _state: Dict[str, float] = cb_state,
        ) -> None:
            stats = progress.compute_step_stats(current_step, total_steps, elapsed_s)
            # steps/s over the most recent window (ignores one-off warmup such as
            # JIT compilation / first CUDA launch) so the rate and ETA are honest.
            rate = progress.windowed_rate(
                current_step - int(_state["step"]),
                elapsed_s - float(_state["t"]),
                fallback_cur=current_step, fallback_elapsed=elapsed_s,
            )
            _state["step"], _state["t"] = current_step, elapsed_s
            remaining_steps = max(0, stats["total_steps"] - stats["current_step"])
            model_eta = progress.eta_from_rate(remaining_steps, rate)
            print(
                f"[gpu-batch][{_name}] step {stats['current_step']}/{stats['total_steps']} "
                f"| {stats['percent']:.1f}% | elapsed {progress.format_duration(elapsed_s)} "
                f"| ETA {progress.format_eta(model_eta)} "
                f"| {rate:.1f} steps/s",
                flush=True,
            )
            progress.emit_progress(
                "gpu_model", model=_name,
                current_step=stats["current_step"], total_steps=stats["total_steps"],
                percent=stats["percent"], elapsed_s=elapsed_s,
                eta_s=model_eta, steps_per_s=rate,
                device=str(device), dtype=str(args.torch_dtype),
                n_scenarios=len(model_scenarios),
            )
            model_frac = stats["current_step"] / max(1, stats["total_steps"])
            gpu_frac = ((_idx - 1) + model_frac) / n_gpu_models
            ov = overall.update("gpu", gpu_frac)
            # Honest, time-based overall ETA: precise remaining-of-current-model
            # plus the average wall time of finished models for the not-yet-started
            # ones. Avoids the phase-weight-vs-wall-time mismatch of a linear
            # extrapolation on the weighted bar percentage.
            elapsed_gpu = time.perf_counter() - t_gpu_start
            completed_models = _idx - 1
            future: Optional[float] = None
            if completed_models >= 1:
                avg_done = max(0.0, elapsed_gpu - elapsed_s) / completed_models
                future = (n_gpu_models - _idx) * avg_done
            elif model_frac > 1e-3:
                future = (n_gpu_models - _idx) * (elapsed_s / model_frac)
            overall_eta = None
            if model_eta is not None or future is not None:
                overall_eta = (model_eta or 0.0) + (future or 0.0)
            progress.emit_progress_total(
                ov, "gpu_model", model=_name,
                elapsed_s=overall.elapsed_s(), eta_s=overall_eta,
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
                finite_check_mode=str(getattr(args, "gpu_finite_check_mode", "snapshot")),
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
        live_failed = sorted({r.display_name for r in results if getattr(r, "status", "") == "failed"})
        run_context = {
            "source": "live",
            "rebuilt_from_cache": True,
            "n_scenarios_new_this_run": len(run_scenarios),
            "truth_generated_this_run": len(run_scenarios),
            "failed_models": live_failed,
            "extra_warnings": cache_warnings,
        }
        aggregate_rows, runtime_rows, equivalent, selected = rebuild_gpu_batch_metrics_from_cache(
            args, scenarios, cache_dir, gpu_cache_names, metrics_dir, plots_dir, reports_dir,
            run_context=run_context,
        )
        _print_gpu_batch_summary(args, aggregate_rows, runtime_rows, equivalent, selected,
                                 _load_written_summary(metrics_dir))
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
    runtime_rows = build_gpu_runtime_metrics(
        results, truth,
        evals_per_step=_gpu_integrator_evals_per_step(getattr(args, "gpu_integrator", "medium")),
    )
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
    result_status = {r.display_name: getattr(r, "status", "") for r in results}
    models_in_agg = {str(r.get("model")) for r in aggregate_rows}
    status_by_model: Dict[str, str] = {}
    for task in gpu_tasks:
        disp = task.display_name
        st = result_status.get(disp)
        if st == "ok":
            status_by_model[disp] = "completed"
        elif st == "failed":
            status_by_model[disp] = "failed"
        elif disp in models_in_agg:
            # resume: no new compute this run, metrics carried from stored rows.
            status_by_model[disp] = "completed"
        else:
            status_by_model[disp] = "skipped"
    cache_provenance = _cache_provenance(
        args, cache_dir, enabled=False,
        truth_counts={
            "requested": len(scenarios),
            "loaded": 0,
            "missing": 0,
            "generated_this_run": len(run_scenarios),
        },
        model_entries={},
    )
    summary = _build_gpu_batch_summary(
        args,
        aggregate_rows=aggregate_rows,
        runtime_rows=runtime_rows,
        gpu_models=gpu_models,
        requested_display=[task.display_name for task in gpu_tasks],
        status_by_model=status_by_model,
        n_scenarios_total=len(scenarios),
        n_scenarios_new_this_run=len(run_scenarios),
        truth_total_runtime_s=truth.total_runtime_s,
        truth_mean_runtime_per_scenario_s=truth.mean_runtime_s,
        equivalent=equivalent,
        selected=selected,
        cache_provenance=cache_provenance,
        rebuilt_from_cache=False,
        source="live",
        extra_warnings=cache_warnings,
    )
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
    _print_gpu_batch_summary(args, aggregate_rows, runtime_rows, equivalent, selected, summary)

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
        for _w in _validate_cache_compatibility(args, cache_dir):
            print(f"[cache] WARNING: {_w}", flush=True)
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
