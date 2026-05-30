# -*- coding: utf-8 -*-
"""Internal module of the lunar gravity-model benchmark harness.

Part of :mod:`lunaris.surrogate.st_lrps.evaluation.compare_gravity_models`;
this is an implementation detail, not a public API. See that module's
docstring for CLI usage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lunaris.common.constants import R_MOON

# --- intra-package wiring (auto-generated split) ---
from .types import (
    BatchModelResult,
    CachedTrajectory,
    Scenario,
    TruthTrajectorySet,
    _BATCH_METRICS_FIELDNAMES,
    _GPU_BATCH_METRICS_FIELDNAMES,
    _METRICS_FIELDNAMES,
    compute_ric_errors,
    interpolate_state_to_times,
)
from .compute import (
    _model_display_name,
)
from .results_io import (
    _build_gpu_batch_summary,
    _cache_provenance,
    _cached_model_path,
    _cached_truth_path,
    _load_cached_trajectory,
    _model_cache_completion,
    _truth_cache_completion,
    _truth_cache_name,
    _write_csv,
)
from .plotting import (
    estimate_stlrps_equivalent_sh_degree,
    plot_gpu_batch_report_figures,
    select_stlrps_scenarios,
    write_gpu_batch_report_pdf,
)

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
            # ``steps`` is already the total trajectory-step count summed over all
            # scenarios, so throughput is steps/runtime (do NOT multiply by n again).
            "trajectory_steps_per_second": steps / max(runtime, 1e-9),
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
    *,
    run_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    print("[cache] Rebuilding metrics from cached trajectories.", flush=True)
    manifest_path = cache_dir / "cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_duration = manifest.get("metadata", {}).get("duration_days")
        if cache_duration is not None:
            args.duration_days = float(cache_duration)
            print(f"[cache] Overriding duration_days to {args.duration_days} from cache manifest.", flush=True)

    truth = load_cached_truth_set(args, scenarios, cache_dir, strict=True)
    if truth.t_by_scenario:
        first_scen = next(iter(truth.t_by_scenario.values()))
        if len(first_scen) > 0:
            args.duration_days = float(first_scen[-1]) / 86400.0
            print(f"[cache] Recovered true duration_days = {args.duration_days} from trajectory arrays.", flush=True)

    run_ctx: Dict[str, Any] = dict(run_context or {})
    failed_models = {str(x) for x in run_ctx.get("failed_models", [])}

    all_rows: List[Dict[str, Any]] = []
    model_entries: Dict[str, Dict[str, Any]] = {}
    status_by_model: Dict[str, str] = {}
    for model in gpu_models:
        complete, missing = _model_cache_completion(cache_dir, model, scenarios)
        print(f"[cache] Model {model}: {complete}/{len(scenarios)} complete.", flush=True)
        if missing and getattr(args, "strict_complete", False):
            raise RuntimeError(
                f"Model {model} is missing {len(missing)} cached scenario trajectories."
            )
        disp = _model_display_name(model)
        sample_meta: Dict[str, Any] = {}
        for scenario in scenarios:
            cached = _load_cached_trajectory(_cached_model_path(cache_dir, model, scenario.scenario_id))
            if cached is None:
                continue
            if not sample_meta:
                sample_meta = dict(cached.metadata)
            result = BatchModelResult(
                model_name=model,
                display_name=disp,
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
        total = len(scenarios)
        if total > 0 and complete == total:
            status = "completed"
        elif complete == 0:
            status = "failed" if disp in failed_models else "skipped"
        else:
            status = "partial"
        status_by_model[disp] = status
        model_entries[disp] = {
            "cache_name": str(model),
            "backend": sample_meta.get("backend"),
            "integrator": sample_meta.get("gpu_integrator") or sample_meta.get("integrator"),
            "rk4_dt_s": sample_meta.get("rk4_dt_s"),
            "output_dt_s": sample_meta.get("dt_out"),
            "dtype": sample_meta.get("dtype"),
            "device": sample_meta.get("device"),
            "requested": total,
            "loaded": int(complete),
            "missing": int(total - complete),
            "status": status,
        }

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
    truth_complete, truth_missing = _truth_cache_completion(cache_dir, args, scenarios)
    truth_counts = {
        "requested": len(scenarios),
        "loaded": int(truth_complete),
        "missing": int(len(truth_missing)),
        "generated_this_run": int(run_ctx.get("truth_generated_this_run", 0) or 0),
    }
    cache_provenance = _cache_provenance(
        args, cache_dir, enabled=True,
        truth_counts=truth_counts, model_entries=model_entries,
    )
    summary = _build_gpu_batch_summary(
        args,
        aggregate_rows=aggregate_rows,
        runtime_rows=runtime_rows,
        gpu_models=gpu_models,
        requested_display=[_model_display_name(m) for m in gpu_models],
        status_by_model=status_by_model,
        n_scenarios_total=len(scenarios),
        n_scenarios_new_this_run=int(run_ctx.get("n_scenarios_new_this_run", 0) or 0),
        truth_total_runtime_s=truth.total_runtime_s,
        truth_mean_runtime_per_scenario_s=truth.mean_runtime_s,
        equivalent=equivalent,
        selected=selected,
        cache_provenance=cache_provenance,
        rebuilt_from_cache=bool(run_ctx.get("rebuilt_from_cache", True)),
        source=str(run_ctx.get("source", "rebuild_from_cache")),
        extra_warnings=run_ctx.get("extra_warnings"),
    )
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
    evals_per_step: int = 4,
) -> List[Dict[str, Any]]:
    """Build per-model runtime and speedup rows.

    ``evals_per_step`` is the RHS (acceleration) evaluations per output step for
    the active GPU integrator (light=2, medium=4, robust=12); it scales the
    acceleration-evaluation throughput so that figure is not mis-reported.
    """

    base_rows: List[Dict[str, Any]] = []
    by_model: Dict[str, Dict[str, Any]] = {}
    truth_total = truth.total_runtime_s
    truth_mean = truth.mean_runtime_s
    evals = max(1, int(evals_per_step))
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
            "acceleration_evaluations_per_second": n_scenarios * n_steps * evals / max(runtime, 1e-9),
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
