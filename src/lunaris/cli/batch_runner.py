#!/usr/bin/env python
"""
batch_runner.py — CLI entry point for batch/ensemble propagation.

This script mirrors main.py's orbit/physics/timeline argument interface and
adds batch uncertainty-propagation flags.  It is invoked by the GUI as a subprocess
so that progress can be streamed line-by-line and the main application stays
responsive.

Progress lines (stdout, consumed by the UI)
-------------------------------------------
    [BATCH] N=500  backend=GPU  T=1.00 d  step=60.0 s  snap=600.0 s
    [BATCH_PROGRESS] {"stage": "propagating", "percent": 42.5, ...}
    [BATCH] Batch 1/5  samples 0-99
    [BATCH] Batch 2/5  samples 100-199
    ...
    [BATCH] Done. Wall=42.3s  impacts=3/500 (0.6%)
    [BATCH_METRICS] {"n_samples": 500, "n_impacts": 3, ...}

Exit codes
----------
    0 — success
    1 — configuration / validation error
    2 — runtime error
    3 — requested UQ report generation failed
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared, pure orbit/physics CLI helpers live in cli.common_args (import-safe;
# no dependency on the sibling main.py entry point).
# ---------------------------------------------------------------------------
from lunaris.cli.common_args import (  # noqa: E402
    apply_args_to_config,
    init_surface_provider,
    parse_adaptive_table,
    resolve_orbit_elements,
    str2bool,
)
from lunaris.common.batch_defs import (  # noqa: E402
    BATCH_SAMPLING_METHODS,
    BatchPropagationConfig,
    BatchPropagationResult,
    SpacecraftUncertainty,
    StateUncertainty,
)
from lunaris.common.constants import DAY_S, DEG2RAD, MU_MOON, R_MOON  # noqa: E402
from lunaris.core.config import load_default_config, replace_sim_config  # noqa: E402

# =============================================================================
# 1.                            ARGUMENT PARSER
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Return the combined sim + batch argument parser."""
    p = argparse.ArgumentParser(
        description="Lunaris batch/ensemble propagation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Time ---------------------------------------------------------------
    g = p.add_argument_group("Time")
    g.add_argument("--start-date", type=str,
                   help=(
                       "UTC start date. Naive timestamps are interpreted as UTC; "
                       "explicit offsets are accepted and normalized to UTC."
                   ))
    dur = g.add_mutually_exclusive_group()
    dur.add_argument("--days",  type=float, help="Simulation duration [days]")
    dur.add_argument("--hours", type=float, help="Simulation duration [hours]")
    g.add_argument("--output-dt-s",        type=float, help="Output spacing [s]")
    g.add_argument("--samples-per-period", type=int)

    # ---- Orbit init ---------------------------------------------------------
    g = p.add_argument_group("Orbit Init (choose one)")
    g.add_argument("--hp-km",   type=float)
    g.add_argument("--ha-km",   type=float)
    g.add_argument("--a-km",    type=float)
    g.add_argument("--e",       type=float)
    g.add_argument("--alt-km",  type=float)
    g.add_argument("--inc-deg", type=float)
    g.add_argument("--raan-deg", type=float)
    g.add_argument("--argp-deg", type=float)
    g.add_argument("--ta-deg",   type=float)

    # ---- Physics ------------------------------------------------------------
    g = p.add_argument_group("Physics Flags")
    g.add_argument("--enable-sh",              type=str2bool)
    g.add_argument("--enable-3rd-body-sun",    type=str2bool)
    g.add_argument("--enable-3rd-body-earth",  type=str2bool)
    g.add_argument("--enable-earth-j2",        type=str2bool)
    g.add_argument("--enable-srp",             type=str2bool)
    g.add_argument("--enable-albedo",          type=str2bool)
    g.add_argument("--enable-thermal",         type=str2bool)
    g.add_argument("--enable-tides",           type=str2bool)
    g.add_argument("--tides-kind",             choices=("k2", "k3"))
    g.add_argument("--enable-relativity-1pn",  type=str2bool)

    # ---- Gravity ------------------------------------------------------------
    g = p.add_argument_group("Gravity Model")
    g.add_argument("--gravity-backend",      choices=("classic_sh", "st_lrps"))
    g.add_argument("--gravity-file-path", type=str)
    g.add_argument("--surrogate-gravity-model-dir", type=str)
    g.add_argument("--degree",            type=int)
    g.add_argument("--adaptive-enabled",  type=str2bool)
    g.add_argument("--adaptive-table",    type=parse_adaptive_table)

    # ---- Spacecraft ---------------------------------------------------------
    g = p.add_argument_group("Spacecraft")
    g.add_argument("--mass-kg",  type=float)
    g.add_argument("--area-m2",  type=float)
    g.add_argument("--cd",       type=float)
    g.add_argument("--cr",       type=float)

    # ---- I/O & Assets -------------------------------------------------------
    g = p.add_argument_group("I/O & Assets")
    g.add_argument("--kernel-dir",   type=str)
    g.add_argument("--ldem-root",    type=str)
    g.add_argument("--albedo-root",  type=str)
    g.add_argument("--ldem-ppd",     type=int)
    # Accepted but unused in batch path (orbit output goes to batch-output-path)
    g.add_argument("--out-dir",        type=str)
    g.add_argument("--make-3d-plots",  type=str2bool)
    g.add_argument("--downsample-3d",  type=int)

    # ---- Numerics (CPU propagator compat) -----------------------------------
    g = p.add_argument_group("Numerics (CPU propagator)")
    g.add_argument("--method",           type=str)
    g.add_argument("--user-max-step-s",  type=float)
    g.add_argument("--rtol",             type=float)
    g.add_argument("--atol",             type=float)

    # ---- Batch / ensemble ---------------------------------------------------
    g = p.add_argument_group("Batch / ensemble")
    g.add_argument("--n-samples",             type=int,   default=500,
                   help="Number of ensemble trajectories (>= 2)")
    g.add_argument(
        "--sampling-method",
        choices=BATCH_SAMPLING_METHODS,
        default="random",
        help=(
            "Ensemble sampling design. 'random' is the classical Monte Carlo "
            "option; 'lhs' and Sobol variants are space-filling designs for "
            "validation and benchmark coverage."
        ),
    )
    g.add_argument("--seed",                  type=int,   default=42,
                   help="RNG seed for reproducibility")
    g.add_argument("--sigma-r-m",             type=float, default=500.0,
                   help="Position 1-sigma [m]")
    g.add_argument("--sigma-v-m-s",           type=float, default=0.5,
                   help="Velocity 1-sigma [m/s]")
    g.add_argument("--sigma-mass-kg",         type=float, default=0.0)
    g.add_argument("--sigma-area-m2",         type=float, default=0.0)
    g.add_argument("--sigma-cd",              type=float, default=0.0)
    g.add_argument("--sigma-cr",              type=float, default=0.0)
    g.add_argument("--use-gpu",               type=str2bool, default=True,
                   help="Use CUDA RK4 propagator (on/off)")
    g.add_argument(
        "--batch-backend",
        choices=[
            "auto", "cpu_sh", "gpu_sh", "numba_cuda_sh", "torch_cuda_sh",
            "torch_cpu_sh", "gpu_st_lrps_potential", "gpu_st_lrps_direct",
        ],
        default="auto",
        help=(
            "Explicit batch propagation backend. 'auto' preserves use-gpu + gravity-mode "
            "routing. 'numba_cuda_sh' (alias 'gpu_sh') is the degree<=24 Numba CUDA "
            "screening kernel; 'torch_cuda_sh' is the high-degree PyTorch CUDA "
            "classic-SH path (gravity-only). Requested SH degree is never clipped."
        ),
    )
    g.add_argument(
        "--gpu-sh-fallback-policy",
        choices=["compatible_gpu", "cpu", "error"],
        default="compatible_gpu",
        help=(
            "When an explicit numba_cuda_sh request exceeds degree 24: "
            "'compatible_gpu' tries torch_cuda_sh then CPU, 'cpu' forces CPU, "
            "'error' raises instead of substituting. Degree is never clipped."
        ),
    )
    g.add_argument("--torch-dtype", choices=["float32", "float64"], default="float64",
                   help="Floating-point dtype for the torch_cuda_sh classic-SH path.")
    g.add_argument("--torch-sh-chunk-size", type=int, default=0,
                   help="Samples per GPU chunk on the torch_cuda_sh path (0 = auto/VRAM-aware).")
    g.add_argument("--gpu-device-id",         type=int,   default=0)
    g.add_argument("--gpu-sh-degree",         type=int,   default=10,
                   help="Requested SH degree. numba_cuda_sh supports degree <=24; higher degrees use torch_cuda_sh (PyTorch CUDA) or fall back explicitly. Never clipped.")
    g.add_argument("--gpu-threads-per-block", type=int,   default=128)
    g.add_argument(
        "--batch-gravity-mode",
        choices=["follow_mission", "classic_sh", "st_lrps"],
        default="follow_mission",
        help="Whether batch propagation follows the mission gravity setup or forces classical/ST-LRPS gravity.",
    )
    g.add_argument("--batch-dt-s",               type=float, default=60.0,
                   help="Fixed RK4 step size [s]")
    g.add_argument("--max-vram-gb",           type=float, default=4.0)
    g.add_argument("--batch-output-format",      choices=["hdf5", "npz"],
                   default="hdf5")
    g.add_argument("--batch-output-path",        type=str,
                   default="outputs/ensemble/batch_output.h5")
    g.add_argument(
        "--result-storage-mode",
        choices=["auto", "memory", "disk"],
        default="auto",
    )
    g.add_argument("--max-result-memory-gb", type=float, default=1.0)
    g.add_argument("--detect-impact", type=str2bool, default=True)
    g.add_argument("--compute-impact-statistics", type=str2bool, default=True)
    g.add_argument("--impact-alt-km",         type=float, default=0.0,
                   help="Impact detection threshold altitude [km]")
    g.add_argument("--uq-report-dir",         type=str, default=None,
                   help="Write a provenance-stamped UQ report (covariance history, "
                        "RIC sigmas, error-ellipsoid figures, manifest) to this directory "
                        "after the run. See docs/UQ_COVARIANCE.md.")

    return p


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(args=argv)


# =============================================================================
# 2.                           METRICS HELPERS
# =============================================================================

def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% confidence interval for impact probability."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _build_metrics(result: BatchPropagationResult, wall_time_s: float, batch_cfg: BatchPropagationConfig) -> dict:
    """Extract summary statistics from an BatchPropagationResult for the UI metrics panel."""
    import numpy as np

    Y          = result.Y            # (T, N, 6)
    impact     = result.impact_mask  # (N,)
    t_impact   = result.t_impact     # (N,)

    N = int(Y.shape[1])
    valid = result.valid_sample_mask()
    n_valid = int(np.sum(valid))
    n_hit = int(np.sum(valid & (impact > 0.5)))
    p_imp = n_hit / n_valid if n_valid > 0 else math.nan

    ci_lo, ci_hi = _wilson_ci(n_hit, n_valid)

    # Mean impact time [days]
    hit_times = t_impact[np.isfinite(t_impact) & valid & (impact > 0.5)]
    t_imp_mean_days = float(np.mean(hit_times) / DAY_S) if hit_times.size > 0 else None

    # Honor the compute_impact_statistics contract. When enabled, run the
    # canonical analysis-layer statistics (Wilson CI + geographic impact-site
    # availability) and surface their outputs; when disabled, skip the
    # computation entirely and record *why*, so the flag is observable rather
    # than inert provenance. (The richer statistics live in the analysis layer,
    # which the headless runner — an entry point — is allowed to import.)
    if batch_cfg.impact_statistics_enabled:
        try:
            from lunaris.analysis.ensemble.statistics import compute_impact_statistics

            imp = compute_impact_statistics(result)
            p_imp = float(imp.p_impact)
            ci_lo, ci_hi = float(imp.p_impact_ci95[0]), float(imp.p_impact_ci95[1])
            t_imp_mean_days = (
                float(imp.t_impact_mean / DAY_S)
                if math.isfinite(imp.t_impact_mean)
                else None
            )
            impact_stats_block: dict = {
                "impact_statistics_computed": True,
                "impact_geographic_available": bool(imp.lat_deg.size > 0),
                "n_impact_sites": int(imp.lat_deg.size),
            }
        except Exception as exc:
            impact_stats_block = {
                "impact_statistics_computed": False,
                "impact_statistics_skip_reason": f"computation failed: {exc}",
            }
    else:
        impact_stats_block = {
            "impact_statistics_computed": False,
            "impact_statistics_skip_reason": "compute_impact_statistics disabled in config",
        }

    # Altitude at t=0 and t=-1
    def _alt_stats(step: int):
        if n_valid < 1:
            raise ValueError("altitude metrics require at least one valid batch sample")
        r = np.linalg.norm(Y[step, np.where(valid)[0], :3], axis=1)
        alt_km = (r - float(R_MOON)) / 1000.0
        return float(np.mean(alt_km)), float(np.std(alt_km))

    alt_mean_0, alt_std_0 = _alt_stats(0)
    alt_mean_f, alt_std_f = _alt_stats(-1)

    diagnostics = getattr(result, "diagnostics", {}) or {}
    backend_name = str(diagnostics.get("backend", "GPU" if batch_cfg.use_gpu else "CPU"))
    backend_note = str(diagnostics.get("backend_note", "") or "")
    backend_diag = diagnostics.get("backend_diagnostics", {}) or {}

    return {
        "n_samples":        N,
        "sampling_method":  str(getattr(batch_cfg, "sampling_method", "random")),
        "sampling_note":    str(diagnostics.get("sampling_note", "") or ""),
        "n_valid_samples":  n_valid,
        "n_impacts":        n_hit,
        "p_impact":         p_imp,
        "p_impact_ci95":    [round(ci_lo, 6), round(ci_hi, 6)],
        "t_impact_mean_days": t_imp_mean_days,
        "alt_mean_0_km":    round(alt_mean_0, 3),
        "alt_std_0_km":     round(alt_std_0, 4),
        "alt_mean_f_km":    round(alt_mean_f, 3),
        "alt_std_f_km":     round(alt_std_f, 4),
        "wall_time_s":      round(wall_time_s, 2),
        "backend":          backend_name,
        "backend_note":     backend_note,
        "actual_batch_backend": diagnostics.get("actual_batch_backend"),
        "requested_batch_backend": diagnostics.get("requested_batch_backend"),
        "runtime_model_kind": diagnostics.get("runtime_model_kind"),
        "fallback_reason": diagnostics.get("fallback_reason"),
        "requested_sh_degree": diagnostics.get("requested_sh_degree"),
        "actual_sh_degree": diagnostics.get("actual_sh_degree"),
        "device_name":      str(backend_diag.get("device_name", "") or ""),
        "threads_per_block": backend_diag.get("threads_per_block"),
        **impact_stats_block,
        "output_path":      str(batch_cfg.output_path_resolved),
    }


def _emit_progress_line(payload: dict) -> None:
    """
    Stream one structured batch/ensemble progress update to stdout.

    The desktop UI treats ``[BATCH_PROGRESS]`` as a machine-readable control line
    rather than as human log text.  Keeping this emission centralized ensures
    every backend phase uses the same JSON envelope.
    """

    print(f"[BATCH_PROGRESS] {json.dumps(payload)}", flush=True)


# =============================================================================
# 3.                                 MAIN
# =============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # ---- Build SimConfig ----------------------------------------------------
    try:
        cfg = load_default_config()
        cfg = apply_args_to_config(cfg, args)
    except Exception as exc:
        print(f"[BATCH][FATAL] Config init failed: {exc}", flush=True)
        return 1

    if str(args.batch_gravity_mode) != "follow_mission":
        try:
            from dataclasses import replace
            forced_backend = str(args.batch_gravity_mode)
            cfg = replace_sim_config(
                cfg,
                gravity=replace(cfg.gravity, backend=forced_backend),
                flags=replace(cfg.flags, enable_sh=True),
            )
        except Exception as exc:
            print(f"[BATCH][FATAL] Gravity override failed: {exc}", flush=True)
            return 1

    if str(args.batch_backend) != "auto":
        try:
            from dataclasses import replace

            explicit_backend = str(args.batch_backend)
            if explicit_backend in {"cpu_sh", "gpu_sh", "numba_cuda_sh", "torch_cuda_sh", "torch_cpu_sh"}:
                forced_backend = "classic_sh"
            elif explicit_backend in {"gpu_st_lrps_potential", "gpu_st_lrps_direct"}:
                forced_backend = "st_lrps"
            else:
                forced_backend = str(getattr(cfg.gravity, "backend", "classic_sh"))
            cfg = replace_sim_config(
                cfg,
                gravity=replace(cfg.gravity, backend=forced_backend),
                flags=replace(cfg.flags, enable_sh=True),
            )
        except Exception as exc:
            print(f"[BATCH][FATAL] batch backend gravity override failed: {exc}", flush=True)
            return 1

    # ---- Resolve orbit → InitialState ---------------------------------------
    orbit_init_given = any(
        getattr(args, k, None) is not None
        for k in ("hp_km", "ha_km", "a_km", "e", "alt_km",
                  "inc_deg", "raan_deg", "argp_deg", "ta_deg")
    )
    if orbit_init_given:
        try:
            op = resolve_orbit_elements(args)
            a_m = op["a_km"] * 1000.0
            e   = op["e"]
            inc  = op["inc_deg"]  * DEG2RAD
            raan = op["raan_deg"] * DEG2RAD
            argp = op["argp_deg"] * DEG2RAD
            ta   = op["ta_deg"]   * DEG2RAD

            # Canonical SSOT conversion (no silent fallback): a failure here is fatal.
            from lunaris.core.state import create_state_from_keplerian
            mu = float(MU_MOON)
            y0 = create_state_from_keplerian(
                semi_major_axis=a_m, eccentricity=e,
                inclination=inc, raan=raan, argp=argp,
                true_anomaly=ta, mu=mu,
            )

            from dataclasses import replace
            cfg = replace_sim_config(cfg, initial_state=y0)
        except Exception as exc:
            print(f"[BATCH][FATAL] Orbit init failed: {exc}", flush=True)
            return 1

    # ---- Build BatchPropagationConfig ---------------------------------------------
    try:
        batch_cfg = BatchPropagationConfig(
            n_samples             = int(args.n_samples),
            seed                  = int(args.seed),
            state                 = StateUncertainty(
                sigma_r_m   = float(args.sigma_r_m),
                sigma_v_m_s = float(args.sigma_v_m_s),
            ),
            spacecraft            = SpacecraftUncertainty(
                sigma_mass_kg = float(args.sigma_mass_kg),
                sigma_cd      = float(args.sigma_cd),
                sigma_cr      = float(args.sigma_cr),
                sigma_area_m2 = float(args.sigma_area_m2),
            ),
            sampling_method       = str(args.sampling_method),
            use_gpu               = bool(args.use_gpu),
            batch_backend            = str(args.batch_backend),
            gpu_device_id         = int(args.gpu_device_id),
            gpu_sh_degree         = int(args.gpu_sh_degree),
            gpu_sh_fallback_policy = str(args.gpu_sh_fallback_policy),
            torch_dtype           = str(args.torch_dtype),
            torch_sh_chunk_size   = int(args.torch_sh_chunk_size),
            gpu_threads_per_block = int(args.gpu_threads_per_block),
            gravity_mode_override = str(args.batch_gravity_mode),
            st_lrps_model_dir       = (
                str(Path(str(args.surrogate_gravity_model_dir)).expanduser().resolve())
                if args.surrogate_gravity_model_dir
                else None
            ),
            dt_s                  = float(args.batch_dt_s),
            max_vram_gb           = float(args.max_vram_gb),
            output_format         = str(args.batch_output_format),
            output_path           = str(args.batch_output_path),
            result_storage_mode   = str(args.result_storage_mode),
            max_result_memory_gb  = float(args.max_result_memory_gb),
            detect_impact         = bool(args.detect_impact),
            compute_impact_statistics = bool(args.compute_impact_statistics),
            impact_alt_km         = float(args.impact_alt_km),
        )
    except Exception as exc:
        print(f"[BATCH][FATAL] BatchPropagationConfig build failed: {exc}", flush=True)
        return 1

    # ---- Surface / terrain assets -------------------------------------------
    surface_provider = None
    topo_grid = None
    try:
        surface_provider = init_surface_provider(args)
        if surface_provider is not None and hasattr(surface_provider, "grids"):
            try:
                topo_grid = surface_provider.grids().topo
            except Exception:
                topo_grid = None
    except Exception as exc:
        print(f"[BATCH][FATAL] Surface asset init failed: {exc}", flush=True)
        return 1

    # ---- Run -----------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        from lunaris.batch import BatchPropagationEngine
        engine = BatchPropagationEngine(
            cfg,
            batch_cfg,
            surface_provider=surface_provider,
            topo_grid=topo_grid,
            progress_callback=_emit_progress_line,
        )
        result = engine.run()
    except Exception as exc:
        print(f"[BATCH][FATAL] batch run failed: {exc}", flush=True)
        return 2

    wall_time = time.perf_counter() - t0

    # ---- Emit metrics line (consumed by UI) ---------------------------------
    try:
        metrics = _build_metrics(result, wall_time, batch_cfg)
        print(f"[BATCH_METRICS] {json.dumps(metrics)}", flush=True)
    except Exception as exc:
        print(f"[BATCH][WARN] Could not build metrics: {exc}", flush=True)

    # ---- Optional UQ report (explicitly requested => fail loudly) -----------
    if args.uq_report_dir:
        try:
            from dataclasses import asdict

            from lunaris.analysis.ensemble.uq_report import build_uq_report

            manifest = build_uq_report(
                result,
                args.uq_report_dir,
                run_config=asdict(batch_cfg),
                source_archive=result.archive_path or batch_cfg.output_path_resolved,
            )
            print(
                f"[BATCH] UQ report written: {Path(args.uq_report_dir) / 'uq_manifest.json'} "
                f"(content hash {manifest['covariance_content_sha256'][:12]}...)",
                flush=True,
            )
        except Exception as exc:
            print(f"[BATCH][FATAL] UQ report failed: {exc}", flush=True)
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
