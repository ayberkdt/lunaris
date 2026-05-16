#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mc_runner.py — CLI entry point for Monte Carlo ensemble propagation.

This script mirrors main.py's orbit/physics/timeline argument interface and
adds Monte Carlo-specific flags.  It is invoked by the GUI as a subprocess
so that progress can be streamed line-by-line and the main application stays
responsive.

Progress lines (stdout, consumed by the UI)
-------------------------------------------
    [MC] N=500  backend=GPU  T=1.00 d  step=60.0 s  snap=600.0 s
    [MC_PROGRESS] {"stage": "propagating", "percent": 42.5, ...}
    [MC] Batch 1/5  samples 0-99
    [MC] Batch 2/5  samples 100-199
    ...
    [MC] Done. Wall=42.3s  impacts=3/500 (0.6%)
    [MC_METRICS] {"n_samples": 500, "n_impacts": 3, ...}

Exit codes
----------
    0 — success
    1 — configuration / validation error
    2 — runtime error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import math
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Re-use the orbit/physics helpers from main.py (pure functions, no side fx)
# ---------------------------------------------------------------------------
from main import (                                      # noqa: E402
    apply_args_to_config,
    init_surface_provider,
    resolve_orbit_elements,
    str2bool,
    parse_adaptive_table,
    _initial_state_from_keplerian_fallback,
)
from config import load_default_config                  # noqa: E402
from common.constants import R_MOON, MU_MOON, DEG2RAD  # noqa: E402
from common.montecarlo_defs import (                   # noqa: E402
    MonteCarloConfig,
    StateUncertainty,
    SpacecraftUncertainty,
)


# =============================================================================
# 1.                            ARGUMENT PARSER
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Return the combined sim + MC argument parser."""
    p = argparse.ArgumentParser(
        description="LunarSim Monte Carlo Runner",
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
    # Accepted but unused in MC path (orbit output goes to mc-output-path)
    g.add_argument("--out-dir",        type=str)
    g.add_argument("--make-3d-plots",  type=str2bool)
    g.add_argument("--downsample-3d",  type=int)

    # ---- Numerics (CPU propagator compat) -----------------------------------
    g = p.add_argument_group("Numerics (CPU propagator)")
    g.add_argument("--method",           type=str)
    g.add_argument("--user-max-step-s",  type=float)
    g.add_argument("--rtol",             type=float)
    g.add_argument("--atol",             type=float)

    # ---- Monte Carlo --------------------------------------------------------
    g = p.add_argument_group("Monte Carlo")
    g.add_argument("--n-samples",             type=int,   default=500,
                   help="Number of MC trajectories (>= 2)")
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
    g.add_argument("--gpu-device-id",         type=int,   default=0)
    g.add_argument("--gpu-sh-degree",         type=int,   default=10,
                   help="SH degree on GPU (0-24)")
    g.add_argument("--gpu-threads-per-block", type=int,   default=128)
    g.add_argument(
        "--mc-gravity-mode",
        choices=["follow_mission", "classic_sh", "st_lrps"],
        default="follow_mission",
        help="Whether Monte Carlo follows the mission gravity setup or forces classical/ST-LRPS gravity.",
    )
    g.add_argument("--mc-dt-s",               type=float, default=60.0,
                   help="Fixed RK4 step size [s]")
    g.add_argument("--max-vram-gb",           type=float, default=4.0)
    g.add_argument("--mc-output-format",      choices=["hdf5", "npz"],
                   default="hdf5")
    g.add_argument("--mc-output-path",        type=str,
                   default="mc_results/mc_output.h5")
    g.add_argument("--impact-alt-km",         type=float, default=0.0,
                   help="Impact detection threshold altitude [km]")

    return p


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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


def _build_metrics(result: object, wall_time_s: float, mc_cfg: MonteCarloConfig) -> dict:
    """Extract summary statistics from an MCRunResult for the UI metrics panel."""
    import numpy as np

    t          = result.t            # (T,)
    Y          = result.Y            # (T, N, 6)
    impact     = result.impact_mask  # (N,)
    t_impact   = result.t_impact     # (N,)

    N     = int(Y.shape[1])
    n_hit = int(np.sum(impact > 0.5))
    p_imp = n_hit / N if N > 0 else 0.0

    ci_lo, ci_hi = _wilson_ci(n_hit, N)

    # Mean impact time [days]
    hit_times = t_impact[np.isfinite(t_impact) & (impact > 0.5)]
    t_imp_mean_days = float(np.mean(hit_times) / 86400.0) if hit_times.size > 0 else None

    # Altitude at t=0 and t=-1
    def _alt_stats(step: int):
        r = np.linalg.norm(Y[step, :, :3], axis=1)   # (N,)
        alt_km = (r - float(R_MOON)) / 1000.0
        return float(np.mean(alt_km)), float(np.std(alt_km))

    alt_mean_0, alt_std_0 = _alt_stats(0)
    alt_mean_f, alt_std_f = _alt_stats(-1)

    diagnostics = getattr(result, "diagnostics", {}) or {}
    backend_name = str(diagnostics.get("backend", "GPU" if mc_cfg.use_gpu else "CPU"))
    backend_note = str(diagnostics.get("backend_note", "") or "")
    backend_diag = diagnostics.get("backend_diagnostics", {}) or {}

    return {
        "n_samples":        N,
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
        "device_name":      str(backend_diag.get("device_name", "") or ""),
        "threads_per_block": backend_diag.get("threads_per_block"),
        "output_path":      str(mc_cfg.output_path_resolved),
    }


def _emit_progress_line(payload: dict) -> None:
    """
    Stream one structured Monte Carlo progress update to stdout.

    The desktop UI treats ``[MC_PROGRESS]`` as a machine-readable control line
    rather than as human log text.  Keeping this emission centralized ensures
    every backend phase uses the same JSON envelope.
    """

    print(f"[MC_PROGRESS] {json.dumps(payload)}", flush=True)


# =============================================================================
# 3.                                 MAIN
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    # ---- Build SimConfig ----------------------------------------------------
    try:
        cfg = load_default_config()
        cfg = apply_args_to_config(cfg, args)
    except Exception as exc:
        print(f"[MC][FATAL] Config init failed: {exc}", flush=True)
        return 1

    if str(args.mc_gravity_mode) != "follow_mission":
        try:
            from dataclasses import replace

            forced_backend = str(args.mc_gravity_mode)
            cfg = replace(
                cfg,
                gravity=replace(cfg.gravity, backend=forced_backend),
                flags=replace(cfg.flags, enable_sh=True),
            )
        except Exception as exc:
            print(f"[MC][FATAL] Gravity override failed: {exc}", flush=True)
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

            try:
                from core.state import create_state_from_keplerian
                mu = float(MU_MOON)
                y0 = create_state_from_keplerian(
                    semi_major_axis=a_m, eccentricity=e,
                    inclination=inc, raan=raan, argp=argp,
                    true_anomaly=ta, mu=mu,
                )
            except Exception:
                y0 = _initial_state_from_keplerian_fallback(
                    a_m=a_m, e=e,
                    inc_rad=inc, raan_rad=raan, argp_rad=argp, ta_rad=ta,
                    mu=float(MU_MOON),
                )

            from dataclasses import replace
            cfg = replace(cfg, initial_state=y0)
        except Exception as exc:
            print(f"[MC][FATAL] Orbit init failed: {exc}", flush=True)
            return 1

    # ---- Build MonteCarloConfig ---------------------------------------------
    try:
        mc_cfg = MonteCarloConfig(
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
            use_gpu               = bool(args.use_gpu),
            gpu_device_id         = int(args.gpu_device_id),
            gpu_sh_degree         = int(args.gpu_sh_degree),
            gpu_threads_per_block = int(args.gpu_threads_per_block),
            gravity_mode_override = str(args.mc_gravity_mode),
            st_lrps_model_dir       = (
                str(Path(str(args.surrogate_gravity_model_dir)).expanduser().resolve())
                if args.surrogate_gravity_model_dir
                else None
            ),
            dt_s                  = float(args.mc_dt_s),
            max_vram_gb           = float(args.max_vram_gb),
            output_format         = str(args.mc_output_format),
            output_path           = str(args.mc_output_path),
            impact_alt_km         = float(args.impact_alt_km),
        )
    except Exception as exc:
        print(f"[MC][FATAL] MonteCarloConfig build failed: {exc}", flush=True)
        return 1

    # ---- Surface / terrain assets -------------------------------------------
    surface_provider = None
    topo_grid = None
    try:
        surface_provider = init_surface_provider(args)
        if surface_provider is not None and hasattr(surface_provider, "grids"):
            try:
                topo_grid = surface_provider.grids().topo  # type: ignore[attr-defined]
            except Exception:
                topo_grid = None
    except Exception as exc:
        print(f"[MC][FATAL] Surface asset init failed: {exc}", flush=True)
        return 1

    # ---- Run -----------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        from core.monte_carlo_engine import MonteCarloEngine
        engine = MonteCarloEngine(
            cfg,
            mc_cfg,
            surface_provider=surface_provider,
            topo_grid=topo_grid,
            progress_callback=_emit_progress_line,
        )
        result = engine.run()
    except Exception as exc:
        print(f"[MC][FATAL] MC run failed: {exc}", flush=True)
        return 2

    wall_time = time.perf_counter() - t0

    # ---- Emit metrics line (consumed by UI) ---------------------------------
    try:
        metrics = _build_metrics(result, wall_time, mc_cfg)
        print(f"[MC_METRICS] {json.dumps(metrics)}", flush=True)
    except Exception as exc:
        print(f"[MC][WARN] Could not build metrics: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
