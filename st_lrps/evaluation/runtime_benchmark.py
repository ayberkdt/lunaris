"""Minimal runtime benchmark scaffold for ST-LRPS artifacts.

This is intentionally pointwise/batch inference only: no propagation, no metric
claims, and no physics changes.  The fuller profiling CLI lives at
``st_lrps.runtime.profiling``; this module gives evaluation workflows a small
benchmark entry point with stable JSON output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from st_lrps.artifacts.manager import make_run_layout, resolve_run_dir, update_run_manifest
from st_lrps.data.dataset_parameters import R_MOON_SI
from st_lrps.runtime.force_model import load_surrogate_force_model


def _queries(n: int, seed: int, alt_min_km: float, alt_max_km: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(int(n), 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-30)
    alt_m = rng.uniform(float(alt_min_km), float(alt_max_km), size=(int(n), 1)) * 1000.0
    return ((float(R_MOON_SI) + alt_m) * directions).astype(np.float64)


def _time(fn, repeat: int) -> dict[str, float]:
    times = []
    for _ in range(max(1, int(repeat))):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=float)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    runtime = load_surrogate_force_model(
        args.model_dir,
        device=args.device,
        chunk_size=int(args.chunk_size),
        allow_config_mismatch=bool(args.allow_config_mismatch),
    )
    q = _queries(args.n, args.seed, args.alt_min_km, args.alt_max_km)
    runtime.predict_residual_potential(q)
    runtime.predict_residual_accel(q)

    potential = _time(lambda: runtime.predict_residual_potential(q), args.repeat)
    residual_accel = _time(lambda: runtime.predict_residual_accel(q), args.repeat)
    total_accel: dict[str, float] | None = None
    try:
        total_accel = _time(lambda: runtime.predict_total_accel(q), args.repeat)
    except Exception as exc:
        total_accel = {"skipped": str(exc)}  # type: ignore[assignment]

    report: dict[str, Any] = {
        "model_dir": str(Path(args.model_dir).expanduser()),
        "checkpoint_path": runtime.checkpoint_path,
        "device": str(runtime.device),
        "runtime_model_kind": getattr(runtime, "runtime_model_kind", "potential_autograd"),
        "n": int(args.n),
        "repeat": int(args.repeat),
        "chunk_size": int(args.chunk_size),
        "potential_prediction": potential,
        "residual_acceleration_prediction": residual_accel,
        "total_acceleration_prediction": total_accel,
        "samples_per_s_residual_accel": float(int(args.n) / max(residual_accel["median_s"], 1e-30)),
        "ms_per_sample_residual_accel": float(residual_accel["median_s"] * 1000.0 / max(int(args.n), 1)),
    }
    out = Path(args.out).expanduser().resolve() if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        try:
            layout = make_run_layout(resolve_run_dir(args.model_dir))
            update_run_manifest(layout, {"runtime_benchmark": str(out)})
        except Exception:
            pass
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch runtime benchmark for ST-LRPS inference.")
    ap.add_argument("--model-dir", required=True, help="Trained ST-LRPS run directory or checkpoint path.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n", type=int, default=4096, help="Number of synthetic shell positions.")
    ap.add_argument("--repeat", type=int, default=20, help="Measured repeats after one warmup.")
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--alt-min-km", type=float, default=100.0)
    ap.add_argument("--alt-max-km", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="results/benchmarks/st_lrps_runtime_benchmark.json",
        help="JSON output path. Defaults under ignored results/.",
    )
    ap.add_argument("--allow-config-mismatch", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_benchmark(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
