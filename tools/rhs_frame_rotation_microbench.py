"""R14 acceptance micro-benchmark: RHS frame-rotation cost.

Times single-RHS evaluations for (a) the SH-only Numba fast path (R13) and
(b) the general classical RHS carrying the same physics (Earth-J2 term enabled
with ``j2_coeff=0.0`` — the R13 parity-test construction), after JIT warm-up,
on the same gravity model, ephemeris rotation, and state.

The R14 refactor prepares the Moon-fixed frame vectors once per RHS
evaluation in both paths; this benchmark records the resulting per-eval cost
as evidence (no double rotation => the general-path overhead stays a
bookkeeping delta, not a rotation multiple).

Usage:
    python tools/rhs_frame_rotation_microbench.py [--degree 50] [--n 2000]

Writes outputs/optimization/rhs_frame_rotation_microbench.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _ConstantEphem:
    """Constant-attitude, constant-Earth ephemeris stub (parity-test twin)."""

    def __init__(self, q_i2f: np.ndarray) -> None:
        self.q_i2f = np.asarray(q_i2f, dtype=np.float64)

    def get_data_provider(self):
        earth = np.array([384_400_000.0, 0.0, 0.0], dtype=np.float64)
        return {
            "dt_s": 1.0,
            "r_sun_tab_m": np.zeros((2, 3), dtype=np.float64),
            "r_earth_tab_m": np.vstack([earth, earth]),
            "q_i2f_tab": np.vstack([self.q_i2f, self.q_i2f]),
        }


def _time_rhs(rhs, y0: np.ndarray, n: int) -> dict[str, float]:
    for _ in range(50):  # JIT warm-up
        rhs(0.0, y0)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for k in range(n):
            rhs(float(k), y0)
        samples.append((time.perf_counter() - t0) / n)
    return {
        "per_eval_us_median": statistics.median(samples) * 1e6,
        "per_eval_us_min": min(samples) * 1e6,
        "n_evals_per_repeat": int(n),
        "repeats": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=50)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/optimization/rhs_frame_rotation_microbench.json"),
    )
    args = parser.parse_args()

    from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
    from lunaris.core.config import load_default_config
    from lunaris.core.dynamics import DynamicsEngine
    from lunaris.physics.spherical_harmonics import GravityModel

    cfg = load_default_config()
    model = GravityModel.from_file(
        path=str(cfg.gravity.file_path), requested_degree=int(args.degree)
    )

    r0 = float(model.R_ref_m) + 100_000.0
    v0 = float(np.sqrt(model.GM_m3s2 / r0))
    y0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0], dtype=np.float64)
    q_i2f = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=np.float64)
    ephem = _ConstantEphem(q_i2f)
    sc = SpacecraftProps(mass_kg=100.0, area_m2=1.0, cr=1.3)

    engines = {
        "sh_only_fast_path": DynamicsEngine(
            sc_props=sc,
            flags=PerturbationFlags(enable_sh=True),
            gravity_model=model,
            ephem_manager=ephem,
            allow_identity_rotation=False,
        ),
        "general_rhs_zero_extra_force": DynamicsEngine(
            sc_props=sc,
            flags=PerturbationFlags(enable_sh=True, enable_earth_j2=True),
            gravity_model=model,
            ephem_manager=ephem,
            earth_j2=SimpleNamespace(
                j2_coeff=0.0, r_eq_m=6_378_137.0, spin_axis_i=(0.0, 0.0, 1.0)
            ),
            allow_identity_rotation=False,
        ),
    }

    results: dict[str, dict] = {}
    for label, engine in engines.items():
        rhs = engine.build_rhs(force_rebuild=True)
        rhs_path = str(engine._prep.get("rhs_path", "unknown"))
        results[label] = {**_time_rhs(rhs, y0, int(args.n)), "rhs_path": rhs_path}
        print(
            f"{label}: {results[label]['per_eval_us_median']:.2f} us/eval "
            f"(path={rhs_path})"
        )

    fast = results["sh_only_fast_path"]["per_eval_us_median"]
    general = results["general_rhs_zero_extra_force"]["per_eval_us_median"]
    payload = {
        "benchmark": "rhs_frame_rotation_microbench (R14 acceptance)",
        "note": (
            "Moon-fixed frame vectors are prepared once per RHS evaluation in "
            "both paths (R14). Same gravity model, constant-attitude ephemeris "
            "rotation, same state; general path carries a zero-magnitude "
            "Earth-J2 term (R13 parity-test construction)."
        ),
        "degree": int(args.degree),
        "general_over_fast_ratio": float(general / fast) if fast > 0 else None,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
