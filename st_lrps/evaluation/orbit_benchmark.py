"""Orbit-level benchmark scaffold for ST-LRPS.

The pointwise evaluator and runtime profiler are ready today.  Orbit-level
claims require a carefully specified propagation contract: initial states,
frames, force model stack, tolerances, truth model, and sampling cadence.  This
module provides the stable CLI entry point and writes that contract skeleton
without pretending to run a full validation suite yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "scaffold",
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "duration_orbits": float(args.duration_orbits),
        "sample_dt_s": float(args.sample_dt_s),
        "expected_contract": {
            "frame": "moon_fixed_cartesian or explicitly transformed inertial/fixed pair",
            "truth_model": "high-degree spherical harmonics",
            "baseline_model": "lower-degree spherical harmonics or point-mass, as declared by target_contract",
            "surrogate_runtime": "st_lrps.runtime.force_model potential_autograd",
            "metrics": ["position_error_m", "velocity_error_m_s", "acceleration_error_m_s2"],
        },
        "note": (
            "Full orbit benchmark implementation is intentionally pending so "
            "runtime/validation claims remain tied to an explicit propagation contract."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Orbit benchmark scaffold for ST-LRPS.")
    ap.add_argument("--model-dir", default=None, help="Trained ST-LRPS run directory.")
    ap.add_argument("--duration-orbits", type=float, default=1.0)
    ap.add_argument("--sample-dt-s", type=float, default=60.0)
    ap.add_argument("--out", default=None, help="Optional JSON output path for the benchmark contract.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    contract = build_contract(args)
    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
