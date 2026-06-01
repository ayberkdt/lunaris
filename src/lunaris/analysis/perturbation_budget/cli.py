"""Command-line entry point for Perturbation Budget Analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable, Optional

from .config import (
    PerturbationBudgetConfig,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    parse_on_off,
)
from .reporting import run_perturbation_budget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunaris-perturbation-budget",
        description="Run Lunaris Perturbation Budget Analysis.",
    )
    parser.add_argument("--altitudes-km", default=None, help="Comma-separated altitudes in km.")
    parser.add_argument("--inclinations-deg", default=None, help="Comma-separated inclinations in degrees.")
    parser.add_argument("--true-anomalies-deg", default=None, help="Comma-separated true anomalies in degrees.")
    parser.add_argument("--epochs", default=None, help="Comma-separated UTC epoch labels.")
    parser.add_argument("--sh-degrees", default=None, help="Comma-separated SH degrees, e.g. 20,30,60,100,200.")
    parser.add_argument("--gravity-model", default=None, help="Optional gravity model file path.")
    parser.add_argument("--out-dir", default=None, help="Output directory.")

    for flag in ("srp", "albedo", "thermal", "tides", "third-body", "relativity", "earth-j2"):
        parser.add_argument(f"--include-{flag}", choices=["on", "off"], default=None)

    parser.add_argument("--spacecraft-area-m2", type=float, default=None)
    parser.add_argument("--spacecraft-mass-kg", type=float, default=None)
    parser.add_argument("--srp-coefficient", type=float, default=None)
    parser.add_argument("--srp-uncertainty", type=float, default=None)
    parser.add_argument("--albedo-uncertainty", type=float, default=None)
    parser.add_argument("--thermal-uncertainty", type=float, default=None)
    parser.add_argument("--tide-uncertainty", type=float, default=None)
    parser.add_argument("--absolute-threshold", type=float, default=None)
    parser.add_argument("--uncertainty-fraction", type=float, default=None)
    parser.add_argument("--synthetic-fallback", choices=["on", "off"], default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> PerturbationBudgetConfig:
    cfg = PerturbationBudgetConfig()
    updates = {}
    if args.altitudes_km:
        updates["altitudes_km"] = parse_csv_floats(args.altitudes_km)
    if args.inclinations_deg:
        updates["inclinations_deg"] = parse_csv_floats(args.inclinations_deg)
    if args.true_anomalies_deg:
        updates["true_anomalies_deg"] = parse_csv_floats(args.true_anomalies_deg)
    if args.epochs:
        updates["epochs_utc"] = parse_csv_strings(args.epochs)
    if args.sh_degrees:
        updates["sh_degrees"] = parse_csv_ints(args.sh_degrees)
    if args.gravity_model:
        updates["gravity_model_path"] = args.gravity_model
    if args.out_dir:
        updates["output_dir"] = args.out_dir

    bool_map = {
        "include_srp": args.include_srp,
        "include_albedo": args.include_albedo,
        "include_thermal_ir": args.include_thermal,
        "include_tides": args.include_tides,
        "include_third_body": args.include_third_body,
        "include_relativity": args.include_relativity,
        "include_earth_j2": args.include_earth_j2,
        "use_synthetic_geometry_fallback": args.synthetic_fallback,
    }
    for field, value in bool_map.items():
        if value is not None:
            updates[field] = parse_on_off(value)

    scalar_map = {
        "spacecraft_area_m2": args.spacecraft_area_m2,
        "spacecraft_mass_kg": args.spacecraft_mass_kg,
        "srp_coefficient": args.srp_coefficient,
        "srp_uncertainty": args.srp_uncertainty,
        "albedo_uncertainty": args.albedo_uncertainty,
        "thermal_uncertainty": args.thermal_uncertainty,
        "tide_uncertainty": args.tide_uncertainty,
        "recommendation_absolute_threshold_m_s2": args.absolute_threshold,
        "recommendation_uncertainty_fraction": args.uncertainty_fraction,
    }
    for field, value in scalar_map.items():
        if value is not None:
            updates[field] = value
    return replace(cfg, **updates)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_perturbation_budget(config_from_args(args))
    print(f"Perturbation Budget Analysis written to: {result.output_dir}")
    print(f"Summary: {result.summary_md}")
    if result.warnings:
        print(f"Warnings: {len(result.warnings)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
