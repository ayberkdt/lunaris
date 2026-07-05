"""``lunaris-frozen-search`` — surrogate-assisted frozen-orbit search (R04).

One command runs the staged pipeline end to end:

    Sobol element sampling -> batch screening (torch classical SH, CPU/CUDA)
    -> domain guard + scoring + top-K -> classical SH CPU validation
    -> (optional) local refinement -> family JSON report -> figures.

The run directory is a resumable file contract (`manifest.json` +
`stage*_*.{npz,json}`); re-running with ``--resume`` picks up where it left
off. Statuses obey the R21 rule: only classical SH validation can produce
``strict_frozen`` / ``quasi_frozen``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunaris-frozen-search",
        description="Surrogate-assisted global search for lunar frozen-orbit candidates.",
    )
    parser.add_argument("--out", type=Path, required=True, help="run output directory")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampling-method",
        choices=("sobol", "sobol_scrambled", "lhs"),
        default="sobol_scrambled",
    )
    parser.add_argument("--a-km", type=float, nargs=2, default=(1_838.0, 2_238.0),
                        metavar=("LO", "HI"), help="semi-major axis bounds [km]")
    parser.add_argument("--e", type=float, nargs=2, default=(0.0, 0.25),
                        metavar=("LO", "HI"), help="eccentricity bounds [-]")
    parser.add_argument("--i-deg", type=float, nargs=2, default=(60.0, 120.0),
                        metavar=("LO", "HI"), help="inclination bounds [deg]")
    parser.add_argument("--screening-days", type=float, default=7.0)
    parser.add_argument("--screening-degree", type=int, default=8)
    parser.add_argument("--screening-dt-s", type=float, default=60.0)
    parser.add_argument("--screening-output-dt-s", type=float, default=3_600.0)
    parser.add_argument(
        "--screening-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--validation-days", type=float, default=30.0)
    parser.add_argument("--validation-degree", type=int, default=50)
    parser.add_argument("--validation-output-dt-s", type=float, default=3_600.0)
    parser.add_argument(
        "--sensitivity-degree",
        type=int,
        default=None,
        help="optional second classical SH degree for the R21 sensitivity check",
    )
    parser.add_argument("--gravity-file", type=str, default=None)
    parser.add_argument("--domain-alt-min-km", type=float, default=20.0)
    parser.add_argument("--domain-alt-max-km", type=float, default=20_000.0)
    parser.add_argument("--perilune-safety-km", type=float, default=20.0)
    parser.add_argument("--refine-top-n", type=int, default=0,
                        help="stage-4 Nelder-Mead refinement for the top N candidates (0 = off)")
    parser.add_argument("--refine-max-iterations", type=int, default=60)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("lunaris-frozen-search")

    from lunaris.analysis.frozen.domain_guard import FrozenSearchDomainGuard
    from lunaris.analysis.frozen.refine import RefinementBounds, RefinementConfig
    from lunaris.analysis.frozen.search import (
        ElementBounds,
        FrozenSearchConfig,
        FrozenSearchPipeline,
    )
    from lunaris.analysis.frozen.search_backends import (
        ClassicalSHValidationPropagator,
        TorchSHScreeningPropagator,
    )

    bounds = ElementBounds(
        a_km=tuple(args.a_km),
        e=tuple(args.e),
        i_deg=tuple(args.i_deg),
    )
    guard = FrozenSearchDomainGuard(
        altitude_min_km=float(args.domain_alt_min_km),
        altitude_max_km=float(args.domain_alt_max_km),
    )
    refinement = None
    if int(args.refine_top_n) > 0:
        refinement = RefinementConfig(
            bounds=RefinementBounds(
                a_km=tuple(args.a_km),
                e=tuple(args.e),
                i_deg=tuple(args.i_deg),
                raan_deg=(0.0, 360.0),
                argp_deg=(0.0, 360.0),
                ta_deg=(0.0, 360.0),
            ),
            max_iterations=int(args.refine_max_iterations),
            seed=int(args.seed),
        )
    config = FrozenSearchConfig(
        bounds=bounds,
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        sampling_method=str(args.sampling_method),
        screening_duration_s=float(args.screening_days) * 86_400.0,
        screening_output_dt_s=float(args.screening_output_dt_s),
        top_k=int(args.top_k),
        validation_duration_s=float(args.validation_days) * 86_400.0,
        validation_output_dt_s=float(args.validation_output_dt_s),
        guard=guard,
        perilune_safety_min_m=float(args.perilune_safety_km) * 1_000.0,
        refine_top_n=int(args.refine_top_n),
        refinement=refinement,
    )

    try:
        screening = TorchSHScreeningPropagator(
            degree=int(args.screening_degree),
            dt_s=float(args.screening_dt_s),
            device=("cuda:0" if args.screening_device == "cuda" else args.screening_device),
            gravity_file=args.gravity_file,
        )
    except ImportError as exc:
        print(f"[FATAL] screening backend unavailable: {exc}", file=sys.stderr)
        return 2

    validation = ClassicalSHValidationPropagator(
        degree=int(args.validation_degree), gravity_file=args.gravity_file
    )
    sensitivity = []
    if args.sensitivity_degree is not None:
        sensitivity.append(
            ClassicalSHValidationPropagator(
                degree=int(args.sensitivity_degree), gravity_file=args.gravity_file
            )
        )

    pipeline = FrozenSearchPipeline(
        config,
        screening=screening,
        validation=validation,
        out_dir=args.out,
        sensitivity_validations=sensitivity,
    )
    products = pipeline.run(resume=bool(args.resume))

    n_figures = 0
    if not args.no_figures:
        from lunaris.analysis.frozen.plots import generate_frozen_report_figures

        n_figures = len(generate_frozen_report_figures(args.out))

    report = products["family_report"]
    logger.info("run directory: %s", args.out)
    print(
        f"[frozen-search] {int(args.n_samples)} samples -> "
        f"{len(products['candidates'])} candidates -> "
        f"{len(products['validated'])} validated -> "
        f"{len(report.get('families', []))} families; {n_figures} figures. "
        f"Report: {Path(args.out) / 'stage4_families.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
