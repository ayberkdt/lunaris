"""``lunaris-frozen-search`` — surrogate-assisted frozen-orbit search (R04).

One command runs the staged pipeline end to end:

    Sobol element sampling -> batch screening (ST-LRPS CUDA when a model
    directory is provided, otherwise torch classical SH) -> domain guard +
    scoring + top-K -> classical SH CPU validation -> (optional) local
    refinement -> family JSON report -> figures.

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
    parser.add_argument(
        "--screening-backend",
        choices=("auto", "torch-sh", "st-lrps"),
        default="auto",
        help="stage-1 backend: auto uses ST-LRPS when --st-lrps-model-dir is set",
    )
    parser.add_argument(
        "--st-lrps-model-dir",
        type=Path,
        default=None,
        help="trained ST-LRPS run directory for stage-1 screening",
    )
    parser.add_argument(
        "--screening-torch-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="ST-LRPS screening tensor dtype",
    )
    parser.add_argument(
        "--screening-chunk-size",
        type=int,
        default=None,
        help="optional torch backend chunk size",
    )
    parser.add_argument(
        "--screening-third-body",
        default="none",
        help="ST-LRPS screening third-body terms: none, sun, earth, or sun,earth",
    )
    parser.add_argument(
        "--screening-output-mode",
        choices=("summary_only", "full"),
        default="summary_only",
        help="stage-1 artifact mode; summary_only avoids storing full (T,N,6)",
    )
    parser.add_argument(
        "--screening-summary-batch-size",
        type=int,
        default=4_096,
        help="sub-batch size used by frozen-search summary_only screening",
    )
    parser.add_argument(
        "--stage1-history-top-k",
        type=int,
        default=None,
        help="number of stage-1 full histories retained in summary_only mode",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--validation-days", type=float, default=30.0)
    parser.add_argument("--validation-degree", type=int, default=50)
    parser.add_argument("--validation-output-dt-s", type=float, default=3_600.0)
    parser.add_argument(
        "--validation-third-body",
        default="none",
        help="classical SH validation third-body terms: none, sun, earth, or sun,earth",
    )
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
    parser.add_argument(
        "--ephemeris-start-date",
        default=None,
        help="UTC start date for ephemeris-wired ST-LRPS/third-body runs",
    )
    parser.add_argument(
        "--paper-safe",
        action="store_true",
        help="reject identity/unresolved rotation-frame backends (paper-grade runs)",
    )
    parser.add_argument(
        "--strict-frame",
        action="store_true",
        help="same frame guard as --paper-safe without the paper-safe label",
    )
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
        STLRPSScreeningPropagator,
        TorchSHScreeningPropagator,
        build_ephemeris_manager_for_frozen_search,
        normalize_third_body_selection,
    )

    try:
        screening_third_body = normalize_third_body_selection(args.screening_third_body)
        validation_third_body = normalize_third_body_selection(args.validation_third_body)
    except ValueError as exc:
        print(f"[FATAL] invalid third-body selector: {exc}", file=sys.stderr)
        return 2

    screening_backend = str(args.screening_backend)
    if screening_backend == "auto":
        screening_backend = "st-lrps" if args.st_lrps_model_dir is not None else "torch-sh"
    if screening_backend == "st-lrps" and args.st_lrps_model_dir is None:
        print(
            "[FATAL] --screening-backend st-lrps requires --st-lrps-model-dir",
            file=sys.stderr,
        )
        return 2
    if screening_third_body and screening_backend != "st-lrps":
        print(
            "[FATAL] --screening-third-body requires ST-LRPS screening",
            file=sys.stderr,
        )
        return 2
    if screening_backend == "st-lrps" and args.screening_device == "cpu":
        print("[FATAL] ST-LRPS screening requires a CUDA device", file=sys.stderr)
        return 2

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
        screening_output_mode=str(args.screening_output_mode),
        screening_summary_batch_size=int(args.screening_summary_batch_size),
        stage1_history_top_k=args.stage1_history_top_k,
        top_k=int(args.top_k),
        validation_duration_s=float(args.validation_days) * 86_400.0,
        validation_output_dt_s=float(args.validation_output_dt_s),
        guard=guard,
        perilune_safety_min_m=float(args.perilune_safety_km) * 1_000.0,
        refine_top_n=int(args.refine_top_n),
        refinement=refinement,
        paper_safe=bool(args.paper_safe),
        strict_frame=bool(args.strict_frame),
    )

    ephem_manager = None
    needs_ephemeris = screening_backend == "st-lrps" or bool(validation_third_body)
    if needs_ephemeris:
        try:
            ephem_dt_s = min(
                float(args.screening_dt_s),
                float(args.screening_output_dt_s),
                float(args.validation_output_dt_s),
            )
            ephem_manager = build_ephemeris_manager_for_frozen_search(
                duration_s=max(
                    float(args.screening_days) * 86_400.0,
                    float(args.validation_days) * 86_400.0,
                ),
                output_dt_s=ephem_dt_s,
                start_date=args.ephemeris_start_date,
                include_third_body=bool(screening_third_body or validation_third_body),
            )
        except Exception as exc:
            print(f"[FATAL] could not build ephemeris: {exc}", file=sys.stderr)
            return 2

    try:
        if screening_backend == "st-lrps":
            screening = STLRPSScreeningPropagator(
                model_dir=args.st_lrps_model_dir,
                dt_s=float(args.screening_dt_s),
                device_id=0,
                torch_dtype=str(args.screening_torch_dtype),
                chunk_size=args.screening_chunk_size,
                ephem_manager=ephem_manager,
                allow_identity_rotation=False,
                third_body=screening_third_body,
                strict_domain=True,
            )
        else:
            screening = TorchSHScreeningPropagator(
                degree=int(args.screening_degree),
                dt_s=float(args.screening_dt_s),
                device=(
                    "cuda:0" if args.screening_device == "cuda" else args.screening_device
                ),
                gravity_file=args.gravity_file,
                chunk_size=args.screening_chunk_size,
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"[FATAL] screening backend unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        validation = ClassicalSHValidationPropagator(
            degree=int(args.validation_degree),
            gravity_file=args.gravity_file,
            third_body=validation_third_body,
            ephem_manager=ephem_manager if validation_third_body else None,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[FATAL] validation backend unavailable: {exc}", file=sys.stderr)
        return 2
    sensitivity = []
    if args.sensitivity_degree is not None:
        sensitivity.append(
            ClassicalSHValidationPropagator(
                degree=int(args.sensitivity_degree),
                gravity_file=args.gravity_file,
                third_body=validation_third_body,
                ephem_manager=ephem_manager if validation_third_body else None,
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
