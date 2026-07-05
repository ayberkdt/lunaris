"""CLI contract tests for the frozen-orbit search entry point."""

from __future__ import annotations

from pathlib import Path

from lunaris.cli.frozen_search import build_parser


def test_frozen_search_parser_accepts_ui_command_contract(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    gravity_file = tmp_path / "gravity.sha"
    st_lrps_dir = tmp_path / "st_lrps_run"

    args = build_parser().parse_args(
        [
            "--out",
            str(out_dir),
            "--n-samples",
            "64",
            "--seed",
            "0",
            "--sampling-method",
            "sobol_scrambled",
            "--a-km",
            "1838",
            "2238",
            "--e",
            "0",
            "0.25",
            "--i-deg",
            "60",
            "120",
            "--screening-days",
            "7",
            "--screening-degree",
            "8",
            "--screening-dt-s",
            "60",
            "--screening-output-dt-s",
            "3600",
            "--screening-device",
            "auto",
            "--screening-backend",
            "st-lrps",
            "--st-lrps-model-dir",
            str(st_lrps_dir),
            "--screening-torch-dtype",
            "float64",
            "--screening-chunk-size",
            "512",
            "--screening-third-body",
            "sun,earth",
            "--screening-output-mode",
            "summary_only",
            "--screening-summary-batch-size",
            "1024",
            "--stage1-history-top-k",
            "12",
            "--top-k",
            "8",
            "--validation-days",
            "30",
            "--validation-degree",
            "50",
            "--validation-output-dt-s",
            "3600",
            "--validation-third-body",
            "earth",
            "--sensitivity-degree",
            "100",
            "--gravity-file",
            str(gravity_file),
            "--domain-alt-min-km",
            "20",
            "--domain-alt-max-km",
            "20000",
            "--perilune-safety-km",
            "20",
            "--ephemeris-start-date",
            "2027-03-02T00:00:00Z",
            "--refine-top-n",
            "2",
            "--refine-max-iterations",
            "60",
            "--no-figures",
            "--no-resume",
            "--verbose",
        ]
    )

    assert args.out == out_dir
    assert args.n_samples == 64
    assert args.seed == 0
    assert args.sampling_method == "sobol_scrambled"
    assert tuple(args.a_km) == (1838.0, 2238.0)
    assert tuple(args.e) == (0.0, 0.25)
    assert tuple(args.i_deg) == (60.0, 120.0)
    assert args.screening_device == "auto"
    assert args.screening_backend == "st-lrps"
    assert args.st_lrps_model_dir == st_lrps_dir
    assert args.screening_torch_dtype == "float64"
    assert args.screening_chunk_size == 512
    assert args.screening_third_body == "sun,earth"
    assert args.screening_output_mode == "summary_only"
    assert args.screening_summary_batch_size == 1024
    assert args.stage1_history_top_k == 12
    assert args.top_k == 8
    assert args.validation_degree == 50
    assert args.validation_third_body == "earth"
    assert args.sensitivity_degree == 100
    assert args.gravity_file == str(gravity_file)
    assert args.ephemeris_start_date == "2027-03-02T00:00:00Z"
    assert args.refine_top_n == 2
    assert args.no_figures is True
    assert args.resume is False
    assert args.verbose is True


def test_frozen_search_parser_resume_flag_defaults_and_override(tmp_path: Path) -> None:
    parser = build_parser()

    default_args = parser.parse_args(["--out", str(tmp_path / "default")])
    assert default_args.resume is True
    assert default_args.screening_backend == "auto"
    assert default_args.st_lrps_model_dir is None
    assert default_args.screening_output_mode == "summary_only"
    assert default_args.screening_third_body == "none"
    assert default_args.validation_third_body == "none"

    no_resume_args = parser.parse_args(["--out", str(tmp_path / "fresh"), "--no-resume"])
    assert no_resume_args.resume is False

    explicit_resume_args = parser.parse_args(["--out", str(tmp_path / "resume"), "--resume"])
    assert explicit_resume_args.resume is True
