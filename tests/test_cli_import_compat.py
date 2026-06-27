from __future__ import annotations

import importlib


def test_main_facade_reexports_split_cli_surfaces() -> None:
    common_args = importlib.import_module("lunaris.cli.common_args")
    main = importlib.import_module("lunaris.cli.main")
    options = importlib.import_module("lunaris.cli.options")
    run = importlib.import_module("lunaris.cli.run")
    summary = importlib.import_module("lunaris.cli.summary")

    assert main.parse_args is options.parse_args
    assert main.validate_args is options.validate_args
    assert main.main is run.main
    assert main.main_entry is run.main_entry
    assert main.init_ephemeris is run.init_ephemeris
    assert main.print_summary is summary.print_summary
    assert main.median_dt is summary.median_dt
    assert main.apply_args_to_config is common_args.apply_args_to_config
    assert main.resolve_orbit_elements is common_args.resolve_orbit_elements


def test_options_parse_empty_args_without_runtime_setup() -> None:
    options = importlib.import_module("lunaris.cli.options")

    args = options.parse_args([])

    assert args.start_date is None
    assert args.surrogate_gravity_model_dir is None


def test_cli_batch_surface_points_at_batch_engine() -> None:
    cli_batch = importlib.import_module("lunaris.cli.batch")
    batch_engine = importlib.import_module("lunaris.batch.engine")

    assert cli_batch.mc_entry is batch_engine.mc_entry
    assert cli_batch.batch_entry is batch_engine.batch_entry
