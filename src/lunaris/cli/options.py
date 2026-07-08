"""Argument parser for the main ``lunaris`` CLI command.

This module is intentionally import-safe: it defines argparse surfaces and
performs lightweight validation only. Heavy artifact helpers are imported lazily
inside validation branches that need them.

Structure
---------
``parse_args`` composes a set of small ``_add_*_args`` group builders (one per
config surface) and then runs a set of focused ``_validate_*`` checks. Splitting
the previously monolithic parser/validator keeps each argument group and each
validation concern independently readable and testable.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lunaris.cli.common_args import parse_adaptive_table, parse_tide_bodies, str2bool
from lunaris.common.time_utils import normalize_iso_datetime_to_utc_string


# ---------------------------------------------------------------------------
# Argument-group builders (one per config surface)
# ---------------------------------------------------------------------------
def _add_time_args(parser: argparse.ArgumentParser) -> None:
    """Time (TimeConfig)."""
    g_time = parser.add_argument_group("Time")
    g_time.add_argument(
        "--start-date",
        type=str,
        help=(
            "UTC start date in ISO format. Naive timestamps are interpreted as UTC; "
            "explicit offsets are accepted and normalized to UTC "
            "(e.g. 2026-01-19T12:00:00Z or 2026-01-19T15:00:00+03:00)."
        ),
    )
    dur = g_time.add_mutually_exclusive_group()
    dur.add_argument("--days", type=float, help="Simulation duration [days]")
    dur.add_argument("--hours", type=float, help="Simulation duration [hours]")

    # aligned with TimeConfig.output_dt_s
    g_time.add_argument("--output-dt-s", type=float, help="Fixed output spacing [s] (omit to keep config default)")
    g_time.add_argument("--samples-per-period", type=int, help="Used when output_dt_s is None")


def _add_orbit_args(parser: argparse.ArgumentParser) -> None:
    """Orbit init."""
    g_orbit = parser.add_argument_group("Orbit Init (choose one)")
    g_orbit.add_argument("--hp-km", type=float, help="Periselene altitude [km]")
    g_orbit.add_argument("--ha-km", type=float, help="Aposelene altitude [km]")
    g_orbit.add_argument("--a-km", type=float, help="Semi-major axis [km]")
    g_orbit.add_argument("--e", type=float, help="Eccentricity (0 <= e < 1)")
    g_orbit.add_argument("--alt-km", type=float, help="Circular orbit altitude [km]")

    g_orbit.add_argument("--inc-deg", type=float, help="Inclination [deg]")
    g_orbit.add_argument("--raan-deg", type=float, help="RAAN [deg]")
    g_orbit.add_argument("--argp-deg", type=float, help="Argument of periapsis [deg]")
    g_orbit.add_argument("--ta-deg", type=float, help="True anomaly [deg]")


def _add_physics_args(parser: argparse.ArgumentParser) -> None:
    """Physics flags (PerturbationFlags) plus thermal/albedo/tide model knobs."""
    g_phys = parser.add_argument_group("Physics Flags")
    g_phys.add_argument("--enable-sh", type=str2bool, help="Enable spherical harmonics gravity (on/off)")
    g_phys.add_argument("--enable-3rd-body-sun", type=str2bool, help="Enable Sun third-body (on/off)")
    g_phys.add_argument("--enable-3rd-body-earth", type=str2bool, help="Enable Earth third-body (on/off)")
    g_phys.add_argument("--enable-earth-j2", type=str2bool, help="Enable differential Earth J2 (on/off)")
    g_phys.add_argument("--enable-srp", type=str2bool, help="Enable SRP (on/off)")
    g_phys.add_argument("--enable-albedo", type=str2bool, help="Enable lunar albedo pressure (on/off)")
    g_phys.add_argument("--enable-thermal", type=str2bool, help="Enable lunar thermal pressure (on/off)")
    g_phys.add_argument("--enable-thermal-ir", dest="enable_thermal", type=str2bool, help="Alias for --enable-thermal")
    g_phys.add_argument(
        "--thermal-mode",
        choices=("constant_temperature", "equilibrium_temperature", "temperature_grid"),
        help="Lunar thermal IR mode.",
    )
    g_phys.add_argument("--thermal-temperature-k", type=float, help="Constant-mode surface temperature [K]")
    g_phys.add_argument("--thermal-night-temperature-k", type=float, help="Equilibrium-mode night/floor temperature [K]")
    g_phys.add_argument("--thermal-emissivity", type=float, help="Thermal surface emissivity [0,1]")
    g_phys.add_argument("--thermal-surface-albedo", type=float, help="Thermal equilibrium absorbed-solar albedo [0,1]")
    g_phys.add_argument("--thermal-ir-coefficient", type=float, help="Spacecraft IR pressure coefficient [-]")
    g_phys.add_argument("--thermal-floor-flux-w-m2", type=float, help="Minimum thermal exitance floor [W/m^2]")
    g_phys.add_argument("--thermal-facet-lat-count", type=int, help="Thermal facet latitude count")
    g_phys.add_argument("--thermal-facet-lon-count", type=int, help="Thermal facet longitude count")

    # Lunar albedo (reflected solar) radiation pressure
    g_phys.add_argument(
        "--albedo-model",
        choices=("lambert_facets", "simple"),
        help="Albedo backend: 'lambert_facets' (default, facet Lambertian) or legacy 'simple' (cannonball).",
    )
    g_phys.add_argument(
        "--albedo-mode",
        choices=("constant_albedo", "albedo_grid", "scaled_dn_grid"),
        help="Per-facet albedo source (grid modes require --albedo-root).",
    )
    g_phys.add_argument("--albedo-const", type=float, help="Constant lunar albedo in [0,1] (default 0.12)")
    g_phys.add_argument("--albedo-pressure-coefficient", type=float, help="Albedo radiation-pressure coefficient C_R_albedo [-]")
    g_phys.add_argument("--albedo-facet-lat-count", type=int, help="Albedo facet latitude count")
    g_phys.add_argument("--albedo-facet-lon-count", type=int, help="Albedo facet longitude count")
    g_phys.add_argument("--albedo-require-provider", type=str2bool, help="Require a surface provider for albedo (on/off)")
    g_phys.add_argument("--albedo-enable-eclipse", type=str2bool, help="Apply lunar-eclipse (Earth-umbra) dimming (on/off)")

    # clean tides contract -> maps to enable_tides_k2/enable_tides_k3
    g_phys.add_argument("--enable-tides", type=str2bool, help="Enable solid tides (on/off)")
    g_phys.add_argument("--tides-kind", choices=("k2", "k3"), help="Tides model kind (k2 or k3)")
    g_phys.add_argument("--tide-bodies", type=parse_tide_bodies, help="Comma-separated tide bodies: earth,sun")
    g_phys.add_argument("--tide-k2", type=float, help="Degree-2 lunar potential Love number k2")
    g_phys.add_argument("--tide-k3", type=float, help="Degree-3 lunar potential Love number k3 (required for --tides-kind k3)")
    g_phys.add_argument("--tide-r-ref-m", type=float, help="Lunar tide reference radius [m]")

    g_phys.add_argument("--enable-relativity-1pn", type=str2bool, help="Enable relativity 1PN (on/off)")


def _add_gravity_args(parser: argparse.ArgumentParser) -> None:
    """Gravity model (GravityConfig)."""
    g_grav = parser.add_argument_group("Gravity Model")
    g_grav.add_argument(
        "--gravity-backend",
        choices=("classic_sh", "st_lrps"),
        help="Central gravity backend: classical spherical harmonics or ST-LRPS surrogate.",
    )
    g_grav.add_argument("--gravity-file-path", type=str, help="Gravity model file path (.tab/.gfc/.shbdr)")
    g_grav.add_argument(
        "--surrogate-gravity-model-dir",
        type=str,
        help="Trained ST-LRPS gravity run directory (config.json + a checkpoint under checkpoints/)",
    )
    g_grav.add_argument("--degree", type=int, help="Max SH degree (Nmax)")
    g_grav.add_argument("--adaptive-enabled", type=str2bool, help="Enable adaptive SH degree (on/off)")
    g_grav.add_argument("--adaptive-table", type=parse_adaptive_table, help="alt:deg,alt:deg (ascending)")


def _add_spacecraft_args(parser: argparse.ArgumentParser) -> None:
    """Spacecraft (SpacecraftProps)."""
    g_sc = parser.add_argument_group("Spacecraft")
    g_sc.add_argument("--mass-kg", type=float, help="Mass [kg]")
    g_sc.add_argument("--area-m2", type=float, help="Area [m^2]")
    g_sc.add_argument("--cd", type=float, help="Cd [-]")
    g_sc.add_argument("--cr", type=float, help="Cr [-]")


def _add_numerics_args(parser: argparse.ArgumentParser) -> None:
    """Numerics (PropagatorConfig)."""
    g_num = parser.add_argument_group("Numerics")
    g_num.add_argument("--method", type=str, help="Integrator method string (e.g. DOP853, RK45, VV)")
    g_num.add_argument("--user-max-step-s", type=float, help="Max internal solver step [s]")
    g_num.add_argument("--rtol", type=float, help="Relative tolerance")
    g_num.add_argument("--atol", type=float, help="Absolute tolerance")
    g_num.add_argument(
        "--compute-2body-baseline",
        type=str2bool,
        help="Run an extra two-body diagnostic propagation after the main solve (on/off)",
    )
    g_num.add_argument("--enable-telemetry", type=str2bool, help="Stream JSON telemetry to stdout (on/off)")
    g_num.add_argument("--telem-cadence-s", type=float, help="Telemetry stdout cadence [s]")
    g_num.add_argument(
        "--telemetry-cadence-s",
        dest="telem_cadence_s",
        type=float,
        help="Alias for --telem-cadence-s",
    )


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    """Output & Assets (OutputConfig + assets)."""
    g_io = parser.add_argument_group("I/O & Assets")
    g_io.add_argument("--out-dir", type=str, help="Output directory")
    g_io.add_argument("--make-3d-plots", type=str2bool, help="Generate 3D plots/animation outputs (on/off)")
    g_io.add_argument("--downsample-3d", type=int, help="3D plot downsample factor")
    g_io.add_argument("--kernel-dir", type=str, help="Directory containing SPICE kernels (renames by filename match)")
    g_io.add_argument("--ldem-root", type=str, help="LOLA LDEM root directory")
    g_io.add_argument("--albedo-root", type=str, help="LOLA Albedo root directory")
    g_io.add_argument("--ldem-ppd", type=int, help="Surface resolution (pixels per degree)")
    g_io.add_argument(
        "--debug-tracebacks",
        action="store_true",
        help="Print tracebacks for unexpected CLI runtime failures.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lunaris Runner (STRICT; config.py + common.type_defs aligned)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _add_time_args(parser)
    _add_orbit_args(parser)
    _add_physics_args(parser)
    _add_gravity_args(parser)
    _add_spacecraft_args(parser)
    _add_numerics_args(parser)
    _add_io_args(parser)

    args = parser.parse_args(args=argv)
    validate_args(parser, args)
    return args


# ---------------------------------------------------------------------------
# Validation (focused checks, each raising via ``parser.error``)
# ---------------------------------------------------------------------------
def _validate_orbit_init(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    hp_ha_any = (args.hp_km is not None) or (args.ha_km is not None)
    a_e_any = (args.a_km is not None) or (args.e is not None)
    alt = args.alt_km is not None

    if hp_ha_any and (args.hp_km is None or args.ha_km is None):
        parser.error("Provide BOTH --hp-km and --ha-km together.")
    if a_e_any and (args.a_km is None or args.e is None):
        parser.error("Provide BOTH --a-km and --e together.")
    if args.e is not None and not (0.0 <= args.e < 1.0):
        parser.error("--e must satisfy 0 <= e < 1.")
    if (args.hp_km is not None and args.ha_km is not None) and (args.a_km is not None or args.e is not None):
        parser.error("Choose ONE orbit init mode: (--hp-km,--ha-km) OR (--a-km,--e) OR (--alt-km).")
    if alt and (hp_ha_any or a_e_any):
        parser.error("Choose ONE orbit init mode: (--alt-km) cannot be combined with other orbit init flags.")

    # If user provides orbital angles without an explicit orbit-init mode, fail fast.
    angles_any = any(
        getattr(args, k) is not None
        for k in ("inc_deg", "raan_deg", "argp_deg", "ta_deg")
    )
    base_any = (
        (args.hp_km is not None and args.ha_km is not None)
        or (args.a_km is not None and args.e is not None)
        or alt
    )
    if angles_any and not base_any:
        parser.error(
            "Orbit angle flags (--inc-deg/--raan-deg/--argp-deg/--ta-deg) require an orbit init mode "
            "(--hp-km/--ha-km or --a-km/--e or --alt-km)."
        )


def _validate_start_date(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.start_date is None:
        return
    s = str(args.start_date).strip()
    try:
        normalize_iso_datetime_to_utc_string(s, precision=0)
    except (ValueError, TypeError) as exc:
        parser.error(
            "--start-date must be an ISO-like timestamp such as "
            "yyyy-MM-ddTHH:mm:ssZ or yyyy-MM-ddTHH:mm:ss+03:00 "
            f"(details: {exc})"
        )


def _validate_numeric_ranges(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.days is not None and args.days <= 0:
        parser.error("--days must be positive.")
    if args.hours is not None and args.hours <= 0:
        parser.error("--hours must be positive.")
    if args.output_dt_s is not None and args.output_dt_s <= 0:
        parser.error("--output-dt-s must be positive.")
    if args.samples_per_period is not None and args.samples_per_period < 2:
        parser.error("--samples-per-period must be >= 2.")
    if args.degree is not None and args.degree < 0:
        parser.error("--degree must be >= 0.")
    if args.downsample_3d is not None and args.downsample_3d < 1:
        parser.error("--downsample-3d must be >= 1.")
    if args.ldem_ppd is not None and args.ldem_ppd <= 0:
        parser.error("--ldem-ppd must be positive.")
    if args.user_max_step_s is not None and args.user_max_step_s <= 0:
        parser.error("--user-max-step-s must be positive.")
    if args.telem_cadence_s is not None and args.telem_cadence_s <= 0:
        parser.error("--telem-cadence-s must be positive.")
    if args.tide_k2 is not None and args.tide_k2 < 0.0:
        parser.error("--tide-k2 must be >= 0.")
    if args.tide_k3 is not None and args.tide_k3 < 0.0:
        parser.error("--tide-k3 must be >= 0.")
    if args.tide_r_ref_m is not None and args.tide_r_ref_m <= 0.0:
        parser.error("--tide-r-ref-m must be positive.")
    if args.thermal_temperature_k is not None and args.thermal_temperature_k < 0.0:
        parser.error("--thermal-temperature-k must be >= 0.")
    if args.thermal_night_temperature_k is not None and args.thermal_night_temperature_k < 0.0:
        parser.error("--thermal-night-temperature-k must be >= 0.")
    if args.thermal_emissivity is not None and not (0.0 <= args.thermal_emissivity <= 1.0):
        parser.error("--thermal-emissivity must be in [0, 1].")
    if args.thermal_surface_albedo is not None and not (0.0 <= args.thermal_surface_albedo <= 1.0):
        parser.error("--thermal-surface-albedo must be in [0, 1].")
    if args.thermal_ir_coefficient is not None and args.thermal_ir_coefficient < 0.0:
        parser.error("--thermal-ir-coefficient must be >= 0.")
    if args.thermal_floor_flux_w_m2 is not None and args.thermal_floor_flux_w_m2 < 0.0:
        parser.error("--thermal-floor-flux-w-m2 must be >= 0.")
    if args.thermal_facet_lat_count is not None and args.thermal_facet_lat_count < 1:
        parser.error("--thermal-facet-lat-count must be >= 1.")
    if args.thermal_facet_lon_count is not None and args.thermal_facet_lon_count < 1:
        parser.error("--thermal-facet-lon-count must be >= 1.")

    if args.albedo_const is not None and not (0.0 <= args.albedo_const <= 1.0):
        parser.error("--albedo-const must be in [0, 1].")
    if args.albedo_pressure_coefficient is not None and args.albedo_pressure_coefficient < 0.0:
        parser.error("--albedo-pressure-coefficient must be >= 0.")
    if args.albedo_facet_lat_count is not None and args.albedo_facet_lat_count < 1:
        parser.error("--albedo-facet-lat-count must be >= 1.")
    if args.albedo_facet_lon_count is not None and args.albedo_facet_lon_count < 1:
        parser.error("--albedo-facet-lon-count must be >= 1.")


def _validate_dependent_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # adaptive table implies adaptive enabled unless user explicitly disabled
    if args.adaptive_table is not None and args.adaptive_enabled is False:
        parser.error("--adaptive-table requires --adaptive-enabled on (or omit --adaptive-enabled).")

    # tides-kind implies enable-tides unless user explicitly forced off
    if args.tides_kind is not None and args.enable_tides is False:
        parser.error("--tides-kind requires --enable-tides on (or omit --enable-tides).")
    if args.tides_kind == "k3" and args.tide_k3 is None:
        parser.error("--tides-kind k3 requires an explicit --tide-k3 value.")


def _validate_paths(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.kernel_dir is not None:
        kd = Path(str(args.kernel_dir)).expanduser()
        if not kd.exists() or not kd.is_dir():
            parser.error(f"--kernel-dir must be an existing directory: {kd}")

    if args.ldem_root is not None:
        p = Path(str(args.ldem_root)).expanduser()
        if not p.exists() or not p.is_dir():
            parser.error(f"--ldem-root must be an existing directory: {p}")

    if args.albedo_root is not None:
        p = Path(str(args.albedo_root)).expanduser()
        if not p.exists() or not p.is_dir():
            parser.error(f"--albedo-root must be an existing directory: {p}")


def _validate_surrogate_dir(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.surrogate_gravity_model_dir is None:
        return

    # Artifact validation: directory existence, config.json, and a usable
    # checkpoint (ckpt_best.pt OR ckpt_last.pt) are delegated to the
    # canonical helper. Do NOT reimplement these checks here.
    from lunaris.common.batch_defs import validate_st_lrps_model_dir

    try:
        model_dir = validate_st_lrps_model_dir(args.surrogate_gravity_model_dir)
    except ValueError as exc:
        parser.error(f"--surrogate-gravity-model-dir: {exc}")

    # Semantic validation (distinct from artifact validation above): confirm
    # the run was trained on a lunar gravity config. Not covered by the
    # artifact helper, so it is kept here and clearly separated.
    looks_like_lunar_run_config: Callable[[Mapping[str, Any]], bool] | None
    try:
        from lunaris.surrogate.st_lrps.data.dataset_parameters import (
            looks_like_lunar_run_config,
        )
    except ImportError:
        looks_like_lunar_run_config = None
    if looks_like_lunar_run_config is not None:
        run_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if not looks_like_lunar_run_config(run_cfg):
            parser.error(
                "--surrogate-gravity-model-dir does not look like a lunar-trained ST-LRPS run: "
                f"{model_dir}"
            )


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _validate_orbit_init(parser, args)
    _validate_start_date(parser, args)
    _validate_numeric_ranges(parser, args)
    _validate_dependent_flags(parser, args)
    _validate_paths(parser, args)
    _validate_surrogate_dir(parser, args)


__all__ = ["parse_args", "validate_args"]
