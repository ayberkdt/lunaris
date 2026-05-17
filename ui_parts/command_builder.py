# LUNAR_SIMULATION/ui_parts/command_builder.py
# -*- coding: utf-8 -*-
"""
UI -> CLI bridge helpers for Lunar Mission Studio.

This module centralizes the translation between page-owned UI state and the
strict command-line interface exposed by `main.py`. Keeping the mapping in one
place has two benefits:

1. The host window no longer needs to know every CLI flag detail.
2. The translation logic becomes easy to unit test without booting a Qt window.

Input model
-----------
The helpers intentionally operate on plain dictionaries/dataclasses collected
from the page widgets:
- Orbit page: `OrbitPage.get_data()`
- Force page: `ForceModelsPage.get_data()`
- Propagation page: `MissionPropagationPage.to_dict()`
- Output page: `OutputPageState`
- Data page: `DataFilesState`

This keeps the bridge free of direct widget access and makes the boundary
between UI presentation and backend orchestration more explicit.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from common.time_utils import normalize_iso_datetime_to_utc_string

from .data_files_page import DataFilesState
from .result_exports_page import OutputPageState
from .solver_policy import (
    DEFAULT_MAX_STEP_S,
    choose_max_step,
    choose_solver_tolerances,
    coerce_positive_float,
)
from .ui_commons import bool_to_onoff


def _warn(log_warning: Optional[Callable[[str], None]], message: str) -> None:
    """
    Forward a non-fatal warning to the optional host callback.

    The command builder should stay pure and should not talk to Qt directly, so
    warnings are pushed back to the caller through a simple callable.
    """

    if log_warning is not None:
        log_warning(message)


def build_preflight_snapshot(
    *,
    orbit: Mapping[str, Any],
    forces: Mapping[str, Any],
    propagation: Mapping[str, Any],
    output: OutputPageState,
    data_files: DataFilesState,
    spacecraft_cfg: Any,
    solver_cfg: Any,
    gravity_cfg: Any,
    albedo_cfg: Any,
) -> dict[str, Any]:
    """
    Flatten UI state into the validation payload consumed by `PreFlightWorker`.

    The worker performs lightweight checks before launching the simulation. It
    does not need the full richness of the UI, so this helper deliberately
    serializes only the fields that matter for preflight validation.
    """

    integrator = propagation.get("integrator", {}) or {}
    method_label = str(integrator.get("method", "") or "DOP853 (Adaptive)")
    rtol_value, atol_value = choose_solver_tolerances(
        method_label,
        rtol=integrator.get("rtol", getattr(solver_cfg, "rtol", None)),
        atol=getattr(solver_cfg, "atol", None),
    )
    max_step_value = choose_max_step(
        getattr(solver_cfg, "max_step", DEFAULT_MAX_STEP_S),
        default=DEFAULT_MAX_STEP_S,
    )

    snapshot: dict[str, Any] = {
        "orbit_mode": orbit.get("mode", "hp_ha"),
        "inc_deg": float(orbit.get("inc_deg", 0.0)),
        "raan_deg": float(orbit.get("raan_deg", 0.0)),
        "argp_deg": float(orbit.get("argp_deg", 0.0)),
        "ta_deg": float(orbit.get("ta_deg", 0.0)),
        "gravity_enabled": bool((forces.get("gravity", {}) or {}).get("enabled", True)),
        "gravity_backend": str(getattr(gravity_cfg, "backend", "classic_sh") or "classic_sh"),
        "gravity_file": str(getattr(gravity_cfg, "file_path", "") or ""),
        "st_lrps_model_dir": str(getattr(gravity_cfg, "st_lrps_model_dir", "") or ""),
        "sun_enabled": bool(forces.get("sun", True)),
        "earth_enabled": bool(forces.get("earth", True)),
        "earth_j2_enabled": bool(forces.get("earth_j2", False)),
        "tides_k2_enabled": bool(forces.get("tides_k2", True)),
        "tides_k3_enabled": bool(forces.get("tides_k3", False)),
        "relativity_1pn_enabled": bool(forces.get("relativity_1pn", False)),
        "albedo_enabled": bool(forces.get("albedo", False)),
        "albedo_label": str(getattr(albedo_cfg, "label_path", "") or ""),
        "albedo_img": str(getattr(albedo_cfg, "img_path", "") or ""),
        "mass_kg": float(getattr(spacecraft_cfg, "mass_kg", 1000.0)),
        "area_m2": float(getattr(spacecraft_cfg, "area_m2", 5.0)),
        "cd": float(getattr(spacecraft_cfg, "cd", 2.2)),
        "cr": float(getattr(spacecraft_cfg, "cr", 1.5)),
        "rtol": float(rtol_value),
        "atol": float(atol_value),
        "max_step": (float(max_step_value) if max_step_value is not None else None),
        "output_dir": output.output_dir,
        "ldem_root": data_files.ldem_root,
        "albedo_root": data_files.albedo_root,
        "kernel_dir": data_files.kernel_dir,
        "ldem_ppd": int(data_files.ldem_ppd),
    }

    if snapshot["orbit_mode"] == "circular":
        snapshot["alt_km"] = float(orbit.get("alt_km", 100.0))
    elif snapshot["orbit_mode"] == "hp_ha":
        snapshot["hp_km"] = float(orbit.get("hp_km", 0.0))
        snapshot["ha_km"] = float(orbit.get("ha_km", orbit.get("hp_km", 0.0)))
    else:
        snapshot["a_km"] = float(orbit.get("a_km", 0.0))
        snapshot["e"] = float(orbit.get("e", 0.0))

    timeline = propagation.get("timeline", {}) or {}
    snapshot["duration_val"] = float(timeline.get("duration", 10.0))
    snapshot["duration_unit"] = str(timeline.get("unit", "Days")).lower()
    return snapshot


def build_command(
    *,
    python_executable: str,
    main_script_path: Path,
    orbit: Mapping[str, Any],
    forces: Mapping[str, Any],
    propagation: Mapping[str, Any],
    output: OutputPageState,
    data_files: DataFilesState,
    gravity_cfg: Any,
    solver_cfg: Any,
    spacecraft_cfg: Any,
    log_warning: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Convert the current UI state into the strict backend CLI command.

    Parameters
    ----------
    python_executable:
        Usually `sys.executable`, kept injectable for tests.
    main_script_path:
        Absolute path to the backend entry point.
    orbit / forces / propagation:
        Plain state snapshots emitted by page-level helper methods.
    output / data_files:
        Dataclass snapshots owned by their respective pages.
    gravity_cfg / solver_cfg / spacecraft_cfg:
        Shared mutable config objects edited by the advanced dialogs.
    log_warning:
        Optional callback used when the UI provided a recoverable-but-invalid
        value and the builder had to fall back to a safer default.
    """

    command: list[str] = [python_executable, str(main_script_path)]

    orbit_mode = orbit.get("mode", "hp_ha")
    if orbit_mode == "circular":
        command.extend(["--alt-km", str(orbit.get("alt_km", 100.0))])
    elif orbit_mode == "hp_ha":
        command.extend(["--hp-km", str(orbit["hp_km"]), "--ha-km", str(orbit["ha_km"])])
    else:
        command.extend(["--a-km", str(orbit["a_km"]), "--e", str(orbit["e"])])

    command.extend(["--inc-deg", str(orbit["inc_deg"])])
    command.extend(["--raan-deg", str(orbit["raan_deg"])])
    command.extend(["--argp-deg", str(orbit["argp_deg"])])
    command.extend(["--ta-deg", str(orbit["ta_deg"])])

    timeline = propagation.get("timeline", {}) or {}
    integrator = propagation.get("integrator", {}) or {}

    epoch = str(timeline.get("epoch", "")).strip()
    if epoch:
        command.extend(["--start-date", normalize_iso_datetime_to_utc_string(epoch, precision=0)])

    duration_value = str(timeline.get("duration", "")).strip()
    duration_unit = str(timeline.get("unit", "Days")).lower().strip()
    if duration_value:
        if duration_unit.startswith("hour"):
            command.extend(["--hours", duration_value])
        else:
            command.extend(["--days", duration_value])

    output_mode = str(integrator.get("output_mode", "dt") or "dt")
    dt_out = str(integrator.get("dt_out", "")).strip()
    samples_per_period = str(integrator.get("samples_per_period", "")).strip()
    if output_mode == "spp" and samples_per_period:
        command.extend(["--samples-per-period", samples_per_period])
    elif dt_out:
        command.extend(["--output-dt-s", dt_out])

    gravity_section = forces.get("gravity", {}) or {}
    gravity_enabled = bool(gravity_section.get("enabled", True))
    command.extend(["--enable-sh", bool_to_onoff(gravity_enabled)])

    if gravity_enabled:
        gravity_backend = str(getattr(gravity_cfg, "backend", "classic_sh") or "classic_sh")
        command.extend(["--gravity-backend", gravity_backend])

        if gravity_backend == "st_lrps":
            surrogate_dir = str(getattr(gravity_cfg, "st_lrps_model_dir", "") or "").strip()
            if surrogate_dir:
                command.extend(["--surrogate-gravity-model-dir", surrogate_dir])
        else:
            degree = getattr(gravity_cfg, "degree", None)
            if degree is not None:
                command.extend(["--degree", str(int(degree))])

            gravity_path = str(getattr(gravity_cfg, "file_path", "") or "").strip()
            if gravity_path:
                command.extend(["--gravity-file-path", gravity_path])

            adaptive_enabled = bool(getattr(gravity_cfg, "adaptive_enabled", False))
            command.extend(["--adaptive-enabled", bool_to_onoff(adaptive_enabled)])

            if adaptive_enabled:
                adaptive_table = getattr(gravity_cfg, "adaptive_table", None)
                if adaptive_table:
                    table_parts = [f"{float(alt)}:{int(deg)}" for alt, deg in adaptive_table]
                    command.extend(["--adaptive-table", ",".join(table_parts)])
    else:
        command.extend(["--adaptive-enabled", "off"])

    command.extend(["--enable-3rd-body-sun", bool_to_onoff(bool(forces.get("sun", True)))])
    command.extend(["--enable-3rd-body-earth", bool_to_onoff(bool(forces.get("earth", True)))])
    command.extend(["--enable-earth-j2", bool_to_onoff(bool(forces.get("earth_j2", False)))])
    command.extend(["--enable-srp", bool_to_onoff(bool(forces.get("srp", False)))])
    command.extend(["--enable-albedo", bool_to_onoff(bool(forces.get("albedo", False)))])
    command.extend(["--enable-thermal", bool_to_onoff(bool(forces.get("thermal", False)))])
    surface_albedo_needed = bool(forces.get("albedo", False) or forces.get("thermal", False))

    tides_k2 = bool(forces.get("tides_k2", True))
    tides_k3 = bool(forces.get("tides_k3", False))
    if tides_k3:
        command.extend(["--enable-tides", "on", "--tides-kind", "k3"])
    elif tides_k2:
        command.extend(["--enable-tides", "on", "--tides-kind", "k2"])
    else:
        command.extend(["--enable-tides", "off"])

    command.extend(
        ["--enable-relativity-1pn", bool_to_onoff(bool(forces.get("relativity_1pn", False)))]
    )

    command.extend(["--mass-kg", str(getattr(spacecraft_cfg, "mass_kg", 1000.0))])
    command.extend(["--area-m2", str(getattr(spacecraft_cfg, "area_m2", 5.0))])
    command.extend(["--cd", str(getattr(spacecraft_cfg, "cd", 2.2))])
    command.extend(["--cr", str(getattr(spacecraft_cfg, "cr", 1.5))])

    integrator_label = str(integrator.get("method", "") or "").strip()
    integrator_method = (integrator_label.split()[0] if integrator_label else "DOP853").strip()
    if integrator_method:
        command.extend(["--method", integrator_method])

    max_step_raw = str(integrator.get("max_step", "") or "").strip()
    max_step_value: Optional[float] = None
    if max_step_raw:
        try:
            max_step_value = float(max_step_raw)
        except Exception:
            _warn(
                log_warning,
                f"[UI] Invalid max step: '{max_step_raw}'. Using solver settings value instead.",
            )

    if max_step_value is None:
        max_step_value = choose_max_step(
            getattr(solver_cfg, "max_step", None),
            default=DEFAULT_MAX_STEP_S,
        )

    if max_step_value is not None and max_step_value > 0.0:
        command.extend(["--user-max-step-s", str(max_step_value)])

    if "Adaptive" in integrator_label:
        rtol_raw = str(integrator.get("rtol", "") or "").strip()
        if rtol_raw and coerce_positive_float(rtol_raw) is None:
            _warn(
                log_warning,
                f"[UI] Invalid rtol: '{rtol_raw}'. Using a safe default value instead.",
            )

        atol_raw = getattr(solver_cfg, "atol", None)
        if coerce_positive_float(atol_raw) is None:
            _warn(
                log_warning,
                "[UI] Invalid atol in solver settings. Using a safe default value instead.",
            )

        rtol_value, atol_value = choose_solver_tolerances(
            integrator_label,
            rtol=(rtol_raw if rtol_raw else getattr(solver_cfg, "rtol", None)),
            atol=atol_raw,
        )
        command.extend(["--rtol", str(rtol_value)])
        command.extend(["--atol", str(atol_value)])

    if output.output_dir.strip():
        command.extend(["--out-dir", output.output_dir.strip()])

    command.extend(["--make-3d-plots", bool_to_onoff(bool(output.generate_3d_plots))])
    if output.generate_3d_plots and int(output.downsample_3d) > 1:
        command.extend(["--downsample-3d", str(int(output.downsample_3d))])

    if data_files.ldem_root:
        command.extend(["--ldem-root", data_files.ldem_root])
        command.extend(["--ldem-ppd", str(int(data_files.ldem_ppd or 4))])

    if surface_albedo_needed and data_files.albedo_root:
        command.extend(["--albedo-root", data_files.albedo_root])

    if data_files.kernel_dir:
        command.extend(["--kernel-dir", data_files.kernel_dir])

    return [str(item) for item in command]


def build_mc_command(
    *,
    python_executable: str,
    mc_runner_path: Path,
    orbit: Mapping[str, Any],
    forces: Mapping[str, Any],
    propagation: Mapping[str, Any],
    mc_data: Mapping[str, Any],
    data_files: DataFilesState,
    gravity_cfg: Any,
    solver_cfg: Any,
    spacecraft_cfg: Any,
    log_warning: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Build the CLI command for ``mc_runner.py`` from modular UI state.

    Parameters mirror ``build_command()`` but target the MC runner script and
    append Monte Carlo specific flags from ``mc_data`` (MonteCarloPage.get_data()).
    """

    command: list[str] = [python_executable, str(mc_runner_path)]

    # -- Orbit ----------------------------------------------------------------
    orbit_mode = orbit.get("mode", "hp_ha")
    if orbit_mode == "circular":
        command.extend(["--alt-km", str(orbit.get("alt_km", 100.0))])
    elif orbit_mode == "hp_ha":
        command.extend(["--hp-km", str(orbit.get("hp_km", 50.0))])
        command.extend(["--ha-km", str(orbit.get("ha_km", orbit.get("hp_km", 50.0)))])
    else:
        command.extend(["--a-km", str(orbit.get("a_km", 0.0))])
        command.extend(["--e",    str(orbit.get("e",    0.0))])

    command.extend(["--inc-deg",  str(orbit.get("inc_deg",  0.0))])
    command.extend(["--raan-deg", str(orbit.get("raan_deg", 0.0))])
    command.extend(["--argp-deg", str(orbit.get("argp_deg", 0.0))])
    command.extend(["--ta-deg",   str(orbit.get("ta_deg",   0.0))])

    # -- Timeline -------------------------------------------------------------
    timeline   = propagation.get("timeline", {}) or {}
    integrator = propagation.get("integrator", {}) or {}

    epoch = str(timeline.get("epoch", "")).strip()
    if epoch:
        command.extend(["--start-date", normalize_iso_datetime_to_utc_string(epoch, precision=0)])

    duration_value = str(timeline.get("duration", "")).strip()
    duration_unit  = str(timeline.get("unit", "Days")).lower().strip()
    if duration_value:
        if duration_unit.startswith("hour"):
            command.extend(["--hours", duration_value])
        else:
            command.extend(["--days", duration_value])

    dt_out = str(integrator.get("dt_out", "")).strip()
    if dt_out:
        command.extend(["--output-dt-s", dt_out])
    else:
        samples_per_period = str(timeline.get("samples_per_period", "")).strip()
        if samples_per_period:
            command.extend(["--samples-per-period", samples_per_period])

    # -- Physics flags --------------------------------------------------------
    gravity_section = forces.get("gravity", {}) or {}
    gravity_mode_override = str(mc_data.get("gravity_mode_override", "follow_mission") or "follow_mission")
    gravity_enabled = bool(gravity_section.get("enabled", True))
    gravity_backend = str(getattr(gravity_cfg, "backend", "classic_sh") or "classic_sh")
    if gravity_mode_override == "classic_sh":
        gravity_enabled = True
        gravity_backend = "classic_sh"
    elif gravity_mode_override == "st_lrps":
        gravity_enabled = True
        gravity_backend = "st_lrps"
    command.extend(["--enable-sh", bool_to_onoff(gravity_enabled)])

    if gravity_enabled:
        command.extend(["--gravity-backend", gravity_backend])
        if gravity_backend == "st_lrps":
            surrogate_dir = ""
            if gravity_mode_override == "st_lrps":
                surrogate_dir = str(mc_data.get("st_lrps_model_dir", "") or "").strip()
            if not surrogate_dir:
                surrogate_dir = str(getattr(gravity_cfg, "st_lrps_model_dir", "") or "").strip()
            if surrogate_dir:
                command.extend(["--surrogate-gravity-model-dir", surrogate_dir])
            elif log_warning is not None:
                log_warning(
                    "Monte Carlo surrogate gravity mode is selected, but no surrogate run "
                    "directory was provided. Set one in the MC page or Force Models page."
                )
        else:
            degree = getattr(gravity_cfg, "degree", None)
            if degree is not None:
                command.extend(["--degree", str(int(degree))])
            gravity_path = str(getattr(gravity_cfg, "file_path", "") or "").strip()
            if gravity_path:
                command.extend(["--gravity-file-path", gravity_path])
            adaptive_enabled = bool(getattr(gravity_cfg, "adaptive_enabled", False))
            command.extend(["--adaptive-enabled", bool_to_onoff(adaptive_enabled)])
            if adaptive_enabled:
                adaptive_table = getattr(gravity_cfg, "adaptive_table", None)
                if adaptive_table:
                    table_parts = [f"{float(alt)}:{int(deg)}" for alt, deg in adaptive_table]
                    command.extend(["--adaptive-table", ",".join(table_parts)])
    else:
        command.extend(["--adaptive-enabled", "off"])

    command.extend(["--enable-3rd-body-sun",   bool_to_onoff(bool(forces.get("sun",    True)))])
    command.extend(["--enable-3rd-body-earth",  bool_to_onoff(bool(forces.get("earth",  True)))])
    command.extend(["--enable-earth-j2",        bool_to_onoff(bool(forces.get("earth_j2", False)))])
    command.extend(["--enable-srp",             bool_to_onoff(bool(forces.get("srp",    False)))])
    command.extend(["--enable-albedo",          bool_to_onoff(bool(forces.get("albedo", False)))])
    command.extend(["--enable-thermal",         bool_to_onoff(bool(forces.get("thermal", False)))])
    surface_albedo_needed = bool(forces.get("albedo", False) or forces.get("thermal", False))

    tides_k2 = bool(forces.get("tides_k2", True))
    tides_k3 = bool(forces.get("tides_k3", False))
    if tides_k3:
        command.extend(["--enable-tides", "on", "--tides-kind", "k3"])
    elif tides_k2:
        command.extend(["--enable-tides", "on", "--tides-kind", "k2"])
    else:
        command.extend(["--enable-tides", "off"])

    command.extend(["--enable-relativity-1pn",
                    bool_to_onoff(bool(forces.get("relativity_1pn", False)))])

    # -- Spacecraft nominal props (center of ensemble) ------------------------
    command.extend(["--mass-kg", str(getattr(spacecraft_cfg, "mass_kg", 1000.0))])
    command.extend(["--area-m2", str(getattr(spacecraft_cfg, "area_m2", 5.0))])
    command.extend(["--cd",      str(getattr(spacecraft_cfg, "cd",      2.2))])
    command.extend(["--cr",      str(getattr(spacecraft_cfg, "cr",      1.5))])

    # -- Data paths -----------------------------------------------------------
    if data_files.kernel_dir:
        command.extend(["--kernel-dir", data_files.kernel_dir])
    if data_files.ldem_root:
        command.extend(["--ldem-root",  data_files.ldem_root])
        command.extend(["--ldem-ppd",   str(int(data_files.ldem_ppd or 4))])
    if surface_albedo_needed and data_files.albedo_root:
        command.extend(["--albedo-root", data_files.albedo_root])

    # -- Monte Carlo specific flags -------------------------------------------
    integrator_label = str(integrator.get("method", "") or "").strip()
    integrator_method = (integrator_label.split()[0] if integrator_label else "DOP853").strip()
    if integrator_method:
        command.extend(["--method", integrator_method])

    max_step_raw = str(integrator.get("max_step", "") or "").strip()
    max_step_value = choose_max_step(
        max_step_raw or getattr(solver_cfg, "max_step", None),
        default=getattr(solver_cfg, "max_step", DEFAULT_MAX_STEP_S),
    )
    if max_step_value is not None:
        command.extend(["--user-max-step-s", f"{float(max_step_value):g}"])

    rtol_value, atol_value = choose_solver_tolerances(
        integrator_label or "DOP853 (Adaptive)",
        rtol=integrator.get("rtol", getattr(solver_cfg, "rtol", None)),
        atol=getattr(solver_cfg, "atol", None),
    )
    command.extend(["--rtol", f"{float(rtol_value):g}"])
    command.extend(["--atol", f"{float(atol_value):g}"])

    command.extend(["--n-samples",             str(mc_data.get("n_samples",  500))])
    command.extend(["--seed",                  str(mc_data.get("seed",        42))])
    command.extend(["--sigma-r-m",             str(mc_data.get("sigma_r_m", 500.0))])
    command.extend(["--sigma-v-m-s",           str(mc_data.get("sigma_v_m_s", 0.5))])
    command.extend(["--sigma-mass-kg",         str(mc_data.get("sigma_mass_kg", 0.0))])
    command.extend(["--sigma-area-m2",         str(mc_data.get("sigma_area_m2", 0.0))])
    command.extend(["--sigma-cd",              str(mc_data.get("sigma_cd",  0.0))])
    command.extend(["--sigma-cr",              str(mc_data.get("sigma_cr",  0.0))])
    # use_gpu is forwarded as-is. When ST-LRPS gravity is selected the backend
    # policy resolver (core.mc_backend_policy) automatically routes to the
    # TorchBatchPropagator GPU path when PyTorch CUDA is available, and falls
    # back to CPU DOP853 when it is not.  No command-side override is needed.
    use_gpu = bool(mc_data.get("use_gpu", True))
    command.extend(["--use-gpu",               bool_to_onoff(use_gpu)])
    command.extend(["--gpu-device-id",         str(mc_data.get("gpu_device_id",  0))])
    command.extend(["--gpu-sh-degree",         str(mc_data.get("gpu_sh_degree", 10))])
    command.extend(["--gpu-threads-per-block", str(mc_data.get("gpu_threads_per_block", 128))])
    command.extend(["--mc-gravity-mode",       gravity_mode_override])
    command.extend(["--mc-dt-s",               str(mc_data.get("dt_s",       60.0))])
    command.extend(["--max-vram-gb",           str(mc_data.get("max_vram_gb", 4.0))])
    command.extend(["--mc-output-format",      str(mc_data.get("output_format", "hdf5"))])
    command.extend(["--mc-output-path",        str(mc_data.get("output_path",
                                                               "mc_results/mc_output.h5"))])
    command.extend(["--impact-alt-km",         str(mc_data.get("impact_alt_km", 0.0))])

    return [str(item) for item in command]


def build_command_preview(command: Sequence[str]) -> str:
    """
    Render a shell-safe preview string for the current platform.

    Windows and POSIX shells quote arguments differently. Centralizing the
    formatting here ensures the preview matches what the host process launcher
    will actually execute.
    """

    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(list(command))
