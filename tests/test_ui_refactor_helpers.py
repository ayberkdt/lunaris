"""
Focused regression tests for the modular UI helper layer.

These tests intentionally avoid booting the full Qt application. The goal is to
verify the new pure/helper modules introduced during the UI refactor:
- command construction from page/config snapshots
- repository-aware auto-detection of data directories
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

from lunaris.loaders.io_surface import _iter_label_candidates
from lunaris.ui.core.command_builder import build_batch_command, build_command
from lunaris.ui.core.session_persistence import autodetect_data_state
from lunaris.ui.pages.data_files_page import DataFilesState
from lunaris.ui.pages.force_models_page import UIGravityConfig
from lunaris.ui.pages.result_exports_page import OutputPageState


class _DummyGravityConfig:
    degree = 660
    file_path = r"C:\models\moon_660.gfc"
    backend = "classic_sh"
    st_lrps_model_dir = r"C:\models\st_lrps_run"
    adaptive_enabled = True
    adaptive_table = ((10.0, 660), (100.0, 140))


class _DummySolverConfig:
    rtol = 1e-10
    atol = 1e-12
    max_step = 3600.0


class _DummySpacecraftConfig:
    mass_kg = 1200.0
    area_m2 = 7.5
    cd = 2.2
    cr = 1.6


def test_build_command_uses_modular_state_objects() -> None:
    orbit = {
        "mode": "hp_ha",
        "hp_km": 50.0,
        "ha_km": 65.0,
        "inc_deg": 90.0,
        "raan_deg": 15.0,
        "argp_deg": 0.0,
        "ta_deg": 180.0,
    }
    forces = {
        "gravity": {"enabled": True},
        "sun": True,
        "earth": False,
        "earth_j2": False,
        "srp": True,
        "albedo": False,
        "thermal": False,
        "tides_k2": True,
        "tides_k3": False,
        "relativity_1pn": True,
    }
    propagation = {
        "timeline": {
            "epoch": "2026-05-03 12:34:56",
            "duration": "48",
            "unit": "Hours",
        },
        "integrator": {
            "method": "DOP853 (Adaptive)",
            "rtol": "1e-10",
            "dt_out": "30",
            "max_step": "120",
        },
    }
    output = OutputPageState(
        output_dir=r"C:\results",
        generate_3d_plots=True,
        downsample_3d=4,
    )
    data_state = DataFilesState(
        ldem_root=r"C:\data\topography_models",
        albedo_root=r"C:\data\albedo_models",
        kernel_dir=r"C:\data\ephemeris_models",
        ldem_ppd=16,
    )

    command = build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        output=output,
        data_files=data_state,
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    assert command[:2] == ["python", "main.py"]
    assert "--start-date" in command
    assert "2026-05-03T12:34:56Z" in command
    assert "--hours" in command
    assert "--make-3d-plots" in command
    assert "--downsample-3d" in command
    assert "--kernel-dir" in command
    assert "--gravity-file-path" in command
    assert "--adaptive-table" in command
    assert command[command.index("--enable-telemetry") + 1] == "on"
    assert command[command.index("--telem-cadence-s") + 1] == "300"
    assert "--enable-relativity-1pn" in command
    assert "--albedo-root" not in command
    assert "--save-csv" not in command


class _DummyAlbedoConfig:
    model = "lambert_facets"
    source = "scaled_dn_grid"
    albedo_const = 0.11
    pressure_coefficient = 1.3
    facet_lat_count = 24
    facet_lon_count = 48
    enable_eclipse = True


def _albedo_command(albedo_cfg: object) -> list[str]:
    forces = {
        "gravity": {"enabled": True},
        "sun": True, "earth": True, "earth_j2": False,
        "srp": False, "albedo": True, "thermal": False,
        "tides_k2": False, "tides_k3": False, "relativity_1pn": False,
    }
    orbit = {"mode": "circular", "alt_km": 100.0, "inc_deg": 90.0,
             "raan_deg": 0.0, "argp_deg": 0.0, "ta_deg": 0.0}
    propagation = {"timeline": {"duration": "1", "unit": "Days"},
                   "integrator": {"method": "DOP853"}}
    return build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        output=OutputPageState(output_dir=r"C:\results", generate_3d_plots=False),
        data_files=DataFilesState(albedo_root=r"C:\data\albedo_models"),
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
        albedo_cfg=albedo_cfg,
    )


def test_build_command_emits_albedo_model_flags() -> None:
    command = _albedo_command(_DummyAlbedoConfig())

    def _val(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert _val("--enable-albedo") == "on"
    assert _val("--albedo-model") == "lambert_facets"
    assert _val("--albedo-mode") == "scaled_dn_grid"
    assert _val("--albedo-const") == "0.11"
    assert _val("--albedo-pressure-coefficient") == "1.3"
    assert _val("--albedo-facet-lat-count") == "24"
    assert _val("--albedo-facet-lon-count") == "48"
    assert _val("--albedo-enable-eclipse") == "on"
    # Grid source -> the surface raster directory must be forwarded.
    assert "--albedo-root" in command


def test_build_command_constant_albedo_skips_albedo_root() -> None:
    class _Const(_DummyAlbedoConfig):
        source = "constant_albedo"

    command = _albedo_command(_Const())
    assert command[command.index("--albedo-mode") + 1] == "constant_albedo"
    # Constant albedo needs no surface raster.
    assert "--albedo-root" not in command


def test_build_command_emits_albedo_require_provider_flag() -> None:
    class _FailClosed(_DummyAlbedoConfig):
        require_provider = True

    command = _albedo_command(_FailClosed())
    assert command[command.index("--albedo-require-provider") + 1] == "on"

    command_default = _albedo_command(_DummyAlbedoConfig())
    assert command_default[command_default.index("--albedo-require-provider") + 1] == "off"


class _DummyThermalConfig:
    mode = "equilibrium_temperature"
    temperature_k = 260.0
    night_temperature_k = 95.0
    emissivity = 0.9
    surface_albedo = 0.15
    ir_coefficient = 1.2
    floor_flux_w_m2 = 5.0
    facet_lat_count = 20
    facet_lon_count = 40


def _minimal_mission_state() -> dict:
    return {
        "orbit": {"mode": "circular", "alt_km": 100.0, "inc_deg": 90.0,
                  "raan_deg": 0.0, "argp_deg": 0.0, "ta_deg": 0.0},
        "propagation": {"timeline": {"duration": "1", "unit": "Days"},
                        "integrator": {"method": "DOP853"}},
    }


def test_build_command_emits_thermal_ir_flags() -> None:
    state = _minimal_mission_state()
    forces = {
        "gravity": {"enabled": True},
        "sun": True, "earth": True, "earth_j2": False,
        "srp": False, "albedo": False, "thermal": True,
        "tides_k2": False, "tides_k3": False, "relativity_1pn": False,
    }
    command = build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=state["orbit"],
        forces=forces,
        propagation=state["propagation"],
        output=OutputPageState(output_dir=r"C:\results", generate_3d_plots=False),
        data_files=DataFilesState(),
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
        thermal_cfg=_DummyThermalConfig(),
    )

    def _val(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert _val("--enable-thermal") == "on"
    assert _val("--thermal-mode") == "equilibrium_temperature"
    assert _val("--thermal-temperature-k") == "260"
    assert _val("--thermal-night-temperature-k") == "95"
    assert _val("--thermal-emissivity") == "0.9"
    assert _val("--thermal-surface-albedo") == "0.15"
    assert _val("--thermal-ir-coefficient") == "1.2"
    assert _val("--thermal-floor-flux-w-m2") == "5"
    assert _val("--thermal-facet-lat-count") == "20"
    assert _val("--thermal-facet-lon-count") == "40"


def test_build_command_thermal_flags_absent_when_disabled() -> None:
    state = _minimal_mission_state()
    forces = {
        "gravity": {"enabled": True},
        "sun": True, "earth": True, "earth_j2": False,
        "srp": False, "albedo": False, "thermal": False,
        "tides_k2": False, "tides_k3": False, "relativity_1pn": False,
    }
    command = build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=state["orbit"],
        forces=forces,
        propagation=state["propagation"],
        output=OutputPageState(output_dir=r"C:\results", generate_3d_plots=False),
        data_files=DataFilesState(),
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
        thermal_cfg=_DummyThermalConfig(),
    )
    assert command[command.index("--enable-thermal") + 1] == "off"
    assert "--thermal-mode" not in command


def test_build_command_emits_tide_value_flags_only_when_set() -> None:
    state = _minimal_mission_state()
    forces = {
        "gravity": {"enabled": True},
        "sun": True, "earth": True, "earth_j2": False,
        "srp": False, "albedo": False, "thermal": False,
        "tides_k2": True, "tides_k3": True, "relativity_1pn": False,
        "tide_k2_value": "0.025",
        "tide_k3_value": "0.009",
        "tide_r_ref_m": "1737400",
        "tide_bodies": "earth,sun",
    }
    command = build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=state["orbit"],
        forces=forces,
        propagation=state["propagation"],
        output=OutputPageState(output_dir=r"C:\results", generate_3d_plots=False),
        data_files=DataFilesState(),
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    def _val(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert _val("--tides-kind") == "k3"
    assert _val("--tide-k2") == "0.025"
    assert _val("--tide-k3") == "0.009"
    assert _val("--tide-r-ref-m") == "1.7374e+06"
    assert _val("--tide-bodies") == "earth,sun"

    # Blank values keep engine defaults: no value flags emitted.
    forces_blank = dict(forces, tide_k2_value="", tide_k3_value="",
                        tide_r_ref_m="", tide_bodies="")
    command_blank = build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=state["orbit"],
        forces=forces_blank,
        propagation=state["propagation"],
        output=OutputPageState(output_dir=r"C:\results", generate_3d_plots=False),
        data_files=DataFilesState(),
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )
    for flag in ("--tide-k2", "--tide-k3", "--tide-r-ref-m", "--tide-bodies"):
        assert flag not in command_blank


def test_autodetect_data_state_understands_repository_folder_names(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    topo_dir = data_root / "topography_models"
    albedo_dir = data_root / "albedo_models"
    kernel_dir = data_root / "ephemeris_models"

    topo_dir.mkdir(parents=True)
    albedo_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)

    (topo_dir / "ldem_64_float.img").write_bytes(b"topography")
    (albedo_dir / "ldam_8_float.img").write_bytes(b"albedo")
    (kernel_dir / "de440.bsp").write_bytes(b"kernel")

    detected_state, messages = autodetect_data_state(tmp_path, DataFilesState())

    assert Path(detected_state.ldem_root) == topo_dir.resolve()
    assert Path(detected_state.albedo_root) == albedo_dir.resolve()
    assert Path(detected_state.kernel_dir) == kernel_dir.resolve()
    assert detected_state.use_ldem_for_albedo is False
    assert any("LDEM auto-filled" in message for message in messages)
    assert any("Kernels auto-filled" in message for message in messages)


def test_autodetect_data_state_prefers_dedicated_albedo_dir_over_legacy_ldem_reuse(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    topo_dir = data_root / "topography_models"
    albedo_dir = data_root / "albedo_models"

    topo_dir.mkdir(parents=True)
    albedo_dir.mkdir(parents=True)

    (topo_dir / "ldem_64_float.img").write_bytes(b"topography")
    (albedo_dir / "ldam_8_float.img").write_bytes(b"albedo")

    initial = DataFilesState(
        ldem_root=str(topo_dir.resolve()),
        albedo_root=str(topo_dir.resolve()),
        use_ldem_for_albedo=True,
    )

    detected_state, _messages = autodetect_data_state(tmp_path, initial)

    assert Path(detected_state.albedo_root) == albedo_dir.resolve()
    assert detected_state.use_ldem_for_albedo is False


def test_iter_label_candidates_accepts_compound_lbl_txt_names(tmp_path: Path) -> None:
    lbl = tmp_path / "ldem_64_float.lbl.txt"
    lbl.write_text("PDS_VERSION_ID = PDS3\n", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("not a label\n", encoding="utf-8")

    labels = _iter_label_candidates(tmp_path)

    assert lbl in labels


def test_ui_gravity_config_clamps_adaptive_rules_to_base_degree() -> None:
    cfg = UIGravityConfig(
        degree=100,
        adaptive_table=[(200.0, 660), (10.0, 1000), (1000.0, 20)],
    )

    cfg.sort_and_validate()

    assert cfg.adaptive_table == [(10.0, 100), (200.0, 100), (1000.0, 20)]


def test_build_batch_command_includes_solver_and_output_controls() -> None:
    orbit = {
        "mode": "hp_ha",
        "hp_km": 100.0,
        "ha_km": 100.0,
        "inc_deg": 90.0,
        "raan_deg": 0.0,
        "argp_deg": 0.0,
        "ta_deg": 0.0,
    }
    forces = {
        "gravity": {"enabled": True},
        "sun": False,
        "earth": False,
        "earth_j2": False,
        "srp": False,
        "albedo": False,
        "thermal": False,
        "tides_k2": False,
        "tides_k3": False,
        "relativity_1pn": False,
    }
    propagation = {
        "timeline": {
            "epoch": "2027-03-02 23:32:37",
            "duration": "0.5",
            "unit": "Days",
        },
        "integrator": {
            "method": "DOP853 (Adaptive)",
            "rtol": "1e-9",
            "dt_out": "15",
            "max_step": "30",
        },
    }
    batch_data = {
        "n_samples": 8,
        "seed": 7,
        "use_gpu": False,
        "output_format": "npz",
        "output_path": r"C:\results\batch_case.npz",
        "dt_s": 20.0,
        "impact_alt_km": 2.0,
    }
    data_state = DataFilesState(
        ldem_root=r"C:\data\topography_models",
        albedo_root=r"C:\data\albedo_models",
        kernel_dir=r"C:\data\ephemeris_models",
        ldem_ppd=16,
    )

    command = build_batch_command(
        python_executable="python",
        batch_runner_path=Path("batch_runner.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        batch_data=batch_data,
        data_files=data_state,
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    assert command[:2] == ["python", "batch_runner.py"]
    assert "--output-dt-s" in command
    assert "15" in command
    assert "--method" in command
    assert "DOP853" in command
    assert "--rtol" in command
    assert "1e-09" in command
    assert "--atol" in command
    assert "1e-12" in command
    assert "--user-max-step-s" in command
    assert "30" in command
    assert "--use-gpu" in command
    assert "off" in command
    assert "2027-03-02T23:32:37Z" in command
    # Torch tuning flags carry safe defaults; UQ report is opt-in only.
    assert command[command.index("--torch-dtype") + 1] == "float64"
    assert command[command.index("--torch-sh-chunk-size") + 1] == "0"
    assert "--uq-report-dir" not in command


def test_build_batch_command_emits_torch_and_uq_flags() -> None:
    orbit = {"mode": "circular", "alt_km": 100.0, "inc_deg": 90.0,
             "raan_deg": 0.0, "argp_deg": 0.0, "ta_deg": 0.0}
    forces = {
        "gravity": {"enabled": True},
        "sun": False, "earth": False, "earth_j2": False,
        "srp": False, "albedo": False, "thermal": False,
        "tides_k2": False, "tides_k3": False, "relativity_1pn": False,
    }
    propagation = {"timeline": {"duration": "1", "unit": "Days"},
                   "integrator": {"method": "DOP853"}}
    batch_data = {
        "n_samples": 8,
        "torch_dtype": "float32",
        "torch_sh_chunk_size": 4096,
        "uq_report_dir": r"C:\results\uq_report",
    }

    command = build_batch_command(
        python_executable="python",
        batch_runner_path=Path("batch_runner.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        batch_data=batch_data,
        data_files=DataFilesState(),
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    assert command[command.index("--torch-dtype") + 1] == "float32"
    assert command[command.index("--torch-sh-chunk-size") + 1] == "4096"
    assert command[command.index("--uq-report-dir") + 1] == r"C:\results\uq_report"


def test_batch_runner_parser_accepts_ui_batch_command() -> None:
    """CLI-parity contract: every flag the UI emits must parse in batch_runner.

    argparse exits with 'unrecognized arguments' on any flag the runner does
    not declare, so parsing the full UI-built command guards the UI→CLI seam
    (the tide value flags were once emitted before the runner accepted them).
    """
    from lunaris.cli.batch_runner import _parse_args

    orbit = {"mode": "circular", "alt_km": 100.0, "inc_deg": 90.0,
             "raan_deg": 0.0, "argp_deg": 0.0, "ta_deg": 0.0}
    forces = {
        "gravity": {"enabled": True},
        "sun": False, "earth": False, "earth_j2": False,
        "srp": False, "albedo": False, "thermal": False,
        "tides_k2": True, "tides_k3": True, "relativity_1pn": False,
        "tide_k2_value": "0.025",
        "tide_k3_value": "0.009",
        "tide_r_ref_m": "1737400",
        "tide_bodies": "earth,sun",
    }
    propagation = {"timeline": {"duration": "1", "unit": "Days"},
                   "integrator": {"method": "DOP853"}}
    batch_data = {
        "n_samples": 8,
        "torch_dtype": "float32",
        "torch_sh_chunk_size": 4096,
        "uq_report_dir": r"C:\results\uq_report",
    }

    command = build_batch_command(
        python_executable="python",
        batch_runner_path=Path("batch_runner.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        batch_data=batch_data,
        data_files=DataFilesState(),
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    args = _parse_args(command[2:])  # skip interpreter + script path
    assert args.tide_k2 == pytest.approx(0.025)
    assert args.tide_k3 == pytest.approx(0.009)
    assert args.tide_r_ref_m == pytest.approx(1_737_400.0)
    assert args.tide_bodies == ("earth", "sun")
    assert args.torch_dtype == "float32"
    assert args.torch_sh_chunk_size == 4096
    assert args.uq_report_dir == r"C:\results\uq_report"


def test_build_command_uses_surrogate_gravity_flags_when_requested() -> None:
    orbit = {
        "mode": "hp_ha",
        "hp_km": 50.0,
        "ha_km": 50.0,
        "inc_deg": 90.0,
        "raan_deg": 0.0,
        "argp_deg": 0.0,
        "ta_deg": 0.0,
    }
    forces = {"gravity": {"enabled": True}}
    propagation = {
        "timeline": {"epoch": "2026-05-03 12:34:56", "duration": "1", "unit": "Days"},
        "integrator": {"method": "DOP853 (Adaptive)", "rtol": "1e-10", "dt_out": "30", "max_step": "120"},
    }
    output = OutputPageState(output_dir=r"C:\results")
    data_state = DataFilesState()
    gravity_cfg = UIGravityConfig(
        degree=100,
        file_path=r"C:\models\moon_660.gfc",
        backend="st_lrps",
        st_lrps_model_dir=r"C:\models\st_lrps_run",
    )

    command = build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        output=output,
        data_files=data_state,
        gravity_cfg=gravity_cfg,
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    assert "--gravity-backend" in command
    assert "st_lrps" in command
    assert "--surrogate-gravity-model-dir" in command
    assert r"C:\models\st_lrps_run" in command
    assert "--gravity-file-path" not in command
    assert "--adaptive-table" not in command


def test_build_batch_command_can_force_surrogate_gravity_override() -> None:
    orbit = {
        "mode": "hp_ha",
        "hp_km": 100.0,
        "ha_km": 100.0,
        "inc_deg": 90.0,
        "raan_deg": 0.0,
        "argp_deg": 0.0,
        "ta_deg": 0.0,
    }
    forces = {"gravity": {"enabled": False}}
    propagation = {
        "timeline": {"epoch": "2027-03-02 23:32:37", "duration": "0.5", "unit": "Days"},
        "integrator": {"method": "DOP853 (Adaptive)", "rtol": "1e-9", "dt_out": "15", "max_step": "30"},
    }
    batch_data = {
        "n_samples": 8,
        "seed": 7,
        "use_gpu": True,
        "gravity_mode_override": "st_lrps",
        "output_format": "npz",
        "output_path": r"C:\results\batch_case.npz",
        "dt_s": 20.0,
        "impact_alt_km": 2.0,
    }

    gravity_cfg = UIGravityConfig(
        degree=100,
        file_path=r"C:\models\moon_660.gfc",
        backend="st_lrps",
        st_lrps_model_dir=r"C:\models\st_lrps_run",
    )

    command = build_batch_command(
        python_executable="python",
        batch_runner_path=Path("batch_runner.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        batch_data=batch_data,
        data_files=DataFilesState(),
        gravity_cfg=gravity_cfg,
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    assert "--enable-sh" in command
    assert "on" in command
    assert "--batch-gravity-mode" in command
    assert "st_lrps" in command
    assert "--surrogate-gravity-model-dir" in command
