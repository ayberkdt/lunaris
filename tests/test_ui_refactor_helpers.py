# -*- coding: utf-8 -*-
"""
Focused regression tests for the modular UI helper layer.

These tests intentionally avoid booting the full Qt application. The goal is to
verify the new pure/helper modules introduced during the UI refactor:
- command construction from page/config snapshots
- repository-aware auto-detection of data directories
"""

from __future__ import annotations

from pathlib import Path

from lunaris.ui.widgets.command_builder import build_command, build_mc_command
from lunaris.ui.widgets.data_files_page import DataFilesState
from lunaris.ui.widgets.force_models_page import UIGravityConfig
from lunaris.ui.widgets.result_exports_page import OutputPageState
from lunaris.ui.widgets.session_persistence import autodetect_data_state
from lunaris.loaders.io_surface import _iter_label_candidates


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
        ldem_root=r"C:\data\topografy_models",
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
    assert "--enable-relativity-1pn" in command
    assert "--albedo-root" not in command
    assert "--save-csv" not in command


def test_autodetect_data_state_understands_repository_folder_names(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    topo_dir = data_root / "topografy_models"
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
    topo_dir = data_root / "topografy_models"
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


def test_build_mc_command_includes_solver_and_output_controls() -> None:
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
    mc_data = {
        "n_samples": 8,
        "seed": 7,
        "use_gpu": False,
        "output_format": "npz",
        "output_path": r"C:\results\mc_case.npz",
        "dt_s": 20.0,
        "impact_alt_km": 2.0,
    }
    data_state = DataFilesState(
        ldem_root=r"C:\data\topografy_models",
        albedo_root=r"C:\data\albedo_models",
        kernel_dir=r"C:\data\ephemeris_models",
        ldem_ppd=16,
    )

    command = build_mc_command(
        python_executable="python",
        mc_runner_path=Path("mc_runner.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        mc_data=mc_data,
        data_files=data_state,
        gravity_cfg=_DummyGravityConfig(),
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    assert command[:2] == ["python", "mc_runner.py"]
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


def test_build_mc_command_can_force_surrogate_gravity_override() -> None:
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
    mc_data = {
        "n_samples": 8,
        "seed": 7,
        "use_gpu": True,
        "gravity_mode_override": "st_lrps",
        "output_format": "npz",
        "output_path": r"C:\results\mc_case.npz",
        "dt_s": 20.0,
        "impact_alt_km": 2.0,
    }

    gravity_cfg = UIGravityConfig(
        degree=100,
        file_path=r"C:\models\moon_660.gfc",
        backend="st_lrps",
        st_lrps_model_dir=r"C:\models\st_lrps_run",
    )

    command = build_mc_command(
        python_executable="python",
        mc_runner_path=Path("mc_runner.py"),
        orbit=orbit,
        forces=forces,
        propagation=propagation,
        mc_data=mc_data,
        data_files=DataFilesState(),
        gravity_cfg=gravity_cfg,
        solver_cfg=_DummySolverConfig(),
        spacecraft_cfg=_DummySpacecraftConfig(),
    )

    assert "--enable-sh" in command
    assert "on" in command
    assert "--mc-gravity-mode" in command
    assert "st_lrps" in command
    assert "--surrogate-gravity-model-dir" in command
