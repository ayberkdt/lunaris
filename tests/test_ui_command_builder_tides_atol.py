"""UI -> CLI seam: tide Love numbers and the Propagation page's atol.

Regression tests for three defects in ``lunaris.ui.core.command_builder``:

* k3 tides with a blank k3 field built a command the runner refuses
  (``--tides-kind k3 requires an explicit --tide-k3 value``, exit 2): the
  engine has no default degree-3 Love number.
* A Love number of 0 (which the CLI accepts: ``--tide-k2/--tide-k3 >= 0``)
  was dropped as "invalid" because the builder required a positive value.
* The absolute tolerance typed on the Propagation page never reached the
  run; the command always carried ``solver_cfg.atol``.

The single-run commands are parsed with ``lunaris.cli.options.parse_args``,
the parser the launched process runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# lunaris.ui imports PySide6; without it this module must skip, not error.
pytest.importorskip("PySide6")

from lunaris.cli.options import parse_args  # noqa: E402
from lunaris.ui.core.command_builder import (  # noqa: E402
    build_batch_command,
    build_command,
    build_preflight_snapshot,
)
from lunaris.ui.pages.data_files_page import DataFilesState  # noqa: E402
from lunaris.ui.pages.result_exports_page import OutputPageState  # noqa: E402


class _Gravity:
    degree = 20
    file_path = ""
    backend = "classic_sh"
    st_lrps_model_dir = ""
    adaptive_enabled = False
    adaptive_table = None


class _Solver:
    rtol = 1e-10
    atol = 1e-12
    max_step = 3600.0


class _Spacecraft:
    mass_kg = 1000.0
    area_m2 = 5.0
    cd = 2.2
    cr = 1.5


_ORBIT = {"mode": "circular", "alt_km": 100.0, "inc_deg": 90.0,
          "raan_deg": 0.0, "argp_deg": 0.0, "ta_deg": 0.0}


def _forces(**tides: object) -> dict[str, object]:
    forces: dict[str, object] = {
        "gravity": {"enabled": True},
        "sun": True, "earth": True, "earth_j2": False,
        "srp": False, "albedo": False, "thermal": False,
        "tides_k2": True, "tides_k3": False, "relativity_1pn": False,
    }
    forces.update(tides)
    return forces


def _propagation(atol: str = "") -> dict[str, object]:
    integrator = {"method": "DOP853 (Adaptive)", "rtol": "1e-10",
                  "dt_out": "60", "max_step": "120"}
    if atol:
        integrator["atol"] = atol
    return {"timeline": {"duration": "1", "unit": "Days"}, "integrator": integrator}


def _single(forces: dict[str, object], propagation: dict[str, object] | None = None,
            warnings: list[str] | None = None) -> list[str]:
    return build_command(
        python_executable="python",
        main_script_path=Path("main.py"),
        orbit=_ORBIT,
        forces=forces,
        propagation=propagation or _propagation(),
        output=OutputPageState(output_dir="", generate_3d_plots=False),
        data_files=DataFilesState(),
        gravity_cfg=_Gravity(),
        solver_cfg=_Solver(),
        spacecraft_cfg=_Spacecraft(),
        log_warning=None if warnings is None else warnings.append,
    )


def _batch(forces: dict[str, object], propagation: dict[str, object] | None = None) -> list[str]:
    return build_batch_command(
        python_executable="python",
        batch_runner_path=Path("batch_runner.py"),
        orbit=_ORBIT,
        forces=forces,
        propagation=propagation or _propagation(),
        batch_data={"n_samples": 4, "use_gpu": False},
        data_files=DataFilesState(),
        gravity_cfg=_Gravity(),
        solver_cfg=_Solver(),
        spacecraft_cfg=_Spacecraft(),
    )


def _val(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("builder", [_single, _batch])
def test_k3_tides_without_a_k3_value_are_refused_in_the_ui(builder) -> None:
    forces = _forces(tides_k3=True, tide_k3_value="")
    with pytest.raises(ValueError, match="k3"):
        builder(forces)


@pytest.mark.parametrize("builder", [_single, _batch])
def test_k3_tides_with_invalid_k3_text_are_refused(builder) -> None:
    forces = _forces(tides_k3=True, tide_k3_value="abc")
    with pytest.raises(ValueError, match="k3"):
        builder(forces)


def test_k3_tides_with_a_value_build_a_command_the_cli_accepts() -> None:
    command = _single(_forces(tides_k3=True, tide_k3_value="0.009"))
    args = parse_args(command[2:])
    assert args.tides_kind == "k3"
    assert args.tide_k3 == pytest.approx(0.009)


def test_love_number_zero_is_accepted_like_the_cli() -> None:
    warnings: list[str] = []
    command = _single(
        _forces(tides_k3=True, tide_k2_value="0", tide_k3_value="0"), warnings=warnings
    )
    assert _val(command, "--tide-k2") == "0"
    assert _val(command, "--tide-k3") == "0"
    assert warnings == []
    args = parse_args(command[2:])
    assert args.tide_k2 == 0.0
    assert args.tide_k3 == 0.0


def test_zero_reference_radius_is_still_rejected() -> None:
    warnings: list[str] = []
    command = _single(_forces(tide_r_ref_m="0"), warnings=warnings)
    assert "--tide-r-ref-m" not in command
    assert any("--tide-r-ref-m" in message for message in warnings)


@pytest.mark.parametrize("raw", ["nan", "inf", "-0.1"])
def test_non_finite_or_negative_love_numbers_are_skipped(raw: str) -> None:
    warnings: list[str] = []
    command = _single(_forces(tide_k2_value=raw), warnings=warnings)
    assert "--tide-k2" not in command
    assert any("--tide-k2" in message for message in warnings)


def test_k2_only_with_blank_values_keeps_engine_defaults() -> None:
    command = _single(_forces(tide_k2_value="", tide_k3_value="", tide_r_ref_m=""))
    for flag in ("--tide-k2", "--tide-k3", "--tide-r-ref-m"):
        assert flag not in command
    parse_args(command[2:])


def test_propagation_page_atol_reaches_the_single_run() -> None:
    command = _single(_forces(), _propagation(atol="1e-11"))
    assert float(_val(command, "--atol")) == pytest.approx(1e-11)
    assert parse_args(command[2:]).atol == pytest.approx(1e-11)


def test_propagation_page_atol_reaches_the_batch_run() -> None:
    command = _batch(_forces(), _propagation(atol="1e-11"))
    assert float(_val(command, "--atol")) == pytest.approx(1e-11)


def test_blank_page_atol_falls_back_to_solver_settings() -> None:
    command = _single(_forces(), _propagation(atol=""))
    assert float(_val(command, "--atol")) == pytest.approx(_Solver.atol)


def test_propagation_page_atol_reaches_the_preflight_snapshot() -> None:
    snapshot = build_preflight_snapshot(
        orbit=_ORBIT,
        forces=_forces(),
        propagation=_propagation(atol="1e-11"),
        output=OutputPageState(output_dir="", generate_3d_plots=False),
        data_files=DataFilesState(),
        spacecraft_cfg=_Spacecraft(),
        solver_cfg=_Solver(),
        gravity_cfg=_Gravity(),
        albedo_cfg=None,
    )
    assert snapshot["atol"] == pytest.approx(1e-11)


def test_batch_command_reports_invalid_solver_text() -> None:
    """Regression: the batch builder replaced invalid max step / rtol / atol
    text with defaults without a word (the single-run builder warns)."""
    warnings: list[str] = []
    propagation = {
        "timeline": {"duration": "1", "unit": "Days"},
        "integrator": {"method": "DOP853 (Adaptive)", "rtol": "abc", "atol": "x", "max_step": "fast"},
    }
    build_batch_command(
        python_executable="python",
        batch_runner_path=Path("batch_runner.py"),
        orbit=_ORBIT,
        forces=_forces(),
        propagation=propagation,
        batch_data={"n_samples": 4, "use_gpu": False},
        data_files=DataFilesState(),
        gravity_cfg=_Gravity(),
        solver_cfg=_Solver(),
        spacecraft_cfg=_Spacecraft(),
        log_warning=warnings.append,
    )
    joined = " ".join(warnings)
    assert "max step" in joined and "rtol" in joined and "atol" in joined


def test_batch_command_keeps_full_precision_solver_values() -> None:
    """Regression: ``:g`` formatting kept six significant digits."""
    propagation = {
        "timeline": {"duration": "1", "unit": "Days"},
        "integrator": {"method": "DOP853 (Adaptive)", "rtol": "1.23456789e-10",
                       "atol": "1e-12", "max_step": "123.4567891"},
    }
    command = _batch(_forces(), propagation)
    assert float(_val(command, "--user-max-step-s")) == 123.4567891
    assert float(_val(command, "--rtol")) == 1.23456789e-10
    assert _val(command, "--atol") == "1e-12"
