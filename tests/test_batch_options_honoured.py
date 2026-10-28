"""``lunaris-batch`` options are honoured, refused, or recorded as not applied.

Each test pins one silent trap found in the 2026-10-28 pre-run audit:

* ``--enable-sh off`` was turned back on whenever ``--batch-backend`` or
  ``--batch-gravity-mode`` named an SH path, so the run equalled the SH-on run;
* ``--batch-dt-s`` has no effect on the CPU backend, which steps with
  ``--method`` / ``--user-max-step-s``, yet the archive said ``adaptive (DOP853)``
  whatever ran;
* the archive carried no units, frame label, UTC start epoch, ``sc_samples``
  column names or software identity, and ``frame_mode`` read ``unknown`` on CPU;
* a run with no path option replaced the shared default archive.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lunaris.batch as batch_pkg
from lunaris.batch import engine as batch_engine
from lunaris.batch import storage
from lunaris.cli import batch_runner
from lunaris.common.batch_defs import BatchPropagationResult
from lunaris.common.constants import MU_MOON
from lunaris.common.type_defs import GravityConfig, InitialState, PerturbationFlags, TimeConfig
from lunaris.core.config import SimConfig

R0_M = 1_838_000.0
V0_M_S = math.sqrt(MU_MOON / R0_M)


def _cfg() -> SimConfig:
    return SimConfig(
        gravity=GravityConfig(file_path="gravity.tab"),
        spice=SimpleNamespace(include_third_body=True, kernels=()),
        initial_state=InitialState(x=R0_M, y=0.0, z=0.0, vx=0.0, vy=V0_M_S, vz=0.0),
        time=TimeConfig(duration_s=120.0, output_dt_s=60.0),
        flags=PerturbationFlags(enable_sh=False),
    )


def _result(diagnostics: dict | None = None) -> BatchPropagationResult:
    n = 2
    return BatchPropagationResult(
        t=np.array([0.0, 60.0, 120.0]),
        Y=np.tile(np.array([R0_M, 0.0, 0.0, 0.0, V0_M_S, 0.0]), (3, n, 1)),
        sc_samples=np.ones((n, 4)),
        impact_mask=np.zeros(n),
        t_impact=np.full(n, np.nan),
        valid_mask=np.array([True, True]),
        diagnostics=dict(diagnostics or {}),
    )


class _Run:
    """Stub engine capturing what the runner built."""

    def __init__(self, monkeypatch, diagnostics: dict | None = None) -> None:
        self.engines: list[tuple[SimConfig, object]] = []
        outer = self

        class _Engine:
            def __init__(self, cfg, batch_cfg, **kwargs) -> None:
                outer.engines.append((cfg, batch_cfg))

            def run(self) -> BatchPropagationResult:
                return _result(diagnostics)

        monkeypatch.setattr(batch_runner, "load_default_config", lambda **kwargs: _cfg())
        monkeypatch.setattr(batch_pkg, "BatchPropagationEngine", _Engine)

    def main(self, tmp_path: Path, *extra: str) -> int:
        return batch_runner.main(
            [
                "--n-samples", "2", "--use-gpu", "off", "--batch-output-format", "npz",
                "--batch-output-path", str(tmp_path / "out.npz"),
                *extra,
            ]
        )


# --------------------------------------------------------------------------- --enable-sh


@pytest.mark.parametrize("extra", [
    ("--batch-backend", "torch_cuda_sh"),
    ("--batch-backend", "cpu_sh"),
    ("--batch-backend", "gpu_st_lrps_potential"),
    ("--batch-gravity-mode", "classic_sh"),
    ("--batch-gravity-mode", "st_lrps"),
])
def test_enable_sh_off_with_an_sh_backend_is_refused(monkeypatch, tmp_path, capsys, extra) -> None:
    run = _Run(monkeypatch)

    code = run.main(tmp_path, "--enable-sh", "off", *extra)

    out = capsys.readouterr().out
    assert code == 1
    assert run.engines == []
    assert "[BATCH][FATAL]" in out and "--enable-sh off" in out


def test_enable_sh_off_alone_stays_off(monkeypatch, tmp_path) -> None:
    run = _Run(monkeypatch)

    assert run.main(tmp_path, "--enable-sh", "off") == 0

    assert run.engines[0][0].flags.enable_sh is False


def test_enable_sh_on_with_an_sh_backend_is_unchanged(monkeypatch, tmp_path) -> None:
    run = _Run(monkeypatch)

    assert run.main(tmp_path, "--enable-sh", "on", "--batch-backend", "cpu_sh") == 0

    assert run.engines[0][0].flags.enable_sh is True


# --------------------------------------------------------------------------- --batch-dt-s


_CPU_DIAG = {
    "integrator": "fixed-step RK4",
    "batch_dt_s_applied": False,
    "requested_dt_s": 10.0,
}


def test_batch_dt_s_not_applied_on_cpu_is_reported(monkeypatch, tmp_path, capsys) -> None:
    run = _Run(monkeypatch, _CPU_DIAG)

    assert run.main(tmp_path, "--method", "RK4", "--batch-dt-s", "10") == 0

    out = capsys.readouterr().out
    assert "[BATCH][WARN] --batch-dt-s 10 was not applied" in out
    assert "fixed-step RK4" in out
    metrics = json.loads(out.split("[BATCH_METRICS] ", 1)[1].splitlines()[0])
    assert metrics["integrator"] == "fixed-step RK4"
    assert metrics["batch_dt_s_applied"] is False
    assert metrics["requested_dt_s"] == 10.0


def test_default_batch_dt_s_on_cpu_is_not_a_warning(monkeypatch, tmp_path, capsys) -> None:
    run = _Run(monkeypatch, _CPU_DIAG)

    assert run.main(tmp_path) == 0

    assert "[BATCH][WARN]" not in capsys.readouterr().out
    assert run.engines[0][1].dt_s == 60.0


@pytest.mark.parametrize(("method", "label"), [
    ("RK4", "fixed-step RK4"),
    ("DOP853", "adaptive (DOP853)"),
])
def test_cpu_integrator_label_follows_the_propagator_method(method, label) -> None:
    cfg = _cfg()
    cfg = replace(cfg, propagator=replace(cfg.propagator, method=method, user_max_step_s=15.0))

    meta = batch_engine._cpu_integrator_provenance(cfg, duration_s=120.0)

    assert meta["integrator"] == label
    assert meta["batch_dt_s_applied"] is False
    assert meta["cpu_user_max_step_s"] == 15.0


# --------------------------------------------------------------------------- self-describing archive


def test_archive_self_description_names_units_frames_epoch_and_software() -> None:
    cfg = _cfg()
    cfg = replace(cfg, time=replace(cfg.time, start_date="2027-03-03T02:32:37+03:00"))
    tables = SimpleNamespace(inertial_frame="J2000", observer="MOON", fixed_frame="MOON_PA")

    meta = batch_engine._archive_self_description(cfg, SimpleNamespace(tables=tables))

    assert meta["start_epoch_utc"] == "2027-03-02T23:32:37Z"
    assert meta["Y_columns"] == ["x", "y", "z", "vx", "vy", "vz"]
    assert meta["Y_units"] == ["m", "m", "m", "m/s", "m/s", "m/s"]
    assert meta["Y_frame"] == "MOON-centred J2000"
    assert meta["impact_position_fixed_frame"] == "MOON_PA"
    assert meta["sc_samples_columns"] == ["mass_kg", "area_m2", "cd", "cr"]
    assert meta["t_units"] == "s since start_epoch_utc"
    assert meta["lunaris_version"]
    assert meta["git_commit"]


def test_archive_self_description_without_ephemeris_does_not_invent_a_frame() -> None:
    meta = batch_engine._archive_self_description(_cfg(), None)

    assert "J2000" not in meta["Y_frame"]
    assert "no ephemeris" in meta["Y_frame"]


@pytest.fixture(scope="module")
def cpu_rk4_archive(tmp_path_factory) -> dict:
    """A real tiny CPU run (RK4) and the manifest it archived."""
    path = tmp_path_factory.mktemp("batch") / "cpu.npz"
    code = batch_runner.main(
        [
            "--start-date", "2027-03-02T23:32:37", "--days", "0.0002", "--output-dt-s", "5",
            "--hp-km", "100", "--ha-km", "100", "--inc-deg", "90", "--raan-deg", "0",
            "--argp-deg", "0", "--ta-deg", "0", "--degree", "4", "--enable-sh", "on",
            "--enable-3rd-body-sun", "off", "--enable-3rd-body-earth", "off",
            "--enable-earth-j2", "off", "--enable-srp", "off", "--enable-albedo", "off",
            "--enable-thermal", "off", "--enable-tides", "off", "--enable-relativity-1pn", "off",
            "--method", "RK4", "--user-max-step-s", "15",
            "--n-samples", "2", "--seed", "11", "--use-gpu", "off", "--batch-dt-s", "10",
            "--batch-output-format", "npz", "--batch-output-path", str(path),
        ]
    )
    assert code == 0
    with np.load(str(path), allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"]))


@pytest.mark.requires_data
def test_cpu_archive_records_the_integrator_that_ran(cpu_rk4_archive) -> None:
    assert cpu_rk4_archive["integrator"] == "fixed-step RK4"
    assert cpu_rk4_archive["batch_dt_s_applied"] is False
    assert cpu_rk4_archive["requested_dt_s"] == 10.0
    assert cpu_rk4_archive["cpu_user_max_step_s"] == 15.0


@pytest.mark.requires_data
def test_cpu_archive_is_self_describing(cpu_rk4_archive) -> None:
    meta = cpu_rk4_archive
    assert meta["frame_mode"] == "moon_fixed_ephemeris"
    assert meta["start_epoch_utc"] == "2027-03-02T23:32:37Z"
    assert meta["Y_units"] == ["m", "m", "m", "m/s", "m/s", "m/s"]
    assert "J2000" in meta["Y_frame"]
    assert meta["sc_samples_columns"] == ["mass_kg", "area_m2", "cd", "cr"]
    assert meta["lunaris_version"] and meta["git_commit"]


# --------------------------------------------------------------------------- default archive


def test_existing_archive_warning_names_its_sha256(tmp_path, caplog) -> None:
    old = tmp_path / "batch_output.h5"
    old.write_bytes(b"previous archive bytes")

    with caplog.at_level(logging.WARNING, logger="lunaris.batch.storage"):
        storage._check_existing_archive(old, True)

    assert hashlib.sha256(b"previous archive bytes").hexdigest() in caplog.text
    assert "will be replaced" in caplog.text


def _existing_default_archive(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "outputs" / "ensemble" / "batch_output.h5"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"results of an earlier run")
    return archive


def test_run_without_a_path_option_does_not_replace_the_default_archive(
    monkeypatch, tmp_path, capsys
) -> None:
    archive = _existing_default_archive(tmp_path, monkeypatch)
    run = _Run(monkeypatch)

    code = batch_runner.main(["--n-samples", "2", "--use-gpu", "off"])

    assert code == 1
    assert run.engines == []
    assert archive.read_bytes() == b"results of an earlier run"
    assert "already exists" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["--out-dir", "elsewhere"],
    ["--batch-output-path", "mine.h5"],
    ["--batch-output-path", "outputs/ensemble/batch_output.h5"],
])
def test_a_chosen_path_is_still_replaced_on_purpose(monkeypatch, tmp_path, extra) -> None:
    _existing_default_archive(tmp_path, monkeypatch)
    run = _Run(monkeypatch)

    code = batch_runner.main(["--n-samples", "2", "--use-gpu", "off", *extra])

    assert code == 0
    assert run.engines[0][1].overwrite_output is True
