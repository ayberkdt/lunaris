"""The single-run CLI reports what actually happened.

* A SciPy integration that fails returns a partial ``PropagationResult`` with
  ``stop_reason == "integration failed"`` instead of raising; the runner used
  to print ``[OK] Finished.`` and exit 0 for it.
* A requested SH degree above the gravity file's maximum is clamped by the
  model factory; the runner used to record the requested degree.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.cli import run as run_mod
from lunaris.cli.options import parse_args
from lunaris.common.constants import MU_MOON
from lunaris.common.type_defs import (
    GravityConfig,
    InitialState,
    PerturbationFlags,
    PropagationResult,
    TimeConfig,
)
from lunaris.core.config import OutputConfig, SimConfig

R0_M = 1_838_000.0
V0_M_S = float(np.sqrt(MU_MOON / R0_M))


def _cfg(tmp_path: Path, *, gravity_file: str = "gravity.tab", degree: int | None = None,
         enable_sh: bool = False) -> SimConfig:
    return SimConfig(
        gravity=GravityConfig(file_path=gravity_file, degree=degree),
        spice=SimpleNamespace(include_third_body=True, kernels=()),
        initial_state=InitialState(x=R0_M, y=0.0, z=0.0, vx=0.0, vy=V0_M_S, vz=0.0),
        time=TimeConfig(duration_s=600.0, output_dt_s=60.0),
        flags=PerturbationFlags(enable_sh=enable_sh),
        output=OutputConfig(out_dir=tmp_path, make_3d_plots=False),
    )


def _result(*, failed: bool) -> PropagationResult:
    t = np.array([0.0, 60.0])
    y = np.tile(np.array([R0_M, 0.0, 0.0, 0.0, V0_M_S, 0.0]), (2, 1))
    ode = SimpleNamespace(success=not failed, message="Required step size is less than spacing")
    return PropagationResult(
        t=t,
        y=y,
        ode=ode,
        stopped_early=failed,
        stop_reason="integration failed" if failed else None,
        t_stop_s=60.0 if failed else None,
    )


def _patch_pipeline(monkeypatch, tmp_path: Path, result: PropagationResult) -> list[str]:
    reports: list[str] = []
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(run_mod, "load_runtime_config", lambda args: cfg)
    monkeypatch.setattr(run_mod, "build_gravity_provider", lambda cfg: (None, float(MU_MOON)))
    monkeypatch.setattr(
        run_mod,
        "build_surface_provider_if_needed",
        lambda cfg, args: run_mod.SurfaceSetup(provider=None, topo_grid=None, topo_requested=False),
    )
    monkeypatch.setattr(run_mod, "build_ephemeris_if_needed", lambda cfg, topo_requested: None)
    monkeypatch.setattr(run_mod, "print_summary", lambda *a, **k: None)
    monkeypatch.setattr(run_mod, "build_engine", lambda cfg, **k: object())
    monkeypatch.setattr(
        run_mod, "run_propagation",
        lambda engine, cfg, **k: run_mod.PropagationRun(result=result, elapsed_s=0.1),
    )
    monkeypatch.setattr(run_mod, "render_reports", lambda **k: reports.append("report"))
    return reports


def test_failed_integration_exits_nonzero_and_skips_the_report(monkeypatch, tmp_path, capsys) -> None:
    reports = _patch_pipeline(monkeypatch, tmp_path, _result(failed=True))

    code = run_mod.run_pipeline(parse_args([]))

    out = capsys.readouterr().out
    assert code == 1
    assert "[OK] Finished." not in out
    assert "[FAIL] Integration failed" in out
    assert "Required step size" in out
    assert reports == []
    # The partial run record is kept for inspection.
    assert list(tmp_path.rglob("run_config.json"))


def test_completed_integration_still_succeeds(monkeypatch, tmp_path, capsys) -> None:
    reports = _patch_pipeline(monkeypatch, tmp_path, _result(failed=False))

    code = run_mod.run_pipeline(parse_args([]))

    assert code == 0
    assert "[OK] Finished." in capsys.readouterr().out
    assert reports == ["report"]


def test_event_stop_is_not_an_integration_failure() -> None:
    result = _result(failed=False)
    result.stopped_early = True
    result.stop_reason = "impact"
    assert run_mod.integration_failure_message(result) is None


def _write_degree4_shadr(path: Path) -> Path:
    rows = [f"{n}, {m}, {1e-6 / n:.6e}, {(5e-7 / n) if m else 0.0:.6e}"
            for n in range(1, 5) for m in range(0, n + 1)]
    path.write_text("1738.0, 4902.8, 0, 4, 4, 1, 0, 0\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_requested_degree_above_file_maximum_records_the_effective_degree(tmp_path, capsys) -> None:
    gravity_file = _write_degree4_shadr(tmp_path / "deg4.tab")
    cfg = _cfg(tmp_path, gravity_file=str(gravity_file), degree=10, enable_sh=True)

    gravity_core, _mu = run_mod.build_gravity_provider(cfg)
    assert gravity_core.degree_max == 4  # the factory evaluates degree 4, not 10

    recorded = run_mod.record_effective_sh_degree(cfg, gravity_core)

    assert recorded.gravity.degree == 4
    out = capsys.readouterr().out
    assert "[WARN]" in out and "10" in out and "4" in out


def test_requested_degree_within_file_maximum_is_unchanged(tmp_path, capsys) -> None:
    gravity_file = _write_degree4_shadr(tmp_path / "deg4.tab")
    cfg = _cfg(tmp_path, gravity_file=str(gravity_file), degree=4, enable_sh=True)

    gravity_core, _mu = run_mod.build_gravity_provider(cfg)

    assert run_mod.record_effective_sh_degree(cfg, gravity_core) is cfg
    assert "[WARN]" not in capsys.readouterr().out


@pytest.mark.parametrize("gravity_core", [None])
def test_sh_off_leaves_the_degree_alone(tmp_path, gravity_core) -> None:
    cfg = _cfg(tmp_path, degree=10)
    assert run_mod.record_effective_sh_degree(cfg, gravity_core) is cfg
