"""The single-run CLI reports the outputs it did not deliver and resolves its grid.

Each test pins one silent trap found in the 2026-10-28 pre-run audit:

* the 3D render never succeeded on matplotlib 3.10 (``add_collection3d`` of an
  empty collection raises in ``auto_scale_xyz``) while ``lunaris`` printed
  ``[OK] Finished.`` and exited 0;
* a requested output that failed was not recorded in the run's diagnostics;
* ``--samples-per-period`` left ``output_dt_s`` unset when the ephemeris tables
  were built, which need a fixed spacing: ``[FATAL] Ephemeris init failed``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.cli import run as run_mod
from lunaris.cli.common_args import apply_args_to_config
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
PERIOD_S = 2.0 * math.pi * math.sqrt(R0_M**3 / MU_MOON)


def _cfg(tmp_path: Path, *, make_3d: bool = False) -> SimConfig:
    return SimConfig(
        gravity=GravityConfig(file_path="gravity.tab"),
        spice=SimpleNamespace(include_third_body=True, kernels=()),
        initial_state=InitialState(x=R0_M, y=0.0, z=0.0, vx=0.0, vy=V0_M_S, vz=0.0),
        time=TimeConfig(duration_s=600.0, output_dt_s=60.0),
        flags=PerturbationFlags(enable_sh=False),
        output=OutputConfig(out_dir=tmp_path, make_3d_plots=make_3d),
    )


def _result() -> PropagationResult:
    y = np.tile(np.array([R0_M, 0.0, 0.0, 0.0, V0_M_S, 0.0]), (2, 1))
    return PropagationResult(
        t=np.array([0.0, 60.0]), y=y, ode=SimpleNamespace(success=True, message=""),
        stopped_early=False, stop_reason=None, t_stop_s=None,
    )


def _patch_pipeline(monkeypatch, cfg: SimConfig, *, ephemeris=None) -> None:
    monkeypatch.setattr(run_mod, "load_runtime_config", lambda args: apply_args_to_config(cfg, args))
    monkeypatch.setattr(run_mod, "build_gravity_provider", lambda cfg: (None, float(MU_MOON)))
    monkeypatch.setattr(
        run_mod, "build_surface_provider_if_needed",
        lambda cfg, args: run_mod.SurfaceSetup(provider=None, topo_grid=None, topo_requested=False),
    )
    monkeypatch.setattr(
        run_mod, "build_ephemeris_if_needed",
        ephemeris or (lambda cfg, topo_requested: None),
    )
    monkeypatch.setattr(run_mod, "print_summary", lambda *a, **k: None)
    monkeypatch.setattr(run_mod, "build_engine", lambda cfg, **k: object())
    monkeypatch.setattr(
        run_mod, "run_propagation",
        lambda engine, cfg, **k: run_mod.PropagationRun(result=_result(), elapsed_s=0.1),
    )
    monkeypatch.setattr(run_mod, "render_reports", lambda **k: None)


# --------------------------------------------------------------------------- 3D render


def _history() -> dict:
    t = np.linspace(0.0, 7200.0, 200)
    w = math.sqrt(MU_MOON / R0_M**3)
    r = np.vstack([R0_M * np.cos(w * t), R0_M * np.sin(w * t), 0.0 * t])
    v = np.vstack([-R0_M * w * np.sin(w * t), R0_M * w * np.cos(w * t), 0.0 * t])
    return {"t_s": t, "r_m": r, "v_mps": v, "meta": {"body_radius_m": 1.7374e6, "mu_m3s2": MU_MOON}}


def test_orbit_animation_builds_its_empty_collections_and_reaches_the_save_step(
    monkeypatch, tmp_path
) -> None:
    """Building the scene raised on matplotlib 3.10; the encode itself is not under test."""
    from lunaris.visualization import orbit_animation

    updated: list[int] = []

    class _FakeAnimation:
        def __init__(self, fig, update, frames, **kwargs) -> None:
            self._update, self._frames = update, int(frames)

        def save(self, path, **kwargs) -> None:
            for i in (0, 5, self._frames - 1):  # the trail collections are filled per frame
                self._update(i)
                updated.append(i)
            Path(path).write_bytes(b"GIF89a")

    monkeypatch.setattr(orbit_animation.animation, "FuncAnimation", _FakeAnimation)
    out = tmp_path / "orbit_3d.gif"

    produced = orbit_animation.render_orbit_animation_from_history(
        _history(), str(out), frames=120, moon_mesh_u=12, moon_mesh_v=6, n_stars=10, dpi=20
    )

    assert produced == str(out)
    assert out.is_file()
    assert updated == [0, 5, 119]


def test_a_renderer_that_produced_nothing_is_a_failure(monkeypatch, tmp_path) -> None:
    from lunaris.visualization import orbit_animation

    monkeypatch.setattr(orbit_animation, "render_orbit_animation", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="produced no file"):
        run_mod.render_optional_3d(_result(), _cfg(tmp_path, make_3d=True), tmp_path)


def test_failed_3d_render_is_recorded_and_not_reported_as_finished(
    monkeypatch, tmp_path, capsys
) -> None:
    _patch_pipeline(monkeypatch, _cfg(tmp_path, make_3d=True))

    def _boom(*args, **kwargs):
        raise RuntimeError("3D exploded")

    monkeypatch.setattr(run_mod, "render_optional_3d", _boom)

    code = run_mod.run_pipeline(parse_args([]))

    out = capsys.readouterr().out
    assert code == 1
    assert "[OK] Finished." not in out
    assert "[FAIL] Propagation completed but 1 requested output(s) failed" in out
    (diagnostics_path,) = tmp_path.rglob("run_diagnostics.json")
    failures = json.loads(diagnostics_path.read_text(encoding="utf-8"))["output_failures"]
    assert [f["stage"] for f in failures] == ["3D render failed"]
    assert "3D exploded" in failures[0]["error"]
    assert list(tmp_path.rglob("run_config.json"))  # the run record is kept


def test_run_with_every_output_delivered_still_finishes_cleanly(
    monkeypatch, tmp_path, capsys
) -> None:
    _patch_pipeline(monkeypatch, _cfg(tmp_path))
    monkeypatch.setattr(run_mod, "render_optional_3d", lambda *a, **k: None)

    code = run_mod.run_pipeline(parse_args([]))

    assert code == 0
    assert "[OK] Finished." in capsys.readouterr().out
    (diagnostics_path,) = tmp_path.rglob("run_diagnostics.json")
    assert "output_failures" not in json.loads(diagnostics_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- --samples-per-period


def test_samples_per_period_gives_the_ephemeris_a_fixed_spacing(monkeypatch, tmp_path, capsys) -> None:
    seen: list[float | None] = []

    def _ephemeris(cfg, topo_requested):
        # What EphemerisManager.from_time_and_spice enforces.
        seen.append(cfg.time.output_dt_s)
        if cfg.time.output_dt_s is None:
            raise ValueError("TimeConfig.output_dt_s is None, but ephemeris tables require a fixed dt.")
        return None

    _patch_pipeline(monkeypatch, _cfg(tmp_path), ephemeris=_ephemeris)

    code = run_mod.run_pipeline(parse_args(["--samples-per-period", "100"]))

    assert code == 0, capsys.readouterr().out
    assert seen == [pytest.approx(PERIOD_S / 100.0, rel=1e-12)]
