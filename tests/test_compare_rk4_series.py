"""tools/compare_rk4_series.py: pairwise RK4 trajectory differences.

The tool reproduced the 2026-09-23 Plan D comparison (comparison_rk4_dt15) field
for field. These tests pin the arithmetic on synthetic caches: a known offset
along each RIC axis, the scenario overlap, and the refusal to compare series
integrated with different RK4 steps (the integrator error would then not cancel).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "compare_rk4_series.py"


def _tool():
    spec = importlib.util.spec_from_file_location("compare_rk4_series", _TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _orbit(n: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 7000.0, n)
    w = 2.0 * np.pi / 7000.0
    r = 1.838e6
    pos = np.stack([r * np.cos(w * t), r * np.sin(w * t), np.zeros(n)], axis=1)
    vel = np.stack([-r * w * np.sin(w * t), r * w * np.cos(w * t), np.zeros(n)], axis=1)
    return t, pos, vel


def _write(series_dir: Path, sid: int, t, pos, vel, dt: float = 15.0) -> None:
    series_dir.mkdir(parents=True, exist_ok=True)
    np.savez(series_dir / f"scenario_{sid:06d}.npz", t=t, position=pos, velocity=vel, rk4_dt_s=np.float64(dt),
             device=np.array("cpu"), hp_km=np.float64(100.0), ha_km=np.float64(100.0), e=np.float64(0.0),
             inc_deg=np.float64(90.0))


def test_a_known_offset_lands_on_the_right_ric_axes(tmp_path: Path) -> None:
    t, pos, vel = _orbit()
    r_hat = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    i_hat = vel / np.linalg.norm(vel, axis=1, keepdims=True)
    c_hat = np.array([0.0, 0.0, 1.0])
    shifted = pos + 3.0 * r_hat + 4.0 * i_hat + 12.0 * c_hat  # |offset| = 13 m
    shifted[0] = pos[0]  # both series start from the scenario's initial state
    w = np.sqrt((len(t) - 1) / len(t))  # RMS over the samples after t0
    for sid in (0, 1):
        _write(tmp_path / "ref", sid, t, pos, vel)
        _write(tmp_path / "model", sid, t, shifted, vel)
    _write(tmp_path / "model", 2, t, shifted, vel)  # not in the reference: ignored
    out = tmp_path / "out"
    code = _tool().main(["--out", str(out), "--series", f"m={tmp_path / 'model'}", "--series", f"r={tmp_path / 'ref'}",
                         "--pair", "A=m,r"])
    assert code == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["A"]
    assert s["n_scenarios"] == 2 and s["rk4_dt_s"] == 15.0 and s["series"] == ["m", "r"]
    assert s["final_pos_err_m"]["median"] == pytest.approx(13.0, rel=1e-9)
    assert s["ric_rms_m"]["radial"] == pytest.approx(3.0 * w, rel=1e-9)
    assert s["ric_rms_m"]["along"] == pytest.approx(4.0 * w, rel=1e-9)
    assert s["ric_rms_m"]["cross"] == pytest.approx(12.0 * w, rel=1e-9)
    curves = np.load(out / "ensemble_rms_vs_time.npz")
    assert curves["A"][0] == 0.0
    np.testing.assert_allclose(curves["A"][1:], 13.0, rtol=1e-9)
    np.testing.assert_array_equal(curves["t_s"], t)
    assert len((out / "per_scenario.csv").read_text(encoding="utf-8").splitlines()) == 3


def test_series_integrated_with_different_steps_are_refused(tmp_path: Path) -> None:
    t, pos, vel = _orbit()
    _write(tmp_path / "a", 0, t, pos, vel, dt=15.0)
    _write(tmp_path / "b", 0, t, pos, vel, dt=60.0)
    with pytest.raises(SystemExit, match="RK4 steps differ"):
        _tool().main(["--out", str(tmp_path / "out"), "--series", f"a={tmp_path / 'a'}",
                      "--series", f"b={tmp_path / 'b'}", "--pair", "X=a,b"])


def test_series_from_different_scenario_sets_are_refused(tmp_path: Path) -> None:
    t, pos, vel = _orbit()
    _write(tmp_path / "a", 0, t, pos, vel)
    _write(tmp_path / "b", 0, t, pos * 1.001, vel)  # same id, another orbit
    with pytest.raises(SystemExit, match="initial states differ"):
        _tool().main(["--out", str(tmp_path / "out"), "--series", f"a={tmp_path / 'a'}",
                      "--series", f"b={tmp_path / 'b'}", "--pair", "X=a,b"])


def test_a_pair_must_name_two_series(tmp_path: Path) -> None:
    t, pos, vel = _orbit()
    _write(tmp_path / "a", 0, t, pos, vel)
    with pytest.raises(SystemExit, match="two --series names"):
        _tool().main(["--out", str(tmp_path / "out"), "--series", f"a={tmp_path / 'a'}", "--pair", "X=a,missing"])
