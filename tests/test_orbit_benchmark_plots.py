"""Robustness tests for the Orbit-Level Benchmark plotting / report styling.

These exercise the publication-grade figure generation
(:func:`plot_gpu_batch_report_figures`) and the PDF writer with fabricated
metric rows — no propagation, no GPU, no SPICE. They verify the figures and PDF
are produced without exceptions for awkward inputs (tiny / zero errors, small
and large N, missing models, log-scale with zeros) rather than comparing pixels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm  # noqa: E402


# ---------------------------------------------------------------------------
# Fake-data builders
# ---------------------------------------------------------------------------

def _make_args(**over) -> argparse.Namespace:
    base = dict(
        plot_theme="report_light", plot_error_logscale=False, plot_3d=False,
        truth="sh200", truth_integrator="DOP853", duration_days=1.0,
        random_scenarios=4, scenario_seed=42, scenario_mode="near_circular_altitude",
        sampling_method="random", inclination_sampling="uniform_deg",
        altitude_min_km=200.0, altitude_max_km=400.0, dt_out=60.0,
        gpu_integrator="medium", workers=1, torch_dtype="float64",
        gpu_models="sh20,sh80,st_lrps", batch_frame_mode="match_dynamics_engine",
        rk4_dt_s=10.0, st_lrps_rk4_dt=30.0,
        plot_best_scenario_id=None, plot_worst_scenario_id=None,
        plot_representative_scenario_id=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _build(n=4, err_scale_km=5e-4, T=10,
           models=("GPU_SH20_RK4", "GPU_SH80_RK4", "GPU_ST_LRPS_RK4")):
    """Construct a self-consistent fake GPU-batch dataset."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 86400.0, T)
    scenarios, truth_t, truth_y, truth_rt = [], {}, {}, {}
    for sid in range(n):
        st0 = np.array([1.9e6, 0.0, 0.0, 0.0, 1.6e3, 0.0])
        scenarios.append(cgm.Scenario(
            scenario_id=sid, hp_km=200.0 + 10 * sid, ha_km=260.0, a_km=1937.0, e=0.001,
            inc_deg=20.0 + 5 * sid, raan_deg=0.0, argp_deg=0.0, ta_deg=0.0,
            initial_state=st0, raw_unit_sample=None, sampling_method="random"))
        yt = np.tile(st0, (T, 1)).astype(float)
        yt[:, 0] += 1.0e3 * np.sin(np.linspace(0, 6, T))
        truth_t[sid], truth_y[sid], truth_rt[sid] = t, yt, 2.0
    truth = cgm.TruthTrajectorySet("sh200_dop853", truth_t, truth_y, truth_rt)

    results, metrics_rows, agg, runtime = [], [], [], []
    for mi, name in enumerate(models):
        scale = err_scale_km * (0.3 if "ST_LRPS" in name else (1.0 + mi))
        y = np.zeros((T, n, 6))
        rmss = []
        for sid in range(n):
            y[:, sid, :] = truth_y[sid].copy()
            y[:, sid, 0] += scale * 1.0e3 * np.linspace(0, 1, T)  # add metres of drift
            dr = np.linalg.norm(y[:, sid, :3] - truth_y[sid][:, :3], axis=1) / 1000.0
            rms = float(np.sqrt(np.mean(dr ** 2)))
            rmss.append(rms)
            metrics_rows.append(dict(
                model=name, scenario_id=sid, status="ok", rms_pos_err_km=rms,
                p95_pos_err_km=rms * 1.2, inc_deg=20.0 + 5 * sid, hp_km=200.0 + 10 * sid,
                along_rms_km=rms * 0.5, radial_rms_km=rms * 0.3, cross_rms_km=rms * 0.2))
        results.append(cgm.BatchModelResult(
            model_name=name.lower(), display_name=name, backend="torch_sh", device="cpu",
            dtype="float64", t=t, y=y, runtime_s=1.0 + mi, n_steps=T - 1, n_scenarios=n,
            rk4_dt_s=10.0, output_dt_s=60.0, status="ok"))
        agg.append(dict(
            model=name, median_rms_pos_err_km=float(np.median(rmss)),
            p95_rms_pos_err_km=float(np.percentile(rmss, 95)),
            max_rms_pos_err_km=float(np.max(rmss)),
            median_along_rms_km=float(np.median(rmss)) * 0.5, n_scenarios_ok=n))
        runtime.append(dict(
            model=name, total_runtime_s=1.0 + mi, runtime_per_scenario_s=(1.0 + mi) / n,
            trajectory_steps_per_second=1000.0, speedup_vs_truth_total=2.0 + mi))
    agg = sorted(agg, key=lambda r: r["median_rms_pos_err_km"])
    equivalent = cgm.estimate_stlrps_equivalent_sh_degree(agg)
    selected = cgm.select_stlrps_scenarios(
        metrics_rows, {s.scenario_id: s for s in scenarios}, _make_args(random_scenarios=n))
    return dict(agg=agg, runtime=runtime, metrics=metrics_rows, results=results,
                truth=truth, scenarios=scenarios, selected=selected, equivalent=equivalent)


def _run(tmp_path, ds, args):
    plots = tmp_path / "plots"
    reports = tmp_path / "reports"
    saved = cgm.plot_gpu_batch_report_figures(
        ds["agg"], ds["runtime"], ds["metrics"], ds["results"], ds["truth"],
        ds["scenarios"], ds["selected"], ds["equivalent"], plots, args)
    cgm.write_gpu_batch_report_pdf(
        args, ds["agg"], ds["runtime"], ds["equivalent"], ds["selected"], plots, reports)
    return saved, reports / "gpu_batch_validation_report.pdf"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_unit_autoscaling_picks_sensible_units():
    assert cgm.select_length_unit(5.0) == ("km", 1.0)
    assert cgm.select_length_unit(0.5) == ("km", 1.0)
    assert cgm.select_length_unit(5e-3)[0] == "m"     # < 0.01 km -> metres
    assert cgm.select_length_unit(5e-4)[0] == "cm"    # < 1e-3 km -> centimetres
    # Non-finite / non-positive falls back to km without raising.
    assert cgm.select_length_unit(float("nan")) == ("km", 1.0)
    assert cgm.select_length_unit(0.0) == ("km", 1.0)
    # Multiplier converts correctly.
    unit, mult = cgm.select_length_unit(5e-3)
    assert abs(0.005 * mult - 5.0) < 1e-9 and unit == "m"


def test_model_color_marker_make_st_lrps_stand_out():
    st = cgm.model_color("GPU_ST_LRPS_RK4")
    sh20 = cgm.model_color("GPU_SH20_RK4")
    sh200 = cgm.model_color("GPU_SH200_RK4")
    assert st == cgm._ST_LRPS_COLOR
    assert st != sh20 and st != sh200
    # SH family is degree-ordered (low != high) and consistent across name forms.
    assert cgm.model_color("sh20") == cgm.model_color("GPU_SH20_RK4")
    assert sh20 != sh200
    # ST-LRPS uses the star marker and a heavier line; SH uses non-star markers.
    assert cgm.model_marker("GPU_ST_LRPS_RK4") == "*"
    assert cgm.model_marker("GPU_SH20_RK4") != "*"
    assert cgm.model_linewidth("GPU_ST_LRPS_RK4") > cgm.model_linewidth("GPU_SH20_RK4")
    assert cgm.display_label("GPU_ST_LRPS_RK4") == "ST-LRPS"
    assert cgm.display_label("GPU_SH20_RK4") == "SH20"


def test_should_log_handles_zeros_without_error():
    assert cgm._should_log([0.0, 0.0, 0.0]) is False     # no positive values
    assert cgm._should_log([1e-6, 1.0]) is True           # spans orders of magnitude
    assert cgm._should_log([1.0, 1.2]) is False
    assert cgm._should_log([]) is False


def test_fmt_km_does_not_collapse_tiny_values_to_zero():
    assert cgm._fmt_km(0.0) == "0"
    assert cgm._fmt_km(1.2345) == "1.2345"
    assert cgm._fmt_km(float("nan")) == "n/a"
    # Sub-metre error: %.4f would render "0.0000"; we keep it visible.
    tiny = cgm._fmt_km(3e-5)        # 3e-5 km = 3 cm
    assert tiny != "0.0000" and "e-" in tiny


def test_gpu_integrator_eval_counts_match_integrators():
    assert cgm._gpu_integrator_evals_per_step("light") == 2
    assert cgm._gpu_integrator_evals_per_step("medium") == 4
    assert cgm._gpu_integrator_evals_per_step("robust") == 12
    assert cgm._gpu_integrator_evals_per_step("unknown") == 4


def test_runtime_metrics_throughput_and_eval_scaling():
    import numpy as np
    T, N = 11, 8
    res = cgm.BatchModelResult(
        model_name="sh20", display_name="GPU_SH20_RK4", backend="b", device="cuda:0",
        dtype="float64", t=np.linspace(0, 600, T), y=np.zeros((T, N, 6)),
        runtime_s=2.0, n_steps=60, n_scenarios=N, rk4_dt_s=10.0, output_dt_s=60.0,
        status="ok")
    truth = cgm.TruthTrajectorySet(
        "sh200_dop853", {0: np.array([0.0])}, {0: np.zeros((1, 6))}, {0: 5.0})

    rows4 = cgm.build_gpu_runtime_metrics([res], truth, evals_per_step=4)
    rows2 = cgm.build_gpu_runtime_metrics([res], truth, evals_per_step=2)
    tps = rows4[0]["trajectory_steps_per_second"]
    assert abs(tps - N * 60 / 2.0) < 1e-6                       # 8*60/2 = 240
    # Acceleration-eval throughput scales with the integrator, not a hardcoded 4.
    assert abs(rows4[0]["acceleration_evaluations_per_second"] - tps * 4) < 1e-6
    assert abs(rows2[0]["acceleration_evaluations_per_second"] - tps * 2) < 1e-6


# ---------------------------------------------------------------------------
# Figure + PDF generation robustness
# ---------------------------------------------------------------------------

def test_plots_generated_for_small_errors(tmp_path):
    ds = _build(n=4, err_scale_km=5e-4)        # tiny -> centimetre units
    saved, pdf = _run(tmp_path, ds, _make_args(random_scenarios=4))
    assert len(saved) > 0
    assert all(Path(p).exists() for p in saved)
    assert pdf.exists() and pdf.stat().st_size > 1000


def test_plots_generated_for_zero_errors(tmp_path):
    ds = _build(n=4, err_scale_km=0.0)         # exercises empty-note / no-data paths
    saved, pdf = _run(tmp_path, ds, _make_args(random_scenarios=4))
    assert all(Path(p).exists() for p in saved)
    assert pdf.exists()


@pytest.mark.parametrize("n", [4, 128])
def test_plots_generated_for_varied_n(tmp_path, n):
    ds = _build(n=n, err_scale_km=0.05, T=8)
    saved, pdf = _run(tmp_path, ds, _make_args(random_scenarios=n))
    assert all(Path(p).exists() for p in saved)
    assert pdf.exists()
    # The distribution figure is always produced.
    assert any("distribution" in p.name for p in saved)


def test_st_lrps_figures_present_and_selected(tmp_path):
    ds = _build(n=12, err_scale_km=0.05)
    saved, _ = _run(tmp_path, ds, _make_args(random_scenarios=12))
    names = {p.name for p in saved}
    assert "stlrps_equivalent_sh_degree.png" in names
    # Selected ST-LRPS scenario figures are emitted.
    assert any(n.startswith("selected_") and "position_error" in n for n in names)
    assert ds["selected"]  # ST-LRPS scenarios were selected


def test_missing_optional_models_do_not_crash(tmp_path):
    # No ST-LRPS at all — only SH baselines.
    ds = _build(n=6, err_scale_km=0.05, models=("GPU_SH20_RK4", "GPU_SH80_RK4"))
    saved, pdf = _run(tmp_path, ds, _make_args(random_scenarios=6, gpu_models="sh20,sh80"))
    names = {p.name for p in saved}
    assert all(Path(p).exists() for p in saved)
    assert "ensemble_mean_position_error_vs_time.png" in names
    assert "selected_representative_position_error_all_models.png" in names
    assert ds["selected"]["_selection_source"] == "comparison set"
    assert pdf.exists()


def test_single_model_does_not_crash(tmp_path):
    ds = _build(n=4, err_scale_km=0.02, models=("GPU_ST_LRPS_RK4",))
    saved, pdf = _run(tmp_path, ds, _make_args(random_scenarios=4, gpu_models="st_lrps"))
    assert all(Path(p).exists() for p in saved)
    assert pdf.exists()


def test_log_scale_with_zero_values_does_not_fail(tmp_path):
    # Force the log-scale code paths via the CLI flag while values include zeros.
    ds = _build(n=4, err_scale_km=0.0)
    saved, pdf = _run(tmp_path, ds, _make_args(random_scenarios=4, plot_error_logscale=True))
    assert all(Path(p).exists() for p in saved)
    assert pdf.exists()


def test_both_themes_render(tmp_path):
    ds = _build(n=8, err_scale_km=0.05)
    for theme in ("report_light", "technical_dark"):
        sub = tmp_path / theme
        saved, pdf = _run(sub, ds, _make_args(random_scenarios=8, plot_theme=theme))
        assert all(Path(p).exists() for p in saved)
        assert pdf.exists()


# ---------------------------------------------------------------------------
# GPU batch frame mode: precomputed_slerp vs dynamic (CPU torch; no SH/SPICE)
# ---------------------------------------------------------------------------

class _FakeEphem:
    """Minimal stand-in for EphemerisManager.get_data_provider()."""

    def __init__(self, q_tab, dt_s):
        self._p = {"q_i2f_tab": q_tab, "dt_s": float(dt_s)}

    def get_data_provider(self):
        return self._p


def _rot_z_quat_table(n: int, max_angle: float = 0.8):
    """A table of unit quaternions (rotation about +z by an increasing angle)."""
    ang = np.linspace(0.0, max_angle, n)
    return np.stack([np.cos(ang / 2), np.zeros(n), np.zeros(n), np.sin(ang / 2)], axis=1)


_STAGE_OFFSETS = {
    "light": [0.0, 0.5],
    "medium": [0.0, 0.5, 0.5, 1.0],
    "robust": [0.0, 0.5, 0.5, 1.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0],
}


@pytest.mark.parametrize("integrator", ["light", "medium", "robust"])
def test_frame_cache_matches_dynamic_quaternions(integrator):
    torch = pytest.importorskip("torch")
    eph = _FakeEphem(_rot_z_quat_table(10), dt_s=100.0)
    dev, dtype = torch.device("cpu"), torch.float64
    dyn = cgm.TorchFrameProvider(eph, device=dev, dtype=dtype, mode="match_dynamics_engine")
    pre = cgm.TorchFrameProvider(eph, device=dev, dtype=dtype, mode="precomputed_slerp")

    dt_eff, total_steps = 10.0, 20
    cache = pre.precompute_rk_stage_quaternions(total_steps, dt_eff, integrator)
    offsets = _STAGE_OFFSETS[integrator]
    assert cache.q_i2f.shape == (total_steps, len(offsets), 4)

    err = 0.0
    for step in range(total_steps):
        for si, off in enumerate(offsets):
            t = step * dt_eff + off * dt_eff
            err = max(err, float((dyn.quat_i2f(t) - cache.q_i2f[step, si]).abs().max()))
    assert err < 1e-12, f"{integrator} cache deviates from dynamic quat_i2f: {err}"


def test_frame_cache_inverse_is_conjugate():
    torch = pytest.importorskip("torch")
    eph = _FakeEphem(_rot_z_quat_table(10), dt_s=100.0)
    pre = cgm.TorchFrameProvider(eph, device=torch.device("cpu"), dtype=torch.float64,
                                 mode="precomputed_slerp")
    cache = pre.precompute_rk_stage_quaternions(8, 10.0, "medium")
    assert bool((cache.q_f2i[..., 0] == cache.q_i2f[..., 0]).all())
    assert bool((cache.q_f2i[..., 1:] == -cache.q_i2f[..., 1:]).all())


def test_frame_cache_out_of_range_holds_last_quaternion():
    """Beyond the ephemeris table the cache must hold the last quaternion
    (like the dynamic path), not extrapolate."""
    torch = pytest.importorskip("torch")
    n, dt_s = 10, 100.0
    eph = _FakeEphem(_rot_z_quat_table(n), dt_s=dt_s)
    dev, dtype = torch.device("cpu"), torch.float64
    dyn = cgm.TorchFrameProvider(eph, device=dev, dtype=dtype, mode="match_dynamics_engine")
    pre = cgm.TorchFrameProvider(eph, device=dev, dtype=dtype, mode="precomputed_slerp")

    dt_eff = 10.0
    big_steps = int((n + 3) * dt_s / dt_eff) + 2   # last stage time far beyond the table
    cache = pre.precompute_rk_stage_quaternions(big_steps, dt_eff, "medium")
    last_t = (big_steps - 1) * dt_eff + dt_eff
    assert float((dyn.quat_i2f(last_t) - dyn.q_tab[-1]).abs().max()) < 1e-12
    assert float((cache.q_i2f[-1, -1] - pre.q_tab[-1]).abs().max()) < 1e-12


def _run_two_modes(integrator, dtype_name, monkeypatch):
    """Propagate a deterministic point-mass system in both frame modes."""
    torch = pytest.importorskip("torch")
    from lunaris.common.constants import MU_MOON

    # Replace the SH/ST-LRPS accelerator with a cheap, deterministic point mass
    # so the test needs no gravity data or checkpoint.
    def _fake_accel(model_name, gravity_model, *, device, dtype):
        mu = torch.tensor(float(MU_MOON), device=device, dtype=dtype)

        def accel(pos_fixed):
            r = torch.linalg.norm(pos_fixed, dim=1, keepdim=True).clamp_min(1.0)
            return -mu * pos_fixed / (r ** 3)

        return accel, "fake_pointmass"

    monkeypatch.setattr(
        "lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.compute._make_gpu_accelerator",
        _fake_accel,
    )

    eph = _FakeEphem(_rot_z_quat_table(40, max_angle=1.2), dt_s=50.0)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    y0 = np.array([
        [1.90e6, 0.0, 0.0, 0.0, 1.60e3, 0.0],
        [0.0, 1.95e6, 0.0, -1.55e3, 0.0, 0.0],
    ], dtype=np.float64)

    kw = dict(duration_s=600.0, rk4_dt_s=10.0, output_dt_s=60.0, ephem=eph,
              device=torch.device("cpu"), dtype=dtype, dtype_name=dtype_name,
              gpu_integrator=integrator)
    res_dyn = cgm.propagate_gpu_batch_model(
        "sh20", None, y0, frame_mode="match_dynamics_engine", **kw)
    res_pre = cgm.propagate_gpu_batch_model(
        "sh20", None, y0, frame_mode="precomputed_slerp", **kw)
    assert res_dyn.status == "ok" and res_pre.status == "ok"
    return res_dyn, res_pre


@pytest.mark.parametrize("integrator", ["light", "medium", "robust"])
def test_precomputed_matches_dynamic_trajectory(integrator, monkeypatch):
    res_dyn, res_pre = _run_two_modes(integrator, "float64", monkeypatch)
    assert res_pre.y.shape == res_dyn.y.shape
    max_abs = float(np.max(np.abs(res_pre.y - res_dyn.y)))
    # Same quaternions + same RK math + same accel -> equivalent within float64.
    assert max_abs < 1e-8, f"{integrator}: precomputed vs dynamic diff {max_abs}"


def test_precomputed_matches_dynamic_trajectory_float32(monkeypatch):
    res_dyn, res_pre = _run_two_modes("medium", "float32", monkeypatch)
    rel = float(np.max(np.abs(res_pre.y - res_dyn.y)) / (np.max(np.abs(res_dyn.y)) + 1e-30))
    assert rel < 1e-5, f"float32 precomputed vs dynamic rel diff {rel}"


@pytest.mark.skipif(
    not (lambda: __import__("importlib").util.find_spec("torch"))(),
    reason="torch not installed",
)
def test_precomputed_cuda_smoke(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from lunaris.common.constants import MU_MOON

    def _fake_accel(model_name, gravity_model, *, device, dtype):
        mu = torch.tensor(float(MU_MOON), device=device, dtype=dtype)

        def accel(pos_fixed):
            r = torch.linalg.norm(pos_fixed, dim=1, keepdim=True).clamp_min(1.0)
            return -mu * pos_fixed / (r ** 3)
        return accel, "fake_pointmass"

    monkeypatch.setattr(
        "lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.compute._make_gpu_accelerator",
        _fake_accel,
    )
    eph = _FakeEphem(_rot_z_quat_table(40, max_angle=1.2), dt_s=50.0)
    y0 = np.array([[1.90e6, 0.0, 0.0, 0.0, 1.60e3, 0.0]], dtype=np.float64)
    res = cgm.propagate_gpu_batch_model(
        "sh20", None, y0, duration_s=300.0, rk4_dt_s=10.0, output_dt_s=60.0, ephem=eph,
        device=torch.device("cuda:0"), dtype=torch.float64, dtype_name="float64",
        frame_mode="precomputed_slerp", gpu_integrator="medium")
    assert res.status == "ok"
    assert np.isfinite(res.y).all()
