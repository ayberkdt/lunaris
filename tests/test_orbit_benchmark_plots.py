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

from st_lrps.evaluation import compare_gravity_models as cgm  # noqa: E402


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
