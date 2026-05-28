# -*- coding: utf-8 -*-
"""
Regression tests for the standard reporting / plotting pipeline.

These tests focus on the user-facing PDF/report layer rather than the Monte
Carlo-specific workspace. The goal is to ensure the refreshed report-manager
layout still works end-to-end with a minimal but valid simulation history.
"""

from __future__ import annotations

from pathlib import Path

from lunaris.analysis.reporting.plotting import effects_from_meta_history, figure_perturbation_magnitude
from lunaris.analysis.reporting.manager import (
    figure_run_config_page,
    figure_summary_page,
    make_report_pdf,
    plot_all,
)


def _minimal_history() -> dict:
    return {
        "t_s": [0.0, 60.0, 120.0, 180.0],
        "y": [
            [1.8374e6, 1.8375e6, 1.8376e6, 1.8377e6],
            [0.0, 1500.0, 3000.0, 4500.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1630.0, 1630.1, 1630.2, 1630.3],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "mu_m3s2": 4.9048695e12,
        "R_body_m": 1.7374e6,
        "events": {
            "peri_idx": [0, 2],
            "apo_idx": [1, 3],
            "impact_idx": None,
            "impact_alt_km": 0.0,
        },
    }


def _minimal_meta(tmp_path: Path) -> dict:
    return {
        "integrator_method": "DOP853",
        "rtol": 1e-10,
        "atol": 1e-12,
        "degree": 20,
        "output_dt_s": 60.0,
        "wall_time_s": 1.23,
        "propagation_time_s": 0.91,
        "_report_search_dir": str(tmp_path),
        "spacecraft": {
            "mass_kg": 1000.0,
            "area_m2": 5.0,
            "cd": 2.2,
            "cr": 1.5,
        },
        "flags": {
            "enable_sh": True,
            "enable_srp": True,
            "enable_relativity_1pn": False,
        },
    }


def test_report_pages_build_from_minimal_history(tmp_path: Path) -> None:
    history = _minimal_history()
    meta = _minimal_meta(tmp_path)

    fig_summary = figure_summary_page(history, meta=meta, pdf_or_out_dir_hint=str(tmp_path))
    fig_config = figure_run_config_page(history, meta=meta, pdf_or_out_dir_hint=str(tmp_path))

    assert fig_summary is not None
    assert fig_config is not None
    assert len(fig_summary.axes) >= 4
    assert len(fig_config.axes) >= 4


def test_make_report_pdf_creates_standard_report(tmp_path: Path) -> None:
    history = _minimal_history()
    meta = _minimal_meta(tmp_path)
    out_pdf = tmp_path / "standard_report.pdf"

    result_path = make_report_pdf(str(out_pdf), history, meta=meta, ctx=None)

    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0


def test_plot_all_creates_timestamped_report_subdirectory(tmp_path: Path) -> None:
    history = _minimal_history()
    meta = _minimal_meta(tmp_path)

    results = plot_all(
        history=history,
        out_dir=str(tmp_path),
        meta=meta,
        save_png=False,
        save_pdf=False,
        use_run_subdir=True,
    )

    out_dir = Path(results["out_dir"])
    assert results["status"] == "success"
    assert out_dir.exists()
    assert out_dir.parent == tmp_path
    assert out_dir.name.startswith("run_")


def test_effects_from_meta_history_reports_individual_force_toggles() -> None:
    effects, _source = effects_from_meta_history(
        meta={
            "flags": {
                "enable_sh": True,
                "enable_3rd_body_sun": True,
                "enable_3rd_body_earth": False,
                "enable_srp": True,
                "enable_albedo": True,
                "enable_tides_k2": True,
                "enable_tides_k3": False,
                "enable_relativity_1pn": True,
            }
        },
        history={},
    )

    assert effects["Spherical Harmonics"] is True
    assert effects["Third-Body Sun"] is True
    assert effects["Third-Body Earth"] is False
    assert effects["Solar Radiation Pressure"] is True
    assert effects["Lunar Albedo"] is True
    assert effects["Solid Tides (k2)"] is True
    assert effects["Solid Tides (k3)"] is False
    assert effects["General Relativity"] is True


def test_figure_perturbation_magnitude_uses_ctx_breakdown_hook() -> None:
    class _DummyCtx:
        def get_acceleration_breakdown(self, t: float, y) -> dict:
            return {
                "Gravity (SH)": 1.0e-3,
                "SRP": 2.0e-7 + float(t) * 0.0,
            }

    fig = figure_perturbation_magnitude(_minimal_history(), _DummyCtx())

    assert fig is not None
    assert len(fig.axes) >= 1
    assert fig.axes[0].get_title() == "Perturbation Acceleration Budget"
