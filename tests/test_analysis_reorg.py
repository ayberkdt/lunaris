import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_analysis_import():
    # Should be minimal and functional
    import lunaris.analysis as analysis

    # Assert every name in analysis.__all__ is accessible
    for name in analysis.__all__:
        assert hasattr(analysis, name)

    assert sorted(analysis.__all__) == sorted([
        "process_simulation_results",
        "compute_history",
        "summarize_history"
    ])
    assert not hasattr(analysis, "plot_all")
    assert not hasattr(analysis, "compute_mc_statistics")

def test_removed_modules_raise_importerror():
    import importlib
    import sys

    modules_to_check = [
        "lunaris.analysis.report_manager",
        "lunaris.analysis.plotting",
        "lunaris.analysis.styling",
        "lunaris.analysis.mc_analysis",
        "lunaris.analysis.mc_plotting",
        "lunaris.analysis.monte_carlo",
        "lunaris.analysis.monte_carlo.linear_check",
        "lunaris.analysis.monte_carlo.plotting",
        "lunaris.analysis.monte_carlo.result_audit",
        "lunaris.analysis.monte_carlo.statistics",
        "lunaris.analysis.monte_carlo.uq_report",
        "lunaris.analysis.compare_gravity_models",
        "lunaris.analysis.threeD_animation",
    ]
    for mod in modules_to_check:
        sys.modules.pop(mod, None)
        with pytest.raises(ImportError):
            importlib.import_module(mod)

def test_new_canonical_paths_importable():
    from lunaris.analysis.ensemble.plotting import plot_mc_report
    from lunaris.analysis.ensemble.statistics import compute_mc_statistics
    from lunaris.analysis.postprocess import (
        compute_history,
        process_simulation_results,
        summarize_history,
    )
    from lunaris.analysis.reporting.manager import plot_all
    from lunaris.analysis.reporting.plotting import figure_orbit_3d
    from lunaris.analysis.reporting.styling import apply_rcparams
    from lunaris.visualization.orbit_animation import render_orbit_animation

def test_stale_names_absent():
    # Allowlist these strings for this test file itself
    # so we don't fail parsing our own banned list
    banned = [
        "LUNAR_SIMULATION",
        "LunarSim",
        "LUNARSIM_",
        "analysis/report_manager.py",
        "analysis/mc_analysis.py",
        "analysis/mc_plotting.py",
        "analysis/compare_gravity_models.py",
        "analysis/threeD_animation.py",
        "Legacy API",
        "animate_orbit"
    ]

    base_dir = Path(__file__).parent.parent
    directories = [
        base_dir / "src" / "lunaris" / "analysis",
        base_dir / "validation" / "gravity",
        base_dir / "src" / "lunaris" / "visualization"
    ]

    for directory in directories:
        if not directory.exists():
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    with open(os.path.join(root, file), encoding="utf-8") as f:
                        content = f.read()
                        for stale in banned:
                            assert stale not in content, f"Found stale string '{stale}' in {file}"

def test_formatting_ssot():
    import lunaris.analysis.reporting.manager as m
    assert not hasattr(m, "_format_percent")
    assert not hasattr(m, "_format_days")
    assert not hasattr(m, "_format_km")
    assert not hasattr(m, "_safe_float")
    assert not hasattr(m, "_format_duration")
    assert not hasattr(m, "_format_count")
    assert not hasattr(m, "_format_sci_or_na")

    import lunaris.analysis.formatting as fmt
    assert hasattr(fmt, "format_percent")

def test_strict_mode_postprocess():
    import numpy as np

    from lunaris.analysis.postprocess import compute_history, process_simulation_results

    # Test strict mode failure on compute_history with wrong shape
    t_s = np.array([0.0, 1.0])
    y = np.zeros((3, 2))  # too small

    with pytest.raises(ValueError, match="y must be a 2D array with shape"):
        compute_history(t_s, y, mu=1.0, R_body=1.0)

    y_good = np.zeros((6, 2))

    class DummyCtx:
        pass

    # Optional products fail-soft
    res = compute_history(t_s, y_good, mu=1.0, R_body=1.0, ctx=DummyCtx())
    assert "accel_mag" not in res

    # Optional products fail-fast in strict mode
    with pytest.raises(Exception):  # noqa: B017
        compute_history(t_s, y_good, mu=1.0, R_body=1.0, ctx=DummyCtx(), strict=True)


def test_process_simulation_results_uses_configured_output_cap():
    import numpy as np

    from lunaris.analysis.postprocess import process_simulation_results

    n = 60_000
    t_s = np.arange(n, dtype=float)
    y = np.zeros((n, 6), dtype=float)
    y[:, 0] = 1_837_400.0
    y[:, 4] = 1_600.0

    cfg = SimpleNamespace(
        time=SimpleNamespace(max_points_cap=70_000),
        propagator=SimpleNamespace(
            events=SimpleNamespace(
                detect_impact=True,
                enable_peri_apo_events=True,
                detect_eclipse=False,
                impact_alt_km=0.0,
            )
        ),
        flags=SimpleNamespace(enable_srp=False, enable_albedo=False, enable_thermal=False),
    )
    result = SimpleNamespace(t=t_s, y=y)

    hist = process_simulation_results(result, cfg=cfg)

    assert len(hist["t_s"]) == n
    assert hist["sample_count_raw"] == n
    assert hist["sample_count_returned"] == n
    assert hist["downsampled_for_reporting"] is False
