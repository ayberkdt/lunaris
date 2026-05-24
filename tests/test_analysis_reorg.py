import os
import pytest
from pathlib import Path

def test_analysis_import():
    # Should be minimal and functional
    import analysis
    assert hasattr(analysis, "process_simulation_results")
    assert hasattr(analysis, "plot_all")

def test_removed_modules_raise_importerror():
    with pytest.raises(ImportError):
        import analysis.report_manager
    with pytest.raises(ImportError):
        import analysis.plotting
    with pytest.raises(ImportError):
        import analysis.styling
    with pytest.raises(ImportError):
        import analysis.mc_analysis
    with pytest.raises(ImportError):
        import analysis.mc_plotting
    with pytest.raises(ImportError):
        import analysis.compare_gravity_models
    with pytest.raises(ImportError):
        import analysis.threeD_animation

def test_new_canonical_paths_importable():
    import analysis.reporting.manager
    import analysis.reporting.plotting
    import analysis.reporting.styling
    import analysis.monte_carlo.statistics
    import analysis.monte_carlo.plotting
    import validation.gravity.compare_gravity_models
    import visualization.orbit_animation
    import analysis.formatting

def test_stale_names_absent():
    stale_names = ["LUNAR_SIMULATION", "LunarSim", "LUNARSIM_"]
    # Check all python files in analysis, validation/gravity, and visualization
    base_dir = Path(__file__).parent.parent
    directories = [
        base_dir / "analysis",
        base_dir / "validation" / "gravity",
        base_dir / "visualization"
    ]
    
    for directory in directories:
        if not directory.exists():
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()
                        for stale in stale_names:
                            # We might have ST_LRPS or STLRPS_ now
                            assert stale not in content, f"Found stale name '{stale}' in {file}"

def test_strict_mode_postprocess():
    import numpy as np
    from analysis.postprocess import process_simulation_results, compute_history
    
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
    with pytest.raises(Exception):
        compute_history(t_s, y_good, mu=1.0, R_body=1.0, ctx=DummyCtx(), strict=True)
