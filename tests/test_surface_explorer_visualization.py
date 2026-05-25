import pytest
from pathlib import Path
import numpy as np

# 1. Import safety
try:
    from visualization import surface_explorer
    import_safe = True
except ImportError:
    import_safe = False

def test_import_safety():
    assert import_safe, "Importing visualization.surface_explorer failed."

# 2. CLI help
def test_cli_help():
    try:
        surface_explorer.main(["--help"])
    except SystemExit as e:
        assert e.code == 0

# 3. No hardcoded personal paths
def test_no_hardcoded_paths():
    p = Path(surface_explorer.__file__)
    content = p.read_text(encoding="utf-8")
    assert "C:\\\\Users" not in content
    assert "Ay Modeli" not in content
    assert "Lunar_Simulation" not in content

# 4. Helper behavior
def test_downsample_helper():
    arr = np.arange(16).reshape(4, 4)
    ds = surface_explorer._downsample(arr, 2)
    assert ds.shape == (2, 2)
    assert ds[0, 0] == 0
    assert ds[0, 1] == 2
    assert ds[1, 0] == 8
    assert ds[1, 1] == 10

def test_select_lon_slices_helper():
    lon_deg = np.array([-10, 0, 10])
    # 350 to 10 in 360 mode means it interprets query as [350, 10] using mod 360
    slices, use_360 = surface_explorer._select_lon_slices(lon_deg, (350, 10))
    assert use_360 is True

# 5. Old path removed
def test_old_path_removed():
    old_p = Path(__file__).parent.parent / "analysis" / "extras" / "surface_topography_explorer.py"
    assert not old_p.exists()

# 6. Visualization README
def test_visualization_readme():
    readme_p = Path(__file__).parent.parent / "visualization" / "README.md"
    assert readme_p.exists()
    content = readme_p.read_text(encoding="utf-8")
    assert "surface_explorer" in content
    assert "analysis/extras/surface_topography_explorer.py" not in content
