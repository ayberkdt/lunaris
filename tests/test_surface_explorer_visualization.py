from pathlib import Path

import numpy as np

# 1. Import safety
try:
    from lunaris.visualization import surface_explorer
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
    readme_p = Path(__file__).parent.parent / "src" / "lunaris" / "visualization" / "README.md"
    assert readme_p.exists()
    content = readme_p.read_text(encoding="utf-8")
    assert "surface_explorer" in content
    assert "analysis/extras/surface_topography_explorer.py" not in content

# 7. Maximum-height report on a loaded TopographyGrid
def _write_ldem(tmp_path: Path, *, unit: str, scaling: float, offset: float, dn: np.ndarray):
    from lunaris.loaders.io_surface import TopographyGrid

    lines, samples = dn.shape
    (tmp_path / "t.img").write_bytes(np.ascontiguousarray(dn.astype("<i2")).tobytes())
    (tmp_path / "t.lbl").write_text(
        f"""PDS_VERSION_ID = PDS3
RECORD_BYTES = {samples * 2}
^IMAGE = "t.img"
PRODUCT_ID = "LDEM_SYNTH"
OBJECT = IMAGE
  LINES = {lines}
  LINE_SAMPLES = {samples}
  SAMPLE_TYPE = LSB_INTEGER
  SAMPLE_BITS = 16
  UNIT = {unit}
  SCALING_FACTOR = {scaling}
  OFFSET = {offset}
END_OBJECT = IMAGE
MAP_PROJECTION_TYPE = "SIMPLE_CYLINDRICAL"
MAP_RESOLUTION = 1 <PIX/DEG>
MAXIMUM_LATITUDE = 20
MINIMUM_LATITUDE = 10
WESTERNMOST_LONGITUDE = 30
EASTERNMOST_LONGITUDE = 50
POSITIVE_LONGITUDE_DIRECTION = "EAST"
CENTER_LATITUDE = 15 <DEG>
CENTER_LONGITUDE = 40 <DEG>
END
""",
        encoding="utf-8",
    )
    return TopographyGrid(tmp_path / "t.lbl", tmp_path / "t.img", mmap=False)


def test_report_topography_max_runs_on_a_topography_grid(tmp_path):
    """TopographyGrid exposed no public lat/lon centres, so the report raised
    AttributeError on every real product (seen on the 16 ppd LDEM)."""
    dn = np.zeros((10, 20), dtype=np.int16)
    dn[3, 7] = 1234  # line 3 -> lat 20 - 3.5 = 16.5; sample 7 -> lon 30 + 7.5 = 37.5
    for sub, unit, scaling, offset in (
        ("km", "KILOMETER", 0.001, 1737.4),  # DN in m, label in km
        ("m", "METER", 1.0, 1737400.0),      # same heights, label in m
    ):
        (tmp_path / sub).mkdir()
        topo = _write_ldem(tmp_path / sub, unit=unit, scaling=scaling, offset=offset, dn=dn)
        np.testing.assert_allclose(topo.lat_centers_deg, 19.5 - np.arange(10))
        np.testing.assert_allclose(topo.lon_centers_deg, 30.5 + np.arange(20))

        h_m, lat, lon = surface_explorer.report_topography_max(topo)
        assert (lat, lon) == (16.5, 37.5)
        assert abs(h_m - 1234.0) < 1e-9
        assert abs(h_m - topo.sample_nearest(lat, lon, kind="height_m")) < 1e-9
