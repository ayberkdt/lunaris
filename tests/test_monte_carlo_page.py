import pytest
from pathlib import Path

try:
    from lunaris.ui.widgets.monte_carlo_page import (
        _normalize_output_path_for_format,
        _format_clock_span,
        MonteCarloPage,
        UIMonteCarloConfig
    )
    from PySide6.QtWidgets import QApplication
    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False

@pytest.fixture(scope="session")
def qapp():
    if not HAS_PYSIDE:
        return None
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_normalize_output_path():
    if not HAS_PYSIDE:
        pytest.skip("lunaris.ui.widgets not available")
    def norm(s, fmt):
        return _normalize_output_path_for_format(s, fmt).replace("\\", "/")
    assert norm("", "hdf5") == "outputs/monte_carlo/mc_output.h5"
    assert norm("foo/bar.npz", "hdf5") == "foo/bar.h5"
    assert norm("foo/bar.h5", "npz") == "foo/bar.npz"
    assert norm("foo/bar.hdf5", "hdf5") == "foo/bar.hdf5"
    assert norm("foo/bar.hdf5", "npz") == "foo/bar.npz"
    assert norm("foo/bar_custom_name", "hdf5") == "foo/bar_custom_name"

def test_format_clock_span():
    if not HAS_PYSIDE:
        pytest.skip("lunaris.ui.widgets not available")
    assert _format_clock_span(None) == "—"
    assert _format_clock_span(-1.0) == "—"
    assert _format_clock_span(59) == "00:59"
    assert _format_clock_span(61) == "01:01"
    assert _format_clock_span(3661) == "1:01:01"

@pytest.fixture
def mc_page(qapp):
    if not HAS_PYSIDE:
        pytest.skip("PySide6 not available")
    cfg = UIMonteCarloConfig()
    page = MonteCarloPage(mc_cfg=cfg)
    return page

def test_validate_page_inputs_default(mc_page):
    ok, errors, warnings = mc_page.validate_page_inputs()
    assert ok is True
    assert not errors

def test_validate_page_inputs_errors(mc_page):
    mc_page.ent_n_samples.setText("1")
    mc_page.ent_sigma_r.setText("-10")
    mc_page.toggle_gpu.setChecked(True)
    
    # Try to set gravity mode to classic_sh
    mc_page.cb_mc_gravity_mode.addItem("Classic", "classic_sh")
    idx = mc_page.cb_mc_gravity_mode.findData("classic_sh")
    mc_page.cb_mc_gravity_mode.setCurrentIndex(idx)
    
    mc_page.ent_gpu_sh.setText("30")
    
    ok, errors, warnings = mc_page.validate_page_inputs()
    assert not ok
    assert any("Ensemble must have at least 2 samples" in e for e in errors)
    assert any("Position uncertainty" in e for e in errors)
    assert any("Classic-SH GPU mode only supports SH degree <= 24" in e for e in errors)

def test_validate_page_inputs_warnings(mc_page):
    mc_page.ent_dt.setText("400")
    
    mc_page.cb_mc_gravity_mode.addItem("ST-LRPS", "st_lrps")
    idx = mc_page.cb_mc_gravity_mode.findData("st_lrps")
    mc_page.cb_mc_gravity_mode.setCurrentIndex(idx)
    
    mc_page.ent_mc_st_lrps_model_dir.setText("")
    
    ok, errors, warnings = mc_page.validate_page_inputs()
    assert ok is True # Warnings don't block
    assert any("Large dt" in w for w in warnings)
    assert any("ST-LRPS model dir is blank" in w for w in warnings)

def test_stale_identity_not_present():
    page_path = Path(__file__).parent.parent / "src" / "lunaris" / "ui" / "widgets" / "monte_carlo_page.py"
    if page_path.exists():
        content = page_path.read_text(encoding="utf-8")
        assert "LUNAR_SIMULATION" not in content, "Stale LUNAR_SIMULATION identity found in monte_carlo_page.py"
