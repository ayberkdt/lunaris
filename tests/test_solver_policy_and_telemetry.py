# -*- coding: utf-8 -*-
"""
Focused regression tests for restart safety and terrain-aware telemetry helpers.

These tests target the small pure/helper functions that back the UI fixes:
- tolerance normalization shared across the desktop workflow
- optional topography fields added to live telemetry
"""

from __future__ import annotations

import os

import numpy as np
from PySide6 import QtWidgets

from lunaris.core.dynamics import _select_adaptive_sh_degree
from lunaris.core.propagator import _make_telem_dict, build_events
from lunaris.common.type_defs import PropagatorConfig
from lunaris.ui.widgets.live_telemetry_page import HAS_PYQTGRAPH, MultiTelemetryPlot
from lunaris.ui.widgets.solver_policy import (
    DEFAULT_ADAPTIVE_ATOL,
    DEFAULT_ADAPTIVE_RTOL,
    LEGACY_ADAPTIVE_ATOL,
    LEGACY_ADAPTIVE_RTOL,
    choose_solver_tolerances,
    normalize_solver_config_object,
    uses_legacy_adaptive_defaults,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_choose_solver_tolerances_falls_back_to_safe_defaults() -> None:
    rtol, atol = choose_solver_tolerances(
        "DOP853 (Adaptive)",
        rtol="",
        atol=None,
    )

    assert rtol == DEFAULT_ADAPTIVE_RTOL
    assert atol == DEFAULT_ADAPTIVE_ATOL


def test_choose_solver_tolerances_derives_atol_from_valid_rtol() -> None:
    rtol, atol = choose_solver_tolerances(
        "RK45",
        rtol="1e-8",
        atol="",
    )

    assert rtol == 1e-8
    assert atol == 1e-10


def test_choose_solver_tolerances_relaxes_stale_legacy_atol_when_rtol_is_invalid() -> None:
    rtol, atol = choose_solver_tolerances(
        "DOP853 (Adaptive)",
        rtol="0",
        atol=LEGACY_ADAPTIVE_ATOL,
    )

    assert rtol == DEFAULT_ADAPTIVE_RTOL
    assert atol == DEFAULT_ADAPTIVE_ATOL


def test_normalize_solver_config_object_upgrades_legacy_defaults_on_request() -> None:
    class _Cfg:
        rtol = LEGACY_ADAPTIVE_RTOL
        atol = LEGACY_ADAPTIVE_ATOL
        max_step = 3600.0

    cfg = _Cfg()

    assert uses_legacy_adaptive_defaults(cfg.rtol, cfg.atol) is True

    normalize_solver_config_object(
        cfg,
        method_label="DOP853 (Adaptive)",
        upgrade_legacy_defaults=True,
    )

    assert cfg.rtol == DEFAULT_ADAPTIVE_RTOL
    assert cfg.atol == DEFAULT_ADAPTIVE_ATOL


def test_select_adaptive_sh_degree_clamps_table_rules_to_loaded_degree() -> None:
    degree = _select_adaptive_sh_degree(
        r_norm_m=1_737_400.0 + 20_000.0,
        r_ref_m=1_737_400.0,
        degree_max=100,
        adaptive_mode=1,
        adaptive_power=2.5,
        adaptive_min_degree=20,
        quantization_step=10,
        table_alt_km=np.asarray([10.0, 50.0, 200.0, 1000.0], dtype=np.float64),
        table_degree=np.asarray([100.0, 660.0, 80.0, 20.0], dtype=np.float64),
        table_len=4,
    )

    assert degree == 100


def test_select_adaptive_sh_degree_power_mode_quantizes_downward() -> None:
    degree = _select_adaptive_sh_degree(
        r_norm_m=2.0 * 1_737_400.0,
        r_ref_m=1_737_400.0,
        degree_max=100,
        adaptive_mode=2,
        adaptive_power=1.0,
        adaptive_min_degree=10,
        quantization_step=10,
        table_alt_km=np.zeros(0, dtype=np.float64),
        table_degree=np.zeros(0, dtype=np.float64),
        table_len=0,
    )

    assert degree == 50


def test_build_events_uses_topography_for_impact_detection_when_available() -> None:
    class _Grav:
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12
        degree_max = 100

    class _Tables:
        dt_s = 60.0
        q_i2f_tab = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    class _Ephem:
        tables = _Tables()

    class _Topo:
        def radius_m(self, lat_rad: float, lon_rad: float) -> float:
            return 1_737_900.0

    class _Dynamics:
        grav = _Grav()
        ephem = _Ephem()

    events = build_events(_Dynamics(), PropagatorConfig(), topo_grid=_Topo(), add_stop_event=False)
    impact_event = next(ev for ev in events if getattr(ev, "_event_role", "") == "impact")
    y = np.asarray([1_737_600.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Mean-radius altitude is still positive (+200 m), but the synthetic terrain
    # sits 500 m higher, so a topo-aware impact event must already be negative.
    assert impact_event(0.0, y) < 0.0


def test_make_telem_dict_includes_topography_clearance_when_available() -> None:
    y = np.array([1_738_400.0, 0.0, 0.0, 0.0, 1_000.0, 0.0], dtype=np.float64)

    telem = _make_telem_dict(
        t_s=42.0,
        y=y,
        R_ref_m=1_737_400.0,
        mu_m3s2=4.9048695e12,
        t_frame_s=123.0,
        r_i_to_bf=lambda t, r: np.asarray(r, dtype=np.float64),
        surface_radius_m=lambda lat_rad, lon_rad: 1_737_900.0,
    )

    assert telem is not None
    assert telem["surface_r_km"] == 1737.9
    assert telem["surface_alt_km"] == 0.5
    assert telem["terrain_clearance_km"] == 0.5


def test_live_telemetry_auto_y_range_keeps_following_new_data() -> None:
    if not HAS_PYQTGRAPH:
        return

    app = _app()
    widget = MultiTelemetryPlot()
    widget.show()
    app.processEvents()

    widget.plot_type_combo.setCurrentText("Altitude vs Time")
    widget.chk_auto_y.setChecked(True)

    widget.add_datapoint({"t_s": 0.0, "alt_km": 100.0})
    widget._flush_buffer()
    app.processEvents()
    first_y_range = widget.alt_viewbox.viewRange()[1]

    widget.add_datapoint({"t_s": 60.0, "alt_km": 250.0})
    widget._flush_buffer()
    app.processEvents()
    second_y_range = widget.alt_viewbox.viewRange()[1]

    assert second_y_range[1] > first_y_range[1]
    assert second_y_range[0] <= first_y_range[0]

    widget.close()
