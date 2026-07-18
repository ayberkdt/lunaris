"""
Tests for the orbit-preview update logic (objective 3).

The state derivation and update scheduling are validated independently of
OpenGL rendering: rapid changes coalesce into a single scheduled update,
invalid intermediate input preserves the last valid orbit, the metric strip and
preview consume the same validated state, and a session load triggers exactly
one refresh. GL-object reuse / camera-stability are checked only when an OpenGL
backend is actually available.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

from lunaris.ui.core.ui_commons import R_MOON_KM as R
from lunaris.ui.pages.orbit_config_page import OrbitPage, OrbitViz3D, _OrbitState


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _page() -> OrbitPage:
    _app()
    return OrbitPage()


def test_rapid_changes_coalesce_into_one_scheduled_update() -> None:
    p = _page()
    try:
        p.btn_mode_classical.setChecked(True)
        p.ent_a.setText("2000")
        for _ in range(12):
            p._update_orbit_3d()
        # Many requests, one pending single-shot timer.
        assert p._orbit_update_timer.isActive()
        assert p._orbit_update_timer.isSingleShot()
    finally:
        p.deleteLater()


def test_invalid_intermediate_input_preserves_last_valid_state() -> None:
    p = _page()
    try:
        p.ent_hp.setText("120")
        p.ent_ha.setText("120")
        p._apply_orbit_update()
        good = p._last_valid_orbit
        assert good is not None

        # Temporary unparseable text must not overwrite the cached good orbit.
        p.ent_hp.setText("1e")
        p._apply_orbit_update()
        assert p._last_valid_orbit is good

        # Nor should an empty active field.
        p.ent_hp.setText("")
        assert p._compute_orbit_state() is None
        p._apply_orbit_update()
        assert p._last_valid_orbit is good
    finally:
        p.deleteLater()


def test_inverted_altitudes_are_not_silently_reinterpreted() -> None:
    p = _page()
    try:
        p.ent_hp.setText("200")
        p.ent_ha.setText("100")
        p._update_ghost_orbit()
        p._apply_orbit_update()

        assert not p.validate_inputs()
        assert p._compute_orbit_state() is None
        assert p.ent_hp.text() == "200"
        assert p.ent_ha.text() == "100"
        assert p.ent_a.text() == "—"
        assert p.ent_e.text() == "—"
        assert p.orbit_validation_notice.isVisibleTo(p)
        assert p.preview_validation_notice.isVisibleTo(p)
        assert all(
            label.text() == "—"
            for label in (
                p.lbl_period,
                p.lbl_hp,
                p.lbl_ha,
                p.lbl_ecc,
                p.lbl_inc,
                p.lbl_energy,
            )
        )

        p.btn_swap_apsides.click()
        assert p.validate_inputs()
        assert p.ent_hp.text() == "100"
        assert p.ent_ha.text() == "200"
        assert p.ent_a.text() != "—"
        assert p.ent_e.text() != "—"
        assert p.orbit_validation_notice.isHidden()
        assert p.preview_validation_notice.isHidden()
    finally:
        p.deleteLater()


def test_metric_strip_and_preview_use_same_validated_state() -> None:
    p = _page()
    try:
        p.btn_mode_classical.setChecked(True)
        p.ent_a.setText("3000")
        p.ent_e.setText("0.2")
        state = p._compute_orbit_state()
        assert isinstance(state, _OrbitState)
        assert abs(state.a_km - 3000.0) < 1e-9
        assert abs(state.e - 0.2) < 1e-9

        p._apply_orbit_update()
        # Eccentricity chip reflects exactly the validated state.
        assert p.lbl_ecc.text() == f"{state.e:.4f}"
        # Periselene chip derives from the same a/e.
        assert p.lbl_hp.text() == f"{state.a_km * (1 - state.e) - R:,.0f}"
    finally:
        p.deleteLater()


def test_compute_orbit_state_rejects_nonphysical_values() -> None:
    p = _page()
    try:
        p.btn_mode_classical.setChecked(True)
        p.ent_a.setText("0")            # non-positive semi-major axis
        assert p._compute_orbit_state() is None
        p.ent_a.setText("-100")
        assert p._compute_orbit_state() is None
    finally:
        p.deleteLater()


def test_session_load_results_in_one_validated_refresh() -> None:
    p = _page()
    try:
        p.load_data({"mode": "circular", "alt_km": 90.0, "inc_deg": 30.0})
        # load_data schedules a single debounced refresh.
        assert p._orbit_update_timer.isActive()
        p._apply_orbit_update()
        state = p._compute_orbit_state()
        assert abs(state.a_km - (R + 90.0)) < 1e-6
        assert abs(state.inc_deg - 30.0) < 1e-9
        assert abs(state.e) < 1e-9
    finally:
        p.deleteLater()


def test_wide_workspace_prioritizes_preview_and_compacts_metrics() -> None:
    app = _app()
    p = _page()
    p.resize(1400, 760)
    p.show()
    app.processEvents()
    try:
        p._update_responsive_layout()
        app.processEvents()
        form_width, preview_width = p._split.sizes()
        assert p._split.orientation() == QtCore.Qt.Horizontal
        assert form_width <= 620
        assert preview_width > form_width
        assert p.group_viz.property("elevated") is True
        assert p._metric_columns == 6
    finally:
        p.close()
        p.deleteLater()
        app.processEvents()


def test_orbit_toolbar_actions_fit_at_minimum_two_pane_width() -> None:
    app = _app()
    p = _page()
    p.resize(980, 720)
    p.show()
    app.processEvents()
    try:
        buttons = (
            *p.orbit_viz_3d._cam_buttons,
            p.orbit_viz_3d._fit_button,
            p.orbit_viz_3d._focus_button,
        )
        assert p.orbit_viz_3d._compact_controls
        assert all(button.width() >= button.sizeHint().width() for button in buttons)
        assert p.orbit_viz_3d._focus_button.accessibleName()
    finally:
        p.close()
        p.deleteLater()
        app.processEvents()


def test_preview_focus_mode_is_reversible_without_recreating_widgets() -> None:
    app = _app()
    p = _page()
    p.resize(1280, 760)
    p.show()
    app.processEvents()
    try:
        form_id = id(p.group_params)
        preview_id = id(p.orbit_viz_3d)
        original_sizes = p._split.sizes()

        p._toggle_preview_focus()
        app.processEvents()
        assert p._preview_focus_active
        assert p._params_scroll.isHidden()
        assert p.orbit_viz_3d._focus_button.text() == "Exit focus"

        p._exit_preview_focus()
        app.processEvents()
        assert not p._preview_focus_active
        assert p._params_scroll.isVisibleTo(p)
        assert p._split.sizes() == original_sizes
        assert id(p.group_params) == form_id
        assert id(p.orbit_viz_3d) == preview_id
    finally:
        p.close()
        p.deleteLater()
        app.processEvents()


def test_camera_fit_accounts_for_wide_shallow_viewport() -> None:
    class FakeScene:
        def __init__(self, width: int, height: int):
            self._width = width
            self._height = height
            self.opts = {"fov": 60.0}
            self.distance = 0.0

        def width(self) -> int:
            return self._width

        def height(self) -> int:
            return self._height

        def setCameraPosition(self, **kwargs) -> None:
            self.distance = float(kwargs["distance"])

    _app()
    viz = OrbitViz3D()
    try:
        viz._a_km = 2000.0
        viz._e = 0.1
        square = FakeScene(600, 600)
        viz.gl_widget = square
        viz._fit_camera(mark_user=False)

        wide = FakeScene(900, 300)
        viz.gl_widget = wide
        viz._fit_camera(mark_user=False)

        assert wide.distance > square.distance
        assert wide.distance > 3.4 * viz._a_km * (1.0 + viz._e)
    finally:
        viz.deleteLater()


# --------------------------------------------------------------------------- #
# OpenGL-dependent reuse / camera stability (skipped without a GL backend)
# --------------------------------------------------------------------------- #
def _viz_with_gl():
    _app()
    viz = OrbitViz3D()
    if getattr(viz, "gl_widget", None) is None:
        viz.deleteLater()
        pytest.skip("OpenGL backend unavailable in this environment")
    return viz


def test_orbit_line_object_is_reused_across_updates() -> None:
    viz = _viz_with_gl()
    try:
        viz.set_orbit_params(2000.0, 0.0, 90.0, 0.0, 0.0, 0.0)
        line_id = id(viz.orbit_line)
        glow_id = id(viz.orbit_glow)
        marker_id = id(viz.periapsis_marker)
        viz.set_orbit_params(2500.0, 0.1, 80.0, 10.0, 20.0, 30.0)
        # Same Python objects updated in place, not recreated.
        assert id(viz.orbit_line) == line_id
        assert id(viz.orbit_glow) == glow_id
        assert id(viz.periapsis_marker) == marker_id
    finally:
        viz.deleteLater()


def test_camera_is_not_reset_when_parameters_change() -> None:
    viz = _viz_with_gl()
    try:
        viz.set_orbit_params(2000.0, 0.0, 90.0, 0.0, 0.0, 0.0)
        opts = dict(viz.gl_widget.opts)
        before = (opts.get("distance"), opts.get("elevation"), opts.get("azimuth"))
        viz.set_orbit_params(5000.0, 0.3, 45.0, 33.0, 12.0, 60.0)
        after = (
            viz.gl_widget.opts.get("distance"),
            viz.gl_widget.opts.get("elevation"),
            viz.gl_widget.opts.get("azimuth"),
        )
        assert before == after
    finally:
        viz.deleteLater()


def test_disk_meshdata_faces_index_only_existing_vertices() -> None:
    """Regression: the plane-disk fan indexed vertex ``n+1`` (out of bounds)
    and built degenerate ``[0, k, k]`` triangles, crashing GLMeshItem.paint
    with ``IndexError: index 97 is out of bounds for axis 0 with size 97``."""
    from lunaris.ui.pages.orbit_config_page import HAS_OPENGL

    if not HAS_OPENGL:
        pytest.skip("pyqtgraph.opengl not available")

    md = OrbitViz3D._make_disk_meshdata(n=96)
    verts = md.vertexes()
    faces = md.faces()
    assert verts.shape == (97, 3)
    assert faces.shape == (96, 3)
    assert faces.min() >= 0
    assert faces.max() < verts.shape[0]
    # No degenerate triangles: all three corners distinct in every face.
    assert (faces[:, 0] != faces[:, 1]).all()
    assert (faces[:, 1] != faces[:, 2]).all()
    assert (faces[:, 0] != faces[:, 2]).all()
    # Fan closes: last face wraps back to the first ring vertex.
    assert faces[-1].tolist() == [0, 96, 1]
