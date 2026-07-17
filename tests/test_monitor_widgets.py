"""Mission Monitor widget tests (groups F/G/H: states, honesty, presentation).

Runs offscreen (QT_QPA_PLATFORM=offscreen). Covers: honest empty states (no
fake axes/zeros), singular orbital elements, provenance unavailability,
event deduplication, the widget-level error boundary, and the visibility of
unit/frame/mode badges.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtWidgets

from lunaris.common.telemetry_contract import (
    TelemetryProvenance,
    TelemetrySample,
    encode_meta_line,
    encode_sample_line,
)
from lunaris.ui.monitor.formatting import UNAVAILABLE
from lunaris.ui.monitor.widgets.altitude import ALTITUDE_SPEC, AltitudeWidget
from lunaris.ui.monitor.widgets.backend_provenance import (
    BACKEND_PROVENANCE_SPEC,
    BackendProvenanceWidget,
)
from lunaris.ui.monitor.widgets.base import MonitorWidgetFrame
from lunaris.ui.monitor.widgets.event_timeline import EVENT_TIMELINE_SPEC, EventTimelineWidget
from lunaris.ui.monitor.widgets.integrator_health import (
    INTEGRATOR_HEALTH_SPEC,
    IntegratorHealthWidget,
)
from lunaris.ui.monitor.widgets.orbital_elements import (
    ORBITAL_ELEMENTS_SPEC,
    OrbitalElementsWidget,
)
from lunaris.ui.monitor.workspace import MonitorController


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def controller():
    app = _app()
    ctrl = MonitorController()
    yield ctrl
    app.processEvents()


def make_sample(seq: int, t: float, **extra) -> TelemetrySample:
    extra.setdefault("sample_kind", "output_state")
    return TelemetrySample(run_id="run_t", sequence_id=seq, simulation_time_s=t, **extra)


def feed_samples(ctrl: MonitorController, *samples: TelemetrySample) -> None:
    ctrl.begin_live_run()
    for sample in samples:
        ctrl.feed_line(encode_sample_line(sample))
    ctrl.flush_now()
    _app().processEvents()


def _shown(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    widget.resize(600, 400)
    widget.show()
    _app().processEvents()
    return widget


class TestEmptyStates:
    def test_altitude_shows_waiting_state_not_fake_axes(self, controller):
        widget = _shown(AltitudeWidget(ALTITUDE_SPEC, controller))
        assert widget._stack.currentWidget() is widget._empty
        assert "Waiting for telemetry" in widget._empty.title_label.text()

    def test_missing_channel_shows_unavailable_not_zero(self, controller):
        widget = _shown(AltitudeWidget(ALTITUDE_SPEC, controller))
        # A run with samples but no altitude/radius/terrain channels at all.
        feed_samples(controller, make_sample(0, 10.0, diagnostics={"x": 1}))
        widget._do_refresh()
        assert widget._stack.currentWidget() is widget._empty
        assert "Channel unavailable" in widget._empty.title_label.text()

    def test_elements_widget_without_element_channels_is_unavailable(self, controller):
        widget = _shown(OrbitalElementsWidget(ORBITAL_ELEMENTS_SPEC, controller))
        feed_samples(controller, make_sample(0, 10.0, altitude_m=1000.0))
        widget._do_refresh()
        assert widget._stack.currentWidget() is widget._empty


class TestAltitudeWidget:
    def test_values_and_badges_after_samples(self, controller):
        widget = _shown(AltitudeWidget(ALTITUDE_SPEC, controller))
        feed_samples(
            controller,
            make_sample(0, 0.0, altitude_m=50_000.0, radius_m=1_787_400.0),
            make_sample(1, 60.0, altitude_m=48_000.0, radius_m=1_785_400.0),
        )
        widget._do_refresh()
        assert widget._stack.currentWidget() is widget._content
        assert widget.value_labels["current"].text() != UNAVAILABLE
        assert "48" in widget.value_labels["min"].text()
        # Unit and mode are visible on the widget (group H).
        assert widget.badge_label.isVisibleTo(widget)
        assert "km" in widget.badge_label.text()
        assert widget.mode_badge.text() == "LIVE"
        # The altitude definition names R_ref explicitly.
        assert "R_ref" in widget.definition_label.text()

    def test_metric_selector_only_offers_present_channels(self, controller):
        widget = _shown(AltitudeWidget(ALTITUDE_SPEC, controller))
        feed_samples(controller, make_sample(0, 0.0, altitude_m=50_000.0))
        widget._do_refresh()
        options = [widget.metric_combo.itemData(i) for i in range(widget.metric_combo.count())]
        assert "altitude_m" in options
        assert "terrain_clearance_m" not in options  # not provided -> not offered


class TestOrbitalElements:
    def test_circular_orbit_shows_undefined_not_zero(self, controller):
        widget = _shown(OrbitalElementsWidget(ORBITAL_ELEMENTS_SPEC, controller))
        feed_samples(controller, make_sample(
            0, 0.0,
            orbital_elements={"sma_m": 1.8e6, "ecc": 1e-12, "inc_rad": 1.5, "nu_rad": 0.4},
        ))
        widget._do_refresh()
        assert widget._stack.currentWidget() is widget._content
        assert "undefined (circular orbit)" in widget.value_labels["elements.argp_rad"].text()
        assert widget.value_labels["elements.ecc"].text() != UNAVAILABLE

    def test_equatorial_orbit_marks_raan_undefined(self, controller):
        widget = _shown(OrbitalElementsWidget(ORBITAL_ELEMENTS_SPEC, controller))
        feed_samples(controller, make_sample(
            0, 0.0,
            orbital_elements={"sma_m": 1.8e6, "ecc": 0.2, "inc_rad": 0.0,
                              "argp_rad": 1.0, "nu_rad": 0.4},
        ))
        widget._do_refresh()
        assert "undefined (equatorial orbit)" in widget.value_labels["elements.raan_rad"].text()

    def test_frame_badge_visible(self, controller):
        widget = _shown(OrbitalElementsWidget(ORBITAL_ELEMENTS_SPEC, controller))
        feed_samples(controller, make_sample(
            0, 0.0, frame_inertial="moon_centered_inertial",
            orbital_elements={"sma_m": 1.8e6, "ecc": 0.01},
        ))
        widget._do_refresh()
        assert "moon_centered_inertial" in widget.badge_label.text()
        assert "derived" in widget.badge_label.text()


class TestIntegratorHealth:
    def test_only_reported_fields_appear(self, controller):
        widget = _shown(IntegratorHealthWidget(INTEGRATOR_HEALTH_SPEC, controller))
        feed_samples(controller, make_sample(0, 120.0, wall_time_s=2.0))
        widget._do_refresh()
        assert "sim_time" in widget._row_labels
        assert "throughput" in widget._row_labels
        assert "diag.nfev" not in widget._row_labels  # no [DIAG] yet -> no fake row

    def test_diag_merge_adds_engine_rows(self, controller):
        widget = _shown(IntegratorHealthWidget(INTEGRATOR_HEALTH_SPEC, controller))
        feed_samples(controller, make_sample(0, 120.0))
        controller.set_run_diagnostics({
            "nfev": 4213.0, "integration_backend": "scipy", "stop_reason": "impact",
        })
        controller.flush_now()
        widget._do_refresh()
        assert widget._row_labels["diag.nfev"].text() == "4,213"
        assert widget._row_labels["diag.integration_backend"].text() == "scipy"
        assert widget._row_labels["diag.stop_reason"].text() == "impact"


class TestEventTimeline:
    def test_events_are_deduplicated_and_ordered(self, controller):
        from lunaris.common.telemetry_contract import TelemetryEvent

        widget = _shown(EventTimelineWidget(EVENT_TIMELINE_SPEC, controller))
        controller.begin_live_run()
        controller.store.append(make_sample(0, 100.0))
        controller.store.add_event(TelemetryEvent("periselene", 60.0, "pass 1"))
        controller.store.add_event(TelemetryEvent("periselene", 60.0, "pass 1"))  # dup
        controller.store.add_event(TelemetryEvent("impact", 90.0, "surface hit", "critical"))
        controller.flush_now()
        widget._do_refresh()
        assert widget.table.topLevelItemCount() == 2
        assert widget.table.topLevelItem(0).text(1) == "periselene"
        assert widget.table.topLevelItem(1).text(1) == "impact"


class TestBackendProvenance:
    def test_meta_fields_and_unavailable_rows(self, controller):
        widget = _shown(BackendProvenanceWidget(BACKEND_PROVENANCE_SPEC, controller))
        controller.begin_live_run()
        controller.feed_line(encode_meta_line(TelemetryProvenance(
            run_id="run_t",
            integrator="DOP853",
            gravity_backend="classic_sh",
            gravity_model="gggrx_1200a.tab",
            sh_degree=120,
            fallback_reason="CUDA unavailable; using CPU reference",
        )))
        controller.flush_now()
        widget._do_refresh()
        assert widget._stack.currentWidget() is widget._content
        assert widget._row_labels["integrator"].text() == "DOP853"
        assert widget._row_labels["sh_degree"].text() == "120"
        # Unknown facts stay visibly unavailable, never guessed.
        assert widget._row_labels["device"].text() == UNAVAILABLE
        assert widget._row_labels["requested_backend"].text() == UNAVAILABLE
        # Fallback is surfaced prominently.
        assert widget.fallback_notice.isVisibleTo(widget)
        assert "CUDA unavailable" in widget.fallback_notice.text()

    def test_effective_backend_comes_from_diag(self, controller):
        widget = _shown(BackendProvenanceWidget(BACKEND_PROVENANCE_SPEC, controller))
        controller.begin_live_run()
        controller.store.append(make_sample(0, 1.0))
        controller.set_run_diagnostics({"rhs_path": "numba_sh", "integration_backend": "scipy"})
        controller.flush_now()
        widget._do_refresh()
        assert widget._row_labels["effective_backend"].text() == "numba_sh"


class TestErrorBoundary:
    def test_one_broken_widget_does_not_break_others(self, controller):
        class BrokenWidget(MonitorWidgetFrame):
            def build_content(self):
                return QtWidgets.QWidget()

            def refresh(self, store):
                raise RuntimeError("intentional test failure")

        broken = _shown(BrokenWidget(INTEGRATOR_HEALTH_SPEC, controller))
        healthy = _shown(IntegratorHealthWidget(INTEGRATOR_HEALTH_SPEC, controller))
        feed_samples(controller, make_sample(0, 10.0))
        broken._do_refresh()
        healthy._do_refresh()
        assert broken._errored
        assert "Widget error" in broken._empty.title_label.text()
        assert not healthy._errored
        assert healthy._stack.currentWidget() is healthy._content

    def test_new_run_clears_the_error_latch(self, controller):
        class FlakyWidget(MonitorWidgetFrame):
            fail = True

            def build_content(self):
                return QtWidgets.QWidget()

            def refresh(self, store):
                if self.fail:
                    raise RuntimeError("boom")

        widget = _shown(FlakyWidget(INTEGRATOR_HEALTH_SPEC, controller))
        feed_samples(controller, make_sample(0, 10.0))
        widget._do_refresh()
        assert widget._errored
        widget.fail = False
        controller.begin_live_run()
        assert not widget._errored


class TestStateVector:
    def _state(self, scale: float = 1.0) -> tuple[float, ...]:
        return (1.8e6 * scale, 2.0e5, -3.0e5, 100.0, -1600.0, 30.0)

    def test_values_norms_and_epoch(self, controller):
        from lunaris.ui.monitor.widgets.state_vector import (
            STATE_VECTOR_SPEC,
            StateVectorWidget,
        )

        widget = _shown(StateVectorWidget(STATE_VECTOR_SPEC, controller))
        feed_samples(controller, make_sample(
            0, 120.0, state_inertial=self._state(), frame_inertial="moon_centered_inertial",
        ))
        widget._do_refresh()
        assert widget._stack.currentWidget() is widget._content
        assert widget.value_labels["x"].text().endswith("km")
        assert "1,800.000000 km" == widget.value_labels["x"].text()
        assert widget.value_labels["v_norm"].text().endswith("km/s")
        assert "2.0 min" in widget.value_labels["epoch"].text()
        assert "moon_centered_inertial" in widget.badge_label.text()

    def test_body_fixed_segment_disabled_when_channel_missing(self, controller):
        from lunaris.ui.monitor.widgets.state_vector import (
            STATE_VECTOR_SPEC,
            StateVectorWidget,
        )

        widget = _shown(StateVectorWidget(STATE_VECTOR_SPEC, controller))
        feed_samples(controller, make_sample(0, 1.0, state_inertial=self._state()))
        widget._do_refresh()
        fixed_button = widget.frame_control.buttons[1]
        assert not fixed_button.isEnabled()
        assert "unavailable" in fixed_button.toolTip().lower()

    def test_body_fixed_channel_enables_the_segment(self, controller):
        from lunaris.ui.monitor.widgets.state_vector import (
            STATE_VECTOR_SPEC,
            StateVectorWidget,
        )

        widget = _shown(StateVectorWidget(STATE_VECTOR_SPEC, controller))
        feed_samples(controller, make_sample(
            0, 1.0, state_inertial=self._state(), state_fixed=self._state(2.0),
            frame_fixed="moon_fixed",
        ))
        widget._do_refresh()
        assert widget.frame_control.buttons[1].isEnabled()
        widget.frame_control.set_current_index(1)
        widget._do_refresh()
        assert "3,600.000000 km" == widget.value_labels["x"].text()
        assert "moon_fixed" in widget.badge_label.text()


class TestOrbitView:
    def test_offscreen_platform_gets_an_explicit_fallback(self, controller):
        from lunaris.ui.monitor.widgets.orbit_view import ORBIT_VIEW_SPEC, OrbitViewWidget

        widget = _shown(OrbitViewWidget(ORBIT_VIEW_SPEC, controller))
        # Offscreen CI: no GL surface -> honest note, no fake scene, no crash.
        assert widget.gl_widget is None
        feed_samples(controller, make_sample(
            0, 0.0, state_inertial=(1.8e6, 0.0, 0.0, 0.0, 1600.0, 0.0),
        ))
        widget._do_refresh()  # refresh with data must stay safe without GL
        assert not widget._errored
        assert "km" in widget.badge_label.text()

    def test_impact_marker_position_comes_from_nearest_state(self, controller):
        from lunaris.common.telemetry_contract import TelemetryEvent
        from lunaris.ui.monitor.widgets.orbit_view import ORBIT_VIEW_SPEC, OrbitViewWidget

        widget = _shown(OrbitViewWidget(ORBIT_VIEW_SPEC, controller))
        feed_samples(
            controller,
            make_sample(0, 0.0, state_inertial=(1.8e6, 0.0, 0.0, 0.0, 1600.0, 0.0)),
            make_sample(1, 60.0, state_inertial=(1.75e6, 1.0e5, 0.0, 0.0, 1600.0, 0.0)),
        )
        controller.store.add_event(TelemetryEvent("impact", 60.0, "hit", "critical"))
        pos = widget._impact_position(controller.store)
        assert pos is not None
        assert pos[0] == pytest.approx(1750.0)  # km


class TestModeBadge:
    def test_live_then_ended(self, controller):
        widget = _shown(IntegratorHealthWidget(INTEGRATOR_HEALTH_SPEC, controller))
        feed_samples(controller, make_sample(0, 10.0))
        widget._update_mode_badge()
        assert widget.mode_badge.text() == "LIVE"
        controller.finish_live_run(exit_code=0)
        widget._update_mode_badge()
        assert widget.mode_badge.text() == "LIVE · ENDED"
