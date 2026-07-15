"""Widget registry tests (Mission Monitor test group D, registry half).

Preset integrity tests live with the presets (Phase 2); this file covers the
registry contract itself: unique IDs, availability semantics, and graceful
handling of unknown/unimplemented widget ids.
"""

from __future__ import annotations

import pytest

from lunaris.ui.monitor.registry import MonitorWidgetRegistry, MonitorWidgetSpec


def spec(widget_id: str = "altitude", **overrides) -> MonitorWidgetSpec:
    base = dict(
        widget_id=widget_id,
        title="Altitude / Radius",
        category="Trajectory",
        description="Altitude history",
        required_channels=("altitude_m",),
        factory=lambda spec, controller: object(),
    )
    base.update(overrides)
    return MonitorWidgetSpec(**base)


class TestRegistration:
    def test_duplicate_widget_id_is_rejected(self):
        registry = MonitorWidgetRegistry()
        registry.register(spec())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec())

    def test_empty_widget_id_is_rejected(self):
        with pytest.raises(ValueError):
            spec(widget_id="")

    def test_unknown_widget_id_returns_none_for_placeholder_restore(self):
        registry = MonitorWidgetRegistry()
        assert registry.get("widget_that_was_removed") is None

    def test_registration_order_is_preserved(self):
        registry = MonitorWidgetRegistry()
        registry.register(spec("b_widget", category="Numerics"))
        registry.register(spec("a_widget", category="Trajectory"))
        assert [s.widget_id for s in registry.specs()] == ["b_widget", "a_widget"]
        assert registry.categories() == ("Numerics", "Trajectory")


class TestAvailability:
    def test_unimplemented_spec_is_declared_but_not_creatable(self):
        registry = MonitorWidgetRegistry()
        reserved = registry.register(spec("invariant_monitor", factory=None))
        assert not reserved.implemented
        assert registry.get("invariant_monitor") is reserved
        assert reserved not in registry.implemented_specs()
        assert reserved not in registry.available_specs(mode="live")

    def test_mode_filtering(self):
        registry = MonitorWidgetRegistry()
        registry.register(spec("live_only", supports_replay=False))
        registry.register(spec("replay_only", supports_live=False))
        live_ids = [s.widget_id for s in registry.available_specs(mode="live")]
        replay_ids = [s.widget_id for s in registry.available_specs(mode="replay")]
        assert live_ids == ["live_only"]
        assert replay_ids == ["replay_only"]

    def test_channel_filter_skips_guaranteed_empty_widgets(self):
        registry = MonitorWidgetRegistry()
        registry.register(spec("altitude", required_channels=("altitude_m",)))
        registry.register(spec("forces", required_channels=("force_components",)))
        registry.register(spec("provenance", required_channels=()))
        available = registry.available_specs(
            mode="live", available_channels=("altitude_m", "events")
        )
        ids = [s.widget_id for s in available]
        assert "altitude" in ids
        assert "provenance" in ids  # no requirements -> always openable
        assert "forces" not in ids

    def test_required_channel_partial_match_is_enough(self):
        # A widget that can render *any* of its channels is worth opening.
        registry = MonitorWidgetRegistry()
        registry.register(spec("alt_or_radius", required_channels=("altitude_m", "radius_m")))
        available = registry.available_specs(mode="live", available_channels=("radius_m",))
        assert [s.widget_id for s in available] == ["alt_or_radius"]
