"""Typed widget registry for the Mission Monitor (Qt-free).

Widgets are never constructed through if/elif chains: each widget kind is a
:class:`MonitorWidgetSpec` registered here, and the workspace builds the
"Add Widget" menu, presets, availability checks, and layout restoration from
the registry alone. A spec whose ``factory`` is ``None`` is *declared but not
implemented* (reserved second-phase widgets): it can appear in documentation
and be restored as a graceful placeholder, but never instantiated as live UI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

MonitorMode = Literal["live", "replay"]


@dataclass(frozen=True, slots=True)
class MonitorWidgetSpec:
    widget_id: str
    title: str
    category: str
    description: str
    #: Channels the widget needs to show *data*. The widget itself must still
    #: render an honest empty state while the channels have no values yet.
    required_channels: tuple[str, ...]
    supports_live: bool = True
    supports_replay: bool = True
    #: Singleton widgets can be open at most once per dashboard tab.
    singleton: bool = True
    #: ``factory(controller) -> QWidget``; None marks a reserved/unimplemented
    #: widget that restores as a placeholder.
    factory: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if not self.widget_id:
            raise ValueError("MonitorWidgetSpec.widget_id must be non-empty")
        if not self.title:
            raise ValueError(f"MonitorWidgetSpec {self.widget_id!r}: title must be non-empty")

    def supports_mode(self, mode: MonitorMode) -> bool:
        return self.supports_live if mode == "live" else self.supports_replay

    @property
    def implemented(self) -> bool:
        return self.factory is not None


class MonitorWidgetRegistry:
    """Ordered, duplicate-rejecting registry of widget specs."""

    def __init__(self) -> None:
        self._specs: dict[str, MonitorWidgetSpec] = {}

    def register(self, spec: MonitorWidgetSpec) -> MonitorWidgetSpec:
        if spec.widget_id in self._specs:
            raise ValueError(f"widget_id {spec.widget_id!r} is already registered")
        self._specs[spec.widget_id] = spec
        return spec

    def get(self, widget_id: str) -> MonitorWidgetSpec | None:
        """Spec for an id, or None for unknown ids (restore shows a placeholder)."""
        return self._specs.get(widget_id)

    def specs(self) -> tuple[MonitorWidgetSpec, ...]:
        return tuple(self._specs.values())

    def categories(self) -> tuple[str, ...]:
        seen: list[str] = []
        for spec in self._specs.values():
            if spec.category not in seen:
                seen.append(spec.category)
        return tuple(seen)

    def implemented_specs(self) -> tuple[MonitorWidgetSpec, ...]:
        return tuple(spec for spec in self._specs.values() if spec.implemented)

    def available_specs(
        self,
        *,
        mode: MonitorMode,
        available_channels: Iterable[str] | None = None,
    ) -> tuple[MonitorWidgetSpec, ...]:
        """Implemented specs usable in ``mode``.

        When ``available_channels`` is given, specs whose required channels are
        all absent are still returned — availability limits what a widget can
        *show*, not whether it can be opened (it opens into its honest
        "Channel unavailable" state). The channel filter is exposed for preset
        builders that prefer to skip guaranteed-empty widgets.
        """
        specs = [s for s in self.implemented_specs() if s.supports_mode(mode)]
        if available_channels is None:
            return tuple(specs)
        channel_set = set(available_channels)
        return tuple(
            s for s in specs
            if not s.required_channels or any(c in channel_set for c in s.required_channels)
        )


#: Process-wide default registry. Widget modules register their specs into it
#: at import time (see lunaris.ui.monitor.widgets).
DEFAULT_REGISTRY = MonitorWidgetRegistry()


__all__ = [
    "DEFAULT_REGISTRY",
    "MonitorMode",
    "MonitorWidgetRegistry",
    "MonitorWidgetSpec",
]
