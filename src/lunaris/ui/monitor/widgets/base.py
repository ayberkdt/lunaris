"""Shared frame, badges, empty states and error boundary for monitor widgets.

Every Mission Monitor widget lives inside a :class:`MonitorWidgetFrame`:

* a header with the widget title, its **unit/frame/source badges** (scientific
  honesty: the semantics are visible on the widget, not hidden in docs), and a
  **Live/Replay/Idle mode badge**;
* a body that switches between the real content and an honest
  :class:`EmptyState` ("Waiting for telemetry", "Channel unavailable ...") —
  no fake axes, no zero placeholders;
* an **error boundary**: an exception inside one widget's refresh turns that
  widget into a labelled error placeholder instead of crashing the workspace.

Subclasses implement :meth:`build_content` and :meth:`refresh`.
"""

from __future__ import annotations

import logging

from PySide6 import QtWidgets

from lunaris.ui.components.primitives import EmptyState
from lunaris.ui.monitor.registry import MonitorWidgetSpec
from lunaris.ui.monitor.store import TelemetryStore

_log = logging.getLogger(__name__)


def _repolish(widget: QtWidgets.QWidget) -> None:
    """Re-evaluate dynamic QSS properties (the repo's kind-property pattern)."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class MonitorWidgetFrame(QtWidgets.QFrame):
    """Base chrome for one monitor widget (see module docstring)."""

    def __init__(
        self,
        spec: MonitorWidgetSpec,
        controller,  # MonitorController; untyped to avoid an import cycle
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.controller = controller
        self._errored = False
        self._needs_refresh = False

        self.setObjectName("section")
        self.setProperty("elevated", True)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)
        title = QtWidgets.QLabel(spec.title)
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        self.badge_label = QtWidgets.QLabel("")
        self.badge_label.setObjectName("headerContextChip")
        self.badge_label.setVisible(False)
        header.addWidget(self.badge_label)
        header.addStretch(1)
        self.mode_badge = QtWidgets.QLabel("IDLE")
        self.mode_badge.setObjectName("dashBadge")
        header.addWidget(self.mode_badge)
        root.addLayout(header)

        self._stack = QtWidgets.QStackedWidget()
        self._empty = EmptyState("Waiting for telemetry",
                                 "Start a propagation run to populate this widget.")
        self._stack.addWidget(self._empty)
        self._content = self.build_content()
        self._stack.addWidget(self._content)
        root.addWidget(self._stack, 1)

        controller.updated.connect(self._on_update)
        controller.run_started.connect(self._on_run_started)
        controller.mode_changed.connect(self._update_mode_badge)
        self._update_mode_badge()
        self._on_update()

    # ------------------------------------------------------------ subclass API
    def build_content(self) -> QtWidgets.QWidget:
        raise NotImplementedError

    def refresh(self, store: TelemetryStore) -> None:
        raise NotImplementedError

    def set_badges(self, text: str, *, tooltip: str = "") -> None:
        """Show the unit/frame/source badge chip next to the title."""
        self.badge_label.setText(text)
        self.badge_label.setToolTip(tooltip or text)
        self.badge_label.setVisible(bool(text))

    def has_data(self, store: TelemetryStore) -> bool:
        """Whether the required channels carry any values yet."""
        required = self.spec.required_channels
        if not required:
            return store.n_samples > 0 or store.provenance is not None
        return any(store.has_channel(channel) for channel in required)

    # -------------------------------------------------------------- lifecycle
    def _on_run_started(self) -> None:
        # A new run clears a previous widget error: give the widget a fresh try.
        self._errored = False
        self._update_mode_badge()
        self._show_waiting_state()

    def _on_update(self) -> None:
        if self._errored:
            return
        if not self.isVisible():
            # Skip derived-metric work for hidden widgets; catch up on show.
            self._needs_refresh = True
            return
        self._do_refresh()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        if self._needs_refresh and not self._errored:
            self._do_refresh()

    def _do_refresh(self) -> None:
        self._needs_refresh = False
        store: TelemetryStore = self.controller.store
        try:
            if not self.has_data(store):
                if store.mode == "idle":
                    self._show_waiting_state()
                else:
                    self._show_unavailable_state(store)
                return
            self.refresh(store)
            self._stack.setCurrentWidget(self._content)
        except Exception:
            self._errored = True
            _log.exception("[monitor] widget %s failed to refresh", self.spec.widget_id)
            self._empty.set_message(
                "Widget error",
                f"The {self.spec.title} widget hit an internal error and was paused "
                "for this run. Other widgets are unaffected; details are in the log.",
            )
            self._stack.setCurrentWidget(self._empty)

    def _show_waiting_state(self) -> None:
        self._empty.set_message("Waiting for telemetry",
                                "Start a propagation run to populate this widget.")
        self._stack.setCurrentWidget(self._empty)

    def _show_unavailable_state(self, store: TelemetryStore) -> None:
        missing = ", ".join(self.spec.required_channels) or "n/a"
        self._empty.set_message(
            "Channel unavailable",
            f"This run/backend does not provide: {missing}. "
            "No substitute values are shown.",
        )
        self._stack.setCurrentWidget(self._empty)

    def _update_mode_badge(self) -> None:
        store: TelemetryStore = self.controller.store
        mode = store.mode
        if mode == "live":
            text, kind = "LIVE", "running"
            if store.outcome is not None:
                text, kind = "LIVE · ENDED", "completed"
                if store.outcome.success is False:
                    kind = "failed"
        elif mode == "replay":
            text, kind = "REPLAY", "completed"
        else:
            text, kind = "IDLE", ""
        self.mode_badge.setText(text)
        self.mode_badge.setProperty("kind", kind)
        _repolish(self.mode_badge)


class MissingWidgetPlaceholder(QtWidgets.QFrame):
    """Graceful stand-in for an unknown or not-yet-implemented widget id."""

    def __init__(self, widget_id: str, title: str | None = None,
                 description: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("section")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        shown_title = title or f"Unknown widget: {widget_id}"
        body = description or (
            "This widget is not available in this build. Its saved layout slot "
            "is preserved; remove it or update Lunaris to restore it."
        )
        layout.addWidget(EmptyState(shown_title, body))


__all__ = ["MissingWidgetPlaceholder", "MonitorWidgetFrame"]
