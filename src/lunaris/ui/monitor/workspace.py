"""Mission Monitor controller, dockable workspace and navigation page.

Architecture (one direction, loosely coupled):

    stdout lines (app.py, already line-assembled)
        → MonitorController.feed_line()   [classify → bounded store]
        → QTimer-batched ``updated`` signal (UI cadence, ≤ ~16 Hz)
        → MonitorWidgetFrame subclasses re-render from store snapshots

The controller owns the store and the paint cadence: widgets never see raw
protocol lines, and the propagation process never sees widgets. Everything
runs on the Qt main thread (QProcess ``readyRead`` handlers are delivered
there), so the store needs no locks.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lunaris.ui.monitor.persistence import MonitorLayout, TabLayout

from PySide6 import QtCore, QtWidgets

from lunaris.common.telemetry_contract import TelemetryEvent
from lunaris.ui.monitor import widgets as _monitor_widgets  # registers specs
from lunaris.ui.monitor.presets import (
    DEFAULT_PRESET_ID,
    PRESETS,
    preset_by_id,
    split_preset,
)
from lunaris.ui.monitor.protocol import (
    MetaMessage,
    ProtocolProblem,
    SampleMessage,
    TelemetryLineClassifier,
    TelemetryMessage,
)
from lunaris.ui.monitor.registry import DEFAULT_REGISTRY, MonitorWidgetRegistry
from lunaris.ui.monitor.store import RunOutcome, TelemetryStore
from lunaris.ui.monitor.widgets.base import MissingWidgetPlaceholder

#: Repaint batching interval — decouples UI cadence from telemetry cadence.
UI_BATCH_INTERVAL_MS = 60


class MonitorController(QtCore.QObject):
    """Owns the telemetry store, the protocol classifier and the UI cadence."""

    updated = QtCore.Signal()
    run_started = QtCore.Signal()
    run_finished = QtCore.Signal()
    meta_received = QtCore.Signal()
    mode_changed = QtCore.Signal()
    protocol_problem = QtCore.Signal(str)
    replay_loaded = QtCore.Signal()
    replay_failed = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        *,
        store: TelemetryStore | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store or TelemetryStore()
        self._classifier = TelemetryLineClassifier()
        #: Replay cursor (simulation seconds); None = follow the latest sample.
        self.cursor_time_s: float | None = None
        #: Last replay artifact the user opened (persisted with the layout).
        self.last_replay_path: str | None = None
        #: Batch/ensemble observability. Deliberately *not* stored in the
        #: telemetry store: a batch subprocess can run concurrently with a
        #: single-run live session, and its payloads must never overwrite the
        #: live run's provenance or samples.
        self.batch_progress: dict[str, Any] | None = None
        self.batch_metrics: dict[str, Any] | None = None
        self._dirty = False
        self._problem_kinds_warned: set[str] = set()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(UI_BATCH_INTERVAL_MS)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._flush)

    # ------------------------------------------------------------- lifecycle
    def begin_live_run(self, *, expected_duration_s: float | None = None) -> str:
        """Reset for a freshly launched propagation process."""
        run_id = f"live-{time.strftime('%Y%m%d-%H%M%S')}"
        self._classifier.begin_run(run_id)
        self._problem_kinds_warned.clear()
        self.cursor_time_s = None
        self.store.begin_run(run_id, mode="live", expected_duration_s=expected_duration_s)
        self.run_started.emit()
        self.mode_changed.emit()
        self._mark_dirty()
        return run_id

    def finish_live_run(self, *, exit_code: int | None = None, reason: str = "") -> None:
        if self.store.mode == "idle":
            return
        success = (exit_code == 0) if exit_code is not None else None
        text = reason or (
            "completed" if success else
            f"exited with code {exit_code}" if exit_code is not None else "finished"
        )
        self.store.finish_run(RunOutcome(reason=text, exit_code=exit_code, success=success))
        bounds = self.store.time_bounds()
        end_t = bounds[1] if bounds is not None else 0.0
        self.store.add_event(TelemetryEvent(
            event_type="run_finished" if success in (True, None) else "run_failed",
            simulation_time_s=end_t,
            message=text,
            severity="info" if success in (True, None) else "warning",
        ))
        self.run_finished.emit()
        self.mode_changed.emit()
        self._mark_dirty()

    # ----------------------------------------------------------------- feeds
    def feed_line(self, line: str) -> TelemetryMessage | None:
        """Classify one complete stdout line; returns None for ordinary logs."""
        message = self._classifier.classify(line)
        if message is None:
            return None
        if isinstance(message, SampleMessage):
            self.store.append(message.sample)
            self._mark_dirty()
        elif isinstance(message, MetaMessage):
            self.store.set_provenance(message.provenance)
            if message.provenance.fallback_reason:
                self.store.add_event(TelemetryEvent(
                    event_type="backend_fallback",
                    simulation_time_s=0.0,
                    message=message.provenance.fallback_reason,
                    severity="warning",
                ))
            self.meta_received.emit()
            self._mark_dirty()
        elif isinstance(message, ProtocolProblem):
            if message.kind not in self._problem_kinds_warned:
                self._problem_kinds_warned.add(message.kind)
                self.protocol_problem.emit(message.detail)
        return message

    def feed_legacy_mapping(self, payload: Mapping[str, Any]) -> bool:
        """Adopt an already-parsed legacy telemetry dict (python-repr lines)."""
        message = self._classifier.adapt_legacy_mapping(payload)
        if message is None:
            return False
        self.store.append(message.sample)
        self._mark_dirty()
        return True

    def set_run_diagnostics(self, payload: Mapping[str, Any]) -> None:
        """Merge the end-of-run [DIAG] payload and derive its discrete events."""
        self.store.set_run_diagnostics(dict(payload))
        t_impact = payload.get("t_impact_s")
        if payload.get("impacted") and isinstance(t_impact, int | float):
            self.store.add_event(TelemetryEvent(
                event_type="impact",
                simulation_time_s=float(t_impact),
                message="Impact detected by the engine.",
                severity="critical",
            ))
        stop_reason = payload.get("stop_reason")
        if isinstance(stop_reason, str) and stop_reason:
            bounds = self.store.time_bounds()
            self.store.add_event(TelemetryEvent(
                event_type="terminal_event",
                simulation_time_s=bounds[1] if bounds else 0.0,
                message=stop_reason,
                severity="warning",
            ))
        self._mark_dirty()

    # ----------------------------------------------------------------- batch
    def begin_batch_run(self) -> None:
        """Reset batch observability when a new batch subprocess starts."""
        self.batch_progress = None
        self.batch_metrics = None
        self._mark_dirty()

    def set_batch_progress(self, payload: Mapping[str, Any]) -> None:
        self.batch_progress = dict(payload)
        self._mark_dirty()

    def set_batch_metrics(self, payload: Mapping[str, Any]) -> None:
        self.batch_metrics = dict(payload)
        self._mark_dirty()

    # ---------------------------------------------------------------- replay
    def jump_to_time(self, t_s: float) -> None:
        """Move the shared timeline cursor (replay mode only)."""
        if self.store.mode != "replay":
            return
        self.cursor_time_s = float(t_s)
        self._mark_dirty()

    def enter_replay_of_live(self) -> None:
        """Review the finished live run in place (same store, replay controls)."""
        if self.store.n_samples == 0:
            return
        self.store.enter_replay()
        self.mode_changed.emit()
        self._mark_dirty()

    def open_replay_file(self, path: str) -> None:
        """Load a telemetry.ndjson artifact in a worker thread (fail-closed)."""
        from lunaris.ui.monitor.replay import ReplayLoader

        self.stop_replay_loader()
        self.cursor_time_s = None
        self.last_replay_path = str(path)
        loader = ReplayLoader(str(path), self)
        loader.count_ready.connect(self._on_replay_count)
        loader.meta_ready.connect(self._on_replay_meta)
        loader.batch_ready.connect(self._on_replay_batch)
        loader.finished_ok.connect(self._on_replay_finished)
        loader.failed.connect(self._on_replay_failed)
        self._replay_loader = loader
        loader.start()

    def stop_replay_loader(self) -> None:
        loader = getattr(self, "_replay_loader", None)
        if loader is not None and loader.isRunning():
            loader.requestInterruption()
            loader.wait(2000)
        self._replay_loader = None

    def _on_replay_count(self, n_samples: int) -> None:
        # Fresh, right-sized store: replay must hold the entire (capped)
        # artifact, not compete with the live-run ring capacity.
        self.store = TelemetryStore(capacity=max(int(n_samples), 16))
        self.store.begin_run("replay", mode="replay")
        self.mode_changed.emit()
        self._mark_dirty()

    def _on_replay_meta(self, provenance) -> None:
        self.store.set_provenance(provenance)
        if provenance.fallback_reason:
            self.store.add_event(TelemetryEvent(
                event_type="backend_fallback",
                simulation_time_s=0.0,
                message=provenance.fallback_reason,
                severity="warning",
            ))
        self.meta_received.emit()
        self._mark_dirty()

    def _on_replay_batch(self, samples) -> None:
        self.store.extend(list(samples))
        self._mark_dirty()

    def _on_replay_finished(self, _delivered: int) -> None:
        self.replay_loaded.emit()
        self._mark_dirty()

    def _on_replay_failed(self, detail: str) -> None:
        self.replay_failed.emit(detail)
        self._mark_dirty()

    # -------------------------------------------------------------- plumbing
    def _mark_dirty(self) -> None:
        self._dirty = True
        if not self._timer.isActive():
            self._timer.start()

    def _flush(self) -> None:
        if self._dirty:
            self._dirty = False
            self.updated.emit()

    def flush_now(self) -> None:
        """Synchronous flush for tests and mode transitions."""
        self._timer.stop()
        self._flush()


class MonitorWorkspace(QtWidgets.QWidget):
    """Dockable multi-widget dashboard (nested QMainWindow + QDockWidgets)."""

    def __init__(
        self,
        controller: MonitorController,
        registry: MonitorWidgetRegistry | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.registry = registry or DEFAULT_REGISTRY
        self._docks: dict[str, QtWidgets.QDockWidget] = {}
        self.active_preset_id = DEFAULT_PRESET_ID

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("toolbar")
        bar = QtWidgets.QHBoxLayout(toolbar)
        bar.setContentsMargins(8, 6, 8, 6)
        bar.setSpacing(8)

        preset_label = QtWidgets.QLabel("Preset")
        preset_label.setObjectName("keyLabel")
        bar.addWidget(preset_label)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setAccessibleName("Workspace preset")
        self.preset_combo.setToolTip("Workspace preset (which widgets are open)")
        for preset in PRESETS:
            self.preset_combo.addItem(preset.title, preset.preset_id)
        self.preset_combo.activated.connect(self._on_preset_activated)
        bar.addWidget(self.preset_combo)

        self.add_button = QtWidgets.QToolButton()
        self.add_button.setText("Add Widget")
        self.add_button.setToolTip("Open a monitor widget from the registry")
        self.add_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.add_menu = QtWidgets.QMenu(self.add_button)
        self.add_menu.aboutToShow.connect(self._rebuild_add_menu)
        self.add_button.setMenu(self.add_menu)
        bar.addWidget(self.add_button)

        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setText("Reset Layout")
        self.reset_button.setToolTip("Re-apply the active preset's default layout")
        self.reset_button.clicked.connect(lambda: self.apply_preset(self.active_preset_id))
        bar.addWidget(self.reset_button)

        self.open_replay_button = QtWidgets.QToolButton()
        self.open_replay_button.setText("Open Replay…")
        self.open_replay_button.setToolTip(
            "Open a completed run's telemetry.ndjson artifact for replay"
        )
        self.open_replay_button.clicked.connect(self._on_open_replay)
        bar.addWidget(self.open_replay_button)

        self.review_button = QtWidgets.QToolButton()
        self.review_button.setText("Review Run")
        self.review_button.setToolTip(
            "Switch the finished live run into replay mode (same data, timeline controls)"
        )
        self.review_button.setEnabled(False)
        self.review_button.clicked.connect(self.controller.enter_replay_of_live)
        self.controller.run_finished.connect(lambda: self.review_button.setEnabled(True))
        self.controller.run_started.connect(lambda: self.review_button.setEnabled(False))
        bar.addWidget(self.review_button)

        bar.addStretch(1)
        self.skipped_label = QtWidgets.QLabel("")
        self.skipped_label.setObjectName("fieldHint")
        self.skipped_label.setProperty("kind", "info")
        self.skipped_label.setVisible(False)
        bar.addWidget(self.skipped_label)
        root.addWidget(toolbar)

        self._host = QtWidgets.QMainWindow()
        self._host.setWindowFlags(QtCore.Qt.Widget)
        self._host.setDockNestingEnabled(True)
        self._host.setDockOptions(
            QtWidgets.QMainWindow.AnimatedDocks
            | QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
        )
        central = QtWidgets.QWidget()
        central.setMaximumSize(0, 0)  # docks own the full area
        self._host.setCentralWidget(central)
        root.addWidget(self._host, 1)

        self.apply_preset(DEFAULT_PRESET_ID)

    # ---------------------------------------------------------------- docks
    def add_widget(self, widget_id: str) -> QtWidgets.QDockWidget:
        """Open (or re-show) the widget with this id; unknown ids get a placeholder."""
        existing = self._docks.get(widget_id)
        if existing is not None:
            existing.show()
            existing.raise_()
            return existing

        spec = self.registry.get(widget_id)
        if spec is not None and spec.implemented and spec.factory is not None:
            content: QtWidgets.QWidget = spec.factory(spec, self.controller)
            title = spec.title
        else:
            title = spec.title if spec is not None else f"Unknown: {widget_id}"
            description = spec.description if spec is not None else ""
            content = MissingWidgetPlaceholder(widget_id, title=title, description=description)

        dock = QtWidgets.QDockWidget(title, self._host)
        dock.setObjectName(f"monitor_dock_{widget_id}")
        dock.setWidget(content)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        area = (QtCore.Qt.LeftDockWidgetArea if len(self._docks) % 2 == 0
                else QtCore.Qt.RightDockWidgetArea)
        self._host.addDockWidget(area, dock)
        self._docks[widget_id] = dock
        return dock

    def open_widget_ids(self) -> tuple[str, ...]:
        return tuple(wid for wid, dock in self._docks.items() if dock.isVisible())

    def clear_docks(self) -> None:
        for dock in self._docks.values():
            self._host.removeDockWidget(dock)
            dock.deleteLater()
        self._docks.clear()

    def apply_preset(self, preset_id: str) -> None:
        preset = preset_by_id(preset_id) or preset_by_id(DEFAULT_PRESET_ID)
        assert preset is not None
        self.active_preset_id = preset.preset_id
        index = self.preset_combo.findData(preset.preset_id)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)

        self.clear_docks()
        openable, skipped = split_preset(preset, self.registry)
        for spec in openable:
            self.add_widget(spec.widget_id)
        if skipped:
            names = ", ".join(skipped)
            self.skipped_label.setText(f"Not in this build: {names}")
            self.skipped_label.setToolTip(
                "These preset widgets are declared for a later phase and are "
                "skipped instead of being faked."
            )
            self.skipped_label.setVisible(True)
        else:
            self.skipped_label.setVisible(False)

    def _on_preset_activated(self, index: int) -> None:
        preset_id = self.preset_combo.itemData(index)
        if preset_id:
            self.apply_preset(str(preset_id))

    # ------------------------------------------------------- layout persistence
    def capture_tab_layout(self, title: str) -> TabLayout:
        """Snapshot this dashboard (open widgets + dock geometry) for saving."""
        from lunaris.ui.monitor.persistence import TabLayout

        state = self._host.saveState()
        dock_b64 = bytes(state.toBase64()).decode("ascii")
        return TabLayout(
            title=title,
            preset_id=self.active_preset_id,
            widget_ids=tuple(self._docks.keys()),
            dock_state_b64=dock_b64,
        )

    def restore_tab_layout(self, tab: TabLayout) -> None:
        """Rebuild this dashboard from a saved tab layout.

        Unknown/removed widget ids come back as graceful placeholders (their
        dock slot is preserved); a stale dock-geometry blob simply falls back
        to the default arrangement — restore never raises.
        """
        from lunaris.ui.monitor.presets import preset_by_id as _preset_by_id

        self.active_preset_id = (
            tab.preset_id if _preset_by_id(tab.preset_id) is not None
            else DEFAULT_PRESET_ID
        )
        index = self.preset_combo.findData(self.active_preset_id)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self.clear_docks()
        self.skipped_label.setVisible(False)
        for widget_id in tab.widget_ids:
            self.add_widget(widget_id)
        blob = tab.dock_state_bytes()
        if blob:
            # restoreState returns False for a stale blob; the freshly added
            # docks then keep their default arrangement.
            self._host.restoreState(QtCore.QByteArray(blob))

    def _on_open_replay(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open run telemetry artifact",
            "",
            "Telemetry artifacts (telemetry.ndjson *.ndjson);;All files (*)",
        )
        if path:
            self.controller.open_replay_file(path)

    def _rebuild_add_menu(self) -> None:
        self.add_menu.clear()
        mode = "replay" if self.controller.store.mode == "replay" else "live"
        by_category: dict[str, list] = {}
        for spec in self.registry.available_specs(mode=mode):
            by_category.setdefault(spec.category, []).append(spec)
        for category in self.registry.categories():
            specs = by_category.get(category)
            if not specs:
                continue
            submenu = self.add_menu.addMenu(category)
            for spec in specs:
                action = submenu.addAction(spec.title)
                action.setToolTip(spec.description)
                action.triggered.connect(
                    lambda _checked=False, wid=spec.widget_id: self.add_widget(wid)
                )
        if self.add_menu.isEmpty():
            action = self.add_menu.addAction("No widgets available")
            action.setEnabled(False)


class ReplayBar(QtWidgets.QFrame):
    """Playback controls bound to one shared TimelineController.

    Every widget follows the same cursor; the bar never owns its own time.
    Hidden outside replay mode.
    """

    def __init__(
        self,
        controller: MonitorController,
        timeline,  # TimelineController
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("toolbar")
        self.controller = controller
        self.timeline = timeline

        bar = QtWidgets.QHBoxLayout(self)
        bar.setContentsMargins(8, 6, 8, 6)
        bar.setSpacing(8)

        def tool_button(text: str, tooltip: str) -> QtWidgets.QToolButton:
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setToolTip(tooltip)
            bar.addWidget(button)
            return button

        self.btn_start = tool_button("|<", "Jump to start")
        self.btn_step_back = tool_button("< Step", "Step one sample backward")
        self.btn_play = tool_button("Play", "Play / pause the replay")
        self.btn_step_fwd = tool_button("Step >", "Step one sample forward")
        self.btn_end = tool_button(">|", "Jump to end")

        self.speed_control = QtWidgets.QComboBox()
        self.speed_control.setAccessibleName("Playback speed")
        for speed in timeline.SPEEDS:
            self.speed_control.addItem(f"{speed:g}×", speed)
        self.speed_control.setCurrentIndex(list(timeline.SPEEDS).index(1.0))
        self.speed_control.setToolTip(
            "Playback speed. 1× replays the full run in about one minute of "
            "wall time (not real time)."
        )
        bar.addWidget(self.speed_control)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setAccessibleName("Replay timeline cursor")
        self.slider.setRange(0, 10_000)
        self.slider.setToolTip("Timeline cursor (simulation time)")
        bar.addWidget(self.slider, 1)

        self.time_label = QtWidgets.QLabel("—")
        self.time_label.setObjectName("valueLabel")
        bar.addWidget(self.time_label)

        self.event_button = QtWidgets.QToolButton()
        self.event_button.setText("Jump to Event")
        self.event_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.event_menu = QtWidgets.QMenu(self.event_button)
        self.event_menu.aboutToShow.connect(self._rebuild_event_menu)
        self.event_button.setMenu(self.event_menu)
        bar.addWidget(self.event_button)

        self.btn_start.clicked.connect(timeline.jump_start)
        self.btn_end.clicked.connect(timeline.jump_end)
        self.btn_step_back.clicked.connect(lambda: timeline.step(-1))
        self.btn_step_fwd.clicked.connect(lambda: timeline.step(+1))
        self.btn_play.clicked.connect(timeline.toggle)
        self.speed_control.currentIndexChanged.connect(
            lambda index: timeline.set_speed(float(self.speed_control.itemData(index)))
        )
        self.slider.sliderMoved.connect(self._on_slider_moved)
        timeline.cursor_changed.connect(lambda _t: self._sync())
        timeline.playing_changed.connect(self._on_playing_changed)
        controller.updated.connect(self._sync)
        controller.mode_changed.connect(self._on_mode_changed)
        self._on_mode_changed()

    # ---------------------------------------------------------------- helpers
    def _on_mode_changed(self) -> None:
        is_replay = self.controller.store.mode == "replay"
        self.setVisible(is_replay)
        if is_replay:
            self._sync()

    def _on_playing_changed(self, playing: bool) -> None:
        self.btn_play.setText("Pause" if playing else "Play")

    def _on_slider_moved(self, value: int) -> None:
        bounds = self.timeline.bounds()
        if bounds is None:
            return
        t0, t1 = bounds
        span = max(t1 - t0, 1e-12)
        self.timeline.set_cursor(t0 + span * (value / 10_000.0))

    def _sync(self) -> None:
        if self.controller.store.mode != "replay":
            return
        bounds = self.timeline.bounds()
        cursor = self.timeline.cursor()
        if bounds is None or cursor is None:
            self.time_label.setText("—")
            return
        t0, t1 = bounds
        span = max(t1 - t0, 1e-12)
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(int(round((cursor - t0) / span * 10_000.0)))
            self.slider.blockSignals(False)
        from lunaris.ui.monitor.formatting import format_duration

        self.time_label.setText(
            f"t = {format_duration(cursor)} / {format_duration(t1)}"
        )

    def _rebuild_event_menu(self) -> None:
        from lunaris.ui.monitor.formatting import format_duration

        self.event_menu.clear()
        events = self.controller.store.events()
        if not events:
            action = self.event_menu.addAction("No events in this run")
            action.setEnabled(False)
            return
        for event in events:
            action = self.event_menu.addAction(
                f"{format_duration(event.simulation_time_s)} — {event.event_type}"
            )
            action.triggered.connect(
                lambda _checked=False, t=event.simulation_time_s:
                self.timeline.jump_to_event(t)
            )


class MonitorPage(QtWidgets.QWidget):
    """Navigation page: problem banner + tabbed dashboards + shared replay bar.

    Each tab is one independent :class:`MonitorWorkspace` (its own docks and
    preset) over the *same* controller/store, so every dashboard observes the
    same run and the same replay cursor. The tab set, per-tab widgets and dock
    geometry persist through ``lunaris_monitor_layout_v1``.
    """

    def __init__(
        self,
        controller: MonitorController,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        from lunaris.ui.monitor.replay import TimelineController

        self.controller = controller
        self.timeline = TimelineController(controller, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.problem_banner = QtWidgets.QLabel("")
        self.problem_banner.setObjectName("fieldHint")
        self.problem_banner.setProperty("kind", "warning")
        self.problem_banner.setWordWrap(True)
        self.problem_banner.setVisible(False)
        layout.addWidget(self.problem_banner)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setAccessibleName("Monitor dashboards")
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self.add_tab_button = QtWidgets.QToolButton()
        self.add_tab_button.setText("+")
        self.add_tab_button.setAccessibleName("Add dashboard tab")
        self.add_tab_button.setToolTip("Add a new dashboard tab")
        self.add_tab_button.clicked.connect(lambda: self.add_dashboard_tab())
        self.tabs.setCornerWidget(self.add_tab_button, QtCore.Qt.TopRightCorner)
        layout.addWidget(self.tabs, 1)

        self.add_dashboard_tab("Dashboard")

        self.replay_bar = ReplayBar(controller, self.timeline, parent=self)
        layout.addWidget(self.replay_bar)

        controller.protocol_problem.connect(self._on_problem)
        controller.replay_failed.connect(self._on_replay_failed)
        controller.run_started.connect(lambda: self.problem_banner.setVisible(False))

    # ------------------------------------------------------------------ tabs
    @property
    def workspace(self) -> MonitorWorkspace:
        """The active dashboard tab's workspace."""
        current = self.tabs.currentWidget()
        assert isinstance(current, MonitorWorkspace)
        return current

    def workspaces(self) -> tuple[MonitorWorkspace, ...]:
        return tuple(
            w for w in (self.tabs.widget(i) for i in range(self.tabs.count()))
            if isinstance(w, MonitorWorkspace)
        )

    def add_dashboard_tab(self, title: str | None = None) -> MonitorWorkspace:
        workspace = MonitorWorkspace(self.controller, parent=self)
        index = self.tabs.addTab(workspace, title or f"Dashboard {self.tabs.count() + 1}")
        self.tabs.setCurrentIndex(index)
        return workspace

    def _on_tab_close_requested(self, index: int) -> None:
        if self.tabs.count() <= 1:
            return  # the monitor always keeps one dashboard
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()

    # ------------------------------------------------------- layout persistence
    def capture_layout(self) -> MonitorLayout:
        """Snapshot every tab into a versioned MonitorLayout."""
        from lunaris.ui.monitor.persistence import MonitorLayout

        tabs = tuple(
            workspace.capture_tab_layout(self.tabs.tabText(i))
            for i, workspace in enumerate(self.workspaces())
        )
        return MonitorLayout(
            tabs=tabs,
            active_tab=max(0, self.tabs.currentIndex()),
            last_replay_path=self.controller.last_replay_path,
        )

    def restore_layout(self, layout: MonitorLayout) -> None:
        """Rebuild all tabs from a saved MonitorLayout (never raises)."""
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()
        for tab in layout.tabs:
            workspace = self.add_dashboard_tab(tab.title)
            workspace.restore_tab_layout(tab)
        if not self.tabs.count():
            self.add_dashboard_tab("Dashboard")
        self.tabs.setCurrentIndex(min(layout.active_tab, self.tabs.count() - 1))
        if layout.last_replay_path:
            self.controller.last_replay_path = layout.last_replay_path

    # ----------------------------------------------------------------- banner
    def _on_problem(self, detail: str) -> None:
        self.problem_banner.setText(
            f"Telemetry protocol warning: {detail} — affected lines were routed "
            "to the Execution Console instead."
        )
        self.problem_banner.setVisible(True)

    def _on_replay_failed(self, detail: str) -> None:
        self.problem_banner.setText(f"Replay could not be loaded: {detail}")
        self.problem_banner.setVisible(True)


# Re-exported for the package-level lazy import in lunaris.ui.monitor.__init__.
_ = _monitor_widgets

__all__ = ["UI_BATCH_INTERVAL_MS", "MonitorController", "MonitorPage", "MonitorWorkspace"]
