"""Small workspace-level widgets for the ST-LRPS Studio.

These widgets stay inside the ST-LRPS UI package so the Studio can share the
Lunar Graphite token system without importing mission-UI internals.
"""

from __future__ import annotations

from collections.abc import Sequence

from lunaris.ui_foundation import DESIGN_TOKENS

from .qt_common import (
    THEME,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    QWidget,
    repolish,
)

_KIND_COLOR = {
    "info": THEME["info"],
    "success": THEME["success"],
    "warning": THEME["warning"],
    "error": THEME["error"],
    "blocked": THEME["error"],
    "active": THEME["accent"],
    "done": THEME["success"],
    "pending": THEME["fg_muted"],
}


def _kind_color(kind: str) -> str:
    return _KIND_COLOR.get(kind, THEME["fg_muted"])


class StudioStatusBadge(QLabel):
    """Compact semantic status label with text, not color alone."""

    def __init__(self, text: str = "Idle", kind: str = "info", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("studioStatusBadge")
        self.setMinimumHeight(DESIGN_TOKENS.controls.status_badge_height)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.set_status(kind, text)

    def set_status(self, kind: str, text: str) -> None:
        # Colors live in the shared #studioStatusBadge[kind=...] rules.
        self.setText(f"{kind.upper()}: {text}")
        self.setProperty("kind", kind)
        repolish(self)


class StudioNotice(QFrame):
    """Token-based inline notice for scientific workflow guidance."""

    def __init__(
        self,
        title: str,
        body: str,
        *,
        kind: str = "info",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("studioNotice")
        self._title = QLabel()
        self._title.setObjectName("studioNoticeTitle")
        self._body = QLabel()
        self._body.setObjectName("studioNoticeBody")
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(3)
        layout.addWidget(self._title)
        layout.addWidget(self._body)
        self.set_notice(title, body, kind=kind)

    def set_notice(self, title: str, body: str, *, kind: str = "info") -> None:
        # Frame border and title color follow the shared [kind=...] rules.
        self._title.setText(f"{kind.upper()} - {title}")
        self._body.setText(body)
        for widget in (self, self._title):
            widget.setProperty("kind", kind)
            repolish(widget)


class StudioWorkflowOverview(QFrame):
    """Horizontal task strip that makes the active workflow visible."""

    def __init__(
        self,
        steps: Sequence[tuple[str, str]],
        *,
        current_index: int = 0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("studioWorkflowOverview")
        self._cells: list[tuple[QLabel, QLabel, QLabel]] = []

        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setHorizontalSpacing(10)
        self._layout.setVerticalSpacing(8)

        for index, (title, detail) in enumerate(steps):
            cell = QFrame()
            cell.setObjectName("studioWorkflowCell")
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(10, 8, 10, 8)
            cell_layout.setSpacing(8)

            number = QLabel(str(index + 1))
            number.setObjectName("workflowStepNumber")
            number.setAlignment(Qt.AlignmentFlag.AlignCenter)
            number.setMinimumWidth(24)
            number.setMinimumHeight(24)

            text_box = QVBoxLayout()
            text_box.setContentsMargins(0, 0, 0, 0)
            text_box.setSpacing(1)
            title_label = QLabel(title)
            title_label.setObjectName("workflowStepTitle")
            detail_label = QLabel(detail)
            detail_label.setObjectName("workflowStepDetail")
            detail_label.setWordWrap(True)
            text_box.addWidget(title_label)
            text_box.addWidget(detail_label)

            cell_layout.addWidget(number)
            cell_layout.addLayout(text_box, 1)
            self._layout.addWidget(cell, 0, index)
            self._layout.setColumnStretch(index, 1)
            self._cells.append((number, title_label, detail_label))

        self.set_current(current_index)

    def set_current(self, current_index: int) -> None:
        for index, (number, title, _detail) in enumerate(self._cells):
            if index < current_index:
                kind = "done"
                state = "DONE"
            elif index == current_index:
                kind = "active"
                state = "ACTIVE"
            else:
                kind = "pending"
                state = "NEXT"
            number.setText(str(index + 1))
            number.setToolTip(state)
            number.setProperty("kind", kind)
            title.setProperty("reached", "true" if index <= current_index else "false")
            for widget in (number, title):
                repolish(widget)


class StudioReadinessPanel(QFrame):
    """Launch gate summary with explicit blocking/warning/info rows."""

    def __init__(self, title: str = "Launch Readiness", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("studioReadinessPanel")

        self._title = QLabel(title)
        self._title.setObjectName("readinessTitle")
        self._summary = StudioStatusBadge("Waiting for inputs", "info")
        self._body = QLabel("")
        self._body.setObjectName("readinessBody")
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._summary)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(7)
        root.addLayout(top)
        root.addWidget(self._body)

        self.set_items([])

    def set_items(self, items: Sequence[tuple[str, str]]) -> None:
        blocking = sum(1 for kind, _text in items if kind in {"error", "blocked"})
        warnings = sum(1 for kind, _text in items if kind == "warning")
        if blocking:
            self._summary.set_status("blocked", f"{blocking} blocking")
        elif warnings:
            self._summary.set_status("warning", f"{warnings} warning")
        elif items:
            self._summary.set_status("success", "Ready")
        else:
            self._summary.set_status("info", "No checks yet")

        lines = []
        for kind, text in items:
            label = {
                "success": "OK",
                "warning": "WARN",
                "error": "BLOCK",
                "blocked": "BLOCK",
                "info": "INFO",
            }.get(kind, kind.upper())
            color = _kind_color(kind)
            lines.append(f'<span style="color:{color}; font-weight:700">{label}</span> {text}')
        self._body.setText("<br>".join(lines) if lines else "Checks appear after fields are available.")
        self.setVisible(bool(items))
