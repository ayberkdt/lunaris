"""Reusable PySide6 building blocks for Lunaris desktop pages."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui_foundation import DESIGN_TOKENS

_ACCESSIBLE_INTERACTIVE = (
    QtWidgets.QAbstractButton,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QLineEdit,
    QtWidgets.QPlainTextEdit,
    QtWidgets.QTextEdit,
    QtWidgets.QAbstractSlider,
    QtWidgets.QAbstractItemView,
)


def _is_composite_child(widget: QtWidgets.QWidget) -> bool:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(
            parent,
            QtWidgets.QComboBox
            | QtWidgets.QAbstractSpinBox
            | QtWidgets.QAbstractItemView
            | QtWidgets.QCalendarWidget,
        ):
            return True
        if parent.metaObject().className() == "QComboBoxPrivateContainer":
            return True
        parent = parent.parentWidget()
    return False


def _focusable_controls(widget: QtWidgets.QWidget) -> list[QtWidgets.QWidget]:
    controls: list[QtWidgets.QWidget] = []
    candidates = [widget, *widget.findChildren(QtWidgets.QWidget)]
    for candidate in candidates:
        if not isinstance(candidate, _ACCESSIBLE_INTERACTIVE):
            continue
        if isinstance(candidate, QtWidgets.QScrollBar):
            continue
        if candidate.focusPolicy() == QtCore.Qt.NoFocus or _is_composite_child(candidate):
            continue
        if isinstance(candidate, QtWidgets.QAbstractButton) and candidate.text().strip():
            continue
        controls.append(candidate)
    return controls


def _clean_label(text: str) -> str:
    return text.replace("&", "").rstrip(": *").strip()


def _nearest_accessible_context(widget: QtWidgets.QWidget) -> str:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QtWidgets.QGroupBox) and parent.title().strip():
            return _clean_label(parent.title())
        for label in parent.findChildren(QtWidgets.QLabel, options=QtCore.Qt.FindDirectChildrenOnly):
            if label.objectName() in {
                "pageTitle",
                "panelTitle",
                "sectionTitle",
            } and label.text().strip():
                return _clean_label(label.text())
        parent = parent.parentWidget()
    return "Workspace"


def assign_accessible_names(root: QtWidgets.QWidget) -> None:
    """Bind form labels and panel context to otherwise unnamed controls.

    QFormLayout labels point at wrapper widgets when a field is composed from a
    path edit plus a Browse button. Screen readers therefore miss the actual
    edit. This pass assigns the visible row label to the wrapper's focusable
    field while leaving explicitly authored names untouched.
    """

    for form in root.findChildren(QtWidgets.QFormLayout):
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
            field_item = form.itemAt(row, QtWidgets.QFormLayout.FieldRole)
            label = label_item.widget() if label_item is not None else None
            field = field_item.widget() if field_item is not None else None
            if not isinstance(label, QtWidgets.QLabel) or field is None:
                continue
            name = _clean_label(label.text())
            if not name:
                continue
            for control in _focusable_controls(field):
                if not control.accessibleName().strip():
                    control.setAccessibleName(name)

    for grid in root.findChildren(QtWidgets.QGridLayout):
        for row in range(grid.rowCount()):
            label_item = grid.itemAtPosition(row, 0)
            field_item = grid.itemAtPosition(row, 1)
            label = label_item.widget() if label_item is not None else None
            field = field_item.widget() if field_item is not None else None
            if not isinstance(label, QtWidgets.QLabel) or field is None:
                continue
            name = _clean_label(label.text())
            if not name:
                continue
            for control in _focusable_controls(field):
                if not control.accessibleName().strip():
                    control.setAccessibleName(name)

    for control in _focusable_controls(root):
        if control.accessibleName().strip():
            continue
        context = _nearest_accessible_context(control)
        if isinstance(control, QtWidgets.QTableView):
            role = "table"
        elif isinstance(control, QtWidgets.QAbstractItemView):
            role = "list"
        elif isinstance(control, QtWidgets.QPlainTextEdit | QtWidgets.QTextEdit):
            role = "output"
        elif isinstance(control, QtWidgets.QComboBox):
            role = "selector"
        elif isinstance(control, QtWidgets.QLineEdit):
            role = "input"
        else:
            role = "control"
        control.setAccessibleName(f"{context} {role}")


class StatusBadge(QtWidgets.QLabel):
    def __init__(self, text: str = "WAITING", kind: str = "info", parent=None):
        super().__init__(text, parent)
        self.setObjectName("statusBadge")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setContentsMargins(10, 4, 10, 4)
        self.setFixedHeight(24)
        self.set_status(kind, text)

    def set_status(self, kind: str, text: str):
        self.setProperty("kind", kind.lower())
        self.setText(text.upper())
        self.style().unpolish(self)
        self.style().polish(self)


class PageHeader(QtWidgets.QFrame):
    def __init__(
        self,
        title: str,
        description: str = "",
        *,
        status: QtWidgets.QWidget | None = None,
        action: QtWidgets.QWidget | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("pageHeader")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.spacing.md)

        text_box = QtWidgets.QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(DESIGN_TOKENS.spacing.xxs)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("pageTitle")
        self.title_label.setAccessibleName(f"{title} page")
        text_box.addWidget(self.title_label)
        self.description_label = QtWidgets.QLabel(description)
        self.description_label.setObjectName("pageDescription")
        self.description_label.setWordWrap(True)
        self.description_label.setVisible(bool(description))
        text_box.addWidget(self.description_label)
        layout.addLayout(text_box, 1)

        if status is not None:
            layout.addWidget(status, 0, QtCore.Qt.AlignTop)
        if action is not None:
            layout.addWidget(action, 0, QtCore.Qt.AlignTop)


class PageShell(QtWidgets.QWidget):
    """Standard page hierarchy with a header and one scroll owner."""

    def __init__(
        self,
        title: str,
        description: str = "",
        *,
        content: QtWidgets.QWidget | None = None,
        status: QtWidgets.QWidget | None = None,
        action: QtWidgets.QWidget | None = None,
        scrollable: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("pageShell")
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll_area: QtWidgets.QScrollArea | None = None
        if scrollable:
            self.scroll_area = QtWidgets.QScrollArea()
            self.scroll_area.setObjectName("pageScroll")
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            body = QtWidgets.QWidget()
            body.setObjectName("pageShellBody")
            self.body_layout = QtWidgets.QVBoxLayout(body)
            # Readable page width: cap the column and centre it when the workspace
            # is wider, so dense forms stay scannable on ultrawide displays.
            body.setMaximumWidth(DESIGN_TOKENS.layout.page_max_width)
            body.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
            )
            centerer = QtWidgets.QWidget()
            centerer.setObjectName("pageShellCenterer")
            center_layout = QtWidgets.QHBoxLayout(centerer)
            center_layout.setContentsMargins(0, 0, 0, 0)
            center_layout.setSpacing(0)
            # The body should *expand* to fill the workspace up to its readable
            # max-width, then centre — not sit at its narrow natural width with
            # large dead bands on both sides. Giving the body a high stretch
            # weight against the two spacers makes it grow until it hits the
            # max-width cap, after which the spacers share the remainder.
            center_layout.addStretch(1)
            center_layout.addWidget(body, 50)
            center_layout.addStretch(1)
            self.scroll_area.setWidget(centerer)
            root.addWidget(self.scroll_area)
        else:
            body = QtWidgets.QWidget()
            body.setObjectName("pageShellBody")
            self.body_layout = QtWidgets.QVBoxLayout(body)
            root.addWidget(body)

        margin = DESIGN_TOKENS.layout.shell_margin
        self.body_layout.setContentsMargins(margin, margin, margin, margin)
        self.body_layout.setSpacing(DESIGN_TOKENS.layout.page_gap)
        self.header = PageHeader(
            title,
            description,
            status=status,
            action=action,
        )
        self.body_layout.addWidget(self.header)

        self.content_host = QtWidgets.QWidget()
        self.content_host.setObjectName("pageContentHost")
        self.content_layout = QtWidgets.QVBoxLayout(self.content_host)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(DESIGN_TOKENS.layout.page_gap)
        self.body_layout.addWidget(self.content_host, 1)
        if content is not None:
            self.add_widget(content)

    def add_widget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        self.content_layout.addWidget(widget, stretch)


class Section(QtWidgets.QFrame):
    def __init__(
        self,
        title: str = "",
        description: str = "",
        *,
        elevated: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("section")
        self.setProperty("elevated", bool(elevated))
        root = QtWidgets.QVBoxLayout(self)
        pad = DESIGN_TOKENS.layout.section_padding
        root.setContentsMargins(pad, pad, pad, pad)
        root.setSpacing(DESIGN_TOKENS.spacing.md)
        if title:
            title_label = QtWidgets.QLabel(title)
            # A card/panel heading sits one tier above the inner-group labels
            # (which use ``#sectionTitle``); ``#panelTitle`` renders at the
            # 14 pt subsection size so the hierarchy page -> panel -> group is
            # perceptible by size, not just weight.
            title_label.setObjectName("panelTitle")
            root.addWidget(title_label)
        if description:
            description_label = QtWidgets.QLabel(description)
            description_label.setObjectName("sectionDescription")
            description_label.setWordWrap(True)
            root.addWidget(description_label)
        self.content_layout = QtWidgets.QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(DESIGN_TOKENS.spacing.md)
        root.addLayout(self.content_layout)

    def add_widget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        self.content_layout.addWidget(widget, stretch)


class Subsection(Section):
    def __init__(self, title: str = "", description: str = "", **kwargs):
        super().__init__(title, description, **kwargs)
        self.setObjectName("subsection")


class FormGrid(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("formGrid")
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(DESIGN_TOKENS.spacing.md)
        self.grid.setVerticalSpacing(DESIGN_TOKENS.spacing.sm)
        self.grid.setColumnStretch(1, 1)
        self._row = 0

    def add_row(
        self,
        label: str,
        field: QtWidgets.QWidget,
        unit: str = "",
        hint: str = "",
    ) -> None:
        label_widget = QtWidgets.QLabel(label)
        label_widget.setObjectName("fieldLabel")
        label_widget.setMinimumWidth(DESIGN_TOKENS.controls.form_label_width)
        label_widget.setBuddy(field)
        # Mirror the visible label into the field's accessible name so screen
        # readers announce a meaningful name, not just the control type. Never
        # override a name the caller set explicitly.
        if label and not field.accessibleName():
            field.setAccessibleName(label.rstrip(": ").strip())
        self.grid.addWidget(label_widget, self._row, 0, QtCore.Qt.AlignVCenter)
        self.grid.addWidget(field, self._row, 1)
        unit_widget = QtWidgets.QLabel(unit)
        unit_widget.setObjectName("fieldUnit")
        unit_widget.setVisible(bool(unit))
        self.grid.addWidget(unit_widget, self._row, 2, QtCore.Qt.AlignVCenter)
        if hint:
            hint_widget = QtWidgets.QLabel(hint)
            hint_widget.setObjectName("fieldHint")
            hint_widget.setWordWrap(True)
            self.grid.addWidget(hint_widget, self._row + 1, 1, 1, 2)
            self._row += 1
        self._row += 1


class LabeledField(QtWidgets.QWidget):
    def __init__(
        self,
        label: str,
        field: QtWidgets.QWidget,
        *,
        hint: str = "",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.spacing.xxs)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setObjectName("fieldLabel")
        label_widget.setBuddy(field)
        if label and not field.accessibleName():
            field.setAccessibleName(label.rstrip(": ").strip())
        layout.addWidget(label_widget)
        layout.addWidget(field)
        if hint:
            hint_widget = QtWidgets.QLabel(hint)
            hint_widget.setObjectName("fieldHint")
            hint_widget.setWordWrap(True)
            layout.addWidget(hint_widget)


class UnitField(QtWidgets.QWidget):
    def __init__(
        self,
        field: QtWidgets.QWidget,
        unit: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.spacing.sm)
        layout.addWidget(field, 1)
        unit_label = QtWidgets.QLabel(unit)
        unit_label.setObjectName("fieldUnit")
        layout.addWidget(unit_label)


class KeyValueList(QtWidgets.QWidget):
    def __init__(
        self,
        items: Iterable[tuple[str, str]] = (),
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("keyValueList")
        self.layout_grid = QtWidgets.QGridLayout(self)
        self.layout_grid.setContentsMargins(0, 0, 0, 0)
        self.layout_grid.setHorizontalSpacing(DESIGN_TOKENS.spacing.lg)
        self.layout_grid.setVerticalSpacing(DESIGN_TOKENS.spacing.sm)
        self.layout_grid.setColumnStretch(1, 1)
        for key, value in items:
            self.add_item(key, value)

    def add_item(self, key: str, value: str) -> QtWidgets.QLabel:
        row = self.layout_grid.rowCount()
        key_label = QtWidgets.QLabel(key)
        key_label.setObjectName("keyLabel")
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("valueLabel")
        value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.layout_grid.addWidget(key_label, row, 0)
        self.layout_grid.addWidget(value_label, row, 1, QtCore.Qt.AlignRight)
        return value_label


class MetricRow(QtWidgets.QFrame):
    def __init__(
        self,
        metrics: Iterable[tuple[str, str]] = (),
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("metricRow")
        self.row_layout = QtWidgets.QHBoxLayout(self)
        self.row_layout.setContentsMargins(0, 0, 0, 0)
        self.row_layout.setSpacing(DESIGN_TOKENS.spacing.sm)
        for label, value in metrics:
            self.add_metric(label, value)

    def add_metric(self, label: str, value: str) -> QtWidgets.QLabel:
        cell = QtWidgets.QFrame()
        cell.setObjectName("metricCell")
        cell_layout = QtWidgets.QVBoxLayout(cell)
        cell_layout.setContentsMargins(12, 8, 12, 8)
        cell_layout.setSpacing(2)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setObjectName("metricLabel")
        value_widget = QtWidgets.QLabel(value)
        value_widget.setObjectName("metricValue")
        cell_layout.addWidget(label_widget)
        cell_layout.addWidget(value_widget)
        self.row_layout.addWidget(cell, 1)
        return value_widget


class InlineNotice(QtWidgets.QFrame):
    def __init__(
        self,
        text: str,
        kind: str = "info",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("inlineNotice")
        self.setProperty("kind", kind)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.label = QtWidgets.QLabel(text)
        self.label.setObjectName("inlineNoticeText")
        self.label.setWordWrap(True)
        self.setAccessibleName(f"{kind} notice")
        layout.addWidget(self.label)

    def set_notice(self, text: str, kind: str) -> None:
        """Update notice content without replacing the live widget."""
        self.label.setText(text)
        self.setProperty("kind", kind)
        self.setAccessibleName(f"{kind} notice")
        self.style().unpolish(self)
        self.style().polish(self)


class ResponsiveColumns(QtWidgets.QWidget):
    """Lay out two live widgets side-by-side when their minima genuinely fit.

    The child widgets are never reconstructed, so their signals, text, and
    process state survive direction changes.  The breakpoint is measured from
    the children's minimum size hints rather than from a magic window width.
    """

    def __init__(
        self,
        first: QtWidgets.QWidget,
        second: QtWidgets.QWidget,
        *,
        spacing: int | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._first = first
        self._second = second
        self._spacing = spacing if spacing is not None else DESIGN_TOKENS.spacing.md
        self._layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(self._spacing)
        self._layout.addWidget(first, 1)
        self._layout.addWidget(second, 1)
        self._stacked: bool | None = None
        QtCore.QTimer.singleShot(0, self._refresh_direction)

    @property
    def stacked(self) -> bool:
        return bool(self._stacked)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_direction()

    def event(self, event: QtCore.QEvent) -> bool:  # noqa: A003
        handled = super().event(event)
        if event.type() in (
            QtCore.QEvent.Type.LayoutRequest,
            QtCore.QEvent.Type.PolishRequest,
            QtCore.QEvent.Type.Show,
        ):
            QtCore.QTimer.singleShot(0, self._refresh_direction)
        return handled

    def _refresh_direction(self) -> None:
        first_min = max(
            self._first.minimumWidth(), self._first.minimumSizeHint().width()
        )
        second_min = max(
            self._second.minimumWidth(), self._second.minimumSizeHint().width()
        )
        required = first_min + second_min + self._spacing
        stacked = self.width() < required
        if stacked == self._stacked:
            return
        self._stacked = stacked
        direction = (
            QtWidgets.QBoxLayout.TopToBottom
            if stacked
            else QtWidgets.QBoxLayout.LeftToRight
        )
        self._layout.setDirection(direction)
        self._layout.invalidate()
        self._layout.activate()
        self.updateGeometry()
        self._first.update()
        self._second.update()
        self.update()
        self.window().update()


class ActionBar(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("actionBar")
        self.action_layout = QtWidgets.QHBoxLayout(self)
        self.action_layout.setContentsMargins(0, 8, 0, 0)
        self.action_layout.setSpacing(DESIGN_TOKENS.spacing.sm)
        self.action_layout.addStretch(1)

    def add_action(self, widget: QtWidgets.QWidget) -> None:
        self.action_layout.addWidget(widget)


class Toolbar(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("toolbar")
        self.toolbar_layout = QtWidgets.QHBoxLayout(self)
        self.toolbar_layout.setContentsMargins(8, 6, 8, 6)
        self.toolbar_layout.setSpacing(DESIGN_TOKENS.spacing.sm)

    def add_widget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        self.toolbar_layout.addWidget(widget, stretch)


class SegmentedControl(QtWidgets.QFrame):
    current_changed = QtCore.Signal(int)

    def __init__(
        self,
        labels: Iterable[str],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("segmentedControl")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QtWidgets.QPushButton] = []
        for index, label in enumerate(labels):
            button = QtWidgets.QPushButton(label)
            button.setObjectName("segmentButton")
            button.setCheckable(True)
            self.group.addButton(button, index)
            layout.addWidget(button)
            self.buttons.append(button)
        if self.buttons:
            self.buttons[0].setChecked(True)
        self.group.idClicked.connect(self.current_changed)

    def current_index(self) -> int:
        return self.group.checkedId()

    def set_current_index(self, index: int, *, emit: bool = False) -> None:
        previous = self.current_index()
        button = self.group.button(index)
        if button is not None:
            button.setChecked(True)
            if emit and index != previous:
                self.current_changed.emit(index)


class EmptyState(QtWidgets.QFrame):
    def __init__(
        self,
        title: str,
        description: str = "",
        *,
        action: QtWidgets.QWidget | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("emptyState")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(DESIGN_TOKENS.spacing.sm)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.material_mark = QtWidgets.QFrame()
        self.material_mark.setObjectName("emptyStateMark")
        self.material_mark.setFixedSize(30, 30)
        self.material_mark.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        mark_layout = QtWidgets.QHBoxLayout(self.material_mark)
        mark_layout.setContentsMargins(0, 0, 0, 0)
        self.material_node = QtWidgets.QFrame()
        self.material_node.setObjectName("emptyStateNode")
        self.material_node.setFixedSize(6, 6)
        mark_layout.addWidget(
            self.material_node, 0, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        layout.addWidget(
            self.material_mark, 0, QtCore.Qt.AlignmentFlag.AlignHCenter
        )
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("emptyStateTitle")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.title_label)
        self.description_label = QtWidgets.QLabel(description)
        self.description_label.setObjectName("emptyStateDescription")
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(QtCore.Qt.AlignCenter)
        self.description_label.setVisible(bool(description))
        layout.addWidget(self.description_label)
        if action is not None:
            layout.addWidget(action, 0, QtCore.Qt.AlignCenter)

    def set_message(self, title: str, description: str = "") -> None:
        """Update the empty-state text in place (so one widget can cover several cases)."""
        self.title_label.setText(title)
        self.description_label.setText(description)
        self.description_label.setVisible(bool(description))


class CompactSearchField(QtWidgets.QLineEdit):
    def __init__(
        self,
        placeholder: str = "Search",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("compactSearch")
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self.setAccessibleName(placeholder)
        self.setMinimumWidth(140)
        self.setMaximumWidth(320)


class OverflowMenuButton(QtWidgets.QToolButton):
    def __init__(
        self,
        menu: QtWidgets.QMenu | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("overflowMenuButton")
        self.setText("More")
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setMenu(menu or QtWidgets.QMenu(self))
        self.setAccessibleName("More actions")
        self.setToolTip("More actions")


class DataTable(QtWidgets.QTableWidget):
    """Mission-analysis data table: sortable, copyable, CSV-exportable.

    Adds the behaviours every results/ephemeris/event table should share:
    sorting, row selection, read-only cells, right-aligned monospace numeric
    columns, unit-bearing headers, ``Ctrl+C`` copy (TSV) of the selection, and
    a ``to_csv()`` export. Surface styling comes from the global ``#dataTable``
    QSS so no inline colors are needed.
    """

    def __init__(
        self,
        headers: Sequence[str | tuple[str, str]] = (),
        *,
        numeric_columns: Iterable[int] = (),
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("dataTable")
        self._numeric_columns = set(numeric_columns)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setHighlightSections(False)
        self.setWordWrap(False)
        if headers:
            self.set_headers(headers)

    def set_headers(self, headers: Sequence[str | tuple[str, str]]) -> None:
        """Set column headers; a ``(label, unit)`` tuple renders as ``label [unit]``."""
        labels: list[str] = []
        for header in headers:
            if isinstance(header, tuple):
                label, unit = header
                labels.append(f"{label} [{unit}]" if unit else label)
            else:
                labels.append(header)
        self.setColumnCount(len(labels))
        self.setHorizontalHeaderLabels(labels)

    def append_row(self, values: Sequence[object]) -> int:
        """Append a row; numeric columns are right-aligned and monospaced."""
        # Insert with sorting disabled so the row index stays stable while filling.
        was_sorting = self.isSortingEnabled()
        self.setSortingEnabled(False)
        row = self.rowCount()
        self.insertRow(row)
        for col, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(value))
            if col in self._numeric_columns:
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                font = item.font()
                # setFamilies()+Monospace hint (setFamily with the comma-joined
                # token string silently falls back to the UI font).
                font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
                font.setFamilies(
                    [f.strip().strip('"') for f in DESIGN_TOKENS.typography.family_mono.split(",")]
                )
                item.setFont(font)
            self.setItem(row, col, item)
        self.setSortingEnabled(was_sorting)
        return row

    def _cell_text(self, row: int, col: int) -> str:
        item = self.item(row, col)
        return item.text() if item is not None else ""

    def _header_text(self, col: int) -> str:
        item = self.horizontalHeaderItem(col)
        return item.text() if item is not None else ""

    def to_csv(self, *, include_header: bool = True) -> str:
        """Return the full table as CSV text."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        cols = range(self.columnCount())
        if include_header:
            writer.writerow([self._header_text(c) for c in cols])
        for row in range(self.rowCount()):
            writer.writerow([self._cell_text(row, c) for c in cols])
        return buffer.getvalue()

    def copy_selection(self) -> None:
        """Copy the selected cells to the clipboard as tab-separated rows."""
        items = self.selectedItems()
        if not items:
            return
        rows: dict[int, dict[int, str]] = {}
        for item in items:
            rows.setdefault(item.row(), {})[item.column()] = item.text()
        lines = [
            "\t".join(cols[c] for c in sorted(cols))
            for _, cols in sorted(rows.items())
        ]
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.Copy):
            self.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = [
    "ActionBar",
    "assign_accessible_names",
    "CompactSearchField",
    "DataTable",
    "EmptyState",
    "FormGrid",
    "InlineNotice",
    "KeyValueList",
    "ResponsiveColumns",
    "LabeledField",
    "MetricRow",
    "OverflowMenuButton",
    "PageHeader",
    "PageShell",
    "Section",
    "SegmentedControl",
    "StatusBadge",
    "Subsection",
    "Toolbar",
    "UnitField",
]
