"""Reusable PySide6 building blocks for Lunaris desktop pages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui.core.ui_commons import StatusBadge
from lunaris.ui.theme.tokens import DESIGN_TOKENS


class PageHeader(QtWidgets.QFrame):
    def __init__(
        self,
        title: str,
        description: str = "",
        *,
        status: Optional[QtWidgets.QWidget] = None,
        action: Optional[QtWidgets.QWidget] = None,
        parent: Optional[QtWidgets.QWidget] = None,
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
        content: Optional[QtWidgets.QWidget] = None,
        status: Optional[QtWidgets.QWidget] = None,
        action: Optional[QtWidgets.QWidget] = None,
        scrollable: bool = True,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("pageShell")
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll_area: Optional[QtWidgets.QScrollArea] = None
        if scrollable:
            self.scroll_area = QtWidgets.QScrollArea()
            self.scroll_area.setObjectName("pageScroll")
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            body = QtWidgets.QWidget()
            body.setObjectName("pageShellBody")
            self.body_layout = QtWidgets.QVBoxLayout(body)
            self.scroll_area.setWidget(body)
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
        parent: Optional[QtWidgets.QWidget] = None,
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
            title_label.setObjectName("sectionTitle")
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
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
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
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.spacing.xxs)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setObjectName("fieldLabel")
        label_widget.setBuddy(field)
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
        parent: Optional[QtWidgets.QWidget] = None,
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
        parent: Optional[QtWidgets.QWidget] = None,
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
        parent: Optional[QtWidgets.QWidget] = None,
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
        parent: Optional[QtWidgets.QWidget] = None,
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


class ActionBar(QtWidgets.QFrame):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("actionBar")
        self.action_layout = QtWidgets.QHBoxLayout(self)
        self.action_layout.setContentsMargins(0, 8, 0, 0)
        self.action_layout.setSpacing(DESIGN_TOKENS.spacing.sm)
        self.action_layout.addStretch(1)

    def add_action(self, widget: QtWidgets.QWidget) -> None:
        self.action_layout.addWidget(widget)


class Toolbar(QtWidgets.QFrame):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
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
        parent: Optional[QtWidgets.QWidget] = None,
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

    def set_current_index(self, index: int) -> None:
        button = self.group.button(index)
        if button is not None:
            button.setChecked(True)


class EmptyState(QtWidgets.QFrame):
    def __init__(
        self,
        title: str,
        description: str = "",
        *,
        action: Optional[QtWidgets.QWidget] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("emptyState")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(DESIGN_TOKENS.spacing.sm)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("emptyStateTitle")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        description_label = QtWidgets.QLabel(description)
        description_label.setObjectName("emptyStateDescription")
        description_label.setWordWrap(True)
        description_label.setAlignment(QtCore.Qt.AlignCenter)
        description_label.setVisible(bool(description))
        layout.addWidget(description_label)
        if action is not None:
            layout.addWidget(action, 0, QtCore.Qt.AlignCenter)


class CompactSearchField(QtWidgets.QLineEdit):
    def __init__(
        self,
        placeholder: str = "Search",
        parent: Optional[QtWidgets.QWidget] = None,
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
        menu: Optional[QtWidgets.QMenu] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("overflowMenuButton")
        self.setText("More")
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setMenu(menu or QtWidgets.QMenu(self))
        self.setAccessibleName("More actions")
        self.setToolTip("More actions")


__all__ = [
    "ActionBar",
    "CompactSearchField",
    "EmptyState",
    "FormGrid",
    "InlineNotice",
    "KeyValueList",
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
