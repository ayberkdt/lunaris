"""Reusable PySide6 building blocks for Lunaris desktop pages."""

from __future__ import annotations

import csv
import io
import itertools
from collections.abc import Iterable, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui.core.ui_commons import StatusBadge
from lunaris.ui.theme.tokens import DESIGN_TOKENS


def _repolish(widget: QtWidgets.QWidget) -> None:
    """Re-evaluate a widget's QSS after a dynamic property change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


class ElidedLabel(QtWidgets.QLabel):
    """A label that renders ``…`` instead of a half glyph when squeezed.

    A plain ``QLabel`` in a crowded ``QHBoxLayout`` clips mid-glyph once the
    layout drops it below its size hint, which reads as a broken app rather
    than as truncation. This paints the elided form of the text at whatever
    width it is actually given, and reports ``minimumSizeHint`` width 0 so a
    layout can shrink it without first squeezing its neighbours.

    ``full_text`` stays authoritative: it is what the tooltip and accessible
    name expose, so the elided pixels never become the only copy of the value.
    """

    def __init__(
        self,
        text: str = "",
        *,
        mode: QtCore.Qt.TextElideMode = QtCore.Qt.ElideRight,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(text, parent)
        self._full_text = text
        self._elide_mode = mode
        # Preferred, not Ignored: Ignored lets a sibling stretch claim *all*
        # the space, so the label renders at width 0 and the text disappears
        # even when the layout had room for it. Preferred asks for the full
        # text (sizeHint) and yields it only under pressure — the shrink is
        # unbounded because minimumSizeHint below reports width 0.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full_text = text
        super().setText(text)
        self.setToolTip(text)
        self.updateGeometry()
        self.update()

    def full_text(self) -> str:
        """Return the unelided text (what the label *means*, not what it shows)."""
        return self._full_text

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        # Width 0: the label yields space rather than forcing siblings to clip.
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        metrics = QtGui.QFontMetrics(self.font())
        return QtCore.QSize(
            metrics.horizontalAdvance(self._full_text),
            super().sizeHint().height(),
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QtGui.QPainter(self)
        metrics = QtGui.QFontMetrics(self.font())
        rect = self.contentsRect()
        elided = metrics.elidedText(self._full_text, self._elide_mode, rect.width())
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(rect, int(self.alignment()), elided)


def cap_input_width(
    *widgets: QtWidgets.QWidget, width: int | None = None
) -> None:
    """Cap single-value form inputs at the standard input width.

    Form panes stretch to fill their page, and an uncapped input stretches with
    them: a 400px box for a seed of ``42`` or a 590px box for ``10.00`` breaks
    the value's visual attachment to its label and makes the form read as
    unfinished. Capping turns that slack into a right margin instead.

    Apply to scalars (numbers, short strings, units). Do *not* apply to inputs
    whose content genuinely earns the width — filesystem paths, long free
    text — where a cap would just force scrolling inside the field.

    ``width`` overrides the token for the rare field that needs a different
    cap; prefer the default so the pages stay visually consistent.
    """
    capped = width if width is not None else DESIGN_TOKENS.controls.input_width_standard
    for widget in widgets:
        widget.setMaximumWidth(capped)


def scrollable(
    content: QtWidgets.QWidget, *, parent: QtWidgets.QWidget | None = None
) -> QtWidgets.QScrollArea:
    """Wrap ``content`` in a transparent, frameless vertical scroll area.

    Use this for any dialog tab or panel whose content can outgrow its box.
    Without it, a ``QVBoxLayout`` that cannot reach its minimum height does not
    scroll — Qt compresses the children past their minimum and they *overlap*,
    which reads as a rendering bug rather than as "there is more below". Labels
    with ``setWordWrap(True)`` make this worse, because their height-for-width
    minimum is unreliable, so they are usually where the collision lands.

    The scroll area is styled to disappear: no frame, transparent viewport, so
    the wrapped content keeps the surface it was designed against.

    Both scrollbars are ``AsNeeded``. Pinning the horizontal one off is
    tempting — content "should" compress to the viewport — but content with an
    incompressible minimum (a row of fixed chips, say) then gets silently
    clipped at the right edge instead, which is the same class of defect this
    helper exists to remove. A scrollbar that appears only when the content
    genuinely cannot fit is honest; a clipped control is not. Size the host so
    that it is rare, not so that it is hidden.
    """
    area = QtWidgets.QScrollArea(parent)
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.viewport().setAutoFillBackground(False)
    content.setAutoFillBackground(False)
    area.setWidget(content)
    return area


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
            body.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
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


def apply_tab_order(widgets: Sequence[QtWidgets.QWidget | None]) -> None:
    """Chain keyboard tab focus through *widgets* in the given visual order.

    Qt's default tab order follows widget *construction* order, which drifts
    from visual order whenever a page builds controls out of order or across
    several containers. Passing the widgets in the order they should be reached
    makes ``Tab`` predictable. ``None`` entries and duplicates are skipped.
    """
    chain: list[QtWidgets.QWidget] = []
    seen: set[int] = set()
    for widget in widgets:
        if widget is None or id(widget) in seen:
            continue
        seen.add(id(widget))
        chain.append(widget)
    for earlier, later in itertools.pairwise(chain):
        QtWidgets.QWidget.setTabOrder(earlier, later)


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
        # Field widgets in add (visual) order, for deterministic tab order.
        self.fields: list[QtWidgets.QWidget] = []
        self._required: set[QtWidgets.QWidget] = set()
        self._invalid: list[QtWidgets.QWidget] = []
        self._error_labels: dict[QtWidgets.QWidget, QtWidgets.QLabel] = {}
        self._base_tooltips: dict[QtWidgets.QWidget, str] = {}
        self._base_descriptions: dict[QtWidgets.QWidget, str] = {}

    def add_row(
        self,
        label: str,
        field: QtWidgets.QWidget,
        unit: str = "",
        hint: str = "",
        *,
        required: bool = False,
    ) -> None:
        label_text = f"{label} *" if (required and label) else label
        label_widget = QtWidgets.QLabel(label_text)
        label_widget.setObjectName("fieldLabel")
        label_widget.setMinimumWidth(DESIGN_TOKENS.controls.form_label_width)
        label_widget.setBuddy(field)
        # Mirror the visible label into the field's accessible name so screen
        # readers announce a meaningful name, not just the control type. Never
        # override a name the caller set explicitly.
        if label and not field.accessibleName():
            field.setAccessibleName(label.rstrip(": ").strip())
        if required:
            self._required.add(field)
            field.setProperty("required", True)
            if not field.accessibleDescription():
                field.setAccessibleDescription("required")
        self._base_tooltips[field] = field.toolTip()
        self._base_descriptions[field] = field.accessibleDescription()
        self.grid.addWidget(label_widget, self._row, 0, QtCore.Qt.AlignVCenter)
        # Keep feedback inside the field cell. Showing an error then grows only
        # this field block, preserving the form's label/field/unit columns.
        field_block = QtWidgets.QWidget()
        field_layout = QtWidgets.QVBoxLayout(field_block)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(DESIGN_TOKENS.spacing.xxs)
        field_layout.addWidget(field)
        error_label = QtWidgets.QLabel("")
        error_label.setObjectName("fieldErrorText")
        error_label.setWordWrap(True)
        error_label.setVisible(False)
        field_layout.addWidget(error_label)
        self.grid.addWidget(field_block, self._row, 1)
        self.fields.append(field)
        self._error_labels[field] = error_label
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
        # Reassert the full chain whenever a visual row is added, so every
        # FormGrid consumer gets deterministic keyboard navigation by default.
        self.apply_tab_order()

    def apply_tab_order(self) -> None:
        """Wire Tab focus through this grid's fields in visual (add) order."""
        apply_tab_order(self.fields)

    def set_error(self, field: QtWidgets.QWidget, message: str | None) -> None:
        """Mark *field* valid (``None``) or invalid (a message).

        A dense form grid keeps its label/field/unit columns aligned by placing
        the visible error message inside the field's own grid cell. The message,
        tooltip, and accessible description identify the error beyond the red
        ``fieldError`` border; ``focus_first_invalid`` jumps to the first one.
        """
        is_error = bool(message)
        field.setProperty("fieldError", True if is_error else False)
        field.setToolTip(message or self._base_tooltips.get(field, ""))
        field.setAccessibleDescription(message or self._base_descriptions.get(field, ""))
        error_label = self._error_labels.get(field)
        if error_label is not None:
            error_label.setText(message or "")
            error_label.setVisible(is_error)
        _repolish(field)
        if is_error:
            if field not in self._invalid:
                self._invalid.append(field)
        elif field in self._invalid:
            self._invalid.remove(field)

    def clear_errors(self) -> None:
        for field in list(self._invalid):
            self.set_error(field, None)

    def focus_first_invalid(self) -> bool:
        """Focus the first field currently marked invalid; return whether one existed."""
        for field in self.fields:
            if field in self._invalid:
                field.setFocus(QtCore.Qt.OtherFocusReason)
                return True
        return False


class ResponsiveColumns(QtWidgets.QWidget):
    """Keep live column widgets side-by-side only while their minima fit."""

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
        self._layout = QtWidgets.QBoxLayout(
            QtWidgets.QBoxLayout.Direction.LeftToRight,
            self,
        )
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
        self._layout.setDirection(
            QtWidgets.QBoxLayout.Direction.TopToBottom
            if stacked
            else QtWidgets.QBoxLayout.Direction.LeftToRight
        )
        self._layout.invalidate()
        self._layout.activate()
        self.updateGeometry()
        self._first.update()
        self._second.update()
        self.update()
        self.window().update()


class LabeledField(QtWidgets.QWidget):
    def __init__(
        self,
        label: str,
        field: QtWidgets.QWidget,
        *,
        hint: str = "",
        required: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.field = field
        self._required = required
        self._has_error = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.spacing.xxs)
        label_widget = QtWidgets.QLabel(f"{label} *" if (required and label) else label)
        label_widget.setObjectName("fieldLabel")
        label_widget.setBuddy(field)
        if label and not field.accessibleName():
            field.setAccessibleName(label.rstrip(": ").strip())
        if required:
            field.setProperty("required", True)
            if not field.accessibleDescription():
                field.setAccessibleDescription("required")
        self._base_tooltip = field.toolTip()
        self._base_accessible_description = field.accessibleDescription()
        layout.addWidget(label_widget)
        layout.addWidget(field)
        if hint:
            hint_widget = QtWidgets.QLabel(hint)
            hint_widget.setObjectName("fieldHint")
            hint_widget.setWordWrap(True)
            layout.addWidget(hint_widget)
        # Inline error text, below the field. Hidden until set_error() so it
        # adds no height in the valid state (this is a VBox, so revealing it
        # grows the field's own block without disturbing sibling widgets).
        self._error_label = QtWidgets.QLabel("")
        self._error_label.setObjectName("fieldErrorText")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

    def set_error(self, message: str | None) -> None:
        """Show an inline error under the field (message) or clear it (``None``)."""
        is_error = bool(message)
        self._has_error = is_error
        self.field.setProperty("fieldError", True if is_error else False)
        self.field.setToolTip(message or self._base_tooltip)
        self.field.setAccessibleDescription(message or self._base_accessible_description)
        _repolish(self.field)
        self._error_label.setText(message or "")
        self._error_label.setVisible(is_error)

    def has_error(self) -> bool:
        """Return the validation state even when this field's parent is hidden."""
        return self._has_error


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
            # Arrow keys move selection within the group (WAI-ARIA radiogroup
            # pattern); the frame filters their key events below.
            button.installEventFilter(self)
            self.group.addButton(button, index)
            layout.addWidget(button)
            self.buttons.append(button)
        if self.buttons:
            self.buttons[0].setChecked(True)
        self.group.idClicked.connect(self.current_changed)

    def eventFilter(self, obj, event):
        """Left/Up select the previous segment, Right/Down the next (wrapping).

        A segmented control is a single mutually-exclusive choice, so arrow keys
        should move the selection the way they do in a native radio group rather
        than doing nothing. Selection and keyboard focus move together and
        ``current_changed`` fires, matching a mouse click.
        """
        if event.type() == QtCore.QEvent.KeyPress and obj in self.buttons:
            key = event.key()
            if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Up):
                self._step_selection(-1)
                return True
            if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Down):
                self._step_selection(1)
                return True
        return super().eventFilter(obj, event)

    def _step_selection(self, delta: int) -> None:
        if not self.buttons:
            return
        current = max(0, self.current_index())
        target = (current + delta) % len(self.buttons)
        if target == current:
            return
        self.buttons[target].setChecked(True)
        self.buttons[target].setFocus(QtCore.Qt.TabFocusReason)
        self.current_changed.emit(target)

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
        # No layout-level AlignCenter: centered children are laid out at their
        # size-hint width, so a word-wrapped description gets a height computed
        # for a wider line and clips mid-line (seen on the Run Diagnostics
        # card). Children span the full width — the text centers itself — and
        # the stretches keep the block vertically centered in tall frames.
        layout.addStretch(1)
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
        layout.addStretch(1)

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
        lines = ["\t".join(cols[c] for c in sorted(cols)) for _, cols in sorted(rows.items())]
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.Copy):
            self.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = [
    "ActionBar",
    "apply_tab_order",
    "CompactSearchField",
    "DataTable",
    "ElidedLabel",
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
    "cap_input_width",
    "scrollable",
]
