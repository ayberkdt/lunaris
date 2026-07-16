"""Shared Qt event filters used by both Lunaris desktop applications.

Kept out of ``lunaris.ui_foundation.__init__`` (like ``fonts``) so importing
the foundation package stays possible without a Qt installation.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


class NoWheelOnSpinFilter(QObject):
    """App-level event filter: prevents accidental spinbox value changes via scroll wheel."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.Wheel
            and isinstance(obj, QSpinBox | QDoubleSpinBox)
        ):
            event.ignore()
            return True
        return False
