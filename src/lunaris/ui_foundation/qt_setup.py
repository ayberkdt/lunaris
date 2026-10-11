"""Process-wide Qt setup both Lunaris desktop apps run before ``QApplication``.

* **C locale for numbers.** Qt number widgets follow the system locale. On a
  Turkish or German desktop a ``QDoubleSpinBox`` showed ``10,000`` for ten,
  and typing ``0.5`` (as every line edit, log and config file in the app
  expects) gave 5.0 because ``.`` is that locale's group separator. The C
  locale makes every number widget read and write ``.`` decimals.
* **Shared OpenGL contexts.** pyqtgraph compiles each GL shader program once
  and keeps the program id on a module-level object, not per context. A
  ``GLViewWidget`` in a second, unshared context (a floated Mission Monitor
  dock, another top-level window) then binds an id that does not exist there
  and draws nothing but its text labels. Qt also requires the attribute
  before ``QApplication`` for Qt WebEngine (the launcher's Moon preview).

This module imports Qt, so it is intentionally NOT re-exported from
``lunaris.ui_foundation.__init__``.
"""

from __future__ import annotations

import logging

from PySide6 import QtCore

_log = logging.getLogger(__name__)


def prepare_qt_application() -> bool:
    """Apply the process-wide Qt setup; call it before creating ``QApplication``.

    Returns whether OpenGL contexts are shared. When an application object
    already exists the attribute can no longer be set; that is logged and
    ``False`` is returned unless it was set earlier.
    """
    QtCore.QLocale.setDefault(QtCore.QLocale.c())
    attribute = QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    if QtCore.QCoreApplication.instance() is None:
        QtCore.QCoreApplication.setAttribute(attribute, True)
        return True
    if QtCore.QCoreApplication.testAttribute(attribute):
        return True
    _log.warning(
        "QApplication already exists without AA_ShareOpenGLContexts; "
        "3D views in separate windows may render blank."
    )
    return False


__all__ = ["prepare_qt_application"]
