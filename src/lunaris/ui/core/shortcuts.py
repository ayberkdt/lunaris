"""Single source of truth for application keyboard shortcuts (W2.3).

Every user-facing key binding in the desktop app is declared here once and
consumed by the menu bar / global-shortcut builders in :mod:`lunaris.ui.app`.
Keeping the inventory Qt-free (plain strings, parsed by ``QKeySequence`` at the
call site) lets the uniqueness and conflict tests run headless and keeps this
module importable from anywhere without a QApplication.

Scopes
------
``menu``
    Bound to a visible ``QAction`` in the menu bar; Qt renders the key label
    next to the menu entry, which is what makes the binding discoverable.
``window``
    A window-level ``QShortcut`` with no menu entry (fast navigation chords).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ShortcutSpec", "SHORTCUTS", "spec", "keys", "primary_key"]


@dataclass(frozen=True)
class ShortcutSpec:
    """One logical action and the key sequence(s) bound to it."""

    action_id: str
    #: ``QKeySequence``-parseable strings; the first entry is the primary
    #: binding shown in menus.
    key_sequences: tuple[str, ...]
    label: str
    scope: str  # "menu" | "window"


#: The full application shortcut inventory. ``nav.page`` is a family: its
#: entries map positionally onto the first nine workspace pages in nav order.
SHORTCUTS: tuple[ShortcutSpec, ...] = (
    ShortcutSpec("file.load_profile", ("Ctrl+O",), "Load Mission Profile...", "menu"),
    ShortcutSpec("file.save_profile", ("Ctrl+S",), "Save Mission Profile", "menu"),
    ShortcutSpec("file.open_results", ("Ctrl+Shift+O",), "Open Results Folder", "menu"),
    ShortcutSpec("file.exit", ("Alt+F4",), "Exit", "menu"),
    ShortcutSpec("analysis.run", ("F5",), "Start Propagation", "menu"),
    ShortcutSpec("analysis.stop", ("Shift+F5",), "Abort Propagation", "menu"),
    ShortcutSpec("view.toggle_log", ("Ctrl+`", "Ctrl+L"), "Toggle Log Panel", "menu"),
    ShortcutSpec("view.clear_log", ("Ctrl+K",), "Clear Log", "menu"),
    ShortcutSpec("view.compact_density", ("Ctrl+Shift+D",), "Compact Density", "menu"),
    ShortcutSpec(
        "nav.page",
        tuple(f"Ctrl+{i}" for i in range(1, 10)),
        "Go to workspace page 1-9",
        "window",
    ),
    ShortcutSpec(
        "console.focus_search", ("Ctrl+Shift+F",), "Focus console search", "window"
    ),
)

_BY_ID = {item.action_id: item for item in SHORTCUTS}


def spec(action_id: str) -> ShortcutSpec:
    """Return the :class:`ShortcutSpec` for ``action_id`` (KeyError if unknown)."""
    return _BY_ID[action_id]


def keys(action_id: str) -> tuple[str, ...]:
    """All key sequences bound to ``action_id``, primary first."""
    return spec(action_id).key_sequences


def primary_key(action_id: str) -> str:
    """The primary (menu-visible) key sequence for ``action_id``."""
    return spec(action_id).key_sequences[0]
