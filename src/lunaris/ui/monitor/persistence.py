"""Versioned Mission Monitor layout persistence (``lunaris_monitor_layout_v1``).

The layout file is separate from the Studio session snapshot: it stores which
dashboard tabs exist, which widgets each tab has open, the dock geometry blob
(Qt ``QMainWindow.saveState`` bytes, base64), the active preset per tab, and
the last replay artifact. Failure policy is explicit:

* a corrupt or foreign-schema file is renamed to ``<name>.bak`` and ignored —
  the monitor opens with the default preset and the app never fails to start;
* writes are atomic (temp file + replace), so a crash cannot half-write the
  layout.

This module is JSON/dataclass only (Qt-free); the workspace produces and
consumes the payloads.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MONITOR_LAYOUT_SCHEMA_VERSION = "lunaris_monitor_layout_v1"
MONITOR_LAYOUT_FILENAME = "monitor_layout.json"


class LayoutError(ValueError):
    """The layout payload is unusable (corrupt, foreign schema, wrong types)."""


@dataclass(frozen=True, slots=True)
class TabLayout:
    """One dashboard tab: identity, open widgets and dock geometry."""

    title: str
    preset_id: str
    widget_ids: tuple[str, ...] = ()
    #: Base64 of QMainWindow.saveState(); empty means "default dock layout".
    dock_state_b64: str = ""

    def dock_state_bytes(self) -> bytes:
        if not self.dock_state_b64:
            return b""
        try:
            return base64.b64decode(self.dock_state_b64, validate=True)
        except (binascii.Error, ValueError):
            return b""


@dataclass(frozen=True, slots=True)
class MonitorLayout:
    tabs: tuple[TabLayout, ...]
    active_tab: int = 0
    last_replay_path: str | None = None
    schema_version: str = MONITOR_LAYOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.tabs:
            raise LayoutError("MonitorLayout requires at least one tab")
        if not 0 <= self.active_tab < len(self.tabs):
            object.__setattr__(self, "active_tab", 0)


def layout_to_payload(layout: MonitorLayout) -> dict[str, Any]:
    return {
        "schema_version": layout.schema_version,
        "active_tab": int(layout.active_tab),
        "last_replay_path": layout.last_replay_path,
        "tabs": [
            {
                "title": tab.title,
                "preset_id": tab.preset_id,
                "widget_ids": list(tab.widget_ids),
                "dock_state_b64": tab.dock_state_b64,
            }
            for tab in layout.tabs
        ],
    }


def layout_from_payload(payload: Any) -> MonitorLayout:
    if not isinstance(payload, dict):
        raise LayoutError("layout payload must be a JSON object")
    version = payload.get("schema_version")
    if version != MONITOR_LAYOUT_SCHEMA_VERSION:
        raise LayoutError(
            f"unsupported monitor layout schema {version!r} "
            f"(expected {MONITOR_LAYOUT_SCHEMA_VERSION!r})"
        )
    tabs_raw = payload.get("tabs")
    if not isinstance(tabs_raw, list) or not tabs_raw:
        raise LayoutError("layout payload has no tabs")
    tabs: list[TabLayout] = []
    for entry in tabs_raw:
        if not isinstance(entry, dict):
            raise LayoutError("layout tab entry must be an object")
        title = entry.get("title")
        preset_id = entry.get("preset_id")
        if not isinstance(title, str) or not isinstance(preset_id, str):
            raise LayoutError("layout tab entry is missing title/preset_id")
        widget_ids_raw = entry.get("widget_ids", [])
        widget_ids = tuple(
            wid for wid in widget_ids_raw if isinstance(wid, str) and wid
        ) if isinstance(widget_ids_raw, list) else ()
        dock_b64 = entry.get("dock_state_b64", "")
        tabs.append(TabLayout(
            title=title,
            preset_id=preset_id,
            widget_ids=widget_ids,
            dock_state_b64=dock_b64 if isinstance(dock_b64, str) else "",
        ))
    active_raw = payload.get("active_tab", 0)
    active = active_raw if isinstance(active_raw, int) and not isinstance(active_raw, bool) else 0
    replay_raw = payload.get("last_replay_path")
    return MonitorLayout(
        tabs=tuple(tabs),
        active_tab=active if 0 <= active < len(tabs) else 0,
        last_replay_path=replay_raw if isinstance(replay_raw, str) and replay_raw else None,
    )


def save_layout(path: Path | str, layout: MonitorLayout) -> None:
    """Atomic write: temp sibling + replace, so a crash cannot half-write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(layout_to_payload(layout), handle, indent=2)
    os.replace(tmp, target)


def load_layout(path: Path | str) -> MonitorLayout:
    """Strict load; raises OSError / LayoutError."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LayoutError(f"layout file is not valid JSON: {exc}") from exc
    return layout_from_payload(payload)


def load_layout_or_quarantine(
    path: Path | str,
    *,
    log_warning: Callable[[str], None] | None = None,
) -> MonitorLayout | None:
    """Load the saved layout; on corruption, back it up and return None.

    A missing file returns None silently (first run). A broken/foreign file is
    renamed to ``<name>.bak`` so the user's data is preserved for inspection
    while the application still opens with the default layout.
    """
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return load_layout(target)
    except (OSError, LayoutError) as exc:
        # Quarantine FIRST: preserving the user's file must not depend on the
        # logging callback (which may itself fail during early startup).
        backup = target.with_suffix(target.suffix + ".bak")
        with contextlib.suppress(OSError):
            os.replace(target, backup)
        if log_warning is not None:
            with contextlib.suppress(Exception):
                log_warning(
                    f"Saved Mission Monitor layout could not be used ({exc}); "
                    "it was backed up and the default layout was applied."
                )
        return None


__all__ = [
    "MONITOR_LAYOUT_FILENAME",
    "MONITOR_LAYOUT_SCHEMA_VERSION",
    "LayoutError",
    "MonitorLayout",
    "TabLayout",
    "layout_from_payload",
    "layout_to_payload",
    "load_layout",
    "load_layout_or_quarantine",
    "save_layout",
]
