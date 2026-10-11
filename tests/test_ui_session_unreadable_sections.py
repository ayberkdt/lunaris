"""Mission Studio session: a page that cannot be read is never saved blank.

Regression tests: ``collect_session_snapshot`` swallowed a failing page
accessor and substituted blank defaults (empty orbit strings, empty forces), so
the next auto-save replaced the user's last good session with a blank section
and nothing said so. The snapshot now names the unreadable sections in
``meta.unreadable_sections`` and both save paths refuse to write it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.ui.core.session_persistence import collect_session_snapshot


@dataclass
class _MiniCfg:
    value: int = 1


class _BrokenOrbitPage:
    def get_data(self) -> dict:
        raise ValueError("half-typed altitude")


class _Page:
    def __init__(self, data: dict) -> None:
        self._data = data

    def get_data(self) -> dict:
        return dict(self._data)

    def to_dict(self) -> dict:
        return {"timeline": {"duration": "1"}, "integrator": {"method": "DOP853"}}


def _snapshot(orbit_page) -> dict:
    from lunaris.ui.pages.data_files_page import DataFilesState
    from lunaris.ui.pages.result_exports_page import OutputPageState

    output_page = SimpleNamespace(
        get_state=lambda: OutputPageState(
            output_dir="out", generate_3d_plots=False, downsample_3d=1, report_preset="standard"
        )
    )
    data_page = SimpleNamespace(get_state=lambda: DataFilesState())
    return collect_session_snapshot(
        orbit_page=orbit_page,
        propagation_page=_Page({}),
        force_page=_Page({"gravity": {"enabled": True}}),
        output_page=output_page,
        data_page=data_page,
        gravity_cfg=SimpleNamespace(to_dict=lambda: {}),
        albedo_cfg=_MiniCfg(),
        solver_cfg=_MiniCfg(),
        spacecraft_cfg=_MiniCfg(),
        app_version="test",
    )


def test_unreadable_page_is_named_in_the_snapshot() -> None:
    snap = _snapshot(_BrokenOrbitPage())
    unreadable = snap["meta"].get("unreadable_sections")
    assert unreadable, "a failing page accessor must be reported, not saved blank silently"
    assert len(unreadable) == 1
    assert unreadable[0].startswith("orbit:")
    assert "half-typed altitude" in unreadable[0]


def test_readable_pages_report_nothing() -> None:
    snap = _snapshot(_Page({"mode": "hp_ha", "hp_km": "100"}))
    assert "unreadable_sections" not in snap["meta"]
    assert snap["orbit"]["hp_km"] == "100"


def _window(session_path: Path, snapshot: dict) -> SimpleNamespace:
    win = SimpleNamespace(session_path=session_path, messages=[])
    win._collect_preset_dict = lambda: snapshot
    win._log_message = lambda msg, severity="info": win.messages.append((severity, msg))
    return win


def test_auto_save_keeps_the_previous_session_when_a_page_is_unreadable(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from lunaris.ui.app import MainWindow

    path = tmp_path / "studio_session.json"
    path.write_text(json.dumps({"orbit": {"hp_km": "100"}}), encoding="utf-8")
    win = _window(path, _snapshot(_BrokenOrbitPage()))
    win._session_payload_for_save = lambda: MainWindow._session_payload_for_save(win)

    assert MainWindow._auto_save_session(win) is False
    assert json.loads(path.read_text(encoding="utf-8")) == {"orbit": {"hp_km": "100"}}
    warnings = [msg for sev, msg in win.messages if sev == "warning"]
    assert warnings and "orbit" in warnings[0]


def test_auto_save_writes_a_fully_readable_snapshot(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from lunaris.ui.app import MainWindow

    path = tmp_path / "studio_session.json"
    snapshot = _snapshot(_Page({"mode": "hp_ha", "hp_km": "250"}))
    win = _window(path, snapshot)
    win._session_payload_for_save = lambda: MainWindow._session_payload_for_save(win)

    assert MainWindow._auto_save_session(win) is True
    assert json.loads(path.read_text(encoding="utf-8"))["orbit"]["hp_km"] == "250"
