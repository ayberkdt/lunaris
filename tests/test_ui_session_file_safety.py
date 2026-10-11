"""Mission Studio session file: unreadable files are kept, writes are atomic.

Regression tests: ``_try_load_last_session`` opened the file with plain
``utf-8`` (a UTF-8 BOM failed) and on any parse error kept the defaults, and
30 s later the auto-save truncated ``studio_session.json`` and wrote those
defaults over the user's profile. Writes were ``open('w')`` + ``json.dump``,
so an exception mid-dump left a truncated file. The legacy migration logged
"Migrated" even when the forward copy failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.ui.core.session_persistence import (
    SessionFileUnreadable,
    quarantine_session_file,
    read_session_file,
    write_session_file,
)


def test_bom_session_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "studio_session.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"a": 1}).encode("utf-8"))
    assert read_session_file(path) == {"a": 1}


@pytest.mark.parametrize("payload", [b'{"a": 1,,}', b"\xff\xfe\x00garbage", b'{"a": '])
def test_unreadable_session_file_raises(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "studio_session.json"
    path.write_bytes(payload)
    with pytest.raises(SessionFileUnreadable):
        read_session_file(path)


def test_quarantine_never_overwrites_an_earlier_backup(tmp_path: Path) -> None:
    path = tmp_path / "studio_session.json"
    path.write_text("first", encoding="utf-8")
    first = quarantine_session_file(path)
    path.write_text("second", encoding="utf-8")
    second = quarantine_session_file(path)
    assert first.name == "studio_session.json.bak"
    assert second.name == "studio_session.json.bak.1"
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"
    assert not path.exists()


def test_failed_write_keeps_the_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "studio_session.json"
    write_session_file(path, {"good": True})
    with pytest.raises(TypeError):
        write_session_file(path, {"bad": object()})  # not JSON serializable mid-dump
    assert json.loads(path.read_text(encoding="utf-8")) == {"good": True}
    assert [p.name for p in tmp_path.iterdir()] == ["studio_session.json"]


def _window(session_path: Path, legacy: Path | None = None) -> SimpleNamespace:
    win = SimpleNamespace(
        session_path=session_path,
        _legacy_session_path=legacy,
        applied=[],
        messages=[],
    )
    win._apply_preset_dict = lambda data: win.applied.append(data)
    win._log_message = lambda msg, severity="info": win.messages.append((severity, msg))
    return win


def test_unreadable_last_session_is_moved_aside_not_overwritten(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from lunaris.ui.app import MainWindow

    path = tmp_path / "studio_session.json"
    path.write_text('{"orbit": {"hp_km": 42},}', encoding="utf-8")  # stray comma
    win = _window(path)
    MainWindow._try_load_last_session(win)
    assert win.applied == []
    backup = tmp_path / "studio_session.json.bak"
    assert backup.read_text(encoding="utf-8") == '{"orbit": {"hp_km": 42},}'
    assert not path.exists()  # the next auto-save writes a new file, not over the user's
    assert any("studio_session.json.bak" in msg for _sev, msg in win.messages)


def test_bom_last_session_is_restored(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from lunaris.ui.app import MainWindow

    path = tmp_path / "studio_session.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"orbit": {"hp_km": 42}}')
    win = _window(path)
    MainWindow._try_load_last_session(win)
    assert win.applied == [{"orbit": {"hp_km": 42}}]


def test_failed_legacy_copy_is_not_reported_as_migrated(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from lunaris.ui.app import MainWindow

    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"x": 1}', encoding="utf-8")
    target = tmp_path / "missing_dir" / "studio_session.json"  # parent does not exist
    win = _window(target, legacy)
    MainWindow._try_load_last_session(win)
    assert win.applied == [{"x": 1}]
    assert not any("Migrated" in msg for _sev, msg in win.messages)
    assert any(sev == "warning" for sev, _msg in win.messages)
