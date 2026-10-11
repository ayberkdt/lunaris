"""The result auditor reports an HDF5 archive it cannot read instead of skipping it.

Regression test: ``_is_batch_hdf5`` returned False when h5py was missing or
the file could not be opened, so the archive was silently left out and the
audit looked clean.
"""

from __future__ import annotations

import builtins
from pathlib import Path

from lunaris.analysis.ensemble import result_audit as ra


def test_hdf5_archive_without_h5py_is_reported_invalid(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "batch_output.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)
    real_import = builtins.__import__

    def _no_h5py(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("no h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_h5py)
    report = ra.audit_root(tmp_path)
    invalid = report["invalid"]
    assert [Path(e["path"]).name for e in invalid] == ["batch_output.h5"]
    assert "h5py" in invalid[0]["reasons"][0]


def test_unreadable_hdf5_is_reported_invalid(tmp_path: Path) -> None:
    (tmp_path / "broken.h5").write_bytes(b"not an hdf5 file")
    probe = ra._is_batch_hdf5(tmp_path / "broken.h5")
    assert isinstance(probe, str)
