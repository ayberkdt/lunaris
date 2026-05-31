"""Safety-contract tests for the ``lunaris-data`` downloader.

Complements ``test_data_downloader.py`` / ``test_lunaris_data_cli.py`` (which
already cover the happy paths, selection semantics, hash verification, and
manifest schema). This module focuses on the *refusal* and *cleanup* contracts
that protect against silently installing corrupt or untrusted data:

- non-http(s)/file URL schemes are refused (no SSRF / arbitrary-scheme fetch),
- a failed or hash-mismatched download never leaves a ``.part`` turd or a
  corrupt final file behind,
- downloaded files always land under the resolved data root, never in ``src/``.

All tests are network-free (``file://`` URLs + temp dirs).
"""

from __future__ import annotations

import hashlib

import pytest

from lunaris.cli import data as data_cli


# --------------------------------------------------------------------------- #
# URL-scheme refusal
# --------------------------------------------------------------------------- #
def test_stream_to_file_refuses_unsupported_scheme(tmp_path):
    dest = tmp_path / "out.bin"
    with pytest.raises(ValueError, match="unsupported scheme"):
        data_cli._stream_to_file("ftp://example.com/payload.bin", dest)
    assert not dest.exists()


def test_download_entry_unsupported_scheme_returns_error_and_no_files(tmp_path):
    root = tmp_path / "root"
    entry = {
        "name": "evil", "group": "gravity", "filename": "g.tab",
        "target_subdir": "gravity_models", "url": "ftp://example.com/g.tab",
    }
    status = data_cli.download_entry(entry, root, overwrite=True, verify=True)
    assert status == "error"
    target = data_cli.dataset_target_path(root, entry)
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


# --------------------------------------------------------------------------- #
# .part / corrupt-file cleanup
# --------------------------------------------------------------------------- #
def test_failed_download_leaves_no_part_file(tmp_path):
    # A file:// URL to a non-existent source fails inside urlopen; the partial
    # file must not be left behind.
    missing = tmp_path / "does_not_exist.bin"
    root = tmp_path / "root"
    entry = {
        "name": "gone", "group": "ephemeris", "filename": "d.bin",
        "target_subdir": "ephemeris_models", "url": missing.as_uri(),
    }
    status = data_cli.download_entry(entry, root, overwrite=True, verify=True)
    assert status == "error"
    target = data_cli.dataset_target_path(root, entry)
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_hash_mismatch_removes_part_and_does_not_install(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"the-real-bytes")
    root = tmp_path / "root"
    entry = {
        "name": "bad", "group": "gravity", "filename": "bad.bin",
        "target_subdir": "gravity_models", "url": src.as_uri(),
        "sha256": "f" * 64,  # deliberately wrong
    }
    status = data_cli.download_entry(entry, root, overwrite=True, verify=True)
    assert status == "hash_mismatch"
    target = data_cli.dataset_target_path(root, entry)
    assert not target.exists()                                   # corrupt file not installed
    assert not target.with_name(target.name + ".part").exists()  # .part cleaned up


def test_successful_download_is_atomic_and_under_root(tmp_path):
    payload = b"lunaris" * 64
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    root = tmp_path / "root"
    entry = {
        "name": "ok", "group": "ephemeris", "filename": "ok.bin",
        "target_subdir": "ephemeris_models", "url": src.as_uri(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    status = data_cli.download_entry(entry, root, overwrite=True, verify=True)
    assert status == "ok"

    target = data_cli.dataset_target_path(root, entry)
    assert target.read_bytes() == payload
    assert not target.with_name(target.name + ".part").exists()
    # The installed file is under the data root and never under a src/ tree.
    assert root.resolve() in target.resolve().parents
    assert "src" not in target.resolve().relative_to(root.resolve()).parts
