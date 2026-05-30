"""Network-free tests for the ``lunaris-data`` external-data CLI.

All download paths are exercised with local ``file://`` URLs and temporary
directories; no test touches a live network server.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lunaris.cli import data as data_cli


def _write_manifest(tmp_path: Path, datasets) -> Path:
    manifest = {"schema_version": data_cli.SCHEMA_VERSION, "datasets": datasets}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #
def test_repo_manifest_loads_and_is_well_formed():
    manifest = data_cli.load_manifest(data_cli.default_manifest_path())
    assert manifest["schema_version"] == data_cli.SCHEMA_VERSION
    groups = {d["group"] for d in manifest["datasets"]}
    assert {"ephemeris", "gravity", "topography"}.issubset(groups)
    for entry in manifest["datasets"]:
        assert entry["group"] in data_cli.GROUPS
        assert entry["target_subdir"] in data_cli.CANONICAL_SUBDIRS
        assert "filename" in entry


def test_load_manifest_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")  # no datasets list
    try:
        data_cli.load_manifest(bad)
    except ValueError:
        return
    raise AssertionError("malformed manifest should raise ValueError")


# --------------------------------------------------------------------------- #
# Data-root resolution
# --------------------------------------------------------------------------- #
def test_resolve_data_root_cli_arg_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_DATA_DIR", str(tmp_path / "env"))
    assert data_cli.resolve_data_root(str(tmp_path / "cli")) == (tmp_path / "cli").resolve()


def test_resolve_data_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_DATA_DIR", str(tmp_path / "env"))
    assert data_cli.resolve_data_root(None) == (tmp_path / "env").resolve()


def test_resolve_data_root_default_fallback(monkeypatch):
    monkeypatch.delenv("LUNARIS_DATA_DIR", raising=False)
    monkeypatch.delenv("STLRPS_DATA_DIR", raising=False)
    assert data_cli.resolve_data_root(None).name == "data"


def test_target_path_construction(tmp_path):
    entry = {"filename": "x.bsp", "target_subdir": "ephemeris_models"}
    assert data_cli.dataset_target_path(tmp_path, entry) == tmp_path / "ephemeris_models" / "x.bsp"


# --------------------------------------------------------------------------- #
# Download behavior (file:// URLs)
# --------------------------------------------------------------------------- #
def test_download_via_file_url_with_sha256_ok(tmp_path):
    src = tmp_path / "src.bin"
    payload = b"lunaris-test-bytes" * 100
    src.write_bytes(payload)
    root = tmp_path / "root"
    entry = {
        "name": "demo", "group": "ephemeris", "filename": "demo.bin",
        "target_subdir": "ephemeris_models", "url": src.as_uri(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert data_cli.download_entry(entry, root, overwrite=True, verify=True) == "ok"
    out = root / "ephemeris_models" / "demo.bin"
    assert out.read_bytes() == payload
    assert not out.with_name("demo.bin.part").exists()  # temp file cleaned up


def test_download_sha256_mismatch_removes_corrupt_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    root = tmp_path / "root"
    entry = {
        "name": "bad", "group": "gravity", "filename": "bad.bin",
        "target_subdir": "gravity_models", "url": src.as_uri(), "sha256": "0" * 64,
    }
    assert data_cli.download_entry(entry, root, overwrite=True, verify=True) == "hash_mismatch"
    assert not (root / "gravity_models" / "bad.bin").exists()


def test_download_url_null_is_manual(tmp_path):
    root = tmp_path / "root"
    entry = {"name": "man", "group": "gravity", "filename": "g.tab",
             "target_subdir": "gravity_models", "url": None, "notes": "place manually"}
    assert data_cli.download_entry(entry, root, overwrite=True) == "manual"
    assert not (root / "gravity_models" / "g.tab").exists()


def test_download_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "s.bin"
    src.write_bytes(b"q")
    root = tmp_path / "root"
    entry = {"name": "d", "group": "ephemeris", "filename": "d.bin",
             "target_subdir": "ephemeris_models", "url": src.as_uri()}
    assert data_cli.download_entry(entry, root, dry_run=True) == "dry-run"
    assert not (root / "ephemeris_models" / "d.bin").exists()


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def test_verify_entry_statuses(tmp_path):
    root = tmp_path / "root"
    (root / "ephemeris_models").mkdir(parents=True)
    payload = b"xyz"
    (root / "ephemeris_models" / "present.bin").write_bytes(payload)
    good_hash = hashlib.sha256(payload).hexdigest()

    valid = {"name": "v", "filename": "present.bin", "target_subdir": "ephemeris_models",
             "url": "https://x/y", "sha256": good_hash}
    present = {"name": "p", "filename": "present.bin", "target_subdir": "ephemeris_models",
               "url": "https://x/y", "sha256": None}
    missing = {"name": "m", "filename": "absent.bin", "target_subdir": "gravity_models",
               "url": "https://x/y"}
    manual = {"name": "mm", "filename": "absent.tab", "target_subdir": "gravity_models",
              "url": None}
    mismatch = {"name": "b", "filename": "present.bin", "target_subdir": "ephemeris_models",
                "url": "https://x/y", "sha256": "0" * 64}

    assert data_cli.verify_entry(valid, root) == "valid"
    assert data_cli.verify_entry(present, root) == "present"
    assert data_cli.verify_entry(missing, root) == "missing"
    assert data_cli.verify_entry(manual, root) == "manual_missing"
    assert data_cli.verify_entry(mismatch, root) == "hash_mismatch"


def test_verify_command_optional_missing_exits_zero(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"name": "opt", "group": "gravity", "filename": "g.tab",
         "target_subdir": "gravity_models", "url": None, "required": False},
    ])
    rc = data_cli.main(["--data-dir", str(tmp_path / "root"),
                        "--manifest", str(manifest), "verify"])
    assert rc == 0


def test_verify_command_required_missing_exits_nonzero(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"name": "req", "group": "ephemeris", "filename": "naif0012.tls",
         "target_subdir": "ephemeris_models", "url": "https://naif.example/naif0012.tls",
         "required": True},
    ])
    rc = data_cli.main(["--data-dir", str(tmp_path / "root"),
                        "--manifest", str(manifest), "verify"])
    assert rc == 1


# --------------------------------------------------------------------------- #
# list / path commands
# --------------------------------------------------------------------------- #
def test_list_command_reports_groups_and_names(tmp_path, capsys):
    manifest = _write_manifest(tmp_path, [
        {"name": "alpha", "group": "ephemeris", "filename": "a.tls",
         "target_subdir": "ephemeris_models", "url": "https://x/a", "required": True},
        {"name": "beta", "group": "gravity", "filename": "b.tab",
         "target_subdir": "gravity_models", "url": None, "required": False},
    ])
    rc = data_cli.main(["--manifest", str(manifest), "--data-dir", str(tmp_path), "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[ephemeris]" in out and "[gravity]" in out
    assert "alpha" in out and "beta" in out
    assert "manual" in out  # beta has no URL


def test_path_command_prints_subdirs(tmp_path, capsys):
    rc = data_cli.main(["--data-dir", str(tmp_path / "root"), "path"])
    out = capsys.readouterr().out
    assert rc == 0
    for sub in data_cli.CANONICAL_SUBDIRS:
        assert sub in out
