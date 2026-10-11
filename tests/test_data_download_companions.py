"""``lunaris-data download`` fetches companion files, as ``verify`` expects.

Regression test: ``download_entry`` fetched only the entry's main URL and
returned ``present``/``ok``; detached labels listed under ``companion_files``
(the ``.lbl`` every LOLA/GRAIL product needs) were never downloaded, so
``verify --strict`` failed right after a fresh download.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from lunaris.cli import data as data_cli


def _entry(src_dir: Path, *, label_sha: str | None = None, required: bool = True) -> dict:
    companion: dict = {"filename": "prod.lbl", "required": required}
    if label_sha is not None:
        companion["sha256"] = label_sha
    return {
        "name": "prod", "group": "topography", "filename": "prod.img",
        "target_subdir": "topography_models", "url": (src_dir / "prod.img").as_uri(),
        "companion_files": [companion],
    }


def _sources(tmp_path: Path) -> Path:
    src = tmp_path / "server"
    src.mkdir()
    (src / "prod.img").write_bytes(b"raster")
    (src / "prod.lbl").write_text("PDS_VERSION_ID = PDS3\n", encoding="utf-8")
    return src


def test_download_fetches_the_companion_from_the_main_files_folder(tmp_path: Path) -> None:
    src = _sources(tmp_path)
    root = tmp_path / "data"
    sha = hashlib.sha256((src / "prod.lbl").read_bytes()).hexdigest()
    entry = _entry(src, label_sha=sha)
    assert data_cli.download_entry(entry, root) == "ok"
    assert (root / "topography_models" / "prod.lbl").read_text(encoding="utf-8").startswith("PDS_VERSION_ID")
    detail = data_cli.verify_entry_detail(entry, root)
    assert [c["status"] for c in detail["companions"]] == ["valid"]


def test_present_main_file_still_gets_its_missing_companion(tmp_path: Path) -> None:
    src = _sources(tmp_path)
    root = tmp_path / "data"
    (root / "topography_models").mkdir(parents=True)
    (root / "topography_models" / "prod.img").write_bytes(b"raster")
    assert data_cli.download_entry(_entry(src), root) == "present"
    assert (root / "topography_models" / "prod.lbl").exists()


def test_required_companion_hash_mismatch_is_reported(tmp_path: Path) -> None:
    src = _sources(tmp_path)
    root = tmp_path / "data"
    assert data_cli.download_entry(_entry(src, label_sha="0" * 64), root) == "hash_mismatch"
    assert not (root / "topography_models" / "prod.lbl").exists()


def test_optional_companion_failure_does_not_fail_the_entry(tmp_path: Path) -> None:
    src = _sources(tmp_path)
    (src / "prod.lbl").unlink()
    root = tmp_path / "data"
    assert data_cli.download_entry(_entry(src, required=False), root) == "ok"


def test_dry_run_writes_no_companion(tmp_path: Path) -> None:
    src = _sources(tmp_path)
    root = tmp_path / "data"
    assert data_cli.download_entry(_entry(src), root, dry_run=True) == "dry-run"
    assert not (root / "topography_models").exists()
