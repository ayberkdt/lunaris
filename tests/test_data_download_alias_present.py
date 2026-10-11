"""``lunaris-data download`` must honour a dataset already present under an alias.

``verify`` resolves an entry through its canonical filename and then its
``aliases`` (``resolve_dataset_path``), but ``download_entry`` only checked the
canonical path, so a product installed under a documented alias name was
downloaded again next to it.
"""

from __future__ import annotations

from pathlib import Path

from lunaris.cli import data as data_cli


def test_alias_copy_counts_as_present(tmp_path: Path) -> None:
    src = tmp_path / "server"
    src.mkdir()
    (src / "PROD.IMG").write_bytes(b"raster")
    entry = {
        "name": "prod", "group": "topography", "filename": "PROD.IMG", "aliases": ["prod_alias.img"],
        "target_subdir": "topography_models", "url": (src / "PROD.IMG").as_uri(),
    }
    root = tmp_path / "data"
    (root / "topography_models").mkdir(parents=True)
    (root / "topography_models" / "prod_alias.img").write_bytes(b"raster")

    assert data_cli.download_entry(entry, root) == "present"
    assert not (root / "topography_models" / "PROD.IMG").exists()
    # --overwrite still refetches the canonical file.
    assert data_cli.download_entry(entry, root, overwrite=True) == "ok"
    assert (root / "topography_models" / "PROD.IMG").read_bytes() == b"raster"
