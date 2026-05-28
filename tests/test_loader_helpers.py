# -*- coding: utf-8 -*-
"""
Regression tests for the shared loader helper layer.

These tests focus on the small discovery/resolution helpers that were moved out
of UI/analysis/model modules into `loaders`. They intentionally avoid SPICE and
Qt so they can validate filesystem policy in isolation.
"""

from __future__ import annotations

from pathlib import Path

from lunaris.loaders.io_helpers import (
    DataRootHints,
    autodetect_repository_data_roots,
    find_lunar_map_path,
)
from lunaris.loaders.spice_builder import maybe_autoinclude_lunar_fk, resolve_kernel_paths


def test_autodetect_repository_data_roots_prefers_split_albedo_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "LUNAR_SIMULATION"
    data_root = project_root / "data"
    topo_dir = data_root / "topografy_models"
    albedo_dir = data_root / "albedo_models"
    kernel_dir = data_root / "ephemeris_models"

    topo_dir.mkdir(parents=True)
    albedo_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)

    (topo_dir / "ldem_64_float.img").write_bytes(b"topography")
    (albedo_dir / "ldam_8_float.img").write_bytes(b"albedo")
    (kernel_dir / "de440.bsp").write_bytes(b"kernel")

    detected, messages = autodetect_repository_data_roots(
        project_root,
        current=DataRootHints(
            ldem_root=str(topo_dir),
            albedo_root=str(topo_dir),
            kernel_dir="",
            use_ldem_for_albedo=True,
        ),
    )

    assert Path(detected.ldem_root) == topo_dir.resolve()
    assert Path(detected.albedo_root) == albedo_dir.resolve()
    assert Path(detected.kernel_dir) == kernel_dir.resolve()
    assert detected.use_ldem_for_albedo is False
    assert any("Albedo auto-filled" in message for message in messages)


def test_find_lunar_map_path_uses_canonical_assets_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "LUNAR_SIMULATION"
    assets_dir = project_root / "data" / "assets"
    start_dir = project_root / "analysis"
    assets_dir.mkdir(parents=True)
    start_dir.mkdir(parents=True)

    texture = assets_dir / "lroc_color_2k.jpg"
    texture.write_bytes(b"not-a-real-image-but-path-discovery-only")

    found = find_lunar_map_path(start_dir=start_dir)

    assert found == str(texture.resolve())


def test_resolve_kernel_paths_accepts_optional_text_wrapped_kernel_files(tmp_path: Path) -> None:
    actual = tmp_path / "naif0012.tls.txt"
    actual.write_text("LSK", encoding="utf-8")

    resolved = resolve_kernel_paths([str(tmp_path / "naif0012.tls")])

    assert resolved == [str(actual)]


def test_maybe_autoinclude_lunar_fk_injects_best_colocated_frame_kernel(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()

    bpc = kernel_dir / "moon_pa_de440_200625.bpc"
    tf = kernel_dir / "moon_de440_220930.tf.txt"
    bpc.write_text("bpc", encoding="utf-8")
    tf.write_text("tf", encoding="utf-8")

    out = maybe_autoinclude_lunar_fk([str(bpc)], "MOON_PA")

    assert out[0] == str(tf)
    assert out[1] == str(bpc)
