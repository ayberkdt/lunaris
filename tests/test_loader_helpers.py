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
from lunaris.loaders.io_surface import find_lola_albedo_product
from lunaris.loaders.spice_builder import maybe_autoinclude_lunar_fk, resolve_kernel_paths


def _write_cyl_label_pair(root: Path, stem: str, *, product_id: str, ppd: int) -> Path:
    img = root / f"{stem}.img"
    lbl = root / f"{stem}.lbl"
    img.write_bytes(b"\0" * 16)
    lbl.write_text(
        "\n".join(
            [
                f'PRODUCT_ID = "{product_id}"',
                f'^IMAGE = "{img.name}"',
                "LINES = 2",
                "LINE_SAMPLES = 2",
                "SAMPLE_TYPE = PC_REAL",
                "SAMPLE_BITS = 32",
                "UNIT = UNITLESS",
                "SCALING_FACTOR = 1.0",
                "OFFSET = 0.0",
                'MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"',
                f"MAP_RESOLUTION = {ppd} <PIXEL/DEGREE>",
                "MAXIMUM_LATITUDE = 90.0",
                "MINIMUM_LATITUDE = -90.0",
                "WESTERNMOST_LONGITUDE = 0.0",
                "EASTERNMOST_LONGITUDE = 360.0",
                'POSITIVE_LONGITUDE_DIRECTION = "EAST"',
                "CENTER_LATITUDE = 0.0 <DEGREE>",
                "CENTER_LONGITUDE = 0.0 <DEGREE>",
            ]
        ),
        encoding="utf-8",
    )
    return lbl


def test_autodetect_repository_data_roots_prefers_split_albedo_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "lunaris"
    data_root = project_root / "data"
    topo_dir = data_root / "topography_models"
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


def test_find_lola_albedo_product_ignores_diviner_dgdr_rasters(tmp_path: Path) -> None:
    root = tmp_path / "albedo_models"
    root.mkdir()
    _write_cyl_label_pair(root, "dgdr_ra_avg_cyl_032_img", product_id="DGDR_RA_AVG_CYL_032_IMG", ppd=32)
    ldam_lbl = _write_cyl_label_pair(root, "ldam_8_float", product_id="LDAM_8_FLOAT", ppd=8)

    lbl, img = find_lola_albedo_product(root)

    assert lbl == ldam_lbl
    assert img.name == "ldam_8_float.img"


def test_find_lola_albedo_product_rejects_direct_dgdr_label(tmp_path: Path) -> None:
    root = tmp_path / "albedo_models"
    root.mkdir()
    dgdr_lbl = _write_cyl_label_pair(root, "dgdr_ra_avg_cyl_032_img", product_id="DGDR_RA_AVG_CYL_032_IMG", ppd=32)

    try:
        find_lola_albedo_product(dgdr_lbl)
    except FileNotFoundError as exc:
        assert "LDAM" in str(exc)
    else:
        raise AssertionError("Diviner DGDR labels must not be accepted as albedo products")


def test_find_lunar_map_path_uses_canonical_assets_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "lunaris"
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
