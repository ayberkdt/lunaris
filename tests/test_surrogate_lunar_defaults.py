# -*- coding: utf-8 -*-
"""
Regression tests for the lunarized surrogate-gravity configuration layer.

These tests focus on the "glue" that keeps the experimental ST-LRPS tooling aligned
with the main Moon-centric simulation stack:

- default dataset parameters must point to the lunar body
- preset names / defaults must no longer advertise Earth/LEO workflows
- auto-discovery must ignore legacy runs that do not look lunar-compatible
"""

from __future__ import annotations

import json
from pathlib import Path

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.physics.surrogate_gravity import discover_st_lrps_model_dirs
from lunaris.surrogate.st_lrps.data.dataset_parameters import (
    DEFAULT_DATASET_CONFIG,
    is_lunar_body_signature,
    looks_like_lunar_run_config,
)
from lunaris.surrogate.st_lrps.data.spatial_cloud_parameters import (
    DEFAULT_SPATIAL_CLOUD_CONFIG,
    PRESETS,
)


def test_default_surrogate_dataset_parameters_point_to_the_moon() -> None:
    assert DEFAULT_DATASET_CONFIG.central_body == "moon"
    assert DEFAULT_DATASET_CONFIG.mu_si == float(MU_MOON)
    assert DEFAULT_DATASET_CONFIG.r_ref_m == float(R_MOON)
    assert Path(DEFAULT_DATASET_CONFIG.gravity_gfc_path).is_file()


def test_spatial_cloud_presets_are_lunar_and_default_preset_is_lunar() -> None:
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.coeff_source == "gfc"
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.alt_min_km == 200.0
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.alt_max_km == 600.0
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.resolved_out_path().startswith("potential_cloud_moon_")
    assert PRESETS
    assert all(name.startswith(("moon_", "debug_")) for name in PRESETS)
    assert all("earth" not in name for name in PRESETS)


def test_lunar_run_config_detection_requires_actual_moon_evidence() -> None:
    assert looks_like_lunar_run_config({"central_body": "moon"}) is False
    assert looks_like_lunar_run_config({"central_body": "moon", "resolved_mu_si": float(MU_MOON)}) is True
    assert looks_like_lunar_run_config({"resolved_mu_si": float(MU_MOON)}) is True
    assert looks_like_lunar_run_config({"dataset_meta": {"r_ref_m": float(R_MOON)}}) is True
    assert looks_like_lunar_run_config(
        {"central_body": "moon", "resolved_mu_si": float(MU_MOON), "r_ref_m": 6_378_137.0}
    ) is False
    assert looks_like_lunar_run_config({"data": r"C:\legacy\earth_cloud.h5"}) is False


def test_lunar_body_signature_requires_consistent_mu_and_radius() -> None:
    assert is_lunar_body_signature(mu_si=float(MU_MOON), r_ref_m=float(R_MOON)) is True
    assert is_lunar_body_signature(mu_si=float(MU_MOON), r_ref_m=6_378_137.0) is False
    assert is_lunar_body_signature(mu_si=3.986004418e14, r_ref_m=float(R_MOON)) is False


def test_discover_st_lrps_model_dirs_filters_non_lunar_runs(tmp_path: Path) -> None:
    lunar_run = tmp_path / "run_lunar"
    earth_run = tmp_path / "run_legacy"

    for run_dir, config in (
        (
            lunar_run,
            {
                "central_body": "moon",
                "resolved_mu_si": float(MU_MOON),
            },
        ),
        (
            earth_run,
            {
                "data": r"C:\old\earth_dataset.h5",
            },
        ),
    ):
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (run_dir / "checkpoints" / "ckpt_best.pt").write_text("placeholder", encoding="utf-8")

    discovered = discover_st_lrps_model_dirs(tmp_path)

    assert discovered == [lunar_run.resolve()]
