"""
Regression tests for the lunarized surrogate-gravity configuration layer.

These tests focus on the "glue" that keeps the experimental ST-LRPS tooling aligned
with the main Moon-centric simulation stack:

- default dataset parameters must point to the lunar body
- preset names / defaults must no longer advertise Earth/LEO workflows
- auto-discovery must ignore legacy runs that do not look lunar-compatible
"""

from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.lunar_data import (
    DEFAULT_LUNAR_GRAVITY_PATH as COMMON_DEFAULT_LUNAR_GRAVITY_PATH,
)
from lunaris.common.lunar_data import (
    is_lunar_body_signature as common_is_lunar_body_signature,
)
from lunaris.common.lunar_data import (
    looks_like_lunar_run_config as common_looks_like_lunar_run_config,
)
from lunaris.common.lunar_data import (
    resolve_lunar_gravity_path as common_resolve_lunar_gravity_path,
)
from lunaris.surrogate.runtime import discover_st_lrps_model_dirs
from lunaris.surrogate.st_lrps.data.dataset_parameters import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_LUNAR_GRAVITY_PATH,
    is_lunar_body_signature,
    looks_like_lunar_run_config,
    resolve_lunar_gravity_path,
)
from lunaris.surrogate.st_lrps.data.spatial_cloud_parameters import (
    DEFAULT_CLOUD_SUITE_CONFIG,
    DEFAULT_SPATIAL_CLOUD_CONFIG,
    PRESETS,
    STRONG_BENCHMARK_ALT_MAX_KM,
    STRONG_BENCHMARK_ALT_MIN_KM,
    STRONG_BENCHMARK_DEGREE_MAX,
    STRONG_BENCHMARK_DEGREE_MIN,
)


@pytest.mark.requires_data
def test_default_surrogate_dataset_parameters_point_to_the_moon() -> None:
    assert DEFAULT_DATASET_CONFIG.central_body == "moon"
    assert DEFAULT_DATASET_CONFIG.mu_si == float(MU_MOON)
    assert DEFAULT_DATASET_CONFIG.r_ref_m == float(R_MOON)
    assert Path(DEFAULT_DATASET_CONFIG.gravity_gfc_path).is_file()


def test_dataset_parameters_reexports_common_lunar_helpers() -> None:
    assert DEFAULT_LUNAR_GRAVITY_PATH == COMMON_DEFAULT_LUNAR_GRAVITY_PATH
    assert resolve_lunar_gravity_path is common_resolve_lunar_gravity_path
    assert is_lunar_body_signature is common_is_lunar_body_signature
    assert looks_like_lunar_run_config is common_looks_like_lunar_run_config


def test_spatial_cloud_presets_are_lunar_and_default_preset_is_lunar() -> None:
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.coeff_source == "gfc"
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.degree_min == STRONG_BENCHMARK_DEGREE_MIN
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.degree_max == STRONG_BENCHMARK_DEGREE_MAX
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.alt_min_km == STRONG_BENCHMARK_ALT_MIN_KM
    assert DEFAULT_SPATIAL_CLOUD_CONFIG.alt_max_km == STRONG_BENCHMARK_ALT_MAX_KM
    assert DEFAULT_CLOUD_SUITE_CONFIG.degree_min == STRONG_BENCHMARK_DEGREE_MIN
    assert DEFAULT_CLOUD_SUITE_CONFIG.degree_max == STRONG_BENCHMARK_DEGREE_MAX
    assert DEFAULT_CLOUD_SUITE_CONFIG.train_alt_min_km == STRONG_BENCHMARK_ALT_MIN_KM
    assert DEFAULT_CLOUD_SUITE_CONFIG.train_alt_max_km == STRONG_BENCHMARK_ALT_MAX_KM
    assert DEFAULT_CLOUD_SUITE_CONFIG.train_total_n == 10_000_000
    assert DEFAULT_CLOUD_SUITE_CONFIG.val_n == 2_000_000
    assert DEFAULT_CLOUD_SUITE_CONFIG.test_n == 2_000_000
    assert DEFAULT_CLOUD_SUITE_CONFIG.ood_low_n == 500_000
    assert DEFAULT_CLOUD_SUITE_CONFIG.ood_high_n == 500_000
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
