"""CloudSuiteConfig guards that reject a bad suite before any point is generated."""

from __future__ import annotations

import pytest

from lunaris.surrogate.st_lrps.data.spatial_cloud_parameters import (
    DEFAULT_CLOUD_SUITE_CONFIG,
    SUITE_PRESETS,
    CloudSuiteConfig,
)


def test_default_and_preset_suites_stay_valid():
    # Constructing them at import already ran __post_init__; re-check the
    # derived OOD-low floor explicitly so a default change cannot slip past.
    for cfg in (DEFAULT_CLOUD_SUITE_CONFIG, *SUITE_PRESETS.values()):
        assert cfg.ood_low_alt_min_km >= 0.0


def test_ood_low_band_below_the_surface_is_rejected():
    # 20 km - 40 km = -20 km: the OOD-low shell would sit inside the Moon.
    with pytest.raises(ValueError, match="below the lunar surface"):
        CloudSuiteConfig(train_alt_min_km=20.0, ood_margin_km=40.0)


def test_ood_low_band_touching_the_surface_is_allowed():
    cfg = CloudSuiteConfig(train_alt_min_km=40.0, ood_margin_km=40.0)
    assert cfg.ood_low_alt_min_km == 0.0
