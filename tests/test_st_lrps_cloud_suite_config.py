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


@pytest.mark.parametrize(
    ("overrides", "pair"),
    [
        ({"val_seed": 42}, ("val_seed", "train_uniform_seed")),
        ({"test_seed": 42}, ("test_seed", "train_uniform_seed")),
        ({"val_seed": 7, "test_seed": 7}, ("test_seed", "val_seed")),
    ],
)
def test_same_shell_seed_collision_is_rejected(overrides, pair):
    # train_stratified_uniform, val and test sample the same shell with the
    # same strategy, so a shared seed replays the training rows as val/test.
    with pytest.raises(ValueError, match=rf"{pair[0]} and {pair[1]} both sample"):
        CloudSuiteConfig(**overrides)


def test_seed_reuse_across_different_distributions_is_allowed():
    # inverse_r2 / residual_mag / boundary / OOD components draw from another
    # distribution or shell, so Sidera's rule (ported as-is) leaves them free.
    cfg = CloudSuiteConfig(
        train_inverse_r2_seed=42,
        train_residual_mag_seed=42,
        train_boundary_seed=42,
        ood_low_seed=42,
        ood_high_seed=42,
    )
    assert cfg.train_uniform_seed == 42


def test_ood_low_band_below_the_surface_is_rejected():
    # 20 km - 40 km = -20 km: the OOD-low shell would sit inside the Moon.
    with pytest.raises(ValueError, match="below the lunar surface"):
        CloudSuiteConfig(train_alt_min_km=20.0, ood_margin_km=40.0)


def test_ood_low_band_touching_the_surface_is_allowed():
    cfg = CloudSuiteConfig(train_alt_min_km=40.0, ood_margin_km=40.0)
    assert cfg.ood_low_alt_min_km == 0.0
