"""Finiteness validation for batch/ensemble configuration dataclasses.

IEEE-754 makes ``NaN < 0`` and ``NaN <= 0`` both False, so plain sign checks
silently accept NaN. These tests pin that every physical/numeric field rejects
non-finite values at construction instead of letting them reach Cholesky
factorization, sampling, memory budgeting, or the integrator step.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.batch_defs import (
    BatchPropagationConfig,
    SpacecraftUncertainty,
    StateUncertainty,
)

NON_FINITE = (float("nan"), float("inf"), float("-inf"))


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("field_name", ["sigma_r_m", "sigma_v_m_s"])
def test_state_uncertainty_rejects_non_finite_sigmas(field_name, bad):
    with pytest.raises(ValueError, match="finite"):
        StateUncertainty(**{field_name: bad})


def test_state_uncertainty_rejects_non_finite_covariance():
    cov = np.eye(6, dtype=np.float64)
    cov[2, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        StateUncertainty(covariance_6x6=cov)


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize(
    "field_name", ["sigma_mass_kg", "sigma_cd", "sigma_cr", "sigma_area_m2"]
)
def test_spacecraft_uncertainty_rejects_non_finite_sigmas(field_name, bad):
    with pytest.raises(ValueError, match="finite"):
        SpacecraftUncertainty(**{field_name: bad})


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize(
    "field_name",
    ["dt_s", "max_vram_gb", "max_result_memory_gb", "impact_alt_km"],
)
def test_batch_config_rejects_non_finite_numeric_fields(field_name, bad):
    with pytest.raises(ValueError, match="finite"):
        BatchPropagationConfig(**{field_name: bad})


def test_batch_config_valid_defaults_still_construct():
    cfg = BatchPropagationConfig()
    assert cfg.dt_s > 0.0
    assert StateUncertainty().to_covariance().shape == (6, 6)
