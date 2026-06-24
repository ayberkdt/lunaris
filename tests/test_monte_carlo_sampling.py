"""Core Monte Carlo sampling design contracts."""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.core.monte_carlo_engine import generate_standard_normal_design


def test_standard_normal_design_lhs_and_sobol_are_finite_and_deterministic() -> None:
    lhs = generate_standard_normal_design(7, 3, "lhs", seed=123)
    sobol_a = generate_standard_normal_design(7, 3, "sobol_scrambled", seed=123)
    sobol_b = generate_standard_normal_design(7, 3, "sobol_scrambled", seed=123)
    sobol_c = generate_standard_normal_design(7, 3, "sobol_scrambled", seed=124)

    assert lhs.shape == (7, 3)
    assert sobol_a.shape == (7, 3)
    assert np.isfinite(lhs).all()
    assert np.isfinite(sobol_a).all()
    assert np.allclose(sobol_a, sobol_b)
    assert not np.allclose(sobol_a, sobol_c)


def test_standard_normal_design_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="sampling_method must be one of"):
        generate_standard_normal_design(8, 3, "grid", seed=123)
