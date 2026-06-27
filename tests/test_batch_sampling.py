from __future__ import annotations

import numpy as np
import pytest

from lunaris.batch.sampling import generate_standard_normal_design


def test_random_standard_normal_design_is_deterministic_with_seed() -> None:
    first = generate_standard_normal_design(8, 3, "random", seed=1234)
    second = generate_standard_normal_design(8, 3, "random", seed=1234)

    assert first.shape == (8, 3)
    np.testing.assert_allclose(first, second)


def test_random_standard_normal_design_uses_supplied_rng_sequence() -> None:
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)

    first = generate_standard_normal_design(5, 2, "random", seed=0, rng=rng_a)
    second = generate_standard_normal_design(5, 2, "random", seed=999, rng=rng_b)

    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize("method", ["lhs", "sobol", "sobol_scrambled"])
def test_space_filling_designs_have_expected_shape(method: str) -> None:
    pytest.importorskip("scipy")

    design = generate_standard_normal_design(7, 4, method, seed=42)

    assert design.shape == (7, 4)
    assert np.isfinite(design).all()
