from __future__ import annotations

import numpy as np
import pytest

from lunaris.batch.sampling import (
    generate_standard_normal_design,
    sample_spacecraft_props,
)


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


def test_unscrambled_sobol_discards_the_pathological_all_zero_point() -> None:
    """The inverse-normal design must not contain the all-coordinate -8σ point."""
    pytest.importorskip("scipy")

    design = generate_standard_normal_design(8, 10, "sobol", seed=42)

    # After the all-zero Sobol point is dropped, the next deterministic point is
    # the unit-hypercube centre. This both preserves determinism and proves the
    # first propagated sample is not an artificial joint tail event.
    np.testing.assert_allclose(design[0], np.zeros(10), atol=1e-15)
    assert float(np.min(design)) > -8.0


def test_spacecraft_sampling_is_truncated_not_clipped() -> None:
    """A high-uncertainty draw has no artificial point mass at the lower floor."""
    pytest.importorskip("scipy")

    n_samples = 4096
    uncertainty = type(
        "Uncertainty",
        (),
        {
            "sigma_mass_kg": 5.0,
            "sigma_area_m2": 5.0,
            "sigma_cd": 5.0,
            "sigma_cr": 5.0,
        },
    )()
    samples = sample_spacecraft_props(
        1.0,
        1.0,
        1.0,
        1.0,
        uncertainty,
        n_samples,
        np.random.default_rng(7),
    )

    lower = 0.01
    assert np.all(samples > lower)
    assert not np.any(samples == lower)
