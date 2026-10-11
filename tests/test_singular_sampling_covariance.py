"""A positive-semidefinite but singular covariance must be sampleable.

``__post_init__`` accepts any PSD ``covariance_6x6`` (eigenvalues >= -1e-10
relative), but ``cholesky_factor`` added a 1e-30 jitter and called
``np.linalg.cholesky``: lost to rounding next to 1e6 m^2 entries, so a
rank-deficient covariance (perfectly correlated components) passed validation
and then raised ``LinAlgError`` when the batch run sampled it.
"""

from __future__ import annotations

import numpy as np

from lunaris.batch.sampling import sample_initial_states
from lunaris.common.batch_defs import StateUncertainty


def _singular_cov() -> np.ndarray:
    s = np.array([1.0e3, 1.0e3, 0.0, 0.1, -0.1, 0.0])  # x == y, vx == -vy (rank 1)
    cov = np.outer(s, s)
    cov[2, 2] = 500.0**2  # independent z: rank 2 overall
    return cov


def test_singular_psd_covariance_factor_reproduces_the_covariance() -> None:
    spec = StateUncertainty(covariance_6x6=_singular_cov())
    L = spec.cholesky_factor()
    np.testing.assert_allclose(L @ L.T, _singular_cov(), rtol=0, atol=1e-6)


def test_singular_psd_covariance_samples_have_that_covariance() -> None:
    spec = StateUncertainty(covariance_6x6=_singular_cov())
    y = sample_initial_states(np.zeros(6), spec, 20000, np.random.default_rng(3))
    d = y - y.mean(axis=0)
    # The degenerate directions hold for every sample (x == y, vx == -vy) up to the
    # float64 eigen-decomposition floor: sqrt(eps * 1e6 m^2) ~ 1e-5 m on a 1 km sigma.
    np.testing.assert_allclose(d[:, 0], d[:, 1], atol=1e-3)
    np.testing.assert_allclose(d[:, 3], -d[:, 4], atol=1e-6)
    assert abs(d[:, 0].std() / 1.0e3 - 1.0) < 0.03


def test_positive_definite_covariance_keeps_the_cholesky_factor() -> None:
    cov = np.diag([1.0e6, 2.0e6, 3.0e6, 1.0, 2.0, 3.0])
    cov[0, 1] = cov[1, 0] = 5.0e5
    L = StateUncertainty(covariance_6x6=cov).cholesky_factor()
    np.testing.assert_array_equal(L, np.linalg.cholesky(cov + np.eye(6) * 1e-30))
