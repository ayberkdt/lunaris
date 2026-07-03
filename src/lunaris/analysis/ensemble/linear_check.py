"""Linear (STM) covariance cross-check for propagated ensembles.

Validates the ensemble sample covariance against linear covariance
propagation ``P(t) = Φ(t) P₀ Φ(t)ᵀ`` in the small-dispersion regime, where the
two must agree up to sampling error. The state-transition matrices are built by
central finite differences of an arbitrary propagation callable, so the check
is propagator-agnostic (analytic two-body, CPU SH dynamics, or a surrogate
backend) and never touches the propagator internals.

Agreement in the linear regime supports the ellipsoid figures; the epoch where
the two histories diverge is itself a result (the onset of non-linearity for
the given dispersion), not a failure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from lunaris.common.type_defs import F64Array

from .statistics import propagate_covariance_linear

# propagate_fn(y0: (6,)) -> (T, 6) trajectory on a fixed shared time grid.
PropagateFn = Callable[[F64Array], F64Array]

#: Default central-difference steps: [m, m, m, m/s, m/s, m/s]. Sized for lunar
#: orbital scales (r ~ 1.8e6 m, v ~ 1.7e3 m/s): large enough to dominate
#: float64 truncation in the propagator, small enough to stay in the linear
#: regime for the small dispersions this check targets.
DEFAULT_FD_EPS = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3], dtype=np.float64)


def finite_difference_stm(
    propagate_fn: PropagateFn,
    y0: F64Array,
    *,
    eps: F64Array | None = None,
) -> F64Array:
    """State-transition matrices Φ(t₀→tₖ) by central finite differences.

    Runs ``propagate_fn`` twelve times (±eps on each of the 6 state
    components); column ``i`` of ``Φ(t)`` is the central difference of the
    trajectory with respect to ``y0[i]``.

    Returns
    -------
    Phi : (T, 6, 6)
    """
    y0 = np.asarray(y0, dtype=np.float64).reshape(6)
    step = DEFAULT_FD_EPS if eps is None else np.asarray(eps, dtype=np.float64).reshape(6)
    if np.any(step <= 0.0):
        raise ValueError("finite-difference steps must all be positive")

    columns: list[F64Array] = []
    n_epochs: int | None = None
    for i in range(6):
        dy = np.zeros(6, dtype=np.float64)
        dy[i] = step[i]
        y_plus = np.asarray(propagate_fn(y0 + dy), dtype=np.float64)
        y_minus = np.asarray(propagate_fn(y0 - dy), dtype=np.float64)
        if y_plus.ndim != 2 or y_plus.shape[1] != 6 or y_plus.shape != y_minus.shape:
            raise ValueError(
                f"propagate_fn must return (T, 6) on a fixed grid; got {y_plus.shape} / {y_minus.shape}"
            )
        if n_epochs is None:
            n_epochs = int(y_plus.shape[0])
        elif int(y_plus.shape[0]) != n_epochs:
            raise ValueError("propagate_fn returned inconsistent epoch counts across FD runs")
        columns.append((y_plus - y_minus) / (2.0 * step[i]))

    assert n_epochs is not None
    Phi = np.empty((n_epochs, 6, 6), dtype=np.float64)
    for i, col in enumerate(columns):
        Phi[:, :, i] = col
    return Phi


def linear_covariance_history(
    propagate_fn: PropagateFn,
    y0: F64Array,
    P0: F64Array,
    *,
    eps: F64Array | None = None,
) -> F64Array:
    """``P(t) = Φ(t) P₀ Φ(t)ᵀ`` with finite-difference STMs. Returns (T, 6, 6)."""
    P0 = np.asarray(P0, dtype=np.float64)
    if P0.shape != (6, 6):
        raise ValueError(f"P0 must be (6, 6), got {P0.shape}")
    Phi = finite_difference_stm(propagate_fn, y0, eps=eps)
    return propagate_covariance_linear(P0, Phi)


def compare_covariance_histories(
    P_lin: F64Array,
    P_ens: F64Array,
) -> dict[str, Any]:
    """Per-epoch agreement metrics between two (T, 6, 6) covariance histories.

    Returns
    -------
    dict with:
    - ``frobenius_rel_diff``: (T,) ‖P_ens − P_lin‖_F / ‖P_lin‖_F
    - ``pos_eig_ratio``: (T, 3) ensemble/linear position-block eigenvalue ratios
      (ascending pairing)
    - ``max_frobenius_rel_diff`` / ``median_frobenius_rel_diff``: scalars
    - ``pos_eig_ratio_range``: (min, max) over all epochs/axes
    """
    P_lin = np.asarray(P_lin, dtype=np.float64)
    P_ens = np.asarray(P_ens, dtype=np.float64)
    if P_lin.shape != P_ens.shape or P_lin.ndim != 3 or P_lin.shape[1:] != (6, 6):
        raise ValueError(
            f"covariance histories must both be (T, 6, 6); got {P_lin.shape} vs {P_ens.shape}"
        )

    diff_norm = np.linalg.norm(P_ens - P_lin, axis=(1, 2))
    lin_norm = np.linalg.norm(P_lin, axis=(1, 2))
    frob = diff_norm / np.maximum(lin_norm, 1e-300)

    def _pos_eigs(P: F64Array) -> F64Array:
        block = P[:, :3, :3]
        sym = 0.5 * (block + np.transpose(block, (0, 2, 1)))
        return np.linalg.eigvalsh(sym)  # (T, 3) ascending

    eig_lin = _pos_eigs(P_lin)
    eig_ens = _pos_eigs(P_ens)
    ratio = eig_ens / np.maximum(eig_lin, 1e-300)

    return {
        "frobenius_rel_diff": frob,
        "pos_eig_ratio": ratio,
        "max_frobenius_rel_diff": float(np.max(frob)),
        "median_frobenius_rel_diff": float(np.median(frob)),
        "pos_eig_ratio_range": (float(np.min(ratio)), float(np.max(ratio))),
    }


__all__ = [
    "DEFAULT_FD_EPS",
    "PropagateFn",
    "compare_covariance_histories",
    "finite_difference_stm",
    "linear_covariance_history",
]
