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


def characteristic_state_scales(y0: F64Array) -> F64Array:
    """Return blockwise SI scales ``[R,R,R,V,V,V]`` for nondimensionalization."""
    state = np.asarray(y0, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(state)):
        raise ValueError("y0 must contain only finite values")
    r_scale = max(float(np.linalg.norm(state[:3])), 1.0)
    v_scale = max(float(np.linalg.norm(state[3:])), 1.0e-3)
    return np.array([r_scale] * 3 + [v_scale] * 3, dtype=np.float64)


def resolve_fd_steps(
    y0: F64Array,
    *,
    eps: F64Array | None = None,
    eps_mode: str = "absolute",
    rel_step: float = 1.0e-6,
    state_scales: F64Array | None = None,
) -> F64Array:
    """Resolve finite-difference steps without component-axis bias.

    ``absolute`` preserves the historical defaults. ``relative`` uses one
    characteristic scale for all three position axes and one for all three
    velocity axes; this remains rotationally neutral unlike ``rel*abs(y0_i)``,
    which collapses steps on components that happen to be zero.
    """
    state = np.asarray(y0, dtype=np.float64).reshape(6)
    mode = str(eps_mode).strip().lower()
    if mode not in {"absolute", "relative"}:
        raise ValueError("eps_mode must be 'absolute' or 'relative'")
    if eps is not None:
        step = np.asarray(eps, dtype=np.float64).reshape(6).copy()
    elif mode == "absolute":
        step = DEFAULT_FD_EPS.copy()
    else:
        rel = float(rel_step)
        if not np.isfinite(rel) or rel <= 0.0:
            raise ValueError("rel_step must be finite and positive")
        scales = (
            characteristic_state_scales(state)
            if state_scales is None
            else np.asarray(state_scales, dtype=np.float64).reshape(6)
        )
        step = np.maximum(DEFAULT_FD_EPS, rel * scales)
    if not np.all(np.isfinite(step)) or np.any(step <= 0.0):
        raise ValueError("finite-difference steps must all be finite and positive")
    return step


def finite_difference_stm(
    propagate_fn: PropagateFn,
    y0: F64Array,
    *,
    eps: F64Array | None = None,
    eps_mode: str = "absolute",
    rel_step: float = 1.0e-6,
    state_scales: F64Array | None = None,
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
    step = resolve_fd_steps(
        y0,
        eps=eps,
        eps_mode=eps_mode,
        rel_step=rel_step,
        state_scales=state_scales,
    )

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
    eps_mode: str = "absolute",
    rel_step: float = 1.0e-6,
    state_scales: F64Array | None = None,
) -> F64Array:
    """``P(t) = Φ(t) P₀ Φ(t)ᵀ`` with finite-difference STMs. Returns (T, 6, 6)."""
    P0 = np.asarray(P0, dtype=np.float64)
    if P0.shape != (6, 6):
        raise ValueError(f"P0 must be (6, 6), got {P0.shape}")
    Phi = finite_difference_stm(
        propagate_fn,
        y0,
        eps=eps,
        eps_mode=eps_mode,
        rel_step=rel_step,
        state_scales=state_scales,
    )
    return propagate_covariance_linear(P0, Phi)


def _nondimensionalize_stm(Phi: F64Array, state_scales: F64Array) -> F64Array:
    scales = np.asarray(state_scales, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("state_scales must contain finite positive values")
    if not (
        np.allclose(scales[:3], scales[0], rtol=0.0, atol=0.0)
        and np.allclose(scales[3:], scales[3], rtol=0.0, atol=0.0)
    ):
        raise ValueError("symplectic metrics require blockwise [R,R,R,V,V,V] scales")
    return Phi * scales[np.newaxis, np.newaxis, :] / scales[np.newaxis, :, np.newaxis]


def stm_quality(
    Phi: F64Array,
    *,
    symplectic_applicable: bool,
    state_scales: F64Array | None = None,
    Phi_half_step: F64Array | None = None,
) -> dict[str, Any]:
    """Return dimensionless STM quality diagnostics.

    Symplecticity is evaluated only after converting ``(r,v)`` to blockwise
    dimensionless coordinates. ``symplectic_applicable`` must describe the
    complete smooth Hamiltonian force stack, not merely the gravity provider.
    """
    phi = np.asarray(Phi, dtype=np.float64)
    if phi.ndim != 3 or phi.shape[1:] != (6, 6) or phi.shape[0] < 1:
        raise ValueError(f"Phi must be (T, 6, 6), got {phi.shape}")
    if not np.all(np.isfinite(phi)):
        raise ValueError("Phi must contain only finite values")

    result: dict[str, Any] = {
        "symplectic_applicable": bool(symplectic_applicable),
        "symplecticity_status": "not_applicable",
        "symplecticity_error": None,
        "det_deviation": None,
        "eps_halving_rel_diff": None,
    }
    phi_nd: F64Array | None = None
    if symplectic_applicable:
        if state_scales is None:
            raise ValueError("state_scales are required for dimensionless symplectic metrics")
        phi_nd = _nondimensionalize_stm(phi, state_scales)
        eye3 = np.eye(3, dtype=np.float64)
        zero3 = np.zeros((3, 3), dtype=np.float64)
        J = np.block([[zero3, eye3], [-eye3, zero3]])
        residual = np.transpose(phi_nd, (0, 2, 1)) @ J @ phi_nd - J
        result.update(
            {
                "symplecticity_status": "evaluated_dimensionless",
                "symplecticity_error": float(
                    np.max(np.linalg.norm(residual, axis=(1, 2)) / np.linalg.norm(J))
                ),
                "det_deviation": float(np.max(np.abs(np.linalg.det(phi_nd) - 1.0))),
            }
        )

    if Phi_half_step is not None:
        half = np.asarray(Phi_half_step, dtype=np.float64)
        if half.shape != phi.shape or not np.all(np.isfinite(half)):
            raise ValueError("Phi_half_step must be finite and have the same shape as Phi")
        if state_scales is None:
            raise ValueError("state_scales are required for a dimensionless eps-halving check")
        base = phi_nd if phi_nd is not None else _nondimensionalize_stm(phi, state_scales)
        half_nd = _nondimensionalize_stm(half, state_scales)
        denom = np.maximum(np.linalg.norm(half_nd, axis=(1, 2)), 1.0e-300)
        result["eps_halving_rel_diff"] = float(
            np.max(np.linalg.norm(base - half_nd, axis=(1, 2)) / denom)
        )
    return result


def finite_difference_stm_with_quality(
    propagate_fn: PropagateFn,
    y0: F64Array,
    *,
    eps: F64Array | None = None,
    eps_mode: str = "relative",
    rel_step: float = 1.0e-6,
    state_scales: F64Array | None = None,
    symplectic_applicable: bool = False,
    check_eps_halving: bool = True,
) -> tuple[F64Array, dict[str, Any]]:
    """Build an FD STM and optionally spend twelve extra runs on step convergence."""
    scales = (
        characteristic_state_scales(y0)
        if state_scales is None
        else np.asarray(state_scales, dtype=np.float64).reshape(6)
    )
    steps = resolve_fd_steps(
        y0,
        eps=eps,
        eps_mode=eps_mode,
        rel_step=rel_step,
        state_scales=scales,
    )
    phi = finite_difference_stm(propagate_fn, y0, eps=steps)
    phi_half = finite_difference_stm(propagate_fn, y0, eps=0.5 * steps) if check_eps_halving else None
    quality = stm_quality(
        phi,
        symplectic_applicable=symplectic_applicable,
        state_scales=scales,
        Phi_half_step=phi_half,
    )
    quality["fd_steps"] = steps.tolist()
    quality["state_scales"] = scales.tolist()
    return phi, quality


def compare_covariance_histories(
    P_lin: F64Array,
    P_ens: F64Array,
    *,
    stm_quality_metrics: dict[str, Any] | None = None,
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

    result = {
        "frobenius_rel_diff": frob,
        "pos_eig_ratio": ratio,
        "max_frobenius_rel_diff": float(np.max(frob)),
        "median_frobenius_rel_diff": float(np.median(frob)),
        "pos_eig_ratio_range": (float(np.min(ratio)), float(np.max(ratio))),
    }
    if stm_quality_metrics is not None:
        result["stm_quality"] = dict(stm_quality_metrics)
    return result


__all__ = [
    "DEFAULT_FD_EPS",
    "PropagateFn",
    "characteristic_state_scales",
    "compare_covariance_histories",
    "finite_difference_stm",
    "finite_difference_stm_with_quality",
    "linear_covariance_history",
    "resolve_fd_steps",
    "stm_quality",
]
