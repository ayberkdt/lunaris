"""Sampling designs and perturbation draws for batch propagation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lunaris.common.batch_defs import BATCH_SAMPLING_METHODS, StateUncertainty
from lunaris.common.type_defs import F64Array


def _sobol_size_note(method: str, n_samples: int) -> str:
    if str(method).startswith("sobol") and n_samples > 0:
        # The deterministic Sobol design omits its all-zero first point before
        # the inverse-normal transform, so it needs one extra base-2 point.
        required = int(n_samples) + (1 if method == "sobol" else 0)
        power = 1 << int(math.ceil(math.log2(max(1, required))))
        if power != int(n_samples):
            discarded = " and discarded the all-zero first point" if method == "sobol" else ""
            return (
                f"{method} generated {power} base-2 design points{discarded}, then kept "
                f"{int(n_samples)}."
            )
    return ""


def generate_standard_normal_design(
    n_samples: int,
    n_dim: int,
    method: str,
    seed: int,
    rng: np.random.Generator | None = None,
) -> F64Array:
    """
    Generate standardized normal samples for ensemble propagation.

    ``random`` is the classical Monte Carlo draw. Space-filling
    methods generate unit-hypercube designs and transform them with the inverse
    normal CDF so the existing covariance machinery can be reused.
    """

    method = str(method or "random")
    if method not in BATCH_SAMPLING_METHODS:
        raise ValueError(
            "sampling_method must be one of: "
            + ", ".join(repr(item) for item in BATCH_SAMPLING_METHODS)
            + f". Got {method!r}"
        )
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if n_dim <= 0:
        raise ValueError(f"n_dim must be > 0, got {n_dim}")

    if method == "random":
        active_rng = rng if rng is not None else np.random.default_rng(int(seed))
        return np.ascontiguousarray(
            active_rng.standard_normal((int(n_samples), int(n_dim))),
            dtype=np.float64,
        )

    from scipy import special
    from scipy.stats import qmc

    if method == "lhs":
        unit = qmc.LatinHypercube(d=int(n_dim), seed=int(seed)).random(int(n_samples))
    else:
        scramble = method == "sobol_scrambled"
        sampler = qmc.Sobol(
            d=int(n_dim),
            scramble=scramble,
            seed=int(seed) if scramble else None,
        )
        # The first point of an unscrambled Sobol sequence is exactly zero in
        # every dimension. Mapping it through ndtri would make one artificial
        # joint ~-8 sigma draw after endpoint clipping. Preserve the useful,
        # deterministic Sobol option while discarding that pathological point.
        # Scrambled Sobol has no fixed all-zero first point and retains the
        # standard first-N base-2 prefix behavior.
        required = int(n_samples) + (1 if not scramble else 0)
        m = int(math.ceil(math.log2(max(1, required))))
        unit = sampler.random_base2(m=m)
        if not scramble:
            unit = unit[1 : int(n_samples) + 1]
        else:
            unit = unit[: int(n_samples)]

    eps = np.finfo(np.float64).eps
    clipped = np.clip(np.asarray(unit, dtype=np.float64), eps, 1.0 - eps)
    return np.ascontiguousarray(special.ndtri(clipped), dtype=np.float64)


def sample_initial_states(
    nominal_state: F64Array,  # (6,) [x,y,z,vx,vy,vz]
    uncertainty: StateUncertainty,
    n_samples: int,
    rng: np.random.Generator,
    *,
    sampling_method: str = "random",
    seed: int = 0,
    standard_normal_samples: F64Array | None = None,
) -> F64Array:
    """
    Draw N Gaussian samples around the nominal state.

    ``sampling_method`` can be ``random`` (classical Monte Carlo), ``lhs``,
    ``sobol``, or ``sobol_scrambled``. Non-random methods are transformed into
    standard-normal samples before the covariance factor is applied.

    Returns
    ----------
    Y0 : (N, 6) float64 perturbed initial states
    """
    L = uncertainty.cholesky_factor()  # (6, 6) lower-triangular
    if standard_normal_samples is None:
        Z = generate_standard_normal_design(n_samples, 6, sampling_method, seed, rng)
    else:
        Z = np.asarray(standard_normal_samples, dtype=np.float64)
        if Z.shape != (int(n_samples), 6):
            raise ValueError(f"standard_normal_samples must be ({n_samples}, 6), got {Z.shape}")
    delta = Z @ L.T  # (N, 6) perturbation
    return np.ascontiguousarray(nominal_state[None, :] + delta, dtype=np.float64)


def sample_spacecraft_props(
    nominal_mass: float,
    nominal_area: float,
    nominal_cd: float,
    nominal_cr: float,
    uncertainty: Any,  # SpacecraftUncertainty
    n_samples: int,
    rng: np.random.Generator,
    *,
    sampling_method: str = "random",
    seed: int = 0,
    standard_normal_samples: F64Array | None = None,
) -> F64Array:
    """
    Sample spacecraft physical properties from lower-truncated Gaussians.

    Each positive physical property is conditioned to be at or above a small
    positive floor (one percent of its nominal value). This is a true
    truncated-normal transform, not a clipped or winsorized Gaussian: samples
    below the floor are never accumulated into a point mass at that floor.

    Returns
    ----------
    sc_samples : (N, 4) float64 - columns [mass_kg, area_m2, cd, cr]
    """
    sc = np.zeros((n_samples, 4), dtype=np.float64)
    if standard_normal_samples is not None:
        z_sc = np.asarray(standard_normal_samples, dtype=np.float64)
        if z_sc.shape != (int(n_samples), 4):
            raise ValueError(f"standard_normal_samples must be ({n_samples}, 4), got {z_sc.shape}")
    elif str(sampling_method or "random") == "random":
        z_sc = None
    else:
        z_sc = generate_standard_normal_design(n_samples, 4, sampling_method, seed)

    def _trunc_normal(mu: float, sigma: float, col: int) -> np.ndarray:
        """Draw a Gaussian conditioned on a small, positive lower bound."""
        if sigma <= 0.0:
            return np.full(n_samples, mu, dtype=np.float64)

        from scipy import special

        lower = 0.01 * max(mu, 1e-30)
        cdf_lower = float(special.ndtr((lower - mu) / sigma))
        if z_sc is None:
            unit = rng.random(n_samples)
        else:
            # QMC designs arrive as standard-normal coordinates. Map them
            # back to the unit interval before applying the conditional CDF.
            unit = special.ndtr(z_sc[:, col])

        eps = np.finfo(np.float64).eps
        unit = np.clip(np.asarray(unit, dtype=np.float64), eps, 1.0 - eps)
        conditional_cdf = cdf_lower + (1.0 - cdf_lower) * unit
        conditional_cdf = np.clip(
            conditional_cdf,
            np.nextafter(cdf_lower, 1.0),
            np.nextafter(1.0, 0.0),
        )
        return np.ascontiguousarray(
            mu + sigma * special.ndtri(conditional_cdf),
            dtype=np.float64,
        )

    sc[:, 0] = _trunc_normal(nominal_mass, float(getattr(uncertainty, "sigma_mass_kg", 0.0)), 0)
    sc[:, 1] = _trunc_normal(nominal_area, float(getattr(uncertainty, "sigma_area_m2", 0.0)), 1)
    sc[:, 2] = _trunc_normal(nominal_cd, float(getattr(uncertainty, "sigma_cd", 0.0)), 2)
    sc[:, 3] = _trunc_normal(nominal_cr, float(getattr(uncertainty, "sigma_cr", 0.0)), 3)

    return sc


__all__ = [
    "_sobol_size_note",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
]
