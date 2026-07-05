"""Local refinement of frozen-orbit candidates (roadmap R22).

Given a screened candidate's orbital elements, refine them by minimizing the
frozen score under hard feasibility constraints. The objective is injected
(propagate -> metrics -> ``frozen_score``), so this module stays independent of
the propagation backend: unit tests use analytic objectives, the pipeline uses
its screening/validation propagators.

Constraint handling is by penalty: any evaluation outside the bounds or
violating a hard constraint returns ``+inf``, so both supported optimizers
(Nelder-Mead, differential evolution) treat it as infeasible. Refined elements
are *candidates only*: ``validation_status`` always starts as
``requires_classical_sh_validation`` (R21 language rule).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

ELEMENT_NAMES = ("a_km", "e", "i_deg", "raan_deg", "argp_deg", "ta_deg")

OPTIMIZER_NELDER_MEAD = "nelder_mead"
OPTIMIZER_DIFFERENTIAL_EVOLUTION = "differential_evolution"
_OPTIMIZERS = (OPTIMIZER_NELDER_MEAD, OPTIMIZER_DIFFERENTIAL_EVOLUTION)

REFINEMENT_VALIDATION_STATUS = "requires_classical_sh_validation"


@dataclass(frozen=True, slots=True)
class RefinementBounds:
    """Element-space box bounds for refinement, ``{name: (lo, hi)}`` ordered
    as :data:`ELEMENT_NAMES`."""

    a_km: tuple[float, float]
    e: tuple[float, float]
    i_deg: tuple[float, float]
    raan_deg: tuple[float, float]
    argp_deg: tuple[float, float]
    ta_deg: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ELEMENT_NAMES:
            lo, hi = getattr(self, name)
            if not (np.isfinite(lo) and np.isfinite(hi)) or float(lo) > float(hi):
                raise ValueError(f"bounds for {name} must satisfy lo <= hi, got ({lo}, {hi})")

    def as_pairs(self) -> list[tuple[float, float]]:
        return [tuple(map(float, getattr(self, name))) for name in ELEMENT_NAMES]

    def contains(self, x: np.ndarray) -> bool:
        pairs = self.as_pairs()
        return all(pairs[k][0] <= float(x[k]) <= pairs[k][1] for k in range(len(pairs)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    """Optimizer + constraint configuration for one refinement run."""

    bounds: RefinementBounds
    optimizer: str = OPTIMIZER_NELDER_MEAD
    max_iterations: int = 200
    seed: int = 0
    # Hard feasibility constraints (checked by the pipeline's objective too;
    # duplicated here so a misbehaving objective cannot smuggle infeasible
    # elements through the optimizer).
    e_max: float = 0.5
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {_OPTIMIZERS}, got {self.optimizer!r}")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1")
        if not np.isfinite(self.e_max) or self.e_max <= 0.0:
            raise ValueError("e_max must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["bounds"] = self.bounds.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class RefinementResult:
    """Outcome of one candidate refinement."""

    original_elements: dict[str, float]
    refined_elements: dict[str, float]
    original_score: float
    refined_score: float
    improved: bool
    validation_status: str
    optimizer_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _elements_dict(x: np.ndarray) -> dict[str, float]:
    return {name: float(x[k]) for k, name in enumerate(ELEMENT_NAMES)}


def refine_candidate(
    elements0: dict[str, float] | np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    config: RefinementConfig,
) -> RefinementResult:
    """Minimize ``score_fn`` over the element box starting from ``elements0``.

    ``score_fn`` receives a ``(6,)`` element vector ordered as
    :data:`ELEMENT_NAMES` and returns a lower-is-better finite score, or
    ``+inf`` for infeasible/unstable evaluations (impact, escape, domain exit,
    perilune below the safety floor — the pipeline's objective is responsible
    for those propagation-level checks; see R27).
    """
    if isinstance(elements0, dict):
        x0 = np.array([float(elements0[name]) for name in ELEMENT_NAMES], dtype=np.float64)
    else:
        x0 = np.asarray(elements0, dtype=np.float64).reshape(-1)
        if x0.size != len(ELEMENT_NAMES):
            raise ValueError(f"elements0 must have {len(ELEMENT_NAMES)} entries, got {x0.size}")
    if not config.bounds.contains(x0):
        raise ValueError("initial elements lie outside the refinement bounds")

    n_evals = 0

    def _objective(x: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)) or not config.bounds.contains(x):
            return float("inf")
        if float(x[1]) > float(config.e_max):
            return float("inf")
        value = float(score_fn(x))
        return value if np.isfinite(value) else float("inf")

    original_score = _objective(x0)

    from scipy import optimize

    if config.optimizer == OPTIMIZER_NELDER_MEAD:
        res = optimize.minimize(
            _objective,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": int(config.max_iterations),
                "xatol": 1e-6,
                "fatol": 1e-9,
                **dict(config.extra_options),
            },
        )
        converged = bool(res.success)
        message = str(res.message)
        x_best = np.asarray(res.x, dtype=np.float64)
        f_best = float(res.fun)
    else:
        res = optimize.differential_evolution(
            _objective,
            bounds=config.bounds.as_pairs(),
            x0=x0,
            seed=int(config.seed),
            maxiter=int(config.max_iterations),
            polish=False,
            **dict(config.extra_options),
        )
        converged = bool(res.success)
        message = str(res.message)
        x_best = np.asarray(res.x, dtype=np.float64)
        f_best = float(res.fun)

    # Never return something worse than the starting point.
    if not np.isfinite(f_best) or f_best > original_score:
        x_best = x0
        f_best = original_score

    return RefinementResult(
        original_elements=_elements_dict(x0),
        refined_elements=_elements_dict(x_best),
        original_score=float(original_score),
        refined_score=float(f_best),
        improved=bool(f_best < original_score),
        validation_status=REFINEMENT_VALIDATION_STATUS,
        optimizer_metadata={
            "optimizer": config.optimizer,
            "n_evaluations": int(n_evals),
            "max_iterations": int(config.max_iterations),
            "converged": converged,
            "message": message,
            "seed": int(config.seed),
        },
    )


__all__ = [
    "ELEMENT_NAMES",
    "OPTIMIZER_DIFFERENTIAL_EVOLUTION",
    "OPTIMIZER_NELDER_MEAD",
    "REFINEMENT_VALIDATION_STATUS",
    "RefinementBounds",
    "RefinementConfig",
    "RefinementResult",
    "refine_candidate",
]
