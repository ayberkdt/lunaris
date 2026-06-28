"""Pure-Python cost / accuracy estimates and validation for the solver UI.

No Qt import, so this is unit-testable in isolation. The mission-propagation page
uses it to give the operator immediate feedback on what a chosen step size or
tolerance implies (number of steps, force-evaluation budget, accuracy band) and
to flag obviously unsafe inputs before launch.

The per-step force-evaluation counts mirror the actual implementation in
``lunaris.core.propagation.propagator`` (the Yoshida compositions reuse no evaluations across
their velocity-Verlet sub-steps, so the counts grow with the number of
sub-steps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lunaris.ui.core.integrator_catalog import IntegratorSpec

# Force/acceleration evaluations per fixed step (matches propagator.py).
#   VV = 2 (a0, a1) · Yoshida-k = 3^levels sub-steps × 2 · PEFRL = 4 kicks
#   RKN4 = 3 (k2≡k3) · RK4 = 4 · RK8 = Σ(n_i+1) over (2,4,6,8) = 24
_EVALS_PER_STEP: dict[str, int] = {
    "VV": 2,
    "PEFRL": 4,
    "YOSHIDA4": 6,
    "YOSHIDA6": 18,
    "YOSHIDA8": 54,
    "RKN4": 3,
    "RK4": 4,
    "RK8": 24,
}

# Below this step count a run is almost certainly under-resolved.
_MIN_REASONABLE_STEPS = 20
# Adaptive relative-tolerance sanity band.
_RTOL_TOO_LOOSE = 1e-3
_RTOL_TOO_TIGHT = 1e-14


def evals_per_step(key: str) -> int | None:
    """Force evaluations per fixed step for a method key, or ``None`` if adaptive."""
    return _EVALS_PER_STEP.get(str(key).upper())


def _coerce_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parsed = float(text)
        if not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def accuracy_label(rtol: object) -> str:
    """Map a relative tolerance to a short qualitative accuracy band."""
    value = _coerce_float(rtol)
    if value is None or value <= 0.0:
        return "—"
    if value <= 1e-11:
        return "Very high accuracy"
    if value <= 1e-9:
        return "High accuracy"
    if value <= 1e-7:
        return "Balanced"
    if value <= 1e-4:
        return "Coarse"
    return "Very coarse"


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Result of :func:`estimate_fixed_step_cost`."""

    mode: str                 # "adaptive" | "auto" | "fixed" | "unknown"
    n_steps: int | None
    total_evals: int | None
    summary: str


def _format_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")  # thin-ish grouping that reads well in the UI


def estimate_fixed_step_cost(
    spec: IntegratorSpec | None,
    duration_s: float | None,
    step_s: float | None,
) -> CostEstimate:
    """Estimate step count and force-evaluation budget for the chosen method."""
    if spec is None:
        return CostEstimate("unknown", None, None, "Select an integration method.")

    if spec.is_adaptive:
        return CostEstimate(
            "adaptive", None, None,
            "Adaptive step — the solver controls cost through the tolerance, not a "
            "fixed step.",
        )

    dur = _coerce_float(duration_s)
    step = _coerce_float(step_s)

    if step is None or step <= 0.0:
        return CostEstimate(
            "auto", None, None,
            "Auto step — a Nyquist-safe step is derived from the gravity field and "
            "orbit at launch.",
        )

    if dur is None or dur <= 0.0:
        return CostEstimate("unknown", None, None, "Set a positive duration to estimate cost.")

    n_steps = max(1, int(math.ceil(dur / step)))
    eps = evals_per_step(spec.key) or 1
    total = n_steps * eps
    summary = (
        f"≈ {_format_int(n_steps)} steps over the run · "
        f"~{_format_int(total)} force evaluations ({eps}/step)."
    )
    return CostEstimate("fixed", n_steps, total, summary)


def validate_solver_inputs(
    spec: IntegratorSpec | None,
    *,
    duration_s: float | None,
    step_s: float | None,
    rtol: object = None,
) -> list[tuple[str, str]]:
    """Return ``[(severity, message), ...]`` for unsafe / questionable inputs.

    ``severity`` is ``"error"`` (block-worthy) or ``"warning"`` (proceed with
    caution). An empty list means the inputs look sane.
    """
    if spec is None:
        return []

    issues: list[tuple[str, str]] = []

    if spec.is_adaptive:
        r = _coerce_float(rtol)
        if rtol is not None and str(rtol).strip() and r is None:
            issues.append(("error", "Relative tolerance is not a valid number."))
        elif r is not None:
            if r <= 0.0:
                issues.append(("error", "Relative tolerance must be positive."))
            elif r > _RTOL_TOO_LOOSE:
                issues.append((
                    "warning",
                    f"Relative tolerance {r:g} is very loose (> {_RTOL_TOO_LOOSE:g}); "
                    "the trajectory may be inaccurate.",
                ))
            elif r < _RTOL_TOO_TIGHT:
                issues.append((
                    "warning",
                    f"Relative tolerance {r:g} is near machine precision; the solver "
                    "may stall or waste steps.",
                ))
        return issues

    # Fixed-step family.
    step = _coerce_float(step_s)
    dur = _coerce_float(duration_s)
    if step_s is not None and str(step_s).strip() and step is None:
        issues.append(("error", "Step size is not a valid number."))
    elif step is not None:
        if step <= 0.0:
            issues.append(("error", "Step size must be positive."))
        elif dur is not None and dur > 0.0:
            if step >= dur:
                issues.append((
                    "warning",
                    "Step size is larger than the whole run — that is a single step. "
                    "Reduce the step size.",
                ))
            else:
                n_steps = int(math.ceil(dur / step))
                if n_steps < _MIN_REASONABLE_STEPS:
                    issues.append((
                        "warning",
                        f"Only {n_steps} steps over the run; the trajectory is likely "
                        "under-resolved. Use a smaller step.",
                    ))
    return issues


__all__ = [
    "CostEstimate",
    "evals_per_step",
    "accuracy_label",
    "estimate_fixed_step_cost",
    "validate_solver_inputs",
]
