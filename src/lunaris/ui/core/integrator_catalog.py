"""Presentation metadata for the propagation integrators offered in the UI.

This is a *pure-Python* catalog (no Qt import) so it can be unit-tested and
reused by any surface that needs to describe an integrator. Each entry maps a
combobox label to the backend method token plus the characteristics shown in the
mission-propagation "integrator card": family, order, step-control mode, and a
short recommended-use blurb.

Contract with the rest of the app
---------------------------------
- ``label`` is what the combobox shows. Command building and session restore
  extract the backend method by taking the first whitespace token of the label
  (e.g. ``"RK4 (Fixed-step)" -> "RK4"``), so every label must start with its
  ``key``.
- ``key`` is the canonical method string the propagator understands
  (``lunaris.core.propagation.propagator``): DOP853/RK45/RK23/RADAU/BDF/LSODA (adaptive,
  SciPy) and VV/PEFRL/YOSHIDA4/YOSHIDA6/YOSHIDA8/RKN4/RK4/RK8 (in-house
  fixed-step).
- ``family == "adaptive"`` must agree with
  ``lunaris.ui.core.solver_policy.solver_method_is_adaptive(label)`` so the
  tolerance controls and the card stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntegratorSpec:
    """Display metadata for a single integrator option."""

    key: str            # backend method token (propagator.py)
    label: str          # combobox display label (must start with ``key``)
    family: str         # "adaptive" | "symplectic" | "nystrom" | "runge_kutta"
    family_label: str   # short badge text
    badge_kind: str     # StatusBadge kind (themed)
    title: str          # full method name shown as the card subtitle
    order: str          # convergence order, e.g. "8(7)", "4", "6"
    metric_type: str    # short type for the metric row, e.g. "Symplectic"
    step_mode: str      # "Adaptive" | "Fixed"
    metric_error: str   # short local-error descriptor, e.g. "Embedded" / "None"
    recommended: str    # one-sentence recommended-use blurb
    notice_kind: str    # InlineNotice kind for the recommendation ("info"/"warning"/...)

    @property
    def is_adaptive(self) -> bool:
        return self.family == "adaptive"

    @property
    def uses_fixed_step(self) -> bool:
        return self.step_mode == "Fixed"


# Ordered by family; the UI inserts a separator between families.
INTEGRATOR_CATALOG: tuple[IntegratorSpec, ...] = (
    # --- Adaptive (SciPy solve_ivp) ----------------------------------------
    IntegratorSpec(
        key="DOP853", label="DOP853 (Adaptive)", family="adaptive",
        family_label="Adaptive", badge_kind="info",
        title="Dormand–Prince 8(7)", order="8(7)",
        metric_type="Explicit RK", step_mode="Adaptive", metric_error="Embedded",
        recommended="High-accuracy general-purpose propagation of smooth, non-stiff "
                    "dynamics. The project default when tight error control matters.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="RK45", label="RK45 (Adaptive)", family="adaptive",
        family_label="Adaptive", badge_kind="info",
        title="Dormand–Prince 5(4)", order="5(4)",
        metric_type="Explicit RK", step_mode="Adaptive", metric_error="Embedded",
        recommended="Lighter general-purpose adaptive solver. A solid default when "
                    "moderate accuracy is enough and speed matters.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="RK23", label="RK23 (Adaptive)", family="adaptive",
        family_label="Adaptive", badge_kind="info",
        title="Bogacki–Shampine 3(2)", order="3(2)",
        metric_type="Explicit RK", step_mode="Adaptive", metric_error="Embedded",
        recommended="Low-accuracy or quick exploratory runs where speed beats precision.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="RADAU", label="RADAU (Adaptive · stiff)", family="adaptive",
        family_label="Adaptive", badge_kind="info",
        title="Radau IIA (implicit)", order="5",
        metric_type="Implicit RK", step_mode="Adaptive", metric_error="Embedded",
        recommended="Stiff dynamics where explicit solvers are forced into tiny steps.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="BDF", label="BDF (Adaptive · stiff)", family="adaptive",
        family_label="Adaptive", badge_kind="info",
        title="Backward Differentiation Formulas", order="1–5",
        metric_type="Multistep", step_mode="Adaptive", metric_error="Variable",
        recommended="Stiff problems; variable-order implicit multistep integration.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="LSODA", label="LSODA (Adaptive · auto)", family="adaptive",
        family_label="Adaptive", badge_kind="info",
        title="LSODA (auto stiff / non-stiff)", order="variable",
        metric_type="Multistep", step_mode="Adaptive", metric_error="Variable",
        recommended="Use when stiffness is unknown — it switches between stiff and "
                    "non-stiff methods automatically.",
        notice_kind="info",
    ),
    # --- Symplectic (in-house fixed step) ----------------------------------
    IntegratorSpec(
        key="VV", label="VV (Symplectic)", family="symplectic",
        family_label="Symplectic", badge_kind="success",
        title="Velocity Verlet", order="2",
        metric_type="Symplectic", step_mode="Fixed", metric_error="Bounded drift",
        recommended="Low-cost, energy-stable long-duration runs in smooth, conservative "
                    "fields. Bounded drift holds only for conservative forces (gravity, "
                    "third-body, J2); enabling SRP, albedo, thermal IR or relativity voids "
                    "the symplectic guarantee — prefer RK4/DOP853 there.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="PEFRL", label="PEFRL (Symplectic)", family="symplectic",
        family_label="Symplectic", badge_kind="success",
        title="Position-Extended Forest–Ruth-Like", order="4",
        metric_type="Symplectic", step_mode="Fixed", metric_error="Bounded drift",
        recommended="Strong 4th-order symplectic default: a low error constant makes it "
                    "excellent for long, smooth arcs. Valid only for conservative forces "
                    "(gravity, third-body, J2); enabling SRP, albedo, thermal IR or "
                    "relativity voids the symplectic guarantee — prefer RK4/DOP853 there.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="YOSHIDA4", label="YOSHIDA4 (Symplectic)", family="symplectic",
        family_label="Symplectic", badge_kind="success",
        title="Yoshida (4th order)", order="4",
        metric_type="Symplectic", step_mode="Fixed", metric_error="Bounded drift",
        recommended="4th-order symplectic with bounded energy drift over many revolutions "
                    "— for conservative forces only (gravity, third-body, J2). SRP, albedo, "
                    "thermal IR or relativity void the symplectic guarantee; prefer "
                    "RK4/DOP853 there.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="YOSHIDA6", label="YOSHIDA6 (Symplectic)", family="symplectic",
        family_label="Symplectic", badge_kind="success",
        title="Yoshida (6th order)", order="6",
        metric_type="Symplectic", step_mode="Fixed", metric_error="Bounded drift",
        recommended="High-precision long-arc symplectic propagation for conservative "
                    "forces only (gravity, third-body, J2). SRP, albedo, thermal IR or "
                    "relativity void the symplectic guarantee; prefer RK4/DOP853 there.",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="YOSHIDA8", label="YOSHIDA8 (Symplectic)", family="symplectic",
        family_label="Symplectic", badge_kind="success",
        title="Yoshida (8th order)", order="8",
        metric_type="Symplectic", step_mode="Fixed", metric_error="Bounded drift",
        recommended="Maximum-precision symplectic integration for very long arcs — "
                    "conservative forces only (gravity, third-body, J2). SRP, albedo, "
                    "thermal IR or relativity void the symplectic guarantee; prefer "
                    "RK4/DOP853 there.",
        notice_kind="info",
    ),
    # --- Runge–Kutta–Nyström -----------------------------------------------
    IntegratorSpec(
        key="RKN4", label="RKN4 (Nyström)", family="nystrom",
        family_label="Nyström", badge_kind="ready",
        title="Runge–Kutta–Nyström (4th order)", order="4",
        metric_type="Nyström", step_mode="Fixed", metric_error="None",
        recommended="Efficient for position/time-only gravity. Accuracy drops when "
                    "velocity-dependent forces (drag, relativity) dominate — prefer RK4 "
                    "or a symplectic method there.",
        notice_kind="warning",
    ),
    # --- Classical fixed-step Runge–Kutta ----------------------------------
    IntegratorSpec(
        key="RK4", label="RK4 (Fixed-step)", family="runge_kutta",
        family_label="Fixed-step RK", badge_kind="completed",
        title="Classical Runge–Kutta (4th order)", order="4",
        metric_type="Explicit RK", step_mode="Fixed", metric_error="None",
        recommended="Simple, deterministic fixed-step reference. General-purpose and "
                    "supports augmented states (e.g. mass).",
        notice_kind="info",
    ),
    IntegratorSpec(
        key="RK8", label="RK8 (Fixed-step)", family="runge_kutta",
        family_label="Fixed-step RK", badge_kind="completed",
        title="8th-order extrapolation (Gragg–Bulirsch–Stoer)", order="8",
        metric_type="Extrapolation", step_mode="Fixed", metric_error="None",
        recommended="High-order fixed-step reference for tight non-symplectic accuracy "
                    "without adaptivity.",
        notice_kind="info",
    ),
)


_BY_LABEL: dict[str, IntegratorSpec] = {spec.label: spec for spec in INTEGRATOR_CATALOG}
_BY_KEY: dict[str, IntegratorSpec] = {spec.key.upper(): spec for spec in INTEGRATOR_CATALOG}


def catalog_labels() -> list[str]:
    """Return every combobox label in catalog (family) order."""
    return [spec.label for spec in INTEGRATOR_CATALOG]


def spec_for_label(label: str | None) -> IntegratorSpec | None:
    """Resolve a (possibly legacy/bare) method label to its spec.

    Matches the full label first, then falls back to the leading whitespace
    token (the backend method key), so both ``"DOP853 (Adaptive)"`` and a bare
    ``"DOP853"`` resolve correctly.
    """
    if not label:
        return None
    text = str(label).strip()
    spec = _BY_LABEL.get(text)
    if spec is not None:
        return spec
    token = text.split()[0].upper() if text.split() else ""
    return _BY_KEY.get(token)


def grouped_labels() -> list[tuple[str, list[str]]]:
    """Return ``[(family_label, [labels...]), ...]`` preserving catalog order."""
    groups: list[tuple[str, list[str]]] = []
    for spec in INTEGRATOR_CATALOG:
        if groups and groups[-1][0] == spec.family_label:
            groups[-1][1].append(spec.label)
        else:
            groups.append((spec.family_label, [spec.label]))
    return groups


__all__ = [
    "IntegratorSpec",
    "INTEGRATOR_CATALOG",
    "catalog_labels",
    "spec_for_label",
    "grouped_labels",
]
