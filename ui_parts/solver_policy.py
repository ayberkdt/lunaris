# LUNAR_SIMULATION/ui_parts/solver_policy.py
# -*- coding: utf-8 -*-
"""
Shared solver-default and normalization policy for the desktop UI.

Why this module exists
----------------------
The project exposes solver settings through multiple surfaces:
- the propagation page text inputs
- the advanced solver dialog
- session save/load
- command building and preflight validation

Historically those call sites each carried their own fallback values. That made
it easy for the UI to drift away from the backend defaults and occasionally
launch runs with overly aggressive or invalid tolerance pairs.

This module centralizes the "safe default" policy so every layer uses the same
answer when the user leaves a field blank, restores an old session, or enters an
invalid tolerance.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple


# These defaults intentionally match the stricter-but-stable backend SSOT in
# `common.type_defs.PropagatorConfig` rather than the older UI-local values.
DEFAULT_ADAPTIVE_RTOL = 1e-10
DEFAULT_ADAPTIVE_ATOL = 1e-12
DEFAULT_MAX_STEP_S = 3600.0
DEFAULT_SOLVER_METHOD = "DOP853 (Adaptive)"
LEGACY_ADAPTIVE_RTOL = 1e-12
LEGACY_ADAPTIVE_ATOL = 1e-14

_ADAPTIVE_METHOD_HINTS = (
    "ADAPTIVE",
    "DOP853",
    "RK45",
    "RK23",
    "RADAU",
    "BDF",
    "LSODA",
)


def solver_method_is_adaptive(method_label: Any) -> bool:
    """
    Return True when the selected method uses tolerance-driven step control.

    The UI may pass a friendly label such as ``"DOP853 (Adaptive)"`` while the
    backend may pass a bare method token like ``"RK45"``. A substring-based
    check keeps both forms compatible without requiring every caller to normalize
    labels first.
    """

    text = str(method_label or "").strip().upper()
    return any(token in text for token in _ADAPTIVE_METHOD_HINTS)


def coerce_positive_float(value: Any) -> Optional[float]:
    """
    Convert an arbitrary UI/session value into a strictly positive float.

    Returns ``None`` for empty, invalid, non-finite, or non-positive values so
    callers can cleanly fall back to shared defaults.
    """

    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parsed = float(text)
        if parsed <= 0.0:
            return None
        return parsed
    except Exception:
        return None


def choose_solver_tolerances(
    method_label: Any,
    *,
    rtol: Any = None,
    atol: Any = None,
) -> Tuple[float, float]:
    """
    Produce a restart-safe `(rtol, atol)` pair for the selected solver.

    Rules
    -----
    - Invalid or missing `rtol` falls back to a conservative adaptive default.
    - Invalid or missing `atol` is derived from `rtol`, but never tighter than
      the shared backend default and never looser than `rtol` itself.
    - Fixed-step methods still receive a valid pair so session restore and
      preflight validation remain internally consistent, even if the backend does
      not actively consume those values for the chosen integrator.
    """

    # The current UI ships one solver policy for all adaptive methods. If we add
    # method-specific defaults later, the branch point already exists here.
    if solver_method_is_adaptive(method_label):
        default_rtol = DEFAULT_ADAPTIVE_RTOL
        default_atol = DEFAULT_ADAPTIVE_ATOL
    else:
        default_rtol = DEFAULT_ADAPTIVE_RTOL
        default_atol = DEFAULT_ADAPTIVE_ATOL

    raw_rtol = coerce_positive_float(rtol)
    rtol_was_invalid = raw_rtol is None
    rtol_value = raw_rtol
    if rtol_value is None:
        rtol_value = default_rtol

    derived_atol = min(max(default_atol, float(rtol_value) * 1e-2), float(rtol_value))
    atol_value = coerce_positive_float(atol)
    if atol_value is None:
        atol_value = derived_atol
    else:
        # Absolute tolerance should never be looser than the relative target.
        atol_value = min(float(atol_value), float(rtol_value))

        # If the visible rtol had to fall back to a default, a very small
        # carried-over atol is usually a stale legacy value rather than fresh
        # user intent. In that case prefer the derived safe pair.
        if rtol_was_invalid and float(atol_value) < float(derived_atol):
            atol_value = derived_atol

    if atol_value <= 0.0:
        atol_value = default_atol

    return float(rtol_value), float(atol_value)


def uses_legacy_adaptive_defaults(rtol: Any, atol: Any) -> bool:
    """
    Detect the older UI default pair that tended to over-constrain adaptive runs.

    The desktop historically shipped `1e-12 / 1e-14` as its visible defaults
    even though the backend SSOT later moved to `1e-10 / 1e-12`. We only use
    this detector during session migration, not for general user-input parsing.
    """

    rtol_value = coerce_positive_float(rtol)
    atol_value = coerce_positive_float(atol)
    if rtol_value is None or atol_value is None:
        return False

    return bool(
        math.isclose(float(rtol_value), LEGACY_ADAPTIVE_RTOL, rel_tol=0.0, abs_tol=1e-18)
        and math.isclose(float(atol_value), LEGACY_ADAPTIVE_ATOL, rel_tol=0.0, abs_tol=1e-20)
    )


def choose_max_step(value: Any, *, default: Any = DEFAULT_MAX_STEP_S) -> Optional[float]:
    """
    Normalize a maximum-step value while preserving the "auto" option.

    Returning ``None`` means "let the backend choose through its Nyquist-based
    logic". Returning a positive float means the UI explicitly wants a cap.
    """

    max_step = coerce_positive_float(value)
    if max_step is not None:
        return max_step

    if default is None:
        return None
    return coerce_positive_float(default)


def normalize_solver_config_object(
    solver_cfg: Any,
    *,
    method_label: Any = DEFAULT_SOLVER_METHOD,
    upgrade_legacy_defaults: bool = False,
) -> Any:
    """
    Mutate a UI-facing solver config object in place so it stays self-consistent.

    The function intentionally works with ``Any`` because the desktop UI uses a
    lightweight mutable dataclass, while tests occasionally pass small stand-in
    objects with the same attribute names.
    """

    raw_rtol = getattr(solver_cfg, "rtol", None)
    raw_atol = getattr(solver_cfg, "atol", None)
    if upgrade_legacy_defaults and solver_method_is_adaptive(method_label):
        if uses_legacy_adaptive_defaults(raw_rtol, raw_atol):
            raw_rtol = DEFAULT_ADAPTIVE_RTOL
            raw_atol = DEFAULT_ADAPTIVE_ATOL

    rtol_value, atol_value = choose_solver_tolerances(
        method_label,
        rtol=raw_rtol,
        atol=raw_atol,
    )
    setattr(solver_cfg, "rtol", rtol_value)
    setattr(solver_cfg, "atol", atol_value)

    max_step_value = choose_max_step(getattr(solver_cfg, "max_step", DEFAULT_MAX_STEP_S))
    setattr(solver_cfg, "max_step", max_step_value if max_step_value is not None else DEFAULT_MAX_STEP_S)
    return solver_cfg
