"""Neutral lunar gravity-data helpers shared across layers.

These helpers (the default lunar gravity coefficient path, its resolver, and the
lunar run-config heuristics) were previously defined in
``lunaris.surrogate.st_lrps.data.dataset_parameters``. That forced
``lunaris.physics`` and, transitively, ``lunaris.core`` via the batch propagation
engine to import the ST-LRPS ML subsystem just to resolve a data path and
sniff a run config.

They are dependency-light (standard library + ``lunaris.common``) and belong in
the shared layer. ``dataset_parameters`` now re-exports them, so existing
imports keep working unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.paths import data_dir_from_root, project_root_from_file

# Lunar body signature constants (SI), aliased for explicitness.
MU_MOON_SI: float = float(MU_MOON)
R_MOON_SI: float = float(R_MOON)

# Project root for editable checkouts; external data can be overridden by env.
_REPO_ROOT = project_root_from_file(__file__)
DEFAULT_LUNAR_GRAVITY_PATH = (
    data_dir_from_root(_REPO_ROOT) / "gravity_models" / "jggrx_1800f_sha.tab.txt"
)


def resolve_lunar_gravity_path(path: str | Path | None = None) -> Path:
    """
    Resolve the lunar gravity coefficient file inside the repository.

    If ``path`` is omitted, the canonical JGGRX lunar model shipped with the
    project is used.
    """

    candidate = Path(path) if path is not None else Path(DEFAULT_LUNAR_GRAVITY_PATH)
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Lunar gravity coefficient file not found: {candidate}")
    return candidate


def looks_lunar_like(
    *,
    mu_si: float | None = None,
    r_ref_m: float | None = None,
    rel_tol: float = 0.20,
) -> bool:
    """
    Return ``True`` when a body signature looks broadly consistent with the Moon.

    The tolerance is intentionally loose because some legacy artifacts store
    rounded constants, but it is still tight enough to reject Earth-scale runs.
    """

    tol = max(float(rel_tol), 0.0)

    def _close(val: float | None, ref: float) -> bool:
        if val is None:
            return False
        v = float(val)
        return abs(v - ref) / max(abs(ref), 1.0) <= tol

    checks = []
    if mu_si is not None:
        checks.append(_close(mu_si, MU_MOON_SI))
    if r_ref_m is not None:
        checks.append(_close(r_ref_m, R_MOON_SI))
    return bool(checks) and all(checks)


def validate_lunar_contract(
    *,
    mu_si: float | None = None,
    r_ref_m: float | None = None,
    gm_rel_tol: float = 1.0e-6,
    radius_rel_tol: float = 1.0e-6,
) -> bool:
    """Validate the numeric core of a strict lunar artifact contract.

    The tolerances admit the small, source-specific GM differences among
    high-resolution lunar gravity products while rejecting the legacy 20%
    discovery envelope. Model identity, source hash, coefficient frame,
    normalization, and tide system remain separate required metadata fields at
    the gravity/artifact contract boundary.
    """

    checks: list[bool] = []
    for value, reference, tolerance in (
        (mu_si, MU_MOON_SI, gm_rel_tol),
        (r_ref_m, R_MOON_SI, radius_rel_tol),
    ):
        if value is None:
            continue
        try:
            numeric = float(value)
            tol = max(float(tolerance), 0.0)
        except (TypeError, ValueError):
            checks.append(False)
            continue
        checks.append(abs(numeric - reference) / max(abs(reference), 1.0) <= tol)
    return bool(checks) and all(checks)


def is_lunar_body_signature(
    *,
    mu_si: float | None = None,
    r_ref_m: float | None = None,
    rel_tol: float = 0.20,
) -> bool:
    """Compatibility alias for the legacy lunar-discovery heuristic."""

    return looks_lunar_like(mu_si=mu_si, r_ref_m=r_ref_m, rel_tol=rel_tol)


def _safe_float(mapping: Mapping[str, Any], key: str) -> float | None:
    """Best-effort float extraction from loosely typed JSON-like mappings."""

    value = mapping.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def looks_like_lunar_run_config(config: Mapping[str, Any]) -> bool:
    """
    Decide whether a surrogate training config is explicitly lunar-oriented.

    Accepted evidence, in order of strength:
    1. explicit ``central_body`` plus a non-contradictory lunar numeric signature
    2. resolved / dataset-level GM close to lunar GM
    3. resolved / dataset-level reference radius close to lunar radius
    """

    body_name = str(config.get("central_body", "") or "").strip().lower()
    dataset_meta = config.get("dataset_meta")
    dataset_meta = dataset_meta if isinstance(dataset_meta, Mapping) else {}

    mu_candidates = (
        _safe_float(config, "resolved_mu_si"),
        _safe_float(dataset_meta, "mu_si"),
    )
    r_candidates = (
        _safe_float(config, "resolved_r_ref_m"),
        _safe_float(config, "r_ref_m"),
        _safe_float(dataset_meta, "r_ref_m"),
        _safe_float(dataset_meta, "r_ref_m_fallback"),
    )

    mu_values = [float(mu) for mu in mu_candidates if mu is not None]
    r_values = [float(r_ref) for r_ref in r_candidates if r_ref is not None]

    mu_checks = [looks_lunar_like(mu_si=mu) for mu in mu_values]
    r_checks = [looks_lunar_like(r_ref_m=r_ref) for r_ref in r_values]

    has_numeric_evidence = bool(mu_checks or r_checks) and all(mu_checks) and all(r_checks)

    if body_name in {"moon", "lunar", "selene"}:
        if has_numeric_evidence:
            return True
        # A bare label is no longer enough because older training scripts could
        # stamp ``central_body="moon"`` even when the underlying dataset did
        # not prove it numerically.
        return False

    if body_name:
        return False

    return has_numeric_evidence


__all__ = [
    "MU_MOON_SI",
    "R_MOON_SI",
    "DEFAULT_LUNAR_GRAVITY_PATH",
    "resolve_lunar_gravity_path",
    "looks_lunar_like",
    "is_lunar_body_signature",
    "validate_lunar_contract",
    "looks_like_lunar_run_config",
]
