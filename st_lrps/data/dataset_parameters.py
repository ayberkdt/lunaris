#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lunar Surrogate Dataset Parameters
=================================

This module is the Single Source of Truth (SSOT) for the experimental
``st_lrps`` pipeline. The rest of the project already uses the
Moon as the primary central body, so the ST-LRPS tooling must follow the same rule:

- dataset generation must sample the lunar gravity field
- training metadata must record lunar body constants
- evaluation / auto-detect helpers must prefer lunar-compatible artifacts

Why this file exists
--------------------
The original neural-surrogate experiments were developed in an Earth-centric sandbox and
several scripts carried over Earth defaults, Earth/LEO preset names, and even
an old built-in EGM96 convenience path. Those leftovers are dangerous because a
gravity surrogate can appear to "work" while silently learning the wrong body.

This module removes that ambiguity by exposing:

- authoritative lunar constants for the surrogate workflow
- the canonical lunar gravity coefficient file inside this repository
- reusable helpers for unit scaling and body-compatibility checks

Design principles
-----------------
1. Repository-local: all default paths resolve inside this project tree.
2. Lunar-first: every implicit default points to the Moon, not Earth.
3. Backward-safe: helpers are conservative and prefer rejecting ambiguous
   legacy artifacts over silently accepting them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    from common.constants import MU_MOON, R_MOON
    from loaders.io_gravity import load_gravity_model
except Exception:  # pragma: no cover
    import sys as _sys

    _REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT_FALLBACK) not in _sys.path:
        _sys.path.insert(0, str(_REPO_ROOT_FALLBACK))
    from common.constants import MU_MOON, R_MOON
    from loaders.io_gravity import load_gravity_model


# =============================================================================
# 1.                         LUNAR PHYSICS CONSTANTS
# =============================================================================

# These aliases keep the surrogate workflow explicit about the central body.
MU_MOON_SI: float = float(MU_MOON)
R_MOON_SI: float = float(R_MOON)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUNAR_GRAVITY_PATH = (
    _REPO_ROOT / "data" / "gravity_models" / "jggrx_1800f_sha.tab.txt"
)


# =============================================================================
# 2.                      DATASET / COEFFICIENT CONFIG SSOT
# =============================================================================


@dataclass(frozen=True)
class DatasetParameters:
    """
    Immutable surrogate-dataset configuration for lunar gravity generation.

    The generator, trainer, evaluator, and analysis helpers all read the same
    object so a change to the central-body assumptions happens in exactly one
    place.
    """

    central_body: str = "moon"
    mu_si: float = MU_MOON_SI
    r_ref_m: float = R_MOON_SI
    gravity_gfc_path: str = str(DEFAULT_LUNAR_GRAVITY_PATH)
    gravity_expected_norm: str = "fully_normalized"
    gravity_strict_norm: bool = True

    @property
    def mu_moon_si(self) -> float:
        """Return the lunar GM using a body-specific attribute name."""

        return float(self.mu_si)

    @property
    def r_moon_si(self) -> float:
        """Return the lunar reference radius using a body-specific attribute name."""

        return float(self.r_ref_m)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable mapping for provenance snapshots."""

        return asdict(self)


DEFAULT_DATASET_CONFIG = DatasetParameters()


# =============================================================================
# 3.                         UNIT-SCALING CONVENIENCE
# =============================================================================


def canonical_scales(*, mu_si: float, du_m: float) -> Tuple[float, float, float]:
    """
    Compute canonical length / time / velocity scales for gravity datasets.

    Parameters
    ----------
    mu_si:
        Central-body gravitational parameter in SI units [m^3/s^2].
    du_m:
        Characteristic length scale in metres. For the lunar workflow this is
        almost always the reference radius.

    Returns
    -------
    DU_m, TU_s, VU_m_s
        Distance, time, and velocity canonical scales.
    """

    mu_val = float(mu_si)
    du_val = float(du_m)
    if mu_val <= 0.0:
        raise ValueError(f"mu_si must be positive. Got {mu_si!r}.")
    if du_val <= 0.0:
        raise ValueError(f"du_m must be positive. Got {du_m!r}.")

    tu_s = (du_val**3 / mu_val) ** 0.5
    vu_m_s = du_val / tu_s
    return du_val, tu_s, vu_m_s


# =============================================================================
# 4.                        GRAVITY COEFFICIENT LOADING
# =============================================================================


def resolve_lunar_gravity_path(path: str | Path | None = None) -> Path:
    """
    Resolve the lunar gravity coefficient file inside the repository.

    If ``path`` is omitted, the canonical JGGRX lunar model shipped with the
    project is used.
    """

    candidate = Path(path) if path is not None else Path(DEFAULT_DATASET_CONFIG.gravity_gfc_path)
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Lunar gravity coefficient file not found: {candidate}")
    return candidate


def load_icgem_gfc(
    *,
    file_path: str | Path,
    max_degree: Optional[int] = None,
    expected_norm: str = "fully_normalized",
    strict: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load the repository's lunar gravity coefficient file.

    The historical function name is preserved because the surrogate scripts
    already call ``load_icgem_gfc(...)``. Internally we delegate to the main
    project loader, which supports the lunar ASCII tables used here.
    """

    resolved = resolve_lunar_gravity_path(file_path)
    n_use, r_ref_m, gm_m3s2, c_nm, s_nm = load_gravity_model(
        str(resolved),
        degree_max=max_degree,
        ascii_strict=bool(strict),
        ascii_require_normalization_state=(1 if strict else None),
    )
    meta = {
        "modelname": resolved.name,
        "path": str(resolved),
        "norm": str(expected_norm),
        "degree": int(n_use),
        "r_ref_m": float(r_ref_m),
        "mu_si": float(gm_m3s2),
        "central_body": "moon",
    }
    return c_nm, s_nm, meta


# =============================================================================
# 5.                        LUNAR COMPATIBILITY HELPERS
# =============================================================================


def is_lunar_body_signature(
    *,
    mu_si: Optional[float] = None,
    r_ref_m: Optional[float] = None,
    rel_tol: float = 0.20,
) -> bool:
    """
    Return ``True`` when a body signature looks consistent with the Moon.

    The tolerance is intentionally loose because some legacy artifacts store
    rounded constants, but it is still tight enough to reject Earth-scale runs.
    """

    tol = max(float(rel_tol), 0.0)

    def _close(val: Optional[float], ref: float) -> bool:
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


def _safe_float(mapping: Mapping[str, Any], key: str) -> Optional[float]:
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

    mu_checks = [is_lunar_body_signature(mu_si=mu) for mu in mu_values]
    r_checks = [is_lunar_body_signature(r_ref_m=r_ref) for r_ref in r_values]

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


def load_run_config(path: str | Path) -> Dict[str, Any]:
    """Load a surrogate run ``config.json`` using UTF-8 with fail-fast errors."""

    cfg_path = Path(path).expanduser().resolve()
    return json.loads(cfg_path.read_text(encoding="utf-8"))


__all__ = [
    "DatasetParameters",
    "DEFAULT_DATASET_CONFIG",
    "DEFAULT_LUNAR_GRAVITY_PATH",
    "MU_MOON_SI",
    "R_MOON_SI",
    "canonical_scales",
    "resolve_lunar_gravity_path",
    "load_icgem_gfc",
    "is_lunar_body_signature",
    "looks_like_lunar_run_config",
    "load_run_config",
]
