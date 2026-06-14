#!/usr/bin/env python3
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
from typing import Any

try:
    from lunaris.common.lunar_data import (
        DEFAULT_LUNAR_GRAVITY_PATH,
        MU_MOON_SI,
        R_MOON_SI,
        is_lunar_body_signature,
        looks_like_lunar_run_config,
        resolve_lunar_gravity_path,
    )
    from lunaris.loaders.io_gravity import load_gravity_model
except Exception:  # pragma: no cover
    from lunaris.common.lunar_data import (
        DEFAULT_LUNAR_GRAVITY_PATH,
        MU_MOON_SI,
        R_MOON_SI,
        is_lunar_body_signature,
        looks_like_lunar_run_config,
        resolve_lunar_gravity_path,
    )
    from lunaris.loaders.io_gravity import load_gravity_model


# =============================================================================
# 1.                      DATASET / COEFFICIENT CONFIG SSOT
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping for provenance snapshots."""

        return asdict(self)


DEFAULT_DATASET_CONFIG = DatasetParameters()


# =============================================================================
# 2.                         UNIT-SCALING CONVENIENCE
# =============================================================================


def canonical_scales(*, mu_si: float, du_m: float) -> tuple[float, float, float]:
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
# 3.                        GRAVITY COEFFICIENT LOADING
# =============================================================================

def load_icgem_gfc(
    *,
    file_path: str | Path,
    max_degree: int | None = None,
    expected_norm: str = "fully_normalized",
    strict: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
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
# 4.                           CONFIG FILE LOADING
# =============================================================================

def load_run_config(path: str | Path) -> dict[str, Any]:
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
