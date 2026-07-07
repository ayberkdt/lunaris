"""Validated provider contracts for dynamics inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from lunaris.common.force_requirements import force_requirements
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps


def _as_f64_c(a: Any, name: str) -> np.ndarray:
    """Return float64, C-contiguous numpy array; reject empty inputs."""
    arr = np.asarray(a, dtype=np.float64)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _require_attr(obj: Any, attr: str, who: str) -> Any:
    """Get required attribute or raise a clear AttributeError (property-safe)."""
    try:
        return getattr(obj, attr)
    except AttributeError as e:
        raise AttributeError(f"{who} is missing required attribute: {attr}") from e

def extract_gravity_strict(g: Any) -> tuple[Any, ...]:
    """
    STRICT gravity contract (aligned with lunaris.physics.spherical_harmonics.GravityModel).

    Required attributes
    -------------------
    - degree_max : int
    - R_ref_m    : float
    - GM_m3s2    : float
    - Cnm, Snm, diag, subdiag, A, B, scale_m : array-like

    Workspace
    ---------
    - Prefer an existing `ws` attribute (preallocated; avoids per-call allocation).
    - Otherwise call `make_workspace()` once at engine init/build time.
    - Workspace must provide: P, dP, cos_m, sin_m

    Returns
    -------
    Tuple of Numba-friendly values (float64, C-contiguous arrays).
    """
    if g is None:
        raise ValueError("gravity_model is None")

    nmax = int(_require_attr(g, "degree_max", "gravity_model"))
    if nmax < 0:
        raise ValueError(f"gravity_model.degree_max must be >= 0, got {nmax}")

    r_ref = float(_require_attr(g, "R_ref_m", "gravity_model"))
    gm = float(_require_attr(g, "GM_m3s2", "gravity_model"))
    if r_ref <= 0.0 or gm <= 0.0:
        raise ValueError(
            f"gravity_model scalars must be positive (R_ref_m={r_ref}, GM_m3s2={gm})"
        )

    # Workspace: use preallocated ws if present; else build once via make_workspace().
    if hasattr(g, "ws"):
        ws_obj = g.ws
    elif hasattr(g, "make_workspace"):
        ws_obj = g.make_workspace()
    else:
        raise AttributeError("gravity_model must define `ws` or `make_workspace()`.")

    if isinstance(ws_obj, Mapping):
        raise TypeError("gravity_model workspace must be an object (no dict/mapping legacy).")

    try:
        ws_P = ws_obj.P
        ws_dP = ws_obj.dP
        ws_cos = ws_obj.cos_m
        ws_sin = ws_obj.sin_m
    except Exception as e:
        raise ValueError("gravity_model workspace must provide: P, dP, cos_m, sin_m.") from e

    # Kernel arrays (float64, contiguous)
    Cnm = _as_f64_c(_require_attr(g, "Cnm", "gravity_model"), "gravity_model.Cnm")
    Snm = _as_f64_c(_require_attr(g, "Snm", "gravity_model"), "gravity_model.Snm")
    diag = _as_f64_c(_require_attr(g, "diag", "gravity_model"), "gravity_model.diag")
    subdiag = _as_f64_c(_require_attr(g, "subdiag", "gravity_model"), "gravity_model.subdiag")
    A = _as_f64_c(_require_attr(g, "A", "gravity_model"), "gravity_model.A")
    B = _as_f64_c(_require_attr(g, "B", "gravity_model"), "gravity_model.B")
    scale_m = _as_f64_c(_require_attr(g, "scale_m", "gravity_model"), "gravity_model.scale_m")

    # Workspace arrays (float64, contiguous)
    ws_P = _as_f64_c(ws_P, "gravity_model.ws.P")
    ws_dP = _as_f64_c(ws_dP, "gravity_model.ws.dP")
    ws_cos = _as_f64_c(ws_cos, "gravity_model.ws.cos_m")
    ws_sin = _as_f64_c(ws_sin, "gravity_model.ws.sin_m")

    # Minimal sanity checks (avoid over-assuming exact internal layout)
    if Cnm.ndim < 2 or Snm.ndim < 2:
        raise ValueError("Cnm/Snm must be at least 2D arrays.")
    if Cnm.shape[0] < (nmax + 1) or Cnm.shape[1] < (nmax + 1):
        raise ValueError(f"Cnm shape too small for nmax={nmax}: got {Cnm.shape}")
    if Snm.shape[0] < (nmax + 1) or Snm.shape[1] < (nmax + 1):
        raise ValueError(f"Snm shape too small for nmax={nmax}: got {Snm.shape}")
    if scale_m.ndim != 1 or scale_m.shape[0] < (nmax + 1):
        raise ValueError(f"scale_m must be 1D with len>=nmax+1 (nmax={nmax}), got {scale_m.shape}")
    if ws_cos.ndim != 1 or ws_sin.ndim != 1:
        raise ValueError("ws.cos_m and ws.sin_m must be 1D arrays.")
    if ws_cos.shape[0] < (nmax + 1) or ws_sin.shape[0] < (nmax + 1):
        raise ValueError(f"workspace sin/cos arrays too small for nmax={nmax}")

    return (
        nmax, r_ref, gm,
        Cnm, Snm, diag, subdiag,
        A, B, scale_m,
        ws_P, ws_dP, ws_cos, ws_sin,
    )


def extract_ephem_tables_strict(ephem: Any) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    STRICT ephemeris contract (aligned with lunaris.physics.ephemeris.EphemerisManager).

    ephem.get_data_provider() must return a mapping with either:
      A) dt, sun_table, earth_table, rot_table
      B) dt_s, r_sun_tab_m, r_earth_tab_m, q_i2f_tab

    Returns (dt_s, sun_tab, earth_tab, q_tab) as float64 C-contiguous arrays.

    Table length contract
    ---------------------
    - `q_tab` is always the master timeline and must be shape `(N, 4)`.
    - `sun_tab` / `earth_tab` may either match that same `N` or collapse to a
      single constant row `(1, 3)` when third-body sampling was intentionally
      disabled during ephemeris construction.
    """
    if ephem is None:
        raise ValueError("ephem_manager is None")
    if not hasattr(ephem, "get_data_provider"):
        raise TypeError("ephem_manager must implement get_data_provider().")

    d = ephem.get_data_provider()
    if not isinstance(d, Mapping):
        raise TypeError("ephem_manager.get_data_provider() must return a mapping/dict.")

    # Resolve keys (accept both naming conventions)
    dt_val = d.get("dt", d.get("dt_s", None))
    sun = d.get("sun_table", d.get("r_sun_tab_m", None))
    earth = d.get("earth_table", d.get("r_earth_tab_m", None))
    qtab = d.get("rot_table", d.get("q_i2f_tab", None))

    if dt_val is None:
        raise KeyError(f"ephem_manager provider missing dt/dt_s. Got keys: {list(d.keys())}")

    dt_s = float(dt_val)
    if dt_s <= 0.0:
        raise ValueError(f"ephem_manager dt must be > 0, got {dt_s}")

    if sun is None or earth is None or qtab is None:
        raise KeyError(
            "ephem_manager provider missing required tables. "
            "Expected either (dt, sun_table, earth_table, rot_table) "
            "or (dt_s, r_sun_tab_m, r_earth_tab_m, q_i2f_tab). "
            f"Got keys: {list(d.keys())}"
        )

    sun_tab = _as_f64_c(sun, "sun_table/r_sun_tab_m")
    earth_tab = _as_f64_c(earth, "earth_table/r_earth_tab_m")
    q_tab = _as_f64_c(qtab, "rot_table/q_i2f_tab")

    if sun_tab.ndim != 2 or sun_tab.shape[1] != 3:
        raise ValueError(f"sun_table must have shape (N,3), got {sun_tab.shape}")
    if earth_tab.ndim != 2 or earth_tab.shape[1] != 3:
        raise ValueError(f"earth_table must have shape (N,3), got {earth_tab.shape}")
    if q_tab.ndim != 2 or q_tab.shape[1] != 4:
        raise ValueError(f"rotation table must have shape (N,4), got {q_tab.shape}")
    q_count = int(q_tab.shape[0])
    if int(sun_tab.shape[0]) not in (1, q_count):
        raise ValueError(
            "sun_table must either match the quaternion timeline or provide a "
            f"single constant row. Got sun={sun_tab.shape[0]}, q={q_count}."
        )
    if int(earth_tab.shape[0]) not in (1, q_count):
        raise ValueError(
            "earth_table must either match the quaternion timeline or provide a "
            f"single constant row. Got earth={earth_tab.shape[0]}, q={q_count}."
        )
    if sun_tab.shape[0] != earth_tab.shape[0] and 1 not in (sun_tab.shape[0], earth_tab.shape[0]):
        raise ValueError(
            "ephem vector tables must either share the same sample count or one "
            "side must be a single constant row. "
            f"Got sun={sun_tab.shape[0]}, earth={earth_tab.shape[0]}."
        )

    return dt_s, sun_tab, earth_tab, q_tab


def extract_surface_provider_strict(surface_provider: Any) -> dict[str, Any]:
    """
    STRICT surface provider contract.

    Accepts:
      - mapping/dict directly, OR
      - an object implementing as_numba_dict() -> mapping/dict

    No grids() legacy path.
    """
    if surface_provider is None:
        raise ValueError("surface_provider is None")

    if isinstance(surface_provider, Mapping):
        return dict(surface_provider)

    if hasattr(surface_provider, "as_numba_dict"):
        p = surface_provider.as_numba_dict()
        if not isinstance(p, Mapping):
            raise TypeError("surface_provider.as_numba_dict() must return a mapping/dict.")
        return dict(p)

    raise TypeError(
        "surface_provider must be a mapping/dict or implement as_numba_dict()."
    )

def need_ephemeris(flags: PerturbationFlags) -> bool:
    """Return True if any enabled perturbation requires ephemeris tables."""
    return force_requirements(
        flags,
        request_external_relativity=True,
    ).need_ephem


def require_srp_props(sc: SpacecraftProps) -> tuple[float, float, float]:
    """Validate and return (mass_kg, area_m2, cr) required by SRP/albedo models."""
    if sc.mass_kg <= 0.0:
        raise ValueError(f"mass_kg must be > 0, got {sc.mass_kg}")
    if sc.area_m2 <= 0.0:
        raise ValueError(f"area_m2 must be > 0 for SRP/Albedo, got {sc.area_m2}")
    if not (0.0 < sc.cr <= 2.5):
        raise ValueError(f"cr looks invalid, got {sc.cr}")
    return float(sc.mass_kg), float(sc.area_m2), float(sc.cr)

__all__ = [
    "extract_gravity_strict",
    "extract_ephem_tables_strict",
    "extract_surface_provider_strict",
    "need_ephemeris",
    "require_srp_props",
    "_require_attr",
    "_as_f64_c",
]
