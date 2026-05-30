# ST_LRPS/core/dynamics.py
# -*- coding: utf-8 -*-
"""
Core Dynamics Engine (EOM / RHS Builder)
=======================================

This module builds the equations of motion (RHS) used by numerical integrators
(e.g. SciPy ``solve_ivp`` or fixed-step propagators).

Architecture
------------
High-level Python objects (gravity, ephemeris, surface data, etc.) are kept
outside the inner loop. ``build_rhs()`` extracts all required inputs into
Numba-friendly primitives (floats, arrays, booleans) and constructs a JIT-
compiled RHS closure. The compiled RHS must not allocate heap memory or access
Python objects.

Reference frames
----------------
- Integration frame: Moon-Centered Inertial (MCI, J2000-like).
- Body-fixed frame: Moon-Centered Fixed (MCF).
- The ephemeris provides the inertial→fixed attitude quaternion ``q_i2f``,
  stored scalar-first as ``(w, x, y, z)``.

State convention
----------------
The state vector is

    ``y = [rx, ry, rz, vx, vy, vz]``

with an optional 7th element interpreted as spacecraft mass (``dm/dt = 0``).

Implementation notes
--------------------
- Ephemeris sampling uses an allocation-free kernel (unpacked float return).
- Perturbation models used in the RHS are written as allocation-free kernels.
- A single internal contract is enforced for provider inputs (gravity/ephem/
  surface) to keep the propagation core consistent across modules.
"""


# =============================================================================
# 0.                                IMPORTS
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Mapping

import time
import math
import numpy as np

from numba import njit 


from lunaris.common.constants import AU, MU_EARTH, MU_MOON, MU_SUN, P_SUN_1AU, R_MOON, R_EARTH_MEAN
from lunaris.common.type_defs import SpacecraftProps, PerturbationFlags, F64Array
from lunaris.common.math_utils import (
    quat_rotate_vec,
    latlon_from_xyz_m,
    wrap_lon_deg,
    clamp,
    sample_2d_scaled_bilinear,
    sample_grid_bilinear,
)

from lunaris.physics.ephemeris import get_ephem_state
from lunaris.physics.spherical_harmonics import sh_accel_fixed_numba, compute_point_mass_acceleration
from lunaris.physics.third_body_effects import accel_third_body_numba, accel_j2_oblate_diff_numba
from lunaris.physics.solar_effects import accel_srp
from lunaris.physics.relativity_effects import _schwarzschild_components
from lunaris.physics.surface_effects import accel_albedo_simple


# =============================================================================
# 1.                             SMALL HELPERS
# =============================================================================

@njit(cache=True)
def _sample_albedo_dn_scaled(
    lat_deg: float,
    lon_deg: float,
    dn_grid: np.ndarray,
    n_rows: int,
    n_cols: int,
    deg_per_px: float,
    lon_ref_deg: float,
    lat_ref_deg: float,
    flip_lat: int,
    scale: float,
    bias: float,
    nodata_dn: float,
    lat_min_deg: float,
    lat_max_deg: float,
) -> float:
    """
    Bilinearly sample a scaled-DN albedo grid at (lat, lon).

    Longitude is wrapped to [0, 360) and latitude is clamped to [lat_min_deg, lat_max_deg].
    Returns scale * DN + bias via a scaled bilinear sampler (nodata handled by the sampler).
    """

    # Guard against degenerate grids / invalid resolution
    if deg_per_px <= 0.0 or n_rows <= 0 or n_cols <= 0:
        return bias

    # These helpers must be Numba-compatible
    lon_wrapped = wrap_lon_deg(lon_deg)
    lat_clamped = clamp(lat_deg, lat_min_deg, lat_max_deg)

    # Latitude -> fractional row index
    # flip_lat == 0: i_f = (lat_ref - lat)/deg_per_px
    # else        : i_f = (lat - lat_ref)/deg_per_px
    lat_sign = 1.0 if flip_lat == 0 else -1.0
    i_f = (lat_sign * (lat_ref_deg - lat_clamped)) / deg_per_px

    # Longitude -> fractional col index (periodic)
    j_f = (lon_wrapped - lon_ref_deg) / deg_per_px
    j_f = j_f % n_cols

    return sample_2d_scaled_bilinear(
        dn_grid, i_f, j_f, n_rows, n_cols, scale, bias, nodata_dn
    )


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


def _is_surrogate_gravity_provider(obj: Any) -> bool:
    """
    Return ``True`` when the gravity object exposes the surrogate-runtime API.

    The classical SH path is fully Numba-compiled. ST-LRPS gravity must
    be evaluated through Python/PyTorch, so the dynamics engine needs to detect
    that provider class and route the RHS build accordingly.
    """

    return bool(
        obj is not None 
        and getattr(obj, "model_kind", None) == "st_lrps"
        and hasattr(obj, "acceleration_fixed")
    )


@njit(cache=True)
def _select_adaptive_sh_degree(
    r_norm_m: float,
    r_ref_m: float,
    degree_max: int,
    adaptive_mode: int,
    adaptive_power: float,
    adaptive_min_degree: int,
    quantization_step: int,
    table_alt_km: np.ndarray,
    table_degree: np.ndarray,
    table_len: int,
) -> int:
    """
    Pick the spherical-harmonic degree to evaluate for the current radius.

    Modes
    -----
    0:
        Adaptive logic disabled; always return `degree_max`.
    1:
        Piecewise altitude table. The first threshold where `alt_km <= threshold`
        wins; above the last threshold we fall back to `adaptive_min_degree`.
    2:
        Smooth power-law heuristic based on `(R_ref / r) ** power`.

    Notes
    -----
    The return value is always clamped to `[adaptive_min_degree, degree_max]` and
    quantized downward when `quantization_step > 1`. Quantizing downward avoids
    unexpectedly increasing runtime cost above the policy's requested level.
    """

    if degree_max <= 0:
        return 0

    n_eval = int(degree_max)
    min_degree = int(adaptive_min_degree)
    if min_degree < 0:
        min_degree = 0
    if min_degree > degree_max:
        min_degree = int(degree_max)

    if adaptive_mode == 1 and table_len > 0:
        alt_km = (float(r_norm_m) - float(r_ref_m)) / 1000.0
        if alt_km < 0.0:
            alt_km = 0.0

        n_eval = min_degree
        for i in range(int(table_len)):
            if alt_km <= float(table_alt_km[i]):
                n_eval = int(table_degree[i])
                break
    elif adaptive_mode == 2:
        safe_r = float(r_norm_m)
        if safe_r < float(r_ref_m):
            safe_r = float(r_ref_m)
        ratio = float(r_ref_m) / safe_r
        n_eval = int(float(degree_max) * (ratio ** float(adaptive_power)))

    if n_eval < min_degree:
        n_eval = min_degree
    if n_eval > degree_max:
        n_eval = int(degree_max)

    qstep = int(quantization_step)
    if qstep > 1:
        n_eval = (n_eval // qstep) * qstep
        if n_eval < min_degree:
            n_eval = min_degree

    return int(n_eval)


def extract_gravity_strict(g: Any) -> Tuple[Any, ...]:
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
        ws_obj = getattr(g, "ws")
    elif hasattr(g, "make_workspace"):
        ws_obj = g.make_workspace()  # type: ignore[attr-defined]
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


def extract_ephem_tables_strict(ephem: Any) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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

    d = ephem.get_data_provider()  # type: ignore[attr-defined]
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


def extract_surface_provider_strict(surface_provider: Any) -> Dict[str, Any]:
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
        p = surface_provider.as_numba_dict()  # type: ignore[attr-defined]
        if not isinstance(p, Mapping):
            raise TypeError("surface_provider.as_numba_dict() must return a mapping/dict.")
        return dict(p)

    raise TypeError(
        "surface_provider must be a mapping/dict or implement as_numba_dict()."
    )



# =============================================================================
# 2.                               PACKINGS
# =============================================================================

def need_ephemeris(flags: PerturbationFlags) -> bool:
    """Return True if any enabled perturbation requires ephemeris tables."""
    return bool(flags.enable_third_body or flags.enable_srp or flags.enable_albedo)


def require_srp_props(sc: SpacecraftProps) -> Tuple[float, float, float]:
    """Validate and return (mass_kg, area_m2, cr) required by SRP/albedo models."""
    if sc.mass_kg <= 0.0:
        raise ValueError(f"mass_kg must be > 0, got {sc.mass_kg}")
    if sc.area_m2 <= 0.0:
        raise ValueError(f"area_m2 must be > 0 for SRP/Albedo, got {sc.area_m2}")
    if not (0.0 < sc.cr <= 2.5):
        raise ValueError(f"cr looks invalid, got {sc.cr}")
    return float(sc.mass_kg), float(sc.area_m2), float(sc.cr)


@dataclass(frozen=True, slots=True, kw_only=True)
class _EphemPack:
    """
    Engine-internal ephemeris pack (validated, float64, C-contiguous).

    `q_i2f_tab` owns the master sample cadence. Sun/Earth tables may either be
    sampled on the same cadence or collapse to a single constant row when the
    caller built a q-only ephemeris set.
    """
    dt_s: float
    r_sun_tab_m: F64Array     # (N,3) or (1,3)
    r_earth_tab_m: F64Array   # (N,3) or (1,3)
    q_i2f_tab: F64Array       # (N,4)

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")

        sun = _as_f64_c(self.r_sun_tab_m, "r_sun_tab_m")
        earth = _as_f64_c(self.r_earth_tab_m, "r_earth_tab_m")
        q = _as_f64_c(self.q_i2f_tab, "q_i2f_tab")

        if sun.ndim != 2 or sun.shape[1] != 3:
            raise ValueError(f"r_sun_tab_m must be (N,3), got {sun.shape}")
        if earth.ndim != 2 or earth.shape[1] != 3:
            raise ValueError(f"r_earth_tab_m must be (N,3), got {earth.shape}")
        if q.ndim != 2 or q.shape[1] != 4:
            raise ValueError(f"q_i2f_tab must be (N,4), got {q.shape}")
        q_count = int(q.shape[0])
        if int(sun.shape[0]) not in (1, q_count):
            raise ValueError(f"r_sun_tab_m must be (1,3) or ({q_count},3), got {sun.shape}")
        if int(earth.shape[0]) not in (1, q_count):
            raise ValueError(f"r_earth_tab_m must be (1,3) or ({q_count},3), got {earth.shape}")
        if sun.shape[0] != earth.shape[0] and 1 not in (sun.shape[0], earth.shape[0]):
            raise ValueError(
                "ephem vector N mismatch: "
                f"sun={sun.shape[0]}, earth={earth.shape[0]} (expected same N or a single constant row)"
            )

        object.__setattr__(self, "r_sun_tab_m", sun)
        object.__setattr__(self, "r_earth_tab_m", earth)
        object.__setattr__(self, "q_i2f_tab", q)


@dataclass(frozen=True, slots=True, kw_only=True)
class _GravPack:
    """Engine-internal gravity pack (validated, float64, C-contiguous; GravityModel)."""
    nmax: int
    r_ref_m: float
    gm_m3s2: float

    Cnm: F64Array
    Snm: F64Array
    diag: F64Array
    subdiag: F64Array
    A: F64Array
    B: F64Array
    scale_m: F64Array

    ws_P: F64Array
    ws_dP: F64Array
    ws_cos_m: F64Array
    ws_sin_m: F64Array

    # Optional runtime degree policy for adaptive SH evaluation.
    adaptive_enabled: bool = False
    adaptive_mode: int = 0  # 0=off, 1=table, 2=power-law
    adaptive_power: float = 2.5
    adaptive_min_degree: int = 4
    adaptive_quantization_step: int = 10
    adaptive_table_alt_km: Optional[F64Array] = None
    adaptive_table_degree: Optional[np.ndarray] = None
    adaptive_table_len: int = 0

    def __post_init__(self) -> None:
        if self.nmax < 0:
            raise ValueError(f"nmax must be >= 0, got {self.nmax}")
        if self.r_ref_m <= 0.0 or self.gm_m3s2 <= 0.0:
            raise ValueError(
                f"r_ref_m and gm_m3s2 must be positive (r={self.r_ref_m}, gm={self.gm_m3s2})"
            )

        Cnm = _as_f64_c(self.Cnm, "Cnm")
        Snm = _as_f64_c(self.Snm, "Snm")
        scale_m = _as_f64_c(self.scale_m, "scale_m")

        if Cnm.ndim < 2 or Snm.ndim < 2:
            raise ValueError("Cnm/Snm must be at least 2D arrays.")
        if Cnm.shape[0] < self.nmax + 1 or Cnm.shape[1] < self.nmax + 1:
            raise ValueError(f"Cnm too small for nmax={self.nmax}: {Cnm.shape}")
        if Snm.shape[0] < self.nmax + 1 or Snm.shape[1] < self.nmax + 1:
            raise ValueError(f"Snm too small for nmax={self.nmax}: {Snm.shape}")
        if scale_m.ndim != 1 or scale_m.shape[0] < self.nmax + 1:
            raise ValueError(f"scale_m must be 1D len>=nmax+1, got {scale_m.shape}")

        object.__setattr__(self, "Cnm", Cnm)
        object.__setattr__(self, "Snm", Snm)
        object.__setattr__(self, "diag", _as_f64_c(self.diag, "diag"))
        object.__setattr__(self, "subdiag", _as_f64_c(self.subdiag, "subdiag"))
        object.__setattr__(self, "A", _as_f64_c(self.A, "A"))
        object.__setattr__(self, "B", _as_f64_c(self.B, "B"))
        object.__setattr__(self, "scale_m", scale_m)

        object.__setattr__(self, "ws_P", _as_f64_c(self.ws_P, "ws_P"))
        object.__setattr__(self, "ws_dP", _as_f64_c(self.ws_dP, "ws_dP"))
        object.__setattr__(self, "ws_cos_m", _as_f64_c(self.ws_cos_m, "ws_cos_m"))
        object.__setattr__(self, "ws_sin_m", _as_f64_c(self.ws_sin_m, "ws_sin_m"))

        if self.ws_cos_m.ndim != 1 or self.ws_sin_m.ndim != 1:
            raise ValueError("ws_cos_m/ws_sin_m must be 1D.")
        if self.ws_cos_m.shape[0] < self.nmax + 1 or self.ws_sin_m.shape[0] < self.nmax + 1:
            raise ValueError(f"workspace sin/cos too small for nmax={self.nmax}")

        alt_km = (
            np.zeros(1, dtype=np.float64)
            if self.adaptive_table_alt_km is None
            else _as_f64_c(self.adaptive_table_alt_km, "adaptive_table_alt_km")
        )
        deg = (
            np.zeros(1, dtype=np.int64)
            if self.adaptive_table_degree is None
            else np.ascontiguousarray(np.asarray(self.adaptive_table_degree, dtype=np.int64))
        )

        if alt_km.ndim != 1:
            raise ValueError("adaptive_table_alt_km must be 1D.")
        if deg.ndim != 1:
            raise ValueError("adaptive_table_degree must be 1D.")
        if alt_km.shape[0] != deg.shape[0]:
            raise ValueError("adaptive_table_alt_km and adaptive_table_degree must have the same length.")

        table_len = int(self.adaptive_table_len)
        if table_len < 0:
            raise ValueError(f"adaptive_table_len must be >= 0, got {table_len}")
        if table_len > int(alt_km.shape[0]):
            raise ValueError("adaptive_table_len exceeds the provided adaptive table storage.")

        object.__setattr__(self, "adaptive_power", float(self.adaptive_power))
        object.__setattr__(self, "adaptive_min_degree", int(self.adaptive_min_degree))
        object.__setattr__(self, "adaptive_quantization_step", max(1, int(self.adaptive_quantization_step)))
        object.__setattr__(self, "adaptive_table_alt_km", alt_km)
        object.__setattr__(self, "adaptive_table_degree", deg)
        object.__setattr__(self, "adaptive_table_len", table_len)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AlbedoPack:
    """
    Engine-internal albedo configuration.

    mode:
      0 = albedo grid (grid_alb)
      1 = scaled DN grid (dn; albedo = sf*DN + off)
      2 = constant albedo (alb_const)
    """
    mode: int
    alb_const: float
    alb_scale: float
    k_lambert: float

    grid_alb: Optional[F64Array] = None
    dn: Optional[F64Array] = None

    n_lines: int = 0
    n_samples: int = 0
    res_deg: float = 0.0
    lon0_deg: float = 0.0
    lat0_deg: float = 0.0
    sf: float = 1.0
    off: float = 0.0
    missing: float = -9999.0
    flip: int = 0
    latmin: float = -90.0
    latmax: float = 90.0

    def __post_init__(self) -> None:
        if self.mode not in (0, 1, 2):
            raise ValueError(f"albedo mode must be 0/1/2, got {self.mode}")

        if self.mode == 2:
            return

        if self.res_deg <= 0.0 or self.n_lines <= 0 or self.n_samples <= 0:
            raise ValueError("albedo grid params invalid (res_deg, n_lines, n_samples must be positive)")
        if not (self.latmin < self.latmax):
            raise ValueError(f"latmin must be < latmax (latmin={self.latmin}, latmax={self.latmax})")

        if self.mode == 0:
            if self.grid_alb is None:
                raise ValueError("mode=0 requires grid_alb")
            grid = _as_f64_c(self.grid_alb, "grid_alb")
            if grid.ndim != 2:
                raise ValueError(f"grid_alb must be 2D, got ndim={grid.ndim}")
            if grid.shape != (self.n_lines, self.n_samples):
                raise ValueError(
                    f"grid_alb shape mismatch: expected {(self.n_lines, self.n_samples)}, got {grid.shape}"
                )
            object.__setattr__(self, "grid_alb", grid)

        else:  # mode == 1
            if self.dn is None:
                raise ValueError("mode=1 requires dn")
            dn = _as_f64_c(self.dn, "dn")
            if dn.ndim != 2:
                raise ValueError(f"dn must be 2D, got ndim={dn.ndim}")
            if dn.shape != (self.n_lines, self.n_samples):
                raise ValueError(
                    f"dn shape mismatch: expected {(self.n_lines, self.n_samples)}, got {dn.shape}"
                )
            object.__setattr__(self, "dn", dn)


@dataclass(frozen=True, slots=True, kw_only=True)
class _EarthJ2Pack:
    """Engine-internal Earth J2 configuration (validated axis and reference radius)."""
    j2: float
    r_ref_m: float
    ax: float
    ay: float
    az: float

    def __post_init__(self) -> None:
        if self.r_ref_m <= 0.0:
            raise ValueError(f"EarthJ2 r_ref_m must be > 0, got {self.r_ref_m}")
        n = (self.ax * self.ax + self.ay * self.ay + self.az * self.az) ** 0.5
        if n <= 1e-15:
            raise ValueError("EarthJ2 axis vector is degenerate (norm ~ 0).")



# =============================================================================
# 3.                             DYNAMICS ENGINE
# =============================================================================

class DynamicsEngine:
    """
    Dynamics RHS builder for high-performance orbit propagation.

    State (SI):
        y = [rx, ry, rz, vx, vy, vz]  (+ optional mass_kg in y[6])
    """

    def __init__(
        self,
        sc_props: "SpacecraftProps",
        flags: "PerturbationFlags",
        *,
        gravity_model: Any = None,
        gravity_adaptive: Any = None,
        ephem_manager: Any = None,
        surface_provider: Any = None,
        earth_j2: Any = None,
        allow_identity_rotation: bool = False,
    ) -> None:
        self.sc_props = sc_props
        self.flags = flags

        self.grav = gravity_model
        self.gravity_adaptive = gravity_adaptive
        self.ephem = ephem_manager
        self.surf = surface_provider
        self.earth_j2 = earth_j2

        # If True, q_i2f is treated as identity when ephemeris is absent.
        # This only substitutes for frame rotation (q), NOT for Sun/Earth vectors.
        self.allow_identity_rotation = bool(allow_identity_rotation)

        self._rhs_cache: Optional[Callable[[float, np.ndarray], np.ndarray]] = None
        self._prep: Dict[str, Any] = {}  # debug/reporting packs + requirements

        self._validate_dependencies()

    # -------------------------------------------------------------------------
    # Requirements / validation
    # -------------------------------------------------------------------------
    def _requirements(self) -> Dict[str, bool]:
        f = self.flags

        use_sh = bool(getattr(f, "enable_sh", False))
        use_surrogate_gravity = bool(use_sh and _is_surrogate_gravity_provider(self.grav))
        use_albedo = bool(getattr(f, "enable_albedo", False))
        use_srp = bool(getattr(f, "enable_srp", False))
        use_3rd_sun = bool(getattr(f, "enable_3rd_body_sun", False))
        use_3rd_earth = bool(getattr(f, "enable_3rd_body_earth", False))

        # Backwards compatibility: enable_relativity_1pn or enable_relativity
        use_rel = bool(getattr(f, "enable_relativity_1pn", getattr(f, "enable_relativity", False)))

        use_earth_j2_flag = bool(getattr(f, "enable_earth_j2", False))
        use_earth_j2 = bool(use_earth_j2_flag and (self.earth_j2 is not None))

        # What data do we need from ephemeris?
        need_sun = bool(use_srp or use_3rd_sun or use_albedo)
        need_earth = bool(use_3rd_earth or use_earth_j2)
        need_q = bool(use_sh or use_surrogate_gravity or use_albedo)

        # Ephemeris manager is required if Sun/Earth vectors are needed.
        need_vectors = bool(need_sun or need_earth)

        # Quaternion can be substituted with identity if explicitly allowed.
        need_quat_from_ephem = bool(need_q and (not self.allow_identity_rotation))

        need_ephem = bool(need_vectors or need_quat_from_ephem)

        return {
            "use_sh": use_sh,
            "use_surrogate_gravity": use_surrogate_gravity,
            "use_albedo": use_albedo,
            "use_srp": use_srp,
            "use_3rd_sun": use_3rd_sun,
            "use_3rd_earth": use_3rd_earth,
            "use_rel": use_rel,
            "use_earth_j2": use_earth_j2,
            "need_sun": need_sun,
            "need_earth": need_earth,
            "need_q": need_q,
            "need_vectors": need_vectors,
            "need_quat_from_ephem": need_quat_from_ephem,
            "need_ephem": need_ephem,
        }

    def _validate_dependencies(self) -> None:
        req = self._requirements()

        if req["use_sh"] and self.grav is None:
            raise ValueError("enable_sh=True but gravity_model is None.")

        if req["use_albedo"] and self.surf is None:
            raise ValueError("enable_albedo=True but surface_provider is None.")

        # SRP / Albedo require valid spacecraft optical area and mass
        if req["use_srp"] or req["use_albedo"]:
            if self.sc_props.mass_kg <= 0.0:
                raise ValueError(f"mass_kg must be > 0, got {self.sc_props.mass_kg}")
            if self.sc_props.area_m2 <= 0.0:
                raise ValueError(f"area_m2 must be > 0 for SRP/Albedo, got {self.sc_props.area_m2}")
            if not (0.0 < self.sc_props.cr <= 2.5):
                raise ValueError(f"cr looks invalid, got {self.sc_props.cr}")

        if req["need_ephem"] and (self.ephem is None):
            reasons: list[str] = []
            if req["need_vectors"]:
                if req["need_sun"]:
                    reasons.append("Sun vector (SRP / 3rd-body Sun / Albedo)")
                if req["need_earth"]:
                    reasons.append("Earth vector (3rd-body Earth / Earth J2)")
            if req["need_quat_from_ephem"]:
                reasons.append("q_i2f (SH / Albedo)")

            why = "; ".join(reasons) if reasons else "Sun/Earth vectors and/or q_i2f"
            raise ValueError(
                f"Ephemeris is required for selected perturbations: {why}. "
                "Provide ephem_manager, or disable those perturbations. "
                "Note: allow_identity_rotation only replaces q_i2f, not Sun/Earth vectors."
            )

        # Features that may exist on flags but are not implemented here
        f = self.flags
        if bool(getattr(f, "enable_thermal", getattr(f, "enable_thermal_ir", False))):
            raise NotImplementedError(
                "Thermal perturbation enabled but not implemented in core.dynamics."
            )
        if bool(getattr(f, "enable_tides_k2", False)) or bool(getattr(f, "enable_tides_k3", False)):
            raise NotImplementedError(
                "Solid tides enabled but not implemented in core.dynamics."
            )

        if bool(getattr(f, "enable_earth_j2", False)) and (self.earth_j2 is None):
            raise ValueError(
                "enable_earth_j2=True but earth_j2 params are None."
            )

    # -------------------------------------------------------------------------
    # Providers -> prepared packs (strict)
    # -------------------------------------------------------------------------
    def _prepare_adaptive_gravity_policy(self, nmax: int) -> Dict[str, Any]:
        """
        Normalize optional adaptive-degree settings into kernel-friendly arrays.

        The backend config already validates table ordering, but the dynamics
        layer still clamps each requested degree to the loaded model's actual
        maximum. This keeps runtime evaluation robust even when older session
        files or UI presets request degrees higher than the active gravity file.
        """

        adaptive = self.gravity_adaptive
        if adaptive is None and self.grav is not None:
            adaptive = getattr(self.grav, "adaptive", None)

        disabled = {
            "adaptive_enabled": False,
            "adaptive_mode": 0,
            "adaptive_power": 2.5,
            "adaptive_min_degree": max(0, min(int(nmax), 4)),
            "adaptive_quantization_step": 1,
            "adaptive_table_alt_km": np.zeros(1, dtype=np.float64),
            "adaptive_table_degree": np.zeros(1, dtype=np.int64),
            "adaptive_table_len": 0,
        }
        if adaptive is None or not bool(getattr(adaptive, "enabled", False)) or int(nmax) <= 0:
            return disabled

        min_degree = max(0, min(int(nmax), int(getattr(adaptive, "min_degree", 4) or 4)))
        quant_step = max(1, int(getattr(adaptive, "quantization_step", 10) or 10))
        power = float(getattr(adaptive, "power", 2.5) or 2.5)

        raw_table = getattr(adaptive, "altitude_table", None)
        if raw_table:
            parsed_rows: list[tuple[float, int]] = []
            for row in raw_table:
                try:
                    alt_km = max(0.0, float(row[0]))
                    degree = max(min_degree, min(int(nmax), int(row[1])))
                except Exception:
                    continue
                parsed_rows.append((alt_km, degree))

            parsed_rows.sort(key=lambda item: item[0])
            cleaned_rows: list[tuple[float, int]] = []
            prev_alt = -1.0
            for alt_km, degree in parsed_rows:
                if alt_km <= prev_alt:
                    continue
                cleaned_rows.append((alt_km, degree))
                prev_alt = alt_km

            if cleaned_rows:
                return {
                    "adaptive_enabled": True,
                    "adaptive_mode": 1,
                    "adaptive_power": power,
                    "adaptive_min_degree": min_degree,
                    "adaptive_quantization_step": quant_step,
                    "adaptive_table_alt_km": np.ascontiguousarray(
                        np.asarray([row[0] for row in cleaned_rows], dtype=np.float64)
                    ),
                    "adaptive_table_degree": np.ascontiguousarray(
                        np.asarray([row[1] for row in cleaned_rows], dtype=np.int64)
                    ),
                    "adaptive_table_len": len(cleaned_rows),
                }

        return {
            "adaptive_enabled": True,
            "adaptive_mode": 2,
            "adaptive_power": power,
            "adaptive_min_degree": min_degree,
            "adaptive_quantization_step": quant_step,
            "adaptive_table_alt_km": np.zeros(1, dtype=np.float64),
            "adaptive_table_degree": np.zeros(1, dtype=np.int64),
            "adaptive_table_len": 0,
        }

    def _prepare_gravity(self) -> _GravPack:
        if self.grav is None:
            z11 = np.zeros((1, 1), dtype=np.float64)
            z1 = np.zeros(1, dtype=np.float64)
            return _GravPack(
                nmax=0,
                r_ref_m=float(R_MOON),
                gm_m3s2=float(MU_MOON),
                Cnm=z11,
                Snm=z11,
                diag=z1,
                subdiag=z1,
                A=z1,
                B=z1,
                scale_m=z1,
                ws_P=z11,
                ws_dP=z11,
                ws_cos_m=z1,
                ws_sin_m=z1,
                adaptive_enabled=False,
                adaptive_mode=0,
                adaptive_power=2.5,
                adaptive_min_degree=0,
                adaptive_quantization_step=1,
                adaptive_table_alt_km=np.zeros(1, dtype=np.float64),
                adaptive_table_degree=np.zeros(1, dtype=np.int64),
                adaptive_table_len=0,
            )

        if _is_surrogate_gravity_provider(self.grav):
            z11 = np.zeros((1, 1), dtype=np.float64)
            z1 = np.zeros(1, dtype=np.float64)
            return _GravPack(
                nmax=0,
                r_ref_m=float(getattr(self.grav, "R_ref_m", getattr(self.grav, "r_ref_m", R_MOON))),
                gm_m3s2=float(getattr(self.grav, "GM_m3s2", getattr(self.grav, "gm_m3s2", MU_MOON))),
                Cnm=z11,
                Snm=z11,
                diag=z1,
                subdiag=z1,
                A=z1,
                B=z1,
                scale_m=np.ones(1, dtype=np.float64),
                ws_P=z11,
                ws_dP=z11,
                ws_cos_m=np.ones(1, dtype=np.float64),
                ws_sin_m=np.zeros(1, dtype=np.float64),
                adaptive_enabled=False,
                adaptive_mode=0,
                adaptive_power=2.5,
                adaptive_min_degree=0,
                adaptive_quantization_step=1,
                adaptive_table_alt_km=np.zeros(1, dtype=np.float64),
                adaptive_table_degree=np.zeros(1, dtype=np.int64),
                adaptive_table_len=0,
            )

        nmax, r_ref, gm, Cnm, Snm, diag, subdiag, A, B, scale_m, ws_P, ws_dP, ws_cos, ws_sin = extract_gravity_strict(
            self.grav
        )
        adaptive_policy = self._prepare_adaptive_gravity_policy(int(nmax))

        return _GravPack(
            nmax=int(nmax),
            r_ref_m=float(r_ref),
            gm_m3s2=float(gm),
            Cnm=Cnm,
            Snm=Snm,
            diag=diag,
            subdiag=subdiag,
            A=A,
            B=B,
            scale_m=scale_m,
            ws_P=ws_P,
            ws_dP=ws_dP,
            ws_cos_m=ws_cos,
            ws_sin_m=ws_sin,
            adaptive_enabled=bool(adaptive_policy["adaptive_enabled"]),
            adaptive_mode=int(adaptive_policy["adaptive_mode"]),
            adaptive_power=float(adaptive_policy["adaptive_power"]),
            adaptive_min_degree=int(adaptive_policy["adaptive_min_degree"]),
            adaptive_quantization_step=int(adaptive_policy["adaptive_quantization_step"]),
            adaptive_table_alt_km=adaptive_policy["adaptive_table_alt_km"],
            adaptive_table_degree=adaptive_policy["adaptive_table_degree"],
            adaptive_table_len=int(adaptive_policy["adaptive_table_len"]),
        )

    def _prepare_ephem(self, req: Dict[str, bool]) -> _EphemPack:
        if self.ephem is None:
            # Only valid if we do NOT need Sun/Earth vectors AND we allow identity rotation for q_i2f.
            q_ident = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
            z23 = np.zeros((2, 3), dtype=np.float64)
            return _EphemPack(dt_s=1.0, r_sun_tab_m=z23, r_earth_tab_m=z23, q_i2f_tab=q_ident)

        dt_s, sun_tab, earth_tab, qtab = extract_ephem_tables_strict(self.ephem)
        return _EphemPack(dt_s=float(dt_s), r_sun_tab_m=sun_tab, r_earth_tab_m=earth_tab, q_i2f_tab=qtab)

    def _prepare_albedo(self, req: Dict[str, bool]) -> _AlbedoPack:
        if not req["use_albedo"]:
            return _AlbedoPack(mode=2, alb_const=0.12, alb_scale=1.0, k_lambert=1.0)

        surf = extract_surface_provider_strict(self.surf)

        alb_const = float(surf.get("albedo_const", 0.12))
        alb_scale = float(surf.get("scale", 1.0))
        k_lambert = float(surf.get("k_lambert", 1.0))

        if "albedo_grid" in surf:
            return _AlbedoPack(
                mode=0,
                alb_const=alb_const,
                alb_scale=alb_scale,
                k_lambert=k_lambert,
                grid_alb=surf["albedo_grid"],
                n_lines=int(surf["n_lines"]),
                n_samples=int(surf["n_samples"]),
                res_deg=float(surf["res_deg"]),
                lon0_deg=float(surf["lon0_deg"]),
                lat0_deg=float(surf["lat0_deg"]),
            )

        if "dn" in surf:
            return _AlbedoPack(
                mode=1,
                alb_const=alb_const,
                alb_scale=alb_scale,
                k_lambert=k_lambert,
                dn=surf["dn"],
                n_lines=int(surf["n_lines"]),
                n_samples=int(surf["n_samples"]),
                res_deg=float(surf["res_deg"]),
                lon0_deg=float(surf["lon0_deg"]),
                lat0_deg=float(surf["lat0_deg"]),
                sf=float(surf.get("scale_factor", 1.0)),
                off=float(surf.get("offset", 0.0)),
                missing=float(surf.get("missing_dn", -1.0)),
                flip=int(surf.get("flip_lat", 0)),
                latmin=float(surf.get("lat_min_deg", -90.0)),
                latmax=float(surf.get("lat_max_deg", 90.0)),
            )

        # Constant-only fallback
        return _AlbedoPack(mode=2, alb_const=alb_const, alb_scale=alb_scale, k_lambert=k_lambert)

    def _prepare_earth_j2(self, req: Dict[str, bool]) -> _EarthJ2Pack:
        if not req["use_earth_j2"] or (self.earth_j2 is None):
            return _EarthJ2Pack(j2=0.0, r_ref_m=1.0, ax=0.0, ay=0.0, az=1.0)

        ej2 = self.earth_j2
        j2 = float(ej2.j2_coeff)
        r_ref = float(ej2.r_eq_m)
        kx, ky, kz = ej2.spin_axis_i
        return _EarthJ2Pack(j2=j2, r_ref_m=r_ref, ax=float(kx), ay=float(ky), az=float(kz))

    # -------------------------------------------------------------------------
    # Public: build RHS
    # -------------------------------------------------------------------------
    def build_rhs(self, *, force_rebuild: bool = False) -> Callable[[float, np.ndarray], np.ndarray]:
        if self._rhs_cache is not None and not force_rebuild:
            return self._rhs_cache

        t0 = time.perf_counter()

        req = self._requirements()
        gp = self._prepare_gravity()
        ep = self._prepare_ephem(req)
        ap = self._prepare_albedo(req)
        ej = self._prepare_earth_j2(req)

        # Cache for debug/reporting
        self._prep = {"req": req, "grav": gp, "eph": ep, "alb": ap, "earth_j2": ej}

        # Flags captured into closure
        USE_SH = bool(req["use_sh"])
        USE_SURROGATE = bool(req.get("use_surrogate_gravity", False))
        USE_3RD_SUN = bool(req["use_3rd_sun"])
        USE_3RD_EARTH = bool(req["use_3rd_earth"])
        USE_SRP = bool(req["use_srp"])
        USE_ALBEDO = bool(req["use_albedo"])
        USE_REL = bool(req["use_rel"])
        USE_EJ2 = bool(req["use_earth_j2"])

        # Ephemeris fetch needed inside kernel only if we actually have ephem_manager.
        HAVE_EPH = bool(self.ephem is not None)
        NEED_EPH = bool(HAVE_EPH and (req["need_sun"] or req["need_earth"] or req["need_q"]))

        # Spacecraft constants
        SC_MASS = float(self.sc_props.mass_kg)
        SC_AREA = float(self.sc_props.area_m2)
        SC_CR = float(self.sc_props.cr)

        # Core constants
        MU_M = float(gp.gm_m3s2)  # prefer model GM even if SH is disabled
        MU_S = float(MU_SUN)
        MU_E = float(MU_EARTH)

        RMOON = float(R_MOON)
        R_EARTH = float(R_EARTH_MEAN)
        AU_ = float(AU)
        P1AU = float(P_SUN_1AU)

        ENABLE_ECLIPSE = True

        # Ephemeris arrays/scalars
        EPH_DT_S = float(ep.dt_s)
        EPH_SUN = ep.r_sun_tab_m
        EPH_EARTH = ep.r_earth_tab_m
        EPH_QTAB = ep.q_i2f_tab

        # Gravity scalars/arrays
        G_NMAX = int(gp.nmax)
        G_RREF = float(gp.r_ref_m)
        G_GM = float(gp.gm_m3s2)
        G_CNM = gp.Cnm
        G_SNM = gp.Snm
        G_DIAG = gp.diag
        G_SUB = gp.subdiag
        G_A = gp.A
        G_B = gp.B
        G_SCL = gp.scale_m
        G_ADAPTIVE_ENABLED = bool(gp.adaptive_enabled)
        G_ADAPTIVE_MODE = int(gp.adaptive_mode)
        G_ADAPTIVE_POWER = float(gp.adaptive_power)
        G_ADAPTIVE_MIN_DEG = int(gp.adaptive_min_degree)
        G_ADAPTIVE_QSTEP = int(gp.adaptive_quantization_step)
        G_ADAPTIVE_ALT_KM = gp.adaptive_table_alt_km
        G_ADAPTIVE_DEG = gp.adaptive_table_degree
        G_ADAPTIVE_TABLE_LEN = int(gp.adaptive_table_len)

        WS_P = gp.ws_P
        WS_DP = gp.ws_dP
        WS_COS = gp.ws_cos_m
        WS_SIN = gp.ws_sin_m

        # Albedo scalars/arrays (provide stable arrays to closure)
        ALB_MODE = int(ap.mode)
        ALB_CONST = float(ap.alb_const)
        ALB_SCALE = float(ap.alb_scale)
        ALB_KLAMB = float(ap.k_lambert)

        ALB_GRID = ap.grid_alb if ap.grid_alb is not None else np.zeros((1, 1), dtype=np.float64)
        ALB_DN = ap.dn if ap.dn is not None else np.zeros((1, 1), dtype=np.float64)

        ALB_NLINES = int(ap.n_lines)
        ALB_NSAMPLES = int(ap.n_samples)
        ALB_RES = float(ap.res_deg)
        ALB_LON0 = float(ap.lon0_deg)
        ALB_LAT0 = float(ap.lat0_deg)
        ALB_SF = float(ap.sf)
        ALB_OFF = float(ap.off)
        ALB_MISSING = float(ap.missing)
        ALB_FLIP = int(ap.flip)
        ALB_LATMIN = float(ap.latmin)
        ALB_LATMAX = float(ap.latmax)

        # Earth J2 axis normalize once outside kernel
        EJ2_J2 = float(ej.j2)
        EJ2_RREF = float(ej.r_ref_m)
        kx, ky, kz = float(ej.ax), float(ej.ay), float(ej.az)
        k2 = kx * kx + ky * ky + kz * kz
        if k2 > 0.0:
            invk = 1.0 / math.sqrt(k2)
            EJ2_KX, EJ2_KY, EJ2_KZ = kx * invk, ky * invk, kz * invk
        else:
            EJ2_KX, EJ2_KY, EJ2_KZ = 0.0, 0.0, 1.0

        if USE_SURROGATE:
            surrogate = self.grav

            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                rx, ry, rz = float(y[0]), float(y[1]), float(y[2])
                vx, vy, vz = float(y[3]), float(y[4]), float(y[5])

                n = int(y.shape[0])
                mass = float(SC_MASS if n <= 6 else y[6])

                ax = 0.0
                ay = 0.0
                az = 0.0

                sunx = 0.0
                suny = 0.0
                sunz = 0.0
                earthx = 0.0
                earthy = 0.0
                earthz = 0.0
                q0 = 1.0
                q1 = 0.0
                q2 = 0.0
                q3 = 0.0

                if NEED_EPH:
                    sunx, suny, sunz, earthx, earthy, earthz, q0, q1, q2, q3 = get_ephem_state(
                        float(t), EPH_DT_S, EPH_SUN, EPH_EARTH, EPH_QTAB
                    )

                if USE_SH:
                    rfx, rfy, rfz = quat_rotate_vec(q0, q1, q2, q3, rx, ry, rz)
                    s_ax, s_ay, s_az = surrogate.acceleration_fixed((rfx, rfy, rfz))
                    agx, agy, agz = quat_rotate_vec(
                        q0, -q1, -q2, -q3, float(s_ax), float(s_ay), float(s_az)
                    )
                    ax += agx
                    ay += agy
                    az += agz
                else:
                    gax, gay, gaz = compute_point_mass_acceleration(rx, ry, rz, MU_M)
                    ax += gax
                    ay += gay
                    az += gaz

                if USE_3RD_SUN:
                    a3x, a3y, a3z = accel_third_body_numba(rx, ry, rz, sunx, suny, sunz, MU_S)
                    ax += a3x
                    ay += a3y
                    az += a3z

                if USE_3RD_EARTH:
                    a3x, a3y, a3z = accel_third_body_numba(rx, ry, rz, earthx, earthy, earthz, MU_E)
                    ax += a3x
                    ay += a3y
                    az += a3z

                if USE_EJ2:
                    j2x, j2y, j2z = accel_j2_oblate_diff_numba(
                        rx, ry, rz, earthx, earthy, earthz, MU_E, EJ2_RREF, EJ2_J2, EJ2_KX, EJ2_KY, EJ2_KZ
                    )
                    ax += j2x
                    ay += j2y
                    az += j2z

                if USE_SRP:
                    earth_r2 = earthx * earthx + earthy * earthy + earthz * earthz
                    enable_earth = ENABLE_ECLIPSE and (earth_r2 > 1.0e12)
                    asx, asy, asz = accel_srp(
                        rx, ry, rz, sunx, suny, sunz, earthx, earthy, earthz,
                        RMOON, R_EARTH, AU_, P1AU, SC_CR, SC_AREA, mass,
                        ENABLE_ECLIPSE, enable_earth,
                    )
                    ax += asx
                    ay += asy
                    az += asz

                if USE_ALBEDO:
                    rfx, rfy, rfz = quat_rotate_vec(q0, q1, q2, q3, rx, ry, rz)
                    lat_deg, lon_deg, _ = latlon_from_xyz_m(rfx, rfy, rfz)

                    alb_val = ALB_CONST
                    if ALB_MODE == 0:
                        alb_val = sample_grid_bilinear(
                            lat_deg, lon_deg, ALB_GRID, ALB_NLINES, ALB_NSAMPLES, ALB_RES, ALB_LON0, ALB_LAT0
                        )
                    elif ALB_MODE == 1:
                        alb_val = _sample_albedo_dn_scaled(
                            lat_deg, lon_deg, ALB_DN, ALB_NLINES, ALB_NSAMPLES, ALB_RES,
                            ALB_LON0, ALB_LAT0, ALB_FLIP, ALB_SF, ALB_OFF, ALB_MISSING, ALB_LATMIN, ALB_LATMAX,
                        )

                    if math.isnan(alb_val):
                        alb_val = ALB_CONST
                    if alb_val < 0.0:
                        alb_val = 0.0
                    elif alb_val > 1.0:
                        alb_val = 1.0
                    alb_val *= ALB_SCALE

                    sfx, sfy, sfz = quat_rotate_vec(q0, q1, q2, q3, sunx, suny, sunz)
                    aax_f, aay_f, aaz_f = accel_albedo_simple(
                        rfx, rfy, rfz, sfx, sfy, sfz, RMOON, AU_, P1AU,
                        alb_val, ALB_KLAMB, SC_CR, SC_AREA, mass, 1 if ENABLE_ECLIPSE else 0,
                    )
                    aax, aay, aaz = quat_rotate_vec(q0, -q1, -q2, -q3, aax_f, aay_f, aaz_f)
                    ax += aax
                    ay += aay
                    az += aaz

                if USE_REL:
                    arx, ary, arz = _schwarzschild_components(rx, ry, rz, vx, vy, vz, MU_M)
                    ax += arx
                    ay += ary
                    az += arz

                dydt = np.empty_like(y)
                dydt[0] = vx
                dydt[1] = vy
                dydt[2] = vz
                dydt[3] = ax
                dydt[4] = ay
                dydt[5] = az
                if n > 6:
                    dydt[6] = 0.0
                return dydt

            self._rhs_cache = rhs
            dt_build = time.perf_counter() - t0
            print(f"[Dynamics] RHS ready. (build={dt_build:.3f}s | surrogate gravity)")
            return rhs

        # This closure captures runtime-sized arrays/config values, so Numba
        # cannot persist it to disk cache reliably. Disabling cache avoids a
        # noisy warning on every run without changing numerical behavior.
        @njit(cache=False, nogil=True)
        def _rhs_kernel_numba(
            t: float,
            y: np.ndarray,
            WS_P: np.ndarray,
            WS_DP: np.ndarray,
            WS_COS: np.ndarray,
            WS_SIN: np.ndarray,
        ) -> np.ndarray:
            rx, ry, rz = y[0], y[1], y[2]
            vx, vy, vz = y[3], y[4], y[5]

            n = y.shape[0]
            mass = SC_MASS if n <= 6 else y[6]

            ax = 0.0
            ay = 0.0
            az = 0.0

            # Ephemeris state (defaults)
            sunx = 0.0
            suny = 0.0
            sunz = 0.0
            earthx = 0.0
            earthy = 0.0
            earthz = 0.0
            q0 = 1.0
            q1 = 0.0
            q2 = 0.0
            q3 = 0.0

            if NEED_EPH:
                sunx, suny, sunz, earthx, earthy, earthz, q0, q1, q2, q3 = get_ephem_state(
                    t, EPH_DT_S, EPH_SUN, EPH_EARTH, EPH_QTAB
                )

            # A) Central gravity
            if USE_SH:
                rfx, rfy, rfz = quat_rotate_vec(q0, q1, q2, q3, rx, ry, rz)
                n_eval = G_NMAX
                if G_ADAPTIVE_ENABLED:
                    r_norm = math.sqrt(rx * rx + ry * ry + rz * rz)
                    n_eval = _select_adaptive_sh_degree(
                        r_norm,
                        G_RREF,
                        G_NMAX,
                        G_ADAPTIVE_MODE,
                        G_ADAPTIVE_POWER,
                        G_ADAPTIVE_MIN_DEG,
                        G_ADAPTIVE_QSTEP,
                        G_ADAPTIVE_ALT_KM,
                        G_ADAPTIVE_DEG,
                        G_ADAPTIVE_TABLE_LEN,
                    )
                afx, afy, afz = sh_accel_fixed_numba(
                    rfx,
                    rfy,
                    rfz,
                    n_eval,
                    G_RREF,
                    G_GM,
                    G_CNM,
                    G_SNM,
                    G_DIAG,
                    G_SUB,
                    G_A,
                    G_B,
                    G_SCL,
                    WS_P,
                    WS_DP,
                    WS_COS,
                    WS_SIN,
                )
                agx, agy, agz = quat_rotate_vec(q0, -q1, -q2, -q3, afx, afy, afz)
                ax += agx
                ay += agy
                az += agz
            else:
                gax, gay, gaz = compute_point_mass_acceleration(rx, ry, rz, MU_M)
                ax += gax
                ay += gay
                az += gaz

            # B) Third-body
            if USE_3RD_SUN:
                a3x, a3y, a3z = accel_third_body_numba(rx, ry, rz, sunx, suny, sunz, MU_S)
                ax += a3x
                ay += a3y
                az += a3z

            if USE_3RD_EARTH:
                a3x, a3y, a3z = accel_third_body_numba(rx, ry, rz, earthx, earthy, earthz, MU_E)
                ax += a3x
                ay += a3y
                az += a3z

            if USE_EJ2:
                j2x, j2y, j2z = accel_j2_oblate_diff_numba(
                    rx,
                    ry,
                    rz,
                    earthx,
                    earthy,
                    earthz,
                    MU_E,
                    EJ2_RREF,
                    EJ2_J2,
                    EJ2_KX,
                    EJ2_KY,
                    EJ2_KZ,
                )
                ax += j2x
                ay += j2y
                az += j2z

            # C) SRP
            if USE_SRP:
                earth_r2 = earthx * earthx + earthy * earthy + earthz * earthz
                enable_earth = ENABLE_ECLIPSE and (earth_r2 > 1.0e12)

                asx, asy, asz = accel_srp(
                    rx,
                    ry,
                    rz,
                    sunx,
                    suny,
                    sunz,
                    earthx,
                    earthy,
                    earthz,
                    RMOON,
                    R_EARTH,
                    AU_,
                    P1AU,
                    SC_CR,
                    SC_AREA,
                    mass,
                    ENABLE_ECLIPSE,
                    enable_earth,
                )
                ax += asx
                ay += asy
                az += asz

            # D) Albedo
            if USE_ALBEDO:
                rfx, rfy, rfz = quat_rotate_vec(q0, q1, q2, q3, rx, ry, rz)
                lat_deg, lon_deg, _ = latlon_from_xyz_m(rfx, rfy, rfz)

                alb_val = ALB_CONST
                if ALB_MODE == 0:
                    alb_val = sample_grid_bilinear(
                        lat_deg,
                        lon_deg,
                        ALB_GRID,
                        ALB_NLINES,
                        ALB_NSAMPLES,
                        ALB_RES,
                        ALB_LON0,
                        ALB_LAT0,
                    )
                elif ALB_MODE == 1:
                    alb_val = _sample_albedo_dn_scaled(
                        lat_deg,
                        lon_deg,
                        ALB_DN,
                        ALB_NLINES,
                        ALB_NSAMPLES,
                        ALB_RES,
                        ALB_LON0,
                        ALB_LAT0,
                        ALB_FLIP,
                        ALB_SF,
                        ALB_OFF,
                        ALB_MISSING,
                        ALB_LATMIN,
                        ALB_LATMAX,
                    )

                if math.isnan(alb_val):
                    alb_val = ALB_CONST
                if alb_val < 0.0:
                    alb_val = 0.0
                elif alb_val > 1.0:
                    alb_val = 1.0
                alb_val *= ALB_SCALE

                # Sun inertial -> fixed
                sfx, sfy, sfz = quat_rotate_vec(q0, q1, q2, q3, sunx, suny, sunz)

                aax_f, aay_f, aaz_f = accel_albedo_simple(
                    rfx,
                    rfy,
                    rfz,
                    sfx,
                    sfy,
                    sfz,
                    RMOON,
                    AU_,
                    P1AU,
                    alb_val,
                    ALB_KLAMB,
                    SC_CR,
                    SC_AREA,
                    mass,
                    1 if ENABLE_ECLIPSE else 0,
                )
                aax, aay, aaz = quat_rotate_vec(q0, -q1, -q2, -q3, aax_f, aay_f, aaz_f)
                ax += aax
                ay += aay
                az += aaz

            # E) Relativity
            if USE_REL:
                arx, ary, arz = _schwarzschild_components(rx, ry, rz, vx, vy, vz, MU_M)
                ax += arx
                ay += ary
                az += arz

            dydt = np.empty_like(y)
            dydt[0] = vx
            dydt[1] = vy
            dydt[2] = vz
            dydt[3] = ax
            dydt[4] = ay
            dydt[5] = az
            if n > 6:
                dydt[6] = 0.0
            return dydt

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return _rhs_kernel_numba(t, y, WS_P, WS_DP, WS_COS, WS_SIN)

        self._rhs_cache = rhs

        dt_build = time.perf_counter() - t0
        print(f"[Dynamics] RHS ready. (build={dt_build:.3f}s)")

        return rhs

    # -------------------------------------------------------------------------
    # Debug / reporting
    # -------------------------------------------------------------------------
    def get_acceleration_breakdown(self, t: float, y: np.ndarray) -> Dict[str, float]:
        """Return acceleration component norms at epoch t (debug/reporting)."""
        if not self._prep:
            self.build_rhs(force_rebuild=False)

        req: Dict[str, bool] = self._prep["req"]
        gp: _GravPack = self._prep["grav"]
        ep: _EphemPack = self._prep["eph"]
        ap: _AlbedoPack = self._prep["alb"]
        ej: _EarthJ2Pack = self._prep["earth_j2"]

        r = np.asarray(y[0:3], dtype=float)
        v = np.asarray(y[3:6], dtype=float)
        mass = float(y[6]) if (y.size > 6) else float(self.sc_props.mass_kg)

        mu_m = float(gp.gm_m3s2)  # consistent with RHS

        out: Dict[str, float] = {}

        # Ephemeris (Python-side fetch)
        sun = np.zeros(3, dtype=float)
        earth = np.zeros(3, dtype=float)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        have_eph = bool(self.ephem is not None)
        need_eph = bool(have_eph and (req["need_sun"] or req["need_earth"] or req["need_q"]))
        if need_eph:
            sx, sy, sz, ex, ey, ez, q0, q1, q2, q3 = get_ephem_state(
                float(t),
                float(ep.dt_s),
                np.ascontiguousarray(ep.r_sun_tab_m, dtype=np.float64),
                np.ascontiguousarray(ep.r_earth_tab_m, dtype=np.float64),
                np.ascontiguousarray(ep.q_i2f_tab, dtype=np.float64),
            )
            sun[:] = (sx, sy, sz)
            earth[:] = (ex, ey, ez)
            q[:] = (q0, q1, q2, q3)

        def _norm3(ax: float, ay: float, az: float) -> float:
            return float(math.sqrt(ax * ax + ay * ay + az * az))

        # Gravity
        if req["use_sh"]:
            if bool(req.get("use_surrogate_gravity", False)):
                rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])
                ax_f, ay_f, az_f = self.grav.acceleration_fixed((rfx, rfy, rfz))
                ax_i, ay_i, az_i = quat_rotate_vec(
                    q[0], -q[1], -q[2], -q[3], float(ax_f), float(ay_f), float(az_f)
                )
                out["Gravity (ST-LRPS)"] = _norm3(ax_i, ay_i, az_i)
            else:
                rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])

                # Copy workspace to avoid contaminating runtime scratch
                WP = np.ascontiguousarray(gp.ws_P, dtype=np.float64).copy()
                WDP = np.ascontiguousarray(gp.ws_dP, dtype=np.float64).copy()
                WC = np.ascontiguousarray(gp.ws_cos_m, dtype=np.float64).copy()
                WS = np.ascontiguousarray(gp.ws_sin_m, dtype=np.float64).copy()
                n_eval = int(gp.nmax)
                if bool(gp.adaptive_enabled):
                    n_eval = _select_adaptive_sh_degree(
                        float(np.linalg.norm(r)),
                        float(gp.r_ref_m),
                        int(gp.nmax),
                        int(gp.adaptive_mode),
                        float(gp.adaptive_power),
                        int(gp.adaptive_min_degree),
                        int(gp.adaptive_quantization_step),
                        np.ascontiguousarray(gp.adaptive_table_alt_km, dtype=np.float64),
                        np.ascontiguousarray(gp.adaptive_table_degree, dtype=np.int64),
                        int(gp.adaptive_table_len),
                    )

                ax_f, ay_f, az_f = sh_accel_fixed_numba(
                    rfx,
                    rfy,
                    rfz,
                    n_eval,
                    float(gp.r_ref_m),
                    float(gp.gm_m3s2),
                    gp.Cnm,
                    gp.Snm,
                    gp.diag,
                    gp.subdiag,
                    gp.A,
                    gp.B,
                    gp.scale_m,
                    WP,
                    WDP,
                    WC,
                    WS,
                )
                ax_i, ay_i, az_i = quat_rotate_vec(q[0], -q[1], -q[2], -q[3], ax_f, ay_f, az_f)
                out["Gravity (SH)"] = _norm3(ax_i, ay_i, az_i)
        else:
            ax0, ay0, az0 = compute_point_mass_acceleration(r[0], r[1], r[2], float(mu_m))
            out["Gravity (PM)"] = _norm3(ax0, ay0, az0)

        # Third body
        if req["use_3rd_sun"]:
            ax3, ay3, az3 = accel_third_body_numba(r[0], r[1], r[2], sun[0], sun[1], sun[2], float(MU_SUN))
            out["3rd Body (Sun)"] = _norm3(ax3, ay3, az3)

        if req["use_3rd_earth"]:
            ax3, ay3, az3 = accel_third_body_numba(r[0], r[1], r[2], earth[0], earth[1], earth[2], float(MU_EARTH))
            out["3rd Body (Earth)"] = _norm3(ax3, ay3, az3)

        if req["use_earth_j2"]:
            j2x, j2y, j2z = accel_j2_oblate_diff_numba(
                float(r[0]),
                float(r[1]),
                float(r[2]),
                float(earth[0]),
                float(earth[1]),
                float(earth[2]),
                float(MU_EARTH),
                float(ej.r_ref_m),
                float(ej.j2),
                float(ej.ax),
                float(ej.ay),
                float(ej.az),
            )
            out["3rd Body (Earth J2)"] = _norm3(j2x, j2y, j2z)

        # SRP
        if req["use_srp"]:
            earth_r2 = float(earth[0] * earth[0] + earth[1] * earth[1] + earth[2] * earth[2])
            enable_earth = bool(earth_r2 > 1.0e12)

            asx, asy, asz = accel_srp(
                r[0],
                r[1],
                r[2],
                sun[0],
                sun[1],
                sun[2],
                earth[0],
                earth[1],
                earth[2],
                float(R_MOON),
                float(R_EARTH_MEAN),
                float(AU),
                float(P_SUN_1AU),
                float(self.sc_props.cr),
                float(self.sc_props.area_m2),
                float(mass),
                True,
                enable_earth,
            )
            out["SRP"] = _norm3(asx, asy, asz)

        # Albedo
        if req["use_albedo"] and (ap.mode != 2):
            rfx, rfy, rfz = quat_rotate_vec(q[0], q[1], q[2], q[3], r[0], r[1], r[2])
            lat_deg, lon_deg, _ = latlon_from_xyz_m(rfx, rfy, rfz)

            aval = float(ap.alb_const)
            if ap.mode == 0 and ap.grid_alb is not None:
                aval = float(
                    sample_grid_bilinear(
                        lat_deg,
                        lon_deg,
                        np.ascontiguousarray(ap.grid_alb, dtype=np.float64),
                        int(ap.n_lines),
                        int(ap.n_samples),
                        float(ap.res_deg),
                        float(ap.lon0_deg),
                        float(ap.lat0_deg),
                    )
                )
            elif ap.mode == 1 and ap.dn is not None:
                aval = float(
                    _sample_albedo_dn_scaled(
                        lat_deg,
                        lon_deg,
                        np.ascontiguousarray(ap.dn, dtype=np.float64),
                        int(ap.n_lines),
                        int(ap.n_samples),
                        float(ap.res_deg),
                        float(ap.lon0_deg),
                        float(ap.lat0_deg),
                        int(ap.flip),
                        float(ap.sf),
                        float(ap.off),
                        float(ap.missing),
                        float(ap.latmin),
                        float(ap.latmax),
                    )
                )

            if math.isnan(aval):
                aval = float(ap.alb_const)
            aval = max(0.0, min(1.0, aval)) * float(ap.alb_scale)

            sfx, sfy, sfz = quat_rotate_vec(q[0], q[1], q[2], q[3], sun[0], sun[1], sun[2])

            aax_f, aay_f, aaz_f = accel_albedo_simple(
                rfx,
                rfy,
                rfz,
                sfx,
                sfy,
                sfz,
                float(R_MOON),
                float(AU),
                float(P_SUN_1AU),
                float(aval),
                float(ap.k_lambert),
                float(self.sc_props.cr),
                float(self.sc_props.area_m2),
                float(mass),
                1,
            )
            aax_i, aay_i, aaz_i = quat_rotate_vec(q[0], -q[1], -q[2], -q[3], aax_f, aay_f, aaz_f)
            out["Albedo"] = _norm3(aax_i, aay_i, aaz_i)

        # Relativity
        if req["use_rel"]:
            arx, ary, arz = _schwarzschild_components(r[0], r[1], r[2], v[0], v[1], v[2], float(mu_m))
            out["Relativity (1PN)"] = _norm3(arx, ary, arz)

        return out



# =============================================================================
# 4.                                PUBLIC API
# =============================================================================

__all__ = (
    # Main engine
    "DynamicsEngine",
)
