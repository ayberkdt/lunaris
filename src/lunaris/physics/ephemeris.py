# ST_LRPS/models/ephemeris.py
"""
Ephemeris (SPICE) Tables + Runtime Interpolator.

This module is the project's single, strict interface between:
- CSPICE kernels (via spiceypy) used at initialization time, and
- the allocation-free, Numba-compiled dynamics loop used at runtime.

At startup, it samples SPICE on a uniform time grid and materializes all
ephemeris/attitude data into contiguous float64 arrays. At runtime, it provides
fast interpolation (no SPICE calls) and optional frame transforms.

Responsibilities
----------------
1) Kernel lifecycle (initialization-time only)
   - Resolves/normalizes kernel paths and optionally auto-fixes common path issues.
   - Loads the requested SPICE kernels (LSK/SPK/PCK/BPC/FK/TF).
   - Optionally starts from a clean SPICE state (`clean_kernels_before=True`).
   - Optionally clears the SPICE kernel pool after table generation (`clear_kernels_after=True`).

   Notes on kernel paths:
   - Some distributions ship text-wrapped kernels (e.g., `.tls.txt`, `.tf.txt`, `.bpc.txt`).
     Path resolution may transparently accept these variants depending on the resolver config.

2) Table generation (one-time)
   - Samples SPICE at a fixed step `output_dt_s` over `duration_s`.
   - Uses a deterministic grid: t[k] = k * dt, with the last sample guaranteed <= duration_s.
   - Stores:
       * Observer→Earth and Observer→Sun position vectors in the inertial frame (meters)
       * Inertial→body-fixed attitude as a quaternion time series [w, x, y, z]

   Performance note:
   - Third-body position sampling attempts a vectorized SPICE call for ET arrays when available;
     if unsupported or shape-mismatched, it falls back to a robust scalar loop.

3) Runtime access (no SPICE dependency)
   - High-level: `EphemerisManager` interpolation + frame transforms (Python API)
   - Low-level: `get_ephem_state()` allocation-free sampler for Numba loops

Data contract (strict, canonical keys)
--------------------------------------
`EphemerisManager.get_data_provider()` returns a dict intended for the
dynamics-layer ephemeris extractor (e.g., `core.dynamics._extract_ephem_tables`).
The provider contains only canonical keys (no aliases):

- "dt_s"            : float
    Sampling step in seconds (uniform).

- "t_tab_s"         : ndarray (N,), float64
    Table timestamps in seconds relative to table start (0 ... <= duration_s).

- "et0"             : float
    SPICE ephemeris time (ET) at table start.

- "q_i2f_tab"       : ndarray (N, 4), float64
    Inertial→fixed rotation quaternion series, scalar-first [w, x, y, z].
    Convention: v_fixed = q ⊗ v_inertial ⊗ conj(q)

- "r_earth_tab_m"   : ndarray (N, 3) or (1, 3), float64
    Observer→Earth in the inertial frame, meters [m].
    If third-body ephemerides are disabled, the table may be (1,3) zeros and
    interpolation degenerates to a constant return.

- "r_sun_tab_m"     : ndarray (N, 3) or (1, 3), float64
    Observer→Sun in the inertial frame, meters [m].
    Same degeneracy behavior as above when disabled.

- "mu_earth_m3s2"   : float
    Earth GM in SI units [m^3/s^2] (from kernel pool if available; fallback otherwise).

- "mu_sun_m3s2"     : float
    Sun GM in SI units [m^3/s^2].

- "inertial_frame"  : str
    Inertial reference frame used for sampling (default: "J2000").

- "fixed_frame"     : str
    Body-fixed frame used for sampling (default: "MOON_PA" when available).
    If `need_moon_fixed_rotation=False`, fixed_frame is set to the inertial frame
    and quaternions are identity.

- "observer"        : str
    SPICE observer body (default: "MOON").

Reference frames and kernels
----------------------------
- Inertial frame: "J2000" (SPICE inertial frame; often treated as ICRF for engineering use).
- Moon-fixed frame (high-fidelity): requires BOTH
  * a binary PCK (.bpc) for lunar orientation, AND
  * a frame kernel (.tf/.fk) defining the requested lunar frame chain (e.g., MOON_PA / DE440).

Auto-inclusion:
- If enabled, the builder may auto-include an appropriate lunar frame kernel from the same
  directory as provided kernels when a lunar fixed frame is requested.

Typical kernel set:
  1) LSK  (leapseconds)            e.g., naif0012.tls
  2) SPK  (ephemerides)            e.g., de440.bsp
  3) PCK  (planetary constants)    e.g., pck00011.tpc (optional but recommended)
  4) BPC  (lunar orientation)      e.g., moon_pa_de440_*.bpc
  5) FK/TF (frame definitions)     e.g., moon_de440_*.tf

Typical usage
-------------
    # Build once (initialization)
    tables = build_tables(
        start_utc="2026-01-01T00:00:00",
        duration_s=7 * 86400.0,
        output_dt_s=60.0,
        kernels=(...),
        fixed_frame="MOON_PA",
        include_third_body=True,
        clean_kernels_before=True,
        clear_kernels_after=True,
    )

    # Runtime (no SPICE calls)
    mgr = EphemerisManager(tables)
    r_sun = mgr.get_sun_position(t_s=1234.0)

    # Inject into dynamics
    provider = mgr.get_data_provider()
"""


# =============================================================================
# 0.                                IMPORTS
# =============================================================================

from __future__ import annotations

import warnings
from contextlib import contextmanager

from dataclasses import dataclass
from collections.abc import Sequence, Iterator
from typing import Optional, Tuple, Mapping, Any

import numpy as np

from numba import njit

import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError

from lunaris.common.constants import MU_EARTH, MU_SUN, KM_TO_M, KM3_TO_M3
from lunaris.common.type_defs import F64Array, TimeConfig
from lunaris.common.math_utils import quat_rotate_vec, quat_conj, interp_quat_slerp, interp_vec3_catmull
from lunaris.loaders.spice_builder import maybe_autoinclude_lunar_fk, resolve_kernel_paths



# =============================================================================
# 1.                           CONSTANTS & CONFIG
# =============================================================================

DEFAULT_INERTIAL_FRAME = "J2000"
DEFAULT_FIXED_FRAME = "MOON_PA"
DEFAULT_OBSERVER = "MOON"


# Fallback GM values (m^3/s^2) if the SPICE kernel pool is missing data.
_FALLBACK_GM_M3S2: dict[str, float] = {
    "EARTH": float(MU_EARTH),
    "SUN": float(MU_SUN),
}



# =============================================================================
# 2.                        DATA STRUCTURES (CONFIG)
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class SpiceBuildConfig:
    """
    Configuration for SPICE kernels and frames (time-agnostic).

    This recipe defines which kernels to load and what frames to use.
    Simulation duration and step size belong to TimeConfig.
    """
    kernels: Tuple[str, ...] = (
        "naif0012.tls",
        "de440.bsp",
        "moon_pa_de440_200625.bpc",
    )

    inertial_frame: str = DEFAULT_INERTIAL_FRAME
    fixed_frame: str = DEFAULT_FIXED_FRAME
    observer: str = DEFAULT_OBSERVER

    include_third_body: bool = True
    clear_kernels_after: bool = True

    earth_target: str = "EARTH"
    sun_target: str = "SUN"

    def __post_init__(self) -> None:
        if not self.kernels:
            raise ValueError("SpiceBuildConfig.kernels cannot be empty.")


@dataclass(frozen=True, slots=True, kw_only=True)
class EphemerisTables:
    """
    Immutable container holding precomputed, time-aligned ephemeris tables.

    Intended for high-performance interpolation in the simulation loop.
    Arrays are expected to be float64 and C-contiguous where possible.
    """
    dt_s: float
    t_tab_s: F64Array        # shape (N,)
    et0: float

    q_i2f_tab: F64Array     # shape (N, 4) [w,x,y,z]
    r_earth_tab_m: F64Array  # shape (N, 3) or (1, 3) if disabled
    r_sun_tab_m: F64Array    # shape (N, 3) or (1, 3) if disabled

    mu_earth_m3s2: float
    mu_sun_m3s2: float

    inertial_frame: str = DEFAULT_INERTIAL_FRAME
    fixed_frame: str = DEFAULT_FIXED_FRAME
    observer: str = DEFAULT_OBSERVER

    def __post_init__(self) -> None:
        # Basic numeric sanity
        if not (self.dt_s > 0.0):
            raise ValueError("EphemerisTables.dt_s must be > 0.")

        # Shape sanity (cheap checks)
        if self.t_tab_s.ndim != 1:
            raise ValueError("EphemerisTables.t_tab_s must be 1D (N,).")
        n = int(self.t_tab_s.shape[0])

        if self.q_i2f_tab.shape != (n, 4):
            raise ValueError(f"q_i2f_tab must have shape (N,4); got {self.q_i2f_tab.shape}.")

        if self.r_earth_tab_m.shape not in ((n, 3), (1, 3)):
            raise ValueError(f"r_earth_tab_m must be (N,3) or (1,3); got {self.r_earth_tab_m.shape}.")

        if self.r_sun_tab_m.shape not in ((n, 3), (1, 3)):
            raise ValueError(f"r_sun_tab_m must be (N,3) or (1,3); got {self.r_sun_tab_m.shape}.")



# =============================================================================
# 3.                PRIVATE SPICE POOL QUERIES (GM, ETC)
# =============================================================================

def get_body_gm_m3s2(
    body_name: str,
    fallback_m3s2: Optional[float] = None,
    *,
    allow_fallback: bool = True,
    warn_on_fallback: bool = True,
) -> float:
    body = str(body_name).strip().upper()

    effective_fallback = fallback_m3s2
    if effective_fallback is None:
        effective_fallback = _FALLBACK_GM_M3S2.get(body)

    try:
        dim, values = spice.bodvrd(body, "GM", 1)
        if int(dim) != 1 or values is None or len(values) < 1:
            raise SpiceyError("GM returned with unexpected dimension/values.")
        gm_km3s2 = float(values[0])
        return gm_km3s2 * KM3_TO_M3

    except SpiceyError as e:
        if (not allow_fallback) or (effective_fallback is None):
            raise RuntimeError(
                f"GM not found in kernel pool for '{body}'. "
                f"Load a PCK (.tpc) or provide an explicit fallback. "
                f"SPICE Error: {type(e).__name__}"
            ) from e

        if warn_on_fallback:
            warnings.warn(
                f"[SPICE] GM for '{body}' missing from kernel pool; using fallback "
                f"{float(effective_fallback)} m^3/s^2.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        return float(effective_fallback)



# =============================================================================
# 6.                       PUBLIC BUILDERS (MAIN API)
# =============================================================================

def _require_positive(name: str, value: float) -> float:
    v = float(value)
    if v <= 0.0:
        raise ValueError(f"{name} must be positive. Got {value!r}")
    return v


def _build_time_grid(duration_s: float, output_dt_s: float) -> np.ndarray:
    """Deterministic uniform grid: 0, dt, 2dt, ... <= duration."""
    dur = float(duration_s)
    dt = float(output_dt_s)
    # +1e-12 to reduce off-by-one from binary float ratios.
    n = int(np.floor(dur / dt + 1.0e-12)) + 1
    return np.arange(n, dtype=np.float64) * dt


def _normalize_quat_series_inplace(q_tab: np.ndarray) -> None:
    """Normalizes quaternions and enforces sign continuity in-place."""
    norms = np.linalg.norm(q_tab, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    q_tab /= norms
    # Enforce sign continuity: q and -q are the same rotation.
    for k in range(1, int(q_tab.shape[0])):
        if float(np.dot(q_tab[k - 1], q_tab[k])) < 0.0:
            q_tab[k] *= -1.0


def _try_fill_spkpos_table_m(
    out_m: np.ndarray,
    *,
    spkpos,
    target: str,
    et_tab: np.ndarray,
    frame: str,
    observer: str,
) -> bool:
    """
    Attempts vectorized SPICE sampling for positions.

    Returns True on success and fills `out_m` (shape (N,3)) in meters.
    Returns False if vectorized sampling is unsupported or returns an unexpected shape.
    """
    try:
        pos_km, _ = spkpos(target, et_tab, frame, "NONE", observer)
        arr = np.asarray(pos_km, dtype=np.float64)
        if arr.ndim != 2:
            return False

        n = int(out_m.shape[0])
        if arr.shape == (3, n):
            arr = arr.T
        if arr.shape != (n, 3):
            return False

        out_m[:] = arr
        out_m *= KM_TO_M
        return True
    except Exception:
        # Any failure here should fall back to the scalar loop path.
        return False


@contextmanager
def _spice_kernels_loaded(
    kernels: Sequence[str],
    *,
    clean_before: bool = True,
    clear_after: bool = True,
) -> Iterator[None]:
    """Loads kernels into a predictable SPICE state and guarantees cleanup."""
    if clean_before:
        spice.kclear()

    try:
        for k in kernels:
            spice.furnsh(str(k))
        yield
    finally:
        if clear_after:
            spice.kclear()


def build_tables(
    *,
    start_utc: str,
    duration_s: float,
    output_dt_s: float,
    kernels: tuple[str, ...] | list[str],

    inertial_frame: str = DEFAULT_INERTIAL_FRAME,
    fixed_frame: str = DEFAULT_FIXED_FRAME,
    observer: str = DEFAULT_OBSERVER,

    include_third_body: bool = True,
    clear_kernels_after: bool = True,
    clean_kernels_before: bool = True,

    earth_target: str = "EARTH",
    sun_target: str = "SUN",

    auto_fix_kernel_paths: bool = True,
    need_moon_fixed_rotation: bool = True,
) -> EphemerisTables:
    """Core builder: samples SPICE at uniform output_dt_s to build ephemeris tables."""

    dt = _require_positive("output_dt_s", output_dt_s)
    _require_positive("duration_s", duration_s)
    if not kernels:
        raise ValueError("Kernel list cannot be empty.")

    inertial_frame = str(inertial_frame).strip()
    fixed_frame = str(fixed_frame).strip()
    observer = str(observer).strip()
    earth_target = str(earth_target).strip()
    sun_target = str(sun_target).strip()

    # Resolve and normalize kernel paths (and optionally auto-include lunar FK/TF).
    k_list = resolve_kernel_paths(list(kernels), auto_fix=bool(auto_fix_kernel_paths))
    k_list = maybe_autoinclude_lunar_fk(k_list, fixed_frame)

    # Deterministic time grid
    t_tab = _build_time_grid(float(duration_s), dt)
    Nt = int(t_tab.size)

    # Pre-allocate outputs
    q_tab = np.empty((Nt, 4), dtype=np.float64)
    if need_moon_fixed_rotation:
        out_fixed_frame = fixed_frame
    else:
        q_tab[:] = (1.0, 0.0, 0.0, 0.0)
        out_fixed_frame = inertial_frame

    if include_third_body:
        rE = np.empty((Nt, 3), dtype=np.float64)
        rS = np.empty((Nt, 3), dtype=np.float64)
    else:
        rE = np.zeros((1, 3), dtype=np.float64)
        rS = np.zeros((1, 3), dtype=np.float64)

    mu_earth = 0.0
    mu_sun = 0.0

    # Import here to avoid hard dependency at module import time in some environments.
    try:
        from spiceypy.utils.exceptions import SpiceyError  # type: ignore
    except Exception:  # pragma: no cover
        SpiceyError = Exception  # type: ignore

    with _spice_kernels_loaded(
        k_list,
        clean_before=bool(clean_kernels_before),
        clear_after=bool(clear_kernels_after),
    ):
        try:
            et0 = float(spice.str2et(str(start_utc)))
        except SpiceyError as e:
            raise ValueError(f"SPICE failed to parse start_utc={start_utc!r}") from e

        # Uniform ET grid
        et_tab = et0 + t_tab

        # Local bindings (tiny speedup in Python loops)
        pxform = spice.pxform
        m2q = spice.m2q
        spkpos = spice.spkpos

        # A) Attitude: inertial -> fixed (pxform is scalar ET; must loop)
        if need_moon_fixed_rotation:
            for i in range(Nt):
                et = float(et_tab[i])
                try:
                    mat_i2f = pxform(inertial_frame, fixed_frame, et)
                except SpiceyError as e:
                    raise RuntimeError(
                        f"Frame transform failed ({inertial_frame} -> {fixed_frame}). "
                        "Likely missing .tf/.fk or .bpc kernel."
                    ) from e
                q_tab[i] = m2q(mat_i2f)

        # B) Third-body states: target relative to observer
        if include_third_body:
            ok_e = _try_fill_spkpos_table_m(
                rE, spkpos=spkpos, target=earth_target, et_tab=et_tab, frame=inertial_frame, observer=observer
            )
            ok_s = _try_fill_spkpos_table_m(
                rS, spkpos=spkpos, target=sun_target, et_tab=et_tab, frame=inertial_frame, observer=observer
            )

            if not (ok_e and ok_s):
                # Fallback scalar path (robust, slightly slower)
                for i in range(Nt):
                    et = float(et_tab[i])
                    try:
                        pos_earth_km, _ = spkpos(earth_target, et, inertial_frame, "NONE", observer)
                        pos_sun_km, _ = spkpos(sun_target, et, inertial_frame, "NONE", observer)
                    except SpiceyError as e:
                        raise RuntimeError(f"SPICE position lookup failed at i={i}, et={et:.3f}") from e

                    rE[i, 0] = float(pos_earth_km[0]) * KM_TO_M
                    rE[i, 1] = float(pos_earth_km[1]) * KM_TO_M
                    rE[i, 2] = float(pos_earth_km[2]) * KM_TO_M

                    rS[i, 0] = float(pos_sun_km[0]) * KM_TO_M
                    rS[i, 1] = float(pos_sun_km[1]) * KM_TO_M
                    rS[i, 2] = float(pos_sun_km[2]) * KM_TO_M

        # GM retrieval (must occur while kernels are loaded).
        # Populate even if third-body ephemerides are disabled (zeros hide misconfigurations).
        mu_earth = get_body_gm_m3s2(earth_target, _FALLBACK_GM_M3S2.get("EARTH"))
        mu_sun = get_body_gm_m3s2(sun_target, _FALLBACK_GM_M3S2.get("SUN"))

    # Normalize quaternions (safety)
    if need_moon_fixed_rotation:
        _normalize_quat_series_inplace(q_tab)

    return EphemerisTables(
        dt_s=float(dt),
        t_tab_s=t_tab,
        et0=float(et0),
        q_i2f_tab=q_tab,
        r_earth_tab_m=rE,
        r_sun_tab_m=rS,
        mu_earth_m3s2=float(mu_earth),
        mu_sun_m3s2=float(mu_sun),
        inertial_frame=str(inertial_frame),
        fixed_frame=str(out_fixed_frame),
        observer=str(observer),
    )


def build_spice_tables(
    time_cfg: TimeConfig,
    spice_cfg: SpiceBuildConfig,
    *,
    auto_fix_kernel_paths: bool = True,
    need_moon_fixed_rotation: bool = True,
) -> EphemerisTables:
    """High-level public builder: merges TimeConfig + SpiceBuildConfig."""

    sd = time_cfg.start_date
    start_utc = sd.isoformat() if hasattr(sd, "isoformat") else str(sd)

    dt = time_cfg.output_dt_s
    if dt is None:
        raise ValueError(
            "TimeConfig.output_dt_s is None, but ephemeris tables require a fixed dt.\n"
            "Set output_dt_s (e.g., 60.0) for SPICE sampling."
        )

    return build_tables(
        start_utc=start_utc,
        duration_s=_require_positive("duration_s", time_cfg.duration_s),
        output_dt_s=_require_positive("output_dt_s", dt),
        kernels=spice_cfg.kernels,
        inertial_frame=str(spice_cfg.inertial_frame),
        fixed_frame=str(spice_cfg.fixed_frame),
        observer=str(spice_cfg.observer),
        include_third_body=bool(spice_cfg.include_third_body),
        clear_kernels_after=bool(spice_cfg.clear_kernels_after),
        earth_target=str(spice_cfg.earth_target),
        sun_target=str(spice_cfg.sun_target),
        auto_fix_kernel_paths=bool(auto_fix_kernel_paths),
        need_moon_fixed_rotation=bool(need_moon_fixed_rotation),
    )




# =============================================================================
# 7.                  HIGH-LEVEL MANAGER (RUNTIME INTERFACE)
# =============================================================================

class EphemerisManager:
    """Runtime interface for querying ephemeris tables (read-only)."""

    __slots__ = ("tables",)

    def __init__(self, tables: EphemerisTables) -> None:
        self.tables = tables

    # --- Strict factories -------------------------------------------------

    @classmethod
    def from_time_and_spice(
        cls,
        time_cfg: TimeConfig,
        spice_cfg: SpiceBuildConfig,
        *,
        auto_fix_kernel_paths: bool = True,
        need_moon_fixed_rotation: bool = True,
    ) -> "EphemerisManager":
        tables = build_spice_tables(
            time_cfg,
            spice_cfg,
            auto_fix_kernel_paths=auto_fix_kernel_paths,
            need_moon_fixed_rotation=need_moon_fixed_rotation,
        )
        return cls(tables)

    @classmethod
    def from_tables(cls, tables: EphemerisTables) -> "EphemerisManager":
        return cls(tables)

    # --- Canonical provider (no aliases) ----------------------------------

    def get_data_provider(self) -> Mapping[str, Any]:
        """
        Returns direct references to internal arrays for performance.
        Treat returned arrays as READ-ONLY.
        """
        t = self.tables
        return {
            "dt_s": float(t.dt_s),
            "t_tab_s": t.t_tab_s,
            "et0": float(t.et0),
            "q_i2f_tab": t.q_i2f_tab,
            "r_earth_tab_m": t.r_earth_tab_m,
            "r_sun_tab_m": t.r_sun_tab_m,
            "mu_earth_m3s2": float(t.mu_earth_m3s2),
            "mu_sun_m3s2": float(t.mu_sun_m3s2),
            "inertial_frame": t.inertial_frame,
            "fixed_frame": t.fixed_frame,
            "observer": t.observer,
        }

    # --- Properties --------------------------------------------------------

    @property
    def dt_ephem_s(self) -> float:
        return float(self.tables.dt_s)

    # --- Internals (DRY helpers) ------------------------------------------

    @staticmethod
    def _ensure_vec3(v: F64Array, *, name: str) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"{name} must be shape (3,), got {arr.shape}")
        return arr

    @staticmethod
    def _write_vec3(out: Optional[F64Array], x: float, y: float, z: float) -> F64Array:
        if out is None:
            out = np.empty(3, dtype=np.float64)
        out[0], out[1], out[2] = x, y, z
        return out

    def _interp_vec3_table(self, t_s: float, tab: np.ndarray, *, out: Optional[F64Array]) -> F64Array:
        # third-body disabled: table holds a single zero row
        if tab.shape[0] == 1:
            if out is None:
                return tab[0].copy()
            out[:] = tab[0]
            return out

        x, y, z = interp_vec3_catmull(float(t_s), float(self.tables.dt_s), tab)
        return self._write_vec3(out, x, y, z)

    def _interp_quat_i2f(self, t_s: float) -> tuple[float, float, float, float]:
        # interp_quat_slerp -> (w,x,y,z)
        return interp_quat_slerp(float(t_s), float(self.tables.dt_s), self.tables.q_i2f_tab)

    # --- Interpolation: public --------------------------------------------

    def get_inertial_to_fixed_rotation(
        self,
        t_s: float,
        *,
        out: Optional[F64Array] = None,
    ) -> F64Array:
        w, x, y, z = self._interp_quat_i2f(t_s)
        if out is None:
            out = np.empty(4, dtype=np.float64)
        out[0], out[1], out[2], out[3] = w, x, y, z
        return out

    def get_earth_position(
        self,
        t_s: float,
        *,
        out: Optional[F64Array] = None,
    ) -> F64Array:
        return self._interp_vec3_table(t_s, self.tables.r_earth_tab_m, out=out)

    def get_sun_position(
        self,
        t_s: float,
        *,
        out: Optional[F64Array] = None,
    ) -> F64Array:
        return self._interp_vec3_table(t_s, self.tables.r_sun_tab_m, out=out)

    # --- Frame transforms --------------------------------------------------

    def transform_inertial_to_fixed(
        self,
        t_s: float,
        v_inertial: F64Array,
        *,
        out: Optional[F64Array] = None,
    ) -> F64Array:
        v = self._ensure_vec3(v_inertial, name="v_inertial")

        w, x, y, z = self._interp_quat_i2f(t_s)
        xo, yo, zo = quat_rotate_vec(
            float(w), float(x), float(y), float(z),
            float(v[0]), float(v[1]), float(v[2]),
        )
        return self._write_vec3(out, xo, yo, zo)

    def transform_fixed_to_inertial(
        self,
        t_s: float,
        v_fixed: F64Array,
        *,
        out: Optional[F64Array] = None,
    ) -> F64Array:
        v = self._ensure_vec3(v_fixed, name="v_fixed")

        w, x, y, z = self._interp_quat_i2f(t_s)
        cw, cx, cy, cz = quat_conj(float(w), float(x), float(y), float(z))

        xo, yo, zo = quat_rotate_vec(
            cw, cx, cy, cz,
            float(v[0]), float(v[1]), float(v[2]),
        )
        return self._write_vec3(out, xo, yo, zo)



# =============================================================================
# 8.           NUMBA-FRIENDLY EPHEMERIS SAMPLER (CORE KERNEL API)
# =============================================================================

@njit(cache=True, nogil=True, inline="always")
def _row3(tab: np.ndarray, i: int) -> Tuple[float, float, float]:
    return float(tab[i, 0]), float(tab[i, 1]), float(tab[i, 2])


@njit(cache=True, nogil=True, inline="always")
def _row4(tab: np.ndarray, i: int) -> Tuple[float, float, float, float]:
    return float(tab[i, 0]), float(tab[i, 1]), float(tab[i, 2]), float(tab[i, 3])


@njit(cache=True, nogil=True, inline="always")
def _clamp_u_to_index_and_frac(u: float, n: int) -> Tuple[int, float]:
    """
    For a table of length n (n>=2), clamp u to [0, n-1] and return:
      i0 in [0, n-2] and frac f in [0,1], so that i0+1 is valid.
    """
    if u <= 0.0:
        return 0, 0.0
    umax = float(n - 1)
    if u >= umax:
        return n - 2, 1.0

    i0 = int(u)            # truncation == floor for u>=0
    if i0 > n - 2:
        i0 = n - 2
        return i0, 1.0
    f = u - float(i0)      # in [0,1)
    return i0, f


@njit(cache=True, nogil=True, inline="always")
def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f



@njit(cache=True, nogil=True)
def interp_vec3_safe(t_s: float, dt_s: float, tab: np.ndarray) -> Tuple[float, float, float]:
    """
    Safe vec3 interpolation:
      - n<=1 or dt<=0 : constant tab[0]
      - 2<=n<4        : linear (clamped)
      - n>=4          : Catmull-Rom (via interp_vec3_catmull)
    """
    n = int(tab.shape[0])

    if n <= 1 or dt_s <= 0.0:
        return _row3(tab, 0)

    if n < 4:
        u = t_s / dt_s
        i0, f = _clamp_u_to_index_and_frac(u, n)

        x0, y0, z0 = _row3(tab, i0)
        x1, y1, z1 = _row3(tab, i0 + 1)

        return _lerp(x0, x1, f), _lerp(y0, y1, f), _lerp(z0, z1, f)

    return interp_vec3_catmull(t_s, dt_s, tab)


@njit(cache=True, nogil=True)
def interp_quat_safe(t_s: float, dt_s: float, tab: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Safe quaternion interpolation:
      - n<=1 or dt<=0 : constant tab[0]
      - else          : SLERP (via interp_quat_slerp)
    """
    n = int(tab.shape[0])
    if n <= 1 or dt_s <= 0.0:
        return _row4(tab, 0)
    return interp_quat_slerp(t_s, dt_s, tab)


@njit(cache=True, nogil=True)
def get_ephem_state(
    t_s: float,
    dt_s: float,
    sun_tab_m: np.ndarray,
    earth_tab_m: np.ndarray,
    q_i2f_tab: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Unified, allocation-free ephemeris sampler for Numba loops.

    Returns:
      (sx, sy, sz, ex, ey, ez, qw, qx, qy, qz)
    """
    sx, sy, sz = interp_vec3_safe(t_s, dt_s, sun_tab_m)
    ex, ey, ez = interp_vec3_safe(t_s, dt_s, earth_tab_m)
    qw, qx, qy, qz = interp_quat_safe(t_s, dt_s, q_i2f_tab)
    return sx, sy, sz, ex, ey, ez, qw, qx, qy, qz



# =============================================================================
# 9.                          PUBLIC API
# =============================================================================

__all__ = (
    # --- Configuration & data structures ---
    "SpiceBuildConfig",     # Build recipe: which kernels/frames to load (time-agnostic)
    "EphemerisTables",      # Immutable precomputed tables (quats + Sun/Earth positions + GM)

    # --- Builders ---
    "build_spice_tables",   # Recommended builder: TimeConfig + SpiceBuildConfig -> EphemerisTables
    "build_tables",         # Low-level/legacy builder: explicit args -> EphemerisTables

    # --- Runtime manager ---
    "EphemerisManager",     # Runtime interface: interpolation, queries, and frame transforms

    # --- Low-level kernel (dynamics loop) ---
    "get_ephem_state",      # Numba-friendly, allocation-free ephemeris sampler for the integrator loop

    # --- Utilities (debug/tools) ---
    "resolve_kernel_paths", # Validates kernel paths and tries common extension fixes (e.g., .tls vs .tls.txt)
    "get_body_gm_m3s2",     # Reads GM from SPICE kernel pool; falls back to defaults (returns SI m^3/s^2)
)
