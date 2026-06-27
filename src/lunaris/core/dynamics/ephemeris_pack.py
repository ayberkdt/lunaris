"""Validated ephemeris data pack for dynamics RHS construction."""

from __future__ import annotations

from dataclasses import dataclass

from lunaris.common.type_defs import F64Array
from lunaris.core.dynamics.requirements import _as_f64_c

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

__all__ = ["_EphemPack"]
