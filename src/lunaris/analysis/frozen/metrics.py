"""Frozen-orbit screening metrics.

The routines here reduce one propagated candidate's element history to the
boundedness and secular-drift metrics used by the Sprint 5 frozen-orbit
classifier. They deliberately stay in ``lunaris.analysis`` and depend only on
NumPy so they can consume either CPU validation histories or summary/top-K
screening products.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True, slots=True)
class FrozenOrbitMetrics:
    """Scalar metrics for one frozen-orbit candidate trajectory."""

    duration_s: float
    sample_count: int
    valid_sample_count: int

    e_min: float
    e_max: float
    e_range: float
    de_dt_per_s: float

    h_peri_min_m: float
    h_peri_max_m: float
    h_peri_range_m: float
    dh_peri_dt_m_per_s: float

    inclination_min_rad: float
    inclination_max_rad: float
    inclination_range_rad: float
    dinclination_dt_rad_per_s: float

    omega_min_rad: float
    omega_max_rad: float
    omega_span_rad: float
    domega_dt_rad_per_s: float
    omega_behavior: str

    hk_loop_radius: float
    hk_loop_drift: float

    impact_time_s: float | None = None
    domain_exit_time_s: float | None = None
    escape: bool = False

    @property
    def has_impact(self) -> bool:
        return self.impact_time_s is not None

    @property
    def has_domain_exit(self) -> bool:
        return self.domain_exit_time_s is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_1d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape={arr.shape}")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    return arr


def _optional_time(value: float | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _envelope(series: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    vals = series[valid & np.isfinite(series)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    return lo, hi, hi - lo


def _linear_slope_per_s(t_s: np.ndarray, series: np.ndarray, valid: np.ndarray) -> float:
    mask = valid & np.isfinite(t_s) & np.isfinite(series)
    if int(mask.sum()) < 2:
        return float("nan")
    t = t_s[mask]
    y = series[mask]
    tc = t - float(np.mean(t))
    denom = float(np.sum(tc * tc))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(tc * (y - float(np.mean(y)))) / denom)


def _omega_metrics(
    t_s: np.ndarray,
    omega_rad: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, float, float, str]:
    mask = valid & np.isfinite(omega_rad)
    if int(mask.sum()) < 2:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            "indeterminate",
        )

    t = t_s[mask]
    unwrapped = np.unwrap(omega_rad[mask])
    lo = float(np.min(unwrapped))
    hi = float(np.max(unwrapped))
    span = hi - lo
    slope = _linear_slope_per_s(t, unwrapped, np.ones_like(unwrapped, dtype=bool))

    if int(mask.sum()) < 3:
        behavior = "indeterminate"
    elif span >= _TWO_PI:
        behavior = "circulation"
    elif span <= np.pi:
        behavior = "libration"
    else:
        behavior = "mixed"
    return lo, hi, float(span), slope, behavior


def _hk_loop_metrics(
    eccentricity: np.ndarray,
    omega_rad: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    mask = valid & np.isfinite(eccentricity) & np.isfinite(omega_rad)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    e = eccentricity[mask]
    omega = omega_rad[mask]
    h_vec = e * np.sin(omega)
    k_vec = e * np.cos(omega)
    h_center = float(np.mean(h_vec))
    k_center = float(np.mean(k_vec))
    radius = np.sqrt((h_vec - h_center) ** 2 + (k_vec - k_center) ** 2)
    drift = float(np.hypot(h_vec[-1] - h_vec[0], k_vec[-1] - k_vec[0]))
    return float(np.max(radius)), drift


def compute_frozen_metrics(
    t_s: Any,
    *,
    eccentricity: Any,
    inclination_rad: Any,
    omega_rad: Any,
    h_peri_m: Any | None = None,
    semi_major_axis_m: Any | None = None,
    reference_radius_m: float | None = None,
    impact_time_s: float | None = None,
    domain_exit_time_s: float | None = None,
    escape: bool = False,
    valid_mask: Any | None = None,
) -> FrozenOrbitMetrics:
    """Compute frozen-orbit metrics from osculating element histories.

    Parameters are SI/radian unless the name states otherwise. ``h_peri_m`` is
    the perilune altitude above the reference surface. If it is omitted, callers
    must provide ``semi_major_axis_m`` and ``reference_radius_m`` so it can be
    derived as ``a * (1 - e) - R_ref``.
    """

    t = _as_1d_float("t_s", t_s)
    e = _as_1d_float("eccentricity", eccentricity)
    inc = _as_1d_float("inclination_rad", inclination_rad)
    omega = _as_1d_float("omega_rad", omega_rad)
    n = int(t.size)
    if e.size != n or inc.size != n or omega.size != n:
        raise ValueError("t_s, eccentricity, inclination_rad, and omega_rad must match")
    if not np.all(np.isfinite(t)):
        raise ValueError("t_s must be finite")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("t_s must be monotonically nondecreasing")
    duration_s = float(t[-1] - t[0])
    if duration_s <= 0.0:
        raise ValueError("t_s must span a positive duration")

    if h_peri_m is None:
        if semi_major_axis_m is None or reference_radius_m is None:
            raise ValueError(
                "provide h_peri_m or both semi_major_axis_m and reference_radius_m"
            )
        a = _as_1d_float("semi_major_axis_m", semi_major_axis_m)
        if a.size != n:
            raise ValueError("semi_major_axis_m must match t_s")
        h_peri = a * (1.0 - e) - float(reference_radius_m)
    else:
        h_peri = _as_1d_float("h_peri_m", h_peri_m)
        if h_peri.size != n:
            raise ValueError("h_peri_m must match t_s")

    valid = np.isfinite(e) & np.isfinite(h_peri) & np.isfinite(inc) & np.isfinite(omega)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim != 1 or mask.size != n:
            raise ValueError("valid_mask must be a 1D array matching t_s")
        valid &= mask

    impact_time = _optional_time(impact_time_s)
    domain_exit_time = _optional_time(domain_exit_time_s)
    if impact_time is not None:
        valid &= t <= impact_time
    if domain_exit_time is not None:
        valid &= t <= domain_exit_time

    e_min, e_max, e_range = _envelope(e, valid)
    hp_min, hp_max, hp_range = _envelope(h_peri, valid)
    inc_min, inc_max, inc_range = _envelope(inc, valid)
    omega_min, omega_max, omega_span, domega_dt, omega_behavior = _omega_metrics(
        t, omega, valid
    )
    hk_radius, hk_drift = _hk_loop_metrics(e, omega, valid)

    return FrozenOrbitMetrics(
        duration_s=duration_s,
        sample_count=n,
        valid_sample_count=int(valid.sum()),
        e_min=e_min,
        e_max=e_max,
        e_range=e_range,
        de_dt_per_s=_linear_slope_per_s(t, e, valid),
        h_peri_min_m=hp_min,
        h_peri_max_m=hp_max,
        h_peri_range_m=hp_range,
        dh_peri_dt_m_per_s=_linear_slope_per_s(t, h_peri, valid),
        inclination_min_rad=inc_min,
        inclination_max_rad=inc_max,
        inclination_range_rad=inc_range,
        dinclination_dt_rad_per_s=_linear_slope_per_s(t, inc, valid),
        omega_min_rad=omega_min,
        omega_max_rad=omega_max,
        omega_span_rad=omega_span,
        domega_dt_rad_per_s=domega_dt,
        omega_behavior=omega_behavior,
        hk_loop_radius=hk_radius,
        hk_loop_drift=hk_drift,
        impact_time_s=impact_time,
        domain_exit_time_s=domain_exit_time,
        escape=bool(escape),
    )


__all__ = ["FrozenOrbitMetrics", "compute_frozen_metrics"]
