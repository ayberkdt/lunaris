# lunaris.batch.summary
"""
Summary-only ensemble output (roadmap R23).

Large screening runs (10^5 orbits) must not materialize or store the full
``(T, N, 6)`` trajectory tensor. This module reduces each sample's trajectory
to a fixed, versioned set of frozen-orbit screening metrics — orbital-element
envelopes, secular trends, apsidal (omega) behavior, impact bookkeeping, and a
screening score — plus a top-K selection whose *full* histories are worth
keeping for diagnostics.

Everything here is pure NumPy over one sub-batch at a time, so the engine can
stream: summarize each completed sub-batch, keep full history only for the
current top-K candidates, and drop the rest.

Schema discipline: consumers key off ``schema_version``; adding a field bumps
the version and this docstring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

BATCH_SUMMARY_SCHEMA_VERSION = 1

# Screening score v1 (lower = more frozen): eccentricity envelope width plus
# the periapsis-altitude envelope width normalized to 100 km, plus the 30-day
# projected periapsis drift normalized the same way. Impacted or invalid
# samples score +inf (they are never frozen candidates).
SCORE_DEFINITION = (
    "score = e_range + h_peri_range_km/100 + |trend_h_peri_km_per_day|*30/100; "
    "impacted/invalid samples = +inf; lower is better (screening v1)"
)

_DAY_S = 86_400.0


def _osculating_elements(
    r: np.ndarray, v: np.ndarray, mu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized osculating ``(a_m, e, i_rad, argp_rad)`` for ``(..., 3)`` states."""
    rn = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = 0.5 * v2 - mu / rn
        a = -mu / (2.0 * eps)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    e_vec = np.cross(v, h) / mu - r / rn[..., None]
    e = np.linalg.norm(e_vec, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = np.arccos(np.clip(h[..., 2] / np.where(hn > 0.0, hn, np.nan), -1.0, 1.0))
    # Node vector n = z x h; argp = angle(n -> e_vec) with quadrant from e_z.
    n_vec = np.stack([-h[..., 1], h[..., 0], np.zeros_like(hn)], axis=-1)
    nn = np.linalg.norm(n_vec, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_argp = np.sum(n_vec * e_vec, axis=-1) / (
            np.where(nn > 0.0, nn, np.nan) * np.where(e > 0.0, e, np.nan)
        )
    argp = np.arccos(np.clip(cos_argp, -1.0, 1.0))
    argp = np.where(e_vec[..., 2] < 0.0, 2.0 * np.pi - argp, argp)
    # Near-equatorial orbits have no ascending node (|n| ~ 0): fall back to the
    # longitude of periapsis so apsidal-drift screening still works there.
    equatorial = nn <= 1e-12 * np.maximum(hn, 1.0)
    lon_peri = np.mod(np.arctan2(e_vec[..., 1], e_vec[..., 0]), 2.0 * np.pi)
    argp = np.where(equatorial, lon_peri, argp)
    return a, e, inc, argp


def _linear_trend_per_day(t_s: np.ndarray, series: np.ndarray, valid: np.ndarray) -> float:
    """Least-squares slope of ``series`` vs time, in units/day; NaN if <2 points."""
    mask = valid & np.isfinite(series)
    if int(mask.sum()) < 2:
        return float("nan")
    tt = t_s[mask] / _DAY_S
    yy = series[mask]
    tm = tt - tt.mean()
    denom = float(np.sum(tm * tm))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(tm * (yy - yy.mean())) / denom)


def _omega_behavior(argp: np.ndarray, valid: np.ndarray) -> str:
    """Classify apsidal motion from the unwrapped argument of periapsis."""
    mask = valid & np.isfinite(argp)
    if int(mask.sum()) < 3:
        return "indeterminate"
    unwrapped = np.unwrap(argp[mask])
    span = float(unwrapped.max() - unwrapped.min())
    if span >= 2.0 * np.pi:
        return "circulating"
    if span <= np.pi:
        return "librating"
    return "mixed"


def summarize_ensemble(
    t_s: np.ndarray,
    Y: np.ndarray,
    impact_flags: np.ndarray,
    t_impact: np.ndarray,
    *,
    mu_m3s2: float,
    r_ref_m: float,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reduce an ensemble trajectory block ``(T, N, 6)`` to per-sample metrics.

    Post-impact snapshots (frozen surface states) are excluded from the
    element envelopes and trends; the impact itself is reported through the
    impact fields. Domain-exit fields are carried in the schema but resolved
    by the caller when a domain guard is active (NaN/False otherwise).
    """
    t_arr = np.asarray(t_s, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    n_t, n_samples = Y_arr.shape[0], Y_arr.shape[1]
    impacts = np.asarray(impact_flags, dtype=np.float64) > 0.5
    t_imp = np.asarray(t_impact, dtype=np.float64)
    sample_valid = (
        np.ones(n_samples, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=np.float64) > 0.5
    )

    a_m, e, inc, argp = _osculating_elements(Y_arr[:, :, :3], Y_arr[:, :, 3:], float(mu_m3s2))
    h_peri_km = (a_m * (1.0 - e) - float(r_ref_m)) / 1_000.0

    # Snapshot validity: finite AND pre-impact (frozen post-impact rows would
    # fake a "perfectly stable" orbit at the surface).
    snap_valid = np.isfinite(a_m) & np.isfinite(e) & np.isfinite(h_peri_km)
    pre_impact = np.ones((n_t, n_samples), dtype=bool)
    for j in np.nonzero(impacts & np.isfinite(t_imp))[0]:
        pre_impact[:, j] = t_arr <= float(t_imp[j])
    snap_valid &= pre_impact

    def _envelope(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masked = np.where(snap_valid, series, np.nan)
        with np.errstate(all="ignore"):
            lo = np.nanmin(masked, axis=0)
            hi = np.nanmax(masked, axis=0)
        return lo, hi, hi - lo

    e_min, e_max, e_range = _envelope(e)
    hp_min, hp_max, hp_range = _envelope(h_peri_km)

    trend_e = np.full(n_samples, np.nan, dtype=np.float64)
    trend_hp = np.full(n_samples, np.nan, dtype=np.float64)
    omega = np.array(["indeterminate"] * n_samples, dtype=object)
    for j in range(n_samples):
        col_valid = snap_valid[:, j]
        trend_e[j] = _linear_trend_per_day(t_arr, e[:, j], col_valid)
        trend_hp[j] = _linear_trend_per_day(t_arr, h_peri_km[:, j], col_valid)
        omega[j] = _omega_behavior(argp[:, j], col_valid)

    with np.errstate(invalid="ignore"):
        score = e_range + hp_range / 100.0 + np.abs(trend_hp) * 30.0 / 100.0
    score = np.where(impacts | ~sample_valid | ~np.isfinite(score), np.inf, score)

    def _row_elements(row: int) -> dict[str, np.ndarray]:
        return {
            "a_km": a_m[row] / 1_000.0,
            "e": e[row],
            "i_deg": np.degrees(inc[row]),
            "argp_deg": np.degrees(argp[row]),
            "h_peri_km": h_peri_km[row],
        }

    return {
        "schema_version": BATCH_SUMMARY_SCHEMA_VERSION,
        "score_definition": SCORE_DEFINITION,
        "n_samples": int(n_samples),
        "fields": {
            "initial_elements": _row_elements(0),
            "final_elements": _row_elements(n_t - 1),
            "e_min": e_min,
            "e_max": e_max,
            "e_range": e_range,
            "h_peri_min_km": hp_min,
            "h_peri_max_km": hp_max,
            "h_peri_range_km": hp_range,
            "trend_e_per_day": trend_e,
            "trend_h_peri_km_per_day": trend_hp,
            "omega_behavior": omega,
            "impact_flag": impacts.astype(np.float64),
            "t_impact_s": t_imp,
            "domain_exit_flag": np.zeros(n_samples, dtype=np.float64),
            "t_domain_exit_s": np.full(n_samples, np.nan, dtype=np.float64),
            "score": score,
            "validation_stage": np.array(["screened"] * n_samples, dtype=object),
            "valid": sample_valid.astype(np.float64),
        },
    }


def merge_summaries(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate per-sub-batch summaries (same schema) into one ensemble summary."""
    if not parts:
        raise ValueError("merge_summaries needs at least one part")
    versions = {int(p["schema_version"]) for p in parts}
    if versions != {BATCH_SUMMARY_SCHEMA_VERSION}:
        raise ValueError(f"summary schema mismatch across parts: {sorted(versions)}")
    merged_fields: dict[str, Any] = {}
    first_fields = parts[0]["fields"]
    for key, value in first_fields.items():
        if isinstance(value, dict):
            merged_fields[key] = {
                sub: np.concatenate([np.asarray(p["fields"][key][sub]) for p in parts])
                for sub in value
            }
        else:
            merged_fields[key] = np.concatenate([np.asarray(p["fields"][key]) for p in parts])
    return {
        "schema_version": BATCH_SUMMARY_SCHEMA_VERSION,
        "score_definition": SCORE_DEFINITION,
        "n_samples": int(sum(int(p["n_samples"]) for p in parts)),
        "fields": merged_fields,
    }


class TopKTrajectoryBuffer:
    """Streaming top-K retention of full trajectories by screening score.

    The engine feeds each completed sub-batch (summary scores + the block's
    full trajectories); only the current best ``k`` full histories are kept,
    so peak memory is one sub-batch plus ``k`` trajectories — never
    ``(T, N, 6)`` for the whole ensemble.
    """

    def __init__(self, k: int) -> None:
        self.k = max(1, int(k))
        self._entries: list[dict[str, Any]] = []

    def offer_batch(
        self,
        *,
        global_start: int,
        scores: np.ndarray,
        Y_batch: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
    ) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        order = np.argsort(scores, kind="stable")[: self.k]
        for j in order.tolist():
            if not np.isfinite(scores[j]):
                break  # sorted: everything after is +inf too
            self._entries.append(
                {
                    "sample_index": int(global_start + j),
                    "score": float(scores[j]),
                    "trajectory": np.ascontiguousarray(Y_batch[:, j, :], dtype=np.float64),
                    "impact_flag": float(impact_flags[j]),
                    "t_impact_s": float(t_impact[j]),
                }
            )
        self._entries.sort(key=lambda item: item["score"])
        del self._entries[self.k :]

    @property
    def selected_indices(self) -> list[int]:
        return [entry["sample_index"] for entry in self._entries]

    @property
    def scores(self) -> list[float]:
        return [entry["score"] for entry in self._entries]

    def stacked_trajectories(self, n_t: int) -> np.ndarray:
        """Return the retained histories as ``(T, K_kept, 6)`` (K_kept <= k)."""
        if not self._entries:
            return np.empty((n_t, 0, 6), dtype=np.float64)
        return np.stack([entry["trajectory"] for entry in self._entries], axis=1)

    def entry_arrays(self) -> dict[str, np.ndarray]:
        return {
            "sample_indices": np.asarray(self.selected_indices, dtype=np.int64),
            "scores": np.asarray(self.scores, dtype=np.float64),
            "impact_flags": np.asarray(
                [entry["impact_flag"] for entry in self._entries], dtype=np.float64
            ),
            "t_impact_s": np.asarray(
                [entry["t_impact_s"] for entry in self._entries], dtype=np.float64
            ),
        }


__all__ = [
    "BATCH_SUMMARY_SCHEMA_VERSION",
    "SCORE_DEFINITION",
    "TopKTrajectoryBuffer",
    "merge_summaries",
    "summarize_ensemble",
]
