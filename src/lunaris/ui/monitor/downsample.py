"""Display-resolution downsampling for monitor time series.

The store keeps full-cadence samples (bounded ring); widgets render at most a
few thousand points. Plain striding can silently drop short-lived spikes
(e.g. a single low-periapsis dip), so the scalar path uses a *bucketed min/max
envelope*: the index range is split into buckets and each bucket contributes
its minimum and maximum sample, preserving every extreme the display could
possibly show. Endpoints are always kept so the "current value" is never
downsampled away.
"""

from __future__ import annotations

import itertools

import numpy as np

#: Smallest sensible display budget (first/last + one min/max bucket).
_MIN_POINTS = 4


def envelope_downsample(
    t: np.ndarray,
    v: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``(t, v)`` to at most ``max_points`` points, keeping extrema.

    Both arrays must be 1-D and equally long, with ``t`` non-decreasing.
    Returns copies; the inputs are never modified. When the series already
    fits the budget it is returned as (copied) is.
    """
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if t.ndim != 1 or v.ndim != 1 or t.shape[0] != v.shape[0]:
        raise ValueError(f"t/v must be equal-length 1-D arrays, got {t.shape} / {v.shape}")
    n = int(t.shape[0])
    budget = max(int(max_points), _MIN_POINTS)
    if n <= budget:
        return t.copy(), v.copy()

    # Reserve the two endpoints; every remaining pair of slots is one bucket.
    n_buckets = max(1, (budget - 2) // 2)
    edges = np.linspace(0, n, n_buckets + 1).astype(np.int64)

    keep: list[int] = [0, n - 1]
    for start, stop in itertools.pairwise(edges):
        if stop <= start:
            continue
        segment = v[start:stop]
        keep.append(int(start + np.argmin(segment)))
        keep.append(int(start + np.argmax(segment)))
    idx = np.unique(np.asarray(keep, dtype=np.int64))
    return t[idx], v[idx]


def decimate_indices(n: int, max_points: int) -> np.ndarray:
    """Uniform index decimation that always keeps the first and last element.

    Used for vector-valued series (3-D trajectories) where a min/max envelope
    has no meaning. Returns sorted unique indices into a length-``n`` array.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    budget = max(int(max_points), 2)
    if n <= budget:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, budget).round().astype(np.int64)
    return np.unique(idx)


__all__ = ["decimate_indices", "envelope_downsample"]
