"""Mandatory domain guard for the frozen-orbit search (roadmap R27).

The surrogate screening stage is only trustworthy inside the model's training
domain (altitude envelope) and inside basic physical sanity bounds. This module
evaluates those bounds over screening trajectories and enforces two rules:

1. A sample that leaves the domain is *invalid* (score ``+inf``) or
   *low-confidence* (score penalty), per configured policy — never silently a
   good candidate.
2. A domain-exited trajectory can never be promoted to a final candidate
   (:func:`assert_candidate_domain_clean` is the choke point the pipeline calls
   before validation/refinement).

Everything is pure NumPy over ``(T, N, 6)`` blocks so it works on summary-mode
top-K products and on full screening blocks alike.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DOMAIN_POLICY_INVALID = "invalid"
DOMAIN_POLICY_LOW_CONFIDENCE = "low_confidence"
_POLICIES = (DOMAIN_POLICY_INVALID, DOMAIN_POLICY_LOW_CONFIDENCE)


@dataclass(frozen=True, slots=True)
class FrozenSearchDomainGuard:
    """Domain envelope + failure policy for surrogate-assisted screening.

    ``altitude_min_km`` / ``altitude_max_km`` bound the screening model's valid
    altitude band above ``reference_radius_m`` (for ST-LRPS these must match the
    artifact's training envelope). ``escape_radius_km`` is a radial escape
    proxy. ``policy`` decides what a domain exit does to the screening score:
    ``"invalid"`` forces ``+inf``; ``"low_confidence"`` adds
    ``domain_exit_penalty`` and keeps the candidate rankable but flagged.
    """

    altitude_min_km: float
    altitude_max_km: float
    escape_radius_km: float = 100_000.0
    domain_exit_penalty: float = 1_000.0
    policy: str = DOMAIN_POLICY_INVALID

    def __post_init__(self) -> None:
        lo = float(self.altitude_min_km)
        hi = float(self.altitude_max_km)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError(
                f"altitude envelope must satisfy min < max, got [{lo}, {hi}] km"
            )
        if not np.isfinite(self.escape_radius_km) or self.escape_radius_km <= 0.0:
            raise ValueError("escape_radius_km must be positive and finite")
        if not np.isfinite(self.domain_exit_penalty) or self.domain_exit_penalty < 0.0:
            raise ValueError("domain_exit_penalty must be non-negative and finite")
        if self.policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}, got {self.policy!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DomainGuardResult:
    """Per-sample domain-guard outcome for one screening block."""

    domain_exit_flag: np.ndarray  # (N,) bool
    t_domain_exit_s: np.ndarray   # (N,) float64, NaN when no exit
    escape_flag: np.ndarray       # (N,) bool
    nonfinite_flag: np.ndarray    # (N,) bool
    reasons: np.ndarray           # (N,) object ("" when clean)

    @property
    def n_exits(self) -> int:
        return int(np.count_nonzero(self.domain_exit_flag))

    def sample_metadata(self, j: int) -> dict[str, Any]:
        """JSON-ready per-sample guard metadata (pipeline candidate records)."""
        exit_t = float(self.t_domain_exit_s[j])
        return {
            "domain_exit": bool(self.domain_exit_flag[j]),
            "t_domain_exit_s": exit_t if np.isfinite(exit_t) else None,
            "escape": bool(self.escape_flag[j]),
            "nonfinite_state": bool(self.nonfinite_flag[j]),
            "domain_exit_reason": str(self.reasons[j]),
        }


def evaluate_domain_guard(
    t_s: Any,
    Y: Any,
    *,
    reference_radius_m: float,
    guard: FrozenSearchDomainGuard,
    impact_flags: Any | None = None,
    t_impact_s: Any | None = None,
) -> DomainGuardResult:
    """Evaluate the domain guard over a ``(T, N, 6)`` trajectory block.

    Post-impact snapshots are excluded (the impact itself is bookkept by the
    propagation loop, not double-counted as a domain exit). The first violating
    snapshot time is reported per sample.
    """
    t_arr = np.asarray(t_s, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    if Y_arr.ndim != 3 or Y_arr.shape[2] < 6:
        raise ValueError(f"Y must be (T, N, 6), got shape {Y_arr.shape}")
    n_t, n_samples = Y_arr.shape[0], Y_arr.shape[1]
    if t_arr.ndim != 1 or t_arr.size != n_t:
        raise ValueError("t_s must be 1D and match Y's time axis")
    r_ref = float(reference_radius_m)
    if not np.isfinite(r_ref) or r_ref <= 0.0:
        raise ValueError("reference_radius_m must be positive and finite")

    considered = np.ones((n_t, n_samples), dtype=bool)
    if impact_flags is not None and t_impact_s is not None:
        impacts = np.asarray(impact_flags, dtype=np.float64) > 0.5
        t_imp = np.asarray(t_impact_s, dtype=np.float64)
        for j in np.nonzero(impacts & np.isfinite(t_imp))[0]:
            considered[:, j] = t_arr <= float(t_imp[j])

    finite = np.all(np.isfinite(Y_arr), axis=2)  # (T, N)
    radius_m = np.linalg.norm(Y_arr[:, :, :3], axis=2)
    with np.errstate(invalid="ignore"):
        alt_km = (radius_m - r_ref) / 1_000.0
        below = alt_km < float(guard.altitude_min_km)
        above = alt_km > float(guard.altitude_max_km)
        escaped = radius_m > float(guard.escape_radius_km) * 1_000.0

    nonfinite_viol = considered & ~finite
    below_viol = considered & finite & below
    above_viol = considered & finite & above
    escape_viol = considered & finite & escaped
    any_viol = nonfinite_viol | below_viol | above_viol | escape_viol

    exit_flag = np.any(any_viol, axis=0)
    escape_flag = np.any(escape_viol, axis=0)
    nonfinite_flag = np.any(nonfinite_viol, axis=0)
    t_exit = np.full(n_samples, np.nan, dtype=np.float64)
    reasons = np.array([""] * n_samples, dtype=object)
    for j in np.nonzero(exit_flag)[0]:
        first = int(np.argmax(any_viol[:, j]))
        t_exit[j] = float(t_arr[first])
        if nonfinite_viol[first, j]:
            reasons[j] = "non-finite state"
        elif escape_viol[first, j]:
            reasons[j] = "escape radius exceeded"
        elif below_viol[first, j]:
            reasons[j] = "altitude below domain envelope"
        else:
            reasons[j] = "altitude above domain envelope"

    return DomainGuardResult(
        domain_exit_flag=exit_flag,
        t_domain_exit_s=t_exit,
        escape_flag=escape_flag,
        nonfinite_flag=nonfinite_flag,
        reasons=reasons,
    )


def apply_domain_guard_to_scores(
    scores: Any,
    result: DomainGuardResult,
    guard: FrozenSearchDomainGuard,
) -> np.ndarray:
    """Return screening scores with the guard's policy applied.

    ``invalid`` policy: domain-exited samples score ``+inf`` (never candidates).
    ``low_confidence`` policy: ``score + domain_exit_penalty`` (rankable but
    dominated by clean candidates of comparable quality).
    """
    penalized = np.asarray(scores, dtype=np.float64).copy()
    exits = np.asarray(result.domain_exit_flag, dtype=bool)
    if penalized.shape != exits.shape:
        raise ValueError(
            f"scores shape {penalized.shape} != guard result shape {exits.shape}"
        )
    if guard.policy == DOMAIN_POLICY_INVALID:
        penalized[exits] = np.inf
    else:
        penalized[exits] += float(guard.domain_exit_penalty)
    return penalized


def assert_candidate_domain_clean(candidate: dict[str, Any]) -> None:
    """Enforce R27: a domain-exited trajectory can never be a final candidate.

    ``candidate`` is a pipeline candidate record whose ``domain_guard`` block
    came from :meth:`DomainGuardResult.sample_metadata`. Raises ``RuntimeError``
    — this is a hard rule, not a warning.
    """
    block = candidate.get("domain_guard")
    if block is None:
        raise RuntimeError(
            "candidate has no domain_guard block; the frozen search requires the "
            "domain guard to run on every screening trajectory (R27)"
        )
    if bool(block.get("domain_exit", False)):
        raise RuntimeError(
            "domain-exited trajectory cannot be promoted to a final candidate "
            f"(reason: {block.get('domain_exit_reason', 'unknown')!r}); "
            "re-screen inside the model domain or validate with classical SH"
        )


__all__ = [
    "DOMAIN_POLICY_INVALID",
    "DOMAIN_POLICY_LOW_CONFIDENCE",
    "DomainGuardResult",
    "FrozenSearchDomainGuard",
    "apply_domain_guard_to_scores",
    "assert_candidate_domain_clean",
    "evaluate_domain_guard",
]
