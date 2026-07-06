# st_lrps/evaluation/phase_diagnostics.py
"""
Phase-drift diagnostics for surrogate-vs-truth trajectory comparisons.

Answers one question with evidence instead of a single Cartesian RMS number:
*is the surrogate's position error a genuine orbit-shape error, or is the
surrogate flying the right orbit slightly ahead/behind in time (phase drift)?*

Diagnostic chain (each layer feeds the next):

1. **RIC error decomposition** — split ``e(t) = r_st(t) - r_gt(t)`` into
   radial / along-track / cross-track components of the ground-truth RIC
   frame (:func:`compute_ric_error_history`).
2. **Phase-lag estimation** — the first-order optimal local time shift
   ``tau(t) = e(t)·v_gt(t) / |v_gt(t)|²`` and a low-order polynomial fit of
   its secular trend (:func:`estimate_phase_lag`).
3. **Phase-corrected residuals** — what remains after (a) removing the
   instantaneous along-velocity projection (upper bound of the phase
   explanation) and (b) re-sampling the truth at ``t + tau_fit(t)`` with a
   degree ≤ 2 polynomial shift, i.e. at most 3 fitted scalars
   (:func:`compute_phase_corrected_errors`).
4. **Tangential-bias diagnosis** — signed RIC decomposition of the
   model-minus-truth acceleration error along the truth trajectory, its
   orbit-averaged mean, and the measured osculating semi-major-axis drift
   (:func:`diagnose_tangential_bias`).
5. **Gauss-VE phase prediction** — the *causal* test: integrate the Gauss
   variational equations with the measured tangential acceleration bias and
   compare the predicted phase lag against the measured one
   (:func:`predict_phase_drift_from_tangential_bias`).
6. **UQ alignment** — how well the surrogate error vector aligns with the
   leading eigenvector of a reference-ensemble position covariance
   (:func:`compute_uq_alignment`; see :data:`UQ_ALIGNMENT_CAVEAT` for the
   claim this metric can and cannot support).

:func:`compute_phase_diagnostics` runs the full chain and returns a
:class:`PhaseDiagnosticsReport` whose :meth:`~PhaseDiagnosticsReport.summary`
is a flat scalar dict ready for a per-scenario CSV row.
:func:`phase_scenario_metrics` is the benchmark-harness adapter producing the
``scenario_results.csv`` phase columns (:data:`PHASE_SCENARIO_COLUMNS`).

Conventions (used consistently everywhere in this module)
----------------------------------------------------------
- Frame: Moon-centred inertial integration frame (same frame the benchmark
  trajectories are produced in). RIC component order is
  ``[radial, along, cross]``, identical to the orbit-benchmark decomposition
  and to :func:`lunaris.analysis.ensemble.statistics.ric_basis_from_state`.
- Units: SI throughout (m, s, m/s, m/s²). Summary keys carry unit suffixes;
  acceleration biases are additionally reported in mGal (1 mGal = 1e-5 m/s²).
- Error sign: ``e(t) = r_st(t) - r_gt(t)`` (surrogate minus truth).
- Phase-lag sign: ``tau(t) > 0`` means the surrogate *leads* the truth,
  i.e. ``r_st(t) ≈ r_gt(t + tau)``. A positive mean tangential bias
  (surrogate over-accelerates along the velocity) raises the semi-major
  axis, lowers the mean motion, and therefore produces ``tau < 0``
  (surrogate falls behind) — the sign the Gauss-VE prediction reproduces.

Near-circular caveat: the ``delta_n -> delta_M -> tau`` step of the
prediction uses the near-circular relations ``delta_n = -(3/2)(n/a) delta_a``
and ``e_along ≈ a·delta_M``; the semi-major-axis rate itself uses the exact
tangential Gauss VE ``da/dt = 2 a² v Δa_t / mu``. Short-period transients at
the orbital frequency are intentionally not modelled — the comparison targets
the secular drift.

All computations are pure NumPy; importing this module never pulls torch,
matplotlib, or the surrogate runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunaris.common.constants import DAY_S, MU_MOON
from lunaris.common.type_defs import F64Array

M_S2_PER_MGAL = 1.0e-5


def _ric_basis(r_m: F64Array, v_ms: F64Array) -> F64Array:
    """Per-epoch RIC basis (rows [r̂; î; ĉ]) of the given state history.

    Delegates to the canonical implementation in
    :func:`lunaris.analysis.ensemble.statistics.ric_basis_from_state`. Imported
    lazily because that module pulls ``lunaris.core.state`` (and with it
    numba), which must not become an import-time dependency of this pure-NumPy
    diagnostics module.
    """
    from lunaris.analysis.ensemble.statistics import ric_basis_from_state

    return ric_basis_from_state(np.hstack([r_m, v_ms]))

# =============================================================================
# 1.              INPUT VALIDATION AND SMALL NUMERIC HELPERS
# =============================================================================


def _as_f64(a: object, name: str, shape_desc: str) -> F64Array:
    arr = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if shape_desc == "(T,)" and arr.ndim != 1:
        raise ValueError(f"{name} must be (T,), got {arr.shape}")
    if shape_desc == "(T,3)" and (arr.ndim != 2 or arr.shape[1] != 3):
        raise ValueError(f"{name} must be (T, 3), got {arr.shape}")
    return arr


def _validate_trajectories(
    t_s: object,
    *named_vectors: tuple[str, object],
) -> tuple[F64Array, list[F64Array]]:
    """Validate a shared strictly-increasing time grid and (T, 3) vector series."""
    t = _as_f64(t_s, "t_s", "(T,)")
    if t.size < 4:
        raise ValueError(f"t_s must have at least 4 epochs, got {t.size}")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("t_s must be strictly increasing")
    out: list[F64Array] = []
    for name, vec in named_vectors:
        arr = _as_f64(vec, name, "(T,3)")
        if arr.shape[0] != t.size:
            raise ValueError(f"{name} has {arr.shape[0]} epochs but t_s has {t.size}")
        out.append(arr)
    return t, out


def _cumtrapz0(y: F64Array, t: F64Array) -> F64Array:
    """Cumulative trapezoid integral of ``y(t)`` with a leading zero (same length)."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(t)
    out = np.empty_like(y)
    out[0] = 0.0
    np.cumsum(seg, out=out[1:])
    return out


def _rms(x: F64Array) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def osculating_sma(r_m: F64Array, v_ms: F64Array, mu_m3_s2: float) -> F64Array:
    """Osculating semi-major axis history [m] via vis-viva: ``a = 1/(2/r - v²/mu)``."""
    r = np.linalg.norm(r_m, axis=1)
    v2 = np.einsum("ij,ij->i", v_ms, v_ms)
    inv_a = 2.0 / r - v2 / mu_m3_s2
    if np.any(inv_a <= 0.0):
        raise ValueError("osculating_sma: trajectory contains non-elliptic states (1/a <= 0)")
    return 1.0 / inv_a


def hermite_resample(
    t_s: F64Array,
    r_m: F64Array,
    v_ms: F64Array,
    t_query_s: F64Array,
) -> tuple[F64Array, F64Array]:
    """
    Cubic-Hermite position resampling of a trajectory at arbitrary query times.

    Uses position *and* velocity at the bracketing grid points, so the local
    interpolation error is O(dt⁴) — small enough that the phase-aligned
    residual is not contaminated by interpolation error at typical benchmark
    output rates.

    Returns
    -------
    (r_query, valid) : positions (Q, 3) and a boolean mask (Q,). Entries with
    ``t_query`` outside ``[t_s[0], t_s[-1]]`` are marked invalid and hold the
    nearest-endpoint position (callers must exclude them from statistics).
    """
    tq = np.asarray(t_query_s, dtype=np.float64)
    valid = (tq >= t_s[0]) & (tq <= t_s[-1])
    tq_c = np.clip(tq, t_s[0], t_s[-1])

    idx = np.clip(np.searchsorted(t_s, tq_c, side="right") - 1, 0, t_s.size - 2)
    h = t_s[idx + 1] - t_s[idx]
    s = ((tq_c - t_s[idx]) / h)[:, None]  # normalized abscissa in [0, 1]

    r0, r1 = r_m[idx], r_m[idx + 1]
    m0, m1 = v_ms[idx] * h[:, None], v_ms[idx + 1] * h[:, None]

    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    r_query = h00 * r0 + h10 * m0 + h01 * r1 + h11 * m1
    return r_query, valid


# =============================================================================
# 2.              LAYER 1 — RIC ERROR DECOMPOSITION
# =============================================================================


@dataclass(frozen=True)
class RICErrorHistory:
    """
    Position error history in the ground-truth RIC frame.

    Component order is ``[radial, along, cross]``. ``ms_fraction`` is each
    component's share of the total mean-square position error (sums to 1),
    i.e. the number behind statements like "92% of the error is along-track".
    """

    t_s:          F64Array   # (T,)
    e_ric_m:      F64Array   # (T, 3) signed components
    rms_ric_m:    F64Array   # (3,)
    max_abs_ric_m: F64Array  # (3,)
    ms_fraction:  F64Array   # (3,)


def compute_ric_error_history(
    t_s: object,
    r_st_m: object,
    r_gt_m: object,
    v_gt_ms: object,
) -> RICErrorHistory:
    """Decompose ``r_st - r_gt`` into the RIC frame of the ground-truth state."""
    t, (r_st, r_gt, v_gt) = _validate_trajectories(
        t_s, ("r_st_m", r_st_m), ("r_gt_m", r_gt_m), ("v_gt_ms", v_gt_ms)
    )
    basis = _ric_basis(r_gt, v_gt)  # (T, 3, 3), rows [r̂; î; ĉ]
    e = r_st - r_gt
    e_ric = np.einsum("tij,tj->ti", basis, e)

    ms_total = float(np.mean(np.einsum("ij,ij->i", e, e)))
    ms_comp = np.mean(np.square(e_ric), axis=0)
    frac = ms_comp / ms_total if ms_total > 0.0 else np.zeros(3)
    return RICErrorHistory(
        t_s=t,
        e_ric_m=e_ric,
        rms_ric_m=np.sqrt(ms_comp),
        max_abs_ric_m=np.max(np.abs(e_ric), axis=0),
        ms_fraction=frac,
    )


# =============================================================================
# 3.              LAYER 2 — PHASE-LAG ESTIMATION AND CORRECTION
# =============================================================================


@dataclass(frozen=True)
class PhaseLagEstimate:
    """
    First-order phase lag ``tau(t) = e·v_gt/|v_gt|²`` and its secular fit.

    ``tau > 0`` means the surrogate leads the truth: ``r_st(t) ≈ r_gt(t+tau)``.
    ``tau_fit_coeffs`` are polynomial coefficients over ``(t - t[0])`` in
    ascending order (``numpy.polynomial.polynomial`` convention). A constant
    tangential acceleration bias produces quadratic phase growth, hence the
    default fit degree of 2 — still only 3 fitted scalars, so the fit cannot
    absorb genuine orbit-shape error.
    """

    t_s:            F64Array  # (T,)
    tau_s:          F64Array  # (T,)
    tau_final_s:    float
    tau_abs_max_s:  float
    tau_fit_coeffs: F64Array  # (degree+1,), ascending powers of (t - t[0])
    tau_fit_degree: int
    tau_fit_rms_residual_s: float

    def tau_fit_s(self, t_s: F64Array) -> F64Array:
        """Evaluate the fitted secular phase-lag model at times ``t_s``."""
        return np.polynomial.polynomial.polyval(
            np.asarray(t_s, dtype=np.float64) - self.t_s[0], self.tau_fit_coeffs
        )


def estimate_phase_lag(
    t_s: object,
    r_st_m: object,
    r_gt_m: object,
    v_gt_ms: object,
    *,
    fit_degree: int = 2,
) -> PhaseLagEstimate:
    """Estimate the instantaneous phase lag and fit its secular trend."""
    if not 0 <= int(fit_degree) <= 3:
        raise ValueError(f"fit_degree must be in [0, 3], got {fit_degree}")
    t, (r_st, r_gt, v_gt) = _validate_trajectories(
        t_s, ("r_st_m", r_st_m), ("r_gt_m", r_gt_m), ("v_gt_ms", v_gt_ms)
    )
    e = r_st - r_gt
    v2 = np.einsum("ij,ij->i", v_gt, v_gt)
    if np.any(v2 <= 0.0):
        raise ValueError("v_gt_ms contains zero-velocity states; phase lag is undefined")
    tau = np.einsum("ij,ij->i", e, v_gt) / v2

    coeffs = np.polynomial.polynomial.polyfit(t - t[0], tau, deg=int(fit_degree))
    residual = tau - np.polynomial.polynomial.polyval(t - t[0], coeffs)
    return PhaseLagEstimate(
        t_s=t,
        tau_s=tau,
        tau_final_s=float(tau[-1]),
        tau_abs_max_s=float(np.max(np.abs(tau))),
        tau_fit_coeffs=np.asarray(coeffs, dtype=np.float64),
        tau_fit_degree=int(fit_degree),
        tau_fit_rms_residual_s=_rms(residual),
    )


@dataclass(frozen=True)
class PhaseCorrectedErrors:
    """
    Residual position error after removing the phase component, two ways.

    ``pointwise_detrended_rms_m`` removes the instantaneous along-velocity
    projection at every epoch (T degrees of freedom): it is the *upper bound*
    of what a phase explanation could ever account for, never a performance
    claim. ``aligned_rms_m`` re-samples the truth at ``t + tau_fit(t)`` using
    the ≤ 3-parameter secular fit: the defensible "error is phase-dominated"
    number. Both are diagnostics computed against ground truth — neither may
    be reported as model accuracy.
    """

    t_s:                  F64Array
    raw_rms_m:            float
    final_pos_err_m:      float
    pointwise_detrended_rms_m: float
    aligned_rms_m:        float
    aligned_valid_fraction: float
    phase_explained_fraction: float   # 1 - aligned_ms / raw_ms, clipped to [0, 1]
    e_aligned_m:          F64Array    # (T, 3); invalid epochs hold NaN
    ric_rms_before_m:     F64Array    # (3,)
    ric_rms_after_m:      F64Array    # (3,) over valid epochs


def compute_phase_corrected_errors(
    t_s: object,
    r_st_m: object,
    r_gt_m: object,
    v_gt_ms: object,
    *,
    fit_degree: int = 2,
) -> tuple[PhaseLagEstimate, PhaseCorrectedErrors]:
    """
    Estimate the phase lag, then measure what error survives its removal.

    Returns the :class:`PhaseLagEstimate` together with the corrected-error
    figures so callers never pair a correction with a mismatched fit.
    """
    t, (r_st, r_gt, v_gt) = _validate_trajectories(
        t_s, ("r_st_m", r_st_m), ("r_gt_m", r_gt_m), ("v_gt_ms", v_gt_ms)
    )
    phase = estimate_phase_lag(t, r_st, r_gt, v_gt, fit_degree=fit_degree)
    ric = compute_ric_error_history(t, r_st, r_gt, v_gt)

    e = r_st - r_gt
    raw_ms = float(np.mean(np.einsum("ij,ij->i", e, e)))
    raw_rms = float(np.sqrt(raw_ms))

    # (a) pointwise: remove the along-velocity projection at every epoch.
    v_hat = v_gt / np.linalg.norm(v_gt, axis=1, keepdims=True)
    e_perp = e - np.einsum("ij,ij->i", e, v_hat)[:, None] * v_hat
    pointwise_rms = _rms(np.linalg.norm(e_perp, axis=1))

    # (b) aligned: compare against the truth re-sampled at t + tau_fit(t).
    tau_model = phase.tau_fit_s(t)
    r_gt_shifted, valid = hermite_resample(t, r_gt, v_gt, t + tau_model)
    e_aligned = np.full_like(e, np.nan)
    e_aligned[valid] = r_st[valid] - r_gt_shifted[valid]
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError("phase alignment left no valid epochs; time grid too short")
    aligned_ms = float(np.mean(np.einsum("ij,ij->i", e_aligned[valid], e_aligned[valid])))

    basis = _ric_basis(r_gt, v_gt)
    e_aligned_ric = np.einsum("tij,tj->ti", basis[valid], e_aligned[valid])

    return phase, PhaseCorrectedErrors(
        t_s=t,
        raw_rms_m=raw_rms,
        final_pos_err_m=float(np.linalg.norm(e[-1])),
        pointwise_detrended_rms_m=pointwise_rms,
        aligned_rms_m=float(np.sqrt(aligned_ms)),
        aligned_valid_fraction=n_valid / t.size,
        phase_explained_fraction=float(np.clip(1.0 - aligned_ms / raw_ms, 0.0, 1.0))
        if raw_ms > 0.0
        else 0.0,
        e_aligned_m=e_aligned,
        ric_rms_before_m=ric.rms_ric_m,
        ric_rms_after_m=np.sqrt(np.mean(np.square(e_aligned_ric), axis=0)),
    )


# =============================================================================
# 4.              LAYER 3 — TANGENTIAL-BIAS DIAGNOSIS (Faz 2)
# =============================================================================


@dataclass(frozen=True)
class TangentialBiasDiagnosis:
    """
    Signed decomposition of the acceleration error along the truth trajectory.

    ``da_ric_m_s2`` uses the RIC axes; ``da_tangential_m_s2`` is the
    projection onto the velocity direction ``v̂`` — the component that pumps
    orbital energy and therefore drives phase drift (identical to the RIC
    along-track component only for circular orbits). Means are *signed*: an
    RMS hides exactly the systematic bias this layer exists to expose.

    ``orbit_mean_da_tangential_m_s2`` is a centred rolling mean over one
    osculating orbital period; ``orbit_mean_valid`` marks epochs whose window
    lies fully inside the data span. ``delta_sma_measured_m`` is the measured
    osculating semi-major-axis difference (surrogate minus truth) — the
    intermediate link of the causal chain Δa_T → δa → δn → τ.
    """

    t_s:                  F64Array   # (T,)
    da_ric_m_s2:          F64Array   # (T, 3) signed [radial, along, cross]
    da_tangential_m_s2:   F64Array   # (T,) signed Δa·v̂
    mean_da_ric_m_s2:     F64Array   # (3,) signed time means
    mean_da_tangential_m_s2: float
    rms_da_m_s2:          float
    orbit_mean_da_tangential_m_s2: F64Array  # (T,)
    orbit_mean_valid:     F64Array   # (T,) bool
    delta_sma_measured_m: F64Array   # (T,)


def diagnose_tangential_bias(
    t_s: object,
    r_gt_m: object,
    v_gt_ms: object,
    da_m_s2: object,
    *,
    r_st_m: object | None = None,
    v_st_ms: object | None = None,
    mu_m3_s2: float = MU_MOON,
) -> TangentialBiasDiagnosis:
    """
    Diagnose systematic acceleration bias from ``da = a_model - a_truth``.

    ``da_m_s2`` must be evaluated along the ground-truth trajectory (model
    minus truth at ``r_gt``), which is first-order equivalent to the bias
    acting on the surrogate trajectory. When the surrogate state history is
    provided, the measured osculating SMA drift is included; otherwise it is
    a zero array (and callers should treat it as unavailable).
    """
    t, (r_gt, v_gt, da) = _validate_trajectories(
        t_s, ("r_gt_m", r_gt_m), ("v_gt_ms", v_gt_ms), ("da_m_s2", da_m_s2)
    )
    basis = _ric_basis(r_gt, v_gt)
    da_ric = np.einsum("tij,tj->ti", basis, da)
    v_norm = np.linalg.norm(v_gt, axis=1)
    da_tan = np.einsum("ij,ij->i", da, v_gt) / v_norm

    # Centred rolling mean over one osculating period (window [t-P/2, t+P/2]).
    sma_gt = osculating_sma(r_gt, v_gt, mu_m3_s2)
    period = 2.0 * np.pi * np.sqrt(sma_gt**3 / mu_m3_s2)
    csum = np.concatenate([[0.0], np.cumsum(da_tan)])
    lo = np.searchsorted(t, t - 0.5 * period, side="left")
    hi = np.searchsorted(t, t + 0.5 * period, side="right")
    counts = np.maximum(hi - lo, 1)
    orbit_mean = (csum[hi] - csum[lo]) / counts
    orbit_valid = (t - 0.5 * period >= t[0]) & (t + 0.5 * period <= t[-1])

    if r_st_m is not None and v_st_ms is not None:
        _, (r_st, v_st) = _validate_trajectories(t, ("r_st_m", r_st_m), ("v_st_ms", v_st_ms))
        delta_sma = osculating_sma(r_st, v_st, mu_m3_s2) - sma_gt
    else:
        delta_sma = np.zeros_like(t)

    return TangentialBiasDiagnosis(
        t_s=t,
        da_ric_m_s2=da_ric,
        da_tangential_m_s2=da_tan,
        mean_da_ric_m_s2=np.mean(da_ric, axis=0),
        mean_da_tangential_m_s2=float(np.mean(da_tan)),
        rms_da_m_s2=_rms(np.linalg.norm(da, axis=1)),
        orbit_mean_da_tangential_m_s2=orbit_mean,
        orbit_mean_valid=orbit_valid,
        delta_sma_measured_m=delta_sma,
    )


@dataclass(frozen=True)
class PhaseLagPrediction:
    """
    Gauss-VE prediction of the phase drift caused by the tangential bias.

    Integration chain (see module docstring for the near-circular caveat):
    ``d(δa)/dt = 2 a² v Δa_t / mu`` → ``δn = -(3/2)(n/a) δa`` →
    ``δM = ∫ δn dt`` → ``tau_pred = δM / n``, ``e_along_pred = a δM``.
    Sign convention matches :class:`PhaseLagEstimate` (positive = leads).
    """

    t_s:                 F64Array  # (T,)
    delta_sma_pred_m:    F64Array  # (T,)
    delta_mean_anomaly_pred_rad: F64Array  # (T,)
    tau_pred_s:          F64Array  # (T,)
    e_along_pred_m:      F64Array  # (T,)
    tau_pred_final_s:    float


def predict_phase_drift_from_tangential_bias(
    t_s: object,
    r_gt_m: object,
    v_gt_ms: object,
    da_tangential_m_s2: object,
    *,
    mu_m3_s2: float = MU_MOON,
) -> PhaseLagPrediction:
    """Integrate the Gauss VEs with the measured tangential bias to predict tau(t)."""
    t = _as_f64(t_s, "t_s", "(T,)")
    r_gt = _as_f64(r_gt_m, "r_gt_m", "(T,3)")
    v_gt = _as_f64(v_gt_ms, "v_gt_ms", "(T,3)")
    da_tan = _as_f64(da_tangential_m_s2, "da_tangential_m_s2", "(T,)")
    if not (t.size == r_gt.shape[0] == v_gt.shape[0] == da_tan.size):
        raise ValueError("prediction inputs must share the same number of epochs")

    sma = osculating_sma(r_gt, v_gt, mu_m3_s2)
    n = np.sqrt(mu_m3_s2 / sma**3)
    v = np.linalg.norm(v_gt, axis=1)

    dadt = 2.0 * sma**2 * v * da_tan / mu_m3_s2   # exact tangential Gauss VE
    delta_sma = _cumtrapz0(dadt, t)
    delta_n = -1.5 * (n / sma) * delta_sma        # near-circular
    delta_m = _cumtrapz0(delta_n, t)
    tau_pred = delta_m / n
    return PhaseLagPrediction(
        t_s=t,
        delta_sma_pred_m=delta_sma,
        delta_mean_anomaly_pred_rad=delta_m,
        tau_pred_s=tau_pred,
        e_along_pred_m=sma * delta_m,
        tau_pred_final_s=float(tau_pred[-1]),
    )


# =============================================================================
# 5.              LAYER 4 — UQ COVARIANCE ALIGNMENT (Faz 3)
# =============================================================================

UQ_ALIGNMENT_CAVEAT = (
    "Along-track dominance of the leading covariance eigenvector is a generic "
    "property of orbital dynamics: any dispersion that perturbs the semi-major "
    "axis grows fastest along-track. High error/eigenvector alignment therefore "
    "shows that the surrogate error grows along the dynamically natural "
    "uncertainty direction; it is consistency evidence, not a causal "
    "explanation. The causal claim belongs to the Gauss-VE tangential-bias "
    "prediction (tau_pred vs measured tau)."
)


@dataclass(frozen=True)
class UQAlignment:
    """
    Alignment of the surrogate position error with a reference UQ covariance.

    ``alignment(t) = |e(t)·q1(t)| / |e(t)|`` where ``q1`` is the leading
    (largest-eigenvalue) eigenvector of the 3×3 position covariance ``P(t)``;
    values near 1 mean the error grows along the direction the reference
    dynamics naturally amplifies. Epochs with ``|e| = 0`` hold NaN and are
    excluded from ``mean_alignment``. ``q1_ric`` expresses the eigenvector in
    the reference RIC frame, sign-canonicalised so the along-track component
    is non-negative (eigenvectors are direction-ambiguous).

    Any report that quotes these numbers must also carry
    :data:`UQ_ALIGNMENT_CAVEAT` — see its text for what this metric does NOT
    show.
    """

    t_s:               F64Array  # (T,)
    alignment:         F64Array  # (T,) in [0, 1]; NaN where |e| == 0
    eigvals_pos_m2:    F64Array  # (T, 3) position-covariance eigenvalues, descending
    q1_ric:            F64Array  # (T, 3) leading eigenvector in RIC, along >= 0
    mean_alignment:    float
    final_alignment:   float
    mean_q1_along_abs: float     # mean |q1 · î|: how along-track the UQ growth is
    leading_eigval_growth: float  # eigvals[-1, 0] / eigvals[0, 0]


def compute_uq_alignment(
    t_s: object,
    e_m: object,
    cov_pos_m2: object,
    r_ref_m: object,
    v_ref_ms: object,
) -> UQAlignment:
    """
    Align a position-error history with a reference covariance history.

    ``cov_pos_m2`` is the (T, 3, 3) position covariance of the *reference*
    ensemble on the same epoch grid (see :func:`load_uq_covariance_history`
    and :func:`interpolate_covariance_to_times`), in the same Moon-centred
    inertial frame as the error vector ``e_m = r_st - r_gt``.
    """
    t, (e, r_ref, v_ref) = _validate_trajectories(
        t_s, ("e_m", e_m), ("r_ref_m", r_ref_m), ("v_ref_ms", v_ref_ms)
    )
    cov = np.asarray(cov_pos_m2, dtype=np.float64)
    if cov.shape != (t.size, 3, 3):
        raise ValueError(f"cov_pos_m2 must be (T, 3, 3) with T={t.size}, got {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("cov_pos_m2 contains non-finite values")

    w, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
    if np.any(w[:, -1] <= 0.0):
        raise ValueError(
            "cov_pos_m2 has a non-positive leading eigenvalue; the leading "
            "uncertainty direction is undefined"
        )
    eigvals = np.ascontiguousarray(w[:, ::-1])
    q1 = vecs[:, :, -1]                    # (T, 3) leading eigenvector

    e_norm = np.linalg.norm(e, axis=1)
    alignment = np.full(t.size, np.nan)
    nz = e_norm > 0.0
    alignment[nz] = np.abs(np.einsum("ij,ij->i", e[nz], q1[nz])) / e_norm[nz]

    basis = _ric_basis(r_ref, v_ref)
    q1_ric = np.einsum("tij,tj->ti", basis, q1)
    q1_ric[q1_ric[:, 1] < 0.0] *= -1.0

    return UQAlignment(
        t_s=t,
        alignment=alignment,
        eigvals_pos_m2=eigvals,
        q1_ric=np.ascontiguousarray(q1_ric),
        mean_alignment=float(np.nanmean(alignment)) if bool(nz.any()) else float("nan"),
        final_alignment=float(alignment[-1]),
        mean_q1_along_abs=float(np.mean(np.abs(q1_ric[:, 1]))),
        leading_eigval_growth=float(eigvals[-1, 0] / eigvals[0, 0]),
    )


def load_uq_covariance_history(npz_path: object) -> tuple[F64Array, F64Array, F64Array]:
    """
    Load ``(t_s, mean_state, cov_pos)`` from a ``uq_covariance.npz`` bundle.

    The bundle is written by :mod:`lunaris.analysis.ensemble.uq_report`
    (keys ``t_s`` (T,), ``mean_state`` (T, 6), ``cov`` (T, 6, 6)); the 3×3
    position block of the covariance is returned.
    """
    with np.load(npz_path) as data:
        missing = [k for k in ("t_s", "mean_state", "cov") if k not in data.files]
        if missing:
            raise ValueError(
                f"{npz_path} is not a uq_covariance.npz bundle: missing keys {missing}"
            )
        t = np.asarray(data["t_s"], dtype=np.float64)
        mean_state = np.asarray(data["mean_state"], dtype=np.float64)
        cov = np.asarray(data["cov"], dtype=np.float64)
    if cov.ndim != 3 or cov.shape[1:] != (6, 6):
        raise ValueError(f"cov must be (T, 6, 6), got {cov.shape}")
    if mean_state.ndim != 2 or mean_state.shape != (t.size, 6):
        raise ValueError(f"mean_state must be (T, 6) with T={t.size}, got {mean_state.shape}")
    return t, mean_state, np.ascontiguousarray(cov[:, :3, :3])


def interpolate_covariance_to_times(
    t_uq_s: object,
    cov_pos_m2: object,
    t_query_s: object,
) -> F64Array:
    """
    Linear per-element interpolation of a covariance history onto a new grid.

    Covariances do not propagate linearly between epochs, so this is a
    diagnostic approximation — adequate for eigen-direction alignment when the
    UQ output grid is dense relative to the orbital period. Query times must
    lie inside the UQ epoch span (no extrapolation).
    """
    t_uq = _as_f64(t_uq_s, "t_uq_s", "(T,)")
    tq = _as_f64(t_query_s, "t_query_s", "(T,)")
    cov = np.asarray(cov_pos_m2, dtype=np.float64)
    if cov.shape != (t_uq.size, 3, 3):
        raise ValueError(f"cov_pos_m2 must be (T, 3, 3) with T={t_uq.size}, got {cov.shape}")
    if t_uq.size < 2 or not np.all(np.diff(t_uq) > 0.0):
        raise ValueError("t_uq_s must be strictly increasing with at least 2 epochs")
    if np.any(tq < t_uq[0]) or np.any(tq > t_uq[-1]):
        raise ValueError(
            "t_query_s extends outside the UQ epoch span; covariance "
            "extrapolation is not supported"
        )
    idx = np.clip(np.searchsorted(t_uq, tq, side="right") - 1, 0, t_uq.size - 2)
    w = ((tq - t_uq[idx]) / (t_uq[idx + 1] - t_uq[idx]))[:, None, None]
    return (1.0 - w) * cov[idx] + w * cov[idx + 1]


@dataclass(frozen=True)
class CovarianceShapeSimilarity:
    """Per-epoch shape comparison of two position-covariance histories."""

    t_s:                    F64Array  # (T,)
    leading_angle_deg:      F64Array  # (T,) angle between leading eigenvectors
    trace_ratio:            F64Array  # (T,) trace(A) / trace(B)
    mean_leading_angle_deg: float


def compute_covariance_shape_similarity(
    t_s: object,
    cov_a_m2: object,
    cov_b_m2: object,
) -> CovarianceShapeSimilarity:
    """Compare two covariance histories (e.g. surrogate vs reference ensemble)."""
    t = _as_f64(t_s, "t_s", "(T,)")
    cov_a = np.asarray(cov_a_m2, dtype=np.float64)
    cov_b = np.asarray(cov_b_m2, dtype=np.float64)
    for name, cov in (("cov_a_m2", cov_a), ("cov_b_m2", cov_b)):
        if cov.shape != (t.size, 3, 3):
            raise ValueError(f"{name} must be (T, 3, 3) with T={t.size}, got {cov.shape}")
        if not np.all(np.isfinite(cov)):
            raise ValueError(f"{name} contains non-finite values")

    q1_a = np.linalg.eigh(cov_a)[1][:, :, -1]
    q1_b = np.linalg.eigh(cov_b)[1][:, :, -1]
    cos_angle = np.clip(np.abs(np.einsum("ij,ij->i", q1_a, q1_b)), 0.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))

    tr_a = np.einsum("tii->t", cov_a)
    tr_b = np.einsum("tii->t", cov_b)
    if np.any(tr_b <= 0.0):
        raise ValueError("cov_b_m2 has a non-positive trace; ratio is undefined")

    return CovarianceShapeSimilarity(
        t_s=t,
        leading_angle_deg=angle_deg,
        trace_ratio=tr_a / tr_b,
        mean_leading_angle_deg=float(np.mean(angle_deg)),
    )


# =============================================================================
# 6.              BENCHMARK-HARNESS ADAPTER (Faz 4)
# =============================================================================

PHASE_SCENARIO_COLUMNS = (
    "phase_lag_final_s",
    "phase_lag_slope_s_per_day",
    "phase_corrected_rms_km",
    "phase_explained_fraction",
)


def phase_scenario_metrics(
    t_s: object,
    r_st_m: object,
    r_gt_m: object,
    v_gt_ms: object,
) -> dict[str, float]:
    """
    Flat phase-metric columns for a benchmark per-scenario CSV row.

    Distances are km (the scenario-CSV convention), the lag is seconds, and
    the slope is a linear least-squares rate of ``tau(t)`` in s/day. Returns
    NaN columns instead of raising, so one degenerate scenario cannot abort a
    multi-hundred-scenario benchmark run — output validation then flags the
    non-finite cells, keeping the failure visible instead of fatal.
    """
    try:
        phase, corrected = compute_phase_corrected_errors(t_s, r_st_m, r_gt_m, v_gt_ms)
    except ValueError:
        return {key: float("nan") for key in PHASE_SCENARIO_COLUMNS}
    t = phase.t_s
    slope_s_per_s = float(np.polynomial.polynomial.polyfit(t - t[0], phase.tau_s, 1)[1])
    return {
        "phase_lag_final_s": phase.tau_final_s,
        "phase_lag_slope_s_per_day": slope_s_per_s * DAY_S,
        "phase_corrected_rms_km": corrected.aligned_rms_m / 1_000.0,
        "phase_explained_fraction": corrected.phase_explained_fraction,
    }


# =============================================================================
# 7.              FULL DIAGNOSTIC REPORT
# =============================================================================


@dataclass(frozen=True)
class PhaseDiagnosticsReport:
    """Aggregated phase diagnostics.

    ``bias``/``prediction`` require Δa samples; ``uq`` requires a reference
    covariance history (both optional inputs of
    :func:`compute_phase_diagnostics`).
    """

    ric:        RICErrorHistory
    phase:      PhaseLagEstimate
    corrected:  PhaseCorrectedErrors
    bias:       TangentialBiasDiagnosis | None
    prediction: PhaseLagPrediction | None
    tau_pred_measured_corr: float | None   # Pearson r(tau_pred, tau)
    tau_pred_final_ratio:   float | None   # tau_pred_final / tau_final
    uq:         UQAlignment | None = None

    def summary(self) -> dict[str, float]:
        """Flat scalar summary (SI units + mGal), one CSV row per scenario."""
        out: dict[str, float] = {
            "raw_rms_m": self.corrected.raw_rms_m,
            "final_pos_err_m": self.corrected.final_pos_err_m,
            "ric_rms_radial_m": float(self.ric.rms_ric_m[0]),
            "ric_rms_along_m": float(self.ric.rms_ric_m[1]),
            "ric_rms_cross_m": float(self.ric.rms_ric_m[2]),
            "ms_fraction_radial": float(self.ric.ms_fraction[0]),
            "ms_fraction_along": float(self.ric.ms_fraction[1]),
            "ms_fraction_cross": float(self.ric.ms_fraction[2]),
            "tau_final_s": self.phase.tau_final_s,
            "tau_abs_max_s": self.phase.tau_abs_max_s,
            "tau_fit_degree": float(self.phase.tau_fit_degree),
            "tau_fit_rms_residual_s": self.phase.tau_fit_rms_residual_s,
            "pointwise_detrended_rms_m": self.corrected.pointwise_detrended_rms_m,
            "aligned_rms_m": self.corrected.aligned_rms_m,
            "aligned_valid_fraction": self.corrected.aligned_valid_fraction,
            "phase_explained_fraction": self.corrected.phase_explained_fraction,
            "ric_rms_after_radial_m": float(self.corrected.ric_rms_after_m[0]),
            "ric_rms_after_along_m": float(self.corrected.ric_rms_after_m[1]),
            "ric_rms_after_cross_m": float(self.corrected.ric_rms_after_m[2]),
        }
        if self.bias is not None:
            out.update(
                {
                    "mean_da_tangential_m_s2": self.bias.mean_da_tangential_m_s2,
                    "mean_da_tangential_mGal": self.bias.mean_da_tangential_m_s2
                    / M_S2_PER_MGAL,
                    "mean_da_radial_m_s2": float(self.bias.mean_da_ric_m_s2[0]),
                    "mean_da_along_m_s2": float(self.bias.mean_da_ric_m_s2[1]),
                    "mean_da_cross_m_s2": float(self.bias.mean_da_ric_m_s2[2]),
                    "rms_da_m_s2": self.bias.rms_da_m_s2,
                    "delta_sma_final_m": float(self.bias.delta_sma_measured_m[-1]),
                }
            )
        if self.prediction is not None:
            out["tau_pred_final_s"] = self.prediction.tau_pred_final_s
            out["delta_sma_pred_final_m"] = float(self.prediction.delta_sma_pred_m[-1])
        if self.tau_pred_measured_corr is not None:
            out["tau_pred_measured_corr"] = self.tau_pred_measured_corr
        if self.tau_pred_final_ratio is not None:
            out["tau_pred_final_ratio"] = self.tau_pred_final_ratio
        if self.uq is not None:
            out.update(
                {
                    "uq_alignment_mean": self.uq.mean_alignment,
                    "uq_alignment_final": self.uq.final_alignment,
                    "uq_q1_along_abs_mean": self.uq.mean_q1_along_abs,
                    "uq_leading_eigval_growth": self.uq.leading_eigval_growth,
                }
            )
        return out


def compute_phase_diagnostics(
    t_s: object,
    r_st_m: object,
    v_st_ms: object,
    r_gt_m: object,
    v_gt_ms: object,
    *,
    da_m_s2: object | None = None,
    cov_pos_m2: object | None = None,
    mu_m3_s2: float = MU_MOON,
    tau_fit_degree: int = 2,
) -> PhaseDiagnosticsReport:
    """
    Run the full phase-diagnostic chain on a surrogate-vs-truth trajectory pair.

    Parameters
    ----------
    t_s, r_st_m, v_st_ms, r_gt_m, v_gt_ms
        Shared strictly-increasing epoch grid [s] and the surrogate / truth
        state histories [m, m/s] in the Moon-centred inertial frame.
    da_m_s2
        Optional (T, 3) model-minus-truth acceleration evaluated along the
        ground-truth trajectory. Enables the tangential-bias diagnosis and
        the Gauss-VE phase prediction; without it those fields are ``None``.
    cov_pos_m2
        Optional (T, 3, 3) reference-ensemble position covariance on the same
        epoch grid (see :func:`load_uq_covariance_history` /
        :func:`interpolate_covariance_to_times`). Enables the UQ alignment
        layer; without it ``report.uq`` is ``None``.
    tau_fit_degree
        Polynomial degree (0-3) of the secular phase-lag fit used for the
        aligned residual. Default 2: constant tangential bias ⇒ quadratic
        phase growth.
    """
    t, (r_st, v_st, r_gt, v_gt) = _validate_trajectories(
        t_s,
        ("r_st_m", r_st_m),
        ("v_st_ms", v_st_ms),
        ("r_gt_m", r_gt_m),
        ("v_gt_ms", v_gt_ms),
    )
    ric = compute_ric_error_history(t, r_st, r_gt, v_gt)
    phase, corrected = compute_phase_corrected_errors(
        t, r_st, r_gt, v_gt, fit_degree=tau_fit_degree
    )

    bias: TangentialBiasDiagnosis | None = None
    prediction: PhaseLagPrediction | None = None
    corr: float | None = None
    ratio: float | None = None
    if da_m_s2 is not None:
        bias = diagnose_tangential_bias(
            t, r_gt, v_gt, da_m_s2, r_st_m=r_st, v_st_ms=v_st, mu_m3_s2=mu_m3_s2
        )
        prediction = predict_phase_drift_from_tangential_bias(
            t, r_gt, v_gt, bias.da_tangential_m_s2, mu_m3_s2=mu_m3_s2
        )
        if np.std(prediction.tau_pred_s) > 0.0 and np.std(phase.tau_s) > 0.0:
            corr = float(np.corrcoef(prediction.tau_pred_s, phase.tau_s)[0, 1])
        if phase.tau_final_s != 0.0:
            ratio = prediction.tau_pred_final_s / phase.tau_final_s

    uq: UQAlignment | None = None
    if cov_pos_m2 is not None:
        uq = compute_uq_alignment(t, r_st - r_gt, cov_pos_m2, r_gt, v_gt)

    return PhaseDiagnosticsReport(
        ric=ric,
        phase=phase,
        corrected=corrected,
        bias=bias,
        prediction=prediction,
        tau_pred_measured_corr=corr,
        tau_pred_final_ratio=ratio,
        uq=uq,
    )
