"""Surrogate-assisted frozen-orbit search pipeline (roadmap R04, R21, R27).

Staged, resumable search over lunar orbital-element space:

    stage 0  Sobol element sampling            -> stage0_samples.npz
    stage 1  batch screening propagation       -> stage1_screening.npz
    stage 2  domain guard + scoring + top-K    -> stage2_candidates.json
    stage 3  classical SH validation           -> stage3_validation.json
    stage 4  (optional) refinement + families  -> stage4_families.json

Each stage's output is a file contract: re-running the pipeline with
``resume=True`` loads finished stages instead of recomputing them. Sobol seed,
sample count, and backend provenance are recorded in ``manifest.json``.
Stage 1 defaults to a summary-only artifact: per-sample metrics plus top-K full
histories, not the full ``(T, N, 6)`` ensemble tensor.

Backend policy: propagation is injected via two small protocols so the same
pipeline runs on the batch GPU/CPU screening backends and on the classical
CPU SH reference path. Enforcement rules (not conventions):

- R27 — every screening trajectory passes the domain guard; domain-exited
  samples can never be promoted to final candidates.
- R21 — ``strict_frozen`` / ``quasi_frozen`` statuses can only come from
  :func:`lunaris.analysis.frozen.classify.classify_candidate` with a classical
  SH validation backend; :func:`enforce_classical_validation_rule` re-checks
  every record before it is written to disk.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .classify import (
    QUASI_FROZEN,
    STRICT_FROZEN,
    FrozenClassificationConfig,
    classify_candidate,
    frozen_score,
    is_classical_validation_backend,
)
from .domain_guard import (
    DomainGuardResult,
    FrozenSearchDomainGuard,
    apply_domain_guard_to_scores,
    assert_candidate_domain_clean,
    evaluate_domain_guard,
)
from .family_report import (
    build_family_report,
    group_candidates_into_families,
)
from .metrics import compute_frozen_metrics
from .refine import (
    ELEMENT_NAMES,
    RefinementConfig,
    refine_candidate,
)

logger = logging.getLogger(__name__)

_VALIDATED_STATUSES = frozenset({STRICT_FROZEN, QUASI_FROZEN})

STAGE0_SAMPLES = "stage0_samples.npz"
STAGE1_SCREENING = "stage1_screening.npz"
STAGE2_CANDIDATES = "stage2_candidates.json"
STAGE3_VALIDATION = "stage3_validation.json"
STAGE4_FAMILIES = "stage4_families.json"
MANIFEST = "manifest.json"

STAGE1_SCREENING_SCHEMA_VERSION = 2
STAGE1_OUTPUT_FULL = "full"
STAGE1_OUTPUT_SUMMARY_ONLY = "summary_only"
_STAGE1_OUTPUT_MODES = frozenset({STAGE1_OUTPUT_FULL, STAGE1_OUTPUT_SUMMARY_ONLY})

_SCREENING_SUMMARY_FIELDS = (
    "e_min",
    "e_max",
    "e_range",
    "h_peri_min_km",
    "h_peri_max_km",
    "h_peri_range_km",
    "trend_e_per_day",
    "trend_h_peri_km_per_day",
    "omega_behavior",
)


# ---------------------------------------------------------------------------
# Element sampling (stage 0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ElementBounds:
    """Sampled element box, ordered as :data:`ELEMENT_NAMES` (km / deg)."""

    a_km: tuple[float, float]
    e: tuple[float, float]
    i_deg: tuple[float, float]
    raan_deg: tuple[float, float] = (0.0, 360.0)
    argp_deg: tuple[float, float] = (0.0, 360.0)
    ta_deg: tuple[float, float] = (0.0, 360.0)

    def __post_init__(self) -> None:
        for name in ELEMENT_NAMES:
            lo, hi = getattr(self, name)
            if not (np.isfinite(lo) and np.isfinite(hi)) or float(lo) > float(hi):
                raise ValueError(f"bounds for {name} must satisfy lo <= hi, got ({lo}, {hi})")
        if float(self.e[0]) < 0.0:
            raise ValueError("eccentricity lower bound must be >= 0")
        if float(self.a_km[0]) <= 0.0:
            raise ValueError("semi-major axis lower bound must be > 0 km")

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in ELEMENT_NAMES], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sobol_element_samples(
    bounds: ElementBounds,
    n_samples: int,
    *,
    seed: int,
    method: str = "sobol_scrambled",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw ``(N, 6)`` element samples in the box, with full provenance.

    ``method`` is ``"sobol"``, ``"sobol_scrambled"``, or ``"lhs"``. Sobol draws
    a base-2 design and keeps the first ``n_samples`` (recorded in provenance).
    """
    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1")
    from scipy.stats import qmc

    dim = len(ELEMENT_NAMES)
    if method == "lhs":
        unit = qmc.LatinHypercube(d=dim, seed=int(seed)).random(int(n_samples))
        base2_note = ""
    elif method in ("sobol", "sobol_scrambled"):
        scramble = method == "sobol_scrambled"
        sampler = qmc.Sobol(d=dim, scramble=scramble, seed=int(seed) if scramble else None)
        m = int(math.ceil(math.log2(max(1, int(n_samples)))))
        unit = sampler.random_base2(m=m)[: int(n_samples)]
        base2_note = f"base-2 design 2^{m}, first {int(n_samples)} kept"
    else:
        raise ValueError(f"unknown sampling method {method!r}")

    box = bounds.as_array()
    elements = box[:, 0][None, :] + unit * (box[:, 1] - box[:, 0])[None, :]
    provenance = {
        "sampling_method": method,
        "sobol_seed": int(seed),
        "sample_count": int(n_samples),
        "bounds": bounds.to_dict(),
        "note": base2_note,
    }
    return np.ascontiguousarray(elements, dtype=np.float64), provenance


def elements_to_states(elements: np.ndarray, mu_m3s2: float) -> np.ndarray:
    """Convert ``(N, 6)`` element rows (km/deg) to Cartesian states ``(N, 6)`` (SI)."""
    from lunaris.common.math_utils import coe_to_rv

    el = np.asarray(elements, dtype=np.float64)
    if el.ndim != 2 or el.shape[1] != len(ELEMENT_NAMES):
        raise ValueError(f"elements must be (N, {len(ELEMENT_NAMES)}), got {el.shape}")
    out = np.empty((el.shape[0], 6), dtype=np.float64)
    deg2rad = np.pi / 180.0
    for j in range(el.shape[0]):
        r, v = coe_to_rv(
            el[j, 0] * 1_000.0,
            el[j, 1],
            el[j, 2] * deg2rad,
            el[j, 3] * deg2rad,
            el[j, 4] * deg2rad,
            el[j, 5] * deg2rad,
            float(mu_m3s2),
        )
        out[j, :3] = r
        out[j, 3:] = v
    return out


# ---------------------------------------------------------------------------
# Propagation protocols (injected backends)
# ---------------------------------------------------------------------------


class ScreeningPropagator(Protocol):
    """Batch screening propagation contract (stage 1)."""

    backend_name: str
    provenance: dict[str, Any]

    def propagate(
        self, Y0: np.ndarray, duration_s: float, output_dt_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(t_out (T,), Y_out (T,N,6), impact_flags (N,), t_impact (N,))``."""
        ...


class ValidationPropagator(Protocol):
    """Single-orbit validation propagation contract (stage 3)."""

    backend_label: str
    provenance: dict[str, Any]

    def propagate(
        self, y0: np.ndarray, duration_s: float, output_dt_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t (T,), y (T,6))`` for one initial state."""
        ...


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrozenSearchConfig:
    """Configuration for one staged frozen-orbit search run."""

    bounds: ElementBounds
    n_samples: int
    seed: int = 0
    sampling_method: str = "sobol_scrambled"

    screening_duration_s: float = 7.0 * 86_400.0
    screening_output_dt_s: float = 3_600.0
    screening_output_mode: str = STAGE1_OUTPUT_SUMMARY_ONLY
    screening_summary_batch_size: int = 4_096
    stage1_history_top_k: int | None = None

    top_k: int = 10

    validation_duration_s: float = 30.0 * 86_400.0
    validation_output_dt_s: float = 3_600.0

    guard: FrozenSearchDomainGuard = field(
        default_factory=lambda: FrozenSearchDomainGuard(
            altitude_min_km=20.0, altitude_max_km=20_000.0
        )
    )
    perilune_safety_min_m: float = 20_000.0
    eccentricity_upper_bound: float = 0.5

    refine_top_n: int = 0  # 0 disables stage-4 refinement
    refinement: RefinementConfig | None = None

    mu_m3s2: float = 4.9028001224453001e12
    reference_radius_m: float = 1.7374e6

    def __post_init__(self) -> None:
        if int(self.n_samples) < 2:
            raise ValueError("n_samples must be >= 2")
        if int(self.top_k) < 1:
            raise ValueError("top_k must be >= 1")
        mode = str(self.screening_output_mode).strip().lower()
        if mode not in _STAGE1_OUTPUT_MODES:
            raise ValueError(
                "screening_output_mode must be one of "
                + ", ".join(sorted(_STAGE1_OUTPUT_MODES))
            )
        object.__setattr__(self, "screening_output_mode", mode)
        if int(self.screening_summary_batch_size) < 1:
            raise ValueError("screening_summary_batch_size must be >= 1")
        if self.stage1_history_top_k is not None and int(self.stage1_history_top_k) < 1:
            raise ValueError("stage1_history_top_k must be >= 1 when set")
        for name in ("screening_duration_s", "validation_duration_s"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if int(self.refine_top_n) > 0 and self.refinement is None:
            raise ValueError("refine_top_n > 0 requires a RefinementConfig")

    def classification_config(self) -> FrozenClassificationConfig:
        return FrozenClassificationConfig.for_mission_duration(
            float(self.validation_duration_s),
            perilune_safety_min_m=float(self.perilune_safety_min_m),
            eccentricity_upper_bound=float(self.eccentricity_upper_bound),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["bounds"] = self.bounds.to_dict()
        out["guard"] = self.guard.to_dict()
        out["refinement"] = self.refinement.to_dict() if self.refinement else None
        return out


# ---------------------------------------------------------------------------
# R21 enforcement
# ---------------------------------------------------------------------------


def enforce_classical_validation_rule(record: dict[str, Any]) -> None:
    """R21 choke point: refuse to persist a validated frozen status without a
    classical SH validation backend. Raises ``RuntimeError``.

    ``record`` is a candidate record carrying a ``classification`` block (from
    ``classify_candidate().to_dict()``). This duplicates the classifier's own
    rule on purpose — a bug or manual edit upstream must not survive the write.
    """
    classification = record.get("classification")
    if not isinstance(classification, dict):
        raise RuntimeError("candidate record has no classification block")
    status = str(classification.get("status", ""))
    if status in _VALIDATED_STATUSES:
        backend = classification.get("validation_backend")
        if not is_classical_validation_backend(backend):
            raise RuntimeError(
                f"status {status!r} requires a classical SH validation backend, "
                f"got {backend!r} (R21)"
            )
        if not bool(classification.get("validated", False)):
            raise RuntimeError(
                f"status {status!r} written without validated=True (R21)"
            )


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and NaN/Inf to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.floating | float):
        f = float(value)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # R29b-justified: provenance nicety, never fatal
        return "unknown"


def _element_history(t: np.ndarray, y: np.ndarray, mu: float, r_ref: float) -> dict[str, np.ndarray]:
    """Osculating element histories for one ``(T, 6)`` trajectory."""
    from lunaris.batch.summary import _osculating_elements

    a_m, e, inc, argp = _osculating_elements(y[:, :3], y[:, 3:], float(mu))
    return {
        "t_s": np.asarray(t, dtype=np.float64),
        "a_m": a_m,
        "e": e,
        "inc_rad": inc,
        "argp_rad": argp,
        "h_peri_m": a_m * (1.0 - e) - float(r_ref),
    }


def _scalar_string(arrays: dict[str, np.ndarray], key: str, default: str = "") -> str:
    if key not in arrays:
        return default
    value = np.asarray(arrays[key])
    if value.shape == ():
        return str(value.item())
    if value.size == 0:
        return default
    return str(value.ravel()[0])


def _stage1_output_mode(arrays: dict[str, np.ndarray]) -> str:
    if "Y_out" in arrays:
        return STAGE1_OUTPUT_FULL
    return _scalar_string(arrays, "output_mode", STAGE1_OUTPUT_SUMMARY_ONLY)


def _stage1_is_summary_only(arrays: dict[str, np.ndarray]) -> bool:
    return _stage1_output_mode(arrays) == STAGE1_OUTPUT_SUMMARY_ONLY and "Y_out" not in arrays


def _string_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.str_)


def _summary_arrays(
    *,
    summary: dict[str, Any],
    guard_result: DomainGuardResult,
    scores: np.ndarray,
    raw_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    """Flatten the batch summary + guard result into the stage-1 npz contract."""
    fields = summary["fields"]
    arrays: dict[str, np.ndarray] = {}
    for key in _SCREENING_SUMMARY_FIELDS:
        value = fields[key]
        if key == "omega_behavior":
            arrays[f"summary__{key}"] = _string_array(value)
        else:
            arrays[f"summary__{key}"] = np.asarray(value, dtype=np.float64)
    arrays.update(
        {
            "summary__impact_flag": np.asarray(fields["impact_flag"], dtype=np.float64),
            "summary__t_impact_s": np.asarray(fields["t_impact_s"], dtype=np.float64),
            "summary__domain_exit_flag": np.asarray(
                guard_result.domain_exit_flag, dtype=np.float64
            ),
            "summary__t_domain_exit_s": np.asarray(
                guard_result.t_domain_exit_s, dtype=np.float64
            ),
            "summary__escape_flag": np.asarray(guard_result.escape_flag, dtype=np.float64),
            "summary__nonfinite_flag": np.asarray(
                guard_result.nonfinite_flag, dtype=np.float64
            ),
            "summary__domain_exit_reason": _string_array(guard_result.reasons),
            "summary__score_raw": np.asarray(raw_scores, dtype=np.float64),
            "summary__score": np.asarray(scores, dtype=np.float64),
            "summary__valid": np.asarray(fields["valid"], dtype=np.float64),
        }
    )
    return arrays


def _concat_summary_array_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("cannot concatenate an empty stage-1 summary")
    merged: dict[str, np.ndarray] = {}
    for key in parts[0]:
        merged[key] = np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
    return merged


def _guard_from_stage1_summary(screening: dict[str, np.ndarray]) -> DomainGuardResult:
    return DomainGuardResult(
        domain_exit_flag=np.asarray(screening["summary__domain_exit_flag"], dtype=np.float64)
        > 0.5,
        t_domain_exit_s=np.asarray(screening["summary__t_domain_exit_s"], dtype=np.float64),
        escape_flag=np.asarray(screening["summary__escape_flag"], dtype=np.float64) > 0.5,
        nonfinite_flag=np.asarray(screening["summary__nonfinite_flag"], dtype=np.float64)
        > 0.5,
        reasons=np.asarray(screening["summary__domain_exit_reason"], dtype=np.str_),
    )


def _summary_fields_from_stage1(screening: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(screening[f"summary__{key}"])
        for key in _SCREENING_SUMMARY_FIELDS
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class FrozenSearchPipeline:
    """Staged, resumable surrogate-assisted frozen-orbit search (R04)."""

    def __init__(
        self,
        config: FrozenSearchConfig,
        *,
        screening: ScreeningPropagator,
        validation: ValidationPropagator,
        out_dir: str | Path,
        sensitivity_validations: list[ValidationPropagator] | None = None,
    ) -> None:
        self.config = config
        self.screening = screening
        self.validation = validation
        self.sensitivity_validations = list(sensitivity_validations or [])
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:12]

    # -- manifest ----------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.out_dir / MANIFEST

    def _load_manifest(self) -> dict[str, Any]:
        path = self._manifest_path()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "run_id": self.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git_commit(),
            "config": _json_safe(self.config.to_dict()),
            "screening_backend": str(self.screening.backend_name),
            "screening_provenance": _json_safe(dict(self.screening.provenance)),
            "validation_backend": str(self.validation.backend_label),
            "validation_provenance": _json_safe(dict(self.validation.provenance)),
            "stages": {},
        }

    def _mark_stage(self, manifest: dict[str, Any], stage: str, filename: str, **extra: Any) -> None:
        manifest["stages"][stage] = {
            "file": filename,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **_json_safe(extra),
        }
        _atomic_write_json(self._manifest_path(), manifest)

    # -- stage 0: sampling ---------------------------------------------------

    def stage0_samples(self, manifest: dict[str, Any], *, resume: bool) -> np.ndarray:
        path = self.out_dir / STAGE0_SAMPLES
        if resume and path.exists():
            with np.load(path, allow_pickle=False) as data:
                return np.asarray(data["elements"], dtype=np.float64)
        elements, provenance = sobol_element_samples(
            self.config.bounds,
            self.config.n_samples,
            seed=self.config.seed,
            method=self.config.sampling_method,
        )
        np.savez_compressed(path, elements=elements)
        manifest["sampling_provenance"] = _json_safe(provenance)
        self._mark_stage(manifest, "stage0", STAGE0_SAMPLES, n_samples=elements.shape[0])
        return elements

    # -- stage 1: screening propagation ---------------------------------------

    def stage1_screening(
        self, manifest: dict[str, Any], elements: np.ndarray, *, resume: bool
    ) -> dict[str, np.ndarray]:
        path = self.out_dir / STAGE1_SCREENING
        if resume and path.exists():
            with np.load(path, allow_pickle=False) as data:
                return {key: np.asarray(data[key]) for key in data.files}

        if str(self.config.screening_output_mode) == STAGE1_OUTPUT_SUMMARY_ONLY:
            arrays = self._stage1_screening_summary_only(elements)
            np.savez_compressed(path, **arrays)
            self._mark_stage(
                manifest,
                "stage1",
                STAGE1_SCREENING,
                backend=str(self.screening.backend_name),
                output_mode=STAGE1_OUTPUT_SUMMARY_ONLY,
                full_trajectory_stored=False,
                n_snapshots=int(arrays["t_out"].shape[0]),
                n_histories=int(arrays["topk_sample_indices"].shape[0]),
                batch_size=int(self.config.screening_summary_batch_size),
            )
            return arrays

        Y0 = elements_to_states(elements, self.config.mu_m3s2)
        t_out, Y_out, impact_flags, t_impact = self.screening.propagate(
            Y0,
            float(self.config.screening_duration_s),
            float(self.config.screening_output_dt_s),
        )
        arrays = {
            "schema_version": np.asarray(STAGE1_SCREENING_SCHEMA_VERSION, dtype=np.int64),
            "output_mode": np.asarray(STAGE1_OUTPUT_FULL),
            "t_out": np.asarray(t_out, dtype=np.float64),
            "Y_out": np.asarray(Y_out, dtype=np.float64),
            "impact_flags": np.asarray(impact_flags, dtype=np.float64),
            "t_impact": np.asarray(t_impact, dtype=np.float64),
        }
        np.savez_compressed(path, **arrays)
        self._mark_stage(
            manifest,
            "stage1",
            STAGE1_SCREENING,
            backend=str(self.screening.backend_name),
            output_mode=STAGE1_OUTPUT_FULL,
            full_trajectory_stored=True,
            n_snapshots=int(arrays["t_out"].shape[0]),
        )
        return arrays

    def _stage1_history_top_k(self) -> int:
        configured = self.config.stage1_history_top_k
        return max(
            int(self.config.top_k),
            int(self.config.refine_top_n),
            int(configured) if configured is not None else 0,
            1,
        )

    def _screening_scores_for_block(
        self,
        t_out: np.ndarray,
        Y_out: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
    ) -> tuple[dict[str, Any], DomainGuardResult, np.ndarray, np.ndarray]:
        from lunaris.batch.summary import summarize_ensemble

        summary = summarize_ensemble(
            t_out,
            Y_out,
            impact_flags,
            t_impact,
            mu_m3s2=float(self.config.mu_m3s2),
            r_ref_m=float(self.config.reference_radius_m),
        )
        guard_result = evaluate_domain_guard(
            t_out,
            Y_out,
            reference_radius_m=float(self.config.reference_radius_m),
            guard=self.config.guard,
            impact_flags=impact_flags,
            t_impact_s=t_impact,
        )
        raw_scores = np.asarray(summary["fields"]["score"], dtype=np.float64)
        scores = apply_domain_guard_to_scores(raw_scores, guard_result, self.config.guard)
        h_peri_min_km = np.asarray(summary["fields"]["h_peri_min_km"], dtype=np.float64)
        below_floor = h_peri_min_km < float(self.config.perilune_safety_min_m) / 1_000.0
        scores = np.where(below_floor, np.inf, scores)
        return summary, guard_result, raw_scores, scores

    def _stage1_screening_summary_only(self, elements: np.ndarray) -> dict[str, np.ndarray]:
        from lunaris.batch.summary import TopKTrajectoryBuffer

        Y0 = elements_to_states(elements, self.config.mu_m3s2)
        n_samples = int(Y0.shape[0])
        batch_size = int(self.config.screening_summary_batch_size)
        topk = TopKTrajectoryBuffer(self._stage1_history_top_k())
        summary_parts: list[dict[str, np.ndarray]] = []
        t_ref: np.ndarray | None = None

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            t_out, Y_out, impact_flags, t_impact = self.screening.propagate(
                Y0[start:end],
                float(self.config.screening_duration_s),
                float(self.config.screening_output_dt_s),
            )
            t_arr = np.asarray(t_out, dtype=np.float64)
            Y_arr = np.asarray(Y_out, dtype=np.float64)
            impacts = np.asarray(impact_flags, dtype=np.float64)
            t_imp = np.asarray(t_impact, dtype=np.float64)
            if t_ref is None:
                t_ref = t_arr
            elif t_arr.shape != t_ref.shape or not np.allclose(t_arr, t_ref, rtol=0.0, atol=1e-9):
                raise RuntimeError(
                    "screening propagator returned inconsistent time grids across "
                    "summary batches"
                )

            summary, guard_result, raw_scores, scores = self._screening_scores_for_block(
                t_arr,
                Y_arr,
                impacts,
                t_imp,
            )
            summary_parts.append(
                _summary_arrays(
                    summary=summary,
                    guard_result=guard_result,
                    scores=scores,
                    raw_scores=raw_scores,
                )
            )
            topk.offer_batch(
                global_start=start,
                scores=scores,
                Y_batch=Y_arr,
                impact_flags=impacts,
                t_impact=t_imp,
            )

        if t_ref is None:
            raise RuntimeError("stage-1 screening produced no batches")

        topk_arrays = topk.entry_arrays()
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(STAGE1_SCREENING_SCHEMA_VERSION, dtype=np.int64),
            "output_mode": np.asarray(STAGE1_OUTPUT_SUMMARY_ONLY),
            "t_out": np.asarray(t_ref, dtype=np.float64),
            **_concat_summary_array_parts(summary_parts),
            "topk_sample_indices": topk_arrays["sample_indices"],
            "topk_scores": topk_arrays["scores"],
            "topk_Y_out": topk.stacked_trajectories(int(t_ref.shape[0])),
            "topk_impact_flags": topk_arrays["impact_flags"],
            "topk_t_impact_s": topk_arrays["t_impact_s"],
        }
        return arrays

    # -- stage 2: guard + score + top-K ---------------------------------------

    def stage2_candidates(
        self,
        manifest: dict[str, Any],
        elements: np.ndarray,
        screening: dict[str, np.ndarray],
        *,
        resume: bool,
    ) -> list[dict[str, Any]]:
        path = self.out_dir / STAGE2_CANDIDATES
        if resume and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["candidates"]

        from lunaris.batch.summary import SCORE_DEFINITION

        if _stage1_is_summary_only(screening):
            fields = _summary_fields_from_stage1(screening)
            raw_scores = np.asarray(screening["summary__score_raw"], dtype=np.float64)
            scores = np.asarray(screening["summary__score"], dtype=np.float64)
            guard_result = _guard_from_stage1_summary(screening)
        else:
            t_out = screening["t_out"]
            Y_out = screening["Y_out"]
            impact_flags = screening["impact_flags"]
            t_impact = screening["t_impact"]
            summary, guard_result, raw_scores, scores = self._screening_scores_for_block(
                t_out,
                Y_out,
                impact_flags,
                t_impact,
            )
            fields = summary["fields"]

        order = np.argsort(scores, kind="stable")
        candidates: list[dict[str, Any]] = []
        for j in order.tolist():
            if len(candidates) >= int(self.config.top_k):
                break
            if not np.isfinite(scores[j]):
                break  # remaining samples are +inf (impacted/invalid/domain-exit)
            record = {
                "sample_index": int(j),
                "elements": {name: float(elements[j, k]) for k, name in enumerate(ELEMENT_NAMES)},
                "screening_score": float(scores[j]),
                "screening_score_raw": float(raw_scores[j]),
                "screening_backend": str(self.screening.backend_name),
                "summary": {
                    "e_min": float(fields["e_min"][j]),
                    "e_max": float(fields["e_max"][j]),
                    "e_range": float(fields["e_range"][j]),
                    "h_peri_min_km": float(fields["h_peri_min_km"][j]),
                    "h_peri_max_km": float(fields["h_peri_max_km"][j]),
                    "h_peri_range_km": float(fields["h_peri_range_km"][j]),
                    "trend_e_per_day": float(fields["trend_e_per_day"][j]),
                    "trend_h_peri_km_per_day": float(fields["trend_h_peri_km_per_day"][j]),
                    "omega_behavior": str(fields["omega_behavior"][j]),
                },
                "domain_guard": guard_result.sample_metadata(j),
                "validation_stage": "screened",
            }
            candidates.append(record)

        payload = {
            "run_id": manifest["run_id"],
            "score_definition": SCORE_DEFINITION,
            "domain_guard": self.config.guard.to_dict(),
            "domain_exit_count": guard_result.n_exits,
            "candidates": _json_safe(candidates),
        }
        _atomic_write_json(path, payload)
        self._mark_stage(
            manifest,
            "stage2",
            STAGE2_CANDIDATES,
            n_candidates=len(candidates),
            domain_exit_count=guard_result.n_exits,
        )
        return payload["candidates"]

    # -- stage 3: classical SH validation --------------------------------------

    def _validate_one(
        self,
        y0: np.ndarray,
        propagator: ValidationPropagator,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Propagate + classify one candidate with one validation backend."""
        t, y = propagator.propagate(
            np.asarray(y0, dtype=np.float64),
            float(self.config.validation_duration_s),
            float(self.config.validation_output_dt_s),
        )
        hist = _element_history(t, y, self.config.mu_m3s2, self.config.reference_radius_m)
        guard_result = evaluate_domain_guard(
            hist["t_s"],
            np.asarray(y, dtype=np.float64)[:, None, :],
            reference_radius_m=float(self.config.reference_radius_m),
            guard=self.config.guard,
        )
        domain_exit_t = (
            float(guard_result.t_domain_exit_s[0])
            if bool(guard_result.domain_exit_flag[0])
            else None
        )
        metrics = compute_frozen_metrics(
            hist["t_s"],
            eccentricity=hist["e"],
            inclination_rad=hist["inc_rad"],
            omega_rad=hist["argp_rad"],
            h_peri_m=hist["h_peri_m"],
            domain_exit_time_s=domain_exit_t,
            escape=bool(guard_result.escape_flag[0]),
        )
        cls_config = self.config.classification_config()
        passed = (
            not metrics.has_impact
            and not metrics.has_domain_exit
            and not metrics.escape
            and np.isfinite(frozen_score(metrics, cls_config))
        )
        classification = classify_candidate(
            metrics,
            cls_config,
            validation_backend=str(propagator.backend_label),
            long_horizon_validation_passed=bool(passed),
        )
        result = {
            "backend": str(propagator.backend_label),
            "duration_days": float(self.config.validation_duration_s) / 86_400.0,
            "passed": bool(passed),
            "metrics": _json_safe(metrics.to_dict()),
            "classification": _json_safe(classification.to_dict()),
        }
        return result, {"metrics": metrics, "classification": classification}


    def stage3_validation(
        self,
        manifest: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        resume: bool,
    ) -> list[dict[str, Any]]:
        path = self.out_dir / STAGE3_VALIDATION
        if resume and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["candidates"]

        validated: list[dict[str, Any]] = []
        for record in candidates:
            # R27: a domain-exited screening trajectory never reaches validation.
            assert_candidate_domain_clean(record)
            elements = np.array(
                [record["elements"][name] for name in ELEMENT_NAMES], dtype=np.float64
            )
            y0 = elements_to_states(elements[None, :], self.config.mu_m3s2)[0]

            primary, _raw = self._validate_one(y0, self.validation)
            sensitivity: list[dict[str, Any]] = []
            for extra in self.sensitivity_validations:
                extra_result, _ = self._validate_one(y0, extra)
                sensitivity.append(
                    {
                        "backend": extra_result["backend"],
                        "status": extra_result["classification"]["status"],
                        "score": extra_result["classification"]["score"],
                        "status_agrees": (
                            extra_result["classification"]["status"]
                            == primary["classification"]["status"]
                        ),
                    }
                )

            out = dict(record)
            out["validation"] = {k: v for k, v in primary.items() if k != "classification"}
            out["classification"] = primary["classification"]
            out["sensitivity"] = sensitivity
            out["validation_stage"] = "validated"
            enforce_classical_validation_rule(out)  # R21 choke point
            validated.append(out)

        payload = {
            "run_id": manifest["run_id"],
            "validation_backend": str(self.validation.backend_label),
            "sensitivity_backends": [
                str(p.backend_label) for p in self.sensitivity_validations
            ],
            "candidates": _json_safe(validated),
        }
        _atomic_write_json(path, payload)
        self._mark_stage(
            manifest, "stage3", STAGE3_VALIDATION, n_candidates=len(validated)
        )
        return payload["candidates"]

    # -- stage 4: refinement + families ----------------------------------------

    def stage4_families(
        self,
        manifest: dict[str, Any],
        validated: list[dict[str, Any]],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        path = self.out_dir / STAGE4_FAMILIES
        if resume and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        from lunaris.batch.summary import SCORE_DEFINITION

        refinements: list[dict[str, Any]] = []
        if int(self.config.refine_top_n) > 0 and self.config.refinement is not None:
            for record in validated[: int(self.config.refine_top_n)]:
                assert_candidate_domain_clean(record)  # R27 again before refinement
                refinements.append(
                    self._refine_record(record, self.config.refinement)
                )

        for record in validated:
            enforce_classical_validation_rule(record)  # R21 before family grouping

        families = group_candidates_into_families(
            [
                {
                    "sample_index": r["sample_index"],
                    "elements": r["elements"],
                    "classification": r["classification"],
                    "metrics": r.get("validation", {}).get("metrics", {}),
                    "validation": r.get("validation"),
                }
                for r in validated
            ],
            screening_backend=str(self.screening.backend_name),
            gravity_model=dict(self.validation.provenance.get("gravity_model", {})),
            third_body=dict(self.validation.provenance.get("third_body", {"earth": False, "sun": False})),
            provenance={
                "run_id": manifest["run_id"],
                "sobol_seed": int(self.config.seed),
                "sample_count": int(self.config.n_samples),
                "git_commit": manifest.get("git_commit", "unknown"),
            },
        )
        report = build_family_report(
            run_id=manifest["run_id"],
            score_definition=SCORE_DEFINITION,
            families=_json_safe(families),
        )
        report["refinements"] = _json_safe(refinements)
        _atomic_write_json(path, report)
        self._mark_stage(
            manifest,
            "stage4",
            STAGE4_FAMILIES,
            n_families=len(families),
            n_refinements=len(refinements),
        )
        return report

    def _refine_record(
        self, record: dict[str, Any], refinement: RefinementConfig
    ) -> dict[str, Any]:
        """Stage-4 refinement objective: short screening re-propagation."""

        def _score_fn(x: np.ndarray) -> float:
            y0 = elements_to_states(x[None, :], self.config.mu_m3s2)
            t, Y, impacts, t_imp = self.screening.propagate(
                y0,
                float(self.config.screening_duration_s),
                float(self.config.screening_output_dt_s),
            )
            if bool(np.asarray(impacts, dtype=np.float64)[0] > 0.5):
                return float("inf")
            guard_result = evaluate_domain_guard(
                t,
                Y,
                reference_radius_m=float(self.config.reference_radius_m),
                guard=self.config.guard,
            )
            if bool(guard_result.domain_exit_flag[0]):
                return float("inf")
            hist = _element_history(
                t, np.asarray(Y)[:, 0, :], self.config.mu_m3s2, self.config.reference_radius_m
            )
            metrics = compute_frozen_metrics(
                hist["t_s"],
                eccentricity=hist["e"],
                inclination_rad=hist["inc_rad"],
                omega_rad=hist["argp_rad"],
                h_peri_m=hist["h_peri_m"],
            )
            return frozen_score(metrics, self.config.classification_config())

        result = refine_candidate(record["elements"], _score_fn, refinement)
        out = result.to_dict()
        out["sample_index"] = int(record["sample_index"])
        return out

    # -- orchestration -----------------------------------------------------------

    def run(self, *, resume: bool = True) -> dict[str, Any]:
        """Run all stages (respecting existing stage files when ``resume``)."""
        manifest = self._load_manifest()
        t0 = time.perf_counter()
        elements = self.stage0_samples(manifest, resume=resume)
        screening = self.stage1_screening(manifest, elements, resume=resume)
        candidates = self.stage2_candidates(manifest, elements, screening, resume=resume)
        validated = self.stage3_validation(manifest, candidates, resume=resume)
        report = self.stage4_families(manifest, validated, resume=resume)
        elapsed = time.perf_counter() - t0
        logger.info(
            "frozen search complete: %d samples -> %d candidates -> %d families (%.1f s)",
            elements.shape[0],
            len(candidates),
            len(report.get("families", [])),
            elapsed,
        )
        manifest["elapsed_s"] = float(elapsed)
        _atomic_write_json(self._manifest_path(), manifest)
        return {
            "manifest": manifest,
            "candidates": candidates,
            "validated": validated,
            "family_report": report,
        }


__all__ = [
    "MANIFEST",
    "STAGE0_SAMPLES",
    "STAGE1_SCREENING",
    "STAGE1_SCREENING_SCHEMA_VERSION",
    "STAGE1_OUTPUT_FULL",
    "STAGE1_OUTPUT_SUMMARY_ONLY",
    "STAGE2_CANDIDATES",
    "STAGE3_VALIDATION",
    "STAGE4_FAMILIES",
    "ElementBounds",
    "FrozenSearchConfig",
    "FrozenSearchPipeline",
    "ScreeningPropagator",
    "ValidationPropagator",
    "elements_to_states",
    "enforce_classical_validation_rule",
    "sobol_element_samples",
]
