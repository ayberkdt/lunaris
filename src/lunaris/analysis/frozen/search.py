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
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
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
        Y0 = elements_to_states(elements, self.config.mu_m3s2)
        t_out, Y_out, impact_flags, t_impact = self.screening.propagate(
            Y0,
            float(self.config.screening_duration_s),
            float(self.config.screening_output_dt_s),
        )
        arrays = {
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
            n_snapshots=int(arrays["t_out"].shape[0]),
        )
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

        from lunaris.batch.summary import SCORE_DEFINITION, summarize_ensemble

        t_out = screening["t_out"]
        Y_out = screening["Y_out"]
        impact_flags = screening["impact_flags"]
        t_impact = screening["t_impact"]

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
        # Perilune safety floor is a hard candidate constraint (same rule the
        # stage-3 classifier applies); filtering here keeps the top-K honest.
        h_peri_min_km = np.asarray(summary["fields"]["h_peri_min_km"], dtype=np.float64)
        below_floor = h_peri_min_km < float(self.config.perilune_safety_min_m) / 1_000.0
        scores = np.where(below_floor, np.inf, scores)

        order = np.argsort(scores, kind="stable")
        candidates: list[dict[str, Any]] = []
        for j in order.tolist():
            if len(candidates) >= int(self.config.top_k):
                break
            if not np.isfinite(scores[j]):
                break  # remaining samples are +inf (impacted/invalid/domain-exit)
            fields = summary["fields"]
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
