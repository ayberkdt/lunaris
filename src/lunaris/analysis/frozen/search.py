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

Stage 1 is also resumable *within itself* (``screening_checkpoint_interval_s``),
because a 60-day screening of thousands of orbits is hours-to-days of compute
that used to be lost whole when a process died. Two nested levels, both under
``stage1_checkpoint/``: each finished screening block writes a summary partial
(``block_<lo>_<hi>_summary.npz``), and inside a block the shared batched loop
periodically checkpoints its integrator state
(:mod:`lunaris.core.batch_checkpoint`). A resumed run replays finished blocks,
continues the interrupted one from its last snapshot boundary, and produces a
stage-1 artifact identical to an uninterrupted run. The directory is deleted
once ``stage1_screening.npz`` exists.

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

import inspect
import json
import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping
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
# Stage-1 resumable work directory: per-block summary partials plus the shared
# batched loop integrator checkpoints (lunaris.core.batch_checkpoint). Removed
# once stage1_screening.npz is written.
STAGE1_CHECKPOINT_DIR = "stage1_checkpoint"

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

# Extra per-sample metrics written by the "frozen" screening score (optional keys).
_OPTIONAL_SUMMARY_FIELDS = (
    "de_dt_per_s",
    "dh_peri_dt_m_per_s",
    "inclination_range_rad",
    "dinclination_dt_rad_per_s",
    "omega_span_rad",
    "domega_dt_rad_per_s",
    "hk_loop_drift",
)
SCREENING_SCORE_V1 = "v1"
SCREENING_SCORE_FROZEN = "frozen"
SCREENING_SCORES = (SCREENING_SCORE_V1, SCREENING_SCORE_FROZEN)
FROZEN_SCREENING_SCORE_DEFINITION = (
    "screening v2: frozen_score(compute_frozen_metrics(history), "
    "FrozenClassificationConfig.for_mission_duration(screening_duration)) = sum of "
    "quasi-threshold ratios of e_range, |de/dt|, h_peri range, |dh_peri/dt|, omega span, "
    "|domega/dt|, h-k loop drift; +inf for impact / domain exit / escape / unsafe perilune / "
    "e above bound; lower is better"
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
        if float(self.e[1]) >= 1.0:
            # e >= 1 is not a bound orbit; it would fail later in the element
            # conversion as a degenerate conic.
            raise ValueError("eccentricity upper bound must be < 1 (elliptic orbits)")
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
    hp_km: tuple[float, float] | None = None,
    argp_choices: tuple[float, ...] | None = None,
    r_ref_km: float = 1_737.4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw ``(N, 6)`` element samples in the box, with full provenance.

    ``method`` is ``"sobol"``, ``"sobol_scrambled"``, or ``"lhs"``. Sobol draws
    a base-2 design and keeps the first ``n_samples`` (recorded in provenance).
    With ``hp_km`` the first Sobol dimension samples the perilune altitude and
    ``a = (R + h_p) / (1 - e)`` is derived (``bounds.a_km`` is ignored for the
    draw); with ``argp_choices`` the argp dimension is mapped onto that discrete
    set with equal measure.
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
    provenance: dict[str, Any] = {
        "sampling_method": method,
        "sobol_seed": int(seed),
        "sample_count": int(n_samples),
        "bounds": bounds.to_dict(),
        "note": base2_note,
    }
    if hp_km is not None:
        hp_lo, hp_hi = float(hp_km[0]), float(hp_km[1])
        hp = hp_lo + unit[:, 0] * (hp_hi - hp_lo)
        elements[:, 0] = (float(r_ref_km) + hp) / (1.0 - elements[:, 1])
        provenance["hp_km"] = [hp_lo, hp_hi]
        provenance["a_km_derived_range"] = [float(elements[:, 0].min()), float(elements[:, 0].max())]
    if argp_choices is not None:
        choices = np.asarray(argp_choices, dtype=np.float64)
        idx = np.minimum((unit[:, 4] * choices.size).astype(int), choices.size - 1)
        elements[:, 4] = choices[idx]
        provenance["argp_choices_deg"] = [float(c) for c in choices]
    return np.ascontiguousarray(elements, dtype=np.float64), provenance


def elements_to_states(
    elements: np.ndarray, mu_m3s2: float, *, frame_rotation: Any = None
) -> np.ndarray:
    """Convert ``(N, 6)`` element rows (km/deg) to Cartesian states ``(N, 6)`` (SI).

    The elements are measured in the element frame; ``frame_rotation`` is the
    inertial -> element-frame matrix (see :func:`rotate_states_element_frame`)
    and the returned states are in the inertial integration frame.
    """
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
    return rotate_states_element_frame(out, frame_rotation, to_inertial=True)


# ---------------------------------------------------------------------------
# Element frame (lunar equator of epoch vs. the integration frame)
# ---------------------------------------------------------------------------

#: Elements are measured in the integration (inertial J2000) frame. Correct only
#: when the gravity field is fixed in that frame (identity diagnostic frame).
ELEMENT_FRAME_INERTIAL_IDENTITY = "inertial_identity"
#: Elements are measured against the Moon's equator at the run's t = 0. When the
#: field rotates with the Moon-fixed frame, inclination and argument of
#: perilune only mean "against the lunar equator" in this frame; the J2000
#: equator is tilted about 23.5 deg from it.
ELEMENT_FRAME_LUNAR_EQUATOR = "lunar_equator_of_epoch"


def _as_rotation_matrix(rotation: Any) -> np.ndarray | None:
    """Validate a ``(3, 3)`` rotation; ``None`` for the exact identity (no-op)."""
    if rotation is None:
        return None
    mat = np.asarray(rotation, dtype=np.float64)
    if mat.shape != (3, 3) or not np.all(np.isfinite(mat)):
        raise ValueError(f"element-frame rotation must be a finite (3, 3) matrix, got {mat.shape}")
    if not np.allclose(mat @ mat.T, np.eye(3), rtol=0.0, atol=1e-9) or np.linalg.det(mat) < 0.0:
        raise ValueError("element-frame rotation must be a proper rotation matrix")
    if np.array_equal(mat, np.eye(3)):
        return None
    return mat


def rotate_states_element_frame(
    states: np.ndarray, rotation: Any, *, to_inertial: bool = False
) -> np.ndarray:
    """Rotate ``(..., 6)`` states between the integration and element frames.

    ``rotation`` is the constant inertial -> element-frame matrix ``R``. By
    default ``r' = R r`` and ``v' = R v``; ``to_inertial=True`` applies ``R^T``.
    A ``None`` or exactly-identity ``rotation`` returns *states* unchanged, so
    identity-frame numbers stay bit-identical.
    """
    mat = _as_rotation_matrix(rotation)
    if mat is None:
        return states
    states = np.asarray(states, dtype=np.float64)
    op = mat if to_inertial else mat.T  # row-vector form: x @ R.T == (R x)
    return np.concatenate([states[..., :3] @ op, states[..., 3:] @ op], axis=-1)


def quaternion_i2f_to_matrix(q_i2f: Any) -> np.ndarray:
    """Matrix ``R`` with ``r_fixed = R r_inertial`` for a scalar-first ``q_i2f``.

    Built column by column with :func:`lunaris.common.math_utils.quat_rotate_np`,
    the same rotation the propagators apply, so no second quaternion convention
    is introduced here.
    """
    from lunaris.common.math_utils import quat_rotate_np

    q = np.asarray(q_i2f, dtype=np.float64).reshape(4)
    return np.column_stack([quat_rotate_np(q, axis) for axis in np.eye(3)])


def resolve_element_frame(
    *backends: tuple[str, Any], explicit_rotation: Any = None
) -> tuple[np.ndarray | None, dict[str, str]]:
    """Pick the element frame from ``(role, backend)`` pairs; the first rotating one wins.

    A backend whose provenance reports ``uses_frame_rotation`` flies a gravity
    field tied to the Moon-fixed frame; its ``ephem_manager`` supplies the
    inertial -> Moon-fixed quaternion at ``t = 0`` (the epoch every backend
    propagates from), which defines the lunar equator of epoch. Backends that
    report no rotation keep the integration frame. Returns the matrix (``None``
    for identity) and a ``{"frame", "rotation_source"}`` record.
    """
    if explicit_rotation is not None:
        mat = _as_rotation_matrix(explicit_rotation)
        if mat is not None:
            return mat, {
                "frame": ELEMENT_FRAME_LUNAR_EQUATOR,
                "rotation_source": "explicit rotation matrix",
            }
        return None, {
            "frame": ELEMENT_FRAME_INERTIAL_IDENTITY,
            "rotation_source": "explicit identity rotation",
        }
    for role, backend in backends:
        prov = getattr(backend, "provenance", None)
        if not (isinstance(prov, Mapping) and prov.get("uses_frame_rotation") is True):
            continue
        ephem = getattr(backend, "ephem_manager", None)
        if ephem is None:
            raise RuntimeError(
                f"{role} backend reports a rotating Moon-fixed frame but exposes no "
                "ephem_manager, so the lunar equator of epoch cannot be resolved."
            )
        mat = _as_rotation_matrix(
            quaternion_i2f_to_matrix(ephem.get_inertial_to_fixed_rotation(0.0))
        )
        return mat, {
            "frame": ELEMENT_FRAME_LUNAR_EQUATOR,
            "rotation_source": f"{role} backend ephemeris q_i2f at t=0",
        }
    return None, {
        "frame": ELEMENT_FRAME_INERTIAL_IDENTITY,
        "rotation_source": "no backend reports a rotating Moon-fixed frame",
    }


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
    # Optional perilune-altitude sampling (km): ``a`` is derived as (R + h_p)/(1 - e)
    # after ``e`` is drawn; ``bounds.a_km`` then only documents the derived range.
    hp_km: tuple[float, float] | None = None
    # Optional discrete argument-of-perilune set (deg) mapped onto the Sobol dimension.
    argp_choices: tuple[float, ...] | None = None
    # Stage-1 ranking: "v1" (e/h_peri envelope score) or "frozen" (classification score).
    screening_score: str = SCREENING_SCORE_V1

    screening_duration_s: float = 7.0 * 86_400.0
    screening_output_dt_s: float = 3_600.0
    screening_output_mode: str = STAGE1_OUTPUT_SUMMARY_ONLY
    screening_summary_batch_size: int = 4_096
    stage1_history_top_k: int | None = None
    # Stage-1 resumability: the shared batched loop checkpoints its integrator
    # state at least this often (wall-clock seconds) and every finished
    # screening block is persisted; ``0`` disables both. A killed run then
    # resumes from the last boundary with bitwise-identical results.
    screening_checkpoint_interval_s: float = 1_200.0

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

    # Paper-safe / strict-frame guard: Moon-fixed gravity evaluated in an
    # identity (frozen) integration frame is acceptable for smoke/exploratory
    # runs, never for paper-grade claims. When either flag is set, backends
    # whose frame provenance is identity/unresolved are rejected outright.
    paper_safe: bool = False
    strict_frame: bool = False

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
        if float(self.screening_checkpoint_interval_s) < 0.0:
            raise ValueError("screening_checkpoint_interval_s must be >= 0")
        for name in ("screening_duration_s", "validation_duration_s"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if int(self.refine_top_n) > 0 and self.refinement is None:
            raise ValueError("refine_top_n > 0 requires a RefinementConfig")
        if self.screening_score not in SCREENING_SCORES:
            raise ValueError(f"screening_score must be one of {SCREENING_SCORES}, got {self.screening_score!r}")
        if self.hp_km is not None:
            lo, hi = float(self.hp_km[0]), float(self.hp_km[1])
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
                raise ValueError(f"hp_km must satisfy lo <= hi, got {self.hp_km}")
        if self.argp_choices is not None and len(self.argp_choices) == 0:
            raise ValueError("argp_choices must be non-empty when given")

    @property
    def strict_frame_required(self) -> bool:
        """True when identity/unresolved rotation frames must be rejected."""
        return bool(self.paper_safe) or bool(self.strict_frame)

    def score_definition(self) -> str:
        if self.screening_score == SCREENING_SCORE_FROZEN:
            return FROZEN_SCREENING_SCORE_DEFINITION
        from lunaris.batch.summary import SCORE_DEFINITION

        return str(SCORE_DEFINITION)

    def classification_config(self) -> FrozenClassificationConfig:
        return FrozenClassificationConfig.for_mission_duration(
            float(self.validation_duration_s),
            perilune_safety_min_m=float(self.perilune_safety_min_m),
            eccentricity_upper_bound=float(self.eccentricity_upper_bound),
        )

    def screening_classification_config(self) -> FrozenClassificationConfig:
        """Classification thresholds scaled to the (shorter) screening duration."""
        return FrozenClassificationConfig.for_mission_duration(
            float(self.screening_duration_s),
            perilune_safety_min_m=float(self.perilune_safety_min_m),
            eccentricity_upper_bound=float(self.eccentricity_upper_bound),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["bounds"] = self.bounds.to_dict()
        out["guard"] = self.guard.to_dict()
        out["refinement"] = self.refinement.to_dict() if self.refinement else None
        out["strict_frame_required"] = self.strict_frame_required
        return out


# ---------------------------------------------------------------------------
# R21 + strict-frame enforcement
# ---------------------------------------------------------------------------

#: Frame-provenance markers that mean "Moon-fixed gravity was evaluated
#: without a real inertial->body rotation" (see search_backends._frame_provenance).
_IDENTITY_FRAME_MARKERS = (
    "identity",
    "unresolved",
    "gravity field fixed in the integration frame",
)


def is_identity_or_unresolved_frame(provenance: Mapping[str, Any] | None) -> bool:
    """True when a backend's frame provenance is identity/unresolved.

    Fail-closed: a missing provenance block or a missing/empty ``frame`` entry
    counts as unresolved, because an unlabeled frame can never support a
    paper-grade Moon-fixed gravity claim.
    """
    if not isinstance(provenance, Mapping):
        return True
    frame = str(provenance.get("frame", "")).strip().lower()
    if not frame:
        return True
    return any(marker in frame for marker in _IDENTITY_FRAME_MARKERS)


def enforce_strict_frame_rule(
    provenance: Mapping[str, Any] | None,
    *,
    strict_frame_required: bool,
    role: str,
) -> None:
    """Refuse identity/unresolved rotation frames under paper_safe/strict_frame.

    Raises ``RuntimeError`` when *strict_frame_required* is true and the
    backend's frame provenance is identity, unresolved, or absent. Called at
    pipeline construction (fail-fast) and again before stage-2/3 persistence
    (belt: a manually constructed pipeline or edited manifest must not survive
    the write).
    """
    if not strict_frame_required:
        return
    if is_identity_or_unresolved_frame(provenance):
        frame = (
            provenance.get("frame") if isinstance(provenance, Mapping) else None
        )
        raise RuntimeError(
            f"paper_safe/strict_frame forbids the {role} backend's rotation frame "
            f"provenance {frame!r}: Moon-fixed gravity evaluated in an identity/"
            "unresolved integration frame can never back a paper-grade frozen-orbit "
            "claim. Wire an ephemeris manager (moon_fixed_ephemeris) or drop the "
            "paper_safe/strict_frame flag for exploratory runs."
        )


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
    # Resolve the commit of the checkout this package runs from, not of the
    # process's working directory (which may be another repository).
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # R29b-justified: provenance nicety, never fatal
        return "unknown"


def _element_history(
    t: np.ndarray, y: np.ndarray, mu: float, r_ref: float, frame_rotation: Any = None
) -> dict[str, np.ndarray]:
    """Osculating element histories for one ``(T, 6)`` trajectory (element frame)."""
    from lunaris.common.math_utils import screening_orbital_elements_vec

    y = rotate_states_element_frame(y, frame_rotation)
    a_m, e, inc, argp = screening_orbital_elements_vec(
        y[:, :3], y[:, 3:], float(mu)
    )
    return {
        "t_s": np.asarray(t, dtype=np.float64),
        "a_m": a_m,
        "e": e,
        "inc_rad": inc,
        "argp_rad": argp,
        "h_peri_m": a_m * (1.0 - e) - float(r_ref),
    }


def frozen_screening_scores(
    t_out: np.ndarray,
    Y_out: np.ndarray,
    impact_flags: np.ndarray,
    t_impact: np.ndarray,
    guard_result: DomainGuardResult,
    *,
    mu: float,
    r_ref: float,
    cls_config: FrozenClassificationConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-sample :func:`frozen_score` of a ``(T, N, 6)`` screening block.

    Same metric set and thresholds as stage-3 classification (scaled to the
    screening duration), so stage-1 ranks by the objective stage 3 judges.
    Returns the raw scores (``+inf`` for impact / domain exit / escape /
    unsafe perilune) and the extra per-sample metrics to persist.
    """
    from lunaris.common.math_utils import screening_orbital_elements_vec

    t = np.asarray(t_out, dtype=np.float64)
    Y = np.asarray(Y_out, dtype=np.float64)
    n = int(Y.shape[1])
    a_m, e, inc, argp = screening_orbital_elements_vec(Y[:, :, :3], Y[:, :, 3:], float(mu))
    h_peri = a_m * (1.0 - e) - float(r_ref)
    impacts = np.asarray(impact_flags, dtype=np.float64) > 0.5
    t_imp = np.asarray(t_impact, dtype=np.float64)
    scores = np.full(n, np.inf, dtype=np.float64)
    extra = {key: np.full(n, np.nan, dtype=np.float64) for key in _OPTIONAL_SUMMARY_FIELDS}
    for j in range(n):
        exit_t = (
            float(guard_result.t_domain_exit_s[j])
            if bool(guard_result.domain_exit_flag[j])
            else None
        )
        impact_t = float(t_imp[j]) if bool(impacts[j]) and np.isfinite(t_imp[j]) else None
        valid = np.isfinite(e[:, j]) & np.isfinite(h_peri[:, j]) & np.isfinite(inc[:, j]) & np.isfinite(argp[:, j])
        if impact_t is not None:
            valid &= t <= impact_t
        if int(np.count_nonzero(valid)) < int(cls_config.min_valid_samples):
            continue
        try:
            metrics = compute_frozen_metrics(
                t,
                eccentricity=e[:, j],
                inclination_rad=inc[:, j],
                omega_rad=argp[:, j],
                h_peri_m=h_peri[:, j],
                impact_time_s=impact_t,
                domain_exit_time_s=exit_t,
                escape=bool(guard_result.escape_flag[j]),
                valid_mask=valid,
            )
        except ValueError:
            continue
        scores[j] = float(frozen_score(metrics, cls_config))
        for key in _OPTIONAL_SUMMARY_FIELDS:
            extra[key][j] = float(getattr(metrics, key))
    return scores, extra


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
    for key in _OPTIONAL_SUMMARY_FIELDS:
        if key in fields:
            arrays[f"summary__{key}"] = np.asarray(fields[key], dtype=np.float64)
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
    fields = {
        key: np.asarray(screening[f"summary__{key}"])
        for key in _SCREENING_SUMMARY_FIELDS
    }
    for key in _OPTIONAL_SUMMARY_FIELDS:
        if f"summary__{key}" in screening:
            fields[key] = np.asarray(screening[f"summary__{key}"])
    return fields


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class FrozenSearchPipeline:
    """Staged, resumable surrogate-assisted frozen-orbit search (R04)."""

    def _score_definition(self) -> str:
        return self.config.score_definition()

    def __init__(
        self,
        config: FrozenSearchConfig,
        *,
        screening: ScreeningPropagator,
        validation: ValidationPropagator,
        out_dir: str | Path,
        sensitivity_validations: list[ValidationPropagator] | None = None,
        element_frame_rotation: Any = None,
    ) -> None:
        self.config = config
        self.screening = screening
        self.validation = validation
        self.sensitivity_validations = list(sensitivity_validations or [])
        # Paper-safe / strict-frame guard: fail at construction, before any
        # stage file is written (belt checks re-run at stage-2/3 persistence).
        strict = config.strict_frame_required
        enforce_strict_frame_rule(
            getattr(screening, "provenance", None),
            strict_frame_required=strict,
            role="screening",
        )
        enforce_strict_frame_rule(
            getattr(validation, "provenance", None),
            strict_frame_required=strict,
            role="validation",
        )
        for extra in self.sensitivity_validations:
            enforce_strict_frame_rule(
                getattr(extra, "provenance", None),
                strict_frame_required=strict,
                role=f"sensitivity validation ({getattr(extra, 'backend_label', '?')})",
            )
        # Inclination / argument of perilune are measured against the lunar
        # equator whenever the field rotates with the Moon; the validation
        # backend (which makes the classification claims) decides first.
        self.element_frame_rotation, self.element_frame = resolve_element_frame(
            ("validation", validation),
            ("screening", screening),
            explicit_rotation=element_frame_rotation,
        )
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
            "element_frame": self.element_frame["frame"],
            "element_frame_rotation_source": self.element_frame["rotation_source"],
            "element_frame_rotation": _json_safe(self.element_frame_rotation),
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
            hp_km=self.config.hp_km,
            argp_choices=self.config.argp_choices,
            r_ref_km=float(self.config.reference_radius_m) / 1_000.0,
        )
        np.savez_compressed(path, elements=elements)
        provenance["element_frame"] = self.element_frame["frame"]
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
            arrays, checkpoint_info = self._stage1_screening_summary_only(
                elements, resume=resume
            )
            np.savez_compressed(path, **arrays)
            self._discard_stage1_checkpoints()
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
                checkpoint=checkpoint_info,
            )
            return arrays

        Y0 = elements_to_states(
            elements, self.config.mu_m3s2, frame_rotation=self.element_frame_rotation
        )
        t_out, Y_out, impact_flags, t_impact = self._propagate_screening_block(
            Y0, 0, int(Y0.shape[0])
        )
        self._discard_stage1_checkpoints()
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
            checkpoint={
                "interval_s": float(self.config.screening_checkpoint_interval_s),
                "loop_checkpoint_supported": self._screening_supports_checkpoint(),
            },
        )
        return arrays

    # -- stage 1 resumability --------------------------------------------------

    def _stage1_checkpoint_dir(self) -> Path:
        return self.out_dir / STAGE1_CHECKPOINT_DIR

    def _screening_supports_checkpoint(self) -> bool:
        """True when the injected screening backend accepts ``checkpoint=``."""
        try:
            params = inspect.signature(self.screening.propagate).parameters
        except (TypeError, ValueError):
            return False
        return "checkpoint" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def _propagate_screening_block(
        self, Y0: np.ndarray, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propagate samples ``[start, end)`` with loop checkpoints when enabled."""
        interval_s = float(self.config.screening_checkpoint_interval_s)
        if interval_s > 0.0 and self._screening_supports_checkpoint():
            from lunaris.core.batch_checkpoint import BatchCheckpointSpec

            spec = BatchCheckpointSpec(
                directory=self._stage1_checkpoint_dir() / f"block_{start:07d}_{end:07d}",
                interval_s=interval_s,
            )
            return self.screening.propagate(  # type: ignore[call-arg]
                Y0,
                float(self.config.screening_duration_s),
                float(self.config.screening_output_dt_s),
                checkpoint=spec,
            )
        return self.screening.propagate(
            Y0,
            float(self.config.screening_duration_s),
            float(self.config.screening_output_dt_s),
        )

    def _stage1_block_partial_path(self, start: int, end: int) -> Path:
        return self._stage1_checkpoint_dir() / f"block_{start:07d}_{end:07d}_summary.npz"

    def _discard_stage1_checkpoints(self) -> None:
        """Remove the stage-1 work directory once stage1_screening.npz exists."""
        cdir = self._stage1_checkpoint_dir()
        if cdir.exists():
            shutil.rmtree(cdir, ignore_errors=True)

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

        # Stored states stay inertial; every element-based quantity below is
        # measured in the element frame. Radii (domain guard) are invariant.
        Y_elem = rotate_states_element_frame(Y_out, self.element_frame_rotation)
        summary = summarize_ensemble(
            t_out,
            Y_elem,
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
        if self.config.screening_score == SCREENING_SCORE_FROZEN:
            raw_scores, extra = frozen_screening_scores(
                t_out,
                Y_elem,
                impact_flags,
                t_impact,
                guard_result,
                mu=float(self.config.mu_m3s2),
                r_ref=float(self.config.reference_radius_m),
                cls_config=self.config.screening_classification_config(),
            )
            summary["fields"].update(extra)
            summary["fields"]["score"] = raw_scores
            summary["score_definition"] = FROZEN_SCREENING_SCORE_DEFINITION
        else:
            raw_scores = np.asarray(summary["fields"]["score"], dtype=np.float64)
        scores = apply_domain_guard_to_scores(raw_scores, guard_result, self.config.guard)
        h_peri_min_km = np.asarray(summary["fields"]["h_peri_min_km"], dtype=np.float64)
        below_floor = h_peri_min_km < float(self.config.perilune_safety_min_m) / 1_000.0
        scores = np.where(below_floor, np.inf, scores)
        return summary, guard_result, raw_scores, scores

    def _stage1_screening_summary_only(
        self, elements: np.ndarray, *, resume: bool = True
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        from lunaris.batch.summary import TopKTrajectoryBuffer

        Y0 = elements_to_states(
            elements, self.config.mu_m3s2, frame_rotation=self.element_frame_rotation
        )
        n_samples = int(Y0.shape[0])
        batch_size = int(self.config.screening_summary_batch_size)
        history_k = self._stage1_history_top_k()
        topk = TopKTrajectoryBuffer(history_k)
        summary_parts: list[dict[str, np.ndarray]] = []
        t_ref: np.ndarray | None = None
        persist_blocks = float(self.config.screening_checkpoint_interval_s) > 0.0
        blocks_total = 0
        blocks_restored = 0

        def _check_grid(t_arr: np.ndarray) -> None:
            nonlocal t_ref
            if t_ref is None:
                t_ref = t_arr
            elif t_arr.shape != t_ref.shape or not np.allclose(t_arr, t_ref, rtol=0.0, atol=1e-9):
                raise RuntimeError(
                    "screening propagator returned inconsistent time grids across "
                    "summary batches"
                )

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            blocks_total += 1
            partial_path = self._stage1_block_partial_path(start, end)
            if persist_blocks and resume and partial_path.exists():
                # Finished block from an interrupted run: replay its summary
                # rows and its block-local top-K histories, skip propagation.
                with np.load(partial_path, allow_pickle=False) as data:
                    part = {key: np.asarray(data[key]) for key in data.files}
                _check_grid(np.asarray(part.pop("t_out"), dtype=np.float64))
                idx = np.asarray(part.pop("topk_sample_indices"), dtype=np.int64)
                sc = np.asarray(part.pop("topk_scores"), dtype=np.float64)
                traj = np.asarray(part.pop("topk_Y_out"), dtype=np.float64)
                fl = np.asarray(part.pop("topk_impact_flags"), dtype=np.float64)
                ti = np.asarray(part.pop("topk_t_impact_s"), dtype=np.float64)
                for j in range(int(idx.shape[0])):
                    topk.offer_batch(
                        global_start=int(idx[j]),
                        scores=sc[j : j + 1],
                        Y_batch=traj[:, j : j + 1, :],
                        impact_flags=fl[j : j + 1],
                        t_impact=ti[j : j + 1],
                    )
                summary_parts.append(part)
                blocks_restored += 1
                logger.info(
                    "[FROZEN-SEARCH] stage 1: block [%d, %d) restored from %s",
                    start, end, partial_path.name,
                )
                continue

            t_out, Y_out, impact_flags, t_impact = self._propagate_screening_block(
                Y0[start:end], start, end
            )
            t_arr = np.asarray(t_out, dtype=np.float64)
            Y_arr = np.asarray(Y_out, dtype=np.float64)
            impacts = np.asarray(impact_flags, dtype=np.float64)
            t_imp = np.asarray(t_impact, dtype=np.float64)
            _check_grid(t_arr)

            summary, guard_result, raw_scores, scores = self._screening_scores_for_block(
                t_arr,
                Y_arr,
                impacts,
                t_imp,
            )
            part = _summary_arrays(
                summary=summary,
                guard_result=guard_result,
                scores=scores,
                raw_scores=raw_scores,
            )
            summary_parts.append(part)
            if persist_blocks:
                # Block-local top-K (same K as the global buffer) is sufficient
                # to rebuild the global top-K on resume.
                local = TopKTrajectoryBuffer(history_k)
                local.offer_batch(
                    global_start=start,
                    scores=scores,
                    Y_batch=Y_arr,
                    impact_flags=impacts,
                    t_impact=t_imp,
                )
                local_arrays = local.entry_arrays()
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = partial_path.with_name(partial_path.name + ".tmp.npz")
                np.savez(
                    tmp_path,
                    t_out=t_arr,
                    topk_sample_indices=local_arrays["sample_indices"],
                    topk_scores=local_arrays["scores"],
                    topk_Y_out=local.stacked_trajectories(int(t_arr.shape[0])),
                    topk_impact_flags=local_arrays["impact_flags"],
                    topk_t_impact_s=local_arrays["t_impact_s"],
                    **part,
                )
                os.replace(tmp_path, partial_path)
            topk.offer_batch(
                global_start=start,
                scores=scores,
                Y_batch=Y_arr,
                impact_flags=impacts,
                t_impact=t_imp,
            )

        if t_ref is None:
            raise RuntimeError("stage-1 screening produced no batches")
        checkpoint_info = {
            "interval_s": float(self.config.screening_checkpoint_interval_s),
            "loop_checkpoint_supported": self._screening_supports_checkpoint(),
            "blocks_total": int(blocks_total),
            "blocks_restored": int(blocks_restored),
        }

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
        return arrays, checkpoint_info

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
            for key in _OPTIONAL_SUMMARY_FIELDS:
                if key in fields:
                    record["summary"][key] = float(fields[key][j])
            candidates.append(record)

        # Strict-frame belt: screened candidates from an identity/unresolved
        # frame must not be persisted under paper_safe/strict_frame.
        enforce_strict_frame_rule(
            getattr(self.screening, "provenance", None),
            strict_frame_required=self.config.strict_frame_required,
            role="screening",
        )
        payload = {
            "run_id": manifest["run_id"],
            "score_definition": self._score_definition(),
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
        hist = _element_history(
            t,
            y,
            self.config.mu_m3s2,
            self.config.reference_radius_m,
            self.element_frame_rotation,
        )
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
            y0 = elements_to_states(
                elements[None, :], self.config.mu_m3s2, frame_rotation=self.element_frame_rotation
            )[0]

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
            # Strict-frame belt: a validated status must never be written from
            # an identity/unresolved validation frame under paper_safe.
            enforce_strict_frame_rule(
                getattr(self.validation, "provenance", None),
                strict_frame_required=self.config.strict_frame_required,
                role="validation",
            )
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
            score_definition=self._score_definition(),
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
            y0 = elements_to_states(
                x[None, :], self.config.mu_m3s2, frame_rotation=self.element_frame_rotation
            )
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
                t,
                np.asarray(Y)[:, 0, :],
                self.config.mu_m3s2,
                self.config.reference_radius_m,
                self.element_frame_rotation,
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
        # A manifest without "element_frame" predates the element-frame fix and
        # measured elements in the integration frame, which is the current
        # identity-frame behaviour but not the lunar-equator one.
        stored_frame = manifest.get("element_frame", ELEMENT_FRAME_INERTIAL_IDENTITY)
        if resume and stored_frame != self.element_frame["frame"]:
            raise RuntimeError(
                f"cannot resume {self.out_dir}: its stage files were written with "
                f"element_frame={stored_frame!r} but this run uses "
                f"{self.element_frame['frame']!r}; stage files from two element frames "
                "cannot be mixed. Use a new out_dir or resume=False."
            )
        manifest["element_frame"] = self.element_frame["frame"]
        manifest["element_frame_rotation_source"] = self.element_frame["rotation_source"]
        manifest["element_frame_rotation"] = _json_safe(self.element_frame_rotation)
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
    "ELEMENT_FRAME_INERTIAL_IDENTITY",
    "ELEMENT_FRAME_LUNAR_EQUATOR",
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
    "enforce_strict_frame_rule",
    "is_identity_or_unresolved_frame",
    "quaternion_i2f_to_matrix",
    "resolve_element_frame",
    "rotate_states_element_frame",
    "sobol_element_samples",
]
