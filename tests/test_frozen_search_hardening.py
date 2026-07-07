"""Scientific-hardening tests for the frozen-search pipeline.

Two contracts are locked here:

1. Stage-1 summary-only output (``screening_output_mode="summary_only"``) is a
   real artifact contract, not an optimization: no full ``(T, N, 6)`` ensemble
   tensor, per-sample ``summary__*`` arrays, top-K retained histories, and a
   hard failure on inconsistent time grids across summary batches. Stage 2
   must be able to consume it without ever touching ``Y_out``.

2. The paper-safe / strict-frame guard: with ``paper_safe=True`` or
   ``strict_frame=True`` no stage may run or persist candidates from a backend
   whose frame provenance is identity/unresolved (Moon-fixed gravity evaluated
   in the integration frame). Exploratory identity-frame runs stay allowed
   without the flags, but the manifest records the frame verbatim.

The fake propagators below are deterministic and analytic-free: static states
plus a tiny per-sample radial drift so screening scores are distinct (no
top-K tie ambiguity).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lunaris.analysis.frozen.domain_guard import FrozenSearchDomainGuard
from lunaris.analysis.frozen.search import (
    STAGE1_OUTPUT_FULL,
    STAGE1_OUTPUT_SUMMARY_ONLY,
    STAGE1_SCREENING,
    ElementBounds,
    FrozenSearchConfig,
    FrozenSearchPipeline,
    enforce_strict_frame_rule,
    is_identity_or_unresolved_frame,
)

MU = 4.9028001224453001e12
R_REF = 1.7374e6

MOON_FIXED_FRAME = "moon_fixed_slerp (ephemeris-wired q_i2f)"
IDENTITY_FRAME = "identity (gravity field fixed in the integration frame)"
UNRESOLVED_FRAME = "unresolved (ephemeris required)"


# ---------------------------------------------------------------------------
# Fake backends (ScreeningPropagator / ValidationPropagator protocols)
# ---------------------------------------------------------------------------


class FakeScreeningPropagator:
    """Deterministic fake stage-1 backend.

    Returns each sample's initial state at every snapshot, scaled by a tiny
    per-sample radial drift (distinct screening scores, no impacts). With
    ``shift_second_batch=True`` the second ``propagate`` call returns a shifted
    time grid, which the summary-only accumulator must reject.
    """

    backend_name = "fake_screening"

    def __init__(
        self,
        *,
        frame: str = MOON_FIXED_FRAME,
        shift_second_batch: bool = False,
        drift_scale: float = 1e-5,
    ) -> None:
        self.provenance = {"backend": "fake_screening", "frame": frame}
        self._calls = 0
        self._shift = bool(shift_second_batch)
        self._drift = float(drift_scale)

    def propagate(self, Y0, duration_s, output_dt_s):
        self._calls += 1
        Y0 = np.asarray(Y0, dtype=np.float64)
        n_snaps = int(round(duration_s / output_dt_s))
        t_out = np.linspace(0.0, float(duration_s), n_snaps + 1)
        if self._shift and self._calls > 1:
            t_out = t_out + 7.0  # inconsistent grid on the second batch
        n = Y0.shape[0]
        Y_out = np.repeat(Y0[None, :, :], t_out.size, axis=0).copy()
        # Per-sample radial drift, distinct because Sobol radii are distinct.
        r0 = np.linalg.norm(Y0[:, :3], axis=1)
        rate = self._drift * (r0 / float(r0.max()))
        scale = 1.0 + rate[None, :] * (t_out[:, None] / max(float(t_out[-1]), 1.0))
        Y_out[:, :, :3] *= scale[:, :, None]
        impact = np.zeros(n, dtype=np.float64)
        t_imp = np.full(n, np.nan, dtype=np.float64)
        return t_out, Y_out, impact, t_imp


class FakeValidationPropagator:
    """Deterministic fake stage-3 backend: exactly constant states."""

    def __init__(
        self,
        *,
        frame: str = MOON_FIXED_FRAME,
        backend_label: str = "classical_sh_deg8",
    ) -> None:
        self.backend_label = backend_label
        self.provenance = {
            "backend": backend_label,
            "frame": frame,
            "gravity_model": {"name": "fake", "degree": 8},
            "third_body": {"earth": False, "sun": False},
        }

    def propagate(self, y0, duration_s, output_dt_s):
        n_snaps = int(round(duration_s / output_dt_s))
        t = np.linspace(0.0, float(duration_s), n_snaps + 1)
        y = np.repeat(np.asarray(y0, dtype=np.float64)[None, :], t.size, axis=0)
        return t, y


def _config(**overrides) -> FrozenSearchConfig:
    kwargs = dict(
        bounds=ElementBounds(a_km=(2_000.0, 2_400.0), e=(0.01, 0.08), i_deg=(70.0, 110.0)),
        n_samples=16,
        seed=7,
        screening_duration_s=6 * 3_600.0,
        screening_output_dt_s=600.0,
        screening_summary_batch_size=8,  # 16 samples -> 2 summary batches
        top_k=4,
        validation_duration_s=12 * 3_600.0,
        validation_output_dt_s=600.0,
        guard=FrozenSearchDomainGuard(altitude_min_km=20.0, altitude_max_km=20_000.0),
        perilune_safety_min_m=10_000.0,
        mu_m3s2=MU,
        reference_radius_m=R_REF,
    )
    kwargs.update(overrides)
    return FrozenSearchConfig(**kwargs)


def _pipeline(tmp_path, config=None, *, screening=None, validation=None, **kwargs):
    return FrozenSearchPipeline(
        config or _config(),
        screening=screening or FakeScreeningPropagator(),
        validation=validation or FakeValidationPropagator(),
        out_dir=tmp_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Phase 4 — stage-1 summary-only contract
# ---------------------------------------------------------------------------


def test_summary_only_stage1_writes_no_full_ensemble(tmp_path):
    pipeline = _pipeline(tmp_path)
    manifest = pipeline._load_manifest()
    elements = pipeline.stage0_samples(manifest, resume=False)
    pipeline.stage1_screening(manifest, elements, resume=False)

    with np.load(tmp_path / STAGE1_SCREENING, allow_pickle=False) as stage1:
        assert "schema_version" in stage1.files
        assert str(stage1["output_mode"].item()) == STAGE1_OUTPUT_SUMMARY_ONLY
        assert "Y_out" not in stage1.files
        assert "summary__score" in stage1.files
        assert "summary__valid" in stage1.files
        assert "summary__domain_exit_flag" in stage1.files
        assert "topk_sample_indices" in stage1.files
        assert "topk_Y_out" in stage1.files
        n_t = int(stage1["t_out"].shape[0])
        k = int(stage1["topk_sample_indices"].shape[0])
        assert stage1["topk_Y_out"].shape == (n_t, k, 6)
        assert 1 <= k <= 4  # k_hist = max(top_k, refine_top_n, stage1_history_top_k, 1)
        # Per-sample arrays cover every sample across both batches.
        assert stage1["summary__score"].shape == (16,)


def test_full_stage1_mode_still_writes_full_ensemble(tmp_path):
    pipeline = _pipeline(tmp_path, _config(screening_output_mode=STAGE1_OUTPUT_FULL))
    manifest = pipeline._load_manifest()
    elements = pipeline.stage0_samples(manifest, resume=False)
    pipeline.stage1_screening(manifest, elements, resume=False)

    with np.load(tmp_path / STAGE1_SCREENING, allow_pickle=False) as stage1:
        assert str(stage1["output_mode"].item()) == STAGE1_OUTPUT_FULL
        assert "Y_out" in stage1.files
        assert stage1["Y_out"].shape[1] == 16


def test_inconsistent_time_grids_across_summary_batches_fail(tmp_path):
    pipeline = _pipeline(
        tmp_path, screening=FakeScreeningPropagator(shift_second_batch=True)
    )
    manifest = pipeline._load_manifest()
    elements = pipeline.stage0_samples(manifest, resume=False)
    with pytest.raises(RuntimeError, match="inconsistent time grids"):
        pipeline.stage1_screening(manifest, elements, resume=False)


def test_stage2_consumes_summary_only_output(tmp_path):
    pipeline = _pipeline(tmp_path)
    manifest = pipeline._load_manifest()
    elements = pipeline.stage0_samples(manifest, resume=False)
    screening = pipeline.stage1_screening(manifest, elements, resume=False)

    assert "Y_out" not in screening  # stage 2 has no full tensor to lean on
    candidates = pipeline.stage2_candidates(
        manifest, elements, screening, resume=False
    )
    assert 1 <= len(candidates) <= 4
    scores = np.asarray(screening["summary__score"], dtype=np.float64)
    for record in candidates:
        # Scores must come from the summary arrays, not a recomputation.
        assert record["screening_score"] == pytest.approx(
            float(scores[int(record["sample_index"])])
        )


def test_topk_histories_cover_stage2_candidates(tmp_path):
    """Candidate histories must be findable in topk_Y_out when Y_out is absent."""
    pipeline = _pipeline(tmp_path)
    manifest = pipeline._load_manifest()
    elements = pipeline.stage0_samples(manifest, resume=False)
    screening = pipeline.stage1_screening(manifest, elements, resume=False)
    candidates = pipeline.stage2_candidates(
        manifest, elements, screening, resume=False
    )

    retained = set(np.asarray(screening["topk_sample_indices"], dtype=np.int64).tolist())
    for record in candidates:
        assert int(record["sample_index"]) in retained
    # And each retained history is a full (T, 6) trajectory.
    n_t = int(screening["t_out"].shape[0])
    assert screening["topk_Y_out"].shape[0] == n_t
    assert screening["topk_Y_out"].shape[2] == 6


# ---------------------------------------------------------------------------
# Phase 5 — paper-safe / strict-frame guard
# ---------------------------------------------------------------------------


def test_is_identity_or_unresolved_frame_matrix():
    assert is_identity_or_unresolved_frame(None) is True
    assert is_identity_or_unresolved_frame({}) is True  # missing frame = fail closed
    assert is_identity_or_unresolved_frame({"frame": ""}) is True
    assert is_identity_or_unresolved_frame({"frame": IDENTITY_FRAME}) is True
    assert is_identity_or_unresolved_frame({"frame": UNRESOLVED_FRAME}) is True
    assert is_identity_or_unresolved_frame({"frame": MOON_FIXED_FRAME}) is False


def test_enforce_strict_frame_rule_is_noop_when_not_required():
    enforce_strict_frame_rule(
        {"frame": IDENTITY_FRAME}, strict_frame_required=False, role="screening"
    )  # no raise


def test_strict_frame_required_property():
    assert _config().strict_frame_required is False
    assert _config(paper_safe=True).strict_frame_required is True
    assert _config(strict_frame=True).strict_frame_required is True
    assert _config(paper_safe=True).to_dict()["strict_frame_required"] is True


def test_paper_safe_rejects_identity_screening_frame(tmp_path):
    with pytest.raises(RuntimeError, match="screening backend"):
        _pipeline(
            tmp_path,
            _config(paper_safe=True),
            screening=FakeScreeningPropagator(frame=IDENTITY_FRAME),
        )


def test_strict_frame_rejects_identity_validation_frame(tmp_path):
    with pytest.raises(RuntimeError, match="validation backend"):
        _pipeline(
            tmp_path,
            _config(strict_frame=True),
            validation=FakeValidationPropagator(frame=IDENTITY_FRAME),
        )


def test_paper_safe_rejects_unresolved_sensitivity_frame(tmp_path):
    with pytest.raises(RuntimeError, match="sensitivity validation"):
        _pipeline(
            tmp_path,
            _config(paper_safe=True),
            sensitivity_validations=[FakeValidationPropagator(frame=UNRESOLVED_FRAME)],
        )


def test_exploratory_identity_frame_allowed_and_recorded(tmp_path):
    """Without paper_safe/strict_frame the identity frame stays allowed for
    smoke/exploratory runs, but the manifest records it verbatim."""
    pipeline = _pipeline(
        tmp_path,
        screening=FakeScreeningPropagator(frame=IDENTITY_FRAME),
        validation=FakeValidationPropagator(frame=IDENTITY_FRAME),
    )
    products = pipeline.run(resume=False)
    assert len(products["candidates"]) >= 1

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["screening_provenance"]["frame"] == IDENTITY_FRAME
    assert manifest["validation_provenance"]["frame"] == IDENTITY_FRAME
    assert manifest["config"]["strict_frame_required"] is False


def test_paper_safe_run_with_ephemeris_wired_frames_succeeds(tmp_path):
    pipeline = _pipeline(tmp_path, _config(paper_safe=True))
    products = pipeline.run(resume=False)
    assert len(products["validated"]) >= 1

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["paper_safe"] is True
    assert manifest["config"]["strict_frame_required"] is True
    assert manifest["validation_provenance"]["frame"] == MOON_FIXED_FRAME


def test_validated_status_cannot_be_persisted_with_identity_frame(tmp_path):
    """Belt check: even if the construction guard is bypassed (backend swapped
    after init), stage 3 must refuse to write validated candidates."""
    pipeline = _pipeline(tmp_path, _config(paper_safe=True))
    manifest = pipeline._load_manifest()
    elements = pipeline.stage0_samples(manifest, resume=False)
    screening = pipeline.stage1_screening(manifest, elements, resume=False)
    candidates = pipeline.stage2_candidates(manifest, elements, screening, resume=False)
    assert candidates

    pipeline.validation = FakeValidationPropagator(frame=IDENTITY_FRAME)
    with pytest.raises(RuntimeError, match="validation backend"):
        pipeline.stage3_validation(manifest, candidates, resume=False)
    assert not (tmp_path / "stage3_validation.json").exists()


def test_screened_candidates_cannot_be_persisted_with_identity_frame(tmp_path):
    """Same belt at stage 2 for the screening backend."""
    pipeline = _pipeline(tmp_path, _config(paper_safe=True))
    manifest = pipeline._load_manifest()
    elements = pipeline.stage0_samples(manifest, resume=False)
    screening = pipeline.stage1_screening(manifest, elements, resume=False)

    pipeline.screening = FakeScreeningPropagator(frame=IDENTITY_FRAME)
    with pytest.raises(RuntimeError, match="screening backend"):
        pipeline.stage2_candidates(manifest, elements, screening, resume=False)
    assert not (tmp_path / "stage2_candidates.json").exists()
