"""ST-LRPS error-decomposition schema: paper-safe reports must separate
field error, orbit error, integrator error, phase-corrected error, and
runtime — a trajectory number can never masquerade as a field-accuracy claim.

Pure-schema tests: no torch, no benchmark computation, no fabricated numbers.
"""

from __future__ import annotations

import pytest

from lunaris.surrogate.st_lrps.evaluation.error_decomposition import (
    ERROR_DECOMPOSITION_SCHEMA_VERSION,
    REQUIRED_TOP_LEVEL_BLOCKS,
    build_error_decomposition_schema,
    validate_error_decomposition,
    validate_paper_safe_error_decomposition,
)


def _complete_block(**overrides):
    kwargs = dict(
        field_error={"accel_rms_mgal": 1.2, "n_points": 4096},
        orbit_error={"pos_err_rms_m": 3.4, "n_scenarios": 32},
        integrator_error={"pos_err_rms_m": 0.1},
        phase_corrected_error={"pos_err_rms_m": 1.1},
        runtime={"wall_s": 12.5, "samples_per_s": 320.0},
        requested_backend="gpu_st_lrps_potential",
        actual_backend="gpu_st_lrps_potential",
        requested_dtype="float64",
        effective_dtype="float64",
        frame_mode="moon_fixed_slerp",
        identity_rotation_used=False,
        synthetic=False,
        paper_safe=True,
    )
    kwargs.update(overrides)
    return build_error_decomposition_schema(**kwargs)


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------


def test_build_schema_returns_all_required_blocks():
    block = build_error_decomposition_schema()
    assert block["schema_version"] == ERROR_DECOMPOSITION_SCHEMA_VERSION
    for name in REQUIRED_TOP_LEVEL_BLOCKS:
        assert name in block, name
    for name in ("field_error", "orbit_error", "integrator_error",
                 "phase_corrected_error", "runtime"):
        assert block[name]["present"] is False  # nothing measured -> explicit
        assert block[name]["metrics"] == {}
        assert block[name]["description"]
    assert set(block["provenance"]) >= {
        "requested_backend", "actual_backend", "requested_dtype",
        "effective_dtype", "frame_mode", "identity_rotation_used",
        "synthetic", "paper_safe",
    }


def test_build_schema_marks_supplied_metrics_present():
    block = _complete_block()
    assert block["field_error"]["present"] is True
    assert block["field_error"]["metrics"]["accel_rms_mgal"] == 1.2
    assert block["orbit_error"]["present"] is True
    assert block["runtime"]["present"] is True
    assert block["scientific_evidence"] is True


def test_synthetic_build_is_never_scientific_evidence():
    block = _complete_block(synthetic=True)
    assert block["provenance"]["synthetic"] is True
    assert block["scientific_evidence"] is False


# ---------------------------------------------------------------------------
# Structural validation (quick mode included)
# ---------------------------------------------------------------------------


def test_structural_validation_passes_for_built_blocks():
    assert validate_error_decomposition(_complete_block()) == []
    assert validate_error_decomposition(build_error_decomposition_schema()) == []


def test_structural_validation_flags_missing_blocks_and_version():
    block = _complete_block()
    del block["orbit_error"]
    block["schema_version"] = "bogus_v0"
    problems = validate_error_decomposition(block)
    assert any("orbit_error" in p for p in problems)
    assert any("schema_version" in p for p in problems)


def test_quick_mode_allowed_only_when_marked_non_scientific():
    honest = _complete_block(synthetic=True)  # build stamps scientific_evidence=False
    assert validate_error_decomposition(honest) == []

    dishonest = _complete_block(synthetic=True)
    dishonest["scientific_evidence"] = True  # manual promotion attempt
    problems = validate_error_decomposition(dishonest)
    assert any("scientific_evidence" in p for p in problems)


# ---------------------------------------------------------------------------
# Paper-safe validation
# ---------------------------------------------------------------------------


def test_paper_safe_passes_with_complete_non_synthetic_evidence():
    validate_paper_safe_error_decomposition(_complete_block())  # no raise


def test_paper_safe_fails_without_field_error():
    block = _complete_block(field_error=None)
    with pytest.raises(ValueError, match="field_error"):
        validate_paper_safe_error_decomposition(block)


def test_paper_safe_trajectory_only_claim_does_not_require_field_error():
    # A trajectory_only claim relaxes the field_error requirement but still
    # demands orbit_error, runtime, provenance, non-synthetic, non-identity.
    block = _complete_block(field_error=None, paper_safe_claim_type="trajectory_only")
    validate_paper_safe_error_decomposition(block, claim_type="trajectory_only")  # no raise


def test_paper_safe_trajectory_only_still_requires_orbit_and_runtime():
    block = _complete_block(
        field_error=None, orbit_error=None, paper_safe_claim_type="trajectory_only"
    )
    with pytest.raises(ValueError, match="orbit_error"):
        validate_paper_safe_error_decomposition(block, claim_type="trajectory_only")


def test_paper_safe_field_claim_requires_field_error_explicitly():
    block = _complete_block(field_error=None)
    with pytest.raises(ValueError, match="field_error"):
        validate_paper_safe_error_decomposition(block, claim_type="field_accuracy")


def test_build_records_claim_type_in_provenance():
    block = _complete_block(paper_safe_claim_type="trajectory_only")
    assert block["provenance"]["paper_safe_claim_type"] == "trajectory_only"
    # Default resolves to the strict full_surrogate_validation.
    assert _complete_block()["provenance"]["paper_safe_claim_type"] == (
        "full_surrogate_validation"
    )


def test_paper_safe_fails_without_orbit_error():
    block = _complete_block(orbit_error=None)
    with pytest.raises(ValueError, match="orbit_error"):
        validate_paper_safe_error_decomposition(block)


def test_paper_safe_fails_without_runtime():
    block = _complete_block(runtime=None)
    with pytest.raises(ValueError, match="runtime"):
        validate_paper_safe_error_decomposition(block)


def test_paper_safe_fails_without_provenance_block():
    block = _complete_block()
    del block["provenance"]
    with pytest.raises(ValueError, match="provenance"):
        validate_paper_safe_error_decomposition(block)


@pytest.mark.parametrize(
    "key",
    ["requested_backend", "actual_backend", "requested_dtype",
     "effective_dtype", "frame_mode"],
)
def test_paper_safe_fails_on_missing_provenance_field(key):
    block = _complete_block(**{key: None})
    with pytest.raises(ValueError, match=key):
        validate_paper_safe_error_decomposition(block)


def test_paper_safe_fails_when_synthetic():
    block = _complete_block(synthetic=True)
    with pytest.raises(ValueError, match="synthetic"):
        validate_paper_safe_error_decomposition(block)


def test_paper_safe_fails_on_identity_rotation():
    block = _complete_block(identity_rotation_used=True)
    with pytest.raises(ValueError, match="identity"):
        validate_paper_safe_error_decomposition(block)


def test_paper_safe_error_lists_every_violation_at_once():
    block = _complete_block(
        field_error=None,
        orbit_error=None,
        runtime=None,
        identity_rotation_used=True,
        synthetic=True,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_paper_safe_error_decomposition(block)
    message = str(excinfo.value)
    for marker in ("field_error", "orbit_error", "runtime", "identity", "synthetic"):
        assert marker in message, marker
