# tests/test_st_lrps_kind_preflight.py
"""Audit F13 — the GPU ST-LRPS artifact-kind preflight must fail closed.

``_st_lrps_kind_mismatch`` guards the GPU batch path: the backend policy's
resolved ``runtime_model_kind`` (the *expectation*) is compared against what
the loaded artifact actually declares. An explicit direct-force request must
be provable from the artifact — legacy kind-less artifacts are potential-only
by construction, so assuming ``force_direct`` would run the wrong physics.
"""

from __future__ import annotations

from lunaris.batch.engine import _st_lrps_kind_mismatch


def test_matching_kinds_pass():
    assert _st_lrps_kind_mismatch("force_direct", "force_direct") is None
    assert _st_lrps_kind_mismatch("potential_autograd", "potential_autograd") is None


def test_no_expectation_passes():
    assert _st_lrps_kind_mismatch("", "force_direct") is None
    assert _st_lrps_kind_mismatch(None, "") is None


def test_declared_mismatch_fails():
    msg = _st_lrps_kind_mismatch("potential_autograd", "force_direct")
    assert msg is not None and "mismatch" in msg
    msg = _st_lrps_kind_mismatch("force_direct", "potential_autograd")
    assert msg is not None and "mismatch" in msg


def test_direct_expectation_with_kindless_artifact_fails_closed():
    msg = _st_lrps_kind_mismatch("force_direct", "")
    assert msg is not None and "refusing to assume" in msg.lower()
    assert _st_lrps_kind_mismatch("force_direct", None) is not None


def test_potential_expectation_with_kindless_artifact_passes():
    # Legacy kind-less artifacts are potential-only by construction, so a
    # potential_autograd expectation is satisfiable without a declaration.
    assert _st_lrps_kind_mismatch("potential_autograd", "") is None


def test_whitespace_is_normalized():
    assert _st_lrps_kind_mismatch(" force_direct ", "force_direct") is None
    assert _st_lrps_kind_mismatch("force_direct", "   ") is not None
