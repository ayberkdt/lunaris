# tests/test_st_lrps_kind_preflight.py
"""Audit F13 — the GPU ST-LRPS artifact-kind preflight.

``_st_lrps_kind_mismatch`` guards the GPU batch path: the backend policy's
resolved ``runtime_model_kind`` (the *expectation*) is compared against what the
loaded artifact actually declares. Only ``potential_autograd`` is supported on
main (the direct-force variant is archived), so this is a generic
declared-vs-declared consistency check: a kind-less legacy artifact is
potential-only by construction and passes, while two conflicting declarations
fail.
"""

from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



from lunaris.batch.engine import _st_lrps_kind_mismatch


def test_matching_kinds_pass():
    assert _st_lrps_kind_mismatch("potential_autograd", "potential_autograd") is None


def test_no_expectation_passes():
    assert _st_lrps_kind_mismatch("", "potential_autograd") is None
    assert _st_lrps_kind_mismatch(None, "") is None


def test_kindless_artifact_passes():
    # Legacy kind-less artifacts are potential-only by construction, so any
    # expectation is satisfiable without a declaration on the artifact.
    assert _st_lrps_kind_mismatch("potential_autograd", "") is None
    assert _st_lrps_kind_mismatch("potential_autograd", None) is None


def test_declared_mismatch_fails():
    msg = _st_lrps_kind_mismatch("potential_autograd", "some_other_kind")
    assert msg is not None and "mismatch" in msg


def test_whitespace_is_normalized():
    assert _st_lrps_kind_mismatch(" potential_autograd ", "potential_autograd") is None
    assert _st_lrps_kind_mismatch("potential_autograd", "   ") is None
