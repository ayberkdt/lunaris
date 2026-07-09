"""Guards for canonical shared constants and canonical-JSON hashing.

These constants are scientifically load-bearing (frame identity, derivative
convention, content digests). They must have exactly one definition; the
aliases kept for backward compatibility must stay identical to the canonical
values in ``lunaris.surrogate.st_lrps.shared.contracts`` and
``lunaris.common.hashing``.
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


import pytest

from lunaris.common.hashing import canonical_json_sha256


def test_frame_and_convention_constants_are_single_sourced():
    from lunaris.surrogate.st_lrps.data import dataset_contract as dc
    from lunaris.surrogate.st_lrps.shared import contracts as sc

    assert sc.MOON_FIXED_FRAME == "moon_fixed_cartesian"
    assert dc.DEFAULT_COORDINATE_FRAME == sc.MOON_FIXED_FRAME

    assert sc.REQUIRED_DERIVATIVE_CONVENTION == "dP_dphi_corrected_v1"
    assert dc.REQUIRED_DERIVATIVE_CONVENTION is sc.REQUIRED_DERIVATIVE_CONVENTION

    assert dc.TARGET_MODES == sc.TARGET_MODES
    assert dc.BASELINE_KINDS == sc.BASELINE_KINDS


def test_runtime_frame_constant_matches_contracts():
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.runtime.force_model import SUPPORTED_RUNTIME_FRAME
    from lunaris.surrogate.st_lrps.shared.contracts import MOON_FIXED_FRAME

    assert SUPPORTED_RUNTIME_FRAME == MOON_FIXED_FRAME


def test_benchmark_payload_hash_delegates_to_common_hashing():
    from lunaris.surrogate.st_lrps.evaluation.provenance import sha256_payload

    payload = {"b": [1, 2], "a": 1}
    assert sha256_payload(payload) == canonical_json_sha256(payload)
    # Key order must not matter.
    assert sha256_payload({"a": 1, "b": [1, 2]}) == sha256_payload(payload)


def test_canonical_json_sha256_byte_stability():
    # Frozen digest of the canonical form (sorted keys, indent=2, ASCII,
    # trailing newline). If this changes, every recorded resolved-config /
    # artifact content hash silently changes — that must never happen
    # accidentally.
    payload = {"b": [1, 2], "a": 1}
    assert (
        canonical_json_sha256(payload)
        == "0bdfe8b5d2224246c36a713bae07d9eaddf349205c98e51f117a353646dee2c6"
    )
