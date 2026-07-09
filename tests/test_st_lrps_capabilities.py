"""Phase 0: the ST-LRPS capability registry is the single source of truth and
the checked-in matrix document never drifts from it."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from lunaris.surrogate.st_lrps.shared.capabilities import (
    CAPABILITIES,
    CapabilityStatus,
    UnsupportedCapability,
    render_capability_matrix_markdown,
    require,
    require_baseline_supported,
    require_runtime_potential,
)

_MATRIX_DOC = Path(__file__).resolve().parents[1] / "docs" / "ST_LRPS_CAPABILITY_MATRIX.md"


def test_unsupported_capability_is_notimplementederror_subclass() -> None:
    # Backward compatibility: existing `except NotImplementedError` / pytest
    # raises sites must keep catching the registry-backed exception.
    assert issubclass(UnsupportedCapability, NotImplementedError)


def test_checked_in_matrix_matches_registry() -> None:
    rendered = render_capability_matrix_markdown()
    on_disk = _MATRIX_DOC.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert on_disk == rendered, (
        "docs/ST_LRPS_CAPABILITY_MATRIX.md is stale. Regenerate it from "
        "render_capability_matrix_markdown() and commit."
    )


def test_force_direct_runtime_kind_is_not_registered() -> None:
    # force_direct is archived in experimental/force-direct-archive; it must not
    # appear anywhere in the capability registry.
    assert not any(c.subject == "force_direct" for c in CAPABILITIES)


def test_potential_autograd_potential_is_supported() -> None:
    cap = require_runtime_potential("potential_autograd")
    assert cap.status is CapabilityStatus.SUPPORTED


def test_baseline_kinds_are_all_supported() -> None:
    assert require_baseline_supported("residual", "spherical_harmonics").status is CapabilityStatus.SUPPORTED
    assert require_baseline_supported("full", "none").status is CapabilityStatus.SUPPORTED
    assert require_baseline_supported("full", "point_mass").status is CapabilityStatus.SUPPORTED
    assert require_baseline_supported("full", "spherical_harmonics").status is CapabilityStatus.SUPPORTED


def test_unknown_capability_key_is_keyerror_not_unsupported() -> None:
    # A typo'd feature is a programming error and must surface loudly, not as the
    # softer UnsupportedCapability.
    with pytest.raises(KeyError):
        require("runtime", "potential_autograd", "no_such_feature")


def test_registry_keys_are_unique() -> None:
    keys = [c.key for c in CAPABILITIES]
    assert len(keys) == len(set(keys))
