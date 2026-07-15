"""Paper-safe / benchmark / strict runs must forbid adaptive SH degree blending.

A reference result must use a single fixed degree so its error is attributable to
one model, not to an altitude-dependent blend of two degrees. The guard lives in
``prepare_adaptive_gravity_policy`` and is threaded from the batch engine via the
same posture that forbids a silent GPU->CPU fallback.
"""

from __future__ import annotations

import pytest

from lunaris.common.type_defs import AdaptiveDegreeConfig
from lunaris.core.dynamics.preparation import prepare_adaptive_gravity_policy


def test_strict_posture_rejects_enabled_adaptive_blend():
    adaptive = AdaptiveDegreeConfig(enabled=True)
    with pytest.raises(ValueError, match="forbidden in paper-safe"):
        prepare_adaptive_gravity_policy(
            64, gravity_model=None, gravity_adaptive=adaptive, strict_fixed_degree=True
        )


def test_strict_posture_allows_fixed_degree():
    # Adaptive disabled -> a strict reference run proceeds with a fixed degree.
    adaptive = AdaptiveDegreeConfig(enabled=False)
    policy = prepare_adaptive_gravity_policy(
        64, gravity_model=None, gravity_adaptive=adaptive, strict_fixed_degree=True
    )
    assert policy["adaptive_enabled"] is False


def test_non_strict_posture_permits_adaptive_blend():
    # Exploratory (non-strict) runs may still use the altitude-aware blend.
    adaptive = AdaptiveDegreeConfig(enabled=True)
    policy = prepare_adaptive_gravity_policy(
        64, gravity_model=None, gravity_adaptive=adaptive, strict_fixed_degree=False
    )
    assert policy["adaptive_enabled"] is True


def test_strict_posture_ignores_adaptive_when_no_degrees():
    # nmax == 0 (point-mass / surrogate) can never blend, so strict is a no-op.
    adaptive = AdaptiveDegreeConfig(enabled=True)
    policy = prepare_adaptive_gravity_policy(
        0, gravity_model=None, gravity_adaptive=adaptive, strict_fixed_degree=True
    )
    assert policy["adaptive_enabled"] is False
