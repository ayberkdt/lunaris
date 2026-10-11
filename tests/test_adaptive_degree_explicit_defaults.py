"""Adaptive-degree policy: explicit values are honoured, invalid ones raise.

``min_degree=0`` is a valid floor (``AdaptiveDegreeConfig`` allows >= 0: degree
0 is the point-mass term) but ``... or 4`` silently replaced it with 4.
``power`` must be > 0 (``AdaptiveDegreeConfig`` rejects 0); a duck-typed policy
carrying 0 was silently replaced with 2.5 instead of being refused.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lunaris.common.type_defs import AdaptiveDegreeConfig
from lunaris.core.dynamics.preparation import prepare_adaptive_gravity_policy


def _policy(adaptive):
    return prepare_adaptive_gravity_policy(50, gravity_model=None, gravity_adaptive=adaptive)


def test_min_degree_zero_is_honoured():
    policy = _policy(AdaptiveDegreeConfig(enabled=True, min_degree=0))
    assert policy["adaptive_min_degree"] == 0
    assert policy["adaptive_mode"] == 2


def test_defaults_apply_only_when_attributes_are_missing_or_none():
    policy = _policy(SimpleNamespace(enabled=True, min_degree=None, power=None))
    assert policy["adaptive_min_degree"] == 4
    assert policy["adaptive_power"] == 2.5
    policy = _policy(SimpleNamespace(enabled=True))
    assert policy["adaptive_min_degree"] == 4
    assert policy["adaptive_power"] == 2.5


def test_explicit_power_is_honoured():
    assert _policy(AdaptiveDegreeConfig(enabled=True, power=1.5))["adaptive_power"] == 1.5


@pytest.mark.parametrize("power", [0.0, -1.0, float("nan")])
def test_invalid_power_raises(power):
    with pytest.raises(ValueError, match="power"):
        _policy(SimpleNamespace(enabled=True, power=power))

