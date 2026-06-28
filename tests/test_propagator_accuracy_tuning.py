"""Accuracy-tuning surface of the propagator:

1. Per-component (vector) ``atol`` resolution for solve_ivp.
2. Energy / angular-momentum drift diagnostics (on the run and the 2-body baseline).
3. SH truncation-degree adequacy warning for the orbit periapsis altitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.type_defs import PropagatorConfig
from lunaris.core.propagation.propagator import _resolve_atol

# ---------------------------------------------------------------------------
# 1) Vector atol
# ---------------------------------------------------------------------------

def test_resolve_atol_scalar_by_default():
    cfg = PropagatorConfig(atol=1e-12)
    out = _resolve_atol(cfg, 6)
    assert isinstance(out, float)
    assert out == 1e-12


def test_resolve_atol_builds_position_velocity_vector():
    cfg = PropagatorConfig(atol=1e-12, atol_pos=1e-6, atol_vel=1e-9)
    out = _resolve_atol(cfg, 6)
    assert isinstance(out, np.ndarray)
    assert out.shape == (6,)
    np.testing.assert_array_equal(out[0:3], 1e-6)
    np.testing.assert_array_equal(out[3:6], 1e-9)


def test_resolve_atol_augmented_state_keeps_scalar_for_extra_components():
    cfg = PropagatorConfig(atol=1e-12, atol_pos=1e-6, atol_vel=1e-9)
    out = _resolve_atol(cfg, 9)  # 6D state + 3 augmented (e.g. STM/surrogate) channels
    assert out.shape == (9,)
    np.testing.assert_array_equal(out[0:3], 1e-6)
    np.testing.assert_array_equal(out[3:6], 1e-9)
    np.testing.assert_array_equal(out[6:9], 1e-12)  # extra components keep scalar atol


def test_resolve_atol_only_one_of_pos_vel():
    cfg = PropagatorConfig(atol=1e-12, atol_pos=1e-6)  # vel left as scalar
    out = _resolve_atol(cfg, 6)
    np.testing.assert_array_equal(out[0:3], 1e-6)
    np.testing.assert_array_equal(out[3:6], 1e-12)


def test_propagator_config_rejects_nonpositive_component_atol():
    with pytest.raises(ValueError):
        PropagatorConfig(atol_pos=0.0)
    with pytest.raises(ValueError):
        PropagatorConfig(atol_vel=-1e-9)
