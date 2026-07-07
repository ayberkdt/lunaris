"""Contract tests for the shared Cartesian state-vector normalization.

These pin the P0 contract: the dynamics RHS supports exactly 6-state
[x,y,z,vx,vy,vz] or 7-state (same + mass); every entry path (single-run
propagate, batch nominal state, CLI y0) must enforce that instead of
accepting oversized states or silently truncating them.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.state_vector import (
    STATE_SIZE_POS_VEL,
    STATE_SIZE_WITH_MASS,
    normalize_cartesian_state,
    normalize_position_velocity_state,
)

_STATE6 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
_STATE7 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]


class _ToArrayState:
    def __init__(self, values):
        self._values = list(values)

    def to_array(self):
        return np.asarray(self._values, dtype=np.float64)


class _PackedYState:
    """OrbitState-like: packed vector exposed via ``.y``."""

    def __init__(self, values):
        self.y = np.asarray(values, dtype=np.float64)


class _ComponentRecord:
    """Plain record with x,y,z,vx,vy,vz attributes (scalar ``.y``)."""

    def __init__(self, values):
        self.x, self.y, self.z, self.vx, self.vy, self.vz = values


# =============================================================================
# normalize_cartesian_state — size contract
# =============================================================================

def test_accepts_exactly_six():
    out = normalize_cartesian_state(_STATE6)
    assert out.shape == (STATE_SIZE_POS_VEL,)
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, _STATE6)


def test_accepts_seven_with_mass_when_allowed():
    out = normalize_cartesian_state(_STATE7, allow_mass=True)
    assert out.shape == (STATE_SIZE_WITH_MASS,)
    assert out[6] == 100.0


def test_rejects_seven_when_mass_not_allowed():
    with pytest.raises(ValueError, match="exactly 6 elements"):
        normalize_cartesian_state(_STATE7, allow_mass=False)


@pytest.mark.parametrize("size", [5, 8, 12])
def test_rejects_wrong_sizes(size):
    state = list(range(size))
    with pytest.raises(ValueError, match=rf"got {size}"):
        normalize_cartesian_state(state, allow_mass=True)
    with pytest.raises(ValueError, match=rf"got {size}"):
        normalize_cartesian_state(state, allow_mass=False)


def test_rejects_none():
    with pytest.raises(ValueError, match="is None"):
        normalize_cartesian_state(None)


def test_error_message_names_expected_and_received_size():
    with pytest.raises(ValueError, match=r"exactly 6 .* or 7 .*got 8"):
        normalize_cartesian_state(list(range(8)), name="Initial state")


# =============================================================================
# normalize_cartesian_state — container styles
# =============================================================================

def test_accepts_object_with_to_array():
    out = normalize_cartesian_state(_ToArrayState(_STATE6))
    np.testing.assert_array_equal(out, _STATE6)


def test_accepts_object_with_packed_y():
    out = normalize_cartesian_state(_PackedYState(_STATE7))
    np.testing.assert_array_equal(out, _STATE7)


def test_accepts_component_record_despite_scalar_y_attribute():
    # A plain record has `.y` too (the position component); the packed-vector
    # branch must fall through to the component attributes instead of failing.
    out = normalize_cartesian_state(_ComponentRecord(_STATE6))
    np.testing.assert_array_equal(out, _STATE6)


def test_returns_independent_copy():
    src = np.asarray(_STATE6, dtype=np.float64)
    out = normalize_cartesian_state(src)
    out[0] = -999.0
    assert src[0] == 1.0


# =============================================================================
# normalize_position_velocity_state — documented mass dropping only
# =============================================================================

def test_pos_vel_passes_six_through():
    out = normalize_position_velocity_state(_STATE6, drop_mass=True)
    np.testing.assert_array_equal(out, _STATE6)


def test_pos_vel_drops_mass_only_when_documented():
    out = normalize_position_velocity_state(_STATE7, drop_mass=True)
    assert out.shape == (STATE_SIZE_POS_VEL,)
    np.testing.assert_array_equal(out, _STATE6)


def test_pos_vel_rejects_mass_without_drop_flag():
    with pytest.raises(ValueError, match="exactly 6 elements"):
        normalize_position_velocity_state(_STATE7, drop_mass=False)


@pytest.mark.parametrize("size", [5, 8, 12])
def test_pos_vel_never_truncates_wrong_sizes(size):
    with pytest.raises(ValueError, match=rf"got {size}"):
        normalize_position_velocity_state(list(range(size)), drop_mass=True)


# =============================================================================
# Entry-path integration: single-run, batch, CLI
# =============================================================================

def test_propagation_entry_rejects_oversized_state():
    from lunaris.core.propagation.result import _as_state_array

    np.testing.assert_array_equal(_as_state_array(_STATE7), _STATE7)
    with pytest.raises(ValueError, match="Initial state.*got 8"):
        _as_state_array(list(range(8)))


def test_batch_nominal_state_no_silent_truncation():
    from lunaris.batch.requirements import _state_to_array

    # 7-state: mass dropped via the documented 6D contract.
    np.testing.assert_array_equal(_state_to_array(_STATE7), _STATE6)
    # 8+ must fail loudly instead of being truncated to 6.
    with pytest.raises(ValueError, match="Nominal state.*got 8"):
        _state_to_array(list(range(8)))
    with pytest.raises(ValueError, match="got 5"):
        _state_to_array(list(range(5)))


def test_cli_y0_rejects_oversized_state():
    from lunaris.cli.run import _y0_to_array

    np.testing.assert_array_equal(_y0_to_array(_ComponentRecord(_STATE6)), _STATE6)
    np.testing.assert_array_equal(_y0_to_array(_STATE7), _STATE7)
    with pytest.raises(ValueError, match="got 9"):
        _y0_to_array(list(range(9)))
