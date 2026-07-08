from __future__ import annotations

import math

import numpy as np

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.contracts import (
    BATCH_ARCHIVE_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    PROPAGATION_DIAGNOSTICS_SCHEMA_VERSION,
    REQUIRED_ARCHIVE_V2_ARRAYS,
    REQUIRED_ARCHIVE_V2_FIELDS,
)
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.propagation.checkpoint import _atomic_save_npz
from lunaris.core.propagation.propagator import propagate


class FakePointMassDynamics:
    grav = None
    ephem = None

    def build_rhs(self):
        def rhs(_t, y):
            y_arr = np.asarray(y, dtype=np.float64)
            r = y_arr[:3]
            v = y_arr[3:6]
            rn = float(np.linalg.norm(r))
            dy = np.empty_like(y_arr)
            dy[:3] = v
            dy[3:6] = -float(MU_MOON) * r / (rn**3)
            if y_arr.size > 6:
                dy[6:] = 0.0
            return dy

        return rhs


def _state() -> np.ndarray:
    r0 = float(R_MOON) + 120_000.0
    v0 = math.sqrt(float(MU_MOON) / r0)
    return np.asarray([r0, 0.0, 0.0, 0.0, v0, 0.0], dtype=np.float64)


def _cfg() -> PropagatorConfig:
    return PropagatorConfig(
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        verbose=False,
        compute_2body_baseline=False,
        use_nyquist_max_step=False,
        events=EventConfig(detect_impact=False, enable_peri_apo_events=False),
    )


def test_contract_registry_reexports_and_legacy_locations_share_values() -> None:
    from lunaris.analysis.ensemble import result_audit
    from lunaris.batch import storage
    from lunaris.core.propagation import checkpoint

    assert storage.BATCH_ARCHIVE_SCHEMA_VERSION == BATCH_ARCHIVE_SCHEMA_VERSION
    assert storage.REQUIRED_ARCHIVE_V2_FIELDS is REQUIRED_ARCHIVE_V2_FIELDS
    assert storage.REQUIRED_ARCHIVE_V2_ARRAYS is REQUIRED_ARCHIVE_V2_ARRAYS
    assert result_audit._REQUIRED_V2_FIELDS is REQUIRED_ARCHIVE_V2_FIELDS
    assert checkpoint.CHECKPOINT_SCHEMA_VERSION == CHECKPOINT_SCHEMA_VERSION


def test_checkpoint_writer_stamps_registry_schema_version(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint.npz"
    t = np.asarray([0.0, 10.0], dtype=np.float64)
    y_row = np.vstack([_state(), _state()])

    _atomic_save_npz(ckpt, t=t, y_row=y_row, method="DOP853", config_hash="abc123")

    with np.load(ckpt, allow_pickle=False) as data:
        assert int(data["checkpoint_schema_version"]) == CHECKPOINT_SCHEMA_VERSION


def test_propagation_diagnostics_schema_uses_registry_version() -> None:
    res = propagate(
        FakePointMassDynamics(),
        _state(),
        _cfg(),
        time_cfg=TimeConfig(duration_s=40.0, output_dt_s=20.0, samples_per_period=2),
    )

    assert (
        res.diagnostics["diagnostics_schema_version"]
        == PROPAGATION_DIAGNOSTICS_SCHEMA_VERSION
    )
