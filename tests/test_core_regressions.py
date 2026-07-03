"""
Focused regression tests for core-layer fixes.

These tests target small but user-visible behaviors that are easy to regress
when the core propagation/batch code is refactored:

- short-run ephemeris rotation tables with a single quaternion sample
- batch archive metadata round-tripping
- CPU batch preservation of refined impact times
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lunaris.batch.storage import _NPZWriter, load_batch_result
from lunaris.common.batch_defs import BatchPropagationConfig
from lunaris.common.type_defs import EventConfig, PropagationResult, PropagatorConfig, TimeConfig
from lunaris.core.batch_propagator import CPUBatchPropagator
from lunaris.core.propagation.propagator import _build_r_i_to_bf_from_rot_table


def test_single_sample_quaternion_table_still_builds_fixed_frame_mapper() -> None:
    dynamics = SimpleNamespace(
        ephem=SimpleNamespace(
            tables=SimpleNamespace(
                dt_s=60.0,
                q_i2f_tab=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            )
        )
    )

    mapper = _build_r_i_to_bf_from_rot_table(dynamics)

    assert mapper is not None
    out = mapper(123.0, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(out, np.asarray([1.0, 2.0, 3.0], dtype=np.float64), atol=1e-12)


def test_load_batch_result_restores_npz_metadata_into_diagnostics(tmp_path: Path) -> None:
    output_path = tmp_path / "batch_metadata.npz"
    t = np.asarray([0.0, 30.0], dtype=np.float64)
    writer = _NPZWriter(output_path, n_samples=1, t_grid=t)
    writer.write_metadata(
        archive_schema_version=2,
        n_samples=1,
        seed=7,
        duration_s=30.0,
        output_dt_s=30.0,
        backend="cpu",
        requested_batch_backend="cpu_sh",
        actual_batch_backend="cpu_sh",
        batch_backend="cpu_sh",
        detect_impact=True,
        compute_impact_statistics=True,
    )
    writer.write_sample_batch(
        0,
        1,
        np.asarray(
            [np.zeros((1, 6), dtype=np.float64), np.ones((1, 6), dtype=np.float64)]
        ),
    )
    writer.write_final(
        sc_samples=np.asarray([[1000.0, 5.0, 2.2, 1.5]], dtype=np.float64),
        impact_flags=np.asarray([0.0], dtype=np.float64),
        t_impact=np.asarray([np.nan], dtype=np.float64),
        valid_mask=np.asarray([1.0], dtype=np.float64),
        impact_position_inertial_m=np.full((1, 3), np.nan),
        impact_position_fixed_m=np.full((1, 3), np.nan),
    )
    writer.finalize()

    loaded = load_batch_result(str(output_path))

    assert loaded.diagnostics["seed"] == 7
    assert loaded.diagnostics["output_dt_s"] == 30.0
    assert loaded.diagnostics["backend"] == "cpu"


def test_cpu_batch_propagator_preserves_precise_impact_times(monkeypatch) -> None:
    sim_cfg = SimpleNamespace(
        time=TimeConfig(start_date="2027-03-02T23:32:37", duration_s=10.0, output_dt_s=5.0),
        propagator=PropagatorConfig(
            method="DOP853",
            verbose=False,
            use_nyquist_max_step=False,
            compute_2body_baseline=False,
            heartbeat_hours=1.0,
            events=EventConfig(detect_impact=True, impact_alt_km=0.0),
        ),
        flags=SimpleNamespace(),
    )
    batch_cfg = BatchPropagationConfig(
        n_samples=2,
        use_gpu=False,
        output_format="npz",
        output_path="outputs/ensemble/test_cpu_batch.npz",
    )

    batch = CPUBatchPropagator(sim_cfg, batch_cfg)
    monkeypatch.setattr(batch, "_make_sample_dynamics", lambda **kwargs: object())

    def fake_propagate(*, dynamics, y0, cfg, time_cfg, topo_grid=None):
        _ = (dynamics, cfg, time_cfg, topo_grid)
        y_row = np.vstack([y0, y0, y0]).astype(np.float64, copy=False)
        return PropagationResult(
            t=np.asarray([0.0, 5.0, 10.0], dtype=np.float64),
            y=y_row,
            impacted=True,
            t_impact_s=7.5,
        )

    monkeypatch.setattr("lunaris.core.propagation.propagator.propagate", fake_propagate)

    t_out, Y_out, impact_flags, t_impact = batch.propagate(
        Y0=np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64),
        masses=np.asarray([1000.0], dtype=np.float64),
        areas=np.asarray([5.0], dtype=np.float64),
        cds=np.asarray([2.2], dtype=np.float64),
        crs=np.asarray([1.5], dtype=np.float64),
        duration_s=10.0,
        output_dt_s=5.0,
    )

    np.testing.assert_allclose(t_out, np.asarray([0.0, 5.0, 10.0], dtype=np.float64))
    assert Y_out.shape == (3, 1, 6)
    assert impact_flags[0] == 1.0
    assert t_impact[0] == 7.5
