from __future__ import annotations

import json

import numpy as np
import pytest

from lunaris.batch.storage import (
    HDF5TrajectoryView,
    _HDF5Writer,
    load_batch_result,
)

h5py = pytest.importorskip("h5py")


def _archive_arrays() -> tuple[np.ndarray, ...]:
    t = np.array([0.0, 10.0, 20.0])
    Y = np.arange(3 * 4 * 6, dtype=np.float64).reshape(3, 4, 6)
    sc = np.ones((4, 4), dtype=np.float64)
    impact = np.array([0.0, 1.0, 0.0, 1.0])
    t_impact = np.array([np.nan, 10.0, np.nan, 20.0])
    valid = np.array([1.0, 1.0, 0.0, 1.0])
    impact_i = np.full((4, 3), np.nan)
    impact_f = np.full((4, 3), np.nan)
    impact_i[[1, 3]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    impact_f[[1, 3]] = [[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]]
    return t, Y, sc, impact, t_impact, valid, impact_i, impact_f


def _v2_metadata(**overrides):
    metadata = {
        "archive_schema_version": 2,
        "n_samples": 4,
        "seed": 42,
        "duration_s": 20.0,
        "output_dt_s": 10.0,
        "backend": "CPU",
        "requested_batch_backend": "cpu_sh",
        "actual_batch_backend": "cpu_sh",
        "batch_backend": "cpu_sh",
        "detect_impact": True,
        "compute_impact_statistics": True,
    }
    metadata.update(overrides)
    return metadata


def test_hdf5_archive_v2_batch_streaming_and_lazy_load(tmp_path) -> None:
    path = tmp_path / "batch.h5"
    t, Y, sc, impact, t_impact, valid, impact_i, impact_f = _archive_arrays()
    writer = _HDF5Writer(path, n_samples=4, t_grid=t)
    writer.write_sample_batch(0, 2, Y[:, :2, :])
    writer.write_sample_batch(2, 4, Y[:, 2:, :])
    writer.write_metadata(**_v2_metadata(backend_diagnostics={"nested": [1, 2, 3]}, ordinary_string="cpu_sh"))
    writer.write_final(sc, impact, t_impact, valid, impact_i, impact_f)
    writer.finalize()

    assert path.exists()
    assert not (tmp_path / "batch.h5.part").exists()

    eager = load_batch_result(str(path))
    lazy = load_batch_result(str(path), lazy=True)
    np.testing.assert_allclose(eager.Y, Y)
    assert isinstance(lazy.Y, HDF5TrajectoryView)
    assert lazy.is_lazy
    np.testing.assert_allclose(lazy.Y[:, 1:3, :], Y[:, 1:3, :])
    np.testing.assert_allclose(lazy.valid_mask, valid)
    np.testing.assert_allclose(
        lazy.impact_position_fixed_m,
        impact_f,
        equal_nan=True,
    )
    assert eager.diagnostics["backend_diagnostics"] == {"nested": [1, 2, 3]}
    assert eager.diagnostics["ordinary_string"] == "cpu_sh"
    assert eager.diagnostics["archive_schema_version"] == 2


def test_hdf5_writer_rejects_incomplete_or_out_of_order_stream(tmp_path) -> None:
    path = tmp_path / "incomplete.h5"
    t, Y, *_ = _archive_arrays()
    writer = _HDF5Writer(path, n_samples=4, t_grid=t)
    writer.write_sample_batch(0, 2, Y[:, :2, :])

    with pytest.raises(ValueError, match="expected start=2"):
        writer.write_sample_batch(3, 4, Y[:, 3:4, :])
    with pytest.raises(RuntimeError, match="incomplete HDF5 trajectory stream"):
        writer.finalize()

    assert not path.exists()
    assert (tmp_path / "incomplete.h5.part").exists()
    writer.abort()
    assert not (tmp_path / "incomplete.h5.part").exists()


def test_hdf5_writer_does_not_swallow_metadata_errors(tmp_path) -> None:
    class BrokenMetadata:
        def __str__(self) -> str:
            raise RuntimeError("cannot serialize")

    path = tmp_path / "bad_metadata.h5"
    writer = _HDF5Writer(path, n_samples=4, t_grid=np.array([0.0]))
    with pytest.raises(ValueError, match="'broken'"):
        writer.write_metadata(broken=BrokenMetadata())
    writer.abort()

    assert not path.exists()
    assert not (tmp_path / "bad_metadata.h5.part").exists()


def test_schema_v2_npz_requires_current_result_arrays(tmp_path) -> None:
    path = tmp_path / "v2_missing_result_arrays.npz"
    t, Y, sc, impact, t_impact, *_ = _archive_arrays()
    Y[:, 2, :] = np.nan
    np.savez_compressed(
        path,
        t=t,
        Y=Y,
        sc_samples=sc,
        impact_flags=impact,
        t_impact=t_impact,
        metadata_json=np.asarray(json.dumps(_v2_metadata()), dtype=np.str_),
    )

    with pytest.raises(ValueError, match="missing required dataset"):
        load_batch_result(str(path))


def test_archive_v2_strict_rejects_missing_required_field(tmp_path) -> None:
    path = tmp_path / "partial.h5"
    t, Y, sc, impact, t_impact, valid, impact_i, impact_f = _archive_arrays()
    writer = _HDF5Writer(path, n_samples=4, t_grid=t)
    writer.write_sample_batch(0, 4, Y)
    # Intentionally incomplete manifest: finalize() still stamps schema v2, but
    # the required provenance fields are absent.
    writer.write_metadata(backend="CPU", n_samples=4)
    writer.write_final(sc, impact, t_impact, valid, impact_i, impact_f)
    writer.finalize()

    with pytest.raises(ValueError, match="missing required"):
        load_batch_result(str(path))

    # The strict flag is ignored; manifest enforcement cannot be bypassed.
    with pytest.raises(ValueError, match="missing required"):
        load_batch_result(str(path), strict=False)


def test_lazy_oe_dispersion_matches_eager_block_read(tmp_path) -> None:
    """C2: disk-backed OE dispersion matches the eager trajectory result."""
    from lunaris.analysis.ensemble.statistics import compute_oe_dispersion

    path = tmp_path / "oe.h5"
    t, _Y, sc, impact, t_impact, valid, impact_i, impact_f = _archive_arrays()
    # Physically-plausible orbital states so the Keplerian conversion is finite.
    rng = np.random.default_rng(3)
    r0 = 1.838e6
    base = np.array([r0, 0.0, 0.0, 0.0, (4.9e12 / r0) ** 0.5, 0.0])
    Y = np.broadcast_to(base, (len(t), 4, 6)).astype(np.float64).copy()
    Y[:, :, :3] += rng.normal(0.0, 2_000.0, size=(len(t), 4, 3))

    writer = _HDF5Writer(path, n_samples=4, t_grid=t)
    writer.write_sample_batch(0, 4, Y)
    writer.write_metadata(**_v2_metadata())
    writer.write_final(sc, impact, t_impact, valid, impact_i, impact_f)
    writer.finalize()

    eager = load_batch_result(str(path))
    lazy = load_batch_result(str(path), lazy=True)
    assert lazy.is_lazy

    oe_eager = compute_oe_dispersion(eager)
    oe_lazy = compute_oe_dispersion(lazy)
    np.testing.assert_allclose(oe_lazy.a_mean_km, oe_eager.a_mean_km, equal_nan=True)
    np.testing.assert_allclose(oe_lazy.e_mean, oe_eager.e_mean, equal_nan=True)
    np.testing.assert_allclose(oe_lazy.inc_mean_deg, oe_eager.inc_mean_deg, equal_nan=True)


def test_archive_without_schema_version_is_rejected(tmp_path) -> None:
    path = tmp_path / "missing_schema.npz"
    t, Y, sc, impact, t_impact, *_ = _archive_arrays()
    np.savez_compressed(
        path,
        t=t,
        Y=Y,
        sc_samples=sc,
        impact_flags=impact,
        t_impact=t_impact,
        metadata_json=np.asarray(json.dumps({"backend": "cpu"}), dtype=np.str_),
    )
    with pytest.raises(ValueError, match="missing archive_schema_version"):
        load_batch_result(str(path))
