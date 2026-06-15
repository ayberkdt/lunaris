from __future__ import annotations

import json

import numpy as np
import pytest

from lunaris.core.monte_carlo_engine import (
    HDF5TrajectoryView,
    _HDF5Writer,
    load_mc_result,
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


def test_hdf5_archive_v2_batch_streaming_and_lazy_load(tmp_path) -> None:
    path = tmp_path / "mc.h5"
    t, Y, sc, impact, t_impact, valid, impact_i, impact_f = _archive_arrays()
    writer = _HDF5Writer(path, n_samples=4, t_grid=t)
    writer.write_sample_batch(0, 2, Y[:, :2, :])
    writer.write_sample_batch(2, 4, Y[:, 2:, :])
    writer.write_metadata(
        backend_diagnostics={"nested": [1, 2, 3]},
        ordinary_string="cpu_sh",
    )
    writer.write_final(sc, impact, t_impact, valid, impact_i, impact_f)
    writer.finalize()

    assert path.exists()
    assert not (tmp_path / "mc.h5.part").exists()

    eager = load_mc_result(str(path))
    lazy = load_mc_result(str(path), lazy=True)
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


def test_legacy_npz_infers_valid_mask_and_missing_impact_positions(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    t, Y, sc, impact, t_impact, *_ = _archive_arrays()
    Y[:, 2, :] = np.nan
    np.savez_compressed(
        path,
        t=t,
        Y=Y,
        sc_samples=sc,
        impact_flags=impact,
        t_impact=t_impact,
        metadata_json=np.asarray(json.dumps({"legacy": True}), dtype=np.str_),
    )

    result = load_mc_result(str(path))
    np.testing.assert_array_equal(result.valid_mask, [1.0, 1.0, 0.0, 1.0])
    assert result.impact_position_inertial_m is None
    assert result.impact_position_fixed_m is None
