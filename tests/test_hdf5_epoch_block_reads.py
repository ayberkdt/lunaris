"""``HDF5TrajectoryView.iter_epoch_sample_blocks`` reads chunk tiles once.

The values must be bit-identical to a per-epoch point selection; only the
read pattern changes (each chunk decompressed once instead of once per epoch).
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from lunaris.batch.storage import HDF5TrajectoryView  # noqa: E402


@pytest.mark.parametrize("chunks", [(16, 128, 6), (7, 50, 6), None])
@pytest.mark.parametrize("n_sel", [0, 1, 37, 300])
def test_epoch_blocks_match_point_selection(tmp_path, chunks, n_sel):
    rng = np.random.default_rng(7)
    T, N = 41, 300
    Y = rng.standard_normal((T, N, 6))
    path = tmp_path / "y.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("Y", data=Y, chunks=chunks, compression="lzf" if chunks else None)
    indices = np.sort(rng.choice(N, size=n_sel, replace=False)).astype(np.int64)

    blocks = list(HDF5TrajectoryView(path).iter_epoch_sample_blocks(indices))

    assert len(blocks) == T
    for epoch, block in enumerate(blocks):
        assert block.shape == (n_sel, 6)
        assert block.dtype == np.float64
        np.testing.assert_array_equal(block, Y[epoch, indices, :])


def test_unsorted_indices_are_still_rejected(tmp_path):
    path = tmp_path / "y.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("Y", data=np.zeros((2, 4, 6)))
    with pytest.raises(ValueError, match="strictly increasing"):
        list(HDF5TrajectoryView(path).iter_epoch_sample_blocks(np.array([2, 1])))
