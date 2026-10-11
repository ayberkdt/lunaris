"""Archive writer paths: format suffix, per-process part files, overwrite policy.

- ``output_path_resolved`` carries the format's suffix: numpy appends ``.npz``
  to any other name, so an ``npz`` run at ``*.h5`` wrote ``*.part.h5.npz`` and
  the ``.part`` rename failed.
- Part files carry the PID, so two runs on one path never delete each other's
  in-progress archive.
- An existing archive is replaced (with a logged warning) by default, refused
  with ``overwrite_output=False``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from lunaris.batch import storage
from lunaris.batch.storage import _make_writer, _NPZWriter, load_batch_result
from lunaris.common.batch_defs import BatchPropagationConfig


def _archive_arrays() -> tuple[np.ndarray, ...]:
    t = np.array([0.0, 10.0, 20.0])
    Y = np.arange(3 * 4 * 6, dtype=np.float64).reshape(3, 4, 6)
    nan3 = np.full((4, 3), np.nan)
    return t, Y, np.ones((4, 4)), np.zeros(4), np.full(4, np.nan), np.ones(4), nan3, nan3.copy()


def _v2_metadata() -> dict:
    return {
        "archive_schema_version": 2, "n_samples": 4, "seed": 42, "duration_s": 20.0,
        "output_dt_s": 10.0, "backend": "CPU", "requested_batch_backend": "cpu_sh",
        "actual_batch_backend": "cpu_sh", "batch_backend": "cpu_sh",
        "detect_impact": True, "compute_impact_statistics": True,
    }


def _finish(writer) -> None:
    t, Y, sc, impact, t_impact, valid, impact_i, impact_f = _archive_arrays()
    writer.write_sample_batch(0, 4, Y)
    writer.write_metadata(**_v2_metadata())
    writer.write_final(sc, impact, t_impact, valid, impact_i, impact_f)
    writer.finalize()


@pytest.mark.parametrize(
    "fmt, given, written",
    [
        ("npz", "run.h5", "run.npz"),
        ("npz", "run", "run.npz"),
        ("npz", "run.npz", "run.npz"),
        ("hdf5", "run.npz", "run.h5"),
        ("hdf5", "run.hdf5", "run.hdf5"),
    ],
)
def test_output_path_resolved_carries_the_format_suffix(tmp_path, fmt, given, written):
    cfg = BatchPropagationConfig(n_samples=4, seed=1, output_format=fmt, output_path=str(tmp_path / given))
    assert cfg.output_path_resolved == (tmp_path / written).resolve()


def test_npz_run_at_an_h5_path_writes_a_loadable_npz(tmp_path):
    cfg = BatchPropagationConfig(
        n_samples=4, seed=1, output_format="npz", output_path=str(tmp_path / "run.h5"),
    )
    t, *_ = _archive_arrays()
    _finish(_make_writer(cfg, 4, t))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.npz"]
    result = load_batch_result(str(tmp_path / "run.npz"))
    np.testing.assert_array_equal(result.Y, _archive_arrays()[1])


def test_concurrent_writers_on_one_path_do_not_delete_each_other(tmp_path, monkeypatch):
    path = tmp_path / "run.npz"
    t, *_ = _archive_arrays()
    monkeypatch.setattr(storage.os, "getpid", lambda: 1111)
    first = _NPZWriter(path, 4, t)
    t, Y, sc, impact, t_impact, valid, impact_i, impact_f = _archive_arrays()
    first.write_sample_batch(0, 4, Y)
    first.write_metadata(**_v2_metadata())
    first.write_final(sc, impact, t_impact, valid, impact_i, impact_f)
    monkeypatch.setattr(storage.os, "getpid", lambda: 2222)
    second = _NPZWriter(path, 4, t)
    assert first._part_path != second._part_path
    assert first._part_path.exists()
    first.finalize()
    assert path.exists()
    second.abort()
    assert path.exists()


def test_existing_archive_is_refused_without_overwrite(tmp_path):
    path = tmp_path / "run.h5"
    path.write_bytes(b"previous archive")
    t, *_ = _archive_arrays()
    cfg = BatchPropagationConfig(
        n_samples=4, seed=1, output_format="npz", output_path=str(tmp_path / "run.npz"),
        overwrite_output=False,
    )
    (tmp_path / "run.npz").write_bytes(b"previous archive")
    with pytest.raises(FileExistsError, match="already exists"):
        _make_writer(cfg, 4, t)
    assert (tmp_path / "run.npz").read_bytes() == b"previous archive"


def test_existing_archive_is_replaced_by_default_with_a_warning(tmp_path, caplog):
    path = tmp_path / "run.npz"
    path.write_bytes(b"previous archive")
    t, *_ = _archive_arrays()
    cfg = BatchPropagationConfig(n_samples=4, seed=1, output_format="npz", output_path=str(path))
    assert cfg.overwrite_output is True
    with caplog.at_level(logging.WARNING, logger="lunaris.batch.storage"):
        writer = _make_writer(cfg, 4, t)
    assert "will be replaced" in caplog.text
    _finish(writer)
    assert load_batch_result(str(path)).Y.shape == (3, 4, 6)
