"""Stage-1 resumability of the frozen-orbit search pipeline.

Finished screening blocks are persisted under ``stage1_checkpoint/`` and
replayed on ``--resume``; the resumed stage-1 product is identical to an
uninterrupted run and the work directory is removed once the stage file exists.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.analysis.frozen.search import (  # noqa: E402
    STAGE1_CHECKPOINT_DIR,
    STAGE1_SCREENING,
    FrozenSearchConfig,
    FrozenSearchPipeline,
)
from test_frozen_search_pipeline import (  # noqa: E402
    KeplerScreeningPropagator,
    KeplerValidationPropagator,
    _config,
)


class _CountingKepler(KeplerScreeningPropagator):
    """Kepler screening backend that can be killed on a given block."""

    def __init__(self, *, fail_on_block: int | None = None) -> None:
        self.blocks: list[tuple[int, int]] = []
        self.fail_on_block = fail_on_block

    def propagate(self, Y0, duration_s, output_dt_s):
        index = len(self.blocks)
        self.blocks.append((index, int(Y0.shape[0])))
        if self.fail_on_block is not None and index == self.fail_on_block:
            raise RuntimeError("simulated kill")
        return super().propagate(Y0, duration_s, output_dt_s)


def _pipeline(tmp_path, screening, **overrides):
    cfg = _config(screening_summary_batch_size=4, **overrides)  # 16 samples -> 4 blocks
    return FrozenSearchPipeline(
        cfg,
        screening=screening,
        validation=KeplerValidationPropagator("classical_sh_deg8"),
        out_dir=tmp_path,
    )


def _stage1(path):
    with np.load(path / STAGE1_SCREENING, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def test_killed_stage1_resumes_from_finished_blocks(tmp_path) -> None:
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    ref_backend = _CountingKepler()
    _pipeline(ref_dir, ref_backend).run(resume=True)
    assert len(ref_backend.blocks) == 4
    assert not (ref_dir / STAGE1_CHECKPOINT_DIR).exists()  # cleaned after stage 1

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    killed = _CountingKepler(fail_on_block=2)
    with pytest.raises(RuntimeError, match="simulated kill"):
        _pipeline(run_dir, killed).run(resume=True)
    partials = sorted((run_dir / STAGE1_CHECKPOINT_DIR).glob("block_*_summary.npz"))
    assert [p.name for p in partials] == [
        "block_0000000_0000004_summary.npz",
        "block_0000004_0000008_summary.npz",
    ]
    assert not (run_dir / STAGE1_SCREENING).exists()

    resumed = _CountingKepler()
    _pipeline(run_dir, resumed).run(resume=True)
    assert len(resumed.blocks) == 2  # only the two unfinished blocks were propagated
    assert not (run_dir / STAGE1_CHECKPOINT_DIR).exists()

    ref = _stage1(ref_dir)
    got = _stage1(run_dir)
    assert set(ref) == set(got)
    for key in ref:
        if ref[key].dtype.kind in "fc":
            assert np.array_equal(np.nan_to_num(ref[key], nan=-7.0), np.nan_to_num(got[key], nan=-7.0)), key
        else:
            assert np.array_equal(ref[key], got[key]), key

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    info = manifest["stages"]["stage1"]["checkpoint"]
    assert info["blocks_total"] == 4
    assert info["blocks_restored"] == 2
    assert info["loop_checkpoint_supported"] is False  # the Kepler fake has no checkpoint kwarg


def test_no_resume_ignores_finished_blocks(tmp_path) -> None:
    killed = _CountingKepler(fail_on_block=1)
    with pytest.raises(RuntimeError):
        _pipeline(tmp_path, killed).run(resume=True)
    fresh = _CountingKepler()
    _pipeline(tmp_path, fresh).run(resume=False)
    assert len(fresh.blocks) == 4


def test_zero_interval_disables_block_persistence(tmp_path) -> None:
    killed = _CountingKepler(fail_on_block=1)
    with pytest.raises(RuntimeError):
        _pipeline(tmp_path, killed, screening_checkpoint_interval_s=0.0).run(resume=True)
    assert not (tmp_path / STAGE1_CHECKPOINT_DIR).exists()
    manifest_stage1_missing = not (tmp_path / STAGE1_SCREENING).exists()
    assert manifest_stage1_missing


def test_config_rejects_negative_checkpoint_interval() -> None:
    with pytest.raises(ValueError, match="screening_checkpoint_interval_s"):
        _config(screening_checkpoint_interval_s=-1.0)


def test_cli_checkpoint_flag_maps_to_seconds() -> None:
    from lunaris.cli.frozen_search import build_parser

    args = build_parser().parse_args(["--out", "x"])
    assert args.screening_checkpoint_minutes == 20.0
    args = build_parser().parse_args(["--out", "x", "--screening-checkpoint-minutes", "0"])
    assert args.screening_checkpoint_minutes == 0.0
    assert FrozenSearchConfig.__dataclass_fields__["screening_checkpoint_interval_s"].default == 1_200.0
