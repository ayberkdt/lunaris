"""Batch engine: t0_s refusal and the summary_only sub-batch cap.

- Every batch backend archives on a ``[0, duration_s]`` axis from
  ``start_date``; the GPU/torch loops start their clock at 0 and the CPU path
  resampled a ``t0_s``-offset solution onto that axis (clamping every time
  before ``t0_s`` to the initial state). ``t0_s != 0`` is refused.
- summary_only mode set the host budget to 0, so every sub-batch held one
  sample. The cap now uses the configured host budget; the summary is
  unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lunaris.batch.engine import BatchPropagationEngine
from lunaris.batch.storage import _host_memory_budget_bytes
from lunaris.common.batch_defs import BatchPropagationConfig, StateUncertainty

MU = 4.9028e12


class _Time:
    t0_s = 3_600.0


class _SimCfg:
    time = _Time()


def test_nonzero_t0_is_refused_before_any_work():
    engine = BatchPropagationEngine.__new__(BatchPropagationEngine)
    engine._cfg = BatchPropagationConfig(n_samples=2, seed=1)
    engine._sim_cfg = _SimCfg()
    engine._cb = None
    with pytest.raises(ValueError, match="t0_s = 3600.0"):
        engine.run()


def test_host_budget_is_capped_by_free_ram():
    cfg = BatchPropagationConfig(n_samples=2, seed=1, max_result_memory_gb=1.0)
    gib = 1024 ** 3
    assert _host_memory_budget_bytes(cfg, available_host_memory_bytes=lambda: None) == gib
    assert _host_memory_budget_bytes(cfg, available_host_memory_bytes=lambda: 10 * gib) == gib
    small = _host_memory_budget_bytes(cfg, available_host_memory_bytes=lambda: gib // 4)
    assert 0 < small < gib // 4


def _summary_run(max_vram_gb: float):
    try:
        from lunaris.common.constants import R_MOON
        from lunaris.core.config import load_default_config, replace_sim_config

        cfg = load_default_config()
    except Exception as exc:  # optional data/config assets
        pytest.skip(f"default config assets unavailable: {exc}")
    r0 = float(R_MOON) + 100_000.0
    v0 = float(np.sqrt(MU / r0))
    cfg = replace_sim_config(cfg, initial_state=np.array([r0, 0.0, 0.0, 0.0, v0, 0.0]))
    cfg = replace_sim_config(cfg, time=replace(cfg.time, duration_s=1200.0, output_dt_s=300.0))
    batch_cfg = BatchPropagationConfig(
        n_samples=5,
        seed=3,
        state=StateUncertainty(sigma_r_m=200.0, sigma_v_m_s=0.2),
        use_gpu=False,
        dt_s=60.0,
        output_format="hdf5",
        result_storage_mode="memory",
        output_mode="summary_only",
        summary_top_k=3,
        max_vram_gb=max_vram_gb,
    )
    batch_counts: list[int] = []

    def progress(payload):
        if payload.get("batch_count"):
            batch_counts.append(int(payload["batch_count"]))

    try:
        result = BatchPropagationEngine(cfg, batch_cfg, progress_callback=progress).run()
    except RuntimeError as exc:
        if "bootstrap" in str(exc).lower():
            pytest.skip(f"simulation assets unavailable: {exc}")
        raise
    return result, max(batch_counts)


def test_summary_only_batches_samples_and_keeps_the_summary_identical():
    # max_vram_gb so small that effective_max_batch() == 1: the old one-sample
    # sub-batches, the reference for "identical results".
    one_by_one, n_batches_ref = _summary_run(max_vram_gb=1e-9)
    batched, n_batches = _summary_run(max_vram_gb=4.0)
    assert n_batches_ref == 5
    assert n_batches == 1

    ref = one_by_one.diagnostics
    new = batched.diagnostics
    assert new["summary_selected_indices"] == ref["summary_selected_indices"]
    np.testing.assert_array_equal(batched.Y, one_by_one.Y)
    ref_fields = ref["batch_summary"]["fields"]
    new_fields = new["batch_summary"]["fields"]
    assert set(new_fields) == set(ref_fields)
    for key, value in ref_fields.items():
        if isinstance(value, dict):
            for sub, arr in value.items():
                np.testing.assert_array_equal(new_fields[key][sub], arr, err_msg=f"{key}.{sub}")
        else:
            np.testing.assert_array_equal(np.asarray(new_fields[key]), np.asarray(value), err_msg=key)
