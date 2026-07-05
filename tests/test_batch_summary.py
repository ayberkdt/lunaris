"""R23: summary-only ensemble output — schema, reduction, top-K, engine wiring."""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.batch.summary import (
    BATCH_SUMMARY_SCHEMA_VERSION,
    TopKTrajectoryBuffer,
    merge_summaries,
    summarize_ensemble,
)
from lunaris.common.batch_defs import BatchPropagationConfig, StateUncertainty

MU = 4.9028e12
R_REF = 1.738e6


def _kepler_states(a_m: float, e: float, n_t: int, *, argp_rad: float = 0.3) -> np.ndarray:
    """Planar Kepler ellipse sampled over one anomaly sweep -> (T, 6) states."""
    nu = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
    p = a_m * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))
    theta = nu + argp_rad
    pos = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(r)], axis=1)
    vr = np.sqrt(MU / p) * e * np.sin(nu)
    vt = np.sqrt(MU / p) * (1.0 + e * np.cos(nu))
    vel = np.stack(
        [
            vr * np.cos(theta) - vt * np.sin(theta),
            vr * np.sin(theta) + vt * np.cos(theta),
            np.zeros_like(r),
        ],
        axis=1,
    )
    return np.concatenate([pos, vel], axis=1)


def _ensemble(n_t: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 6_000.0, n_t)
    stable = _kepler_states(R_REF + 100_000.0, 0.02, n_t)
    dead = np.tile(np.array([R_REF - 5_000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (n_t, 1))
    Y = np.stack([stable, dead], axis=1)  # (T, 2, 6)
    impact = np.array([0.0, 1.0])
    t_imp = np.array([np.nan, 0.0])
    return t, Y, impact, t_imp


def test_summary_schema_and_stable_sample_metrics():
    t, Y, impact, t_imp = _ensemble()
    summary = summarize_ensemble(t, Y, impact, t_imp, mu_m3s2=MU, r_ref_m=R_REF)

    assert summary["schema_version"] == BATCH_SUMMARY_SCHEMA_VERSION
    assert summary["n_samples"] == 2
    f = summary["fields"]
    # Exact Kepler states: element envelopes are numerically tight.
    assert f["e_range"][0] < 1e-6
    assert abs(f["e_min"][0] - 0.02) < 1e-6
    assert f["h_peri_range_km"][0] < 1e-3
    # Periapsis altitude of a(1-e) ellipse above R_REF.
    expected_hp_km = ((R_REF + 100_000.0) * 0.98 - R_REF) / 1_000.0
    assert abs(f["h_peri_min_km"][0] - expected_hp_km) < 1e-3
    assert f["omega_behavior"][0] == "librating"
    assert np.isfinite(f["score"][0])
    assert f["validation_stage"][0] == "screened"
    # Initial/final element blocks are present and self-consistent.
    assert abs(f["initial_elements"]["e"][0] - 0.02) < 1e-6
    assert abs(f["final_elements"]["a_km"][0] - (R_REF + 100_000.0) / 1_000.0) < 1e-3


def test_summary_impacted_sample_scores_inf_and_excludes_frozen_rows():
    t, Y, impact, t_imp = _ensemble()
    summary = summarize_ensemble(t, Y, impact, t_imp, mu_m3s2=MU, r_ref_m=R_REF)
    f = summary["fields"]
    assert f["impact_flag"][1] == 1.0
    assert f["t_impact_s"][1] == 0.0
    assert np.isinf(f["score"][1])
    # All rows are post-impact (t_impact=0 keeps only t=0): envelopes empty->NaN.
    assert np.isnan(f["trend_e_per_day"][1])
    assert f["omega_behavior"][1] == "indeterminate"


def test_merge_summaries_concatenates_fields():
    t, Y, impact, t_imp = _ensemble()
    part = summarize_ensemble(t, Y, impact, t_imp, mu_m3s2=MU, r_ref_m=R_REF)
    merged = merge_summaries([part, part])
    assert merged["n_samples"] == 4
    assert merged["fields"]["score"].shape == (4,)
    assert merged["fields"]["initial_elements"]["e"].shape == (4,)


def test_top_k_buffer_streams_best_scores_across_batches():
    n_t = 5
    buf = TopKTrajectoryBuffer(2)

    def _offer(start: int, scores: list[float]) -> None:
        n = len(scores)
        Y = np.full((n_t, n, 6), float(start), dtype=np.float64)
        buf.offer_batch(
            global_start=start,
            scores=np.asarray(scores),
            Y_batch=Y,
            impact_flags=np.zeros(n),
            t_impact=np.full(n, np.nan),
        )

    _offer(0, [0.5, np.inf, 0.9])       # candidates 0 (0.5), 2 (0.9)
    _offer(3, [0.1, 0.7, np.inf])       # candidate 3 (0.1) displaces 2

    assert buf.selected_indices == [3, 0]
    assert buf.scores == [0.1, 0.5]
    stacked = buf.stacked_trajectories(n_t)
    assert stacked.shape == (n_t, 2, 6)
    # Trajectories follow their entries (batch start value marks provenance).
    assert stacked[0, 0, 0] == 3.0 and stacked[0, 1, 0] == 0.0
    arrays = buf.entry_arrays()
    assert arrays["sample_indices"].tolist() == [3, 0]


def test_config_validates_output_mode_and_top_k():
    with pytest.raises(ValueError, match="output_mode"):
        BatchPropagationConfig(n_samples=4, seed=1, output_mode="bogus")
    with pytest.raises(ValueError, match="summary_top_k"):
        BatchPropagationConfig(n_samples=4, seed=1, output_mode="summary_only", summary_top_k=0)
    cfg = BatchPropagationConfig(n_samples=4, seed=1, output_mode="summary_only", summary_top_k=3)
    assert cfg.output_mode == "summary_only"


def test_engine_summary_only_mode_end_to_end(tmp_path):
    """Engine wiring: summary mode returns top-K trajectories + full-N summary
    and writes NO trajectory archive."""
    try:
        from lunaris.core.config import load_default_config, replace_sim_config
    except Exception as exc:  # pragma: no cover - optional data/config assets
        pytest.skip(f"default config unavailable: {exc}")

    from dataclasses import replace

    from lunaris.batch.engine import BatchPropagationEngine
    from lunaris.common.constants import R_MOON

    try:
        cfg = load_default_config()
    except Exception as exc:
        pytest.skip(f"default config assets unavailable: {exc}")

    r0 = float(R_MOON) + 100_000.0
    v0 = float(np.sqrt(MU / r0))
    cfg = replace_sim_config(cfg, initial_state=np.array([r0, 0.0, 0.0, 0.0, v0, 0.0]))
    cfg = replace_sim_config(cfg, time=replace(cfg.time, duration_s=1200.0, output_dt_s=300.0))

    out_path = tmp_path / "summary_batch.h5"
    batch_cfg = BatchPropagationConfig(
        n_samples=4,
        seed=3,
        state=StateUncertainty(sigma_r_m=200.0, sigma_v_m_s=0.2),
        use_gpu=False,
        dt_s=60.0,
        output_format="hdf5",
        output_path=str(out_path),
        result_storage_mode="memory",
        output_mode="summary_only",
        summary_top_k=2,
    )
    try:
        result = BatchPropagationEngine(cfg, batch_cfg).run()
    except RuntimeError as exc:
        if "bootstrap" in str(exc).lower():
            pytest.skip(f"simulation assets unavailable: {exc}")
        raise

    diag = result.diagnostics
    assert diag["output_mode"] == "summary_only"
    summary = diag["batch_summary"]
    assert summary["schema_version"] == BATCH_SUMMARY_SCHEMA_VERSION
    assert summary["n_samples"] == 4
    assert summary["fields"]["score"].shape == (4,)
    # Only the top-K full histories are retained.
    k_kept = result.Y.shape[1]
    assert k_kept <= 2
    assert len(diag["summary_selected_indices"]) == k_kept
    assert result.archive_path is None
    # No trajectory archive is written in summary mode.
    assert not out_path.exists()
