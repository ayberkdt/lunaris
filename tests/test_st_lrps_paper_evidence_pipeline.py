"""Parts 2/3 — ST-LRPS paper evidence pipeline (field validation, orbit benchmark
wiring, worst-case, multi-seed, tables, configs)."""

from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_ = pytest.importorskip("torch.nn")

from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.evaluation.validation_suite import (
    compute_field_metrics,
    write_field_validation_csvs,
)
from lunaris.surrogate.st_lrps.paper_evidence import runner as R
from lunaris.surrogate.st_lrps.paper_evidence.multi_seed import (
    aggregate_multi_seed,
    collect_seed_entry,
    write_multi_seed_outputs,
)
from lunaris.surrogate.st_lrps.paper_evidence.paper_tables import (
    csv_to_markdown_table,
    generate_paper_figures,
    generate_paper_tables,
)
from lunaris.surrogate.st_lrps.paper_evidence.worst_case import (
    analyze_worst_cases,
    run_worst_case_from_benchmark_dir,
)

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO / "configs" / "st_lrps" / "paper"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Task 1 — configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "field_validation.json", "benchmark_1day_high_degree.json", "benchmark_5day_general.json",
    "worst_case_analysis.json", "ablation_suite.json",
])
def test_pipeline_configs_exist_and_parse(name):
    cfg = json.loads((_CONFIGS / name).read_text(encoding="utf-8"))
    assert cfg.get("name")


def test_benchmark_configs_are_paper_safe():
    from lunaris.surrogate.st_lrps.evaluation.benchmark_config import (
        is_paper_safe_requested,
        load_benchmark_config,
    )

    for name in ("benchmark_1day_high_degree.json", "benchmark_5day_general.json"):
        cfg = load_benchmark_config(_CONFIGS / name)
        assert is_paper_safe_requested(cfg)
        assert cfg["run_options"]["synthetic"] is False
        assert cfg["validation"]["strict_domain"] is True
        # truth-duplicate baselines must carry a justification.
        assert cfg["validation"].get("truth_baseline_justification")


# ---------------------------------------------------------------------------
# Task 4 — field metric additions + CSV writers
# ---------------------------------------------------------------------------

def test_field_metrics_new_keys():
    n = 100
    xyz = np.zeros((n, 3))
    xyz[:, 0] = R_MOON_SI + np.linspace(100e3, 300e3, n)
    a_true = np.tile([1.0, 0.0, 0.0], (n, 1))
    a_pred = a_true + np.tile([0.0, 2.0e-3, 0.0], (n, 1))
    u = np.zeros(n)
    m = compute_field_metrics(xyz, u, a_true, u, a_pred, r_ref_m=R_MOON_SI)
    assert m["residual_accel_mae_m_s2"] == pytest.approx(2.0e-3, rel=1e-6)
    for key in ("accel_error_p50_m_s2", "accel_error_p90_m_s2", "non_finite_prediction_count",
                "radius_domain_warning_count"):
        assert key in m
    assert m["non_finite_prediction_count"] == 0


def test_field_metrics_counts_non_finite():
    xyz = np.array([[R_MOON_SI + 1e5, 0, 0], [R_MOON_SI + 2e5, 0, 0]])
    a_true = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    a_pred = np.array([[np.nan, 0, 0], [1.0, 0, 0]])
    m = compute_field_metrics(xyz, np.zeros(2), a_true, np.zeros(2), a_pred, r_ref_m=R_MOON_SI)
    assert m["non_finite_prediction_count"] == 1


def test_write_field_validation_csvs(tmp_path):
    report = {
        "field_validation": {
            "seeded_random": {
                "kind": "interpolation", "count": 10, "residual_accel_rmse_m_s2": 1e-6,
                "residual_accel_mae_m_s2": 8e-7, "accel_error_p50_m_s2": 7e-7,
                "altitude_binned_error": [{"altitude_km_min": 100, "altitude_km_max": 200, "count": 5, "accel_rmse": 1e-6}],
                "latitude_binned_error": [{"latitude_deg_min": -90, "latitude_deg_max": 0, "count": 5, "accel_rmse": 1e-6}],
                "longitude_binned_error": [{"longitude_deg_min": -180, "longitude_deg_max": 0, "count": 5, "accel_rmse": 1e-6}],
            },
            "ood_low_altitude": {"kind": "altitude_extrapolation", "error": "empty validation split"},
        }
    }
    paths = write_field_validation_csvs(report, tmp_path)
    for key in ("metrics", "by_altitude", "by_lat_lon", "summary"):
        assert paths[key].exists()
    metrics_rows = list(csv.DictReader((tmp_path / "field_validation_metrics.csv").open(encoding="utf-8")))
    assert {r["policy"] for r in metrics_rows} == {"seeded_random", "ood_low_altitude"}
    latlon_rows = list(csv.DictReader((tmp_path / "field_validation_by_lat_lon.csv").open(encoding="utf-8")))
    assert {r["dimension"] for r in latlon_rows} == {"latitude_deg", "longitude_deg"}


# ---------------------------------------------------------------------------
# Task 6 — worst-case analysis
# ---------------------------------------------------------------------------

def _synthetic_scenarios():
    return [
        # scenario 0: along-track dominated (phase drift)
        {"scenario_id": 0, "model": "ST-LRPS", "rms_pos_err_km": 2.0, "max_pos_err_km": 5.0,
         "final_pos_err_km": 4.0, "radial_rms_km": 0.1, "along_rms_km": 1.9, "cross_rms_km": 0.05,
         "rms_vel_err_ms": 0.2, "domain_warning": ""},
        # scenario 1: radial dominated, OOD (domain warning)
        {"scenario_id": 1, "model": "ST-LRPS", "rms_pos_err_km": 3.0, "max_pos_err_km": 9.0,
         "final_pos_err_km": 6.0, "radial_rms_km": 2.5, "along_rms_km": 0.4, "cross_rms_km": 0.1,
         "rms_vel_err_ms": 0.4, "domain_warning": "altitude outside training envelope"},
        # scenario 2: small error
        {"scenario_id": 2, "model": "ST-LRPS", "rms_pos_err_km": 0.5, "max_pos_err_km": 1.0,
         "final_pos_err_km": 0.8, "radial_rms_km": 0.1, "along_rms_km": 0.4, "cross_rms_km": 0.05,
         "rms_vel_err_ms": 0.05, "domain_warning": ""},
        # a competing baseline row that must be ignored
        {"scenario_id": 0, "model": "SH20", "rms_pos_err_km": 50.0, "max_pos_err_km": 99.0,
         "final_pos_err_km": 80.0, "radial_rms_km": 10, "along_rms_km": 40, "cross_rms_km": 5,
         "rms_vel_err_ms": 4.0, "domain_warning": ""},
    ]


def test_analyze_worst_cases_ranks_and_flags():
    defs = [{"scenario_id": 0, "hp_km": 100, "ha_km": 120, "a_km": 1837.4, "e": 0.01, "inc_deg": 90},
            {"scenario_id": 1, "hp_km": 30, "ha_km": 50, "a_km": 1777.4, "e": 0.0, "inc_deg": 45}]
    analysis = analyze_worst_cases(_synthetic_scenarios(), model="ST-LRPS", scenario_defs=defs,
                                   train_alt_min_km=100.0, train_alt_max_km=500.0, top_n=2)
    assert analysis["n_scenarios"] == 3  # SH20 row excluded
    worst_max = analysis["rankings"]["max_pos_err_km"]
    assert worst_max[0]["scenario_id"] == 1 and worst_max[1]["scenario_id"] == 0
    # scenario 0: along-track dominated phase drift.
    s0 = next(r for r in analysis["worst_scenarios"] if r["scenario_id"] == 0)
    assert s0["phase_drift_dominated"] is True and s0["radial_dominated"] is False
    assert s0["orbital_period_s"] and s0["orbital_period_s"] > 0
    # scenario 1: radial dominated + OOD (domain warning AND below envelope).
    s1 = next(r for r in analysis["worst_scenarios"] if r["scenario_id"] == 1)
    assert s1["radial_dominated"] is True and s1["is_ood"] is True
    assert s1["leaves_training_envelope"] is True


def test_run_worst_case_from_benchmark_dir(tmp_path):
    bdir = tmp_path / "bench"
    _write_csv(bdir / "scenario_results.csv", _synthetic_scenarios())
    _write_csv(bdir / "scenarios.csv", [
        {"scenario_id": 0, "hp_km": 100, "ha_km": 120, "a_km": 1837.4, "e": 0.01, "inc_deg": 90},
        {"scenario_id": 1, "hp_km": 30, "ha_km": 50, "a_km": 1777.4, "e": 0.0, "inc_deg": 45},
        {"scenario_id": 2, "hp_km": 200, "ha_km": 250, "a_km": 1962.4, "e": 0.01, "inc_deg": 60},
    ])
    paths = run_worst_case_from_benchmark_dir(bdir, tmp_path / "wc", train_alt_min_km=100, train_alt_max_km=500)
    assert paths["csv"].exists() and paths["summary"].exists()
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "Worst" in summary and "phase-drift" in summary


# ---------------------------------------------------------------------------
# Task 7 — multi-seed
# ---------------------------------------------------------------------------

def test_collect_and_aggregate_multi_seed(tmp_path):
    # Synthetic per-seed field + benchmark CSVs.
    entries = []
    for i, seed in enumerate((42, 123, 2026)):
        field = tmp_path / f"field_{seed}.csv"
        _write_csv(field, [
            {"policy": "seeded_random", "residual_accel_rmse_m_s2": 1e-6 + i * 1e-7},
            {"policy": "spatial_block", "residual_accel_rmse_m_s2": 2e-6 + i * 1e-7},
            {"policy": "ood_low_altitude", "residual_accel_rmse_m_s2": 5e-6 + i * 1e-7},
        ])
        bench = tmp_path / f"bench_{seed}.csv"
        _write_csv(bench, [{"model": "ST-LRPS", "median_rms_pos_err_km": 1.0 + i,
                            "p95_rms_pos_err_km": 2.0 + i, "max_rms_pos_err_km": 3.0 + i, "runtime_s": 10.0}])
        entries.append(collect_seed_entry(seed, field_metrics_csv=field,
                                          benchmark_metrics={"1day": bench}, artifact_hash=f"hash{seed}"))
    summary = aggregate_multi_seed(entries)
    assert summary["n_seeds"] == 3
    assert summary["single_seed_limitation"] is False
    assert summary["best_seed"] == 42 and summary["worst_seed"] == 2026
    stats = summary["statistics"]["bench1day_median_rms_km"]
    assert stats["mean"] == pytest.approx(2.0) and stats["min"] == 1.0 and stats["max"] == 3.0
    paths = write_multi_seed_outputs(summary, tmp_path / "ms")
    assert paths["csv"].exists()
    assert "MEAN" in (tmp_path / "ms" / "multi_seed_summary.csv").read_text(encoding="utf-8")


def test_single_seed_limitation_labelled():
    summary = aggregate_multi_seed([{"seed": 42, "bench1day_median_rms_km": 1.0}])
    assert summary["single_seed_limitation"] is True
    assert "single" in summary["single_seed_note"].lower()


# ---------------------------------------------------------------------------
# Task 9 — tables + figures
# ---------------------------------------------------------------------------

def test_csv_to_markdown_table(tmp_path):
    p = tmp_path / "x.csv"
    _write_csv(p, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    md = csv_to_markdown_table(p)
    assert "| a | b |" in md and "| 1 | 2 |" in md
    assert csv_to_markdown_table(tmp_path / "missing.csv") == "_(no data)_\n"


def test_generate_paper_tables_from_csvs(tmp_path):
    # Place a couple of recognised CSVs in the evidence tree.
    _write_csv(tmp_path / "orbit_benchmarks" / "b1" / "metrics_summary.csv",
               [{"model": "ST-LRPS", "median_rms_pos_err_km": 1.2}])
    _write_csv(tmp_path / "field_validation" / "seed42" / "field_validation_metrics.csv",
               [{"policy": "spatial_block", "residual_accel_rmse_m_s2": 2e-6}])
    result = generate_paper_tables(tmp_path, tmp_path / "tables")
    assert "orbit_benchmark_summary" in result["written"]
    assert "field_validation_summary" in result["written"]
    assert (tmp_path / "tables" / "TABLES_INDEX.md").exists()
    # numbers come from the CSV, not hardcoded.
    assert "1.2" in (tmp_path / "tables" / "table_orbit_benchmark_summary.md").read_text(encoding="utf-8")


def test_generate_paper_figures_safe(tmp_path):
    _write_csv(tmp_path / "metrics_summary.csv",
               [{"model": "SH20", "median_rms_pos_err_km": 5.0}, {"model": "ST-LRPS", "median_rms_pos_err_km": 1.0}])
    result = generate_paper_figures(tmp_path, tmp_path / "figures")
    # Either rendered (matplotlib present) or cleanly skipped — never crash.
    assert "skipped" in result


# ---------------------------------------------------------------------------
# Task 2/5 — runner stage wiring
# ---------------------------------------------------------------------------

def test_worst_case_stage_end_to_end(tmp_path):
    bdir = tmp_path / "bench_out"
    _write_csv(bdir / "scenario_results.csv", _synthetic_scenarios())
    cfg = {"schema_version": 1, "name": "wc", "benchmark_dir": str(bdir), "model": "ST-LRPS", "top_n": 3}
    cfg_path = tmp_path / "wc.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    rc = R.run_worst_case_stage(cfg_path, evidence_root=tmp_path / "ev", dry_run=False)
    assert rc == 0
    assert (tmp_path / "ev" / "worst_case_analysis" / "worst_case_scenarios.csv").exists()
    manifest = json.loads((tmp_path / "ev" / "manifests" / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert "worst_case" in manifest["runs"]


def test_tables_stage_end_to_end(tmp_path):
    ev = tmp_path / "ev"
    _write_csv(ev / "orbit_benchmarks" / "b" / "runtime_summary.csv",
               [{"model": "ST-LRPS", "total_runtime_s": 0.5, "n_scenarios": 2}])
    rc = R.run_tables_stage(evidence_root=ev, dry_run=False)
    assert rc == 0
    assert (ev / "tables" / "table_runtime_speedup_summary.md").exists()


def test_stage_dry_runs_record_manifest(tmp_path):
    for stage in ("field-validation", "worst-case", "multi-seed", "tables"):
        rc = R.main(["--stage", stage, "--evidence-root", str(tmp_path), "--dry-run"])
        assert rc == 0
    manifest = json.loads((tmp_path / "manifests" / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert {"field_validation", "worst_case", "multi_seed", "tables"}.issubset(set(manifest["runs"].keys()))


# ---------------------------------------------------------------------------
# field-validation stage real run (needs torch + a contract model)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
from dataset_pipeline_test_utils import write_toy_contract_h5  # noqa: E402
from st_lrps_contract_test_utils import make_contract_run  # noqa: E402


def test_field_validation_stage_real_run(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=500.0)
    dataset = write_toy_contract_h5(tmp_path / "cloud.h5", n=400, alt_min_km=100.0, alt_max_km=500.0)
    cfg = {
        "schema_version": 1, "name": "fv", "model_dir": str(run["run_dir"]), "dataset": str(dataset),
        "device": "cpu", "split_seed": 7, "val_fraction": 0.2,
        "policies": ["seeded_random", "spatial_block", "ood_low_altitude"],
    }
    cfg_path = tmp_path / "fv.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    ev = tmp_path / "ev"
    rc = R.run_field_validation_stage(cfg_path, evidence_root=ev, dry_run=False)
    assert rc == 0
    out = ev / "field_validation" / "contract_run"
    assert (out / "field_validation_metrics.csv").exists()
    assert (out / "field_validation_by_altitude.csv").exists()
    rows = list(csv.DictReader((out / "field_validation_metrics.csv").open(encoding="utf-8")))
    policies = {r["policy"] for r in rows}
    assert {"seeded_random", "spatial_block", "ood_low_altitude"}.issubset(policies)
