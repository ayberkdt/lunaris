"""Paper-evidence multi-seed stage: every trained seed is aggregated with its own results.

Regression tests for ``lunaris.surrogate.st_lrps.paper_evidence.runner``:

* the multi-seed stage read the seed from the train entry's ``config`` record,
  which holds only path and hashes, so every train entry was skipped and the
  summary reported 0 seeds;
* field-validation entries were all written under the constant manifest key
  ``field_validation``, so a second model overwrote the first, and the
  per-seed lookup used a ``seed{N}`` directory the stage never writes;
* shared orbit-benchmark CSVs were attached to every seed (std 0);
* ``_run_key`` returned the config name before the seed, so a ``--seed`` sweep
  of one named config collapsed onto one manifest entry.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lunaris.surrogate.st_lrps.paper_evidence import runner as R


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _train_entry(ev: Path, seed: int, ckpt: Path) -> None:
    config = {"name": "final", "seed": seed}
    R._write_manifest(
        ev / "manifests" / "evidence_manifest.json", R._run_key(config), config, None,
        ev / f"out{seed}", ["python", "train"], artifacts={"checkpoint": ckpt}, dry_run=False,
        extra={"evidence_dir": str(ev / "training" / R._run_key(config))},
    )


def _field_entry(ev: Path, model: str, seed: int, rmse: float) -> None:
    csv_path = _write_csv(
        ev / "field_validation" / model / "field_validation_metrics.csv",
        [{"policy": "seeded_random", "residual_accel_rmse_m_s2": rmse}],
    )
    R._record_stage(
        ev, "field-validation", f"field_validation:{model}", out_dir=csv_path.parent,
        artifacts={"field_metrics_csv": csv_path}, extra={"seed": seed},
    )


def test_run_key_keeps_a_seed_sweep_apart() -> None:
    assert R._run_key({"name": "final", "seed": 1}) == "final_seed1"
    assert R._run_key({"name": "final", "seed": 2}) == "final_seed2"
    assert R._run_key({"name": "st_lrps_paper_full_seed42", "seed": 42}) == "st_lrps_paper_full_seed42"
    assert R._run_key({"name": "final"}) == "final"
    assert R._run_key({"seed": 7}) == "seed7"


def test_multi_seed_stage_pairs_each_seed_with_its_own_field_metrics(tmp_path: Path) -> None:
    ev = tmp_path / "ev"
    for seed, rmse in ((1, 1.0e-6), (2, 3.0e-6)):
        ckpt = tmp_path / f"ckpt{seed}.pt"
        ckpt.write_bytes(f"checkpoint {seed}".encode())
        _train_entry(ev, seed, ckpt)
        _field_entry(ev, f"model_seed{seed}", seed, rmse)
    # A shared benchmark CSV names no candidate: it must not be given to both seeds.
    _write_csv(ev / "orbit_benchmarks" / "benchmark_1day_high_degree" / "metrics_summary.csv",
               [{"model": "ST-LRPS", "median_rms_pos_err_km": 1.0}])

    assert R.run_multi_seed_stage(evidence_root=ev) == 0

    manifest = json.loads((ev / "manifests" / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert {"final_seed1", "final_seed2", "field_validation:model_seed1",
            "field_validation:model_seed2"}.issubset(manifest["runs"])
    assert manifest["runs"]["multi_seed"]["extra"]["n_seeds"] == 2

    rows = list(csv.DictReader((ev / "tables" / "multi_seed_summary.csv").open(encoding="utf-8")))
    per_seed = {r["seed"]: r for r in rows if r["seed"] in {"1", "2"}}
    assert float(per_seed["1"]["field_random_accel_rmse"]) == 1.0e-6
    assert float(per_seed["2"]["field_random_accel_rmse"]) == 3.0e-6
    assert per_seed["1"]["bench1day_median_rms_km"] in ("", "nan", "None")


def test_single_seed_may_borrow_shared_outputs(tmp_path: Path) -> None:
    ev = tmp_path / "ev"
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"checkpoint")
    _train_entry(ev, 42, ckpt)
    _write_csv(ev / "field_validation" / "field_validation_metrics.csv",
               [{"policy": "seeded_random", "residual_accel_rmse_m_s2": 2.0e-6}])
    _write_csv(ev / "orbit_benchmarks" / "benchmark_1day_high_degree" / "metrics_summary.csv",
               [{"model": "ST-LRPS", "median_rms_pos_err_km": 1.5}])

    assert R.run_multi_seed_stage(evidence_root=ev) == 0

    rows = list(csv.DictReader((ev / "tables" / "multi_seed_summary.csv").open(encoding="utf-8")))
    (row,) = [r for r in rows if r["seed"] == "42"]
    assert float(row["field_random_accel_rmse"]) == 2.0e-6
    assert float(row["bench1day_median_rms_km"]) == 1.5
