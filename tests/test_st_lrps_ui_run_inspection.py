"""Pure, Qt-free coverage for the Studio run inspection contract."""

from __future__ import annotations

import json
from pathlib import Path

from lunaris.surrogate.st_lrps.ui.studio_parts.run_inspection import (
    config_diff,
    load_run_records,
    provenance_items,
    read_periodic_evals,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_run_records_skip_partial_manifest_and_normalize_fields(tmp_path: Path) -> None:
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "run_manifest.json").write_text("{", encoding="utf-8")
    run = tmp_path / "run_a"
    run.mkdir()
    _write_json(
        run / "run_manifest.json",
        {
            "run_id": "run-a",
            "status": "COMPLETED",
            "best_score": 1.25e-4,
            "best_epoch": 8,
            "latest_epoch": 10,
            "dataset_meta": {"dataset_name": "cloud_a"},
            "compute_accounting": {"device": "cpu"},
        },
    )
    records = load_run_records(tmp_path)
    assert len(records) == 1
    assert records[0]["name"] == "run-a"
    assert records[0]["status"] == "completed"
    assert records[0]["dataset"] == "cloud_a"
    assert records[0]["device"] == "cpu"


def test_periodic_eval_and_config_diff_are_stable(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "periodic_evals").mkdir()
    (first / "periodic_evals" / "history.jsonl").write_text(
        '{"epoch": 4, "rmse_u": 0.1, "rmse_a": 0.2, "angle": 1.5}\nmalformed\n',
        encoding="utf-8",
    )
    _write_json(first / "config.json", {"model": {"hidden": 128}, "lr": 1e-3})
    _write_json(second / "config.json", {"model": {"hidden": 256}, "lr": 1e-3})
    rows = read_periodic_evals(first)
    assert rows == [{"epoch": 4, "rmse_u": 0.1, "rmse_a": 0.2, "angle": 1.5, "raw": rows[0]["raw"]}]
    diff = config_diff([first, second])
    assert diff == [{"field": "model.hidden", "first": "128", "second": "256"}]


def test_provenance_marks_non_train_scaler_as_warning(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "run_manifest.json",
        {
            "split_policy": "fixed",
            "split_counts": {"train": 8, "val": 2, "test": 2},
            "scaler_fit_scope": "all_rows",
            "requested_device": "cuda",
            "compute_accounting": {"device": "cpu"},
            "dataset_sha256": "0123456789abcdef0123",
        },
    )
    items = dict((label, (value, kind)) for label, value, kind in provenance_items(tmp_path))
    assert items["Scaler fit scope"] == ("all_rows", "warning")
    assert items["Device"] == ("cuda → cpu", "info")
    assert items["Dataset hash"][0] == "0123456789abcdef"
