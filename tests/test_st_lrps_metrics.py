# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import json
from types import SimpleNamespace

import pytest

import lunaris.surrogate.st_lrps.training.metrics as metrics_mod
from lunaris.surrogate.st_lrps.training.metrics import (
    LOWER_IS_BETTER,
    compute_checkpoint_score,
    flatten_epoch_metrics,
)


def _cfg(metric: str = "hybrid", alpha: float = 0.30):
    return SimpleNamespace(best_metric=metric, hybrid_direction_alpha=alpha)


def _val_stats():
    return {
        "val_total_loss": 1.20,
        "val_base_loss": 1.00,
        "val_physics_loss": 0.20,
        "mse_u": 0.40,
        "mse_a": 0.60,
        "loss_dir": 0.50,
    }


@pytest.mark.parametrize(
    ("metric", "expected", "formula"),
    [
        ("val_total_loss", 1.20, "val_total_loss"),
        ("val_base_loss", 1.00, "val_base_loss"),
        ("hybrid", 1.15, "val_base_loss + 0.30 * val_loss_dir"),
        ("direction_loss", 0.50, "val_loss_dir"),
    ],
)
def test_compute_checkpoint_score_modes(metric, expected, formula):
    score, report = compute_checkpoint_score(_val_stats(), _cfg(metric))
    assert score == pytest.approx(expected)
    assert report["score"] == pytest.approx(expected)
    assert report["formula"] == formula
    assert report["best_metric"] == metric
    assert report["lower_is_better"] is LOWER_IS_BETTER is True


def test_compute_checkpoint_score_total_loss_alias_warns_once(caplog):
    metrics_mod._WARNED_ALIASES.clear()
    caplog.set_level(logging.WARNING)
    score, report = compute_checkpoint_score(_val_stats(), _cfg("total_loss"))
    assert score == pytest.approx(1.20)
    assert report["best_metric"] == "val_total_loss"
    assert "deprecated" in caplog.text


def test_compute_checkpoint_score_missing_required_metric_fails():
    stats = _val_stats()
    stats.pop("val_total_loss")
    with pytest.raises(KeyError):
        compute_checkpoint_score(stats, _cfg("val_total_loss"))


def test_flatten_epoch_metrics_stable_keys_and_optional_missing():
    score, report = compute_checkpoint_score(_val_stats(), _cfg("hybrid"))
    report.update(
        {
            "eligible_for_best": True,
            "is_best_update": True,
            "best_epoch": 3,
            "best_score": score,
        }
    )
    row = flatten_epoch_metrics(
        2,
        {"loss": 2.0, "loss_opt": 1.8, "mse_u": 0.7, "mse_a": 1.1, "lr": 1e-4},
        _val_stats(),
        report,
        _cfg("hybrid"),
    )
    for key in (
        "epoch",
        "epoch_display",
        "train_loss_total",
        "train_loss_objective",
        "val_loss_total",
        "val_loss_base",
        "checkpoint_score",
        "checkpoint_formula",
        "best_metric",
        "is_best_eligible",
        "is_best_update",
        "best_epoch",
        "best_score",
        "samples_seen",
        "optimizer_steps",
        "epoch_time_s",
    ):
        assert key in row
    assert row["epoch"] == 2
    assert row["epoch_display"] == 3
    assert row["checkpoint_score"] == pytest.approx(score)
    assert row["train_loss_dir"] == pytest.approx(0.0)


def test_publication_eval_suite_outputs_summary_files(tmp_path):
    from lunaris.surrogate.st_lrps.evaluation.cli import _write_publication_eval_suite

    eval_dir = tmp_path / "eval" / "test"
    eval_dir.mkdir(parents=True)
    (eval_dir / "eval_report.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "n_samples": 2,
                    "U": {"rmse": 1.0, "mae": 0.5},
                    "residual_vector_metrics": {
                        "rmse": 2.0,
                        "mae": 1.0,
                        "linf": 3.0,
                        "rel_mean_pct": 4.0,
                        "rel_median": 0.03,
                        "percentiles": {"p50": 1.0, "p90": 2.0, "p95": 2.5, "p99": 2.9, "max": 3.0},
                    },
                    "angular_metrics": {"residual_all": {"mean_deg": 1.2, "p90_deg": 2.3, "p95_deg": 3.4}},
                    "a_directional": {
                        "accel_err_radial_rmse": 0.1,
                        "accel_err_cross_radial_rmse": 0.2,
                        "accel_err_radial_mae": 0.05,
                        "accel_err_cross_radial_mae": 0.06,
                    },
                    "altitude_min_km": 100.0,
                    "altitude_max_km": 500.0,
                    "inference_samples_per_sec": 123.0,
                    "inference_time_s": 0.1,
                    "device": "cpu",
                    "dtype": "float32",
                }
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "altitude_binned_metrics.csv").write_text(
        "alt_km_lo,alt_km_hi,n,rmse_U,rmse_accel,mae_a_vec,p95_a_error,angular_mean_deg,angular_p90_deg,radial_rmse,cross_rmse\n"
        "100,150,2,1,2,1,2.5,1.2,2.3,0.1,0.2\n",
        encoding="utf-8",
    )

    _write_publication_eval_suite(
        tmp_path / "eval",
        [("test", tmp_path / "test.h5", eval_dir)],
        model_dir=tmp_path / "run",
        alt_bin_km=50.0,
    )

    assert (tmp_path / "eval" / "summary_metrics.json").exists()
    assert (tmp_path / "eval" / "summary_metrics.csv").exists()
    assert (tmp_path / "eval" / "altitude_binned_metrics.csv").exists()
    summary = json.loads((tmp_path / "eval" / "summary_metrics.json").read_text(encoding="utf-8"))
    assert summary[0]["split"] == "test"
    assert summary[0]["rmse_a_vec"] == pytest.approx(2.0)
