from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from st_lrps.runtime import profiling


def test_import_safety():
    assert profiling.RuntimeProfileReport is not None


def test_generate_lunar_shell_queries_shape_and_radius():
    r_ref_m = 1_737_400.0
    queries = profiling.generate_lunar_shell_queries(
        100,
        r_ref_m=r_ref_m,
        alt_min_km=100.0,
        alt_max_km=200.0,
        seed=123,
    )

    assert isinstance(queries, np.ndarray)
    assert queries.shape == (100, 3)
    assert np.all(np.isfinite(queries))
    radii = np.linalg.norm(queries, axis=1)
    alt_km = (radii - r_ref_m) / 1000.0
    assert alt_km.min() >= 100.0 - 1e-9
    assert alt_km.max() <= 200.0 + 1e-9


def test_summarize_times_keys():
    summary = profiling.summarize_times([1.0, 2.0, 3.0])

    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    assert "p95" in summary
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_report_json_and_csv_serialization(tmp_path):
    report = profiling.RuntimeProfileReport(
        config=profiling.RuntimeProfileConfig(
            model_dir="runs/st_lrps_train_fake",
            batch_sizes=[1],
            n_warmup=0,
            n_repeat=1,
        ),
        model_dir="runs/st_lrps_train_fake",
        checkpoint_kind="best",
        checkpoint_path="runs/st_lrps_train_fake/checkpoints/ckpt_best.pt",
        device="cpu",
        dtype="torch.float32",
        torch_version="test",
        cuda_available=False,
        cuda_device_name=None,
        created_at_utc="2026-01-01T00:00:00Z",
        load=profiling.ProfileTimerResult(total_load_s=0.1),
        inference_results=[
            profiling.InferenceProfileResult(
                batch_size=1,
                n_warmup=0,
                n_repeat=1,
                chunk_size=None,
                input_source="synthetic",
                device="cpu",
                total_wall_s=0.01,
                mean_call_s=0.01,
                median_call_s=0.01,
                p95_call_s=0.01,
                min_call_s=0.01,
                max_call_s=0.01,
                samples_per_s=100.0,
                microseconds_per_sample=10_000.0,
                output_shape="(1, 3)",
                output_dtype="float64",
                finite_output_fraction=1.0,
                accel_norm_mean=1.0,
                accel_norm_max=1.0,
            )
        ],
    )

    outputs = profiling.write_runtime_profile_outputs(report, tmp_path, make_plots=False)

    json_path = outputs["json"]
    csv_path = outputs["csv"]
    summary_path = outputs["summary"]
    assert json_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model_dir"] == "runs/st_lrps_train_fake"
    assert payload["inference_results"][0]["batch_size"] == 1
    assert "mean_call_s" in csv_path.read_text(encoding="utf-8")


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "st_lrps.runtime.profiling", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--model-dir" in result.stdout
    assert "--batch-sizes" in result.stdout


def test_synthetic_query_generation_does_not_require_hdf5():
    queries = profiling.generate_lunar_shell_queries(
        4,
        r_ref_m=1_737_400.0,
        alt_min_km=10.0,
        alt_max_km=20.0,
        seed=1,
    )

    assert queries.shape == (4, 3)
