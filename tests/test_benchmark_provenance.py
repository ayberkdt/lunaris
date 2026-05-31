from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lunaris.surrogate.st_lrps.evaluation.provenance import (
    artifact_record,
    build_benchmark_manifest,
    sha256_file,
    sha256_payload,
)


def _config(tmp_path: Path, gravity_file: Path, model_dir: Path) -> dict:
    return {
        "schema_version": 1,
        "name": "prov_fixture",
        "scenario": {
            "seed": 42,
            "count": 1,
            "type": "bounded_keplerian",
            "altitude_min_km": 100.0,
            "altitude_max_km": 200.0,
        },
        "propagation": {"duration_days": 0.01, "output_dt_s": 60.0, "integrator": "RK4", "dt_s": 30.0, "dtype": "float64"},
        "truth": {"model": "spherical_harmonics", "degree": 20, "gravity_file": str(gravity_file)},
        "baselines": [{"name": "SH20", "model": "spherical_harmonics", "degree": 20}],
        "surrogate": {"enabled": True, "name": "ST-LRPS", "model_dir": str(model_dir), "baseline_degree": 20},
        "outputs": {"out_dir": str(tmp_path), "write_figures": True, "write_csv": True, "write_json": True},
    }


def test_sha256_file_uses_file_bytes(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"lunaris")
    assert sha256_file(path) == hashlib.sha256(b"lunaris").hexdigest()


def test_artifact_record_records_missing_reason(tmp_path):
    record = artifact_record(tmp_path / "missing.pt", label="checkpoint")
    assert record["sha256"] is None
    assert "does not exist" in record["missing_reason"]


def test_manifest_captures_config_model_and_environment_hashes(tmp_path):
    config_path = tmp_path / "benchmark.json"
    gravity_file = tmp_path / "gravity.grv"
    gravity_file.write_text("gravity", encoding="utf-8")
    model_dir = tmp_path / "model"
    (model_dir / "checkpoints").mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (model_dir / "checkpoints" / "ckpt_best.pt").write_bytes(b"checkpoint")

    config = _config(tmp_path, gravity_file, model_dir)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    resolved_hash = sha256_payload(config)
    manifest = build_benchmark_manifest(
        config=config,
        config_path=config_path,
        resolved_config_sha256=resolved_hash,
        output_dir=tmp_path,
        cwd=tmp_path,
    )

    assert manifest["benchmark_name"] == "prov_fixture"
    assert manifest["config"]["sha256"] == sha256_file(config_path)
    assert manifest["config"]["resolved_config_sha256"] == resolved_hash
    assert manifest["models"]["truth"]["gravity_file"]["sha256"] == sha256_file(gravity_file)
    assert manifest["models"]["surrogate"]["checkpoint"]["sha256"] == sha256_file(model_dir / "checkpoints" / "ckpt_best.pt")
    assert manifest["environment"]["python_version"]
