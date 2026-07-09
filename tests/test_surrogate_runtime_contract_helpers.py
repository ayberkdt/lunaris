from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import json
import os
from pathlib import Path

import numpy as np
import pytest

from lunaris.surrogate.runtime import artifact, device, metadata, scalers


def _run_dir(root: Path, name: str, *, checkpoint: str = "ckpt_last.pt") -> Path:
    run_dir = root / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / checkpoint).write_bytes(b"checkpoint")
    return run_dir


def test_artifact_discovery_filters_and_sorts_runs(tmp_path, monkeypatch) -> None:
    old = _run_dir(tmp_path, "old")
    new = _run_dir(tmp_path, "new", checkpoint="ckpt_best.pt")
    earth = _run_dir(tmp_path, "earth")
    (tmp_path / "incomplete").mkdir()

    os.utime(old, (10.0, 10.0))
    os.utime(new, (20.0, 20.0))
    os.utime(earth, (30.0, 30.0))
    monkeypatch.setattr(artifact, "_looks_like_lunar_run", lambda path: path.name != "earth")

    found = artifact.discover_st_lrps_model_dirs(tmp_path)

    assert found == [new.resolve(), old.resolve()]
    assert artifact.find_latest_st_lrps_model_dir(tmp_path) == new.resolve()
    assert artifact._is_valid_surrogate_run(new)
    assert artifact._find_checkpoint_for_run(new).name == "ckpt_best.pt"
    with pytest.raises(FileNotFoundError):
        artifact._find_checkpoint_for_run(tmp_path / "incomplete")


def test_metadata_degree_and_path_resolution(monkeypatch) -> None:
    assert metadata._extract_degree_metadata({"degree_min": 4, "degree_max": 12}) == (4, 12)
    assert metadata._extract_degree_metadata(
        {"dataset_meta": {"degree_min": 10, "requested_degree": 50}}
    ) == (10, 50)
    with pytest.raises(ValueError, match="degree metadata"):
        metadata._extract_degree_metadata({})

    nested = {"cloud_config_json": json.dumps({"gravity_model_path": "nested.gfc"})}
    assert metadata._config_path_value(nested, "gravity_model_path") == "nested.gfc"
    assert metadata._config_path_value({"cloud_config_json": "not-json"}, "missing") is None

    monkeypatch.setattr(
        metadata,
        "resolve_lunar_gravity_path",
        lambda value=None: Path("default.gfc") if value is None else Path(str(value)),
    )
    assert metadata._resolve_baseline_gravity_path({"gravity_file_path": "explicit.gfc"}) == Path(
        "explicit.gfc"
    )
    assert metadata._resolve_baseline_gravity_path({}) == Path("default.gfc")


def test_scaler_normalization_and_file_fallback(tmp_path) -> None:
    x = scalers._normalize_scale_mapping({"mean": [1, 2, 3], "scale": [10]}, 3, "x")
    assert x.is_isometric
    np.testing.assert_allclose(x.mean, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(x.scale, [10.0])

    u = scalers._normalize_scale_mapping({"mean": [0], "std": [2]}, 1, "u")
    np.testing.assert_allclose(u.scale, [2.0])

    with pytest.raises(ValueError, match="missing 'mean'"):
        scalers._normalize_scale_mapping({"scale": [1]}, 3, "x")
    with pytest.raises(ValueError, match="positive finite"):
        scalers._normalize_scale_mapping({"mean": [0, 0, 0], "scale": [0]}, 3, "x")

    checkpoint_bundle = scalers._load_scaler_bundle(
        tmp_path,
        {
            "scaler": {
                "x": {"mean": [0, 0, 0], "scale": [1]},
                "u": {"mean": [0], "scale": [2]},
                "a": {"mean": [0, 0, 0], "scale": [3, 3, 3]},
            }
        },
    )
    assert checkpoint_bundle.a is not None
    np.testing.assert_allclose(checkpoint_bundle.a.scale, [3.0, 3.0, 3.0])

    (tmp_path / "scaler.json").write_text(
        json.dumps(
            {
                "x": {"mean": [0, 0, 0], "scale": [1]},
                "u": {"mean": [0], "scale": [2]},
            }
        ),
        encoding="utf-8",
    )
    file_bundle = scalers._load_scaler_bundle(tmp_path, {})
    assert file_bundle.a is None
    np.testing.assert_allclose(file_bundle.u.scale, [2.0])

    with pytest.raises(FileNotFoundError):
        scalers._load_scaler_bundle(tmp_path / "missing", {})


def test_require_torch_reports_optional_dependency_guidance(monkeypatch) -> None:
    monkeypatch.setattr(device, "torch", None)
    monkeypatch.setattr(device, "nn", None)
    monkeypatch.setattr(device, "_TORCH_IMPORT_ERROR", ImportError("no torch"))

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        device._require_torch()
