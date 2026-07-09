from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import importlib


def test_surrogate_runtime_package_exports_public_api() -> None:
    adapter = importlib.import_module("lunaris.surrogate.runtime.adapter")
    package = importlib.import_module("lunaris.surrogate.runtime")

    assert package.SurrogateGravityModel is adapter.SurrogateGravityModel


def test_surrogate_runtime_responsibility_modules_import() -> None:
    artifact = importlib.import_module("lunaris.surrogate.runtime.artifact")
    metadata = importlib.import_module("lunaris.surrogate.runtime.metadata")
    networks = importlib.import_module("lunaris.surrogate.runtime.networks")
    scalers = importlib.import_module("lunaris.surrogate.runtime.scalers")
    device = importlib.import_module("lunaris.surrogate.runtime.device")

    assert callable(artifact.find_latest_st_lrps_model_dir)
    assert metadata._extract_degree_metadata(
        {"dataset_meta": {"degree_min": 10, "requested_degree": 50}}
    ) == (10, 50)
    assert callable(networks._build_model_from_config)
    assert hasattr(scalers, "_ScalerBundle")
    assert callable(device._require_torch)
