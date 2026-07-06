from __future__ import annotations

import importlib

PUBLIC_BATCH_SYMBOLS = (
    "BatchPropagationConfig",
    "BatchPropagationEngine",
    "BatchPropagationResult",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
    "HDF5TrajectoryView",
    "load_batch_result",
    "batch_entry",
)

PRIVATE_ENGINE_SYMBOLS = (
    "_BACKEND_DISPLAY_NAMES",
    "_st_lrps_kind_mismatch",
)


def test_public_batch_symbols_import_from_package() -> None:
    batch = importlib.import_module("lunaris.batch")

    for name in PUBLIC_BATCH_SYMBOLS:
        assert hasattr(batch, name), name


def test_private_engine_symbols_importable() -> None:
    engine = importlib.import_module("lunaris.batch.engine")

    for name in PRIVATE_ENGINE_SYMBOLS:
        assert hasattr(engine, name), name


def test_responsibility_modules_expose_expected_surfaces() -> None:
    modules = {
        "lunaris.batch.engine": ("BatchPropagationEngine", "batch_entry"),
        "lunaris.batch.sampling": (
            "generate_standard_normal_design",
            "sample_initial_states",
            "sample_spacecraft_props",
        ),
        "lunaris.batch.storage": ("HDF5TrajectoryView", "load_batch_result"),
        "lunaris.batch.memory_policy": ("_available_host_memory_bytes",),
        "lunaris.batch.provenance": ("_sha256_file", "_metadata_value_to_jsonable"),
        "lunaris.batch.requirements": ("_need_ephemeris", "_need_body_vectors"),
        "lunaris.batch.backend_policy": ("resolve_batch_backend_policy", "BatchBackendPlan"),
        "lunaris.batch.types": (
            "BatchPropagationConfig",
            "BatchPropagationResult",
        ),
        "lunaris.common.batch_defs": (
            "BatchPropagationConfig",
            "BatchPropagationResult",
        ),
    }

    for module_name, names in modules.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name}:{name}"


def test_batch_types_share_identity_with_batch_defs() -> None:
    batch_defs = importlib.import_module("lunaris.common.batch_defs")
    batch_types = importlib.import_module("lunaris.batch.types")
    batch = importlib.import_module("lunaris.batch")

    assert batch_types.BatchPropagationConfig is batch_defs.BatchPropagationConfig
    assert batch_types.BatchPropagationResult is batch_defs.BatchPropagationResult
    assert batch.BatchPropagationConfig is batch_defs.BatchPropagationConfig
    assert batch.BatchPropagationResult is batch_defs.BatchPropagationResult
