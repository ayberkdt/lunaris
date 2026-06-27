from __future__ import annotations

import importlib

PUBLIC_BATCH_SYMBOLS = (
    "BatchPropagationConfig",
    "BatchPropagationEngine",
    "BatchPropagationResult",
    "MonteCarloEngine",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
    "HDF5TrajectoryView",
    "load_mc_result",
    "mc_entry",
    "batch_entry",
)

LEGACY_PRIVATE_SYMBOLS = (
    "_BACKEND_DISPLAY_NAMES",
    "REQUIRED_ARCHIVE_V2_FIELDS",
    "_HDF5Writer",
    "_NPZWriter",
    "_make_writer",
    "_resolve_result_storage",
    "_allocate_result_buffer",
    "_need_body_vectors",
    "_build_ephemeris_manager",
    "_impact_positions_fixed",
)


def test_public_batch_symbols_import_from_new_and_legacy_paths() -> None:
    legacy = importlib.import_module("lunaris.core.monte_carlo_engine")
    batch = importlib.import_module("lunaris.batch")

    for name in PUBLIC_BATCH_SYMBOLS:
        assert hasattr(batch, name), name
        assert hasattr(legacy, name), name


def test_legacy_private_symbols_remain_importable() -> None:
    legacy = importlib.import_module("lunaris.core.monte_carlo_engine")

    for name in LEGACY_PRIVATE_SYMBOLS:
        assert hasattr(legacy, name), name


def test_responsibility_modules_expose_expected_surfaces() -> None:
    modules = {
        "lunaris.batch.engine": ("BatchPropagationEngine", "MonteCarloEngine", "mc_entry", "batch_entry"),
        "lunaris.batch.sampling": (
            "generate_standard_normal_design",
            "sample_initial_states",
            "sample_spacecraft_props",
        ),
        "lunaris.batch.storage": ("HDF5TrajectoryView", "load_mc_result"),
        "lunaris.batch.memory_policy": ("_available_host_memory_bytes",),
        "lunaris.batch.provenance": ("_sha256_file", "_metadata_value_to_jsonable"),
        "lunaris.batch.requirements": ("_need_ephemeris", "_need_body_vectors"),
        "lunaris.batch.backend_policy": ("resolve_mc_backend_policy", "MCBackendPlan"),
        "lunaris.batch.types": (
            "BatchPropagationConfig",
            "BatchPropagationResult",
            "MonteCarloConfig",
            "MCRunResult",
        ),
        "lunaris.common.batch_defs": (
            "BatchPropagationConfig",
            "BatchPropagationResult",
            "MonteCarloConfig",
            "MCRunResult",
        ),
    }

    for module_name, names in modules.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name}:{name}"


def test_batch_type_aliases_keep_legacy_identity() -> None:
    batch_defs = importlib.import_module("lunaris.common.batch_defs")
    legacy_defs = importlib.import_module("lunaris.common.montecarlo_defs")
    batch = importlib.import_module("lunaris.batch")

    assert batch_defs.BatchPropagationConfig is batch_defs.MonteCarloConfig
    assert batch_defs.BatchPropagationResult is batch_defs.MCRunResult
    assert legacy_defs.MonteCarloConfig is batch_defs.MonteCarloConfig
    assert legacy_defs.MCRunResult is batch_defs.MCRunResult
    assert batch.BatchPropagationEngine is batch.MonteCarloEngine
