# lunaris.core.monte_carlo_engine
"""Compatibility shim for the historical Monte Carlo engine import path.

The canonical implementation now lives under ``lunaris.batch``. Console
entry points and downstream imports continue to resolve through this module.

The fold is lazy (PEP 562 module ``__getattr__``): importing this module must
not pull the batch orchestration layer into ``core`` at import time. Names
resolve on first access and are cached in the module globals, so
``from ... import X``, ``hasattr``, and monkeypatch paths behave exactly as
they did with the old eager fold. The remaining call-time edges to
``lunaris.batch`` are declared in the import-linter contract's
``ignore_imports``.
"""

from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module
from typing import Any

import numpy as np

# Canonical homes of the public batch surface re-exported through this shim.
_PUBLIC_EXPORT_HOMES: dict[str, str] = {
    "BatchPropagationEngine": "lunaris.batch.engine",
    "MonteCarloEngine": "lunaris.batch.engine",
    "batch_entry": "lunaris.batch.engine",
    "mc_entry": "lunaris.batch.engine",
    "generate_standard_normal_design": "lunaris.batch.sampling",
    "sample_initial_states": "lunaris.batch.sampling",
    "sample_spacecraft_props": "lunaris.batch.sampling",
    "HDF5TrajectoryView": "lunaris.batch.storage",
    "load_mc_result": "lunaris.batch.storage",
    "BatchPropagationConfig": "lunaris.batch.types",
    "BatchPropagationResult": "lunaris.batch.types",
}

_COMPAT_PRIVATE_EXPORT_NAMES = (
    "_BACKEND_DISPLAY_NAMES",
    "REQUIRED_ARCHIVE_V2_FIELDS",
    "_HOST_MEMORY_SAFETY_FACTOR",
    "_available_host_memory_bytes",
    "_active_physics_capabilities",
    "_decode_archive_metadata",
    "_decode_metadata_value",
    "_metadata_value_to_jsonable",
    "_sha256_file",
    "_build_ephemeris_manager",
    "_impact_positions_fixed",
    "_need_body_vectors",
    "_need_ephemeris",
    "_state_to_array",
    "_surface_topography_requested",
    "_sobol_size_note",
    "_HDF5Writer",
    "_infer_valid_mask_from_dataset",
    "_make_writer",
    "_NPZWriter",
    "_validate_archive_v2_manifest",
)

# Search order for the legacy private names above, in their canonical batch
# homes. Resolution is dynamic (importlib + getattr, not ``from X import _Y``)
# so import-pruning linters (``ruff --fix``) cannot strip these re-exports.
_BATCH_SUBMODULES = (
    "lunaris.batch.engine",
    "lunaris.batch.storage",
    "lunaris.batch.sampling",
    "lunaris.batch.provenance",
    "lunaris.batch.requirements",
    "lunaris.batch.memory_policy",
    "lunaris.batch.backend_policy",
)


def _resolve_result_storage(mc_cfg: Any, n_steps: int) -> tuple[str, int, int]:
    """Compatibility proxy preserving monkeypatches on this legacy module."""
    from lunaris.batch import storage as _storage

    # Look the memory probe up through THIS module so a monkeypatch of
    # ``lunaris.core.monte_carlo_engine._available_host_memory_bytes`` keeps
    # working exactly as it did with the eager fold.
    # noqa rationale: plain attribute access would also trigger the module
    # __getattr__, but getattr keeps mypy quiet about ModuleType attributes.
    mem_probe = getattr(_sys.modules[__name__], "_available_host_memory_bytes")  # noqa: B009
    return _storage._resolve_result_storage(
        mc_cfg,
        n_steps,
        available_host_memory_bytes=mem_probe,
    )


def _allocate_result_buffer(
    storage_mode: str,
    writer_buffer: Any,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Compatibility proxy preserving monkeypatches on this legacy module."""
    if storage_mode == "disk":
        return None
    if storage_mode != "memory":
        raise ValueError(f"Unsupported result storage mode: {storage_mode!r}")
    if isinstance(writer_buffer, np.ndarray):
        if tuple(writer_buffer.shape) != tuple(shape):
            raise ValueError(
                f"Writer result buffer must have shape {shape}, "
                f"got {writer_buffer.shape}"
            )
        return writer_buffer
    return np.empty(shape, dtype=np.float64)


def _resolve_compat_export(name: str) -> Any:
    home = _PUBLIC_EXPORT_HOMES.get(name)
    if home is not None:
        return getattr(_import_module(home), name)
    if name in _COMPAT_PRIVATE_EXPORT_NAMES:
        for modname in _BATCH_SUBMODULES:
            sub = _import_module(modname)
            if hasattr(sub, name):
                return getattr(sub, name)
        raise ImportError(  # pragma: no cover - guards future renames in lunaris.batch
            "lunaris.core.monte_carlo_engine compatibility shim could not resolve: "
            + name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:
    value = _resolve_compat_export(name)
    globals()[name] = value  # cache: snapshot semantics + stable monkeypatch target
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# Derived from the lazy-fold tables: exactly the names resolvable via the
# PEP 562 __getattr__ above (public batch surface + legacy private exports).
__all__ = sorted(set(_PUBLIC_EXPORT_HOMES) | set(_COMPAT_PRIVATE_EXPORT_NAMES))
