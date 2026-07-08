"""Disk and memory archive IO for batch propagation results."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.batch.memory_policy import (
    _HOST_MEMORY_SAFETY_FACTOR,
    _available_host_memory_bytes,
)
from lunaris.batch.provenance import (
    _decode_archive_metadata,
    _decode_metadata_value,
    _metadata_value_to_jsonable,
)
from lunaris.common.batch_defs import BatchPropagationConfig, BatchPropagationResult
from lunaris.common.contracts.batch_archive import (
    BATCH_ARCHIVE_SCHEMA_VERSION,
    REQUIRED_ARCHIVE_V2_ARRAYS,
    REQUIRED_ARCHIVE_V2_FIELDS,
)


def _resolve_result_storage(
    batch_cfg: BatchPropagationConfig,
    n_steps: int,
    *,
    available_host_memory_bytes: Callable[[], int | None] | None = None,
) -> tuple[str, int, int]:
    """Resolve eager versus disk-backed result storage before opening a writer."""
    result_bytes = batch_cfg.estimated_result_bytes(n_steps)
    memory_limit_bytes = int(float(batch_cfg.max_result_memory_gb) * (1024.0 ** 3))
    # Effective host budget: the configured cap, further bounded by a safety
    # fraction of the RAM actually free now (when measurable). This is the budget
    # the auto storage decision and the per-batch host buffer must both respect.
    memory_probe = available_host_memory_bytes or _available_host_memory_bytes
    available = memory_probe()
    host_budget_bytes = memory_limit_bytes
    if available is not None:
        host_budget_bytes = min(memory_limit_bytes, int(available * _HOST_MEMORY_SAFETY_FACTOR))
    storage_mode = str(batch_cfg.result_storage_mode)
    if storage_mode == "auto":
        storage_mode = (
            "disk"
            if batch_cfg.output_format == "hdf5" and result_bytes > host_budget_bytes
            else "memory"
        )
    if storage_mode not in {"memory", "disk"}:
        raise ValueError(f"Unsupported result storage mode: {storage_mode!r}")
    if (
        batch_cfg.output_format == "npz"
        and result_bytes > host_budget_bytes
        and batch_cfg.result_storage_mode == "auto"
    ):
        limit_gib = host_budget_bytes / (1024.0 ** 3)
        budget_note = (
            f"host memory safety budget ({limit_gib:.2f} GiB = "
            f"{_HOST_MEMORY_SAFETY_FACTOR:.0%} of free RAM)"
            if available is not None and host_budget_bytes < memory_limit_bytes
            else f"max_result_memory_gb={batch_cfg.max_result_memory_gb:g}"
        )
        raise MemoryError(
            "Estimated eager batch/ensemble trajectory size "
            f"({result_bytes / (1024.0 ** 3):.2f} GiB) exceeds {budget_note}. "
            "Use HDF5 output for disk-backed streaming or explicitly choose "
            "result_storage_mode='memory'."
        )
    if (
        storage_mode == "memory"
        and available is not None
        and result_bytes > int(available * _HOST_MEMORY_SAFETY_FACTOR)
    ):
        warnings.warn(
            "Eager batch/ensemble result "
            f"({result_bytes / (1024.0 ** 3):.2f} GiB) exceeds the host memory "
            f"safety budget ({_HOST_MEMORY_SAFETY_FACTOR:.0%} of "
            f"{available / (1024.0 ** 3):.2f} GiB free RAM); the run may exhaust "
            "host memory. Prefer HDF5 disk-backed output for large ensembles.",
            RuntimeWarning,
            stacklevel=2,
        )
    return storage_mode, result_bytes, host_budget_bytes


def _allocate_result_buffer(
    storage_mode: str,
    writer_buffer: Any,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Allocate the full ensemble only for the explicit eager result path."""
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


class HDF5TrajectoryView:
    """Read-only, path-backed view of an HDF5 trajectory dataset."""

    _lunaris_lazy_trajectory = True

    def __init__(self, path: str | Path, dataset: str = "Y") -> None:
        self.path = Path(path).expanduser().resolve()
        self.dataset = str(dataset)
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for lazy HDF5 batch results.") from None
        with h5py.File(str(self.path), "r") as f:
            ds = f[self.dataset]
            self.shape = tuple(int(v) for v in ds.shape)
            self.ndim = int(ds.ndim)
            self.dtype = np.dtype(ds.dtype)

    def __getitem__(self, key: Any) -> np.ndarray:
        import h5py

        with h5py.File(str(self.path), "r") as f:
            return np.asarray(f[self.dataset][key], dtype=np.float64)

    def iter_epoch_sample_blocks(
        self,
        sample_indices: np.ndarray,
    ) -> Iterator[np.ndarray]:
        """Yield one ``(n_samples, 6)`` block per epoch using one file handle."""
        import h5py

        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("sample_indices must be one-dimensional")
        if indices.size > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError("sample_indices must be strictly increasing")

        with h5py.File(str(self.path), "r") as f:
            dataset = f[self.dataset]
            for epoch in range(self.shape[0]):
                yield np.asarray(dataset[epoch, indices, :], dtype=np.float64)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        import h5py

        with h5py.File(str(self.path), "r") as f:
            arr = np.asarray(f[self.dataset], dtype=dtype or np.float64)
        if copy:
            return arr.copy()
        return arr


class _HDF5Writer:
    """Atomic HDF5 writer with sample-axis batch streaming."""

    def __init__(
        self,
        path: Path,
        n_samples: int,
        t_grid: np.ndarray,
        n_state: int = 6,
    ) -> None:
        try:
            import h5py
            self._h5py = h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 output. "
                "Install via:  pip install h5py"
            ) from None

        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._part_path = Path(f"{path}.part")
        if self._part_path.exists():
            self._part_path.unlink()
        self._f = h5py.File(str(self._part_path), "w")
        self._n = n_samples
        self._s = n_state
        self._t = np.ascontiguousarray(t_grid, dtype=np.float64)
        self._next_sample = 0
        self._final_payload_written = False
        self._ds_t = self._f.create_dataset("t", data=self._t)
        self._ds_Y = self._f.create_dataset(
            "Y", shape=(len(self._t), n_samples, n_state),
            dtype=np.float64,
            chunks=(min(len(self._t), 16), min(n_samples, 128), n_state),
            compression="lzf",
        )

    def write_sample_batch(self, start: int, end: int, Y: np.ndarray) -> None:
        """Write ``Y[:, start:end, :]`` without retaining the full ensemble."""
        start = int(start)
        end = int(end)
        if start != self._next_sample:
            raise ValueError(
                "HDF5 sample batches must be contiguous and ordered: "
                f"expected start={self._next_sample}, got {start}"
            )
        if end <= start or end > self._n:
            raise ValueError(
                f"HDF5 sample batch [{start}:{end}] is outside [0:{self._n}]"
            )
        expected = (len(self._t), int(end - start), self._s)
        if tuple(Y.shape) != expected:
            raise ValueError(f"HDF5 batch must have shape {expected}, got {Y.shape}")
        self._ds_Y[:, start:end, :] = Y
        self._next_sample = end

    @property
    def memory_buffer(self) -> None:
        return None

    def write_metadata(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            try:
                payload = _metadata_value_to_jsonable(v)
                if payload is None:
                    continue
                if isinstance(payload, dict | list):
                    payload = json.dumps(payload, sort_keys=True)
                self._f.attrs[k] = payload
            except Exception as exc:
                raise ValueError(
                    f"Could not write HDF5 metadata field {k!r}"
                ) from exc

    def write_final(
        self,
        sc_samples: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
        valid_mask: np.ndarray,
        impact_position_inertial_m: np.ndarray,
        impact_position_fixed_m: np.ndarray,
    ) -> None:
        expected_shapes = {
            "sc_samples": (self._n, 4),
            "impact_flags": (self._n,),
            "t_impact": (self._n,),
            "valid_mask": (self._n,),
            "impact_position_inertial_m": (self._n, 3),
            "impact_position_fixed_m": (self._n, 3),
        }
        values = {
            "sc_samples": sc_samples,
            "impact_flags": impact_flags,
            "t_impact": t_impact,
            "valid_mask": valid_mask,
            "impact_position_inertial_m": impact_position_inertial_m,
            "impact_position_fixed_m": impact_position_fixed_m,
        }
        for name, expected in expected_shapes.items():
            actual = tuple(np.shape(values[name]))
            if actual != expected:
                raise ValueError(
                    f"HDF5 final payload {name!r} must have shape "
                    f"{expected}, got {actual}"
                )
        self._f.create_dataset("sc_samples",  data=sc_samples)
        self._f.create_dataset("impact_flags", data=np.asarray(impact_flags, dtype=np.uint8))
        self._f.create_dataset("t_impact",    data=t_impact)
        self._f.create_dataset("valid_mask", data=np.asarray(valid_mask, dtype=np.uint8))
        self._f.create_dataset(
            "impact_position_inertial_m", data=impact_position_inertial_m
        )
        self._f.create_dataset("impact_position_fixed_m", data=impact_position_fixed_m)
        self._final_payload_written = True

    def finalize(self) -> None:
        if self._next_sample != self._n:
            raise RuntimeError(
                "Cannot finalize incomplete HDF5 trajectory stream: "
                f"wrote samples [0:{self._next_sample}], expected [0:{self._n}]"
            )
        if not self._final_payload_written:
            raise RuntimeError(
                "Cannot finalize HDF5 archive before final result arrays are written"
            )
        self._f.attrs["archive_schema_version"] = BATCH_ARCHIVE_SCHEMA_VERSION
        self._f.flush()
        self._f.close()
        self._part_path.replace(self._path)

    def abort(self) -> None:
        # R29b-justified: abort() runs on the failure path; best-effort cleanup
        # of the .part file must never mask the original exception being raised.
        try:
            self._f.close()
        except Exception:
            pass
        try:
            self._part_path.unlink(missing_ok=True)
        except Exception:
            pass


class _SummaryOnlyWriter:
    """R23 screening writer: a no-op — summary mode never archives trajectories.

    The batch engine reduces every sub-batch to the versioned screening
    summary (``lunaris.batch.summary``) and keeps only the top-K full
    histories in the in-memory result; there is deliberately no ``(T, N, 6)``
    archive to write.
    """

    memory_buffer = None

    def write_sample_batch(self, *args: Any, **kwargs: Any) -> None:
        return None

    def write_metadata(self, **kwargs: Any) -> None:
        return None

    def write_final(self, *args: Any) -> None:
        return None

    def finalize(self) -> None:
        return None

    def abort(self) -> None:
        return None


class _NPZWriter:
    """Writes an eager trajectory archive through an atomic ``.part.npz`` file."""

    def __init__(
        self,
        path: Path,
        n_samples: int,
        t_grid: np.ndarray,
        n_state: int = 6,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._part_path = path.with_name(f"{path.stem}.part{path.suffix or '.npz'}")
        if self._part_path.exists():
            self._part_path.unlink()
        self._t = np.ascontiguousarray(t_grid, dtype=np.float64)
        self._Y = np.empty((len(self._t), n_samples, n_state), dtype=np.float64)
        self._metadata: dict[str, Any] = {}
        self._final_payload_written = False

    def write_sample_batch(self, start: int, end: int, Y: np.ndarray) -> None:
        self._Y[:, start:end, :] = Y

    @property
    def memory_buffer(self) -> np.ndarray:
        return self._Y

    def write_metadata(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self._metadata[str(key)] = _metadata_value_to_jsonable(value)

    def write_final(
        self,
        sc_samples: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
        valid_mask: np.ndarray,
        impact_position_inertial_m: np.ndarray,
        impact_position_fixed_m: np.ndarray,
    ) -> None:
        np.savez_compressed(
            str(self._part_path),
            t=self._t,
            Y=self._Y,
            sc_samples=sc_samples,
            impact_flags=np.asarray(impact_flags, dtype=np.uint8),
            t_impact=t_impact,
            valid_mask=np.asarray(valid_mask, dtype=np.uint8),
            impact_position_inertial_m=impact_position_inertial_m,
            impact_position_fixed_m=impact_position_fixed_m,
            metadata_json=np.asarray(json.dumps(self._metadata, sort_keys=True), dtype=np.str_),
        )
        self._final_payload_written = True

    def finalize(self) -> None:
        if not self._final_payload_written:
            raise RuntimeError(
                "Cannot finalize NPZ archive before final result arrays are written"
            )
        self._part_path.replace(self._path)

    def abort(self) -> None:
        try:
            self._part_path.unlink(missing_ok=True)
        except Exception:
            pass


def _make_writer(
    batch_cfg: BatchPropagationConfig,
    n_samples: int,
    t_grid: np.ndarray,
) -> Any:
    """Factory: return the appropriate writer based on output_format."""
    p = batch_cfg.output_path_resolved
    if batch_cfg.output_format == "hdf5":
        return _HDF5Writer(p, n_samples, t_grid)
    return _NPZWriter(p, n_samples, t_grid)


def _validate_archive_v2_manifest(metadata: dict[str, Any]) -> None:
    """Enforce the required schema-v2 manifest for batch archives."""
    raw_version = metadata.get("archive_schema_version")
    if raw_version is None:
        raise ValueError(
            "Batch archive is missing archive_schema_version. Current Lunaris "
            "requires schema v2 archives produced by BatchPropagationEngine; "
            "regenerate the run with the current batch pipeline."
        )
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        raise ValueError(
            f"Batch archive has invalid archive_schema_version={raw_version!r}. "
            "Current Lunaris requires schema v2 archives."
        ) from None
    if version < BATCH_ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported batch archive schema v{version}. Current Lunaris "
            f"requires schema v{BATCH_ARCHIVE_SCHEMA_VERSION} archives; regenerate the run."
        )
    if version > BATCH_ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported batch archive schema v{version}. This loader expects "
            f"schema v{BATCH_ARCHIVE_SCHEMA_VERSION}; upgrade the loader or regenerate the archive."
        )
    missing = [f for f in REQUIRED_ARCHIVE_V2_FIELDS if f not in metadata]
    if missing:
        raise ValueError(
            f"Batch archive declares schema v{version} but is missing required "
            f"manifest field(s): {', '.join(sorted(missing))}. The archive is incomplete "
            "or was not produced by a current BatchPropagationEngine run."
        )


def _require_archive_v2_arrays(names: Any, *, format_label: str) -> None:
    available = {str(name) for name in names}
    missing = [name for name in REQUIRED_ARCHIVE_V2_ARRAYS if name not in available]
    if missing:
        raise ValueError(
            f"Batch {format_label} archive is missing required dataset(s): "
            f"{', '.join(sorted(missing))}. Regenerate the run with the current "
            "BatchPropagationEngine writer."
        )


def load_batch_result(path: str, *, lazy: bool = False, strict: bool = True) -> BatchPropagationResult:
    """
    Reload a saved batch/ensemble result from HDF5 or NPZ file.

    Parameters
    ----------
    path : str
        Path produced by ``BatchPropagationEngine.run()``.
    lazy : bool
        Return a disk-backed, read-only trajectory view instead of loading the
        full ``Y`` ensemble into memory (HDF5 only).
    strict : bool
        Deprecated. ``False`` no longer relaxes validation; the loader always
        requires a complete schema-v2 manifest and the current result datasets.

    Returns
    ----------
    BatchPropagationResult
    """
    p = Path(path).expanduser().resolve()
    suffix = p.suffix.lower()
    if strict is not True:
        warnings.warn(
            "load_batch_result(strict=False) is deprecated and no longer relaxes "
            "schema-v2 archive validation.",
            DeprecationWarning,
            stacklevel=2,
        )

    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required to read HDF5 batch output.") from None
        with h5py.File(str(p), "r") as f:
            diagnostics = {
                str(key): _decode_metadata_value(value)
                for key, value in dict(f.attrs).items()
            }
            _validate_archive_v2_manifest(diagnostics)
            _require_archive_v2_arrays(f.keys(), format_label="HDF5")
            t_arr  = np.asarray(f["t"],           dtype=np.float64)
            Y_arr: Any = (
                HDF5TrajectoryView(p)
                if lazy
                else np.asarray(f["Y"], dtype=np.float64)
            )
            sc     = np.asarray(f["sc_samples"],  dtype=np.float64)
            imask  = np.asarray(f["impact_flags"], dtype=np.float64)
            t_imp  = np.asarray(f["t_impact"],    dtype=np.float64)
            valid = np.asarray(f["valid_mask"], dtype=np.float64)
            impact_i = np.asarray(f["impact_position_inertial_m"], dtype=np.float64)
            impact_f = np.asarray(f["impact_position_fixed_m"], dtype=np.float64)
        return BatchPropagationResult(
            t=t_arr, Y=Y_arr, sc_samples=sc,
            impact_mask=imask, t_impact=t_imp,
            valid_mask=valid,
            impact_position_inertial_m=impact_i,
            impact_position_fixed_m=impact_f,
            archive_path=str(p),
            diagnostics=diagnostics,
        )

    if suffix == ".npz":
        with np.load(str(p), allow_pickle=False) as data:
            diagnostics = {}
            if "metadata_json" in data.files:
                diagnostics = _decode_archive_metadata(data["metadata_json"])
            _validate_archive_v2_manifest(diagnostics)
            _require_archive_v2_arrays(data.files, format_label="NPZ")
            return BatchPropagationResult(
                t=data["t"],
                Y=data["Y"],
                sc_samples=data["sc_samples"],
                impact_mask=data["impact_flags"],
                t_impact=data["t_impact"],
                valid_mask=data["valid_mask"],
                impact_position_inertial_m=data["impact_position_inertial_m"],
                impact_position_fixed_m=data["impact_position_fixed_m"],
                archive_path=str(p),
                diagnostics=diagnostics,
            )

    raise ValueError(f"Unrecognised batch output format: {suffix!r} (expected .h5 or .npz)")


__all__ = [
    "BATCH_ARCHIVE_SCHEMA_VERSION",
    "REQUIRED_ARCHIVE_V2_FIELDS",
    "REQUIRED_ARCHIVE_V2_ARRAYS",
    "HDF5TrajectoryView",
    "_HDF5Writer",
    "_NPZWriter",
    "_make_writer",
    "_resolve_result_storage",
    "_allocate_result_buffer",
    "_validate_archive_v2_manifest",
    "load_batch_result",
]
