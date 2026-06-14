"""Small, Qt-independent HDF5 metadata inspection surface for ST-LRPS Studio."""

from __future__ import annotations

import json
from typing import Any

try:
    import h5py
except ImportError:  # pragma: no cover - depends on optional installation
    h5py = None  # type: ignore[assignment]

HAS_H5PY = h5py is not None


def inspect_h5_metadata(path: str) -> dict[str, Any] | None:
    """Read dataset metadata without loading the full HDF5 payload."""

    if h5py is None:
        return None
    try:
        with h5py.File(path, "r") as handle:
            info: dict[str, Any] = {"attrs": {}}

            for key in handle.attrs:
                value = handle.attrs[key]
                if hasattr(value, "item"):
                    value = value.item()
                elif isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                info["attrs"][key] = value

            dataset = None
            dataset_name = ""
            for name in ("data", "dataset", "train"):
                if name in handle:
                    dataset = handle[name]
                    dataset_name = name
                    break
            if dataset is None:
                for name in handle:
                    candidate = handle[name]
                    if isinstance(candidate, h5py.Dataset) and len(candidate.shape) >= 2:
                        dataset = candidate
                        dataset_name = name
                        break

            if dataset is not None:
                info["dataset_name"] = dataset_name
                info["rows"] = dataset.shape[0]
                info["cols"] = dataset.shape[1] if len(dataset.shape) > 1 else 1
                info["dtype"] = str(dataset.dtype)
                info["shape"] = list(dataset.shape)
                for key in dataset.attrs:
                    value = dataset.attrs[key]
                    if hasattr(value, "item"):
                        value = value.item()
                    elif isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    info["attrs"][key] = value

            attributes = json.dumps(info["attrs"]).lower()
            if "si" in attributes or "meter" in attributes or "m/s" in attributes:
                info["is_si"] = True
            elif "canonical" in attributes or "dimensionless" in attributes:
                info["is_si"] = False
            else:
                info["is_si"] = None

            if "columns" in info["attrs"]:
                info["col_names"] = info["attrs"]["columns"]
            elif "column_names" in info["attrs"]:
                info["col_names"] = info["attrs"]["column_names"]

            return info
    except Exception:
        return None


__all__ = ["HAS_H5PY", "inspect_h5_metadata"]
