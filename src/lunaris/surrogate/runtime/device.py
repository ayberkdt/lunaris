"""Optional PyTorch import guard for surrogate runtime inference."""

from __future__ import annotations

from typing import Any

torch: Any | None
nn: Any | None
_TORCH_IMPORT_ERROR: Exception | None
try:
    import torch as _torch
    import torch.nn as _nn
except Exception as exc:  # pragma: no cover - exercised only on machines without torch
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    torch = _torch
    nn = _nn
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> tuple[Any, Any]:
    """Return the torch modules required for inference, or fail with install guidance."""

    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for ST-LRPS surrogate inference. "
            "Install the optional ML stack with `pip install lunaris[ml]` "
            "or `pip install lunaris[hpc]`."
        ) from _TORCH_IMPORT_ERROR
    return torch, nn


__all__ = ["torch", "nn", "_TORCH_IMPORT_ERROR", "_require_torch"]
