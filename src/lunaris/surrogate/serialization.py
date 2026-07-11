"""Trust-gated deserialization for torch ``.pt`` artifacts.

Untrusted ``.pt`` files are a code-execution boundary: ``torch.load`` with
``weights_only=False`` runs the full pickle machinery, which executes code
embedded in the file. Every Lunaris loader therefore goes through
:func:`safe_torch_load`, which only ever falls back to full unpickling when
the caller explicitly opted in (``trust_artifact=True``) or the operator set
``LUNARIS_TRUST_ARTIFACT=1`` in the environment. Without that opt-in a payload
that cannot be loaded tensor-only raises :class:`UntrustedArtifactError`
instead of silently unpickling.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import pickle
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

TRUST_ARTIFACT_ENV_VAR = "LUNARIS_TRUST_ARTIFACT"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class UntrustedArtifactError(RuntimeError):
    """A ``.pt`` payload needs full unpickling but no trust opt-in was given."""


def artifact_trust_from_env() -> bool:
    """Return True when ``LUNARIS_TRUST_ARTIFACT`` enables full unpickling."""

    return os.environ.get(TRUST_ARTIFACT_ENV_VAR, "").strip().lower() in _TRUE_VALUES


@lru_cache(maxsize=1)
def _default_safe_globals() -> tuple[Any, ...]:
    """Data-container globals every pipeline checkpoint may legitimately hold.

    Checkpoints written by this pipeline carry numpy payloads (e.g. the
    ``rng_state`` block stores numpy RNG state as ``ndarray``s). Reconstructing
    these classes cannot execute code — they are pure data containers — so
    allowlisting them keeps the tensor-only pass working without weakening the
    trust boundary.
    """

    out: list[Any] = []
    try:
        import numpy as np
    except ImportError:
        return ()
    # numpy 2 moved the array-reconstruction helper from numpy.core to numpy._core.
    for mod_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        reconstruct = getattr(mod, "_reconstruct", None)
        if reconstruct is not None and reconstruct not in out:
            out.append(reconstruct)
    out.extend((np.ndarray, np.dtype))
    try:
        dtypes_mod = importlib.import_module("numpy.dtypes")
    except ImportError:
        pass
    else:
        out.extend(cls for cls in vars(dtypes_mod).values() if isinstance(cls, type))
    return tuple(out)


def _refusal_message(path: Path, reason: str) -> str:
    return (
        f"Refusing to fully unpickle {path}: {reason}. Loading this file with "
        "weights_only=False executes code embedded in it. If the artifact comes "
        "from a source you trust, opt in explicitly with trust_artifact=True "
        f"(CLI: --trust-artifact) or by setting {TRUST_ARTIFACT_ENV_VAR}=1."
    )


def safe_torch_load(
    path: str | Path,
    *,
    map_location: Any,
    trust_artifact: bool = False,
    safe_globals: tuple[type, ...] = (),
) -> Any:
    """Load a ``.pt`` file tensor-only; full unpickling requires a trust opt-in.

    The ``weights_only=True`` loader is always tried first — every artifact
    written by this pipeline (tensors + JSON-safe metadata) loads through it.
    ``safe_globals`` may allowlist additional benign classes for that pass
    (e.g. ``TorchVersion``). When tensor-only loading is impossible (legacy
    payloads with arbitrary pickled objects, or a torch too old to know
    ``weights_only``), the full unpickle only runs if ``trust_artifact`` is
    True or ``LUNARIS_TRUST_ARTIFACT=1``; otherwise
    :class:`UntrustedArtifactError` is raised.
    """

    import torch

    path = Path(path)
    trusted = bool(trust_artifact) or artifact_trust_from_env()
    allowlist = [*_default_safe_globals(), *safe_globals]
    try:
        if allowlist and hasattr(torch.serialization, "safe_globals"):
            ctx = torch.serialization.safe_globals(allowlist)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch too old to know the weights_only kwarg: any load is a full unpickle.
        if not trusted:
            raise UntrustedArtifactError(
                _refusal_message(
                    path, "this PyTorch version has no weights_only (tensor-only) loader"
                )
            ) from None
        return torch.load(path, map_location=map_location)
    except (pickle.UnpicklingError, RuntimeError) as safe_load_error:
        if not trusted:
            raise UntrustedArtifactError(
                _refusal_message(
                    path, f"safe (weights_only=True) load failed ({safe_load_error})"
                )
            ) from safe_load_error
        warnings.warn(
            f"Safe (weights_only=True) load of {path} failed; falling back to "
            "full unpickling because the artifact was explicitly trusted. This "
            "executes code embedded in the checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


__all__ = [
    "TRUST_ARTIFACT_ENV_VAR",
    "UntrustedArtifactError",
    "artifact_trust_from_env",
    "safe_torch_load",
]
