"""Host-memory probes and budget constants for batch storage policy."""

from __future__ import annotations

_HOST_MEMORY_SAFETY_FACTOR = 0.8


def _available_host_memory_bytes() -> int | None:
    """Bytes of host RAM available right now, or ``None`` if it cannot be measured.

    Uses psutil when present; degrades gracefully (returns ``None``) so the memory
    safety factor is a best-effort guard, never a hard dependency.
    """
    try:
        import psutil
    except Exception:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


__all__ = ["_HOST_MEMORY_SAFETY_FACTOR", "_available_host_memory_bytes"]
