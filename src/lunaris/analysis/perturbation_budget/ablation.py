"""Optional propagation-level ablation hooks for future expansion."""

from __future__ import annotations


def propagation_ablation_not_run() -> list[dict[str, str]]:
    """Return an explicit placeholder row for MVP reports."""
    return [
        {
            "status": "not_run",
            "reason": "Propagation ablation is optional and not executed by the MVP.",
        }
    ]
