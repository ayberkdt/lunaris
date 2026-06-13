# -*- coding: utf-8 -*-
"""
Shared pre-flight validation service
====================================

Flow-agnostic checks that run **before** a propagation/MC/benchmark/training job
starts, so invalid combinations are caught early instead of failing mid-run or —
worse — silently dropping physics. This module has no UI/Qt dependency so the
CLI, the desktop UI, the Monte Carlo engine, the benchmark runner, and the
training/evaluation pipelines can all share one source of truth.

Phase 2 §4/§5 requirements served here:

* invalid backend/degree/force-model combinations are reported before the run
  (``Geçersiz kombinasyon çalışma başlamadan yakalanmalıdır``);
* no silent SH-degree clipping (``Sessiz derece kırpılması yapılmamalıdır``);
* no silent force-model disabling (``Sessiz force-model kapatma yapılmamalıdır``);
* the user-facing message and the technical log line are kept separate.

The backend checks delegate to the central capability registry
(:mod:`lunaris.core.backend_capabilities`) so the UI and CLI reach the *same*
decision. This service is additive: it does not change how any existing flow
runs until that flow opts in to calling it.

Broader §5 checks (dataset contract, artifact/frame/unit validation, checkpoint
output dimension, memory estimate, legacy-artifact permission, paper-safe mode)
attach as additional ``check_*`` functions following the same pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from lunaris.core.backend_capabilities import (
    get_capabilities,
    unsupported_force_models,
)

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class PreflightIssue:
    """One pre-flight finding.

    ``message`` is the user-facing sentence; ``detail`` is the technical log line
    (``key=value`` form) for ``run.log`` / artifact metadata. ``code`` is a stable
    machine identifier so callers can branch (e.g. resolve a fallback) without
    string-matching the message.
    """

    severity: str
    code: str
    message: str
    detail: str = ""


@dataclass
class PreflightReport:
    """Aggregated pre-flight result.

    ``ok`` is ``True`` only when there are no ``error``-severity issues; warnings
    and info notes do not block a run but should be surfaced and recorded.
    """

    issues: List[PreflightIssue] = field(default_factory=list)

    def add(self, issues: "PreflightIssue | List[PreflightIssue]") -> None:
        if isinstance(issues, PreflightIssue):
            self.issues.append(issues)
        else:
            self.issues.extend(issues)

    @property
    def errors(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def infos(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.severity == INFO]

    @property
    def ok(self) -> bool:
        return not self.errors

    def user_messages(self) -> List[str]:
        """User-facing lines, errors first."""
        order = {ERROR: 0, WARNING: 1, INFO: 2}
        return [i.message for i in sorted(self.issues, key=lambda x: order.get(x.severity, 9))]

    def technical_log(self) -> List[str]:
        """``severity code | detail`` lines for run.log / metadata."""
        return [f"{i.severity} {i.code} | {i.detail or i.message}" for i in self.issues]


# =============================================================================
# Individual checks
# =============================================================================


def check_backend_capability(
    *,
    requested_backend: str,
    requested_sh_degree: int = 0,
    flags: Any = None,
) -> List[PreflightIssue]:
    """Validate a concrete backend request against the capability registry.

    Returns errors when the named backend is unknown, when the requested SH
    degree exceeds the backend's on-device limit (reported, never silently
    clipped), or when an active force model is unsupported by the backend
    (reported, never silently dropped). For an explicit backend request these
    are errors; an ``auto`` resolver may instead consult these results to choose
    an explicit, recorded fallback.
    """
    issues: List[PreflightIssue] = []

    try:
        caps = get_capabilities(requested_backend)
    except KeyError as exc:
        return [
            PreflightIssue(
                ERROR,
                "backend_unknown",
                f"Unknown backend '{requested_backend}'.",
                str(exc),
            )
        ]

    degree = int(requested_sh_degree or 0)
    if caps.max_sh_degree is not None and degree > caps.max_sh_degree:
        issues.append(
            PreflightIssue(
                ERROR,
                "sh_degree_exceeds_backend",
                (
                    f"Backend '{requested_backend}' supports spherical-harmonics "
                    f"degree up to {caps.max_sh_degree}, but degree {degree} was "
                    "requested. Use a CPU backend or a lower degree (the degree is "
                    "not silently reduced)."
                ),
                f"requested_backend={requested_backend} requested_degree={degree} "
                f"max_sh_degree={caps.max_sh_degree}",
            )
        )

    if flags is not None:
        blocked = unsupported_force_models(requested_backend, flags)
        if blocked:
            pretty = ", ".join(blocked)
            issues.append(
                PreflightIssue(
                    ERROR,
                    "force_model_unsupported",
                    (
                        f"Backend '{requested_backend}' does not model: {pretty}. "
                        "Use a CPU backend (these force models are not silently disabled)."
                    ),
                    f"requested_backend={requested_backend} unsupported_force_models={pretty}",
                )
            )

    return issues


def check_output_dir_writable(output_dir: "str | Path | None") -> List[PreflightIssue]:
    """Ensure the output directory exists (or can be created) and is writable."""
    text = str(output_dir or "").strip()
    if not text:
        return [PreflightIssue(ERROR, "output_dir_missing", "No output directory specified.", "output_dir=<empty>")]

    path = Path(text)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return [
            PreflightIssue(
                ERROR,
                "output_dir_uncreatable",
                f"Cannot create output directory: {path}",
                f"output_dir={path} error={exc}",
            )
        ]

    probe = path / ".lunaris_write_test"
    try:
        probe.touch()
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return [
            PreflightIssue(
                ERROR,
                "output_dir_unwritable",
                f"No write permission in output directory: {path}",
                f"output_dir={path} error={exc}",
            )
        ]

    return []


# =============================================================================
# Orchestrator
# =============================================================================


def backend_preflight(
    *,
    requested_backend: str,
    requested_sh_degree: int = 0,
    flags: Any = None,
    output_dir: "str | Path | None" = None,
) -> PreflightReport:
    """Run the backend-capability checks (and, if given, the output-dir check).

    A convenience aggregator for the common "is this backend request runnable?"
    question shared by the CLI, UI, MC, and benchmark flows.
    """
    report = PreflightReport()
    report.add(
        check_backend_capability(
            requested_backend=requested_backend,
            requested_sh_degree=requested_sh_degree,
            flags=flags,
        )
    )
    if output_dir is not None:
        report.add(check_output_dir_writable(output_dir))
    return report


__all__ = [
    "ERROR",
    "WARNING",
    "INFO",
    "PreflightIssue",
    "PreflightReport",
    "check_backend_capability",
    "check_output_dir_writable",
    "backend_preflight",
]
