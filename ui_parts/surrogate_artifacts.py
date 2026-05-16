# LUNAR_SIMULATION/ui_parts/surrogate_artifacts.py
# -*- coding: utf-8 -*-
"""
Shared surrogate gravity run artifact resolver.

All UI components that need to validate or locate a surrogate gravity run
should go through this module.  It keeps the acceptance policy in one place
so preflight, force-models dialog, and MC page all agree on what constitutes
a valid run.

Design rules
------------
- Filesystem + JSON only.  No PyTorch, no checkpoint weight loading.
- ckpt_last.pt is accepted when ckpt_best.pt is absent (emit a warning).
- config.json absence is a warning, not an error (checkpoint may be self-contained).
- scaler.json absence is a warning (backend may embed scaler in checkpoint).
- Run directory can be supplied as:
    1. The run directory itself.
    2. The checkpoints/ sub-directory.
    3. A direct ckpt_best.pt or ckpt_last.pt file path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SurrogateArtifacts:
    """Resolved paths for a trained surrogate gravity run."""

    run_dir: Path
    checkpoint_path: Path
    checkpoint_kind: str               # "best" | "last"
    config_path: Optional[Path] = None
    scaler_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_run_dir(path: Path) -> Path:
    """
    Normalise any of the three supported path forms into a run directory.

    Accepted inputs
    ---------------
    - Run dir  (contains config.json / checkpoints/)
    - checkpoints/ sub-dir  (parent is run dir)
    - Direct ckpt_best.pt / ckpt_last.pt file  (parent.parent is run dir)
    """

    p = path.expanduser().resolve()

    if p.is_file():
        # Direct checkpoint file → run dir is two levels up
        if p.name in ("ckpt_best.pt", "ckpt_last.pt"):
            return p.parent.parent
        # Unknown file — use its parent as a best-effort guess
        return p.parent

    if p.is_dir():
        # If the directory is named "checkpoints", its parent is the run dir
        if p.name == "checkpoints":
            return p.parent

    return p


def _find_checkpoint(run_dir: Path) -> Tuple[Optional[Path], str]:
    """
    Return the best available checkpoint under *run_dir/checkpoints/*.

    Preference: ckpt_best.pt > ckpt_last.pt
    """

    ckpt_dir = run_dir / "checkpoints"
    best = ckpt_dir / "ckpt_best.pt"
    last = ckpt_dir / "ckpt_last.pt"

    if best.is_file():
        return best, "best"
    if last.is_file():
        return last, "last"
    return None, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_surrogate_artifacts(
    path: "str | Path",
) -> Tuple[Optional[SurrogateArtifacts], List[str]]:
    """
    Resolve a user-supplied path into a ``SurrogateArtifacts`` object.

    Returns
    -------
    (artifacts, errors)
        ``artifacts`` is ``None`` when the run is fatally invalid.
        ``errors`` lists all fatal problems found.
        Non-fatal issues are recorded in ``artifacts.warnings``.
    """

    errors: List[str] = []
    warnings_list: List[str] = []

    run_dir = _resolve_run_dir(Path(path))

    if not run_dir.exists():
        errors.append(f"Path does not exist: {run_dir}")
        return None, errors

    if not run_dir.is_dir():
        errors.append(f"Path is not a directory: {run_dir}")
        return None, errors

    # Checkpoint — required
    ckpt_path, ckpt_kind = _find_checkpoint(run_dir)
    if ckpt_path is None:
        ckpt_dir = run_dir / "checkpoints"
        errors.append(
            f"No checkpoint found in {ckpt_dir}. "
            "Expected ckpt_best.pt or ckpt_last.pt."
        )
        return None, errors

    if ckpt_kind == "last":
        warnings_list.append(
            "Using ckpt_last.pt because ckpt_best.pt was not found. "
            "This may be an in-progress training run."
        )

    # config.json — soft-required (warn if absent)
    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        config_out: Optional[Path] = cfg_path
    else:
        config_out = None
        warnings_list.append(
            f"config.json not found in {run_dir}. "
            "Lunar body validation will be skipped."
        )

    # scaler.json — soft-required (warn if absent)
    scaler_path = run_dir / "scaler.json"
    if scaler_path.is_file():
        scaler_out: Optional[Path] = scaler_path
    else:
        scaler_out = None
        warnings_list.append(
            f"scaler.json not found in {run_dir}. "
            "Backend may use scaler embedded in the checkpoint."
        )

    artifacts = SurrogateArtifacts(
        run_dir=run_dir,
        checkpoint_path=ckpt_path,
        checkpoint_kind=ckpt_kind,
        config_path=config_out,
        scaler_path=scaler_out,
        warnings=warnings_list,
    )
    return artifacts, errors


def is_valid_surrogate_run(path: "str | Path") -> bool:
    """
    Return ``True`` when the path resolves to a usable surrogate run.

    Accepts ckpt_last.pt as a fallback when ckpt_best.pt is absent.
    """

    artifacts, errors = resolve_surrogate_artifacts(path)
    return artifacts is not None and not errors


def resolve_st_lrps_model_dir(path: "str | Path") -> Path:
    """
    Return the canonical run directory for any supported path form.

    Does not validate whether the run is complete.
    """

    return _resolve_run_dir(Path(path))


def validate_surrogate_run_preflight(
    path: "str | Path",
) -> Tuple[bool, str, List[str]]:
    """
    Full preflight check suitable for ``PreFlightWorker``.

    Returns
    -------
    (ok, summary_message, warnings)
        ``ok`` is ``False`` when the run cannot be used.
        ``summary_message`` describes the outcome.
        ``warnings`` is a list of non-fatal caveats.
    """

    if not str(path).strip():
        return False, "Surrogate gravity run directory is not set.", []

    artifacts, errors = resolve_surrogate_artifacts(path)
    if errors:
        return False, "\n".join(errors), []

    assert artifacts is not None
    return (
        True,
        f"Surrogate gravity run validated: {artifacts.run_dir.name} "
        f"(checkpoint: {artifacts.checkpoint_kind})",
        artifacts.warnings,
    )


def looks_like_lunar_surrogate_run(path: "str | Path") -> bool:
    """
    Return ``True`` when the run's config.json indicates a Moon-targeted model.

    Falls back to ``False`` on any error (missing config, bad JSON, etc.).
    """

    run_dir = _resolve_run_dir(Path(path))
    cfg_path = run_dir / "config.json"

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    body_name = str(cfg.get("central_body", "") or "").strip().lower()
    if body_name in {"moon", "lunar", "selene"}:
        return True

    dataset_meta = cfg.get("dataset_meta") or {}
    if not isinstance(dataset_meta, dict):
        dataset_meta = {}

    def _close(val: object, ref: float, tol: float = 0.05) -> bool:
        try:
            v = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return abs(v - ref) / max(abs(ref), 1.0) <= tol

    # Moon μ ≈ 4.905e12 m³/s², R ≈ 1.738e6 m
    for candidate in (cfg.get("resolved_mu_si"), dataset_meta.get("mu_si")):
        if _close(candidate, 4_904_869_500_000.0):
            return True

    for candidate in (
        cfg.get("resolved_r_ref_m"),
        cfg.get("r_ref_m"),
        dataset_meta.get("r_ref_m"),
        dataset_meta.get("r_ref_m_fallback"),
    ):
        if _close(candidate, 1_738_000.0):
            return True

    return False
