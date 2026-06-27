"""ST-LRPS artifact discovery helpers for surrogate runtime inference."""

from __future__ import annotations

import json
from pathlib import Path

from lunaris.common.lunar_data import looks_like_lunar_run_config
from lunaris.common.paths import project_root_from_file

_REPO_ROOT = project_root_from_file(__file__)
DEFAULT_ST_LRPS_RUNS_DIR = _REPO_ROOT / "st_lrps" / "runs"

def _is_valid_surrogate_run(path: Path) -> bool:
    """
    Return ``True`` when ``path`` looks like a complete surrogate gravity run.

    Accepts either ``ckpt_best.pt`` (fully trained) or ``ckpt_last.pt``
    (in-progress / interrupted run) so the UI can offer recently-started
    runs for inspection without requiring a completed best-checkpoint.
    """

    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    ckpt_dir = path / "checkpoints"
    return (ckpt_dir / "ckpt_best.pt").is_file() or (ckpt_dir / "ckpt_last.pt").is_file()


def _find_checkpoint_for_run(run_dir: Path) -> Path:
    """
    Return the best available checkpoint inside *run_dir/checkpoints/*.

    Preference order: ``ckpt_best.pt`` (finished training) then
    ``ckpt_last.pt`` (interrupted run, suitable for inference with a warning).
    Raises ``FileNotFoundError`` when neither exists.
    """

    ckpt_dir = run_dir / "checkpoints"
    for name in ("ckpt_best.pt", "ckpt_last.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. Expected ckpt_best.pt or ckpt_last.pt."
    )


def find_checkpoint_for_st_lrps_run(run_dir: Path | str) -> Path:
    """Public wrapper used by validation tools to report the selected weights."""

    return _find_checkpoint_for_run(Path(run_dir).expanduser().resolve())

def _looks_like_lunar_run(path: Path) -> bool:
    """
    Return ``True`` when run metadata clearly targets the Moon.

    Discovery helpers should avoid auto-selecting old Earth-era experiments,
    even if those folders still contain syntactically valid checkpoints.
    """

    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(looks_like_lunar_run_config(cfg))


def discover_st_lrps_model_dirs(root: Path | str = DEFAULT_ST_LRPS_RUNS_DIR) -> list[Path]:
    """
    Discover available surrogate gravity run directories.

    Results are sorted by modification time (newest first) so the UI can offer
    the most recent trained model as the first suggestion.
    """

    runs_root = Path(root).expanduser().resolve()
    search_dirs = [runs_root]

    # Also check the newer `outputs/training` directory if using the default root
    if runs_root == DEFAULT_ST_LRPS_RUNS_DIR.resolve():
        search_dirs.append((_REPO_ROOT / "outputs" / "training").resolve())

    candidates = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        candidates.extend([
            p.resolve()
            for p in d.iterdir()
            if _is_valid_surrogate_run(p) and _looks_like_lunar_run(p)
        ])

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates


def find_latest_st_lrps_model_dir(root: Path | str = DEFAULT_ST_LRPS_RUNS_DIR) -> Path | None:
    """Return the newest valid surrogate run directory, if any."""

    candidates = discover_st_lrps_model_dirs(root)
    return candidates[0] if candidates else None


# =============================================================================
# 2.                          SCALER NORMALIZATION
# =============================================================================

__all__ = [
    "DEFAULT_ST_LRPS_RUNS_DIR",
    "_is_valid_surrogate_run",
    "_find_checkpoint_for_run",
    "find_checkpoint_for_st_lrps_run",
    "_looks_like_lunar_run",
    "discover_st_lrps_model_dirs",
    "find_latest_st_lrps_model_dir",
]
