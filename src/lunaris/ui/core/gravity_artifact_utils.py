# ST_LRPS/ui_parts/gravity_artifact_utils.py
# -*- coding: utf-8 -*-
"""
Pure gravity artifact helper functions for UI modules.

These helpers are filesystem-only and do not require PyTorch or heavy
backend imports.  They were extracted from force_models_page.py so that
other UI modules can use them without importing the full page widget tree.

Backward-compat note: force_models_page.py still re-exports everything
from here under the same names so existing call sites don't break.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

try:
    from lunaris.ui.core.ui_commons import normalize_path, find_project_root
    from lunaris.ui.core.surrogate_artifacts import is_valid_surrogate_run, looks_like_lunar_surrogate_run
except ImportError:
    if __name__ == "__main__":
        import sys
        print("Run as: python -m lunaris.ui.core.gravity_artifact_utils", file=sys.stderr)
        raise SystemExit(2)
    raise


PROJECT_ROOT = find_project_root()
ST_LRPS_RUNS_DIR = PROJECT_ROOT / "st_lrps" / "runs"

GRAVITY_EXTENSIONS = (".shbdr", ".dat", ".txt", ".tab", ".gfc")


def extract_sh_degree(filename: str) -> Optional[int]:
    """
    Heuristic to extract the SH degree from a gravity model filename.
    Matches patterns like 'GRGM1200B', 'deg100', 'L180'.
    """
    if not filename:
        return None

    match = re.search(
        r"(?:^|[^a-z0-9])(?:l|deg|degree|d)[\s_\-]*([0-9]{2,4})(?:[^0-9]|$)",
        filename,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(r"([0-9]{3,4})", filename)

    try:
        return int(match.group(1)) if match else None
    except (ValueError, IndexError):
        return None


def find_best_gravity_file(
    root_dir: Path, preferred_degree: Optional[int] = None
) -> Optional[str]:
    """
    Automatically scan the project for gravity model files.
    Scores candidates by extension, degree match, and file size.
    """

    search_paths = [
        root_dir,
        root_dir / "data" / "gravity",
        root_dir / "models" / "gravity",
        root_dir / "assets",
    ]

    candidates = []

    for env_var in ["STLRPS_GRAVITY_PATH", "STLRPS_SHBDR"]:
        env_val = os.environ.get(env_var)
        if env_val and Path(env_val).is_file():
            return normalize_path(env_val)

    for folder in search_paths:
        if not folder.is_dir():
            continue
        for file_path in folder.rglob("*"):
            if file_path.suffix.lower() not in GRAVITY_EXTENSIONS:
                continue
            fn = file_path.name
            size = file_path.stat().st_size if file_path.exists() else 0
            ext_score = 2 if file_path.suffix.lower() == ".shbdr" else 1
            detected_deg = extract_sh_degree(fn)
            exact_match = 1 if (preferred_degree and detected_deg == preferred_degree) else 0
            if preferred_degree and detected_deg:
                diff_score = -abs(detected_deg - preferred_degree)
            else:
                diff_score = -9999
            candidates.append((ext_score, exact_match, diff_score, size, str(file_path)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    return normalize_path(candidates[0][4])


def list_st_lrps_model_dirs(root_dir: Path = ST_LRPS_RUNS_DIR) -> List[Path]:
    """
    Discover available surrogate gravity runs sorted newest-first.

    Only returns runs that pass `is_valid_surrogate_run` (accepts ckpt_last)
    and appear to be Moon-targeted.
    """

    if not root_dir.is_dir():
        return []

    runs = [
        path.resolve()
        for path in root_dir.iterdir()
        if path.is_dir()
        and is_valid_surrogate_run(path)
        and looks_like_lunar_surrogate_run(path)
    ]
    runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return runs
