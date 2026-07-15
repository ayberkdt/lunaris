"""Terminology hygiene: the framework uses batch/ensemble naming only.

Monte Carlo / MC terminology was removed from all framework-level names
(classes, modules, CLI, config/schema fields, docs). "Monte Carlo" may appear
in prose only as the statistical description of the ``random`` sampling
design. Banned strings below are assembled from parts so this file passes its
own scan.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Assembled from parts so the scan does not trip on this test's own source.
_MC = "mc"
_MONTE = "monte"
_CARLO = "carlo"
_BANNED_GPU_ALIAS = "gpu" + "_sh"

BANNED_SUBSTRINGS: tuple[str, ...] = (
    "lunaris-" + _MC,
    "Monte" + "Carlo" + "Config",
    "Monte" + "Carlo" + "Engine",
    "Monte" + "Carlo" + "Page",
    "Monte" + "Carlo" + "AnalysisPanel",
    _MC.upper() + "RunResult",
    _MC.upper() + "Backend",
    _MC.upper() + "Statistics",
    _MONTE + "_" + _CARLO,
    _MONTE + _CARLO + "_defs",
    _MC + "_backend",
    _MC + "_runner",
    _MC + "_propagator",
    _MC + "_config",
    _MC + "_result",
    _MC + "_cfg",
    _MC + "_entry",
    "load_" + _MC + "_result",
    "plot_" + _MC + "_report",
    "compute_" + _MC + "_statistics",
    "build_" + _MC + "_output_grid",
    "[" + _MC.upper() + "_PROGRESS]",
    "[" + _MC.upper() + "_METRICS]",
    _BANNED_GPU_ALIAS,
)

BANNED_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("local variable named mc", re.compile(r"\b" + _MC + r"\s*=")),
    ("mc attribute alias", re.compile(r"\b" + _MC + r"\.")),
    (_MC + "_ prefix identifier", re.compile(r"(?<![A-Za-z0-9_])" + _MC + r"_")),
    (
        "banned backend-name construction",
        re.compile(
            r"['\"]gpu['\"]\s*\+\s*['\"]_sh['\"]"
        ),
    ),
)

SCAN_DIRS = ("src/lunaris", "tests", "docs", "examples", "hpc", "validation", "tools")
SCAN_FILES = ("README.md", "pyproject.toml")
SCAN_EXTS = {".py", ".md", ".toml", ".cfg", ".txt", ".yaml", ".yml", ".sh", ".sbatch", ".ini"}
EXCLUDE_PARTS = {"node_modules", "__pycache__", ".git", "web", "out", ".next"}
# CHANGELOG records history verbatim and is not part of the active surface.
EXCLUDE_FILES = {"CHANGELOG.md"}


def _iter_scan_files():
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for root, dirs, files in os.walk(base):
            parts = set(Path(root).parts)
            if parts & EXCLUDE_PARTS:
                dirs[:] = []
                continue
            for name in files:
                p = Path(root) / name
                if p.suffix.lower() in SCAN_EXTS and p.name not in EXCLUDE_FILES:
                    yield p
    for f in SCAN_FILES:
        p = REPO_ROOT / f
        if p.exists():
            yield p


def test_no_banned_mc_terminology_in_active_sources() -> None:
    offenders: list[str] = []
    for path in _iter_scan_files():
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lowered = content.lower()
        offenders.extend(
            f"{path.relative_to(REPO_ROOT)}: {banned}"
            for banned in BANNED_SUBSTRINGS
            if banned.lower() in lowered
        )
        for label, pattern in BANNED_REGEXES:
            if path.name == "test_terminology_hygiene.py" and label == "banned backend-name construction":
                continue
            if pattern.search(content):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {label}")
    assert not offenders, "banned Monte-Carlo-era names found:\n" + "\n".join(offenders)


def test_no_mc_named_files_remain() -> None:
    offenders: list[str] = []
    needles = ("_" + _MC + "_", _MONTE, _MC + "_")
    for d in ("src/lunaris", "tests", "docs"):
        base = REPO_ROOT / d
        for root, dirs, files in os.walk(base):
            parts = set(Path(root).parts)
            if parts & EXCLUDE_PARTS:
                dirs[:] = []
                continue
            for name in files:
                low = name.lower()
                if any(n in low for n in needles):
                    offenders.append(str((Path(root) / name).relative_to(REPO_ROOT)))
    assert not offenders, "files with MC-era names remain:\n" + "\n".join(offenders)
