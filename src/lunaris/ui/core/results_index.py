"""Read-only index of completed mission runs for the Results zone (W6.1).

A *run directory* is any directory that contains a ``run_config.json`` written
by :func:`lunaris.cli.run.write_run_artifacts`. This module scans a results
root for such directories and returns plain, Qt-free records the Results page
can render (run selector, KPI summary, figure gallery).

Safety contract (Results-zone risk decisions)
---------------------------------------------
* **Read-only** — nothing here creates, moves, or deletes files.
* **No symlink traversal** — symlinked directories are skipped so the scan
  cannot escape the chosen root.
* **Bounded** — the walk is depth-limited and caps the number of runs, so a
  mistakenly selected huge directory cannot hang the caller.
* **Tolerant** — a corrupt ``run_config.json`` still yields a record (with
  ``config_ok=False``); one bad run must not hide the others.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["RunRecord", "index_runs", "run_kpis"]

_RUN_CONFIG_NAME = "run_config.json"
_DIAGNOSTICS_NAME = "run_diagnostics.json"
_FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg")
_REPORT_SUFFIXES = (".pdf",)
#: Directory-name tokens that mark output as demonstration/smoke material.
#: Demo output must never be presentable as solver evidence, so the page
#: badges these records visibly (scientific-UX rule).
_DEMO_TOKENS = ("demo", "smoke", "example", "sample", "tutorial")


@dataclass(frozen=True)
class RunRecord:
    """One completed run directory, described without interpretation."""

    run_dir: Path
    label: str
    created_at: datetime
    config: dict[str, Any] = field(default_factory=dict)
    #: False when run_config.json existed but could not be parsed.
    config_ok: bool = True
    #: Short sha256 prefix of the raw run_config.json bytes (provenance id).
    config_sha256: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    figures: tuple[Path, ...] = ()
    reports: tuple[Path, ...] = ()
    is_demo: bool = False


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        # utf-8-sig also accepts a Windows BOM (Notepad / PowerShell exports).
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}, False
    if not isinstance(payload, dict):
        return {}, False
    return payload, True


def _sha256_prefix(path: Path, *, length: int = 12) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""
    return digest[:length]


def _looks_demo(run_dir: Path) -> bool:
    name = run_dir.name.lower()
    return any(token in name for token in _DEMO_TOKENS)


def _record_for(run_dir: Path) -> RunRecord:
    config_path = run_dir / _RUN_CONFIG_NAME
    config, config_ok = _load_json(config_path)
    diagnostics, _ = _load_json(run_dir / _DIAGNOSTICS_NAME)

    figures: list[Path] = []
    reports: list[Path] = []
    try:
        for entry in sorted(run_dir.iterdir()):
            if not entry.is_file() or entry.is_symlink():
                continue
            suffix = entry.suffix.lower()
            if suffix in _FIGURE_SUFFIXES:
                figures.append(entry)
            elif suffix in _REPORT_SUFFIXES:
                reports.append(entry)
    except OSError:
        pass

    try:
        created = datetime.fromtimestamp(config_path.stat().st_mtime)
    except OSError:
        created = datetime.fromtimestamp(0)

    return RunRecord(
        run_dir=run_dir,
        label=run_dir.name,
        created_at=created,
        config=config,
        config_ok=config_ok,
        config_sha256=_sha256_prefix(config_path),
        diagnostics=diagnostics,
        figures=tuple(figures),
        reports=tuple(reports),
        is_demo=_looks_demo(run_dir),
    )


#: Hard cap on directories examined during a single scan. Bounds the cost of a
#: mistakenly selected huge tree while staying far above any realistic run count.
_MAX_DIRS_VISITED = 20_000


def index_runs(root: Path | str, *, max_depth: int = 2, limit: int = 200) -> list[RunRecord]:
    """Scan ``root`` for run directories and return the newest ``limit``.

    ``max_depth`` counts directory levels below ``root`` (the root itself is
    depth 0 and is also considered — a flat layout that writes straight into
    the results directory is a valid single run).

    The scan collects *all* candidate run directories (up to a directory-visit
    cap) and only then keeps the newest ``limit`` by modification time. Ranking
    after collection is what makes "newest first" a real guarantee: stopping the
    walk at the first ``limit`` matches would drop newer runs that happen to sort
    late alphabetically.
    """

    root_path = Path(root)
    if not root_path.is_dir():
        return []

    # (mtime, run_dir) candidates; records are built only for the survivors so a
    # huge tree does not pay JSON-parse + hash cost for runs it will discard.
    candidates: list[tuple[float, Path]] = []
    visited = 0

    def _mtime(directory: Path) -> float:
        try:
            return (directory / _RUN_CONFIG_NAME).stat().st_mtime
        except OSError:
            return 0.0

    def _scan(directory: Path, depth: int) -> None:
        nonlocal visited
        if visited >= _MAX_DIRS_VISITED:
            return
        visited += 1
        if (directory / _RUN_CONFIG_NAME).is_file():
            candidates.append((_mtime(directory), directory))
            # A run directory is a leaf: runs are not nested inside runs.
            return
        if depth >= int(max_depth):
            return
        try:
            with os.scandir(directory) as entries:
                subdirs = [
                    entry
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False) and not entry.is_symlink()
                ]
        except OSError:
            return
        for entry in sorted(subdirs, key=lambda e: e.name):
            if visited >= _MAX_DIRS_VISITED:
                return
            _scan(Path(entry.path), depth + 1)

    _scan(root_path, 0)

    newest = heapq.nlargest(int(limit), candidates, key=lambda item: item[0])
    records = [_record_for(run_dir) for _, run_dir in newest]
    records.sort(key=lambda record: record.created_at, reverse=True)
    return records


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------

def _config_get(config: dict[str, Any], *path: str) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _fmt(value: Any, fmt: str, unit: str) -> str | None:
    if value is None:
        return None
    try:
        text = fmt.format(value)
    except (TypeError, ValueError):
        text = str(value)
    return f"{text} {unit}".strip()


def run_kpis(record: RunRecord) -> list[tuple[str, str]]:
    """Derive the KPI summary rows for one run.

    Single formatting path for every physical value (unit stated explicitly);
    values come only from the run's own ``run_config.json`` and the
    engine-reported ``run_diagnostics.json`` — nothing is estimated here.
    """

    cfg = record.config
    diag = record.diagnostics
    rows: list[tuple[str, str]] = []

    def add(label: str, rendered: str | None) -> None:
        if rendered:
            rows.append((label, rendered))

    add("Integrator", _fmt(diag.get("method") or _config_get(cfg, "propagator", "method"), "{}", ""))
    add("SH degree", _fmt(diag.get("degree") if diag.get("degree") is not None else _config_get(cfg, "gravity", "degree"), "{}", ""))
    add("Duration", _fmt(_config_get(cfg, "time", "duration_days"), "{}", "days"))
    add("Output interval", _fmt(diag.get("output_dt_s") if diag.get("output_dt_s") is not None else _config_get(cfg, "time", "output_dt_s"), "{}", "s"))
    add("Wall time", _fmt(diag.get("wall_time_s"), "{:.3f}", "s"))
    add("RHS evaluations", _fmt(diag.get("nfev"), "{:,.0f}", ""))
    add("Energy drift (rel.)", _fmt(diag.get("kepler_energy_rel_drift"), "{:.3e}", ""))
    add("Stop reason", _fmt(diag.get("stop_reason"), "{}", ""))
    return rows
