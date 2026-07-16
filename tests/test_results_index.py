"""Unit tests for the Qt-free results index (Results zone W6.1).

The index is the read-only data layer under the Run History card: it must find
run directories (any directory holding ``run_config.json``), tolerate corrupt
JSON, respect the depth bound, skip symlinked directories, and never touch the
scanned files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from lunaris.ui.core.results_index import index_runs, run_kpis


def _make_run(
    root: Path,
    name: str,
    *,
    config: dict | str = "default",
    diagnostics: dict | None = None,
    figures: int = 2,
    reports: int = 1,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    if config == "default":
        config = {
            "propagator": {"method": "DOP853"},
            "gravity": {"degree": 64},
            "time": {"duration_days": 10.0, "output_dt_s": 60.0},
        }
    if isinstance(config, str) and config != "default":
        (run_dir / "run_config.json").write_text(config, encoding="utf-8")
    else:
        (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    if diagnostics is not None:
        (run_dir / "run_diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    for i in range(figures):
        (run_dir / f"figure_{i}.png").write_bytes(b"\x89PNG\r\n")
    for i in range(reports):
        (run_dir / f"report_{i}.pdf").write_bytes(b"%PDF-1.4")
    return run_dir


def test_index_empty_root_returns_no_runs(tmp_path: Path) -> None:
    assert index_runs(tmp_path) == []
    assert index_runs(tmp_path / "does-not-exist") == []


def test_index_finds_flat_and_nested_runs(tmp_path: Path) -> None:
    _make_run(tmp_path, "run_a")
    _make_run(tmp_path, "run_b")
    records = index_runs(tmp_path)
    assert sorted(r.label for r in records) == ["run_a", "run_b"]
    rec = next(r for r in records if r.label == "run_a")
    assert rec.config_ok is True
    assert len(rec.figures) == 2
    assert len(rec.reports) == 1
    assert rec.config_sha256


def test_index_exposes_canonical_analysis_artifacts(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "run_analysis", figures=0, reports=0)
    figures = run_dir / "figures"
    figures.mkdir()
    (figures / "orbit_overview.png").write_bytes(b"\x89PNG\r\n")
    for name in (
        "metrics.json",
        "config.json",
        "diagnostics.json",
        "provenance.json",
    ):
        (run_dir / name).write_text("{}", encoding="utf-8")
    for name in ("events.csv", "orbital_elements.csv", "force_budget.csv"):
        (run_dir / name).write_text("header\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# report", encoding="utf-8")
    (run_dir / "report.pdf").write_bytes(b"%PDF-1.4")

    (record,) = index_runs(tmp_path)
    assert record.analysis_ready is True
    assert record.report_markdown == run_dir / "report.md"
    assert record.metrics_json == run_dir / "metrics.json"
    assert record.force_budget_csv == run_dir / "force_budget.csv"
    assert record.figures_dir == figures
    assert figures / "orbit_overview.png" in record.figures
    assert run_dir / "report.pdf" in record.reports


def test_index_root_itself_can_be_a_run(tmp_path: Path) -> None:
    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "alt.png").write_bytes(b"\x89PNG")
    records = index_runs(tmp_path)
    assert len(records) == 1
    assert records[0].run_dir == tmp_path


def test_index_tolerates_corrupt_config(tmp_path: Path) -> None:
    _make_run(tmp_path, "broken", config="{not json", figures=1, reports=0)
    _make_run(tmp_path, "good")
    records = index_runs(tmp_path)
    assert len(records) == 2
    broken = next(r for r in records if r.label == "broken")
    assert broken.config_ok is False
    assert broken.config == {}
    assert len(broken.figures) == 1  # artifacts still inventoried


def test_index_respects_depth_bound(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c"
    _make_run(deep.parent, "c")  # depth 3 below root -> beyond max_depth=2
    _make_run(tmp_path, "shallow")
    records = index_runs(tmp_path, max_depth=2)
    assert [r.label for r in records] == ["shallow"]


def test_index_skips_symlinked_directories(tmp_path: Path) -> None:
    real = _make_run(tmp_path / "elsewhere", "real_run")
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    link = scan_root / "linked"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        import pytest

        pytest.skip("symlinks unavailable (Windows without developer mode)")
    assert index_runs(scan_root) == []


def test_index_orders_newest_first(tmp_path: Path) -> None:
    old = _make_run(tmp_path, "old")
    new = _make_run(tmp_path, "new")
    past = 1_000_000_000
    os.utime(old / "run_config.json", (past, past))
    records = index_runs(tmp_path)
    assert [r.label for r in records] == ["new", "old"]
    assert records[0].run_dir == new


def test_index_keeps_newest_across_the_limit(tmp_path: Path) -> None:
    """The ``limit`` must retain the newest runs, not the first ``limit`` found.

    Regression for the alphabetical-DFS early-stop bug: a newer run whose name
    sorts late must not be dropped in favour of older, alphabetically-earlier
    runs. Here ``zzz_new`` is newest but sorts last; with limit=1 it must win.
    """
    older = _make_run(tmp_path, "aaa_old")
    newer = _make_run(tmp_path, "zzz_new")
    old_t = 1_000_000_000
    new_t = 2_000_000_000
    os.utime(older / "run_config.json", (old_t, old_t))
    os.utime(newer / "run_config.json", (new_t, new_t))

    records = index_runs(tmp_path, limit=1)
    assert [r.label for r in records] == ["zzz_new"]


def test_demo_marker_from_directory_name(tmp_path: Path) -> None:
    _make_run(tmp_path, "smoke_test_run")
    _make_run(tmp_path, "mission_2026")
    records = {r.label: r for r in index_runs(tmp_path)}
    assert records["smoke_test_run"].is_demo is True
    assert records["mission_2026"].is_demo is False


def test_run_kpis_report_units_and_engine_values(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        "run",
        diagnostics={
            "method": "DOP853",
            "wall_time_s": 12.3456,
            "nfev": 120000,
            "kepler_energy_rel_drift": 1.2e-9,
            "stop_reason": "completed",
        },
    )
    (record,) = index_runs(tmp_path)
    kpis = dict(run_kpis(record))
    assert kpis["Integrator"] == "DOP853"
    assert kpis["Duration"] == "10.0 days"
    assert kpis["Output interval"] == "60.0 s"
    assert kpis["Wall time"] == "12.346 s"
    assert kpis["Energy drift (rel.)"] == "1.200e-09"


def test_run_kpis_omit_absent_values(tmp_path: Path) -> None:
    _make_run(tmp_path, "run", config={}, diagnostics=None)
    (record,) = index_runs(tmp_path)
    assert run_kpis(record) == []


def test_index_accepts_bom_prefixed_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "bom_run"
    run_dir.mkdir()
    payload = json.dumps({"propagator": {"method": "RK4"}})
    (run_dir / "run_config.json").write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
    (record,) = index_runs(tmp_path)
    assert record.config_ok is True
    assert record.config["propagator"]["method"] == "RK4"
