"""Run History card behaviour on the Results & Export page (W6.2-W6.4).

Headless (offscreen Qt): the card shows an EmptyState until a run directory
exists, lists indexed runs newest-first, renders the KPI summary from the
run's own files, and badges demo output so it cannot pass as evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.ui_qt_helpers import QtWidgets

pytest.importorskip("PySide6.QtWidgets")

from lunaris.ui.pages.result_exports_page import ResultsExportPage  # noqa: E402


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _page(tmp_path: Path) -> ResultsExportPage:
    _app()
    page = ResultsExportPage(
        project_root=tmp_path,
        create_card=lambda title: QtWidgets.QGroupBox(title),
    )
    page.set_output_dir(str(tmp_path / "missions"))
    return page


def _write_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "propagator": {"method": "RK4"},
                "gravity": {"degree": 32},
                "time": {"duration_days": 5.0, "output_dt_s": 30.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_diagnostics.json").write_text(
        json.dumps({"method": "RK4", "wall_time_s": 3.21, "stop_reason": "completed"}),
        encoding="utf-8",
    )
    (run_dir / "altitude.png").write_bytes(b"\x89PNG\r\n")
    (run_dir / "report.pdf").write_bytes(b"%PDF-1.4")
    return run_dir


def _kpi_texts(page: ResultsExportPage) -> dict[str, str]:
    grid = page.kpi_run.layout_grid
    labels = [
        grid.itemAt(i).widget().text()
        for i in range(grid.count())
        if grid.itemAt(i).widget() is not None
    ]
    return dict(zip(labels[::2], labels[1::2], strict=False))


def test_empty_directory_shows_empty_state(tmp_path: Path) -> None:
    page = _page(tmp_path)
    try:
        page.refresh_runs()
        assert page.runs_empty.isVisibleTo(page)
        assert not page.run_history_body.isVisibleTo(page)
        assert page.list_runs.count() == 0
    finally:
        page.deleteLater()


def test_runs_listed_and_kpis_rendered(tmp_path: Path) -> None:
    _write_run(tmp_path / "missions", "mission_2026_07")
    page = _page(tmp_path)
    try:
        page.refresh_runs()
        assert page.list_runs.count() == 1
        assert not page.runs_empty.isVisibleTo(page)
        assert page.run_history_body.isVisibleTo(page)

        kpis = _kpi_texts(page)
        assert kpis["Integrator"] == "RK4"
        assert kpis["Duration"] == "5.0 days"
        assert kpis["Wall time"] == "3.210 s"
        # Gallery lists the figure and the report.
        names = {page.gallery_runs.item(i).text() for i in range(page.gallery_runs.count())}
        assert names == {"altitude.png", "report.pdf"}
        # Provenance line carries the config hash and the run path.
        assert "sha256" in page.lbl_run_provenance.text()
        assert not page.badge_run_demo.isVisibleTo(page)
    finally:
        page.deleteLater()


def test_demo_run_is_badged(tmp_path: Path) -> None:
    _write_run(tmp_path / "missions", "demo_orbit")
    page = _page(tmp_path)
    try:
        page.refresh_runs()
        assert page.list_runs.count() == 1
        assert page.badge_run_demo.isVisibleTo(page)
    finally:
        page.deleteLater()
