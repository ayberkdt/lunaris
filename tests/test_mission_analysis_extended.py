from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests.test_orbit_analysis_contract import (
    MU,
    RADIUS_M,
    _circular_result,
    _config,
)

from lunaris.analysis.orbit_analysis import build_orbit_analysis
from lunaris.analysis.reporting import mission_report
from lunaris.analysis.reporting.manager import generate_run_package
from lunaris.cli.run import render_reports


def _eccentric_result(*, samples: int = 241, eccentricity: float = 0.2) -> SimpleNamespace:
    semi_major_axis = RADIUS_M + 300_000.0
    mean_motion = np.sqrt(MU / semi_major_axis**3)
    period = 2.0 * np.pi / mean_motion
    t = np.linspace(0.0, 2.0 * period, samples)
    mean_anomaly = mean_motion * t
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(12):
        eccentric_anomaly -= (
            eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))
    denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
    x = semi_major_axis * (np.cos(eccentric_anomaly) - eccentricity)
    y = semi_major_axis * np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly)
    vx = -semi_major_axis * mean_motion * np.sin(eccentric_anomaly) / denominator
    vy = (
        semi_major_axis
        * mean_motion
        * np.sqrt(1.0 - eccentricity**2)
        * np.cos(eccentric_anomaly)
        / denominator
    )
    inclination = 0.3
    state = np.column_stack(
        (
            x,
            y * np.cos(inclination),
            y * np.sin(inclination),
            vx,
            vy * np.cos(inclination),
            vy * np.sin(inclination),
        )
    )
    return SimpleNamespace(
        t=t,
        y=state,
        diagnostics={
            "wall_time_s": 1.0,
            "nfev": 900,
            "accepted_steps": 120,
            "rejected_steps": 3,
            "internal_step_min_s": 0.2,
            "internal_step_median_s": 12.0,
            "internal_step_max_s": 30.0,
            "integrator": "DOP853",
            "integration_backend": "scipy",
            "rhs_path": "general_python",
            "degree": 20,
        },
        impacted=False,
        stopped_early=False,
        stop_reason=None,
        ode=SimpleNamespace(success=True),
        t_events=[np.asarray([period])],
    )


def test_eccentric_metrics_extrema_wrapping_and_secular_fit() -> None:
    analysis = build_orbit_analysis(
        _eccentric_result(),
        config=_config(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
    )
    metrics = analysis.metric_map
    assert metrics["orbit.e.initial"].value == pytest.approx(0.2, rel=1.0e-8)
    assert metrics["orbit.altitude.minimum"].value == pytest.approx(
        (RADIUS_M + 300_000.0) * 0.8 - RADIUS_M,
        abs=1.0e-6,
    )
    assert metrics["orbit.altitude.maximum"].value == pytest.approx(
        (RADIUS_M + 300_000.0) * 1.2 - RADIUS_M,
        abs=1.0e-6,
    )
    assert metrics["orbit.raan.initial"].status == "ok"
    assert metrics["orbit.argp.initial"].status == "ok"
    assert metrics["orbit.secular.semi_major_axis"].value == pytest.approx(0.0, abs=1.0e-5)
    assert metrics["numerical.accepted_steps"].value == 120
    assert metrics["numerical.internal_step.median"].value == 12.0
    assert metrics["numerical.event_location_quality"].status == "ok"
    event_ids = [event.event_id for event in analysis.events]
    assert len(event_ids) == len(set(event_ids))
    assert analysis.events[-1].event_type == "completed"


def test_low_degree_recommendation_is_a_visible_warning() -> None:
    result = _eccentric_result()
    result.diagnostics["recommended_degree"] = 124
    analysis = build_orbit_analysis(
        result,
        config=_config(enable_sh=True),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
    )
    metric = analysis.metric_map["physics.gravity_degree.recommended"]
    assert metric.status == "warning"
    assert any("degree 20" in warning and "124" in warning for warning in analysis.warnings)


def test_terminal_impact_event_is_critical_and_ordered() -> None:
    result = _eccentric_result()
    result.impacted = True
    result.t_impact_s = float(result.t[-1])
    result.stop_reason = "impact"
    analysis = build_orbit_analysis(
        result,
        config=_config(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
    )
    impact = next(event for event in analysis.events if event.event_type == "impact")
    assert impact.severity == "critical"
    assert analysis.metric_map["run.status"].value == "impact"
    assert list(analysis.events) == sorted(
        analysis.events,
        key=lambda event: (event.simulation_time_s, event.event_id),
    )


def test_optional_pdf_page_failure_yields_unavailable_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = build_orbit_analysis(
        _circular_result(samples=25),
        config=_config(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
        preset="quick",
    )

    def fail_page(*_args, **_kwargs):
        raise RuntimeError("intentional optional-page failure")

    monkeypatch.setitem(mission_report._PAGE_FACTORIES, "orbit", fail_page)
    path = mission_report.write_report_pdf(analysis, tmp_path / "failure-safe.pdf")
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") >= 5


def test_paper_pdf_formatting_is_deterministic(tmp_path: Path) -> None:
    analysis = build_orbit_analysis(
        _circular_result(samples=25),
        config=_config(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
        preset="paper",
        run_id="deterministic-paper",
    )
    analysis = replace(analysis, generated_at_utc="2026-01-01T00:00:00Z")
    first = mission_report.write_report_pdf(analysis, tmp_path / "first.pdf")
    second = mission_report.write_report_pdf(analysis, tmp_path / "second.pdf")
    assert first.read_bytes() == second.read_bytes()
    assert b"/MediaBox [ 0 0 595.44 841.68 ]" in first.read_bytes()
    assert first.read_bytes().count(b"/Type /Page") >= 13
    assert "spatial" in mission_report.REPORT_PRESETS["paper"].page_sections
    assert "force_dynamics" in mission_report.REPORT_PRESETS["paper"].page_sections
    assert mission_report.REPORT_PRESETS["paper"].page_sections[-1] == "assets"


def test_operational_orbit_envelope_and_force_status_are_derived() -> None:
    config = _config(enable_sh=True, enable_srp=True)
    analysis = build_orbit_analysis(
        _eccentric_result(samples=49),
        config=config,
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
        preset="paper",
    )

    periselene_km, aposelene_km, period_h = mission_report._osculating_envelope(analysis)
    assert periselene_km.shape == aposelene_km.shape == period_h.shape
    assert np.all(np.isfinite(period_h))
    assert np.all(periselene_km < aposelene_km)

    analysis = replace(
        analysis,
        config_snapshot={
            "flags": {
                "enable_sh": True,
                "enable_srp": True,
                "enable_relativity_1pn": False,
            }
        },
    )
    statuses = {item.label: item for item in mission_report._force_model_statuses(analysis)}
    assert statuses["Spherical harmonics"].active is True
    assert statuses["Solar radiation"].active is True
    assert statuses["Relativity 1PN"].active is False


def test_analysis_layer_does_not_import_ui() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "lunaris" / "analysis"
    offenders = [
        path
        for path in root.rglob("*.py")
        if "lunaris.ui" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_public_manager_generates_run_package(tmp_path: Path) -> None:
    outputs = generate_run_package(
        result=_circular_result(samples=17),
        config=_config(),
        out_dir=tmp_path,
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
        preset="quick",
    )
    assert outputs["status"] == "success"
    assert (tmp_path / "report.pdf").is_file()


def test_cli_report_boundary_emits_structured_notification(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outputs = render_reports(
        result=_circular_result(samples=17),
        engine=None,
        cfg=_config(),
        out_dir=tmp_path,
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
        preset="quick",
    )
    captured = capsys.readouterr().out
    assert outputs["status"] == "success"
    assert "[REPORT]" in captured
    payload = json.loads(captured.split("[REPORT]", 1)[1])
    assert payload["report_pdf"] == str((tmp_path / "report.pdf").resolve())
