from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.analysis.artifacts import load_analysis_artifacts, write_analysis_artifacts
from lunaris.analysis.contracts import MetricValue
from lunaris.analysis.orbit_analysis import build_orbit_analysis
from lunaris.analysis.reporting.mission_report import generate_analysis_package
from lunaris.analysis.run_comparison import compare_run_packages

MU = 4.902801e12
RADIUS_M = 1_737_400.0


def _config(*, enable_sh: bool = False, enable_srp: bool = False, enable_tides: bool = False):
    return SimpleNamespace(
        time=SimpleNamespace(start_date="2026-01-01T00:00:00Z"),
        propagator=SimpleNamespace(
            method="DOP853",
            events=SimpleNamespace(
                detect_impact=True,
                impact_alt_km=0.0,
                enable_peri_apo_events=True,
                detect_eclipse=False,
            ),
        ),
        gravity=SimpleNamespace(backend="classic_sh", degree=20, file_path=None),
        spice=SimpleNamespace(inertial_frame="J2000", kernel_paths=()),
        flags=SimpleNamespace(
            enable_sh=enable_sh,
            enable_srp=enable_srp,
            enable_tides=enable_tides,
            enable_tides_k2=enable_tides,
            enable_tides_k3=False,
            enable_3rd_body_sun=False,
            enable_3rd_body_earth=False,
            enable_earth_j2=False,
            enable_albedo=False,
            enable_thermal=False,
            enable_relativity_1pn=False,
        ),
    )


def _circular_result(*, samples: int = 65):
    radius = RADIUS_M + 100_000.0
    period = 2.0 * np.pi * np.sqrt(radius**3 / MU)
    t = np.linspace(0.0, 2.0 * period, samples)
    theta = 2.0 * np.pi * t / period
    speed = np.sqrt(MU / radius)
    state = np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta),
            -speed * np.sin(theta),
            speed * np.cos(theta),
            np.zeros_like(theta),
        )
    )
    return SimpleNamespace(
        t=t,
        y=state,
        diagnostics={
            "wall_time_s": 2.0,
            "nfev": 512,
            "integrator": "DOP853",
            "integration_backend": "scipy",
            "rhs_path": "general_python",
            "degree": 20,
        },
        impacted=False,
        stopped_early=False,
        stop_reason=None,
        ode=SimpleNamespace(success=True),
    )


def test_metric_contract_rejects_nonfinite_and_requires_unavailable_reason() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        MetricValue("bad", "Bad", float("nan"), "m", "ok", "test")
    with pytest.raises(ValueError, match="availability_reason"):
        MetricValue("missing", "Missing", None, "m", "unavailable", "test")


def test_circular_equatorial_analysis_marks_singular_angles_unavailable() -> None:
    analysis = build_orbit_analysis(
        _circular_result(),
        config=_config(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
        run_id="unit_run",
    )

    metrics = analysis.metric_map
    assert metrics["orbit.raan.initial"].status == "unavailable"
    assert metrics["orbit.argp.initial"].status == "unavailable"
    assert metrics["orbit.altitude.minimum"].value == pytest.approx(100_000.0)
    assert metrics["orbit.period"].value is not None
    assert metrics["orbit.completed_count"].value == pytest.approx(2.0, rel=0.08)
    assert metrics["diagnostic.energy.max_relative_drift"].kind == "invariant"
    assert metrics["numerical.accepted_steps"].availability_reason == "Unavailable for this integrator."
    assert all(
        left.simulation_time_s <= right.simulation_time_s
        for left, right in zip(analysis.events, analysis.events[1:], strict=False)
    )
    assert analysis.metrics_payload()["analysis_schema_version"] == 2


def test_maneuver_diagnostics_become_events_metrics_and_disable_invariant_claim() -> None:
    result = _circular_result()
    burn_t = float(result.t[10])
    result.diagnostics["maneuvers_applied"] = [
        {
            "t_burn_s": burn_t,
            "frame": "ric",
            "dv_norm_mps": 12.5,
            "mass_before_kg": 1000.0,
            "mass_after_kg": 995.0,
        }
    ]
    analysis = build_orbit_analysis(
        result,
        config=_config(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
    )
    maneuver = next(event for event in analysis.events if event.event_type == "maneuver")
    assert maneuver.simulation_time_s == pytest.approx(burn_t)
    assert "delta-v=12.5 m/s" in (maneuver.note or "")
    assert analysis.metric_map["mission.maneuver.count"].value == 1
    assert analysis.metric_map["mission.maneuver.total_delta_v"].value == pytest.approx(12.5)
    assert analysis.metric_map["mission.maneuver.propellant_used"].value == pytest.approx(5.0)
    assert analysis.metric_map["diagnostic.energy.max_relative_drift"].kind == "diagnostic"


def test_live_force_budget_excludes_aggregate_terms_from_ranking() -> None:
    class Context:
        R_body_m = RADIUS_M

        @staticmethod
        def get_acceleration_breakdown(t: float, state: np.ndarray) -> dict[str, float]:
            del state
            return {
                "Gravity (SH)": 1.5,
                "SRP": 1.0e-7 * (1.0 + 0.01 * np.cos(t)),
                "Solid Tides (Earth)": 4.0e-9,
                "Solid Tides (Sun)": 2.0e-9,
                "Solid Tides": 5.0e-9,
            }

    analysis = build_orbit_analysis(
        _circular_result(),
        config=_config(enable_sh=True, enable_srp=True, enable_tides=True),
        ctx=Context(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
    )
    budget = {item.force_id: item for item in analysis.force_contributions}

    assert budget["gravity_sh"].included_in_noncentral_ranking is False
    assert budget["solid_tides"].included_in_noncentral_ranking is False
    assert budget["solid_tides_earth"].included_in_noncentral_ranking is True
    assert budget["spherical_harmonic_residual"].available is False
    assert "No magnitude subtraction" in (budget["spherical_harmonic_residual"].interpretation or "")
    assert analysis.metric_map["diagnostic.energy.max_relative_drift"].kind == "diagnostic"


def test_vector_force_budget_forms_physical_total_and_signed_ric(tmp_path) -> None:
    class Context:
        @staticmethod
        def get_acceleration_vector_breakdown(
            t: float,
            state: np.ndarray,
        ) -> dict[str, np.ndarray]:
            del t, state
            return {
                "Gravity (SH)": np.asarray((-1.5, 0.0, 0.0)),
                "SRP": np.asarray((1.0e-7, 2.0e-7, 3.0e-7)),
                "3rd Body (Earth)": np.asarray((4.0e-7, -1.0e-7, 2.0e-7)),
            }

    config = _config(enable_sh=True, enable_srp=True)
    config.flags.enable_3rd_body_earth = True
    analysis = build_orbit_analysis(
        _circular_result(samples=17),
        config=config,
        ctx=Context(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
    )

    expected = np.asarray((5.0e-7, 1.0e-7, 5.0e-7))
    total = analysis.force_vectors_m_s2["Total non-central acceleration"]
    assert total[0] == pytest.approx(expected)
    assert analysis.force_ric_m_s2["Total non-central acceleration"][0] == pytest.approx(
        expected
    )
    budget = {item.force_id: item for item in analysis.force_contributions}
    assert budget["total_noncentral_acceleration"].available is True
    assert budget["total_noncentral_acceleration"].median_m_s2 == pytest.approx(
        np.linalg.norm(expected)
    )

    write_analysis_artifacts(analysis, tmp_path)
    loaded = load_analysis_artifacts(tmp_path)
    assert loaded.force_vectors_m_s2["SRP"] == pytest.approx(
        analysis.force_vectors_m_s2["SRP"]
    )
    assert loaded.force_ric_m_s2["Total non-central acceleration"] == pytest.approx(
        analysis.force_ric_m_s2["Total non-central acceleration"]
    )


def test_analysis_rejects_nonfinite_source_state() -> None:
    result = _circular_result()
    result.y[4, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_orbit_analysis(result, config=_config())


def test_run_comparison_persists_typed_metric_deltas(tmp_path) -> None:
    baseline = build_orbit_analysis(
        _circular_result(samples=17),
        config=_config(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
        run_id="baseline",
    )
    candidate_metrics = tuple(
        replace(metric, value=float(metric.value) + 250.0)
        if metric.metric_id == "orbit.altitude.minimum" and metric.value is not None
        else metric
        for metric in baseline.metrics
    )
    candidate = replace(baseline, run_id="candidate", metrics=candidate_metrics)
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    write_analysis_artifacts(baseline, baseline_dir)
    write_analysis_artifacts(candidate, candidate_dir)

    outputs = compare_run_packages(baseline_dir, candidate_dir)
    payload = json.loads(Path(outputs["comparison_json"]).read_text(encoding="utf-8"))
    altitude = next(
        item for item in payload["metrics"] if item["metric_id"] == "orbit.altitude.minimum"
    )
    assert altitude["delta"] == pytest.approx(250.0)
    assert altitude["comparable"] is True
    assert Path(outputs["comparison_csv"]).is_file()
    assert Path(outputs["comparison_markdown"]).is_file()


def test_canonical_package_writes_all_formats_from_one_contract(tmp_path) -> None:
    analysis = build_orbit_analysis(
        _circular_result(samples=33),
        config=_config(),
        meta={"mu_m3s2": MU, "body_radius_m": RADIUS_M},
        preset="quick",
        run_id="package_smoke",
    )

    outputs = generate_analysis_package(analysis, tmp_path)

    assert outputs["status"] == "success"
    for relative in (
        "config.json",
        "diagnostics.json",
        "provenance.json",
        "metrics.json",
        "events.csv",
        "orbital_elements.csv",
        "force_budget.csv",
        "report.md",
        "report.pdf",
        "tables/summary.csv",
        "tables/extrema.csv",
        "tables/integrator.csv",
        "tables/provenance.csv",
        "figures/orbit_overview.png",
        "figures/altitude_history.png",
        "figures/orbital_elements.png",
        "figures/orbit_envelope.png",
        "figures/spatial_context.png",
        "figures/numerical_health.png",
        "figures/force_ric.png",
        "figures/event_timeline.png",
        "artifact_manifest.json",
    ):
        path = tmp_path / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 0, relative

    metrics_text = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert '"value": NaN' not in metrics_text
    assert '"value": Infinity' not in metrics_text
    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## 1. Executive Mission Summary" in report_text
    assert "Circular singularity" in report_text

    pdf_bytes = (tmp_path / "report.pdf").read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_bytes.count(b"/Type /Page") >= 5

    loaded = load_analysis_artifacts(tmp_path)
    assert loaded.run_id == analysis.run_id
    assert loaded.series.state_m_mps.shape == analysis.series.state_m_mps.shape
    assert loaded.metric_map["orbit.altitude.minimum"].value == pytest.approx(
        analysis.metric_map["orbit.altitude.minimum"].value
    )
