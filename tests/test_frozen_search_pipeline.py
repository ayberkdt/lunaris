"""Sprint 5 frozen-orbit search: domain guard (R27), family report (R30),
refinement (R22), R21 enforcement, and the staged pipeline (R04).

The pipeline tests inject analytic two-body (Kepler) propagators, so orbital
elements are exactly constant: clean Kepler orbits classify as strict/candidate
frozen, and the tests stay backend- and torch-independent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lunaris.analysis.frozen.classify import (
    QUASI_FROZEN,
    STRICT_FROZEN,
    FrozenClassificationConfig,
    classify_candidate,
)
from lunaris.analysis.frozen.domain_guard import (
    DOMAIN_POLICY_LOW_CONFIDENCE,
    FrozenSearchDomainGuard,
    apply_domain_guard_to_scores,
    assert_candidate_domain_clean,
    evaluate_domain_guard,
)
from lunaris.analysis.frozen.family_report import (
    FAMILY_REPORT_SCHEMA_VERSION,
    build_family_report,
    group_candidates_into_families,
    validate_family_report,
)
from lunaris.analysis.frozen.metrics import compute_frozen_metrics
from lunaris.analysis.frozen.refine import (
    RefinementBounds,
    RefinementConfig,
    refine_candidate,
)
from lunaris.analysis.frozen.search import (
    STAGE0_SAMPLES,
    STAGE1_OUTPUT_FULL,
    STAGE1_OUTPUT_SUMMARY_ONLY,
    STAGE1_SCREENING,
    STAGE2_CANDIDATES,
    STAGE3_VALIDATION,
    STAGE4_FAMILIES,
    ElementBounds,
    FrozenSearchConfig,
    FrozenSearchPipeline,
    elements_to_states,
    enforce_classical_validation_rule,
    sobol_element_samples,
)

MU = 4.9028001224453001e12
R_REF = 1.7374e6


# ---------------------------------------------------------------------------
# Analytic Kepler batch propagator (test backend)
# ---------------------------------------------------------------------------


def _solve_kepler(M: np.ndarray, e: float) -> np.ndarray:
    E = M.copy()
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def _kepler_states(el: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Exact two-body states (T, 6) for one element row (km/deg)."""
    from lunaris.common.math_utils import coe_to_rv

    a_m = float(el[0]) * 1_000.0
    e = float(el[1])
    n_motion = np.sqrt(MU / a_m**3)
    nu0 = np.deg2rad(float(el[5]))
    E0 = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(nu0 / 2.0), np.sqrt(1.0 + e) * np.cos(nu0 / 2.0)
    )
    M0 = E0 - e * np.sin(E0)
    M = M0 + n_motion * t_s
    E = _solve_kepler(np.mod(M, 2.0 * np.pi), e)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0)
    )
    out = np.empty((t_s.size, 6), dtype=np.float64)
    for k in range(t_s.size):
        r, v = coe_to_rv(
            a_m,
            e,
            np.deg2rad(float(el[2])),
            np.deg2rad(float(el[3])),
            np.deg2rad(float(el[4])),
            float(nu[k]),
            MU,
        )
        out[k, :3] = r
        out[k, 3:] = v
    return out


class KeplerScreeningPropagator:
    """Analytic two-body screening backend (elements exactly constant)."""

    backend_name = "test_kepler_batch"
    provenance = {"backend": "test_kepler_batch", "physics": "two-body analytic"}

    def propagate(self, Y0, duration_s, output_dt_s):
        from lunaris.core.state import cartesian_to_keplerian

        n_snaps = int(round(duration_s / output_dt_s))
        t_out = np.linspace(0.0, duration_s, n_snaps + 1)
        N = Y0.shape[0]
        Y_out = np.empty((t_out.size, N, 6), dtype=np.float64)
        for j in range(N):
            a, e, inc, raan, argp, ta = cartesian_to_keplerian(Y0[j, :3], Y0[j, 3:], mu=MU)
            el = np.array(
                [
                    a / 1_000.0,
                    e,
                    np.rad2deg(inc),
                    np.rad2deg(raan),
                    np.rad2deg(argp),
                    np.rad2deg(ta),
                ]
            )
            Y_out[:, j, :] = _kepler_states(el, t_out)
        impact = np.zeros(N, dtype=np.float64)
        t_imp = np.full(N, np.nan, dtype=np.float64)
        return t_out, Y_out, impact, t_imp


class KeplerValidationPropagator:
    """Analytic two-body validation backend with a configurable label."""

    def __init__(self, backend_label: str = "classical_sh_deg8") -> None:
        self.backend_label = backend_label
        self.provenance = {
            "backend": backend_label,
            "gravity_model": {"name": "test", "degree": 8},
            "third_body": {"earth": False, "sun": False},
        }
        self._screen = KeplerScreeningPropagator()

    def propagate(self, y0, duration_s, output_dt_s):
        t, Y, _imp, _t_imp = self._screen.propagate(
            np.asarray(y0)[None, :], duration_s, output_dt_s
        )
        return t, Y[:, 0, :]


def _bounds() -> ElementBounds:
    return ElementBounds(a_km=(2_000.0, 2_400.0), e=(0.01, 0.08), i_deg=(70.0, 110.0))


def _config(tmp_kwargs=None, **overrides) -> FrozenSearchConfig:
    kwargs = dict(
        bounds=_bounds(),
        n_samples=16,
        seed=7,
        screening_duration_s=6 * 3_600.0,
        screening_output_dt_s=600.0,
        top_k=4,
        validation_duration_s=12 * 3_600.0,
        validation_output_dt_s=600.0,
        guard=FrozenSearchDomainGuard(altitude_min_km=20.0, altitude_max_km=20_000.0),
        perilune_safety_min_m=10_000.0,
        mu_m3s2=MU,
        reference_radius_m=R_REF,
    )
    kwargs.update(overrides)
    return FrozenSearchConfig(**kwargs)


# ---------------------------------------------------------------------------
# R27 — domain guard
# ---------------------------------------------------------------------------


def _circular_block(radius_m: float, n_t: int = 5) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 3_600.0, n_t)
    v = np.sqrt(MU / radius_m)
    Y = np.zeros((n_t, 1, 6))
    Y[:, 0, 0] = radius_m
    Y[:, 0, 4] = v
    return t, Y


def test_domain_guard_flags_altitude_exit_below_and_above():
    guard = FrozenSearchDomainGuard(altitude_min_km=50.0, altitude_max_km=500.0)
    t, Y_low = _circular_block(R_REF + 30_000.0)   # 30 km < 50 km floor
    result = evaluate_domain_guard(t, Y_low, reference_radius_m=R_REF, guard=guard)
    assert bool(result.domain_exit_flag[0])
    assert "below" in str(result.reasons[0])
    assert result.t_domain_exit_s[0] == t[0]

    _, Y_high = _circular_block(R_REF + 600_000.0)  # 600 km > 500 km ceiling
    result = evaluate_domain_guard(t, Y_high, reference_radius_m=R_REF, guard=guard)
    assert bool(result.domain_exit_flag[0])
    assert "above" in str(result.reasons[0])


def test_domain_guard_flags_nonfinite_and_escape():
    guard = FrozenSearchDomainGuard(
        altitude_min_km=10.0, altitude_max_km=1e6, escape_radius_km=5_000.0
    )
    t, Y = _circular_block(R_REF + 100_000.0)
    Y_nan = Y.copy()
    Y_nan[3, 0, 2] = np.nan
    result = evaluate_domain_guard(t, Y_nan, reference_radius_m=R_REF, guard=guard)
    assert bool(result.nonfinite_flag[0])
    assert result.t_domain_exit_s[0] == t[3]

    _, Y_far = _circular_block(6_000_000.0)  # radius 6000 km > 5000 km escape proxy
    result = evaluate_domain_guard(t, Y_far, reference_radius_m=R_REF, guard=guard)
    assert bool(result.escape_flag[0])
    assert "escape" in str(result.reasons[0])


def test_domain_guard_ignores_post_impact_rows():
    guard = FrozenSearchDomainGuard(altitude_min_km=50.0, altitude_max_km=500.0)
    t, Y = _circular_block(R_REF + 100_000.0)
    # After the recorded impact the state freezes at the surface (alt ~ 0 < floor).
    Y[3:, 0, 0] = R_REF
    result = evaluate_domain_guard(
        t,
        Y,
        reference_radius_m=R_REF,
        guard=guard,
        impact_flags=np.array([1.0]),
        t_impact_s=np.array([t[2]]),
    )
    assert not bool(result.domain_exit_flag[0])


def test_domain_guard_score_policies():
    guard_invalid = FrozenSearchDomainGuard(altitude_min_km=20.0, altitude_max_km=100.0)
    guard_soft = FrozenSearchDomainGuard(
        altitude_min_km=20.0,
        altitude_max_km=100.0,
        policy=DOMAIN_POLICY_LOW_CONFIDENCE,
        domain_exit_penalty=42.0,
    )
    t, Y = _circular_block(R_REF + 300_000.0)  # above the 100 km ceiling
    result = evaluate_domain_guard(t, Y, reference_radius_m=R_REF, guard=guard_invalid)
    scores = np.array([1.5])
    assert np.isinf(apply_domain_guard_to_scores(scores, result, guard_invalid)[0])
    assert apply_domain_guard_to_scores(scores, result, guard_soft)[0] == pytest.approx(43.5)


def test_domain_exited_candidate_cannot_be_final():
    with pytest.raises(RuntimeError, match="cannot be promoted"):
        assert_candidate_domain_clean(
            {"domain_guard": {"domain_exit": True, "domain_exit_reason": "above"}}
        )
    with pytest.raises(RuntimeError, match="requires the domain guard"):
        assert_candidate_domain_clean({})
    assert_candidate_domain_clean({"domain_guard": {"domain_exit": False}})


# ---------------------------------------------------------------------------
# R30 — family report schema
# ---------------------------------------------------------------------------


def _family(**overrides):
    fam = {
        "family_id": "F001",
        "status": "quasi_frozen_candidate",
        "screening_backend": "torch_cpu_sh",
        "validation_backend": "classical_sh_deg50",
        "gravity_model": {"name": "jggrx", "degree": 50},
        "third_body": {"earth": False, "sun": False},
        "validation_days": 30.0,
        "member_count": 1,
        "member_sample_indices": [3],
        "element_ranges": {
            "a_km": [2_000.0, 2_100.0],
            "e": [0.02, 0.03],
            "i_deg": [85.0, 86.0],
            "argp_deg": [88.0, 92.0],
        },
        "stability_metrics": {"score_min": 0.4},
        "provenance": {"sobol_seed": 7, "sample_count": 16},
    }
    fam.update(overrides)
    return fam


def test_family_report_roundtrip_and_schema_version():
    report = build_family_report(
        run_id="testrun", score_definition="score v1", families=[_family()]
    )
    assert report["schema_version"] == FAMILY_REPORT_SCHEMA_VERSION
    validate_family_report(report)  # no raise


@pytest.mark.parametrize(
    "mutation, match",
    [
        ({"member_count": 2}, "member_count"),
        ({"element_ranges": {"a_km": [2.0, 1.0], "e": [0, 1], "i_deg": [0, 1], "argp_deg": [0, 1]}}, "lo > hi"),
        ({"status": None}, "must not be null"),
    ],
)
def test_family_report_rejects_bad_fields(mutation, match):
    report = build_family_report(
        run_id="testrun", score_definition="score v1", families=[_family()]
    )
    report["families"][0].update(mutation)
    with pytest.raises(ValueError, match=match):
        validate_family_report(report)


def test_family_report_enforces_r21_language_rule():
    """A validated frozen status without a classical SH backend is a schema error."""
    bad = _family(status=QUASI_FROZEN, validation_backend="gpu_st_lrps_potential")
    with pytest.raises(ValueError, match="classical SH"):
        build_family_report(run_id="t", score_definition="s", families=[bad])
    ok = _family(status=QUASI_FROZEN, validation_backend="classical_sh_deg100")
    build_family_report(run_id="t", score_definition="s", families=[ok])


def test_group_candidates_uses_weakest_member_status():
    members = []
    for idx, status in enumerate(["strict_frozen", "quasi_frozen_candidate"]):
        members.append(
            {
                "sample_index": idx,
                "elements": {"a_km": 2_050.0, "e": 0.02, "i_deg": 85.0, "argp_deg": 90.0,
                             "raan_deg": 0.0, "ta_deg": 0.0},
                "classification": {
                    "status": status,
                    "score": 0.5,
                    "validation_backend": "classical_sh_deg50" if idx == 0 else None,
                },
                "metrics": {"e_range": 1e-4},
                "validation": {"duration_days": 30.0},
            }
        )
    families = group_candidates_into_families(
        members,
        screening_backend="torch_cpu_sh",
        gravity_model={"name": "jggrx"},
        third_body={"earth": False, "sun": False},
        provenance={"sobol_seed": 1},
    )
    assert len(families) == 1
    assert families[0]["status"] == "quasi_frozen_candidate"
    assert families[0]["member_count"] == 2


# ---------------------------------------------------------------------------
# R22 — refinement
# ---------------------------------------------------------------------------


def _refine_bounds() -> RefinementBounds:
    return RefinementBounds(
        a_km=(2_000.0, 2_400.0),
        e=(0.0, 0.2),
        i_deg=(70.0, 110.0),
        raan_deg=(0.0, 360.0),
        argp_deg=(0.0, 360.0),
        ta_deg=(0.0, 360.0),
    )


def test_refine_quadratic_objective_improves_toward_minimum():
    target = np.array([2_200.0, 0.05, 90.0, 180.0, 90.0, 0.0])

    def score_fn(x):
        return float(np.sum(((x - target) / (target + 1.0)) ** 2))

    x0 = {"a_km": 2_100.0, "e": 0.08, "i_deg": 80.0, "raan_deg": 170.0,
          "argp_deg": 80.0, "ta_deg": 10.0}
    result = refine_candidate(
        x0, score_fn, RefinementConfig(bounds=_refine_bounds(), max_iterations=400)
    )
    assert result.improved
    assert result.refined_score < result.original_score
    assert result.refined_elements["a_km"] == pytest.approx(2_200.0, abs=20.0)
    assert result.validation_status == "requires_classical_sh_validation"
    assert result.optimizer_metadata["n_evaluations"] > 0


def test_refine_differential_evolution_backend():
    target = np.array([2_200.0, 0.05, 90.0, 180.0, 90.0, 0.0])

    def score_fn(x):
        return float(np.sum(((x - target) / (target + 1.0)) ** 2))

    x0 = {"a_km": 2_050.0, "e": 0.15, "i_deg": 75.0, "raan_deg": 10.0,
          "argp_deg": 300.0, "ta_deg": 200.0}
    result = refine_candidate(
        x0,
        score_fn,
        RefinementConfig(
            bounds=_refine_bounds(),
            optimizer="differential_evolution",
            max_iterations=30,
            seed=3,
            extra_options={"popsize": 8, "tol": 1e-6},
        ),
    )
    assert result.refined_score <= result.original_score


def test_refine_never_returns_worse_than_start_and_penalizes_infeasible():
    calls = {"n": 0}

    def hostile_fn(x):
        calls["n"] += 1
        return float("inf")  # everything infeasible

    x0 = {"a_km": 2_100.0, "e": 0.05, "i_deg": 90.0, "raan_deg": 0.0,
          "argp_deg": 0.0, "ta_deg": 0.0}
    result = refine_candidate(
        x0, hostile_fn, RefinementConfig(bounds=_refine_bounds(), max_iterations=10)
    )
    assert result.refined_elements == result.original_elements
    assert not result.improved


def test_refine_rejects_out_of_bounds_start():
    with pytest.raises(ValueError, match="outside the refinement bounds"):
        refine_candidate(
            {"a_km": 9_999.0, "e": 0.05, "i_deg": 90.0, "raan_deg": 0.0,
             "argp_deg": 0.0, "ta_deg": 0.0},
            lambda x: 1.0,
            RefinementConfig(bounds=_refine_bounds()),
        )


# ---------------------------------------------------------------------------
# R21 — enforcement
# ---------------------------------------------------------------------------


def test_enforce_classical_validation_rule_blocks_non_classical():
    record = {
        "classification": {
            "status": STRICT_FROZEN,
            "validated": True,
            "validation_backend": "gpu_st_lrps_potential",
        }
    }
    with pytest.raises(RuntimeError, match="classical SH"):
        enforce_classical_validation_rule(record)

    record["classification"]["validation_backend"] = "classical_sh_deg100"
    enforce_classical_validation_rule(record)  # no raise

    record["classification"]["validated"] = False
    with pytest.raises(RuntimeError, match="validated=True"):
        enforce_classical_validation_rule(record)


def test_classifier_never_grants_frozen_status_without_classical_backend():
    """Belt: classify_candidate itself refuses (upstream of the choke point)."""
    t = np.linspace(0.0, 86_400.0, 25)
    metrics = compute_frozen_metrics(
        t,
        eccentricity=np.full(25, 0.02),
        inclination_rad=np.full(25, np.deg2rad(85.0)),
        omega_rad=np.full(25, np.deg2rad(90.0)),
        h_peri_m=np.full(25, 150_000.0),
    )
    config = FrozenClassificationConfig.for_mission_duration(86_400.0)
    via_surrogate = classify_candidate(
        metrics, config,
        validation_backend="gpu_st_lrps_potential",
        long_horizon_validation_passed=True,
    )
    assert via_surrogate.status not in (STRICT_FROZEN, QUASI_FROZEN)
    assert not via_surrogate.validated


def test_third_body_selector_normalization():
    from lunaris.analysis.frozen.search_backends import normalize_third_body_selection

    assert normalize_third_body_selection("none") == ()
    assert normalize_third_body_selection("sun,earth") == ("sun", "earth")
    assert normalize_third_body_selection("earth+sun") == ("sun", "earth")
    assert normalize_third_body_selection(True) == ("sun", "earth")
    with pytest.raises(ValueError, match="third-body selector"):
        normalize_third_body_selection("mars")


def test_classical_validation_third_body_requires_ephemeris(monkeypatch):
    import lunaris.analysis.frozen.search_backends as backends

    monkeypatch.setattr(backends, "_load_gravity_model", lambda *_args, **_kwargs: object())
    with pytest.raises(RuntimeError, match="ephemeris-wired"):
        backends.ClassicalSHValidationPropagator(degree=8, third_body="sun")


def test_classical_validation_third_body_wires_flags_and_provenance(monkeypatch):
    import lunaris.analysis.frozen.search_backends as backends
    import lunaris.core.dynamics as dynamics_mod

    class DummyEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(backends, "_load_gravity_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dynamics_mod, "DynamicsEngine", DummyEngine)
    ephem = object()
    prop = backends.ClassicalSHValidationPropagator(
        degree=8,
        third_body="sun,earth",
        ephem_manager=ephem,
    )

    assert prop.backend_label == "classical_sh_deg8_3b_sun_earth"
    assert prop.provenance["third_body"] == {"earth": True, "sun": True}
    assert "ephemeris-wired" in prop.provenance["frame"]
    flags = prop._engine.kwargs["flags"]
    assert flags.enable_3rd_body_sun is True
    assert flags.enable_3rd_body_earth is True
    assert prop._engine.kwargs["ephem_manager"] is ephem


# ---------------------------------------------------------------------------
# R04 — staged pipeline end-to-end (analytic backends)
# ---------------------------------------------------------------------------


def test_sobol_samples_are_deterministic_with_provenance():
    b = _bounds()
    el1, prov1 = sobol_element_samples(b, 8, seed=11)
    el2, _ = sobol_element_samples(b, 8, seed=11)
    el3, _ = sobol_element_samples(b, 8, seed=12)
    np.testing.assert_array_equal(el1, el2)
    assert not np.array_equal(el1, el3)
    assert prov1["sobol_seed"] == 11
    assert prov1["sample_count"] == 8
    assert prov1["sampling_method"] == "sobol_scrambled"
    box = b.as_array()
    assert np.all(el1 >= box[:, 0]) and np.all(el1 <= box[:, 1])


def test_elements_to_states_roundtrip():
    el, _ = sobol_element_samples(_bounds(), 4, seed=1)
    states = elements_to_states(el, MU)
    from lunaris.core.state import cartesian_to_keplerian

    for j in range(4):
        a, e, _inc, _raan, _argp, _ta = cartesian_to_keplerian(
            states[j, :3], states[j, 3:], mu=MU
        )
        assert a / 1_000.0 == pytest.approx(el[j, 0], rel=1e-9)
        assert e == pytest.approx(el[j, 1], abs=1e-9)


def test_pipeline_end_to_end_with_analytic_backends(tmp_path):
    pipeline = FrozenSearchPipeline(
        _config(),
        screening=KeplerScreeningPropagator(),
        validation=KeplerValidationPropagator("classical_sh_deg8"),
        out_dir=tmp_path,
        sensitivity_validations=[KeplerValidationPropagator("classical_sh_deg16")],
    )
    products = pipeline.run(resume=True)

    # File contracts exist.
    for name in (STAGE0_SAMPLES, STAGE2_CANDIDATES, STAGE3_VALIDATION, STAGE4_FAMILIES,
                 "manifest.json", STAGE1_SCREENING):
        assert (tmp_path / name).exists(), name

    with np.load(tmp_path / STAGE1_SCREENING, allow_pickle=False) as stage1:
        assert str(stage1["output_mode"].item()) == STAGE1_OUTPUT_SUMMARY_ONLY
        assert "Y_out" not in stage1.files
        assert "summary__score" in stage1.files
        assert "topk_Y_out" in stage1.files
        assert stage1["topk_Y_out"].shape[1] <= 4

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sampling_provenance"]["sobol_seed"] == 7
    assert manifest["sampling_provenance"]["sample_count"] == 16
    assert manifest["stages"]["stage1"]["output_mode"] == STAGE1_OUTPUT_SUMMARY_ONLY
    assert manifest["stages"]["stage1"]["full_trajectory_stored"] is False
    assert set(manifest["stages"]) == {"stage0", "stage1", "stage2", "stage3", "stage4"}

    candidates = products["candidates"]
    assert 1 <= len(candidates) <= 4
    for record in candidates:
        assert record["domain_guard"]["domain_exit"] is False  # R27 ran

    validated = products["validated"]
    assert len(validated) == len(candidates)
    for record in validated:
        status = record["classification"]["status"]
        # Kepler orbits are exactly frozen; classical label => validated statuses.
        assert status in (STRICT_FROZEN, QUASI_FROZEN)
        assert record["classification"]["validated"] is True
        assert record["sensitivity"][0]["status_agrees"] is True
        enforce_classical_validation_rule(record)

    report = products["family_report"]
    validate_family_report(report)
    assert len(report["families"]) >= 1


def test_pipeline_full_stage1_mode_keeps_legacy_trajectory_contract(tmp_path):
    pipeline = FrozenSearchPipeline(
        _config(screening_output_mode=STAGE1_OUTPUT_FULL),
        screening=KeplerScreeningPropagator(),
        validation=KeplerValidationPropagator("classical_sh_deg8"),
        out_dir=tmp_path,
    )
    products = pipeline.run(resume=True)
    assert len(products["candidates"]) >= 1

    with np.load(tmp_path / STAGE1_SCREENING, allow_pickle=False) as stage1:
        assert str(stage1["output_mode"].item()) == STAGE1_OUTPUT_FULL
        assert "Y_out" in stage1.files
        assert stage1["Y_out"].ndim == 3

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["stage1"]["full_trajectory_stored"] is True


def test_pipeline_resume_reuses_stage_files(tmp_path):
    config = _config()
    screening = KeplerScreeningPropagator()
    validation = KeplerValidationPropagator("classical_sh_deg8")
    pipeline = FrozenSearchPipeline(
        config, screening=screening, validation=validation, out_dir=tmp_path
    )
    pipeline.run(resume=True)

    class ExplodingScreening:
        backend_name = "must_not_run"
        provenance = {}

        def propagate(self, *args, **kwargs):
            raise AssertionError("stage 1 must be loaded from disk on resume")

    resumed = FrozenSearchPipeline(
        config, screening=ExplodingScreening(), validation=validation, out_dir=tmp_path
    )
    products = resumed.run(resume=True)  # must not call ExplodingScreening.propagate
    assert len(products["candidates"]) >= 1


def test_pipeline_negative_no_frozen_status_without_classical_backend(tmp_path):
    """G5 gate negative test: a surrogate-labeled validation backend can never
    produce strict_frozen / quasi_frozen statuses anywhere in the products."""
    pipeline = FrozenSearchPipeline(
        _config(),
        screening=KeplerScreeningPropagator(),
        validation=KeplerValidationPropagator("gpu_st_lrps_potential"),
        out_dir=tmp_path,
    )
    products = pipeline.run(resume=True)
    for record in products["validated"]:
        assert record["classification"]["status"] not in (STRICT_FROZEN, QUASI_FROZEN)
        assert record["classification"]["validated"] is False
    for family in products["family_report"]["families"]:
        assert family["status"] not in (STRICT_FROZEN, QUASI_FROZEN)


def test_pipeline_domain_guard_excludes_out_of_envelope_samples(tmp_path):
    """Samples propagating outside the guard envelope never become candidates."""
    config = _config(
        guard=FrozenSearchDomainGuard(altitude_min_km=20.0, altitude_max_km=280.0),
        top_k=16,
    )
    # Bounds put apolune well above 280 km for most samples: those must be
    # guard-excluded, and every surviving candidate must be domain-clean.
    pipeline = FrozenSearchPipeline(
        config,
        screening=KeplerScreeningPropagator(),
        validation=KeplerValidationPropagator("classical_sh_deg8"),
        out_dir=tmp_path,
    )
    products = pipeline.run(resume=True)
    stage2 = json.loads((tmp_path / STAGE2_CANDIDATES).read_text(encoding="utf-8"))
    assert stage2["domain_exit_count"] > 0
    for record in products["candidates"]:
        assert record["domain_guard"]["domain_exit"] is False


def test_pipeline_refinement_stage_runs_with_analytic_objective(tmp_path):
    config = _config(
        refine_top_n=1,
        refinement=RefinementConfig(bounds=_refine_bounds(), max_iterations=15),
    )
    pipeline = FrozenSearchPipeline(
        config,
        screening=KeplerScreeningPropagator(),
        validation=KeplerValidationPropagator("classical_sh_deg8"),
        out_dir=tmp_path,
    )
    products = pipeline.run(resume=True)
    refinements = products["family_report"]["refinements"]
    assert len(refinements) == 1
    assert refinements[0]["validation_status"] == "requires_classical_sh_validation"
    assert refinements[0]["refined_score"] <= refinements[0]["original_score"]
