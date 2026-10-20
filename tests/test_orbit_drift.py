"""Analytic, model-independent tests for the ST-LRPS orbit-drift harness."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from lunaris.surrogate.st_lrps.evaluation.orbit_drift import (
    circular_orbit_state,
    energy_drift,
    orbit_drift,
    propagate_orbit,
)

MU = 4.9048695e12  # Moon GM [m^3/s^2]
RADIUS = 1.938e6  # ~200 km altitude [m]


def _kepler(mu: float):
    def accel(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        return -mu * r / rn ** 3
    return accel


def _kepler_potential(mu: float):
    def potential(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        return float(-mu / np.linalg.norm(r))
    return potential


def _period(mu: float, radius: float) -> float:
    return 2.0 * np.pi * np.sqrt(radius ** 3 / mu)


def test_propagate_circular_orbit_closes_and_conserves_energy():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    period = _period(MU, RADIUS)
    n = 2000
    orbit = propagate_orbit(_kepler(MU), r0, v0, dt_s=period / n, n_steps=n)

    # After one period the orbit should return close to its start.
    closure = np.linalg.norm(orbit["positions"][-1] - r0)
    assert closure / RADIUS < 1e-4

    # And RK4 should conserve mechanical energy to truncation floor.
    e = energy_drift(_kepler(MU), _kepler_potential(MU), r0, v0, dt_s=period / n, n_steps=n)
    assert e["relative_energy_drift_max_abs"] < 1e-6


def test_orbit_drift_identical_models_is_zero():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    period = _period(MU, RADIUS)
    out = orbit_drift(_kepler(MU), _kepler(MU), r0, v0, dt_s=period / 500, n_steps=500)
    assert out["position_drift_summary_m"]["max"] == 0.0
    assert out["velocity_drift_summary_m_s"]["max"] == 0.0
    assert out["position_drift_rel_max"] == 0.0


def test_orbit_drift_grows_for_mismatched_models():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    period = _period(MU, RADIUS)
    n = 1500  # 3 orbits
    out = orbit_drift(
        _kepler(MU), _kepler(MU * 1.001), r0, v0, dt_s=3 * period / n, n_steps=n,
    )
    drift = out["position_drift_m"]
    assert drift[0] == 0.0
    assert out["position_drift_summary_m"]["final"] > 0.0
    # Divergence accumulates: the last quarter drifts more than the first quarter.
    assert drift[-1] > drift[len(drift) // 4]


def test_energy_drift_detects_nonconservative_field():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    period = _period(MU, RADIUS)
    n = 4000  # 5 orbits
    dt = 5 * period / n

    pure = energy_drift(_kepler(MU), _kepler_potential(MU), r0, v0, dt_s=dt, n_steps=n)

    # Add a position-only rotational (non-conservative) perturbation a = eps*(-y, x, 0).
    eps = 1.0e-9

    def nonconservative(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        base = -MU * r / rn ** 3
        rot = np.zeros_like(r)
        rot[:, 0] = -eps * r[:, 1]
        rot[:, 1] = eps * r[:, 0]
        return base + rot

    drifted = energy_drift(nonconservative, _kepler_potential(MU), r0, v0, dt_s=dt, n_steps=n)

    # The conservative field stays at the integrator floor; the rotational field
    # pumps energy secularly and is unmistakably larger.
    assert pure["relative_energy_drift_max_abs"] < 1e-6
    assert drifted["relative_energy_drift_max_abs"] > 1e-4
    assert drifted["relative_energy_drift_max_abs"] > 100.0 * pure["relative_energy_drift_max_abs"]


def test_propagate_orbit_rejects_bad_input():
    r0, v0 = circular_orbit_state(MU, RADIUS)
    with pytest.raises(ValueError, match="dt_s must be positive"):
        propagate_orbit(_kepler(MU), r0, v0, dt_s=0.0, n_steps=10)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        propagate_orbit(_kepler(MU), r0, v0, dt_s=1.0, n_steps=0)
    with pytest.raises(ValueError, match="3 elements"):
        propagate_orbit(_kepler(MU), [1.0, 2.0], v0, dt_s=1.0, n_steps=10)


def test_circular_orbit_state_validation():
    with pytest.raises(ValueError, match="must be positive"):
        circular_orbit_state(-1.0, RADIUS)
    r0, v0 = circular_orbit_state(MU, RADIUS)
    # Circular speed: |v| = sqrt(mu / r), perpendicular to r.
    np.testing.assert_allclose(np.linalg.norm(v0), np.sqrt(MU / RADIUS), rtol=1e-12)
    assert abs(float(np.dot(r0, v0))) < 1e-6


class _TotalOnlyRuntime:
    """Runtime whose total differs from any point-mass-plus-residual reconstruction."""

    def domain_status(self, r):
        return {"recommended_fallback": False, "reason": "ok"}

    def predict_total_accel_fixed(self, r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        return np.full_like(r, 7.0e-3)

    def predict_residual_accel_fixed(self, r):  # pragma: no cover - must not be used
        raise AssertionError("orbit_drift must take the runtime's total, not rebuild it")


def test_domain_tracking_total_accel_returns_the_runtimes_total_unchanged():
    """Ported from Sidera 4a8784c1: no point mass is added on top of the runtime total.

    The old wrapper returned ``-mu r/r^3 + residual`` for every artifact, counting
    the monopole twice for a full-field artifact and dropping degrees
    2..base_degree for an SH-baseline residual artifact.
    """
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _DomainTrackingTotalAccel

    accel = _DomainTrackingTotalAccel(_TotalOnlyRuntime(), strict_domain=False)
    out = accel(np.array([[RADIUS, 0.0, 0.0], [0.0, RADIUS, 0.0]]))
    np.testing.assert_array_equal(out, np.full((2, 3), 7.0e-3))
    assert accel.domain_report()["domain_check_failures"] == 0


class _BrokenDomainRuntime(_TotalOnlyRuntime):
    def domain_status(self, r):
        raise KeyError("domain_contract")


def test_failed_domain_check_is_counted_not_read_as_in_domain():
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _DomainTrackingTotalAccel

    accel = _DomainTrackingTotalAccel(_BrokenDomainRuntime(), strict_domain=False)
    out = accel(np.array([[RADIUS, 0.0, 0.0], [0.0, RADIUS, 0.0]]))
    np.testing.assert_array_equal(out, np.full((2, 3), 7.0e-3))
    report = accel.domain_report()
    assert report["evaluations"] == 2
    assert report["domain_check_failures"] == 2
    assert report["extrapolating_evaluations"] == 0


def test_failed_domain_check_raises_under_strict_domain():
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _DomainTrackingTotalAccel

    accel = _DomainTrackingTotalAccel(_BrokenDomainRuntime(), strict_domain=True)
    with pytest.raises(RuntimeError, match="domain_status"):
        accel(np.array([[RADIUS, 0.0, 0.0]]))


def test_orbit_drift_runtime_uses_the_artifacts_own_base(tmp_path):
    """Real artifacts: a full-field model gets no extra monopole; an SH baseline
    without its gravity file is refused."""
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _runtime_total_accel
    from st_lrps_contract_test_utils import make_contract_run

    r = np.array([[RADIUS, 0.0, 0.0]])  # Moon-fixed, m

    # Residual over an SH degree-20 baseline whose recorded source is not a
    # file here: refused at load instead of silently using a point mass.
    sh_run = make_contract_run(tmp_path / "sh", degree_min=20, degree_max=60)
    with pytest.raises(FileNotFoundError, match="--gravity-file"):
        _runtime_total_accel(str(sh_run["run_dir"]), "cpu")

    # Without a caller base the runtime itself still refuses the SH baseline.
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _DomainTrackingTotalAccel
    from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model

    runtime = load_surrogate_force_model(str(sh_run["run_dir"]), device="cpu")
    with pytest.raises(ValueError, match="base_accel_fn"):
        _DomainTrackingTotalAccel(runtime)(r)

    # Full field with no baseline: the zero-weight network's output is the total.
    # (The fixture builder only writes residual datasets, so the loaded runtime's
    # target contract is switched to the full-field semantics it would carry.)
    import dataclasses

    full_accel = _DomainTrackingTotalAccel(runtime)
    mu = float(runtime.artifact_contract.mu_si)
    full_accel.runtime.target_contract = dataclasses.replace(
        full_accel.runtime.target_contract, target_mode="full", baseline_kind="none"
    )
    a_net = np.asarray(
        full_accel.runtime.predict_residual_accel_fixed(r), dtype=np.float64
    ).reshape(-1, 3)
    out = full_accel(r)
    np.testing.assert_allclose(out, a_net, rtol=0, atol=1e-12)
    # The old wrapper added mu/r^2 ~ 1.3 m/s^2 here; the total must not carry it.
    assert np.max(np.abs(out)) < 1e-6 * mu / RADIUS ** 2

    # Point-mass base: exactly one monopole, -mu r / r^3.
    full_accel.runtime.target_contract = dataclasses.replace(
        full_accel.runtime.target_contract, baseline_kind="point_mass"
    )
    np.testing.assert_allclose(
        full_accel(r), a_net - full_accel.runtime.mu_si * r / RADIUS ** 3, rtol=1e-12, atol=0
    )


def _write_shadr(path, *, degree, gm_m3s2, radius_m, degree1=0.0):
    """Small SHADR-format field: degree-1 terms ``degree1``, deterministic higher terms."""
    rows = [f"{radius_m / 1e3!r}, {gm_m3s2 / 1e9!r}, 0, {degree}, {degree}, 1, 0, 0"]
    for n in range(1, degree + 1):
        for m in range(n + 1):
            c = degree1 if n == 1 else (-1.0) ** (n + m) * 9.0e-5 / (n * n + m + 1)
            s = 0.0 if m == 0 else (degree1 if n == 1 else 4.0e-5 / (n * n + 2 * m))
            rows.append(f"{n}, {m}, {c!r}, {s!r}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _stub_runtime(*, base_degree, dataset_contract, mu, r_ref, baseline_kind="spherical_harmonics"):
    from types import SimpleNamespace

    return SimpleNamespace(
        target_contract=SimpleNamespace(baseline_kind=baseline_kind, base_degree=base_degree),
        artifact_contract=SimpleNamespace(
            dataset_contract=dict(dataset_contract), mu_si=mu, r_ref_m=r_ref
        ),
    )


def test_sh_base_matches_the_generator_label_convention(tmp_path):
    """SH(base) + labels (base, target] is the full field: the base the residual excludes."""
    import hashlib

    from lunaris.physics.spherical_harmonics import (
        GravityModel,
        precompute_legendre_constants,
        sh_potential_accel_batch_serial,
    )
    from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import sh_base_accel_from_artifact

    # Nonzero degree-1 terms: the base must carry degrees 1..base_degree, not 2..
    path = _write_shadr(
        tmp_path / "field.tab", degree=8, gm_m3s2=MU_MOON_SI, radius_m=R_MOON_SI, degree1=2.0e-6
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = _stub_runtime(
        base_degree=3,
        dataset_contract={"source_gravity_file_path": str(path), "source_gravity_file_sha256": sha},
        mu=MU_MOON_SI,
        r_ref=R_MOON_SI,
    )
    base_fn, info = sh_base_accel_from_artifact(runtime)
    assert info["base_source"] == "gravity_file" and info["sha256_verified"] is True

    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(16, 3))
    r = RADIUS * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    full = GravityModel.from_file(str(path), requested_degree=8)
    tables = precompute_legendre_constants(8)

    def kernel(degree_min):
        return sh_potential_accel_batch_serial(
            r, full.c_coeffs, full.s_coeffs, *tables, full.mu, full.r_ref, 8, degree_min
        )[1]

    base = base_fn(r)
    truth, labels = kernel(-1), kernel(3)
    scale = float(np.max(np.linalg.norm(truth, axis=1)))
    np.testing.assert_allclose(base + labels, truth, rtol=0, atol=1e-14 * scale)
    # The degree-2..3 terms a point-mass base drops are present and not negligible.
    point_mass = -MU_MOON_SI * r / np.linalg.norm(r, axis=1, keepdims=True) ** 3
    assert np.min(np.linalg.norm(base - point_mass, axis=1)) > 1e-7 * scale


def test_sh_base_refuses_a_file_that_is_not_the_labels_field(tmp_path):
    import hashlib

    from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import sh_base_accel_from_artifact

    path = _write_shadr(tmp_path / "field.tab", degree=4, gm_m3s2=MU_MOON_SI, radius_m=R_MOON_SI)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    def runtime(**contract):
        return _stub_runtime(base_degree=3, dataset_contract=contract, mu=MU_MOON_SI, r_ref=R_MOON_SI)

    with pytest.raises(ValueError, match="--gravity-file"):
        sh_base_accel_from_artifact(runtime())
    with pytest.raises(FileNotFoundError, match="--gravity-file"):
        sh_base_accel_from_artifact(runtime(source_gravity_file_path=str(tmp_path / "elsewhere.tab")))
    with pytest.raises(ValueError, match="sha256"):
        sh_base_accel_from_artifact(
            runtime(source_gravity_file_path=str(path), source_gravity_file_sha256="1" * 64)
        )
    # The override replaces a path from another machine; the sha still has to match.
    _, info = sh_base_accel_from_artifact(
        runtime(source_gravity_file_path="/cluster/data/field.tab", source_gravity_file_sha256=sha),
        gravity_file=path,
    )
    assert info["gravity_file_overridden"] is True and info["sha256_verified"] is True

    other_gm = _write_shadr(
        tmp_path / "other_gm.tab", degree=4, gm_m3s2=MU_MOON_SI * 1.001, radius_m=R_MOON_SI
    )
    with pytest.raises(ValueError, match="GM"):
        sh_base_accel_from_artifact(runtime(), gravity_file=other_gm)
    other_radius = _write_shadr(
        tmp_path / "other_r.tab", degree=4, gm_m3s2=MU_MOON_SI, radius_m=R_MOON_SI + 1000.0
    )
    with pytest.raises(ValueError, match="reference radius"):
        sh_base_accel_from_artifact(runtime(), gravity_file=other_radius)

    # A degree-0 or point-mass baseline is the runtime's own job.
    pm_runtime = _stub_runtime(
        base_degree=0, dataset_contract={}, mu=MU_MOON_SI, r_ref=R_MOON_SI
    )
    assert sh_base_accel_from_artifact(pm_runtime) == (
        None,
        {"baseline_kind": "spherical_harmonics", "base_degree": 0, "base_source": "runtime"},
    )


def test_orbit_drift_runtime_adds_the_sh_base_from_the_gravity_file(tmp_path):
    """Real artifact over an SH(3) baseline: total = SH(3) from the file + network output."""
    import hashlib

    from lunaris.physics.spherical_harmonics import GravityModel
    from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _runtime_total_accel
    from st_lrps_contract_test_utils import make_contract_run

    path = _write_shadr(tmp_path / "field.tab", degree=8, gm_m3s2=MU_MOON_SI, radius_m=R_MOON_SI)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    plain = make_contract_run(tmp_path / "plain", degree_min=3, degree_max=8)
    dataset_contract = {
        **plain["cfg"]["artifact_contract"]["dataset_contract"],
        "source_gravity_file_path": str(path),
        "source_gravity_file_sha256": sha,
    }
    run = make_contract_run(
        tmp_path / "sh3",
        degree_min=3,
        degree_max=8,
        contract_overrides={"dataset_contract": dataset_contract},
    )

    accel, _, _ = _runtime_total_accel(str(run["run_dir"]), "cpu")
    assert accel.base_info["sha256_verified"] is True
    r = np.array([[RADIUS, 0.0, 0.0], [0.0, 0.6 * RADIUS, 0.8 * RADIUS]])
    net = np.asarray(accel.runtime.predict_residual_accel_fixed(r), dtype=np.float64)
    model = GravityModel.from_file(str(path), requested_degree=3)
    base = np.stack([model.accel_fixed(p, degree=3) for p in r])
    np.testing.assert_allclose(accel(r), base + net, rtol=1e-12, atol=0)
