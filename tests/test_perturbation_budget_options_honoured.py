"""Regression tests: the perturbation-budget CLI/config options must be honoured.

Each test pins a silent trap found in the pre-run audit:

1. ``--sh-degrees`` above the gravity file's maximum degree was clipped silently
   (SH30 == SH60 bit-for-bit) and the tool then recommended the wrong degree.
2. ``--epochs`` strings were hashed by character code, not used as times.
3. ``--include-earth-j2 on`` was dropped when ``--include-third-body off``.
4. CSVs did not say which gravity coefficients / geometry produced them.
5. Force magnitudes had no tight oracle (only decade bands).
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lunaris.analysis.perturbation_budget.acceleration_budget import (
    load_gravity_model_for_budget,
    non_gravity_vectors,
)
from lunaris.analysis.perturbation_budget.cli import main as budget_main
from lunaris.analysis.perturbation_budget.config import PerturbationBudgetConfig
from lunaris.analysis.perturbation_budget.reporting import run_perturbation_budget
from lunaris.analysis.perturbation_budget.sampling import SampleState, generate_sample_states
from lunaris.common.constants import AU, MU_EARTH, P_SUN_1AU
from lunaris.physics.third_body_effects import EarthJ2Params, calc_j2_oblate_diff_accel

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO_ROOT / "data" / "ephemeris_models"

_ALL_FORCE_FLAGS = (
    "include_srp",
    "include_albedo",
    "include_thermal_ir",
    "include_tides",
    "include_third_body",
    "include_relativity",
    "include_earth_j2",
)


def _cfg(**overrides: object) -> PerturbationBudgetConfig:
    """Tiny single-sample config with every force off unless overridden."""
    base: dict[str, object] = {
        "altitudes_km": [100.0],
        "inclinations_deg": [0.0],
        "true_anomalies_deg": [0.0],
        "epochs_utc": ["synthetic"],
        "sh_degrees": [2, 4],
        **dict.fromkeys(_ALL_FORCE_FLAGS, False),
    }
    base.update(overrides)
    return PerturbationBudgetConfig(**base)  # type: ignore[arg-type]


def _write_shadr(path: Path, degree: int) -> Path:
    rows = [
        f"{n}, {m}, {1e-6 / n:.6e}, {(5e-7 / n) if m else 0.0:.6e}"
        for n in range(1, degree + 1)
        for m in range(0, n + 1)
    ]
    path.write_text(f"1738.0, 4902.8, 0, {degree}, {degree}, 1, 0, 0\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sample(r_m: np.ndarray, sun_m: np.ndarray, earth_m: np.ndarray) -> SampleState:
    return SampleState(
        sample_id="s0",
        altitude_km=100.0,
        inclination_deg=0.0,
        true_anomaly_deg=0.0,
        epoch_utc="synthetic",
        r_m=np.asarray(r_m, dtype=np.float64),
        v_m_s=np.array([0.0, 1600.0, 0.0]),
        sun_m=np.asarray(sun_m, dtype=np.float64),
        earth_m=np.asarray(earth_m, dtype=np.float64),
        geometry_source="synthetic_geometry",
    )


# --------------------------------------------------------------------------- #
# 1. SH degrees above the model's maximum degree are refused
# --------------------------------------------------------------------------- #


def test_sh_degree_above_gravity_file_max_degree_is_refused(tmp_path: Path) -> None:
    path = _write_shadr(tmp_path / "deg4.tab", 4)
    cfg = _cfg(sh_degrees=[2, 4, 6], gravity_model_path=str(path), use_synthetic_gravity_fallback=False)

    with pytest.raises(ValueError, match=r"degree 6.*maximum degree 4"):
        load_gravity_model_for_budget(cfg)
    with pytest.raises(ValueError, match="maximum degree 4"):
        run_perturbation_budget(replace(cfg, output_dir=str(tmp_path / "out")))


def test_sh_degree_cli_above_gravity_file_max_degree_is_refused(tmp_path: Path) -> None:
    path = _write_shadr(tmp_path / "deg4.tab", 4)
    with pytest.raises(ValueError, match="maximum degree 4"):
        budget_main(
            [
                "--altitudes-km", "100",
                "--inclinations-deg", "0",
                "--true-anomalies-deg", "0",
                "--epochs", "synthetic",
                "--sh-degrees", "2,4,8",
                "--gravity-model", str(path),
                "--synthetic-fallback", "off",
                "--out-dir", str(tmp_path / "out"),
            ]
        )


def test_sh_degree_at_gravity_file_max_degree_is_accepted(tmp_path: Path) -> None:
    path = _write_shadr(tmp_path / "deg4.tab", 4)
    cfg = _cfg(sh_degrees=[2, 4], gravity_model_path=str(path), use_synthetic_gravity_fallback=False)
    assert load_gravity_model_for_budget(cfg).model.max_degree == 4


# --------------------------------------------------------------------------- #
# 2. --epochs is a time or it is refused
# --------------------------------------------------------------------------- #


def test_synthetic_geometry_refuses_calendar_epochs() -> None:
    with pytest.raises(ValueError, match="geometry_mode"):
        _cfg(epochs_utc=["2027-01-01T00:00:00Z"])


def test_synthetic_geometry_epoch_labels_do_not_change_geometry() -> None:
    a = generate_sample_states(_cfg(epochs_utc=["label-a"]))[0]
    b = generate_sample_states(_cfg(epochs_utc=["label-b"]))[0]
    np.testing.assert_array_equal(a.sun_m, b.sun_m)
    np.testing.assert_array_equal(a.earth_m, b.earth_m)
    assert a.geometry_source == "synthetic_geometry"


def test_ephemeris_geometry_refuses_non_date_epochs() -> None:
    with pytest.raises(ValueError, match="epoch"):
        _cfg(geometry_mode="ephemeris", epochs_utc=["synthetic"])


def test_unknown_geometry_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="geometry_mode"):
        _cfg(geometry_mode="magic")


def test_cli_calendar_epochs_without_ephemeris_geometry_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="geometry_mode"):
        budget_main(["--epochs", "2027-01-01", "--out-dir", str(tmp_path / "out")])


@pytest.mark.requires_data
def test_ephemeris_geometry_matches_spice_at_the_epoch() -> None:
    import spiceypy as spice

    epochs = ["2026-01-02T00:00:00Z", "2027-01-01T00:00:00Z"]
    cfg = _cfg(geometry_mode="ephemeris", epochs_utc=epochs)
    samples = generate_sample_states(cfg)
    assert {s.epoch_utc for s in samples} == set(epochs)

    kernels = [
        KERNEL_DIR / "naif0012.tls",
        KERNEL_DIR / "de440.bsp",
        KERNEL_DIR / "moon_pa_de440_200625.bpc",
        KERNEL_DIR / "moon_de440_250416.tf",
    ]
    for k in kernels:
        spice.furnsh(str(k))
    try:
        for epoch in epochs:
            sample = next(s for s in samples if s.epoch_utc == epoch)
            et = spice.str2et(epoch)
            rot = np.asarray(spice.pxform("J2000", "MOON_PA", et))
            sun, _ = spice.spkezr("SUN", et, "J2000", "NONE", "MOON")
            earth, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "MOON")
            np.testing.assert_allclose(sample.sun_m, rot @ np.asarray(sun[:3]) * 1e3, rtol=1e-9)
            np.testing.assert_allclose(sample.earth_m, rot @ np.asarray(earth[:3]) * 1e3, rtol=1e-9)
            np.testing.assert_allclose(sample.sun_vel_m_s, rot @ np.asarray(sun[3:]) * 1e3, rtol=1e-9, atol=1e-6)
            assert sample.geometry_source.startswith("ephemeris")
    finally:
        for k in kernels:
            spice.unload(str(k))

    by_epoch = {s.epoch_utc: s for s in samples}
    assert not np.allclose(by_epoch[epochs[0]].sun_m, by_epoch[epochs[1]].sun_m, rtol=1e-3)
    assert not np.allclose(by_epoch[epochs[0]].earth_m, by_epoch[epochs[1]].earth_m, rtol=1e-3)


# --------------------------------------------------------------------------- #
# 3. --include-earth-j2 is independent of --include-third-body
# --------------------------------------------------------------------------- #


def test_earth_j2_is_honoured_when_third_body_is_off() -> None:
    r = np.array([1.838e6, 0.0, 0.0])
    sun = np.array([1.0e11, 5.0e10, 2.0e10])
    earth = np.array([3.0e8, 2.0e8, 1.0e8])
    sample = _sample(r, sun, earth)

    vectors = non_gravity_vectors(_cfg(include_third_body=False, include_earth_j2=True), sample)

    expected = calc_j2_oblate_diff_accel(r, earth, mu_body=MU_EARTH, params=EarthJ2Params())
    assert set(vectors) == {"Earth J2 Differential"}
    np.testing.assert_array_equal(vectors["Earth J2 Differential"], expected)


# --------------------------------------------------------------------------- #
# 4. Every CSV says which gravity coefficients and geometry produced it
# --------------------------------------------------------------------------- #


def test_every_csv_carries_gravity_and_geometry_source(tmp_path: Path) -> None:
    cfg = _cfg(include_srp=True, include_third_body=True, output_dir=str(tmp_path))
    result = run_perturbation_budget(cfg)

    csvs = [
        result.perturbation_budget_csv,
        result.gravity_degree_sensitivity_csv,
        result.force_model_uncertainty_budget_csv,
        result.recommended_gravity_degree_by_altitude_csv,
        result.propagation_ablation_csv,
        result.runtime_budget_csv,
    ]
    for path in csvs:
        rows = _read_csv(path)
        assert rows, path
        assert {r["gravity_source"] for r in rows} == {"synthetic_gravity_coefficients"}, path
        assert {r["geometry_source"] for r in rows} == {"synthetic_geometry"}, path


def test_gravity_source_column_names_the_gravity_file(tmp_path: Path) -> None:
    path = _write_shadr(tmp_path / "deg4.tab", 4)
    cfg = _cfg(
        gravity_model_path=str(path),
        use_synthetic_gravity_fallback=False,
        output_dir=str(tmp_path / "out"),
    )
    result = run_perturbation_budget(cfg)
    rows = _read_csv(result.perturbation_budget_csv)
    assert {r["gravity_source"] for r in rows} == {str(path)}


# --------------------------------------------------------------------------- #
# 5. Force-magnitude oracles and on/off flags
# --------------------------------------------------------------------------- #


def test_collinear_earth_third_body_matches_closed_form() -> None:
    b = 3.844e8  # Moon-centred Earth distance [m]
    r = 1.838e6  # spacecraft radius, on the Moon->Earth line [m]
    sample = _sample(np.array([r, 0.0, 0.0]), np.array([0.0, AU, 0.0]), np.array([b, 0.0, 0.0]))

    acc = non_gravity_vectors(_cfg(include_third_body=True), sample)["Third Body Earth"]

    expected = float(MU_EARTH) * (1.0 / (b - r) ** 2 - 1.0 / b**2)
    assert math.isclose(float(acc[0]), expected, rel_tol=1e-10)
    assert abs(float(acc[1])) <= 1e-10 * expected
    assert abs(float(acc[2])) <= 1e-10 * expected


def test_unshadowed_srp_matches_closed_form() -> None:
    cr, area, mass = 1.8, 2.0, 1000.0
    r = np.array([0.0, 1.838e6, 0.0])
    sun = np.array([1.49e11, 0.0, 0.0])
    earth = np.array([0.0, 0.0, 3.844e8])  # far from the Sun-craft line: no Earth shadow
    sample = _sample(r, sun, earth)

    acc = non_gravity_vectors(
        _cfg(include_srp=True, srp_coefficient=cr, spacecraft_area_m2=area, spacecraft_mass_kg=mass),
        sample,
    )["SRP"]

    d_vec = r - sun
    d = float(np.linalg.norm(d_vec))
    expected_mag = float(P_SUN_1AU) * cr * area / mass * (float(AU) / d) ** 2
    assert math.isclose(float(np.linalg.norm(acc)), expected_mag, rel_tol=1e-12)
    np.testing.assert_allclose(acc / np.linalg.norm(acc), d_vec / d, rtol=1e-12)


_FLAG_TO_FORCES = {
    "include_srp": {"SRP"},
    "include_albedo": {"Lunar Albedo"},
    "include_thermal_ir": {"Thermal IR"},
    "include_tides": {"Solid Tides", "Solid Tides Earth", "Solid Tides Sun"},
    "include_third_body": {"Third Body Earth", "Third Body Sun"},
    "include_relativity": {"1PN Relativity"},
    "include_earth_j2": {"Earth J2 Differential"},
}


@pytest.mark.parametrize("flag", _ALL_FORCE_FLAGS)
def test_each_force_flag_adds_only_its_own_force(flag: str) -> None:
    sample = _sample(
        np.array([1.838e6, 0.0, 0.0]),
        np.array([1.0e11, 9.0e10, 4.0e10]),
        np.array([3.0e8, 2.0e8, 1.0e8]),
    )
    all_on = non_gravity_vectors(_cfg(**dict.fromkeys(_ALL_FORCE_FLAGS, True)), sample)
    only_this = non_gravity_vectors(_cfg(**{flag: True}), sample)
    all_but_this = non_gravity_vectors(_cfg(**{**dict.fromkeys(_ALL_FORCE_FLAGS, True), flag: False}), sample)

    assert set(only_this) == _FLAG_TO_FORCES[flag]
    assert set(all_but_this) == set(all_on) - _FLAG_TO_FORCES[flag]
    # Switching one flag leaves every other force bit-identical.
    for name, vec in all_but_this.items():
        np.testing.assert_array_equal(vec, all_on[name])
    for name, vec in only_this.items():
        np.testing.assert_array_equal(vec, all_on[name])
    assert non_gravity_vectors(_cfg(), sample) == {}
