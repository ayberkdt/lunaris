"""Generate an independent gravity-only reference trajectory (GRAIL degree 32).

The reference force is computed with **pyshtools** (an external, globally
recognised spherical-harmonic library) and the orbit is integrated with an
**independent** high-order solver (SciPy ``DOP853`` at tighter tolerances than
the Lunaris run it will validate). The committed artifact therefore exercises
Lunaris' full propagation pipeline — force evaluation inside the RHS, the
integrator, and the state plumbing — against software that does not share
Lunaris code.

Honesty note
------------
NASA/JPL does not publish a "gravity-only truth trajectory": all public mission
products (SPK/OEM, reconstructed orbits) carry full force models and orbit
determination. This is an *independent-tool-generated* gravity-only reference,
not NASA-published truth. The pyshtools->Cartesian acceleration convention used
here is the same one guarded by the degree-120 pyshtools *field* benchmark,
which matches Lunaris to machine precision; if that convention ever breaks, the
field benchmark fails first.

The system is the well-posed *non-rotating* gravity-only problem: the field is
fixed in the integration frame (``state_frame == gravity_fixed_frame``).
Rotating body-fixed propagation (SPICE lunar orientation) is a separate layer.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pyshtools as pysh
from scipy.integrate import solve_ivp

from lunaris.validation.gravity_reference.source_hashes import sha256_file

REPO = Path(__file__).resolve().parents[4]
BENCHMARK_ID = "grail_degree32_pyshtools_trajectory"
DEGREE = 32
CREATED_AT_UTC = "2026-06-17T00:00:00Z"
GRAIL_TAB_PATH = REPO / "data" / "gravity_models" / "jggrx_1800f_sha.tab.txt"

# Non-rotating gravity-only arc: ~2 revolutions of a low circular lunar orbit.
ALT_M = 100_000.0
INC_DEG = 45.0
STEP_S = 30.0
N_STEPS = 480  # 480 * 30 s = 14400 s (~2 orbits); duration is an exact multiple of step


def parse_grail_coefficients(tab_path: Path, max_degree: int) -> dict:
    with open(tab_path, encoding="utf-8") as f:
        header = [p.strip() for p in f.readline().split(",")]
        r_ref_m = float(header[0]) * 1000.0
        mu_m3_s2 = float(header[1]) * 1e9
        coeffs = [{"n": 0, "m": 0, "c": 1.0, "s": 0.0}]
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            n, m = int(parts[0]), int(parts[1])
            if n > max_degree:
                continue
            coeffs.append({"n": n, "m": m, "c": float(parts[2]), "s": float(parts[3])})
    return {
        "schema_version": "lunaris_synthetic_sh_coefficients_v1",
        "title": f"Truncated GRAIL JGGRX_1800F (Degree {max_degree})",
        "normalization": "fully_normalized_4pi",
        "degree": max_degree,
        "order": max_degree,
        "mu_m3_s2": mu_m3_s2,
        "reference_radius_m": r_ref_m,
        "coefficients": coeffs,
    }


def _clm_from_payload(payload: dict, degree: int) -> np.ndarray:
    clm = np.zeros((2, degree + 1, degree + 1), dtype=np.float64)
    for row in payload["coefficients"]:
        n, m = int(row["n"]), int(row["m"])
        if n <= degree and m <= n:
            clm[0, n, m] = float(row["c"])
            clm[1, n, m] = float(row["s"])
    return clm


def pyshtools_acceleration(
    position: np.ndarray, mu_m3_s2: float, reference_radius_m: float, clm: np.ndarray
) -> np.ndarray:
    """a = +grad U via pyshtools (same convention as the field benchmark)."""
    r = float(np.linalg.norm(position))
    lat_rad = math.asin(position[2] / r)
    lon_rad = math.atan2(position[1], position[0])
    a_r, a_theta, a_phi = pysh.gravmag.MakeGravGridPoint(
        clm, mu_m3_s2, reference_radius_m, r, math.degrees(lat_rad), math.degrees(lon_rad)
    )
    colat = math.pi / 2.0 - lat_rad
    e_r = np.array([math.sin(colat) * math.cos(lon_rad), math.sin(colat) * math.sin(lon_rad), math.cos(colat)])
    e_theta = np.array([math.cos(colat) * math.cos(lon_rad), math.cos(colat) * math.sin(lon_rad), -math.sin(colat)])
    e_phi = np.array([-math.sin(lon_rad), math.cos(lon_rad), 0.0])
    return a_r * e_r + a_theta * e_theta + a_phi * e_phi


def main() -> int:
    ref_dir = REPO / "validation" / "gravity_reference" / "reference_data" / "trajectory"
    bench_dir = REPO / "validation" / "gravity_reference" / "benchmarks" / "trajectory"
    ref_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    coeff_payload = parse_grail_coefficients(GRAIL_TAB_PATH, DEGREE)
    mu = float(coeff_payload["mu_m3_s2"])
    r_ref = float(coeff_payload["reference_radius_m"])
    coeff_path = ref_dir / "grail_degree32_coefficients.json"
    coeff_path.write_text(json.dumps(coeff_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clm = _clm_from_payload(coeff_payload, DEGREE)

    # Initial state: inclined ~circular low lunar orbit.
    r0 = r_ref + ALT_M
    v_circ = math.sqrt(mu / r0)
    inc = math.radians(INC_DEG)
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circ * math.cos(inc), v_circ * math.sin(inc)], dtype=np.float64)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        a = pyshtools_acceleration(y[:3], mu, r_ref, clm)
        return np.array([y[3], y[4], y[5], a[0], a[1], a[2]], dtype=np.float64)

    epochs = np.arange(N_STEPS + 1, dtype=np.float64) * STEP_S
    print(f"Integrating {N_STEPS} steps with pyshtools degree-{DEGREE} force (DOP853, rtol=1e-13)...")
    sol = solve_ivp(
        rhs, (float(epochs[0]), float(epochs[-1])), y0, method="DOP853",
        t_eval=epochs, rtol=1e-13, atol=1e-16, max_step=STEP_S,
    )
    if not sol.success:
        raise RuntimeError(f"Reference integration failed: {sol.message}")
    states = np.asarray(sol.y, dtype=np.float64).T  # (N,6)

    csv_path = ref_dir / "grail_degree32_trajectory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
        for i in range(epochs.shape[0]):
            writer.writerow([f"{epochs[i]:.6f}"] + [f"{states[i, j]:.17e}" for j in range(6)])

    manifest = {
        "schema_version": "lunaris_gravity_trajectory_reference_v1",
        "benchmark_id": BENCHMARK_ID,
        "reference_class": "external_tool_generated_trajectory",
        "title": "GRAIL JGGRX_1800F degree-32 gravity-only arc (pyshtools force, SciPy DOP853, non-rotating)",
        "source": {
            "organization": "NASA/JPL (coefficients), SHTOOLS + SciPy (reference integration)",
            "project": "Independent gravity-only trajectory reference",
            "document_or_repository": "https://pyshtools.github.io/pyshtools/",
            "release_or_commit": f"pyshtools={pysh.__version__}",
            "retrieved_at_utc": CREATED_AT_UTC,
            "license_note": (
                "Independent-tool-generated gravity-only arc; NOT NASA-published truth. "
                "Mission ephemerides are not gravity-only and are not used here."
            ),
        },
        "reference_file": {
            "path": "validation/gravity_reference/reference_data/trajectory/grail_degree32_trajectory.csv",
            "format": "csv",
            "sha256": sha256_file(csv_path),
            "size_bytes": csv_path.stat().st_size,
        },
        "generator": {
            "name": "pyshtools_force_scipy_dop853",
            "version": "v1",
            "script_path": "validation/gravity_reference/generators/trajectory/generate_grail_degree32_pyshtools.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "integrator": "scipy.solve_ivp DOP853 rtol=1e-13 atol=1e-16",
        },
        "time": {
            "initial_epoch": CREATED_AT_UTC,
            "time_scale": "TDB",
            "duration_s": float(epochs[-1]),
            "output_step_s": float(STEP_S),
        },
        "frames": {
            "state_center": "MOON",
            "state_frame": "MOON_PA",
            "gravity_fixed_frame": "MOON_PA",
            "comparison_frame": "MOON_PA",
        },
        "initial_state": {
            "representation": "cartesian",
            "position_unit": "m",
            "velocity_unit": "m/s",
            "state": [float(v) for v in y0],
        },
        "gravity": {
            "model_name": "jggrx_1800f_truncated32",
            "model_release": "v1",
            "coefficient_file": "validation/gravity_reference/reference_data/trajectory/grail_degree32_coefficients.json",
            "coefficient_sha256": sha256_file(coeff_path),
            "degree": DEGREE,
            "order": DEGREE,
            "normalization": "fully_normalized_4pi",
            "mu_m3_s2": mu,
            "reference_radius_m": r_ref,
            "tide_system": "tide_free",
            "coefficient_frame": "MOON_PA",
        },
        "dynamics": {
            "gravity_only": True,
            "third_body": False,
            "solar_radiation_pressure": False,
            "relativity": False,
            "solid_tides": False,
            "albedo": False,
            "thermal_ir": False,
            "maneuvers": False,
            "thrust": False,
            "empirical_accelerations": False,
        },
        "integration": {"rtol": 1e-12, "atol": 1e-15},
        # Observed agreement on the reference platform: max position error
        # ~5e-8 m, max velocity error ~4e-11 m/s, Lunaris energy drift ~4e-15
        # over ~2 orbits. The frozen limits below sit far above that (regression
        # guards with generous cross-platform margin) yet are tight enough that a
        # sign/frame/kernel defect — which produces km-scale divergence — fails
        # immediately.
        "comparison": {
            "threshold_origin": "convergence_derived_and_frozen",
            "absolute_tolerances": {"position_m": 1.0e-2, "velocity_m_s": 1.0e-5},
            "relative_tolerances": {"energy_relative_drift": 1.0e-10},
        },
    }
    manifest_path = bench_dir / f"{BENCHMARK_ID}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Reference rows: {epochs.shape[0]}")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
