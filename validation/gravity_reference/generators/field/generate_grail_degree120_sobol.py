"""Statistical GRAIL degree-120 field benchmark: many Sobol points vs pyshtools.

The eight hand-picked points in ``grail_degree120_pyshtools_oracle`` are a fast
smoke check. This generator produces a *statistical* reference instead: a large,
deterministic Sobol point cloud spread over the sphere and across altitude
strata (plus explicit polar and equatorial sets), so the field validation
reports max / P95 / P99 errors and latitude/altitude-binned error tables rather
than a claim resting on eight points.

Reference values come from **pyshtools** (independent SH library); this is an
``independent_high_precision_field_oracle``, not NASA-published vectors. Same
no-Condon-Shortley convention as the smoke benchmark (guarded by its test).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pyshtools as pysh
from scipy.stats import qmc

from lunaris.validation.gravity_reference.source_hashes import sha256_file

REPO = Path(__file__).resolve().parents[4]
BENCHMARK_ID = "grail_degree120_pyshtools_sobol"
DEGREE = 120
N_SOBOL = 2048
CREATED_AT_UTC = "2026-06-18T00:00:00Z"
GRAIL_TAB_PATH = REPO / "data" / "gravity_models" / "jggrx_1800f_sha.tab.txt"
ALT_MIN_KM = 50.0
ALT_MAX_KM = 2000.0


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


def _clm(payload: dict, degree: int) -> np.ndarray:
    arr = np.zeros((2, degree + 1, degree + 1), dtype=np.float64)
    for row in payload["coefficients"]:
        n, m = int(row["n"]), int(row["m"])
        if n <= degree and m <= n:
            arr[0, n, m] = float(row["c"])
            arr[1, n, m] = float(row["s"])
    return arr


def evaluate_pyshtools(pos: np.ndarray, mu: float, r_ref: float, clm: np.ndarray, degree: int):
    r = float(np.linalg.norm(pos))
    lat = math.degrees(math.asin(pos[2] / r))
    lon = math.degrees(math.atan2(pos[1], pos[0]))
    a_r, a_th, a_ph = pysh.gravmag.MakeGravGridPoint(clm, mu, r_ref, r, lat, lon)
    colat = math.pi / 2.0 - math.radians(lat)
    ph = math.radians(lon)
    e_r = np.array([math.sin(colat) * math.cos(ph), math.sin(colat) * math.sin(ph), math.cos(colat)])
    e_t = np.array([math.cos(colat) * math.cos(ph), math.cos(colat) * math.sin(ph), -math.sin(colat)])
    e_p = np.array([-math.sin(ph), math.cos(ph), 0.0])
    accel = a_r * e_r + a_th * e_t + a_ph * e_p
    scaled = np.zeros_like(clm)
    for ell in range(degree + 1):
        scaled[:, ell, :] = clm[:, ell, :] * (r_ref / r) ** ell
    potential = (mu / r) * pysh.expand.MakeGridPoint(scaled, lat=lat, lon=lon)
    return potential, accel


def sample_points(r_ref: float) -> list[tuple[str, np.ndarray]]:
    pts: list[tuple[str, np.ndarray]] = []
    # Deterministic Sobol cloud: uniform on the sphere x log-uniform altitude.
    sob = qmc.Sobol(d=3, scramble=True, seed=42).random(N_SOBOL)
    z = 2.0 * sob[:, 0] - 1.0
    lon = 2.0 * math.pi * sob[:, 1]
    alt_km = ALT_MIN_KM * (ALT_MAX_KM / ALT_MIN_KM) ** sob[:, 2]
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    for i in range(N_SOBOL):
        r = r_ref + alt_km[i] * 1000.0
        u = np.array([rho[i] * math.cos(lon[i]), rho[i] * math.sin(lon[i]), z[i]])
        pts.append((f"sobol_{i:05d}", u * r))
    # Explicit polar + equatorial strata so those regions are never undersampled.
    for j, alt in enumerate((50e3, 200e3, 1000e3, 2000e3)):
        pts.append((f"npole_{j}", np.array([1e-6, -2e-6, 1.0]) / np.linalg.norm([1e-6, -2e-6, 1.0]) * (r_ref + alt)))
        pts.append((f"spole_{j}", np.array([2e-6, 1e-6, -1.0]) / np.linalg.norm([2e-6, 1e-6, 1.0]) * (r_ref + alt)))
        for k, lam in enumerate((0.0, 90.0, 180.0, 270.0)):
            ll = math.radians(lam)
            pts.append((f"eq_{j}_{k}", np.array([math.cos(ll), math.sin(ll), 0.0]) * (r_ref + alt)))
    return pts


def main() -> int:
    ref_dir = REPO / "validation" / "gravity_reference" / "reference_data" / "field"
    bench_dir = REPO / "validation" / "gravity_reference" / "benchmarks" / "field"
    coeff_payload = parse_grail_coefficients(GRAIL_TAB_PATH, DEGREE)
    mu = float(coeff_payload["mu_m3_s2"])
    r_ref = float(coeff_payload["reference_radius_m"])
    coeff_path = ref_dir / "grail_degree120_sobol_coefficients.json"
    coeff_path.write_text(json.dumps(coeff_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clm = _clm(coeff_payload, DEGREE)

    points = sample_points(r_ref)
    print(f"Evaluating {len(points)} points with pyshtools degree-{DEGREE}...")
    csv_path = ref_dir / "grail_degree120_sobol_field_vectors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["point_id", "x_m", "y_m", "z_m", "potential_m2_s2", "ax_m_s2", "ay_m_s2", "az_m_s2"])
        for pid, pos in points:
            pot, acc = evaluate_pyshtools(pos, mu, r_ref, clm, DEGREE)
            w.writerow([pid] + [f"{v:.17e}" for v in (*pos, pot, *acc)])

    manifest = {
        "schema_version": "lunaris_gravity_field_reference_v1",
        "benchmark_id": BENCHMARK_ID,
        "reference_class": "independent_high_precision_field_oracle",
        "title": f"GRAIL JGGRX_1800F degree-120 statistical field validation ({len(points)} Sobol+strata points, pyshtools oracle)",
        "source": {
            "organization": "NASA/JPL (coefficients), SHTOOLS (computation)",
            "project": "Independent high-precision field oracle (statistical)",
            "document_or_repository": "https://pyshtools.github.io/pyshtools/",
            "release_or_commit": f"pyshtools={pysh.__version__}",
            "retrieved_at_utc": CREATED_AT_UTC,
            "license_note": "Independent open-source SH computation; not NASA-published vectors.",
        },
        "reference_file": {
            "path": "validation/gravity_reference/reference_data/field/grail_degree120_sobol_field_vectors.csv",
            "format": "csv",
            "sha256": sha256_file(csv_path),
            "size_bytes": csv_path.stat().st_size,
        },
        "generator": {
            "name": "pyshtools_gravmag_MakeGravGridPoint_sobol",
            "version": "v1",
            "script_path": "validation/gravity_reference/generators/field/generate_grail_degree120_sobol.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "arithmetic": "external_shtools_float64",
            "precision_digits": 16,
        },
        "coordinates": {"center": "MOON", "frame": "MOON_PA", "position_unit": "m"},
        "gravity": {
            "model_name": "jggrx_1800f_truncated120",
            "model_release": "v1",
            "coefficient_file": "validation/gravity_reference/reference_data/field/grail_degree120_sobol_coefficients.json",
            "coefficient_sha256": sha256_file(coeff_path),
            "degree": DEGREE,
            "order": DEGREE,
            "normalization": "fully_normalized_4pi",
            "mu_m3_s2": mu,
            "reference_radius_m": r_ref,
            "tide_system": "tide_free",
            "coefficient_frame": "MOON_PA",
        },
        "quantities": {
            "potential": True,
            "acceleration": True,
            "gradient_or_hessian": False,
            "potential_sign_convention": "positive_geodesy_mu_over_r",
            "acceleration_sign_convention": "a_equals_positive_gradient_U",
        },
        "comparison": {
            "absolute_tolerances": {"acceleration_norm_m_s2": 1e-8, "acceleration_component_m_s2": 1e-8},
            "relative_tolerances": {"acceleration_norm": 1e-7},
            "threshold_origin": "official_software_validation",
        },
    }
    (bench_dir / f"{BENCHMARK_ID}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(points)} reference rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
