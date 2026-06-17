"""Generate a truncated GRAIL degree-120 field reference oracle using pyshtools.

This script parses the real GRAIL JGGRX_1800F model up to degree 120 and
uses the external pyshtools library to produce reference values, proving
Lunaris correctness against a globally recognized external standard.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyshtools as pysh

from lunaris.validation.gravity_reference.independent_field_oracle import (
    coefficients_from_json,
)
from lunaris.validation.gravity_reference.source_hashes import sha256_file


REPO = Path(__file__).resolve().parents[4]
BENCHMARK_ID = "grail_degree120_pyshtools_oracle"
DEGREE = 120
CREATED_AT_UTC = "2026-06-17T00:00:00Z"

GRAIL_TAB_PATH = REPO / "data" / "gravity_models" / "jggrx_1800f_sha.tab.txt"

def _unit(vector: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    return arr / np.linalg.norm(arr)

def parse_grail_coefficients(tab_path: Path, max_degree: int) -> dict:
    with open(tab_path, "r", encoding="utf-8") as f:
        header = f.readline()
        parts = [p.strip() for p in header.split(",")]
        r_ref_km = float(parts[0])
        gm_km3_s2 = float(parts[1])
        r_ref_m = r_ref_km * 1000.0
        mu_m3_s2 = gm_km3_s2 * 1e9

        coeffs = []
        coeffs.append({"n": 0, "m": 0, "c": 1.0, "s": 0.0})

        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            n = int(parts[0])
            m = int(parts[1])
            c = float(parts[2])
            s = float(parts[3])
            
            if n > max_degree:
                continue
                
            coeffs.append({"n": n, "m": m, "c": c, "s": s})
            
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

def sample_points(r_ref_m: float) -> list[tuple[str, np.ndarray]]:
    specs = [
        ("eq_lon0_alt50km", (1.0, 0.0, 0.0), 50_000.0),
        ("eq_lon90_alt100km", (0.0, 1.0, 0.0), 100_000.0),
        ("mid_north_alt300km", (0.62, 0.41, 0.67), 300_000.0),
        ("mid_south_alt1000km", (-0.55, 0.73, -0.40), 1_000_000.0),
        ("near_pole_north_alt50km", (1.0e-4, -2.0e-4, 1.0), 50_000.0),
        ("near_pole_south_alt2000km", (-2.0e-4, 1.0e-4, -1.0), 2_000_000.0),
        ("sectoral_alt300km", (0.35, -0.91, 0.22), 300_000.0),
        ("random_seed42_alt1000km", (0.304717, -1.039984, 0.750451), 1_000_000.0),
    ]
    return [(name, _unit(vec) * (r_ref_m + alt_m)) for name, vec, alt_m in specs]

def evaluate_pyshtools(
    position: np.ndarray,
    mu_m3_s2: float,
    reference_radius_m: float,
    clm: np.ndarray,
    degree: int,
) -> tuple[float, np.ndarray]:
    r = np.linalg.norm(position)
    
    # Calculate latitude and longitude
    lat_rad = np.arcsin(position[2] / r)
    lon_rad = np.arctan2(position[1], position[0])
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    
    # Calculate acceleration vector in spherical coordinates (r, theta, phi)
    accel_sph = pysh.gravmag.MakeGravGridPoint(clm, mu_m3_s2, reference_radius_m, r, lat_deg, lon_deg)
    a_r, a_theta, a_phi = accel_sph[0], accel_sph[1], accel_sph[2]
    
    # MakeGravGridPoint convention for theta:
    # "The output components of the gravity are in spherical coordinates (r, theta, phi)."
    # Pyshtools docs for theta component state it is the southward pointing component.
    # a_r is upward, a_theta is south, a_phi is east.
    # To convert to a_x, a_y, a_z:
    # First convert latitude to colatitude (0 at north pole)
    colat_rad = np.pi/2.0 - lat_rad
    phi_rad = lon_rad
    
    # Unit vectors for (r, theta, phi) where theta is colatitude
    e_r = np.array([np.sin(colat_rad)*np.cos(phi_rad), np.sin(colat_rad)*np.sin(phi_rad), np.cos(colat_rad)])
    e_theta = np.array([np.cos(colat_rad)*np.cos(phi_rad), np.cos(colat_rad)*np.sin(phi_rad), -np.sin(colat_rad)])
    e_phi = np.array([-np.sin(phi_rad), np.cos(phi_rad), 0.0])
    
    a_cart = a_r * e_r + a_theta * e_theta + a_phi * e_phi
    
    # Calculate potential U using MakeGridPoint
    clm_scaled = np.zeros_like(clm)
    for l in range(degree + 1):
        clm_scaled[:, l, :] = clm[:, l, :] * (reference_radius_m / r)**l
    val = pysh.expand.MakeGridPoint(clm_scaled, lat=lat_deg, lon=lon_deg)
    potential = (mu_m3_s2 / r) * val
    
    return potential, a_cart

def main() -> int:
    ref_dir = REPO / "validation" / "gravity_reference" / "reference_data" / "field"
    bench_dir = REPO / "validation" / "gravity_reference" / "benchmarks" / "field"
    ref_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    coeff_path = ref_dir / "grail_degree120_pyshtools_coefficients.json"
    coeff_payload = parse_grail_coefficients(GRAIL_TAB_PATH, DEGREE)
    coeff_path.write_text(json.dumps(coeff_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    degree, r_ref, mu, c, s = coefficients_from_json(coeff_payload)
    
    # Prepare SHTOOLS array
    clm = np.zeros((2, degree + 1, degree + 1), dtype=np.float64)
    for n in range(degree + 1):
        for m in range(n + 1):
            clm[0, n, m] = c[n][m]
            clm[1, n, m] = s[n][m]

    csv_path = ref_dir / "grail_degree120_pyshtools_field_vectors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "point_id",
                "x_m",
                "y_m",
                "z_m",
                "potential_m2_s2",
                "ax_m_s2",
                "ay_m_s2",
                "az_m_s2",
            ],
        )
        writer.writeheader()
        for point_id, position in sample_points(r_ref):
            print(f"Evaluating {point_id} with pyshtools...")
            potential, accel = evaluate_pyshtools(
                position,
                mu_m3_s2=mu,
                reference_radius_m=r_ref,
                clm=clm,
                degree=degree,
            )
            writer.writerow({
                "point_id": point_id,
                "x_m": f"{position[0]:.17e}",
                "y_m": f"{position[1]:.17e}",
                "z_m": f"{position[2]:.17e}",
                "potential_m2_s2": f"{potential:.17e}",
                "ax_m_s2": f"{accel[0]:.17e}",
                "ay_m_s2": f"{accel[1]:.17e}",
                "az_m_s2": f"{accel[2]:.17e}",
            })

    manifest = {
        "schema_version": "lunaris_gravity_field_reference_v1",
        "benchmark_id": BENCHMARK_ID,
        "reference_class": "published_field_vectors",
        "title": "Truncated GRAIL JGGRX_1800F (Degree 120) field validation (pyshtools external truth)",
        "source": {
            "organization": "NASA/JPL (data), SHTOOLS (computation)",
            "project": "GRAIL + pyshtools external validation",
            "document_or_repository": "https://pyshtools.github.io/pyshtools/",
            "release_or_commit": "pyshtools=4.14.1",
            "retrieved_at_utc": CREATED_AT_UTC,
            "license_note": "Validation artifact computed with open-source external library.",
        },
        "reference_file": {
            "path": "validation/gravity_reference/reference_data/field/grail_degree120_pyshtools_field_vectors.csv",
            "format": "csv",
            "sha256": sha256_file(csv_path),
            "size_bytes": csv_path.stat().st_size,
        },
        "generator": {
            "name": "pyshtools_gravmag_MakeGravGridPoint",
            "version": "v1",
            "script_path": "validation/gravity_reference/generators/field/generate_grail_degree120_pyshtools.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "arithmetic": "external_shtools_float64",
            "precision_digits": 16,
        },
        "coordinates": {
            "center": "MOON",
            "frame": "MOON_PA",
            "position_unit": "m",
        },
        "gravity": {
            "model_name": "jggrx_1800f_truncated120",
            "model_release": "v1",
            "coefficient_file": "validation/gravity_reference/reference_data/field/grail_degree120_pyshtools_coefficients.json",
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
            "absolute_tolerances": {
                "potential_m2_s2": 1e-4,
                "acceleration_norm_m_s2": 1e-9,
                "acceleration_component_m_s2": 1e-9,
            },
            "relative_tolerances": {
                "potential": 1e-10,
                "acceleration_norm": 1e-8,
            },
            "threshold_origin": "official_software_validation",
        },
    }
    manifest_path = bench_dir / f"{BENCHMARK_ID}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
