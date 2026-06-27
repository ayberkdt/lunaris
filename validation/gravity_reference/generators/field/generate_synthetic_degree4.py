"""Generate the committed synthetic degree-4 gravity-field reference.

This is a maintainer script. It intentionally uses the independent direct-formula
oracle, not the production Lunaris spherical-harmonic evaluator.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from lunaris.validation.gravity_reference.independent_field_oracle import (
    coefficients_from_json,
    evaluate_point,
)
from lunaris.validation.gravity_reference.source_hashes import sha256_file

REPO = Path(__file__).resolve().parents[4]
BENCHMARK_ID = "synthetic_degree4_oracle"
R_REF_M = 1_738_000.0
MU_M3_S2 = 4_904_869_500_000.0
DEGREE = 4
CREATED_AT_UTC = "2026-06-17T00:00:00Z"


def _unit(vector: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def coefficient_fixture() -> dict:
    return {
        "schema_version": "lunaris_synthetic_sh_coefficients_v1",
        "title": "Synthetic fully normalized degree-4 lunar-like SH fixture",
        "normalization": "fully_normalized_4pi",
        "degree": DEGREE,
        "order": DEGREE,
        "mu_m3_s2": MU_M3_S2,
        "reference_radius_m": R_REF_M,
        "coefficients": [
            {"n": 0, "m": 0, "c": 1.0, "s": 0.0},
            {"n": 2, "m": 0, "c": -9.08e-5, "s": 0.0},
            {"n": 2, "m": 2, "c": 2.0e-5, "s": -1.3e-5},
            {"n": 3, "m": 1, "c": 8.0e-6, "s": -5.0e-6},
            {"n": 3, "m": 3, "c": 2.5e-6, "s": 1.1e-6},
            {"n": 4, "m": 0, "c": 1.0e-6, "s": 0.0},
            {"n": 4, "m": 2, "c": -7.0e-7, "s": 4.0e-7},
            {"n": 4, "m": 4, "c": 5.0e-7, "s": -2.0e-7},
        ],
    }


def sample_points() -> list[tuple[str, np.ndarray]]:
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
    return [(name, _unit(vec) * (R_REF_M + alt_m)) for name, vec, alt_m in specs]


def main() -> int:
    ref_dir = REPO / "validation" / "gravity_reference" / "reference_data" / "field"
    bench_dir = REPO / "validation" / "gravity_reference" / "benchmarks" / "field"
    ref_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    coeff_path = ref_dir / "synthetic_degree4_coefficients.json"
    coeff_payload = coefficient_fixture()
    coeff_path.write_text(json.dumps(coeff_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    degree, r_ref, mu, c, s = coefficients_from_json(coeff_payload)
    csv_path = ref_dir / "synthetic_degree4_field_vectors.csv"
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
        for point_id, position in sample_points():
            got = evaluate_point(
                position,
                mu_m3_s2=mu,
                reference_radius_m=r_ref,
                c_coeffs=c,
                s_coeffs=s,
                degree=degree,
                rel_step=1e-4,
                precision_digits=16,
            )
            writer.writerow({
                "point_id": point_id,
                "x_m": f"{position[0]:.17e}",
                "y_m": f"{position[1]:.17e}",
                "z_m": f"{position[2]:.17e}",
                "potential_m2_s2": f"{got.potential_m2_s2:.17e}",
                "ax_m_s2": f"{got.acceleration_m_s2[0]:.17e}",
                "ay_m_s2": f"{got.acceleration_m_s2[1]:.17e}",
                "az_m_s2": f"{got.acceleration_m_s2[2]:.17e}",
            })

    manifest = {
        "schema_version": "lunaris_gravity_field_reference_v1",
        "benchmark_id": BENCHMARK_ID,
        "reference_class": "independent_high_precision_field_oracle",
        "title": "Synthetic degree-4 field oracle smoke benchmark",
        "source": {
            "organization": "Lunaris project",
            "project": "gravity_reference",
            "document_or_repository": "https://github.com/ayberkdt/lunaris",
            "release_or_commit": "generated from committed script",
            "retrieved_at_utc": CREATED_AT_UTC,
            "license_note": "Synthetic data generated in-tree under the repository license.",
        },
        "reference_file": {
            "path": "validation/gravity_reference/reference_data/field/synthetic_degree4_field_vectors.csv",
            "format": "csv",
            "sha256": sha256_file(csv_path),
            "size_bytes": csv_path.stat().st_size,
        },
        "generator": {
            "name": "lunaris_independent_direct_formula_oracle",
            "version": "v1",
            "script_path": "validation/gravity_reference/generators/field/generate_synthetic_degree4.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "arithmetic": "float64_direct_formula_plus_4th_order_finite_difference",
            "precision_digits": 16,
        },
        "coordinates": {
            "center": "MOON",
            "frame": "MOON_FIXED_SYNTHETIC",
            "position_unit": "m",
        },
        "gravity": {
            "model_name": "synthetic_degree4",
            "model_release": "v1",
            "coefficient_file": "validation/gravity_reference/reference_data/field/synthetic_degree4_coefficients.json",
            "coefficient_sha256": sha256_file(coeff_path),
            "degree": DEGREE,
            "order": DEGREE,
            "normalization": "fully_normalized_4pi",
            "mu_m3_s2": MU_M3_S2,
            "reference_radius_m": R_REF_M,
            "tide_system": "not_applicable_synthetic",
            "coefficient_frame": "MOON_FIXED_SYNTHETIC",
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
                "acceleration_norm_m_s2": 1e-7,
                "acceleration_component_m_s2": 1e-7,
            },
            "relative_tolerances": {
                "potential": 1e-11,
                "acceleration_norm": 1e-7,
            },
            "threshold_origin": "convergence_derived_and_frozen",
        },
    }
    manifest_path = bench_dir / f"{BENCHMARK_ID}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
