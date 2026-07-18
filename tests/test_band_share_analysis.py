from __future__ import annotations

import hashlib

import numpy as np
from validation.gravity.band_share_analysis import (
    COEFFICIENT_NORMALIZATION,
    EVIDENCE_SCHEMA_VERSION,
    build_evidence_payload,
)

from lunaris.physics.spherical_harmonics import GravityModel


def _model() -> GravityModel:
    c = np.zeros((3, 3), dtype=np.float64)
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -2.0e-4
    return GravityModel.from_arrays(
        degree_max=2,
        r_ref=1_737_400.0,
        mu=4.9048695e12,
        c_coeffs_full=c,
        s_coeffs_full=s,
    )


def test_evidence_payload_pins_model_and_execution_provenance(tmp_path) -> None:
    gravity_file = tmp_path / "model.tab"
    gravity_file.write_bytes(b"deterministic gravity coefficients\n")
    payload = build_evidence_payload(
        gravity_file=gravity_file,
        model=_model(),
        band_edges=(60, 100),
        n_points=1000,
        seed=20260718,
        results=[{"altitude_km": 80.0}],
        repo_commit_sha="a" * 40,
        generated_at_utc="2026-07-18T12:00:00+00:00",
    )

    assert payload["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert payload["repo_commit_sha"] == "a" * 40
    assert payload["generated_at_utc"] == "2026-07-18T12:00:00+00:00"
    model = payload["gravity_model"]
    assert model["path_hint"] == "model.tab"
    assert model["sha256"] == hashlib.sha256(gravity_file.read_bytes()).hexdigest()
    assert model["normalization"] == COEFFICIENT_NORMALIZATION
    assert model["mu_m3_s2"] == 4.9048695e12
    assert model["reference_radius_m"] == 1_737_400.0
    assert model["evaluated_max_degree"] == 2
    assert payload["method"]["share_denominator"].startswith("rms_norm")
    assert payload["n_points_per_altitude"] == 1000
