"""Phase 1: the full-field spherical-harmonics baseline is really evaluated.

Covers the promoted physics kernel (``sh_potential_accel_fixed`` +
``GravityModel.potential_*``) and the ``shared.scaling`` baseline branch, with
independent cross-checks: potential against the independent field oracle,
acceleration against the trusted ``GravityModel.accel_fixed`` recurrence, and the
degree-0 limit against the analytical monopole.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.lunar_data import MU_MOON_SI, R_MOON_SI
from lunaris.physics.spherical_harmonics import GravityModel, sh_potential_accel_fixed
from lunaris.surrogate.st_lrps.shared.contracts import TargetContract
from lunaris.validation.gravity_reference.independent_field_oracle import geopotential

torch = pytest.importorskip("torch")

_N = 6


def _synthetic_field(degree: int = _N):
    rng = np.random.default_rng(11)
    c = np.zeros((degree + 1, degree + 1))
    s = np.zeros((degree + 1, degree + 1))
    c[0, 0] = 1.0
    for n in range(2, degree + 1):
        for m in range(0, n + 1):
            c[n, m] = rng.normal() * 1e-4
            if m > 0:
                s[n, m] = rng.normal() * 1e-4
    return c, s


def _query_points():
    return np.array(
        [
            [R_MOON_SI + 50e3, 0.0, 0.0],
            [0.0, R_MOON_SI + 120e3, 0.0],
            [(R_MOON_SI + 80e3) * 0.6, (R_MOON_SI + 80e3) * 0.5, (R_MOON_SI + 80e3) * 0.62],
        ],
        dtype=np.float64,
    )


def test_physics_potential_matches_independent_oracle() -> None:
    c, s = _synthetic_field()
    pts = _query_points()
    V, _ = sh_potential_accel_fixed(pts, c, s, MU_MOON_SI, R_MOON_SI, _N, -1)
    for i, p in enumerate(pts):
        Vo = geopotential(p, mu_m3_s2=MU_MOON_SI, reference_radius_m=R_MOON_SI,
                          c_coeffs=c, s_coeffs=s, degree=_N)
        assert abs(V[i] - Vo) / abs(Vo) < 1e-12


def test_physics_accel_matches_gravitymodel_recurrence() -> None:
    c, s = _synthetic_field()
    pts = _query_points()
    _, a = sh_potential_accel_fixed(pts, c, s, MU_MOON_SI, R_MOON_SI, _N, -1)
    gm = GravityModel.from_arrays(degree_max=_N, r_ref=R_MOON_SI, mu=MU_MOON_SI,
                                  c_coeffs_full=c, s_coeffs_full=s)
    for i, p in enumerate(pts):
        a_ref = gm.accel_fixed(p, degree=_N)
        assert np.linalg.norm(a[i] - a_ref) / np.linalg.norm(a_ref) < 1e-9


def test_degree0_reduces_to_monopole() -> None:
    c, s = _synthetic_field()
    pts = _query_points()
    V, a = sh_potential_accel_fixed(pts, c, s, MU_MOON_SI, R_MOON_SI, 0, -1)
    for i, p in enumerate(pts):
        r = float(np.linalg.norm(p))
        assert abs(V[i] - MU_MOON_SI / r) / (MU_MOON_SI / r) < 1e-12
        a_mono = -MU_MOON_SI * p / r**3
        assert np.linalg.norm(a[i] - a_mono) / np.linalg.norm(a_mono) < 1e-12


def test_gravitymodel_potential_api() -> None:
    c, s = _synthetic_field()
    gm = GravityModel.from_arrays(degree_max=_N, r_ref=R_MOON_SI, mu=MU_MOON_SI,
                                  c_coeffs_full=c, s_coeffs_full=s)
    p = _query_points()[0]
    U, a = gm.potential_accel_fixed(p, degree=_N)
    assert U == pytest.approx(gm.potential_fixed(p, degree=_N))
    assert np.linalg.norm(a - gm.accel_fixed(p, degree=_N)) / np.linalg.norm(a) < 1e-9


def _full_sh_contract(a_sign: float = 1.0) -> TargetContract:
    return TargetContract(
        central_body="moon",
        target_mode="full",
        base_degree=_N,
        target_degree=_N,
        baseline_kind="spherical_harmonics",
        unit_system="si",
        frame="moon_fixed_cartesian",
        derivative_convention_version="dP_dphi_corrected_v1",
        a_sign=a_sign,
        mu_si=MU_MOON_SI,
        r_ref_m=R_MOON_SI,
    )


def test_scaling_sh_baseline_matches_physics_field() -> None:
    from lunaris.surrogate.st_lrps.shared.scaling import (
        compute_base_accel_from_contract,
        compute_base_potential_from_contract,
    )

    c, s = _synthetic_field()
    gm = GravityModel.from_arrays(degree_max=_N, r_ref=R_MOON_SI, mu=MU_MOON_SI,
                                  c_coeffs_full=c, s_coeffs_full=s)
    pts = _query_points()
    x = torch.as_tensor(pts, dtype=torch.float64)
    contract = _full_sh_contract(a_sign=1.0)

    u_base = compute_base_potential_from_contract(x, contract, gm).numpy().reshape(-1)
    a_base = compute_base_accel_from_contract(x, contract, gm).numpy()
    for i, p in enumerate(pts):
        Vo = geopotential(p, mu_m3_s2=MU_MOON_SI, reference_radius_m=R_MOON_SI,
                          c_coeffs=c, s_coeffs=s, degree=_N)
        assert abs(u_base[i] - Vo) / abs(Vo) < 1e-12
        a_ref = gm.accel_fixed(p, degree=_N)
        assert np.linalg.norm(a_base[i] - a_ref) / np.linalg.norm(a_ref) < 1e-9


def test_scaling_sh_baseline_applies_a_sign_to_potential() -> None:
    from lunaris.surrogate.st_lrps.shared.scaling import compute_base_potential_from_contract

    c, s = _synthetic_field()
    gm = GravityModel.from_arrays(degree_max=_N, r_ref=R_MOON_SI, mu=MU_MOON_SI,
                                  c_coeffs_full=c, s_coeffs_full=s)
    x = torch.as_tensor(_query_points(), dtype=torch.float64)
    u_plus = compute_base_potential_from_contract(x, _full_sh_contract(1.0), gm).numpy()
    u_minus = compute_base_potential_from_contract(x, _full_sh_contract(-1.0), gm).numpy()
    assert np.allclose(u_plus, -u_minus)


def test_scaling_sh_baseline_requires_gravity_model() -> None:
    from lunaris.surrogate.st_lrps.shared.scaling import compute_base_accel_from_contract

    x = torch.as_tensor(_query_points(), dtype=torch.float64)
    with pytest.raises(ValueError, match="gravity_model|gravity-model"):
        compute_base_accel_from_contract(x, _full_sh_contract(), gravity_model=None)
