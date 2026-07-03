"""Validation of the external-body Schwarzschild differential term.

The term is evaluated in Moon-centered inertial coordinates and should equal

    a_1PN(spacecraft relative to body) - a_1PN(Moon relative to body)

for the external body's Schwarzschild correction. These tests deliberately keep
the de Sitter term out of scope; it is the sibling term in the combined
external 1PN path.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lunaris.common.constants import AU, C_LIGHT, MU_MOON, MU_SUN
from lunaris.physics.relativity_effects import (
    EPS_1E12,
    _external_schwarzschild_diff_components,
    _schwarzschild_components,
)

C_SQ = float(C_LIGHT) * float(C_LIGHT)


def _schwarzschild_ref(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    """Pure-Python central Schwarzschild reference, independent of the kernel."""
    rx, ry, rz = [float(x) for x in r]
    vx, vy, vz = [float(x) for x in v]
    r2 = rx * rx + ry * ry + rz * rz
    if r2 <= EPS_1E12:
        return np.zeros(3, dtype=np.float64)

    radius = math.sqrt(r2)
    v2 = vx * vx + vy * vy + vz * vz
    rv = rx * vx + ry * vy + rz * vz
    factor = mu / (C_SQ * r2 * radius)
    alpha = 4.0 * mu / radius - v2
    beta = 4.0 * rv
    return factor * np.array(
        [
            alpha * rx + beta * vx,
            alpha * ry + beta * vy,
            alpha * rz + beta * vz,
        ],
        dtype=np.float64,
    )


def _external_schwarzschild_ref(
    r: np.ndarray,
    v: np.ndarray,
    body: np.ndarray,
    body_v: np.ndarray,
    mu_body: float,
) -> np.ndarray:
    """Differential external reference: spacecraft term minus Moon term."""
    if mu_body <= 0.0 or float(np.dot(body, body)) <= EPS_1E12:
        return np.zeros(3, dtype=np.float64)
    spacecraft = _schwarzschild_ref(r - body, v - body_v, mu_body)
    moon = _schwarzschild_ref(-body, -body_v, mu_body)
    return spacecraft - moon


def _external_schwarzschild_kernel(
    r: np.ndarray,
    v: np.ndarray,
    body: np.ndarray,
    body_v: np.ndarray,
    mu_body: float,
) -> np.ndarray:
    return np.array(
        _external_schwarzschild_diff_components(
            float(r[0]),
            float(r[1]),
            float(r[2]),
            float(v[0]),
            float(v[1]),
            float(v[2]),
            float(body[0]),
            float(body[1]),
            float(body[2]),
            float(body_v[0]),
            float(body_v[1]),
            float(body_v[2]),
            float(mu_body),
        ),
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    ("r", "v", "body", "body_v", "mu_body"),
    [
        (
            np.array([2.05e6, -1.2e5, 4.0e4], dtype=np.float64),
            np.array([80.0, 1550.0, -25.0], dtype=np.float64),
            np.array([float(AU), 3.84e8, -1.1e7], dtype=np.float64),
            np.array([0.0, -29_780.0, 4.0], dtype=np.float64),
            float(MU_SUN),
        ),
        (
            np.array([-1.7e6, 8.0e5, 2.5e5], dtype=np.float64),
            np.array([-900.0, -1200.0, 130.0], dtype=np.float64),
            np.array([3.7e8, -1.2e8, 4.5e7], dtype=np.float64),
            np.array([210.0, -860.0, 35.0], dtype=np.float64),
            3.986004418e14,
        ),
        (
            np.array([2.3e6, 2.0e5, -3.0e5], dtype=np.float64),
            np.array([320.0, -1410.0, 70.0], dtype=np.float64),
            np.array([-8.0e9, 1.1e10, 2.0e9], dtype=np.float64),
            np.array([100.0, 9000.0, -250.0], dtype=np.float64),
            6.0e18,
        ),
    ],
)
def test_external_schwarzschild_is_two_central_terms_difference(
    r: np.ndarray,
    v: np.ndarray,
    body: np.ndarray,
    body_v: np.ndarray,
    mu_body: float,
) -> None:
    got = _external_schwarzschild_kernel(r, v, body, body_v, mu_body)
    expected = _external_schwarzschild_ref(r, v, body, body_v, mu_body)

    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=1e-27)

    sc_kernel = np.array(_schwarzschild_components(*(r - body), *(v - body_v), mu_body))
    moon_kernel = np.array(_schwarzschild_components(*(-body), *(-body_v), mu_body))
    np.testing.assert_allclose(got, sc_kernel - moon_kernel, rtol=2e-13, atol=1e-27)


def test_solar_external_schwarzschild_is_small_for_lunar_orbiter() -> None:
    r = np.array([2.0e6, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, math.sqrt(float(MU_MOON) / np.linalg.norm(r)), 0.0], dtype=np.float64)
    body = np.array([float(AU), 0.0, 0.0], dtype=np.float64)
    body_v = np.zeros(3, dtype=np.float64)

    external = _external_schwarzschild_kernel(r, v, body, body_v, float(MU_SUN))
    central_moon = np.array(_schwarzschild_components(*r, *v, float(MU_MOON)), dtype=np.float64)

    external_norm = float(np.linalg.norm(external))
    central_norm = float(np.linalg.norm(central_moon))
    # At lunar-orbiter scale the solar external Schwarzschild differential is a
    # tidal-like residual of two large Sun-relative 1PN terms. It should be near
    # 1e-13 m/s^2 and hundreds of times smaller than the Moon central 1PN term.
    assert external_norm < 3.0e-13
    assert external_norm / central_norm < 2.0e-3


def test_solar_external_schwarzschild_goes_to_zero_with_body_distance() -> None:
    r = np.array([2.0e6, 1.5e5, -8.0e4], dtype=np.float64)
    v = np.array([-120.0, math.sqrt(float(MU_MOON) / np.linalg.norm(r)), 35.0], dtype=np.float64)
    body_1au = np.array([float(AU), 3.84e8, -1.0e7], dtype=np.float64)
    body_v_1au = np.array([0.0, -29_780.0, 4.0], dtype=np.float64)

    norms: list[float] = []
    for scale in (1.0, 10.0, 100.0, 1000.0):
        body = scale * body_1au
        # Keplerian orbital speed around the same central body falls as R^-1/2.
        body_v = body_v_1au / math.sqrt(scale)
        norms.append(float(np.linalg.norm(_external_schwarzschild_kernel(r, v, body, body_v, float(MU_SUN)))))

    assert norms[0] < 1.0e-11
    assert norms[1] < norms[0] * 1.0e-2
    assert norms[2] < norms[1] * 1.0e-2
    assert norms[3] < norms[2] * 1.0e-2


def _function_block(source: str, name: str, next_name: str) -> str:
    start = source.index(f"def {name}")
    end = source.index(f"def {next_name}", start)
    return " ".join(line.strip() for line in source[start:end].splitlines())


def test_cuda_external_schwarzschild_source_uses_same_difference_formula() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "lunaris" / "core" / "batch_propagator.py").read_text()

    central = _function_block(source, "_relativity_1pn_cuda", "_external_schwarzschild_diff_cuda")
    assert "fac = mu * inv_r3 / c2" in central
    assert "A = (4.0 * mu * inv_r - v2)" in central
    assert "B = 4.0 * rdotv" in central
    assert "out[0] = fac * (A * rx + B * vx)" in central
    assert "out[1] = fac * (A * ry + B * vy)" in central
    assert "out[2] = fac * (A * rz + B * vz)" in central

    external = _function_block(source, "_external_schwarzschild_diff_cuda", "_de_sitter_cuda")
    assert "_relativity_1pn_cuda(rx - bx, ry - by, rz - bz, vx - bvx, vy - bvy, vz - bvz, mu_body, sc)" in external
    assert "_relativity_1pn_cuda(-bx, -by, -bz, -bvx, -bvy, -bvz, mu_body, moon)" in external
    assert "out[0] = sc[0] - moon[0]" in external
    assert "out[1] = sc[1] - moon[1]" in external
    assert "out[2] = sc[2] - moon[2]" in external
