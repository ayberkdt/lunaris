"""Parity: ST-LRPS spatial-cloud generator vs the runtime GravityModel.

The generator (`_sh_potential_accel_batch_serial`) produces the gravity labels
(potential + acceleration) that ST-LRPS trains on; the runtime propagator uses
`GravityModel.accel_fixed`. If these two evaluate the spherical-harmonic field
under *different* conventions, a model learns one field while the propagator
applies another.

This guards the Condon-Shortley phase specifically. Both must use the
geodesy/GRAIL convention (no `(-1)^m` phase). A zonal-only (m=0) check cannot
catch a phase error, so these tests deliberately exercise odd-order tesseral and
sectoral coefficients (C/S at m=1, m=3), and a negative control proves the test
has teeth.
"""

from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import numpy as np
import pytest

from lunaris.physics.spherical_harmonics import GravityModel
from lunaris.surrogate.st_lrps.data.spatial_cloud_generator import (
    _sh_potential_accel_batch_serial,
    precompute_legendre_constants,
)

_DEGREE = 4
_R = 1_738_000.0
_MU = 4.9028e12


def _synthetic_field_with_odd_m() -> tuple[np.ndarray, np.ndarray]:
    """A small field whose odd-order terms make the phase convention observable."""
    c = np.zeros((_DEGREE + 1, _DEGREE + 1), dtype=np.float64)
    s = np.zeros((_DEGREE + 1, _DEGREE + 1), dtype=np.float64)
    c[0, 0] = 1.0  # monopole
    c[2, 0] = -9.0e-5  # J2 (zonal, phase-insensitive — included for realism)
    c[2, 1] = 1.1e-5            # odd m
    s[2, 2] = 2.0e-6            # even m sectoral
    c[3, 1] = 2.1e-5
    s[3, 1] = -1.3e-5  # odd m
    c[3, 3] = 4.0e-6
    s[3, 3] = 3.0e-6   # odd m sectoral
    c[4, 1] = -5.0e-6
    s[4, 3] = 1.5e-6  # odd m at higher degree
    return c, s


def _off_axis_points() -> np.ndarray:
    """Points off the equator and prime meridian, so odd-m terms contribute."""
    def unit(v: tuple[float, float, float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64)
        return a / np.linalg.norm(a)

    specs = [
        (unit((0.35, -0.91, 0.22)), 300e3),
        (unit((0.62, 0.41, 0.67)), 1000e3),
        (unit((1.0, 1.0, 0.3)), 100e3),
        (unit((-0.55, 0.73, -0.40)), 2000e3),
    ]
    return np.array([u * (_R + alt) for u, alt in specs], dtype=np.float64)


def _generator_accel(c: np.ndarray, s: np.ndarray, points: np.ndarray) -> np.ndarray:
    a_nm, b_nm, diag_f, subdiag_f, k_ratio = precompute_legendre_constants(_DEGREE)
    # degree_min=-1 includes the structural monopole and every degree n>=1, so the
    # full field matches GravityModel's evaluation.
    _v, accel = _sh_potential_accel_batch_serial(
        points, c, s, a_nm, b_nm, diag_f, subdiag_f, k_ratio, _MU, _R, _DEGREE, -1
    )
    return np.asarray(accel, dtype=np.float64)


def test_generator_matches_engine_on_odd_m_field() -> None:
    """Generator and GravityModel must agree to ~machine precision, odd-m included."""
    c, s = _synthetic_field_with_odd_m()
    points = _off_axis_points()
    gen = _generator_accel(c, s, points)
    model = GravityModel.from_arrays(_DEGREE, _R, _MU, c, s)
    worst = max(
        float(np.linalg.norm(gen[i] - model.accel_fixed(points[i], degree=_DEGREE)))
        for i in range(points.shape[0])
    )
    assert worst < 1e-9, f"generator vs engine acceleration mismatch: {worst:.3e} m/s^2"


def test_parity_test_detects_condon_shortley_phase() -> None:
    """Negative control: a Condon-Shortley field must NOT match the no-CS generator.

    Applying the (-1)^m phase is equivalent to negating every odd-order
    coefficient. Feeding that flipped field to the (no-CS) engine reproduces what
    a CS-phase generator would output; it must disagree with the real generator
    by a wide margin, proving the parity check above is sensitive to the phase.
    """
    c, s = _synthetic_field_with_odd_m()
    points = _off_axis_points()
    gen = _generator_accel(c, s, points)

    c_cs = c.copy()
    s_cs = s.copy()
    for m in range(_DEGREE + 1):
        if m % 2 == 1:
            c_cs[:, m] *= -1.0
            s_cs[:, m] *= -1.0
    model_cs = GravityModel.from_arrays(_DEGREE, _R, _MU, c_cs, s_cs)
    worst_cs = max(
        float(np.linalg.norm(gen[i] - model_cs.accel_fixed(points[i], degree=_DEGREE)))
        for i in range(points.shape[0])
    )
    assert worst_cs > 1e-9, (
        "negative control failed: a Condon-Shortley field matched the no-CS "
        "generator, so the parity test would not catch a phase regression"
    )


@pytest.mark.parametrize("degree_max", [2, 3, 4])
def test_parity_holds_per_degree(degree_max: int) -> None:
    """Agreement must hold at each truncation that introduces new odd-m terms."""
    c, s = _synthetic_field_with_odd_m()
    points = _off_axis_points()
    a_nm, b_nm, diag_f, subdiag_f, k_ratio = precompute_legendre_constants(degree_max)
    _v, gen = _sh_potential_accel_batch_serial(
        points, c[: degree_max + 1, : degree_max + 1], s[: degree_max + 1, : degree_max + 1],
        a_nm, b_nm, diag_f, subdiag_f, k_ratio, _MU, _R, degree_max, -1,
    )
    model = GravityModel.from_arrays(degree_max, _R, _MU, c, s)
    worst = max(
        float(np.linalg.norm(gen[i] - model.accel_fixed(points[i], degree=degree_max)))
        for i in range(points.shape[0])
    )
    assert worst < 1e-9, f"degree {degree_max}: mismatch {worst:.3e} m/s^2"
