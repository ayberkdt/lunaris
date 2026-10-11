"""Frozen-search element frame: elements are measured against the lunar equator
of epoch whenever the gravity field rotates with the Moon-fixed frame."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.analysis.frozen.classify import (
    FrozenClassificationConfig,
    classify_candidate,
)
from lunaris.analysis.frozen.metrics import compute_frozen_metrics
from lunaris.analysis.frozen.search import (
    ELEMENT_FRAME_INERTIAL_IDENTITY,
    ELEMENT_FRAME_LUNAR_EQUATOR,
    ElementBounds,
    elements_to_states,
    quaternion_i2f_to_matrix,
    resolve_element_frame,
    rotate_states_element_frame,
)
from lunaris.common.math_utils import quat_rotate_np

MU = 4902.8003063302e9


def _rotation_about_x(angle_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _inclination_deg(state: np.ndarray) -> float:
    h = np.cross(state[:3], state[3:])
    return math.degrees(math.acos(h[2] / np.linalg.norm(h)))


def test_identity_rotation_is_bit_identical_to_the_plain_conversion():
    el = np.array([[1990.0, 0.06, 83.5, 18.1, 270.0, 356.2]])
    plain = elements_to_states(el, MU)
    assert np.array_equal(plain, elements_to_states(el, MU, frame_rotation=None))
    assert np.array_equal(plain, elements_to_states(el, MU, frame_rotation=np.eye(3)))
    assert rotate_states_element_frame(plain, None) is plain


def test_elements_are_measured_in_the_rotated_frame():
    # An equatorial orbit in a frame tilted 23.5 deg about x is inclined
    # 23.5 deg in the integration frame and 0 deg in the element frame.
    rot = _rotation_about_x(23.5)
    el = np.array([[2000.0, 0.05, 0.0, 0.0, 90.0, 10.0]])
    inertial = elements_to_states(el, MU, frame_rotation=rot)
    assert _inclination_deg(inertial[0]) == pytest.approx(23.5, abs=1e-9)
    back = rotate_states_element_frame(inertial, rot)
    assert _inclination_deg(back[0]) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(back, elements_to_states(el, MU), rtol=0.0, atol=1e-6)


def test_quaternion_matrix_matches_the_propagator_rotation():
    q = np.array([0.9, 0.1, -0.3, 0.2])
    q /= np.linalg.norm(q)
    mat = quaternion_i2f_to_matrix(q)
    v = np.array([0.3, -0.5, 0.81])
    assert np.allclose(mat @ v, quat_rotate_np(q, v), rtol=0.0, atol=1e-15)
    assert np.allclose(mat @ mat.T, np.eye(3), rtol=0.0, atol=1e-15)


def test_non_rotation_matrix_is_rejected():
    with pytest.raises(ValueError, match="proper rotation"):
        rotate_states_element_frame(np.zeros((1, 6)), np.diag([1.0, 1.0, -1.0]))


def test_resolve_element_frame_prefers_a_rotating_backend():
    q = np.array([math.cos(0.2), math.sin(0.2), 0.0, 0.0])
    ephem = SimpleNamespace(get_inertial_to_fixed_rotation=lambda t: q)
    identity = SimpleNamespace(provenance={"uses_frame_rotation": False}, ephem_manager=None)
    rotating = SimpleNamespace(provenance={"uses_frame_rotation": True}, ephem_manager=ephem)

    mat, info = resolve_element_frame(("validation", identity), ("screening", rotating))
    assert info["frame"] == ELEMENT_FRAME_LUNAR_EQUATOR
    assert "screening" in info["rotation_source"]
    assert np.allclose(mat, quaternion_i2f_to_matrix(q))

    mat, info = resolve_element_frame(("validation", identity), ("screening", identity))
    assert mat is None and info["frame"] == ELEMENT_FRAME_INERTIAL_IDENTITY


def test_rotating_backend_without_ephemeris_fails_closed():
    broken = SimpleNamespace(provenance={"uses_frame_rotation": True}, ephem_manager=None)
    with pytest.raises(RuntimeError, match="ephem_manager"):
        resolve_element_frame(("validation", broken))


def test_eccentricity_upper_bound_must_be_elliptic():
    with pytest.raises(ValueError, match="< 1"):
        ElementBounds(a_km=(1800.0, 2000.0), e=(0.0, 1.0), i_deg=(0.0, 180.0))


def _flat_metrics(duration_s: float):
    t = np.linspace(0.0, duration_s, 200)
    return compute_frozen_metrics(
        t,
        eccentricity=np.full_like(t, 0.05),
        inclination_rad=np.full_like(t, math.radians(80.0)),
        omega_rad=np.full_like(t, math.radians(90.0)),
        h_peri_m=np.full_like(t, 100_000.0),
    )


def test_short_history_cannot_be_labelled_validated():
    cfg = FrozenClassificationConfig.for_mission_duration(60 * 86_400.0)
    kw = {"validation_backend": "classical_sh_deg200", "long_horizon_validation_passed": True}

    full = classify_candidate(_flat_metrics(60 * 86_400.0), cfg, **kw)
    assert full.validated and full.status == "strict_frozen"

    short = classify_candidate(_flat_metrics(30 * 86_400.0), cfg, **kw)
    assert not short.validated
    assert short.status == "candidate_frozen_orbit"
    assert any("shorter than the configured mission duration" in r for r in short.reasons)
