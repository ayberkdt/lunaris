"""Task 3 — explicit, safe runtime frame handling for the ST-LRPS force model.

The surrogate is a Moon-fixed (body-fixed) model. These tests pin the frame
contract: the ``_fixed`` methods equal the legacy aliases; the ``_inertial``
helpers reduce to the fixed methods under identity rotation, preserve norms and
potential under a real rotation, and a wrong-frame artifact fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.common.math_utils import quat_rotate_np
from lunaris.surrogate.st_lrps.runtime.force_model import (
    SUPPORTED_RUNTIME_FRAME,
    SurrogateForceModel,
    _quat_to_rotation_matrix,
    _rotate_fixed_to_inertial,
    _rotate_inertial_to_fixed,
)
from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack


def _make_fm(**ctor_overrides):
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    cfg = {
        "resolved_mu_si": 4.902e12,
        "resolved_a_sign": 1.0,
        "resolved_r_ref_m": 1.737e6,
        "degree_min": -1,
        "altitude_min_km": 50.0,
        "altitude_max_km": 600.0,
    }
    return SurrogateForceModel(model=model, scaler=sp, cfg=cfg, device=torch.device("cpu"), **ctor_overrides)


def _sample_positions(n=16, seed=1):
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = 1.737e6 + rng.uniform(100e3, 400e3, (n, 1))
    return r * dirs


def _random_unit_quat(seed=2):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def test_rotation_matrix_matches_canonical_kernel():
    q = _random_unit_quat(seed=5)
    rot = _quat_to_rotation_matrix(q)
    v = np.array([1.2, -3.4, 5.6])
    np.testing.assert_allclose(rot @ v, quat_rotate_np(q, v), rtol=1e-9, atol=1e-9)
    # Orthonormal rotation matrix.
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)


def test_inertial_fixed_round_trip_is_identity():
    q = _random_unit_quat(seed=6)
    v = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    there = _rotate_inertial_to_fixed(v, q)
    back = _rotate_fixed_to_inertial(there, q)
    np.testing.assert_allclose(back, v, rtol=1e-10, atol=1e-9)


def test_fixed_methods_match_legacy_aliases():
    fm = _make_fm()
    x = _sample_positions()
    np.testing.assert_allclose(
        fm.predict_residual_potential_fixed(x), fm.predict_residual_potential(x), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        fm.predict_residual_accel_fixed(x), fm.predict_residual_accel(x), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        fm.predict_total_accel_fixed(x), fm.predict_total_accel(x), rtol=0, atol=0
    )


def test_inertial_identity_quaternion_matches_fixed():
    fm = _make_fm()
    x = _sample_positions()
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(
        fm.predict_residual_accel_inertial(x, q_identity),
        fm.predict_residual_accel_fixed(x),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fm.predict_residual_potential_inertial(x, q_identity),
        fm.predict_residual_potential_fixed(x),
        rtol=1e-10,
        atol=1e-12,
    )


def test_inertial_rotation_preserves_norm_and_potential():
    fm = _make_fm()
    q = _random_unit_quat(seed=9)
    r_inertial = _sample_positions(seed=4)

    a_inertial = fm.predict_residual_accel_inertial(r_inertial, q)
    # Equivalent fixed-frame evaluation at the rotated position.
    r_fixed = _rotate_inertial_to_fixed(r_inertial, q)
    a_fixed = fm.predict_residual_accel_fixed(r_fixed)

    # Rotation is orthogonal: per-row acceleration magnitude is invariant.
    np.testing.assert_allclose(
        np.linalg.norm(a_inertial, axis=1), np.linalg.norm(a_fixed, axis=1), rtol=1e-9, atol=1e-12
    )
    # The inertial accel is exactly the fixed accel rotated back out.
    np.testing.assert_allclose(a_inertial, _rotate_fixed_to_inertial(a_fixed, q), rtol=1e-9, atol=1e-12)
    # Potential is frame-invariant.
    np.testing.assert_allclose(
        fm.predict_residual_potential_inertial(r_inertial, q),
        fm.predict_residual_potential_fixed(r_fixed),
        rtol=1e-10,
        atol=1e-12,
    )


def test_single_vector_inertial_shapes():
    fm = _make_fm()
    q = _random_unit_quat(seed=11)
    r = np.array([1.937e6, 0.0, 0.0])
    a = fm.predict_residual_accel_inertial(r, q)
    assert a.shape == (3,)


def test_wrong_frame_artifact_raises():
    good = _make_fm()
    assert good.frame == SUPPORTED_RUNTIME_FRAME
    bad_contract = good.artifact_contract.to_dict()
    bad_contract["dataset_contract"]["coordinate_frame"] = "moon_centered_inertial"
    with pytest.raises(ValueError, match="moon_fixed_cartesian"):
        _make_fm(artifact_contract=bad_contract)
