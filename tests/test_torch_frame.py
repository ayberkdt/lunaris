from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)


import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.core.torch_frame import (  # noqa: E402
    TorchFrameError,
    TorchMoonFrame,
    quat_conjugate_torch,
    quat_rotate_torch,
)


def _ephem() -> SimpleNamespace:
    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]
    return SimpleNamespace(
        get_data_provider=lambda: {
            "dt_s": 10.0,
            "q_i2f_tab": np.asarray([q0, q1], dtype=np.float64),
        }
    )


def test_frame_requires_ephemeris_unless_identity_is_explicit() -> None:
    with pytest.raises(TorchFrameError, match="ephemeris"):
        TorchMoonFrame(
            None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
    frame = TorchMoonFrame(
        None,
        device=torch.device("cpu"),
        dtype=torch.float64,
        allow_identity=True,
    )
    assert not frame.uses_rotation


def test_nonidentity_frame_round_trip_uses_conjugate() -> None:
    frame = TorchMoonFrame(
        _ephem(),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    vector = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    fixed = frame.inertial_to_fixed(10.0, vector)
    np.testing.assert_allclose(fixed.numpy(), [[0.0, 1.0, 0.0]], atol=1e-12)
    inertial = frame.fixed_to_inertial(10.0, fixed)
    np.testing.assert_allclose(inertial.numpy(), vector.numpy(), atol=1e-12)


def test_stage_epoch_slerp_changes_rotation() -> None:
    frame = TorchMoonFrame(
        _ephem(),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    vector = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    at_start = quat_rotate_torch(frame.quat_i2f(0.0), vector)
    at_mid = quat_rotate_torch(frame.quat_i2f(5.0), vector)
    at_end = quat_rotate_torch(frame.quat_i2f(10.0), vector)
    assert not torch.allclose(at_start, at_mid)
    assert not torch.allclose(at_mid, at_end)
    back = quat_rotate_torch(
        quat_conjugate_torch(frame.quat_i2f(5.0)),
        at_mid,
    )
    np.testing.assert_allclose(back.numpy(), vector.numpy(), atol=1e-12)


def test_precompute_rk_stage_quaternions() -> None:
    frame = TorchMoonFrame(
        _ephem(),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    # 2 steps, dt_eff=5.0. stages: 0.0, 2.5
    stage_time_offsets = [0.0, 2.5]
    dt_eff = 5.0
    cache = frame.precompute_rk_stage_quaternions(2, dt_eff, stage_time_offsets)

    assert cache.q_i2f.shape == (2, 2, 4)
    assert cache.q_f2i.shape == (2, 2, 4)

    # Step 0, stage 0 -> t = 0.0
    expected_0_0 = frame.quat_i2f(0.0)
    np.testing.assert_allclose(cache.q_i2f[0, 0].numpy(), expected_0_0.numpy(), atol=1e-12)

    # Step 0, stage 1 -> t = 2.5
    expected_0_1 = frame.quat_i2f(2.5)
    np.testing.assert_allclose(cache.q_i2f[0, 1].numpy(), expected_0_1.numpy(), atol=1e-12)

    # Step 1, stage 0 -> t = 5.0
    expected_1_0 = frame.quat_i2f(5.0)
    np.testing.assert_allclose(cache.q_i2f[1, 0].numpy(), expected_1_0.numpy(), atol=1e-12)

    # Step 1, stage 1 -> t = 7.5
    expected_1_1 = frame.quat_i2f(7.5)
    np.testing.assert_allclose(cache.q_i2f[1, 1].numpy(), expected_1_1.numpy(), atol=1e-12)

    # Verify conjugate
    np.testing.assert_allclose(
        cache.q_f2i[..., 0].numpy(), cache.q_i2f[..., 0].numpy(), atol=1e-12
    )
    np.testing.assert_allclose(
        cache.q_f2i[..., 1:].numpy(), -cache.q_i2f[..., 1:].numpy(), atol=1e-12
    )

