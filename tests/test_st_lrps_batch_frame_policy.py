"""Regression tests for the ST-LRPS ``--batch-rk4`` frame policy contract."""

from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import argparse

import numpy as np
import pytest

from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.compute import (
    _resolve_batch_frame_modes,
    _run_batch_rk4_cpu,
    run_st_lrps_batch_rk4,
)


def _args(**overrides) -> argparse.Namespace:
    base = {
        "batch_frame_mode": "moon_fixed_ephemeris",
        "batch_size": None,
        "gpu_fallback": "cpu",
        "torch_dtype": "float64",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_batch_precomputed_request_uses_canonical_effective_frame() -> None:
    frame = _resolve_batch_frame_modes(
        _args(batch_frame_mode="precomputed_slerp")
    )
    assert frame.requested_frame_mode == "precomputed_slerp"
    assert frame.effective_frame_mode == "moon_fixed_ephemeris"
    assert frame.frame_interpolation == "dynamic_slerp"
    assert frame.uses_frame_rotation is True


def test_legacy_batch_default_frame_requires_ephemeris_before_cuda_probe() -> None:
    with pytest.raises(RuntimeError, match="requires an ephemeris"):
        run_st_lrps_batch_rk4(
            object(),
            np.zeros((1, 6), dtype=np.float64),
            duration_s=60.0,
            dt_s=10.0,
            output_dt_s=60.0,
            args=_args(batch_frame_mode="moon_fixed_ephemeris"),
            ephem=None,
        )


class _ConstantFixedAccelSurrogate:
    def acceleration_fixed_batch(self, positions_m: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), (positions_m.shape[0], 1))


class _QuarterTurnEphem:
    """90 degree z-rotation: inertial x maps to fixed y."""

    def transform_inertial_to_fixed(self, _t_s: float, v_inertial: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(v_inertial, dtype=np.float64)
        return np.array([-y, x, z], dtype=np.float64)

    def transform_fixed_to_inertial(self, _t_s: float, v_fixed: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(v_fixed, dtype=np.float64)
        return np.array([y, -x, z], dtype=np.float64)


def test_cpu_batch_rk4_rotates_fixed_acceleration_back_to_inertial() -> None:
    y0 = np.zeros((1, 6), dtype=np.float64)
    surrogate = _ConstantFixedAccelSurrogate()

    frame_correct = _run_batch_rk4_cpu(
        surrogate,
        y0,
        duration_s=1.0,
        dt_s=1.0,
        output_dt_s=1.0,
        ephem=_QuarterTurnEphem(),
        requested_frame_mode="moon_fixed_ephemeris",
        effective_frame_mode="moon_fixed_ephemeris",
    )
    legacy = _run_batch_rk4_cpu(
        surrogate,
        y0,
        duration_s=1.0,
        dt_s=1.0,
        output_dt_s=1.0,
        ephem=None,
        requested_frame_mode="identity_diagnostic",
        effective_frame_mode="identity_diagnostic",
    )

    # The surrogate always returns +Y in fixed coordinates. With a 90 degree
    # q_i2f, frame-correct propagation must turn that into +X inertial
    # acceleration; legacy identity propagation leaves it in +Y.
    assert frame_correct["uses_frame_rotation"] is True
    assert legacy["uses_frame_rotation"] is False
    assert frame_correct["Y"][-1, 0, 3] == pytest.approx(1.0)
    assert frame_correct["Y"][-1, 0, 4] == pytest.approx(0.0)
    assert legacy["Y"][-1, 0, 3] == pytest.approx(0.0)
    assert legacy["Y"][-1, 0, 4] == pytest.approx(1.0)
