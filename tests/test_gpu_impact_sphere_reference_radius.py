"""Batch/GPU impact sphere must match the CPU path's radius.

The CPU single-orbit path detects a sphere impact at
``get_ref_radius_and_mu(dynamics)[0] + impact_alt`` -- the loaded gravity
model's reference radius (``R_MOON`` only when no gravity model is loaded).
The torch classic-SH, torch ST-LRPS and numba batch propagators hard-coded
``R_MOON``, so a model whose reference radius is not 1738 km (any non-SHADR
model, or an ST-LRPS artifact with its own ``R_ref_m``) detected impacts on a
different sphere than the CPU path for the same run.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.core.propagation.time_grid import get_ref_radius_and_mu

torch = pytest.importorskip("torch")

from lunaris.common.type_defs import PerturbationFlags  # noqa: E402
from lunaris.core.torch_sh_propagator import TorchSHBatchPropagator  # noqa: E402
from lunaris.physics.spherical_harmonics import GravityModel  # noqa: E402

# A reference radius deliberately different from R_MOON (1738 km).
_R_REF = 1_750_000.0
_DEG = 2


def _grav(r_ref: float) -> GravityModel:
    c = np.zeros((_DEG + 1, _DEG + 1))
    return GravityModel.from_arrays(_DEG, r_ref, float(MU_MOON), c, np.zeros_like(c))


def _torch_sh(r_ref: float) -> TorchSHBatchPropagator:
    cfg = SimpleNamespace(
        sh_degree=_DEG, dt_s=30.0, impact_alt_km=0.0, torch_dtype="float64",
        torch_sh_chunk_size=0, gpu_device_id=0,
    )
    dyn = SimpleNamespace(grav=_grav(r_ref), ephem=None)
    return TorchSHBatchPropagator(
        dyn, cfg, PerturbationFlags(enable_sh=True), device="cpu", dtype=torch.float64
    )


def _state(r: float) -> np.ndarray:
    v = math.sqrt(float(MU_MOON) / r)
    return np.array([[r, 0.0, 0.0, 0.0, v, 0.0]])


def test_cpu_reference_radius_is_the_model_radius() -> None:
    assert get_ref_radius_and_mu(SimpleNamespace(grav=_grav(_R_REF)))[0] == _R_REF
    assert get_ref_radius_and_mu(SimpleNamespace(grav=None))[0] == float(R_MOON)


def test_torch_sh_impact_sphere_uses_model_reference_radius() -> None:
    # Between R_MOON and the model radius: inside the CPU impact sphere.
    y0 = _state(0.5 * (float(R_MOON) + _R_REF))
    prop = _torch_sh(_R_REF)
    _, _, impact, t_imp = prop.propagate(
        y0, *([np.ones(1)] * 4), duration_s=60.0, output_dt_s=30.0
    )
    assert impact[0] == 1.0
    assert t_imp[0] == 0.0


def test_torch_sh_shipped_model_radius_is_unchanged() -> None:
    # With the shipped SHADR radius (== R_MOON) the sphere is bit-identical.
    # 1 km: above the ~183 m chord sagitta of a 30 s step at this radius, which
    # the segment-crossing test would otherwise (correctly by its own rule) flag.
    y0 = _state(float(R_MOON) + 1000.0)
    prop = _torch_sh(float(R_MOON))
    _, _, impact, _ = prop.propagate(y0, *([np.ones(1)] * 4), duration_s=60.0, output_dt_s=30.0)
    assert impact[0] == 0.0


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_torch_st_lrps_impact_sphere_uses_surrogate_reference_radius() -> None:
    from lunaris.core.torch_batch_propagator import TorchBatchPropagator

    class _FakeSurrogate:
        R_ref_m = _R_REF

        def to_device(self, device, dtype=None):
            return self

    cfg = SimpleNamespace(dt_s=30.0, impact_alt_km=0.25, torch_dtype="float64")
    prop = TorchBatchPropagator(_FakeSurrogate(), cfg, allow_identity_rotation=True)
    assert prop._impact_r == pytest.approx(_R_REF + 250.0, abs=0.0)
