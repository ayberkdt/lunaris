# tests/test_backend_fallback_and_scaler_consolidation.py
"""Review findings #3 and #4.

#3 (single scaler source): the potential_autograd GPU-tensor path now scales with
   the canonical runtime's scaler when it is loaded (the model is already shared),
   instead of an independently-loaded second scaler. The two must agree exactly.

#4 (no silent benchmark downgrade): a GPU backend that fails to *initialize* must
   hard-fail -- not silently downgrade to CPU -- when fallback is forbidden
   (``sh_fallback_policy='error'`` or a paper-safe / strict-backend flag).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.batch.engine import BatchPropagationEngine
from lunaris.surrogate.runtime import SurrogateGravityModel
from st_lrps_contract_test_utils import make_contract_run

# ---------------------------------------------------------------------------
# #3 Single scaler source (canonical runtime)
# ---------------------------------------------------------------------------

def test_potential_autograd_torch_uses_canonical_scaler(tmp_path):
    # degree_min=1 keeps the SH baseline (data-file) branch out of the picture
    # while still loading through the canonical runtime.
    art = make_contract_run(tmp_path, degree_min=1)
    model = SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")
    assert model._force_runtime is not None  # canonical runtime loaded

    # Inside the [100, 1000] km training envelope so the domain guard stays quiet.
    x = torch.tensor([[2.038e6, 2.0e4, -1.0e4], [2.20e6, 0.0, 0.0]], dtype=torch.float32)

    # The canonical scaler and gravity_provider's local scaler must be byte-identical,
    # so delegating the tensor path to the canonical scaler is a numerical no-op --
    # it removes a *divergence source*, it does not change results.
    a_canon = model._force_runtime.scaler.scale_x(x)
    a_local = model._scale_x(x)
    assert torch.allclose(a_canon, a_local, atol=0.0)

    fr = model._force_runtime
    assert torch.allclose(fr.scaler._u_scale / fr.scaler._x_scale, model._u_scale / model._x_scale, atol=0.0)

    # End-to-end: the GPU-tensor residual matches the canonical numpy residual.
    a_torch = model.predict_residual_accel_torch(x).cpu().numpy()
    a_numpy = np.asarray(fr.predict_residual_accel(x.cpu().numpy()), dtype=np.float64)
    assert np.allclose(a_torch, a_numpy, atol=1e-9)


# ---------------------------------------------------------------------------
# #4 GPU init-failure must not silently downgrade under a strict policy
# ---------------------------------------------------------------------------

def _engine(**cfg_attrs) -> BatchPropagationEngine:
    eng = BatchPropagationEngine.__new__(BatchPropagationEngine)
    eng._cfg = SimpleNamespace(**cfg_attrs)  # type: ignore
    eng._backend_note = None  # type: ignore
    return eng


def test_fallback_forbidden_on_policy_error():
    assert _engine(sh_fallback_policy="error")._fallback_forbidden() is True


def test_fallback_allowed_by_default():
    assert _engine(sh_fallback_policy="compatible_gpu")._fallback_forbidden() is False
    assert _engine(sh_fallback_policy="cpu")._fallback_forbidden() is False
    assert _engine()._fallback_forbidden() is False  # attribute absent -> default allows


@pytest.mark.parametrize("flag", ["paper_safe", "strict_backend", "benchmark_mode"])
def test_fallback_forbidden_by_explicit_flag(flag):
    assert _engine(sh_fallback_policy="compatible_gpu", **{flag: True})._fallback_forbidden() is True


def test_handle_backend_init_failure_raises_when_forbidden():
    eng = _engine(sh_fallback_policy="error")
    plan = SimpleNamespace(gravity_backend="st_lrps")
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        eng._handle_backend_init_failure(plan, "[BATCH] GPU init failed.", RuntimeError("cuda oom"))
    # The plan must NOT have been quietly downgraded.
    assert not getattr(plan, "fallback_applied", False)


def test_handle_backend_init_failure_downgrades_when_allowed():
    eng = _engine(sh_fallback_policy="compatible_gpu")
    plan = SimpleNamespace(gravity_backend="st_lrps")
    with pytest.warns(RuntimeWarning):
        eng._handle_backend_init_failure(plan, "[BATCH] GPU init failed.", RuntimeError("cuda oom"))
    assert plan.fallback_applied is True
    assert plan.use_gpu is False
    assert eng._backend_note is not None
