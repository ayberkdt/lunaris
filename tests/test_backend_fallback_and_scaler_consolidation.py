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
pytest.importorskip("torch.nn")

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


def test_surrogate_to_device_float64_moves_model_and_scalers(tmp_path):
    art = make_contract_run(tmp_path, degree_min=1)
    model = SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")
    assert model._force_runtime is not None

    model.to_device(torch.device("cpu"), dtype=torch.float64)

    assert next(model.model.parameters()).dtype == torch.float64
    assert model._x_mean.dtype == torch.float64
    assert model._x_scale.dtype == torch.float64
    assert model._u_mean.dtype == torch.float64
    assert model._u_scale.dtype == torch.float64
    assert model._mu_tensor.dtype == torch.float64
    assert model._force_runtime.scaler._x_mean.dtype == torch.float64
    assert model._force_runtime.scaler._u_scale.dtype == torch.float64

    x = torch.tensor([[2.038e6, 2.0e4, -1.0e4]], dtype=torch.float64)
    a_torch = model.predict_residual_accel_torch(x)
    assert a_torch.dtype == torch.float64

    diag = model.dtype_diagnostics(requested_dtype=torch.float64)
    assert diag["requested_dtype"] == "float64"
    assert diag["effective_dtype"] == "float64"
    assert diag["model_dtype"] == "float64"
    assert diag["scaler_dtype"] == "float64"
    assert diag["force_runtime_scaler_dtype"] == "float64"
    assert diag["dtype_downgraded"] is False


def test_surrogate_to_device_float32_roundtrip_moves_all_tensors(tmp_path):
    """float64 -> float32 round-trip: every cached tensor follows, none is left behind."""
    art = make_contract_run(tmp_path, degree_min=1)
    model = SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")
    model.to_device(torch.device("cpu"), dtype=torch.float64)
    model.to_device(torch.device("cpu"), dtype=torch.float32)

    assert next(model.model.parameters()).dtype == torch.float32
    assert model._x_mean.dtype == torch.float32
    assert model._x_scale.dtype == torch.float32
    assert model._u_mean.dtype == torch.float32
    assert model._u_scale.dtype == torch.float32
    assert model._mu_tensor.dtype == torch.float32

    diag = model.dtype_diagnostics(requested_dtype=torch.float32)
    assert diag["requested_dtype"] == "float32"
    assert diag["effective_dtype"] == "float32"
    assert diag["model_dtype"] == "float32"
    assert diag["scaler_dtype"] == "float32"
    assert diag["dtype_downgraded"] is False


def test_surrogate_dtype_diagnostics_flags_downgrade(tmp_path):
    """Requesting float64 against a float32 runtime must be reported as a downgrade."""
    art = make_contract_run(tmp_path, degree_min=1)
    model = SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")
    model.to_device(torch.device("cpu"), dtype=torch.float32)

    diag = model.dtype_diagnostics(requested_dtype=torch.float64)
    assert diag["requested_dtype"] == "float64"
    assert diag["effective_dtype"] == "float32"
    assert diag["dtype_downgraded"] is True


def test_surrogate_runtime_rejects_unsupported_dtype(tmp_path):
    """float16/bfloat16 are not valid surrogate runtime dtypes, via helper AND public path."""
    from lunaris.surrogate.runtime.gravity_provider import _coerce_torch_dtype

    with pytest.raises(ValueError, match="float32 or float64"):
        _coerce_torch_dtype("float16")
    with pytest.raises(ValueError, match="float32 or float64"):
        _coerce_torch_dtype("bfloat16")
    with pytest.raises(ValueError, match="float32 or float64"):
        _coerce_torch_dtype(torch.float16)  # dtype object, not just the string form

    art = make_contract_run(tmp_path, degree_min=1)
    model = SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")
    with pytest.raises(ValueError, match="float32 or float64"):
        model.to_device(torch.device("cpu"), dtype="float16")


def test_torch_batch_diagnostics_report_effective_surrogate_dtype(tmp_path, monkeypatch):
    from lunaris.core.torch_batch_propagator import TorchBatchPropagator

    art = make_contract_run(tmp_path, degree_min=1)
    model = SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")
    model.to_device(torch.device("cpu"), dtype=torch.float64)

    prop = object.__new__(TorchBatchPropagator)
    prop._torch = torch
    prop._device = torch.device("cpu")
    prop._model = model
    prop._dtype = torch.float64
    prop._frame = SimpleNamespace(uses_rotation=False)
    prop._throughput_metrics = {}

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "FakeCUDA")

    diag = prop.diagnostics_snapshot()
    assert diag["requested_dtype"] == "float64"
    assert diag["effective_dtype"] == "float64"
    assert diag["model_dtype"] == "float64"
    assert diag["scaler_dtype"] == "float64"
    assert diag["force_runtime_scaler_dtype"] == "float64"
    assert diag["dtype_downgraded"] is False


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


def _gpu_plan(**overrides):
    """Real BatchBackendPlan resolved to a GPU backend, ready to be downgraded."""
    from lunaris.batch.backend_policy import BatchBackend, BatchBackendPlan

    base = dict(
        final_backend=BatchBackend.GPU_ST_LRPS,
        use_gpu=True,
        gravity_backend="st_lrps",
        torch_cuda_available=True,
        numba_cuda_available=False,
        actual_backend="gpu_st_lrps_potential",
        actual_sh_degree=200,
        actual_device="cuda:0",
        cuda_device_name="FakeGPU",
        dtype="float32",
        integrator="fixed RK4",
    )
    base.update(overrides)
    return BatchBackendPlan(**base)


def test_handle_backend_init_failure_raises_when_forbidden():
    eng = _engine(sh_fallback_policy="error")
    plan = _gpu_plan()
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        eng._handle_backend_init_failure(plan, "[BATCH] GPU init failed.", RuntimeError("cuda oom"))
    # The plan must NOT have been quietly downgraded.
    assert plan.fallback_applied is False
    assert plan.use_gpu is True


def test_handle_backend_init_failure_downgrades_when_allowed():
    eng = _engine(sh_fallback_policy="compatible_gpu")
    plan = _gpu_plan()
    with pytest.warns(RuntimeWarning):
        eng._handle_backend_init_failure(plan, "[BATCH] GPU init failed.", RuntimeError("cuda oom"))
    # Immutable plan: the downgrade replaces the stored plan, the original
    # decision object is untouched.
    assert eng._backend_plan.fallback_applied is True
    assert eng._backend_plan.use_gpu is False
    assert plan.fallback_applied is False
    assert plan.use_gpu is True
    assert eng._backend_note is not None


def test_downgrade_to_cpu_rewrites_full_provenance_st_lrps():
    """A GPU->CPU downgrade must relabel EVERY provenance field so a CPU run is
    never presented as a GPU run (Phase 8: no run labelled GPU when CPU used)."""
    from lunaris.batch.backend_policy import BatchBackend

    plan = _gpu_plan()
    cpu_plan = plan.as_cpu_fallback("cuda unavailable")

    assert cpu_plan.final_backend == BatchBackend.CPU
    assert cpu_plan.use_gpu is False
    assert cpu_plan.actual_backend == "cpu_st_lrps"
    assert cpu_plan.actual_sh_degree is None
    assert cpu_plan.actual_device == "cpu"
    assert cpu_plan.cuda_device_name is None
    assert cpu_plan.dtype == "float64"
    assert cpu_plan.effective_dtype == "float64"
    assert cpu_plan.requested_dtype == "float32"
    assert cpu_plan.dtype_downgraded is True
    assert "DOP853" in cpu_plan.integrator
    assert cpu_plan.fallback_applied is True
    assert cpu_plan.fallback_reason == "cuda unavailable"
    # The original plan is a frozen decision object and must be unchanged.
    assert plan.final_backend == BatchBackend.GPU_ST_LRPS
    assert plan.use_gpu is True
    assert plan.actual_backend == "gpu_st_lrps_potential"
    assert plan.fallback_applied is False


def test_downgrade_to_cpu_uses_cpu_sh_for_classic_backend():
    from lunaris.batch.backend_policy import BatchBackend

    plan = _gpu_plan(
        final_backend=BatchBackend.GPU_CLASSIC_SH,
        gravity_backend="sh",
        torch_cuda_available=False,
        numba_cuda_available=True,
        actual_backend="numba_cuda_sh",
        actual_sh_degree=24,
    )
    cpu_plan = plan.as_cpu_fallback("cuda oom")
    assert cpu_plan.actual_backend == "cpu_sh"
    assert cpu_plan.use_gpu is False
    assert plan.actual_backend == "numba_cuda_sh"


def test_backend_plan_is_immutable():
    """BatchBackendPlan is a frozen decision object: consumers cannot mutate it."""
    import dataclasses

    plan = _gpu_plan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.use_gpu = False  # type: ignore[misc]
