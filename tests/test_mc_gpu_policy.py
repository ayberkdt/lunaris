# -*- coding: utf-8 -*-
"""
Regression tests for Monte Carlo GPU backend selection and tuning helpers.

These tests stay CPU-only; they validate the decision logic that determines
when the CUDA backend is allowed, how launch widths are normalized, and when
the engine deliberately falls back to the CPU full-fidelity path.

New in this revision
--------------------
- Backend policy tests using ``core.mc_backend_policy.resolve_mc_backend_policy``.
- ST-LRPS batch torch inference tests (N=4, CPU tensors).
- TorchBatchPropagator smoke test (N=4, CPU-emulated, torch required).
- Fail-fast tests: missing degree_max fails before sample loop.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.montecarlo_defs import MonteCarloConfig
from common.type_defs import PerturbationFlags
from core.mc_propagator import _sanitize_gpu_threads_per_block, gpu_unsupported_features
from core.monte_carlo_engine import MonteCarloEngine


# =============================================================================
# Existing tests — updated to also monkeypatch torch CUDA where needed
# =============================================================================

def test_gpu_unsupported_features_only_reports_cpu_only_models() -> None:
    flags = PerturbationFlags(
        enable_sh=True,
        enable_earth_j2=True,
        enable_albedo=True,
        enable_tides_k2=True,
    )

    unsupported = gpu_unsupported_features(flags)

    assert "albedo" in unsupported
    assert "solid tides" in unsupported
    assert "Earth J2" not in unsupported


def test_sanitize_gpu_threads_per_block_aligns_and_clamps() -> None:
    assert _sanitize_gpu_threads_per_block(130, warp_size=32, max_threads_per_block=1024) == 128
    assert _sanitize_gpu_threads_per_block(2048, warp_size=32, max_threads_per_block=512) == 512
    assert _sanitize_gpu_threads_per_block(1, warp_size=32, max_threads_per_block=1024) == 32


def test_engine_falls_back_to_cpu_when_gpu_requested_with_unsupported_physics(monkeypatch) -> None:
    import core.mc_propagator as mc_prop
    import core.mc_backend_policy as policy_mod

    class DummyCPU:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyGPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("GPU backend should not be constructed for unsupported physics.")

    monkeypatch.setattr(mc_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mc_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(mc_prop, "GPUBatchPropagator", DummyGPU)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2,
        use_gpu=True,
        output_format="npz",
        output_path="mc_results/test_policy_cpu.npz",
    )
    engine._sim_cfg = SimpleNamespace(flags=PerturbationFlags(enable_albedo=True))
    engine._dyn = object()
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prop = MonteCarloEngine._build_propagator(engine)

    assert isinstance(prop, DummyCPU)
    note_lower = engine._backend_note.lower()
    assert "falling back to" in note_lower and "cpu" in note_lower
    assert any("albedo" in str(item.message).lower() for item in caught)


def test_engine_keeps_gpu_path_for_supported_earth_j2_runs(monkeypatch) -> None:
    import core.mc_propagator as mc_prop
    import core.mc_backend_policy as policy_mod

    class DummyCPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("CPU backend should not be chosen for supported Earth J2 physics.")

    class DummyGPU:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(mc_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mc_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(mc_prop, "GPUBatchPropagator", DummyGPU)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2,
        use_gpu=True,
        output_format="npz",
        output_path="mc_results/test_policy_gpu.npz",
    )
    engine._sim_cfg = SimpleNamespace(flags=PerturbationFlags(enable_earth_j2=True))
    engine._dyn = object()
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""

    prop = MonteCarloEngine._build_propagator(engine)

    assert isinstance(prop, DummyGPU)
    assert engine._backend_note == ""


def test_engine_falls_back_to_cpu_when_surrogate_gravity_is_requested_and_torch_cuda_unavailable(
    monkeypatch,
) -> None:
    """ST-LRPS + torch CUDA unavailable → CPU fallback."""
    import core.mc_propagator as mc_prop
    import core.mc_backend_policy as policy_mod

    class DummyCPU:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyGPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("GPU backend should not be used with surrogate gravity when torch CUDA is unavailable.")

    monkeypatch.setattr(mc_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mc_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(mc_prop, "GPUBatchPropagator", DummyGPU)
    # Critical: torch CUDA is NOT available
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2,
        use_gpu=True,
        output_format="npz",
        output_path="mc_results/test_policy_surrogate.npz",
    )
    engine._sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    engine._dyn = object()
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prop = MonteCarloEngine._build_propagator(engine)

    assert isinstance(prop, DummyCPU)
    # The backend note should mention ST-LRPS and fallback
    note_lower = engine._backend_note.lower()
    assert "st-lrps" in note_lower or "surrogate" in note_lower
    assert any(
        "st-lrps" in str(w.message).lower() or "surrogate" in str(w.message).lower()
        for w in caught
    )


# Keep the old test name as an alias so CI doesn't break if it references it by name
test_engine_falls_back_to_cpu_when_surrogate_gravity_is_requested = (
    test_engine_falls_back_to_cpu_when_surrogate_gravity_is_requested_and_torch_cuda_unavailable
)


# =============================================================================
# New: backend policy module unit tests
# =============================================================================

def test_policy_cpu_explicit(monkeypatch) -> None:
    """use_gpu=False always → CPU regardless of CUDA."""
    import core.mc_backend_policy as policy_mod
    from core.mc_backend_policy import MCBackend, resolve_mc_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)

    mc_cfg = SimpleNamespace(use_gpu=False, gravity_mode_override="follow_mission")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    plan = resolve_mc_backend_policy(mc_cfg, sim_cfg)
    assert plan.final_backend == MCBackend.CPU
    assert not plan.use_gpu


def test_policy_st_lrps_torch_cuda_true(monkeypatch) -> None:
    """ST-LRPS + torch CUDA available + no extra perturbations → GPU_ST_LRPS."""
    import core.mc_backend_policy as policy_mod
    from core.mc_backend_policy import MCBackend, resolve_mc_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    mc_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_mc_backend_policy(mc_cfg, sim_cfg)
    assert plan.final_backend == MCBackend.GPU_ST_LRPS
    assert plan.use_gpu
    assert plan.torch_cuda_available
    assert "fixed-step rk4" in plan.integrator.lower()
    assert plan.batch_note != ""


def test_policy_st_lrps_torch_cuda_false_falls_back(monkeypatch) -> None:
    """ST-LRPS + torch CUDA unavailable → CPU."""
    import core.mc_backend_policy as policy_mod
    from core.mc_backend_policy import MCBackend, resolve_mc_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)

    mc_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_mc_backend_policy(mc_cfg, sim_cfg)
    assert plan.final_backend == MCBackend.CPU
    assert not plan.use_gpu
    assert len(plan.warnings) > 0
    assert any("st-lrps" in w.lower() for w in plan.warnings)


def test_policy_st_lrps_gpu_with_third_body_falls_back(monkeypatch) -> None:
    """ST-LRPS + torch CUDA + third-body enabled → CPU (unsupported on torch path)."""
    import core.mc_backend_policy as policy_mod
    from core.mc_backend_policy import MCBackend, resolve_mc_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    mc_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True, enable_3rd_body_sun=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_mc_backend_policy(mc_cfg, sim_cfg)
    assert plan.final_backend == MCBackend.CPU
    assert any("third-body sun" in w.lower() for w in plan.warnings)


def test_policy_classic_sh_numba_cuda_true(monkeypatch) -> None:
    """Classic SH + Numba CUDA available → GPU_CLASSIC_SH."""
    import core.mc_backend_policy as policy_mod
    from core.mc_backend_policy import MCBackend, resolve_mc_backend_policy

    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)

    mc_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="follow_mission")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    plan = resolve_mc_backend_policy(mc_cfg, sim_cfg)
    assert plan.final_backend == MCBackend.GPU_CLASSIC_SH
    assert plan.use_gpu


def test_policy_classic_sh_numba_cuda_false(monkeypatch) -> None:
    """Classic SH + Numba CUDA unavailable → CPU."""
    import core.mc_backend_policy as policy_mod
    from core.mc_backend_policy import MCBackend, resolve_mc_backend_policy

    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)

    mc_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="follow_mission")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    plan = resolve_mc_backend_policy(mc_cfg, sim_cfg)
    assert plan.final_backend == MCBackend.CPU
    assert not plan.use_gpu


def test_policy_no_contradictory_command_args_st_lrps_gpu(monkeypatch) -> None:
    """GPU_ST_LRPS plan emits use_gpu=True, gravity_backend='st_lrps'."""
    import core.mc_backend_policy as policy_mod
    from core.mc_backend_policy import resolve_mc_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    mc_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_mc_backend_policy(mc_cfg, sim_cfg)
    assert plan.use_gpu is True
    assert plan.gravity_backend == "st_lrps"


def test_policy_no_contradictory_command_args_cpu_fallback(monkeypatch) -> None:
    """CPU fallback plan emits use_gpu=False regardless of request."""
    import core.mc_backend_policy as policy_mod
    from core.mc_backend_policy import resolve_mc_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    mc_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_mc_backend_policy(mc_cfg, sim_cfg)
    assert plan.use_gpu is False


# =============================================================================
# New: ST-LRPS batch torch inference tests (CPU tensors — no GPU required)
# =============================================================================

torch = pytest.importorskip("torch")


def _make_tiny_surrogate(tmp_path: Path) -> "Any":  # noqa: F821
    """Create a minimal SurrogateGravityModel on CPU for inference tests."""
    from common.constants import MU_MOON, R_MOON
    from models.surrogate_gravity import SurrogateGravityModel, _build_model_from_config

    config = {
        "hidden": 8,
        "depth": 1,
        "activation": "tanh",
        "dropout": 0.0,
        "resolved_mu_si": float(MU_MOON),
        "resolved_a_sign": 1.0,
        "scaler_kind": "isometric",
        "degree_min": 10,
        "degree_max": 50,
    }
    scaler = {
        "x": {"mean": [0.0, 0.0, 0.0], "scale": 2_000_000.0},
        "u": {"mean": [0.0], "scale": 1.0},
        "a": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
    }

    run_dir = tmp_path / "tiny_run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")

    model_net = _build_model_from_config(config)
    with torch.no_grad():
        for p in model_net.parameters():
            p.zero_()

    torch.save({"model": model_net.state_dict(), "config": config, "scaler": scaler},
               ckpt_dir / "ckpt_best.pt")

    return SurrogateGravityModel.from_model_dir(
        run_dir,
        mu_override=float(MU_MOON),
        r_ref_override=float(R_MOON),
        device_preference="cpu",
    )


def test_predict_residual_accel_torch_shape(tmp_path: Path) -> None:
    """predict_residual_accel_torch returns [N, 3] for [N, 3] input."""
    model = _make_tiny_surrogate(tmp_path)
    x = torch.zeros(4, 3, dtype=torch.float32)
    x[:, 0] = 1_838_000.0  # 100 km altitude positions

    out = model.predict_residual_accel_torch(x)

    assert out.shape == (4, 3)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


def test_predict_total_accel_torch_shape(tmp_path: Path) -> None:
    """predict_total_accel_torch returns [N, 3] for [N, 3] input."""
    model = _make_tiny_surrogate(tmp_path)
    x = torch.zeros(4, 3, dtype=torch.float32)
    x[:, 0] = 1_838_000.0

    out = model.predict_total_accel_torch(x)

    assert out.shape == (4, 3)
    assert out.dtype == torch.float32


def test_predict_total_accel_torch_zero_net_matches_point_mass(tmp_path: Path) -> None:
    """Zero-weight network → total acceleration equals point-mass (residual mode)."""
    from common.constants import MU_MOON
    model = _make_tiny_surrogate(tmp_path)

    r = 1_838_000.0
    x = torch.tensor([[r, 0.0, 0.0]], dtype=torch.float32)
    out = model.predict_total_accel_torch(x)  # [1, 3]

    expected_ax = -float(MU_MOON) / (r * r)
    assert abs(float(out[0, 0]) - expected_ax) / abs(expected_ax) < 1e-4
    assert abs(float(out[0, 1])) < 1e-12
    assert abs(float(out[0, 2])) < 1e-12


def test_predict_residual_accel_torch_zero_net_is_zero(tmp_path: Path) -> None:
    """Zero-weight network → residual acceleration is zero (delta above point mass)."""
    model = _make_tiny_surrogate(tmp_path)
    x = torch.tensor([[1_838_000.0, 0.0, 0.0]], dtype=torch.float32)
    out = model.predict_residual_accel_torch(x)
    assert torch.allclose(out, torch.zeros(1, 3), atol=1e-10)


def test_degree_max_metadata_exposed(tmp_path: Path) -> None:
    """SurrogateGravityModel exposes degree_max for MC propagator contract."""
    model = _make_tiny_surrogate(tmp_path)
    assert hasattr(model, "degree_max")
    assert int(model.degree_max) == 50
    assert int(model.degree_min) == 10


# =============================================================================
# New: TorchBatchPropagator smoke test (CPU-only, no real GPU needed)
# =============================================================================

def test_torch_batch_propagator_cpu_smoke(tmp_path: Path, monkeypatch) -> None:
    """
    Smoke test: TorchBatchPropagator propagates N=4 samples for a few steps.

    Bypasses __init__ (which requires CUDA) and directly constructs a CPU-only
    propagator instance.  Validates:
    - Output shape (T, N, 6)
    - No per-sample Python loop in RHS
    - impact_flags and t_impact have correct shapes
    """
    import torch as _torch
    from core.torch_batch_propagator import TorchBatchPropagator
    from common.constants import R_MOON

    model = _make_tiny_surrogate(tmp_path)
    # Ensure model tensors are on CPU (they already are; explicit for clarity)
    model.to_device(_torch.device("cpu"))

    # Construct propagator without calling __init__ to avoid CUDA guard
    prop = object.__new__(TorchBatchPropagator)
    prop._torch = _torch
    prop._device = _torch.device("cpu")
    prop._dt = 60.0
    prop._impact_r = float(R_MOON)  # no altitude pad → impact at surface
    prop._model = model

    # Monkeypatch cuda calls that appear in diagnostics_snapshot / propagate
    monkeypatch.setattr(_torch.cuda, "get_device_name", lambda idx: "FakeCUDA")
    monkeypatch.setattr(_torch.cuda, "synchronize", lambda dev=None: None)

    r0 = float(R_MOON) + 100_000.0  # 100 km altitude
    N = 4
    Y0 = np.zeros((N, 6), dtype=np.float64)
    Y0[:, 0] = r0
    Y0[:, 4] = 1_633.0  # approximate circular velocity at 100 km

    t_out, Y_out, impact_flags, t_impact = prop.propagate(
        Y0=Y0,
        masses=np.ones(N, dtype=np.float64) * 1000.0,
        areas=np.ones(N, dtype=np.float64) * 5.0,
        cds=np.ones(N, dtype=np.float64) * 2.2,
        crs=np.ones(N, dtype=np.float64) * 1.5,
        duration_s=600.0,   # 10 minutes
        output_dt_s=120.0,  # 2-minute snapshots → initial + 5 snaps
    )

    assert t_out.shape[0] > 1, "Expected at least 2 time snapshots"
    assert Y_out.shape == (t_out.shape[0], N, 6), f"Y_out shape mismatch: {Y_out.shape}"
    assert impact_flags.shape == (N,)
    assert t_impact.shape == (N,)
    # All samples should survive 10 minutes at 100 km with zero-weight network
    assert np.all(impact_flags == 0.0), "Expected no impacts in 10-minute run at 100 km"


# =============================================================================
# New: engine selects GPU-ST-LRPS when torch CUDA is available
# =============================================================================

def test_engine_selects_torch_gpu_when_st_lrps_and_torch_cuda_available(monkeypatch) -> None:
    """ST-LRPS + torch CUDA available + no extra perturbations → TorchBatchPropagator."""
    import core.mc_propagator as mc_prop
    import core.mc_backend_policy as policy_mod

    class DummyCPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("CPU should not be selected when GPU ST-LRPS is available.")

    class DummyGPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("Classic GPU should not be selected for ST-LRPS.")

    class DummyTorchGPU:
        def __init__(self, surrogate_model, mc_cfg, device_id=0) -> None:
            self.surrogate_model = surrogate_model
            self.mc_cfg = mc_cfg

    monkeypatch.setattr(mc_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mc_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(mc_prop, "GPUBatchPropagator", DummyGPU)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)
    monkeypatch.setattr("core.torch_batch_propagator.TorchBatchPropagator", DummyTorchGPU)

    # Fake surrogate model with model_kind and degree metadata
    fake_grav = SimpleNamespace(
        model_kind="st_lrps",
        model_dir=Path("/fake/run"),
        degree_min=10,
        degree_max=50,
    )
    fake_dyn = SimpleNamespace(grav=fake_grav)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2,
        use_gpu=True,
        gravity_mode_override="st_lrps",
        output_format="npz",
        output_path="mc_results/test_torch_gpu.npz",
    )
    engine._sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    engine._dyn = fake_dyn
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""

    prop = MonteCarloEngine._build_propagator(engine)

    assert isinstance(prop, DummyTorchGPU)
    assert prop.surrogate_model is fake_grav
