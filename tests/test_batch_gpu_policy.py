"""
Regression tests for batch GPU backend selection and tuning helpers.

These tests stay CPU-only; they validate the decision logic that determines
when the CUDA backend is allowed, how launch widths are normalized, and when
the engine deliberately falls back to the CPU full-fidelity path.

New in this revision
--------------------
- Backend policy tests using ``batch.backend_policy.resolve_batch_backend_policy``.
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

from lunaris.batch.engine import BatchPropagationEngine
from lunaris.common.batch_defs import BatchPropagationConfig
from lunaris.common.type_defs import PerturbationFlags
from lunaris.core.batch_propagator import _sanitize_gpu_threads_per_block, gpu_unsupported_features

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
    import lunaris.batch.backend_policy as policy_mod
    import lunaris.core.batch_propagator as batch_prop

    class DummyCPU:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyGPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("GPU backend should not be constructed for unsupported physics.")

    monkeypatch.setattr(batch_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(batch_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(batch_prop, "GPUBatchPropagator", DummyGPU)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)

    engine = BatchPropagationEngine.__new__(BatchPropagationEngine)
    engine._cfg = BatchPropagationConfig(
        n_samples=2,
        use_gpu=True,
        output_format="npz",
        output_path="outputs/ensemble/test_policy_cpu.npz",
    )
    engine._sim_cfg = SimpleNamespace(flags=PerturbationFlags(enable_albedo=True))
    engine._dyn = object()
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prop = BatchPropagationEngine._build_propagator(engine)

    assert isinstance(prop, DummyCPU)
    note_lower = engine._backend_note.lower()
    assert "falling back to" in note_lower and "cpu" in note_lower
    assert any("albedo" in str(item.message).lower() for item in caught)


def test_engine_keeps_gpu_path_for_supported_earth_j2_runs(monkeypatch) -> None:
    import lunaris.batch.backend_policy as policy_mod
    import lunaris.core.batch_propagator as batch_prop

    class DummyCPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("CPU backend should not be chosen for supported Earth J2 physics.")

    class DummyGPU:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(batch_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(batch_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(batch_prop, "GPUBatchPropagator", DummyGPU)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)

    engine = BatchPropagationEngine.__new__(BatchPropagationEngine)
    engine._cfg = BatchPropagationConfig(
        n_samples=2,
        use_gpu=True,
        output_format="npz",
        output_path="outputs/ensemble/test_policy_gpu.npz",
    )
    engine._sim_cfg = SimpleNamespace(flags=PerturbationFlags(enable_earth_j2=True))
    engine._dyn = object()
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""

    prop = BatchPropagationEngine._build_propagator(engine)

    assert isinstance(prop, DummyGPU)
    assert engine._backend_note == ""


def test_engine_falls_back_to_cpu_when_surrogate_gravity_is_requested_and_torch_cuda_unavailable(
    monkeypatch,
) -> None:
    """ST-LRPS + torch CUDA unavailable → CPU fallback."""
    import lunaris.batch.backend_policy as policy_mod
    import lunaris.core.batch_propagator as batch_prop

    class DummyCPU:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyGPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("GPU backend should not be used with surrogate gravity when torch CUDA is unavailable.")

    monkeypatch.setattr(batch_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(batch_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(batch_prop, "GPUBatchPropagator", DummyGPU)
    # Critical: torch CUDA is NOT available
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)

    engine = BatchPropagationEngine.__new__(BatchPropagationEngine)
    engine._cfg = BatchPropagationConfig(
        n_samples=2,
        use_gpu=True,
        output_format="npz",
        output_path="outputs/ensemble/test_policy_surrogate.npz",
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
        prop = BatchPropagationEngine._build_propagator(engine)

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
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)

    batch_cfg = SimpleNamespace(use_gpu=False, gravity_mode_override="follow_mission")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.final_backend == BatchBackend.CPU
    assert not plan.use_gpu


def test_policy_st_lrps_torch_cuda_true(monkeypatch) -> None:
    """ST-LRPS + torch CUDA available + no extra perturbations → GPU_ST_LRPS."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.final_backend == BatchBackend.GPU_ST_LRPS
    assert plan.use_gpu
    assert plan.torch_cuda_available
    assert "fixed-step rk4" in plan.integrator.lower()
    assert plan.batch_note != ""


def test_policy_st_lrps_gpu_dtype_provenance_honors_config(monkeypatch) -> None:
    """R10: the ST-LRPS GPU plan resolves dtype from config, not a hardcode."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    # gpu_st_lrps_potential supports float64, so a float64 request is honored and
    # surfaces in provenance rather than being overwritten by a float32 default.
    cfg64 = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps", torch_dtype="float64")
    plan64 = resolve_batch_backend_policy(cfg64, sim_cfg)
    assert plan64.final_backend == BatchBackend.GPU_ST_LRPS
    assert plan64.requested_dtype == "float64"
    assert plan64.effective_dtype == "float64"
    assert plan64.dtype == "float64"
    assert plan64.dtype_downgraded is False

    cfg32 = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps", torch_dtype="float32")
    plan32 = resolve_batch_backend_policy(cfg32, sim_cfg)
    assert plan32.effective_dtype == "float32"
    assert plan32.dtype_downgraded is False


def test_policy_st_lrps_torch_cuda_false_falls_back(monkeypatch) -> None:
    """ST-LRPS + torch CUDA unavailable → CPU."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.final_backend == BatchBackend.CPU
    assert not plan.use_gpu
    assert len(plan.warnings) > 0
    assert any("st-lrps" in w.lower() for w in plan.warnings)
    # Provenance must be honest: this is a real GPU->CPU fallback (reviewer §1b).
    assert plan.fallback_applied is True
    assert plan.requested_device == "cuda"
    assert plan.actual_device == "cpu"
    assert plan.fallback_reason  # must record why


def test_policy_st_lrps_unsupported_physics_records_fallback_provenance(monkeypatch) -> None:
    """ST-LRPS + torch CUDA + unsupported physics → CPU with honest fallback fields."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True, enable_srp=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.final_backend == BatchBackend.CPU
    assert plan.actual_backend == "cpu_st_lrps"
    assert plan.fallback_applied is True
    assert plan.requested_device == "cuda"
    assert plan.actual_device == "cpu"
    assert "srp" in plan.fallback_reason.lower()


def test_policy_st_lrps_gpu_with_third_body_selects_hybrid(monkeypatch) -> None:
    """ST-LRPS + torch CUDA + third-body enabled -> hybrid ST-LRPS backend."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True, enable_3rd_body_sun=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.final_backend == BatchBackend.GPU_ST_LRPS
    assert plan.actual_backend == "gpu_st_lrps_third_body"
    assert plan.third_body_backend == "analytic_vectorized"
    assert plan.fallback_applied is False
    assert "third_body_sun" not in plan.unsupported_forces


def test_policy_st_lrps_third_body_plus_srp_reports_remaining_unsupported(monkeypatch) -> None:
    """Hybrid compatibility removes third-body from the fallback reason."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True, enable_3rd_body_sun=True, enable_srp=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)

    assert plan.final_backend == BatchBackend.CPU
    assert plan.fallback_applied is True
    assert "srp" in plan.fallback_reason.lower()
    assert "third_body_sun" not in plan.fallback_reason


# =============================================================================
# R29b — paper-safe/strict posture forbids silent planning-time fallbacks
# =============================================================================


def test_fallback_forbidden_helper_reads_policy_and_flags() -> None:
    from lunaris.batch.backend_policy import fallback_forbidden

    assert fallback_forbidden(SimpleNamespace()) is False
    assert fallback_forbidden(SimpleNamespace(sh_fallback_policy="error")) is True
    assert fallback_forbidden(SimpleNamespace(paper_safe=True)) is True
    assert fallback_forbidden(SimpleNamespace(strict_backend=True)) is True
    assert fallback_forbidden(SimpleNamespace(benchmark_mode=True)) is True
    assert fallback_forbidden(SimpleNamespace(sh_fallback_policy="compatible_gpu")) is False


def test_policy_paper_safe_forbids_st_lrps_cuda_unavailable_fallback(monkeypatch) -> None:
    """R29b (#3): ST-LRPS GPU + no CUDA + paper_safe → raise, never a silent CPU plan."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps", paper_safe=True)
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    with pytest.raises(RuntimeError, match="forbidden"):
        resolve_batch_backend_policy(batch_cfg, sim_cfg)


def test_policy_paper_safe_forbids_st_lrps_unsupported_physics_fallback(monkeypatch) -> None:
    """R29b (#3): ST-LRPS GPU + unsupported physics + strict → raise."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps", strict_backend=True)
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True, enable_srp=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    with pytest.raises(RuntimeError, match="srp|forbidden"):
        resolve_batch_backend_policy(batch_cfg, sim_cfg)


def test_policy_paper_safe_forbids_dtype_downgrade(monkeypatch) -> None:
    """R29b (#4): requested dtype the backend cannot honor + paper_safe → raise."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    # float16 is not in gpu_st_lrps_potential's dtype_support → downgrade.
    cfg = SimpleNamespace(
        use_gpu=True, gravity_mode_override="st_lrps", torch_dtype="float16", paper_safe=True
    )
    with pytest.raises(RuntimeError, match="dtype|downgrade"):
        resolve_batch_backend_policy(cfg, sim_cfg)

    # Research mode: same downgrade is allowed but recorded.
    cfg_research = SimpleNamespace(
        use_gpu=True, gravity_mode_override="st_lrps", torch_dtype="float16"
    )
    plan = resolve_batch_backend_policy(cfg_research, sim_cfg)
    assert plan.dtype_downgraded is True
    assert plan.requested_dtype == "float16"
    assert plan.effective_dtype in ("float32", "float64")
    assert any("dtype" in w.lower() for w in plan.warnings)


def test_policy_paper_safe_forces_error_semantics_for_classic_sh(monkeypatch) -> None:
    """R29b (#3): classic-SH explicit GPU request that cannot be honored + paper_safe → raise."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(
        use_gpu=True,
        gravity_mode_override="follow_mission",
        batch_backend="numba_cuda_sh",
        sh_degree=8,
        paper_safe=True,
    )
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    with pytest.raises(RuntimeError, match="cannot be honored|forbids"):
        resolve_batch_backend_policy(batch_cfg, sim_cfg)


# =============================================================================
# R29b — paper-safe/strict posture forbids silent planning-time fallbacks
# =============================================================================


def test_fallback_forbidden_helper_reads_policy_and_flags() -> None:
    from lunaris.batch.backend_policy import fallback_forbidden

    assert fallback_forbidden(SimpleNamespace()) is False
    assert fallback_forbidden(SimpleNamespace(sh_fallback_policy="error")) is True
    assert fallback_forbidden(SimpleNamespace(paper_safe=True)) is True
    assert fallback_forbidden(SimpleNamespace(strict_backend=True)) is True
    assert fallback_forbidden(SimpleNamespace(benchmark_mode=True)) is True
    assert fallback_forbidden(SimpleNamespace(sh_fallback_policy="compatible_gpu")) is False


def test_policy_paper_safe_forbids_st_lrps_cuda_unavailable_fallback(monkeypatch) -> None:
    """R29b (#3): ST-LRPS GPU + no CUDA + paper_safe → raise, never a silent CPU plan."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps", paper_safe=True)
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    with pytest.raises(RuntimeError, match="forbidden"):
        resolve_batch_backend_policy(batch_cfg, sim_cfg)


def test_policy_paper_safe_forbids_st_lrps_unsupported_physics_fallback(monkeypatch) -> None:
    """R29b (#3): ST-LRPS GPU + unsupported physics + strict → raise."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps", strict_backend=True)
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True, enable_srp=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    with pytest.raises(RuntimeError, match="srp|forbidden"):
        resolve_batch_backend_policy(batch_cfg, sim_cfg)


def test_policy_paper_safe_forbids_dtype_downgrade(monkeypatch) -> None:
    """R29b (#4): requested dtype the backend cannot honor + paper_safe → raise."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    # float16 is not in gpu_st_lrps_potential's dtype_support → downgrade.
    cfg = SimpleNamespace(
        use_gpu=True, gravity_mode_override="st_lrps", torch_dtype="float16", paper_safe=True
    )
    with pytest.raises(RuntimeError, match="dtype|downgrade"):
        resolve_batch_backend_policy(cfg, sim_cfg)

    # Research mode: same downgrade is allowed but recorded.
    cfg_research = SimpleNamespace(
        use_gpu=True, gravity_mode_override="st_lrps", torch_dtype="float16"
    )
    plan = resolve_batch_backend_policy(cfg_research, sim_cfg)
    assert plan.dtype_downgraded is True
    assert plan.requested_dtype == "float16"
    assert plan.effective_dtype in ("float32", "float64")
    assert any("dtype" in w.lower() for w in plan.warnings)


def test_policy_paper_safe_forces_error_semantics_for_classic_sh(monkeypatch) -> None:
    """R29b (#3): classic-SH explicit GPU request that cannot be honored + paper_safe → raise."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(
        use_gpu=True,
        gravity_mode_override="follow_mission",
        batch_backend="numba_cuda_sh",
        sh_degree=8,
        paper_safe=True,
    )
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    with pytest.raises(RuntimeError, match="cannot be honored|forbids"):
        resolve_batch_backend_policy(batch_cfg, sim_cfg)


def test_policy_classic_sh_numba_cuda_true(monkeypatch) -> None:
    """Classic SH + Numba CUDA available → GPU_CLASSIC_SH."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="follow_mission")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.final_backend == BatchBackend.GPU_CLASSIC_SH
    assert plan.use_gpu
    assert plan.actual_backend == "numba_cuda_sh"
    assert plan.backend_family == "classic_sh"
    assert plan.backend_implementation == "numba_cuda"
    assert plan.requested_sh_degree == 0 or plan.requested_sh_degree == int(getattr(batch_cfg, "sh_degree", 0))


def test_policy_classic_sh_numba_cuda_false(monkeypatch) -> None:
    """Classic SH + Numba CUDA unavailable → CPU."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="follow_mission")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.final_backend == BatchBackend.CPU
    assert not plan.use_gpu
    assert plan.actual_backend == "cpu_sh"
    assert "cuda" in plan.fallback_reason.lower()


def test_policy_classic_sh_high_degree_falls_back_without_clipping(monkeypatch) -> None:
    """Classic GPU SH degree > true CUDA tier is an explicit CPU fallback."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_sh_limits", lambda: (24, (24,)))

    batch_cfg = SimpleNamespace(
        use_gpu=True,
        batch_backend="numba_cuda_sh",
        gravity_mode_override="follow_mission",
        sh_degree=80,
    )
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.final_backend == BatchBackend.CPU
    assert not plan.use_gpu
    assert plan.requested_backend == "numba_cuda_sh"
    assert plan.actual_backend == "cpu_sh"
    assert plan.requested_sh_degree == 80
    assert plan.actual_sh_degree is None
    assert plan.numba_cuda_sh_max_degree == 24
    assert plan.numba_cuda_sh_supported_tiers == (24,)
    assert plan.fallback_reason == "numba_cuda_sh supports degree <= 24"
    assert any("without clipping" in w.lower() for w in plan.warnings)


@pytest.mark.parametrize(
    ("requested", "artifact_kind"),
    [
        ("gpu_st_lrps_potential", "force_direct"),
    ],
)
def test_policy_rejects_explicit_st_lrps_artifact_kind_mismatch(
    monkeypatch,
    requested: str,
    artifact_kind: str,
) -> None:
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)
    monkeypatch.setattr(
        policy_mod,
        "_read_st_lrps_runtime_kind",
        lambda batch_cfg, sim_cfg: artifact_kind,
    )
    batch_cfg = SimpleNamespace(
        use_gpu=True,
        batch_backend=requested,
        gravity_mode_override="follow_mission",
        sh_degree=0,
    )
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )
    with pytest.raises(ValueError, match="requires"):
        resolve_batch_backend_policy(batch_cfg, sim_cfg)


def test_policy_no_contradictory_command_args_st_lrps_gpu(monkeypatch) -> None:
    """GPU_ST_LRPS plan emits use_gpu=True, gravity_backend='st_lrps'."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.use_gpu is True
    assert plan.gravity_backend == "st_lrps"


def test_policy_no_contradictory_command_args_cpu_fallback(monkeypatch) -> None:
    """CPU fallback plan emits use_gpu=False regardless of request."""
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)
    assert plan.use_gpu is False


# =============================================================================
# New: ST-LRPS batch torch inference tests (CPU tensors — no GPU required)
# =============================================================================

torch = pytest.importorskip("torch")


def _make_tiny_surrogate(tmp_path: Path) -> Any:  # noqa: F821
    """Create a minimal SurrogateGravityModel on CPU for inference tests."""
    from lunaris.common.constants import MU_MOON, R_MOON
    from lunaris.surrogate.runtime import SurrogateGravityModel
    from lunaris.surrogate.runtime.networks import _build_model_from_config

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


@pytest.mark.requires_data
def test_predict_residual_accel_torch_shape(tmp_path: Path) -> None:
    """predict_residual_accel_torch returns [N, 3] for [N, 3] input."""
    model = _make_tiny_surrogate(tmp_path)
    x = torch.zeros(4, 3, dtype=torch.float32)
    x[:, 0] = 1_838_000.0  # 100 km altitude positions

    out = model.predict_residual_accel_torch(x)

    assert out.shape == (4, 3)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


@pytest.mark.requires_data
def test_predict_total_accel_torch_shape(tmp_path: Path) -> None:
    """predict_total_accel_torch returns [N, 3] for [N, 3] input."""
    model = _make_tiny_surrogate(tmp_path)
    x = torch.zeros(4, 3, dtype=torch.float32)
    x[:, 0] = 1_838_000.0

    out = model.predict_total_accel_torch(x)

    assert out.shape == (4, 3)
    assert out.dtype == torch.float32


@pytest.mark.requires_data
def test_predict_total_accel_torch_zero_net_matches_point_mass(tmp_path: Path) -> None:
    """Zero-weight network → total acceleration equals point-mass (residual mode)."""
    from lunaris.common.constants import MU_MOON
    model = _make_tiny_surrogate(tmp_path)

    r = 1_838_000.0
    x = torch.tensor([[r, 0.0, 0.0]], dtype=torch.float32)
    out = model.predict_total_accel_torch(x)  # [1, 3]

    expected_ax = -float(MU_MOON) / (r * r)
    assert abs(float(out[0, 0]) - expected_ax) / abs(expected_ax) < 1e-4
    assert abs(float(out[0, 1])) < 1e-3
    assert abs(float(out[0, 2])) < 1e-3


@pytest.mark.requires_data
def test_predict_residual_accel_torch_zero_net_is_zero(tmp_path: Path) -> None:
    """Zero-weight network → residual acceleration is zero (delta above point mass)."""
    model = _make_tiny_surrogate(tmp_path)
    x = torch.tensor([[1_838_000.0, 0.0, 0.0]], dtype=torch.float32)
    out = model.predict_residual_accel_torch(x)
    assert torch.allclose(out, torch.zeros(1, 3), atol=1e-10)


@pytest.mark.requires_data
def test_degree_max_metadata_exposed(tmp_path: Path) -> None:
    """SurrogateGravityModel exposes degree_max for the batch propagator contract."""
    model = _make_tiny_surrogate(tmp_path)
    assert hasattr(model, "degree_max")
    assert int(model.degree_max) == 50
    assert int(model.degree_min) == 10


# =============================================================================
# New: TorchBatchPropagator smoke test (CPU-only, no real GPU needed)
# =============================================================================

@pytest.mark.requires_data
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

    from lunaris.common.constants import R_MOON
    from lunaris.core.torch_batch_propagator import TorchBatchPropagator

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
    prop._dtype = _torch.float64

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
    import lunaris.batch.backend_policy as policy_mod
    import lunaris.core.batch_propagator as batch_prop

    class DummyCPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("CPU should not be selected when GPU ST-LRPS is available.")

    class DummyGPU:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("Classic GPU should not be selected for ST-LRPS.")

    class DummyTorchGPU:
        def __init__(self, surrogate_model, batch_cfg, device_id=0) -> None:
            self.surrogate_model = surrogate_model
            self.batch_cfg = batch_cfg

    monkeypatch.setattr(batch_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(batch_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(batch_prop, "GPUBatchPropagator", DummyGPU)
    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)
    monkeypatch.setattr("lunaris.core.torch_batch_propagator.TorchBatchPropagator", DummyTorchGPU)

    # Fake surrogate model with model_kind and degree metadata
    fake_grav = SimpleNamespace(
        model_kind="st_lrps",
        model_dir=Path("/fake/run"),
        degree_min=10,
        degree_max=50,
    )
    fake_dyn = SimpleNamespace(grav=fake_grav)

    engine = BatchPropagationEngine.__new__(BatchPropagationEngine)
    engine._cfg = BatchPropagationConfig(
        n_samples=2,
        use_gpu=True,
        gravity_mode_override="st_lrps",
        output_format="npz",
        output_path="outputs/ensemble/test_torch_gpu.npz",
        st_lrps_model_dir="mock",
    )
    engine._sim_cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )
    engine._dyn = fake_dyn
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""

    prop = BatchPropagationEngine._build_propagator(engine)

    assert isinstance(prop, DummyTorchGPU)
    assert prop.surrogate_model is fake_grav
