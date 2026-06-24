"""
Live Monte Carlo backend dispatch for torch_cuda_sh (task §15 / §10 / §16E).

Two layers, both CPU-only (CUDA availability is monkeypatched):

1. ``resolve_mc_backend_policy`` integration — the *real* policy resolver (not
   the standalone ``select_classic_sh_backend``) must route classic-SH requests
   through the shared decision function and expose the right ``final_backend`` /
   ``actual_backend`` / degree provenance.

2. Engine dispatch — when the plan is ``GPU_TORCH_SH`` the engine must construct
   the Torch SH propagator and must NOT construct the Numba or CPU propagators.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.common.montecarlo_defs import MonteCarloConfig
from lunaris.common.type_defs import PerturbationFlags
from lunaris.core.mc_backend_policy import MCBackend, resolve_mc_backend_policy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_availability(monkeypatch, *, torch_avail: bool, numba_avail: bool) -> None:
    import lunaris.core.mc_backend_policy as policy_mod

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: torch_avail)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: numba_avail)
    monkeypatch.setattr(policy_mod, "_gpu_sh_limits", lambda: (24, (24,)))


def _mc(**kw):
    base = dict(
        use_gpu=True,
        mc_backend="auto",
        gravity_mode_override="follow_mission",
        gpu_sh_degree=0,
        gpu_sh_fallback_policy="compatible_gpu",
        torch_dtype="float64",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _sim(flags=None):
    return SimpleNamespace(
        flags=flags if flags is not None else PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )


def test_mc_body_vector_policy_includes_relativity_external_terms() -> None:
    from lunaris.core.monte_carlo_engine import _need_body_vectors

    cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=False, enable_relativity_1pn=True),
        thermal=None,
    )

    assert _need_body_vectors(cfg) is True


# ===========================================================================
# 1. Policy integration (§15)
# ===========================================================================

def test_explicit_torch_routing_low_degree(monkeypatch) -> None:
    """backend=torch_cuda_sh, degree=20, torch+numba available, gravity-only."""
    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    plan = resolve_mc_backend_policy(_mc(mc_backend="torch_cuda_sh", gpu_sh_degree=20), _sim())
    assert plan.final_backend == MCBackend.GPU_TORCH_SH
    assert plan.actual_backend == "torch_cuda_sh"
    assert plan.actual_sh_degree == 20
    assert plan.fallback_applied is False
    assert plan.backend_implementation == "torch"


def test_explicit_torch_low_degree_never_routes_to_numba(monkeypatch) -> None:
    """An explicit torch request must never be silently served by numba_cuda_sh."""
    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    plan = resolve_mc_backend_policy(_mc(mc_backend="torch_cuda_sh", gpu_sh_degree=20), _sim())
    assert plan.actual_backend != "numba_cuda_sh"
    assert plan.final_backend != MCBackend.GPU_CLASSIC_SH


def test_high_degree_auto_selects_torch(monkeypatch) -> None:
    """backend=auto, degree=100, torch+numba available → torch_cuda_sh, no fallback."""
    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    plan = resolve_mc_backend_policy(_mc(mc_backend="auto", gpu_sh_degree=100), _sim())
    assert plan.final_backend == MCBackend.GPU_TORCH_SH
    assert plan.actual_backend == "torch_cuda_sh"
    assert plan.requested_sh_degree == 100
    assert plan.actual_sh_degree == 100
    assert plan.fallback_applied is False


def test_high_degree_auto_torch_unavailable_falls_back_to_cpu(monkeypatch) -> None:
    """backend=auto, degree=100, torch unavailable → cpu_sh with non-empty reason."""
    _patch_availability(monkeypatch, torch_avail=False, numba_avail=True)
    plan = resolve_mc_backend_policy(_mc(mc_backend="auto", gpu_sh_degree=100), _sim())
    assert plan.final_backend == MCBackend.CPU
    assert plan.actual_backend == "cpu_sh"
    assert plan.fallback_applied is True
    assert plan.fallback_reason  # must not be empty
    assert plan.actual_sh_degree is None


def test_physics_mismatch_high_degree_falls_back_to_cpu(monkeypatch) -> None:
    """backend=auto, degree=100, torch available, SRP enabled → cpu_sh (gravity-only torch)."""
    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    plan = resolve_mc_backend_policy(
        _mc(mc_backend="auto", gpu_sh_degree=100),
        _sim(PerturbationFlags(enable_sh=True, enable_srp=True)),
    )
    assert plan.final_backend == MCBackend.CPU
    assert plan.actual_backend == "cpu_sh"
    assert "srp" in plan.fallback_reason.lower()
    assert any("srp" in w.lower() for w in plan.warnings)


def test_explicit_numba_high_degree_compatible_gpu_uses_torch(monkeypatch) -> None:
    """backend=numba_cuda_sh, degree=100, compatible_gpu, torch available → torch_cuda_sh."""
    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    plan = resolve_mc_backend_policy(
        _mc(mc_backend="numba_cuda_sh", gpu_sh_degree=100, gpu_sh_fallback_policy="compatible_gpu"),
        _sim(),
    )
    assert plan.final_backend == MCBackend.GPU_TORCH_SH
    assert plan.actual_backend == "torch_cuda_sh"
    assert plan.actual_sh_degree == 100
    assert plan.fallback_applied is True


def test_explicit_numba_high_degree_error_policy_raises(monkeypatch) -> None:
    """backend=numba_cuda_sh, degree=100, fallback_policy=error → hard error, no substitution."""
    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    with pytest.raises(RuntimeError) as exc:
        resolve_mc_backend_policy(
            _mc(mc_backend="numba_cuda_sh", gpu_sh_degree=100, gpu_sh_fallback_policy="error"),
            _sim(),
        )
    assert "degree" in str(exc.value).lower()


@pytest.mark.parametrize("torch_avail", [True, False])
def test_requested_degree_100_is_never_reported_as_24(monkeypatch, torch_avail: bool) -> None:
    """No routing path may report the requested degree as the Numba limit (24)."""
    _patch_availability(monkeypatch, torch_avail=torch_avail, numba_avail=True)
    plan = resolve_mc_backend_policy(_mc(mc_backend="auto", gpu_sh_degree=100), _sim())
    assert plan.requested_sh_degree == 100
    assert plan.actual_sh_degree in (100, None)
    assert plan.actual_sh_degree != 24


def test_low_degree_auto_still_prefers_numba(monkeypatch) -> None:
    """Regression: the numba_cuda_sh degree<=24 screening path is unchanged."""
    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    plan = resolve_mc_backend_policy(_mc(mc_backend="auto", gpu_sh_degree=20), _sim())
    assert plan.final_backend == MCBackend.GPU_CLASSIC_SH
    assert plan.actual_backend == "numba_cuda_sh"


# ---------------------------------------------------------------------------
# 1b. torch_cpu_sh selection matrix (§2 / §7 backend policy)
# ---------------------------------------------------------------------------

def test_explicit_torch_cpu_sh_uses_torch_on_cpu(monkeypatch) -> None:
    """backend=torch_cpu_sh → TORCH_CPU_SH plan on CPU, fixed-step RK4, no GPU."""
    pytest.importorskip("torch")
    _patch_availability(monkeypatch, torch_avail=False, numba_avail=False)
    plan = resolve_mc_backend_policy(_mc(mc_backend="torch_cpu_sh", gpu_sh_degree=50), _sim())
    assert plan.final_backend == MCBackend.TORCH_CPU_SH
    assert plan.actual_backend == "torch_cpu_sh"
    assert plan.use_gpu is False
    assert plan.actual_device == "cpu"
    assert plan.integrator == "fixed-step RK4"
    assert plan.actual_sh_degree == 50


def test_cuda_unavailable_cpu_policy_falls_back_to_torch_cpu(monkeypatch) -> None:
    """torch_cuda_sh + CUDA unavailable + fallback_policy=cpu → torch_cpu_sh, recorded."""
    pytest.importorskip("torch")
    _patch_availability(monkeypatch, torch_avail=False, numba_avail=False)
    plan = resolve_mc_backend_policy(
        _mc(mc_backend="torch_cuda_sh", gpu_sh_degree=80, gpu_sh_fallback_policy="cpu"),
        _sim(),
    )
    assert plan.final_backend == MCBackend.TORCH_CPU_SH
    assert plan.actual_backend == "torch_cpu_sh"
    assert plan.fallback_applied is True
    assert plan.fallback_reason  # must record why it fell back


def test_cuda_unavailable_error_policy_raises(monkeypatch) -> None:
    """torch_cuda_sh + CUDA unavailable + fallback_policy=error → hard error."""
    _patch_availability(monkeypatch, torch_avail=False, numba_avail=False)
    with pytest.raises(RuntimeError):
        resolve_mc_backend_policy(
            _mc(mc_backend="torch_cuda_sh", gpu_sh_degree=80, gpu_sh_fallback_policy="error"),
            _sim(),
        )


def test_cuda_unavailable_compatible_gpu_does_not_fall_back_to_cpu(monkeypatch) -> None:
    """torch_cuda_sh + CUDA unavailable + compatible_gpu → raise (NO silent CPU fallback)."""
    _patch_availability(monkeypatch, torch_avail=False, numba_avail=False)
    with pytest.raises(RuntimeError):
        resolve_mc_backend_policy(
            _mc(
                mc_backend="torch_cuda_sh",
                gpu_sh_degree=80,
                gpu_sh_fallback_policy="compatible_gpu",
            ),
            _sim(),
        )


# ===========================================================================
# 2. Engine dispatch (§10 / §16E)
# ===========================================================================

def _torch_plan_engine(monkeypatch, tmp_path, *, mc_backend="torch_cuda_sh", degree=100):
    """Build an engine stub whose policy resolves to GPU_TORCH_SH."""
    import lunaris.core.mc_propagator as mc_prop
    import lunaris.core.torch_sh_propagator as torch_sh_mod
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    constructed: dict[str, object] = {}

    class DummyCPU:
        def __init__(self, *a, **k) -> None:
            raise AssertionError("CPU propagator must not be constructed for the torch SH path.")

    class DummyNumbaGPU:
        def __init__(self, *a, **k) -> None:
            raise AssertionError("Numba GPU propagator must not be constructed for the torch SH path.")

    class DummyTorchSH:
        def __init__(self, dyn, mc_cfg, flags, **k) -> None:
            constructed["torch_sh"] = self
            self.dyn = dyn
            self.mc_cfg = mc_cfg

    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    monkeypatch.setattr(mc_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mc_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(mc_prop, "GPUBatchPropagator", DummyNumbaGPU)
    monkeypatch.setattr(torch_sh_mod, "TorchSHBatchPropagator", DummyTorchSH)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2,
        use_gpu=True,
        mc_backend=mc_backend,
        gpu_sh_degree=degree,
        output_format="npz",
        output_path=str(tmp_path / "mc_torch_sh.npz"),
    )
    engine._sim_cfg = _sim()
    engine._dyn = SimpleNamespace(grav=SimpleNamespace(degree_max=degree), ephem=None)
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""
    engine._backend_plan = None
    return engine, constructed, DummyTorchSH


def test_engine_dispatches_torch_sh_and_not_numba_or_cpu(monkeypatch, tmp_path: Path) -> None:
    engine, constructed, DummyTorchSH = _torch_plan_engine(monkeypatch, tmp_path)
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    prop = MonteCarloEngine._build_propagator(engine)

    assert isinstance(prop, DummyTorchSH)
    assert "torch_sh" in constructed
    # The plan recorded on the engine must label the run as torch, not numba/cpu.
    assert engine._backend_plan.final_backend == MCBackend.GPU_TORCH_SH
    assert engine._backend_plan.actual_backend == "torch_cuda_sh"


def test_engine_explicit_numba_high_degree_dispatches_torch(monkeypatch, tmp_path: Path) -> None:
    """numba_cuda_sh + degree 100 (compatible_gpu default) dispatches the torch propagator."""
    engine, constructed, DummyTorchSH = _torch_plan_engine(
        monkeypatch, tmp_path, mc_backend="numba_cuda_sh", degree=100
    )
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    prop = MonteCarloEngine._build_propagator(engine)
    assert isinstance(prop, DummyTorchSH)
    assert engine._backend_plan.actual_backend == "torch_cuda_sh"


def test_engine_dispatches_torch_cpu_sh_with_cpu_device(monkeypatch, tmp_path: Path) -> None:
    """An explicit torch_cpu_sh request builds the Torch propagator with device='cpu'
    and must NOT construct the SciPy CPU or Numba GPU propagators."""
    pytest.importorskip("torch")
    import lunaris.core.mc_propagator as mc_prop
    import lunaris.core.torch_sh_propagator as torch_sh_mod
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    captured: dict[str, object] = {}

    class DummyCPU:
        def __init__(self, *a, **k) -> None:
            raise AssertionError("SciPy CPU propagator must not be built for torch_cpu_sh.")

    class DummyNumbaGPU:
        def __init__(self, *a, **k) -> None:
            raise AssertionError("Numba GPU propagator must not be built for torch_cpu_sh.")

    class DummyTorchSH:
        def __init__(self, dyn, mc_cfg, flags, **k) -> None:
            captured["device"] = k.get("device")

    _patch_availability(monkeypatch, torch_avail=False, numba_avail=False)
    monkeypatch.setattr(mc_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(mc_prop, "GPUBatchPropagator", DummyNumbaGPU)
    monkeypatch.setattr(torch_sh_mod, "TorchSHBatchPropagator", DummyTorchSH)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2,
        use_gpu=True,
        mc_backend="torch_cpu_sh",
        gpu_sh_degree=50,
        output_format="npz",
        output_path=str(tmp_path / "mc_torch_cpu.npz"),
    )
    engine._sim_cfg = _sim()
    engine._dyn = SimpleNamespace(grav=SimpleNamespace(degree_max=50), ephem=None)
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""
    engine._backend_plan = None

    prop = MonteCarloEngine._build_propagator(engine)
    assert isinstance(prop, DummyTorchSH)
    assert captured["device"] == "cpu"
    assert engine._backend_plan.final_backend == MCBackend.TORCH_CPU_SH
    assert engine._backend_plan.actual_backend == "torch_cpu_sh"


def test_engine_torch_preflight_error_is_not_silently_downgraded(monkeypatch, tmp_path: Path) -> None:
    """A TorchSHPreflightError (e.g. degree above the model) must propagate, not fall to CPU."""
    import lunaris.core.mc_propagator as mc_prop
    import lunaris.core.torch_sh_propagator as torch_sh_mod
    from lunaris.core.monte_carlo_engine import MonteCarloEngine
    from lunaris.core.torch_sh_propagator import TorchSHPreflightError

    class DummyCPU:
        def __init__(self, *a, **k) -> None:
            raise AssertionError("Preflight errors must not be downgraded to CPU.")

    class RaisingTorchSH:
        def __init__(self, *a, **k) -> None:
            raise TorchSHPreflightError("Requested SH degree 100, but loaded gravity model supports degree 50.")

    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    monkeypatch.setattr(mc_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mc_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(torch_sh_mod, "TorchSHBatchPropagator", RaisingTorchSH)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2, use_gpu=True, mc_backend="torch_cuda_sh", gpu_sh_degree=100,
        output_format="npz", output_path=str(tmp_path / "mc_torch_err.npz"),
    )
    engine._sim_cfg = _sim()
    engine._dyn = SimpleNamespace(grav=SimpleNamespace(degree_max=50), ephem=None)
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""
    engine._backend_plan = None

    with pytest.raises(TorchSHPreflightError):
        MonteCarloEngine._build_propagator(engine)


def test_engine_torch_cuda_sh_passes_explicit_cuda_device(monkeypatch, tmp_path: Path) -> None:
    """The GPU torch_cuda_sh branch must construct the propagator with an explicit
    CUDA device (cuda:<id>) so it cannot silently degrade to CPU inside the
    propagator's own device-resolution fallback (reviewer §3)."""
    import lunaris.core.mc_propagator as mc_prop
    import lunaris.core.torch_sh_propagator as torch_sh_mod
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    captured: dict[str, object] = {}

    class DummyTorchSH:
        def __init__(self, dyn, mc_cfg, flags, **k) -> None:
            captured["device"] = k.get("device")

    _patch_availability(monkeypatch, torch_avail=True, numba_avail=True)
    monkeypatch.setattr(mc_prop, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch_sh_mod, "TorchSHBatchPropagator", DummyTorchSH)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2, use_gpu=True, mc_backend="torch_cuda_sh", gpu_sh_degree=100,
        gpu_device_id=2, output_format="npz", output_path=str(tmp_path / "mc_cuda_dev.npz"),
    )
    engine._sim_cfg = _sim()
    engine._dyn = SimpleNamespace(grav=SimpleNamespace(degree_max=100), ephem=None)
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""
    engine._backend_plan = None

    prop = MonteCarloEngine._build_propagator(engine)
    assert isinstance(prop, DummyTorchSH)
    assert captured["device"] == "cuda:2"


def test_engine_st_lrps_build_failure_downgrades_plan_to_cpu(monkeypatch, tmp_path: Path) -> None:
    """If the GPU ST-LRPS propagator fails to construct, the engine must fall back
    to CPU AND rewrite the plan provenance — a CPU run must never keep a GPU label
    (reviewer §1a). Mirrors the numba/torch branches which already downgrade."""
    import lunaris.core.mc_propagator as mc_prop
    import lunaris.core.torch_batch_propagator as tbp_mod
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    built: dict[str, object] = {}

    class DummyCPU:
        def __init__(self, *a, **k) -> None:
            built["cpu"] = True

    class RaisingTorchBatch:
        def __init__(self, *a, **k) -> None:
            raise RuntimeError("CUDA OOM while building ST-LRPS propagator")

    _patch_availability(monkeypatch, torch_avail=True, numba_avail=False)
    monkeypatch.setattr(mc_prop, "CPUBatchPropagator", DummyCPU)
    monkeypatch.setattr(tbp_mod, "TorchBatchPropagator", RaisingTorchBatch)

    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._mc = MonteCarloConfig(
        n_samples=2, use_gpu=True, mc_backend="gpu_st_lrps_potential", gpu_sh_degree=0,
        st_lrps_model_dir=str(tmp_path / "surrogate_run"),
        output_format="npz", output_path=str(tmp_path / "mc_stlrps_fail.npz"),
    )
    engine._sim_cfg = _sim()
    engine._dyn = SimpleNamespace(
        grav=SimpleNamespace(model_kind="st_lrps", model_dir="x", degree_min=2, degree_max=50),
        ephem=None,
    )
    engine._surface_provider = None
    engine._topo_grid = None
    engine._backend_note = ""
    engine._backend_plan = None

    prop = MonteCarloEngine._build_propagator(engine)
    assert isinstance(prop, DummyCPU)
    assert built.get("cpu") is True
    plan = engine._backend_plan
    assert plan.final_backend == MCBackend.CPU
    assert plan.actual_backend == "cpu_st_lrps"
    assert plan.actual_device == "cpu"
    assert plan.fallback_applied is True
    assert plan.fallback_reason  # records why the GPU build was abandoned


def _patch_dynamics_gravity_capture(monkeypatch) -> dict:
    """Patch the gravity/dynamics stack so _build_dynamics runs without real data
    and capture the requested coefficient degree passed to GravityModel.from_file."""
    import lunaris.core.dynamics as dyn_mod
    import lunaris.core.monte_carlo_engine as mce
    import lunaris.physics.spherical_harmonics as sh_mod

    captured: dict = {}

    class _FakeGrav:
        @staticmethod
        def from_file(*, path, requested_degree):
            captured["degree"] = requested_degree
            return object()

    monkeypatch.setattr(sh_mod, "GravityModel", _FakeGrav)
    monkeypatch.setattr(dyn_mod, "DynamicsEngine", lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(mce, "_need_ephemeris", lambda cfg, topo_requested: False)
    monkeypatch.setattr(mce, "_surface_topography_requested", lambda sp, tg: False)
    return captured


def _engine_for_dynamics(mc_backend: str, *, use_gpu: bool, gpu_sh_degree: int, degree: int = 25):
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    cfg = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(
            uses_st_lrps=False, degree=degree, file_path="grav.tab",
            st_lrps_model_dir="", adaptive=None,
        ),
        spacecraft=SimpleNamespace(), earth_j2=None, thermal=None, solid_tides=None,
    )
    engine = MonteCarloEngine.__new__(MonteCarloEngine)
    engine._sim_cfg = cfg
    engine._mc = MonteCarloConfig(
        n_samples=2, use_gpu=use_gpu, mc_backend=mc_backend, gpu_sh_degree=gpu_sh_degree,
        output_format="npz", output_path="x.npz",
    )
    engine._surface_provider = None
    engine._topo_grid = None
    return engine


def test_build_dynamics_loads_gpu_sh_degree_for_gpu_classic_path(monkeypatch) -> None:
    """A classic-SH GPU batch request (torch_cuda_sh) loads coefficients to at
    least mc.gpu_sh_degree, not just the mission degree, so the propagator
    preflight does not reject a high gpu_sh_degree (reviewer #9)."""
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    captured = _patch_dynamics_gravity_capture(monkeypatch)
    engine = _engine_for_dynamics("torch_cuda_sh", use_gpu=True, gpu_sh_degree=100, degree=25)
    MonteCarloEngine._build_dynamics(engine)
    assert captured["degree"] == 100  # max(25, 100), not 25


def test_build_dynamics_keeps_mission_degree_for_cpu_path(monkeypatch) -> None:
    """A pure-CPU request must NOT inflate the loaded degree to gpu_sh_degree; the
    mission degree is preserved so CPU physics is unchanged (reviewer #9)."""
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    captured = _patch_dynamics_gravity_capture(monkeypatch)
    engine = _engine_for_dynamics("cpu_sh", use_gpu=False, gpu_sh_degree=100, degree=25)
    MonteCarloEngine._build_dynamics(engine)
    assert captured["degree"] == 25


def test_backend_display_name_distinguishes_torch_cpu_from_cuda() -> None:
    """The run label is derived from plan.actual_backend, not the propagator class
    name (reviewer §2). torch_cpu_sh and torch_cuda_sh share one class, so the
    mapping must keep CPU labeled CPU."""
    from lunaris.core.monte_carlo_engine import _BACKEND_DISPLAY_NAMES

    assert _BACKEND_DISPLAY_NAMES["torch_cpu_sh"] == "CPU-TORCH-SH"
    assert _BACKEND_DISPLAY_NAMES["torch_cuda_sh"] == "GPU-TORCH-SH"
    assert _BACKEND_DISPLAY_NAMES["cpu_st_lrps"] == "CPU-ST-LRPS"
    assert _BACKEND_DISPLAY_NAMES["torch_cpu_sh"] != _BACKEND_DISPLAY_NAMES["torch_cuda_sh"]
