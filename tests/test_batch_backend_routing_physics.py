"""Batch backend routing must not change the physics a run flies.

(a) A GPU-path request that ends on the CPU (no CUDA, or a failed GPU build)
    flies the degree a single run of the same ``SimConfig`` flies, and the
    archive records the degree it actually flew.
"""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.common.batch_defs import (
    BatchPropagationConfig,
    SpacecraftUncertainty,
    StateUncertainty,
)
from lunaris.common.type_defs import PerturbationFlags

MU = 4.9028e12


def _patch_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import lunaris.batch.backend_policy as policy_mod

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)
    monkeypatch.setattr(policy_mod, "_numba_cuda_sh_limits", lambda: (24, (24,)))


# ---------------------------------------------------------------------------
# (a) a GPU request that ends on the CPU flies the single-run degree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch_backend", "widened"),
    [("auto", True), ("numba_cuda_sh", True), ("cpu_sh", False)],
)
def test_cpu_path_rebuilds_a_widened_template_at_the_mission_degree(
    monkeypatch: pytest.MonkeyPatch, batch_backend: str, widened: bool
) -> None:
    """Asset-free routing check: the template is loaded at max(20, 24) for a
    GPU-path request; once the policy plans the CPU, the CPU propagator gets a
    template at the mission degree 20, reusing the ephemeris built once."""
    import lunaris.batch.engine as engine_mod
    import lunaris.core.dynamics as dynamics_mod
    from lunaris.core.batch_propagator import CPUBatchPropagator
    from lunaris.physics.spherical_harmonics import GravityModel

    _patch_no_cuda(monkeypatch)
    loaded: list[int | None] = []
    ephem_builds: list[object] = []

    def _from_file(*, path: str, requested_degree: int | None = None) -> SimpleNamespace:
        loaded.append(requested_degree)
        return SimpleNamespace(max_degree=requested_degree, model_kind="classic_sh")

    class _Dyn:
        def __init__(self, **kwargs: object) -> None:
            self.grav = kwargs["gravity_model"]
            self.ephem = kwargs["ephem_manager"]

    def _ephem(_cfg: object) -> object:
        ephem_builds.append(object())
        return ephem_builds[-1]

    monkeypatch.setattr(GravityModel, "from_file", _from_file)
    monkeypatch.setattr(dynamics_mod, "DynamicsEngine", _Dyn)
    monkeypatch.setattr(engine_mod, "_need_ephemeris", lambda *_a, **_k: True)
    monkeypatch.setattr(engine_mod, "_build_ephemeris_manager", _ephem)

    sim = SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(
            uses_st_lrps=False, degree=20, file_path="unused.tab", adaptive=None,
            st_lrps_model_dir=None,
        ),
        spacecraft=None,
    )
    bcfg = BatchPropagationConfig(
        n_samples=2, use_gpu=True, batch_backend=batch_backend, sh_degree=24,
    )
    engine = engine_mod.BatchPropagationEngine(sim, bcfg)
    assert engine._dyn.grav.max_degree == (24 if widened else 20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the recorded CPU fallback
        prop = engine._build_propagator()

    assert isinstance(prop, CPUBatchPropagator)
    assert prop._dyn_template is engine._dyn
    assert engine._dyn.grav.max_degree == 20
    assert loaded == ([24, 20] if widened else [20])
    assert len(ephem_builds) == 1
    assert engine._dyn.ephem is ephem_builds[0]


@pytest.mark.requires_data
@pytest.mark.parametrize("batch_backend", ["auto", "numba_cuda_sh"])
def test_gpu_request_ending_on_cpu_matches_single_run_degree(
    monkeypatch: pytest.MonkeyPatch, tmp_path, batch_backend: str
) -> None:
    """Mission degree 20, ``sh_degree`` 24, CUDA unavailable: the batch members
    equal single runs of the same SimConfig (same code path, same inputs, same
    output grid, so the difference is exactly zero; 1e-9 m allows for nothing
    but a different summation order). Before the fix the CPU flew degree 24;
    the archive recorded ``actual_sh_degree`` = null."""
    from lunaris.batch.engine import BatchPropagationEngine
    from lunaris.cli.run import build_engine, build_ephemeris_if_needed, build_gravity_provider
    from lunaris.common.constants import R_MOON
    from lunaris.core.propagation.propagator import propagate

    try:
        from lunaris.core.config import load_default_config, replace_sim_config

        base = load_default_config()
    except Exception as exc:  # optional data/config assets
        pytest.skip(f"default config assets unavailable: {exc}")

    _patch_no_cuda(monkeypatch)
    degree, sh_degree = 20, 24
    r0 = float(R_MOON) + 100_000.0
    v0 = float(np.sqrt(MU / r0))
    cfg = replace_sim_config(
        base,
        initial_state=np.array([r0, 0.0, 0.0, 0.0, v0, 0.0]),
        gravity=dataclasses.replace(base.gravity, degree=degree),
        time=dataclasses.replace(base.time, duration_s=3600.0, output_dt_s=60.0),
    )
    bcfg = BatchPropagationConfig(
        n_samples=2, seed=11, output_format="npz", output_path=str(tmp_path / "b.npz"),
        state=StateUncertainty(sigma_r_m=500.0, sigma_v_m_s=0.5),
        spacecraft=SpacecraftUncertainty(sigma_mass_kg=100.0, sigma_area_m2=0.5, sigma_cr=0.2),
        use_gpu=True, batch_backend=batch_backend, sh_degree=sh_degree, dt_s=60.0,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # the recorded fallback warning
            res = BatchPropagationEngine(cfg, bcfg).run()
    except RuntimeError as exc:
        if "bootstrap" in str(exc).lower():
            pytest.skip(f"simulation assets unavailable: {exc}")
        raise

    diag = res.diagnostics
    assert diag["actual_batch_backend"] == "cpu_sh"
    assert diag["fallback_reason"]  # recorded, not silent
    assert diag["requested_sh_degree"] == sh_degree
    assert diag["actual_sh_degree"] == degree

    Y = np.asarray(res.Y)
    t = np.asarray(res.t)
    sc = np.asarray(res.sc_samples)
    for i in range(Y.shape[1]):
        scfg = replace_sim_config(
            cfg,
            spacecraft=dataclasses.replace(
                cfg.spacecraft, mass_kg=float(sc[i][0]), area_m2=float(sc[i][1]),
                cd=float(sc[i][2]), cr=float(sc[i][3]),
            ),
        )
        grav, _mu = build_gravity_provider(scfg)
        ephem = build_ephemeris_if_needed(scfg, topo_requested=False)
        engine = build_engine(scfg, gravity_core=grav, ephem_mgr=ephem, surface_provider=None)
        single = propagate(
            dynamics=engine, y0=np.asarray(Y[0, i], float), cfg=scfg.propagator,
            time_cfg=scfg.time,
        )
        assert np.array_equal(np.asarray(single.t), t)
        dr = np.max(np.linalg.norm(Y[:, i, :3] - np.asarray(single.y)[:, :3], axis=1))
        assert dr <= 1e-9, f"member {i}: batch vs single-run position differs by {dr:.3e} m"


# ---------------------------------------------------------------------------
# (b) the torch classic-SH backends must not silently drop physics
# ---------------------------------------------------------------------------


def _torch_cfg(backend: str, policy: str) -> BatchPropagationConfig:
    return BatchPropagationConfig(
        n_samples=2, use_gpu=True, batch_backend=backend, sh_degree=8,
        sh_fallback_policy=policy,
    )


def _sim(*, adaptive: bool = False, srp: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True, enable_srp=srp),
        gravity=SimpleNamespace(uses_st_lrps=False, adaptive=SimpleNamespace(enabled=adaptive)),
    )


def test_torch_cpu_sh_with_unsupported_physics_is_refused_or_recorded_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit torch_cpu_sh skipped the unsupported-physics check: the plan
    said torch_cpu_sh for a run the gravity-only evaluator cannot fly."""
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    pytest.importorskip("torch")
    _patch_no_cuda(monkeypatch)
    with pytest.raises(RuntimeError, match="not supported by torch_cpu_sh"):
        resolve_batch_backend_policy(_torch_cfg("torch_cpu_sh", "error"), _sim(srp=True))
    plan = resolve_batch_backend_policy(_torch_cfg("torch_cpu_sh", "compatible_gpu"), _sim(srp=True))
    assert plan.actual_backend == "cpu_sh"
    assert plan.requested_backend == "torch_cpu_sh"
    assert plan.fallback_applied is True
    assert "physics not supported" in plan.fallback_reason
    # A gravity-only run keeps the explicit torch_cpu_sh request.
    plan = resolve_batch_backend_policy(_torch_cfg("torch_cpu_sh", "error"), _sim())
    assert plan.actual_backend == "torch_cpu_sh"
    assert plan.fallback_applied is False


def test_torch_cuda_sh_policy_cpu_route_honours_unsupported_physics() -> None:
    from lunaris.batch.backend_policy import select_classic_sh_backend

    pytest.importorskip("torch")
    common = dict(
        requested_backend="torch_cuda_sh", requested_degree=8,
        numba_cuda_available=False, torch_cuda_available=False, fallback_policy="cpu",
    )
    ok = select_classic_sh_backend(torch_physics_ok=True, **common)
    assert (ok.backend, ok.fallback_applied) == ("torch_cpu_sh", True)
    blocked = select_classic_sh_backend(torch_physics_ok=False, **common)
    assert (blocked.backend, blocked.fallback_applied, blocked.requires_error) == ("cpu_sh", True, False)
    assert "physics" in blocked.fallback_reason


def test_torch_sh_preflight_refuses_adaptive_gravity() -> None:
    torch = pytest.importorskip("torch")
    from lunaris.core.torch_sh_propagator import TorchSHBatchPropagator, TorchSHPreflightError

    dyn = SimpleNamespace(gravity_adaptive=SimpleNamespace(enabled=True), grav=None, ephem=None)
    cfg = SimpleNamespace(sh_degree=8, dt_s=60.0, torch_dtype="float64", impact_alt_km=0.0)
    with pytest.raises(TorchSHPreflightError, match="adaptive"):
        TorchSHBatchPropagator(dyn, cfg, PerturbationFlags(enable_sh=True), device=torch.device("cpu"))
