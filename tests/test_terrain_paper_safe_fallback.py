"""R12: terrain-impact requested but no usable topography payload.

The impact freeze silently degrading to a constant sphere is a hidden
simplification. Paper-safe / strict runs must hard-fail; research-mode runs must
warn and record ``terrain_fallback='sphere'`` so the surface used is never
ambiguous. Both are exercised at the ``_build_propagator`` policy layer without
constructing a real GPU propagator.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lunaris.common.batch_defs import BatchPropagationConfig
from lunaris.common.type_defs import PerturbationFlags


def _sim():
    return SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True),
        gravity=SimpleNamespace(uses_st_lrps=False),
    )


def _terrain_engine(tmp_path: Path, *, strict: bool):
    from lunaris.batch.engine import BatchPropagationEngine

    # The batch engine's strict-run signal (no silent simplification) is
    # sh_fallback_policy='error'; _fallback_forbidden also honors paper_safe /
    # strict_backend / benchmark_mode when a caller exposes them on the config.
    engine = BatchPropagationEngine.__new__(BatchPropagationEngine)
    engine._cfg = BatchPropagationConfig(
        n_samples=2,
        use_gpu=False,            # CPU path so no GPU stack is needed
        batch_backend="cpu_sh",
        sh_degree=8,
        detect_impact=True,
        impact_surface_mode="terrain",   # terrain requested...
        sh_fallback_policy="error" if strict else "compatible_gpu",
        output_format="npz",
        output_path=str(tmp_path / "terrain_r12.npz"),
    )
    engine._sim_cfg = _sim()
    engine._dyn = SimpleNamespace(grav=SimpleNamespace(degree_max=8), ephem=None)
    engine._surface_provider = None   # ...but no topography payload available
    engine._topo_grid = None
    engine._backend_note = ""
    engine._backend_plan = None
    engine._terrain_fallback = None
    return engine


def test_strict_terrain_without_payload_hard_fails(tmp_path: Path) -> None:
    from lunaris.batch.engine import BatchPropagationEngine

    engine = _terrain_engine(tmp_path, strict=True)
    with pytest.raises(RuntimeError, match="terrain.*sphere|Paper-safe"):
        BatchPropagationEngine._build_propagator(engine)


def test_research_mode_terrain_without_payload_warns_and_records(tmp_path: Path, monkeypatch) -> None:
    import lunaris.core.batch_propagator as batch_prop
    from lunaris.batch.engine import BatchPropagationEngine

    built = {}

    class DummyCPU:
        def __init__(self, *a, **k) -> None:
            built["topo_payload"] = k.get("topo_payload")

    monkeypatch.setattr(batch_prop, "CPUBatchPropagator", DummyCPU)

    engine = _terrain_engine(tmp_path, strict=False)
    with pytest.warns(RuntimeWarning, match="terrain.*sphere|Falling back to a constant sphere"):
        BatchPropagationEngine._build_propagator(engine)

    # The downgrade is recorded, and the sphere freeze (no topo payload) is used.
    assert engine._terrain_fallback == "sphere"
    assert built["topo_payload"] is None
