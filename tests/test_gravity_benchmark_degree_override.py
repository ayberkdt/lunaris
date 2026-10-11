"""The gravity benchmark must not silently skip its surrogate degree override.

Regression test: ``propagate_for_scenario`` lowered and restored the surrogate's
``degree_max`` inside ``contextlib.suppress(Exception)``; a provider that
refused the assignment was benchmarked at another degree with no warning, and
a failed restore left a cached model altered for later scenarios.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from lunaris.common.type_defs import PropagatorConfig  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark import compute  # noqa: E402


class _ReadOnlyDegreeProvider:
    degree_min = 10

    @property
    def degree_max(self) -> int:
        return 200


class _ReachedPropagation(Exception):
    pass


def _engine_reached(*_a, **_k):
    raise _ReachedPropagation


def test_refused_degree_override_raises(monkeypatch) -> None:
    monkeypatch.setattr(compute, "replace_sim_config", lambda cfg, **_k: cfg)
    # Pre-fix the refusal was swallowed and the run went on to build the engine.
    monkeypatch.setattr(compute, "DynamicsEngine", _engine_reached)
    cache = SimpleNamespace(get=lambda _name: _ReadOnlyDegreeProvider())
    cfg_base = SimpleNamespace(
        propagator=PropagatorConfig(), spacecraft=None, flags=None, earth_j2=None, srp=None,
        thermal=None, albedo=None, solid_tides=None,
    )
    with pytest.raises(AttributeError):
        compute.propagate_for_scenario(
            "st_lrps", np.zeros(6), SimpleNamespace(), cfg_base, None, cache
        )
