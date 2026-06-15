"""A2: a force_direct ST-LRPS artifact must never silently fall back to the
legacy local potential model.

The legacy local runtime path (``_build_model_from_config``) only ever builds a
scalar-potential MLP evaluated via autograd. That is the wrong physics for a
``force_direct`` artifact (which predicts residual acceleration directly from a
3-output head). When the canonical runtime loader fails, a force_direct artifact
must hard-fail rather than degrade to a potential model.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.runtime_adapter import SurrogateGravityModel  # noqa: E402
from st_lrps_contract_test_utils import make_contract_run  # noqa: E402


def _raise_canonical_failure(*_args, **_kwargs):
    raise RuntimeError("simulated canonical force_model loader failure")


def test_force_direct_artifact_has_no_legacy_fallback(tmp_path, monkeypatch) -> None:
    art = make_contract_run(
        tmp_path,
        cfg_overrides={"runtime_model_kind": "force_direct", "output_dim": 3},
        contract_overrides={"runtime_model_kind": "force_direct", "output_dim": 3},
    )

    import lunaris.surrogate.st_lrps.runtime.force_model as fm

    monkeypatch.setattr(fm, "load_surrogate_force_model", _raise_canonical_failure)

    with pytest.raises(RuntimeError, match="force_direct"):
        SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")


def test_potential_autograd_artifact_still_allows_fallback(tmp_path, monkeypatch) -> None:
    # Default fixture is potential_autograd. When the canonical loader fails the
    # legacy fallback path is still permitted, so the force_direct guard must NOT
    # fire (any error raised here must be something other than the guard).
    art = make_contract_run(tmp_path)

    import lunaris.surrogate.st_lrps.runtime.force_model as fm

    monkeypatch.setattr(fm, "load_surrogate_force_model", _raise_canonical_failure)

    try:
        SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")
    except RuntimeError as exc:
        assert "force_direct ST-LRPS artifact could not be loaded" not in str(exc)
