"""R01: a force_direct ST-LRPS artifact is fail-closed on main.

The direct residual-acceleration (force_direct) variant is archived in the
``experimental/force-direct-archive`` branch. On main only the conservative
``potential_autograd`` surrogate is loadable. An artifact whose contract still
declares ``runtime_model_kind='force_direct'`` must be rejected with a clear
error rather than loaded (or silently degraded to a potential model).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
_ = pytest.importorskip("torch.nn")

torch = pytest.importorskip("torch")
_ = pytest.importorskip("torch.nn")

torch = pytest.importorskip("torch")

from lunaris.surrogate.runtime import SurrogateGravityModel  # noqa: E402
from lunaris.surrogate.st_lrps.artifacts.manager import read_artifact_contract  # noqa: E402
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContractError  # noqa: E402
from st_lrps_contract_test_utils import make_contract_run  # noqa: E402


def _force_direct_contract_run(tmp_path):
    # The fixture builds a valid potential_autograd model (build_model_from_config
    # no longer accepts force_direct), then we override ONLY the recorded contract
    # so it declares the archived kind — exactly the "legacy artifact on disk"
    # case the fail-closed guard must catch.
    return make_contract_run(
        tmp_path,
        contract_overrides={"runtime_model_kind": "force_direct", "output_dim": 3},
    )


def test_force_direct_contract_is_rejected_fail_closed(tmp_path) -> None:
    art = _force_direct_contract_run(tmp_path)
    with pytest.raises(ArtifactContractError, match="archive|force_direct"):
        read_artifact_contract(art["run_dir"], strict=True)


def test_force_direct_artifact_does_not_load_as_gravity_model(tmp_path) -> None:
    art = _force_direct_contract_run(tmp_path)
    with pytest.raises(Exception, match="force_direct|archive"):
        SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")


def test_potential_autograd_artifact_still_allows_fallback(tmp_path, monkeypatch) -> None:
    # Default fixture is potential_autograd. When the canonical loader fails the
    # legacy fallback path is still permitted, so the force_direct guard must NOT
    # fire. degree_min < 2 keeps the fallback off the SH-baseline branch so the
    # test never touches the (data-only) lunar gravity coefficient file.
    art = make_contract_run(tmp_path, degree_min=1)

    import lunaris.surrogate.st_lrps.runtime.force_model as fm

    def _raise_canonical_failure(*_args, **_kwargs):
        raise RuntimeError("simulated canonical force_model loader failure")

    monkeypatch.setattr(fm, "load_surrogate_force_model", _raise_canonical_failure)

    # The fallback may still fail for an unrelated reason (e.g. a state-dict
    # mismatch); the only invariant under test is that the force_direct hard-fail
    # guard does NOT fire for a potential_autograd artifact.
    try:
        SurrogateGravityModel.from_model_dir(str(art["run_dir"]), device_preference="cpu")
    except Exception as exc:
        assert "force_direct ST-LRPS artifacts are archived" not in str(exc)
