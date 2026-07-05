"""R11: canonical ST-LRPS runtime is the single source; force_model is a shim.

Acceptance (roadmap): scaler / domain-guard / model-kind logic lives ONLY in
``canonical_runtime.py`` — grep-verifiable — while every existing import path
through ``force_model`` keeps resolving to the same objects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

import lunaris.surrogate.st_lrps.runtime.canonical_runtime as canonical  # noqa: E402
import lunaris.surrogate.st_lrps.runtime.force_model as shim  # noqa: E402

_RUNTIME_DIR = Path(canonical.__file__).parent


def test_shim_forwards_every_public_name_to_canonical():
    for name in (
        "SurrogateForceModel",
        "BaseSurrogateRuntime",
        "PotentialAutogradRuntime",
        "load_surrogate_force_model",
        "SUPPORTED_RUNTIME_FRAME",
        "enforce_altitude_envelope_torch",
    ):
        assert getattr(shim, name) is getattr(canonical, name), name


def test_shim_is_dynamic_fold_not_static_reexport():
    # ruff --fix strips static re-exports (known trap); the shim must forward
    # via module __getattr__ and define no runtime classes of its own.
    source = Path(shim.__file__).read_text(encoding="utf-8")
    assert "def __getattr__" in source
    assert "class SurrogateForceModel" not in source


def test_shim_supports_monkeypatch_through_import(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(shim, "load_surrogate_force_model", sentinel)
    # Attribute access through the shim sees the patch (PEP 562 fallback only
    # fires when the real attribute is absent).
    from lunaris.surrogate.st_lrps.runtime import force_model as reimported

    assert reimported.load_surrogate_force_model is sentinel
    # The canonical module itself is untouched.
    assert canonical.load_surrogate_force_model is not sentinel


def test_domain_guard_logic_single_home():
    """The torch-path envelope guard exists only in canonical_runtime; the
    gravity_provider adapter delegates instead of re-implementing it."""
    import lunaris.surrogate.runtime.gravity_provider as gp

    gp_source = Path(gp.__file__).read_text(encoding="utf-8")
    canonical_source = Path(canonical.__file__).read_text(encoding="utf-8")
    marker = "outside the surrogate \ntraining envelope"  # message text home
    assert "training envelope" in canonical_source
    # The adapter carries no copy of the guard math/message — only the delegate call.
    assert "predictions here are extrapolation" not in gp_source
    assert "enforce_altitude_envelope_torch" in gp_source
    del marker


def test_enforce_altitude_envelope_torch_semantics():
    x_in = torch.tensor([[1_838_000.0, 0.0, 0.0]])  # ~100 km over R_ref=1.738e6
    # Inside envelope: no warning, warn-state unchanged.
    warned = canonical.enforce_altitude_envelope_torch(
        x_in, r_ref_m=1.738e6, alt_min_km=50.0, alt_max_km=150.0,
        strict=False, already_warned=False, caller="t",
    )
    assert warned is False
    # Unknown envelope: no-op even for crazy inputs.
    warned = canonical.enforce_altitude_envelope_torch(
        x_in * 100.0, r_ref_m=1.738e6, alt_min_km=None, alt_max_km=None,
        strict=True, already_warned=False, caller="t",
    )
    assert warned is False
    # Outside + strict -> raise; outside + research -> warn once.
    x_out = torch.tensor([[3_000_000.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="strict_domain"):
        canonical.enforce_altitude_envelope_torch(
            x_out, r_ref_m=1.738e6, alt_min_km=50.0, alt_max_km=150.0,
            strict=True, already_warned=False, caller="t",
        )
    with pytest.warns(RuntimeWarning, match="extrapolation"):
        warned = canonical.enforce_altitude_envelope_torch(
            x_out, r_ref_m=1.738e6, alt_min_km=50.0, alt_max_km=150.0,
            strict=False, already_warned=False, caller="t",
        )
    assert warned is True


def test_loader_through_shim_and_canonical_agree(tmp_path):
    from st_lrps_contract_test_utils import make_contract_run

    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    fm_canonical = canonical.load_surrogate_force_model(run["run_dir"], device="cpu")
    fm_shim = shim.load_surrogate_force_model(run["run_dir"], device="cpu")
    assert type(fm_canonical) is type(fm_shim) is canonical.SurrogateForceModel
    assert fm_canonical.artifact_contract.to_dict() == fm_shim.artifact_contract.to_dict()
