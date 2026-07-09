# tests/test_gravity_provider_fallback_policy.py
"""Audit F3 — the legacy runtime fallback policy in ``SurrogateGravityModel``.

The legacy local loader skips strict checkpoint-contract validation
(``validate_checkpoint_contract``), so it is reserved for pre-contract
artifacts and must warn loudly when taken. An artifact that declares a
versioned contract (``artifact_contract`` / ``target_contract`` /
``runtime_model_kind``) must load through the canonical runtime or fail
closed. A ``config.json`` <-> checkpoint-config divergence must be surfaced,
never merged silently.
"""

from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import json
import warnings
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_data

torch = pytest.importorskip("torch")

from lunaris.common.constants import MU_MOON
from lunaris.surrogate.runtime import SurrogateGravityModel
from lunaris.surrogate.runtime.networks import _build_model_from_config
from st_lrps_contract_test_utils import make_contract_run


def _make_legacy_run(
    tmp_path: Path,
    run_name: str = "legacy_run",
    config_extra: dict | None = None,
) -> Path:
    """Minimal pre-contract run: no artifact_contract / target_contract / kind."""
    run_dir = tmp_path / run_name
    (run_dir / "checkpoints").mkdir(parents=True)
    config = {
        "hidden": 8,
        "depth": 1,
        "activation": "tanh",
        "dropout": 0.0,
        "resolved_mu_si": float(MU_MOON),
        "resolved_a_sign": 1.0,
        "scaler_kind": "isometric",
        "degree_min": 0,
        "degree_max": 50,
    }
    if config_extra:
        config.update(config_extra)
    scaler = {
        "x": {"mean": [0.0, 0.0, 0.0], "scale": 2_000_000.0},
        "u": {"mean": [0.0], "scale": 1.0},
        "a": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
    }
    model = _build_model_from_config(config)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    torch.save(
        {"model": model.state_dict(), "config": config, "scaler": scaler},
        run_dir / "checkpoints" / "ckpt_best.pt",
    )
    return run_dir


def _force_canonical_failure(monkeypatch) -> None:
    import lunaris.surrogate.runtime.force_runtime as fr

    def _boom(*args, **kwargs):
        raise RuntimeError("canonical runtime deliberately unavailable (test)")

    monkeypatch.setattr(fr, "load_force_runtime", _boom)


def test_pre_contract_artifact_falls_back_with_runtime_warning(tmp_path, monkeypatch):
    run_dir = _make_legacy_run(tmp_path)
    _force_canonical_failure(monkeypatch)
    with pytest.warns(RuntimeWarning, match="legacy local runtime"):
        model = SurrogateGravityModel.from_model_dir(run_dir, device_preference="cpu")
    assert model._force_runtime is None  # legacy path was taken


def test_contract_declaring_artifact_never_falls_back(tmp_path, monkeypatch):
    run = make_contract_run(tmp_path)
    _force_canonical_failure(monkeypatch)
    with pytest.raises(RuntimeError, match="Refusing the legacy fallback"):
        SurrogateGravityModel.from_model_dir(str(run["run_dir"]), device_preference="cpu")


def test_runtime_kind_key_alone_blocks_the_fallback(tmp_path, monkeypatch):
    run_dir = _make_legacy_run(
        tmp_path,
        "kind_only_run",
        config_extra={"runtime_model_kind": "potential_autograd"},
    )
    _force_canonical_failure(monkeypatch)
    with pytest.raises(RuntimeError, match="Refusing the legacy fallback"):
        SurrogateGravityModel.from_model_dir(run_dir, device_preference="cpu")


def test_config_checkpoint_divergence_warns(tmp_path):
    run_dir = _make_legacy_run(tmp_path, "diverged_run")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg["degree_max"] = 60  # checkpoint still embeds 50
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="disagree"):
        model = SurrogateGravityModel.from_model_dir(run_dir, device_preference="cpu")
    # The checkpoint values take precedence in the merge — that must not change.
    assert model.degree_max == 50


def test_matching_configs_do_not_warn_about_divergence(tmp_path):
    run_dir = _make_legacy_run(tmp_path, "clean_run")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        SurrogateGravityModel.from_model_dir(run_dir, device_preference="cpu")
    assert not [w for w in rec if "disagree" in str(w.message)]


# ---------------------------------------------------------------------------
# Audit F7 — the legacy builder must not apply silent architecture defaults.
# w0 is not stored in the state_dict, so a wrong default is silent wrong
# physics; Fourier settings are provenance-relevant even when the projection
# matrix is restored by the strict state_dict load.
# ---------------------------------------------------------------------------

def test_legacy_builder_warns_on_unrecorded_sine_w0():
    cfg = {"activation": "sine", "hidden": 8, "depth": 1}
    with pytest.warns(RuntimeWarning, match="w0"):
        _build_model_from_config(cfg)


def test_legacy_builder_silent_when_w0_recorded():
    cfg = {
        "activation": "sine",
        "hidden": 8,
        "depth": 1,
        "w0_first": 30.0,
        "w0_hidden": 30.0,
    }
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _build_model_from_config(cfg)
    assert not [w for w in rec if "w0" in str(w.message)]


def test_legacy_builder_warns_on_unrecorded_fourier_params():
    cfg = {"activation": "tanh", "hidden": 8, "depth": 1, "use_fourier": True}
    with pytest.warns(RuntimeWarning, match="[Ff]ourier"):
        _build_model_from_config(cfg)


def test_legacy_builder_silent_when_fourier_params_recorded():
    cfg = {
        "activation": "tanh",
        "hidden": 8,
        "depth": 1,
        "use_fourier": True,
        "fourier_n_features": 16,
        "fourier_sigma": 1.0,
        "fourier_seed": 7,
        "fourier_append_raw": False,
    }
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _build_model_from_config(cfg)
    assert not [w for w in rec if "Fourier" in str(w.message)]
