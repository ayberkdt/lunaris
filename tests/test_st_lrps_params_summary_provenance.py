"""Unit coverage for ST-LRPS dataset parameters, config summary, and provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.surrogate.st_lrps.data.dataset_parameters import (
    DEFAULT_DATASET_CONFIG,
    MU_MOON_SI,
    R_MOON_SI,
    DatasetParameters,
    canonical_scales,
    is_lunar_body_signature,
    load_run_config,
)
from lunaris.surrogate.st_lrps.evaluation import provenance as prov
from lunaris.surrogate.st_lrps.training.config_summary import (
    build_experiment_feature_summary,
)

# --- dataset_parameters ----------------------------------------------------

def test_dataset_parameters_defaults_are_lunar():
    p = DEFAULT_DATASET_CONFIG
    assert p.central_body == "moon"
    assert is_lunar_body_signature(mu_si=p.mu_si, r_ref_m=p.r_ref_m)
    assert p.mu_moon_si == p.mu_si == MU_MOON_SI
    assert p.r_moon_si == p.r_ref_m == R_MOON_SI
    d = p.to_dict()
    assert d["central_body"] == "moon" and d["mu_si"] == MU_MOON_SI


def test_canonical_scales_math_and_validation():
    du, tu, vu = canonical_scales(mu_si=MU_MOON_SI, du_m=R_MOON_SI)
    assert du == R_MOON_SI
    # TU = sqrt(DU^3 / mu); VU = DU/TU = sqrt(mu/DU).
    np.testing.assert_allclose(tu, np.sqrt(R_MOON_SI**3 / MU_MOON_SI), rtol=1e-12)
    np.testing.assert_allclose(vu, np.sqrt(MU_MOON_SI / R_MOON_SI), rtol=1e-12)
    with pytest.raises(ValueError, match="mu_si must be positive"):
        canonical_scales(mu_si=0.0, du_m=R_MOON_SI)
    with pytest.raises(ValueError, match="du_m must be positive"):
        canonical_scales(mu_si=MU_MOON_SI, du_m=-1.0)


def test_load_run_config_roundtrip(tmp_path):
    cfg = {"data": "x.h5", "out": "run", "depth": 6, "nested": {"a": 1}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert load_run_config(path) == cfg


def test_dataset_parameters_is_frozen():
    p = DatasetParameters()
    with pytest.raises(Exception, match="cannot assign to field"):
        p.mu_si = 1.0  # frozen dataclass


# --- config_summary --------------------------------------------------------

@dataclass
class _ToyContract:
    target_mode: str = "residual"
    base_degree: int = 2


def test_feature_summary_from_dict_reflects_config():
    cfg = {
        "model_preset": "recommended_physical_radial_decay",
        "run_preset": "paper",
        "split_policy": "spatial_block",
        "runtime_model_kind": "potential_autograd",
        "use_residual_blocks": True,
        "n_bands": 3,
        "use_altitude_balanced_loss": True,
        "direction_loss_weight": 0.5,
        "gradnorm_mode": "ntk_init",
    }
    s = build_experiment_feature_summary(cfg)
    assert s["model_preset"] == "recommended_physical_radial_decay"
    assert s["run_preset"] == "paper"
    assert s["split_policy"] == "spatial_block"
    assert s["runtime_model_kind"] == "potential_autograd"
    assert s["residual_blocks"] is True
    assert s["n_bands"] == 3
    assert s["altitude_balanced_loss"] is True
    assert s["direction_loss"]["weight"] == 0.5
    assert s["gradnorm_mode"] == "ntk_init"


def test_feature_summary_from_object_and_model_attrs():
    cfg = SimpleNamespace(model_preset="custom", n_bands=1, input_feature_dim=3)
    model = SimpleNamespace(embedding_type="physical_radial_decay", input_feature_dim=11)
    s = build_experiment_feature_summary(cfg, target_contract=_ToyContract(), model=model)
    # Model attributes win over cfg for encoding identity.
    assert s["input_encoding"] == "physical_radial_decay"
    assert s["input_feature_dim"] == 11
    # Dataclass target contract is serialized via asdict.
    assert s["target_contract"] == {"target_mode": "residual", "base_degree": 2}


def test_contract_dict_variants():
    from lunaris.surrogate.st_lrps.training.config_summary import _contract_dict

    assert _contract_dict(None) is None
    assert _contract_dict({"target_mode": "full"}) == {"target_mode": "full"}
    assert _contract_dict(_ToyContract()) == {"target_mode": "residual", "base_degree": 2}

    class _HasToDict:
        def to_dict(self):
            return {"k": "v"}

    assert _contract_dict(_HasToDict()) == {"k": "v"}
    assert _contract_dict(42) is None  # unsupported type


# --- provenance ------------------------------------------------------------

def test_sha256_helpers_and_payload_is_order_independent():
    assert prov.sha256_text("abc") == prov.sha256_text("abc")
    assert prov.sha256_text("abc") != prov.sha256_text("abd")
    # Canonical payload hash ignores key insertion order.
    a = prov.sha256_payload({"x": 1, "y": 2})
    b = prov.sha256_payload({"y": 2, "x": 1})
    assert a == b


def test_sha256_file_and_artifact_record(tmp_path):
    import hashlib

    f = tmp_path / "blob.bin"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert prov.sha256_file(f) == expected
    assert prov.sha256_file(None) is None
    assert prov.sha256_file(tmp_path / "missing.bin") is None

    rec = prov.artifact_record(f, label="dataset")
    assert rec["sha256"] == expected and rec["missing_reason"] is None
    assert prov.artifact_record(None)["missing_reason"] is not None
    assert prov.artifact_record(tmp_path / "nope")["missing_reason"] is not None
    assert "not a file" in prov.artifact_record(tmp_path)["missing_reason"]


def test_collect_environment_reports_versions():
    env = prov.collect_environment()
    assert "." in env["python_version"]
    assert env["numpy_version"] is not None  # numpy is a hard dependency
    assert isinstance(env["optional_import_errors"], dict)
    assert env["device_name"] in ("cpu",) or isinstance(env["device_name"], str)


def test_collect_git_info_structure():
    info = prov.collect_git_info()
    assert {"commit_sha", "branch", "is_dirty", "repo_root", "errors"} <= set(info)
    assert isinstance(info["errors"], dict)
    # When git resolves cleanly the commit is a hex string.
    if info["errors"]["commit"] is None:
        assert isinstance(info["commit_sha"], str) and len(info["commit_sha"]) >= 7


def test_build_benchmark_manifest_and_write_json(tmp_path):
    cfg = {
        "name": "smoke",
        "truth": {"model": "spherical_harmonics", "degree": 50, "gravity_file": None},
        "surrogate": {"enabled": False},
        "propagation": {"integrator": "rk4", "dt_s": 1.0, "dtype": "float64"},
        "scenario": {"seed": 7, "count": 3, "altitude_min_km": 100.0, "altitude_max_km": 500.0},
        "baselines": [{"name": "sh20", "model": "spherical_harmonics", "degree": 20}],
    }
    cfg_path = tmp_path / "bench.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    manifest = prov.build_benchmark_manifest(
        config=cfg,
        config_path=cfg_path,
        resolved_config_sha256="deadbeef",
        output_dir=tmp_path / "out",
    )
    assert manifest["schema_version"] == 1
    assert manifest["benchmark_name"] == "smoke"
    assert manifest["config"]["sha256"] == prov.sha256_file(cfg_path)
    assert manifest["config"]["resolved_config_sha256"] == "deadbeef"
    assert manifest["models"]["truth"]["degree"] == 50
    assert manifest["models"]["surrogate"]["enabled"] is False
    assert manifest["models"]["baselines"][0]["name"] == "sh20"
    assert manifest["environment"]["numpy_version"] is not None
    assert manifest["scenario"]["seed"] == 7

    written = prov.write_json(tmp_path / "manifest.json", manifest)
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["benchmark_name"] == "smoke"


def test_build_benchmark_manifest_resolves_surrogate_artifacts(tmp_path):
    pytest.importorskip("torch")
    from st_lrps_contract_test_utils import make_contract_run

    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    run_dir = run["run_dir"]
    cfg = {
        "name": "surrogate-bench",
        "truth": {"model": "spherical_harmonics", "degree": 60},
        "surrogate": {"enabled": True, "model_dir": str(run_dir), "baseline_degree": 20},
        "propagation": {"dt_s": 2.0},
        "scenario": {"seed": 1, "count": 1},
    }
    cfg_path = tmp_path / "bench.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    manifest = prov.build_benchmark_manifest(
        config=cfg, config_path=cfg_path, resolved_config_sha256="abc", output_dir=tmp_path / "out",
    )
    surrogate = manifest["models"]["surrogate"]
    assert surrogate["enabled"] is True
    assert surrogate["model_dir_missing_reason"] is None
    # The checkpoint and config are discovered and hashed.
    assert surrogate["checkpoint"]["sha256"] is not None
    assert surrogate["checkpoint"]["missing_reason"] is None
    assert surrogate["config"]["path"].endswith("config.json")
