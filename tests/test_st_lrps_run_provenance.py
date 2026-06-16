"""P3: ST-LRPS run_provenance contract completeness and audit↔training consistency.

These tests pin that:
  * ``build_run_provenance`` records the reproducibility fields the audit gate
    requires, with valid SHA-256 digests for the contract/split.
  * ``run_provenance`` never feeds ``training_config_hash`` (config identity must
    stay stable across runs).
  * ``build_resolved_config`` emits a run_provenance block that
    ``classify_st_lrps_run`` accepts as TRUSTED when full, and degrades cleanly
    otherwise.
"""

from __future__ import annotations

import torch
from tests.st_lrps_contract_test_utils import (
    tiny_dataset_meta,
    tiny_scaler,
    tiny_training_cfg,
)

from lunaris.analysis.monte_carlo.result_audit import (
    QUARANTINED,
    TRUSTED,
    _is_valid_sha256,
    classify_st_lrps_run,
)
from lunaris.common.hashing import canonical_json_sha256
from lunaris.surrogate.st_lrps.artifacts.manager import (
    ST_LRPS_PROVENANCE_VERSION,
    build_resolved_config,
    build_run_provenance,
)
from lunaris.surrogate.st_lrps.networks.models import (
    build_model_from_config,
    compute_architecture_signature,
)

_DETERMINISM = {"use_deterministic_algorithms": True, "tf32": False}
_SPLIT_MANIFEST = {"train_count": 100, "val_count": 20, "index_hashes": {"train": "c" * 64}}


def _resolved_cfg(*, determinism=_DETERMINISM, with_split=True):
    cfg = tiny_training_cfg()
    model = build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)
    cfg["input_feature_dim"] = int(getattr(model, "input_feature_dim", 3))
    cfg["embedding_type"] = str(getattr(model, "embedding_type", "raw"))
    cfg["model_builder_version"] = str(getattr(model, "model_builder_version", "unknown"))
    arch_sig = compute_architecture_signature(cfg)
    dataset_meta = tiny_dataset_meta()
    if with_split:
        dataset_meta["split_manifest"] = dict(_SPLIT_MANIFEST)
    return build_resolved_config(
        cfg, dataset_meta, model, tiny_scaler(), arch_sig, determinism=determinism
    )


def test_build_run_provenance_has_valid_hashes_and_kind() -> None:
    contract = {"x": 1, "y": [2, 3]}
    prov = build_run_provenance(
        artifact_contract=contract,
        split_manifest=_SPLIT_MANIFEST,
        determinism=_DETERMINISM,
        runtime_model_kind="force_direct",
    )
    assert prov["provenance_version"] == ST_LRPS_PROVENANCE_VERSION
    assert prov["model_kind"] == "force_direct"
    assert prov["determinism"] == _DETERMINISM
    assert _is_valid_sha256(prov["artifact_contract_sha256"])
    assert _is_valid_sha256(prov["split_manifest_sha256"])
    # The recorded digest must match a fresh recompute with the shared canonical
    # hasher (the same function result_audit uses to verify it).
    assert prov["artifact_contract_sha256"] == canonical_json_sha256(contract)


def test_build_run_provenance_nullables_when_inputs_absent() -> None:
    prov = build_run_provenance(
        artifact_contract=None,
        split_manifest=None,
        determinism=None,
        runtime_model_kind=None,
    )
    assert prov["artifact_contract_sha256"] is None
    assert prov["split_manifest_sha256"] is None
    assert prov["determinism"] is None
    # model_kind defaults to the autograd runtime, never empty.
    assert prov["model_kind"] == "potential_autograd"


def test_run_provenance_excluded_from_training_config_hash() -> None:
    # Two builds that differ only by determinism (hence run_provenance) must share
    # the same config-identity hash.
    cfg_a = _resolved_cfg(determinism={"a": 1})
    cfg_b = _resolved_cfg(determinism={"b": 2})
    assert cfg_a["run_provenance"]["determinism"] != cfg_b["run_provenance"]["determinism"]
    assert cfg_a["training_config_hash"] == cfg_b["training_config_hash"]


def test_resolved_config_is_trusted_by_audit() -> None:
    cfg = _resolved_cfg()
    prov = cfg["run_provenance"]
    assert _is_valid_sha256(prov["artifact_contract_sha256"])
    assert _is_valid_sha256(prov["split_manifest_sha256"])
    status, reasons = classify_st_lrps_run(cfg, has_checkpoint=True)
    assert status == TRUSTED, reasons


def test_resolved_config_without_split_is_quarantined() -> None:
    cfg = _resolved_cfg(with_split=False)
    assert cfg["run_provenance"]["split_manifest_sha256"] is None
    status, reasons = classify_st_lrps_run(cfg, has_checkpoint=True)
    assert status == QUARANTINED
    assert any("split_manifest" in r for r in reasons)
