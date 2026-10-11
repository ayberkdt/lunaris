"""A strict artifact-contract read refuses run inputs it cannot trust.

``read_artifact_contract(strict=True)`` used to swallow a ``scaler.json`` that
disagreed with the checkpoint and an unreadable ``config.json``, fall back to
the checkpoint's own copies and report the contract valid. The paper-evidence
gate and the benchmark pipeline call it with ``strict=True``. Separately,
``load_scaler_for_run`` ignored a checkpoint's pinned ``scaler_hash`` when the
checkpoint carried no embedded scaler payload.
"""

from __future__ import annotations

import json
import logging

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.artifacts.manager import (  # noqa: E402
    load_scaler_for_run,
    make_run_layout,
    read_artifact_contract,
)
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContractError  # noqa: E402
from st_lrps_contract_test_utils import make_contract_run  # noqa: E402


def _tamper_scaler(run_dir) -> None:
    path = run_dir / "scaler.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["u"]["scale"] = float(payload["u"]["scale"]) * 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_an_intact_run_reads_strictly(tmp_path) -> None:
    run = make_contract_run(tmp_path)
    assert read_artifact_contract(run["run_dir"], strict=True) is not None


def test_a_scaler_file_that_disagrees_with_the_checkpoint_is_refused(tmp_path) -> None:
    run = make_contract_run(tmp_path)
    _tamper_scaler(run["run_dir"])
    with pytest.raises(ArtifactContractError, match="scaler"):
        read_artifact_contract(run["run_dir"], strict=True)


def test_an_unreadable_config_file_is_refused(tmp_path) -> None:
    run = make_contract_run(tmp_path)
    (run["run_dir"] / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="config.json"):
        read_artifact_contract(run["run_dir"], strict=True)


def test_a_lenient_read_falls_back_and_says_so(tmp_path, caplog) -> None:
    run = make_contract_run(tmp_path)
    _tamper_scaler(run["run_dir"])
    (run["run_dir"] / "config.json").write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        contract = read_artifact_contract(run["run_dir"], strict=False)
    assert contract is not None
    assert "config.json" in caplog.text and "scaler" in caplog.text


def _drop_embedded_scaler(run_dir) -> dict:
    ckpt_path = run_dir / "checkpoints" / "ckpt_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt.get("scaler_hash")
    ckpt.pop("scaler", None)
    return ckpt


def test_pinned_scaler_hash_is_checked_without_an_embedded_scaler(tmp_path) -> None:
    run = make_contract_run(tmp_path)
    layout = make_run_layout(run["run_dir"])
    ckpt = _drop_embedded_scaler(run["run_dir"])

    # The matching scaler.json loads and reports the pinned hash.
    _, info = load_scaler_for_run(layout, ckpt, device=torch.device("cpu"))
    assert info["scaler_hash"] == ckpt["scaler_hash"]

    # A scaler.json from another run is refused instead of accepted silently.
    _tamper_scaler(run["run_dir"])
    with pytest.raises(RuntimeError, match="scaler_hash"):
        load_scaler_for_run(layout, ckpt, device=torch.device("cpu"))


def test_no_pinned_hash_and_no_embedded_scaler_still_loads(tmp_path) -> None:
    # Older checkpoints without scaler_hash keep loading from scaler.json.
    run = make_contract_run(tmp_path)
    layout = make_run_layout(run["run_dir"])
    ckpt = _drop_embedded_scaler(run["run_dir"])
    ckpt.pop("scaler_hash")
    _tamper_scaler(run["run_dir"])
    _, info = load_scaler_for_run(layout, ckpt, device=torch.device("cpu"))
    assert info["scaler_source"].endswith("scaler.json")
