"""Warm start (``--init-from``): weights only, into a new run, never writing the source."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from dataset_pipeline_test_utils import write_toy_contract_h5
from lunaris.surrogate.st_lrps.artifacts.manager import resolve_resume_checkpoint
from lunaris.surrogate.st_lrps.shared.scaling import ScalerPack
from lunaris.surrogate.st_lrps.training import engine
from lunaris.surrogate.st_lrps.training.config import parse_args
from test_st_lrps_training_guards import _tiny_cfg


def _tree_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture
def source_run(tmp_path):
    data = write_toy_contract_h5(tmp_path / "toy_train.h5", n=48)
    run_dir = tmp_path / "source"
    engine.train(_tiny_cfg(data, run_dir))
    return data, run_dir


def test_init_loads_the_source_weights_exactly(source_run, tmp_path):
    data, src = source_run
    _, _, ckpt = resolve_resume_checkpoint(src, prefer="best")
    try:
        session = engine.build_training_session(_tiny_cfg(data, tmp_path / "warm", init_from=str(src)))
    finally:
        engine._detach_run_logs()
    loaded = session.model.state_dict()
    for name, tensor in ckpt["model_state_dict"].items():
        assert torch.equal(loaded[name].cpu(), tensor.cpu()), name
    assert session._init_source["checkpoint_kind"] == "best"


def test_warm_start_trains_into_a_new_run_and_leaves_the_source_untouched(source_run, tmp_path):
    data, src = source_run
    before = _tree_hashes(src)
    warm = tmp_path / "warm"
    status = engine.train(_tiny_cfg(data, warm, init_from=str(src)))
    assert status == "completed"
    assert _tree_hashes(src) == before
    manifest = json.loads((warm / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["init_from"]["run_dir"] == str(src.resolve())
    assert set(manifest["init_from"]["sha256"]) == {"checkpoint", "scaler", "config"}
    # Fresh counters: the source finished epoch 0, so a resume would start at 1;
    # the warm run starts at its own epoch 0.
    history = [json.loads(line) for line in (warm / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [h["epoch"] for h in history] == [0]


@pytest.mark.parametrize("where", ["same", "inside", "parent"])
def test_init_refuses_an_output_that_overlaps_the_source(source_run, where):
    data, src = source_run
    before = _tree_hashes(src)
    out = {"same": src, "inside": src / "child", "parent": src.parent}[where]
    with pytest.raises(ValueError, match="overlap"):
        engine.train(_tiny_cfg(data, out, init_from=str(src)))
    assert _tree_hashes(src) == before


def test_init_refuses_a_different_network(source_run, tmp_path):
    data, src = source_run
    with pytest.raises(ValueError, match="architecture signature"):
        engine.train(_tiny_cfg(data, tmp_path / "warm", hidden=16, init_from=str(src)))


def test_init_on_new_data_reuses_the_source_scaler(source_run, tmp_path):
    _, src = source_run
    new_data = write_toy_contract_h5(tmp_path / "toy_new.h5", n=64, alt_min_km=120.0, alt_max_km=450.0)
    warm = tmp_path / "warm"
    assert engine.train(_tiny_cfg(new_data, warm, init_from=str(src))) == "completed"
    source_scaler = ScalerPack.load_json(src / "scaler.json")
    warm_scaler = ScalerPack.load_json(warm / "scaler.json")
    assert (warm_scaler.x, warm_scaler.u, warm_scaler.a) == (source_scaler.x, source_scaler.u, source_scaler.a)
    inherited = warm_scaler.provenance["inherited_from"]
    assert inherited["scaler_path"] == str(src.resolve() / "scaler.json")
    assert inherited["warm_start_dataset_content_sha256"] != source_scaler.provenance["dataset_content_sha256"]


def test_init_and_resume_are_exclusive_on_the_command_line(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["lunaris-train", "--init-from", str(run_dir), "--resume-from", str(run_dir)])
    with pytest.raises(SystemExit):
        parse_args()


def test_init_without_data_uses_the_source_runs_files(tmp_path, monkeypatch):
    train = write_toy_contract_h5(tmp_path / "train.h5", n=48)
    val = write_toy_contract_h5(tmp_path / "val.h5", n=16)
    src = tmp_path / "source"
    (src / "checkpoints").mkdir(parents=True)
    (src / "config.json").write_text(
        json.dumps({"data": str(train), "train_data": str(train), "val_data": str(val)}), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["lunaris-train", "--init-from", str(src), "--out", str(tmp_path / "warm")])
    cfg = parse_args()
    assert (cfg.train_data, cfg.val_data) == (str(train), str(val))
    assert cfg.init_from == str(src)
    assert cfg.init_checkpoint == "best"


def test_a_warm_run_on_new_data_can_be_resumed(source_run, tmp_path):
    # The inherited scaler carries the source's split provenance; a resume of the
    # warm run must check the data the warm run started on instead.
    _, src = source_run
    new_data = write_toy_contract_h5(tmp_path / "toy_new.h5", n=64)
    warm = tmp_path / "warm"
    assert engine.train(_tiny_cfg(new_data, warm, init_from=str(src))) == "completed"
    assert engine.train(_tiny_cfg(new_data, warm, epochs=2, resume_from=str(warm))) == "completed"
    manifest = json.loads((warm / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["init_from"]["run_dir"] == str(src.resolve())  # lineage survives the resume


def test_resuming_a_warm_run_on_other_data_is_refused(source_run, tmp_path):
    _, src = source_run
    new_data = write_toy_contract_h5(tmp_path / "toy_new.h5", n=64)
    other = write_toy_contract_h5(tmp_path / "toy_other.h5", n=80)
    warm = tmp_path / "warm"
    engine.train(_tiny_cfg(new_data, warm, init_from=str(src)))
    with pytest.raises(ValueError, match="training_data_lineage"):
        engine.train(_tiny_cfg(other, warm, epochs=2, resume_from=str(warm)))


def test_init_refuses_data_beyond_the_source_input_scale(source_run, tmp_path):
    _, src = source_run  # source data reach 500 km, so x_scale = r_ref + 500 km
    high = write_toy_contract_h5(tmp_path / "toy_high.h5", n=64, alt_max_km=700.0)
    with pytest.raises(ValueError, match="normalized radius"):
        engine.train(_tiny_cfg(high, tmp_path / "warm", init_from=str(src)))


def test_init_refuses_the_source_training_file_as_held_out_data(source_run, tmp_path):
    src_data, src = source_run
    new_data = write_toy_contract_h5(tmp_path / "toy_new.h5", n=64)
    cfg = _tiny_cfg(None, tmp_path / "warm", init_from=str(src), train_data=str(new_data), val_data=str(src_data))
    with pytest.raises(ValueError, match="cannot be held out"):
        engine.train(cfg)


def test_init_refuses_a_different_split_of_the_source_file(source_run, tmp_path):
    src_data, src = source_run
    with pytest.raises(ValueError, match="re-splits a file the weights were trained on"):
        engine.train(_tiny_cfg(src_data, tmp_path / "warm", split_seed=12, init_from=str(src)))


def test_init_refuses_labels_from_another_gravity_model(source_run, tmp_path):
    _, src = source_run
    other_physics = write_toy_contract_h5(
        tmp_path / "toy_other_model.h5", n=64, contract_overrides={"source_gravity_file_sha256": "0" * 64}
    )
    with pytest.raises(RuntimeError, match="not compatible with the source checkpoint.*source_gravity_file_sha256"):
        engine.train(_tiny_cfg(other_physics, tmp_path / "warm", init_from=str(src)))


def test_the_source_check_fails_and_is_recorded_when_the_source_changed(source_run, tmp_path):
    data, src = source_run
    warm = tmp_path / "warm"
    engine.train(_tiny_cfg(data, warm, init_from=str(src)))
    record = json.loads((warm / "run_manifest.json").read_text(encoding="utf-8"))["init_from"]
    (src / "config.json").write_text("{}", encoding="utf-8")
    layout = engine.ensure_run_layout(warm)
    with pytest.raises(RuntimeError, match="changed during the warm start"):
        engine._verify_init_source_unchanged(record, layout)
    manifest = json.loads((warm / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["init_from_source_unchanged"] is False


def test_a_chained_warm_start_refuses_an_ancestors_training_file(source_run, tmp_path):
    # source -> warm1 (trained on d1) -> warm2: d1 must not become warm2's validation file.
    _, src = source_run
    d1 = write_toy_contract_h5(tmp_path / "toy_d1.h5", n=64)
    d2 = write_toy_contract_h5(tmp_path / "toy_d2.h5", n=72)
    warm1 = tmp_path / "warm1"
    engine.train(_tiny_cfg(d1, warm1, init_from=str(src)))
    lineage = ScalerPack.load_json(warm1 / "scaler.json").provenance["training_data_lineage"]
    assert len(lineage) == 2
    cfg = _tiny_cfg(None, tmp_path / "warm2", init_from=str(warm1), train_data=str(d2), val_data=str(d1))
    with pytest.raises(ValueError, match="cannot be held out"):
        engine.train(cfg)


def test_the_source_is_verified_even_when_the_training_loop_fails(source_run, tmp_path, monkeypatch):
    data, src = source_run
    warm = tmp_path / "warm"

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected failure inside the epoch loop")

    monkeypatch.setattr(engine, "_lr_multiplier_for_epoch", _boom)
    with pytest.raises(RuntimeError, match="injected failure"):
        engine.train(_tiny_cfg(data, warm, init_from=str(src)))
    manifest = json.loads((warm / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["init_from_source_unchanged"] is True
