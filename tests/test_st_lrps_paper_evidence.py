"""Part 1 — ST-LRPS paper evidence pipeline (configs, manifest, runner, guards)."""

from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import copy
import json
from pathlib import Path

import pytest

from lunaris.surrogate.st_lrps.paper_evidence import (
    PaperConfigError,
    build_training_argv,
    collect_environment,
    compute_file_sha256,
    compute_json_hash,
    validate_st_lrps_paper_training_config,
    write_evidence_manifest,
)
from lunaris.surrogate.st_lrps.paper_evidence import runner as paper_runner
from lunaris.surrogate.st_lrps.paper_evidence.config_validation import load_paper_training_config
from lunaris.surrogate.st_lrps.paper_evidence.evidence_manifest import build_evidence_manifest
from lunaris.surrogate.st_lrps.paper_evidence.training_argv import find_unfilled_placeholders

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO / "configs" / "st_lrps" / "paper"
_SEED_CONFIGS = ["train_full_seed42.json", "train_full_seed123.json", "train_full_seed2026.json"]


def _load_raw(name: str) -> dict:
    return json.loads((_CONFIGS / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task 2 — canonical configs exist and are paper-safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _SEED_CONFIGS)
def test_paper_configs_exist_and_validate(name):
    config = load_paper_training_config(_CONFIGS / name)  # raises if unsafe
    assert config["paper_safe"] is True
    assert config["scaler"]["fit_scope"] == "train_only"
    safety = config["contract_safety"]
    assert safety["strict_dataset_contract"] is True


def test_configs_differ_only_by_seed_and_output():
    base = _load_raw("train_full_seed42.json")
    for name, seed in (("train_full_seed123.json", 123), ("train_full_seed2026.json", 2026)):
        other = _load_raw(name)
        assert other["seed"] == seed
        assert other["split"]["split_seed"] == seed
        assert f"seed{seed}" in other["output"]["out_dir"]
        # Everything else identical.
        a, b = copy.deepcopy(base), copy.deepcopy(other)
        for d, _s in ((a, 42), (b, seed)):
            d.pop("name")
            d.pop("seed")
            d["split"].pop("split_seed")
            d["output"].pop("out_dir")
            d.pop("description", None)
        assert a == b


# ---------------------------------------------------------------------------
# Task 6 — config validator rejects unsafe settings
# ---------------------------------------------------------------------------

def _mutate(**path_value):
    cfg = _load_raw("train_full_seed42.json")
    for dotted, value in path_value.items():
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        if value is _DELETE:
            node.pop(keys[-1], None)
        else:
            node[keys[-1]] = value
    return cfg


_DELETE = object()


@pytest.mark.parametrize(
    "mutation",
    [
        {"scaler.fit_scope": "full_dataset"},
        {"split.split_policy": _DELETE},
        {"contract_safety.strict_dataset_contract": False},
        {"output.out_dir": _DELETE},
        {"seed": _DELETE},
        {"target.target_mode": _DELETE},
        {"target.base_sh_degree": _DELETE},
        {"target.target_sh_degree": _DELETE},
        {"artifact_contract_output": False},
    ],
)
def test_validator_rejects_unsafe(mutation):
    cfg = _mutate(**mutation)
    with pytest.raises(PaperConfigError):
        validate_st_lrps_paper_training_config(cfg)


def test_validator_error_lists_the_problem():
    cfg = _mutate(**{"scaler.fit_scope": "full_dataset"})
    with pytest.raises(PaperConfigError, match="train_only"):
        validate_st_lrps_paper_training_config(cfg)


# ---------------------------------------------------------------------------
# Task 4 — flag mapping
# ---------------------------------------------------------------------------

def test_build_training_argv_emits_explicit_flags_and_no_safety_flags():
    cfg = _load_raw("train_full_seed42.json")
    argv = build_training_argv(cfg)
    # Core flags present.
    assert "--out" in argv and "--seed" in argv and "42" in argv
    assert "--split-policy" in argv and "spatial_block" in argv
    assert "--u-scale-mode" in argv and "--epochs" in argv and "--batch-size" in argv
    # Boolean toggles map to the right form.
    assert "--use-residual-blocks" in argv
    assert "--use-altitude-balanced-loss" in argv
    assert "--use-radial-cross-loss" in argv
    # Legacy/safety flags are NEVER emitted (they default to safe/false).
    assert not any(a.startswith("--allow-") for a in argv)


def test_bool_toggle_off_form():
    cfg = _mutate(**{"model.use_residual_blocks": False, "loss.use_radial_cross_loss": False})
    argv = build_training_argv(cfg)
    assert "--no-residual-blocks" in argv and "--use-residual-blocks" not in argv
    assert "--no-radial-cross-loss" in argv


def test_find_unfilled_placeholders():
    cfg = _load_raw("train_full_seed42.json")
    assert any("train_data" in p for p in find_unfilled_placeholders(cfg))
    cfg["dataset"]["train_data"] = "/data/real_cloud.h5"
    assert find_unfilled_placeholders(cfg) == []


def test_mapped_flags_parse_into_valid_trainconfig(monkeypatch):
    """Every emitted flag must be a real trainer flag, and the safety flags must
    default to the safe (false) value because the mapper never emits them."""
    import sys

    from lunaris.surrogate.st_lrps.training import config as tcfg

    cfg = _load_raw("train_full_seed42.json")
    cfg["dataset"]["train_data"] = "C:/tmp/does_not_exist_cloud.h5"  # existence checked later in engine.train
    flags = build_training_argv(cfg)
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *flags])
    tc = tcfg.parse_args()  # raises if any flag name/type is wrong

    assert tc.split_policy == "spatial_block"
    assert tc.seed == 42 and tc.epochs == 400 and tc.n_bands == 3
    assert tc.use_residual_blocks is True and tc.use_radial_cross_loss is True
    assert float(tc.spatial_val_block_fraction) == 0.15


# ---------------------------------------------------------------------------
# Task 3 — evidence manifest helpers
# ---------------------------------------------------------------------------

def test_compute_file_sha256_and_missing(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    assert compute_file_sha256(f) == compute_file_sha256(f)
    assert len(compute_file_sha256(f)) == 64
    assert compute_file_sha256(tmp_path / "nope.txt") is None


def test_compute_json_hash_deterministic():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert compute_json_hash(a) == compute_json_hash(b)


def test_collect_environment_keys():
    env = collect_environment()
    assert "python_version" in env and "git" in env
    assert "commit_sha" in env["git"]


def test_write_evidence_manifest_records_missing_and_merges(tmp_path):
    manifest_path = tmp_path / "manifests" / "evidence_manifest.json"
    entry = build_evidence_manifest(
        stage="train",
        run_key="seed42",
        config_path=None,
        config={"seed": 42},
        out_dir=tmp_path / "out",
        artifacts={"checkpoint": tmp_path / "missing.pt"},
        command=["python", "-m", "trainer"],
        dry_run=True,
    )
    write_evidence_manifest(manifest_path, run_key="seed42", entry=entry)
    write_evidence_manifest(
        manifest_path, run_key="seed123",
        entry=build_evidence_manifest(stage="train", run_key="seed123", config_path=None,
                                      config={"seed": 123}, out_dir=None, dry_run=True),
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(data["runs"].keys()) == {"seed42", "seed123"}
    rec = data["runs"]["seed42"]["artifacts"]["checkpoint"]
    assert rec["sha256"] is None and rec["missing_reason"]  # missing recorded, not ignored


# ---------------------------------------------------------------------------
# Task 4 — runner dry-run + stage dispatch
# ---------------------------------------------------------------------------

def test_runner_dry_run_writes_plan_no_training(tmp_path, capsys):
    rc = paper_runner.main([
        "--stage", "train",
        "--config", str(_CONFIGS / "train_full_seed42.json"),
        "--evidence-root", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "planned command" in out
    evidence_dir = tmp_path / "training" / "st_lrps_paper_full_seed42"
    assert (evidence_dir / "train_command.txt").exists()
    assert (evidence_dir / "environment.json").exists()
    manifest = json.loads((tmp_path / "manifests" / "evidence_manifest.json").read_text(encoding="utf-8"))
    run = manifest["runs"]["st_lrps_paper_full_seed42"]
    assert run["dry_run"] is True
    assert run["extra"]["unfilled_placeholders"]  # placeholders surfaced
    # No trainer output directory was created in a dry run.
    assert not (tmp_path / "outputs").exists()


def test_runner_seed_override(tmp_path):
    rc = paper_runner.main([
        "--stage", "train",
        "--config", str(_CONFIGS / "train_full_seed42.json"),
        "--evidence-root", str(tmp_path),
        "--seed", "999",
        "--out-dir", str(tmp_path / "run999"),
        "--dry-run",
    ])
    assert rc == 0
    manifest = json.loads((tmp_path / "manifests" / "evidence_manifest.json").read_text(encoding="utf-8"))
    run = next(iter(manifest["runs"].values()))
    assert any("999" in c for c in run["command"])


def test_runner_rejects_unsafe_config(tmp_path, capsys):
    bad = _mutate(**{"scaler.fit_scope": "full_dataset"})
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    rc = paper_runner.main(["--stage", "train", "--config", str(bad_path), "--dry-run",
                            "--evidence-root", str(tmp_path)])
    assert rc == 2
    assert "CONFIG REJECTED" in capsys.readouterr().err


def test_runner_train_requires_config(capsys):
    rc = paper_runner.main(["--stage", "train"])
    assert rc == 2
    assert "--config is required" in capsys.readouterr().err


@pytest.mark.parametrize("stage", ["field-validation", "orbit-benchmark", "worst-case", "multi-seed", "tables"])
def test_pipeline_stages_dispatch_in_dry_run(stage, tmp_path):
    # Parts 2/3 implement these stages; they dispatch (no "not implemented") and
    # write into the provided evidence root (never the repo workspace) in dry-run.
    rc = paper_runner.main(["--stage", stage, "--evidence-root", str(tmp_path), "--dry-run"])
    assert rc == 0


def test_mark_pre_hygiene(tmp_path):
    run_dir = tmp_path / "old_run"
    rc = paper_runner.main(["--mark-pre-hygiene", str(run_dir)])
    assert rc == 0
    marker = json.loads((run_dir / "PRE_HYGIENE.json").read_text(encoding="utf-8"))
    assert marker["status"] == "pre_hygiene"
    assert marker["not_for_final_paper_claims"] is True


# ---------------------------------------------------------------------------
# Task 5 — hygiene verification + evidence packaging (needs a real run dir)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
from st_lrps_contract_test_utils import make_contract_run  # noqa: E402


def _make_hygiene_run(tmp_path, *, fit_scope="train_only", with_split_manifest=True):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    run_dir = Path(run["run_dir"])
    scaler = json.loads((run_dir / "scaler.json").read_text(encoding="utf-8"))
    scaler.setdefault("provenance", {})["fit_scope"] = fit_scope
    scaler["provenance"]["split_policy"] = "spatial_block"
    (run_dir / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    if with_split_manifest:
        prov = run_dir / "provenance"
        prov.mkdir(parents=True, exist_ok=True)
        (prov / "split_manifest.json").write_text(
            json.dumps({
                "split_policy": "spatial_block",
                "train_count": 80, "val_count": 20, "test_count": 0, "ood_count": 0,
                "index_hashes": {"train": "a" * 64, "val": "b" * 64, "test": "", "ood": ""},
            }),
            encoding="utf-8",
        )
    return run_dir


def test_verify_passes_for_hygiene_run(tmp_path):
    run_dir = _make_hygiene_run(tmp_path)
    verified = paper_runner.verify_paper_run_artifacts(run_dir)
    assert verified["checkpoint"].exists()
    assert verified["scaler"].exists()


def test_verify_fails_on_non_train_only_scaler(tmp_path):
    run_dir = _make_hygiene_run(tmp_path, fit_scope="full_dataset")
    with pytest.raises(paper_runner.PaperEvidenceError, match="train_only"):
        paper_runner.verify_paper_run_artifacts(run_dir)


def test_verify_fails_on_missing_split_manifest(tmp_path):
    run_dir = _make_hygiene_run(tmp_path, with_split_manifest=False)
    with pytest.raises(paper_runner.PaperEvidenceError, match="split_manifest"):
        paper_runner.verify_paper_run_artifacts(run_dir)


def test_verify_detects_split_hash_collision(tmp_path):
    run_dir = _make_hygiene_run(tmp_path)
    prov = run_dir / "provenance" / "split_manifest.json"
    manifest = json.loads(prov.read_text(encoding="utf-8"))
    manifest["index_hashes"]["val"] = manifest["index_hashes"]["train"]  # same indices -> overlap
    prov.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(paper_runner.PaperEvidenceError, match="overlap"):
        paper_runner.verify_paper_run_artifacts(run_dir)


def test_package_evidence_writes_full_bundle(tmp_path):
    run_dir = _make_hygiene_run(tmp_path)
    config = load_paper_training_config(_CONFIGS / "train_full_seed42.json")
    evidence_dir = tmp_path / "evidence"
    paper_runner.package_evidence(
        run_dir, evidence_dir, config=config, command=["python", "-m", "trainer", "--x"]
    )
    for name in (
        "training_config_resolved.json",
        "artifact_contract.json",
        "scaler.json",
        "split_manifest.json",
        "training_summary.md",
        "environment.json",
        "train_command.txt",
        "paper_evidence.json",
    ):
        assert (evidence_dir / name).exists(), name
    # Large checkpoint is referenced, not copied into the evidence dir.
    assert not (evidence_dir / "ckpt_best.pt").exists()
    marker = json.loads((evidence_dir / "paper_evidence.json").read_text(encoding="utf-8"))
    assert marker["hygiene_compliant"] is True
    assert marker["checkpoint"]["sha256"]
    summary = (evidence_dir / "training_summary.md").read_text(encoding="utf-8")
    assert "PRELIMINARY" in summary and "train_only" in summary
