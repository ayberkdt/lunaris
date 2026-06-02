# -*- coding: utf-8 -*-
"""Staged runner for the ST-LRPS paper evidence pipeline (Part 1).

Only the ``train`` stage is implemented in Part 1. The other stages exist as
explicit placeholders that say they belong to Part 2/3 — they never pretend to
be complete.

Train stage:
  1. load + validate the paper config (rejects unsafe settings),
  2. resolve paths and check for unfilled dataset placeholders,
  3. build the canonical trainer command,
  4. launch the existing ST-LRPS trainer (unless ``--dry-run``),
  5. verify the run is hygiene-compliant (train-only scaler, split manifest,
     artifact contract, checkpoint) — failing loudly otherwise,
  6. package a provenance bundle and update the evidence manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from lunaris.common.paths import project_root_from_file
from lunaris.surrogate.st_lrps.paper_evidence.config_validation import (
    PaperConfigError,
    load_paper_training_config,
)
from lunaris.surrogate.st_lrps.paper_evidence.evidence_manifest import (
    build_evidence_manifest,
    collect_environment,
    compute_file_sha256,
    utc_now_iso,
    write_evidence_manifest,
)
from lunaris.surrogate.st_lrps.paper_evidence.training_argv import (
    build_training_command,
    find_unfilled_placeholders,
    resolve_out_dir,
)

PART = 2
# Stages that make up the core evidence pipeline (run by --stage all, in order).
# Ablation is intentionally excluded from `all` — it is secondary/expensive and is
# run explicitly with --stage ablation.
ALL_STAGES = ("train", "field-validation", "orbit-benchmark", "worst-case", "multi-seed", "tables")
SUPPORTED_STAGES = (*ALL_STAGES, "ablation", "all")

# Default paper configs per stage (overridable with --config).
_DEFAULT_CONFIGS = {
    "field-validation": "configs/st_lrps/paper/field_validation.json",
    "worst-case": "configs/st_lrps/paper/worst_case_analysis.json",
    "ablation": "configs/st_lrps/paper/ablation_suite.json",
}
_DEFAULT_BENCHMARK_CONFIGS = (
    "configs/st_lrps/paper/benchmark_1day_high_degree.json",
    "configs/st_lrps/paper/benchmark_5day_general.json",
)


class PaperEvidenceError(RuntimeError):
    """Raised when a hygiene-compliant final-candidate run cannot be produced."""


# ---------------------------------------------------------------------------
# Evidence workspace
# ---------------------------------------------------------------------------

def default_evidence_root() -> Path:
    return project_root_from_file(__file__) / "validation" / "paper_evidence" / "st_lrps"


def _run_key(config: Mapping[str, Any]) -> str:
    name = str(config.get("name") or "").strip()
    if name:
        return name
    seed = config.get("seed")
    return f"seed{seed}" if seed is not None else "run"


# ---------------------------------------------------------------------------
# Hygiene verification (Task 5 — fail loudly)
# ---------------------------------------------------------------------------

def verify_paper_run_artifacts(run_dir: str | Path) -> dict[str, Path]:
    """Verify a completed run is valid for final paper claims; raise otherwise.

    Returns a mapping of logical artifact name -> path. Fails loudly if the
    scaler was not fit train-only, the split manifest is missing, the artifact
    contract cannot be read, or no checkpoint exists.
    """
    from lunaris.surrogate.st_lrps.artifacts.manager import make_run_layout, read_artifact_contract

    run_dir = Path(run_dir)
    layout = make_run_layout(run_dir)

    # 1. Train-only scaler (the core anti-leakage guarantee).
    if not layout.scaler_json.exists():
        raise PaperEvidenceError(f"missing scaler.json under {run_dir}")
    scaler = json.loads(layout.scaler_json.read_text(encoding="utf-8"))
    provenance = scaler.get("provenance") or {}
    fit_scope = str(provenance.get("fit_scope", "")).strip().lower()
    if fit_scope != "train_only":
        raise PaperEvidenceError(
            f"scaler fit_scope={fit_scope!r} (expected 'train_only'). The run used a non-train-only "
            "scaler and is NOT valid for final paper claims."
        )

    # 2. Split manifest present, with distinct, non-empty split index hashes.
    split_manifest_path = layout.provenance_dir / "split_manifest.json"
    if not split_manifest_path.exists():
        raise PaperEvidenceError(f"missing split_manifest.json under {run_dir}/provenance")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    _verify_split_disjointness(split_manifest)

    # 3. Artifact contract must be readable (i.e. it was written).
    try:
        read_artifact_contract(run_dir, strict=True, allow_legacy_contract=False)
    except Exception as exc:  # noqa: BLE001 - surface as a clear paper error
        raise PaperEvidenceError(f"artifact contract is missing or invalid for {run_dir}: {exc}") from exc

    # 4. A checkpoint must exist.
    checkpoint = layout.ckpt_best if layout.ckpt_best.exists() else layout.ckpt_last
    if not checkpoint.exists():
        raise PaperEvidenceError(f"no checkpoint (ckpt_best/ckpt_last) under {run_dir}/checkpoints")

    return {
        "checkpoint": checkpoint,
        "scaler": layout.scaler_json,
        "split_manifest": split_manifest_path,
        "config_resolved": layout.config_json,
        "history": layout.history_csv,
        "environment": layout.provenance_dir / "environment.json",
        "run_manifest": layout.run_manifest_json,
    }


def _verify_split_disjointness(split_manifest: Mapping[str, Any]) -> None:
    """Sanity-check split separation from the manifest.

    The split system guarantees disjoint indices by construction (and is unit
    tested). Persisted manifests carry only per-split index hashes, so here we
    assert those hashes are pairwise distinct for non-empty splits — identical
    hashes would mean the same indices were reused across splits (a leak).
    """
    hashes = split_manifest.get("index_hashes")
    counts = {
        "train": split_manifest.get("train_count", 0),
        "val": split_manifest.get("val_count", 0),
        "test": split_manifest.get("test_count", 0),
        "ood": split_manifest.get("ood_count", 0),
    }
    if not isinstance(hashes, Mapping):
        # Independent-file splits legitimately omit index hashes.
        if str(split_manifest.get("split_policy")) == "independent_files":
            return
        raise PaperEvidenceError("split_manifest.json is missing index_hashes")
    seen: dict[str, str] = {}
    for name, count in counts.items():
        if int(count or 0) <= 0:
            continue
        digest = hashes.get(name)
        if not digest:
            raise PaperEvidenceError(f"split_manifest.json missing index hash for non-empty split {name!r}")
        if digest in seen:
            raise PaperEvidenceError(
                f"split index hash collision: {name!r} and {seen[digest]!r} share indices "
                "(train/val/test/OOD splits overlap)."
            )
        seen[digest] = name


def mark_run_pre_hygiene(run_dir: str | Path, *, reason: str = "trained before the validation-hygiene refactor") -> Path:
    """Stamp an old run as pre-hygiene / not valid for final paper claims."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "PRE_HYGIENE.json"
    marker.write_text(
        json.dumps(
            {
                "status": "pre_hygiene",
                "not_for_final_paper_claims": True,
                "reason": reason,
                "marked_at_utc": utc_now_iso(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


# ---------------------------------------------------------------------------
# Evidence packaging (Task 5 — full provenance per candidate)
# ---------------------------------------------------------------------------

def package_evidence(
    run_dir: str | Path,
    evidence_dir: str | Path,
    *,
    config: Mapping[str, Any],
    command: list[str],
) -> dict[str, Path]:
    """Copy/derive the canonical evidence bundle. Large checkpoints are referenced
    by path+hash, never copied into the workspace."""
    run_dir = Path(run_dir)
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    verified = verify_paper_run_artifacts(run_dir)

    bundle: dict[str, Path] = {}

    def _copy(src: Path, dst_name: str) -> None:
        if src.exists():
            shutil.copyfile(src, evidence_dir / dst_name)
            bundle[dst_name] = evidence_dir / dst_name

    _copy(verified["config_resolved"], "training_config_resolved.json")
    _copy(verified["scaler"], "scaler.json")
    _copy(verified["split_manifest"], "split_manifest.json")
    _copy(verified["history"], "history.csv")

    # environment.json: copy the engine's snapshot, else write a fresh one.
    if verified["environment"].exists():
        _copy(verified["environment"], "environment.json")
    else:
        (evidence_dir / "environment.json").write_text(
            json.dumps(collect_environment(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        bundle["environment.json"] = evidence_dir / "environment.json"

    # Standalone artifact contract.
    artifact_contract_path = evidence_dir / "artifact_contract.json"
    artifact_contract_path.write_text(
        json.dumps(_read_artifact_contract_dict(run_dir), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    bundle["artifact_contract.json"] = artifact_contract_path

    # train_command.txt (the exact launched command).
    (evidence_dir / "train_command.txt").write_text(
        subprocess.list2cmdline([str(c) for c in command]) + "\n", encoding="utf-8"
    )
    bundle["train_command.txt"] = evidence_dir / "train_command.txt"

    # Human-readable training summary.
    summary_path = evidence_dir / "training_summary.md"
    summary_path.write_text(_render_training_summary(run_dir, config, verified["checkpoint"]), encoding="utf-8")
    bundle["training_summary.md"] = summary_path

    # Hygiene marker (checkpoint referenced by path + hash, not copied).
    marker = {
        "produced_by": "st_lrps_paper_evidence_runner",
        "part": PART,
        "hygiene_compliant": True,
        "not_for_final_paper_claims": False,
        "run_dir": str(run_dir),
        "checkpoint": {
            "path": str(verified["checkpoint"]),
            "sha256": compute_file_sha256(verified["checkpoint"]),
        },
        "created_at_utc": utc_now_iso(),
    }
    (evidence_dir / "paper_evidence.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    bundle["paper_evidence.json"] = evidence_dir / "paper_evidence.json"
    bundle["checkpoint"] = verified["checkpoint"]
    return bundle


def _read_artifact_contract_dict(run_dir: Path) -> dict[str, Any]:
    from lunaris.surrogate.st_lrps.artifacts.manager import read_artifact_contract

    return read_artifact_contract(run_dir, strict=True, allow_legacy_contract=False).to_dict()


def _render_training_summary(run_dir: Path, config: Mapping[str, Any], checkpoint: Path) -> str:
    from lunaris.surrogate.st_lrps.artifacts.manager import make_run_layout, read_run_manifest

    layout = make_run_layout(run_dir)
    manifest = read_run_manifest(layout)
    target = config.get("target", {}) if isinstance(config.get("target"), Mapping) else {}
    split = config.get("split", {}) if isinstance(config.get("split"), Mapping) else {}
    lines = [
        f"# ST-LRPS Training Summary — {config.get('name', run_dir.name)}",
        "",
        "> PRELIMINARY: this is a final-candidate training run. Field validation, "
        "OOD/orbit benchmarks, and ablations (Parts 2/3) are required before any "
        "scientific performance claim.",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Seed: {config.get('seed')}  |  split seed: {split.get('split_seed')}",
        f"- Split policy: {split.get('split_policy')}",
        f"- Target mode: {target.get('target_mode')}  |  base SH degree: {target.get('base_sh_degree')}  |  target SH degree: {target.get('target_sh_degree')}",
        f"- Scaler fit scope: train_only (verified)",
        f"- Epochs (target): {config.get('epochs')}  |  batch size: {config.get('batch_size')}",
        f"- Best epoch: {manifest.get('best_epoch')}  |  best score: {manifest.get('best_score')} ({manifest.get('best_metric') or manifest.get('best_score_name')})",
        f"- Status: {manifest.get('status')}",
        f"- Checkpoint: `{checkpoint}`",
        "",
        "Pre-hygiene checkpoints (trained before the validation-hygiene refactor) "
        "are NOT valid for final paper claims; see the workspace README.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_train_stage(
    config_path: str | Path,
    *,
    seed: Optional[int] = None,
    out_dir: Optional[str | Path] = None,
    evidence_root: Optional[str | Path] = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    require_clean_git: bool = False,
) -> int:
    config = load_paper_training_config(config_path)  # validates (raises on unsafe)

    # Overrides: --seed lets one config produce multiple candidates.
    if seed is not None:
        config["seed"] = int(seed)
        config.setdefault("split", {})["split_seed"] = int(seed)
    if out_dir is not None:
        config.setdefault("output", {})["out_dir"] = str(out_dir)

    run_key = _run_key(config)
    evidence_root = Path(evidence_root) if evidence_root is not None else default_evidence_root()
    evidence_dir = evidence_root / "training" / run_key
    manifest_path = evidence_root / "manifests" / "evidence_manifest.json"
    run_out_dir = resolve_out_dir(config)
    command = build_training_command(config, python=sys.executable)
    placeholders = find_unfilled_placeholders(config)

    if require_clean_git:
        git = collect_environment().get("git", {})
        if git.get("is_dirty"):
            print("[paper-evidence] ERROR: --require-clean-git set but the git tree is dirty.", file=sys.stderr)
            return 2

    expected_artifacts = {
        "checkpoint": run_out_dir / "checkpoints" / "ckpt_best.pt",
        "scaler": run_out_dir / "scaler.json",
        "split_manifest": run_out_dir / "provenance" / "split_manifest.json",
        "config_resolved": run_out_dir / "config.json",
        "history": run_out_dir / "history.csv",
        "environment": run_out_dir / "provenance" / "environment.json",
    }

    if skip_existing and _run_completed(run_out_dir):
        print(f"[paper-evidence] SKIP {run_key}: completed run already exists at {run_out_dir}.")
        _write_manifest(manifest_path, run_key, config, config_path, run_out_dir, command,
                        artifacts=expected_artifacts, dry_run=False, extra={"skipped_existing": True})
        return 0

    if dry_run:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "train_command.txt").write_text(
            subprocess.list2cmdline([str(c) for c in command]) + "\n", encoding="utf-8"
        )
        (evidence_dir / "environment.json").write_text(
            json.dumps(collect_environment(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        _write_manifest(manifest_path, run_key, config, config_path, run_out_dir, command,
                        artifacts=expected_artifacts, dry_run=True,
                        extra={"unfilled_placeholders": placeholders})
        print(f"[paper-evidence] DRY RUN ({run_key}): no training launched.")
        print(f"  planned command: {subprocess.list2cmdline([str(c) for c in command])}")
        print(f"  out dir:         {run_out_dir}")
        print(f"  evidence dir:    {evidence_dir}")
        print(f"  manifest:        {manifest_path}")
        if placeholders:
            print("  WARNING: dataset placeholders are not filled in yet:")
            for ph in placeholders:
                print(f"    - {ph}")
            print("  Fill them before a real (non-dry-run) training launch.")
        return 0

    # Real run: dataset paths must be filled.
    if placeholders:
        print("[paper-evidence] ERROR: dataset paths are still placeholders; fill them before training:",
              file=sys.stderr)
        for ph in placeholders:
            print(f"    - {ph}", file=sys.stderr)
        return 2

    print(f"[paper-evidence] TRAIN {run_key} -> {run_out_dir}")
    print(f"  command: {subprocess.list2cmdline([str(c) for c in command])}")
    result = subprocess.run(command, cwd=str(project_root_from_file(__file__)))
    if result.returncode != 0:
        print(f"[paper-evidence] training FAILED ({run_key}) exit={result.returncode}.", file=sys.stderr)
        _write_manifest(manifest_path, run_key, config, config_path, run_out_dir, command,
                        artifacts=expected_artifacts, dry_run=False,
                        extra={"training_exit_code": int(result.returncode)})
        return 2

    # Verify + package full provenance (fails loudly if not hygiene-compliant).
    try:
        bundle = package_evidence(run_out_dir, evidence_dir, config=config, command=command)
    except PaperEvidenceError as exc:
        print(f"[paper-evidence] HYGIENE CHECK FAILED ({run_key}): {exc}", file=sys.stderr)
        return 3

    _write_manifest(manifest_path, run_key, config, config_path, run_out_dir, command,
                    artifacts={**expected_artifacts, **bundle}, dry_run=False,
                    extra={"evidence_dir": str(evidence_dir)})
    print(f"[paper-evidence] OK ({run_key}). Evidence: {evidence_dir}; manifest: {manifest_path}")
    return 0


def _write_manifest(manifest_path, run_key, config, config_path, run_out_dir, command, *, artifacts, dry_run, extra):
    entry = build_evidence_manifest(
        stage="train",
        run_key=run_key,
        config_path=config_path,
        config=config,
        out_dir=run_out_dir,
        artifacts={k: (str(v) if not isinstance(v, (str, Path)) else v) for k, v in artifacts.items()},
        command=[str(c) for c in command],
        dry_run=dry_run,
        extra=extra,
    )
    write_evidence_manifest(manifest_path, run_key=run_key, entry=entry)


def _run_completed(run_out_dir: Path) -> bool:
    manifest = run_out_dir / "run_manifest.json"
    if not manifest.exists():
        return False
    try:
        return str(json.loads(manifest.read_text(encoding="utf-8")).get("status", "")).lower() == "completed"
    except Exception:
        return False


def run_placeholder_stage(stage: str) -> int:
    print(f"[paper-evidence] stage '{stage}' has no implementation.")
    return 0


# ---------------------------------------------------------------------------
# Shared stage helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_placeholder(value: Any) -> bool:
    from lunaris.surrogate.st_lrps.paper_evidence.training_argv import _is_placeholder as _ph

    return isinstance(value, str) and _ph(value)


def _safe_name(name: Any) -> str:
    text = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name)).strip("_")
    return text or "run"


def _resolve_config(stage: str, config_arg: Optional[str]) -> Optional[Path]:
    if config_arg:
        return Path(config_arg)
    default = _DEFAULT_CONFIGS.get(stage)
    if default is None:
        return None
    repo = project_root_from_file(__file__)
    candidate = repo / default
    return candidate if candidate.exists() else Path(default)


def _model_artifacts(model_dir: str | Path | None) -> dict[str, Any]:
    if not model_dir:
        return {}
    try:
        from lunaris.surrogate.st_lrps.artifacts.manager import make_run_layout

        layout = make_run_layout(Path(model_dir))
        ckpt = layout.ckpt_best if layout.ckpt_best.exists() else layout.ckpt_last
        return {
            "model_checkpoint": ckpt,
            "model_scaler": layout.scaler_json,
            "model_split_manifest": layout.provenance_dir / "split_manifest.json",
            "model_config_resolved": layout.config_json,
        }
    except Exception:
        return {}


def _record_stage(
    evidence_root: Path,
    stage: str,
    run_key: str,
    *,
    config_path: Optional[str | Path] = None,
    config: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[str | Path] = None,
    artifacts: Optional[Mapping[str, str | Path | None]] = None,
    command: Optional[list[str]] = None,
    dry_run: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    manifest_path = Path(evidence_root) / "manifests" / "evidence_manifest.json"
    entry = build_evidence_manifest(
        stage=stage, run_key=run_key, config_path=config_path, config=config,
        out_dir=out_dir, artifacts=artifacts or {}, command=command, dry_run=dry_run, extra=extra,
    )
    write_evidence_manifest(manifest_path, run_key=run_key, entry=entry)


# ---------------------------------------------------------------------------
# Stage: field validation
# ---------------------------------------------------------------------------

def run_field_validation_stage(
    config_path: str | Path,
    *,
    model_dir: Optional[str | Path] = None,
    evidence_root: Path,
    dry_run: bool = False,
) -> int:
    cfg = _load_json(config_path)
    model = str(model_dir) if model_dir else cfg.get("model_dir")
    dataset = cfg.get("dataset")
    # --evidence-root is authoritative for placement; the config out_dir is a
    # documented default for the canonical workspace. A per-model subdir lets the
    # multi-seed stage discover each candidate's field metrics.
    model_name = _safe_name(Path(str(model)).name) if model and not _is_placeholder(model) else "model"
    out_dir = evidence_root / "field_validation" / model_name
    placeholders = [n for n, v in (("model_dir", model), ("dataset", dataset)) if _is_placeholder(v)]

    if dry_run:
        print(f"[paper-evidence] DRY RUN field-validation: model={model} dataset={dataset}")
        print(f"  policies: {cfg.get('policies')}")
        print(f"  out dir:  {out_dir}")
        if placeholders:
            print(f"  WARNING: unfilled placeholders: {placeholders}")
        _record_stage(evidence_root, "field-validation", "field_validation",
                      config_path=config_path, config=cfg, out_dir=out_dir, dry_run=True,
                      extra={"unfilled_placeholders": placeholders})
        return 0

    if placeholders:
        print(f"[paper-evidence] ERROR field-validation: fill {placeholders} before running.", file=sys.stderr)
        return 2

    from lunaris.surrogate.st_lrps.evaluation.validation_suite import (
        DEFAULT_FIELD_POLICIES, run_field_validation, write_field_validation_csvs,
    )

    report = run_field_validation(
        model, dataset,
        policies=cfg.get("policies", list(DEFAULT_FIELD_POLICIES)),
        split_seed=int(cfg.get("split_seed", 1234)),
        val_fraction=float(cfg.get("val_fraction", 0.15)),
        options=cfg.get("options"),
        device=str(cfg.get("device", "cpu")),
    )
    paths = write_field_validation_csvs(report, out_dir)
    artifacts = {"field_metrics_csv": paths.get("metrics"), **_model_artifacts(model)}
    _record_stage(evidence_root, "field-validation", "field_validation",
                  config_path=config_path, config=cfg, out_dir=out_dir, artifacts=artifacts,
                  extra={"policies": list(report.get("field_validation", {}).keys())})
    print(f"[paper-evidence] OK field-validation -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Stage: orbit benchmark (paper-safe)
# ---------------------------------------------------------------------------

def run_orbit_benchmark_stage(
    config_paths: Sequence[str | Path],
    *,
    model_dir: Optional[str | Path] = None,
    evidence_root: Path,
    dry_run: bool = False,
) -> int:
    rc_total = 0
    for cpath in config_paths:
        cfg = _load_json(cpath)
        name = _safe_name(cfg.get("name", Path(cpath).stem))
        out_dir = evidence_root / "orbit_benchmarks" / name
        model = str(model_dir) if model_dir else cfg.get("surrogate", {}).get("model_dir")

        if dry_run:
            print(f"[paper-evidence] DRY RUN orbit-benchmark '{name}': paper_safe={cfg.get('paper_safe')}")
            print(f"  model_dir: {model}")
            print(f"  out dir:   {out_dir}")
            _record_stage(evidence_root, "orbit-benchmark", f"orbit:{name}",
                          config_path=cpath, config=cfg, out_dir=out_dir, dry_run=True,
                          extra={"unfilled_placeholders": [m for m in [model] if _is_placeholder(m)]})
            continue

        if _is_placeholder(model):
            print(f"[paper-evidence] ERROR orbit-benchmark '{name}': surrogate.model_dir is a placeholder.",
                  file=sys.stderr)
            rc_total = rc_total or 2
            continue

        from lunaris.surrogate.st_lrps.evaluation.benchmark_pipeline import run_configured_benchmark

        rc = run_configured_benchmark(cpath, out_dir=out_dir, model_dir=model, paper_safe=True)
        if rc == 0:
            _standardize_orbit_outputs(out_dir)
        artifacts = {
            "benchmark_manifest": out_dir / "benchmark_manifest.json",
            "validation_report": out_dir / "validation_report.json",
            "metrics_summary": out_dir / "metrics_summary.csv",
            "scenario_results": out_dir / "scenario_results.csv",
            **_model_artifacts(model),
        }
        _record_stage(evidence_root, "orbit-benchmark", f"orbit:{name}",
                      config_path=cpath, config=cfg, out_dir=out_dir, artifacts=artifacts,
                      extra={"benchmark_exit_code": int(rc)})
        print(f"[paper-evidence] orbit-benchmark '{name}' exit={rc} -> {out_dir}")
        rc_total = rc_total or rc
    return rc_total


def _standardize_orbit_outputs(out_dir: Path) -> None:
    """Mirror the standardized benchmark CSVs under orbit_benchmark_* names."""
    mapping = {
        "metrics_summary.csv": "orbit_benchmark_metrics.csv",
        "scenario_results.csv": "orbit_benchmark_scenario_results.csv",
        "runtime_summary.csv": "orbit_benchmark_runtime.csv",
        "report.md": "orbit_benchmark_summary.md",
    }
    for src_name, dst_name in mapping.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


# ---------------------------------------------------------------------------
# Stage: worst-case
# ---------------------------------------------------------------------------

def run_worst_case_stage(config_path: str | Path, *, evidence_root: Path, dry_run: bool = False) -> int:
    cfg = _load_json(config_path)
    benchmark_dir = cfg.get("benchmark_dir")
    out_dir = evidence_root / "worst_case_analysis"

    if dry_run:
        print(f"[paper-evidence] DRY RUN worst-case: benchmark_dir={benchmark_dir} -> {out_dir}")
        _record_stage(evidence_root, "worst-case", "worst_case", config_path=config_path, config=cfg,
                      out_dir=out_dir, dry_run=True,
                      extra={"unfilled_placeholders": [benchmark_dir] if _is_placeholder(benchmark_dir) else []})
        return 0

    if _is_placeholder(benchmark_dir) or not benchmark_dir:
        print("[paper-evidence] ERROR worst-case: fill benchmark_dir before running.", file=sys.stderr)
        return 2

    from lunaris.surrogate.st_lrps.paper_evidence.worst_case import run_worst_case_from_benchmark_dir

    paths = run_worst_case_from_benchmark_dir(
        benchmark_dir, out_dir,
        model=str(cfg.get("model", "ST-LRPS")),
        train_alt_min_km=cfg.get("train_altitude_min_km"),
        train_alt_max_km=cfg.get("train_altitude_max_km"),
        top_n=int(cfg.get("top_n", 5)),
    )
    _record_stage(evidence_root, "worst-case", "worst_case", config_path=config_path, config=cfg,
                  out_dir=out_dir,
                  artifacts={"worst_case_csv": paths.get("csv"),
                             "scenario_results": Path(benchmark_dir) / "scenario_results.csv"})
    print(f"[paper-evidence] OK worst-case -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Stage: ablation (secondary)
# ---------------------------------------------------------------------------

def run_ablation_stage(
    config_path: str | Path, *, evidence_root: Path, dry_run: bool = False, execute_override: Optional[bool] = None
) -> int:
    cfg = _load_json(config_path)
    ds = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), Mapping) else {}
    out_root = str(evidence_root / "ablation")  # --evidence-root authoritative
    argv: list[str] = []
    if ds.get("train_data") and ds.get("val_data"):
        argv += ["--train-data", str(ds["train_data"]), "--val-data", str(ds["val_data"])]
    elif ds.get("train_data"):
        argv += ["--train-data", str(ds["train_data"])]
    argv += ["--out-root", out_root, "--seed", str(cfg.get("seed", 42)), "--matrix", str(cfg.get("matrix", "default"))]
    if cfg.get("epochs"):
        argv += ["--epochs", str(cfg["epochs"])]
    if cfg.get("run_eval_after_training"):
        argv.append("--run-eval-after-training")
    execute = bool(execute_override if execute_override is not None else cfg.get("execute", False))
    argv.append("--execute" if (execute and not dry_run) else "--dry-run")

    from lunaris.surrogate.st_lrps.evaluation import ablation as abl

    rc = abl.main(argv)
    _record_stage(evidence_root, "ablation", "ablation", config_path=config_path, config=cfg, out_dir=out_root,
                  artifacts={"ablation_manifest": Path(out_root) / "ablation_manifest.json",
                             "ablation_summary_csv": Path(out_root) / "st_lrps_ablation_summary.csv"},
                  command=["python", "-m", "lunaris.surrogate.st_lrps.evaluation.ablation", *argv],
                  dry_run=bool(dry_run or not execute),
                  extra={"executed": bool(execute and not dry_run)})
    return rc


# ---------------------------------------------------------------------------
# Stage: multi-seed summary
# ---------------------------------------------------------------------------

def run_multi_seed_stage(*, evidence_root: Path, dry_run: bool = False) -> int:
    from lunaris.surrogate.st_lrps.paper_evidence.multi_seed import (
        aggregate_multi_seed, collect_seed_entry, write_multi_seed_outputs,
    )

    out_dir = evidence_root / "tables"
    if dry_run:
        print(f"[paper-evidence] DRY RUN multi-seed: would aggregate per-seed outputs under {evidence_root}.")
        _record_stage(evidence_root, "multi-seed", "multi_seed", out_dir=out_dir, dry_run=True)
        return 0

    # Discover per-seed training runs from the evidence manifest.
    manifest_path = evidence_root / "manifests" / "evidence_manifest.json"
    entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        for run_key, rec in (manifest.get("runs", {}) or {}).items():
            if rec.get("stage") != "train":
                continue
            seed = (rec.get("config", {}) or {}).get("seed")
            if seed is None:
                continue
            checkpoint = (rec.get("artifacts", {}).get("checkpoint", {}) or {}).get("sha256")
            field_csv = evidence_root / "field_validation" / f"seed{seed}" / "field_validation_metrics.csv"
            field_csv = field_csv if field_csv.exists() else (evidence_root / "field_validation" / "field_validation_metrics.csv")
            bench = {}
            for case, sub in (("1day", "benchmark_1day_high_degree"), ("5day", "benchmark_5day_general")):
                p = evidence_root / "orbit_benchmarks" / sub / "metrics_summary.csv"
                if p.exists():
                    bench[case] = p
            entries.append(collect_seed_entry(int(seed), field_metrics_csv=field_csv if field_csv.exists() else None,
                                               benchmark_metrics=bench or None, artifact_hash=checkpoint))
    summary = aggregate_multi_seed(entries) if entries else aggregate_multi_seed([])
    paths = write_multi_seed_outputs(summary, out_dir)
    _record_stage(evidence_root, "multi-seed", "multi_seed", out_dir=out_dir,
                  artifacts={"multi_seed_csv": paths.get("csv")},
                  extra={"n_seeds": summary.get("n_seeds"), "single_seed_limitation": summary.get("single_seed_limitation")})
    print(f"[paper-evidence] OK multi-seed ({summary.get('n_seeds')} seed(s)) -> {paths.get('csv')}")
    return 0


# ---------------------------------------------------------------------------
# Stage: tables + figures
# ---------------------------------------------------------------------------

def run_tables_stage(*, evidence_root: Path, dry_run: bool = False) -> int:
    from lunaris.surrogate.st_lrps.paper_evidence.paper_tables import (
        generate_paper_figures, generate_paper_tables,
    )

    if dry_run:
        print(f"[paper-evidence] DRY RUN tables: would regenerate tables/figures from CSVs under {evidence_root}.")
        _record_stage(evidence_root, "tables", "tables", out_dir=evidence_root / "tables", dry_run=True)
        return 0

    tables = generate_paper_tables(evidence_root, evidence_root / "tables")
    figures = generate_paper_figures(evidence_root, evidence_root / "figures")
    _record_stage(evidence_root, "tables", "tables", out_dir=evidence_root / "tables",
                  artifacts={"tables_index": tables.get("index")},
                  extra={"tables_written": list(tables.get("written", {}).keys()),
                         "tables_missing": tables.get("missing", []),
                         "figures_skipped": figures.get("skipped", True)})
    print(f"[paper-evidence] OK tables ({len(tables.get('written', {}))} written) + figures "
          f"({'skipped' if figures.get('skipped') else len(figures.get('rendered', {}))}).")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lunaris-st-lrps-paper-evidence",
        description="ST-LRPS paper evidence pipeline (train, field validation, orbit benchmarks, "
                    "worst-case, multi-seed, tables; ablation is secondary).",
    )
    ap.add_argument(
        "--stage",
        choices=SUPPORTED_STAGES,
        default="train",
        help="Pipeline stage. 'all' = train, field-validation, orbit-benchmark, worst-case, "
             "multi-seed, tables (ablation is run explicitly).",
    )
    ap.add_argument("--config", default=None, help="Config JSON for the stage (defaults per stage).")
    ap.add_argument("--benchmark-config", action="append", default=None,
                    help="Orbit benchmark config(s); repeatable. Defaults to the 1-day + 5-day paper configs.")
    ap.add_argument("--model-dir", default=None, help="Trained ST-LRPS run dir (field-validation / orbit-benchmark).")
    ap.add_argument("--seed", type=int, default=None, help="Override the train config seed (and split seed).")
    ap.add_argument("--out-dir", default=None, help="Override the train config output directory.")
    ap.add_argument("--evidence-root", default=None, help="Evidence workspace root (default: validation/paper_evidence/st_lrps).")
    ap.add_argument("--dry-run", action="store_true", help="Validate + plan only; do no heavy work.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip a train run whose output already completed.")
    ap.add_argument("--require-clean-git", action="store_true", help="Fail if the git working tree is dirty.")
    ap.add_argument("--execute-ablation", action="store_true", help="ablation stage: launch training (default dry-run).")
    ap.add_argument("--mark-pre-hygiene", default=None, help="Mark a run directory as pre-hygiene and exit.")
    return ap


def _dispatch_stage(stage: str, args, evidence_root: Path) -> int:
    if stage == "train":
        if not args.config:
            print("[paper-evidence] ERROR: --config is required for the train stage.", file=sys.stderr)
            return 2
        try:
            return run_train_stage(
                args.config, seed=args.seed, out_dir=args.out_dir, evidence_root=evidence_root,
                dry_run=bool(args.dry_run), skip_existing=bool(args.skip_existing),
                require_clean_git=bool(args.require_clean_git),
            )
        except PaperConfigError as exc:
            print(f"[paper-evidence] CONFIG REJECTED:\n{exc}", file=sys.stderr)
            return 2

    if stage == "field-validation":
        cfg = _resolve_config(stage, args.config)
        return run_field_validation_stage(cfg, model_dir=args.model_dir, evidence_root=evidence_root,
                                          dry_run=bool(args.dry_run))
    if stage == "orbit-benchmark":
        configs = args.benchmark_config or [
            str(project_root_from_file(__file__) / c) for c in _DEFAULT_BENCHMARK_CONFIGS
        ]
        return run_orbit_benchmark_stage(configs, model_dir=args.model_dir, evidence_root=evidence_root,
                                         dry_run=bool(args.dry_run))
    if stage == "worst-case":
        return run_worst_case_stage(_resolve_config(stage, args.config), evidence_root=evidence_root,
                                    dry_run=bool(args.dry_run))
    if stage == "ablation":
        return run_ablation_stage(_resolve_config(stage, args.config), evidence_root=evidence_root,
                                  dry_run=bool(args.dry_run),
                                  execute_override=True if args.execute_ablation else None)
    if stage == "multi-seed":
        return run_multi_seed_stage(evidence_root=evidence_root, dry_run=bool(args.dry_run))
    if stage == "tables":
        return run_tables_stage(evidence_root=evidence_root, dry_run=bool(args.dry_run))
    return run_placeholder_stage(stage)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.mark_pre_hygiene:
        marker = mark_run_pre_hygiene(args.mark_pre_hygiene)
        print(f"[paper-evidence] marked pre-hygiene: {marker}")
        return 0

    evidence_root = Path(args.evidence_root) if args.evidence_root else default_evidence_root()
    stages = ALL_STAGES if args.stage == "all" else (args.stage,)
    rc = 0
    for stage in stages:
        stage_rc = _dispatch_stage(stage, args, evidence_root)
        rc = rc or stage_rc
    return rc


__all__ = [
    "ALL_STAGES",
    "SUPPORTED_STAGES",
    "PaperEvidenceError",
    "default_evidence_root",
    "main",
    "mark_run_pre_hygiene",
    "package_evidence",
    "run_ablation_stage",
    "run_field_validation_stage",
    "run_multi_seed_stage",
    "run_orbit_benchmark_stage",
    "run_tables_stage",
    "run_train_stage",
    "run_worst_case_stage",
    "verify_paper_run_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
