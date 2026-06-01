# -*- coding: utf-8 -*-
"""Provenance capture for reproducible benchmark runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .benchmark_config import canonical_json_text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(payload))


def artifact_record(path: str | Path | None, *, label: str = "file") -> dict[str, Any]:
    if path is None or str(path) == "":
        return {"path": None, "sha256": None, "missing_reason": f"{label} not configured"}
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "sha256": None, "missing_reason": f"{label} does not exist locally"}
    if not p.is_file():
        return {"path": str(p), "sha256": None, "missing_reason": f"{label} is not a file"}
    return {"path": str(p), "sha256": sha256_file(p), "missing_reason": None}


def collect_git_info(cwd: str | Path | None = None) -> dict[str, Any]:
    root = Path(cwd or Path.cwd())

    def run_git(args: list[str]) -> tuple[str | None, str | None]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception as exc:
            return None, str(exc)
        if proc.returncode != 0:
            return None, (proc.stderr or proc.stdout).strip() or f"git exited {proc.returncode}"
        return proc.stdout.strip(), None

    commit, commit_error = run_git(["rev-parse", "HEAD"])
    branch, branch_error = run_git(["branch", "--show-current"])
    status, status_error = run_git(["status", "--porcelain"])
    root_text, root_error = run_git(["rev-parse", "--show-toplevel"])

    return {
        "commit_sha": commit,
        "branch": branch or None,
        "is_dirty": None if status is None else bool(status),
        "repo_root": root_text,
        "errors": {
            "commit": commit_error,
            "branch": branch_error,
            "dirty": status_error,
            "root": root_error,
        },
    }


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": None,
        "scipy_version": None,
        "torch_version": None,
        "cuda_available": False,
        "device_name": None,
        "optional_import_errors": {},
    }
    try:
        import numpy as np  # type: ignore

        env["numpy_version"] = np.__version__
    except Exception as exc:
        env["optional_import_errors"]["numpy"] = str(exc)
    try:
        import scipy  # type: ignore

        env["scipy_version"] = scipy.__version__
    except Exception as exc:
        env["optional_import_errors"]["scipy"] = str(exc)
    try:
        import torch  # type: ignore

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if env["cuda_available"]:
            try:
                env["device_name"] = torch.cuda.get_device_name(0)
            except Exception as exc:
                env["optional_import_errors"]["torch_device_name"] = str(exc)
        else:
            env["device_name"] = "cpu"
    except Exception as exc:
        env["optional_import_errors"]["torch"] = str(exc)
        env["device_name"] = "cpu"
    return env


def build_benchmark_manifest(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    resolved_config_sha256: str,
    output_dir: str | Path,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    truth = config.get("truth", {}) if isinstance(config.get("truth"), Mapping) else {}
    surrogate = config.get("surrogate", {}) if isinstance(config.get("surrogate"), Mapping) else {}
    propagation = config.get("propagation", {}) if isinstance(config.get("propagation"), Mapping) else {}
    scenario = config.get("scenario", {}) if isinstance(config.get("scenario"), Mapping) else {}
    dataset = config.get("dataset", {}) if isinstance(config.get("dataset"), Mapping) else {}

    model_dir = surrogate.get("model_dir")
    checkpoint = _find_checkpoint(model_dir)
    model_config = _model_config_path(model_dir)
    gravity_file = truth.get("gravity_file") or truth.get("file")
    dataset_path = dataset.get("path") or dataset.get("file")

    manifest = {
        "schema_version": 1,
        "benchmark_name": config.get("name"),
        "created_at_utc": utc_now_iso(),
        "output_dir": str(output_dir),
        "repo": collect_git_info(cwd),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "resolved_config_sha256": resolved_config_sha256,
        },
        "scenario": {
            "seed": scenario.get("seed"),
            "count": scenario.get("count"),
            "type": scenario.get("type"),
            "altitude_min_km": scenario.get("altitude_min_km"),
            "altitude_max_km": scenario.get("altitude_max_km"),
            "eccentricity_mode": scenario.get("eccentricity_mode"),
            "inclination_mode": scenario.get("inclination_mode"),
        },
        "models": {
            "truth": {
                "kind": truth.get("model"),
                "degree": truth.get("degree"),
                "gravity_file": artifact_record(gravity_file, label="gravity file"),
                "integrator": truth.get("integrator"),
                "rtol": truth.get("rtol"),
                "atol": truth.get("atol"),
            },
            "baselines": [
                {
                    "name": item.get("name"),
                    "kind": item.get("model"),
                    "degree": item.get("degree"),
                }
                for item in config.get("baselines", [])
                if isinstance(item, Mapping)
            ],
            "surrogate": {
                "enabled": surrogate.get("enabled"),
                "model_dir": str(model_dir) if model_dir else None,
                "model_dir_missing_reason": _missing_dir_reason(model_dir),
                "checkpoint": artifact_record(checkpoint, label="checkpoint"),
                "config": artifact_record(model_config, label="surrogate config"),
                "runtime_model_kind": surrogate.get("runtime_model_kind", "potential_autograd"),
                "baseline_degree": surrogate.get("baseline_degree"),
            },
        },
        "data": {
            "dataset": artifact_record(dataset_path, label="dataset"),
        },
        "numerics": {
            "integrator": propagation.get("integrator"),
            "dt_s": propagation.get("dt_s"),
            "output_dt_s": propagation.get("output_dt_s"),
            "dtype": propagation.get("dtype"),
            "duration_days": propagation.get("duration_days"),
        },
        "environment": collect_environment(),
    }
    return manifest


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canonical_json_text(payload), encoding="utf-8")
    return p


def _find_checkpoint(model_dir: Any) -> Path | None:
    if not model_dir:
        return None
    path = Path(str(model_dir)).expanduser()
    if path.is_file():
        return path
    candidates = [
        path / "checkpoints" / "ckpt_best.pt",
        path / "checkpoints" / "ckpt_last.pt",
        path / "ckpt_best.pt",
        path / "ckpt_last.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    try:
        all_candidates = sorted(path.glob("**/*.pt")) + sorted(path.glob("**/*.pth")) + sorted(path.glob("**/*.ckpt"))
    except Exception:
        all_candidates = []
    return all_candidates[0] if all_candidates else None


def _model_config_path(model_dir: Any) -> Path | None:
    if not model_dir:
        return None
    path = Path(str(model_dir)).expanduser()
    if path.is_file():
        return path.with_name("config.json")
    candidate = path / "config.json"
    return candidate if candidate.exists() else candidate


def _missing_dir_reason(model_dir: Any) -> str | None:
    if not model_dir:
        return "surrogate model_dir not configured"
    path = Path(str(model_dir)).expanduser()
    if not path.exists():
        return "surrogate model_dir does not exist locally"
    if not path.is_dir() and not path.is_file():
        return "surrogate model_dir is not a directory or checkpoint file"
    return None
